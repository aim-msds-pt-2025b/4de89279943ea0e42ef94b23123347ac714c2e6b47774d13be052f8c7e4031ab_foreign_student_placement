import pandas as pd
import os
import mlflow
from pathlib import Path
import json
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# Add src directory to Python path for imports
import sys
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Comprehensive joblib warning suppression for Windows
import warnings

warnings.filterwarnings(
    "ignore", message=".*Could not find the number of physical cores.*"
)
warnings.filterwarnings(
    "ignore", message=".*The system cannot find the file specified.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Set environment variables before any sklearn imports to prevent CPU detection
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # Disable multiprocessing to avoid warnings

# Redirect stderr temporarily to suppress low-level Windows errors
import io
import contextlib


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


try:
    from .data_preprocessing import preprocess_data
    from .feature_engineering import engineer_features
    from .model_training import (
        train_base_models,
        tune_models,
        build_ensemble,
        save_model,
    )
    from .evaluation import evaluate_models, select_and_save_best
    from .drift_detection import detect_drift
except ImportError:
    from data_preprocessing import preprocess_data
    from feature_engineering import engineer_features
    from model_training import (
        train_base_models,
        tune_models,
        build_ensemble,
        save_model,
    )
    from evaluation import evaluate_models, select_and_save_best
    from drift_detection import detect_drift

# Try optional visualization imports separately to avoid hard failures when seaborn is missing
try:
    from visualization import (
        plot_target_distribution,
        plot_feature_correlations,
        plot_roc_curves,
        plot_confusion_matrix,
    )

    VIS_AVAILABLE = True
except Exception:
    try:
        from .visualization import (
            plot_target_distribution,
            plot_feature_correlations,
            plot_roc_curves,
            plot_confusion_matrix,
        )

        VIS_AVAILABLE = True
    except Exception:
        VIS_AVAILABLE = False

# Optional imports
SHAP_AVAILABLE = False
try:
    # Only import shap if needed, avoiding lint errors
    if False:  # Disabled for now
        import shap
        import matplotlib.pyplot as plt

        SHAP_AVAILABLE = True
except Exception:
    pass


# Set MLflow tracking URI to Docker service for UI visibility
# This must be done before any MLflow operations in imported modules
mlflow.set_tracking_uri("http://localhost:5000")


def main():
    print(">> Starting ML Pipeline...")

    # 1) Load & preprocess (with drifted data generation)
    print(">> Loading and preprocessing data...")
    with suppress_stderr():
        result = preprocess_data("data/global_student_migration.csv", emit_drifted=True)

    # Handle both old and new return formats
    if len(result) == 8:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_drifted,
            y_train_drifted,
            X_test_drifted,
            y_test_drifted,
        ) = result

        # Save original test set for drift detection
        original_test = pd.concat([X_test, y_test], axis=1)
        original_test.to_csv("data/test.csv", index=False)
        print(">> Original and drifted datasets saved")
    else:
        X_train, X_test, y_train, y_test = result
        print(">> Warning: Drifted data not generated")

    # 2) Feature engineering
    with suppress_stderr():
        X_train_fe, X_test_fe = engineer_features(X_train, X_test)
    feature_names = list(X_train_fe.columns)
    print(f"Data shape - Train: {X_train_fe.shape}, Test: {X_test_fe.shape}")
    print(f"Features: {len(feature_names)}")

    # 3) Train & save base models (optional artifacts)
    with suppress_stderr():
        train_base_models(X_train_fe, y_train, models_dir="models")

    # 4) Hyperparameter tuning (with MLflow tracking handled inside if available)
    with suppress_stderr():
        best_estimators = tune_models(
            X_train_fe,
            y_train,
            X_test=X_test_fe,
            y_test=y_test,
            track_mlflow=True,
        )

    # 5) Build and save ensemble
    ensemble = build_ensemble(best_estimators, X_train_fe, y_train, models_dir="models")
    all_models = {**best_estimators, "ensemble": ensemble}

    # 6) Evaluate all tuned models + ensemble
    print(">> Evaluating models...")
    metrics_df = evaluate_models(all_models, X_test_fe, y_test)
    print("\nModel Performance:")
    print(metrics_df)

    # 7) Select & save only the best model’s metrics and artifact
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    best_name, cm = select_and_save_best(
        metrics_df,
        all_models,
        X_test_fe,
        y_test,
        metrics_txt_path="reports/metrics.txt",
    )
    save_model(all_models[best_name], "models/best_model.joblib")
    print(f"\n>> Best model: {best_name}")
    print("Confusion matrix:\n", cm)

    # 8) Check if model meets performance threshold and register if so
    best_accuracy = metrics_df.loc[best_name, "accuracy"]
    print(f">> Best model accuracy: {best_accuracy:.4f}")

    # Allow overriding threshold via environment (default 0.8 per homework)
    try:
        threshold = float(os.environ.get("ML_THRESHOLD", "0.8"))
    except Exception:
        threshold = 0.8

    if best_accuracy > threshold:  # Classification threshold as per homework
        print(">> Performance threshold met, registering model with MLflow...")
        try:
            from .mlflow_config import MLflowTracker
        except Exception:
            from mlflow_config import MLflowTracker
        tracker = MLflowTracker()
        # Log a run for the best model to attach model artifact and then register
        best_model = all_models[best_name]
        run_id = tracker.log_model_run(
            model_name=best_name,
            model=best_model,
            X_test=X_test_fe,
            y_test=y_test,
            hyperparams=None,
            additional_metrics={"selected_best_accuracy": float(best_accuracy)},
        )
        try:
            tracker.register_best_model(
                model_name=best_name, run_id=run_id, stage="Staging"
            )
            print(">> Model registered successfully and moved to Staging!")
        except Exception as e:
            print(f">> Model registration failed: {e}")
    else:
        print(f">> Performance threshold not met (accuracy {best_accuracy:.4f} <= 0.8)")

    # 9) Run drift detection as per homework requirements
    if Path("data/test.csv").exists() and Path("data/drifted_test.csv").exists():
        print(">> Running drift detection on test set...")
        try:
            test_drift_results = detect_drift("data/test.csv", "data/drifted_test.csv")

            # Log drift status to MLflow
            with mlflow.start_run(run_name="drift_detection"):
                mlflow.log_param(
                    "test_drift_detected", test_drift_results["drift_detected"]
                )
                mlflow.log_param(
                    "test_overall_drift_score",
                    test_drift_results["overall_drift_score"],
                )

            print(f">> Drift detected: {test_drift_results['drift_detected']}")
            print(
                f">> Overall drift score: {test_drift_results['overall_drift_score']:.4f}"
            )

            # Raise error if drift detected as per homework requirements
            if test_drift_results["drift_detected"]:
                raise ValueError(
                    "Data drift detected in test set! Model retraining required."
                )

        except Exception as e:
            print(f">> Drift detection failed or drift detected: {e}")
            # Re-raise if it's the drift detection error
            if "Data drift detected" in str(e):
                raise e

    # 8) Visualizations & ROC curves (only if viz module is available)
    if VIS_AVAILABLE:
        raw = pd.read_csv("data/global_student_migration.csv")
        raw["placement_status"] = raw["placement_status"].map(
            {"Placed": 1, "Not Placed": 0}
        )
        plot_target_distribution(raw)
        numeric_cols = [
            "gpa_or_score",
            "test_score",
            "year_of_enrollment",
            "graduation_year",
        ]
        plot_feature_correlations(raw, numeric_cols)
        plot_roc_curves(all_models, X_test_fe, y_test)
        plot_confusion_matrix(all_models[best_name], X_test_fe, y_test, name=best_name)
        print("\n>> All figures exported to reports/figures/")
    else:
        print(">> Visualization module not available, skipping plots.")

    # 9) SHAP interpretability (best model if linear-like)
    if SHAP_AVAILABLE and hasattr(all_models[best_name], "coef_"):
        try:
            print("🔍 Generating SHAP explanations...")
            import shap
            import matplotlib.pyplot as plt

            explainer = shap.LinearExplainer(all_models[best_name], X_train_fe)
            sample = X_test_fe[: min(300, X_test_fe.shape[0])]
            shap_vals = explainer.shap_values(sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_vals, sample, feature_names=feature_names, show=False
            )
            plt.tight_layout()
            plt.savefig(
                "reports/figures/shap_summary.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(">> SHAP plot saved to reports/figures/shap_summary.png")
        except Exception as e:
            print(f">> SHAP analysis failed: {e}")
    else:
        print(">> SHAP analysis skipped (not available or not linear model)")

    # 10) Baseline stats for drift detection (numeric-only; cast bool -> int)
    baseline_stats = {}
    for fname in feature_names:
        col = X_train_fe[fname]
        if is_bool_dtype(col):
            col = col.astype(int)
        if not is_numeric_dtype(col):
            # Skip non-numeric features for numeric drift stats
            continue
        baseline_stats[fname] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "q25": float(col.quantile(0.25)),
            "q50": float(col.quantile(0.50)),
            "q75": float(col.quantile(0.75)),
        }
    with open("models/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(">> Baseline stats saved for drift detection")


if __name__ == "__main__":
    main()
