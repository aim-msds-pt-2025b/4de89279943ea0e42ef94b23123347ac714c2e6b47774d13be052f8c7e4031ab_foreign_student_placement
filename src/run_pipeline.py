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
import pandas as pd
from pathlib import Path
import json
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# Optional imports
try:
    import shap
    import matplotlib.pyplot as plt

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def main():
    print(">> Starting ML Pipeline...")

    # 1) Load & preprocess
    print(">> Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/global_student_migration.csv"
    )

    # 2) Feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)
    feature_names = list(X_train_fe.columns)
    print(f"Data shape - Train: {X_train_fe.shape}, Test: {X_test_fe.shape}")
    print(f"Features: {len(feature_names)}")

    # 3) Train & save base models (optional artifacts)
    train_base_models(X_train_fe, y_train, models_dir="models")

    # 4) Hyperparameter tuning (with MLflow tracking handled inside if available)
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
