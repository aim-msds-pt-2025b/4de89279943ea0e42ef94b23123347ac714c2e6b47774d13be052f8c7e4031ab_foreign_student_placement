from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_base_models, tune_models, build_ensemble
from evaluation import evaluate_models, select_and_save_best
from visualization import (
    plot_target_distribution,
    plot_feature_correlations,
    plot_roc_curves,
    plot_confusion_matrix,
)
from mlflow_config import MLflowTracker
from simple_drift_detection import DriftDetector, create_synthetic_drift_data
import pandas as pd
import os


def main():
    print("🚀 Starting ML Pipeline with MLflow tracking and drift detection...")

    # Initialize MLflow tracker
    try:
        mlflow_tracker = MLflowTracker()
        print("✅ MLflow tracker initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize MLflow: {e}")
        mlflow_tracker = None

    # 1) Load & preprocess
    print("📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/global_student_migration.csv"
    )

    # 2) Feature engineering
    print("🔧 Engineering features...")
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)

    # 3) Train & save base models
    print("🤖 Training base models...")
    train_base_models(X_train_fe, y_train, models_dir="models")

    # 4) Hyperparameter tuning with MLflow tracking
    print("⚙️ Hyperparameter tuning with MLflow tracking...")
    best_estimators = tune_models(
        X_train_fe,
        y_train,
        X_test_fe,
        y_test,
        track_mlflow=(mlflow_tracker is not None),
    )

    # 5) Build and save ensemble
    print("🎯 Building ensemble model...")
    ensemble = build_ensemble(best_estimators, X_train_fe, y_train, models_dir="models")
    all_models = {**best_estimators, "ensemble": ensemble}

    # Track ensemble with MLflow
    if mlflow_tracker:
        try:
            ensemble_run_id = mlflow_tracker.log_model_run(
                model_name="ensemble",
                model=ensemble,
                X_test=X_test_fe,
                y_test=y_test,
                hyperparams={"voting": "soft", "n_estimators": len(best_estimators)},
            )
            print(f"✅ Ensemble model logged to MLflow: {ensemble_run_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not log ensemble to MLflow: {e}")

    # 6) Evaluate all tuned models + ensemble
    print("📈 Evaluating models...")
    metrics_df = evaluate_models(all_models, X_test_fe, y_test)
    print(metrics_df)

    # 7) Select & save only the best model's metrics
    print("🏆 Selecting best model...")
    best_name, cm = select_and_save_best(
        metrics_df,
        all_models,
        X_test_fe,
        y_test,
        metrics_txt_path="reports/metrics.txt",
    )
    print(f"\n🥇 Best model: {best_name}")
    print("Confusion matrix:\n", cm)

    # Register best model in MLflow Model Registry
    if mlflow_tracker:
        try:
            # Find the best model's run ID and register it
            experiment = mlflow_tracker.client.get_experiment_by_name(
                mlflow_tracker.experiment_name
            )
            runs = mlflow_tracker.client.search_runs(
                experiment_ids=[experiment.experiment_id]
            )

            # Find run for best model
            best_run = None
            for run in runs:
                if run.data.params.get("model_name") == best_name:
                    best_run = run
                    break

            if best_run:
                model_version = mlflow_tracker.register_best_model(
                    model_name=best_name,
                    run_id=best_run.info.run_id,
                    stage="Production",
                )
                print(
                    f"✅ Best model registered: {model_version.name} v{model_version.version}"
                )
        except Exception as e:
            print(f"⚠️ Warning: Could not register best model: {e}")

    # 8) Drift Detection Demo
    print("\n🔍 Running drift detection demo...")
    try:
        run_drift_detection_demo()
    except Exception as e:
        print(f"⚠️ Warning: Drift detection failed: {e}")

    # 9) Running Visualizations & Exporting Figures
    print("📊 Generating visualizations...")
    # — re-load original data so we can plot distributions & correlations
    raw = pd.read_csv("data/global_student_migration.csv")
    raw["placement_status"] = raw["placement_status"].map(
        {"Placed": 1, "Not Placed": 0}
    )

    # 9a) Target distribution
    plot_target_distribution(raw)

    # 9b) Feature correlations (numeric columns only)
    numeric_cols = [
        "gpa_or_score",
        "test_score",
        "year_of_enrollment",
        "graduation_year",
    ]
    plot_feature_correlations(raw, numeric_cols)

    # 9c) ROC curves for all tuned models + ensemble
    plot_roc_curves(all_models, X_test_fe, y_test)

    # 9d) Confusion matrix for the best model
    plot_confusion_matrix(all_models[best_name], X_test_fe, y_test, name=best_name)

    print("\n✅ All figures exported to reports/figures/")

    # 10) Generate MLflow comparison report
    if mlflow_tracker:
        print("\n📋 Generating MLflow run comparison...")
        try:
            comparison_df = mlflow_tracker.compare_runs()
            comparison_df.to_csv("reports/mlflow_run_comparison.csv", index=False)
            print("✅ MLflow run comparison saved to reports/mlflow_run_comparison.csv")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate MLflow comparison: {e}")

    print("\n🎉 Pipeline completed successfully!")
    if mlflow_tracker:
        print("\n📊 View MLflow UI with: mlflow ui --backend-store-uri mlruns")


def run_drift_detection_demo():
    """Demonstrate drift detection capabilities."""
    print("🔍 Setting up drift detection demo...")

    # Load original data
    raw_data = pd.read_csv("data/global_student_migration.csv")
    raw_data["placement_status"] = raw_data["placement_status"].map(
        {"Placed": 1, "Not Placed": 0}
    )

    # Define feature types
    numerical_features = [
        "gpa_or_score",
        "test_score",
        "year_of_enrollment",
        "graduation_year",
    ]

    categorical_features = [
        "gender",
        "nationality",
        "destination_country",
        "field_of_study",
        "degree_level",
    ]

    # Initialize drift detector
    drift_detector = DriftDetector(
        reference_data=raw_data,
        target_column="placement_status",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Create synthetic drifted data for demo
    print("📊 Creating synthetic drifted data...")
    drifted_data = create_synthetic_drift_data(
        original_data=raw_data,
        drift_magnitude=0.3,
        drift_features=["gpa_or_score", "test_score"],
    )

    # Detect data drift
    print("🔍 Detecting data drift...")
    drift_results = drift_detector.detect_data_drift(
        drifted_data, significance_level=0.05
    )

    # Detect target drift
    print("🎯 Detecting target drift...")
    target_drift_results = drift_detector.detect_target_drift(
        drifted_data, significance_level=0.05
    )

    # Run data quality tests
    print("✅ Running data quality tests...")
    quality_results = drift_detector.run_data_quality_tests(drifted_data)

    # Generate alerts
    print("🚨 Generating drift alerts...")
    alert = drift_detector.generate_drift_alert(drift_results, alert_threshold=0.2)

    # Combine all results
    complete_drift_report = {
        "data_drift": drift_results,
        "target_drift": target_drift_results,
        "data_quality": quality_results,
        "alert": alert,
    }

    # Save drift report
    os.makedirs("reports", exist_ok=True)
    drift_detector.save_drift_report(complete_drift_report, "reports/drift_report.json")

    # Print summary
    print("\n📋 Drift Detection Summary:")
    print(
        f"   • Dataset drift detected: {drift_results.get('dataset_drift_detected', 'Unknown')}"
    )
    print(f"   • Drift share: {drift_results.get('drift_share', 0):.2%}")
    print(f"   • Drifted features: {len(drift_results.get('drifted_features', []))}")
    print(f"   • Alert level: {alert.get('alert_level', 'Unknown')}")
    print(f"   • Alert triggered: {alert.get('alert_triggered', False)}")

    if alert.get("alert_triggered", False):
        print(f"   • Message: {alert.get('message', '')}")
        print("   • Recommendations:")
        for rec in alert.get("recommendations", []):
            print(f"     - {rec}")

    print("✅ Drift detection demo completed!")
    return complete_drift_report


if __name__ == "__main__":
    main()
