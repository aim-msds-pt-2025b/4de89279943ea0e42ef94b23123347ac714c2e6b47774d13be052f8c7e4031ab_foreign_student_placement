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
import pandas as pd


def main():
    # 1) Load & preprocess
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/global_student_migration.csv"
    )

    # 2) Feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)

    # 3) Train & save base models
    train_base_models(X_train_fe, y_train, models_dir="models")

    # 4) Hyperparameter tuning
    best_estimators = tune_models(X_train_fe, y_train)

    # 5) Build and save ensemble
    ensemble = build_ensemble(best_estimators, X_train_fe, y_train, models_dir="models")
    all_models = {**best_estimators, "ensemble": ensemble}

    # 6) Evaluate all tuned models + ensemble
    metrics_df = evaluate_models(all_models, X_test_fe, y_test)
    print(metrics_df)

    # 7) Select & save only the best model’s metrics
    best_name, cm = select_and_save_best(
        metrics_df,
        all_models,
        X_test_fe,
        y_test,
        metrics_txt_path="reports/metrics.txt",
    )
    print(f"\nBest model: {best_name}")
    print("Confusion matrix:\n", cm)
    # 8) Running Visualizations & Exporting Figures
    # — re-load original data so we can plot distributions & correlations
    raw = pd.read_csv("data/global_student_migration.csv")
    raw["placement_status"] = raw["placement_status"].map(
        {"Placed": 1, "Not Placed": 0}
    )

    # 8a) Target distribution
    plot_target_distribution(raw)

    # 8b) Feature correlations (numeric columns only)
    numeric_cols = [
        "gpa_or_score",
        "test_score",
        "year_of_enrollment",
        "graduation_year",
    ]
    plot_feature_correlations(raw, numeric_cols)

    # 8c) ROC curves for all tuned models  ensemble
    plot_roc_curves(all_models, X_test_fe, y_test)

    # 8d) Confusion matrix for the best model
    plot_confusion_matrix(all_models[best_name], X_test_fe, y_test, name=best_name)

    print("\n✅ All figures exported to reports/figures/")


if __name__ == "__main__":
    main()
