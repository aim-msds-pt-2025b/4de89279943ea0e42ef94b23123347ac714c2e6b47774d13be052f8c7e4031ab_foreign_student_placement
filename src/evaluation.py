import os
import json
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from pathlib import Path


def eval_models(models: dict, X_test, y_test):
    """Simple evaluation function for run_pipeline_new.py"""
    rows = []
    for name, m in models.items():
        y_pred = m.predict(X_test)
        auc = float("nan")
        if hasattr(m, "predict_proba"):
            try:
                y_prob = m.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                pass
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def confusion(m, X, y):
    """Simple confusion matrix helper for run_pipeline_new.py"""
    return confusion_matrix(y, m.predict(X))


def evaluate_models(models_dict, X_test, y_test):
    """
    Returns a DataFrame of metrics for each model with MLflow logging.
    """
    # Set MLflow tracking URI to local directory for testing
    mlflow.set_tracking_uri("file:./mlruns")

    rows = []
    for name, mdl in models_dict.items():
        with mlflow.start_run(run_name=f"evaluate_{name}"):
            y_pred = mdl.predict(X_test)
            proba = (
                mdl.predict_proba(X_test)[:, 1]
                if hasattr(mdl, "predict_proba")
                else None
            )

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, proba) if proba is not None else None

            # Log exactly 2 evaluation metrics for classification
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            rows.append(
                {
                    "model": name,
                    "accuracy": accuracy,
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                }
            )

    # Save evaluation results to reports/evaluation_results.json
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    evaluation_results = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "models_evaluated": len(rows),
        "results": rows,
    }

    with open(reports_dir / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    return pd.DataFrame(rows).set_index("model")


def select_and_save_best(
    metrics_df, models_dict, X_test, y_test, metrics_txt_path="reports/metrics.txt"
):
    """
    Picks the best model by F1, writes out metrics.txt, and returns its name & confusion matrix.
    """
    best_name = metrics_df["f1_score"].idxmax()
    best_model = models_dict[best_name]

    cm = confusion_matrix(y_test, best_model.predict(X_test))

    os.makedirs(os.path.dirname(metrics_txt_path), exist_ok=True)
    with open(metrics_txt_path, "w") as f:
        f.write(f"Best model: {best_name}\n")
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            val = metrics_df.at[best_name, metric]
            line = (
                f"{metric.capitalize()}: {val:.4f}\n"
                if pd.notna(val)
                else f"{metric.capitalize()}: N/A\n"
            )
            f.write(line)
        f.write("Confusion matrix:\n")
        f.write(str(cm.tolist()))

    return best_name, cm
