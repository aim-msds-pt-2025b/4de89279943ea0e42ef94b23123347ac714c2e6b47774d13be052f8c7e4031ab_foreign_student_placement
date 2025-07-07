import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_models(models_dict, X_test, y_test):
    """
    Returns a DataFrame of metrics for each model.
    """
    rows = []
    for name, mdl in models_dict.items():
        y_pred = mdl.predict(X_test)
        proba = (
            mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba") else None
        )

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None,
            }
        )

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
