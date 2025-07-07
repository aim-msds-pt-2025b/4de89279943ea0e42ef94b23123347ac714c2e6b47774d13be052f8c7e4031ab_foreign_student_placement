import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test, metrics_path: str = "../reports/metrics.txt"):
    """
    Compute & save classification metrics to a text file.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy:  {acc:.3f}\n")
        f.write(f"Precision: {prec:.3f}\n")
        f.write(f"Recall:    {rec:.3f}\n")
        f.write(f"F1 Score:  {f1:.3f}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
