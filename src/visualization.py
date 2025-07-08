import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

_fig_dir = "reports/figures"


def _ensure_dir():
    os.makedirs(_fig_dir, exist_ok=True)
    return _fig_dir


def plot_target_distribution(df: pd.DataFrame):
    """
    Bar-plot of how many students were placed vs not placed.
    """
    _ensure_dir()
    fig, ax = plt.subplots()
    sns.countplot(x="placement_status", data=df, ax=ax)
    ax.set_title("Placement Status Distribution")
    ax.set_xlabel("Placed (1) vs Not Placed (0)")
    fig.savefig(os.path.join(_fig_dir, "target_distribution.png"))
    plt.close(fig)


def plot_feature_correlations(df: pd.DataFrame, numeric_cols: list[str]):
    """
    Heatmap of Pearson correlations between numeric features.
    """
    _ensure_dir()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.savefig(os.path.join(_fig_dir, "feature_correlations.png"))
    plt.close(fig)


def plot_roc_curves(models: dict[str, any], X_test: pd.DataFrame, y_test: pd.Series):
    """
    Overlaid ROC curves for any model that supports predict_proba().
    """
    _ensure_dir()
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(_fig_dir, "roc_curves.png"))
    plt.close(fig)


def plot_confusion_matrix(
    model: any, X_test: pd.DataFrame, y_test: pd.Series, name: str = "best_model"
):
    """
    Simple heatmap of the confusion matrix for a single model.
    """
    _ensure_dir()
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {name}")
    fig.savefig(os.path.join(_fig_dir, f"confusion_matrix_{name}.png"))
    plt.close(fig)
