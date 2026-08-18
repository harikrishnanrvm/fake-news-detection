"""Model-agnostic evaluation helpers - usable by the baseline (Phase 4) and
later the LSTM (Phase 5), so both are scored and plotted the same way and
are genuinely comparable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

LABELS = ["Fake", "Real"]


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 (binary, positive class = Real/1)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def save_classification_report(y_true, y_pred, output_path: Path) -> pd.DataFrame:
    """Save sklearn's per-class classification report as a CSV table."""
    report_dict = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    return report_df


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    """Save a confusion matrix plot (raw counts, Fake/Real labels)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title, fontsize=11, wrap=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: Path, title: str) -> float:
    """Save an ROC curve plot and return the AUC score.

    y_proba must be the predicted probability of the positive class (Real/1).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#2980b9")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return roc_auc
