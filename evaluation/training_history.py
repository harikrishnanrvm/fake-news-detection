"""Training-history plotting - the one genuinely deep-learning-specific piece
of evaluation code in this project (the baseline has no "epochs," so nothing
in evaluation/metrics.py covers this). Everything else the LSTM needs
(accuracy/precision/recall/F1, confusion matrix, ROC curve, error analysis,
experiment logging) reuses the existing model-agnostic evaluation/ modules
unchanged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history, output_path: Path) -> None:
    """Save side-by-side accuracy and loss curves (train vs. validation) per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history.history["accuracy"], label="Train", color="#2980b9")
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="#c0392b")
    axes[0].set_title("Accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train", color="#2980b9")
    axes[1].plot(history.history["val_loss"], label="Validation", color="#c0392b")
    axes[1].set_title("Loss per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (binary crossentropy)")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
