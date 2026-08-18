"""Prediction agreement analysis for two classifiers scored on the same test set.

Complements evaluation/significance.py (McNemar's test): McNemar's test answers
"is one model statistically better?", but it collapses everything into just
the disagreement counts. This module answers a different, complementary
question - "do the two models fail on the *same* articles, or different
ones?" - which matters for Phase 6's discussion of whether an ensemble or a
model swap would actually help.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_agreement_counts(y_true, y_pred_a, y_pred_b) -> dict[str, int]:
    """Split test rows into 4 buckets: both correct, both wrong, and each
    model's exclusive wins.

    "Exclusive win" for model A means: A got this row right and B did not -
    exactly the rows that would flip if you swapped B for A in production.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    return {
        "both_correct": int(np.sum(a_correct & b_correct)),
        "both_wrong": int(np.sum(~a_correct & ~b_correct)),
        "a_only_correct": int(np.sum(a_correct & ~b_correct)),
        "b_only_correct": int(np.sum(~a_correct & b_correct)),
    }


def agreement_counts_to_dataframe(
    counts: dict[str, int], label_a: str, label_b: str
) -> pd.DataFrame:
    """Turn the raw counts dict into a report-ready table with percentages."""
    total = sum(counts.values())
    rows = [
        ("Both models correct", counts["both_correct"]),
        ("Both models wrong", counts["both_wrong"]),
        (f"Only {label_a} correct", counts["a_only_correct"]),
        (f"Only {label_b} correct", counts["b_only_correct"]),
    ]
    df = pd.DataFrame(rows, columns=["category", "count"])
    df["percentage"] = (df["count"] / total * 100).round(2)
    return df


def plot_agreement_bar(
    counts: dict[str, int], label_a: str, label_b: str, output_path: Path
) -> None:
    """Save a simple bar chart of the 4 agreement categories.

    A plain bar chart (rather than a proportional Venn diagram, which would
    need the extra matplotlib-venn dependency) keeps this readable without
    adding a new library for a one-off figure.
    """
    categories = [
        "Both\ncorrect",
        "Both\nwrong",
        f"Only {label_a}\ncorrect",
        f"Only {label_b}\ncorrect",
    ]
    values = [
        counts["both_correct"],
        counts["both_wrong"],
        counts["a_only_correct"],
        counts["b_only_correct"],
    ]
    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Number of test articles")
    ax.set_title(f"Prediction agreement: {label_a} vs. {label_b} (test set)")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
