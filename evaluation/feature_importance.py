"""Feature importance for the TF-IDF + Logistic Regression baseline.

Logistic Regression assigns one learned coefficient (weight) to every word
in the TF-IDF vocabulary. Because label=1 is "Real", a large *positive*
coefficient means that word's presence pushes the prediction toward Real;
a large *negative* coefficient pushes it toward Fake. This module just reads
those coefficients back out - no separate importance calculation is needed,
which is exactly why Logistic Regression is described as "interpretable" in
docs/preprocessing_plan.md and docs/baseline_model_report.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_top_features(
    vectorizer: TfidfVectorizer, model: LogisticRegression, n: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_fake_words, top_real_words) DataFrames, each with n rows."""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    order = coefficients.argsort()

    top_fake_idx = order[:n]  # most negative coefficients
    top_real_idx = order[-n:][::-1]  # most positive coefficients, largest first

    fake_df = pd.DataFrame(
        {"word": feature_names[top_fake_idx], "coefficient": coefficients[top_fake_idx]}
    )
    real_df = pd.DataFrame(
        {"word": feature_names[top_real_idx], "coefficient": coefficients[top_real_idx]}
    )
    return fake_df, real_df


def plot_top_features(fake_df: pd.DataFrame, real_df: pd.DataFrame, output_path: Path) -> None:
    """Save a side-by-side horizontal bar chart of the top Fake/Real words."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    axes[0].barh(fake_df["word"][::-1], fake_df["coefficient"][::-1], color="#c0392b")
    axes[0].set_title("Top words indicating Fake")
    axes[0].set_xlabel("Logistic Regression coefficient")

    axes[1].barh(real_df["word"][::-1], real_df["coefficient"][::-1], color="#2980b9")
    axes[1].set_title("Top words indicating Real")
    axes[1].set_xlabel("Logistic Regression coefficient")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
