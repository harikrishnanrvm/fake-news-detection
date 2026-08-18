"""Phase 4 baseline: TF-IDF + Logistic Regression.

Single responsibility per function - load data, vectorize, train, save. The
notebook (notebooks/03_baseline_model.ipynb) calls these; the actual logic
lives here so it can be reused (e.g. by the FastAPI app in Phase 7) without
retraining or duplicating code.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config.settings import (
    LOGISTIC_REGRESSION_MAX_ITER,
    RANDOM_SEED,
    STAGE3_PREPROCESSED_FILE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

LABEL_MAP = {"fake": 0, "real": 1}


def load_baseline_dataset(path: Path = STAGE3_PREPROCESSED_FILE) -> pd.DataFrame:
    """Load Stage 3 output and keep only what the baseline model needs.

    Deliberately drops `lstm_text` here (not just "ignores" it) so it's
    structurally impossible for baseline training code to accidentally read
    the LSTM-specific column.
    """
    df = pd.read_csv(path)
    return df[["title", "text", "label", "baseline_text"]].copy()


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create a TfidfVectorizer using the project's configured vocabulary cap.

    No stop_words='english' here - baseline_text already had stop words
    removed in Stage 3 (see docs/preprocessing_plan.md); repeating it would
    be redundant. Unigrams only (TFIDF_NGRAM_RANGE = (1, 1)) keeps the
    "top words per class" feature-importance section simple to read and
    explain in a viva.
    """
    return TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Fit a Logistic Regression classifier with fixed, simple hyperparameters.

    No hyperparameter search (e.g. grid search over C) - this is intended as
    a baseline, not a fully-tuned model; a fixed, documented configuration is
    easier to reproduce and to explain than a search whose result depends on
    what grid was chosen.
    """
    model = LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def save_baseline_artifacts(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    output_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the fitted vectorizer, model, label map, and metadata.

    Saved as three separate files so the FastAPI app (Phase 7) can load
    exactly what it needs (vectorizer + model) without retraining, per
    the project specification's Model Artifacts standard.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(model, output_dir / "logistic_regression.pkl")

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    metadata = {
        "algorithm": "TF-IDF + Logistic Regression",
        "tfidf_params": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "vocabulary_size_actual": len(vectorizer.vocabulary_),
        },
        "logistic_regression_params": {
            "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
            "random_state": RANDOM_SEED,
            "C": model.C,
            "solver": model.solver,
        },
        "random_seed": RANDOM_SEED,
        "dataset_source": str(STAGE3_PREPROCESSED_FILE),
        "text_column": "baseline_text",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
