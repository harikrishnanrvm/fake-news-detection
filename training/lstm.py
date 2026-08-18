"""Phase 5: LSTM text classifier.

Single responsibility per function - tokenize, pad, build, train, save. The
notebook (notebooks/04_lstm_training.ipynb) calls these; the actual logic
lives here so it can be reused (e.g. by the FastAPI app in Phase 7) without
retraining, per the project specification's Notebook Philosophy.

Mirrors training/baseline.py's structure deliberately, so the two models'
code reads the same way despite using different libraries.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config.settings import (
    DENSE_UNITS,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    LSTM_BATCH_SIZE,
    LSTM_MAX_EPOCHS,
    LSTM_UNITS,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    RANDOM_SEED,
    SPATIAL_DROPOUT_RATE,
    STAGE3_PREPROCESSED_FILE,
    VOCAB_SIZE,
)
from training.baseline import LABEL_MAP


def set_random_seeds() -> None:
    """Fix Python/NumPy/TensorFlow random seeds for reproducible training.

    Matches the RANDOM_SEED used everywhere else in the project. Note (as
    already stated in docs/preprocessing_plan.md): bit-exact determinism
    isn't guaranteed on every hardware/backend combination, but fixed seeds
    still make results reproducible "within small floating-point variance."
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def load_lstm_dataset(path: Path = STAGE3_PREPROCESSED_FILE) -> pd.DataFrame:
    """Load Stage 3 output and keep only what the LSTM needs.

    Deliberately drops `baseline_text` here (not just "ignores" it) - the
    same structural safeguard training/baseline.py uses for `lstm_text`, so
    it's impossible for this code to accidentally train on the wrong column.
    """
    df = pd.read_csv(path)
    return df[["title", "text", "label", "lstm_text"]].copy()


def build_tokenizer(texts) -> Tokenizer:
    """Fit a Keras Tokenizer on training texts only.

    Fitting only on the training set (never on validation/test) mirrors the
    same rule already used for the baseline's TfidfVectorizer
    (docs/preprocessing_plan.md) - fitting on the full dataset before
    splitting would leak validation/test vocabulary into the model's word
    index, a second, subtler form of train/test leakage.
    """
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer: Tokenizer, texts) -> np.ndarray:
    """Convert texts to integer sequences and pad/truncate to MAX_SEQUENCE_LENGTH.

    Padding/truncating "post" (at the end) rather than "pre" keeps the
    beginning of each article - where the most informative words typically
    appear in news writing - intact for both padding and truncation.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")


def build_lstm_model() -> Sequential:
    """Build the Sequential LSTM architecture.

    Embedding -> SpatialDropout1D -> LSTM -> Dropout -> Dense(ReLU) -> Dense(Sigmoid).
    Deliberately a single, unidirectional LSTM - no BiLSTM/attention/CNN-LSTM
    - kept simple and explainable, per this phase's scope.
    """
    model = Sequential([
        Input(shape=(MAX_SEQUENCE_LENGTH,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        SpatialDropout1D(SPATIAL_DROPOUT_RATE),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
):
    """Train with EarlyStopping (on val_loss) and ModelCheckpoint (best val_loss).

    EarlyStopping's restore_best_weights=True means training can run for up
    to LSTM_MAX_EPOCHS, but the model ends up holding the weights from
    whichever epoch had the lowest validation loss, not necessarily the last
    one - guards against overfitting in later epochs. ModelCheckpoint saves
    that same best-so-far model to disk as a safety net during training.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_MAX_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )


def save_lstm_artifacts(
    model: Sequential,
    tokenizer: Tokenizer,
    output_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the trained model, tokenizer, label map, and metadata.

    Saved as separate files, per the project specification's Model Artifacts standard, so
    the FastAPI app (Phase 7) can load exactly what it needs without
    retraining.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "model.keras")
    joblib.dump(tokenizer, output_dir / "tokenizer.pkl")

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    metadata = {
        "algorithm": "LSTM",
        "architecture": [
            f"Embedding(input_dim={VOCAB_SIZE}, output_dim={EMBEDDING_DIM})",
            f"SpatialDropout1D({SPATIAL_DROPOUT_RATE})",
            f"LSTM({LSTM_UNITS})",
            f"Dropout({DROPOUT_RATE})",
            f"Dense({DENSE_UNITS}, activation='relu')",
            "Dense(1, activation='sigmoid')",
        ],
        "tokenizer_params": {
            "vocab_size": VOCAB_SIZE,
            "oov_token": OOV_TOKEN,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "actual_vocabulary_size": len(tokenizer.word_index) + 1,
        },
        "training_params": {
            "batch_size": LSTM_BATCH_SIZE,
            "max_epochs": LSTM_MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
        },
        "random_seed": RANDOM_SEED,
        "dataset_source": str(STAGE3_PREPROCESSED_FILE),
        "text_column": "lstm_text",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
