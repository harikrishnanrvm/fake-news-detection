"""Loads the trained LSTM model, tokenizer, and label map once at startup.

Loading a Keras model from disk is slow enough (a few seconds) that doing it
on every request would make the API unusable. Instead, api/app.py calls
load_artifacts() exactly once, when the server starts, and every /predict
request afterward reuses the same in-memory model and tokenizer via the
get_* functions below.

This module only ever *reads* models/lstm/ - it never trains or modifies
anything there, per Phase 7's "do not retrain" constraint.
"""
from __future__ import annotations

import json

import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from config.settings import LSTM_MODEL_DIR

_model: tf.keras.Model | None = None
_tokenizer: Tokenizer | None = None
_label_map: dict[str, int] | None = None


def load_artifacts() -> None:
    """Load the LSTM model, tokenizer, and label map into memory.

    Called once from api/app.py's startup event, before the app accepts
    any requests.
    """
    global _model, _tokenizer, _label_map

    _model = tf.keras.models.load_model(LSTM_MODEL_DIR / "model.keras")
    _tokenizer = joblib.load(LSTM_MODEL_DIR / "tokenizer.pkl")
    with open(LSTM_MODEL_DIR / "label_map.json", encoding="utf-8") as f:
        _label_map = json.load(f)


def get_model() -> tf.keras.Model:
    """Return the already-loaded LSTM model.

    Raises RuntimeError if called before load_artifacts() - this should never
    happen in normal operation, since the startup event runs first.
    """
    if _model is None:
        raise RuntimeError("LSTM model not loaded - load_artifacts() must run at startup.")
    return _model


def get_tokenizer() -> Tokenizer:
    """Return the already-loaded Keras Tokenizer."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded - load_artifacts() must run at startup.")
    return _tokenizer


def get_label_map() -> dict[str, int]:
    """Return the already-loaded label map, e.g. {"fake": 0, "real": 1}."""
    if _label_map is None:
        raise RuntimeError("Label map not loaded - load_artifacts() must run at startup.")
    return _label_map


def is_ready() -> bool:
    """True once all three artifacts have been loaded - used by /health."""
    return _model is not None and _tokenizer is not None and _label_map is not None
