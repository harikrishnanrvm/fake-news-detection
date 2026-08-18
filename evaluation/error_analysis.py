"""Helpers for inspecting individual misclassified examples.

Aggregate metrics (accuracy, F1, ...) say *how often* a model is wrong;
error analysis looks at *which* articles it gets wrong and why - the part
that actually informs what the LSTM (Phase 5) needs to do differently.
"""
from __future__ import annotations

import pandas as pd


def get_false_positives(test_df: pd.DataFrame, y_true, y_pred, n: int = 5) -> pd.DataFrame:
    """Rows the model predicted Real (1) that are actually Fake (0).

    A false positive here means: a fake article that fooled the model into
    predicting "real."
    """
    mask = (y_true == 0) & (y_pred == 1)
    return test_df.loc[mask].head(n)


def get_false_negatives(test_df: pd.DataFrame, y_true, y_pred, n: int = 5) -> pd.DataFrame:
    """Rows the model predicted Fake (0) that are actually Real (1).

    A false negative here means: a genuine article the model wrongly
    flagged as fake.
    """
    mask = (y_true == 1) & (y_pred == 0)
    return test_df.loc[mask].head(n)
