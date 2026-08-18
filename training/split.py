"""Shared train/validation/test split - used by every model (baseline, LSTM).

Kept separate from any one model's training code so the baseline (Phase 4)
and the LSTM (Phase 5) split the data identically: same rows in the same
sets, given the same input DataFrame and the same RANDOM_SEED. That's what
makes a later baseline-vs-LSTM comparison fair - both are evaluated on
exactly the same held-out test articles.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT


def stratified_three_way_split(
    df: pd.DataFrame, label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, validation, test), stratified on label_col.

    Uses TRAIN_SPLIT/VAL_SPLIT/TEST_SPLIT and RANDOM_SEED from
    config/settings.py (70/15/15 by default). Stratification keeps the
    fake/real ratio consistent across all three sets - without it, a random
    split could accidentally put proportionally more of one class into the
    test set, making test-set metrics noisier and harder to trust.
    """
    train_df, remaining_df = train_test_split(
        df,
        train_size=TRAIN_SPLIT,
        stratify=df[label_col],
        random_state=RANDOM_SEED,
    )

    # remaining_df holds VAL_SPLIT + TEST_SPLIT; split it proportionally
    # between the two so the final three-way ratio matches the config exactly.
    relative_val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=relative_val_size,
        stratify=remaining_df[label_col],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
