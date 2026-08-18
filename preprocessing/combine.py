"""Stage 1: combine Fake.csv and True.csv into a single labeled DataFrame.

Single responsibility: load the two raw files, tag each with a label, and
concatenate them. No cleaning and no text preprocessing happens here - see
cleaning.py (Stage 2) and text_cleaning.py (Stage 3) for those.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import RANDOM_SEED


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load one of the original Kaggle CSVs exactly as downloaded, unmodified."""
    return pd.read_csv(path)


def assign_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Return a copy of df tagged with a constant label column (0=fake, 1=real)."""
    labeled = df.copy()
    labeled["label"] = label
    return labeled


def combine_datasets(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate labeled Fake/Real DataFrames and shuffle reproducibly.

    Shuffling avoids the two classes sitting in two contiguous blocks, which
    would otherwise bias any downstream head()/sample() inspection. The fixed
    RANDOM_SEED keeps this step reproducible run to run.
    """
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    return combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
