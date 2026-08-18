"""Stage 2: data cleaning only - no NLP preprocessing here (see text_cleaning.py).

Single responsibility per function: detect/remove corrupted rows, remove
duplicate articles, remove empty article bodies, drop the leakage columns
identified in docs/label_leakage_analysis.md, and strip the Reuters dateline
from `text`. title and text are kept as separate columns - combining them is
a Stage 3 (NLP preprocessing) concern, see text_cleaning.combine_title_and_text.
"""
from __future__ import annotations

import re

import pandas as pd

from preprocessing.text_cleaning import strip_reuters_prefix

# The standard format ("December 31, 2017") and the one known valid alternate
# format ("19-Feb-18") found during EDA (docs/data_dictionary.md). Anything
# matching neither was manually inspected and confirmed to be genuinely
# corrupted (bare image/article URLs, or a leaked page-builder template in
# place of an article) - see docs/data_cleaning_strategy.md.
_STANDARD_DATE_PATTERN = re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}\s*$")
_ALT_DATE_PATTERN = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")

_LEAKAGE_COLUMNS = ["subject", "date"]


def detect_corrupted_rows(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Flag rows whose `date` field is neither the standard nor known alt format.

    Must run before drop_leakage_columns() removes `date` - this check
    depends on that column still being present.
    """
    date_values = df[date_col].astype(str)
    is_standard = date_values.str.match(_STANDARD_DATE_PATTERN)
    is_alt_format = date_values.str.match(_ALT_DATE_PATTERN)
    return ~is_standard & ~is_alt_format


def remove_corrupted_rows(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Drop rows flagged by detect_corrupted_rows()."""
    corrupted = detect_corrupted_rows(df, date_col=date_col)
    return df.loc[~corrupted].reset_index(drop=True)


def remove_empty_articles(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop rows where the article body is empty (or only whitespace)."""
    is_empty = df[text_col].astype(str).str.strip() == ""
    return df.loc[~is_empty].reset_index(drop=True)


def deduplicate_articles(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop duplicate articles, keeping the first occurrence of each body text.

    Deliberately keyed on `text` alone, not the full row - see
    docs/duplicate_analysis.md for why this single key was chosen over
    trying to separately handle every duplicate category.
    """
    return df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)


def drop_leakage_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Drop the label-leakage columns identified in docs/label_leakage_analysis.md."""
    columns = columns if columns is not None else _LEAKAGE_COLUMNS
    return df.drop(columns=columns)


def strip_reuters_from_column(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply strip_reuters_prefix() to every row of the given text column."""
    df = df.copy()
    df[text_col] = df[text_col].apply(strip_reuters_prefix)
    return df
