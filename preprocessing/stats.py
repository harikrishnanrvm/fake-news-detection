"""Bookkeeping helpers for tracking row/column counts through the preprocessing pipeline.

Every pipeline stage (see combine.py, cleaning.py, text_cleaning.py, pipeline.py)
reports what it did through this module, so the numbers in the report and the
docs are read directly from what the code actually did, not typed in by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class StepRecord:
    """One bookkeeping row describing the effect of a single pipeline step."""

    stage: str
    step: str
    rows_before: int
    rows_after: int
    reason: str
    columns_before: int
    columns_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def record_step(
    records: list[StepRecord],
    stage: str,
    step: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    reason: str,
) -> None:
    """Append one bookkeeping row describing the effect of a single pipeline step."""
    records.append(
        StepRecord(
            stage=stage,
            step=step,
            rows_before=len(df_before),
            rows_after=len(df_after),
            reason=reason,
            columns_before=df_before.shape[1],
            columns_after=df_after.shape[1],
        )
    )


def stats_to_dataframe(records: list[StepRecord]) -> pd.DataFrame:
    """Convert accumulated step records into a flat DataFrame for saving/inspection."""
    return pd.DataFrame(
        [
            {
                "stage": r.stage,
                "step": r.step,
                "rows_before": r.rows_before,
                "rows_after": r.rows_after,
                "rows_removed": r.rows_removed,
                "reason": r.reason,
                "columns_before": r.columns_before,
                "columns_after": r.columns_after,
            }
            for r in records
        ]
    )


def save_stats(records: list[StepRecord], output_path: Path) -> pd.DataFrame:
    """Write accumulated step statistics to a CSV and return them as a DataFrame."""
    stats_df = stats_to_dataframe(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    return stats_df
