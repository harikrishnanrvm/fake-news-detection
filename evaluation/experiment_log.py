"""Lightweight experiment tracking - a single append-only CSV.

See the project specification's Experiment Tracking section for why this is a CSV and not
MLflow/W&B: a solo project with no team to coordinate experiments across
doesn't need a tracking server - a plain CSV opens directly in pandas/Excel
and drops straight into a report comparison table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_COLUMNS = [
    "timestamp",
    "model",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "training_time_seconds",
    "vocabulary_size",
    "notes",
]


def log_experiment(csv_path: Path, record: dict[str, Any]) -> pd.DataFrame:
    """Append one experiment record to csv_path, creating the file if needed.

    `record` should contain (at least) the keys in _COLUMNS; missing keys
    are filled with pd.NA so the CSV's columns stay stable across models
    that don't report every field (e.g. the LSTM will add its own columns
    like epochs, which this function tolerates via a full outer join on
    columns rather than requiring an exact match).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([record])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False)
    return combined
