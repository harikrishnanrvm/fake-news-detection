"""Top-level orchestration for the three-stage preprocessing pipeline.

Stage 1 (combine)    -> dataset/processed/01_combined_raw.csv
Stage 2 (clean)      -> dataset/processed/02_cleaned.csv
Stage 3 (preprocess) -> dataset/processed/03_preprocessed.csv

Each stage function returns (dataframe, stats_records) so a caller (the
notebook, or run_full_pipeline() below) can inspect exactly what happened.
This is where the important logic lives - notebooks/02_preprocessing.ipynb
only calls these functions and narrates the results, per the project specification's
Notebook Philosophy ("never leave important logic only inside notebooks").
"""
from __future__ import annotations

import pandas as pd

from config.settings import (
    DATASET_RAW_DIR,
    REPORT_TABLES_DIR,
    STAGE1_COMBINED_FILE,
    STAGE2_CLEANED_FILE,
    STAGE3_PREPROCESSED_FILE,
)
from preprocessing import cleaning, combine, text_cleaning
from preprocessing.stats import StepRecord, record_step, save_stats
from utils.logger import get_logger

logger = get_logger(__name__)


def run_stage1() -> tuple[pd.DataFrame, list[StepRecord]]:
    """Stage 1: combine Fake.csv + True.csv into one labeled DataFrame. No cleaning."""
    records: list[StepRecord] = []

    fake_raw = combine.load_raw_csv(DATASET_RAW_DIR / "Fake.csv")
    true_raw = combine.load_raw_csv(DATASET_RAW_DIR / "True.csv")
    raw_concat = pd.concat([fake_raw, true_raw], ignore_index=True)

    fake_labeled = combine.assign_label(fake_raw, label=0)
    true_labeled = combine.assign_label(true_raw, label=1)
    combined_df = combine.combine_datasets(fake_labeled, true_labeled)

    record_step(
        records, stage="stage1_combine", step="combine_fake_and_true",
        df_before=raw_concat, df_after=combined_df,
        reason="Concatenated Fake.csv (label=0) and True.csv (label=1) and added a label column; no rows removed, no cleaning applied",
    )

    STAGE1_COMBINED_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(STAGE1_COMBINED_FILE, index=False)
    save_stats(records, REPORT_TABLES_DIR / "stage1_stats.csv")
    logger.info("Stage 1 complete: %d rows -> %s", len(combined_df), STAGE1_COMBINED_FILE)
    return combined_df, records


def run_stage2(df_stage1: pd.DataFrame) -> tuple[pd.DataFrame, list[StepRecord]]:
    """Stage 2: cleaning only. No NLP preprocessing (see run_stage3)."""
    records: list[StepRecord] = []
    df = df_stage1

    before = df
    df = cleaning.remove_corrupted_rows(df)
    record_step(
        records, "stage2_clean", "remove_corrupted_rows", before, df,
        "Rows with a non-standard `date` value whose title/text were manually confirmed "
        "to contain no real article (bare image/article URLs, or a leaked page-builder "
        "template) - see docs/data_cleaning_strategy.md",
    )

    before = df
    df = cleaning.remove_empty_articles(df)
    record_step(
        records, "stage2_clean", "remove_empty_articles", before, df,
        "Rows where the `text` field is empty or whitespace-only",
    )

    before = df
    df = cleaning.deduplicate_articles(df)
    record_step(
        records, "stage2_clean", "deduplicate_articles", before, df,
        "Duplicate `text` values, keeping the first occurrence - see docs/duplicate_analysis.md",
    )

    before = df
    reuters_modified = (
        before["text"].apply(text_cleaning.strip_reuters_prefix) != before["text"]
    ).sum()
    df = cleaning.strip_reuters_from_column(df)
    record_step(
        records, "stage2_clean", "strip_reuters_prefix", before, df,
        f"Removed a leading Reuters wire-service dateline from {reuters_modified} rows "
        "(text modified in place, no rows removed) - see docs/label_leakage_analysis.md",
    )

    before = df
    df = cleaning.drop_leakage_columns(df)
    record_step(
        records, "stage2_clean", "drop_leakage_columns", before, df,
        "Dropped `subject` and `date` columns (label leakage) - see docs/label_leakage_analysis.md",
    )

    STAGE2_CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGE2_CLEANED_FILE, index=False)
    save_stats(records, REPORT_TABLES_DIR / "stage2_stats.csv")
    logger.info("Stage 2 complete: %d rows -> %s", len(df), STAGE2_CLEANED_FILE)
    return df, records


def run_stage3(df_stage2: pd.DataFrame) -> tuple[pd.DataFrame, list[StepRecord]]:
    """Stage 3: NLP preprocessing, applied only after Stage 2.

    Produces two cleaned text columns from the same combined content:
    `baseline_text` (stop-word removal + lemmatization ON, for TF-IDF/Logistic
    Regression) and `lstm_text` (both OFF, to preserve word order/negation for
    the LSTM) - the split recommendation from docs/preprocessing_plan.md.
    """
    records: list[StepRecord] = []
    df = df_stage2.copy()

    before = df.copy()
    df["content"] = df.apply(
        lambda r: text_cleaning.combine_title_and_text(r["title"], r["text"]), axis=1
    )
    record_step(
        records, "stage3_preprocess", "combine_title_and_text", before, df,
        "Concatenated title + text into a single `content` field; no rows removed",
    )

    before = df.copy()
    df["baseline_text"] = df["content"].apply(
        lambda t: text_cleaning.clean_text(t, text_cleaning.BASELINE_STEPS)
    )
    record_step(
        records, "stage3_preprocess", "clean_text_baseline", before, df,
        "Baseline NLP cleaning: lowercase, HTML/URL/punctuation removal, stop-word removal, "
        "lemmatization - see docs/preprocessing_plan.md",
    )

    before = df.copy()
    df["lstm_text"] = df["content"].apply(
        lambda t: text_cleaning.clean_text(t, text_cleaning.LSTM_STEPS)
    )
    record_step(
        records, "stage3_preprocess", "clean_text_lstm", before, df,
        "LSTM NLP cleaning: same as baseline but WITHOUT stop-word removal or lemmatization, "
        "to preserve word order and negation - see docs/preprocessing_plan.md",
    )

    before = df
    empty_after_cleaning = (df["baseline_text"].str.strip() == "") | (df["lstm_text"].str.strip() == "")
    df = df.loc[~empty_after_cleaning].reset_index(drop=True)
    record_step(
        records, "stage3_preprocess", "remove_rows_emptied_by_cleaning", before, df,
        "Dropped rows that became empty after NLP cleaning (e.g. an article consisting only "
        "of stop words/punctuation/URLs)",
    )

    STAGE3_PREPROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGE3_PREPROCESSED_FILE, index=False)
    save_stats(records, REPORT_TABLES_DIR / "stage3_stats.csv")
    logger.info("Stage 3 complete: %d rows -> %s", len(df), STAGE3_PREPROCESSED_FILE)
    return df, records


def run_full_pipeline() -> pd.DataFrame:
    """Run Stage 1 -> Stage 2 -> Stage 3 end-to-end.

    Saves every staged CSV plus one combined statistics CSV
    (report/tables/preprocessing_pipeline_stats.csv). Returns the final
    Stage 3 DataFrame.
    """
    all_records: list[StepRecord] = []

    stage1_df, stage1_records = run_stage1()
    all_records += stage1_records

    stage2_df, stage2_records = run_stage2(stage1_df)
    all_records += stage2_records

    stage3_df, stage3_records = run_stage3(stage2_df)
    all_records += stage3_records

    save_stats(all_records, REPORT_TABLES_DIR / "preprocessing_pipeline_stats.csv")
    logger.info("Full pipeline complete: %d rows in final Stage 3 output", len(stage3_df))
    return stage3_df
