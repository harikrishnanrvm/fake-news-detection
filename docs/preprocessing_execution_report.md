# Preprocessing Pipeline — Execution Report

This document reports what actually happened when the staged preprocessing pipeline
(`preprocessing/` package, run from `notebooks/02_preprocessing.ipynb`) was executed against
the real dataset. Every number below comes directly from `report/tables/preprocessing_pipeline_stats.csv`
(and the per-stage `stage1_stats.csv` / `stage2_stats.csv` / `stage3_stats.csv`), not from
the design estimates in `docs/data_cleaning_strategy.md` — where the two differ slightly, it's
because pipeline **order** changes the exact population each step runs against (explained below).

This is the execution record; the design reasoning behind *why* each step exists is in
`docs/preprocessing_plan.md`, `docs/data_cleaning_strategy.md`, and `docs/label_leakage_analysis.md`.

---

## Pipeline overview

```
dataset/raw/{Fake.csv, True.csv}   (23,481 + 21,417 rows)
        ↓  Stage 1 — combine only
dataset/processed/01_combined_raw.csv        (44,898 rows, 5 columns)
        ↓  Stage 2 — cleaning only
dataset/processed/02_cleaned.csv             (38,638 rows, 3 columns)
        ↓  Stage 3 — NLP preprocessing only
dataset/processed/03_preprocessed.csv        (38,638 rows, 6 columns)
```

**Net result: 44,898 → 38,638 rows retained (86.06%). No rows were removed by Stage 3** — all
row removal happened during Stage 2 cleaning; Stage 3 only transforms text and adds columns.

---

## Stage 1 — Combine

| Step | Rows before | Rows after | Rows removed | Columns before → after |
|---|---|---|---|---|
| `combine_fake_and_true` | 44,898 | 44,898 | 0 | 4 → 5 |

**What happened:** `Fake.csv` tagged `label=0`, `True.csv` tagged `label=1`, concatenated,
shuffled with the project's fixed `RANDOM_SEED`. No rows removed by definition — this stage's
entire job is combination, not judgment.

---

## Stage 2 — Clean

| Step | Rows before | Rows after | Rows removed | Reason |
|---|---|---|---|---|
| `remove_corrupted_rows` | 44,898 | 44,888 | 10 | Non-standard `date` value, confirmed by inspection to contain no real article (bare URLs / a leaked page-builder template) |
| `remove_empty_articles` | 44,888 | 44,257 | 631 | Empty or whitespace-only `text` field |
| `deduplicate_articles` | 44,257 | 38,638 | 5,619 | Duplicate `text`, keeping first occurrence |
| `strip_reuters_prefix` | 38,638 | 38,638 | 0 (20,887 rows text-modified) | Removed leading Reuters wire-service dateline |
| `drop_leakage_columns` | 38,638 | 38,638 | 0 (columns 5 → 3) | Dropped `subject` and `date` |

**Note on the deduplication count:** `docs/duplicate_analysis.md` estimated **6,252** rows would
be removed by text-based deduplication, computed directly on the raw 44,898-row combined
dataset. The actual pipeline removes **5,619** at this step, because by the time deduplication
runs, `remove_corrupted_rows` and `remove_empty_articles` have already run first and removed
641 rows — some of which were themselves members of what would otherwise have been counted as
a duplicate-text group. This is expected and correct: **pipeline order changes exact counts,
even when the conceptual step is the same.** (5,619 + 641 = 6,260, close to the original 6,252
estimate — the small remaining difference comes from duplicate groups where more than one
member was also empty/corrupted.) This exact discrepancy is worth stating plainly in the
report and being ready to explain in the viva: it demonstrates the pipeline was actually run
and measured, not just estimated once and assumed.

**Note on `strip_reuters_prefix`:** this step modifies text in 20,887 rows without removing
any rows — slightly more than the 21,247 raw "(Reuters)"-containing count from
`docs/label_leakage_analysis.md`, because by this point in the pipeline some of those rows
have already been removed by the earlier duplicate/empty/corrupted-row steps.

---

## Stage 3 — NLP Preprocessing

| Step | Rows before | Rows after | Rows removed | Columns before → after |
|---|---|---|---|---|
| `combine_title_and_text` | 38,638 | 38,638 | 0 | 3 → 4 |
| `clean_text_baseline` | 38,638 | 38,638 | 0 | 4 → 5 |
| `clean_text_lstm` | 38,638 | 38,638 | 0 | 5 → 6 |
| `remove_rows_emptied_by_cleaning` | 38,638 | 38,638 | 0 | 6 → 6 |

**What happened:** `title` + `text` combined into `content`; two independently-cleaned text
columns produced from it — `baseline_text` (stop-word removal + lemmatization applied) and
`lstm_text` (both skipped, to preserve word order and negation) — per the split decision in
`docs/preprocessing_plan.md`. **Zero rows became empty purely from NLP cleaning** — every
article that survived Stage 2 had enough real content to survive Stage 3 too.

**Example (real output, row 0 of the final dataset):**

- `content` (combined, uncleaned): *"Ben Stein Calls Out 9th Circuit Court: Committed a "Coup d'état" Against the Constitution 21st Century Wire says Ben Stein, reputable professor from..."*
- `baseline_text`: *"ben stein call 9th circuit court committed coup état constitution 21st century wire say ben stein reputable professor pepperdine university also holl..."*
- `lstm_text`: *"ben stein calls out 9th circuit court committed a coup d état against the constitution 21st century wire says ben stein reputable professor from peppe..."*

Notice `baseline_text` dropped short function words (`"out"`, `"a"`) and reduced inflected
words to base forms (`"calls"` → `"call"`, `"says"` → `"say"`); `lstm_text` kept every word
in its original form and order — the design decision from `preprocessing_plan.md`, now
visible in real output.

---

## Column evolution, end to end

| Stage | Columns |
|---|---|
| Raw (`Fake.csv`/`True.csv`) | `title`, `text`, `subject`, `date` |
| Stage 1 output | `title`, `text`, `subject`, `date`, `label` |
| Stage 2 output | `title`, `text`, `label` |
| Stage 3 output | `title`, `text`, `label`, `content`, `baseline_text`, `lstm_text` |

---

## Educational summary (why / what-if-skipped / baseline vs. LSTM)

A condensed version of the full explanations already written into `notebooks/02_preprocessing.ipynb`'s
markdown cells (which are the canonical, more detailed version — this table is for quick
report/viva reference):

| Step | Why necessary | If skipped | Baseline vs. LSTM |
|---|---|---|---|
| Remove corrupted rows | 10 rows have no real article content | Meaningless rows add noise | Identical |
| Remove empty articles | 631 rows have no body text | Model learns from nothing for these rows | Identical |
| Deduplicate articles | Prevents train/test leakage via memorized duplicates | Inflated, untrustworthy accuracy | Identical |
| Strip Reuters prefix | 99.6%-accurate shortcut unrelated to truthfulness | Model "cheats" instead of learning language | Identical (arguably worse for baseline, which is best at exploiting single-token shortcuts) |
| Drop `subject`/`date` | Perfect / near-perfect label leakage | Model reads metadata, ignores article text | Identical |
| Combine title + text | Captures real title-length/style signal found in EDA | Title signal discarded | Identical |
| Lowercase | Prevents case-variant vocabulary splitting | Larger, sparser vocabulary | Identical (both lose the ALL-CAPS signal) |
| HTML removal | 34–146 rows affected, one-sided | Garbage tokens in vocabulary | Identical |
| URL removal | 14% of Fake rows affected — itself a shortcut | Vocabulary bloat + a new shortcut | Identical |
| Punctuation removal | Punctuation tokens carry little standalone meaning | Larger vocabulary, word-boundary noise | Identical (loses sensational-punctuation signal) |
| **Stop-word removal** | Removes noise for TF-IDF | Diluted bag-of-words features | **Different** — on for baseline, off for LSTM (negation/context) |
| **Lemmatization** | Groups inflected forms for TF-IDF | Related words treated as unrelated | **Different** — on for baseline, off for LSTM (embeddings can learn this; added complexity otherwise) |
| Whitespace normalization | Collapses repeated whitespace | Spurious empty tokens | Identical |
| Number removal (not used) | Numbers can be informative; `VOCAB_SIZE`/`OOV_TOKEN` already caps rare tokens | N/A — deliberately skipped | Identical |
| Stemming (not used) | Redundant with lemmatization, produces non-words | N/A — deliberately skipped | Identical |

---

## Reproducibility note

Every step above is implemented as a pure, reusable function in `preprocessing/` (not
notebook-only code), and the whole pipeline can be re-run end to end with:

```python
from preprocessing.pipeline import run_full_pipeline
final_df = run_full_pipeline()
```

Re-running produces the same 38,638-row output every time, because the only randomized step
(the Stage 1 shuffle) uses the project's fixed `RANDOM_SEED` from `config/settings.py`.

## What this report deliberately does not cover

No train/validation/test split, no tokenizer fitting, no model training. That begins in
Phase 4 (baseline) and Phase 5 (LSTM), both of which consume `dataset/processed/03_preprocessed.csv`
directly.
