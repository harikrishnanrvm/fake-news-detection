# Data Dictionary

**Dataset:** Kaggle — "Fake and Real News Dataset" (Clément Bisaillon)
`clmentbisaillon/fake-and-real-news-dataset`
**Source files:** `Fake.csv` (23,481 rows), `True.csv` (21,417 rows)
**Combined dataset:** 44,898 rows, produced in `notebooks/01_dataset_analysis.ipynb` and saved to `dataset/processed/combined_raw.csv`

This document records every column present in the raw data, plus the `label` column created during combination, along with the decision on whether each one is used by the final model. It is written to be included directly in the BCA final report (Section: Data Understanding / Data Dictionary).

---

## Column Reference

### `title`

| Field | Value |
|---|---|
| Data type | string (object) |
| Example value | `"Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing"` |
| Description | The article headline as scraped from the original source. |
| Missing values | 0 rows with a null title; 0 rows with an empty-string title, in both files. |
| Used by final model | **Yes** |
| Reason if discarded | N/A — retained. EDA (`01_dataset_analysis.ipynb`, Section 5) found a real style difference between classes (Fake titles average 14.7 words vs. Real's 10.0, with a much wider range), so title is combined with the article body as model input rather than discarded (see [`data_cleaning_strategy.md`](data_cleaning_strategy.md)). |

### `text`

| Field | Value |
|---|---|
| Data type | string (object) |
| Example value | `"WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress..."` |
| Description | The full article body as scraped from the original source. |
| Missing values | 0 rows with a null value; **630 rows (Fake) + 1 row (Real) with an empty string** (article has a title but no body). |
| Used by final model | **Yes** — this is the primary input. |
| Reason if discarded | N/A — retained. Empty-string rows are dropped during cleaning (not discarded as a column, discarded as specific rows — see [`data_cleaning_strategy.md`](data_cleaning_strategy.md)). |

### `subject`

| Field | Value |
|---|---|
| Data type | string (object), effectively categorical (6 values in Fake, 2 values in Real) |
| Example value | `"politicsNews"` (Real) / `"News"` (Fake) |
| Description | A topic tag assigned by the original news aggregator/scraper. |
| Missing values | 0 |
| Used by final model | **No** |
| Reason if discarded | Introduces severe label leakage. The set of `subject` values used in `Fake.csv` (`News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east`) and in `True.csv` (`politicsNews`, `worldnews`) has **zero overlap**. A trivial rule using only this column reaches **100.000% classification accuracy** (verified in `01_dataset_analysis.ipynb` / [`label_leakage_analysis.md`](label_leakage_analysis.md)) — meaning the column reflects how the two source files were collected, not any property of the article's truthfulness. |

### `date`

| Field | Value |
|---|---|
| Data type | string (object); mostly `"Month D, YYYY"` |
| Example value | `"December 31, 2017"` |
| Description | Publication date as scraped from the original source. |
| Missing values | 0 rows null; **45 rows (Fake only) with a non-standard value** — 35 are a valid date in an alternate format (`DD-Mon-YY`, e.g. `19-Feb-18`), and **10 are not dates at all** (raw image/article URLs, and in one case a leaked WordPress page-builder template in place of an article). `True.csv` has 0 malformed dates. |
| Used by final model | **No** |
| Reason if discarded | Not needed for a text-classification model, and carries a temporal bias risk: `Fake.csv` spans **2015-03-31 to 2017-12-31**, while `True.csv` spans only **2016-01-13 to 2017-12-31** — meaning any article dated before January 2016 is guaranteed to be Fake by construction of the dataset, not by content. See [`label_leakage_analysis.md`](label_leakage_analysis.md). |

### `label` *(derived column, not in the original CSVs)*

| Field | Value |
|---|---|
| Data type | integer (0 or 1) |
| Example value | `0` (fake) / `1` (real) |
| Description | Created during Task/Phase 2 by tagging `Fake.csv` rows as `0` and `True.csv` rows as `1`, then concatenating both files into one DataFrame (see `01_dataset_analysis.ipynb`, Section 9). This is the model's target/output variable. |
| Missing values | 0 (created by assignment, cannot be missing) |
| Used by final model | **Yes** — this is the prediction target, not an input feature. |
| Reason if discarded | N/A |

---

## Summary Table

See [`report/tables/data_dictionary.csv`](../report/tables/data_dictionary.csv) for the same information in a format suitable for direct import into the report (Word/Excel/LaTeX table generators).

## Net columns used as model input

After this analysis, the model's actual input is: **`title` + `text` (combined into a single text field)**. `subject` and `date` are excluded entirely. This decision is expanded on and justified further in [`label_leakage_analysis.md`](label_leakage_analysis.md) and finalized in [`data_cleaning_strategy.md`](data_cleaning_strategy.md).
