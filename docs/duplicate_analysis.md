# Duplicate Analysis

Computed on the combined, labeled dataset (44,898 rows = `Fake.csv` + `True.csv`) produced in `notebooks/01_dataset_analysis.ipynb`. All counts below are **extra rows** — i.e. rows beyond the first occurrence of each duplicate group, the rows that would actually be removed by deduplication.

---

## Category 1 — Exact duplicate rows

Every column (`title`, `text`, `subject`, `date`) identical.

- **Count:** 209 extra rows
- **Percentage:** 0.47% of 44,898

**Example** — a Reuters live-blog-style article republished identically 8 times:

| title | label |
|---|---|
| "Highlights: The Trump presidency on April 13 at 9:30 P.M. EDT/0130 GMT on Friday" | Real (×8 identical rows) |

This kind of duplication is consistent with wire-service updates being re-saved/re-scraped at different times without any actual content change.

---

## Category 2 — Duplicate article body, different title

Same `text`, but at least one other row sharing that text has a different `title`.

- **Count:** 655 extra rows
- **Percentage:** 1.46% of 44,898

**Example** — the same Fake-labeled article body reused under several different headlines/polls (titles observed for one shared body): *"TAKE OUR POLL: Who Do You Think President Trump Should Pick To Replace James Comey?"*, *"Joe Scarborough BERATES Mika Brzezinski Over 'Cheap Shot' At Ivanka Trump..."*, *"WATCH TUCKER CARLSON Scorch Sanctuary City Mayor..."* — consistent with a templated/boilerplate body reused across multiple listicle-style posts.

> Note: in duplicate groups larger than 2 rows, Category 1 and Category 2 can technically overlap (e.g. a group of 8 identical-text rows might contain a subset that also share a title). The counts above are reported per the filtering method described in the underlying script; the **deduplication recommendation below sidesteps this ambiguity** by keying on `text` alone, in one unambiguous step.

---

## Category 3 — Duplicate title, different article body

Same `title`, but at least one other row sharing that title has different `text`.

- **Count:** 392 extra rows
- **Percentage:** 0.87% of 44,898

**Example** — `"Factbox: Trump fills top jobs for his administration"` used as a headline more than once, each time with an updated body as new appointments were announced (both rows labeled Real) — a legitimate republish-with-update pattern, not a data error.

---

## Category 4 — Near duplicates (normalized-fingerprint approximation)

A full pairwise similarity comparison (e.g. cosine similarity over all ~45k×45k row pairs) is computationally expensive and unnecessary at BCA scope. Instead, a lightweight proxy was used: lowercase the article body, strip all non-alphanumeric characters, and take the first 200 characters as a "fingerprint." Rows sharing a fingerprint but **not** an exact `text` match are flagged as near-duplicates.

- **Count:** 698 extra rows (beyond anything already caught by exact-text matching)
- **Percentage:** 1.56% of 44,898

**Example** — a recurring daily column, *"Factbox: Trump on Twitter (Dec 29) - Approval rating, Amazon"* and *"Trump on Twitter (Dec 28) - Global Warming"* (95 rows in this one series alone) all open with the identical boilerplate sentence *"The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump..."* before diverging into that day's actual content.

**Important caveat:** this method mostly catches **legitimate recurring-column articles** that share a formulaic opening, not accidental duplication. These should **not** be auto-removed — see recommendation below.

---

## Cross-file duplicates (same article in both Fake.csv and True.csv)

Checked separately because this would represent a genuine labeling conflict (the same content assigned both labels).

- **Result: 0 genuine matches.** A raw set-intersection on `text` found exactly one shared value between the two files, but it was a single blank/whitespace-only string (`" "`), i.e. an artifact of the empty-article-body issue already tracked in the EDA notebook's Dataset Quality section — not a real duplicated article. No content-level label conflicts were found.

---

## Recommended deduplication strategy

**Deduplicate on `text` alone, keeping the first occurrence, before the train/validation/test split.**

```
combined_df.drop_duplicates(subset=["text"], keep="first")
```

This single, unambiguous step:
- Removes **6,252 rows (13.92%)** of the 44,898-row combined dataset, leaving **38,646 rows**.
- Fully subsumes Categories 1 and 2 above (both require identical `text`).

**Do not** automatically remove Category 3 (same title, different text) or Category 4 (near-duplicate fingerprints) — both were found on inspection to mostly represent legitimate distinct articles (an updated wire story, a recurring column format) rather than errors. Removing them would delete real, non-redundant training examples. These are documented as a known limitation rather than solved, per BCA-scope practicality.

This decision is carried forward into [`data_cleaning_strategy.md`](data_cleaning_strategy.md).

---

## Why duplicates are dangerous

1. **Train/test leakage inflates reported performance.** If an article (or its exact duplicate) ends up in both the training split and the test split — likely with a naive random split, since ~14% of rows have an exact twin somewhere in the data — the model doesn't need to generalize to that test example. It already memorized the identical text during training. The test accuracy/F1 in that case measures memorization, not the ability to classify genuinely unseen news.
2. **Duplicate weighting bias.** Every duplicate row is a separate training example that contributes its own gradient update. An article repeated 8 times (as seen in Category 1) effectively gets 8× the influence on the learned decision boundary compared to a normal, non-duplicated article — silently over-weighting whatever pattern that one repeated story happens to represent.

## How this affects train/test evaluation specifically

Deduplicating **before** splitting (not after) is what matters — if you split first and deduplicate later, an identical pair can still land on opposite sides of the split. The fix implemented in Phase 3 must be: **drop duplicate text → then perform the stratified train/validation/test split** (see `config.settings.RANDOM_SEED`, `TRAIN_SPLIT`/`VAL_SPLIT`/`TEST_SPLIT`), never the reverse order.

## How to explain this in the viva

> "I checked for duplicate articles before splitting my data and found about 14% of rows were exact duplicates of another row elsewhere in the dataset — mostly republished wire stories. I removed these based on the article text, *before* performing my train/validation/test split, specifically to prevent the same article from appearing in both my training and test sets. Without this step, my reported accuracy would have been inflated by memorization rather than reflecting real generalization to unseen articles."

This is a specific, evidence-backed answer (you have the exact percentage and a concrete example), which is exactly what distinguishes "I ran a tutorial" from "I understand my data."

---

See [`report/tables/duplicate_statistics.csv`](../report/tables/duplicate_statistics.csv) for these figures in a report-ready table format.
