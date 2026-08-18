# Label Leakage Analysis

**Purpose:** to determine, before any model is trained, whether any column or textual pattern lets the label be predicted without genuinely analyzing the article's content. This document is intended to become a dedicated subsection of the final report (Section: Data Understanding → Label Leakage).

**Method used throughout:** for each suspected signal, a trivial rule (a single `if/else`, no machine learning) is built using *only* that signal, and its accuracy against the true label is measured. If a one-line rule can predict the label with near-perfect accuracy, that signal is leaking the label rather than describing the article's truthfulness.

**Nothing is removed in this document** — findings only. Removal decisions are finalized in [`data_cleaning_strategy.md`](data_cleaning_strategy.md).

---

## 1. `subject`

**Rule tested:** *"predict Real if `subject` is one of True.csv's subject values, else predict Fake."*

- Fake.csv subject values: `News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east`
- True.csv subject values: `politicsNews`, `worldnews`
- **Overlap between the two sets: none (empty set).**
- **Rule accuracy: 100.000%**

**Can `subject` alone predict the class?** Yes — perfectly, on this dataset. This is the most severe leakage source found: not a strong correlation, but a mathematically perfect separator, because the two source files were tagged with entirely disjoint vocabularies during collection.

**Scientific justification for removal:** a feature that perfectly separates the classes without reference to the article's actual language is not learning to detect fake news — it is learning to detect *which of the two source files this row came from*, which is a property of how the dataset was assembled, not a property of misinformation. This is a classic case of **leakage via data-collection artifact**, analogous to a medical model accidentally learning "which hospital submitted this record" instead of "is the patient sick."

**Decision:** remove `subject` entirely before modeling.

---

## 2. Reuters prefix — `"(Reuters)"`

**Quantified occurrence:**

| | Fake.csv | True.csv |
|---|---|---|
| Contains `"(Reuters)"` anywhere in `text` | 9 / 23,481 (0.04%) | 21,247 / 21,417 (**99.2%**) |
| Appears as a dateline (within the first ~60 characters) | — | 20,144 / 21,417 (94.1%) |

**Rule tested:** *"predict Real if `text` contains `(Reuters)`, else predict Fake."*

- **Rule accuracy: 99.601%** (tp = 21,247, fp = 9, fn = 170, tn = 23,472)
- Precision for Real: 99.96% — of the rows the rule calls Real, almost all really are.
- Recall for Real: 99.21% — the rule catches almost all of the actual Real articles.

**How predictive is it?** Extremely — a single substring check reproduces the label for 99.6% of the entire 44,898-row dataset, with zero training and zero understanding of language, tone, or factual content.

**Why this is leakage, not signal:** `(Reuters) - ` is a wire-service dateline convention, present because the "Real" class in this dataset happens to be sourced almost entirely from Reuters. It says "this text came from Reuters," not "this text is true." A genuine but non-Reuters real article (or a fake article that happens to fabricate a Reuters-style dateline) would break this rule immediately — the model would have learned a shortcut that does not generalize beyond this specific dataset's sourcing.

**Decision:** strip leading wire-service datelines from `text` during preprocessing, so the model cannot key on this tag.

---

## 3. `pic.twitter.com`

**Quantified occurrence:**

| | Fake.csv | True.csv |
|---|---|---|
| Contains `"pic.twitter.com"` | 3,474 / 23,481 (**14.8%**) | 0 / 21,417 (0%) |
| Of rows containing it, % that are Fake | — | **100%** |

**Why it is a shortcut signal:** unlike the Reuters tag (which is highly predictive in both directions), this one is **one-directional**: it never appears in a Real article, but it also only appears in 14.8% of Fake articles — so on its own it can't classify most of the dataset. However, for the subset of rows where it *is* present, it is a perfect (100%) predictor of Fake. This means a model could learn "if I see this pattern, output Fake with full confidence" — a valid-looking shortcut for roughly one in seven Fake articles, again based on a formatting/sourcing artifact (embedded tweet links common to this era's clickbait-style sites) rather than genuine content analysis.

**Decision:** no special-cased rule needed — standard URL removal (already planned in the project specification's preprocessing steps, see [`preprocessing_plan.md`](preprocessing_plan.md)) will remove this pattern as a side effect of removing all URLs generically. Important detail carried into the cleaning strategy: URLs must be **deleted outright**, not replaced with a placeholder token like `<URL>` — replacing would preserve "a URL was here" as a token, which is exactly the same shortcut under a new name.

---

## 4. `date`

**Does `date` introduce useful information, or unintended bias?** Unintended bias — confirmed two ways:

**a) Date range does not overlap at the low end:**

| | Earliest valid date | Latest valid date | Valid rows |
|---|---|---|---|
| Fake.csv | 2015-03-31 | 2017-12-31 | 23,436 |
| True.csv | **2016-01-13** | 2017-12-31 | 21,417 |

**Rule tested:** *"predict Fake if `date` is earlier than 2016-01-13 (True.csv's earliest date)."*

- **2,891 rows (6.44% of the dataset) satisfy this rule — and every single one of them is genuinely Fake (0 false positives).**

This means roughly 1 in 16 articles can be labeled correctly using nothing but the calendar date, because the two source files were scraped over different time windows, not because articles before January 2016 are inherently less truthful.

**b) Malformed date values are exclusive to Fake.csv:** 45 rows have a non-standard `date` value, and **all 45 are in Fake.csv**; `True.csv` has zero. (Of those 45: 35 are a validly-formatted date in an alternate format, `DD-Mon-YY`; 10 are genuinely corrupted — raw image/article URLs, and in one case an entire scraped WordPress template, sitting in the `date` field — see [`duplicate_analysis.md`](duplicate_analysis.md) / `data_cleaning_strategy.md` for the row-level cleanup decision on those 10.) Simply checking "is this `date` value malformed?" is therefore itself a small but perfect leakage signal (0.1% of rows).

**Decision:** remove `date` entirely before modeling — it is not needed for text classification, and every way it has been examined here shows it reflects *when/how the two files were scraped*, not the truthfulness of the article.

---

## Summary

| Feature | Leakage severity | Standalone rule accuracy | Remove before modeling? |
|---|---|---|---|
| `subject` | Severe (perfect separator) | 100.000% | **Yes** |
| `"(Reuters)"` tag in `text` | Severe | 99.601% | **Yes** — strip the tag, keep the rest of the text |
| `"pic.twitter.com"` in `text` | Moderate (one-directional) | 100% precision, 14.8% recall | Handled generically via URL removal |
| `date` | Moderate (range + malformed-value signal) | 100% precision on 6.44% of rows via date-range rule | **Yes** |

**Scientific justification, stated once for all four:** a supervised model is only doing its intended job if the patterns it learns are properties of the *input we care about* (the article's language) rather than properties of *how the dataset happened to be assembled*. Every signal above is the latter. Leaving them in would produce a model that reports an impressive but scientifically meaningless accuracy number — one that would not survive contact with a single real-world article from outside this dataset's two source pipelines. Identifying and addressing this before training, rather than discovering it after an unbelievably high accuracy score, is the difference between a defensible result and an indefensible one in front of a viva panel.

These findings are carried forward into [`data_cleaning_strategy.md`](data_cleaning_strategy.md) as concrete cleaning decisions.
