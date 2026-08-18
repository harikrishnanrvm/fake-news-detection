# Data Cleaning Strategy

This document records every cleaning **decision** and its justification, before any of it is implemented in code. It draws directly on the findings in [`data_dictionary.md`](data_dictionary.md), [`duplicate_analysis.md`](duplicate_analysis.md), and [`label_leakage_analysis.md`](label_leakage_analysis.md). Implementation happens in `02_preprocessing.ipynb`, only after this document is reviewed.

---

## Columns to keep

| Column | Kept as |
|---|---|
| `title` | Combined into the model's input text (see "Title + body combination" below) |
| `text` | Combined into the model's input text |
| `label` | Kept as-is — the prediction target |

## Columns to remove

| Column | Reason |
|---|---|
| `subject` | Perfect label leakage — Fake and Real subject values never overlap (100.000% standalone rule accuracy). See `label_leakage_analysis.md` §1. |
| `date` | Leaks the label two ways: a date-range rule alone gets 100% precision on 6.44% of rows, and all 45 malformed date values occur only in Fake.csv. Not needed for text classification. See `label_leakage_analysis.md` §4. |

Both columns are dropped **entirely**, not partially masked or bucketed — there is no version of "keep a little of `subject`" that doesn't reintroduce some of the same leakage.

---

## Duplicate handling

**Decision:** `drop_duplicates(subset=["text"], keep="first")`, applied **before** the train/validation/test split, not after.

- Removes 6,252 rows (13.92%), leaving 38,646 rows.
- This single step already covers both "exact duplicate rows" and "duplicate text with a different title" (both require identical `text`).
- **Not** applied to "duplicate title, different text" (392 rows) or the fingerprint-based "near duplicates" (698 rows) — both were manually inspected and found to be legitimate distinct articles (an updated wire story; a recurring daily column sharing a boilerplate opening), not data errors. Automatically removing them would delete real, non-redundant training examples.
- Order matters: deduplicating *before* splitting is what actually prevents train/test leakage. Deduplicating after an already-performed split would not undo any leakage that already happened at split time.

Full reasoning in `duplicate_analysis.md`.

---

## Empty article handling

**Decision:** drop rows where `text` (after stripping whitespace) is an empty string.

- Affects 630 rows in Fake.csv and 1 row in True.csv (~1.4% of the dataset).
- No empty-title rows exist in either file, so no title-based dropping is needed.
- These rows are dropped rather than filled with a placeholder, because a placeholder (e.g. `"[no content]"`) would just become a new, meaningless but perfectly memorizable token — the same shortcut-signal problem in a different shape.

---

## Malformed rows (the 45 non-standard `date` rows)

Since `date` is being dropped as a column entirely, most of these rows don't need special handling *because of their date* — but inspection revealed they are not a uniform group:

- **35 rows** have a validly-formatted date, just in a different format (`DD-Mon-YY` instead of `Month D, YYYY`). Their `title`/`text` are normal, legitimate articles. **No action needed** beyond the column drop already decided above.
- **10 rows** are genuinely corrupted: the `date` field contains a raw image/article URL, and in one case the `title`/`text` fields themselves contain a leaked WordPress page-builder template (`[vc_row][vc_column...]`) instead of an actual article. **Decision: drop these 10 rows entirely** — not because of the `date` field (which is being dropped anyway), but because their `title`/`text` do not contain a real article, confirmed by direct inspection in the deep-dive analysis.

---

## Reuters removal strategy

**Decision:** strip a leading wire-service dateline from `text` using a pattern matching *up to and including* `"(Reuters) - "` at (or very near) the start of the string, leaving the rest of the article body untouched.

- Applies to the ~99.2% of Real articles that carry this tag.
- Rows without the pattern (effectively all of Fake, and the small remainder of Real) are left unchanged — this is a targeted removal, not a blanket rule that could damage unrelated text.
- Explicitly **not** solved by generic punctuation/parenthesis removal — a naive "remove all parentheses" step would strip `(Reuters)` but would also mangle unrelated parenthetical text elsewhere in both classes, and wouldn't remove the `"WASHINGTON -"`-style dateline of city names that often precedes it. A dedicated, narrowly-scoped step is safer and easier to justify in the report than an incidental side-effect of a broader rule.
- The `pic.twitter.com` shortcut (`label_leakage_analysis.md` §3) does **not** need its own dedicated rule — generic URL removal (see `preprocessing_plan.md`) handles it as a side effect, provided URLs are deleted outright rather than replaced with a placeholder token (a placeholder would preserve the same shortcut under a new name).

---

## Title and article body: will they be combined?

**Decision: yes.** `title` and `text` will be concatenated into a single input field (e.g. `content = title + " " + text`) before tokenization.

**Justification:**
- EDA found a real, non-leakage style difference in titles (Fake titles average 14.7 words vs. Real's 10.0, with a much wider range) — genuine signal that would be thrown away if `title` were dropped.
- The project specification's Architecture section already specifies a single linear pipeline (`Tokenizer → Padding → Embedding → LSTM`) with one input stream, not two parallel branches. Combining title and body into one field keeps the model architecture as simple as originally planned — a two-input model (separate title/body encoders merged later) would be a legitimate alternative, but it adds real architectural complexity for a BCA-scope project without a demonstrated need.
- Combining is done as plain string concatenation, not any special weighting of the title — kept simple and explainable.

---

## Expected preprocessing pipeline (design order)

This is the intended order of operations for `02_preprocessing.ipynb`. Per-step justification for the generic NLP steps (lowercasing, stop words, etc.) is in [`preprocessing_plan.md`](preprocessing_plan.md); this list only fixes the *order*, which matters (e.g. deduplication must happen before splitting, not after).

1. Load `Fake.csv` / `True.csv`, assign `label` (0/1), concatenate — already done in `01_dataset_analysis.ipynb`, saved as `dataset/processed/combined_raw.csv`.
2. Drop `subject` and `date` columns.
3. Drop the 10 genuinely-corrupted rows identified above.
4. Drop rows with an empty `text` field.
5. Combine `title` + `text` into a single `content` field.
6. Strip leading Reuters-style datelines from `content`.
7. Deduplicate on `content` (or on the pre-combination `text`, whichever is settled on in implementation — functionally equivalent since `title` is far less often duplicated than `text`), keeping the first occurrence.
8. Apply the general NLP cleaning steps decided in `preprocessing_plan.md` (lowercase, HTML removal, URL removal, punctuation removal, stop-word removal where applicable, lemmatization where applicable, whitespace normalization).
9. Stratified train/validation/test split (70/15/15), using `config.settings.RANDOM_SEED`.
10. Fit the tokenizer **on the training split only**, save `tokenizer.pkl` + `preprocessing_config.json` per the Model Artifacts standard in the project specification.

Step 9 fitting the tokenizer only on the training split (not on the full dataset before splitting) is a reproducibility/leakage point in its own right: fitting on everything first would let the validation/test vocabulary quietly influence the model's word index, which is a subtler, second form of train/test leakage beyond the duplicate-row issue already covered.

---

## What this document deliberately does not do

No code is written here — only design decisions and their justification. Implementation is deferred to `02_preprocessing.ipynb`, pending review of this document and [`preprocessing_plan.md`](preprocessing_plan.md).
