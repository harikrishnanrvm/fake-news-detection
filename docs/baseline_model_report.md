# Baseline Model Report — TF-IDF + Logistic Regression

**Phase:** Phase 4 — Baseline Machine Learning
**Produced by:** `notebooks/03_baseline_model.ipynb`, calling `training/baseline.py` and `evaluation/`
**Artifacts:** `models/baseline/` (trained vectorizer, model, label map, metadata)

This document is written for direct inclusion in the BCA final report (Section: Baseline
Model). It covers why Logistic Regression was selected, its advantages and limitations, the
results actually obtained, and what improvement can realistically be expected from the LSTM in
Phase 5.

---

## Why Logistic Regression was selected

Logistic Regression was chosen over several realistic alternatives for TF-IDF-based text
classification:

| Alternative | Why it was not chosen instead |
|---|---|
| Naive Bayes | Comparable simplicity and speed, but assumes even stronger word-independence and produces less well-calibrated probabilities. A reasonable second choice, not a clearly better first one. |
| Decision Tree | Prone to overfitting on high-dimensional sparse TF-IDF data unless carefully pruned; not a standard text-classification baseline. |
| Random Forest / Gradient Boosting | Competitive accuracy is possible, but slower to train and not as directly interpretable — no single clean per-word weight the way Logistic Regression provides. Starts to blur the line between "baseline" and "another model to compare." |
| SVM | A strong alternative, often similar in accuracy. Not chosen mainly because Logistic Regression's native probability output was needed for the ROC curve, and its coefficients are the most directly interpretable for the Feature Importance analysis. |

The deciding factors were **interpretability** (a single signed weight per word, directly
explaining predictions — see Feature Importance below) and a **native probability output**
(needed for both the ROC curve and the "confidence score" planned for the frontend in
the project specification).

## Why this is "the baseline," not just "a model"

It is the simplest reasonable approach for this task — a linear model over word frequencies —
used deliberately as a reference point. Its purpose is to answer, later: *does the added
complexity of an LSTM (sequence modeling, embeddings, more hyperparameters, longer training
time) actually buy a meaningful improvement over something this simple?* If it doesn't, that is
still a valid, reportable finding — not a failure of the project.

---

## Results

Computed on a held-out test set (15% of the cleaned dataset, 5,796 articles), never seen during
training or TF-IDF fitting:

| Metric | Value |
|---|---|
| Accuracy | 0.9824 |
| Precision | 0.9788 |
| Recall | 0.9893 |
| F1 Score | 0.9840 |
| AUC | 0.9979 |
| Training time | 0.24 seconds |
| Vocabulary size | 20,000 (capped, unigrams only) |

Confusion matrix: 2,549 true negatives (Fake→Fake), 68 false positives (Fake→Real), 34 false
negatives (Real→Fake), 3,145 true positives (Real→Real). Full figures: `report/figures/baseline_confusion_matrix.png`,
`report/figures/baseline_roc_curve.png`. Full per-class report: `report/tables/baseline_classification_report.csv`.

**Interpreting a ~98% result:** this is *after* Phase 3 removed the dataset's most severe
leakage sources (`subject`, `date`, the leading Reuters dateline — see
`docs/label_leakage_analysis.md`). A high score following that cleanup indicates the two
classes still differ in genuine, learnable ways (word choice, structure, source-style
conventions) — not that the leakage removal failed. Section "An important limitation found
during this phase" below documents one specific residual signal found via feature-importance
analysis, in the interest of full transparency.

---

## Advantages (observed, not just theoretical)

- **Fast:** training took under a quarter of a second on ~27,000 articles — enables rapid
  iteration during development.
- **Interpretable:** every prediction can be traced to specific word weights (see Feature
  Importance below) — a property the LSTM will not have without extra work (e.g. attention
  visualization), and a genuine strength to highlight in the report/viva.
- **Strong result with minimal tuning:** no hyperparameter search was needed to reach ~98% —
  the default Logistic Regression configuration was sufficient.

## Limitations (observed, not just theoretical)

- **No use of word order.** TF-IDF treats each article as an unordered bag of words — "the
  senator did not deny the claim" and "the senator did deny the claim" would look almost
  identical to this model, since only individual word frequencies are used, not sequence.
- **Cannot verify facts.** The model learns statistical association between words/topics and
  the label, not truth. Error analysis (in the notebook) found false positives and false
  negatives concentrated in articles with atypical writing style for their class — evidence the
  model is partly keying on *style*, not content-independent truthfulness.
- **Residual leakage sensitivity.** Because it is a linear model over raw word frequencies, it
  is *especially* good at exploiting single-token shortcuts (like a leftover "reuters" mention)
  — see below.

## An important limitation found during this phase

Feature importance analysis (see `notebooks/03_baseline_model.ipynb`, Feature Importance
section) found that **"reuters" is the single strongest word indicating "Real,"** despite Phase
3 explicitly stripping the leading `"(Reuters) - "` dateline. Investigation found **5,186 of
38,638 rows (13.4%) still contain the word "reuters" elsewhere in the article body** — mostly in
Real articles (4,978 Real vs. 208 Fake) — from secondary/corrected datelines or in-body source
citations the prefix-stripping regex (bounded to the first ~80 characters) doesn't reach.

This is reported here deliberately, not omitted: it is a genuine, specific residual leakage
signal, found through model interpretation rather than assumed away. It does not undo the
value of Phase 3's cleaning (the leakage removed there — `subject` at 100% rule accuracy — was
far more severe), but it is an honest, reportable limit on how clean the data actually ended up,
and a concrete example of why feature-importance analysis is worth doing even after a careful
planning-stage cleaning effort.

---

## Expected improvements from the LSTM (Phase 5)

Based specifically on this baseline's observed limitations:

- **Word order and negation.** An LSTM processes text as a sequence and can, in principle,
  distinguish "did not confirm" from "did confirm" — something no bag-of-words model can do
  regardless of how it's tuned. This is the single most concrete, mechanistic reason to expect
  an improvement, not just a general "deep learning is more powerful" assumption.
- **Contextual word meaning.** An embedding layer can represent a word differently depending on
  its neighbors (to a degree), rather than treating "bank" (river) and "bank" (financial)
  identically the way TF-IDF does.
- **What the LSTM will likely *not* fix:** it cannot verify facts either, and it will need to be
  checked for the same residual "reuters" leakage sensitivity found here — the LSTM reads the
  same `lstm_text` column (from the same Stage 3 output), so the same 13.4% residual signal is
  present in its input too. This should be checked explicitly in Phase 5/6, not assumed solved
  by switching architectures.
- **A realistic expectation, stated plainly:** given the baseline already reaches ~98% on this
  particular dataset (partly aided by remaining stylistic/source-format signal), the LSTM may
  show only a modest improvement, or could even perform similarly — and that would still be a
  valid, explainable outcome for the report, not a failed experiment.

---

## What this report deliberately does not include

No LSTM implementation and no baseline-vs-LSTM comparison — both are out of scope for Phase 4.
The comparison happens in Phase 6, after Phase 5 produces its own results under
`evaluation/experiments.csv`.
