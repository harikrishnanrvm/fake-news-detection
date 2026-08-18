# Threats to Validity

**Phase:** Phase 6 — Comparative Analysis & Discussion
**Purpose:** an honest, explicit account of the limits on what this project's results can and
cannot claim — the kind of section a rigorous research paper includes and a viva examiner will
specifically probe for. Every threat named here was either directly observed during this
project's own phases or is a standard, well-known limitation of the methodology used; none is
hypothetical filler.

---

## Internal Validity

*Internal validity asks: within this project's own experiments, can we trust that the measured
differences are actually caused by what we think caused them?*

- **Shared, seeded train/validation/test split.** Both models were trained and evaluated using
  the exact same `training/split.py` function and `RANDOM_SEED = 42`, confirmed directly (Phase
  6, Step 4 of `notebooks/05_model_comparison.ipynb`) to reproduce the identical 5,796-row test
  set both models were originally scored on. This is a strength, not a weakness — it is what
  makes the Section 3/4 comparison in `docs/model_comparison_report.md` valid at all — but it is
  also a single split. A different random seed could, in principle, produce a somewhat different
  train/test partition and a somewhat different accuracy gap; this was not tested with multiple
  seeds (e.g. k-fold cross-validation), which was judged out of scope for a BCA project prioritizing
  simplicity and a single, clearly reproducible run over a more elaborate validation scheme.
- **No hyperparameter tuning for either model.** Both the Logistic Regression's `C` and the
  LSTM's architecture/learning-rate/regularization settings were fixed, documented defaults
  (`config/settings.py`), not the result of a search. This was a deliberate choice (a "baseline"
  is supposed to represent a simple, untuned configuration — see `docs/baseline_model_report.md`),
  but it means the ~1-point accuracy gap found in Phase 6 reflects these *specific*
  configurations, not necessarily the best each architecture could achieve with tuning.
- **Single training run per model.** Neither model's training was repeated across multiple random
  initializations. The LSTM in particular showed a visibly noisy training history
  (`report/figures/lstm_training_history.png`) — a different weight initialization could plausibly
  land on a different final epoch/performance, and this run-to-run variance was not measured.
- **The Reuters ablation study's own finding constrains how the baseline's 98.24% should be
  read.** `docs/reuters_ablation_study.md` established that the baseline's high accuracy is not
  concentrated in one removable token but is redundantly encoded across several correlated
  wire-service style cues. That finding was not repeated for the LSTM in this phase (see External/
  Construct Validity below) — a direct gap worth naming rather than silently leaving unexamined.

## External Validity

*External validity asks: how far do these results generalize beyond this exact dataset and setup?*

- **Single dataset, two fixed sources.** The entire project uses one Kaggle dataset
  (`clmentbisaillon/fake-and-real-news-dataset`), where "Real" is essentially Reuters wire copy
  and "Fake" is a specific set of blog/aggregator sources. Both models' ~97-98% accuracy reflects
  how well each distinguishes *these two particular writing styles* — not a general ability to
  detect fabricated claims. A model trained here would very likely need retraining (at minimum)
  to perform comparably on news from different outlets, a different time period, or a different
  political/topical mix.
- **Time period.** The articles are drawn from a specific historical window (predominantly
  2016-2017 U.S. political news, per the dataset's `date` column explored in
  `docs/data_dictionary.md`). Neither model has been evaluated on more recent news, different
  news cycles, or non-political subject matter.
- **English-only, and a specific register of English.** Per the project specification's explicit scope, no
  multilingual capability was built or tested. Even within English, both models are tuned to the
  specific registers present in this dataset (formal wire-service prose vs. informal blog/opinion
  writing) and may not transfer to, for example, satire, academic writing, or social-media-style
  short text.
- **No cross-dataset evaluation was performed.** This is the single most direct way to test
  external validity (train here, evaluate on an independent fake-news dataset) and was explicitly
  named as recommended future work in `docs/model_comparison_report.md` (Section 11) rather than
  attempted in this phase.

## Construct Validity

*Construct validity asks: does the thing we measured actually reflect the thing we claim to be
measuring* ("is this news fake") *rather than something else that happens to correlate with it?*

- **This is the most significant, already-documented threat in the whole project.** The Reuters
  ablation study (`docs/reuters_ablation_study.md`) found direct evidence that the baseline is
  substantially measuring **"does this read like Reuters wire style vs. this dataset's blog
  style"** — which correlates extremely strongly with the Fake/Real label *in this dataset*, but
  is a different construct from "is this specific factual claim true." A model scoring 98%
  accuracy on this construct is not the same as a model that has learned to fact-check.
- **The label itself is a proxy.** "Real" here means "published by Reuters, as included in this
  dataset's collection process" and "Fake" means "included in this dataset's Fake.csv source
  list" — both are proxies for ground-truth veracity, assembled by the dataset's original
  creators, not independently fact-checked article-by-article as part of this project.
- **Binary labels lose nuance.** Real news can be misleading through selective framing without
  being fabricated; fake news exists on a spectrum from outright fabrication to genuine errors to
  satire. A single Fake/Real label cannot represent that spectrum, and neither model was designed
  or evaluated to attempt to.

## Dataset Limitations

- **Class balance is close but not identical** (2,617 Fake / 3,179 Real in the test set, matching
  the roughly 52%/48% split noted in `docs/data_dictionary.md`) — not severe enough to require
  imbalance-handling techniques (per the project specification's original dataset-selection rationale), but
  still worth naming as a property of the data, not a designed experimental condition.
- **4,317 duplicate/near-duplicate rows and 10 corrupted rows were removed** during Stage 2
  cleaning (`docs/duplicate_analysis.md`) — a necessary cleaning step, but one that changes the
  effective dataset size and composition from the raw download, and is a judgment call (e.g. the
  exact duplicate-detection method) that could reasonably have been made differently.
- **5,186 of 38,638 rows (13.4%) still contained the word "reuters"** even after Stage 3's
  dateline-stripping (`docs/reuters_ablation_study.md`) — an acknowledged residual cleaning
  limitation, not fully resolved, deliberately documented rather than chased further (see that
  study's Question 6 for the reasoning).

## Potential Biases

- **Source-style bias** (already discussed under Construct Validity) is the dominant, most
  consequential bias in this project — both models substantially learn "wire-service style vs.
  blog style," not truthfulness directly.
- **Topical/political bias.** The dataset's articles are concentrated in U.S. political news from
  a specific period; both models' learned vocabulary and associations (e.g. specific politicians'
  names, "said," weekday datelines) are shaped by that topical concentration and may not transfer
  to non-political fake news (e.g. health misinformation, financial scams).
- **Selection bias in the dataset's construction.** This project did not control, and cannot
  fully audit, how the original dataset's creators selected which Fake and Real articles to
  include — any systematic pattern in that original collection process (e.g. which fake-news
  sites were chosen) becomes a bias both models inherit.

## Future Improvements

- **Cross-dataset validation** — train here, test on an independently-sourced fake-news dataset,
  to separate genuine generalization from this dataset's specific source-style confound.
- **Pretrained embeddings or a pretrained transformer encoder** (e.g. GloVe, or transfer learning
  from DistilBERT) for the LSTM, to test whether its current shortfall is a data-scale limitation
  rather than an architectural one — deliberately out of scope for this project's stated goal of
  a simple, fully-explainable-in-a-viva LSTM (the project specification).
- **Multiple random seeds / k-fold cross-validation**, to quantify how sensitive the ~1-point
  accuracy gap (Section 3, `docs/model_comparison_report.md`) is to the specific train/test split
  chosen, rather than reporting a single run's result as if it were exact.
- **A simple ensemble of both models**, motivated directly by Section 4's finding that only 0.76%
  of test articles fool both models simultaneously — a high theoretical ceiling worth testing in
  future work.
- **Human-in-the-loop or fact-checking-based evaluation**, to address the Construct Validity gap
  directly — comparing model predictions against independently-verified factual accuracy, not
  just against this dataset's Fake/Real source labels.
