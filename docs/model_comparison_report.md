# Model Comparison Report — Baseline vs. LSTM

**Phase:** Phase 6 — Comparative Analysis & Discussion
**Produced by:** `notebooks/05_model_comparison.ipynb`, calling `training/split.py`,
`evaluation/metrics.py`, `evaluation/significance.py`, and the new `evaluation/agreement.py`
**Inputs (all frozen, read-only):** `models/baseline/`, `models/lstm/`,
`dataset/processed/03_preprocessed.csv`, `evaluation/experiments.csv`

This document is written for direct inclusion in the BCA final report (Section: Results and
Discussion). No model was retrained to produce it — every number below comes from either a
fresh, timed inference pass of both frozen models over the identical test set, or from
already-recorded metadata (`models/*/metadata.json`, `evaluation/experiments.csv`,
`docs/baseline_model_report.md`, `docs/lstm_model_report.md`).

---

## 1. Overview

This project built two fundamentally different approaches to the same classification problem —
telling a Fake news article from a Real one:

| | Baseline: TF-IDF + Logistic Regression | LSTM |
|---|---|---|
| Category | Traditional Machine Learning | Deep Learning |
| How it represents text | **Feature engineering**: a fixed formula (TF-IDF) computes one frequency-based score per vocabulary word, before training even starts | **Feature learning**: an Embedding layer *learns* a 100-number vector per word, discovered from data during training |
| Structure of the representation | **Bag-of-words** — an article is an unordered count of which words appear, and how often; word order is discarded entirely | **Sequence** — an article is read word-by-word, in order, with the LSTM carrying forward a memory (its cell state) of everything read so far |
| What the model itself learns | One weight per vocabulary word (20,001 numbers total) | Word embeddings + LSTM gate weights + a small classification head (2,044,353 numbers total) |

**Feature engineering vs. feature learning** is the single biggest conceptual difference between
the two. TF-IDF's formula (term frequency × inverse document frequency) was designed by
statisticians decades ago and applies the same fixed calculation regardless of what the text is
about. The LSTM's Embedding layer instead starts with random numbers and *learns* what
representation is useful, purely by seeing which word patterns predict the correct label during
training. This is exactly why Deep Learning is often described as learning its own features,
while traditional Machine Learning typically relies on a human (or a fixed formula) to engineer
them first.

**Bag-of-words vs. sequence modelling** is the second key difference, and it is what motivated
building the LSTM in the first place (`docs/baseline_model_report.md`'s own error analysis
identified this exact gap): TF-IDF cannot tell `"officials did not confirm the claim"` apart from
`"officials did confirm the claim"` — both produce the same set of word counts. An LSTM reads
text in order and can, in principle, use that difference.

**Traditional Machine Learning vs. Deep Learning**, at a broader level, is a distinction of
*how the model arrives at its representation of the input* — a fixed statistical formula vs. a
representation learned end-to-end from data — and, generally, of *how much data and compute*
the approach needs to reach its potential. Section 7 below discusses why, for this particular
dataset and this scope, that extra learning capacity did not translate into a higher score.

---

## 2. Performance Comparison

Full table: `report/tables/model_comparison.csv`. Figure: `report/figures/model_accuracy_comparison.png`.

| Metric | Baseline (TF-IDF + Logistic Regression) | LSTM |
|---|---|---|
| Accuracy | **0.9824** | 0.9720 |
| Precision | **0.9788** | 0.9757 |
| Recall | **0.9893** | 0.9733 |
| F1 Score | **0.9840** | 0.9745 |
| ROC-AUC | **0.9979** | 0.9917 |
| Training Time | **0.25 s** | 599.1 s (~10 min) |
| Inference Time (total, 5,796 articles) | **1.29 s** | 8.26 s |
| Inference Time (per article) | **0.22 ms** | 1.43 ms |
| Model Size (model file only) | **0.16 MB** | 24.57 MB |
| Model Size (total inference artifacts) | **0.90 MB** | 28.69 MB |
| Trainable Parameters / Coefficients | **20,001** | 2,044,353 |
| Vocabulary Size (capped) | 20,000 | 20,000 (of 96,250 distinct words seen) |
| Training Epochs | N/A (single `lbfgs` solver call) | 10 |
| Best Validation Epoch | N/A | 8 (by `val_loss` = 0.1100) |

The baseline wins on every accuracy-style metric, and by a wide margin on every efficiency
metric. This pattern — and whether the accuracy gap is even a *real*, reliable difference or
just test-set noise — is examined statistically in Section 3.

---

## 3. Statistical Comparison

A plain accuracy difference (98.24% vs. 97.20%, a gap of about 1 percentage point) cannot, by
itself, say whether the baseline is *genuinely* the stronger model or whether this particular
5,796-row test set happened to favor it by chance. **McNemar's test** is the correct tool here
because both models were scored on the exact same test rows (a *paired* comparison) —
reused unchanged from `evaluation/significance.py`, the same module used for the Reuters
ablation study.

**Why this test, specifically:** McNemar's test looks only at the *discordant* rows — where the
two models disagree — and ignores every row where they agree (whether both are right or both
are wrong), since agreement rows provide no evidence that one model is better than the other.

**Null hypothesis (H0):** among the rows where the two models disagree, each one is equally
likely to be the one that's correct — i.e., the disagreements look like a fair coin flip, with
no systematic advantage to either model.

**Results:**

| | Count |
|---|---|
| Baseline correct, LSTM wrong | 118 |
| LSTM correct, baseline wrong | 58 |
| Total discordant rows | 176 out of 5,796 |
| McNemar exact p-value | **0.000007** |

**Interpretation:** p = 0.000007 is far below the conventional 0.05 threshold, so H0 is
rejected — the 118-vs-58 split among the disagreement rows is **not** consistent with a coin
flip. This is a genuinely different outcome from the Reuters ablation study
(`docs/reuters_ablation_study.md`), where p = 1.0000 and H0 could not be rejected at all.

**An important nuance — do not confuse "statistically significant" with "a large practical
difference."** Both are true here in a specific, limited sense:

- It **is** statistically significant: with 176 discordant rows to work with, a roughly 2-to-1
  split (118 vs. 58) is unlikely to arise from pure chance, and the test correctly detects that.
- It is **not** a large practical difference: only 176 out of 5,796 test articles (3.0%) are
  discordant at all — the two models agree, one way or the other, on the remaining 97.0%. The
  accuracy gap this produces (about 1 percentage point) is real and statistically defensible, but
  it is not evidence of a dramatically stronger model, only a modestly, reliably stronger one on
  this dataset.

This is a useful, concrete example of a broader statistics lesson worth stating plainly for a
viva: **statistical significance is about confidence that an effect isn't exactly zero, not
about how big that effect is.** A large test set (5,796 rows) can detect a small, genuine
difference with high confidence — which is exactly what happened here — without that difference
being large in absolute terms.

---

## 4. Prediction Agreement Analysis

Full table: `report/tables/prediction_agreement.csv`. Figure: `report/figures/error_overlap.png`.

| Category | Count | Percentage |
|---|---|---|
| Both models correct | 5,576 | 96.20% |
| Both models wrong | 44 | 0.76% |
| Only baseline correct | 118 | 2.04% |
| Only LSTM correct | 58 | 1.00% |

**Do the two models fail on the same articles, or different ones?** Mostly different. Only 44
articles (0.76% of the test set) fool *both* models — genuinely rare, hard cases. The much
larger share of disagreement (176 articles, 3.04%) is split between each model's own exclusive
mistakes, with the baseline holding a roughly 2-to-1 advantage on these disputed rows (matching
the McNemar result above, since these are the same 176 rows).

**What this implies:**

- The two approaches are **not making the same kind of mistake** — if they were, "both wrong"
  would be a much larger share of the 220 total error rows (44 + 118 + 58) than the 20% it
  actually is. Instead, each model has its own distinct blind spot most of the time.
- Because the overlap in *correct* predictions (96.2%) is so high, the bulk of what separates a
  "good" model from a "great" one on this dataset is concentrated in a small, specific set of
  articles — exactly the kind of finding that motivates a closer, qualitative look (Section 5)
  rather than resting on the aggregate metrics alone.
- This pattern is also informative for anyone considering an ensemble (voting between both
  models): since only 0.76% of articles are unrecoverable by either model, an ensemble's
  practical ceiling on this test set would be quite high (up to ~99.24% if every disagreement
  were resolved correctly) — though building one was out of scope for this project, which
  prioritized explaining and comparing two individually-understood models over maximizing a
  combined score.

---

## 5. Error Analysis

Representative examples: `report/tables/comparison_error_examples.csv`. Confusion matrices:
`report/figures/baseline_confusion_matrix.png`, `report/figures/lstm_confusion_matrix.png`.

| | Baseline | LSTM |
|---|---|---|
| True Negatives (Fake→Fake) | 2,549 | 2,540 |
| False Positives (Fake→Real) | 68 | 77 |
| False Negatives (Real→Fake) | 34 | 85 |
| True Positives (Real→Real) | 3,145 | 3,094 |

The LSTM's recall dropped more than its precision relative to the baseline (recall 0.9733 vs.
0.9893, a 1.6-point gap, vs. precision 0.9757 vs. 0.9788, a 0.3-point gap) — it is
disproportionately more likely to call a genuine Real article "Fake" than the baseline is.

### Both models wrong (44 articles — the genuinely hard cases)

Examples: *"Benghazi Survivor On Hillary Clinton: 'I Don't Think She Has A Soul'"* (Fake),
*"Obamas donated less to charities in 2015 as income slipped"* (Real), *"Trump booster
apologizes for Clinton 'blackface' tweet"* (Real), *"Undercutting the Nation State? Chicago
Group Suggests 'Global Cities' Should Run World Affairs"* (Fake). These read as genuinely
ambiguous by *style* regardless of architecture — factual-sounding headlines on politically
charged topics, on both sides of the label. Neither a bag-of-words model nor a sequence model
resolves them, suggesting the underlying difficulty is about the article's content/register, not
a limitation specific to either approach.

### Only the baseline got right — the LSTM's exclusive mistakes (118 articles)

Examples: *"Merkel scolds ally to shield coalition talks from weedkiller row"* (Real), *"Trump
Condemned By Jewish Leaders In Poland After Snubbing Warsaw Ghetto Memorial"* (Fake), *"MITCH
MCCONNELL: The Senate Will Not Take Up Nomination of Merrick Garland"* (Fake), *"Obama warns
Democrats against overconfidence about Clinton victory"* (Real). This matches the pattern already
documented in `docs/lstm_model_report.md`'s own error analysis: several of the LSTM's false
negatives involve **international/foreign-affairs topics with less common names** (Merkel), and
several false positives involve Fake articles written in an unusually **neutral, factual-sounding
register**. A plausible, previously-stated explanation still applies: names and vocabulary that
appear rarely in training are more likely to fall outside the 20,000-word cap or to have a
poorly-learned embedding vector, a limitation more specific to a from-scratch embedding than to
TF-IDF (where a rare word simply gets a small, harmless weight rather than a poorly-positioned
dense vector).

### Only the LSTM got right — the baseline's exclusive mistakes (58 articles)

Examples: *"FLAMING RINO ALERT! LINDSEY GRAHAM: 'TRUMP IS GOING TO KILL MY PARTY'"* (Fake),
*"BREAKING: House Republicans Work To Cut Off Federal Funding For Syrian Refugee Resettlement
Program"* (Fake), *"Fake 'US embassy' Bust in Ghana Exposes Danger of EU Schengen Deal with
Turkey"* (Fake), *"How Trump is Accelerating the Decline of US Global Influence"* (Fake). Several
of these have clickbait-style formatting cues (all-caps phrases, "BREAKING:", exclamation marks)
that TF-IDF also has access to as individual tokens — so this is reported as an **observed
pattern**, not a fully explained mechanism: it is plausible, but not directly verified here, that
the LSTM's sequence-level view of *how* these cues combine with surrounding words gave it an edge
on this specific, smaller set of examples. Given this is only 58 rows (1.0% of the test set),
this pattern should be treated as a lead for further investigation, not a confirmed conclusion.

### Domestic vs. international, named entities, rare vocabulary

Across both the baseline's own error analysis (`docs/baseline_model_report.md`) and the LSTM's
(`docs/lstm_model_report.md`), a consistent theme holds: **both models' errors skew toward
atypical writing style for the article's class**, and the LSTM specifically struggles more with
**international-affairs content and foreign names** than the baseline does — a vocabulary-coverage
issue inherent to a from-scratch (non-pretrained) Embedding layer, not something either model's
architecture can fully avoid on this dataset's moderate size (~27,000 training rows).

---

## 6. Complexity Analysis

Full table: `report/tables/model_comparison.csv`. Figure: `report/figures/model_training_time_comparison.png`.

| Dimension | Baseline | LSTM | Ratio |
|---|---|---|---|
| Training time | 0.25 s | 599.1 s | ~2,440x slower |
| Inference time (per article) | 0.22 ms | 1.43 ms | ~6.4x slower |
| Model size (model file only) | 0.16 MB | 24.57 MB | ~153x larger |
| Model size (total inference artifacts) | 0.90 MB | 28.69 MB | ~32x larger |
| Parameters / coefficients | 20,001 | 2,044,353 | ~102x more |

**Training time** is the starkest difference by far (nearly four orders of magnitude), because
the baseline's `lbfgs` solver converges on a convex optimization problem in under a second, while
the LSTM iterates over the full training set 10 times, each requiring a forward and backward
pass through a 2-million-parameter network.

**Inference complexity** is a smaller but still real gap: a TF-IDF transform plus a sparse
matrix–vector product (baseline) is inherently cheaper per article than tokenizing, padding, and
running a full LSTM forward pass (this notebook's own timed measurement). Both are still fast
enough in absolute terms (well under 2 ms/article) to be practical behind a web API — this
difference matters more for training/iteration speed during development than for a deployed
`/predict` endpoint's user-facing latency.

**Memory usage / model size**: the LSTM's Embedding table alone accounts for 2,000,000 of its
2,044,353 parameters (~98%) — it is, by parameter count, mostly a lookup table of word vectors,
with a comparatively small recurrent/classification component on top. The baseline's vectorizer
(0.73 MB) is actually *larger on disk* than its model file (0.16 MB) because it must store the
full 20,000-word vocabulary and IDF weights — worth noting so the "baseline is small" story isn't
oversimplified to just the model file.

**Deployment complexity**: the baseline needs only scikit-learn + joblib at inference time — a
lightweight, widely-available dependency footprint. The LSTM needs a TensorFlow/Keras runtime, a
meaningfully heavier dependency to install and containerize (see `requirements.txt`'s pinned
`tensorflow==2.21.0`), for a model that scored slightly lower.

**Explainability**: the baseline's Feature Importance analysis (`docs/baseline_model_report.md`)
reads a signed coefficient directly off every vocabulary word — a complete, exact explanation of
what the model learned. The LSTM has no equivalent: explaining one of its predictions would
require additional techniques (e.g. attention visualization) not implemented in this project,
making it materially harder to justify a specific prediction to an end user or examiner.

**Maintainability**: the baseline's single deterministic `.fit()` call has no epoch-to-epoch
variability to monitor; the LSTM's visibly noisy training history
(`report/figures/lstm_training_history.png`) means future retraining runs need to be watched for
instability, and iterating on it (trying a new hyperparameter, debugging an issue) costs ~10
minutes per attempt versus a fraction of a second for the baseline.

---

## 7. Dataset Discussion

Bringing together findings from every previous phase, three separate lines of evidence converge
on the same explanation for **why the simpler model slightly outperformed the more complex one
on this specific dataset:**

**1. The dataset's separating signal is largely surface-level and stylistic, not deeply
semantic.** `docs/label_leakage_analysis.md` found the `subject` column alone reaches 100%
rule-based accuracy and the raw Reuters dateline alone reaches 99.6% — both removed before
modeling (Phase 3). Even after that removal, the Reuters ablation study
(`docs/reuters_ablation_study.md`) found the Real class is, in practice, close to synonymous with
"Reuters wire-service writing style" — a **redundantly encoded** signal (neutral `"X said"`
attribution, weekday datelines, formal titles like `minister`/`spokesman`) that a linear
bag-of-words model is extremely well-suited to detecting, arguably as well as or better than a
sequence model whose main structural advantage (using word order and context) matters more for
subtler, order-dependent distinctions that this dataset does not require much of.

**2. Duplicate removal and preprocessing decisions shaped what remained learnable.** Stage 2
removed 4,317 duplicate/near-duplicate rows and 10 genuinely corrupted rows
(`docs/duplicate_analysis.md`), and Stage 3 stripped the leading Reuters dateline and separated
`baseline_text`/`lstm_text` specifically so each model received appropriately preprocessed input
(stop words and lemmatization removed only for the baseline). None of these choices *caused* the
LSTM's shortfall — both models trained on data cleaned to the same standard — but they do mean
what "signal" was left for either model to find had already been narrowed to genuine,
non-trivial differences between the two classes' writing, which is exactly the situation where a
model's true representational strengths (or lack of reward for extra complexity) show through
most clearly.

**3. Training instability and moderate dataset size limited how much the LSTM's extra capacity
could be realized.** As documented in `docs/lstm_model_report.md`, the training-history plot
shows a visible validation-loss spike at epoch 2 and erratic swings at epochs 9–10 — the LSTM did
not converge as smoothly as it might with further tuning (a lower or scheduled learning rate,
gradient clipping), none of which was pursued, deliberately, per this project's priority on
simplicity and reproducibility over squeezing out maximum accuracy. Separately, ~27,000 training
articles is a reasonable amount of data for a linear model with sparse TF-IDF features, but a
comparatively modest amount for training a ~2-million-parameter embedding + LSTM **from
scratch** (no pretrained embeddings) — deep learning models generally need more data to fully
realize an advantage over a strong linear baseline.

**The central, most important takeaway for the report and viva: more complex models do not
always outperform simpler ones, and this project provides direct, measured evidence of that,
rather than a general claim.** The LSTM was not a wasted effort — building it was the correct
methodological step, because without it, the claim "a baseline is good enough here" could not
have been *checked*, only assumed. The result itself — a well-executed, correctly-built LSTM that
still comes in slightly behind a simple linear model — is a legitimate, well-evidenced finding,
not a failure of implementation.

---

## 8. Strengths & Weaknesses

Full table: `report/tables/model_strengths_weaknesses.csv`.

| Dimension | Logistic Regression Baseline | LSTM |
|---|---|---|
| Accuracy (this dataset) | Higher on every metric | Slightly lower on every metric |
| Interpretability | High — per-word coefficients | Low — no direct equivalent |
| Training speed | Extremely fast (0.25 s) | Slow (~10 min) |
| Inference speed | Fast (0.22 ms/article) | Slower (1.43 ms/article), still real-time-viable |
| Computational cost | Low — single CPU core | Higher — benefits from more compute |
| Scalability to more data | Plateaus as a linear model | More headroom in principle, not yet observed here |
| Robustness (word order/negation) | None — bag-of-words | Structurally capable, unrewarded on this dataset |
| Robustness (rare/foreign vocabulary) | Degrades gracefully | More fragile — cap/embedding-quality limited |
| Training stability | Fully deterministic | Visibly noisy across epochs |
| Ease of deployment | Simple — scikit-learn + joblib | Heavier — needs TensorFlow/Keras runtime |
| Maintainability | Easy to debug/retrain quickly | Harder to debug, slower iteration |

---

## 9. Business Recommendation

**Recommendation: deploy the TF-IDF + Logistic Regression baseline as the production model for
this system, not the LSTM.**

This is **not** a recommendation made simply because the baseline scored a higher number. It is
based on weighing every dimension the business would actually care about:

| Consideration | Favors |
|---|---|
| Performance | Baseline (higher on every metric, and the difference is statistically real per Section 3) |
| Training cost | Baseline (2,440x cheaper to train — matters for retraining as new data arrives) |
| Inference cost | Baseline (6.4x cheaper per prediction, though both are fast enough for this use case) |
| Maintainability | Baseline (deterministic, fast to iterate on, no training-instability to monitor) |
| Interpretability | Baseline (a directly explainable decision — valuable if a user or examiner asks "why was this flagged?") |
| Hardware requirements | Baseline (runs comfortably on a single CPU core; no GPU benefit realized here) |
| Dataset characteristics | Baseline (this dataset's separating signal is largely stylistic/lexical — exactly what a linear bag-of-words model is suited to) |

Every one of these factors points the same direction, which is what makes this a genuinely
balanced recommendation rather than a coincidence of picking whichever model happened to win on
accuracy. Had the LSTM won on accuracy but lost on every other dimension, the recommendation
would have required a real trade-off discussion; here, it does not.

**When would this recommendation change?** If a future dataset had a subtler, less style-driven
distinction between classes (where word order and negation genuinely mattered more), if
significantly more training data became available, or if pretrained embeddings/transformer-based
transfer learning were introduced (out of scope for this project, per the project specification's
"avoid advanced architectures" principle) — any of these could plausibly shift the outcome, and
should be re-evaluated with the same rigor applied here, not assumed.

---

## 10. Threats to Validity

See the dedicated document: `docs/threats_to_validity.md`.

---

## 11. Final Conclusion

**What was achieved:** two complete, independently evaluated models — a TF-IDF + Logistic
Regression baseline and an LSTM — trained on an identical, carefully cleaned dataset, evaluated
on an identical held-out test set, and compared with the same statistical rigor
(`evaluation/significance.py`'s McNemar's test) already established in the Reuters ablation
study. Every deliverable — data cleaning, leakage investigation, baseline, ablation validation,
LSTM, and now this comparison — was backed by real, executed code and honestly reported results.

**What was learned:**
- A simple, fast, fully interpretable linear model can match or exceed a more complex Deep
  Learning model on a dataset whose class-separating signal is largely surface-level/stylistic.
- Statistical significance and practical significance are different questions — this comparison
  found a *statistically significant* baseline advantage (p = 0.000007) that is nonetheless a
  *modest* one in absolute terms (~1 accuracy point, 3.0% of test rows disagreeing).
- The two models fail on largely different articles (only 0.76% fool both), even though their
  aggregate accuracy is close — aggregate metrics alone can hide meaningfully different error
  profiles.

**Why the baseline performed better:** primarily because this dataset's Fake/Real distinction is
dominated by source-style cues (Reuters wire-service conventions vs. blog/aggregator writing)
that a bag-of-words linear model detects very efficiently, combined with the LSTM's from-scratch
embedding needing more data than was available to fully realize its structural advantages, and
some observed training instability that further tuning (out of this project's scope) might
address.

**What the LSTM contributed regardless:** it directly tested whether the baseline's
"good-enough" result was actually good enough, rather than assuming so; its different error
profile (Section 5) revealed a genuine, previously-undocumented weakness around
international/foreign-name articles; and it stands as a complete, correctly-implemented reference
architecture for word-order-sensitive text classification, ready to be revisited if a future
dataset or added tuning budget calls for it.

**Lessons learned during experimentation:** reusing the exact same seeded split
(`training/split.py`) and the same evaluation/statistical-testing code across every model was
what made this comparison trustworthy in the first place — a fair comparison has to be designed
into the pipeline from the start, not bolted on afterward. Honest reporting of a negative result
(the LSTM losing) turned out to be more informative, and more defensible in a viva, than any
attempt to tune until the LSTM "won."

**Possible future work:** pretrained word embeddings (GloVe/Word2Vec) or a pretrained transformer
encoder (e.g. DistilBERT) via transfer learning, to test whether the LSTM's shortfall is really a
data-scale issue rather than an architectural one; a simple ensemble of both models, given the
high theoretical ceiling identified in Section 4; hyperparameter tuning (learning rate scheduling,
gradient clipping) to address the observed training instability; and — most importantly for
external validity — evaluation on a second, independently-sourced fake-news dataset, to test
whether either model's ~97-98% accuracy reflects genuine generalization or is partly an artifact
of this specific dataset's two-source structure (see `docs/threats_to_validity.md`).

---

## 12. Report Assets Generated in This Phase

- `report/tables/model_comparison.csv` — full metrics/complexity comparison table
- `report/tables/model_strengths_weaknesses.csv` — qualitative strengths/weaknesses table
- `report/tables/prediction_agreement.csv` — 4-way agreement breakdown
- `report/tables/comparison_error_examples.csv` — representative titles from each disagreement bucket
- `report/figures/model_accuracy_comparison.png`
- `report/figures/model_training_time_comparison.png`
- `report/figures/error_overlap.png`

`evaluation/experiments.csv` was **not** modified in this phase — its schema records training
runs, and no model was retrained here; the statistical comparison (McNemar's test) and agreement
analysis are recorded instead in the tables above and in this report, consistent with how the
Reuters ablation study's own McNemar result was reported in prose/tables rather than as a new
experiment-log row.
