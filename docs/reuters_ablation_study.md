# Reuters Leakage Validation — Ablation Study

**Status:** additional validation experiment. **Does not replace the official baseline** in
`models/baseline/`, which remains the project's baseline for all future comparisons (Phase 5/6).
**Produced by:** `notebooks/03b_reuters_ablation_experiment.ipynb`.

This document is written for direct inclusion in the BCA final report (Section: Baseline
Validation / Robustness Check).

---

## Background

The official baseline (`docs/baseline_model_report.md`) reached ~98.24% accuracy, but its
feature-importance analysis found "reuters" as the single strongest word predicting "Real" —
despite Stage 2 preprocessing already stripping the leading `"(Reuters) - "` dateline. Further
investigation found **5,186 of 38,638 rows (13.4%)** still contained the word "reuters"
elsewhere in the article body (secondary datelines, correction notices, in-body citations).

Rather than treating a suspiciously strong feature as a good sign, this experiment tests it
directly: **remove every remaining occurrence of "reuters" and re-measure.**

## Experimental design

Only one variable changes. Everything else is reused unmodified:

| Element | Official baseline | Ablation experiment |
|---|---|---|
| Text column | `baseline_text` | `baseline_text` with every `"reuters"` token removed |
| Train/val/test split | `training/split.py`, seed 42 | Identical (same function, same seed, same row order) |
| TF-IDF parameters | `config/settings.py` (unigrams, 20,000 max features) | Unchanged |
| Logistic Regression parameters | `config/settings.py` defaults | Unchanged |
| Model artifacts | `models/baseline/` | Not persisted — this is a one-off validation run, not a candidate model |

Because the row order and `label` column are untouched, re-running the identical split function
with the identical seed reproduces the **exact same 5,796 test rows** the official baseline was
scored on — confirmed directly in the notebook by reloading the official baseline's saved model
and verifying its re-derived test metrics match `models/baseline/metadata.json` exactly before
any new training happened.

---

## 1. How much did performance change?

| Metric | Official Baseline | Reuters-Removed Experiment | Absolute Difference | % Difference |
|---|---|---|---|---|
| Accuracy | 0.98240 | 0.98257 | +0.00017 | +0.018% |
| Precision | 0.97884 | 0.97974 | +0.00090 | +0.092% |
| Recall | 0.98930 | 0.98868 | −0.00063 | −0.064% |
| F1 Score | 0.98404 | 0.98419 | +0.00014 | +0.015% |
| AUC | 0.99793 | 0.99790 | −0.00003 | −0.003% |
| Training time (s) | 0.246 | 0.227 | −0.019 | −7.7% |

(Full table: `report/tables/reuters_ablation_comparison.csv`.)

**Every classification metric changed by less than one-tenth of one percentage point**, in
both directions — two metrics even improved slightly without "reuters." Training time differs
by ~19ms, well within normal run-to-run noise for a sub-second training job, not a meaningful
effect.

## 2. Is the change statistically meaningful?

**No.** Both models were scored on the identical 5,796 test articles, making this a paired
comparison — the correct test is **McNemar's test**, which looks only at rows where the two
models disagree.

- Discordant predictions: **9 out of 5,796** (4 where the baseline was right and the ablation
  wrong; 5 where the reverse was true).
- McNemar's exact p-value: **1.0000**.

A p-value this high means the 9 disagreements are indistinguishable from what you'd expect by
pure chance if the two models were equally accurate — there is no statistical evidence that
removing "reuters" changed the model's real performance at all.

## 3. Does the model still perform well without Reuters?

**Yes — essentially identically well.** 98.26% accuracy, 98.42% F1, 0.9979 AUC. Whatever the
model is learning, it does not collapse, or even measurably weaken, when this one token is
entirely removed.

## 4. Does this suggest Reuters was acting as label leakage?

**A nuanced yes, with an important distinction.** The *word* "reuters" was not functioning as a
*load-bearing, irreplaceable* shortcut — the model doesn't need that specific token to reach
~98%. But feature importance on the ablated model (Step 8 in the notebook) shows the model
simply promotes its **next-strongest, already-present signals** to fill the gap: `said` becomes
the new #1 (its coefficient actually goes up slightly, 18.40 → 18.52), and weekday names
(`wednesday`, `tuesday`, `thursday`, `friday`, `monday`) — already present in the original top
10 — move up. **No genuinely new word enters the top 20.**

This means the leakage was never concentrated in one token to begin with — it was already
**redundantly encoded** across several correlated wire-service writing conventions (neutral
`"X said"` attribution, day-of-week datelines, formal titles like `minister`/`representative`/
`spokesman`). Removing "reuters" didn't remove the leakage; it just revealed that the same
underlying signal was present in multiple other places at once.

## 5. What does this tell us about the dataset?

**The Real class is not just "genuine news" — it is, almost without exception, Reuters wire
copy specifically**, and the Fake class is, stylistically, a different kind of writing entirely
(blog/aggregator conventions: `via`, `video`, `image`, `featured`, `watch`, `breaking`). The
model's ~98% accuracy is real and reproducible, but it is measuring **"does this read like
Reuters wire style vs. this dataset's blog style,"** which correlates extremely strongly with
the Fake/Real label *in this dataset* — without necessarily measuring "is this specific claim
true." This is a structural property of how the dataset was assembled (two source pipelines
with very different house styles), not a bug introduced by this project's preprocessing.

## 6. Would you recommend additional Reuters removal before training the LSTM?

**No — not more token-by-token removal.** This experiment provides direct, empirical evidence
that hunting down and removing more individual wire-service tokens would very likely repeat
this result: the model would simply promote its next-next-strongest correlated signal, with no
statistically meaningful change in outcome. That effort has a demonstrated low return.

**What is recommended instead:**

1. **Document this as an acknowledged structural limitation** of the dataset in the final
   report's Limitations section — not something "fixed" by more preprocessing, but a property
   of the data itself that any model trained on it (baseline or LSTM) needs to be understood
   against.
2. **Run the equivalent check on the LSTM once it exists** (Phase 5/6): does removing "reuters"
   from `lstm_text` change *its* performance meaningfully? Since the LSTM reads text from the
   same two source pipelines, it likely has access to the same redundant style cues (`said`,
   weekday names are not stop words removed from `lstm_text` either) — this should be verified,
   not assumed, once there is an LSTM to test.
3. **Do not chase a "perfectly clean" dataset for a BCA-scope project.** A genuinely
   leakage-free evaluation would require entirely different real/fake sources with comparable
   writing styles — out of scope here, but worth naming explicitly as a direction for future
   work rather than silently ignoring.

---

## Feature Importance: Before vs. After

Full plot: `report/figures/reuters_ablation_top_features.png` (compare against
`report/figures/baseline_top_features.png`).

- **Fake-indicating words: completely unchanged** (`via`, `video`, `image`, `gop`, `read`,
  `featured`, `hillary`, `mr`, `even`, `watch`, ...) — expected, since "reuters" was never a
  Fake-side signal.
- **Real-indicating words: reshuffled, not replaced.** `said` moves from #2 to #1; weekday names
  move up by one or two ranks each; no new word appears in the top 20 that wasn't already
  present in the baseline's top 20.

This "reshuffle, don't replace" pattern is itself the core piece of evidence for the answer to
Question 4 above: it directly shows redundant encoding rather than a single removable leak.

---

## Educational Notes

### What is an ablation study?

Removing one specific component of a system, keeping everything else fixed, and re-measuring —
named after the surgical term for removing tissue to observe the effect. If removing the part
changes the outcome a lot, that part mattered; if removing it barely matters, it wasn't doing
as much independent work as its individual importance score suggested.

### Why is this experiment scientifically useful?

A large Logistic Regression coefficient on "reuters" showed *correlation* with the label, but
correlation alone can't distinguish "the model critically depends on this one word" from "the
model has many correlated signals and this is merely the strongest individual one." Only
actually removing the word and re-measuring can tell those two stories apart — this is the
scientific method applied directly to a trained model: form a hypothesis, change exactly one
variable, measure the result.

### Why is verifying potential leakage better than simply reporting the highest possible accuracy?

Because this project has already shown, twice, that a very high accuracy number can be an
illusion (`subject` alone reached 100% accuracy; the Reuters dateline alone reached 99.6% — see
`docs/label_leakage_analysis.md`). An unexamined "98% accuracy" invites the reasonable viva
question "are you sure that's not another shortcut?" — this experiment answers that question
with direct evidence (a controlled ablation, a paired statistical test, and a before/after
feature-importance comparison) instead of an assumption. That is what turns a good-looking
number into a *defensible* one.

---

## What this study deliberately does not conclude

This is **not** a finding that the baseline is invalid or should be discarded — the opposite:
the model performs equally well without the most suspicious single feature, which is reassuring
evidence about its robustness to that one specific concern. It **is** a finding that the
dataset has a deeper, structural source-style confound that no single-token fix resolves, which
should be documented as a known limitation rather than treated as solved.

The official baseline in `models/baseline/` remains unchanged and remains the project's
baseline for Phase 5/6 comparisons.
