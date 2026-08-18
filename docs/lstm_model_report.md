# LSTM Model Report

**Phase:** Phase 5 — Deep Learning (LSTM)
**Produced by:** `notebooks/04_lstm_training.ipynb`, calling `training/lstm.py` and `evaluation/`
**Artifacts:** `models/lstm/` (trained model, tokenizer, label map, metadata)

This document is written for direct inclusion in the BCA final report (Section: Deep Learning
Model). It covers why LSTM was selected, its advantages and limitations relative to the
baseline, the training process, hyperparameters, final results, and an honest discussion —
including a result that did not go the way a first guess might expect.

---

## Why LSTM was selected

The baseline's own Error Analysis (`docs/baseline_model_report.md`) identified a specific,
mechanistic limitation: TF-IDF has no concept of word order, so it cannot distinguish
`"officials did not confirm the claim"` from `"officials did confirm the claim"`. LSTM was
chosen specifically to address that gap — it reads text as an ordered sequence and maintains a
memory (its cell state) across the sequence, giving it the structural capacity to use word
order and negation that a bag-of-words model lacks entirely.

LSTM was chosen over a plain RNN because plain RNNs suffer from the **vanishing gradient
problem** — the training signal connecting distant words to the final prediction shrinks
exponentially over many time steps. With articles routinely running to hundreds of words
(`MAX_SEQUENCE_LENGTH = 300`), a plain RNN would struggle to learn from anything beyond the
first few dozen words. LSTM's three gates (forget, input, output) give it a more direct path
for preserving information across long sequences.

LSTM was chosen over BiLSTM, Attention, and Transformer architectures per this phase's explicit
scope and the project specification's "avoid advanced architectures unless there is a clear academic benefit"
principle — a plain, unidirectional LSTM can be explained gate-by-gate in a viva in a way a
Transformer's self-attention mechanism would be a much harder claim to fully defend at BCA
level.

---

## Architecture and hyperparameters

```
Embedding(input_dim=20,000, output_dim=100)
        ↓
SpatialDropout1D(0.2)
        ↓
LSTM(64 units)
        ↓
Dropout(0.3)
        ↓
Dense(32, activation="relu")
        ↓
Dense(1, activation="sigmoid")
```

| Hyperparameter | Value | Where set |
|---|---|---|
| Vocabulary size (cap) | 20,000 | `config/settings.py: VOCAB_SIZE` |
| Actual distinct words in training data | 96,250 | Reported by the fitted Tokenizer |
| Max sequence length | 300 | `MAX_SEQUENCE_LENGTH` |
| Embedding dimension | 100 | `EMBEDDING_DIM` |
| LSTM units | 64 | `LSTM_UNITS` |
| Spatial dropout rate | 0.2 | `SPATIAL_DROPOUT_RATE` |
| Dropout rate | 0.3 | `DROPOUT_RATE` |
| Dense layer units | 32 | `DENSE_UNITS` |
| Batch size | 64 | `LSTM_BATCH_SIZE` |
| Max epochs | 10 | `LSTM_MAX_EPOCHS` |
| Early stopping patience | 3 (on `val_loss`) | `EARLY_STOPPING_PATIENCE` |
| Optimizer | Adam | `training/lstm.py` |
| Loss | Binary crossentropy | `training/lstm.py` |
| Total trainable parameters | 2,044,353 | Model summary |

Total parameter count (~2.04M) is dominated by the Embedding layer (2,000,000 of the 2,044,353,
i.e. ~98%) — the vast majority of what this model learns is *word representations*, not the
LSTM's own recurrent weights (42,240) or the classification head (2,113).

---

## Training process

Data: `dataset/processed/03_preprocessed.csv`, `lstm_text` column, split via the **exact same**
`training/split.py` function and seed used for the baseline — 27,046 train / 5,796 validation /
5,796 test rows.

Training ran the full 10 allowed epochs (`LSTM_MAX_EPOCHS`) — EarlyStopping's patience of 3 was
never triggered, because validation loss kept finding new lows intermittently (epoch 8's 0.1100
was the best; epochs 9–10 didn't improve on it, but that's only 2 non-improving epochs, one
short of the patience threshold when training hit its epoch cap). `restore_best_weights=True`
rolled the final model back to **epoch 8's** weights regardless.

| | Value |
|---|---|
| Training time | 599 seconds (~10 minutes) |
| Epochs completed | 10 / 10 (max reached, early stopping not triggered) |
| Best validation loss | 0.1100 (epoch 8) |
| Best validation accuracy | 0.9703 (epoch 9 — see note below) |

**A small, honest note on "best" metrics:** the lowest validation *loss* (0.1100) and the
highest validation *accuracy* (0.9703) occurred on different epochs (8 and 9 respectively) —
tracked independently, as is standard practice. Since `EarlyStopping` monitors `val_loss`
specifically, the model that was actually restored and saved is **epoch 8's** version, whose
own validation accuracy was 0.9674, not 0.9703. This is a normal, expected nuance (loss and
accuracy don't have to peak on the same epoch), not an inconsistency.

The training-history plot (`report/figures/lstm_training_history.png`) shows a visibly **noisy**
training run: validation loss spikes sharply at epoch 2 (0.61) before recovering, and both
training loss and accuracy swing considerably in the final two epochs (training loss jumps from
0.12 back up to 0.31 at epoch 10, while validation loss stays low and stable at 0.11–0.12). This
instability, rather than a smooth monotonic improvement, is itself worth noting honestly in the
report — it reflects the real, sometimes-bumpy nature of training a moderately-sized LSTM on a
moderately-sized dataset with default hyperparameters and no learning-rate tuning, and is
exactly why `restore_best_weights=True` matters: without it, the final epoch's clearly worse
training-loss spike could have been what got shipped.

---

## Final results (test set, 5,796 articles, never used in training or tokenizer fitting)

| Metric | LSTM | Official Baseline (for reference) |
|---|---|---|
| Accuracy | 0.9720 | 0.9824 |
| Precision | 0.9757 | 0.9788 |
| Recall | 0.9733 | 0.9893 |
| F1 Score | 0.9745 | 0.9840 |
| AUC | 0.9917 | 0.9979 |
| Training time | 599 s | 0.25 s |

Confusion matrix: 2,540 true negatives, 77 false positives, 85 false negatives, 3,094 true
positives (`report/figures/lstm_confusion_matrix.png`). Full per-class report:
`report/tables/lstm_classification_report.csv`.

*(Baseline figures shown for context only, read directly from `models/baseline/metadata.json`
and `evaluation/experiments.csv` — not recomputed here. A full, statistically-grounded
comparison, following the same McNemar's-test approach used in `docs/reuters_ablation_study.md`,
is Phase 6's job, not this report's.)*

---

## Discussion — an honest result

**On every metric, the LSTM scored slightly lower than the simple TF-IDF + Logistic Regression
baseline.** This is reported plainly, not minimized: it would have been easy to only report the
LSTM's own ~97% number and let it look impressive in isolation, but this project's whole
methodology (Phase 4's feature importance, the Reuters ablation study) has been built around
checking assumptions rather than accepting a good-looking number at face value. The same
standard applies here.

**Why might this have happened?** Three plausible, non-exclusive explanations, consistent with
findings already documented elsewhere in this project:

1. **The dataset's separating signal is largely surface-level/stylistic**
   (`docs/reuters_ablation_study.md` found the Real class is essentially "Reuters wire style,"
   with multiple redundant cues like `said`, weekday datelines, and formal titles). A linear
   bag-of-words model is *very* good at picking up exactly this kind of surface-level,
   single-word-level signal — arguably as good as, or better suited to it than, a sequence
   model whose main advantage (using word order/context) matters more for subtler,
   order-dependent distinctions this dataset may not require much of.
2. **Training instability.** The visibly noisy training-history plot (large validation loss
   spike at epoch 2, erratic swings at epochs 9–10) suggests the LSTM did not converge as
   smoothly as it might with further tuning (a lower or scheduled learning rate, gradient
   clipping, or more epochs with a more patient schedule) — none of which was pursued here,
   deliberately, per this phase's stated priority on simplicity and reproducibility over
   squeezing out maximum accuracy.
3. **Moderate dataset size relative to model capacity.** ~27,000 training articles is a
   reasonable amount for a linear model with a few hundred thousand sparse TF-IDF features, but
   is a comparatively modest amount of data for training a ~2-million-parameter embedding +
   LSTM from scratch (rather than using pretrained embeddings) — deep learning models generally
   benefit from larger datasets to fully realize an advantage over strong linear baselines.

**Does this mean the LSTM was a failure, or the wrong choice?** No. Building it was still the
right methodological step — Phase 4 already established that a baseline was needed specifically
so a claim like "the LSTM improves things" could be *checked*, not assumed. The result here is
itself informative: for this dataset, given this scope of tuning, the added architectural
complexity of an LSTM did not translate into a measurable accuracy advantage. That is a
legitimate, defensible finding for a BCA report, consistent with the project specification's own framing of
this risk from the very start of the project.

## Error analysis: how the LSTM's mistakes compare to the baseline's

Reusing `evaluation/error_analysis.py` unchanged, the LSTM's misclassified test examples show a
**qualitatively different** pattern from the baseline's (`docs/baseline_model_report.md`):

- **False positives** (Fake called Real): titles like *"Trump Condemned By Jewish Leaders In
  Poland After Snubbing Warsaw Ghetto Memorial"* and *"MITCH MCCONNELL: The Senate Will Not Take
  Up Nomination of Merrick Garland"* — Fake articles written in a comparatively neutral,
  factual-sounding register, similar in spirit to the baseline's false positives.
- **False negatives** (Real called Fake): titles like *"Merkel scolds ally to shield coalition
  talks from weedkiller row,"* *"BOJ governor Kuroda warns against policies unwinding free
  trade,"* and *"Danish fishermen could be hit hard by Brexit: research report"* — genuine
  Reuters articles on **international/foreign-affairs topics** involving less common
  names/terms (Merkel, Kuroda, Danish). This is a different pattern from the baseline's false
  negatives (which skewed toward domestic, politically-charged/opinion-adjacent topics like the
  Clinton Foundation). A plausible explanation: foreign names and less-frequent
  international-affairs vocabulary are more likely to fall outside, or near the edge of, the
  20,000-word vocabulary cap, or to have been seen too rarely during training for the Embedding
  layer to have learned a strong, well-positioned vector for them — a limitation more specific
  to a from-scratch embedding approach than to a bag-of-words model, where a rare word simply
  gets a proportionally small TF-IDF weight rather than a poorly-learned dense vector.

This difference is a genuinely useful, concrete finding for Phase 6: **the two models don't just
differ in aggregate accuracy, they fail on different kinds of articles** — worth a direct,
side-by-side investigation once both models' complete error sets can be cross-referenced.

---

## Advantages of the LSTM (relative to the baseline)

- **Structural capacity for word order and negation** — even though this particular dataset
  didn't reward it with a higher score, the capability itself is real and could matter more on
  a dataset with subtler, less style-driven distinctions between classes.
- **Learned representations** — the Embedding layer builds its own notion of word similarity
  from data, rather than relying on a fixed frequency-based formula.

## Limitations of the LSTM (as observed here, not just in theory)

- **Did not outperform the baseline** on this dataset, at this scope of tuning — the central,
  headline limitation of this run.
- **Noisier, less predictable training** — visible directly in the training-history plot; the
  baseline's training, by contrast, is a single deterministic `.fit()` call with no epoch-to-epoch
  variability to manage at all.
- **Orders of magnitude slower to train** — 599 seconds vs. 0.25 seconds for the baseline, a
  real practical cost for iteration speed during development.
- **Less interpretable** — there is no direct LSTM equivalent of the baseline's per-word
  coefficients (`docs/baseline_model_report.md`'s Feature Importance section); explaining *why*
  the LSTM made a specific prediction would require additional techniques (e.g. attention
  visualization, which this architecture doesn't include) not implemented in this project.
- **More vulnerable on rare/foreign vocabulary**, per the error analysis above — a from-scratch
  embedding table learns weaker representations for infrequent words than a TF-IDF score, which
  degrades gracefully by weighting rare words low rather than representing them poorly.

---

## What this report deliberately does not conclude

This is **not** a claim that "LSTM is worse than Logistic Regression" as a general rule — only
that, on this specific dataset, with this specific (deliberately simple, untuned) configuration,
the baseline performed slightly better. A different dataset, more training data, pretrained
embeddings, or hyperparameter tuning could change this outcome — none of which was in scope for
this phase, whose priority was correctness, reproducibility, and educational clarity over
maximizing accuracy.

No formal statistical comparison (e.g. McNemar's test between the baseline's and LSTM's test
predictions) was performed in this report — that belongs to Phase 6, once both models' results
can be examined side by side with the same rigor already applied in
`docs/reuters_ablation_study.md`.
