# Viva Preparation Notes

This is a **living document** — a new section is appended for every completed project phase.
Earlier phases' content is never overwritten. By the end of the project, this file should be a
complete, self-contained viva revision guide.

---

# Phase 4 — Baseline Machine Learning

## Likely Viva Questions

1. What is Machine Learning?
2. Why is this called a baseline model?
3. What is TF-IDF?
4. Why can't Logistic Regression use raw text?
5. Why convert text into vectors?
6. What is feature extraction?
7. What is vectorization?
8. What is Logistic Regression?
9. Why was Logistic Regression selected for this project?
10. Why not Naive Bayes?
11. Why not Decision Trees (or Random Forest)?
12. What is a train/test split?
13. Why use stratified sampling?
14. What is overfitting?
15. What is underfitting?
16. What is Precision?
17. What is Recall?
18. What is F1 Score?
19. What is a Confusion Matrix?
20. What are False Positives and False Negatives?
21. What is an ROC curve, and what does AUC mean?
22. What does it mean for a TF-IDF vector to be "sparse"?
23. What is regularization, and does Logistic Regression use it?

## Project-Specific Questions

24. Why was the Reuters prefix removed before training?
25. What is label leakage, and where did it appear in this project?
26. Why was the `subject` column removed entirely?
27. Why were duplicate articles removed before splitting the data?
28. Why combine title and article body into one field?
29. Why create multiple staged processed datasets instead of one script?
30. Why build a baseline before implementing the LSTM?
31. Why do `baseline_text` and `lstm_text` differ (stop words/lemmatization)?
32. What did the feature-importance analysis reveal, and why does it matter?

---

## Short Answers

### 1. What is Machine Learning?
**Short answer:** Programming a computer to learn patterns from data, rather than being given
explicit rules for every case.
**Expanded:** Traditional programming is `rules + data → output` (a human writes the logic).
Machine Learning flips this: `data + output (examples) → rules` — the algorithm works out the
pattern itself from labeled examples.
**Example:** Instead of writing "if the text contains 'BREAKING' and three exclamation marks,
call it fake," this project shows the model 38,638 labeled examples and lets it learn which
word patterns correlate with each label.

### 2. Why is this called a baseline model?
**Short answer:** It's the simplest reasonable approach, used as a reference point the LSTM
must beat to justify its extra complexity.
**Expanded:** "Baseline" doesn't mean "weak" — a *strong* baseline is the point. Without one,
there's no way to know if the LSTM's result (Phase 5) is actually good, or just "a number."
**Example:** If the LSTM scores 90% and the baseline scores 98%, that's important information —
it would mean the added complexity wasn't worth it for this dataset.

### 3. What is TF-IDF?
**Short answer:** Term Frequency × Inverse Document Frequency — a way to turn text into numbers
that rewards words that are frequent *in one article* but rare *across all articles*.
**Expanded:** TF = how often a word appears in this article. IDF = how rare that word is across
the whole dataset (common words like "the" get a low IDF). Multiplying them means common
filler words score low everywhere, while distinctive words score high in the articles where
they actually appear.
**Example:** "election" appearing 8 times in one article gets a high TF-IDF score; "the"
appearing 40 times gets a near-zero score because it appears in almost every article (low IDF).

### 4. Why can't Logistic Regression use raw text?
**Short answer:** It only does arithmetic on numbers — text has to be converted to numbers
first.
**Expanded:** Every classical ML algorithm computes weighted sums, sigmoid functions, distances,
or similar numeric operations. A string like `"Trump said Wednesday"` is not something you can
multiply by a weight.
**Example:** TF-IDF converts that string into a row of ~20,000 numbers (mostly zero) before
Logistic Regression ever sees it.

### 5. Why convert text into vectors?
**Short answer:** So a fixed-size, numeric representation of variable-length text can be fed
into a mathematical model.
**Expanded:** Articles have different lengths, but ML algorithms generally need a fixed-size
input. Vectorization solves both problems at once — text → numbers, and variable length →
fixed length (one entry per vocabulary word).
**Example:** A 50-word article and a 5,000-word article both become a vector of exactly 20,000
numbers (the vocabulary size).

### 6. What is feature extraction?
**Short answer:** The general process of turning raw data into the measurable inputs
("features") a model actually learns from.
**Expanded:** TF-IDF is one specific *feature extraction* technique for text. A "feature" here
is one vocabulary word's score in a given article.
**Example:** The feature "reuters" has a high value for an article that mentions Reuters, zero
for one that doesn't.

### 7. What is vectorization?
**Short answer:** The specific act of converting text into a numeric vector — the mechanical
step that produces the features above.
**Expanded:** Vectorization and feature extraction are closely related terms; in this project,
"vectorization" refers specifically to `TfidfVectorizer.transform()` turning a string into a
row of TF-IDF numbers.
**Example:** `vectorizer.transform(["some article text"])` returns one sparse row vector.

### 8. What is Logistic Regression?
**Short answer:** A classification algorithm that computes a weighted sum of input features,
passes it through a sigmoid function, and outputs a probability between 0 and 1.
**Expanded:** Despite the name, it's used for classification, not continuous-value regression.
The sigmoid function `1 / (1 + e^-x)` squashes any real number into the 0–1 range, which is
interpreted as "probability of being Real."
**Example:** An output of 0.93 means the model is 93% confident the article is Real; above 0.5
it predicts Real, otherwise Fake.

### 9. Why was Logistic Regression selected for this project?
**Short answer:** It's fast, interpretable (one weight per word), and produces a probability
directly — three properties this project specifically needed.
**Expanded:** The probability output feeds the ROC curve and the planned "confidence score" in
the frontend; the per-word weights directly power the Feature Importance analysis.
**Example:** See `docs/baseline_model_report.md`'s comparison table against Naive Bayes,
Decision Trees, Random Forest, and SVM.

### 10. Why not Naive Bayes?
**Short answer:** A reasonable alternative, but it assumes even stronger word-independence and
gives less well-calibrated probabilities.
**Expanded:** Naive Bayes computes class probabilities assuming every word is fully independent
of every other — Logistic Regression's weights are learned jointly, which usually gives a
better-calibrated probability estimate for a given fixed feature set.
**Example:** Not chosen here, but would be a legitimate second experiment to add later if
comparing multiple classical baselines.

### 11. Why not Decision Trees (or Random Forest)?
**Short answer:** Decision Trees overfit easily on thousands of sparse TF-IDF features; Random
Forest is harder to interpret and slower, blurring the "simple baseline" purpose.
**Expanded:** A single Decision Tree can create very deep, narrow splits chasing noise in a
high-dimensional sparse matrix without careful pruning. Random Forest fixes overfitting but
loses the clean "one weight per word" interpretability Logistic Regression provides.
**Example:** Logistic Regression's coefficients directly produced the Top-20-words chart;
Random Forest has no equivalent single number per word.

### 12. What is a train/test split?
**Short answer:** Dividing the dataset so the model is evaluated on data it never saw during
training.
**Expanded:** A model evaluated only on its own training data can simply memorize answers,
making its score meaningless as an estimate of real-world performance.
**Example:** This project splits 70% train / 15% validation / 15% test — metrics are always
reported on the untouched 15% test set.

### 13. Why use stratified sampling?
**Short answer:** To keep the same Fake/Real ratio in every split, instead of leaving it to
chance.
**Expanded:** A plain random split could accidentally put more of one class in the test set,
making test metrics noisier and less comparable across runs.
**Example:** Verified in the notebook: train, validation, and test all end up ~54.8% Real,
matching the full dataset almost exactly.

### 14. What is overfitting?
**Short answer:** When a model learns the training data too specifically (including its noise),
performing well on training data but poorly on new data.
**Expanded:** An overfit model has essentially memorized quirks of the training set rather than
learning generalizable patterns.
**Example:** A model that reached 100% training accuracy but only 70% test accuracy would be a
clear sign of overfitting; this project's baseline shows no such gap (~98% on unseen test data).

### 15. What is underfitting?
**Short answer:** When a model is too simple to capture the real pattern in the data, performing
poorly on both training and test data.
**Expanded:** The opposite failure mode of overfitting — the model hasn't learned enough, not
too much.
**Example:** If Logistic Regression had scored only ~55% (barely better than guessing the
majority class), that would suggest underfitting, or that the features (TF-IDF) don't capture
the pattern well.

### 16. What is Precision?
**Short answer:** Of everything the model labeled "Real," what fraction actually was Real?
**Expanded:** `Precision = True Positives / (True Positives + False Positives)`. High precision
means few false alarms.
**Example:** This project's precision is 0.9788 — of every article the model called Real, 97.88%
truly were.

### 17. What is Recall?
**Short answer:** Of everything that actually was Real, what fraction did the model correctly
catch?
**Expanded:** `Recall = True Positives / (True Positives + False Negatives)`. High recall means
few misses.
**Example:** This project's recall is 0.9893 — the model caught 98.93% of all genuinely Real
articles.

### 18. What is F1 Score?
**Short answer:** The harmonic mean of Precision and Recall — one number balancing both.
**Expanded:** A model that is very precise but has poor recall (or vice versa) gets penalized;
F1 only stays high if both are reasonably high together.
**Example:** This project's F1 is 0.9840, close to both its precision and recall, showing
neither was sacrificed for the other.

### 19. What is a Confusion Matrix?
**Short answer:** A table showing counts of correct and incorrect predictions, broken down by
which kind of mistake was made.
**Expanded:** For binary classification it's a 2×2 grid: true negatives, false positives, false
negatives, true positives.
**Example:** This project's test-set confusion matrix: 2,549 true negatives, 68 false
positives, 34 false negatives, 3,145 true positives (`report/figures/baseline_confusion_matrix.png`).

### 20. What are False Positives and False Negatives?
**Short answer:** A false positive is a Fake article wrongly called Real; a false negative is a
Real article wrongly called Fake.
**Expanded:** Which type matters more depends on the application — for fake news detection,
false negatives (missing genuine news as "fake") could be seen as more damaging to user trust
than false positives, though both were investigated equally here.
**Example:** See `notebooks/03_baseline_model.ipynb`'s Error Analysis section for 5 real
examples of each, with an explanation of why each was misclassified.

### 21. What is an ROC curve, and what does AUC mean?
**Short answer:** A curve plotting True Positive Rate against False Positive Rate as the
decision threshold changes; AUC (Area Under the Curve) summarizes it in one number from 0.5
(random) to 1.0 (perfect).
**Expanded:** Unlike accuracy/precision/recall (which use one fixed 0.5 threshold), the ROC
curve shows how the model behaves across *all* possible thresholds — useful for judging the
quality of the underlying probability, not just one cutoff.
**Example:** This project's AUC is 0.9979, very close to a perfect classifier.

### 22. What does it mean for a TF-IDF vector to be "sparse"?
**Short answer:** Almost all entries in each article's vector are zero, because most vocabulary
words don't appear in any given article.
**Expanded:** With a 20,000-word vocabulary, a typical ~360-word article can only have a few
hundred non-zero entries at most — over 99% of the matrix is zero.
**Example:** Verified directly in the notebook by computing sparsity percentage on the training
matrix.

### 23. What is regularization, and does Logistic Regression use it?
**Short answer:** A penalty added during training that discourages excessively large weights,
helping prevent overfitting.
**Expanded:** Scikit-learn's `LogisticRegression` applies L2 regularization by default,
controlled by the parameter `C` (smaller `C` = stronger regularization). This project used the
default `C=1.0` — no tuning was performed, consistent with keeping this a simple baseline.
**Example:** Recorded in `models/baseline/metadata.json` for full reproducibility.

---

### 24. Why was the Reuters prefix removed before training?
**Short answer:** 99.2% of Real articles open with a `"(Reuters) - "` dateline that almost no
Fake articles have — a shortcut unrelated to truthfulness.
**Expanded:** Without removing it, the model could reach ~99.6% accuracy just by detecting one
substring, never learning genuine language patterns (`docs/label_leakage_analysis.md`).
**Example:** Feature importance still found "reuters" as the #1 predictor of Real even *after*
removal, because 13.4% of articles mention it again later in the body — a residual limitation
documented in `docs/baseline_model_report.md`.

### 25. What is label leakage, and where did it appear in this project?
**Short answer:** When a feature lets the model predict the label using information that
reflects how the data was collected, not the real-world property being studied.
**Expanded:** Found in `subject` (100% rule accuracy — the two classes use entirely disjoint
subject tags), the Reuters dateline (99.6% rule accuracy), `pic.twitter.com` links
(one-directional shortcut), and `date` (a range-based shortcut).
**Example:** All four are documented with exact quantified accuracy in `docs/label_leakage_analysis.md`.

### 26. Why was the `subject` column removed entirely?
**Short answer:** It perfectly separates the two classes — Fake and Real use completely
non-overlapping subject values.
**Expanded:** A trivial `if/else` rule on `subject` alone reaches 100.000% accuracy — meaning
the model would learn "which source file this came from," not "is this fake."
**Example:** Fake subjects: `News`, `politics`, `left-news`, `Government News`, `US_News`,
`Middle-east`. Real subjects: `politicsNews`, `worldnews`. Zero overlap.

### 27. Why were duplicate articles removed before splitting the data?
**Short answer:** To prevent the same article appearing in both the training set and the test
set, which would inflate reported accuracy through memorization rather than real generalization.
**Expanded:** ~14% of rows were exact duplicates of another row (mostly republished wire
stories). Deduplicating *before* splitting (not after) is what actually prevents this leakage.
**Example:** `docs/duplicate_analysis.md` — deduplication on `text` alone removed 6,252 rows in
the standalone estimate (5,619 in the actual staged pipeline run, due to order effects — see
`docs/preprocessing_execution_report.md`).

### 28. Why combine title and article body into one field?
**Short answer:** EDA found titles carry real signal (Fake titles average 14.7 words vs. Real's
10.0), and combining keeps a single-input model architecture instead of two.
**Expanded:** Discarding the title would throw away a measurable style difference for free; a
two-input model would add architectural complexity with no demonstrated need.
**Example:** `content = title + " " + text`, implemented in `preprocessing/text_cleaning.py`'s
`combine_title_and_text()`.

### 29. Why create multiple staged processed datasets instead of one script?
**Short answer:** So each stage (combine, clean, preprocess) has a single, independently
verifiable responsibility, and a bug in one stage doesn't hide inside a monolithic transformation.
**Expanded:** `01_combined_raw.csv` → `02_cleaned.csv` → `03_preprocessed.csv`, each produced by
its own pipeline stage (`preprocessing/pipeline.py`), each with its own before/after row counts
recorded (`report/tables/stage*_stats.csv`).
**Example:** If Stage 3 output looked wrong, `02_cleaned.csv` could be reloaded and Stage 3
re-run in isolation, without re-running Stages 1–2 or risking their being silently altered.

### 30. Why build a baseline before implementing the LSTM?
**Short answer:** Without a baseline, there's no reference point to judge whether the LSTM's
result is actually good.
**Expanded:** It also builds understanding of the full ML workflow (vectorize → train →
evaluate) on a fast, simple model before adding the LSTM's extra complexity (tokenizer,
padding, embeddings, epochs) on top.
**Example:** This project's baseline reached ~98% accuracy — the LSTM (Phase 5) must be judged
against this number, not in isolation.

### 31. Why do `baseline_text` and `lstm_text` differ (stop words/lemmatization)?
**Short answer:** TF-IDF has no concept of word order, so stop words are pure noise for it and
removing them helps; an LSTM can use word order and needs stop words like "not"/"never" intact
for negation.
**Expanded:** Applying the same cleaning to both models would either leave avoidable noise in
the baseline or damage the one thing an LSTM is specifically good at (using context and word
order).
**Example:** `preprocessing/text_cleaning.py`'s `BASELINE_STEPS` (stopwords + lemmatization ON)
vs. `LSTM_STEPS` (both OFF) — see `docs/preprocessing_plan.md`.

### 32. What did the feature-importance analysis reveal, and why does it matter?
**Short answer:** It revealed that "reuters" is still the single strongest predictor of Real,
even after removing the leading Reuters dateline — a residual leakage signal missed by the
planning-phase analysis alone.
**Expanded:** 13.4% of articles still mention "reuters" later in the body (corrections,
secondary datelines, citations). This shows model interpretation can catch things a static
data-quality check misses.
**Example:** `notebooks/03_baseline_model.ipynb`'s Feature Importance section walks through
exactly how this was discovered, quantified, and what it means for the LSTM (which reads the
same underlying text and is likely to have the same sensitivity).

---

## Concepts I Must Understand

| Concept | Definition | Why it's used | Analogy | Example in this project |
|---|---|---|---|---|
| **TF-IDF** | Term Frequency × Inverse Document Frequency; scores a word high if frequent in one document and rare overall | Converts text into meaningful numeric features | Highlighting a word in a book only if it's rare across your whole bookshelf but repeated on this one page | `TfidfVectorizer` in `training/baseline.py` |
| **Vectorization** | Converting text into a fixed-length numeric vector | ML models require numeric, fixed-size input | Translating a sentence into a barcode a scanner (the model) can read | `vectorizer.transform(...)` |
| **Logistic Regression** | A linear model + sigmoid function that outputs a class probability | Fast, interpretable, produces probabilities | A tally sheet: add up points for/against "Real," then convert the total into a confidence percentage | `LogisticRegression` in `training/baseline.py` |
| **Sigmoid function** | `1 / (1 + e^-x)`, squashes any number into (0, 1) | Converts a raw weighted sum into a probability | A dimmer switch that maps "how much evidence" to "how bright" (0–100%) | Used internally by `LogisticRegression` |
| **Train/test split** | Dividing data into a portion for learning and a portion for evaluation | Prevents evaluating a model on data it memorized | Studying from one set of practice questions, then being tested on a different, unseen set | `training/split.py` |
| **Stratified sampling** | Splitting data while preserving each class's proportion in every split | Keeps splits comparable and reduces noise from class imbalance | Making sure a class survey samples the same boy/girl ratio as the whole school | `stratified_three_way_split()` |
| **Overfitting** | Model fits training data (including its noise) too closely, hurting new-data performance | A pitfall to recognize and avoid | Memorizing exam answers instead of understanding the subject | Checked by comparing train vs. test performance |
| **Underfitting** | Model is too simple to capture the real pattern | A pitfall to recognize and avoid | Studying so little that even the practice questions go badly | Would show as low accuracy on both train and test |
| **Precision** | Of predicted-positive cases, how many were correct | Measures false-alarm rate | Of everyone you accused, how many were actually guilty | 0.9788 in this project |
| **Recall** | Of actual-positive cases, how many were caught | Measures miss rate | Of everyone actually guilty, how many did you catch | 0.9893 in this project |
| **F1 Score** | Harmonic mean of precision and recall | Balances both into one number | Won't let you game the score by only optimizing one side | 0.9840 in this project |
| **Confusion Matrix** | Table of TP/FP/FN/TN counts | Shows *which kind* of mistake a model makes | A scoreboard broken down by exactly how each point was scored or missed | `report/figures/baseline_confusion_matrix.png` |
| **ROC Curve / AUC** | True-positive-rate vs. false-positive-rate across all thresholds; AUC summarizes it | Evaluates probability quality, not just one cutoff | Judging a weather forecaster across every possible "rain likely" threshold, not just one | AUC = 0.9979 in this project |
| **Regularization** | A penalty on large model weights during training | Reduces overfitting risk | A rule that discourages any one opinion from dominating a group decision too strongly | `LogisticRegression`'s default L2 penalty, `C=1.0` |
| **Label leakage** | A feature that predicts the label via data-collection artifacts, not real signal | Must be found and removed for a trustworthy model | A test where the answer key's font color secretly matches the correct option | `subject`, `date`, Reuters dateline (see `docs/label_leakage_analysis.md`) |
| **Sparse vector/matrix** | A vector/matrix where almost all values are zero | TF-IDF vectors are naturally sparse; storing them efficiently saves memory | A mostly-empty attendance sheet where only a few names are marked present each day | >99% zero entries in `X_train` |

---

## Common Mistakes

- **Fitting the TF-IDF vectorizer on the full dataset before splitting.** This leaks
  information about the test set's vocabulary into training. *Correct approach (used here):*
  `fit_transform()` only on the training set; `transform()` only on validation/test.
- **Removing stop words identically for every model.** Assuming "more cleaning is always
  better" and applying the same stop-word removal to an LSTM as to TF-IDF risks damaging
  negation handling. *Correct approach:* the split decision documented in `docs/preprocessing_plan.md`.
- **Treating accuracy as the only metric that matters.** A 98% accuracy sounds great, but
  without precision/recall/F1/confusion matrix, it's impossible to tell if the errors are
  evenly split or concentrated in one class.
- **Assuming a high accuracy automatically means a good model.** This project's own history is
  the counter-example: this dataset could hit ~99.6% accuracy from the Reuters tag *alone*,
  which would have been a meaningless, non-generalizing result — always ask what the model
  might be "cheating" with before trusting a high score.
- **Forgetting to use a fixed random seed.** Without one, the train/test split (and therefore
  every reported metric) changes slightly on every re-run, making results impossible to
  reproduce or fairly compare against the LSTM later.
- **Not checking for duplicate rows before splitting.** Skipping this can let the same article
  appear in both train and test, inflating the test score without the model actually
  generalizing.
- **Confusing stemming with lemmatization**, or assuming more preprocessing steps are always
  better — see `docs/preprocessing_plan.md` for why stemming was deliberately not used at all.

## Follow-up Questions

- *If asked "why Logistic Regression?", expect:* "What would happen if you used an SVM
  instead?" → Answer: likely similar accuracy, but you'd lose the direct per-word
  interpretability that powers the Feature Importance section.
- *If asked "what is TF-IDF?", expect:* "What's the difference between TF-IDF and just counting
  word frequency?" → Answer: raw counts rank "the"/"a"/"is" as most important in every article;
  IDF specifically down-weights words that appear in most documents.
- *If asked about the ~98% accuracy, expect:* "Isn't that suspiciously high?" → Answer: yes,
  worth being ready to walk through the leakage-removal history (`subject`, Reuters dateline,
  `date`) and the residual "reuters" finding from feature importance, showing you investigated
  rather than accepted the number blindly.
- *If asked "why remove `subject`?", expect:* "Couldn't you just keep it and see if it helps?"
  → Answer: it doesn't "help," it *replaces* the task — a 100%-accurate shortcut, unrelated to
  detecting actual fake news, would make the whole project meaningless as a demonstration of
  text-based detection.
- *If asked about the train/test split, expect:* "Why 70/15/15 and not 80/10/10 or 90/10?" →
  Answer: with ~38,600 rows, even 15% (≈5,800) is large enough for stable test metrics; the
  three-way split also reserves a validation set for the LSTM's early stopping in Phase 5,
  using an identical split methodology for fair comparison.
- *If asked "what's a false negative here?", expect:* "Which is worse, a false positive or a
  false negative, for fake news detection?" → Answer: arguably a false negative (real news
  flagged as fake) risks suppressing legitimate information, while a false positive (fake news
  flagged as real) fails to catch misinformation — both matter, and this project reports both
  rather than optimizing for just one.

## End-of-Phase Summary

**Things I should confidently explain:**
- The full pipeline: `baseline_text` → TF-IDF → Logistic Regression → prediction.
- Why Logistic Regression was chosen over Naive Bayes/Decision Trees/SVM.
- What TF-IDF actually computes and why it beats raw word counts.
- All five evaluation outputs (accuracy, precision, recall, F1, confusion matrix) and what each
  one specifically tells you that the others don't.
- The residual "reuters" leakage finding — this is a strong, specific, honest talking point.

**Topics to revise before moving on:**
- The exact mathematical form of the sigmoid function and what "log-odds" means, if asked to
  go one level deeper than "it outputs a probability."
- The precise mechanics of how `train_test_split`'s stratify parameter works internally.

**Key takeaways:**
- A strong baseline is a deliverable in its own right, not just a formality before the "real"
  model.
- Interpretability (Feature Importance) isn't just a nice-to-have — it directly caught a data
  quality issue (residual Reuters mentions) that pure metrics alone would never have revealed.
- Every design decision in this phase (split ratio, stop-word handling, vectorizer scope) traces
  back to a specific, statable reason — that traceability is what a viva panel is actually
  testing for.

---

## Addendum — Reuters Leakage Validation (Ablation Experiment)

An additional experiment (`notebooks/03b_reuters_ablation_experiment.ipynb`,
`docs/reuters_ablation_study.md`) followed up directly on the "reuters" finding above. Kept
here as an addendum rather than a new phase, since it validates Phase 4 rather than starting a
new one.

### New Likely Viva Questions

33. What is an ablation study?
34. Why did you run a Reuters-removal ablation experiment?
35. What is McNemar's test, and why use it instead of just comparing two accuracy numbers?
36. If removing "reuters" barely changed accuracy, was there no leakage after all?
37. What did feature importance show after removing "reuters"?
38. Why didn't you retrain or replace the official baseline with this experiment's model?

### Short Answers

**33. What is an ablation study?**
**Short answer:** Removing one specific component and re-measuring, to see how much that
component actually mattered.
**Expanded:** Named after the surgical term for removing tissue to observe an effect. If
removing a part changes little, that part wasn't doing as much independent work as it looked
like it was doing.
**Example:** This project removed every occurrence of "reuters" from `baseline_text`, retrained
under otherwise identical conditions, and measured the change.

**34. Why did you run a Reuters-removal ablation experiment?**
**Short answer:** Feature importance showed "reuters" as the #1 predictor of Real even after
Stage 2 stripped the leading dateline — a finding that needed investigating, not celebrating.
**Expanded:** A large coefficient shows correlation, not how much the model actually depends on
that one word. Only removing it and re-measuring can tell the difference.
**Example:** `docs/reuters_ablation_study.md`, Question 4's answer.

**35. What is McNemar's test, and why use it instead of just comparing two accuracy numbers?**
**Short answer:** A paired statistical test for two classifiers scored on the *same* test set —
it looks only at rows where the two models disagree.
**Expanded:** Comparing two raw accuracy numbers doesn't say whether a small difference is real
or just noise. McNemar's test uses the *discordant* rows (one model right, one wrong) to test
that directly.
**Example:** Only 9 of 5,796 test rows were discordant between the official baseline and the
ablation model; the exact p-value was 1.0 — no significant difference.

**36. If removing "reuters" barely changed accuracy, was there no leakage after all?**
**Short answer:** There is still a leakage-like signal, but it isn't concentrated in one
removable word — it's spread redundantly across several correlated Reuters-style cues.
**Expanded:** After removing "reuters," the model simply promoted `said` and weekday names
(already present in its top 10) rather than losing accuracy — proving the signal was never
resting on one token alone.
**Example:** `docs/reuters_ablation_study.md`'s "reshuffle, don't replace" feature-importance
comparison.

**37. What did feature importance show after removing "reuters"?**
**Short answer:** `said` became the new #1 predictor of Real; weekday names moved up; the
Fake-word list didn't change at all.
**Expanded:** No genuinely new word entered the top 20 on either side — everything that
appeared was already present, just one rank higher.
**Example:** `report/figures/reuters_ablation_top_features.png` vs. `report/figures/baseline_top_features.png`.

**38. Why didn't you retrain or replace the official baseline with this experiment's model?**
**Short answer:** This was a validation experiment to test a hypothesis, not a better model —
the official baseline remains the project's reference point.
**Expanded:** The ablation model's artifacts were never saved to `models/baseline/`; the
official vectorizer/model were only ever loaded read-only, for comparison.
**Example:** `models/baseline/*.pkl` timestamps are unchanged from Phase 4's original training run.

### Follow-up Questions

- *If asked "why not just compare the two accuracy numbers directly?", expect:* "What if you'd
  gotten a 2% difference — would you still need McNemar's test?" → Answer: yes, especially for
  a smaller test set; a 2% difference on 5,796 rows could still be within normal statistical
  noise, and only a formal test (or a much larger test set) can say for sure.
- *If asked about the "reshuffle, don't replace" finding, expect:* "Doesn't that mean your
  model still isn't measuring real fake-news detection?" → Answer: partially fair — it means
  the model is measuring "Reuters wire style vs. this dataset's blog style" as much as it's
  measuring anything about truthfulness, which is now a documented, acknowledged limitation
  rather than an unexamined assumption.

### End-of-Addendum Summary

**Things I should confidently explain:** what an ablation study is and why it's the correct
tool here; how McNemar's test works and why it's more rigorous than comparing two accuracy
numbers; the "reshuffle, don't replace" feature-importance finding and what it implies about
the dataset's structural source-style confound.

**Key takeaway:** a good scientist doesn't just report a metric — they try to break it, and
report honestly what happens when they do. This experiment didn't "fix" anything and wasn't
meant to; it turned a suspicious feature-importance finding into a documented, statistically
tested, and honestly-reported limitation.

---

# Phase 5 — Deep Learning (LSTM)

## Likely Viva Questions

1. What is Deep Learning, and how is it different from the Machine Learning used in Phase 4?
2. What is a word embedding?
3. Why do we need an Embedding layer instead of feeding word indices directly into the model?
4. What is tokenization (in the Keras/Deep Learning sense)?
5. What is a vocabulary, and why is it capped at a fixed size?
6. What is an OOV (out-of-vocabulary) token?
7. What is sequence padding, and why is it necessary?
8. Why pad/truncate at the end ("post") instead of the beginning ("pre")?
9. What is an LSTM cell?
10. What is the forget gate?
11. What is the input gate?
12. What is the output gate?
13. What is the vanishing gradient problem?
14. Why was LSTM chosen over a plain RNN?
15. Why was LSTM chosen over BiLSTM, Attention, or a Transformer?
16. Why does the final layer use a Sigmoid activation?
17. Why is Binary Crossentropy used as the loss function?
18. What is Early Stopping?
19. What is Dropout, and why is it used?
20. What is SpatialDropout1D, and how is it different from ordinary Dropout?
21. What is sequence length, and how was 300 chosen for this project?
22. What is batch size?
23. What is an epoch?
24. How is a trained Keras model saved and reloaded?
25. What is the Adam optimizer?
26. What is ReLU, and why is it used in the hidden Dense layer?
27. What does it mean for the LSTM layer to return only its final hidden state?

## Project-Specific Questions

28. Why does the LSTM use `lstm_text` instead of `baseline_text`?
29. Why reuse `training/split.py` unchanged instead of writing a new split for the LSTM?
30. Why reuse `evaluation/metrics.py` and `evaluation/error_analysis.py` instead of writing LSTM-specific versions?
31. Why fit the Tokenizer only on the training set?
32. What does ModelCheckpoint add beyond what EarlyStopping already does?
33. Why compare training and validation curves instead of just looking at the final accuracy?

---

## Short Answers

### 1. What is Deep Learning, and how is it different from the Machine Learning used in Phase 4?
**Short answer:** Deep Learning is Machine Learning using neural networks with multiple learned
layers; the baseline (Logistic Regression) has no hidden layers at all — it's a single linear
weighting of hand-computed TF-IDF features.
**Expanded:** The baseline's "features" (TF-IDF scores) were computed by a fixed formula before
training even began. The LSTM instead *learns its own feature representation* — the Embedding
layer's word vectors and the LSTM's internal gate weights are all learned from data, not
computed by a fixed rule.
**Example:** TF-IDF decides "reuters" is important using a fixed frequency formula; the LSTM's
embedding for "reuters" is a set of 100 numbers learned purely from how that word behaves
during training.
**Follow-up:** "So is Deep Learning always better?" → No — this project's own results (see
`docs/lstm_model_report.md`) are the direct evidence for why that assumption should always be
checked, not assumed.

### 2. What is a word embedding?
**Short answer:** A dense vector of real numbers (length 100 in this project) that represents
one word's meaning, learned during training.
**Expanded:** Unlike a one-hot or TF-IDF representation (mostly zeros, no notion of similarity
between words), an embedding positions semantically similar words closer together in a
continuous vector space — the model can learn, for example, that "minister" and "secretary"
behave similarly, without being told so explicitly.
**Example:** `Embedding(input_dim=20000, output_dim=100)` in `training/lstm.py` creates a
20,000 × 100 table of learnable numbers — one 100-number row per vocabulary word.
**Follow-up:** "Could you use a pretrained embedding like GloVe or Word2Vec instead?" → Yes,
that's a legitimate enhancement, deliberately not used here to keep the pipeline self-contained
and avoid an extra external dependency for a BCA-scope baseline LSTM.

### 3. Why do we need an Embedding layer instead of feeding word indices directly into the model?
**Short answer:** Raw integer word IDs imply a false numeric relationship (word 42 is not "6
more" than word 36 in any meaningful sense).
**Expanded:** The Embedding layer's whole job is to replace that arbitrary integer with a
learned vector whose *distance* to other word vectors actually means something.
**Example:** Without an embedding layer, the model would have to treat "good" (id 501) and
"great" (id 502) as no more related than "good" (id 501) and "banana" (id 9931) just because
502 and 9931 are both "not 501."
**Follow-up:** "What would happen if you skipped the Embedding layer entirely?" → The model
would essentially be unable to learn anything meaningful from word identity, since the input
numbers themselves carry no usable structure.

### 4. What is tokenization (in the Keras/Deep Learning sense)?
**Short answer:** Splitting text into words and assigning each a unique integer ID.
**Expanded:** `Tokenizer.fit_on_texts()` builds the word → integer mapping from the training
data; `texts_to_sequences()` then applies that mapping to turn any text into a list of integers.
**Example:** `"trump said no"` → `[42, 7, 15]`, using `training/lstm.py`'s `build_tokenizer()`
and `texts_to_padded_sequences()`.
**Follow-up:** "How is this different from the tokenization idea in TF-IDF?" → TF-IDF's
"tokenization" splits text into words too, but immediately turns them into frequency-weighted
scores per vocabulary word (a bag, no order); the Keras Tokenizer keeps one integer **per word
position**, preserving order for the LSTM to read sequentially.

### 5. What is a vocabulary, and why is it capped at a fixed size?
**Short answer:** The set of words the model is allowed to recognize; capped at 20,000 here so
the Embedding table and training time stay manageable.
**Expanded:** The training data actually contains 96,249 distinct words (verified directly in
the notebook) — far more than the cap. Keeping all of them would mean an enormous Embedding
table mostly full of rows that see a handful of training examples each, learning little.
**Example:** `Tokenizer(num_words=VOCAB_SIZE, ...)` in `training/lstm.py` keeps only the 20,000
most frequent words; everything else maps to the OOV token.
**Follow-up:** "Why 20,000 specifically, and not 10,000 or 50,000?" → Chosen to match
`VOCAB_SIZE`, the same cap already used for the baseline's TF-IDF vectorizer
(`config/settings.py`), so both models are capped at a comparable vocabulary size even though
they build their vocabularies differently.

### 6. What is an OOV (out-of-vocabulary) token?
**Short answer:** A single placeholder token (`"<OOV>"`) that every word outside the 20,000-word
vocabulary gets mapped to.
**Expanded:** Without an OOV token, an unrecognized word would either crash the pipeline or
silently disappear from the sequence; mapping it to a shared placeholder lets the model at
least register "an unknown word occurred here."
**Example:** `Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)` — every rare word beyond the
cap becomes the same `<OOV>` integer ID.
**Follow-up:** "Doesn't mapping many different rare words to the same token lose information?"
→ Yes, some — an accepted, standard trade-off for keeping the vocabulary a manageable size;
rare words individually carry little statistical weight anyway.

### 7. What is sequence padding, and why is it necessary?
**Short answer:** Forcing every article's integer sequence to the same fixed length (300) by
adding zeros to short ones and cutting long ones.
**Expanded:** A neural network layer processes a batch of inputs at once and requires every
input in that batch to have identical shape; articles naturally have wildly different lengths
(the dataset's word counts range from a few words to thousands).
**Example:** `pad_sequences(sequences, maxlen=300, padding="post", truncating="post")` in
`training/lstm.py`.
**Follow-up:** "Doesn't padding with zeros add meaningless data the model has to process?" →
The zero ID is reserved and never assigned to a real word, so the model can learn to treat it
as "no information here"; in a more advanced setup, a "masking" layer can make the model skip
padded positions entirely, which this project's simple architecture does not use.

### 8. Why pad/truncate at the end ("post") instead of the beginning ("pre")?
**Short answer:** To keep the beginning of each article — where news writing typically
front-loads the most important information — intact in both padding and truncation.
**Expanded:** "Pre" truncation would cut off the start of long articles (losing the most
information-dense part); "pre" padding would push the real content to the end of a fixed-length
window, which behaves differently for short vs. long articles for no good reason.
**Example:** `padding="post", truncating="post"` in `texts_to_padded_sequences()`.
**Follow-up:** "Would 'pre' truncation ever make sense?" → In domains where the ending matters
most (e.g. certain legal or narrative text with concluding statements), but that's not the
convention for news articles.

### 9. What is an LSTM cell?
**Short answer:** A recurrent neural network unit that maintains an internal memory (the "cell
state") and uses three gates to control what enters, stays, and exits that memory as it reads a
sequence one step at a time.
**Expanded:** Unlike a plain RNN cell (which just combines the current input with the previous
step's output through one simple transformation), an LSTM cell's gates let it selectively
preserve information across many time steps without it decaying away.
**Example:** `LSTM(64)` in `training/lstm.py` — 64 is the size of the cell's internal memory and
output vector.
**Follow-up:** "What's actually inside the cell state, concretely?" → A vector of numbers (64
of them here) updated at every time step by the forget and input gates — not directly
human-interpretable, but functioning as the network's working memory of the sequence so far.

### 10. What is the forget gate?
**Short answer:** The part of the LSTM cell that decides what to discard from the memory so far.
**Expanded:** It looks at the current word's embedding and the previous step's output, and
outputs a number between 0 and 1 for each memory dimension — closer to 0 means "forget this,"
closer to 1 means "keep this."
**Example:** If the model has been building up "this looks like a Reuters-style article" and
then encounters strong contradicting evidence, the forget gate can down-weight that
accumulated signal.
**Follow-up:** "What would happen without a forget gate?" → The cell state would just keep
accumulating information forever without ever clearing outdated context — closer to a plain
RNN's behavior, and part of why plain RNNs struggle with longer sequences.

### 11. What is the input gate?
**Short answer:** The part of the LSTM cell that decides what new information from the current
word to add to the memory.
**Expanded:** Works alongside a candidate update (computed from the current input) — the input
gate scales how much of that candidate actually gets written into the cell state.
**Example:** Encountering the word "not" right before a claim might strongly update the memory
to flag an upcoming negation.
**Follow-up:** "How is the input gate different from the forget gate?" → The forget gate
controls what leaves the existing memory; the input gate controls what new information enters
it — both operate every time step, together.

### 12. What is the output gate?
**Short answer:** The part of the LSTM cell that decides what part of the (now-updated) memory
to expose as this time step's output.
**Expanded:** The cell state itself is never directly output — the output gate filters it to
produce the hidden state that gets passed to the next time step (and, at the final step, to the
next layer).
**Example:** The 64-number vector `LSTM(64)` returns after processing all 300 time steps is the
final output gate's result at the last step.
**Follow-up:** "Why not just output the whole cell state directly?" → Separating "internal
memory" from "what's exposed" lets the network keep information in memory for later without
necessarily using it at every single step.

### 13. What is the vanishing gradient problem?
**Short answer:** During training, the error signal used to adjust earlier time steps'
influence shrinks exponentially as it's propagated backward through many steps, so a plain RNN
effectively stops learning from anything more than a few steps back.
**Expanded:** Backpropagation through time repeatedly multiplies gradients by values that are
often less than 1; after enough time steps, the product becomes vanishingly small, so words
near the start of a 300-word sequence get almost no training signal in a plain RNN.
**Example:** A plain RNN trying to connect a negation word near the start of an article to the
classification decision at the end would likely fail to learn that connection at all.
**Follow-up:** "How exactly does LSTM avoid this?" → The cell state has a more direct,
largely-additive update path (via the gates) rather than being repeatedly squashed through a
single nonlinear transformation at every step, which allows gradients to flow backward with
much less decay.

### 14. Why was LSTM chosen over a plain RNN?
**Short answer:** Specifically to avoid the vanishing gradient problem above — this dataset's
articles run to hundreds of words, far beyond what a plain RNN can realistically learn from.
**Expanded:** Given `MAX_SEQUENCE_LENGTH = 300`, a plain RNN would very likely fail to connect
information from the first 50–100 words to the final classification decision.
**Example:** See Question 13's answer for the mechanism.
**Follow-up:** "Are there simpler fixes to the vanishing gradient problem than switching to
LSTM?" → GRU (Gated Recurrent Unit) is a lighter-weight alternative gated architecture with
similar benefits; not used here to keep to the specific architecture requested for this
project, and because LSTM is the more widely recognized, more thoroughly documented choice for a
BCA-level report.

### 15. Why was LSTM chosen over BiLSTM, Attention, or a Transformer?
**Short answer:** All are legitimate techniques that might improve accuracy further, but each
adds architectural complexity and harder-to-explain internals, working against this project's
priority on a clean, fully-explainable architecture.
**Expanded:** A BiLSTM reads the sequence both forward and backward and doubles the recurrent
parameters; Attention/Transformers replace recurrence entirely with a different (and
substantially more complex) mechanism. The project specification explicitly calls for avoiding these unless
there's a clear, demonstrated academic benefit.
**Example:** This project's plain, unidirectional `LSTM(64)` layer can be fully explained
gate-by-gate (Questions 9–12) in a way a Transformer's multi-head self-attention mechanism
would be a much harder claim to defend at BCA level.
**Follow-up:** "Would a BiLSTM likely perform better here?" → Possibly, since it could use
context from both directions (e.g. a word's meaning informed by what comes after it too), but
that's an open question this project deliberately didn't chase, in favor of a simpler, fully
understood baseline-for-deep-learning.

### 16. Why does the final layer use a Sigmoid activation?
**Short answer:** Sigmoid squashes any real number into the range (0, 1), which is exactly what
a "probability the article is Real" needs to be.
**Expanded:** This is the same interpretation the baseline's Logistic Regression output has —
matching activation choices is part of what makes the two models' outputs directly comparable
in Phase 6.
**Example:** `Dense(1, activation="sigmoid")` in `training/lstm.py`; a raw model output of 0.83
means "83% confident this is Real."
**Follow-up:** "What would you use instead for a 3-class problem?" → `softmax` over 3 output
units, not sigmoid — sigmoid is specifically for binary (or independent multi-label) outputs.

### 17. Why is Binary Crossentropy used as the loss function?
**Short answer:** It directly measures how far a predicted probability is from the true 0/1
label, penalizing confident wrong predictions much more heavily than uncertain ones.
**Expanded:** It is the standard, mathematically appropriate loss for a single-output sigmoid
binary classifier — pairing it with a different loss (e.g. mean squared error) would give
weaker gradient signal specifically for the confidently-wrong predictions the model most needs
to correct.
**Example:** `model.compile(loss="binary_crossentropy", ...)` in `build_lstm_model()`.
**Follow-up:** "What loss would you use for the multi-class case?" → Categorical (or sparse
categorical) crossentropy, the direct multi-class generalization of the same idea.

### 18. What is Early Stopping?
**Short answer:** A training callback that halts training once validation loss stops improving
for a set number of epochs, and rolls the model back to its best-validation-loss weights.
**Expanded:** Without it, training would run for the full fixed number of epochs regardless of
whether the model started overfitting partway through — `restore_best_weights=True` specifically
ensures the *final* model in memory is the best one seen, not just the last one trained.
**Example:** `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` in
`training/lstm.py` — training stops if validation loss doesn't improve for 3 consecutive epochs.
**Follow-up:** "Why monitor val_loss instead of val_accuracy?" → Loss is a continuous, more
sensitive signal of model confidence and calibration; accuracy can stay flat while the
underlying probability estimates are quietly getting worse (or better), so loss is generally the
more standard choice to monitor for this callback.

### 19. What is Dropout, and why is it used?
**Short answer:** A regularization technique that randomly zeroes out a fraction of a layer's
output values during training, forcing the network not to over-rely on any single connection.
**Expanded:** Applied here after the LSTM layer (`Dropout(0.3)`), guarding specifically against
the dense classification head overfitting to particular quirks of the LSTM's output on the
training set.
**Example:** During training, roughly 30% of the LSTM's 64 output values are randomly set to 0
on each batch; at prediction time, Dropout is inactive and the full output is used.
**Follow-up:** "Does Dropout apply during prediction/inference?" → No — Keras automatically
disables Dropout outside of training, so predictions use the complete, undropped network.

### 20. What is SpatialDropout1D, and how is it different from ordinary Dropout?
**Short answer:** Instead of zeroing out individual numbers at random (like ordinary Dropout),
SpatialDropout1D zeroes out entire embedding dimensions consistently across the whole sequence.
**Expanded:** For word embeddings specifically, this is the more semantically appropriate form
of regularization — it prevents the LSTM from becoming overly reliant on any single embedding
dimension, rather than corrupting scattered individual numbers inside otherwise-intact word
vectors.
**Example:** `SpatialDropout1D(0.2)` immediately after the Embedding layer in
`build_lstm_model()`.
**Follow-up:** "Why apply it right after the Embedding layer specifically, and not elsewhere?"
→ Because that's precisely where "whole dimensions of a word vector" exist as a concept — later
layers (like the LSTM's own output) don't have the same per-word-per-dimension structure to
usefully drop this way.

### 21. What is sequence length, and how was 300 chosen for this project?
**Short answer:** The fixed number of word positions every input sequence is padded/truncated
to; chosen here as 300 based on the dataset's median article length.
**Expanded:** Phase 2 EDA (`01_dataset_analysis.ipynb`) found articles average ~360–420 words
with a small median gap between classes, but a small number of extreme outliers up to ~8,100
words — using the maximum would waste huge amounts of computation padding nearly every article
far beyond what it needs.
**Example:** `MAX_SEQUENCE_LENGTH = 300` in `config/settings.py`.
**Follow-up:** "What happens to articles longer than 300 words?" → They get truncated (their
words beyond position 300 are simply dropped) — a deliberate, acknowledged trade-off, not an
error.

### 22. What is batch size?
**Short answer:** The number of training examples processed together before the model's
weights are updated once.
**Expanded:** A larger batch size processes more data per weight update (often faster per
epoch, especially on CPU, due to vectorized operations) but gives a noisier-to-smoother
gradient estimate depending on the specific value and dataset; a smaller batch size updates more
often per epoch but each update is based on less data.
**Example:** `LSTM_BATCH_SIZE = 64` in `config/settings.py` — each weight update in this project
is based on 64 articles at a time.
**Follow-up:** "What would happen with batch_size=1?" → Extremely slow training (no benefit
from vectorized batch computation) and very noisy gradient updates — not practical here.

### 23. What is an epoch?
**Short answer:** One complete pass through the entire training dataset.
**Expanded:** `LSTM_MAX_EPOCHS = 10` sets an upper bound; EarlyStopping can (and, in this
project, did) stop training before reaching that maximum once validation loss stopped
improving — see `models/lstm/metadata.json` for the exact number of epochs actually completed.
**Example:** With 27,046 training rows and a batch size of 64, one epoch consists of
`ceil(27046 / 64) = 423` weight-update steps.
**Follow-up:** "Why not just train for a very large fixed number of epochs?" → Without
EarlyStopping, the model would keep training past the point of best generalization and start
overfitting — exactly the failure mode EarlyStopping exists to prevent.

### 24. How is a trained Keras model saved and reloaded?
**Short answer:** `model.save("model.keras")` saves the architecture, weights, and training
configuration into a single file; `keras.models.load_model("model.keras")` reloads all of it.
**Expanded:** The `.keras` format (native to Keras 3) is a single self-contained file — simpler
than the older multi-file `SavedModel` directory format or the older `.h5` format, and is what
the project specification's Model Artifacts standard specifies project-wide.
**Example:** `training/lstm.py`'s `save_lstm_artifacts()` calls `model.save(output_dir /
"model.keras")`.
**Follow-up:** "What else needs to be saved besides the model file itself?" → The fitted
Tokenizer (`tokenizer.pkl`) — without it, new article text at prediction time couldn't be
converted into the same integer sequences the model was trained on, and the label map
(`label_map.json`), so `0`/`1` predictions can be mapped back to "Fake"/"Real" consistently.

### 25. What is the Adam optimizer?
**Short answer:** An optimization algorithm that adapts each parameter's learning rate
individually during training, based on that parameter's recent gradient history.
**Expanded:** Adam combines ideas from two earlier optimizers (momentum and per-parameter
adaptive learning rates), making it a robust, low-maintenance default that rarely needs manual
learning-rate tuning to get reasonable results.
**Example:** `model.compile(optimizer="adam", ...)` in `build_lstm_model()`.
**Follow-up:** "What would you lose by using plain SGD (Stochastic Gradient Descent) instead?"
→ Plain SGD typically needs a carefully hand-tuned (and often scheduled) learning rate to train
well; Adam's adaptivity removes most of that tuning burden, at some cost in raw
computational simplicity.

### 26. What is ReLU, and why is it used in the hidden Dense layer?
**Short answer:** `ReLU(x) = max(0, x)` — outputs the input directly if positive, zero
otherwise; used because it's simple, fast, and avoids some of the vanishing-gradient issues
older activations (sigmoid/tanh) have in deeper networks.
**Expanded:** Unlike sigmoid/tanh, ReLU's gradient doesn't shrink toward zero for large positive
inputs, which generally makes networks with ReLU hidden layers easier to train.
**Example:** `Dense(32, activation="relu")` in `build_lstm_model()` — the hidden layer between
the LSTM's output and the final sigmoid classification.
**Follow-up:** "Does ReLU have any downsides?" → Yes — a "dying ReLU" can occur where a neuron
gets stuck outputting zero for all inputs and stops learning; not specifically addressed in
this project's simple architecture, but a known limitation worth naming if asked.

### 27. What does it mean for the LSTM layer to return only its final hidden state?
**Short answer:** With `return_sequences` left at its default (`False`), `LSTM(64)` outputs a
single 64-number vector summarizing the whole 300-word sequence, not one vector per time step.
**Expanded:** That single final vector is what feeds into the Dropout/Dense layers — appropriate
for this project because the task is "classify the whole article," a single decision per input,
not a per-word/per-time-step output.
**Example:** If `return_sequences=True` were used instead, the LSTM would output a `(300, 64)`
matrix (one 64-number vector per word position) — useful for tasks like per-word tagging, or as
input to a second stacked LSTM layer, neither of which this project needs.
**Follow-up:** "When would you want return_sequences=True?" → Stacking multiple LSTM layers
(each earlier layer needs to pass a full sequence to the next), or for sequence-to-sequence
tasks like translation — not applicable to this single-layer, whole-article classification
setup.

---

### 28. Why does the LSTM use `lstm_text` instead of `baseline_text`?
**Short answer:** `baseline_text` had stop words removed and words lemmatized — both
deliberately skipped for `lstm_text` because the LSTM needs word order and negation words
intact (see `docs/preprocessing_plan.md`).
**Expanded:** Reusing `baseline_text` would silently damage the one thing an LSTM is
specifically good at exploiting (sequence/context), for a noise-reduction benefit that only
matters to a bag-of-words model in the first place.
**Example:** `load_lstm_dataset()` in `training/lstm.py` structurally drops `baseline_text` at
the source, so it's impossible to use it here by accident.
**Follow-up:** "What would you expect if you mistakenly trained on baseline_text instead?" →
Some loss of ability to use negation/context correctly, and a vocabulary already stripped of
common function words the LSTM might otherwise have used structurally — not necessarily a
dramatic accuracy drop, but a design mistake worth being able to identify on sight.

### 29. Why reuse `training/split.py` unchanged instead of writing a new split for the LSTM?
**Short answer:** So the LSTM is evaluated on the **exact same test articles** as the baseline,
making the eventual Phase 6 comparison fair.
**Expanded:** Because the row order and label column are identical between the two notebooks'
input data, calling the same split function with the same seed reproduces the identical
train/validation/test partition — verified directly in `docs/reuters_ablation_study.md` for the
baseline's own re-derivation, and by the same logic here.
**Example:** `stratified_three_way_split(df)` called identically in both
`notebooks/03_baseline_model.ipynb` and `notebooks/04_lstm_training.ipynb`.
**Follow-up:** "What would go wrong if the LSTM used a different split?" → Any accuracy
difference in Phase 6 could then be explained by "the two models were tested on different
articles" rather than a genuine difference in modeling approach — destroying the validity of
the comparison entirely.

### 30. Why reuse `evaluation/metrics.py` and `evaluation/error_analysis.py` instead of writing LSTM-specific versions?
**Short answer:** Both modules were written to be model-agnostic from the start (they operate
on `y_true`/`y_pred`/`y_proba`, not on a specific model type), so no LSTM-specific version was
needed.
**Expanded:** This also guarantees the two models are *scored* the same way — same metric
definitions, same confusion matrix layout, same plot styling — removing any risk that a
difference in results comes from a difference in how they were measured rather than a
difference in the models themselves.
**Example:** `compute_classification_metrics()`, `plot_confusion_matrix()`, `plot_roc_curve()`,
`get_false_positives()`/`get_false_negatives()` are called with zero modification in both
`notebooks/03_baseline_model.ipynb` and `notebooks/04_lstm_training.ipynb`.
**Follow-up:** "What *did* need new code for the LSTM, then?" → Only
`evaluation/training_history.py` (accuracy/loss-per-epoch plots) — a genuinely
deep-learning-specific concept (the baseline has no "epochs") with no equivalent in the
existing evaluation code.

### 31. Why fit the Tokenizer only on the training set?
**Short answer:** The same reproducibility/leakage rule already used for the baseline's
TfidfVectorizer — fitting on the full dataset before splitting would leak validation/test
vocabulary into the model's word index.
**Expanded:** This is a second, subtler form of train/test leakage beyond the row-level
duplicate leakage handled in `docs/duplicate_analysis.md` — one about *vocabulary*, not rows.
**Example:** `build_tokenizer(train_df["lstm_text"])` — never called on `val_df` or `test_df`.
**Follow-up:** "How would you even detect this kind of leakage if it happened?" → Hard to detect
after the fact from metrics alone (it usually just makes results look slightly, misleadingly
better) — which is exactly why it needs to be prevented by design/code structure rather than
caught by inspection.

### 32. What does ModelCheckpoint add beyond what EarlyStopping already does?
**Short answer:** EarlyStopping keeps the best weights *in memory* for the rest of that training
run; ModelCheckpoint independently *saves* the best-so-far model to disk during training, as a
safety net.
**Expanded:** If training were interrupted (crash, manual stop) before EarlyStopping triggered,
`restore_best_weights` would have nothing to restore *from* — ModelCheckpoint's on-disk copy
survives independently of whether the training process itself completes.
**Example:** `ModelCheckpoint(filepath=..., monitor="val_loss", save_best_only=True)` in
`train_lstm_model()`.
**Follow-up:** "Are the two callbacks ever in conflict?" → No — both monitor the same metric
(`val_loss`) and agree on which epoch was "best"; they're complementary safety nets, not
competing mechanisms.

### 33. Why compare training and validation curves instead of just looking at the final accuracy?
**Short answer:** A single final accuracy number can't show *how* the model got there — the
training-history plot reveals whether the model is still improving, has plateaued, or has
started overfitting (training accuracy rising while validation accuracy falls or stalls).
**Expanded:** This is the same underlying concept as the baseline's confusion matrix vs. plain
accuracy (`docs/baseline_model_report.md`) — an aggregate number hides the story a more detailed
view reveals.
**Example:** `evaluation/training_history.py`'s `plot_training_history()`, saved as
`report/figures/lstm_training_history.png`.
**Follow-up:** "What would a clear overfitting signature look like on this plot?" → Training
loss continuing to fall epoch after epoch while validation loss flattens out and then starts
rising — exactly the pattern EarlyStopping is designed to catch and roll back from.

---

## Concepts I Must Understand

| Concept | Definition | Why it's used | Analogy | Example in this project |
|---|---|---|---|---|
| **Word embedding** | A learned dense vector representing a word's meaning | Lets the model use word similarity, unlike TF-IDF/one-hot | Placing related words closer together on a map | `Embedding(20000, 100)` |
| **Tokenizer (Keras)** | Maps words to integer IDs, preserving sequence order | Needed input format for a sequence model | Assigning each person in a queue a numbered ticket, in order | `build_tokenizer()` |
| **Padding** | Forcing all sequences to one fixed length | Neural network layers need uniform input shape per batch | Trimming or extending every plank to the same length before building a fence | `pad_sequences(..., maxlen=300)` |
| **Vocabulary cap** | Limiting the model to the N most frequent words | Keeps the Embedding table a manageable size | A dictionary that only includes the most commonly used words | `VOCAB_SIZE = 20000` |
| **OOV token** | Placeholder for any word outside the vocabulary | Avoids crashing or silently dropping unknown words | A universal "unknown item" barcode at a checkout scanner | `OOV_TOKEN = "<OOV>"` |
| **LSTM cell** | A recurrent unit with gated memory | Preserves information across long sequences without vanishing | A notebook where you actively decide what to erase, add, and read back at every step | `LSTM(64)` |
| **Forget gate** | Decides what to discard from memory | Prevents outdated information from lingering forever | Erasing old, no-longer-relevant notes | Internal to `LSTM(64)` |
| **Input gate** | Decides what new information to add to memory | Lets relevant new context update the memory | Writing a new, important note | Internal to `LSTM(64)` |
| **Output gate** | Decides what part of memory to expose as output | Separates "what's stored" from "what's shared right now" | Choosing what to say out loud vs. what stays in your head | Internal to `LSTM(64)` |
| **Vanishing gradient** | Training signal shrinking to near-zero over many time steps | Explains why plain RNNs fail on long sequences | A whisper passed down a very long line of people, arriving inaudible | Motivates choosing LSTM over plain RNN |
| **Dropout** | Randomly zeroing some values during training | Regularization - reduces overfitting | Randomly benching a few players each practice so the team doesn't over-rely on any one | `Dropout(0.3)` |
| **SpatialDropout1D** | Randomly zeroing entire embedding dimensions | Regularization tailored to embeddings specifically | Temporarily removing a whole category of information, not scattered random facts | `SpatialDropout1D(0.2)` |
| **Early Stopping** | Halting training once validation loss stops improving | Prevents overfitting from training too long | Stopping studying once more practice tests stop improving your score | `EarlyStopping(patience=3)` |
| **ModelCheckpoint** | Saving the best-so-far model during training | Safety net independent of EarlyStopping | Saving your work periodically in case the program crashes | `ModelCheckpoint(...)` |
| **Batch size** | Number of examples processed per weight update | Balances training speed vs. gradient noise | Grading a stack of 64 exams before updating your sense of the class average | `LSTM_BATCH_SIZE = 64` |
| **Epoch** | One full pass through the training data | The basic unit of training duration | Reading an entire textbook once, cover to cover | `LSTM_MAX_EPOCHS = 10` |
| **Binary Crossentropy** | Loss function measuring how far predicted probabilities are from true 0/1 labels | Correct, standard loss for sigmoid binary classification | A stricter penalty for being confidently wrong than for being unsure and wrong | `loss="binary_crossentropy"` |
| **Sigmoid activation** | Squashes any number into (0, 1) | Produces an interpretable probability | A dimmer switch mapping "how much evidence" to "how bright" (0-100%) | `Dense(1, activation="sigmoid")` |
| **ReLU activation** | `max(0, x)` | Simple, fast, avoids some vanishing-gradient issues in hidden layers | A one-way valve - lets positive signal through, blocks negative | `Dense(32, activation="relu")` |
| **Adam optimizer** | Adapts each parameter's learning rate individually during training | Robust default, little manual tuning needed | An automatic transmission adjusting itself to road conditions, vs. manual gear-shifting (SGD) | `optimizer="adam"` |

---

## Common Mistakes

- **Fitting the Tokenizer on the full dataset before splitting.** Same leakage risk as fitting
  TF-IDF on everything in Phase 4 — always fit only on the training set.
- **Using `baseline_text` for the LSTM (or vice versa) out of convenience.** The two text
  columns exist specifically because the two models need different preprocessing
  (`docs/preprocessing_plan.md`) — using the wrong one silently undermines that design decision.
- **Assuming a bigger/deeper network is automatically better.** More LSTM units or additional
  stacked layers increase overfitting risk and training time without a guaranteed accuracy
  gain, especially on a moderately-sized dataset (~27,000 training rows) — always justify
  added complexity against the "avoid advanced architectures" principle in the project specification.
- **Forgetting `restore_best_weights=True` on EarlyStopping.** Without it, training simply stops
  early but keeps whatever weights were current at the stopping epoch, which may already be
  slightly past the best point — not necessarily the best-performing version seen.
- **Comparing the LSTM to the baseline using a different test set (or a different random
  seed).** Destroys the fairness of the comparison entirely — always confirm both models used
  `training/split.py`'s identical, seeded split.
- **Reporting only final accuracy without the training-history plot.** Hides whether the model
  is well-fit, still improving, or already overfitting by the time training stopped.
- **Treating high training accuracy as good news on its own.** Only meaningful in the context of
  the *validation* accuracy alongside it — a growing gap between the two is the overfitting
  signature to watch for, not a reason to celebrate the training number in isolation.

## End-of-Phase Summary

**Things I should confidently explain:**
- The complete pipeline: `lstm_text` → Tokenizer → padded sequences → Embedding →
  SpatialDropout1D → LSTM → Dropout → Dense(ReLU) → Dense(Sigmoid).
- What each of the three LSTM gates does, in plain language, and why gating solves the vanishing
  gradient problem a plain RNN suffers from.
- Why Sigmoid + Binary Crossentropy specifically (matching the baseline's probability
  interpretation, for a fair Phase 6 comparison).
- Why EarlyStopping and ModelCheckpoint are both present and what each independently guards
  against.
- Exactly what was reused unchanged from Phase 4 (`training/split.py`, all of `evaluation/`
  except the new `training_history.py`) and why that reuse matters for a fair comparison.

**Topics to revise before Phase 6:**
- The precise mathematical form of each LSTM gate's equations, if asked to go one level deeper
  than "decides what to forget/add/output."
- Exact epoch count, best validation loss/accuracy, and final test-set metrics for this specific
  training run — see `models/lstm/metadata.json` and `docs/lstm_model_report.md` for the
  authoritative numbers.

**Key takeaways:**
- Every architectural choice in this phase (LSTM over RNN/BiLSTM/Transformer, Sigmoid+BCE,
  SpatialDropout1D placement, sequence length, vocabulary cap) traces back to a specific,
  statable reason — the same "traceability" principle from Phase 4 applies here too.
- Reusing `training/split.py` and most of `evaluation/` unchanged wasn't just convenient — it
  was necessary for the eventual baseline-vs-LSTM comparison in Phase 6 to be scientifically
  valid.
- A Deep Learning model is not automatically "the good one" — Phase 6 will need to earn that
  conclusion with evidence, exactly the same standard this project has already held the baseline
  to.

---

# Phase 6 — Comparative Analysis & Discussion

## Likely Viva Questions

1. Why is a model comparison phase necessary after building both a baseline and a Deep Learning model?
2. What does it mean to compare two models "fairly"?
3. What is McNemar's test, and why is it more appropriate here than just comparing two accuracy numbers?
4. What is a discordant pair in the context of McNemar's test?
5. What is the null hypothesis of McNemar's test in this comparison?
6. What does a p-value actually tell you?
7. What is the difference between statistical significance and practical (effect-size) significance?
8. Why can a large test set detect a "small" difference as statistically significant?
9. What is prediction agreement analysis, and how is it different from McNemar's test?
10. What does it mean if two models rarely fail on the same articles?
11. What is an ensemble model, and why wasn't one built here?
12. What is inference time, and why does it matter separately from training time?
13. Why is training time often more important than inference time during model development?
14. What is model size, and why does it matter for deployment?
15. What counts as a "trainable parameter," and why is Logistic Regression's coefficient count a meaningful analogue?
16. Why is Deep Learning not always superior to traditional Machine Learning?
17. What is the difference between feature engineering and feature learning?
18. What is the difference between a bag-of-words representation and a sequence representation?
19. What role does dataset size play in whether Deep Learning outperforms simpler models?
20. What is model interpretability, and why does the baseline have more of it than the LSTM?
21. What is deployment complexity, and how does it differ from raw accuracy?
22. Why might a business choose a slightly less accurate model over a more accurate one?
23. What is a threats-to-validity discussion, and why include one?
24. What is the difference between internal validity, external validity, and construct validity?
25. What does it mean for a label to be a "proxy" rather than ground truth?

## Project-Specific Questions

26. Why did the TF-IDF + Logistic Regression baseline outperform the LSTM on this dataset?
27. What did the Reuters ablation study contribute to explaining this project's final comparison?
28. Why was the LSTM's training history described as "noisy," and why does that matter for this comparison?
29. Why does the LSTM show more errors on international/foreign-name articles than the baseline?
30. Why wasn't `evaluation/experiments.csv` updated with a new row in Phase 6?
31. Why was McNemar's test result different in Phase 6 (statistically significant) than in the Reuters ablation study (not significant)?
32. What would need to change for the Business Recommendation to favor the LSTM instead?
33. Why does `evaluation/agreement.py` exist as a separate module from `evaluation/significance.py`?
34. What is the theoretical "ensemble ceiling" identified in the Prediction Agreement Analysis, and why wasn't an ensemble built?
35. Why does this project conclude the LSTM was still worth building, despite losing to the baseline?

---

## Short Answers

### 1. Why is a model comparison phase necessary after building both a baseline and a Deep Learning model?
**Short answer:** Because "I built an LSTM" and "the LSTM is better" are two completely different
claims, and only a direct, fair comparison can support the second one.
**Expanded:** Phase 4 built the baseline specifically so that Phase 5's LSTM would have something
to be measured against (`docs/baseline_model_report.md` says this explicitly). Skipping Phase 6
would mean reporting the LSTM's ~97% accuracy in isolation, which sounds impressive until you
learn a simpler model already reached ~98%.
**Example:** `docs/model_comparison_report.md` Section 2 places both models' metrics side by side
— only visible together does the baseline's advantage become apparent.
**Follow-up:** "Couldn't you just look at both reports separately?" → Technically yes, but neither
individual report ran a statistical test against the other, so neither could say whether the gap
was real or just noise — that required this dedicated phase.

### 2. What does it mean to compare two models "fairly"?
**Short answer:** Evaluating both on the exact same test data, prepared and split the exact same
way, so any difference in scores reflects the models, not the evaluation setup.
**Expanded:** This project made "fair" concrete from the start: `training/split.py` was written
once (Phase 4) and reused unchanged (same function, same `RANDOM_SEED`) for the LSTM (Phase 5)
and re-verified again in Phase 6 by reproducing the identical 5,796-row test set.
**Example:** `notebooks/05_model_comparison.ipynb`, Step 1, calls `stratified_three_way_split()`
and confirms the resulting test set size and class balance match what both models were originally
scored on.
**Follow-up:** "What would make a comparison unfair?" → Using a different random seed, a
different split ratio, or (worse) letting one model's tokenizer/vectorizer see test data during
fitting — any of these would let one model have an advantage the other didn't get. Note that
"fair" means the *split* and *evaluation procedure* are identical, not that both models literally
consume byte-identical text — the baseline reads `baseline_text` and the LSTM reads `lstm_text`
(both derived from the same Stage 3 cleaning, differing only in steps each architecture
specifically needs, per `docs/preprocessing_plan.md`), since TF-IDF and an LSTM fundamentally
require different input shapes anyway.

### 3. What is McNemar's test, and why is it more appropriate here than just comparing two accuracy numbers?
**Short answer:** A statistical test for two classifiers scored on the *same* test set that looks
only at the rows where they disagree, to determine whether one is systematically stronger.
**Expanded:** A plain accuracy comparison (98.24% vs. 97.20%) treats the gap as if it were
obviously real. McNemar's test instead asks a sharper question: among the rows where the models
actually disagree, is the split between them (118 baseline-wins vs. 58 LSTM-wins) more lopsided
than pure chance would produce?
**Example:** `evaluation/significance.py`'s `mcnemar_test()`, called in
`notebooks/05_model_comparison.ipynb` Step 9, reused unchanged from the Reuters ablation study.
**Follow-up:** "Why not a simple two-proportion z-test on the two accuracies instead?" → Because
the two accuracies aren't independent — both models were scored on the *same* rows, so their
errors are correlated. McNemar's test is specifically built for this paired situation; treating
the two accuracies as independent samples would be a subtly incorrect statistical test.

### 4. What is a discordant pair in the context of McNemar's test?
**Short answer:** A test row where the two models disagree — one got it right, the other got it wrong.
**Expanded:** Rows where both models agree (both right, or both wrong) carry no information about
which model is *better*, only about how hard the row is — so McNemar's test discards them and
looks only at the 176 discordant rows out of 5,796.
**Example:** `mcnemar_test()`'s `n_discordant = n_b + n_c`, where `n_b` = baseline right/LSTM
wrong (118) and `n_c` = LSTM right/baseline wrong (58).
**Follow-up:** "What if there had been very few discordant rows?" → The test would have very
little statistical power to detect any difference at all — with only a handful of disagreements,
almost any split would look consistent with chance, which is exactly what happened in the Reuters
ablation study (only 9 discordant rows, p = 1.0).

### 5. What is the null hypothesis of McNemar's test in this comparison?
**Short answer:** That among the discordant rows, each model is equally likely to be the one
that's correct — the disagreements look like a fair coin flip.
**Expanded:** Rejecting this null hypothesis means the 118-vs-58 split is unlikely to have arisen
by chance if the two models were truly equally accurate on the rows where they differ.
**Example:** `docs/model_comparison_report.md` Section 3 states this null hypothesis explicitly
before reporting the p-value, rather than jumping straight to "the p-value is small."
**Follow-up:** "What would a p-value close to 1 have meant here, hypothetically?" → It would have
meant the 118-vs-58 split (or something equally or more lopsided) is exactly the kind of outcome
you'd expect from 176 fair coin flips — no evidence of a systematic difference, as was actually
found in the Reuters ablation study.

### 6. What does a p-value actually tell you?
**Short answer:** The probability of seeing a result at least as extreme as what was observed, if
the null hypothesis were actually true.
**Expanded:** A p-value of 0.000007 means: *if* the baseline and LSTM were truly equally likely
to be correct on discordant rows, a split at least as lopsided as 118-vs-58 would happen only
about 7 times in a million repeated experiments. It does **not** mean "there's a 0.0007% chance
the null hypothesis is true" — a common misinterpretation worth explicitly avoiding in a viva.
**Example:** `evaluation/significance.py`'s `exact_p_value`, computed via `scipy.stats.binomtest`.
**Follow-up:** "So a tiny p-value proves the baseline is much better?" → No — it proves the
difference is unlikely to be chance; it says nothing about *how large* the difference is. That's
exactly the distinction Question 7 covers.

### 7. What is the difference between statistical significance and practical (effect-size) significance?
**Short answer:** Statistical significance asks "is this effect probably not exactly zero?";
practical significance asks "is this effect big enough to actually matter?" — and a result can be
one without the other.
**Expanded:** This project's own two statistical tests are a perfect side-by-side illustration:
the Reuters ablation study found *no* statistical significance (p = 1.0, a genuinely negligible
effect), while Phase 6 found statistical significance (p = 0.000007) for a gap that is still only
about 1 accuracy point and affects only 3.0% of test rows — real and reliable, but modest.
**Example:** `docs/model_comparison_report.md` Section 3's explicit callout: "do not confuse
'statistically significant' with 'a large practical difference.'"
**Follow-up:** "Could a difference be practically large but not statistically significant?" → Yes
— with a very small test set, even a large-looking accuracy gap might not survive a formal test,
because there wouldn't be enough discordant rows to rule out chance confidently.

### 8. Why can a large test set detect a "small" difference as statistically significant?
**Short answer:** More data means less uncertainty about the true underlying rates, so even a
modest, consistent effect becomes distinguishable from chance.
**Expanded:** With 5,796 test rows (176 discordant), the 118-vs-58 split had enough data behind it
for the binomial test to confidently rule out "this is just noise" — the same 118:58 *ratio* seen
in only, say, 10 discordant rows (roughly 7:3) would not have been nearly as conclusive.
**Example:** Compare Phase 6's 176 discordant rows (p = 0.000007) against the Reuters ablation's
9 discordant rows (p = 1.0) — a smaller absolute disagreement count made even a real-looking
split statistically inconclusive there.
**Follow-up:** "Does this mean bigger test sets are always better?" → For statistical power, yes;
practically, test set size is constrained by how much labeled data can be held out without
starving the training set — this project's 70/15/15 split (the project specification) already balances that.

### 9. What is prediction agreement analysis, and how is it different from McNemar's test?
**Short answer:** It categorizes every test row into one of four outcomes (both correct, both
wrong, only A correct, only B correct) — McNemar's test only ever looks at the last two categories.
**Expanded:** McNemar's test is a hypothesis test about *which* model is stronger; agreement
analysis is a descriptive breakdown of *how much overlap* exists in what the two models get right
or wrong, which is useful for questions McNemar's test doesn't answer, like "would an ensemble
help much?"
**Example:** `evaluation/agreement.py`'s `compute_agreement_counts()`, purpose-built for Phase 6
and reused nowhere else yet — a genuinely new (small) module, unlike most of `evaluation/`
which was reused unchanged from Phase 4/5.
**Follow-up:** "Could you derive McNemar's test's numbers from the agreement table?" → Yes —
`a_only_correct` and `b_only_correct` in the agreement table are exactly `n_b` and `n_c` in
McNemar's test; the two analyses share the same underlying counts, viewed through different
lenses.

### 10. What does it mean if two models rarely fail on the same articles?
**Short answer:** It suggests the two models have different, largely non-overlapping blind spots
— they are not simply "the same errors, at different rates."
**Expanded:** In this project, only 44 of 5,796 test articles (0.76%) fooled *both* models — the
rest of the 220 total errors are split fairly evenly between each model's own exclusive mistakes.
**Example:** `report/tables/prediction_agreement.csv` and `report/figures/error_overlap.png`.
**Follow-up:** "Why does this matter practically?" → It's the evidence behind the "ensemble
ceiling" discussion (Question 34) — if the errors barely overlapped, combining the two models'
predictions could in principle recover many of the individual mistakes.

### 11. What is an ensemble model, and why wasn't one built here?
**Short answer:** A model that combines multiple individual models' predictions (e.g. by voting)
to try to do better than any single one alone; not built here because this project's scope was
to understand and compare two individual, fully-explainable models, not to maximize accuracy.
**Expanded:** `docs/model_comparison_report.md` Section 4 notes the theoretical ceiling (up to
~99.24% if every disagreement were resolved correctly) as a finding worth naming, explicitly
flagged as future work rather than pursued now.
**Example:** A simple majority-vote or "trust the baseline unless the LSTM is very confident"
rule are both plausible ensemble strategies that were not implemented.
**Follow-up:** "What would an ensemble cost you?" → Interpretability (now two models' worth of
internal logic to explain instead of one) and deployment complexity (both models' dependencies
needed at once) — exactly the trade-offs Section 9's Business Recommendation weighs against.

### 12. What is inference time, and why does it matter separately from training time?
**Short answer:** How long a trained model takes to produce one prediction on new input; it
matters separately because training happens rarely (once per retrain) while inference happens
every time a user submits an article.
**Expanded:** A model could be slow to train but fast enough at inference to be perfectly usable
in production (exactly the LSTM's situation here: 599s to train, but only 1.43ms/article to
predict — both fast enough for a live web API).
**Example:** `notebooks/05_model_comparison.ipynb` Step 3 times both models' full pipeline (TF-IDF
transform + predict, or tokenize + pad + predict) over all 5,796 test articles.
**Follow-up:** "Is 1.43ms/article actually a problem for the FastAPI app planned in Phase 7?" →
No — even the slower LSTM comfortably meets typical web API latency budgets (usually measured in
tens to hundreds of milliseconds); the *training* time gap is the one that matters for developer
iteration speed, not the deployed user experience.

### 13. Why is training time often more important than inference time during model development?
**Short answer:** Because developers retrain models repeatedly while iterating (trying new
features, fixing bugs, tuning hyperparameters), while a deployed model's inference happens once
per prediction and is usually already fast enough for both models here.
**Expanded:** The baseline's 0.25-second training time means a developer can try an idea and see
the result almost instantly; the LSTM's ~10-minute training time turns the same experimentation
loop into a genuine interruption to a workflow.
**Example:** This project's own experience — the baseline notebook (Phase 4) and the LSTM
notebook (Phase 5) took dramatically different amounts of wall-clock time to iterate on during
development.
**Follow-up:** "Does this argument change for a company retraining a model on a schedule (e.g.
nightly)?" → Somewhat — a slow but automated nightly retrain is more tolerable than a slow
interactive development loop, but 10 minutes is still far cheaper to run nightly than to debug
interactively, so the practical cost gap remains real either way.

### 14. What is model size, and why does it matter for deployment?
**Short answer:** How much disk space (and, correspondingly, memory) a trained model's saved
artifacts occupy; it matters because deployment environments (containers, serverless functions,
mobile/edge devices) often have real size and memory constraints.
**Expanded:** The LSTM's total inference artifacts (28.69 MB) are about 32x larger than the
baseline's (0.90 MB) — for this project's scale, both are small in absolute terms, but the ratio
would matter more in a resource-constrained deployment target.
**Example:** `report/tables/model_comparison.csv`'s "Model Size" rows, computed directly from
`Path.stat().st_size` on the saved artifact files in `notebooks/05_model_comparison.ipynb`.
**Follow-up:** "Which artifact accounts for most of the LSTM's size?" → `model.keras` itself
(24.57 MB of the 28.69 MB total) — dominated by the Embedding layer's 2,000,000-number table, the
same component that dominates its parameter count.

### 15. What counts as a "trainable parameter," and why is Logistic Regression's coefficient count a meaningful analogue?
**Short answer:** A trainable parameter is any number the model adjusts during training to fit
the data; Logistic Regression's 20,000 word coefficients plus 1 intercept are learned the same
conceptual way (adjusted to minimize a loss function), even though the optimization method
(`lbfgs`, a single-shot solver) differs from the LSTM's iterative gradient descent.
**Expanded:** Strictly, "trainable parameters" is Deep Learning terminology tied to
backpropagation; this project uses the term loosely across both models because the underlying
idea — a table of learned numeric weights applied to input features — is genuinely comparable,
just arrived at differently.
**Example:** `report/tables/model_comparison.csv` reports 20,001 for the baseline and 2,044,353
for the LSTM, with a footnote-style caveat in `docs/model_comparison_report.md` Section 1
explaining the terminology difference rather than silently treating the two as identical.
**Follow-up:** "Does having ~100x fewer parameters mean the baseline is a 'worse' model?" → No —
parameter count measures model capacity/complexity, not quality; this comparison's whole point
(Section 7) is that more parameters did not translate into a better result here.

### 16. Why is Deep Learning not always superior to traditional Machine Learning?
**Short answer:** Deep Learning's advantage comes from learning its own representations from
data, but that advantage needs enough data (and often enough tuning) to pay off — when the
distinguishing signal in a dataset is already simple/surface-level, a traditional model can match
or beat it while using far fewer resources.
**Expanded:** This project supplies a direct, measured example rather than a general claim: on
this dataset, given this scope of tuning, the LSTM did not outperform TF-IDF + Logistic
Regression on any metric.
**Example:** `docs/model_comparison_report.md` Section 7's three converging lines of evidence
(surface-level dataset signal, moderate data size for a from-scratch embedding, observed training
instability).
**Follow-up:** "Is this a general finding about LSTMs vs. Logistic Regression?" → No — explicitly
stated in Section 7 as specific to this dataset and this configuration; a different dataset, more
data, or pretrained embeddings could change the outcome.

### 17. What is the difference between feature engineering and feature learning?
**Short answer:** Feature engineering computes input features using a fixed, human-designed
formula before training; feature learning lets the model discover its own useful representation
of the input during training.
**Expanded:** TF-IDF is feature engineering — term frequency × inverse document frequency is a
fixed statistical formula, unchanged regardless of what the model eventually learns. The LSTM's
Embedding layer is feature learning — it starts as random numbers and is shaped entirely by what
turns out to help predict the label correctly.
**Example:** `docs/model_comparison_report.md` Section 1's comparison table, and Phase 5's own
Question 1/2 in this document (`## Phase 5`) covering embeddings in more depth.
**Follow-up:** "Is feature learning always better than feature engineering?" → No — feature
learning needs enough data to learn good representations from scratch; feature engineering can
outperform it when the engineered features already capture what matters, exactly the situation
this project found itself in.

### 18. What is the difference between a bag-of-words representation and a sequence representation?
**Short answer:** Bag-of-words represents a document as an unordered collection of word counts/
scores, discarding order; a sequence representation preserves the order words appeared in.
**Expanded:** TF-IDF vectors are bag-of-words — shuffling a document's words produces the exact
same vector. The LSTM's padded integer sequences preserve order — shuffling the words would
produce a different (and likely much worse-scoring) input.
**Example:** `"did not confirm"` and `"did confirm"` produce different TF-IDF-adjacent word
counts only via the presence/absence of "not," while a sequence model can, in principle, use the
specific position of "not" relative to "confirm."
**Follow-up:** "Did this dataset actually need sequence information to be classified well?" →
Apparently not much — Section 7 of `docs/model_comparison_report.md` argues the dominant signal
here is closer to word-choice/style (which bag-of-words captures fine) than order-dependent
meaning.

### 19. What role does dataset size play in whether Deep Learning outperforms simpler models?
**Short answer:** Deep Learning models, especially ones learning representations from scratch
(no pretrained embeddings), generally need larger datasets to fully realize their capacity
advantage over simpler models.
**Expanded:** ~27,000 training articles is enough for a linear model with sparse features to
generalize well, but is a comparatively modest amount for training a ~2-million-parameter
embedding + LSTM entirely from scratch.
**Example:** `docs/lstm_model_report.md`'s Discussion section names this as one of three
plausible, non-exclusive explanations for the LSTM's shortfall, repeated and reinforced in
`docs/model_comparison_report.md` Section 7.
**Follow-up:** "What's the fix, if more data isn't available?" → Pretrained embeddings (GloVe/
Word2Vec) or transfer learning from a pretrained transformer — both let the model start from
representations already learned on a much larger, external corpus, rather than learning word
meaning from this dataset's ~27,000 rows alone. Named as future work in Section 11, deliberately
not implemented here per this project's scope.

### 20. What is model interpretability, and why does the baseline have more of it than the LSTM?
**Short answer:** Interpretability is how directly a model's internal reasoning can be explained
to a human; the baseline has it because every prediction traces back to a specific, signed weight
per word, while the LSTM's internal gate computations have no equally direct human-readable
equivalent.
**Expanded:** `docs/baseline_model_report.md`'s Feature Importance analysis reads coefficients
directly off the fitted Logistic Regression — a complete, exact account of what the model
learned. Explaining one specific LSTM prediction would require additional techniques (e.g.
attention visualization) that this architecture doesn't include.
**Example:** `report/figures/baseline_top_features.png` shows, in one picture, exactly which
words push a prediction toward Fake or Real; no LSTM equivalent exists in this project.
**Follow-up:** "Why does interpretability matter for a fake-news detector specifically?" → A user
told "this article was flagged as Fake" reasonably wants to know *why* — an interpretable model
can answer that directly; an opaque one cannot, which matters for user trust and for defending a
prediction if challenged.

### 21. What is deployment complexity, and how does it differ from raw accuracy?
**Short answer:** Deployment complexity covers everything needed to actually run a model in
production — dependencies, runtime environment, resource requirements — separate from how
accurate its predictions are.
**Expanded:** The baseline needs only scikit-learn + joblib at inference time; the LSTM needs a
full TensorFlow/Keras runtime — a materially heavier dependency to install, containerize, and
keep updated, regardless of either model's accuracy.
**Example:** `requirements.txt`'s pinned `tensorflow==2.21.0` vs. `scikit-learn==1.7.2` —
TensorFlow alone is a much larger package with more transitive dependencies.
**Follow-up:** "Could this tip a close accuracy race in the simpler model's favor?" → Yes — this
is exactly Section 9's point: performance is only one of several considerations in the Business
Recommendation, not the sole deciding factor.

### 22. Why might a business choose a slightly less accurate model over a more accurate one?
**Short answer:** Because raw accuracy is only one of several costs and benefits that matter in
production — training/inference cost, interpretability, maintainability, and hardware
requirements can outweigh a small accuracy gap.
**Expanded:** In this project's case, the *more* accurate model (the baseline) also wins on every
other dimension, so the recommendation isn't even a trade-off — but the general principle (accuracy
isn't the only axis that matters) holds regardless.
**Example:** `docs/model_comparison_report.md` Section 9's Business Recommendation table, which
lists seven separate considerations and checks which model each one favors.
**Follow-up:** "Give a hypothetical where the *less* accurate model would be the right choice
anyway." → If interpretability were a hard legal/regulatory requirement (e.g. a system that must
justify each decision) and the accuracy gap were small, the more interpretable model could be the
correct choice even if a competitor model scored marginally higher.

### 23. What is a threats-to-validity discussion, and why include one?
**Short answer:** An explicit, honest accounting of the specific ways a project's conclusions
might not hold up — under different conditions, different data, or different assumptions —
included so the project's claims aren't overstated.
**Expanded:** Every empirical project has limits; naming them precisely (rather than letting an
examiner discover them unprompted) is a mark of rigor, not weakness.
**Example:** `docs/threats_to_validity.md`, covering internal, external, and construct validity
plus dataset limitations, biases, and future improvements.
**Follow-up:** "Doesn't listing limitations undermine confidence in the results?" → The opposite —
a report that names its own limitations precisely is *more* credible, because it shows the author
understands exactly what was and wasn't established, rather than overclaiming.

### 24. What is the difference between internal validity, external validity, and construct validity?
**Short answer:** Internal validity asks whether the project's own experiments support their
conclusions; external validity asks whether those conclusions generalize beyond this exact setup;
construct validity asks whether what was measured actually reflects what it claims to measure.
**Expanded:** All three are threatened differently in this project: internal validity by the
single train/test split and lack of hyperparameter tuning; external validity by the single,
two-source, English-only, time-bounded dataset; construct validity by the Reuters ablation
study's own finding that the label is substantially a proxy for writing style, not verified truth.
**Example:** `docs/threats_to_validity.md`'s three separate top-level sections, each populated
with concrete, project-specific examples rather than generic textbook statements.
**Follow-up:** "Which of the three is this project's biggest weakness?" → Construct validity —
the Reuters ablation study provides direct, already-collected evidence that both models are
substantially measuring source-style, not truthfulness, which is a more fundamental limitation
than the internal/external validity concerns.

### 25. What does it mean for a label to be a "proxy" rather than ground truth?
**Short answer:** The Fake/Real label reflects which source list an article came from (as
assembled by the dataset's original creators), not an independent, article-by-article
fact-check of whether its claims are true.
**Expanded:** A model can score extremely well at predicting *this proxy label* while not
directly measuring the concept ("is this claim true") the project ultimately cares about — the
gap between the two is exactly what the Reuters ablation study and Section 5's construct-validity
discussion are about.
**Example:** `docs/threats_to_validity.md`'s Construct Validity section states this directly:
"'Real' here means 'published by Reuters, as included in this dataset's collection process.'"
**Follow-up:** "How would you actually measure the true construct (veracity) instead?" →
Independent fact-checking of each article's claims against verified sources — a substantially
larger, human-labor-intensive undertaking, explicitly out of scope here and named as future work.

### 26. Why did the TF-IDF + Logistic Regression baseline outperform the LSTM on this dataset?
**Short answer:** Because this dataset's Fake/Real distinction is dominated by surface-level,
style-based cues that a linear bag-of-words model detects very efficiently, while the LSTM's
main structural advantage (using word order) matters less here and its from-scratch embedding
had a comparatively modest amount of data to learn from.
**Expanded:** Three converging pieces of evidence support this (`docs/model_comparison_report.md`
Section 7): the Reuters ablation study's "redundantly encoded style signal" finding, the LSTM's
visibly noisy training history, and the moderate training-set size relative to a ~2-million-
parameter from-scratch model.
**Example:** The LSTM's own error analysis (`docs/lstm_model_report.md`) found its false
negatives concentrated in international-affairs articles with less-common names — exactly the
kind of vocabulary a from-scratch embedding learns weakest on with limited data.
**Follow-up:** "Is this the LSTM's 'fault,' or the dataset's?" → Neither, exactly — it's a
mismatch between what this specific architecture is good at and what this specific dataset
happens to reward, which is precisely why the comparison needed to be run rather than assumed.

### 27. What did the Reuters ablation study contribute to explaining this project's final comparison?
**Short answer:** It established, with direct evidence (an ablation experiment plus McNemar's
test), that the baseline's high accuracy substantially reflects "Reuters wire style vs. this
dataset's blog style" rather than a single removable shortcut — the key piece of evidence behind
Section 7's explanation of why the baseline performed so well in the first place.
**Expanded:** Without that study, Phase 6 would only be able to say "the baseline scored higher"
without being able to say *why* with any evidence — the ablation study is what turns "the baseline
uses a shortcut" from a suspicion into a documented, tested finding.
**Example:** `docs/reuters_ablation_study.md`'s "reshuffle, don't replace" finding (removing
"reuters" just promoted the next-strongest correlated word) is cited directly in
`docs/model_comparison_report.md` Section 7.
**Follow-up:** "Should the same ablation be run on the LSTM?" → Yes — named explicitly as a
documented gap in `docs/threats_to_validity.md`'s Internal Validity section; it was recommended
back in `docs/reuters_ablation_study.md` itself but not yet carried out.

### 28. Why was the LSTM's training history described as "noisy," and why does that matter for this comparison?
**Short answer:** Because its validation loss spiked sharply at epoch 2 and both training loss
and accuracy swung considerably in the final two epochs, rather than improving smoothly —
evidence the model did not converge as cleanly as it might with further tuning.
**Expanded:** This matters for the comparison because it's one of the three explanations
(Section 7) for the LSTM's shortfall, and because it's a genuine, observed complexity/
maintainability cost (Section 6) distinct from the baseline's single, fully deterministic
training run.
**Example:** `report/figures/lstm_training_history.png` — epoch 2's validation loss of 0.61
against a smoother surrounding trend, and epoch 10's training loss jumping back up to 0.31.
**Follow-up:** "What would you try first to fix this instability?" → A lower or scheduled
learning rate, or gradient clipping — both named as concrete, un-pursued next steps in
`docs/lstm_model_report.md` and `docs/model_comparison_report.md` Section 7.

### 29. Why does the LSTM show more errors on international/foreign-name articles than the baseline?
**Short answer:** A from-scratch Embedding layer learns a weaker, less reliable vector for words
it sees rarely during training, while TF-IDF simply assigns a small (but still usable) weight to
a rare word — a more graceful way to handle scarcity.
**Expanded:** Names like "Merkel" or "Kuroda" appear infrequently in ~27,000 training articles
concentrated on U.S. political news; the LSTM's embedding for such words is correspondingly
poorly positioned, while the baseline's linear weight for the same word is just proportionally
small, not poorly *shaped*.
**Example:** `docs/lstm_model_report.md`'s error analysis cites *"Merkel scolds ally to shield
coalition talks from weedkiller row"* as a real false negative — a genuine Reuters (Real) article
the LSTM called Fake.
**Follow-up:** "Would pretrained embeddings fix this?" → Likely to help significantly — a
pretrained embedding (e.g. GloVe) already encodes reasonable vectors for common foreign names and
international vocabulary from a much larger external corpus, rather than needing to learn them
from this dataset's limited exposure alone.

### 30. Why wasn't `evaluation/experiments.csv` updated with a new row in Phase 6?
**Short answer:** Because its schema (`evaluation/experiment_log.py`) records *training runs*
(accuracy, training time, etc.), and Phase 6 didn't train a new model — it only re-ran inference
on two already-frozen models and performed statistical analysis on the results.
**Expanded:** Adding a row would misrepresent Phase 6 as a new "experiment" in the training sense,
when its actual output (a statistical comparison and a set of report tables/figures) doesn't fit
that CSV's columns meaningfully.
**Example:** The Reuters ablation study set a precedent for this exact judgment call in reverse —
it *did* log a new row, because it *did* train a genuinely new (if temporary) model; Phase 6 has
no equivalent new model to log.
**Follow-up:** "Where does the McNemar's test result get recorded, then?" → In
`report/tables/model_comparison.csv`, `report/tables/prediction_agreement.csv`, and
`docs/model_comparison_report.md` directly — a deliberate choice to keep `experiments.csv`
strictly reserved for training-run records.

### 31. Why was McNemar's test result different in Phase 6 (statistically significant) than in the Reuters ablation study (not significant)?
**Short answer:** Because there were far more discordant rows to work with here (176 vs. 9), and
the split among them was more lopsided (118:58, about 2:1) than the ablation study's near-even
4:5 split — both factors make a systematic difference easier to detect statistically.
**Expanded:** The Reuters ablation study compared two *very similar* models (the same baseline
architecture, with or without one token) — a genuinely tiny, essentially undetectable difference
was expected and found. Phase 6 compares two *architecturally different* models, where a real
(if modest) performance gap exists and had enough discordant rows to be statistically confirmed.
**Example:** `evaluation/significance.py`'s `mcnemar_test()` — the exact same function, same
formula, applied to two different pairs of predictions, producing p = 1.0 in one case and
p = 0.000007 in the other.
**Follow-up:** "Does this mean McNemar's test 'worked' in Phase 6 and 'failed' in the ablation
study?" → No — both results are the test working correctly; they simply reflect two genuinely
different underlying situations (near-zero difference vs. a real, modest difference).

### 32. What would need to change for the Business Recommendation to favor the LSTM instead?
**Short answer:** The LSTM would need to close (or reverse) its accuracy gap, or the other
considerations (training cost, interpretability, deployment complexity) would need to become less
important relative to squeezing out extra accuracy — e.g. a dataset where word order/negation is
genuinely decisive, or far more training data / pretrained embeddings.
**Expanded:** `docs/model_comparison_report.md` Section 9 states this explicitly as the
recommendation's own boundary condition, rather than presenting the recommendation as permanent.
**Example:** A hypothetical dataset built specifically to require negation-sensitivity (e.g.
carefully paired "did/did not confirm" claims) would be a case tailor-made to reward the LSTM's
structural advantage over TF-IDF.
**Follow-up:** "Is the current recommendation dataset-specific or a general claim about LSTMs?" →
Dataset-specific — Section 9 explicitly frames it as conditional on *this* dataset's
characteristics, not a general "Logistic Regression beats LSTM" claim.

### 33. Why does `evaluation/agreement.py` exist as a separate module from `evaluation/significance.py`?
**Short answer:** They answer genuinely different questions from the same underlying data —
`significance.py` performs a hypothesis test (is one model reliably better?), while
`agreement.py` produces a descriptive breakdown (how much do the two models' correct/incorrect
predictions overlap?) — keeping them separate keeps each module's single responsibility clear.
**Expanded:** This mirrors the same modular, single-responsibility design already used throughout
the project (e.g. `preprocessing/text_cleaning.py`'s individually-named functions) — one new,
small module for one new, genuinely new kind of analysis, rather than growing an existing module
beyond its original purpose.
**Example:** `evaluation/agreement.py`'s `compute_agreement_counts()` and `plot_agreement_bar()`,
used only in `notebooks/05_model_comparison.ipynb` — the one genuinely new piece of evaluation
code this phase needed, exactly analogous to how `evaluation/training_history.py` was Phase 5's
one new addition.
**Follow-up:** "Could this functionality have just been written inline in the notebook instead?"
→ It could have, since it's used only once — but keeping it as a small, testable, reusable module
matches this project's established convention (the project specification's Notebook Philosophy) that
non-trivial logic belongs in `evaluation/`/`training/`/`preprocessing/`, not only inside a
notebook cell.

### 34. What is the theoretical "ensemble ceiling" identified in the Prediction Agreement Analysis, and why wasn't an ensemble built?
**Short answer:** If an ensemble of the two models could correctly resolve every one of the 176
discordant rows, accuracy would rise to about 99.24% (only the 44 "both wrong" rows would remain
unrecoverable) — a theoretical upper bound, not a guaranteed outcome, and building one was outside
this project's scope.
**Expanded:** Reaching that ceiling would require an ensemble strategy that reliably knows *which*
model to trust on each disagreement — a nontrivial problem in itself, not solved just by having
two models available.
**Example:** `docs/model_comparison_report.md` Section 4's arithmetic: 5,796 − 44 = 5,752 correct
in a "perfect oracle" ensemble, i.e. 5,752 / 5,796 ≈ 99.24%.
**Follow-up:** "Is 99.24% a realistic ensemble target, or just a ceiling?" → Just a ceiling — a
real ensemble (e.g. majority vote, or a meta-model trained to pick between them) would recover
only some, not all, of the 176 discordant rows, since it can't magically know which model is
right on any given row without additional information.

### 35. Why does this project conclude the LSTM was still worth building, despite losing to the baseline?
**Short answer:** Because building it was the only way to actually test whether the baseline's
strong result was "good enough" or whether a more powerful architecture could do meaningfully
better — an untested assumption either way would have been weaker science than a tested, honestly
reported result.
**Expanded:** `docs/baseline_model_report.md` (Phase 4) explicitly framed the baseline's whole
purpose as a reference point for exactly this later check; the LSTM's shortfall is therefore not
a wasted detour but the answer to the question the project set out to ask.
**Example:** The LSTM's different error profile (Section 5, more errors on international/
foreign-name articles) is a genuinely new finding this project would not have if it had stopped
after Phase 4 — the LSTM contributed real information even while losing on aggregate accuracy.
**Follow-up:** "What's the single biggest methodological lesson from this whole comparison?" →
That a fair, rigorous comparison — same split, same statistical tests, same honesty about
negative results — matters more for a project's credibility than which model happens to "win."

---

## Concepts I Must Understand

| Concept | Definition | Why it's used | Analogy | Example in this project |
|---|---|---|---|---|
| **McNemar's test** | Paired statistical test comparing two classifiers on discordant rows only | Correctly accounts for both models being scored on the same, non-independent test set | Asking only the people who changed their mind between two votes which way they leaned, ignoring everyone who voted the same both times | `evaluation/significance.py: mcnemar_test()` |
| **Discordant pair** | A test row where the two models disagree | The only rows that carry evidence about which model is stronger | The one disputed call in an otherwise unanimous jury | 176 out of 5,796 test rows |
| **p-value** | Probability of an effect this extreme under the null hypothesis | Quantifies how surprising the observed result would be if there were truly no difference | How unlikely a coin would need to be to land the same way 20 times in a row, by chance | 0.000007 in Phase 6's comparison |
| **Statistical vs. practical significance** | Whether an effect is probably nonzero, vs. whether it's big enough to matter | Prevents over- or under-interpreting a p-value on its own | A very precise scale detecting a 1-gram difference vs. whether that gram actually matters for your recipe | p = 0.000007 for only a ~1-point accuracy gap |
| **Prediction agreement analysis** | Categorizing every test row into both-correct/both-wrong/exclusive-win buckets | Reveals whether two models' errors overlap or are largely independent | Comparing two students' wrong answers on the same exam to see if they missed the same questions | `evaluation/agreement.py` |
| **Ensemble ceiling** | The best accuracy achievable if an ensemble always picked the correct model per row | Bounds how much value combining models could realistically add | The best possible team score if you always deferred to whichever teammate happened to be right | ≈99.24% in this comparison |
| **Feature engineering** | Computing input features via a fixed, human-designed formula | Simple, fast, interpretable; doesn't need much data to work well | Using a recipe's fixed measurements rather than tasting and adjusting as you go | TF-IDF (`training/baseline.py`) |
| **Feature learning** | Letting the model discover its own useful input representation from data | Can capture subtler patterns, but needs enough data to learn well | Learning to cook by tasting and adjusting over many attempts, rather than following a fixed recipe | Embedding layer (`training/lstm.py`) |
| **Model interpretability** | How directly a model's reasoning can be explained to a human | Builds trust, aids debugging, supports justifying individual decisions | A teacher who shows their full work vs. one who only gives the final answer | Logistic Regression coefficients |
| **Deployment complexity** | Everything needed to actually run a trained model in production | Distinct from accuracy; affects real operational cost | The difference between shipping a single lightweight tool vs. an entire workshop | scikit-learn+joblib vs. TensorFlow runtime |
| **Threats to validity** | An explicit accounting of a study's specific limitations | Prevents overclaiming; standard practice in rigorous research | A map that also marks where the terrain hasn't been surveyed | `docs/threats_to_validity.md` |
| **Construct validity** | Whether what was measured reflects what it claims to measure | Distinguishes "predicts the label well" from "detects truth" | A thermometer that actually measures room brightness, not temperature, despite correlating with both | Reuters wire-style vs. genuine veracity |
| **Label as proxy** | A recorded label standing in for, but not identical to, the true concept of interest | Clarifies exactly what a model has and hasn't been shown to do | A restaurant's star rating as a proxy for "food quality," shaped by things beyond the food itself | Fake/Real labels reflecting source list, not fact-checking |

---

## Common Mistakes

- **Comparing two accuracy numbers and stopping there.** Without a paired statistical test
  (McNemar's), there's no principled way to say whether a gap like 98.24% vs. 97.20% is real or
  just this particular test set's noise.
- **Treating a small p-value as proof of a large, important difference.** A tiny p-value only
  means the effect is probably not exactly zero — Section 3's distinction between statistical and
  practical significance exists specifically to guard against this.
- **Recommending the higher-accuracy model without weighing anything else.** Section 9's Business
  Recommendation deliberately checks seven separate considerations — accuracy is one input, not
  an automatic verdict.
- **Assuming both models fail on the same articles just because their aggregate accuracy is
  close.** The agreement analysis found the opposite here — only 0.76% of articles fool both
  models, meaning their error profiles are substantially different despite similar overall scores.
- **Logging a non-training analysis into `evaluation/experiments.csv` just because "it's an
  experiment."** The CSV's schema is specifically for training runs; forcing an unrelated
  analysis into it would blur what the file is for.
- **Skipping a threats-to-validity discussion because the results "look good."** A high accuracy
  number is exactly when this project's own history (label leakage, the Reuters ablation finding)
  shows a careful validity check is most needed, not least.

## End-of-Phase Summary

**Things I should confidently explain:**
- Why McNemar's test, not a plain accuracy comparison, is the correct way to compare two models
  scored on the same test set — and what its null hypothesis actually means.
- The difference between statistical significance and practical significance, using this
  project's own two contrasting results (Reuters ablation: p = 1.0; Phase 6: p = 0.000007) as a
  concrete illustration.
- The four-way prediction agreement breakdown and what it implies (largely non-overlapping error
  profiles despite similar aggregate accuracy).
- The full set of reasons (Section 7 of `docs/model_comparison_report.md`) for why the baseline
  outperformed the LSTM on this specific dataset — and why that is not a general claim about LSTMs.
- The Business Recommendation's reasoning: every dimension (not just accuracy) pointed toward the
  baseline, which is what makes it a genuinely balanced recommendation.
- The three categories of threats to validity (internal, external, construct) and at least one
  concrete example of each from this specific project.

**Topics to revise before a viva:**
- The exact numbers in `report/tables/model_comparison.csv` and
  `report/tables/prediction_agreement.csv` — training/inference time, model size, parameter
  counts, and the four agreement-category counts.
- The precise wording of McNemar's null hypothesis and what rejecting it does and doesn't imply.

**Key takeaways:**
- A model comparison is not complete until it is both statistically grounded (McNemar's test) and
  qualitatively examined (error analysis, agreement analysis) — a single accuracy number, however
  confidently reported, is not sufficient evidence on its own.
- This project's central, most defensible finding across all six phases is not "the baseline
  is better" or "the LSTM is worse" in the abstract — it is that **a rigorous, fair, honestly
  reported comparison matters more than which model wins**, and every phase's methodology
  (frozen artifacts, shared seeded splits, reused evaluation code, documented negative results)
  was built specifically to make that comparison trustworthy.
