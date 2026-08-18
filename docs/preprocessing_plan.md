# Preprocessing Design Plan

This document decides, for each candidate text-cleaning technique, whether to use it — before any of it is implemented. No implementation code is written here (see [`data_cleaning_strategy.md`](data_cleaning_strategy.md) for the overall pipeline order this plugs into).

## A note on "tokenization" (read this first)

The word "tokenization" gets used two different ways in this project, and it matters for the "before/after tokenization" question asked below:

1. **Linguistic tokenization** — splitting a string into individual words, e.g. `"Trump said no"` → `["Trump", "said", "no"]`. This is a *preprocessing* step.
2. **Integer tokenization** — the Keras `Tokenizer` converting those words into integer IDs using a vocabulary (`VOCAB_SIZE`, `OOV_TOKEN` in `config/settings.py`), e.g. `["trump", "said", "no"]` → `[42, 7, 15]`. This happens once, right before the Embedding layer.

**Every technique below is a preprocessing step that must happen before integer tokenization (sense 2).** The Keras `Tokenizer` needs to see already-clean words to build a sensible, compact vocabulary — cleaning the text *after* it's already been converted to integers isn't meaningful (you'd have to reverse the encoding first). So "before/after tokenization" is answered once, up front: **all steps below run before integer tokenization**; where a step interacts with linguistic tokenization (sense 1) specifically, that's called out individually.

---

## Summary decision table

| Technique | Decision | Applies to |
|---|---|---|
| Lowercasing | **Use** | Baseline + LSTM |
| HTML removal | **Use** | Baseline + LSTM |
| URL removal | **Use** (delete outright, no placeholder token) | Baseline + LSTM |
| Punctuation removal | **Use** | Baseline + LSTM |
| Number removal | **Do not use** (full removal) | — |
| Stop-word removal | **Use for baseline, do NOT use for LSTM** | Split decision — see below |
| Lemmatization | **Use for baseline; optional/skip for LSTM** | Split decision — see below |
| Stemming | **Do not use** | — |
| Whitespace normalization | **Use** | Baseline + LSTM |

Two techniques get a **split recommendation** between the Phase 4 baseline (TF-IDF + Logistic Regression) and the Phase 5 LSTM, because they interact with word *order*, which only one of those two models can actually use. This is explained in each section.

---

## Lowercasing

- **Why we need it:** without it, `"Trump"`, `"trump"`, and `"TRUMP"` become three separate vocabulary entries instead of one. That splits the training signal for the same word across multiple tokens and wastes embedding capacity.
- **What problem it solves:** vocabulary explosion from case variants of the same word.
- **What happens if skipped:** larger, sparser vocabulary; the model has to independently (re-)learn that `"Trump"` and `"trump"` mean the same thing, using more data and capacity to learn something a one-line preprocessing step gives for free.
- **Trade-off worth naming:** EDA (`01_dataset_analysis.ipynb`) found that ~3.4% of Fake titles are mostly uppercase (clickbait-style headlines), vs. 0% of Real titles — a real signal. Lowercasing discards that specific case-based signal. This is an accepted, deliberate trade-off: the standard benefit (smaller, denser vocabulary) outweighs preserving one narrow stylistic cue, and it keeps the pipeline simple and standard.
- **Decision: Use**, for both models.

---

## HTML removal

- **Why we need it:** scraped web text sometimes retains stray HTML tags (`<br>`, `<div>`) or entities (`&amp;`, `&#39;`).
- **Evidence, not assumption:** checked directly rather than assumed. HTML-like tags appear in 34/23,481 Fake rows and 3/21,417 Real rows; HTML entities appear in 146/23,481 Fake rows and 0/21,417 Real rows. Rare, but real, and one-sided enough that leaving it in is itself a tiny shortcut-signal risk, similar in kind (if far smaller in scale) to the Reuters-tag leakage in `label_leakage_analysis.md`.
- **What happens if skipped:** a handful of garbage tokens (`div`, `amp`, `39`) enter the vocabulary, and — more importantly — an avoidable one-sided artifact is left in the data.
- **Decision: Use**, for both models. Cheap to apply, measurably present, and removes a class-skewed artifact rather than just "cleaning for cleaning's sake."

---

## URL removal

- **Why we need it:** article text contains embedded links (e.g. `pic.twitter.com/...`), which tokenize into meaningless one-off strings.
- **Evidence:** 14.0% of Fake articles contain a URL vs. 0.0% of Real articles (see `label_leakage_analysis.md` §3 for the specific `pic.twitter.com` sub-case).
- **What happens if skipped:** two problems, not one. First, vocabulary bloat from near-unique URL strings. Second — more importantly — URLs are themselves a label-leakage signal in this dataset (only Fake articles have them), so leaving them in lets the model cheat exactly the way `label_leakage_analysis.md` already warned about.
- **Critical implementation detail:** URLs must be **deleted outright**, not replaced with a placeholder token like `<URL>`. A placeholder would still let the model learn "presence of `<URL>` token → Fake" — the identical shortcut, just renamed. Full deletion removes the shortcut, not just its surface form.
- **Decision: Use**, for both models — full deletion, no placeholder.

---

## Punctuation removal

- **Why we need it:** punctuation marks, if left in, become their own vocabulary tokens (`,`, `.`, `!`) without carrying the kind of word-level meaning the model is meant to learn from.
- **What happens if skipped:** larger vocabulary, more noise; word-boundary artifacts (e.g. `"Trump,"` and `"Trump"` treated as different tokens if punctuation isn't separated first).
- **Trade-off worth naming:** sensational punctuation (`"!!!"`, `"?!"`) is a genuinely plausible stylistic marker of tabloid-style writing. Removing punctuation gives that up. Accepted trade-off for the same reason as lowercasing: a simpler, standard pipeline is easier to build, explain, and defend than one trying to hand-engineer a punctuation-based feature on top.
- **Decision: Use**, for both models.

---

## Number removal

- **Why this is different from the others:** numbers in news text are often genuinely informative (dates, statistics, counts, dollar amounts) rather than noise.
- **What happens if we remove them anyway:** real information is thrown away for no clear benefit — arguably worse than doing nothing.
- **What happens if we skip removal entirely:** a legitimate concern is vocabulary bloat from one-off numbers (phone numbers, arbitrary IDs). But this project already has a safety net for exactly that: `VOCAB_SIZE = 20_000` and `OOV_TOKEN = "<OOV>"` in `config/settings.py`. The Keras Tokenizer keeps only the most frequent 20,000 tokens and maps everything else to a single out-of-vocabulary token — so rare numeric tokens get folded into `<OOV>` automatically, without a separate, hand-written number-removal step.
- **Decision: Do not use** (do not blindly strip numbers). Rely on the existing vocabulary cap instead of adding a dedicated cleaning step for a problem that's already handled elsewhere in the pipeline.

---

## Stop-word removal — **split decision between baseline and LSTM**

- **Why it's normally used:** words like `"the"`, `"is"`, `"and"` appear extremely frequently but carry little topic-specific meaning on their own, so a bag-of-words model (like TF-IDF) treats them mostly as noise that dilutes the more informative words.
- **Baseline (TF-IDF + Logistic Regression): Use.** TF-IDF has no concept of word order — it only counts word frequency-weighted-by-rarity. Stop words genuinely add noise here with little to no cost, since the model was never going to use their position anyway.
- **LSTM: Do NOT use.** This is the important nuance: an LSTM is a *sequence* model — it reads words in order and can use function words for grammatical structure and, critically, **negation** (`"not"`, `"never"`, `"no"`). Removing stop words risks turning `"officials did not confirm the claim"` into `"officials confirm claim"` — the literal opposite meaning. Blindly copying the baseline's stop-word list onto the LSTM pipeline would silently damage the one thing an LSTM is specifically good at (using context and order) to fix a problem (bag-of-words noise) that doesn't apply to it in the first place.
- **What happens if we get this backwards:** using stop-word removal on the LSTM risks corrupting negation and grammatical structure; skipping it on the baseline leaves avoidable noise in a model that has no other way to use those words meaningfully. Neither mistake is fatal, but both are avoidable, and getting this right is a genuinely strong point to raise in the viva — it shows the preprocessing was *tailored to each model*, not copy-pasted.

---

## Lemmatization — **split decision, softer than stop-words**

- **Why it's normally used:** reduces inflected forms (`"running"`, `"ran"`, `"runs"`) to a common base form (`"run"`), shrinking the vocabulary and letting a bag-of-words model treat related words as the same feature.
- **Baseline: Use.** TF-IDF benefits directly — without lemmatization, `"run"` and `"running"` are separate columns in the TF-IDF matrix, each seeing less data than their combined form would.
- **LSTM: Optional / skip.** An embedding layer can often learn similar vector representations for inflected forms directly from data (distributional similarity), given enough training examples of each form — lemmatization's benefit is smaller here. It also adds real pipeline complexity: NLTK's default lemmatizer needs part-of-speech information to lemmatize accurately (without it, it mostly only reduces plural nouns correctly), which is one more moving part to build, explain, and potentially get wrong.
- **Decision: Use for the baseline. For the LSTM, skip it (or apply the same simple lemmatizer only if time permits) to keep the pipeline simpler** — this is a "do the simpler thing unless there's a clear benefit" call, consistent with the project specification's Decision Making principle.

---

## Stemming

- **Why it exists:** a cruder, rule-based alternative to lemmatization — chops word endings using fixed rules (e.g. Porter stemmer) without a dictionary.
- **What problem it claims to solve:** the same vocabulary-reduction goal as lemmatization, computed faster and without needing a dictionary.
- **What happens if used:** it frequently produces non-words (`"university"` → `"univers"`, `"argument"` → `"argu"`). These are harder to sanity-check by eye during development, and look unpolished if shown directly in a report or viva demo (e.g. printing top TF-IDF terms).
- **Why not use both:** stemming and lemmatization solve the same problem; using both is redundant, and would make it unclear (to a reader of the report, or to an examiner) which one is actually responsible for a given result.
- **Decision: Do not use.** Lemmatization (used in the baseline, per above) is preferred whenever word-form reduction is wanted, specifically because it produces real dictionary words — easier to inspect, easier to explain, and there is no case in this project where stemming's speed advantage matters (the dataset is not large enough for stemming's speed benefit over lemmatization to matter).

---

## Whitespace normalization

- **Why we need it:** scraped text often contains repeated spaces, tabs, or newlines (e.g. from HTML line breaks). Collapse any run of whitespace into a single space, and strip leading/trailing whitespace.
- **What happens if skipped:** `str.split()`-based word counts (as used throughout the EDA notebook) get slightly distorted by repeated whitespace producing empty strings between separators; downstream tokenization can produce spurious empty tokens.
- **Decision: Use**, for both models. This is the one technique with no real trade-off or debate — always safe, always beneficial, essentially free.

---

## Summary for the report

Two of nine techniques (stop-word removal, lemmatization) are **deliberately applied differently** to the Phase 4 baseline and the Phase 5 LSTM, because they interact with word order and an LSTM can use word order while TF-IDF cannot. The other seven are applied identically to both models. Every "Use" decision above traded off a small, named amount of potentially-useful signal (all-caps styling, sensational punctuation) in exchange for a simpler, more standard, more defensible pipeline — consistent with the project specification's KISS principle and its explicit BCA-viva simplicity check.
