# Dissertation Writing Style Guide

This guide applies to every chapter of the final dissertation (`docs/final_dissertation.md`),
from Part 2 onward. It exists so the whole document reads in one consistent voice, even though
it's being written chapter by chapter across many sessions.

---

## Voice

- Write like a final-year BCA student describing a project they actually built, not like a
  research paper trying to impress a reviewer.
- Use "This project," "The system," or "The model" as the usual subject of a sentence, not "I."
  ("I" is only used in the Declaration and Acknowledgement, which are personal statements.)
- Be formal but conversational enough to sound human. A dissertation chapter can still read
  naturally, the way a clear textbook explanation does.

## Sentence Length

- Prefer short to medium-length sentences. One idea per sentence, most of the time.
- If a sentence has more than two clauses stacked together, split it into two sentences instead.
- Vary sentence length a little so it doesn't feel like a checklist — but never for its own sake;
  clarity always wins over rhythm.

## Word Choices

Prefer plain, direct verbs:

| Prefer | Avoid |
|---|---|
| find, show, use, build, compare, test, check | demonstrate, leverage, facilitate, utilize, thereby, employ |

Avoid exaggerated or promotional language entirely, even when a result is genuinely good:

- revolutionary, cutting-edge, state-of-the-art, groundbreaking
- significantly enhances, robust framework, seamless, highly sophisticated
- "It is important to note that...", "In today's digital age...", overused "Moreover,"/"Furthermore,"
  as paragraph openers

A strong result should be stated plainly with its actual number, not talked up. "The baseline
reached 98.24% accuracy" is better than "The baseline achieved outstanding, industry-leading
accuracy of 98.24%."

## Explaining Technical Concepts

- Introduce a technical term in plain words the first time it appears, then use the short form
  afterward. For example: "TF-IDF, a method that represents text using weighted word
  frequencies," on first use, then just "TF-IDF" from then on.
- Don't over-explain a concept that was already explained earlier in the document — refer back
  instead of repeating the full explanation.
- It's fine for the Testing/Results-style chapters to be a little more technical and numbers-heavy
  than the Introduction — that's expected. Keep sentence structure simple even there.

## Chapter Scope and Repetition

- Each chapter should stick to its own job and not do another chapter's work. Before adding a
  paragraph, check: does this belong here, or does it belong in the Literature Review, System
  Analysis, Methodology, Algorithms, Design, Implementation, Testing, or Discussion chapter
  instead? If it belongs elsewhere, leave it out.
- If a topic is genuinely relevant here but explained in full elsewhere, mention it in one short
  sentence and point to that chapter by name, rather than re-explaining it. This applies both
  ways: refer back to a chapter already written ("as discussed in the Literature Review chapter")
  and forward to one not yet written ("explained in the chapters that follow").
- When revising a chapter for length, repeated or overlapping content with another chapter is the
  first thing to cut — not sentence-level trimming of content that's actually unique to that
  chapter.
- Write each chapter to be concise by default, not padded to hit a page target. A shorter chapter
  that says everything once is better than a longer one that restates things already said
  elsewhere in the dissertation.

## Numbers and Claims

- Every number in the dissertation must come from an actual project artifact (notebooks,
  `evaluation/experiments.csv`, `report/tables/`, `docs/*_report.md`) — never estimated or rounded
  for effect.
- State findings honestly, including the ones that don't flatter the project (the LSTM losing to
  the baseline, McNemar's test being significant but the practical gap being small). This project's
  own findings are already interesting without needing to be oversold.

## Consistency

- Keep terminology identical throughout: "Fake" / "Real" as the two class labels, "LSTM" after
  first introducing it as "Long Short-Term Memory (LSTM)," "the baseline" or "the Logistic
  Regression model" interchangeably but not switching to new names for the same thing.
- Keep the same tone from chapter to chapter — a reader shouldn't be able to tell that different
  chapters were written in different sessions.

## Quick Before/After Reference

| Instead of | Write |
|---|---|
| "This project leverages a robust deep learning framework to significantly enhance detection accuracy." | "This project uses an LSTM model to classify news articles as Fake or Real." |
| "The results demonstrate that the model achieves state-of-the-art performance." | "The model reached 97.20% accuracy on the test set." |
| "It is important to note that the dataset thereby exhibits significant class imbalance." | "The dataset has slightly more Real articles than Fake ones." |
