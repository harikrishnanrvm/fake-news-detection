"""Stage 3: modular, single-purpose NLP text-cleaning functions.

Each function does exactly one thing and can be tested/explained on its own,
per the approved design in docs/preprocessing_plan.md. clean_text() composes
them into one configurable pipeline, driven by a plain dict of booleans, so
any step can be switched on or off independently (e.g. the baseline model
wants stop-word removal and lemmatization; the LSTM does not - see
BASELINE_STEPS / LSTM_STEPS below).
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Matches a leading wire-service dateline such as "WASHINGTON (Reuters) - ".
# Bounded to the first ~80 characters so an unrelated "(Reuters)" mention
# deep inside an article is never touched - only a genuine leading dateline.
_REUTERS_PREFIX_PATTERN = re.compile(r"^.{0,80}?\(Reuters\)\s*-?\s*")

# Full URLs, and the bare "pic.twitter.com/..." embeds that don't have a scheme.
_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.IGNORECASE)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_reuters_prefix(text: str) -> str:
    """Remove a leading Reuters wire-service dateline (e.g. "(Reuters) - ").

    This directly targets the label-leakage signal documented in
    docs/label_leakage_analysis.md (99.2% of Real articles carry this tag,
    0.04% of Fake articles do). If the pattern isn't present, the text is
    returned unchanged.
    """
    return _REUTERS_PREFIX_PATTERN.sub("", text, count=1)


def remove_word_token(text: str, word: str) -> str:
    """Remove every whole-word, case-insensitive occurrence of `word` from text.

    Unlike strip_reuters_prefix() (which only removes a *leading* dateline),
    this removes the token wherever it appears - used by the Reuters ablation
    experiment (docs/reuters_ablation_study.md) to test whether the residual
    "reuters" mentions found by feature importance (docs/baseline_model_report.md)
    are still acting as a shortcut. Only the token itself is removed, not any
    surrounding text; \\b word boundaries prevent partial matches inside other
    words (e.g. "reutersplc" would not lose "reuters").
    """
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return pattern.sub("", text)


def remove_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities (e.g. "&amp;" -> "&")."""
    text = html.unescape(text)
    return _HTML_TAG_PATTERN.sub(" ", text)


def remove_urls(text: str) -> str:
    """Delete URLs and bare link fragments outright (no placeholder token).

    A placeholder like "<URL>" would preserve "a link was here" as a token,
    which is exactly the label-leakage shortcut this step exists to remove
    (see docs/label_leakage_analysis.md - the pic.twitter.com signal).
    """
    return _URL_PATTERN.sub(" ", text)


def lowercase_text(text: str) -> str:
    """Lowercase all characters, so 'Trump' and 'trump' become one token."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation, keeping only word characters and whitespace."""
    return _PUNCTUATION_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines/tabs into single spaces and trim."""
    text = text.replace("\xa0", " ")  # non-breaking space, common after HTML unescaping
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Load NLTK's English stop-word list once and cache it."""
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    """Create (and cache) a single shared WordNetLemmatizer instance."""
    return WordNetLemmatizer()


def remove_stopwords(text: str) -> str:
    """Remove common English stop words (e.g. "the", "is", "and").

    Intended for the baseline (TF-IDF) model only - see BASELINE_STEPS /
    LSTM_STEPS below and docs/preprocessing_plan.md for why the LSTM
    pipeline deliberately skips this step (it needs negation words like
    "not"/"never" for context).
    """
    words = text.split()
    stop_set = get_stopwords()
    return " ".join(w for w in words if w not in stop_set)


def lemmatize_text(text: str) -> str:
    """Reduce words to a common base form (e.g. "running" -> "run").

    Intended for the baseline model only - see docs/preprocessing_plan.md.
    """
    lemmatizer = get_lemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())


def combine_title_and_text(title: str, text: str) -> str:
    """Concatenate title and article body into a single input field.

    Plain concatenation, no special weighting - see docs/data_cleaning_strategy.md
    ("Title and article body: will they be combined?") for the justification.
    """
    return f"{title} {text}".strip()


# Step order matters: HTML/URL removal must happen before lowercasing and
# punctuation removal (so tags/links are recognized in their original form),
# and stop-word removal / lemmatization must happen last, since they operate
# on already-lowercased, already-punctuation-free word tokens.
_STEP_ORDER = [
    "remove_html",
    "remove_urls",
    "lowercase",
    "remove_punctuation",
    "remove_stopwords",
    "lemmatize",
]

_STEP_FUNCTIONS = {
    "remove_html": remove_html,
    "remove_urls": remove_urls,
    "lowercase": lowercase_text,
    "remove_punctuation": remove_punctuation,
    "remove_stopwords": remove_stopwords,
    "lemmatize": lemmatize_text,
}

# Per docs/preprocessing_plan.md: stop-word removal and lemmatization are
# applied for the baseline (TF-IDF + Logistic Regression) model only, and
# skipped for the LSTM, which needs word order/negation intact.
BASELINE_STEPS: dict[str, bool] = {
    "remove_html": True,
    "remove_urls": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
    "lemmatize": True,
}

LSTM_STEPS: dict[str, bool] = {
    "remove_html": True,
    "remove_urls": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": False,
    "lemmatize": False,
}


def clean_text(text: str, steps: dict[str, bool]) -> str:
    """Apply the enabled cleaning steps to one string, in a fixed, sensible order.

    `steps` is a plain dict of booleans (see BASELINE_STEPS / LSTM_STEPS),
    so each technique can be switched on or off independently without
    touching this function - the whole point of keeping preprocessing
    modular rather than one long script.
    """
    for step_name in _STEP_ORDER:
        if steps.get(step_name, False):
            text = _STEP_FUNCTIONS[step_name](text)
    return normalize_whitespace(text)
