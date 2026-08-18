"""Runs the LSTM's clean -> tokenize -> pad -> predict pipeline for one article.

This is the inference-time mirror of training/lstm.py's data preparation -
it must apply the exact same cleaning steps (LSTM_STEPS) and the exact same
tokenizer/padding settings the model was trained with, otherwise the model
would be scoring input that looks nothing like what it learned from.
"""
from __future__ import annotations

from typing import Any

from preprocessing.text_cleaning import LSTM_STEPS, clean_text
from training.lstm import texts_to_padded_sequences

from api.model_loader import get_label_map, get_model, get_tokenizer


def predict_article(raw_text: str) -> dict[str, Any]:
    """Classify one news article as Fake or Real.

    Parameters
    ----------
    raw_text : str
        The article text as pasted by the user (already validated - see
        api/schemas.py's PredictRequest - so this is never empty here).

    Returns
    -------
    dict[str, Any]
        Fields matching api.schemas.PredictResponse.
    """
    cleaned_text = clean_text(raw_text, steps=LSTM_STEPS)

    tokenizer = get_tokenizer()
    padded_sequence = texts_to_padded_sequences(tokenizer, [cleaned_text])

    model = get_model()
    probability_real = float(model.predict(padded_sequence, verbose=0)[0][0])
    probability_fake = 1.0 - probability_real

    # label_map is {"fake": 0, "real": 1} - the model's single sigmoid output
    # is already "probability of the label mapped to 1", i.e. Real.
    get_label_map()  # loaded eagerly at startup; confirms artifacts are ready
    prediction = "Real" if probability_real >= 0.5 else "Fake"
    confidence = probability_real if prediction == "Real" else probability_fake

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probability_real": round(probability_real, 4),
        "probability_fake": round(probability_fake, 4),
        "model": "LSTM",
    }
