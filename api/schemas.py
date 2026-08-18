"""Pydantic request/response models for the Fake News Detection API.

Keeping these in one module (separate from routes.py) makes the API's
"contract" easy to find in one place, and is what Swagger UI (/docs) reads
to generate its example schemas.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from config.settings import MAX_ARTICLE_CHARS, MIN_ARTICLE_CHARS


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(..., description="The news article text to classify.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Enforce the project specification's API validation rules before any prediction runs.

        Stripping first means whitespace-only input (e.g. "   ") is treated
        the same as empty input, rather than slipping past a naive length
        check.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("Article text cannot be empty.")
        if len(stripped) < MIN_ARTICLE_CHARS:
            raise ValueError(
                f"Article text must be at least {MIN_ARTICLE_CHARS} characters long."
            )
        if len(stripped) > MAX_ARTICLE_CHARS:
            raise ValueError(
                f"Article text must not exceed {MAX_ARTICLE_CHARS} characters."
            )
        return stripped


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    prediction: str = Field(..., description='"Real" or "Fake".')
    confidence: float = Field(..., description="Confidence in the predicted label, as a percentage.")
    probability_real: float = Field(..., description="Model's raw probability that the article is Real.")
    probability_fake: float = Field(..., description="Model's raw probability that the article is Fake.")
    model: str = Field(..., description="Which trained model produced this prediction.")


class RootResponse(BaseModel):
    """Response body for GET /."""

    message: str
    endpoints: list[str]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str


class ErrorResponse(BaseModel):
    """Standard error shape for every failure response, per the project specification's API section."""

    error: str
    detail: str
