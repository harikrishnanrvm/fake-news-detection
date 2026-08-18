"""Small shared helpers for the API layer."""
from __future__ import annotations

from typing import Any


def build_error_response(error: str, detail: str) -> dict[str, Any]:
    """Build the standard {"error": ..., "detail": ...} JSON body.

    Used by both exception handlers in api/app.py so every failure - a
    validation error or an unexpected server error - comes back in exactly
    the same shape, per the project specification's API Validation Rules.
    """
    return {"error": error, "detail": detail}


def clean_pydantic_message(raw_message: str) -> str:
    """Strip pydantic v2's "Value error, " prefix from a custom validator message.

    A field_validator that raises ValueError("Article text cannot be empty.")
    comes back from pydantic as "Value error, Article text cannot be empty." -
    trimming the prefix keeps the client-facing message matching the project specification's
    exact wording instead of leaking a pydantic implementation detail.
    """
    prefix = "Value error, "
    if raw_message.startswith(prefix):
        return raw_message[len(prefix):]
    return raw_message
