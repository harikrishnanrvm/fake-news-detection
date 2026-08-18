"""API route handlers - kept separate from app.py so the app factory itself
(startup wiring, exception handlers) stays small and easy to scan."""
from __future__ import annotations

from fastapi import APIRouter

from api.predictor import predict_article
from api.schemas import HealthResponse, PredictRequest, PredictResponse, RootResponse
from api.model_loader import is_ready

router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["General"])
def read_root() -> RootResponse:
    """Welcome message and a list of available endpoints."""
    return RootResponse(
        message="Fake News Detection API - see /docs for interactive API documentation.",
        endpoints=["/", "/health", "/predict", "/docs"],
    )


@router.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    """Liveness/readiness check - also confirms the LSTM model finished loading."""
    return HealthResponse(status="ok" if is_ready() else "loading")


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one news article as Fake or Real using the trained LSTM.

    Input is already validated by PredictRequest (non-empty, within the
    configured length range) before this function runs.
    """
    result = predict_article(request.text)
    return PredictResponse(**result)
