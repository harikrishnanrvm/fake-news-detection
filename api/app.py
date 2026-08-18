"""FastAPI application: wiring for startup, routes, and error handling.

Run with (from the project root, venv activated):
    uvicorn api.app:app --reload

See docs/deployment_guide.md for the full setup walkthrough.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.model_loader import load_artifacts
from api.routes import router
from api.utils import build_error_response, clean_pydantic_message
from utils.logger import get_logger

logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LSTM model and tokenizer once, before the app accepts requests.

    Per Phase 7's requirement: the model/tokenizer must be loaded exactly
    once at startup, never reloaded per request - see api/model_loader.py.
    """
    logger.info("Loading LSTM model and tokenizer...")
    load_artifacts()
    logger.info("Model and tokenizer loaded - ready to serve predictions.")
    yield


app = FastAPI(
    title="Fake News Detection API",
    description="Serves predictions from the project's trained LSTM model (Phases 1-6).",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS: this is a local, single-user BCA demo (no auth, no
# sensitive data), so allowing any origin is a reasonable simplification -
# it also lets the frontend be opened either as a plain file or via /static.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert FastAPI's default validation error shape into the project's
    standard {"error", "detail"} format (the project specification's API Validation Rules),
    covering empty input, too-short/too-long text, null input, missing
    fields, and malformed JSON bodies - all of these surface as a
    RequestValidationError.
    """
    first_error = exc.errors()[0]
    detail = clean_pydantic_message(str(first_error.get("msg", "Invalid request.")))
    logger.warning(f"Validation error on {request.url.path}: {detail}")
    return JSONResponse(
        status_code=422,
        content=build_error_response("validation_error", detail),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch anything unexpected. The real error is logged server-side only -
    the client always gets a generic message, never a stack trace.
    """
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=build_error_response(
            "server_error", "Something went wrong while processing your request."
        ),
    )


app.include_router(router)

# The Bootstrap frontend (frontend/index.html) is served as a static file so
# it can call /predict on the same origin, with no extra web server needed.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/app", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Convenience redirect-style route: open http://127.0.0.1:8000/app
    instead of remembering the full /static/index.html path.
    """
    return FileResponse(FRONTEND_DIR / "index.html")
