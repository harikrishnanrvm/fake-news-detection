"""Project-wide logging setup - see the project specification's Logging section.

Deliberately just the standard library `logging` module: one console
handler, one rotating-free file handler writing to logs/app.log. No
structured/JSON logging, no external log services - overkill for a solo
BCA project.
"""
from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured to write to both the console and logs/app.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured - avoid attaching duplicate handlers

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
