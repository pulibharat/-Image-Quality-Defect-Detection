"""Process-wide singleton wrapper around core.QualityEngine.

Loading the classifiers (joblib) and the autoencoder (torch state_dict) is
the only "model loading" step the service needs -- both are small files
read from backend/models_store/ and loaded once at FastAPI startup (see
app/main.py's lifespan handler), then reused for every request.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from core.quality_engine import QualityEngine  # noqa: E402

from app.config import settings

_engine: QualityEngine | None = None


def load_engine() -> QualityEngine:
    global _engine
    if _engine is None:
        _engine = QualityEngine(settings.model_dir)
    return _engine


def get_engine() -> QualityEngine:
    if _engine is None:
        raise RuntimeError("Model engine not loaded yet.")
    return _engine
