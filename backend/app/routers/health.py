from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthOut
from app.services import ml_engine
from app.services.pipeline import MODEL_VERSION

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health():
    try:
        engine = ml_engine.get_engine()
        loaded = engine.is_ready
    except RuntimeError:
        loaded = False
    return HealthOut(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        app_version=settings.app_version,
        model_version=MODEL_VERSION,
    )
