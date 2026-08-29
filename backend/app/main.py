import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import analyze, health, history
from app.services import ml_engine

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_quality_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized.")
    engine = ml_engine.load_engine()
    if engine.is_ready:
        logger.info("Model artifacts loaded from %s", settings.model_dir)
    else:
        logger.warning(
            "Model artifacts missing/incomplete in %s -- /health will report 'degraded' and "
            "/api/analyze will fail until `ml/train_classifiers.py` and `ml/train_autoencoder.py` "
            "have been run.", settings.model_dir,
        )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Detects blur, exposure, noise, corruption, and defects in uploaded images "
                "using a hybrid engineered-feature + learned-model pipeline (no external AI APIs).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred while processing the request."},
    )


app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(history.router)


@app.get("/api")
def api_info():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


# Convenience for single-process local dev / a standalone backend container
# that happens to have the frontend/ directory alongside it: serve the
# static UI at "/" so `uvicorn app.main:app` alone is a fully working app
# with no reverse proxy needed. In the two-container docker-compose
# deployment this directory isn't copied into the backend image, so this
# is skipped there and the separate nginx service serves the UI instead.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }
