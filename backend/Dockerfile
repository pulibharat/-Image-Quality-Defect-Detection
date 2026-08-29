# Build context is the REPO ROOT (see docker-compose.yml), not backend/,
# because the backend imports the shared `core` package that also powers
# the offline training scripts.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# opencv-python-headless still dlopens a couple of shared libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY core /app/core
COPY backend /app/backend

RUN mkdir -p /app/backend/data /app/backend/uploads/images /app/backend/uploads/heatmaps

WORKDIR /app/backend

ENV MODEL_DIR=/app/backend/models_store \
    UPLOAD_DIR=/app/backend/uploads \
    DATABASE_URL=sqlite:////app/backend/data/app.db \
    CORS_ORIGINS=*

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
