"""End-to-end single-image analysis pipeline: validate -> decode -> run the
hybrid quality engine -> persist image/heatmap to disk -> persist the
structured result to the database. Shared by the single-image and batch
analyze endpoints so there is one code path for "what happens to an
uploaded image".
"""
from __future__ import annotations

import json
import time

from sqlalchemy.orm import Session

from app.config import settings
from app.models import AnalysisResult
from app.services import storage
from app.services.image_validation import downscale_for_processing, validate_and_decode
from app.services.ml_engine import get_engine

MODEL_VERSION = "hybrid-gbc+autoencoder-v1"


def run_analysis(raw_bytes: bytes, filename: str, content_type: str | None, db: Session) -> AnalysisResult:
    engine = get_engine()

    image, width, height = validate_and_decode(
        raw_bytes, content_type, settings.allowed_content_types_list, settings.max_upload_mb * 1024 * 1024,
    )
    processing_image = downscale_for_processing(image)

    t0 = time.time()
    result = engine.analyze(processing_image)
    processing_time_ms = (time.time() - t0) * 1000

    file_id, image_rel_path = storage.save_original(raw_bytes, filename)
    heatmap_rel_path = storage.save_heatmap_overlay(file_id, processing_image, result["heatmap"])

    row = AnalysisResult(
        original_filename=filename,
        stored_image_path=image_rel_path,
        heatmap_path=heatmap_rel_path,
        width=width,
        height=height,
        file_size_bytes=len(raw_bytes),
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues_json=json.dumps(result["issues"]),
        features_json=json.dumps(result["features"]),
        anomaly_score=result["anomaly_score"],
        processing_time_ms=processing_time_ms,
        model_version=MODEL_VERSION,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def to_analysis_out_dict(row: AnalysisResult) -> dict:
    engine = get_engine()
    return {
        "id": row.id,
        "created_at": row.created_at,
        "original_filename": row.original_filename,
        "width": row.width,
        "height": row.height,
        "file_size_bytes": row.file_size_bytes,
        "quality_score": row.quality_score,
        "quality_label": row.quality_label,
        "issues": json.loads(row.issues_json),
        "features": json.loads(row.features_json),
        "anomaly_score": row.anomaly_score,
        "anomaly_threshold": engine.anomaly_threshold if engine else None,
        "processing_time_ms": row.processing_time_ms,
        "model_version": row.model_version,
        "image_url": f"/api/analyses/{row.id}/image",
        "heatmap_url": f"/api/analyses/{row.id}/heatmap" if row.heatmap_path else None,
    }
