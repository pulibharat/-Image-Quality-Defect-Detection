from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

UPLOAD_DIR = Path(settings.upload_dir)


def _ensure_dirs():
    (UPLOAD_DIR / "images").mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)


def save_original(raw_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """Saves the raw uploaded bytes unmodified. Returns (file_id, relative_path)."""
    _ensure_dirs()
    suffix = Path(original_filename).suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        suffix = ".jpg"
    file_id = uuid.uuid4().hex
    rel_path = f"images/{file_id}{suffix}"
    (UPLOAD_DIR / rel_path).write_bytes(raw_bytes)
    return file_id, rel_path


def save_heatmap_overlay(file_id: str, image_rgb: np.ndarray, heatmap: np.ndarray | None) -> str | None:
    """Renders the anomaly reconstruction-error heatmap as a red-hot overlay
    on top of the (resized-for-processing) image, for explainability /
    localization of problematic regions."""
    if heatmap is None:
        return None
    _ensure_dirs()
    heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, colored, 0.45, 0)

    rel_path = f"heatmaps/{file_id}.png"
    out_path = UPLOAD_DIR / rel_path
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return rel_path


def resolve_path(rel_path: str) -> Path:
    return UPLOAD_DIR / rel_path
