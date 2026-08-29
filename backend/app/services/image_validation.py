"""Upload validation and decoding.

Deliberately tolerant of truncated/corrupted-but-partially-decodable image
files (LOAD_TRUNCATED_IMAGES=True): a badly corrupted JPEG is exactly the
kind of input the "corruption / severe degradation" detector is meant to
flag, so we want the quality engine to see it rather than have the API
reject it outright. We only raise (-> HTTP 400) when the bytes genuinely
cannot be decoded as any image at all, or fail basic sanity limits.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_DIMENSION = 4000       # reject absurdly large images (decompression-bomb guard)
MIN_DIMENSION = 8          # reject unusably tiny images
PROCESSING_MAX_SIDE = 1600  # downscale copy used for feature extraction / inference


class InvalidImageError(ValueError):
    pass


def validate_and_decode(raw_bytes: bytes, declared_content_type: str | None,
                         allowed_content_types: list[str], max_bytes: int) -> tuple[np.ndarray, int, int]:
    if not raw_bytes:
        raise InvalidImageError("Uploaded file is empty.")
    if len(raw_bytes) > max_bytes:
        raise InvalidImageError(f"File exceeds the {max_bytes // (1024*1024)}MB upload limit.")
    if declared_content_type and allowed_content_types and declared_content_type not in allowed_content_types:
        raise InvalidImageError(
            f"Unsupported content type '{declared_content_type}'. Allowed: {', '.join(allowed_content_types)}"
        )

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"File could not be decoded as an image: {exc}") from exc

    width, height = img.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise InvalidImageError(f"Image is too small ({width}x{height}); minimum is {MIN_DIMENSION}px per side.")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise InvalidImageError(f"Image is too large ({width}x{height}); maximum is {MAX_DIMENSION}px per side.")

    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    array = np.array(img)

    return array, width, height


def downscale_for_processing(image: np.ndarray, max_side: int = PROCESSING_MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    import cv2
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
