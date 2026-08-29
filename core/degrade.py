"""Synthetic image-quality degradations used to build a labeled training /
evaluation set from clean base images, and procedural generators for extra
clean base scenes so we are not limited to the handful of photos bundled
with scikit-image.

Every degradation takes a `severity` in [0, 1] (0 = no change). This lets
`ml/generate_dataset.py` produce graded (none/low/medium/high) labels for
each issue type instead of only binary present/absent labels.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_blur(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return image.copy()
    ksize = int(1 + round(severity * 14))
    if ksize % 2 == 0:
        ksize += 1
    sigma = 0.5 + severity * 6.0
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_underexposure(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return image.copy()
    rng = _rng(seed)
    gain = 1.0 - 0.85 * severity
    gamma = 1.0 + severity * 2.5
    out = (image.astype(np.float64) / 255.0) * gain
    out = np.power(np.clip(out, 0, 1), gamma) * 255.0
    dark_noise = rng.normal(0, 2 + severity * 4, image.shape)
    out = out + dark_noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return image.copy()
    add = 60 * severity
    gain = 1.0 + 1.2 * severity
    out = image.astype(np.float64) * gain + add
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return image.copy()
    rng = _rng(seed)
    sigma = severity * 45.0
    gauss = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + gauss
    if severity > 0.5:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(max(vals, 2)))
        poisson_part = rng.poisson(np.clip(image, 0, 255) * vals) / float(vals)
        noisy = 0.5 * noisy + 0.5 * poisson_part
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_corruption(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    """Simulates severe compression / transmission corruption: heavy JPEG
    re-quantization plus randomly wiped/shuffled blocks."""
    if severity <= 0:
        return image.copy()
    rng = _rng(seed)
    quality = int(max(1, 30 - severity * 28))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    out_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    h, w = out.shape[:2]
    n_blocks = int(severity * 10)
    for _ in range(n_blocks):
        bs = rng.integers(max(4, int(min(h, w) * 0.03)), max(6, int(min(h, w) * 0.12)))
        y = rng.integers(0, max(1, h - bs))
        x = rng.integers(0, max(1, w - bs))
        choice = rng.random()
        if choice < 0.4:
            out[y:y + bs, x:x + bs] = rng.integers(0, 255, (bs, bs, 3), dtype=np.uint8)
        elif choice < 0.7:
            out[y:y + bs, x:x + bs] = 0
        else:
            sy, sx = rng.integers(0, max(1, h - bs)), rng.integers(0, max(1, w - bs))
            out[y:y + bs, x:x + bs] = out[sy:sy + bs, sx:sx + bs]
    return out


def apply_defect(image: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    """Simulates localized sensor/lens defects: dead-pixel clusters,
    scratches and dust spots -- distinct from global corruption because it
    only affects small regions of an otherwise normal image."""
    if severity <= 0:
        return image.copy()
    rng = _rng(seed)
    out = image.copy()
    h, w = out.shape[:2]

    n_scratches = int(1 + severity * 4)
    for _ in range(n_scratches):
        pt1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        length = int(rng.integers(int(min(h, w) * 0.1), int(min(h, w) * 0.5) + 1))
        angle = rng.uniform(0, 2 * np.pi)
        pt2 = (int(pt1[0] + length * np.cos(angle)), int(pt1[1] + length * np.sin(angle)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(out, pt1, pt2, color, thickness=int(1 + severity * 2))

    n_spots = int(2 + severity * 10)
    overlay = out.copy()
    for _ in range(n_spots):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        radius = int(rng.integers(2, max(3, int(min(h, w) * 0.04))))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(overlay, center, radius, color, -1)
    out = cv2.addWeighted(overlay, 0.5 + 0.4 * severity, out, 0.5 - 0.4 * severity, 0)

    n_dead = int(severity * 200)
    ys = rng.integers(0, h, n_dead)
    xs = rng.integers(0, w, n_dead)
    out[ys, xs] = rng.choice([0, 255], size=(n_dead, 1))
    return out


DEGRADATIONS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
    "defect": apply_defect,
}


# ---------------------------------------------------------------------------
# Procedural "clean" base scene generators.
# These exist to give the synthetic dataset more visual variety than the
# ~15 photos bundled with scikit-image alone provide -- more distinct base
# scenes means a more meaningful held-out split for evaluation.
# ---------------------------------------------------------------------------

def _smooth_noise_field(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    field = np.zeros((h, w), dtype=np.float64)
    for o in range(octaves):
        scale = 2 ** o
        small = rng.random((max(2, h // (4 * scale) + 2), max(2, w // (4 * scale) + 2)))
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        field += big / (o + 1)
    field -= field.min()
    field /= (field.max() + 1e-9)
    return field


def gen_gradient_scene(size: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    colors = rng.integers(30, 225, size=(2, 3))
    t = (xx * rng.uniform(0.3, 1) + yy * rng.uniform(0, 0.7))
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    img = colors[0][None, None, :] * (1 - t[..., None]) + colors[1][None, None, :] * t[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_shapes_scene(size: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    bg = tuple(int(c) for c in rng.integers(40, 220, 3))
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    n_shapes = rng.integers(6, 18)
    for _ in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            r = int(rng.integers(size // 20, size // 5))
            cv2.circle(img, c, r, color, -1)
        elif kind == 1:
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.rectangle(img, p1, p2, color, -1)
        else:
            pts = rng.integers(0, size, size=(3, 2))
            cv2.drawContours(img, [pts], 0, color, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def gen_texture_scene(size: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    base_color = rng.integers(50, 200, 3).astype(np.float64)
    field = _smooth_noise_field(size, size, rng, octaves=5)
    img = np.zeros((size, size, 3), dtype=np.float64)
    for c in range(3):
        variation = (rng.uniform(0.5, 1.5)) * 90
        img[..., c] = base_color[c] + (field - 0.5) * variation
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_checker_scene(size: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    cell = int(rng.integers(8, size // 4))
    colors = rng.integers(20, 235, size=(2, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    mask = ((xx // cell) + (yy // cell)) % 2
    img = np.where(mask[..., None] == 0, colors[0], colors[1]).astype(np.uint8)
    return img


PROCEDURAL_GENERATORS = [gen_gradient_scene, gen_shapes_scene, gen_texture_scene, gen_checker_scene]


def generate_procedural_base_images(n: int, size: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    images = []
    for i in range(n):
        gen = PROCEDURAL_GENERATORS[i % len(PROCEDURAL_GENERATORS)]
        images.append(gen(size, seed=rng.randint(0, 10_000_000)))
    return images
