"""Engineered image-quality feature extraction.

All functions operate on an RGB uint8 numpy array (H, W, 3). Every feature
here is a classical, hand-derived image-quality statistic (no learned
weights) -- this is the "engineered features" half of the hybrid AI/ML
approach described in the README. The learned half (per-issue classifiers
and the anomaly autoencoder, both in this package / backend) consume the
vector produced by `feature_vector()`.
"""
from __future__ import annotations

import cv2
import numpy as np

# Fixed order of features fed into the learned classifiers. Keeping this as
# an explicit list (rather than dict ordering) means the training scripts
# and the serving code can never silently drift out of sync.
FEATURE_NAMES = [
    "sharpness_lap_var",
    "sharpness_tenengrad",
    "brightness_mean",
    "contrast_std",
    "underexposed_ratio",
    "overexposed_ratio",
    "noise_sigma",
    "colorfulness",
    "saturation_mean",
    "entropy",
    "edge_density",
    "blockiness",
]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sharpness_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Low variance == few sharp edges == blur."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def sharpness_tenengrad(gray: np.ndarray) -> float:
    """Mean squared gradient magnitude (Sobel). A second, independent
    sharpness signal that is less sensitive to noise than the Laplacian."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def exposure_stats(gray: np.ndarray) -> dict:
    mean = float(gray.mean())
    std = float(gray.std())
    under = float(np.mean(gray < 16))
    over = float(np.mean(gray > 240))
    return {
        "brightness_mean": mean,
        "contrast_std": std,
        "underexposed_ratio": under,
        "overexposed_ratio": over,
    }


def noise_sigma_estimate(gray: np.ndarray) -> float:
    """Fast per-image noise standard deviation estimate.

    Immerkaer (1996), "Fast Noise Variance Estimation": convolves with a
    Laplacian-like kernel that has zero response on flat/linear regions, so
    its output energy over a natural image is dominated by sensor noise
    rather than real structure.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sqrt(np.pi / 2) * np.sum(np.abs(conv)) / (6 * (w - 2) * (h - 2))
    return float(sigma)


def colorfulness_metric(image: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric. Works on RGB; if the
    input is grayscale-like (channels equal) this naturally comes out ~0."""
    if image.ndim == 2:
        return 0.0
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def saturation_mean(image: np.ndarray) -> float:
    if image.ndim == 2:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels marked as edges by auto-thresholded Canny.
    Distinguishes genuinely low-detail (flat) scenes from blurred ones when
    combined with the sharpness features."""
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    return float(np.mean(edges > 0))


def blockiness_score(gray: np.ndarray) -> float:
    """Rough measure of 8x8 JPEG blocking artifacts: mean absolute pixel
    difference across block boundaries minus the mean difference within
    blocks. Large positive values indicate visible block edges, a hallmark
    of heavy compression / corruption."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    g = gray.astype(np.float64)
    col_diffs = np.abs(np.diff(g, axis=1))
    row_diffs = np.abs(np.diff(g, axis=0))

    boundary_cols = np.arange(7, w - 1, 8)
    boundary_rows = np.arange(7, h - 1, 8)
    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return 0.0

    boundary_energy = col_diffs[:, boundary_cols].mean() + row_diffs[boundary_rows, :].mean()
    overall_energy = col_diffs.mean() + row_diffs.mean() + 1e-6
    return float(max(0.0, (boundary_energy - overall_energy) / overall_energy))


def extract_features(image: np.ndarray) -> dict:
    """Compute the full engineered feature dictionary for an RGB uint8
    image. Returns human-readable keys; use `feature_vector()` to get the
    fixed-order numeric array consumed by the learned models."""
    gray = _to_gray(image)
    feats = {
        "sharpness_lap_var": sharpness_laplacian_variance(gray),
        "sharpness_tenengrad": sharpness_tenengrad(gray),
        "noise_sigma": noise_sigma_estimate(gray),
        "colorfulness": colorfulness_metric(image),
        "saturation_mean": saturation_mean(image),
        "entropy": shannon_entropy(gray),
        "edge_density": edge_density(gray),
        "blockiness": blockiness_score(gray),
    }
    feats.update(exposure_stats(gray))
    return feats


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
