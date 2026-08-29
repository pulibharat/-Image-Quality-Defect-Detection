import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import degrade, features


def _checker(size=128, cell=8):
    yy, xx = np.mgrid[0:size, 0:size]
    mask = ((xx // cell) + (yy // cell)) % 2
    img = np.where(mask[..., None] == 0, 20, 235).astype(np.uint8)
    return np.repeat(img, 3, axis=2)


def test_blur_reduces_sharpness():
    sharp = _checker()
    blurry = degrade.apply_blur(sharp, severity=0.9)
    f_sharp = features.extract_features(sharp)
    f_blurry = features.extract_features(blurry)
    assert f_blurry["sharpness_lap_var"] < f_sharp["sharpness_lap_var"]
    assert f_blurry["edge_density"] < f_sharp["edge_density"]


def test_underexposure_lowers_brightness():
    base = _checker()
    dark = degrade.apply_underexposure(base, severity=0.8)
    f_base = features.extract_features(base)
    f_dark = features.extract_features(dark)
    assert f_dark["brightness_mean"] < f_base["brightness_mean"]
    assert f_dark["underexposed_ratio"] >= f_base["underexposed_ratio"]


def test_overexposure_raises_brightness_and_clipping():
    base = _checker()
    bright = degrade.apply_overexposure(base, severity=0.8)
    f_base = features.extract_features(base)
    f_bright = features.extract_features(bright)
    assert f_bright["brightness_mean"] > f_base["brightness_mean"]
    assert f_bright["overexposed_ratio"] >= f_base["overexposed_ratio"]


def test_noise_increases_noise_sigma():
    base = _checker()
    noisy = degrade.apply_noise(base, severity=0.8)
    f_base = features.extract_features(base)
    f_noisy = features.extract_features(noisy)
    assert f_noisy["noise_sigma"] > f_base["noise_sigma"]


def test_feature_vector_matches_feature_names_order():
    img = _checker()
    feats = features.extract_features(img)
    vec = features.feature_vector(feats)
    assert vec.shape == (len(features.FEATURE_NAMES),)
    assert list(feats[k] for k in features.FEATURE_NAMES) == list(vec)
