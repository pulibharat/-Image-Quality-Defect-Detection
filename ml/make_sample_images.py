"""Generates a small, human-browsable set of demonstration images (one per
required quality condition) for the sample_images/ submission folder. Uses
real photos from scikit-image plus the same synthetic degradations used to
build the training set, so these are representative, honest examples of
what the classifiers actually saw.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.degrade import apply_blur, apply_corruption, apply_defect, apply_noise, apply_overexposure, apply_underexposure
from skimage import data as skdata

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_images"
OUT_DIR.mkdir(exist_ok=True)


def load(name, size=384):
    img = np.asarray(getattr(skdata, name)())
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def save(name, img):
    cv2.imwrite(str(OUT_DIR / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {name}")


base = load("astronaut")
save("01_clean_acceptable.jpg", base)
save("02_blurry.jpg", apply_blur(base, 0.7, seed=1))
save("03_underexposed.jpg", apply_underexposure(base, 0.7, seed=1))
save("04_overexposed.jpg", apply_overexposure(base, 0.7, seed=1))
save("05_noisy.jpg", apply_noise(base, 0.7, seed=1))
save("06_corrupted.jpg", apply_corruption(load("coffee"), 0.8, seed=2))
save("07_defective_scratches.jpg", apply_defect(load("chelsea"), 0.7, seed=3))

combo = apply_noise(apply_blur(load("rocket"), 0.4, seed=4), 0.4, seed=4)
save("08_multiple_issues.jpg", combo)
