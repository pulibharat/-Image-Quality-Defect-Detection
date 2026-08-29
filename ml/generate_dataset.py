"""Builds a labeled image-quality dataset entirely offline, with no external
AI services and no network dependency at *use* time (the only network use
is a one-off `pip install scikit-image` for a few bundled sample photos).

Base "clean" scenes come from two sources:
  1. A handful of real photos bundled with scikit-image (data.astronaut,
     data.chelsea, ...), for realistic texture/content.
  2. Procedurally generated scenes (gradients, shapes, textures, checkers)
     from core.degrade, for volume and diversity.

Each base scene is a `scene_id`. Scenes are split train/val/test BEFORE any
degradation is applied, so every generated sample inherits its scene's
split -- this guarantees the val/test sets contain visual content the
training process never saw, per the assessment's "evaluate on unseen
images" requirement.

For each scene we generate:
  - 1 clean sample (all issues = none)
  - single-issue samples at 3 severities (0.3 / 0.6 / 0.9) for each of the
    6 issue types, so a classifier for issue X sees both "X at various
    strengths" (positives) and "other issues, X absent" (hard negatives)
  - a few two-issue "combo" samples for realism and robustness evaluation

Output: ml/data/images/<split>/<sample_id>.png + ml/data/labels.csv
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.degrade import DEGRADATIONS, ISSUE_TYPES, generate_procedural_base_images

SCENE_SIZE = 256
SEED = 42
SEVERITIES = [0.3, 0.6, 0.9]
N_COMBOS_PER_SCENE = 3
OUT_DIR = Path(__file__).resolve().parent / "data"


def severity_bucket(s: float) -> str:
    if s <= 0:
        return "none"
    if s <= 0.4:
        return "low"
    if s <= 0.7:
        return "medium"
    return "high"


def load_skimage_scenes() -> list[tuple[str, np.ndarray]]:
    scenes = []
    try:
        from skimage import data as skdata
    except ImportError:
        print("scikit-image not installed; skipping real-photo base scenes.")
        return scenes

    candidates = [
        "astronaut", "chelsea", "coffee", "rocket", "hubble_deep_field",
        "camera", "coins", "moon", "page", "text", "checkerboard",
        "brick", "grass", "gravel", "colorwheel", "retina",
    ]
    for name in candidates:
        try:
            fn = getattr(skdata, name, None)
            if fn is None:
                continue
            img = fn()
            img = np.asarray(img)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.resize(img, (SCENE_SIZE, SCENE_SIZE), interpolation=cv2.INTER_AREA)
            scenes.append((f"skimage_{name}", img))
        except Exception as exc:  # pragma: no cover - defensive, dataset fetch is best-effort
            print(f"  skipped skimage.data.{name}: {exc}")
    print(f"Loaded {len(scenes)} real-photo base scenes from scikit-image.")
    return scenes


def load_procedural_scenes(n: int) -> list[tuple[str, np.ndarray]]:
    imgs = generate_procedural_base_images(n, SCENE_SIZE, seed=SEED)
    return [(f"procedural_{i:03d}", img) for i, img in enumerate(imgs)]


def build_scene_list() -> list[tuple[str, np.ndarray]]:
    scenes = load_skimage_scenes()
    scenes += load_procedural_scenes(max(0, 45 - len(scenes)))
    return scenes


def split_scenes(scenes: list, seed: int = SEED):
    rng = random.Random(seed)
    order = scenes[:]
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def save_sample(img: np.ndarray, split: str, sample_id: str):
    out_path = OUT_DIR / "images" / split / f"{sample_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path.relative_to(OUT_DIR)


def main():
    rng = random.Random(SEED)
    scenes = build_scene_list()
    splits = split_scenes(scenes)
    for name, lst in splits.items():
        print(f"{name}: {len(lst)} base scenes")

    rows = []
    sample_counter = 0

    for split, scene_list in splits.items():
        for scene_id, base_img in scene_list:
            # 1) clean sample
            sample_id = f"{sample_counter:05d}_{scene_id}_clean"
            save_sample(base_img, split, sample_id)
            rows.append({"sample_id": sample_id, "split": split, "scene_id": scene_id,
                         **{k: 0.0 for k in ISSUE_TYPES}})
            sample_counter += 1

            # 2) single-issue samples at graded severities
            for issue in ISSUE_TYPES:
                for sev in SEVERITIES:
                    img = DEGRADATIONS[issue](base_img, sev, seed=rng.randint(0, 1_000_000))
                    sample_id = f"{sample_counter:05d}_{scene_id}_{issue}_{int(sev*100)}"
                    save_sample(img, split, sample_id)
                    row = {k: 0.0 for k in ISSUE_TYPES}
                    row[issue] = sev
                    rows.append({"sample_id": sample_id, "split": split, "scene_id": scene_id, **row})
                    sample_counter += 1

            # 3) combo samples (two random issues stacked)
            for c in range(N_COMBOS_PER_SCENE):
                chosen = rng.sample(ISSUE_TYPES, 2)
                img = base_img
                row = {k: 0.0 for k in ISSUE_TYPES}
                for issue in chosen:
                    sev = rng.choice(SEVERITIES)
                    img = DEGRADATIONS[issue](img, sev, seed=rng.randint(0, 1_000_000))
                    row[issue] = sev
                sample_id = f"{sample_counter:05d}_{scene_id}_combo{c}"
                save_sample(img, split, sample_id)
                rows.append({"sample_id": sample_id, "split": split, "scene_id": scene_id, **row})
                sample_counter += 1

    labels_path = OUT_DIR / "labels.csv"
    fieldnames = ["sample_id", "split", "scene_id"] + ISSUE_TYPES + [f"{i}_bucket" for i in ISSUE_TYPES]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for issue in ISSUE_TYPES:
                row[f"{issue}_bucket"] = severity_bucket(row[issue])
            writer.writerow(row)

    print(f"\nGenerated {len(rows)} samples -> {labels_path}")
    for split in ["train", "val", "test"]:
        n = sum(1 for r in rows if r["split"] == split)
        print(f"  {split}: {n} samples")


if __name__ == "__main__":
    main()
