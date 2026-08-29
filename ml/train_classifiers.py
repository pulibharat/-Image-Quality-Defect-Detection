"""Trains one 4-class (none/low/medium/high) severity classifier per issue
type on the engineered feature vectors from core.features, using the
synthetic dataset produced by generate_dataset.py.

This is the "classical machine learning using engineered image features"
half of the hybrid approach: gradient-boosted trees on top of ~12
interpretable image statistics (sharpness, exposure, noise, texture, ...).
Trees over a small, hand-designed feature set generalize well from a
few-thousand-sample synthetic dataset -- a pixel CNN would not.

Artifacts are written to backend/models_store/ so the FastAPI service can
load them directly with no training-time dependencies (scikit-image,
matplotlib, etc. never need to be installed in the serving image).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.degrade import ISSUE_TYPES
from core.features import FEATURE_NAMES, extract_features, feature_vector

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models_store"
BUCKETS = ["none", "low", "medium", "high"]


def load_split(labels: pd.DataFrame, split: str):
    rows = labels[labels["split"] == split]
    X, buckets_by_issue, scene_ids = [], {i: [] for i in ISSUE_TYPES}, []
    for _, row in rows.iterrows():
        img_path = DATA_DIR / "images" / split / f"{row['sample_id']}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        feats = extract_features(img)
        X.append(feature_vector(feats))
        for issue in ISSUE_TYPES:
            buckets_by_issue[issue].append(row[f"{issue}_bucket"])
        scene_ids.append(row["scene_id"])
    return np.vstack(X), buckets_by_issue, scene_ids


def main():
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    print("Extracting features for train split...")
    X_train, y_train, _ = load_split(labels, "train")
    print("Extracting features for val split...")
    X_val, y_val, _ = load_split(labels, "val")
    print(f"train: {X_train.shape}, val: {X_val.shape}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}

    for issue in ISSUE_TYPES:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("gbc", GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.08,
                subsample=0.9, random_state=42,
            )),
        ])
        clf.fit(X_train, y_train[issue])
        train_acc = clf.score(X_train, y_train[issue])
        val_acc = clf.score(X_val, y_val[issue])
        metrics[issue] = {"train_accuracy": train_acc, "val_accuracy": val_acc}
        print(f"  {issue:15s} train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        joblib.dump(clf, MODEL_DIR / f"{issue}_classifier.joblib")

    meta = {
        "feature_names": FEATURE_NAMES,
        "issue_types": ISSUE_TYPES,
        "buckets": BUCKETS,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "val_accuracy": {k: v["val_accuracy"] for k, v in metrics.items()},
    }
    with open(MODEL_DIR / "classifiers_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {len(ISSUE_TYPES)} classifiers + meta to {MODEL_DIR}")


if __name__ == "__main__":
    main()
