"""Trains the anomaly-detection autoencoder ONLY on images that look
"normal": clean scenes plus mild (severity <= 0.3, i.e. bucket "low")
everyday variation in blur/exposure/noise. Corruption and defect samples,
and anything above "low" severity, are deliberately excluded from training
so the network never learns to reconstruct them -- it should reconstruct
those poorly, which is the anomaly signal used at inference.

This is the classic reconstruction-based anomaly-detection formulation:
train a compact autoencoder on the "normal" class only, then use
reconstruction error as the anomaly score for everything else. See
evaluate.py for the ROC-AUC / threshold analysis that validates this
actually separates corrupted+defective images from normal ones.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.autoencoder import IMG_SIZE, ConvAutoencoder

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models_store"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normal_subset(labels: pd.DataFrame, split: str) -> pd.DataFrame:
    mask = (
        (labels["split"] == split)
        & (labels["corruption"] == 0)
        & (labels["defect"] == 0)
        & (labels["blur"] <= 0.3)
        & (labels["underexposure"] <= 0.3)
        & (labels["overexposure"] <= 0.3)
        & (labels["noise"] <= 0.3)
    )
    return labels[mask]


class PatchDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, split: str, train: bool):
        self.paths = [DATA_DIR / "images" / split / f"{r.sample_id}.png" for r in rows.itertuples()]
        self.images = []
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is not None:
                self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.train = train

    def __len__(self):
        # Report more "samples" than images: every epoch re-crops randomly,
        # so this just controls how many random crops we draw per epoch.
        return max(1, len(self.images)) * 8

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        h, w = img.shape[:2]
        crop_size = random.randint(int(min(h, w) * 0.55), min(h, w)) if self.train else min(h, w)
        y = random.randint(0, h - crop_size) if self.train else (h - crop_size) // 2
        x = random.randint(0, w - crop_size) if self.train else (w - crop_size) // 2
        patch = img[y:y + crop_size, x:x + crop_size]
        patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if self.train and random.random() < 0.5:
            patch = np.ascontiguousarray(patch[:, ::-1, :])
        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor


def main(epochs: int = 40, batch_size: int = 16, lr: float = 1e-3):
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    train_rows = normal_subset(labels, "train")
    val_rows = normal_subset(labels, "val")
    print(f"Autoencoder training on {len(train_rows)} 'normal' train images, "
          f"{len(val_rows)} 'normal' val images (device={DEVICE})")

    train_ds = PatchDataset(train_rows, "train", train=True)
    val_ds = PatchDataset(val_rows, "val", train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvAutoencoder(base_channels=16).to(DEVICE)
    print(f"Model parameters: {model.num_parameters():,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                val_losses.append(loss_fn(recon, batch).item())

        tr, va = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  train_loss={tr:.5f}  val_loss={va:.5f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), MODEL_DIR / "autoencoder.pt")

    # Establish an anomaly-score threshold from the "normal" val distribution
    # Threshold selection: score EVERY val image (not just the "normal"
    # training subset) with the same top-fraction aggregate used at
    # inference, then pick the point on the val ROC curve that maximizes
    # Youden's J = TPR - FPR against "is corruption or defect present"
    # (any severity). This is more principled than a normal-only
    # percentile: it directly optimizes the operating point against the
    # target the autoencoder is meant to catch, using only val data (the
    # test split, scored in evaluate.py, remains untouched by this choice).
    model.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pt", map_location=DEVICE))
    model.eval()

    from core.autoencoder import reconstruction_error as _recon_err
    from sklearn.metrics import roc_curve

    all_val_rows = labels[labels["split"] == "val"]
    val_scores, val_binary_labels, normal_scores = [], [], []
    for row in all_val_rows.itertuples():
        img_path = DATA_DIR / "images" / "val" / f"{row.sample_id}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        score, _ = _recon_err(model, img, device=DEVICE)
        val_scores.append(score)
        is_abnormal = int(row.corruption > 0 or row.defect > 0)
        val_binary_labels.append(is_abnormal)
        if not is_abnormal:
            normal_scores.append(score)

    val_scores_arr, val_labels_arr = np.array(val_scores), np.array(val_binary_labels)
    if len(np.unique(val_labels_arr)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(val_labels_arr, val_scores_arr)
        youden_idx = int(np.argmax(tpr - fpr))
        threshold = float(roc_thresholds[youden_idx])
    else:
        threshold = float(np.percentile(normal_scores, 95)) if normal_scores else 0.01

    meta = {
        "img_size": IMG_SIZE,
        "base_channels": 16,
        "num_parameters": model.num_parameters(),
        "epochs": epochs,
        "best_val_loss": best_val,
        "score_aggregation": "mean of top 10% highest-error pixels (see core.autoencoder.reconstruction_error)",
        "anomaly_threshold": threshold,
        "threshold_selection_method": "Youden's J on val ROC (corruption-or-defect vs normal)",
        "normal_val_score_mean": float(np.mean(normal_scores)) if normal_scores else None,
        "normal_val_score_std": float(np.std(normal_scores)) if normal_scores else None,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history": history,
    }
    with open(MODEL_DIR / "autoencoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved autoencoder.pt + meta to {MODEL_DIR}  (threshold={threshold:.6f}, "
          f"selected via Youden's J on {len(val_scores)} val images)")


if __name__ == "__main__":
    main()
