"""Evaluates the trained hybrid quality engine on the held-out TEST split
(scenes never used for training or validation of either the classifiers or
the autoencoder).

Produces, per issue type: accuracy, macro precision/recall/F1, and a
confusion matrix over the 4 severity buckets (none/low/medium/high).
Also produces an anomaly-detection evaluation for the autoencoder in
isolation: ROC-AUC and a confusion matrix at the trained threshold for
"is this image corrupted or defective" (binary), which is the target the
anomaly formulation was designed for.

Writes a JSON report and PNG plots (confusion matrices, ROC curve,
reconstruction-error distribution) to evaluation/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.autoencoder import reconstruction_error
from core.degrade import ISSUE_TYPES
from core.quality_engine import BUCKET_RANK, QualityEngine

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models_store"
OUT_DIR = Path(__file__).resolve().parent.parent / "evaluation"
BUCKETS = ["none", "low", "medium", "high"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    test_rows = labels[labels["split"] == "test"].reset_index(drop=True)
    print(f"Evaluating on {len(test_rows)} held-out TEST samples "
          f"({test_rows['scene_id'].nunique()} unseen scenes)")

    engine = QualityEngine(MODEL_DIR)
    assert engine.is_ready, "Model artifacts missing -- run train_classifiers.py and train_autoencoder.py first."

    y_true = {issue: [] for issue in ISSUE_TYPES}
    y_pred = {issue: [] for issue in ISSUE_TYPES}
    anomaly_scores, anomaly_binary_true = [], []
    quality_scores, quality_labels = [], []

    t0 = time.time()
    for row in test_rows.itertuples():
        img_path = DATA_DIR / "images" / "test" / f"{row.sample_id}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = engine.analyze(img)

        pred_by_type = {i["type"]: i["severity"] for i in result["issues"]}
        for issue in ISSUE_TYPES:
            y_true[issue].append(getattr(row, f"{issue}_bucket"))
            y_pred[issue].append(pred_by_type.get(issue, "none"))

        anomaly_scores.append(result["anomaly_score"])
        anomaly_binary_true.append(1 if (row.corruption > 0 or row.defect > 0) else 0)
        quality_scores.append(result["quality_score"])
        quality_labels.append(result["quality_label"])
    elapsed = time.time() - t0
    print(f"Inference on {len(test_rows)} images took {elapsed:.1f}s "
          f"({1000*elapsed/max(1,len(test_rows)):.1f} ms/image)")

    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_test_samples": len(test_rows), "n_test_scenes": int(test_rows["scene_id"].nunique()),
              "per_issue": {}, "anomaly_detector": {}, "quality_label_distribution": {}}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, issue in zip(axes.flat, ISSUE_TYPES):
        yt, yp = y_true[issue], y_pred[issue]
        acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
        prec, rec, f1, support = precision_recall_fscore_support(
            yt, yp, labels=BUCKETS, average=None, zero_division=0)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            yt, yp, labels=BUCKETS, average="macro", zero_division=0)
        cm = confusion_matrix(yt, yp, labels=BUCKETS)

        report["per_issue"][issue] = {
            "accuracy": acc,
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "per_class": {
                cls: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                for cls, p, r, f, s in zip(BUCKETS, prec, rec, f1, support)
            },
            "confusion_matrix": cm.tolist(),
        }
        print(f"  {issue:15s} acc={acc:.3f}  macro_f1={macro_f1:.3f}")

        ConfusionMatrixDisplay(cm, display_labels=BUCKETS).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{issue} (acc={acc:.2f}, F1={macro_f1:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    # --- Anomaly detector (autoencoder) evaluation ---
    anomaly_scores_arr = np.array(anomaly_scores)
    anomaly_true_arr = np.array(anomaly_binary_true)
    if len(np.unique(anomaly_true_arr)) > 1:
        auc = float(roc_auc_score(anomaly_true_arr, anomaly_scores_arr))
        fpr, tpr, thresholds = roc_curve(anomaly_true_arr, anomaly_scores_arr)
    else:
        auc, fpr, tpr = None, None, None

    pred_binary = (anomaly_scores_arr > engine.anomaly_threshold).astype(int)
    cm_anomaly = confusion_matrix(anomaly_true_arr, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm_anomaly.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    report["anomaly_detector"] = {
        "task": "binary: is corruption or defect present (any severity)",
        "threshold_used": float(engine.anomaly_threshold),
        "roc_auc": auc,
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "confusion_matrix_[[tn,fp],[fn,tp]]": cm_anomaly.tolist(),
    }
    print(f"  anomaly_detector  ROC-AUC={auc}  precision={precision:.3f}  recall={recall:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if fpr is not None:
        axes[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Anomaly detector ROC curve")
        axes[0].legend()
    axes[1].hist(anomaly_scores_arr[anomaly_true_arr == 0], bins=30, alpha=0.6, label="normal")
    axes[1].hist(anomaly_scores_arr[anomaly_true_arr == 1], bins=30, alpha=0.6, label="corrupted/defective")
    axes[1].axvline(engine.anomaly_threshold, color="red", linestyle="--", label="threshold")
    axes[1].set_xlabel("Reconstruction error (anomaly score)")
    axes[1].set_title("Reconstruction error distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "anomaly_roc_and_distribution.png", dpi=130)
    plt.close(fig)

    # --- Overall quality label distribution + failure case sampling ---
    for lbl in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]:
        report["quality_label_distribution"][lbl] = int(sum(1 for q in quality_labels if q == lbl))

    failure_cases = []
    for row, yt_blur, yp_blur in zip(test_rows.itertuples(), y_true["blur"], y_pred["blur"]):
        if BUCKET_RANK.get(yt_blur, 0) >= 2 and yp_blur == "none":
            failure_cases.append({"sample_id": row.sample_id, "issue": "blur",
                                  "true": yt_blur, "predicted": yp_blur})
    report["example_failure_cases"] = failure_cases[:15]

    report_path = OUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")
    print(f"Wrote {OUT_DIR / 'confusion_matrices.png'}")
    print(f"Wrote {OUT_DIR / 'anomaly_roc_and_distribution.png'}")


if __name__ == "__main__":
    main()
