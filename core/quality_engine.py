"""The hybrid decision engine: loads the trained per-issue classifiers and
the anomaly autoencoder once, then combines their outputs with the raw
engineered features into the final structured quality assessment.

Shared by ml/evaluate.py (offline evaluation against the held-out test set)
and backend/app/services (online inference), so there is exactly one
implementation of "how a prediction becomes a quality_score" to keep
train-time evaluation and serving in lockstep.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from core.autoencoder import ConvAutoencoder, reconstruction_error
from core.degrade import ISSUE_TYPES
from core.features import extract_features, feature_vector

BUCKET_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SEVERITY_MULTIPLIER = {"none": 0.0, "low": 0.35, "medium": 0.7, "high": 1.0}
ISSUE_WEIGHT = {
    "blur": 15, "underexposure": 12, "overexposure": 12,
    "noise": 10, "corruption": 25, "defect": 20,
}
EXPLANATIONS = {
    "blur": "Laplacian/Sobel sharpness statistics are below the range seen in sharp training images.",
    "underexposure": "A large share of pixels are near-black and overall brightness is low.",
    "overexposure": "A large share of pixels are near-white (clipped highlights) and/or brightness is high.",
    "noise": "The high-frequency (Immerkaer) noise estimate exceeds typical clean-image levels.",
    "corruption": "Blockiness, entropy, and edge-density statistics match the pattern of heavy compression or transmission corruption.",
    "defect": "Texture, edge, and blockiness statistics match the pattern of a localized scratch, spot, or sensor defect.",
}


class QualityEngine:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device

        self.classifiers = {}
        for issue in ISSUE_TYPES:
            path = self.model_dir / f"{issue}_classifier.joblib"
            self.classifiers[issue] = joblib.load(path) if path.exists() else None

        clf_meta_path = self.model_dir / "classifiers_meta.json"
        self.classifiers_meta = json.loads(clf_meta_path.read_text()) if clf_meta_path.exists() else {}

        ae_meta_path = self.model_dir / "autoencoder_meta.json"
        self.autoencoder_meta = json.loads(ae_meta_path.read_text()) if ae_meta_path.exists() else {}
        self.anomaly_threshold = self.autoencoder_meta.get("anomaly_threshold", 0.01)

        ae_path = self.model_dir / "autoencoder.pt"
        self.autoencoder = None
        if ae_path.exists():
            base_channels = self.autoencoder_meta.get("base_channels", 16)
            model = ConvAutoencoder(base_channels=base_channels)
            model.load_state_dict(torch.load(ae_path, map_location=device))
            model.eval()
            self.autoencoder = model

    @property
    def is_ready(self) -> bool:
        return self.autoencoder is not None and all(self.classifiers.values())

    def _classify_issue(self, issue: str, x: np.ndarray) -> tuple[str, float, dict]:
        clf = self.classifiers[issue]
        bucket = clf.predict(x)[0]
        proba = clf.predict_proba(x)[0]
        proba_map = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
        confidence = float(max(proba))
        return bucket, confidence, proba_map

    def analyze(self, image: np.ndarray) -> dict:
        feats = extract_features(image)
        x = feature_vector(feats).reshape(1, -1)

        anomaly_score, heatmap = (None, None)
        if self.autoencoder is not None:
            anomaly_score, heatmap = reconstruction_error(self.autoencoder, image, device=self.device)

        classifier_outputs = {}
        for issue in ISSUE_TYPES:
            bucket, confidence, proba_map = self._classify_issue(issue, x)
            classifier_outputs[issue] = {"bucket": bucket, "confidence": confidence, "proba": proba_map}

        # NOTE on the anomaly autoencoder's role here: an earlier version of
        # this method let a strong anomaly score (ratio = score/threshold)
        # override a classifier "none" for corruption/defect. That was
        # removed after evaluation showed the reconstruction-error score is
        # confounded by raw image texture/detail as much as by actual
        # defects: e.g. a clean, sharp, real photo (sample_images/01) scores
        # ratio=4.9 (would fire a "strong anomaly"), while a genuinely
        # scratched/defective photo (sample_images/07) scores only 1.7. In
        # other words, on real (non-synthetic-scene) photos the score
        # tracks "how visually complex is this image" more than "how
        # defective is it" -- exactly the failure mode you'd expect from a
        # small autoencoder trained mostly on procedurally-generated normal
        # scenes (see ml/train_autoencoder.py, README "Limitations").
        # Its standalone binary ROC-AUC on the in-domain synthetic held-out
        # test set is a respectable 0.74 (evaluation/evaluation_report.json),
        # but that number does not hold up on out-of-domain real photos, so
        # letting it drive the primary decision was a net accuracy loss.
        # The classifiers (92-98% held-out accuracy, see evaluation/) are
        # therefore the sole basis for `issues`; the anomaly score and its
        # localization heatmap are still returned below as a secondary,
        # clearly-labeled diagnostic signal for explainability, not as a
        # detector in their own right.

        issues = []
        for issue in ISSUE_TYPES:
            output = classifier_outputs[issue]
            if output["bucket"] != "none":
                issues.append({
                    "type": issue,
                    "severity": output["bucket"],
                    "confidence": round(output["confidence"], 3),
                    "confidence_source": "classifier",
                    "explanation": EXPLANATIONS[issue],
                })

        penalty = sum(
            ISSUE_WEIGHT[i["type"]] * SEVERITY_MULTIPLIER[i["severity"]] * i["confidence"]
            for i in issues
        )
        quality_score = float(np.clip(100 - penalty, 0, 100))

        has_high_severe = any(i["severity"] == "high" and i["type"] in ("corruption", "defect") for i in issues)
        if has_high_severe or quality_score < 40:
            quality_label = "DEFECTIVE"
        elif not issues and quality_score >= 75:
            quality_label = "ACCEPTABLE"
        else:
            quality_label = "DEGRADED"

        issues.sort(key=lambda i: (BUCKET_RANK[i["severity"]], i["confidence"]), reverse=True)

        return {
            "quality_score": round(quality_score, 1),
            "quality_label": quality_label,
            "issues": issues,
            "features": {k: round(float(v), 4) for k, v in feats.items()},
            "anomaly_score": round(float(anomaly_score), 6) if anomaly_score is not None else None,
            "anomaly_threshold": round(float(self.anomaly_threshold), 6),
            "heatmap": heatmap,
            "classifier_outputs": classifier_outputs,
        }
