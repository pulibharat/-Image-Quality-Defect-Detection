"""Lightweight convolutional autoencoder used for the anomaly-detection half
of the quality engine.

Formulation: the network is trained ONLY on patches from clean, undegraded
images, so it learns to reconstruct "normal" natural-image statistics
cheaply. At inference time an image that is corrupted, heavily defective, or
otherwise far outside that learned distribution reconstructs poorly -- the
per-pixel reconstruction error is therefore both an anomaly *score*
(mean error) and, upsampled back to the input resolution, a localization
*heatmap* of which regions look abnormal (used for explainability / the
optional quality-heatmap feature).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 128


class ConvAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),        # 128 -> 64
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),  # 64 -> 32
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),  # 16 -> 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 4, 4, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),  # 8 -> 16
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),        # 32 -> 64
            nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1), nn.Sigmoid(),                                        # 64 -> 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def preprocess(image: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalized float tensor (1, 3, size, size)."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def reconstruction_error(model: ConvAutoencoder, image: np.ndarray, device: str = "cpu", top_fraction: float = 0.1):
    """Returns (scalar_anomaly_score, error_heatmap_uint8_at_original_size).

    The scalar score is the mean error over the worst `top_fraction` of
    pixels rather than the whole-image mean. Corruption/defect artifacts
    (a scratch, a wiped block, a dust spot) typically cover only a small
    part of the frame, so a plain global mean dilutes them into the noise
    floor of everything else that reconstructs fine; a top-k aggregate
    stays sensitive to localized damage while still averaging out isolated
    single-pixel reconstruction noise.
    """
    model.eval()
    x = preprocess(image).to(device)
    recon = model(x)
    err = (recon - x).pow(2).mean(dim=1, keepdim=True)  # (1,1,size,size)

    err_flat = err.flatten()
    k = max(1, int(err_flat.numel() * top_fraction))
    score = float(torch.topk(err_flat, k).values.mean().item())

    err_np = err.squeeze().cpu().numpy()
    err_np = err_np / (err_np.max() + 1e-9)
    h, w = image.shape[:2]
    heatmap = cv2.resize(err_np, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    return score, heatmap
