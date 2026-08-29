"""Shared image-quality core: feature extraction, synthetic degradations, and
the autoencoder architecture used by both the offline training scripts (ml/)
and the online inference service (backend/app). Keeping this logic in one
place guarantees train/serve parity.
"""
