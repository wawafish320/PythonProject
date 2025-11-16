"""Shared normalization helpers for vector features (NumPy & Torch)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "VectorTanhNormalizer",
    "VectorTanhNormalizerTorch",
]


class VectorTanhNormalizer:
    """
    NumPy version for dataset preprocessing.
    Applies tanh(x / scale), optional z-score; supports inverse_transform for debugging.
    """

    def __init__(self, scales: np.ndarray, mu: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        scales = np.asarray(scales, dtype=np.float32)
        if scales.ndim != 1:
            raise ValueError(f"scales must be 1-D, got {scales.shape}")
        self.scales = np.clip(scales, 1e-6, None)
        if mu is not None:
            mu = np.asarray(mu, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            if mu.shape != self.scales.shape or std.shape != self.scales.shape:
                raise ValueError("mu/std shape mismatch with scales.")
            self.mu = mu
            self.std = np.clip(std, 1e-6, None)
        else:
            self.mu, self.std = None, None

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        X = np.tanh(arr / self.scales)
        if self.mu is not None and self.std is not None:
            X = (X - self.mu) / self.std
        return X.astype(np.float32, copy=False)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        Y = arr.astype(np.float32, copy=False)
        if self.mu is not None and self.std is not None:
            Y = Y * self.std + self.mu
        Y = np.clip(Y, -0.999999, 0.999999)
        return np.arctanh(Y) * self.scales


class VectorTanhNormalizerTorch(nn.Module):
    """
    Torch version for training-time pose-history normalization.
    Mirrors VectorTanhNormalizer logic with device/dtype alignment.
    """

    def __init__(self, scales: torch.Tensor, mu: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("scales", scales.clone().float().clamp_min(1e-6))
        self.register_buffer("mu", None if mu is None else mu.clone().float())
        self.register_buffer("std", None if std is None else std.clone().float().clamp_min(1e-6))

    def forward(self, raw_flat: torch.Tensor) -> torch.Tensor:
        if raw_flat.numel() == 0:
            return raw_flat
        z = torch.tanh(raw_flat / self.scales)
        if self.mu is not None and self.std is not None:
            z = (z - self.mu) / self.std
        return z

    def inverse(self, norm_flat: torch.Tensor) -> torch.Tensor:
        if norm_flat.numel() == 0:
            return norm_flat
        z = norm_flat
        if self.mu is not None and self.std is not None:
            z = z * self.std + self.mu
        eps = 1.0 - 1e-6
        z = z.clamp(min=-eps, max=eps)
        if hasattr(torch, "atanh"):
            raw = torch.atanh(z) * self.scales
        else:
            raw = 0.5 * (torch.log1p(z) - torch.log1p(-z)) * self.scales
        return raw

