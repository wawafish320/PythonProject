"""Shared normalization helpers for vector features (NumPy & Torch)."""

from __future__ import annotations

from typing import Optional, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "VectorTanhNormalizer",
    "VectorTanhNormalizerTorch",
    "AngvelNormalizer",
    "AngvelNormCfg",
    "_make_angnorm_from_spec",
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


# -----------------------------------------------------------------------------
# Angular velocity normalizer (moved from pretrain_mpl_min.py)
# -----------------------------------------------------------------------------

@dataclass
class AngvelNormCfg:
    s_eff: np.ndarray  # [J*3]
    mu: Optional[np.ndarray]  # [J*3] or None
    std: Optional[np.ndarray]  # [J*3] or None

    def transform(self, W_raw: np.ndarray) -> np.ndarray:
        """tanh-squash (and optional z-score) angular velocity."""
        X = np.tanh(W_raw / self.s_eff)
        if self.mu is not None and self.std is not None:
            X = (X - self.mu) / self.std
        return X.astype(np.float32)

    def inverse(self, X_norm: np.ndarray) -> np.ndarray:
        """Inverse of transform for angular velocity."""
        X = X_norm
        if self.mu is not None and self.std is not None:
            X = X * self.std + self.mu
        X = np.clip(X, -0.999999, 0.999999)
        W_raw = np.arctanh(X) * self.s_eff
        return W_raw.astype(np.float32)

    # Backwards compatibility for call sites using inverse_transform
    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return self.inverse(X_norm)


class AngvelNormalizer:
    """
    Consume ONLY angvel-specific fields from norm_template.json placed beside the npz:
      - REQUIRED: tanh_scales_angvel (or s_eff_angvel)  -> used for tanh compression
      - OPTIONAL: MuAngVel & StdAngVel -> if both present, perform z-score; otherwise skip
    """

    def __init__(self, tpl_path: str, J_times_3: int, require_zscore: bool = False):
        with open(tpl_path, "r", encoding="utf-8") as f:
            TPL = json.load(f)

        def _vec(name):
            v = TPL.get(name, None)
            return None if v is None else np.asarray(v, dtype=np.float32)

        s = _vec("tanh_scales_angvel") or _vec("s_eff_angvel")
        if s is None:
            raise RuntimeError("norm_template.json missing 'tanh_scales_angvel' (or 's_eff_angvel').")
        if s.size != J_times_3:
            raise RuntimeError(f"tanh_scales_angvel length {s.size} != J*3 {J_times_3}")
        self.s_eff = np.clip(s, 1e-6, None).astype(np.float32)
        self.scales = self.s_eff

        muA, sdA = _vec("MuAngVel"), _vec("StdAngVel")
        if (muA is not None) ^ (sdA is not None):
            raise RuntimeError("Both MuAngVel and StdAngVel must exist together, or both be absent.")
        if muA is not None:
            if muA.size != J_times_3 or sdA.size != J_times_3:
                raise RuntimeError("MuAngVel/StdAngVel size must equal J*3.")
            self.mu = muA.astype(np.float32)
            self.std = np.clip(sdA.astype(np.float32), 1e-6, None)
        else:
            if require_zscore:
                raise RuntimeError("require_zscore=True but MuAngVel/StdAngVel not found in template.")
            self.mu, self.std = None, None

        self.require_z = require_zscore

    def transform(self, W_raw: np.ndarray) -> np.ndarray:
        assert W_raw.ndim == 2 and W_raw.shape[1] == self.s_eff.size, \
            f"W_raw shape {W_raw.shape} not compatible with J*3={self.s_eff.size}."
        X = np.tanh(W_raw / self.s_eff)
        if self.mu is not None and self.std is not None:
            X = (X - self.mu) / self.std
        return X.astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2 and X.shape[1] == self.s_eff.size, \
            f"X shape {tuple(X.shape)} not compatible with J*3={self.s_eff.size}."
        Y = X.astype(np.float32)
        if getattr(self, "mu", None) is not None and getattr(self, "std", None) is not None:
            Y = Y * self.std + self.mu
        Y = np.clip(Y, -0.999999, 0.999999)
        W_raw = np.arctanh(Y) * self.s_eff
        return W_raw.astype(np.float32)


def _make_angnorm_from_spec(spec: dict, J_times_3: int, require_zscore: bool):
    s = np.asarray(spec.get("tanh_scales_angvel") or spec.get("s_eff_angvel"), dtype=np.float32)
    if s.size != J_times_3:
        raise RuntimeError(f"norm spec: tanh_scales_angvel length {s.size} != J*3 {J_times_3}")
    mu = spec.get("MuAngVel")
    std = spec.get("StdAngVel")
    if (mu is None) ^ (std is None):
        raise RuntimeError("MuAngVel/StdAngVel must appear together or both absent.")
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mu.size != J_times_3 or std.size != J_times_3:
            raise RuntimeError("MuAngVel/StdAngVel size mismatch with J*3.")
    if require_zscore and (mu is None or std is None):
        raise RuntimeError("require_zscore=True but MuAngVel/StdAngVel missing.")
    return AngvelNormCfg(s_eff=np.clip(s, 1e-6, None), mu=mu, std=std)
