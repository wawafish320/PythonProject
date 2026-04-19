"""Shared normalization helpers for vector features (NumPy & Torch)."""

from __future__ import annotations

from typing import Optional, Sequence
from dataclasses import dataclass

import json
import numpy as np
import torch
import torch.nn as nn

from ..contracts.asset_semantics import require_standard_rotvec_spec

__all__ = [
    "VectorTanhNormalizer",
    "VectorTanhNormalizerTorch",
    "AngvelNormalizer",
    "AngvelNormCfg",
    "normalize_cond_tensor",
    "prepare_runtime_stat_tensor",
    "_make_angnorm_from_spec",
]


class VectorTanhNormalizer:
    """
    NumPy version for dataset preprocessing.
    Applies tanh(x / scale), optional z-score; supports inverse() for debugging.
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

    def inverse(self, arr: np.ndarray) -> np.ndarray:
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


def prepare_runtime_stat_tensor(
    value,
    *,
    ref_tensor: torch.Tensor,
    cache: Optional[dict] = None,
    cache_key: Optional[str] = None,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    device = ref_tensor.device
    dtype = ref_tensor.dtype

    bucket = None
    runtime_key = None
    if isinstance(cache, dict) and cache_key is not None:
        bucket = cache.setdefault(str(cache_key), {})
        runtime_key = (device, dtype)
        tensor = bucket.get(runtime_key)
        if tensor is not None:
            return tensor

    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)

    if bucket is not None and runtime_key is not None:
        bucket[runtime_key] = tensor
    return tensor


def normalize_cond_tensor(
    cond_raw: Optional[torch.Tensor],
    cond_mu: Optional[torch.Tensor],
    cond_std: Optional[torch.Tensor],
    *,
    cond_norm_clip: float = 6.0,
) -> Optional[torch.Tensor]:
    if cond_raw is None or cond_mu is None or cond_std is None:
        return None
    if cond_mu.dim() == 3:
        cond_mu = cond_mu.squeeze(1)
    if cond_std.dim() == 3:
        cond_std = cond_std.squeeze(1)
    if cond_mu.shape != cond_raw.shape:
        if cond_mu.size(0) == 1 and cond_raw.size(0) > 1:
            cond_mu = cond_mu.expand(cond_raw.size(0), -1)
        if cond_std.size(0) == 1 and cond_raw.size(0) > 1:
            cond_std = cond_std.expand(cond_raw.size(0), -1)
    std = cond_std.clamp_min(1e-6)
    cond_norm = (cond_raw - cond_mu) / std
    clamp_val = float(cond_norm_clip or 0.0)
    if clamp_val > 0:
        cond_norm = cond_norm.clamp(-clamp_val, clamp_val)
    return cond_norm


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


class AngvelNormalizer:
    """
    Consume ONLY angvel-specific fields from norm_template.json placed beside the npz:
      - REQUIRED: tanh_scales_angvel (or s_eff_angvel)  -> used for tanh compression
      - OPTIONAL: MuAngVel & StdAngVel -> if both present, perform z-score; otherwise skip
    """

    def __init__(self, tpl_path: str, J_times_3: int, require_zscore: bool = False):
        with open(tpl_path, "r", encoding="utf-8") as f:
            TPL = json.load(f)
        require_standard_rotvec_spec(TPL, context=f"angvel template {tpl_path}")

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

    def inverse(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2 and X.shape[1] == self.s_eff.size, \
            f"X shape {tuple(X.shape)} not compatible with J*3={self.s_eff.size}."
        Y = X.astype(np.float32)
        if getattr(self, "mu", None) is not None and getattr(self, "std", None) is not None:
            Y = Y * self.std + self.mu
        Y = np.clip(Y, -0.999999, 0.999999)
        W_raw = np.arctanh(Y) * self.s_eff
        return W_raw.astype(np.float32)


def _make_angnorm_from_spec(spec: dict, J_times_3: int, require_zscore: bool):
    require_standard_rotvec_spec(spec, context="angvel norm spec")
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
