#!/usr/bin/env python3
"""
Minimal regression checks for MotionJointLoss geometry-path refactor.

Covers:
1) rot6d slice/denorm/reproject helper equivalence
2) geodesic kernel equivalence (legacy acos path vs geometry.geodesic_R)
3) parent-relative matrices equivalence
4) angular-velocity path equivalence under standard rotvec semantics
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.geometry import (
    angvel_vec_from_R_seq,
    angvel_vec_from_delta_R,
    geodesic_R,
    parent_relative_matrices,
    reproject_rot6d,
    rot6d_to_matrix,
    so3_exp_map,
)
from train.models import MotionJointLoss


def _legacy_extract_rot6d_flat(loss_obj: MotionJointLoss, x: torch.Tensor) -> torch.Tensor:
    sl = loss_obj.group_slices.get("BoneRotations6D")
    if not isinstance(sl, slice):
        raise RuntimeError("BoneRotations6D slice is missing.")
    rot = x[..., sl]
    mu = torch.as_tensor(loss_obj.mu_y, device=rot.device, dtype=rot.dtype)[..., sl]
    std = torch.as_tensor(loss_obj.std_y, device=rot.device, dtype=rot.dtype)[..., sl].clamp(min=1e-6)
    while mu.dim() < rot.dim():
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)
    rot = rot * std + mu
    return reproject_rot6d(rot)


def _legacy_geodesic(Rp: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
    RtR = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cos)


def _legacy_parent_relative(R: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    J = int(R.shape[-3])
    out = torch.empty_like(R)
    for j in range(J):
        p = int(parents[j].item())
        if p < 0 or p >= J:
            out[..., j, :, :] = R[..., j, :, :]
        else:
            parent = R[..., p, :, :]
            child = R[..., j, :, :]
            out[..., j, :, :] = torch.matmul(parent.transpose(-1, -2), child)
    return out


def _reference_log_map(R: torch.Tensor) -> torch.Tensor:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    skew = R - R.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    denom = sin_theta.unsqueeze(-1)
    factor = theta.unsqueeze(-1) / denom.clamp_min(1e-6)
    small = (sin_theta.abs() < 1e-4).unsqueeze(-1)
    approx = vec
    exact = vec * factor
    return torch.where(small, approx, exact)


def _report(name: str, diff: torch.Tensor, atol: float) -> None:
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(f"[{name}] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} tol={atol:.6e}")
    if not math.isfinite(max_abs) or max_abs > float(atol):
        raise AssertionError(f"{name} diff too large: {max_abs:.6e} > {atol:.6e}")


def main() -> int:
    torch.manual_seed(42)

    # 1) rot6d extract helper (slice + denorm + reproject) vs legacy path.
    B, T, J = 3, 6, 8
    D = J * 6
    layout = {"output_dim": D, "BoneRotations6D": {"start": 0, "size": D}}
    loss_obj = MotionJointLoss(output_layout=layout)
    loss_obj.mu_y = np.random.randn(D).astype(np.float32)
    loss_obj.std_y = (0.1 + np.random.rand(D)).astype(np.float32)
    x = torch.randn(B, T, D)
    new_flat = loss_obj._extract_rot6d_flat(x, denorm=True, reproject=True, sanitize=False)
    if new_flat is None:
        raise AssertionError("_extract_rot6d_flat unexpectedly returned None")
    old_flat = _legacy_extract_rot6d_flat(loss_obj, x)
    _report("extract_rot6d_flat", (new_flat - old_flat).abs(), atol=0.0)

    # 2) geodesic kernel old/new.
    Rp = rot6d_to_matrix(torch.randn(B, T, J, 6))
    Rg = rot6d_to_matrix(torch.randn(B, T, J, 6))
    theta_old = _legacy_geodesic(Rp, Rg)
    theta_new = geodesic_R(Rp, Rg)
    _report("geodesic_R", (theta_old - theta_new).abs(), atol=2e-3)

    # 3) parent-relative matrices old/new.
    parents = torch.tensor([-1, 0, 1, 2, 1, 4, 4, 0], dtype=torch.long)
    rel_old = _legacy_parent_relative(Rp, parents)
    rel_new = parent_relative_matrices(Rp, parents)
    _report("parent_relative_matrices", (rel_old - rel_new).abs(), atol=0.0)

    # 4) angular velocity reference/new (standard rotvec semantics).
    omega_small = 0.03 * torch.randn(B, T, J, 3)
    R_seq = so3_exp_map(torch.cumsum(omega_small, dim=1))
    w_ref_seq = _reference_log_map(torch.matmul(R_seq[:, 1:], R_seq[:, :-1].transpose(-1, -2))) * 60.0
    w_new_seq = angvel_vec_from_R_seq(R_seq, fps=60.0)
    _report("angvel_vec_from_R_seq", (w_ref_seq - w_new_seq).abs(), atol=1e-5)

    dR = so3_exp_map(0.03 * torch.randn(B, T, J, 3))
    w_old_delta = _reference_log_map(dR) * 60.0
    w_new_delta = angvel_vec_from_delta_R(dR, fps=60.0)
    _report("angvel_vec_from_delta_R", (w_old_delta - w_new_delta).abs(), atol=1e-5)

    print("[OK] MotionJointLoss geometry refactor checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
