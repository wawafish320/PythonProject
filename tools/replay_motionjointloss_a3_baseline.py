#!/usr/bin/env python3
"""
Phase A3 baseline replay for MotionJointLoss geometry paths.

This script uses a fixed synthetic seed/batch to record stable baseline values
for:
  - rot_geo
  - rot_local
  - rot_vel (compute_rot6d_log_loss)

It also reports legacy-vs-refactor diffs to ensure semantic consistency.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.geometry import geodesic_R, reproject_rot6d, root_relative_matrices, rot6d_to_matrix
from train.models import MotionJointLoss


def _legacy_extract_rot6d_flat(loss_obj: MotionJointLoss, x: torch.Tensor) -> torch.Tensor:
    sl = loss_obj.group_slices.get("BoneRotations6D")
    if not isinstance(sl, slice):
        raise RuntimeError("BoneRotations6D slice missing.")
    rot = x[..., sl]
    D = int(rot.shape[-1])
    if D % 6 != 0:
        raise RuntimeError("BoneRotations6D dim is not divisible by 6.")
    if getattr(loss_obj, "mu_y", None) is not None and getattr(loss_obj, "std_y", None) is not None:
        st = int(sl.start or 0)
        mu = torch.as_tensor(loss_obj.mu_y, device=rot.device, dtype=rot.dtype)[..., st:st + D]
        std = torch.as_tensor(loss_obj.std_y, device=rot.device, dtype=rot.dtype)[..., st:st + D].clamp(min=1e-6)
        while mu.dim() < rot.dim():
            mu = mu.unsqueeze(0)
            std = std.unsqueeze(0)
        rot = rot * std + mu
    return reproject_rot6d(rot)


def _legacy_geo_from_rot6d(loss_obj: MotionJointLoss, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pr = _legacy_extract_rot6d_flat(loss_obj, pred)
    gr = _legacy_extract_rot6d_flat(loss_obj, gt)
    J = int(pr.shape[-1]) // 6
    Rp = rot6d_to_matrix(pr.view(*pr.shape[:-1], J, 6))
    Rg = rot6d_to_matrix(gr.view(*gr.shape[:-1], J, 6))
    RtR = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.arccos(cos)
    weights = loss_obj._joint_weight_vector(theta.device, theta.dtype, J)
    view_shape = (1,) * (theta.dim() - 1) + (J,)
    return (theta * weights.view(*view_shape)).mean()


def _legacy_parent_relative(R: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    J = int(R.shape[-3])
    out = torch.empty_like(R)
    for j in range(J):
        p = int(parents[j].item())
        if p < 0 or p >= J:
            out[..., j, :, :] = R[..., j, :, :]
        else:
            out[..., j, :, :] = torch.matmul(R[..., p, :, :].transpose(-1, -2), R[..., j, :, :])
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


def _legacy_rot_local(loss_obj: MotionJointLoss, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    Rp_world = loss_obj._rot6d_matrices(pred)
    Rg_world = loss_obj._rot6d_matrices(gt)
    if Rp_world is None or Rg_world is None:
        return pred.new_tensor(0.0)
    Rp_root = root_relative_matrices(Rp_world, int(getattr(loss_obj, "root_idx", 0)))
    Rg_root = root_relative_matrices(Rg_world, int(getattr(loss_obj, "root_idx", 0)))
    parents = getattr(loss_obj, "_parents_tensor", None)
    J = int(Rp_root.shape[-3])
    if parents is None or parents.device != Rp_root.device or int(parents.numel()) < J:
        parents = torch.as_tensor(loss_obj.parents[:J], device=Rp_root.device, dtype=torch.long)
        loss_obj._parents_tensor = parents
    else:
        parents = parents[:J]
    Rp_local = _legacy_parent_relative(Rp_root, parents)
    Rg_local = _legacy_parent_relative(Rg_root, parents)
    geo_local = geodesic_R(Rp_local, Rg_local)
    weights = loss_obj._joint_weight_vector(geo_local.device, geo_local.dtype, J)
    w = weights.view((1,) * (geo_local.dim() - 1) + (J,))
    return (geo_local * w).mean()


def _legacy_rot_vel(loss_obj: MotionJointLoss, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    Rp = loss_obj._rot6d_matrices(pred)
    Rg = loss_obj._rot6d_matrices(gt)
    if Rp is None or Rg is None or Rp.dim() < 5 or int(Rp.shape[-4]) < 2:
        return pred.new_tensor(0.0)
    dRp = torch.matmul(Rp[..., 1:, :, :, :], Rp[..., :-1, :, :, :].transpose(-1, -2))
    dRg = torch.matmul(Rg[..., 1:, :, :, :], Rg[..., :-1, :, :, :].transpose(-1, -2))
    log_p = _reference_log_map(dRp)
    log_g = _reference_log_map(dRg)
    return F.smooth_l1_loss(log_p, log_g)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_loss(joints: int, seed: int) -> MotionJointLoss:
    D = int(joints) * 6
    layout = {"output_dim": D, "BoneRotations6D": {"start": 0, "size": D}}
    parents = [-1] + [max(0, i - 1) for i in range(1, int(joints))]
    loss_obj = MotionJointLoss(output_layout=layout, meta={"skeleton": {"parents": parents}}, w_rot_local=1.0)

    rng = np.random.default_rng(seed + 17)
    loss_obj.mu_y = rng.normal(0.0, 0.1, size=(D,)).astype(np.float32)
    loss_obj.std_y = rng.uniform(0.1, 1.2, size=(D,)).astype(np.float32)
    return loss_obj


def run(seed: int, batch: int, timesteps: int, joints: int) -> dict:
    _set_seed(seed)
    loss_obj = _build_loss(joints=joints, seed=seed)
    D = joints * 6

    pred = torch.randn(batch, timesteps, D, dtype=torch.float32)
    gt = torch.randn(batch, timesteps, D, dtype=torch.float32)

    rot_geo_new = loss_obj.compute_rot6d_geo_loss(pred, gt)
    rot_local_new = pred.new_tensor(0.0)
    Rp_world = loss_obj._rot6d_matrices(pred)
    Rg_world = loss_obj._rot6d_matrices(gt)
    if Rp_world is not None and Rg_world is not None:
        Rp_root = loss_obj._root_relative(Rp_world)
        Rg_root = loss_obj._root_relative(Rg_world)
        Rp_local = loss_obj._parent_relative_matrices(Rp_root)
        Rg_local = loss_obj._parent_relative_matrices(Rg_root)
        geo_local = geodesic_R(Rp_local, Rg_local)
        weights = loss_obj._joint_weight_vector(geo_local.device, geo_local.dtype, joints)
        rot_local_new = (geo_local * weights.view(1, 1, joints)).mean()
    rot_vel_new = loss_obj.compute_rot6d_log_loss(pred, gt)

    rot_geo_old = _legacy_geo_from_rot6d(loss_obj, pred, gt)
    rot_local_old = _legacy_rot_local(loss_obj, pred, gt)
    rot_vel_old = _legacy_rot_vel(loss_obj, pred, gt)

    def _f(x: torch.Tensor) -> float:
        return float(x.detach().cpu().item())

    out = {
        "meta": {
            "seed": int(seed),
            "batch_size": int(batch),
            "timesteps": int(timesteps),
            "joints": int(joints),
            "rot_dim": int(D),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "baseline": {
            "rot_geo_new": _f(rot_geo_new),
            "rot_geo_old": _f(rot_geo_old),
            "rot_geo_abs_diff": abs(_f(rot_geo_new - rot_geo_old)),
            "rot_local_new": _f(rot_local_new),
            "rot_local_old": _f(rot_local_old),
            "rot_local_abs_diff": abs(_f(rot_local_new - rot_local_old)),
            "rot_vel_new": _f(rot_vel_new),
            "rot_vel_old": _f(rot_vel_old),
            "rot_vel_abs_diff": abs(_f(rot_vel_new - rot_vel_old)),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay fixed seed/batch baseline for MotionJointLoss geometry paths.")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=16)
    ap.add_argument("--joints", type=int, default=46)
    ap.add_argument(
        "--out",
        type=str,
        default="docs/Problems/active/artifacts/2026-02-17_models_geometry_a3_baseline_seed20260223.json",
    )
    args = ap.parse_args()

    payload = run(
        seed=int(args.seed),
        batch=int(args.batch),
        timesteps=int(args.timesteps),
        joints=int(args.joints),
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b = payload["baseline"]
    print(f"[A3] baseline saved: {out_path}")
    print(
        "[A3] "
        f"rot_geo(new/old/diff)={b['rot_geo_new']:.9f}/{b['rot_geo_old']:.9f}/{b['rot_geo_abs_diff']:.3e}; "
        f"rot_local(new/old/diff)={b['rot_local_new']:.9f}/{b['rot_local_old']:.9f}/{b['rot_local_abs_diff']:.3e}; "
        f"rot_vel(new/old/diff)={b['rot_vel_new']:.9f}/{b['rot_vel_old']:.9f}/{b['rot_vel_abs_diff']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
