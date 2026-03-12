#!/usr/bin/env python3
"""
Sanity check for supervised hinge delta_target computation.

We want:
  R_gt = R_base @ exp(delta * axis)
  R_err = R_base^T @ R_gt = exp(delta * axis)
  omega = log(R_err)
  delta_target = omega[axis]  ~= delta
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so add project root for `import train.*`.
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.geometry import so3_exp_map, so3_log_map


def _axis_vec(axis: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = str(axis).strip().lower()
    idx = {"x": 0, "y": 1, "z": 2}.get(a, 2)
    v = torch.zeros(3, device=device, dtype=dtype)
    v[idx] = 1.0
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--delta", type=float, default=0.7, help="Delta in radians.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cpu")
    dtype = torch.float32
    B = int(args.batch)
    delta = float(args.delta)
    tol = float(args.tol)

    axis = _axis_vec(args.axis, device=device, dtype=dtype)
    omega = axis.view(1, 1, 3).expand(B, 1, 3) * delta
    R_delta = so3_exp_map(omega).squeeze(1)  # (B,3,3)

    # Random base rotation via exp map (keeps this test self-contained).
    base_omega = torch.randn(B, 3, device=device, dtype=dtype) * 0.5
    R_base = so3_exp_map(base_omega.view(B, 1, 3)).squeeze(1)

    R_gt = R_base @ R_delta
    R_err = R_base.transpose(-1, -2) @ R_gt
    omega_err = so3_log_map(R_err)  # (B,3)

    axis_i = {"x": 0, "y": 1, "z": 2}[args.axis]
    delta_tgt = omega_err[:, axis_i]
    max_abs_err = (delta_tgt - delta).abs().max().item()

    print(f"axis={args.axis} delta={delta:.6f}  max|delta_target-delta|={max_abs_err:.6e}")
    if not (max_abs_err <= tol):
        raise SystemExit(f"[FAIL] max_abs_err={max_abs_err:.6e} > tol={tol:.6e}")
    print("[OK]")


if __name__ == "__main__":
    main()
