#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.geometry import (
    angvel_vec_from_R_seq,
    angvel_vec_from_delta_R,
    geodesic_R,
    so3_exp_map,
    so3_log_map,
)
from train.rotvec_semantics import (
    require_standard_rotvec_bundle,
    require_standard_rotvec_spec,
)


def _assert_close(name: str, got: torch.Tensor, exp: torch.Tensor, tol: float) -> None:
    err = float((got - exp).abs().max().item())
    if err > tol:
        raise SystemExit(f"[FAIL] {name}: max_abs_err={err:.6e} > tol={tol:.6e}")


def _check_geometry_roundtrip() -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    omega = axis * 1.0
    rt = so3_log_map(so3_exp_map(omega))
    _assert_close("log(exp([0,0,1]))", rt, omega, 1e-5)

    for angle in (0.1, 0.5, 1.0, math.pi / 2.0):
        w = axis * float(angle)
        rt = so3_log_map(so3_exp_map(w))
        _assert_close(f"roundtrip angle={angle:.6f}", rt, w, 1e-5)

    for angle, tol in (
        (0.1, 1e-6),
        (0.5, 1e-6),
        (1.0, 1e-6),
        (math.pi / 2.0, 1e-6),
        (math.pi - 1e-4, 5e-4),
    ):
        w = axis * float(angle)
        R = so3_exp_map(w)
        R_rt = so3_exp_map(so3_log_map(R))
        geo = float(geodesic_R(R_rt.unsqueeze(0), R.unsqueeze(0)).item())
        if geo > tol:
            raise SystemExit(
                f"[FAIL] exp(log(R)) geodesic error too large at angle={angle:.6f}: {geo:.6e} > {tol:.6e}"
            )


def _blend(R_inc: torch.Tensor, R_dir: torch.Tensor, lam: float) -> torch.Tensor:
    R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
    omega = so3_log_map(R_res)
    return torch.matmul(so3_exp_map(omega * lam), R_inc)


def _check_lambda_blend() -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    R_inc = torch.eye(3, dtype=torch.float32)
    R_dir = so3_exp_map(axis * 1.0)
    R_l0 = _blend(R_inc, R_dir, 0.0)
    R_l1 = _blend(R_inc, R_dir, 1.0)
    R_lh = _blend(R_inc, R_dir, 0.5)

    err_l0 = float(geodesic_R(R_l0.unsqueeze(0), R_inc.unsqueeze(0)).item())
    err_l1 = float(geodesic_R(R_l1.unsqueeze(0), R_dir.unsqueeze(0)).item())
    half = float(geodesic_R(R_lh.unsqueeze(0), R_inc.unsqueeze(0)).item())
    full = float(geodesic_R(R_dir.unsqueeze(0), R_inc.unsqueeze(0)).item())
    if err_l0 > 1e-6:
        raise SystemExit(f"[FAIL] lambda=0 blend mismatch: {err_l0:.6e}")
    if err_l1 > 1e-6:
        raise SystemExit(f"[FAIL] lambda=1 blend mismatch: {err_l1:.6e}")
    if abs(half - 0.5 * full) > 1e-6:
        raise SystemExit(f"[FAIL] lambda=0.5 residual angle mismatch: {half:.6e} vs {0.5 * full:.6e}")


def _check_angvel_semantics() -> None:
    fps = 60.0
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    omega = axis * 1.2
    delta_R = so3_exp_map(omega / fps)
    got = angvel_vec_from_delta_R(delta_R, fps=fps)
    _assert_close("angvel_vec_from_delta_R", got, omega, 1e-6)

    R0 = torch.eye(3, dtype=torch.float32)
    R1 = delta_R
    R_seq = torch.stack([R0, R1], dim=0).view(1, 2, 1, 3, 3)
    got_seq = angvel_vec_from_R_seq(R_seq, fps=fps)[0, 0, 0]
    _assert_close("angvel_vec_from_R_seq", got_seq, omega, 1e-6)


def _check_assets() -> None:
    spec_paths = [
        Path("raw_data/processed_data/norm_template.json"),
        Path("temp_pretrain_template.json"),
        Path("models/pretrain_template.json"),
    ]
    for path in spec_paths:
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing required spec asset: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require_standard_rotvec_spec(payload, context=str(path))

    tpl_temp = json.loads(Path("temp_pretrain_template.json").read_text(encoding="utf-8"))
    tpl_model = json.loads(Path("models/pretrain_template.json").read_text(encoding="utf-8"))
    if tpl_temp.get("tanh_scales_angvel") != tpl_model.get("tanh_scales_angvel"):
        raise SystemExit("[FAIL] temp_pretrain_template.json and models/pretrain_template.json are out of sync.")

    for path in (
        Path("models/motion_encoder_equiv_stageA.pt"),
        Path("models/motion_encoder_equiv.pt.best.pt"),
    ):
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing required encoder bundle: {path}")
        payload = torch.load(path, map_location="cpu")
        require_standard_rotvec_bundle(payload, context=str(path))


def main() -> None:
    _check_geometry_roundtrip()
    _check_lambda_blend()
    _check_angvel_semantics()
    _check_assets()
    print("ok: standard rotvec geometry, lambda blend, angvel semantics, and migrated assets verified")


if __name__ == "__main__":
    main()
