#!/usr/bin/env python3
"""
Report magnitude/scale of pose vs SO(3) delta-rotation (velocity) losses on a teacher-forced rollout.

Workflow (two-step; keeps this tool model-agnostic):
  1) Generate teacher predictions:
       python -m train.validate.run_teacher_rollout \
         --model <CKPT.pth> \
         --teacher validate/teacher_batches/Walk_F_teacher.json \
         --bundle raw_data/processed_data/norm_template.json \
         --pretrain-template models/pretrain_template.json \
         --encoder-bundle models/motion_encoder_equiv_stageA.pt \
         --npz-root raw_data/processed_data \
         --out debug_output/rotvel_calib --force

  2) Report loss scales on the produced *_teacher_pred.json:
       python tools/report_rot_vel_loss_scale.py \
         --teacher-pred-json debug_output/rotvel_calib/Walk_F_teacher_pred.json \
         --bundle raw_data/processed_data/norm_template.json \
         --rot_vel_omega_min_deg_s 30

This prints a small JSON payload (easy to paste back into docs).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from train.data.layout import LayoutCenter
from train.models import MotionJointLoss


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _resolve_gt_pred(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pred = payload.get("prediction", {}) if isinstance(payload.get("prediction"), dict) else {}
    teacher = payload.get("teacher", {}) if isinstance(payload.get("teacher"), dict) else {}
    y_pred = _as_f32(pred.get("y_norm", []))
    y_gt = _as_f32(teacher.get("target_norm", []))
    if y_pred.ndim != 2 or y_gt.ndim != 2:
        raise ValueError(f"Invalid shapes: pred={y_pred.shape}, gt={y_gt.shape} (expected (T,D)).")
    T = min(int(y_pred.shape[0]), int(y_gt.shape[0]))
    if T < 2:
        raise ValueError(f"Need T>=2, got T={T}.")
    D = min(int(y_pred.shape[1]), int(y_gt.shape[1]))
    y_pred = y_pred[:T, :D]
    y_gt = y_gt[:T, :D]
    if y_pred.shape != y_gt.shape:
        raise ValueError(f"Shape mismatch after trim: pred={y_pred.shape}, gt={y_gt.shape}")
    return y_pred, y_gt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report relative magnitude of pose vs SO(3) delta-rotation (velocity) losses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--teacher-pred-json", type=str, required=True, help="Path to *_teacher_pred.json from run_teacher_rollout.")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json", help="Normalization bundle (norm_template.json).")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS used for omega gating conversion (defaults to bundle meta fps).")
    ap.add_argument("--rot_vel_log_scale", type=float, default=1.0, help="Scale applied to so3_log_map/_matrix_log_map (1.0 => standard axis-angle).")
    ap.add_argument("--rot_vel_omega_min_deg_s", type=float, default=0.0, help="If >0, gate rot-vel loss to |omega_gt| >= thr (deg/s).")
    ap.add_argument("--rot_vel_loss", type=str, default="smooth_l1", choices=("smooth_l1", "mse"))
    args = ap.parse_args()

    pred_path = Path(args.teacher_pred_json).expanduser()
    bundle_path = Path(args.bundle).expanduser()
    payload = _load_json(pred_path)
    y_pred, y_gt = _resolve_gt_pred(payload)

    dims = payload.get("dims", {}) if isinstance(payload.get("dims"), dict) else {}
    Dx = int(dims.get("Dx", 0) or 0)
    Dy = int(dims.get("Dy", y_pred.shape[1]) or y_pred.shape[1])
    if Dy != int(y_pred.shape[1]):
        Dy = int(y_pred.shape[1])

    center = LayoutCenter(str(bundle_path))
    if Dx > 0:
        center.strict_validate(Dx, Dy)
    else:
        # Fallback: validate only Y (still parse output layout).
        center.output_layout = center.output_layout or None

    fps = float(args.fps) if args.fps is not None else float(center.fps or 60.0)

    loss_fn = MotionJointLoss(
        output_layout=center.output_layout,
        fps=fps,
        rot6d_spec=center.rot6d_spec,
        meta=center.meta,
        w_rot_local=1.0,  # report raw (unweighted) via stats
        w_rot_vel=1.0,
        rot_vel_log_scale=float(args.rot_vel_log_scale),
        rot_vel_omega_min_deg_s=float(args.rot_vel_omega_min_deg_s),
        rot_vel_loss=str(args.rot_vel_loss),
    )
    loss_fn.mu_y = torch.as_tensor(center.mu_y, dtype=torch.float32).view(1, -1)
    loss_fn.std_y = torch.as_tensor(center.std_y, dtype=torch.float32).view(1, -1)

    with torch.no_grad():
        pred_t = torch.from_numpy(y_pred).unsqueeze(0)  # (1,T,Dy)
        gt_t = torch.from_numpy(y_gt).unsqueeze(0)
        total, stats = loss_fn(pred_t, gt_t)

    rot_geo = float(stats.get("rot_geo", float("nan")))
    rot_local = float(stats.get("rot_local", float("nan")))
    rot_local_deg = float(stats.get("rot_local_deg", float("nan")))
    rot_vel = float(stats.get("rot_vel", float("nan")))
    rot_vel_mask_frac = float(stats.get("rot_vel_mask_frac", float("nan")))
    omega_mean = float(stats.get("rot_vel_omega_gt_mean_deg_s", float("nan")))

    # Heuristic: choose lambda so weighted vel ~= local in magnitude.
    lam_eq = float("nan")
    if math.isfinite(rot_local) and math.isfinite(rot_vel) and rot_vel > 1e-12:
        lam_eq = float(rot_local / rot_vel)

    out = {
        "teacher_pred_json": str(pred_path),
        "bundle": str(bundle_path),
        "T": int(y_pred.shape[0]),
        "Dy": int(y_pred.shape[1]),
        "fps": float(fps),
        "rot_geo_rad": float(rot_geo),
        "rot_geo_deg": float(rot_geo * (180.0 / math.pi)) if math.isfinite(rot_geo) else float("nan"),
        "rot_local_rad": float(rot_local),
        "rot_local_deg": float(rot_local_deg),
        "rot_vel": float(rot_vel),
        "rot_vel_loss": str(args.rot_vel_loss),
        "rot_vel_log_scale": float(args.rot_vel_log_scale),
        "rot_vel_omega_min_deg_s": float(args.rot_vel_omega_min_deg_s),
        "rot_vel_mask_frac": float(rot_vel_mask_frac),
        "rot_vel_omega_gt_mean_deg_s": float(omega_mean),
        "lambda_eq__rot_local_over_rot_vel": float(lam_eq),
        "note": "lambda_eq is a rough scale-match heuristic; tune via sweep + freerun dt_frames/GeoLocal tail acceptance.",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
