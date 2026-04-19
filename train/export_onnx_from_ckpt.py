#!/usr/bin/env python3
"""
Re‑export step‑stateful (no phase) ONNX from a trained MPL checkpoint.

This is essentially a thin wrapper around:
    - MotionEventDataset  (to get Dx/Dy/Dc and example tensors)
    - EventMotionModel    (same architecture as training)
    - export_onnx_step_stateful_nophase (from training_MPL.py)

Use this when:
    - raw_data/processed_data/*.npz + norm_template.json 已更新（比如 Dy 从 276→278）
    - 但老的 ONNX 还是旧维度，需要用最新模板 + checkpoint 重新导出。

Example (adapt8_v6):
    python -m train.export_onnx_from_ckpt \\
        --ckpt models/MLPL2_uncertainty_adapt8_v6/exp_phase_MLPL2_uncertainty_adapt8_v6/ckpt_best_exp_phase_MLPL2_uncertainty_adapt8_v6.pth \\
        --bundle raw_data/processed_data/norm_template.json \\
        --data raw_data/processed_data \\
        --out-onnx models/MLPL2_uncertainty_adapt8_v6/exp_phase_MLPL2_uncertainty_adapt8_v6/exp_phase_MLPL2_uncertainty_adapt8_v6_step_stateful_nophase.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.data.dataset import MotionEventDataset, make_fixedlen_collate
from train.checkpoint.compat import (
    DirectPoseBuildOverrides,
    DirectPoseLoadCompatOptions,
    EventClockBuildOverrides,
    LambdaFusionBuildOverrides,
    load_event_motion_ckpt_payload,
    prepare_event_motion_ckpt_state_for_load,
    resolve_event_motion_build_state_from_ckpt,
)
from train.models import EventMotionModel
from train.training_MPL import export_onnx_step_stateful_nophase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-export step-stateful-nophase ONNX from a trained MPL checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path (.pth) with {'model': state_dict} (best or last).",
    )
    p.add_argument(
        "--bundle",
        type=str,
        default="raw_data/processed_data/norm_template.json",
        help="Normalization bundle (norm_template.json) used during training.",
    )
    p.add_argument(
        "--data",
        type=str,
        default="raw_data/processed_data",
        help="Directory with processed *.npz clips (same as training data root).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length for the example DataLoader batch (does not affect ONNX interface).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for the example DataLoader batch (only used for export example shapes).",
    )
    p.add_argument(
        "--out-onnx",
        type=str,
        required=True,
        help="Output ONNX path (will be overwritten if exists).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=512,
        help="Hidden width used during training (must match checkpoint).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help=(
            "Encoder depth used during training (must match checkpoint). "
            "depth<=2 uses the original MLP encoder; depth>2 enables a residual encoder "
            "(2-layer stem + (depth-2) residual blocks)."
        ),
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Attention head count used during training (must match checkpoint).",
    )
    p.add_argument(
        "--context-len",
        type=int,
        default=16,
        help="Context length hyperparameter (kept for completeness).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"[FATAL] checkpoint not found: {ckpt_path}")

    bundle_path = Path(args.bundle).expanduser().resolve()
    if not bundle_path.is_file():
        raise SystemExit(f"[FATAL] bundle (norm_template) not found: {bundle_path}")

    data_root = Path(args.data).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"[FATAL] data directory not found: {data_root}")

    # ---- Load norm_spec to get pose_hist_len (for dataset construction) ----
    with bundle_path.open("r", encoding="utf-8") as f:
        norm_spec = json.load(f)
    pose_hist_len = int(norm_spec.get("pose_hist_len", 0) or 0)

    # ---- Collect some *.npz files as training-like dataset ----
    npz_files: List[Path] = sorted(data_root.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"[FATAL] no *.npz found under {data_root}")

    # Use all npz for normal shape statistics; ONNX export only needs example batch.
    ds_train = MotionEventDataset(
        data_dir=str(data_root),
        seq_len=int(args.seq_len),
        paths=[str(p) for p in npz_files],
        pose_hist_len=pose_hist_len,
        norm_spec=norm_spec,
    )
    print(f"[ExportONNX] Dataset dims: Dx={ds_train.Dx} Dy={ds_train.Dy} Dc={ds_train.Dc}")

    collate_fn = make_fixedlen_collate(int(args.seq_len))
    loader = DataLoader(
        ds_train,
        batch_size=int(args.batch),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    Dx, Dy, Dc = int(ds_train.Dx), int(ds_train.Dy), int(ds_train.Dc)
    contact_dim = int(getattr(ds_train, "contact_dim", 0) or 0)
    angvel_dim = int(getattr(ds_train, "angvel_dim", 0) or 0)
    pose_hist_dim = int(getattr(ds_train, "pose_hist_dim", 0) or 0)
    period_dim = int(getattr(ds_train, "period_dim", 0) or 0)

    ckpt_payload = load_event_motion_ckpt_payload(str(ckpt_path), map_location="cpu")
    if ckpt_payload.stripped_frozen_key_count > 0:
        print(
            f"[ExportONNX][INFO] stripped {ckpt_payload.stripped_frozen_key_count} frozen encoder keys "
            "from checkpoint for runtime load."
        )
    build_state = resolve_event_motion_build_state_from_ckpt(
        ckpt_payload=ckpt_payload,
        in_state_dim=Dx,
        out_motion_dim=Dy,
        cond_dim=Dc,
        contact_dim=contact_dim,
        period_dim=period_dim,
        direct_pose_overrides=DirectPoseBuildOverrides(
            train_direct_pose=False,
            direct_pose_reinit=False,
        ),
        event_clock_overrides=EventClockBuildOverrides(
            mode="auto",
            max_delta=0.5,
            has_encoder_bundle=False,
        ),
        lambda_fusion_overrides=LambdaFusionBuildOverrides(
            train_lambda_head=False,
        ),
    )
    contact_plan_enable = bool(
        build_state.contact_plan_cfg.enable
        or build_state.contact_plan_cfg.inject != "none"
        or build_state.direct_pose_cfg.enable
    )

    # ---- Build model with same architecture as training ----
    model = EventMotionModel(
        in_state_dim=Dx,
        out_motion_dim=Dy,
        cond_dim=Dc,
        period_dim=int(build_state.event_clock_cfg.period_dim_init),
        hidden_dim=int(args.width),
        num_layers=int(args.depth),
        num_heads=int(args.num_heads),
        dropout=0.1,
        context_len=int(args.context_len),
        contact_dim=contact_dim,
        angvel_dim=angvel_dim,
        pose_hist_dim=pose_hist_dim,
        bone_names=getattr(ds_train, "bone_names", None),
        output_layout=getattr(ds_train, "output_layout", None),
        contact_plan_enable=bool(contact_plan_enable),
        contact_plan_hidden=int(build_state.contact_plan_cfg.hidden),
        contact_plan_dropout=0.0,
        contact_plan_inject=str(build_state.contact_plan_cfg.inject),
        contact_plan_inject_detach=True,
        contact_plan_time_pe_dim=int(build_state.contact_plan_cfg.time_pe_dim),
        contact_plan_init_mode=str(build_state.contact_plan_cfg.init_mode),
        contact_plan_init_hidden=int(build_state.contact_plan_cfg.init_hidden),
        contact_plan_init_dropout=float(build_state.contact_plan_cfg.init_dropout),
        use_event_clock=bool(build_state.event_clock_cfg.use_event_clock),
        event_clock_max_delta=float(build_state.event_clock_cfg.max_delta),
        event_clock_hidden_dim=int(build_state.event_clock_cfg.hidden_dim),
        event_clock_gate_hidden_dim=int(build_state.event_clock_cfg.gate_hidden_dim),
        direct_pose_enable=bool(build_state.direct_pose_cfg.enable),
        direct_pose_hidden=int(build_state.direct_pose_cfg.hidden),
        direct_pose_dropout=0.0,
        direct_pose_detach_plan=True,
        direct_pose_meas_mode=str(build_state.direct_pose_cfg.meas_mode),
        direct_pose_meas_drop_prob=0.0,
        direct_pose_meas_noise_std=0.0,
        direct_pose_plan_drop_prob=0.0,
        direct_pose_feat_source=str(build_state.direct_pose_cfg.feat_source),
        direct_pose_time_pe_dim=int(build_state.direct_pose_cfg.time_pe_dim),
        direct_pose_time_pe_base=float(build_state.direct_pose_cfg.time_pe_base),
        direct_pose_use_phase_z=bool(build_state.direct_pose_cfg.use_phase_z),
        direct_pose_phase_z_mode=str(build_state.direct_pose_cfg.phase_z_mode),
        direct_pose_split_enable=bool(build_state.direct_pose_cfg.split_enable),
        direct_pose_nonleg_proj_dim=int(build_state.direct_pose_cfg.nonleg_proj_dim),
        direct_pose_arm_split_enable=bool(build_state.direct_pose_cfg.arm_split_enable),
        direct_pose_arm_bones=build_state.direct_pose_cfg.arm_bones,
        lambda_fusion_enable=bool(build_state.lambda_fusion_cfg.enable),
        lambda_fusion_mode=str(build_state.lambda_fusion_cfg.mode),
        lambda_fusion_hidden=int(build_state.lambda_fusion_cfg.hidden),
        lambda_fusion_dropout=float(build_state.lambda_fusion_cfg.dropout),
        lambda_fusion_detach_err=True,
        lambda_fusion_logit_init=float(build_state.lambda_fusion_cfg.logit_init),
        lambda_fusion_use_rollout_step=bool(build_state.lambda_fusion_cfg.use_rollout_step),
    )

    # ---- Load checkpoint weights ----
    state_dict = prepare_event_motion_ckpt_state_for_load(
        state_dict=build_state.state_dict,
        model=model,
        ckpt_posttrain_cfg=build_state.ckpt_posttrain_cfg,
        contact_dim=contact_dim,
        direct_pose_cfg=build_state.direct_pose_cfg,
        load_options=DirectPoseLoadCompatOptions(
            train_direct_pose=False,
            leg_enable=False,
        ),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[ExportONNX][WARN] state_dict mismatch: missing={missing}, unexpected={unexpected}")

    # Sanity: log output head shape
    out_head = getattr(model, "out_head", None)
    if out_head is not None and hasattr(out_head, "weight"):
        print(f"[ExportONNX] out_head weight shape={tuple(out_head.weight.shape)} (Dy x H)")

    # ---- Export ONNX ----
    out_onnx = Path(args.out_onnx).expanduser().resolve()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ExportONNX] exporting to {out_onnx} ...")
    export_onnx_step_stateful_nophase(
        model.eval().cpu(),
        loader,
        str(out_onnx),
        opset=18,
        dynamic_batch=False,
    )
    print("[ExportONNX] done.")


if __name__ == "__main__":
    main()
