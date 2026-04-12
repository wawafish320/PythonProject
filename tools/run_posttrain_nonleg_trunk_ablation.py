#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import train.posttrain as posttrain


def _enable_trunk(model: nn.Module, mode: str) -> List[str]:
    mode = str(mode or "none").strip().lower()
    if mode in ("", "none", "off", "false", "0"):
        return []
    head = getattr(model, "direct_pose_head", None)
    if head is None:
        raise RuntimeError("model has no direct_pose_head")
    names: List[str] = []
    if mode == "full":
        for name, param in head.named_parameters():
            param.requires_grad_(True)
            names.append(f"direct_pose_head.{name}")
        return names
    if mode == "last":
        target = None
        prefix = "direct_pose_head"
        if isinstance(head, nn.Sequential) and len(head) > 3 and isinstance(head[3], nn.Linear):
            target = head[3]
            prefix = "direct_pose_head.3"
        elif isinstance(head, nn.Sequential):
            for idx in range(len(head) - 1, -1, -1):
                if isinstance(head[idx], nn.Linear):
                    target = head[idx]
                    prefix = f"direct_pose_head.{idx}"
                    break
        if target is None:
            raise RuntimeError("could not resolve last Linear in direct_pose_head")
        for name, param in target.named_parameters():
            param.requires_grad_(True)
            names.append(f"{prefix}.{name}")
        return names
    raise ValueError(f"unsupported trunk mode: {mode}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--trunk-mode", type=str, default="full", choices=("none", "last", "full"))
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--save-step-ckpts", type=str, default="0,1,5,20")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    payload = posttrain.load_json(Path(args.config).expanduser())
    if not isinstance(payload, dict):
        raise SystemExit("[FATAL] config payload must be a dict")
    payload = dict(payload)
    payload["out_dir"] = str(args.out_dir)
    payload["run_name"] = str(args.run_name)
    payload["epochs"] = int(args.epochs)
    payload["steps_per_epoch"] = int(args.steps_per_epoch)
    payload["save_step_ckpts"] = str(args.save_step_ckpts)

    cfg = posttrain._cfg_from_payload(payload)
    posttrain._set_seed(cfg.seed)
    device = posttrain._resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    norm_spec, ds, batch_iter = posttrain._build_dataset_and_loader(cfg)
    (
        model,
        direct_pose_feat_source,
        direct_pose_time_pe_dim,
        direct_pose_time_pe_base,
        direct_pose_use_phase_z,
        direct_pose_phase_z_mode,
        direct_pose_split_enable,
        direct_pose_nonleg_proj_dim,
        direct_pose_leg_gate_mode_model,
        direct_pose_leg_gate_power_model,
    ) = posttrain._build_posttrain_model_from_ckpt(
        cfg=cfg,
        ds=ds,
        device=device,
    )
    trainer = posttrain._build_model_and_trainer(cfg=cfg, ds=ds, model=model, norm_spec=norm_spec)
    train_mode = posttrain._resolve_train_mode(cfg)

    posttrain._freeze_all(model)
    posttrain._unfreeze_for_train_mode(model, cfg, train_mode)
    extra_names = _enable_trunk(model, str(args.trunk_mode))
    model.train()

    params, names = posttrain._select_trainable_params(model)
    if not params:
        raise SystemExit("[FATAL] no trainable params selected")
    print(f"[ablation] trunk_mode={args.trunk_mode} extra_trainable={len(extra_names)}")
    if extra_names:
        print(f"[ablation] extra={', '.join(extra_names[:8])}{' ...' if len(extra_names) > 8 else ''}")
    print(f"[ablation] total_trainable={len(params)}")

    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    columns = ("X", "Z")
    rot6d_spec = getattr(ds, "rot6d_spec", None)
    if isinstance(rot6d_spec, dict):
        cols = rot6d_spec.get("columns")
        if isinstance(cols, (list, tuple)) and len(cols) >= 2:
            a = str(cols[0]).strip().upper()
            b = str(cols[1]).strip().upper()
            if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                columns = (a, b)

    rollout_common_kwargs: Dict[str, Any] = {
        "trainer": trainer,
        "model": model,
        "columns": columns,
        "rollout_steps": cfg.rollout_steps,
        "rollout_cycles": cfg.rollout_cycles,
        "include_boundary": cfg.rollout_include_boundary,
        "boundary_weight": cfg.lambda_boundary_weight,
        "random_offset": cfg.rollout_random_offset,
        "time_index_mode": cfg.time_index_mode,
        "time_weight_max": cfg.lambda_time_weight_max,
        "time_weight_mode": cfg.lambda_time_weight_mode,
        "detach_rollout_state": cfg.detach_rollout_state,
        "contact_meas_weight": cfg.contact_meas_weight,
    }
    rollout_mode_kwargs = posttrain._build_rollout_mode_kwargs(cfg, train_mode)
    log_rows = posttrain._run_training_loop(
        cfg=cfg,
        train_mode=train_mode,
        model=model,
        params=params,
        opt=opt,
        batch_iter=batch_iter,
        rollout_common_kwargs=rollout_common_kwargs,
        rollout_mode_kwargs=rollout_mode_kwargs,
        l2sp_pairs=[],
        l2sp_weight=0.0,
    )
    ckpt_out = posttrain._save_posttrain_outputs(
        cfg=cfg,
        model=model,
        log_rows=log_rows,
        direct_pose_feat_source=direct_pose_feat_source,
        direct_pose_time_pe_dim=direct_pose_time_pe_dim,
        direct_pose_time_pe_base=direct_pose_time_pe_base,
        direct_pose_use_phase_z=direct_pose_use_phase_z,
        direct_pose_phase_z_mode=direct_pose_phase_z_mode,
        direct_pose_split_enable=direct_pose_split_enable,
        direct_pose_nonleg_proj_dim=direct_pose_nonleg_proj_dim,
        direct_pose_leg_gate_mode_model=direct_pose_leg_gate_mode_model,
        direct_pose_leg_gate_power_model=direct_pose_leg_gate_power_model,
    )
    print(f"[ablation][OK] saved: {ckpt_out}")


if __name__ == "__main__":
    main()
