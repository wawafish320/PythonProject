#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_tailk7_vs_baseline_leg_linear_probe import _extract_rot6d_columns, _make_runner_args, _resolve_device
from train import posttrain
from train.validate.run_freerun_cycles import FreeRunCycleRunner


RUN_TAG = "20260404"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_donor_hidden_dynamics_followup_{RUN_TAG}"
DEBUG_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_donor_hidden_dynamics_followup_{RUN_TAG}"
LOG_FILE = DEBUG_ROOT / "lane.log"
SUMMARY_JSON = DEBUG_ROOT / "summary.json"
TRAIN_LOG_JSON = DEBUG_ROOT / "train_log.json"
TRAIN_CFG_JSON = DEBUG_ROOT / "train_config.json"

DONOR_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
CONTROL_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "e3x60_adapter_factorized"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth"
)

TRAINABLE_MODULE_PATHS: Tuple[str, ...] = (
    "shared_encoder",
    "residual_proj",
    "_pasa_lnq",
    "_pasa_q",
    "_pasa_k",
    "_pasa_v",
    "_pasa_o",
    "_pasa_film",
    "coupling_norm",
)
FOCUS_OFFSETS: Tuple[int, ...] = (1, 5, 20)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{ts}] {msg}"
    print(text, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _resolve_path(value: Any, default: Optional[Path] = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("path value is required")
        return default
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _last_step_hidden(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[:, -1]
    if x.ndim == 2:
        return x
    raise RuntimeError(f"unexpected h_final tensor shape: {tuple(x.shape)}")


def _freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _enable_module_paths(model: torch.nn.Module, names: Sequence[str]) -> List[str]:
    enabled: List[str] = []
    for name in names:
        try:
            module = model.get_submodule(str(name))
        except Exception:
            module = None
        if module is None:
            continue
        has_params = False
        for param in module.parameters():
            param.requires_grad_(True)
            has_params = True
        if has_params:
            enabled.append(str(name))
    return enabled


def _select_trainable_params(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[str]]:
    params: List[torch.nn.Parameter] = []
    names: List[str] = []
    for name, param in model.named_parameters():
        if not bool(param.requires_grad):
            continue
        params.append(param)
        names.append(str(name))
    return params, names


def _build_runner(
    *,
    ckpt_path: Path,
    device_pref: str,
) -> Tuple[FreeRunCycleRunner, Any, Dict[str, Any]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    post_cfg = dict(obj.get("posttrain_cfg") or {})
    if not post_cfg:
        raise RuntimeError(f"checkpoint missing posttrain_cfg: {ckpt_path}")

    bundle_json = _resolve_path(post_cfg.get("bundle_json"), ROOT / "raw_data" / "processed_data" / "norm_template.json")
    pretrain_template = _resolve_path(post_cfg.get("pretrain_template"), ROOT / "models" / "pretrain_template.json")
    encoder_bundle = _resolve_path(post_cfg.get("encoder_bundle"), ROOT / "models" / "motion_encoder_equiv.pt.best.pt")

    runner_args = _make_runner_args(
        ckpt_path=ckpt_path,
        posttrain_cfg=post_cfg,
        bundle_json=bundle_json,
        encoder_bundle=encoder_bundle,
        pretrain_template=pretrain_template,
        device=_resolve_device(device_pref),
    )
    runner = FreeRunCycleRunner(runner_args)

    paths = post_cfg.get("paths", None)
    if isinstance(paths, (list, tuple)) and paths:
        npz_path = _resolve_path(paths[0])
    else:
        data_root = _resolve_path(post_cfg.get("data"), ROOT / "raw_data" / "processed_data")
        npz_path = (data_root / "Walk_F.npz").resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"training npz missing: {npz_path}")

    seq_len = int(post_cfg.get("seq_len", 87) or 87)
    ds = runner._build_dataset(npz_path, seq_len)
    runner._ensure_model_ready(ds)
    if runner.model is None or runner.trainer is None:
        raise RuntimeError("failed to reconstruct donor model/trainer")
    return runner, ds, post_cfg


def _build_cfg_payload(
    *,
    base_payload: Mapping[str, Any],
    out_dir: Path,
    run_name: str,
    device: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    seed: int,
) -> Dict[str, Any]:
    payload = dict(base_payload)
    payload["ckpt_in"] = str(DONOR_CKPT)
    payload["out_dir"] = str(out_dir)
    payload["run_name"] = str(run_name)
    payload["device"] = str(device)
    payload["lr"] = float(lr)
    payload["weight_decay"] = float(weight_decay)
    payload["epochs"] = int(epochs)
    payload["steps_per_epoch"] = int(steps_per_epoch)
    payload["seed"] = int(seed)
    return payload


def _rollout_time_base(batch: Mapping[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    base = batch.get("start", None)
    if torch.is_tensor(base):
        return base.to(device=device)
    return None


def _make_yaw_gt_fn(
    *,
    trainer: Any,
    prep_ctx: Mapping[str, Any],
) -> Any:
    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    y0_raw = prep_ctx["y0_raw"]
    gt_seq = prep_ctx["gt_seq"]

    def _yaw_gt(idx_step: int) -> Optional[torch.Tensor]:
        try:
            if include_boundary and y0_raw is not None and int(idx_step) == (cycle_len - 1):
                gt_raw_frame = y0_raw
            else:
                gt_idx = min(int(gt_seq.shape[1]) - 1, int(idx_step))
                gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
            return trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
        except Exception:
            return None

    return _yaw_gt


def _capture_teacher_hfinal_trace(
    *,
    trainer: Any,
    model: torch.nn.Module,
    prep_ctx: Mapping[str, Any],
    batch: Mapping[str, Any],
    time_index_mode: str,
    detach_rollout_state: bool,
) -> List[torch.Tensor]:
    state = copy.deepcopy(prep_ctx["state"])
    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    steps = int(prep_ctx["steps"])
    total_steps = int(prep_ctx["total_steps"])
    offset = int(prep_ctx["offset"])
    denom = cycle_len if include_boundary else steps
    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))
    time_base = _rollout_time_base(batch, prep_ctx["device"])
    yaw_gt_fn = _make_yaw_gt_fn(trainer=trainer, prep_ctx=prep_ctx)
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for t in range(total_steps):
            idx = int((offset + int(t)) % max(1, int(denom)))
            step_common = posttrain._rollout_step_common(
                trainer,
                model,
                state=state,
                t=int(t),
                idx=int(idx),
                total_steps=total_steps,
                cond_seq=prep_ctx["cond_seq"],
                cond_raw_tgt=prep_ctx["cond_raw_tgt"],
                cond_norm_mu=prep_ctx["cond_norm_mu"],
                cond_norm_std=prep_ctx["cond_norm_std"],
                angvel_seq=prep_ctx["angvel_seq"],
                pose_hist_seq=prep_ctx["pose_hist_seq"],
                time_index_mode=str(time_index_mode),
                time_base=time_base,
                enable_reprojection=enable_reprojection,
                include_boundary=include_boundary,
                cycle_len=cycle_len,
                yaw_gt_fn=yaw_gt_fn,
                detach_rollout_state=bool(detach_rollout_state),
                task_callback=None,
            )
            ret = step_common["ret"]
            h_final = ret.get("h_final", None)
            if not torch.is_tensor(h_final):
                raise RuntimeError("teacher rollout missing h_final")
            out.append(_last_step_hidden(h_final).detach())
            if include_boundary and prep_ctx["y0_raw"] is not None and int(idx) == (cycle_len - 1):
                gt_raw = prep_ctx["y0_raw"]
            else:
                gt_raw = trainer._denorm(prep_ctx["gt_seq"][:, idx])
            if int(t) < int(total_steps) - 1:
                posttrain._apply_rollout_carry_state(
                    trainer,
                    state,
                    y_next_raw=gt_raw.detach(),
                    cond_raw_step=step_common.get("cond_raw_step", None),
                )
    return out


def _hidden_dynamics_aux_loss(
    *,
    free_h: Sequence[torch.Tensor],
    teacher_h: Sequence[torch.Tensor],
    prep_ctx: Mapping[str, Any],
    focus_cycle_min: int,
    focus_sic_lo: int,
    focus_sic_hi: int,
    global_weight: float,
    cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if len(free_h) != len(teacher_h):
        raise RuntimeError(f"h_final trace length mismatch: free={len(free_h)} teacher={len(teacher_h)}")
    if not free_h:
        raise RuntimeError("empty h_final trace")

    per_step_terms: List[torch.Tensor] = []
    per_step_norm_l2: List[torch.Tensor] = []
    for free_t, teacher_t in zip(free_h, teacher_h):
        if tuple(free_t.shape) != tuple(teacher_t.shape):
            raise RuntimeError(f"h_final shape mismatch: {tuple(free_t.shape)} vs {tuple(teacher_t.shape)}")
        diff = free_t - teacher_t
        mse = diff.pow(2).mean(dim=-1)
        cos = 1.0 - F.cosine_similarity(free_t, teacher_t, dim=-1, eps=1e-8)
        per_step_terms.append(mse + (float(cosine_weight) * cos))
        per_step_norm_l2.append(diff.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt())

    global_term = torch.stack([term.mean() for term in per_step_terms]).mean()
    global_norm = torch.stack([term.mean() for term in per_step_norm_l2]).mean()

    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    steps = int(prep_ctx["steps"])
    total_steps = int(prep_ctx["total_steps"])
    offset = int(prep_ctx["offset"])
    denom = cycle_len if include_boundary else steps

    offset_losses: Dict[int, torch.Tensor] = {}
    offset_norms: Dict[int, torch.Tensor] = {}
    offset_counts: Dict[int, int] = {}
    for horizon in FOCUS_OFFSETS:
        horizon_terms: List[torch.Tensor] = []
        horizon_norms: List[torch.Tensor] = []
        count = 0
        for base_t in range(max(0, total_steps - int(horizon))):
            step_in_cycle = int((offset + int(base_t)) % max(1, denom))
            cycle = int((offset + int(base_t)) // max(1, denom))
            if cycle < int(focus_cycle_min):
                continue
            if step_in_cycle < int(focus_sic_lo) or step_in_cycle > int(focus_sic_hi):
                continue
            target_t = int(base_t) + int(horizon)
            horizon_terms.append(per_step_terms[target_t].mean())
            horizon_norms.append(per_step_norm_l2[target_t].mean())
            count += 1
        if horizon_terms:
            offset_losses[int(horizon)] = torch.stack(horizon_terms).mean()
            offset_norms[int(horizon)] = torch.stack(horizon_norms).mean()
        else:
            offset_losses[int(horizon)] = global_term.new_tensor(0.0)
            offset_norms[int(horizon)] = global_norm.new_tensor(float("nan"))
        offset_counts[int(horizon)] = int(count)

    local_term = torch.stack([offset_losses[h] for h in FOCUS_OFFSETS]).mean()
    total = local_term + (float(global_weight) * global_term)
    stats = {
        "hfinal_aux_loss": float(total.detach().cpu()),
        "hfinal_aux_global_loss": float(global_term.detach().cpu()),
        "hfinal_aux_global_norm_l2": float(global_norm.detach().cpu()),
        "hfinal_aux_global_weight": float(global_weight),
        "hfinal_aux_cosine_weight": float(cosine_weight),
        "hfinal_aux_focus_cycle_min": float(focus_cycle_min),
        "hfinal_aux_focus_sic_lo": float(focus_sic_lo),
        "hfinal_aux_focus_sic_hi": float(focus_sic_hi),
    }
    for horizon in FOCUS_OFFSETS:
        stats[f"hfinal_aux_offset{int(horizon)}_loss"] = float(offset_losses[horizon].detach().cpu())
        stats[f"hfinal_aux_offset{int(horizon)}_norm_l2"] = float(offset_norms[horizon].detach().cpu())
        stats[f"hfinal_aux_offset{int(horizon)}_samples"] = float(offset_counts[horizon])
    return total, stats


def _build_runtime_and_loss(
    *,
    trainer: Any,
    model: torch.nn.Module,
    cfg: Any,
    batch: Mapping[str, Any],
    columns: Tuple[str, str],
    hidden_aux_weight: float,
    focus_cycle_min: int,
    focus_sic_lo: int,
    focus_sic_hi: int,
    hidden_global_weight: float,
    hidden_cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rollout_mode_kwargs = posttrain._build_rollout_mode_kwargs(cfg, "direct")
    prep_ctx = posttrain._lambda_rollout_prepare_context(
        trainer,
        model,
        batch,
        columns=columns,
        rollout_steps=int(cfg.rollout_steps),
        rollout_cycles=int(cfg.rollout_cycles),
        include_boundary=bool(cfg.rollout_include_boundary),
        boundary_weight=float(getattr(cfg, "lambda_boundary_weight", 0.0) or 0.0),
        random_offset=bool(cfg.rollout_random_offset),
        time_weight_mode=str(getattr(cfg, "lambda_time_weight_mode", "inv") or "inv"),
        time_weight_max=float(getattr(cfg, "lambda_time_weight_max", 2.0) or 2.0),
    )
    reg_ctx = posttrain._lambda_rollout_build_reg_params(
        trainer,
        objective="direct",
        lambda_gate_sup_weight=float(getattr(cfg, "lambda_gate_sup_weight", 0.0) or 0.0),
        lambda_gate_sup_start_step=int(getattr(cfg, "lambda_gate_sup_start_step", -1) or -1),
        lambda_gate_sup_tau_deg=float(getattr(cfg, "lambda_gate_sup_tau_deg", 2.5) or 2.5),
        lambda_gate_sup_margin_deg=float(getattr(cfg, "lambda_gate_sup_margin_deg", 1.0) or 1.0),
        direct_pose_loss_group_norm_enable=bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
        direct_pose_loss_group_norm_w_leg=float(getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0) or 1.0),
        direct_pose_loss_group_norm_w_nonleg=float(
            getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0) or 1.0
        ),
        direct_pose_loss_group_norm_ema_beta=float(getattr(cfg, "direct_pose_loss_group_norm_ema_beta", 0.95) or 0.95),
        direct_pose_loss_group_norm_ratio_min=float(getattr(cfg, "direct_pose_loss_group_norm_ratio_min", 0.2) or 0.2),
        direct_pose_loss_group_norm_ratio_max=float(getattr(cfg, "direct_pose_loss_group_norm_ratio_max", 5.0) or 5.0),
        direct_pose_loss_group_norm_eps=float(getattr(cfg, "direct_pose_loss_group_norm_eps", 1e-6) or 1e-6),
        direct_pose_loss_3way_enable=bool(getattr(cfg, "direct_pose_loss_3way_enable", False)),
        direct_pose_loss_3way_w_leg=float(getattr(cfg, "direct_pose_loss_3way_w_leg", 1.0) or 1.0),
        direct_pose_loss_3way_w_arm=float(getattr(cfg, "direct_pose_loss_3way_w_arm", 1.0) or 1.0),
        direct_pose_loss_3way_w_else=float(getattr(cfg, "direct_pose_loss_3way_w_else", 1.0) or 1.0),
        direct_pose_loss_arm_else_balance_enable=bool(getattr(cfg, "direct_pose_loss_arm_else_balance_enable", False)),
        direct_pose_loss_arm_weight=float(getattr(cfg, "direct_pose_loss_arm_weight", 1.0) or 1.0),
        direct_pose_loss_else_weight=float(getattr(cfg, "direct_pose_loss_else_weight", 1.0) or 1.0),
    )
    nonleg_focus_ctx = posttrain._lambda_rollout_resolve_nonleg_focus(
        trainer,
        objective="direct",
        direct_pose_nonleg_focus_bones=str(getattr(cfg, "direct_pose_nonleg_focus_bones", "") or ""),
        direct_pose_nonleg_focus_weight=float(getattr(cfg, "direct_pose_nonleg_focus_weight", 1.0) or 1.0),
        J=int(prep_ctx["J"]),
        device=prep_ctx["device"],
    )
    weights_ctx = {
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "direct_pose_leg_align_weight": float(rollout_mode_kwargs["direct_pose_leg_align_weight"]),
        "direct_pose_leg_align_oracle_min_deg": float(rollout_mode_kwargs["direct_pose_leg_align_oracle_min_deg"]),
        "direct_pose_leg_align_oracle_weight_deg": float(
            rollout_mode_kwargs["direct_pose_leg_align_oracle_weight_deg"]
        ),
        "direct_pose_leg_align_mode": str(rollout_mode_kwargs["direct_pose_leg_align_mode"]),
        "direct_pose_leg_align_mag_weight": float(rollout_mode_kwargs["direct_pose_leg_align_mag_weight"]),
        "direct_pose_leg_align_res_weight": float(rollout_mode_kwargs["direct_pose_leg_align_res_weight"]),
        "direct_pose_leg_align_sign_weight": float(rollout_mode_kwargs["direct_pose_leg_align_sign_weight"]),
        "direct_pose_leg_align_cos_thresh": float(rollout_mode_kwargs["direct_pose_leg_align_cos_thresh"]),
        "direct_pose_leg_align_target_joints": rollout_mode_kwargs["direct_pose_leg_align_target_joints"],
        "direct_pose_leg_align_anchor_joints": rollout_mode_kwargs["direct_pose_leg_align_anchor_joints"],
        "direct_pose_leg_align_anchor_weight": float(rollout_mode_kwargs["direct_pose_leg_align_anchor_weight"]),
        "direct_pose_leg_gate_sup_weight": float(rollout_mode_kwargs["direct_pose_leg_gate_sup_weight"]),
        "direct_pose_loss_leg_split": bool(rollout_mode_kwargs["direct_pose_loss_leg_split"]),
        "direct_nonleg_focus_mask_j": nonleg_focus_ctx["direct_nonleg_focus_mask_j"],
        "direct_nonleg_focus_resolved": int(nonleg_focus_ctx["direct_nonleg_focus_resolved"]),
        "direct_nonleg_focus_weight_use": float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"]),
        "direct_pose_loss_3way_enable": bool(rollout_mode_kwargs["direct_pose_loss_3way_enable"]),
        "direct_pose_loss_3way_w_leg": float(rollout_mode_kwargs["direct_pose_loss_3way_w_leg"]),
        "direct_pose_loss_3way_w_arm": float(rollout_mode_kwargs["direct_pose_loss_3way_w_arm"]),
        "direct_pose_loss_3way_w_else": float(rollout_mode_kwargs["direct_pose_loss_3way_w_else"]),
        "direct_pose_loss_arm_else_balance_enable": bool(
            rollout_mode_kwargs["direct_pose_loss_arm_else_balance_enable"]
        ),
        "direct_pose_loss_arm_weight": float(rollout_mode_kwargs["direct_pose_loss_arm_weight"]),
        "direct_pose_loss_else_weight": float(rollout_mode_kwargs["direct_pose_loss_else_weight"]),
        "gate_sup_weight": float(reg_ctx["gate_sup_weight"]),
        "gate_sup_start": int(reg_ctx["gate_sup_start"]),
        "tau_rad": float(reg_ctx["tau_rad"]),
        "margin_rad": float(reg_ctx["margin_rad"]),
        "lambda_plan_entropy_weight": 0.0,
        "lambda_plan_dyn_weight": 0.0,
        "lambda_early_weight": 0.0,
        "lambda_early_steps": 0,
        "lambda_entropy_weight": 0.0,
        "lambda_smooth_weight": 0.0,
        "lambda_monotonic_weight": 0.0,
    }
    accum_ctx = posttrain._lambda_fusion_init_accum_ctx()
    state_vars = {
        "meas_used_logits": False,
        "direct_nonleg_focus_applied": float(nonleg_focus_ctx["direct_nonleg_focus_applied"]),
        "lam_prev": None,
        "lam_prev_monot": None,
        "plan_prev": None,
    }
    runtime_ctx = {
        "trainer": trainer,
        "model": model,
        "batch": batch,
        "prep_ctx": prep_ctx,
        "time_index_mode": str(getattr(cfg, "time_index_mode", "auto") or "auto"),
        "enable_reprojection": bool(getattr(trainer, "enable_cond_reprojection", True)),
        "detach_rollout_state": bool(cfg.detach_rollout_state),
        "columns": columns,
        "objective": "direct",
    }

    free_h: List[torch.Tensor] = []

    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if torch.is_tensor(output):
            free_h.append(_last_step_hidden(output))
        return output

    handle = model.coupling_norm.register_forward_hook(_hook)
    try:
        meas_used_logits, direct_nonleg_focus_applied = posttrain._lambda_fusion_run_unroll(
            runtime_ctx=runtime_ctx,
            weights_ctx=weights_ctx,
            accum_ctx=accum_ctx,
            state_vars=state_vars,
        )
    finally:
        handle.remove()
    if len(free_h) != int(prep_ctx["total_steps"]):
        raise RuntimeError(
            f"freerun h_final trace length mismatch: hook={len(free_h)} total_steps={int(prep_ctx['total_steps'])}"
        )

    finalize_ctx = {
        "trainer": trainer,
        "model": model,
        "objective": "direct",
        "direct_pose_leg_gate_sup_weight": float(rollout_mode_kwargs["direct_pose_leg_gate_sup_weight"]),
        "direct_pose_leg_align_weight": float(rollout_mode_kwargs["direct_pose_leg_align_weight"]),
        "direct_pose_leg_align_anchor_weight": float(rollout_mode_kwargs["direct_pose_leg_align_anchor_weight"]),
        "lambda_entropy_weight": 0.0,
        "lambda_smooth_weight": 0.0,
        "lambda_early_weight": 0.0,
        "lambda_monotonic_weight": 0.0,
        "lambda_plan_entropy_weight": 0.0,
        "lambda_plan_dyn_weight": 0.0,
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "include_boundary": bool(prep_ctx["include_boundary"]),
        "random_offset": bool(cfg.rollout_random_offset),
        "offset": int(prep_ctx["offset"]),
        "boundary_weight": float(getattr(cfg, "lambda_boundary_weight", 0.0) or 0.0),
        "boundary_steps": int(prep_ctx["boundary_steps"]),
        "boundary_weighted_sum": float(prep_ctx["boundary_weighted_sum"]),
        "direct_nonleg_focus_requested": int(nonleg_focus_ctx["direct_nonleg_focus_requested"]),
        "direct_nonleg_focus_resolved": int(nonleg_focus_ctx["direct_nonleg_focus_resolved"]),
        "direct_nonleg_focus_weight_use": float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"]),
        "direct_nonleg_focus_applied": float(direct_nonleg_focus_applied),
        "meas_used_logits": bool(meas_used_logits),
        **reg_ctx,
    }
    base_loss, stats, _aux_payload = posttrain._lambda_fusion_finalize(finalize_ctx=finalize_ctx, accum_ctx=accum_ctx)

    teacher_h = _capture_teacher_hfinal_trace(
        trainer=trainer,
        model=model,
        prep_ctx=prep_ctx,
        batch=batch,
        time_index_mode=str(getattr(cfg, "time_index_mode", "auto") or "auto"),
        detach_rollout_state=bool(cfg.detach_rollout_state),
    )
    hidden_aux, hidden_stats = _hidden_dynamics_aux_loss(
        free_h=free_h,
        teacher_h=teacher_h,
        prep_ctx=prep_ctx,
        focus_cycle_min=int(focus_cycle_min),
        focus_sic_lo=int(focus_sic_lo),
        focus_sic_hi=int(focus_sic_hi),
        global_weight=float(hidden_global_weight),
        cosine_weight=float(hidden_cosine_weight),
    )
    total = base_loss + (float(hidden_aux_weight) * hidden_aux)
    stats["base_total"] = float(base_loss.detach().cpu())
    stats["donor_hidden_aux_weight"] = float(hidden_aux_weight)
    stats["donor_hidden_aux_weighted"] = float((float(hidden_aux_weight) * hidden_aux).detach().cpu())
    stats.update(hidden_stats)
    stats["total"] = float(total.detach().cpu())
    return total, stats


def _save_ckpt(
    *,
    model: torch.nn.Module,
    cfg_payload: Mapping[str, Any],
    out_dir: Path,
    run_name: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    torch.save({"model": model.state_dict(), "posttrain_cfg": dict(cfg_payload)}, ckpt_out)
    return ckpt_out


def _build_control_composite_ckpt(
    *,
    donor_ckpt: Path,
    control_ckpt: Path,
    out_dir: Path,
    run_name: str,
) -> Tuple[Path, Dict[str, Any]]:
    donor_obj = torch.load(donor_ckpt, map_location="cpu")
    control_obj = torch.load(control_ckpt, map_location="cpu")
    donor_state = dict(donor_obj.get("model") or {})
    control_state = dict(control_obj.get("model") or {})
    merged_state = dict(donor_state)

    copied_keys: List[str] = []
    for key, value in control_state.items():
        if not str(key).startswith("direct_pose_"):
            continue
        merged_state[str(key)] = value
        copied_keys.append(str(key))

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    torch.save(
        {
            "model": merged_state,
            "posttrain_cfg": dict(control_obj.get("posttrain_cfg") or {}),
        },
        ckpt_out,
    )
    report = {
        "source_donor_ckpt": str(donor_ckpt),
        "source_control_ckpt": str(control_ckpt),
        "output_ckpt": str(ckpt_out),
        "copied_prefixes": ["direct_pose_*"],
        "copied_key_count": int(len(copied_keys)),
        "copied_keys": copied_keys,
    }
    return ckpt_out, report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Donor-only h_final hidden-dynamics follow-up for cp015 tailk7.")
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden-aux-weight", type=float, default=1.0)
    ap.add_argument("--hidden-global-weight", type=float, default=0.15)
    ap.add_argument("--hidden-cosine-weight", type=float, default=0.25)
    ap.add_argument("--focus-cycle-min", type=int, default=1)
    ap.add_argument("--focus-sic-lo", type=int, default=11)
    ap.add_argument("--focus-sic-hi", type=int, default=43)
    ap.add_argument("--seed", type=int, default=20260404)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    posttrain._set_seed(int(args.seed))

    donor_out_dir = MODEL_ROOT / "donor_hfinal_trunk_anchor"
    donor_run_name = (
        "WalkF_stage7_70a_hfinal_dynamics_trunk_anchor_"
        f"lr{str(args.lr).replace('.', 'p')}_e{int(args.epochs)}x{int(args.steps_per_epoch)}_{RUN_TAG}"
    )
    donor_ckpt_out = donor_out_dir / f"ckpt_last_{donor_run_name}.pth"
    composite_out_dir = MODEL_ROOT / "e3x60_adapter_factorized_control_on_donor_hfinal"
    composite_run_name = (
        "WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_control_on_donor_hfinal_"
        f"{RUN_TAG}"
    )
    composite_ckpt_out = composite_out_dir / f"ckpt_last_{composite_run_name}.pth"

    if donor_ckpt_out.is_file() and composite_ckpt_out.is_file() and SUMMARY_JSON.is_file() and not bool(args.force):
        print(SUMMARY_JSON)
        return 0

    runner, ds, base_post_cfg = _build_runner(ckpt_path=DONOR_CKPT, device_pref=str(args.device))
    trainer = runner.trainer
    model = runner.model
    if trainer is None or model is None:
        raise RuntimeError("runner missing trainer/model")
    trainer.device = runner.device
    trainer.lambda_reliability_mode = str(base_post_cfg.get("lambda_reliability_mode", "none") or "none")
    trainer.lambda_reliability_warmup_steps = int(base_post_cfg.get("lambda_reliability_warmup_steps", 0) or 0)
    trainer.lambda_reliability_contact_err_max = float(
        base_post_cfg.get("lambda_reliability_contact_err_max", 1.0) or 1.0
    )
    trainer.lambda_reliability_warmup_joint_scales = base_post_cfg.get("lambda_reliability_warmup_joint_scales", None)

    _freeze_all(model)
    enabled_modules = _enable_module_paths(model, TRAINABLE_MODULE_PATHS)
    params, trainable_names = _select_trainable_params(model)
    if not params:
        raise RuntimeError("no donor trunk params enabled for training")
    trainable_param_count = int(sum(int(p.numel()) for p in params))

    payload = _build_cfg_payload(
        base_payload=base_post_cfg,
        out_dir=donor_out_dir,
        run_name=donor_run_name,
        device=str(args.device),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        seed=int(args.seed),
    )
    payload["donor_hidden_dynamics_followup"] = {
        "enabled": True,
        "trainable_module_paths": list(enabled_modules),
        "hidden_aux_weight": float(args.hidden_aux_weight),
        "hidden_global_weight": float(args.hidden_global_weight),
        "hidden_cosine_weight": float(args.hidden_cosine_weight),
        "focus_cycle_min": int(args.focus_cycle_min),
        "focus_sic_lo": int(args.focus_sic_lo),
        "focus_sic_hi": int(args.focus_sic_hi),
        "focus_offsets": [int(x) for x in FOCUS_OFFSETS],
        "base_objective": "direct_with_frozen_heads",
    }
    cfg = posttrain._cfg_from_payload(payload)
    _write_json(TRAIN_CFG_JSON, payload)

    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    batch_iter = posttrain._iter_infinite(loader)
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    columns = tuple(str(x) for x in _extract_rot6d_columns(trainer))

    log_rows: List[Dict[str, float]] = []
    global_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        epoch_loss = 0.0
        ok_steps = 0
        bad_steps = 0
        for it in range(int(args.steps_per_epoch)):
            batch = next(batch_iter)
            opt.zero_grad(set_to_none=True)
            loss, stats = _build_runtime_and_loss(
                trainer=trainer,
                model=model,
                cfg=cfg,
                batch=batch,
                columns=columns,
                hidden_aux_weight=float(args.hidden_aux_weight),
                focus_cycle_min=int(args.focus_cycle_min),
                focus_sic_lo=int(args.focus_sic_lo),
                focus_sic_hi=int(args.focus_sic_hi),
                hidden_global_weight=float(args.hidden_global_weight),
                hidden_cosine_weight=float(args.hidden_cosine_weight),
            )
            if not bool(torch.isfinite(loss)):
                bad_steps += 1
                global_step += 1
                log(f"[WARN] non-finite loss at step={global_step}; skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            ok_steps += 1
            epoch_loss += float(loss.detach().cpu())
            global_step += 1
            row = dict(stats)
            row["epoch"] = float(epoch)
            row["iter"] = float(it)
            row["step"] = float(global_step)
            log_rows.append(row)

            if int(args.log_every) > 0 and ((it % int(args.log_every)) == 0 or it == int(args.steps_per_epoch) - 1):
                msg = posttrain._format_posttrain_step_msg(
                    train_mode="direct",
                    cfg=cfg,
                    stats=stats,
                    epoch=epoch,
                    it=it,
                    steps_per_epoch=int(args.steps_per_epoch),
                    l2sp_weight=0.0,
                )
                msg += (
                    f" haux={stats.get('hfinal_aux_loss', float('nan')):.6f}"
                    f" off1={stats.get('hfinal_aux_offset1_norm_l2', float('nan')):.6f}"
                    f" off5={stats.get('hfinal_aux_offset5_norm_l2', float('nan')):.6f}"
                    f" off20={stats.get('hfinal_aux_offset20_norm_l2', float('nan')):.6f}"
                )
                log(msg)

        avg = epoch_loss / max(1, ok_steps)
        log(f"[epoch {epoch}] avg_total={avg:.6f} ok_steps={ok_steps} skipped={bad_steps}")

    _write_json(TRAIN_LOG_JSON, {"config": payload, "log": log_rows})
    donor_saved = _save_ckpt(model=model, cfg_payload=payload, out_dir=donor_out_dir, run_name=donor_run_name)
    composite_saved, composite_report = _build_control_composite_ckpt(
        donor_ckpt=donor_saved,
        control_ckpt=CONTROL_CKPT,
        out_dir=composite_out_dir,
        run_name=composite_run_name,
    )

    summary = {
        "analysis": "cp015_tailk7_donor_hidden_dynamics_followup",
        "run_tag": RUN_TAG,
        "inputs": {
            "donor_ckpt": str(DONOR_CKPT),
            "control_ckpt": str(CONTROL_CKPT),
        },
        "training": {
            "device": str(args.device),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "base_objective": "direct_with_frozen_heads",
            "hidden_aux_weight": float(args.hidden_aux_weight),
            "hidden_global_weight": float(args.hidden_global_weight),
            "hidden_cosine_weight": float(args.hidden_cosine_weight),
            "focus_cycle_min": int(args.focus_cycle_min),
            "focus_sic_lo": int(args.focus_sic_lo),
            "focus_sic_hi": int(args.focus_sic_hi),
            "focus_offsets": [int(x) for x in FOCUS_OFFSETS],
            "trainable_module_paths": list(enabled_modules),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_names": trainable_names,
            "last_log_row": log_rows[-1] if log_rows else {},
        },
        "artifacts": {
            "train_config_json": str(TRAIN_CFG_JSON),
            "train_log_json": str(TRAIN_LOG_JSON),
            "donor_ckpt": str(donor_saved),
            "control_composite_ckpt": str(composite_saved),
        },
        "control_transplant": composite_report,
    }
    _write_json(SUMMARY_JSON, summary)
    print(SUMMARY_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
