#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_cp015_oldplan_downstream_chain import (  # noqa: E402
    AFFINE_STATS,
    ENCODER_BUNDLE,
    PRETRAIN_CLAMP,
    create_replace_zerophase_warmstart,
)
from tools.run_cp015_tailk7_donor_hidden_dynamics_followup import (  # noqa: E402
    CONTROL_CKPT,
    DONOR_CKPT,
    TRAINABLE_MODULE_PATHS,
    _build_cfg_payload,
    _build_control_composite_ckpt,
    _build_runner,
    _capture_teacher_hfinal_trace,
    _enable_module_paths,
    _freeze_all,
    _save_ckpt,
    _select_trainable_params,
    _write_json,
)
from tools.run_tailk7_vs_baseline_leg_linear_probe import _extract_rot6d_columns  # noqa: E402
from train import posttrain  # noqa: E402


RUN_TAG = "20260404"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_donor_dynamics_gain_followup_{RUN_TAG}"
DEBUG_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_donor_dynamics_gain_followup_{RUN_TAG}"
LOG_FILE = DEBUG_ROOT / "lane.log"
SUMMARY_JSON = DEBUG_ROOT / "summary.json"
TRAIN_LOG_JSON = DEBUG_ROOT / "train_log.json"
TRAIN_CFG_JSON = DEBUG_ROOT / "train_config.json"
LANE_C_WARMSTART_REPORT = DEBUG_ROOT / "warmstart" / "replace_zerophase_report.json"

CURRENT_CONTROL_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "configs"
    / "posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.json"
)

TRANSITION_OFFSETS: Tuple[int, ...] = (1, 5, 20)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{ts}] {msg}"
    print(text, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _last_step_hidden(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[:, -1]
    if x.ndim == 2:
        return x
    raise RuntimeError(f"unexpected h_final tensor shape: {tuple(x.shape)}")


def _transition_term(
    free_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    *,
    cosine_weight: float,
    magnitude_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if tuple(free_delta.shape) != tuple(teacher_delta.shape):
        raise RuntimeError(
            f"delta shape mismatch: free={tuple(free_delta.shape)} teacher={tuple(teacher_delta.shape)}"
        )
    delta_diff = free_delta - teacher_delta
    mse = delta_diff.pow(2).mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(free_delta, teacher_delta, dim=-1, eps=1e-8)
    free_mag = free_delta.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
    teacher_mag = teacher_delta.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
    mag_abs = (free_mag - teacher_mag).abs()
    total = mse + (float(cosine_weight) * cosine) + (float(magnitude_weight) * mag_abs)
    stats = {
        "mse": mse,
        "cosine": cosine,
        "mag_abs": mag_abs,
        "free_mag": free_mag,
        "teacher_mag": teacher_mag,
        "delta_norm_l2": delta_diff.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt(),
    }
    return total, stats


def _hidden_transition_aux_loss(
    *,
    free_h: Sequence[torch.Tensor],
    teacher_h: Sequence[torch.Tensor],
    prep_ctx: Mapping[str, Any],
    focus_cycle_min: int,
    focus_sic_lo: int,
    focus_sic_hi: int,
    global_weight: float,
    cosine_weight: float,
    magnitude_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if len(free_h) != len(teacher_h):
        raise RuntimeError(f"h_final trace length mismatch: free={len(free_h)} teacher={len(teacher_h)}")
    if not free_h:
        raise RuntimeError("empty h_final trace")

    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    steps = int(prep_ctx["steps"])
    total_steps = int(prep_ctx["total_steps"])
    offset = int(prep_ctx["offset"])
    denom = cycle_len if include_boundary else steps

    global_losses: List[torch.Tensor] = []
    per_offset_focus: Dict[int, List[torch.Tensor]] = {int(span): [] for span in TRANSITION_OFFSETS}
    per_offset_delta_norm: Dict[int, List[torch.Tensor]] = {int(span): [] for span in TRANSITION_OFFSETS}
    per_offset_mag_gap: Dict[int, List[torch.Tensor]] = {int(span): [] for span in TRANSITION_OFFSETS}
    per_offset_free_mag: Dict[int, List[torch.Tensor]] = {int(span): [] for span in TRANSITION_OFFSETS}
    per_offset_teacher_mag: Dict[int, List[torch.Tensor]] = {int(span): [] for span in TRANSITION_OFFSETS}
    per_offset_counts: Dict[int, int] = {int(span): 0 for span in TRANSITION_OFFSETS}

    for span in TRANSITION_OFFSETS:
        span_losses_global: List[torch.Tensor] = []
        for base_t in range(max(0, total_steps - int(span))):
            free_delta = free_h[int(base_t) + int(span)] - free_h[int(base_t)]
            teacher_delta = teacher_h[int(base_t) + int(span)] - teacher_h[int(base_t)]
            term, aux = _transition_term(
                free_delta=free_delta,
                teacher_delta=teacher_delta.detach(),
                cosine_weight=float(cosine_weight),
                magnitude_weight=float(magnitude_weight),
            )
            span_losses_global.append(term.mean())
            step_in_cycle = int((offset + int(base_t)) % max(1, denom))
            cycle = int((offset + int(base_t)) // max(1, denom))
            if cycle < int(focus_cycle_min):
                continue
            if step_in_cycle < int(focus_sic_lo) or step_in_cycle > int(focus_sic_hi):
                continue
            per_offset_focus[int(span)].append(term.mean())
            per_offset_delta_norm[int(span)].append(aux["delta_norm_l2"].mean())
            per_offset_mag_gap[int(span)].append(aux["mag_abs"].mean())
            per_offset_free_mag[int(span)].append(aux["free_mag"].mean())
            per_offset_teacher_mag[int(span)].append(aux["teacher_mag"].mean())
            per_offset_counts[int(span)] += 1
        if span_losses_global:
            global_losses.append(torch.stack(span_losses_global).mean())

    if not global_losses:
        raise RuntimeError("no valid transition terms for global loss")

    focus_terms: List[torch.Tensor] = []
    global_term = torch.stack(global_losses).mean()
    stats: Dict[str, float] = {
        "htransition_aux_global_loss": float(global_term.detach().cpu()),
        "htransition_aux_global_weight": float(global_weight),
        "htransition_aux_cosine_weight": float(cosine_weight),
        "htransition_aux_magnitude_weight": float(magnitude_weight),
        "htransition_aux_focus_cycle_min": float(focus_cycle_min),
        "htransition_aux_focus_sic_lo": float(focus_sic_lo),
        "htransition_aux_focus_sic_hi": float(focus_sic_hi),
    }

    for span in TRANSITION_OFFSETS:
        if per_offset_focus[int(span)]:
            focus_loss = torch.stack(per_offset_focus[int(span)]).mean()
            delta_norm = torch.stack(per_offset_delta_norm[int(span)]).mean()
            mag_gap = torch.stack(per_offset_mag_gap[int(span)]).mean()
            free_mag = torch.stack(per_offset_free_mag[int(span)]).mean()
            teacher_mag = torch.stack(per_offset_teacher_mag[int(span)]).mean()
        else:
            focus_loss = global_term.new_tensor(0.0)
            nan_t = global_term.new_tensor(float("nan"))
            delta_norm = nan_t
            mag_gap = nan_t
            free_mag = nan_t
            teacher_mag = nan_t
        focus_terms.append(focus_loss)
        stats[f"htransition_aux_span{int(span)}_loss"] = float(focus_loss.detach().cpu())
        stats[f"htransition_aux_span{int(span)}_delta_norm_l2"] = float(delta_norm.detach().cpu())
        stats[f"htransition_aux_span{int(span)}_mag_abs"] = float(mag_gap.detach().cpu())
        stats[f"htransition_aux_span{int(span)}_free_mag"] = float(free_mag.detach().cpu())
        stats[f"htransition_aux_span{int(span)}_teacher_mag"] = float(teacher_mag.detach().cpu())
        stats[f"htransition_aux_span{int(span)}_samples"] = float(per_offset_counts[int(span)])

    local_term = torch.stack(focus_terms).mean()
    total = local_term + (float(global_weight) * global_term)
    stats["htransition_aux_focus_loss"] = float(local_term.detach().cpu())
    stats["htransition_aux_total"] = float(total.detach().cpu())
    return total, stats


def _build_runtime_and_loss(
    *,
    trainer: Any,
    model: torch.nn.Module,
    cfg: Any,
    batch: Mapping[str, Any],
    columns: Tuple[str, str],
    transition_aux_weight: float,
    focus_cycle_min: int,
    focus_sic_lo: int,
    focus_sic_hi: int,
    transition_global_weight: float,
    transition_cosine_weight: float,
    transition_magnitude_weight: float,
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
    transition_aux, transition_stats = _hidden_transition_aux_loss(
        free_h=free_h,
        teacher_h=teacher_h,
        prep_ctx=prep_ctx,
        focus_cycle_min=int(focus_cycle_min),
        focus_sic_lo=int(focus_sic_lo),
        focus_sic_hi=int(focus_sic_hi),
        global_weight=float(transition_global_weight),
        cosine_weight=float(transition_cosine_weight),
        magnitude_weight=float(transition_magnitude_weight),
    )
    total = base_loss + (float(transition_aux_weight) * transition_aux)
    stats["base_total"] = float(base_loss.detach().cpu())
    stats["donor_transition_aux_weight"] = float(transition_aux_weight)
    stats["donor_transition_aux_weighted"] = float((float(transition_aux_weight) * transition_aux).detach().cpu())
    stats.update(transition_stats)
    stats["total"] = float(total.detach().cpu())
    return total, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dynamics-focused donor continuation follow-up for cp015 tailk7.")
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--transition-aux-weight", type=float, default=2.0)
    ap.add_argument("--transition-global-weight", type=float, default=0.10)
    ap.add_argument("--transition-cosine-weight", type=float, default=0.25)
    ap.add_argument("--transition-magnitude-weight", type=float, default=0.50)
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

    donor_out_dir = MODEL_ROOT / "donor_transition_gain"
    donor_run_name = (
        "WalkF_stage7_70a_transition_gain_objective_"
        f"lr{str(args.lr).replace('.', 'p')}_e{int(args.epochs)}x{int(args.steps_per_epoch)}_{RUN_TAG}"
    )
    donor_ckpt_out = donor_out_dir / f"ckpt_last_{donor_run_name}.pth"

    lane_f_out_dir = MODEL_ROOT / "laneF_frozen_current_control"
    lane_f_run_name = (
        "WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_"
        f"frozen_current_control_on_transition_gain_{RUN_TAG}"
    )
    lane_f_ckpt_out = lane_f_out_dir / f"ckpt_last_{lane_f_run_name}.pth"

    lane_c_warmstart = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_transition_gain_replace_zerophase_{RUN_TAG}.pth"

    if (
        donor_ckpt_out.is_file()
        and lane_f_ckpt_out.is_file()
        and lane_c_warmstart.is_file()
        and SUMMARY_JSON.is_file()
        and not bool(args.force)
    ):
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
    payload["donor_dynamics_gain_followup"] = {
        "enabled": True,
        "trainable_module_paths": list(enabled_modules),
        "transition_aux_weight": float(args.transition_aux_weight),
        "transition_global_weight": float(args.transition_global_weight),
        "transition_cosine_weight": float(args.transition_cosine_weight),
        "transition_magnitude_weight": float(args.transition_magnitude_weight),
        "focus_cycle_min": int(args.focus_cycle_min),
        "focus_sic_lo": int(args.focus_sic_lo),
        "focus_sic_hi": int(args.focus_sic_hi),
        "transition_offsets": [int(x) for x in TRANSITION_OFFSETS],
        "base_objective": "direct_with_frozen_heads",
        "dynamics_target": "teacher-conditioned_vs_freerun_transition_span_consistency",
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
                transition_aux_weight=float(args.transition_aux_weight),
                focus_cycle_min=int(args.focus_cycle_min),
                focus_sic_lo=int(args.focus_sic_lo),
                focus_sic_hi=int(args.focus_sic_hi),
                transition_global_weight=float(args.transition_global_weight),
                transition_cosine_weight=float(args.transition_cosine_weight),
                transition_magnitude_weight=float(args.transition_magnitude_weight),
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
                    f" taux={stats.get('htransition_aux_total', float('nan')):.6f}"
                    f" d1={stats.get('htransition_aux_span1_delta_norm_l2', float('nan')):.6f}"
                    f" d5={stats.get('htransition_aux_span5_delta_norm_l2', float('nan')):.6f}"
                    f" d20={stats.get('htransition_aux_span20_delta_norm_l2', float('nan')):.6f}"
                    f" mag20={stats.get('htransition_aux_span20_mag_abs', float('nan')):.6f}"
                )
                log(msg)

        avg = epoch_loss / max(1, ok_steps)
        log(f"[epoch {epoch}] avg_total={avg:.6f} ok_steps={ok_steps} skipped={bad_steps}")

    _write_json(TRAIN_LOG_JSON, {"config": payload, "log": log_rows})
    donor_saved = _save_ckpt(model=model, cfg_payload=payload, out_dir=donor_out_dir, run_name=donor_run_name)
    lane_f_saved, lane_f_report = _build_control_composite_ckpt(
        donor_ckpt=donor_saved,
        control_ckpt=CONTROL_CKPT,
        out_dir=lane_f_out_dir,
        run_name=lane_f_run_name,
    )
    create_replace_zerophase_warmstart(donor_saved, lane_c_warmstart, LANE_C_WARMSTART_REPORT)
    warmstart_report = json.loads(LANE_C_WARMSTART_REPORT.read_text(encoding="utf-8"))

    lane_f_eval_dir = DEBUG_ROOT / "eval_model_source" / "laneF_frozen_current_control"
    lane_c_model_dir = MODEL_ROOT / "laneC_coadapt_posttrain"
    lane_c_eval_dir = DEBUG_ROOT / "eval_model_source" / "laneC_coadapt_posttrain"
    lane_c_run_name = (
        "WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_"
        f"coadapt_on_transition_gain_{RUN_TAG}"
    )
    lane_c_ckpt_out = lane_c_model_dir / f"ckpt_last_{lane_c_run_name}.pth"
    lane_f_group_json = DEBUG_ROOT / "eval_model_source" / "laneF_frozen_current_control_group_summary.json"
    lane_c_group_json = DEBUG_ROOT / "eval_model_source" / "laneC_coadapt_posttrain_group_summary.json"
    drift_summary_json = DEBUG_ROOT / "hfinal_drift_summary.json"

    commands = {
        "donor_continuation_and_laneF_assembly": (
            f"python3 tools/run_cp015_tailk7_donor_dynamics_gain_followup.py --device {args.device} --epochs {args.epochs} "
            f"--steps-per-epoch {args.steps_per_epoch} --lr {args.lr} --transition-aux-weight {args.transition_aux_weight} "
            f"--transition-global-weight {args.transition_global_weight} --transition-cosine-weight {args.transition_cosine_weight} "
            f"--transition-magnitude-weight {args.transition_magnitude_weight} --focus-cycle-min {args.focus_cycle_min} "
            f"--focus-sic-lo {args.focus_sic_lo} --focus-sic-hi {args.focus_sic_hi} --force"
        ),
        "laneF_eval": (
            "PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py "
            "-m train.validate.run_freerun_cycles "
            f"--teacher {ROOT / 'validate' / 'teacher_batches' / 'Walk_F_teacher.json'} "
            f"--model {lane_f_saved} "
            "--rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none "
            "--contacts_meas_source model --direct_pose_meas_source model --direct_pose_plan_source model "
            "--pose_hist_source buffer --pose_hist_update_source pred "
            "--lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series "
            f"--out {lane_f_eval_dir} --force"
        ),
        "laneF_group_summary": (
            f"python3 tools/phasea_group_summary.py {lane_f_eval_dir / 'Walk_F_freerun_cycles.json'} "
            f"--cycle_gte 1 --drop_wrap --out {lane_f_group_json}"
        ),
        "laneC_posttrain": (
            "PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py "
            "-m train.posttrain "
            f"--config {CURRENT_CONTROL_CONFIG} "
            f"--ckpt_in {lane_c_warmstart} "
            f"--out_dir {lane_c_model_dir} "
            f"--run_name {lane_c_run_name} "
            "--posttrain_contacts_source pretrain_contact "
            f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
            f"--encoder_bundle {ENCODER_BUNDLE} "
            f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
        ),
        "laneC_eval": (
            "PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py "
            "-m train.validate.run_freerun_cycles "
            f"--teacher {ROOT / 'validate' / 'teacher_batches' / 'Walk_F_teacher.json'} "
            f"--model {lane_c_ckpt_out} "
            "--rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none "
            "--contacts_meas_source model --lambda_fusion_apply --log_contacts "
            "--export_direct_arm_probe --export_joint_direct_geolocal_series "
            f"--out {lane_c_eval_dir} --force"
        ),
        "laneC_group_summary": (
            f"python3 tools/phasea_group_summary.py {lane_c_eval_dir / 'Walk_F_freerun_cycles.json'} "
            f"--cycle_gte 1 --drop_wrap --out {lane_c_group_json}"
        ),
        "drift_audit": (
            "python3 tools/analyze_cp015_tailk7_hfinal_drift.py "
            f"--ckpt {lane_f_saved} "
            f"--eval {lane_f_eval_dir / 'Walk_F_freerun_cycles.json'} "
            f"--teacher {ROOT / 'validate' / 'teacher_batches' / 'Walk_F_teacher.json'} "
            f"--device {args.device} --drop-wrap --out {drift_summary_json}"
        ),
    }

    summary = {
        "analysis": "cp015_tailk7_donor_dynamics_gain_followup",
        "run_tag": RUN_TAG,
        "inputs": {
            "donor_ckpt": str(DONOR_CKPT),
            "control_ckpt": str(CONTROL_CKPT),
            "current_control_config": str(CURRENT_CONTROL_CONFIG),
        },
        "training": {
            "device": str(args.device),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "base_objective": "direct_with_frozen_heads",
            "transition_aux_weight": float(args.transition_aux_weight),
            "transition_global_weight": float(args.transition_global_weight),
            "transition_cosine_weight": float(args.transition_cosine_weight),
            "transition_magnitude_weight": float(args.transition_magnitude_weight),
            "focus_cycle_min": int(args.focus_cycle_min),
            "focus_sic_lo": int(args.focus_sic_lo),
            "focus_sic_hi": int(args.focus_sic_hi),
            "transition_offsets": [int(x) for x in TRANSITION_OFFSETS],
            "trainable_module_paths": list(enabled_modules),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_names": trainable_names,
            "last_log_row": log_rows[-1] if log_rows else {},
        },
        "artifacts": {
            "train_config_json": str(TRAIN_CFG_JSON),
            "train_log_json": str(TRAIN_LOG_JSON),
            "donor_ckpt": str(donor_saved),
            "laneF_composite_ckpt": str(lane_f_saved),
            "laneC_warmstart_ckpt": str(lane_c_warmstart),
            "laneF_eval_dir": str(lane_f_eval_dir),
            "laneC_model_dir": str(lane_c_model_dir),
            "laneC_eval_dir": str(lane_c_eval_dir),
            "laneF_group_json": str(lane_f_group_json),
            "laneC_group_json": str(lane_c_group_json),
            "drift_summary_json": str(drift_summary_json),
        },
        "laneF_control_transplant": lane_f_report,
        "laneC_warmstart": warmstart_report,
        "commands": commands,
    }
    _write_json(SUMMARY_JSON, summary)
    print(SUMMARY_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
