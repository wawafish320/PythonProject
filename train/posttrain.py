#!/usr/bin/env python3
"""
Post-training utilities for MPL checkpoints.

Initial targets:
  - Freeze the base model.
  - Fine-tune the SO(3) delta-corrector head (omega_hat) to reduce free-run drift.
  - Stage2: freeze both experts (incremental + direct) and train lambda_fusion_head only.

This script is intentionally kept separate from the main training entry to keep the
primary training pipeline minimal, and to allow game-specific rollout simulation
later (e.g., action switches / resets).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train.configuration.io import dump_json, load_json
from train.data.dataset import (
    MotionEventDataset,
    build_and_attach_dataset_runtime,
    build_motion_dataloader,
    build_motion_dataset,
)
from train.geometry import (
    geodesic_R_safe as _geodesic_R_safe,
    matrix_to_rot6d,
    normalize_rot6d_delta,
    reproject_rot6d,
    rot6d_to_matrix,
    so3_exp_map,
    so3_log_map,
)
from train import rollout_kernel as _rollout_kernel
from train import posttrain_build_shell as _posttrain_build_shell
from train.data.io import config_to_jsonable as _cfg_to_jsonable
from train.configuration.norm_spec import (
    ContactPretrainRuntime,
    merge_norm_spec,
    resolve_contact_pretrain_runtime,
)
from train.models import EventMotionModel, MotionJointLoss
from train.runtime_attach import (
    apply_contacts_pretrain_runtime,
    apply_loss_runtime_from_trainer,
    apply_shared_trainer_runtime,
    resolve_shared_trainer_runtime,
)
from train.runtime.freeze import (
    _freeze_all,
    _select_trainable_params,
    _unfreeze_for_train_mode,
)
from train.training_MPL import Trainer
from train.utils import (
    apply_cli_overrides as _apply_cli_overrides_shared,
    as_bool as _as_bool,
    as_float_list as _as_float_list,
    as_path as _as_path,
    cfg_from_schema as _cfg_from_schema,
    cfg_get_bool as _cfg_get_bool,
    cfg_get_enum as _cfg_get_enum,
    cfg_get_float as _cfg_get_float,
    cfg_get_float_or as _cfg_get_float_or,
    cfg_get_float_present as _cfg_get_float_present,
    cfg_get_int as _cfg_get_int,
    cfg_get_int_or as _cfg_get_int_or,
    cfg_get_int_present as _cfg_get_int_present,
    cfg_get_or as _cfg_get_or,
    cfg_pick as _cfg_pick,
    cfg_get_str_or as _cfg_get_str_or,
    clamp_float as _clamp_float,
    grad_list_cosine as _grad_list_cosine,
    grad_list_norm as _grad_list_norm,
    grad_norm_of_module as _grad_norm_of_module,
    iter_infinite as _iter_infinite,
    normalize_optional_csv as _normalize_optional_csv,
    parse_int_set_spec as _parse_int_set_spec,
    resolve_device as _resolve_device,
)


_LEG_ALIGN_DISTAL_JOINTS: tuple[str, ...] = (
    "foot_r",
    "ball_r",
    "foot_l",
    "ball_l",
)

_LEG_ALIGN_PROXIMAL_JOINTS: tuple[str, ...] = (
    "thigh_r",
    "calf_r",
    "thigh_l",
    "calf_l",
)

_LEG_ALIGN_FOOT_JOINTS: tuple[str, ...] = (
    "foot_r",
    "foot_l",
)

_LEG_ALIGN_BALL_JOINTS: tuple[str, ...] = (
    "ball_r",
    "ball_l",
)

_LEG_ALIGN_CALF_JOINTS: tuple[str, ...] = (
    "calf_r",
    "calf_l",
)

_LEG_ALIGN_THIGH_JOINTS: tuple[str, ...] = (
    "thigh_r",
    "thigh_l",
)

_LEG_ALIGN_SELECTOR_GROUPS: Dict[str, tuple[str, ...]] = {
    "all": _LEG_ALIGN_PROXIMAL_JOINTS + _LEG_ALIGN_DISTAL_JOINTS,
    "leg": _LEG_ALIGN_PROXIMAL_JOINTS + _LEG_ALIGN_DISTAL_JOINTS,
    "legs": _LEG_ALIGN_PROXIMAL_JOINTS + _LEG_ALIGN_DISTAL_JOINTS,
    "distal": _LEG_ALIGN_DISTAL_JOINTS,
    "distals": _LEG_ALIGN_DISTAL_JOINTS,
    "proximal": _LEG_ALIGN_PROXIMAL_JOINTS,
    "proximals": _LEG_ALIGN_PROXIMAL_JOINTS,
    "foot": _LEG_ALIGN_FOOT_JOINTS,
    "feet": _LEG_ALIGN_FOOT_JOINTS,
    "ball": _LEG_ALIGN_BALL_JOINTS,
    "balls": _LEG_ALIGN_BALL_JOINTS,
    "calf": _LEG_ALIGN_CALF_JOINTS,
    "calves": _LEG_ALIGN_CALF_JOINTS,
    "thigh": _LEG_ALIGN_THIGH_JOINTS,
    "thighs": _LEG_ALIGN_THIGH_JOINTS,
}


@dataclass(frozen=True)
class PostTrainConfig:
    ckpt_in: Path
    out_dir: Path
    run_name: str

    data: Path
    paths: Optional[Tuple[Path, ...]]
    bundle_json: Path
    pretrain_template: Optional[Path]
    encoder_bundle: Optional[Path]

    device: str
    batch: int
    seq_len: int
    # Dataset window sampling mode:
    # - sliding    : enumerate all windows (historical mode; can heavily overweight long clips)
    # - start0     : only start=0 per clip
    # - clip_random: balanced per-clip sampling; random start per clip (train only)
    dataset_index_mode: str
    # Rollout horizon for loss unroll (<= seq_len-1). 0 means "use full window".
    rollout_steps: int
    # Optional multi-cycle unroll for looped clips (pose drift training).
    # When >1, repeats the (seq_len-1) transitions with modulo indexing.
    rollout_cycles: int
    # When multi-cycle unroll is enabled, optionally include the synthetic boundary transition
    # between the last frame and the first frame (wrap). This makes rollout dynamics match
    # freerun_cycles (continuous closed-loop), but the boundary step can be down-weighted for loss.
    rollout_include_boundary: bool
    # Randomize the starting phase (offset) within a cycle for each batch to reduce overfitting to
    # a fixed wrap boundary phase.
    rollout_random_offset: bool
    # How to feed time_index into EventMotionModel (used by contact_plan time-PE):
    # - global: time_index = start + global_step
    # - cycle : time_index = step % (seq_len-1) (keeps time-PE in-range under multi-cycle)
    # - auto  : cycle when rollout_cycles>1 else global
    # - none  : disable time_index (no time-PE bias)
    time_index_mode: str
    depth: int
    num_heads: int
    dropout: float
    context_len: int
    epochs: int
    steps_per_epoch: int
    save_step_ckpts: Optional[str]
    lr: float
    weight_decay: float

    # Optional reset hook kept for checkpoint calibration/debugging.
    so3_corr_gate_logit_reset: Optional[float]
    detach_rollout_state: bool

    # Legacy Stage1-5 target configs are retained for parsing compatibility only;
    # corresponding train_* selectors are retired and rejected via fail-fast checks.
    contact_plan_init_mode: str
    contact_plan_init_hidden: int
    contact_plan_init_dropout: float

    # Event-Clock v3 (contact_plan residual correction). Auto-detected from checkpoint by default.
    # - auto: enable iff ckpt has event_clock_* weights
    # - on  : force-enable (even if weights missing)
    # - off : force-disable (drops weights on save)
    event_clock: str  # auto|on|off
    event_clock_max_delta: float
    event_clock_hidden_dim: Optional[int]
    event_clock_gate_hidden_dim: Optional[int]

    # Stage2: freeze experts, only train lambda_fusion_head (learn when to trust incremental vs direct).
    train_lambda_head: bool
    # Optional: finetune direct_pose_head (cond+plan(+meas) -> absolute pose).
    train_direct_pose: bool
    # Optional: enable a leg-specific residual head for direct pose (extra capacity for lower body).
    direct_pose_leg_enable: bool
    direct_pose_leg_bones: Optional[str]
    # Optional: when train_direct_pose=true, freeze direct_pose_head and train leg residual head only.
    direct_pose_leg_train_only: bool
    # Optional: when train_direct_pose=true, freeze direct_pose_head + direct_pose_leg_head and train the
    # leg gate/scale head only (useful for calibration without re-learning omega direction/magnitude).
    direct_pose_leg_gate_train_only: bool
    # Leg residual mode / decoupling knobs (see train/models.py):
    direct_pose_leg_mode: str
    direct_pose_leg_stopgrad_main: bool
    direct_pose_leg_detach_feat: bool
    direct_pose_leg_max_deg: float
    # Optional: learned gate/scale for leg omega (SO(3) mode only). See train/models.py.
    # - learned      : omega_eff = sigmoid(gate_logits) ** gate_power * omega_raw
    # - scale        : omega_eff = exp(clamp(log_mag, [-clip,+clip])) * omega_raw
    direct_pose_leg_gate_mode: str  # none|learned|scale
    direct_pose_leg_gate_power: float
    # Only used when direct_pose_leg_gate_mode='scale'.
    direct_pose_leg_scale_log_clip: float
    # Optional hard clamp on leg scale magnitude: k>1 => [1/k, k]; 0/1 disables.
    direct_pose_leg_scale_clamp_k: float
    # Optional: supervise learned leg gate using oracle ||omega_oracle|| thresholding (BCEWithLogits).
    # Target: gate=1 if ||omega_oracle|| >= direct_pose_leg_align_oracle_min_deg else 0.
    direct_pose_leg_gate_sup_weight: float
    # Optional: direction alignment loss for leg SO(3) residual omega (see docs/Problems/... 8.10).
    # align_mode='cos':  L_align = relu(-cos(omega_pred, omega_oracle))  (cheatable by ||omega_pred||->0)
    # align_mode='proj': omega_oracle = log(R_gt @ R_base^T); L = w_mag*(proj-||oracle||)^2 + w_res*||res||^2 (+ w_sign*relu(-proj)^2)
    direct_pose_leg_align_weight: float
    direct_pose_leg_align_oracle_min_deg: float
    direct_pose_leg_align_oracle_weight_deg: float
    direct_pose_leg_align_mode: str
    direct_pose_leg_align_mag_weight: float
    direct_pose_leg_align_res_weight: float
    direct_pose_leg_align_sign_weight: float
    # Optional: focus alignment on direction-misaligned cases only by masking joints where
    # cos(omega_pred, omega_oracle) >= thresh. 0 disables.
    direct_pose_leg_align_cos_thresh: float
    # Optional: restrict the main leg_align objective to a joint subset. Supports preset tokens
    # such as "distal", "proximal", "calf", "thigh", "foot", "ball", plus explicit joint names.
    # Empty/None means "all leg joints".
    direct_pose_leg_align_target_joints: Optional[str]
    # Optional: add a small auxiliary anchor on top of the main leg_align target subset.
    # Useful for experiments like "distal-only proj + tiny calf anchor".
    direct_pose_leg_align_anchor_joints: Optional[str]
    direct_pose_leg_align_anchor_weight: float
    # Optional curriculum for leg align weight.
    # - none   : keep direct_pose_leg_align_weight constant
    # - linear : hold start_weight for warmup_steps, then linearly ramp to target weight
    direct_pose_leg_align_schedule: str
    direct_pose_leg_align_start_weight: float
    direct_pose_leg_align_warmup_steps: int
    direct_pose_leg_align_ramp_steps: int
    # Direct pose head config overrides (optional; useful when reinitializing the direct head with a new input layout).
    # - feat_source: auto|cond|hidden|cond+hidden
    # - time_pe_dim: -1 means "auto (infer from checkpoint)", 0 disables
    direct_pose_feat_source: str
    direct_pose_time_pe_dim: int
    direct_pose_time_pe_base: float
    # If true, concatenate phase_z_in (2*contact_dim) into direct head input.
    direct_pose_use_phase_z: bool
    # How to route phase_z_in into direct conditioning:
    # - concat           : append phase_z_in as extra features (compat "add phase")
    # - replace_contacts : use phase_z_in to replace (contacts_plan, contacts_meas) in direct concat mode
    direct_pose_phase_z_mode: str
    # Optional: split direct output into leg/non-leg heads with shared trunk.
    direct_pose_split_enable: bool
    # Optional: non-leg projection bottleneck dim for split head.
    # >0 => h_nonleg=ReLU(Linear(hid, proj)); out_nonleg=Linear(proj, D_nonleg)
    # 0 => compat split (out_nonleg=Linear(hid, D_nonleg))
    direct_pose_nonleg_proj_dim: int
    # Optional: split non-leg branch into arm/else readouts (three-way split with leg branch).
    direct_pose_arm_split_enable: bool
    direct_pose_arm_bones: Optional[str]
    # Optional: when train_direct_pose=true, freeze trunk/leg and train non-leg branch only.
    direct_pose_nonleg_train_only: bool
    direct_pose_reinit: bool
    direct_pose_hidden_override: Optional[int]
    direct_pose_meas_mode_override: Optional[str]
    # Stage7 direct objective: optionally decouple legs vs non-legs (see discussion in Jan 2026 notes).
    direct_pose_loss_leg_split: bool
    # Optional: upweight selected non-leg bones inside the direct non-leg term (signal-focused Experiment 1).
    # Comma-separated bone names or indices. Only effective when objective="direct" and leg-split is active.
    direct_pose_nonleg_focus_bones: Optional[str]
    direct_pose_nonleg_focus_weight: float
    # Optional: group-wise magnitude normalization for direct objective (legs vs non-legs).
    #   L = w_leg * clamp(L_leg / ema_leg) + w_non * clamp(L_non / ema_non)
    # EMA is initialized by the first observed batch value (no warmup switch).
    direct_pose_loss_group_norm_enable: bool
    direct_pose_loss_group_norm_w_leg: float
    direct_pose_loss_group_norm_w_nonleg: float
    direct_pose_loss_group_norm_ema_beta: float
    direct_pose_loss_group_norm_ratio_min: float
    direct_pose_loss_group_norm_ratio_max: float
    direct_pose_loss_group_norm_eps: float
    # Optional: monitor split-head gradient allocation during direct-pose training.
    direct_pose_grad_monitor_enable: bool
    direct_pose_grad_ratio_gate: float
    # Optional: probe early-step gradient conflict between distal/proximal leg_align groups
    # on the shared leg head parameters.
    direct_pose_leg_align_grad_probe_enable: bool
    direct_pose_leg_align_grad_probe_steps: int

    lambda_fusion_mode: str
    lambda_fusion_hidden: int
    lambda_fusion_dropout: float
    lambda_fusion_logit_init: float
    lambda_fusion_use_rollout_step: bool
    lambda_fusion_entropy_weight: float
    lambda_fusion_smooth_weight: float
    lambda_fusion_early_steps: int
    lambda_fusion_early_weight: float
    lambda_fusion_monotonic_weight: float
    lambda_plan_entropy_weight: float
    lambda_plan_dyn_weight: float
    lambda_time_weight_mode: str
    lambda_time_weight_max: float
    lambda_reliability_mode: str
    lambda_reliability_warmup_steps: int
    lambda_reliability_contact_err_max: float
    # Optional: per-joint scaling (J,) applied to warmup r_t, to adapt warmup speed by bone.
    # Accepts list[float] (length=J) or None.
    lambda_reliability_warmup_joint_scales: Optional[list[float]]
    lambda_l2sp_weight: float
    # Loss weight multiplier for boundary (wrap) steps when rollout_include_boundary=true.
    # 0 disables boundary supervision but keeps boundary in state update.
    lambda_boundary_weight: float
    # Stage2: optional gate supervision for lambda_fusion_head (logits).
    # Encourages λ to match which expert has lower per-joint geodesic error:
    #   λ* = sigmoid((err_inc - err_dir) / τ)
    # and supervises lambda_fusion_logits with BCE (soft targets).
    # Disabled by default when lambda_gate_sup_weight=0.
    lambda_gate_sup_weight: float
    lambda_gate_sup_tau_deg: float
    # Only supervise when |err_inc - err_dir| >= margin (deg). Default 1°.
    # Set to 0 to disable the margin mask.
    lambda_gate_sup_margin_deg: float
    # Start rollout step for gate supervision. -1 auto uses lambda_reliability_warmup_steps when warmup is enabled.
    lambda_gate_sup_start_step: int

    # Optional rollout auxiliary supervision on contact_meas prediction (only active in direct/lambda objectives).
    contact_meas_weight: float

    # White-box contacts_meas runtime knobs (P2 ground_z stability / ablations).
    # Used by validation white-box rollout diagnostics.
    contact_meas_gate_by_hit: str  # auto|true|false
    contact_meas_vxy_mode: str  # abs|root_rel
    contact_meas_ground_z_mode: str  # ema|window|slew
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_slew_up_cm: float
    contact_meas_ground_z_slew_down_cm: float

    # Posttrain rollout frozen-contact runtime knobs.
    posttrain_contacts_pretrain_clamp: float
    posttrain_contacts_pretrain_affine_stats: Optional[str]

    seed: int


_CLI_BOOL_OVERRIDE_KEYS: tuple[str, ...] = (
    "rollout_include_boundary",
    "rollout_random_offset",
    "detach_rollout_state",
    "train_direct_pose",
    "train_lambda_head",
)
_CLI_OPTIONAL_FLOAT_OVERRIDE_KEYS: tuple[str, ...] = ("so3_corr_gate_logit_reset",)
_CLI_OVERRIDE_SKIP_KEYS: tuple[str, ...] = ("config", "paths")
_CLI_OVERRIDE_SPECIAL_KEYS: set[str] = set(_CLI_OVERRIDE_SKIP_KEYS) | set(_CLI_BOOL_OVERRIDE_KEYS) | set(
    _CLI_OPTIONAL_FLOAT_OVERRIDE_KEYS
)

_DIRECT_POSE_LEG_GATE_ALIAS_MAP: Dict[str, str] = {
    "": "none",
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "no": "none",
    "n": "none",
    "disable": "none",
    "disabled": "none",
    "learned": "learned",
    "on": "learned",
    "true": "learned",
    "1": "learned",
    "yes": "learned",
    "y": "learned",
    "scale": "scale",
    "mag": "scale",
    "magnitude": "scale",
    "logmag": "scale",
    "log_mag": "scale",
    "exp": "scale",
    "alpha": "scale",
}
_DIRECT_POSE_LEG_GATE_CHOICES: Tuple[str, ...] = ("none", "learned", "scale")


def _cfg_reject_retired_direct_pose_highorder(payload: Dict[str, Any]) -> None:
    active_keys: list[str] = []

    side_bool_keys = (
        "direct_pose_leg_side_routing",
        "direct_pose_leg_side_plan_other",
        "direct_pose_leg_side_phase_other",
        "direct_pose_leg_side_phase_rel",
        "direct_pose_leg_side_sign_gate",
        "direct_pose_leg_side_rank1",
    )
    for key in side_bool_keys:
        if _cfg_get_bool(payload, key, False):
            active_keys.append(key)

    if int(_cfg_get_int(payload, "direct_pose_leg_side_embed_dim", 0, min_value=0)) > 0:
        active_keys.append("direct_pose_leg_side_embed_dim")

    side_cue = str(payload.get("direct_pose_leg_side_cue") or "none").strip().lower()
    if side_cue not in ("", "none", "off", "disable", "disabled"):
        active_keys.append("direct_pose_leg_side_cue")

    if float(_cfg_get_float(payload, "direct_pose_leg_side_sign_gate_reg_weight", 0.0, min_value=0.0)) > 0.0:
        active_keys.append("direct_pose_leg_side_sign_gate_reg_weight")

    sic_raw = payload.get("direct_pose_loss_sics", None)
    if isinstance(sic_raw, (list, tuple)):
        if any(str(x).strip() for x in sic_raw):
            active_keys.append("direct_pose_loss_sics")
    elif sic_raw is not None and str(sic_raw).strip().lower() not in ("", "none", "null"):
        active_keys.append("direct_pose_loss_sics")

    if int(_cfg_get_int(payload, "direct_pose_loss_cycle_gte", 0, min_value=0)) > 0:
        active_keys.append("direct_pose_loss_cycle_gte")

    sic_mode = str(payload.get("direct_pose_loss_sic_mode") or "mask").strip().lower()
    if sic_mode not in ("", "mask", "none", "off", "disable", "disabled"):
        active_keys.append("direct_pose_loss_sic_mode")

    sic_boost = float(_cfg_get_float(payload, "direct_pose_loss_sic_boost", 1.0))
    if math.isfinite(sic_boost) and abs(float(sic_boost) - 1.0) > 1e-12:
        active_keys.append("direct_pose_loss_sic_boost")

    if active_keys:
        keys_txt = ", ".join(sorted(set(active_keys)))
        raise SystemExit(
            "[FATAL][RETIRED_DIRECT_POSE_HIGHORDER] posttrain mainline no longer accepts active direct_pose "
            f"high-order branches: {keys_txt}. "
            "Keep side-routing/sign-gate/rank1/SIC-focus at inert defaults, or use archived repro/validate lanes."
        )
def _cfg_parse_path_basic(payload: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_in = _as_path(payload.get("ckpt_in"))
    if ckpt_in is None:
        raise ValueError("Config must set 'ckpt_in'.")
    out_dir = _as_path(payload.get("out_dir")) or Path("./models/posttrain")

    data = _as_path(payload.get("data")) or Path("./raw_data/processed_data")
    paths_raw = payload.get("paths")
    paths: Optional[Tuple[Path, ...]] = None
    if isinstance(paths_raw, (list, tuple)):
        items: list[Path] = []
        for p in paths_raw:
            pp = _as_path(p)
            if pp is None:
                continue
            items.append(pp)
        if items:
            paths = tuple(items)
    bundle = _as_path(payload.get("bundle_json")) or Path("./raw_data/processed_data/norm_template.json")
    return {
        "ckpt_in": ckpt_in,
        "out_dir": out_dir,
        "run_name": str(payload.get("run_name") or f"posttrain_{time.strftime('%Y%m%d-%H%M%S')}"),
        "data": data,
        "paths": paths,
        "save_step_ckpts": _normalize_optional_csv(payload.get("save_step_ckpts", None)),
        "bundle_json": bundle,
        "pretrain_template": _as_path(payload.get("pretrain_template")),
        "encoder_bundle": _as_path(payload.get("encoder_bundle") or payload.get("encoder_path")),
    }


def _cfg_parse_direct_pose(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _cfg_from_schema(
        payload,
        [
            ("event_clock_hidden_dim", _cfg_get_int, {"key": "event_clock_hidden_dim", "default": None, "allow_none": True}),
            ("event_clock_gate_hidden_dim", _cfg_get_int, {"key": "event_clock_gate_hidden_dim", "default": None, "allow_none": True}),
            ("lambda_gate_sup_weight", _cfg_get_float, {"key": "lambda_gate_sup_weight", "default": 0.0}),
            ("lambda_gate_sup_tau_deg", _cfg_get_float, {"key": "lambda_gate_sup_tau_deg", "default": 2.5}),
            ("lambda_gate_sup_margin_deg", _cfg_get_float, {"key": "lambda_gate_sup_margin_deg", "default": 1.0}),
            ("lambda_gate_sup_start_step", _cfg_get_int, {"key": "lambda_gate_sup_start_step", "default": -1}),
            ("direct_pose_time_pe_dim", _cfg_get_int, {"key": "direct_pose_time_pe_dim", "default": -1}),
            ("direct_pose_hidden_override", _cfg_get_int, {"key": "direct_pose_hidden_override", "default": None, "allow_none": True}),
            ("direct_pose_nonleg_proj_dim", _cfg_get_int, {"key": "direct_pose_nonleg_proj_dim", "default": 0, "min_value": 0}),
            ("direct_pose_split_enable", _cfg_get_bool, {"key": "direct_pose_split_enable", "default": False}),
            ("direct_pose_arm_split_enable", _cfg_get_bool, {"key": "direct_pose_arm_split_enable", "default": False}),
            ("direct_pose_nonleg_train_only", _cfg_get_bool, {"key": "direct_pose_nonleg_train_only", "default": False}),
            ("direct_pose_leg_enable", _cfg_get_bool, {"key": "direct_pose_leg_enable", "default": False}),
            ("direct_pose_leg_train_only", _cfg_get_bool, {"key": "direct_pose_leg_train_only", "default": False}),
            ("direct_pose_leg_gate_train_only", _cfg_get_bool, {"key": "direct_pose_leg_gate_train_only", "default": False}),
            ("direct_pose_leg_mode", _cfg_get_str_or, {"key": "direct_pose_leg_mode", "default": "rot6d_add"}),
            ("direct_pose_leg_stopgrad_main", _cfg_get_bool, {"key": "direct_pose_leg_stopgrad_main", "default": False}),
            ("direct_pose_leg_detach_feat", _cfg_get_bool, {"key": "direct_pose_leg_detach_feat", "default": False}),
            ("direct_pose_leg_max_deg", _cfg_get_float, {"key": "direct_pose_leg_max_deg", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_gate_mode", _cfg_get_enum, {"key": "direct_pose_leg_gate_mode", "default": "none", "alias_map": _DIRECT_POSE_LEG_GATE_ALIAS_MAP, "choices": _DIRECT_POSE_LEG_GATE_CHOICES}),
            ("direct_pose_leg_gate_power", _cfg_get_float, {"key": "direct_pose_leg_gate_power", "default": 1.0, "min_value": 1e-8}),
            ("direct_pose_leg_gate_sup_weight", _cfg_get_float, {"key": "direct_pose_leg_gate_sup_weight", "default": 0.0, "min_value": 0.0}),
        ],
    )
    cfg["direct_pose_arm_bones"] = _normalize_optional_csv(payload.get("direct_pose_arm_bones", None))
    cfg["direct_pose_leg_bones"] = payload.get("direct_pose_leg_bones", None)

    direct_pose_meas_mode_override = payload.get("direct_pose_meas_mode_override", payload.get("direct_pose_meas_mode", None))
    if direct_pose_meas_mode_override is not None:
        direct_pose_meas_mode_override = str(direct_pose_meas_mode_override).strip() or None
    cfg["direct_pose_meas_mode_override"] = direct_pose_meas_mode_override

    cfg["direct_pose_leg_scale_log_clip"] = _cfg_get_float(payload, "direct_pose_leg_scale_log_clip", 4.0, min_value=1e-8)
    direct_pose_leg_scale_clamp_k = float(_cfg_get_float(payload, "direct_pose_leg_scale_clamp_k", 0.0))
    cfg["direct_pose_leg_scale_clamp_k"] = direct_pose_leg_scale_clamp_k if direct_pose_leg_scale_clamp_k > 1.0 else 0.0

    leg_align_cfg = _cfg_from_schema(
        payload,
        [
            ("direct_pose_leg_align_weight", _cfg_get_float, {"key": "direct_pose_leg_align_weight", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_oracle_min_deg", _cfg_get_float, {"key": "direct_pose_leg_align_oracle_min_deg", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_oracle_weight_deg", _cfg_get_float, {"key": "direct_pose_leg_align_oracle_weight_deg", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_mode", _cfg_get_enum, {"key": "direct_pose_leg_align_mode", "default": "cos", "alias_map": {"": "cos", "none": "cos", "off": "cos", "disable": "cos", "disabled": "cos"}, "choices": ("cos", "proj")}),
            ("direct_pose_leg_align_mag_weight", _cfg_get_float, {"key": "direct_pose_leg_align_mag_weight", "default": 1.0, "min_value": 0.0}),
            ("direct_pose_leg_align_res_weight", _cfg_get_float, {"key": "direct_pose_leg_align_res_weight", "default": 1.0, "min_value": 0.0}),
            ("direct_pose_leg_align_sign_weight", _cfg_get_float, {"key": "direct_pose_leg_align_sign_weight", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_cos_thresh", _cfg_get_float, {"key": "direct_pose_leg_align_cos_thresh", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_target_joints", _cfg_get_or, {"key": "direct_pose_leg_align_target_joints", "default": None}),
            ("direct_pose_leg_align_anchor_joints", _cfg_get_or, {"key": "direct_pose_leg_align_anchor_joints", "default": None}),
            ("direct_pose_leg_align_anchor_weight", _cfg_get_float, {"key": "direct_pose_leg_align_anchor_weight", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_schedule", _cfg_get_enum, {"key": "direct_pose_leg_align_schedule", "default": "none", "alias_map": {"": "none", "off": "none", "false": "none", "0": "none"}, "choices": ("none", "linear")}),
            ("direct_pose_leg_align_start_weight", _cfg_get_float, {"key": "direct_pose_leg_align_start_weight", "default": 0.0, "min_value": 0.0}),
            ("direct_pose_leg_align_warmup_steps", _cfg_get_int, {"key": "direct_pose_leg_align_warmup_steps", "default": 0, "min_value": 0}),
            ("direct_pose_leg_align_ramp_steps", _cfg_get_int, {"key": "direct_pose_leg_align_ramp_steps", "default": 0, "min_value": 0}),
        ],
    )
    for key in ("direct_pose_leg_align_target_joints", "direct_pose_leg_align_anchor_joints"):
        value = leg_align_cfg[key]
        if value is None:
            continue
        s = str(value).strip()
        leg_align_cfg[key] = None if s.lower() in ("", "none", "null", "off", "disabled") else s
    cfg.update(leg_align_cfg)

    cfg["direct_pose_loss_leg_split"] = _cfg_get_bool(payload, "direct_pose_loss_leg_split", False)
    cfg["direct_pose_nonleg_focus_bones"] = _normalize_optional_csv(payload.get("direct_pose_nonleg_focus_bones", None))

    direct_pose_loss_cfg = _cfg_from_schema(
        payload,
        [
            ("direct_pose_loss_group_norm_enable", _cfg_get_bool, {"key": "direct_pose_loss_group_norm_enable", "default": False}),
            ("direct_pose_loss_group_norm_w_leg", _cfg_get_float, {"key": "direct_pose_loss_group_norm_w_leg", "default": 1.0, "require_finite": False}),
            ("direct_pose_loss_group_norm_w_nonleg", _cfg_get_float, {"key": "direct_pose_loss_group_norm_w_nonleg", "default": 1.0, "require_finite": False}),
            ("direct_pose_loss_group_norm_ema_beta", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ema_beta", "default": 0.95}),
            ("direct_pose_loss_group_norm_ratio_min", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ratio_min", "default": 0.2}),
            ("direct_pose_loss_group_norm_ratio_max", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ratio_max", "default": 5.0}),
            ("direct_pose_loss_group_norm_eps", _cfg_get_float, {"key": "direct_pose_loss_group_norm_eps", "default": 1e-6}),
            ("direct_pose_nonleg_focus_weight", _cfg_get_float, {"key": "direct_pose_nonleg_focus_weight", "default": 1.0}),
            ("direct_pose_grad_monitor_enable", _cfg_get_bool, {"key": "direct_pose_grad_monitor_enable", "default": False}),
            ("direct_pose_grad_ratio_gate", _cfg_get_float, {"key": "direct_pose_grad_ratio_gate", "default": 0.35}),
            ("direct_pose_leg_align_grad_probe_enable", _cfg_get_bool, {"key": "direct_pose_leg_align_grad_probe_enable", "default": False}),
            ("direct_pose_leg_align_grad_probe_steps", _cfg_get_int, {"key": "direct_pose_leg_align_grad_probe_steps", "default": 0, "min_value": 0}),
        ],
    )
    direct_pose_loss_group_norm_ema_beta = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ema_beta"])
    if (not math.isfinite(direct_pose_loss_group_norm_ema_beta)) or direct_pose_loss_group_norm_ema_beta < 0.0:
        direct_pose_loss_group_norm_ema_beta = 0.95
    direct_pose_loss_cfg["direct_pose_loss_group_norm_ema_beta"] = _clamp_float(
        direct_pose_loss_group_norm_ema_beta,
        min_value=0.0,
        max_value=0.9999,
    )
    direct_pose_loss_group_norm_ratio_min = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_min"])
    direct_pose_loss_group_norm_ratio_max = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_max"])
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_min)) or direct_pose_loss_group_norm_ratio_min <= 0.0:
        direct_pose_loss_group_norm_ratio_min = 0.2
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_max)) or direct_pose_loss_group_norm_ratio_max <= 0.0:
        direct_pose_loss_group_norm_ratio_max = 5.0
    if direct_pose_loss_group_norm_ratio_min > direct_pose_loss_group_norm_ratio_max:
        direct_pose_loss_group_norm_ratio_min, direct_pose_loss_group_norm_ratio_max = direct_pose_loss_group_norm_ratio_max, direct_pose_loss_group_norm_ratio_min
    direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_min"] = direct_pose_loss_group_norm_ratio_min
    direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_max"] = direct_pose_loss_group_norm_ratio_max
    for key, default in (
        ("direct_pose_loss_group_norm_eps", 1e-6),
        ("direct_pose_nonleg_focus_weight", 1.0),
        ("direct_pose_grad_ratio_gate", 0.35),
    ):
        value = float(direct_pose_loss_cfg[key])
        direct_pose_loss_cfg[key] = default if (not math.isfinite(value)) or value <= 0.0 else value
    direct_pose_leg_align_grad_probe_steps = int(direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_steps"])
    if direct_pose_leg_align_grad_probe_steps < 0:
        direct_pose_leg_align_grad_probe_steps = 0
    if bool(direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_enable"]) and direct_pose_leg_align_grad_probe_steps <= 0:
        direct_pose_leg_align_grad_probe_steps = 30
    direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_steps"] = direct_pose_leg_align_grad_probe_steps
    cfg.update(direct_pose_loss_cfg)
    return cfg


def _cfg_parse_lambda_rollout(payload: Dict[str, Any]) -> Dict[str, Any]:
    core_cfg = _cfg_from_schema(
        payload,
        [
            ("device", _cfg_get_str_or, {"key": "device", "default": "auto"}),
            ("batch", _cfg_get_int_or, {"key": "batch", "default": 8}),
            ("seq_len", _cfg_get_int_or, {"key": "seq_len", "default": 180}),
            ("dataset_index_mode", _cfg_get_str_or, {"key": "dataset_index_mode", "default": "sliding"}),
            ("rollout_steps", _cfg_get_int_or, {"key": "rollout_steps", "default": 0}),
            ("rollout_cycles", _cfg_get_int_or, {"key": "rollout_cycles", "default": 1, "min_value": 1}),
            ("rollout_include_boundary_raw", _cfg_pick, {"key": "rollout_include_boundary"}),
            ("rollout_random_offset", _cfg_get_bool, {"key": "rollout_random_offset", "default": False}),
            ("time_index_mode", _cfg_get_str_or, {"key": "time_index_mode", "default": "global"}),
            ("depth", _cfg_get_int_or, {"key": "depth", "default": 3}),
            ("num_heads", _cfg_get_int_or, {"key": "num_heads", "default": 4}),
            ("dropout", _cfg_get_float_or, {"key": "dropout", "default": 0.1}),
            ("context_len", _cfg_get_int_or, {"key": "context_len", "default": 16}),
            ("epochs", _cfg_get_int_present, {"key": "epochs", "default": 1}),
            ("steps_per_epoch", _cfg_get_int_present, {"key": "steps_per_epoch", "default": 200}),
            ("lr", _cfg_get_float_present, {"key": "lr", "default": 2e-4}),
            ("weight_decay", _cfg_get_float_or, {"key": "weight_decay", "default": 0.0}),
            ("so3_corr_gate_logit_reset", _cfg_pick, {"key": "so3_corr_gate_logit_reset"}),
            ("detach_rollout_state", _cfg_get_bool, {"key": "detach_rollout_state", "default": True}),
            ("contact_plan_init_mode", _cfg_get_str_or, {"key": "contact_plan_init_mode", "default": "learnable"}),
            ("contact_plan_init_hidden", _cfg_get_int_or, {"key": "contact_plan_init_hidden", "default": 128}),
            ("contact_plan_init_dropout", _cfg_get_float_or, {"key": "contact_plan_init_dropout", "default": 0.0}),
            ("event_clock", _cfg_get_str_or, {"key": "event_clock", "default": "auto"}),
            ("event_clock_max_delta", _cfg_get_float_or, {"key": "event_clock_max_delta", "default": 0.5}),
            ("train_lambda_head", _cfg_get_bool, {"key": "train_lambda_head", "default": False}),
            ("train_direct_pose", _cfg_get_bool, {"key": "train_direct_pose", "default": False}),
            ("direct_pose_feat_source", _cfg_get_str_or, {"key": "direct_pose_feat_source", "default": "auto"}),
            ("direct_pose_time_pe_base", _cfg_get_float_or, {"key": "direct_pose_time_pe_base", "default": 10000.0}),
            ("direct_pose_use_phase_z", _cfg_get_bool, {"key": "direct_pose_use_phase_z", "default": False}),
            ("direct_pose_phase_z_mode", _cfg_get_str_or, {"key": "direct_pose_phase_z_mode", "default": "concat"}),
            ("direct_pose_reinit", _cfg_get_bool, {"key": "direct_pose_reinit", "default": False}),
            ("lambda_fusion_mode", _cfg_get_str_or, {"key": "lambda_fusion_mode", "default": "per_joint"}),
            ("lambda_fusion_hidden", _cfg_get_int_or, {"key": "lambda_fusion_hidden", "default": 128}),
            ("lambda_fusion_dropout", _cfg_get_float_or, {"key": "lambda_fusion_dropout", "default": 0.0}),
            ("lambda_fusion_logit_init", _cfg_get_float_or, {"key": "lambda_fusion_logit_init", "default": -2.0}),
            ("lambda_fusion_use_rollout_step", _cfg_get_bool, {"key": "lambda_fusion_use_rollout_step", "default": False}),
            ("lambda_fusion_entropy_weight", _cfg_get_float_or, {"key": "lambda_fusion_entropy_weight", "default": 0.0}),
            ("lambda_fusion_smooth_weight", _cfg_get_float_or, {"key": "lambda_fusion_smooth_weight", "default": 0.0}),
            ("lambda_fusion_early_steps", _cfg_get_int_or, {"key": "lambda_fusion_early_steps", "default": 0}),
            ("lambda_fusion_early_weight", _cfg_get_float_or, {"key": "lambda_fusion_early_weight", "default": 0.0}),
            ("lambda_fusion_monotonic_weight", _cfg_get_float_or, {"key": "lambda_fusion_monotonic_weight", "default": 0.0}),
            ("lambda_plan_entropy_weight", _cfg_get_float_or, {"key": "lambda_plan_entropy_weight", "default": 0.0}),
            ("lambda_plan_dyn_weight", _cfg_get_float_or, {"key": "lambda_plan_dyn_weight", "default": 0.0}),
            ("lambda_time_weight_mode", _cfg_get_str_or, {"key": "lambda_time_weight_mode", "default": "inv"}),
            ("lambda_time_weight_max", _cfg_get_float_or, {"key": "lambda_time_weight_max", "default": 2.0}),
            ("lambda_reliability_mode", _cfg_get_str_or, {"key": "lambda_reliability_mode", "default": "none"}),
            ("lambda_reliability_warmup_steps", _cfg_get_int_or, {"key": "lambda_reliability_warmup_steps", "default": 0}),
            ("lambda_reliability_contact_err_max", _cfg_get_float_or, {"key": "lambda_reliability_contact_err_max", "default": 1.0}),
            ("lambda_reliability_warmup_joint_scales_raw", _cfg_pick, {"key": "lambda_reliability_warmup_joint_scales"}),
            ("lambda_l2sp_weight", _cfg_get_float_or, {"key": "lambda_l2sp_weight", "default": 0.0}),
            ("lambda_boundary_weight", _cfg_get_float_or, {"key": "lambda_boundary_weight", "default": 0.0}),
            ("contact_meas_weight", _cfg_get_float_or, {"key": "contact_meas_weight", "default": 0.0}),
            ("contact_meas_gate_by_hit", _cfg_get_str_or, {"key": "contact_meas_gate_by_hit", "default": "auto"}),
            ("contact_meas_vxy_mode", _cfg_get_str_or, {"key": "contact_meas_vxy_mode", "default": "abs"}),
            ("contact_meas_ground_z_mode", _cfg_get_str_or, {"key": "contact_meas_ground_z_mode", "default": "window"}),
            ("contact_meas_ground_z_beta", _cfg_get_float_or, {"key": "contact_meas_ground_z_beta", "default": 0.05}),
            ("contact_meas_ground_z_window", _cfg_get_int_or, {"key": "contact_meas_ground_z_window", "default": 5}),
            ("contact_meas_ground_z_quantile", _cfg_get_float_or, {"key": "contact_meas_ground_z_quantile", "default": 0.2}),
            ("contact_meas_ground_z_slew_up_cm", _cfg_get_float_or, {"key": "contact_meas_ground_z_slew_up_cm", "default": 0.0}),
            ("contact_meas_ground_z_slew_down_cm", _cfg_get_float_or, {"key": "contact_meas_ground_z_slew_down_cm", "default": 0.0}),
            ("posttrain_contacts_pretrain_clamp", _cfg_get_float, {"key": "posttrain_contacts_pretrain_clamp", "default": 1.0, "min_value": 0.0}),
            ("posttrain_contacts_pretrain_affine_stats", _cfg_pick, {"key": "posttrain_contacts_pretrain_affine_stats"}),
            ("seed", _cfg_get_int_or, {"key": "seed", "default": 0}),
        ],
    )
    rollout_cycles_val = int(core_cfg["rollout_cycles"])
    core_cfg["rollout_include_boundary"] = _as_bool(
        core_cfg.pop("rollout_include_boundary_raw"),
        default=(rollout_cycles_val > 1),
    )
    core_cfg["lambda_reliability_warmup_joint_scales"] = _as_float_list(
        core_cfg.pop("lambda_reliability_warmup_joint_scales_raw")
    )
    clamp_v = core_cfg.get("posttrain_contacts_pretrain_clamp", 1.0)
    try:
        clamp_f = float(clamp_v)
    except Exception:
        clamp_f = 1.0
    if not math.isfinite(clamp_f):
        clamp_f = 1.0
    core_cfg["posttrain_contacts_pretrain_clamp"] = max(0.0, float(clamp_f))
    affine_spec = core_cfg.get("posttrain_contacts_pretrain_affine_stats", None)
    if affine_spec is not None:
        affine_spec = str(affine_spec).strip() or None
    core_cfg["posttrain_contacts_pretrain_affine_stats"] = affine_spec
    return core_cfg


def _cfg_from_payload(payload: Dict[str, Any]) -> PostTrainConfig:
    if not isinstance(payload, dict):
        raise TypeError("posttrain config payload must be a dict")
    for reject in (
        _cfg_reject_retired_direct_pose_highorder,
    ):
        reject(payload)
    path_basic_cfg = _cfg_parse_path_basic(payload)
    rollout_lambda_cfg = _cfg_parse_lambda_rollout(payload)
    direct_pose_cfg = _cfg_parse_direct_pose(payload)
    return PostTrainConfig(**path_basic_cfg, **rollout_lambda_cfg, **direct_pose_cfg)


def _resolve_rollout_steps(T: int, rollout_steps: int) -> int:
    max_steps = int(T - 1)
    if max_steps <= 0:
        raise ValueError(f"seq_len must be >=2, got {T}")
    try:
        r = int(rollout_steps or 0)
    except Exception:
        r = 0
    if r > 0:
        return max(1, min(max_steps, r))
    return max_steps


def _make_rollout_step_weights(
    steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    max_val: float,
) -> torch.Tensor:
    mode = str(mode or "inv").strip().lower()
    if mode in ("inv", "inverse", "1/t", "one_over_t"):
        w = 1.0 / (torch.arange(int(steps), device=device, dtype=dtype) + 1.0)
    elif mode in ("linear", "lin"):
        max_val = max(1.0, float(max_val or 1.0))
        w = torch.linspace(1.0, max_val, steps=int(steps), device=device, dtype=dtype)
    else:
        w = torch.ones((int(steps),), device=device, dtype=dtype)
    return w / w.sum().clamp_min(1e-6)


def _prepare_rollout_cond(
    trainer: Trainer,
    *,
    cond_seq: Optional[torch.Tensor],
    cond_raw_tgt: Optional[torch.Tensor],
    cond_norm_mu: Optional[torch.Tensor],
    cond_norm_std: Optional[torch.Tensor],
    idx: int,
    t: int,
    include_boundary: bool,
    cycle_len: int,
    y_prev_raw: torch.Tensor,
    enable_reprojection: bool,
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    cond_t, cond_raw_step, _, _ = _rollout_kernel.prepare_rollout_cond(
        trainer,
        cond_seq=cond_seq,
        cond_raw_seq=cond_raw_tgt,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        step_idx=int(t),
        cond_idx=int(idx),
        cond_has_time_dim=bool(torch.is_tensor(cond_seq) and cond_seq.dim() == 3),
        cond_raw_has_time_dim=bool(torch.is_tensor(cond_raw_tgt) and cond_raw_tgt.dim() == 3),
        include_boundary=bool(include_boundary),
        cycle_len=int(cycle_len),
        y_prev_raw=y_prev_raw,
        allow_reprojection=bool(enable_reprojection and int(t) > 0),
        yaw_gt_fn=yaw_gt_fn,
    )
    return cond_t, cond_raw_step

def _rollout_step_common(
    trainer: Trainer,
    model: EventMotionModel,
    *,
    state: Dict[str, Any],
    t: int,
    idx: int,
    total_steps: int,
    cond_seq: Optional[torch.Tensor],
    cond_raw_tgt: Optional[torch.Tensor],
    cond_norm_mu: Optional[torch.Tensor],
    cond_norm_std: Optional[torch.Tensor],
    angvel_seq: Optional[torch.Tensor],
    pose_hist_seq: Optional[torch.Tensor],
    time_index_mode: str,
    time_base: Optional[torch.Tensor],
    enable_reprojection: bool,
    include_boundary: bool = False,
    cycle_len: int = 1,
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    detach_rollout_state: bool = True,
) -> Dict[str, Any]:
    motion = state["motion"]
    y_prev_raw = state["y_prev_raw"]

    cond_t, cond_raw_step = _prepare_rollout_cond(
        trainer,
        cond_seq=cond_seq,
        cond_raw_tgt=cond_raw_tgt,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        idx=int(idx),
        t=int(t),
        include_boundary=bool(include_boundary),
        cycle_len=int(cycle_len),
        y_prev_raw=y_prev_raw,
        enable_reprojection=bool(enable_reprojection),
        yaw_gt_fn=yaw_gt_fn,
    )

    angvel_t = _rollout_kernel.resolve_rollout_step_angvel(
        trainer,
        motion=motion,
        angvel_seq=angvel_seq,
        step_idx=int(idx),
        has_time_dim=bool(torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3),
    )

    _, pose_hist_t = _rollout_kernel.resolve_rollout_pose_history(
        pose_hist_state=state.get("pose_hist_state", None),
        pose_hist_seq=pose_hist_seq,
        idx=int(idx),
    )

    contacts_in_t = _rollout_kernel.prepare_rollout_contacts_input(
        trainer,
        model,
        motion_t=motion,
        pose_hist_t=pose_hist_t,
    )

    time_base_local, time_index_seed = _rollout_kernel.resolve_rollout_time_controls(
        time_index_mode=str(time_index_mode),
        time_base=time_base,
        frame_idx=int(idx),
        rollout_step_idx=int(t),
    )
    time_index_t, rollout_step_t = _rollout_kernel.build_rollout_step_time_inputs(
        motion,
        total_steps=int(total_steps),
        step_idx=int(t),
        time_base=time_base_local,
        time_index_seed=time_index_seed,
    )

    ret, _, _ = _rollout_kernel.forward_rollout_model_step(
        model,
        motion=motion.unsqueeze(1),
        cond_input=_rollout_kernel.ensure_rollout_time_axis(cond_t),
        contacts_in_t=contacts_in_t,
        angvel_t=_rollout_kernel.ensure_rollout_time_axis(angvel_t),
        pose_history_t=_rollout_kernel.ensure_rollout_time_axis(pose_hist_t),
        plan_z=state.get("plan_z", None),
        meas_logits_prev=state.get("meas_logits_prev", None),
        time_index_t=time_index_t,
        rollout_step_t=rollout_step_t,
    )
    _rollout_kernel.update_rollout_recurrent_state(model, ret, state)

    return {
        "ret": ret,
        "contacts_in_t": contacts_in_t,
        "cond_raw_step": cond_raw_step,
        "time_index_t": time_index_t,
        "rollout_step_t": rollout_step_t,
    }


def _lambda_entropy(p: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


_ROLLOUT_SOFT_FAIL_ERRORS = (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError)


def _record_posttrain_soft_fail(trainer: Trainer, key: str) -> None:
    store = getattr(trainer, "_posttrain_soft_fail_counts", None)
    if not isinstance(store, dict):
        store = {}
    store[key] = int(store.get(key, 0)) + 1
    setattr(trainer, "_posttrain_soft_fail_counts", store)


@dataclass(frozen=True)
class LambdaRolloutPrepContext:
    device: torch.device
    dtype: torch.dtype
    motion_seq: torch.Tensor
    gt_seq: torch.Tensor
    cond_seq: Optional[torch.Tensor]
    cond_raw_tgt: Optional[torch.Tensor]
    cond_norm_mu: Optional[torch.Tensor]
    cond_norm_std: Optional[torch.Tensor]
    contacts_seq: Optional[torch.Tensor]
    angvel_seq: Optional[torch.Tensor]
    pose_hist_seq: Optional[torch.Tensor]
    B: int
    T: int
    Dy: int
    steps: int
    rollout_cycles: int
    include_boundary: bool
    cycle_len: int
    total_steps: int
    offset: int
    y0_raw: Optional[torch.Tensor]
    rot_slice: slice
    rot_len: int
    J: int
    std_y: torch.Tensor
    state: Dict[str, Any]
    step_weights: torch.Tensor
    boundary_steps: int
    boundary_weighted_sum: float


@dataclass(frozen=True)
class LambdaRolloutNonLegFocusContext:
    direct_nonleg_focus_mask_j: Optional[torch.Tensor]
    direct_nonleg_focus_requested: int
    direct_nonleg_focus_resolved: int
    direct_nonleg_focus_weight_use: float
    direct_nonleg_focus_applied: float


@dataclass(frozen=True)
class LambdaRolloutRegParams:
    gate_sup_weight: float
    gate_sup_start: int
    tau_rad: float
    margin_rad: float
    direct_group_norm_enable: bool
    direct_group_w_leg: float
    direct_group_w_nonleg: float
    direct_group_beta: float
    direct_group_ratio_min: float
    direct_group_ratio_max: float
    direct_group_eps: float


@dataclass(frozen=True)
class LambdaRolloutRuntimeContext:
    trainer: Trainer
    model: EventMotionModel
    state: Dict[str, Any]
    total_steps: int
    cycle_len: int
    include_boundary: bool
    steps: int
    offset: int
    time_index_mode: str
    time_base: Any
    enable_reprojection: bool
    detach_rollout_state: bool
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]]
    columns: Tuple[str, str]
    B: int
    J: int
    objective: str
    y0_raw: Optional[torch.Tensor]
    gt_seq: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    rot_len: int


@dataclass(frozen=True)
class LambdaRolloutDataContext:
    cond_seq: Optional[torch.Tensor]
    cond_raw_tgt: Optional[torch.Tensor]
    cond_norm_mu: Optional[torch.Tensor]
    cond_norm_std: Optional[torch.Tensor]
    angvel_seq: Optional[torch.Tensor]
    pose_hist_seq: Optional[torch.Tensor]
    contacts_seq: Optional[torch.Tensor]
    step_weights: torch.Tensor
    std_y: torch.Tensor
    rot_slice: slice


@dataclass(frozen=True)
class LambdaRolloutWeights:
    contact_meas_weight: float
    direct_pose_leg_align_weight: float
    direct_pose_leg_align_oracle_min_deg: float
    direct_pose_leg_align_oracle_weight_deg: float
    direct_pose_leg_align_mode: str
    direct_pose_leg_align_mag_weight: float
    direct_pose_leg_align_res_weight: float
    direct_pose_leg_align_sign_weight: float
    direct_pose_leg_align_cos_thresh: float
    direct_pose_leg_align_target_joints: Optional[str]
    direct_pose_leg_align_anchor_joints: Optional[str]
    direct_pose_leg_align_anchor_weight: float
    direct_pose_leg_gate_sup_weight: float
    direct_pose_loss_leg_split: bool
    direct_nonleg_focus_mask_j: Optional[torch.Tensor]
    direct_nonleg_focus_resolved: int
    direct_nonleg_focus_weight_use: float
    gate_sup_weight: float
    gate_sup_start: int
    tau_rad: float
    margin_rad: float
    lambda_plan_entropy_weight: float
    lambda_plan_dyn_weight: float
    lambda_early_weight: float
    lambda_early_steps: int
    lambda_entropy_weight: float
    lambda_smooth_weight: float
    lambda_monotonic_weight: float


@dataclass
class LambdaFusionAccum:
    loss_terms: list[torch.Tensor] = field(default_factory=list)
    inc_terms: list[torch.Tensor] = field(default_factory=list)
    dir_terms: list[torch.Tensor] = field(default_factory=list)
    dir_base_terms: list[torch.Tensor] = field(default_factory=list)
    dir_leg_base_terms: list[torch.Tensor] = field(default_factory=list)
    dir_nonleg_base_terms: list[torch.Tensor] = field(default_factory=list)
    dir_nonleg_plain_terms: list[torch.Tensor] = field(default_factory=list)
    leg_gate_sup_terms: list[torch.Tensor] = field(default_factory=list)
    leg_gate_sup_tgt_frac_terms: list[torch.Tensor] = field(default_factory=list)
    leg_gate_sup_pred_mean_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_frac_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_joint_num_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_joint_den_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_joint_frac_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_distal_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_distal_frac_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_proximal_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_proximal_frac_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_anchor_terms: list[torch.Tensor] = field(default_factory=list)
    leg_align_anchor_frac_terms: list[torch.Tensor] = field(default_factory=list)
    ent_terms: list[torch.Tensor] = field(default_factory=list)
    smooth_terms: list[torch.Tensor] = field(default_factory=list)
    early_terms: list[torch.Tensor] = field(default_factory=list)
    mono_terms: list[torch.Tensor] = field(default_factory=list)
    plan_ent_terms: list[torch.Tensor] = field(default_factory=list)
    plan_dyn_terms: list[torch.Tensor] = field(default_factory=list)
    plan_ent_stat_terms: list[torch.Tensor] = field(default_factory=list)
    plan_dyn_stat_terms: list[torch.Tensor] = field(default_factory=list)
    meas_terms: list[torch.Tensor] = field(default_factory=list)
    lam_vals: list[torch.Tensor] = field(default_factory=list)
    lam_eff_vals: list[torch.Tensor] = field(default_factory=list)
    lam_rel_vals: list[torch.Tensor] = field(default_factory=list)
    boundary_blend_terms: list[torch.Tensor] = field(default_factory=list)
    boundary_inc_terms: list[torch.Tensor] = field(default_factory=list)
    boundary_dir_terms: list[torch.Tensor] = field(default_factory=list)
    boundary_lam_terms: list[torch.Tensor] = field(default_factory=list)
    boundary_lam_eff_terms: list[torch.Tensor] = field(default_factory=list)
    boundary_r_terms: list[torch.Tensor] = field(default_factory=list)
    gate_sup_terms: list[torch.Tensor] = field(default_factory=list)
    gate_sup_frac_terms: list[torch.Tensor] = field(default_factory=list)
    gate_sup_acc_num_terms: list[torch.Tensor] = field(default_factory=list)
    gate_sup_acc_den_terms: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class LambdaFusionFinalizeContext:
    trainer: Trainer
    model: EventMotionModel
    objective: str
    direct_pose_leg_gate_sup_weight: float = 0.0
    direct_pose_leg_align_weight: float = 0.0
    direct_pose_leg_align_anchor_weight: float = 0.0
    lambda_entropy_weight: float = 0.0
    lambda_smooth_weight: float = 0.0
    lambda_early_weight: float = 0.0
    lambda_monotonic_weight: float = 0.0
    lambda_plan_entropy_weight: float = 0.0
    lambda_plan_dyn_weight: float = 0.0
    contact_meas_weight: float = 0.0
    include_boundary: bool = False
    random_offset: bool = False
    offset: int = 0
    boundary_weight: float = 0.0
    boundary_steps: int = 0
    boundary_weighted_sum: float = 0.0
    direct_nonleg_focus_requested: int = 0
    direct_nonleg_focus_resolved: int = 0
    direct_nonleg_focus_weight_use: float = 1.0
    direct_nonleg_focus_applied: float = 0.0
    meas_used_logits: bool = False
    gate_sup_weight: float = 0.0
    direct_group_norm_enable: bool = False
    direct_group_w_leg: float = 1.0
    direct_group_w_nonleg: float = 1.0
    direct_group_beta: float = 0.95
    direct_group_ratio_min: float = 0.2
    direct_group_ratio_max: float = 5.0
    direct_group_eps: float = 1e-6


@dataclass
class LambdaRolloutStepState:
    meas_used_logits: bool
    direct_nonleg_focus_applied: float
    lam_prev: Optional[torch.Tensor]
    lam_prev_monot: Optional[torch.Tensor]
    plan_prev: Optional[torch.Tensor]


@dataclass
class LambdaRolloutStepContext:
    runtime: LambdaRolloutRuntimeContext
    data: LambdaRolloutDataContext
    weights: LambdaRolloutWeights
    accum: LambdaFusionAccum
    state_vars: LambdaRolloutStepState


def _lambda_rollout_prepare_context(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    rollout_steps: int,
    rollout_cycles: int,
    include_boundary: bool,
    boundary_weight: float,
    random_offset: bool,
    time_weight_mode: str,
    time_weight_max: float,
) -> LambdaRolloutPrepContext:
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch["motion"].to(device=device, dtype=dtype)  # (B,T,Dx)
    gt_seq = batch["gt_motion"].to(device=device, dtype=dtype)  # (B,T,Dy) -> absolute y[t+1]
    cond_seq = batch.get("cond_in")
    if torch.is_tensor(cond_seq):
        cond_seq = cond_seq.to(device=device, dtype=dtype)
    cond_raw_tgt = batch.get("cond_tgt_raw")
    if torch.is_tensor(cond_raw_tgt):
        cond_raw_tgt = cond_raw_tgt.to(device=device, dtype=dtype)
    cond_norm_mu = batch.get("cond_norm_mu")
    if torch.is_tensor(cond_norm_mu):
        cond_norm_mu = cond_norm_mu.to(device=device, dtype=dtype)
    else:
        cond_norm_mu = None
    cond_norm_std = batch.get("cond_norm_std")
    if torch.is_tensor(cond_norm_std):
        cond_norm_std = cond_norm_std.to(device=device, dtype=dtype)
    else:
        cond_norm_std = None
    # Mirror eval_utils / run_freerun_cycles: normalize a reprojected raw condition into cond_input.
    cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, motion_seq) if cond_norm_mu is not None else None
    cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, motion_seq) if cond_norm_std is not None else None

    contacts_seq = batch.get("contacts")
    if torch.is_tensor(contacts_seq):
        contacts_seq = contacts_seq.to(device=device, dtype=dtype)
    angvel_seq = batch.get("angvel")
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    pose_hist_seq = batch.get("pose_hist")
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    B, T, _ = motion_seq.shape
    Dy = int(gt_seq.shape[-1])
    if T < 2:
        raise ValueError(f"seq_len must be >=2, got {T}")
    steps = _resolve_rollout_steps(T, rollout_steps)
    steps = max(1, int(steps))
    rollout_cycles = max(1, int(rollout_cycles or 1))
    include_boundary = bool(include_boundary) and int(rollout_cycles) > 1 and int(steps) == int(T - 1)
    cycle_len = int(T) if include_boundary else int(steps)
    total_steps = (int(rollout_cycles) * int(cycle_len) - 1) if include_boundary else (int(steps) * int(rollout_cycles))

    offset = 0
    if bool(random_offset) and int(rollout_cycles) > 1:
        try:
            offset = int(torch.randint(low=0, high=max(1, int(cycle_len)), size=(1,), device="cpu").item())
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "prep_random_offset"); offset = 0

    # For wrap-boundary steps (idx==T-1), use y0 as the target pose (matches tiled freerun_cycles).
    y0_raw = None
    if include_boundary:
        try:
            motion0 = motion_seq[:, 0]
            motion0_raw = trainer.normalizer.denorm_x(motion0)
            y0_raw = trainer.normalizer.x_to_y(motion0_raw, Dy)
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "prep_boundary_y0"); y0_raw = None

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, Dy)
    rot_len = int(rot_slice.stop - rot_slice.start)
    if rot_len <= 0 or rot_len % 6 != 0:
        raise ValueError(f"Invalid rot slice len={rot_len} (expected multiple of 6)")
    J = rot_len // 6

    std_y_np = getattr(trainer.normalizer, "std_y", None)
    if std_y_np is None:
        raise RuntimeError("Trainer.normalizer.std_y missing; cannot scale delta_norm -> delta_raw")
    std_y = torch.as_tensor(np.asarray(std_y_np, dtype=np.float32), device=device, dtype=dtype)

    motion = motion_seq[:, int(offset)]
    motion_raw = trainer.normalizer.denorm_x(motion)
    y_prev_raw = trainer.normalizer.x_to_y(motion_raw, Dy)
    pose_hist_state = _rollout_kernel.prepare_rollout_pose_hist_state(
        trainer,
        state_seq=motion_seq,
        pose_hist_seq=pose_hist_seq,
        y_raw_local=y_prev_raw,
        rot6d_y_slice=rot_slice,
        offset=int(offset),
    )
    state: Dict[str, Any] = {
        "motion": motion,
        "motion_raw": motion_raw,
        "y_prev_raw": y_prev_raw,
        "plan_z": None,
        "meas_logits_prev": None,
        "rot_slice": rot_slice,
        "pose_hist_state": pose_hist_state,
    }

    step_weights = _make_rollout_step_weights(
        total_steps,
        device=device,
        dtype=dtype,
        mode=str(time_weight_mode or "inv"),
        max_val=float(time_weight_max or 1.0),
    )
    boundary_steps = 0
    boundary_weighted_sum = 0.0
    if include_boundary:
        try:
            idxs = (torch.arange(int(total_steps), device=device) + int(offset)) % int(cycle_len)
            boundary_mask = idxs == (int(cycle_len) - 1)
            boundary_steps = int(boundary_mask.sum().detach().cpu().item())
            bw = float(boundary_weight or 0.0)
            bw = max(0.0, bw)
            if abs(bw - 1.0) > 1e-12:
                factors = torch.ones_like(step_weights)
                factors = torch.where(boundary_mask, step_weights.new_tensor(bw), factors)
                step_weights = step_weights * factors
                step_weights = step_weights / step_weights.sum().clamp_min(1e-6)
            boundary_weighted_sum = float(step_weights[boundary_mask].sum().detach().cpu().item())
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "prep_boundary_weights"); boundary_steps = 0; boundary_weighted_sum = 0.0

    return LambdaRolloutPrepContext(
        device=device,
        dtype=dtype,
        motion_seq=motion_seq,
        gt_seq=gt_seq,
        cond_seq=cond_seq,
        cond_raw_tgt=cond_raw_tgt,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        contacts_seq=contacts_seq,
        angvel_seq=angvel_seq,
        pose_hist_seq=pose_hist_seq,
        B=int(B),
        T=int(T),
        Dy=int(Dy),
        steps=int(steps),
        rollout_cycles=int(rollout_cycles),
        include_boundary=bool(include_boundary),
        cycle_len=int(cycle_len),
        total_steps=int(total_steps),
        offset=int(offset),
        y0_raw=y0_raw,
        rot_slice=rot_slice,
        rot_len=int(rot_len),
        J=int(J),
        std_y=std_y,
        state=state,
        step_weights=step_weights,
        boundary_steps=int(boundary_steps),
        boundary_weighted_sum=float(boundary_weighted_sum),
    )


def _lambda_rollout_resolve_nonleg_focus(
    trainer: Trainer,
    *,
    objective: str,
    direct_pose_nonleg_focus_bones: str,
    direct_pose_nonleg_focus_weight: float,
    J: int,
    device: torch.device,
) -> LambdaRolloutNonLegFocusContext:
    # Optional: focus selected non-leg bones by increasing their contribution inside L_nonleg.
    direct_nonleg_focus_mask_j = None
    direct_nonleg_focus_requested = 0
    direct_nonleg_focus_resolved = 0
    direct_nonleg_focus_weight_use = 1.0
    direct_nonleg_focus_applied = 0.0
    try:
        direct_nonleg_focus_weight_use = float(direct_pose_nonleg_focus_weight or 1.0)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "resolve_nonleg_weight"); direct_nonleg_focus_weight_use = 1.0
    if (not math.isfinite(direct_nonleg_focus_weight_use)) or direct_nonleg_focus_weight_use <= 0.0:
        direct_nonleg_focus_weight_use = 1.0
    try:
        focus_spec = str(direct_pose_nonleg_focus_bones or "").strip()
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "resolve_nonleg_spec"); focus_spec = ""
    focus_tokens = [s.strip() for s in focus_spec.replace(";", ",").split(",") if s.strip()]
    direct_nonleg_focus_requested = int(len(focus_tokens))
    if objective == "direct" and focus_tokens and abs(float(direct_nonleg_focus_weight_use) - 1.0) > 1e-12:
        try:
            name_to_idx: dict[str, int] = {}
            loss_fn_obj = getattr(trainer, "loss_fn", None)
            bone_names = getattr(loss_fn_obj, "bone_names", None)
            if (not isinstance(bone_names, (list, tuple))) or (len(bone_names) <= 0):
                meta = getattr(loss_fn_obj, "meta", None)
                if isinstance(meta, dict):
                    sk = meta.get("skeleton", None)
                    if isinstance(sk, dict):
                        names_meta = sk.get("bone_names", None)
                        if isinstance(names_meta, (list, tuple)) and len(names_meta) > 0:
                            bone_names = names_meta
            if isinstance(bone_names, (list, tuple)):
                for i_name, nm in enumerate(bone_names):
                    key = str(nm).strip()
                    if not key:
                        continue
                    name_to_idx[key] = int(i_name)
                    name_to_idx[key.lower()] = int(i_name)
            idx_set = set()
            for tok in focus_tokens:
                idx_t = None
                if tok.lstrip("-").isdigit():
                    idx_t = int(tok)
                else:
                    idx_t = name_to_idx.get(tok, name_to_idx.get(tok.lower(), None))
                if idx_t is None:
                    continue
                if 0 <= int(idx_t) < int(J):
                    idx_set.add(int(idx_t))
            if idx_set:
                mask_j = torch.zeros((J,), device=device, dtype=torch.bool)
                idx_sorted = sorted(int(i) for i in idx_set)
                mask_j[torch.as_tensor(idx_sorted, device=device, dtype=torch.long)] = True
                direct_nonleg_focus_mask_j = mask_j
                direct_nonleg_focus_resolved = int(mask_j.sum().detach().cpu().item())
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "resolve_nonleg_index"); direct_nonleg_focus_mask_j = None; direct_nonleg_focus_resolved = 0

    return LambdaRolloutNonLegFocusContext(
        direct_nonleg_focus_mask_j=direct_nonleg_focus_mask_j,
        direct_nonleg_focus_requested=int(direct_nonleg_focus_requested),
        direct_nonleg_focus_resolved=int(direct_nonleg_focus_resolved),
        direct_nonleg_focus_weight_use=float(direct_nonleg_focus_weight_use),
        direct_nonleg_focus_applied=float(direct_nonleg_focus_applied),
    )


def _lambda_rollout_build_reg_params(
    trainer: Trainer,
    *,
    objective: str,
    lambda_gate_sup_weight: float,
    lambda_gate_sup_start_step: int,
    lambda_gate_sup_tau_deg: float,
    lambda_gate_sup_margin_deg: float,
    direct_pose_loss_group_norm_enable: bool,
    direct_pose_loss_group_norm_w_leg: float,
    direct_pose_loss_group_norm_w_nonleg: float,
    direct_pose_loss_group_norm_ema_beta: float,
    direct_pose_loss_group_norm_ratio_min: float,
    direct_pose_loss_group_norm_ratio_max: float,
    direct_pose_loss_group_norm_eps: float,
) -> LambdaRolloutRegParams:
    # ---- Stage2: optional gate supervision (λ logits) ----
    gate_sup_weight = float(lambda_gate_sup_weight or 0.0)
    gate_sup_start = int(lambda_gate_sup_start_step or 0)
    if gate_sup_start < 0:
        # Auto: start after reliability warmup, when enabled, to avoid train/infer mismatch.
        mode = str(getattr(trainer, "lambda_reliability_mode", "none") or "none").strip().lower()
        tokens = [s.strip() for s in mode.replace(",", "+").split("+") if s.strip()]
        warmup_steps = int(getattr(trainer, "lambda_reliability_warmup_steps", 0) or 0)
        if ("warmup" in tokens or "step_warmup" in tokens) and warmup_steps > 0:
            gate_sup_start = warmup_steps
        else:
            gate_sup_start = 0
    gate_sup_start = max(0, int(gate_sup_start))
    tau_deg = float(lambda_gate_sup_tau_deg) if lambda_gate_sup_tau_deg is not None else 2.5
    tau_rad = max(1e-6, tau_deg * (math.pi / 180.0))
    margin_deg = float(lambda_gate_sup_margin_deg) if lambda_gate_sup_margin_deg is not None else 1.0
    margin_rad = max(0.0, margin_deg * (math.pi / 180.0))

    # ---- Stage7/B2: optional legs vs non-legs group-wise magnitude normalization ----
    direct_group_norm_enable = bool(objective == "direct" and bool(direct_pose_loss_group_norm_enable))
    try:
        direct_group_w_leg = float(direct_pose_loss_group_norm_w_leg or 1.0)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_w_leg"); direct_group_w_leg = 1.0
    try:
        direct_group_w_nonleg = float(direct_pose_loss_group_norm_w_nonleg or 1.0)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_w_nonleg"); direct_group_w_nonleg = 1.0
    try:
        direct_group_beta = float(direct_pose_loss_group_norm_ema_beta or 0.95)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_beta"); direct_group_beta = 0.95
    if (not math.isfinite(direct_group_beta)) or direct_group_beta < 0.0:
        direct_group_beta = 0.95
    direct_group_beta = max(0.0, min(0.9999, float(direct_group_beta)))
    try:
        direct_group_ratio_min = float(direct_pose_loss_group_norm_ratio_min or 0.2)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_ratio_min"); direct_group_ratio_min = 0.2
    try:
        direct_group_ratio_max = float(direct_pose_loss_group_norm_ratio_max or 5.0)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_ratio_max"); direct_group_ratio_max = 5.0
    if (not math.isfinite(direct_group_ratio_min)) or direct_group_ratio_min <= 0.0:
        direct_group_ratio_min = 0.2
    if (not math.isfinite(direct_group_ratio_max)) or direct_group_ratio_max <= 0.0:
        direct_group_ratio_max = 5.0
    if direct_group_ratio_min > direct_group_ratio_max:
        direct_group_ratio_min, direct_group_ratio_max = direct_group_ratio_max, direct_group_ratio_min
    try:
        direct_group_eps = float(direct_pose_loss_group_norm_eps or 1e-6)
    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "reg_group_eps"); direct_group_eps = 1e-6
    if (not math.isfinite(direct_group_eps)) or direct_group_eps <= 0.0:
        direct_group_eps = 1e-6

    return LambdaRolloutRegParams(
        gate_sup_weight=float(gate_sup_weight),
        gate_sup_start=int(gate_sup_start),
        tau_rad=float(tau_rad),
        margin_rad=float(margin_rad),
        direct_group_norm_enable=bool(direct_group_norm_enable),
        direct_group_w_leg=float(direct_group_w_leg),
        direct_group_w_nonleg=float(direct_group_w_nonleg),
        direct_group_beta=float(direct_group_beta),
        direct_group_ratio_min=float(direct_group_ratio_min),
        direct_group_ratio_max=float(direct_group_ratio_max),
        direct_group_eps=float(direct_group_eps),
    )


def _sanitize_metric_key_suffix(name: str, *, default: str) -> str:
    key_suffix = "".join(ch if str(ch).isalnum() else "_" for ch in str(name).strip())
    while "__" in key_suffix:
        key_suffix = key_suffix.replace("__", "_")
    key_suffix = key_suffix.strip("_")
    return key_suffix or str(default)


def _resolve_leg_align_joint_names(
    *,
    model: EventMotionModel,
    expected_count: int,
    keep_mask: Optional[torch.Tensor],
) -> list[str]:
    joint_names = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
    if int(expected_count) <= 0:
        return []
    if len(joint_names) != int(expected_count):
        return [f"j{i}" for i in range(int(expected_count))]
    if torch.is_tensor(keep_mask) and int(keep_mask.numel()) == int(len(joint_names)):
        try:
            keep_flags = [bool(v) for v in keep_mask.detach().cpu().tolist()]
            joint_names = [name for name, keep in zip(joint_names, keep_flags) if bool(keep)]
        except Exception:
            pass
    if len(joint_names) != int(expected_count):
        return [f"j{i}" for i in range(int(expected_count))]
    return [str(name) for name in joint_names]


def _resolve_leg_align_selector_joints(
    spec: Optional[str],
    *,
    joint_names: list[str],
) -> list[str]:
    joint_names_use = [str(name) for name in joint_names]
    if not joint_names_use:
        return []
    if spec is None:
        return list(joint_names_use)
    raw = str(spec).replace("|", ",").replace("+", ",")
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        return list(joint_names_use)
    selected: set[str] = set()
    for token in tokens:
        group = _LEG_ALIGN_SELECTOR_GROUPS.get(token, None)
        if group is not None:
            selected.update(str(name).lower() for name in group)
        else:
            selected.add(str(token).lower())
    return [name for name in joint_names_use if str(name).lower() in selected]


def _compute_leg_align_subset_term(
    *,
    per: torch.Tensor,
    w: torch.Tensor,
    joint_names: list[str],
    target_joints: Iterable[str],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if per.dim() != 2 or w.dim() != 2:
        return None, None
    if int(per.shape[1]) <= 0 or int(w.shape[1]) != int(per.shape[1]) or len(joint_names) != int(per.shape[1]):
        return None, None
    target_names = [str(name).lower() for name in target_joints]
    if not target_names:
        return None, None
    target_set = set(target_names)
    mask_vals = [str(name).lower() in target_set for name in joint_names]
    if not any(mask_vals):
        return None, None
    mask = torch.tensor(mask_vals, device=per.device, dtype=torch.bool)
    mask_f = mask.to(dtype=per.dtype).view(1, -1)
    w_group = w * mask_f
    denom = w_group.sum().clamp_min(1.0)
    loss = (per * w_group).sum() / denom
    frac = (w[:, mask] > 0.0).to(dtype=dtype).mean()
    return loss, frac


def _lambda_rollout_apply_direct_leg_adjustments(
    *, trainer: Trainer, model: EventMotionModel, ret: Dict[str, Any], direct_raw_base: torch.Tensor, R_gt: torch.Tensor,
    B: int, J: int, device: torch.device, dtype: torch.dtype, columns: Tuple[str, str], rot_slice: slice, rot_len: int,
    direct_pose_leg_align_weight: float, direct_pose_leg_align_oracle_min_deg: float, direct_pose_leg_align_oracle_weight_deg: float,
    direct_pose_leg_align_mode: str, direct_pose_leg_align_mag_weight: float, direct_pose_leg_align_res_weight: float,
    direct_pose_leg_align_sign_weight: float, direct_pose_leg_align_cos_thresh: float,
    direct_pose_leg_align_target_joints: Optional[str], direct_pose_leg_align_anchor_joints: Optional[str],
    direct_pose_leg_align_anchor_weight: float, direct_pose_leg_gate_sup_weight: float,
    step_weight: torch.Tensor, leg_align_terms: list[torch.Tensor],
    leg_align_frac_terms: list[torch.Tensor], leg_align_joint_num_terms: list[torch.Tensor],
    leg_align_joint_den_terms: list[torch.Tensor], leg_align_joint_frac_terms: list[torch.Tensor],
    leg_align_distal_terms: list[torch.Tensor], leg_align_distal_frac_terms: list[torch.Tensor],
    leg_align_proximal_terms: list[torch.Tensor], leg_align_proximal_frac_terms: list[torch.Tensor],
    leg_align_anchor_terms: list[torch.Tensor], leg_align_anchor_frac_terms: list[torch.Tensor],
    leg_gate_sup_terms: list[torch.Tensor], leg_gate_sup_tgt_frac_terms: list[torch.Tensor],
    leg_gate_sup_pred_mean_terms: list[torch.Tensor],
) -> torch.Tensor:
    try:
        omega_leg = ret.get("direct_leg_omega", None)
        if not torch.is_tensor(omega_leg):
            return direct_raw_base
        if omega_leg.dim() == 4 and omega_leg.size(1) == 1:
            omega_leg = omega_leg[:, 0]
        leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
        if not (torch.is_tensor(leg_idx) and int(leg_idx.numel()) > 0):
            return direct_raw_base
        idx_use = leg_idx.to(device=device)
        if not (omega_leg.dim() == 3 and omega_leg.shape[0] == B and omega_leg.shape[-1] == 3):
            return direct_raw_base
        if int(omega_leg.shape[1]) != int(idx_use.numel()):
            return direct_raw_base

        keep_mask = None
        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
        if 0 <= root_idx < J and bool((idx_use == int(root_idx)).any().detach().cpu().item()):
            keep_mask = (idx_use != int(root_idx))
            if bool(keep_mask.any().detach().cpu().item()):
                idx_use = idx_use[keep_mask]
                omega_leg = omega_leg[:, keep_mask, :]
        if int(idx_use.numel()) <= 0:
            return direct_raw_base

        base6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J, 6)
        R_base = rot6d_to_matrix(base6, columns=columns)
        R_leg_base = R_base[:, idx_use, :, :]
        if bool(getattr(model, "direct_pose_leg_stopgrad_main", False)):
            R_leg_base = R_leg_base.detach()

        omega_oracle = None
        oracle_norm = None
        if float(direct_pose_leg_align_weight or 0.0) > 0.0:
            try:
                with torch.no_grad():
                    R_gt_leg = R_gt[:, idx_use, :, :]
                    R_delta_oracle = torch.matmul(R_gt_leg, R_leg_base.transpose(-1, -2))
                    omega_oracle = so3_log_map(R_delta_oracle)
                    oracle_norm = omega_oracle.norm(dim=-1)
                    min_rad = float(direct_pose_leg_align_oracle_min_deg or 0.0) * (math.pi / 180.0)
                    w = (oracle_norm > float(min_rad)).to(dtype=dtype)
                    w_deg = float(direct_pose_leg_align_oracle_weight_deg or 0.0)
                    if w_deg > 0.0 and math.isfinite(w_deg):
                        w = w * (oracle_norm / (float(w_deg) * (math.pi / 180.0))).clamp(0.0, 1.0)
                    w = w.detach()

                p = omega_leg.to(device=device, dtype=dtype)
                try:
                    cos_thr = float(direct_pose_leg_align_cos_thresh or 0.0)
                except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "leg_align_cos_thresh"); cos_thr = 0.0
                if 0.0 < cos_thr < 1.0:
                    with torch.no_grad():
                        dot = (p * omega_oracle).sum(dim=-1)
                        den = (p.norm(dim=-1) * omega_oracle.norm(dim=-1)).clamp_min(1e-8)
                        w = (w * ((dot / den).clamp(-1.0, 1.0) < float(cos_thr)).to(dtype=dtype)).detach()

                align_mode = str(direct_pose_leg_align_mode or "cos").strip().lower()
                if align_mode not in ("cos", "proj"):
                    align_mode = "cos"
                if align_mode == "cos":
                    dot = (p * omega_oracle).sum(dim=-1)
                    den = (p.norm(dim=-1) * omega_oracle.norm(dim=-1)).clamp_min(1e-8)
                    per = F.relu(-(dot / den).clamp(-1.0, 1.0))
                else:
                    oracle_norm_safe = oracle_norm.clamp_min(1e-8)
                    oracle_dir = omega_oracle / oracle_norm_safe.unsqueeze(-1)
                    proj = (p * oracle_dir).sum(dim=-1)
                    res = p - proj.unsqueeze(-1) * oracle_dir
                    mag_w = float(direct_pose_leg_align_mag_weight or 1.0)
                    res_w = float(direct_pose_leg_align_res_weight or 1.0)
                    sign_w = float(direct_pose_leg_align_sign_weight or 0.0)
                    mag_w = 1.0 if (not math.isfinite(mag_w)) or mag_w < 0.0 else mag_w
                    res_w = 1.0 if (not math.isfinite(res_w)) or res_w < 0.0 else res_w
                    sign_w = 0.0 if (not math.isfinite(sign_w)) or sign_w < 0.0 else sign_w
                    per = (mag_w * (proj - oracle_norm).pow(2)) + (res_w * res.pow(2).sum(dim=-1))
                    if sign_w > 0.0:
                        per = per + (sign_w * F.relu(-proj).pow(2))

                denom = w.sum().clamp_min(1.0)
                leg_align_terms.append(((per * w).sum() / denom) * step_weight)
                leg_align_frac_terms.append((w > 0.0).to(dtype=dtype).mean() * step_weight)
                joint_names_use = _resolve_leg_align_joint_names(model=model, expected_count=int(per.shape[1]), keep_mask=keep_mask)
                main_target_joints = _resolve_leg_align_selector_joints(direct_pose_leg_align_target_joints, joint_names=joint_names_use)
                main_loss, main_frac = _compute_leg_align_subset_term(per=per, w=w, joint_names=joint_names_use, target_joints=main_target_joints, dtype=dtype)
                if torch.is_tensor(main_loss):
                    leg_align_terms.append(main_loss * step_weight)
                else:
                    _record_posttrain_soft_fail(trainer, "leg_align_target_spec_empty")
                if torch.is_tensor(main_frac):
                    leg_align_frac_terms.append(main_frac * step_weight)
                for target_joints, loss_terms, frac_terms in (
                    (_LEG_ALIGN_DISTAL_JOINTS, leg_align_distal_terms, leg_align_distal_frac_terms),
                    (_LEG_ALIGN_PROXIMAL_JOINTS, leg_align_proximal_terms, leg_align_proximal_frac_terms),
                ):
                    loss_group, frac_group = _compute_leg_align_subset_term(per=per, w=w, joint_names=joint_names_use, target_joints=target_joints, dtype=dtype)
                    if torch.is_tensor(loss_group):
                        loss_terms.append(loss_group * step_weight)
                    if torch.is_tensor(frac_group):
                        frac_terms.append(frac_group * step_weight)
                try:
                    anchor_weight = float(direct_pose_leg_align_anchor_weight or 0.0)
                except (TypeError, ValueError):
                    anchor_weight = 0.0
                if anchor_weight > 0.0 and direct_pose_leg_align_anchor_joints is not None:
                    anchor_target_joints = _resolve_leg_align_selector_joints(direct_pose_leg_align_anchor_joints, joint_names=joint_names_use)
                    anchor_loss, anchor_frac = _compute_leg_align_subset_term(per=per, w=w, joint_names=joint_names_use, target_joints=anchor_target_joints, dtype=dtype)
                    if torch.is_tensor(anchor_loss):
                        leg_align_terms.append((anchor_weight * anchor_loss) * step_weight)
                        leg_align_anchor_terms.append(anchor_loss * step_weight)
                    else:
                        _record_posttrain_soft_fail(trainer, "leg_align_anchor_spec_empty")
                    if torch.is_tensor(anchor_frac):
                        leg_align_anchor_frac_terms.append(anchor_frac * step_weight)
                elif anchor_weight > 0.0:
                    _record_posttrain_soft_fail(trainer, "leg_align_anchor_missing_joints")
                leg_align_joint_num_terms.append(((per * w).sum(dim=0)).detach() * step_weight.detach())
                leg_align_joint_den_terms.append((w.sum(dim=0)).detach() * step_weight.detach())
                leg_align_joint_frac_terms.append(((w > 0.0).to(dtype=dtype).mean(dim=0)).detach() * step_weight.detach())
            except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "leg_align_supervision")

        if float(direct_pose_leg_gate_sup_weight or 0.0) > 0.0:
            try:
                gate_logits = ret.get("direct_leg_gate_logits", None)
                if torch.is_tensor(gate_logits):
                    if gate_logits.dim() == 3:
                        gate_logits = gate_logits[:, -1]
                    if gate_logits.dim() == 2 and gate_logits.shape[0] == B:
                        if torch.is_tensor(keep_mask) and gate_logits.shape[1] == keep_mask.shape[0]:
                            gate_logits = gate_logits[:, keep_mask]
                        if int(gate_logits.shape[1]) == int(idx_use.numel()):
                            gl = gate_logits.to(device=device, dtype=dtype)
                            if oracle_norm is None:
                                with torch.no_grad():
                                    R_delta_oracle = torch.matmul(R_gt[:, idx_use, :, :], R_leg_base.transpose(-1, -2))
                                    omega_oracle = so3_log_map(R_delta_oracle)
                                    oracle_norm = omega_oracle.norm(dim=-1)
                            if torch.is_tensor(oracle_norm):
                                with torch.no_grad():
                                    min_rad = float(direct_pose_leg_align_oracle_min_deg or 0.0) * (math.pi / 180.0)
                                    tgt = (oracle_norm >= float(min_rad)).to(device=device, dtype=dtype)
                                err = F.binary_cross_entropy_with_logits(gl, tgt, reduction="none")
                                leg_gate_sup_terms.append(err.mean() * step_weight)
                                leg_gate_sup_tgt_frac_terms.append(tgt.mean() * step_weight)
                                leg_gate_sup_pred_mean_terms.append(torch.sigmoid(gl).mean() * step_weight)
            except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "leg_gate_supervision")

        R_leg = torch.matmul(so3_exp_map(omega_leg), R_leg_base)
        R_final = R_base.clone()
        R_final[:, idx_use, :, :] = R_leg
        rot6_final = matrix_to_rot6d(R_final, columns=columns).view(B, rot_len)
        direct_raw_base = direct_raw_base.clone()
        direct_raw_base[..., rot_slice] = rot6_final
    except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "leg_adjustment_main")

    return direct_raw_base


def _lambda_rollout_decode_model_outputs(*, ret: Dict[str, Any], objective: str, B: int, J: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_norm = ret.get("out")
    direct_norm = ret.get("out_direct")
    lam = ret.get("lambda_fusion")
    if delta_norm is None or direct_norm is None:
        raise RuntimeError("Model dict output missing required keys: out / out_direct.")
    if lam is None:
        if objective == "blend":
            raise RuntimeError("Model dict output missing required key: lambda_fusion (needed for objective='blend').")
        lam = delta_norm.new_zeros((B, J))
    if delta_norm.dim() == 3:
        delta_norm = delta_norm[:, -1]
    if direct_norm.dim() == 3:
        direct_norm = direct_norm[:, -1]
    if lam.dim() == 3:
        lam = lam[:, -1]
    return delta_norm, direct_norm, lam


def _lambda_rollout_accumulate_plan_terms(*, trainer: Trainer, ret: Dict[str, Any], weights: LambdaRolloutWeights, accum: LambdaFusionAccum, lam_eff: torch.Tensor, plan_prev: Optional[torch.Tensor], step_weight: torch.Tensor, B: int) -> Optional[torch.Tensor]:
    plan_step = None
    try:
        plan_step = ret.get("contacts_plan", None)
        if torch.is_tensor(plan_step):
            if plan_step.dim() == 3:
                plan_step = plan_step[:, -1]
            if plan_step.dim() != 2:
                plan_step = None
    except (AttributeError, TypeError):
        _record_posttrain_soft_fail(trainer, "unroll_contacts_plan_decode")
        plan_step = None

    if (float(weights.lambda_plan_entropy_weight or 0.0) <= 0.0 and float(weights.lambda_plan_dyn_weight or 0.0) <= 0.0) or (not torch.is_tensor(plan_step)):
        return plan_prev

    lam_eff_mean = lam_eff.mean(dim=-1)
    try:
        plan_det = plan_step.detach()
        ent = _lambda_entropy(plan_det).mean(dim=-1)
        accum.plan_ent_stat_terms.append(ent.mean() * step_weight)
        if float(weights.lambda_plan_entropy_weight or 0.0) > 0.0:
            accum.plan_ent_terms.append((lam_eff_mean * ent).mean() * step_weight)
    except _ROLLOUT_SOFT_FAIL_ERRORS:
        _record_posttrain_soft_fail(trainer, "unroll_plan_entropy")
    try:
        plan_det = plan_step.detach()
        dyn = (plan_det - plan_prev).abs().mean(dim=-1) if torch.is_tensor(plan_prev) and plan_prev.shape == plan_step.shape else plan_det.new_zeros((B,))
        accum.plan_dyn_stat_terms.append(dyn.mean() * step_weight)
        if float(weights.lambda_plan_dyn_weight or 0.0) > 0.0:
            accum.plan_dyn_terms.append((lam_eff_mean * dyn).mean() * step_weight)
        return plan_det
    except _ROLLOUT_SOFT_FAIL_ERRORS:
        _record_posttrain_soft_fail(trainer, "unroll_plan_dynamics")
        return plan_step.detach()


def _lambda_rollout_accumulate_direct_objective(*, trainer: Trainer, model: EventMotionModel, weights: LambdaRolloutWeights, accum: LambdaFusionAccum, objective: str, e_dir: torch.Tensor, step_weight: torch.Tensor, J: int, direct_nonleg_focus_applied: float) -> float:
    e_dir_use = e_dir.mean()
    if objective == "direct":
        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
        root_idx = root_idx if 0 <= root_idx < J else 0
        nr_mask = None
        if J > 1 and 0 <= root_idx < J:
            nr_mask = torch.ones((J,), device=e_dir.device, dtype=torch.bool)
            nr_mask[root_idx] = False
            e = e_dir[:, nr_mask]
        else:
            e = e_dir
        e_dir_use = e.mean()
        L_leg_base = L_nonleg_base = L_nonleg_plain = None
        if bool(weights.direct_pose_loss_leg_split):
            leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
            if torch.is_tensor(leg_idx) and int(leg_idx.numel()) > 0:
                try:
                    leg_mask = torch.zeros((J,), device=e_dir.device, dtype=torch.bool)
                    leg_mask[leg_idx.to(device=e_dir.device)] = True
                    if J > 1 and 0 <= root_idx < J:
                        leg_mask[root_idx] = False
                    if torch.is_tensor(nr_mask) and nr_mask.shape == leg_mask.shape:
                        leg_mask = leg_mask[nr_mask]
                    if bool(leg_mask.any().detach().cpu().item()) and bool((~leg_mask).any().detach().cpu().item()):
                        e_leg, e_nonleg = e[:, leg_mask], e[:, ~leg_mask]
                        L_leg_base = e_leg.mean()
                        L_nonleg_plain = e_nonleg.mean()
                        L_nonleg_base = L_nonleg_plain
                        if torch.is_tensor(weights.direct_nonleg_focus_mask_j) and int(weights.direct_nonleg_focus_resolved) > 0 and abs(float(weights.direct_nonleg_focus_weight_use) - 1.0) > 1e-12:
                            focus_mask = weights.direct_nonleg_focus_mask_j[nr_mask] if torch.is_tensor(nr_mask) and nr_mask.shape == weights.direct_nonleg_focus_mask_j.shape else weights.direct_nonleg_focus_mask_j
                            if focus_mask.shape == leg_mask.shape:
                                focus_nonleg = focus_mask[~leg_mask]
                                if bool(focus_nonleg.any().detach().cpu().item()):
                                    w_non = torch.ones((int(e_nonleg.shape[-1]),), device=e_nonleg.device, dtype=e_nonleg.dtype)
                                    w_non = torch.where(focus_nonleg, w_non * w_non.new_tensor(float(weights.direct_nonleg_focus_weight_use)), w_non)
                                    L_nonleg_base = ((e_nonleg * w_non.unsqueeze(0)).sum(dim=-1) / w_non.sum().clamp_min(1e-6)).mean()
                                    direct_nonleg_focus_applied = 1.0
                        e_dir_use = L_nonleg_base + L_leg_base
                except _ROLLOUT_SOFT_FAIL_ERRORS:
                    _record_posttrain_soft_fail(trainer, "unroll_direct_leg_split")
        accum.dir_base_terms.append(e_dir_use * step_weight)
        if torch.is_tensor(L_leg_base):
            accum.dir_leg_base_terms.append(L_leg_base * step_weight)
        if torch.is_tensor(L_nonleg_base):
            accum.dir_nonleg_base_terms.append(L_nonleg_base * step_weight)
        if torch.is_tensor(L_nonleg_plain):
            accum.dir_nonleg_plain_terms.append(L_nonleg_plain * step_weight)
    accum.dir_terms.append(e_dir_use * step_weight)
    return float(direct_nonleg_focus_applied)


def _lambda_rollout_accumulate_gate_supervision(*, trainer: Trainer, ret: Dict[str, Any], accum: LambdaFusionAccum, weights: LambdaRolloutWeights, e_inc: torch.Tensor, e_dir: torch.Tensor, step_weight: torch.Tensor, t: int, B: int, J: int) -> None:
    if float(weights.gate_sup_weight) <= 0.0 or int(t) < int(weights.gate_sup_start):
        return
    lam_logits = ret.get("lambda_fusion_logits", None)
    if not torch.is_tensor(lam_logits):
        return
    if lam_logits.dim() == 3:
        lam_logits = lam_logits[:, -1]
    if lam_logits.dim() != 2 or lam_logits.shape[0] != B:
        return
    try:
        with torch.no_grad():
            delta = (e_inc - e_dir).detach()
            lam_star = torch.sigmoid(delta / float(weights.tau_rad)).detach()
            mask = (delta.abs() >= float(weights.margin_rad)).to(dtype=lam_star.dtype) if float(weights.margin_rad) > 0.0 else torch.ones_like(lam_star)
        if lam_logits.shape[-1] == 1:
            lam_star, mask = lam_star.mean(dim=-1, keepdim=True), mask.mean(dim=-1, keepdim=True)
        elif lam_logits.shape[-1] != J:
            lam_star = None
        if lam_star is None:
            return
        lam_star = lam_star.to(device=lam_logits.device, dtype=lam_logits.dtype)
        mask = mask.to(device=lam_logits.device, dtype=lam_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(lam_logits, lam_star, reduction="none")
        mask_sum = mask.sum()
        accum.gate_sup_terms.append(((bce * mask).sum() / mask_sum.clamp_min(1e-6)) * step_weight)
        accum.gate_sup_frac_terms.append(mask.mean() * step_weight)
        with torch.no_grad():
            pred = (torch.sigmoid(lam_logits) > 0.5).to(dtype=mask.dtype)
            tgt = (lam_star > 0.5).to(dtype=mask.dtype)
            accum.gate_sup_acc_num_terms.append(((pred == tgt).to(dtype=mask.dtype) * mask).sum() * step_weight)
            accum.gate_sup_acc_den_terms.append(mask_sum * step_weight)
    except _ROLLOUT_SOFT_FAIL_ERRORS:
        _record_posttrain_soft_fail(trainer, "unroll_gate_supervision")


def _lambda_rollout_unroll_single_step(*, t: int, ctx: LambdaRolloutStepContext) -> None:
    runtime = ctx.runtime
    data = ctx.data
    weights = ctx.weights
    accum = ctx.accum
    state_vars = ctx.state_vars

    trainer = runtime.trainer
    model = runtime.model
    state = runtime.state
    total_steps = int(runtime.total_steps)
    cycle_len = int(runtime.cycle_len)
    include_boundary = bool(runtime.include_boundary)
    steps = int(runtime.steps)
    offset = int(runtime.offset)
    B = int(runtime.B)
    J = int(runtime.J)
    objective = str(runtime.objective)
    rot_len = int(runtime.rot_len)
    rot_slice = data.rot_slice
    step_weights = data.step_weights

    meas_used_logits = bool(state_vars.meas_used_logits)
    direct_nonleg_focus_applied = float(state_vars.direct_nonleg_focus_applied)
    lam_prev = state_vars.lam_prev
    lam_prev_monot = state_vars.lam_prev_monot
    plan_prev = state_vars.plan_prev

    denom = cycle_len if include_boundary else steps
    idx = int((offset + int(t)) % max(1, int(denom)))
    step_common = _rollout_step_common(
        trainer,
        model,
        state=state,
        t=int(t),
        idx=int(idx),
        total_steps=total_steps,
        cond_seq=data.cond_seq,
        cond_raw_tgt=data.cond_raw_tgt,
        cond_norm_mu=data.cond_norm_mu,
        cond_norm_std=data.cond_norm_std,
        angvel_seq=data.angvel_seq,
        pose_hist_seq=data.pose_hist_seq,
        time_index_mode=runtime.time_index_mode,
        time_base=runtime.time_base,
        enable_reprojection=bool(runtime.enable_reprojection),
        include_boundary=include_boundary,
        cycle_len=cycle_len,
        yaw_gt_fn=runtime.yaw_gt_fn,
        detach_rollout_state=bool(runtime.detach_rollout_state),
    )
    ret = step_common["ret"]
    contacts_in_t = step_common["contacts_in_t"]
    cond_raw_step = step_common["cond_raw_step"]
    rollout_step_t = step_common["rollout_step_t"]
    y_prev_raw = state["y_prev_raw"]

    delta_norm, direct_norm, lam = _lambda_rollout_decode_model_outputs(ret=ret, objective=objective, B=B, J=J)

    if contacts_in_t is None and float(weights.contact_meas_weight or 0.0) > 0.0 and torch.is_tensor(data.contacts_seq):
        try:
            contacts_seq = data.contacts_seq
            gt_c_t = contacts_seq[:, idx] if contacts_seq.dim() == 3 else contacts_seq
            meas_logits = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(meas_logits):
                if meas_logits.dim() == 3:
                    meas_logits = meas_logits[:, -1]
                if torch.is_tensor(gt_c_t) and gt_c_t.shape == meas_logits.shape:
                    meas_used_logits = True
                    gt = gt_c_t.clamp(0.0, 1.0)
                    accum.meas_terms.append(F.binary_cross_entropy_with_logits(meas_logits, gt) * step_weights[t])
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_contact_meas_logits")

    delta_raw = delta_norm * data.std_y
    prev6 = reproject_rot6d(y_prev_raw[..., rot_slice]).view(B, J, 6)
    R_prev = rot6d_to_matrix(prev6, columns=runtime.columns)

    if include_boundary and runtime.y0_raw is not None and int(idx) == (cycle_len - 1):
        gt_raw = runtime.y0_raw
    else:
        gt_raw = trainer._denorm(runtime.gt_seq[:, idx])
    gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(B, J, 6)
    R_gt = rot6d_to_matrix(gt6, columns=runtime.columns)

    delta6 = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=runtime.columns)
    R_delta = rot6d_to_matrix(delta6, columns=runtime.columns)
    R_inc = torch.matmul(R_delta, R_prev)

    direct_raw_base = trainer._denorm(direct_norm)
    direct_raw_base = _lambda_rollout_apply_direct_leg_adjustments(
        trainer=trainer,
        model=model,
        ret=ret,
        direct_raw_base=direct_raw_base,
        R_gt=R_gt,
        B=B,
        J=J,
        device=runtime.device,
        dtype=runtime.dtype,
        columns=runtime.columns,
        rot_slice=rot_slice,
        rot_len=rot_len,
        direct_pose_leg_align_weight=float(weights.direct_pose_leg_align_weight),
        direct_pose_leg_align_oracle_min_deg=float(weights.direct_pose_leg_align_oracle_min_deg),
        direct_pose_leg_align_oracle_weight_deg=float(weights.direct_pose_leg_align_oracle_weight_deg),
        direct_pose_leg_align_mode=str(weights.direct_pose_leg_align_mode),
        direct_pose_leg_align_mag_weight=float(weights.direct_pose_leg_align_mag_weight),
        direct_pose_leg_align_res_weight=float(weights.direct_pose_leg_align_res_weight),
        direct_pose_leg_align_sign_weight=float(weights.direct_pose_leg_align_sign_weight),
        direct_pose_leg_align_cos_thresh=float(weights.direct_pose_leg_align_cos_thresh),
        direct_pose_leg_align_target_joints=weights.direct_pose_leg_align_target_joints,
        direct_pose_leg_align_anchor_joints=weights.direct_pose_leg_align_anchor_joints,
        direct_pose_leg_align_anchor_weight=float(weights.direct_pose_leg_align_anchor_weight),
        direct_pose_leg_gate_sup_weight=float(weights.direct_pose_leg_gate_sup_weight),
        step_weight=step_weights[t],
        leg_align_terms=accum.leg_align_terms,
        leg_align_frac_terms=accum.leg_align_frac_terms,
        leg_align_joint_num_terms=accum.leg_align_joint_num_terms,
        leg_align_joint_den_terms=accum.leg_align_joint_den_terms,
        leg_align_joint_frac_terms=accum.leg_align_joint_frac_terms,
        leg_align_distal_terms=accum.leg_align_distal_terms,
        leg_align_distal_frac_terms=accum.leg_align_distal_frac_terms,
        leg_align_proximal_terms=accum.leg_align_proximal_terms,
        leg_align_proximal_frac_terms=accum.leg_align_proximal_frac_terms,
        leg_align_anchor_terms=accum.leg_align_anchor_terms,
        leg_align_anchor_frac_terms=accum.leg_align_anchor_frac_terms,
        leg_gate_sup_terms=accum.leg_gate_sup_terms,
        leg_gate_sup_tgt_frac_terms=accum.leg_gate_sup_tgt_frac_terms,
        leg_gate_sup_pred_mean_terms=accum.leg_gate_sup_pred_mean_terms,
    )
    dir6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J, 6)
    R_dir = rot6d_to_matrix(dir6, columns=runtime.columns)

    lam = lam.to(device=runtime.device, dtype=runtime.dtype)
    if lam.ndim == 2 and lam.shape[-1] == 1:
        lam = lam.expand(B, J)
    if lam.shape[-1] != J:
        raise RuntimeError(f"lambda_fusion has wrong shape {tuple(lam.shape)} (expected (B,{J}))")
    lam_raw = lam.clamp(0.0, 1.0)
    lam_eff, lam_rel = lam_raw, None
    try:
        lam_eff, lam_rel = trainer._lambda_fusion_apply_reliability(
            lam_raw,
            step_idx=int(t),
            total_steps=total_steps,
            rollout_step=rollout_step_t,
            ret=ret,
        )
    except _ROLLOUT_SOFT_FAIL_ERRORS:
        _record_posttrain_soft_fail(trainer, "unroll_lambda_reliability")
        lam_eff, lam_rel = lam_raw, None
    if lam_eff is None or (not torch.is_tensor(lam_eff)):
        lam_eff = lam_raw

    accum.lam_vals.append(lam_raw.detach())
    accum.lam_eff_vals.append(lam_eff.detach())
    if torch.is_tensor(lam_rel):
        accum.lam_rel_vals.append(lam_rel.detach())

    plan_prev = _lambda_rollout_accumulate_plan_terms(trainer=trainer, ret=ret, weights=weights, accum=accum, lam_eff=lam_eff, plan_prev=plan_prev, step_weight=step_weights[t], B=B)

    R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
    omega = so3_log_map(R_res)
    R_blend = torch.matmul(so3_exp_map(omega * lam_eff.unsqueeze(-1)), R_inc)

    e_blend = _geodesic_R_safe(R_blend, R_gt)
    e_inc = _geodesic_R_safe(R_inc, R_gt)
    e_dir = _geodesic_R_safe(R_dir, R_gt)
    w_step = step_weights[t]
    accum.loss_terms.append(e_blend.mean() * w_step)
    accum.inc_terms.append(e_inc.mean() * w_step)
    direct_nonleg_focus_applied = _lambda_rollout_accumulate_direct_objective(trainer=trainer, model=model, weights=weights, accum=accum, objective=objective, e_dir=e_dir, step_weight=w_step, J=J, direct_nonleg_focus_applied=direct_nonleg_focus_applied)
    _lambda_rollout_accumulate_gate_supervision(trainer=trainer, ret=ret, accum=accum, weights=weights, e_inc=e_inc, e_dir=e_dir, step_weight=w_step, t=t, B=B, J=J)

    if include_boundary and int(idx) == (cycle_len - 1):
        try:
            accum.boundary_blend_terms.append(_geodesic_R_safe(R_blend, R_gt).mean().detach())
            accum.boundary_inc_terms.append(_geodesic_R_safe(R_inc, R_gt).mean().detach())
            accum.boundary_dir_terms.append(_geodesic_R_safe(R_dir, R_gt).mean().detach())
            accum.boundary_lam_terms.append(lam_raw.mean().detach())
            accum.boundary_lam_eff_terms.append(lam_eff.mean().detach())
            if torch.is_tensor(lam_rel):
                accum.boundary_r_terms.append(lam_rel.mean().detach())
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_boundary_stats")

    if float(weights.lambda_early_weight or 0.0) > 0.0 and int(weights.lambda_early_steps or 0) > 0 and int(t) < int(weights.lambda_early_steps):
        accum.early_terms.append(lam_eff.mean() * w_step)
    if float(weights.lambda_entropy_weight or 0.0) > 0.0:
        accum.ent_terms.append((-_lambda_entropy(lam_eff).mean()) * w_step)
    if float(weights.lambda_smooth_weight or 0.0) > 0.0:
        if lam_prev is not None:
            accum.smooth_terms.append(((lam_eff - lam_prev).pow(2).mean()) * w_step)
        lam_prev = lam_eff.detach()
    if float(weights.lambda_monotonic_weight or 0.0) > 0.0:
        if lam_prev_monot is not None:
            accum.mono_terms.append(F.relu(lam_prev_monot - lam_eff).mean() * w_step)
        lam_prev_monot = lam_eff.detach()

    rot_next6d = matrix_to_rot6d(R_blend, columns=runtime.columns).view(B, rot_len)
    y_next_raw = y_prev_raw + delta_raw
    y_next_raw = y_next_raw.clone()
    y_next_raw[..., rot_slice] = rot_next6d
    if bool(runtime.detach_rollout_state):
        y_next_raw = y_next_raw.detach()
    if t < total_steps - 1:
        _rollout_kernel.apply_rollout_carry_state(
            trainer,
            state,
            y_next_raw=y_next_raw,
            cond_raw_step=cond_raw_step,
        )

    state_vars.meas_used_logits = bool(meas_used_logits)
    state_vars.direct_nonleg_focus_applied = float(direct_nonleg_focus_applied)
    state_vars.lam_prev = lam_prev
    state_vars.lam_prev_monot = lam_prev_monot
    state_vars.plan_prev = plan_prev


def _lambda_fusion_run_unroll(
    *,
    runtime_ctx: Dict[str, Any],
    weights_ctx: LambdaRolloutWeights,
    accum_ctx: LambdaFusionAccum,
    state_vars: LambdaRolloutStepState,
) -> Tuple[bool, float]:
    trainer = runtime_ctx["trainer"]
    model = runtime_ctx["model"]
    batch = runtime_ctx["batch"]
    prep_ctx: LambdaRolloutPrepContext = runtime_ctx["prep_ctx"]
    include_boundary = bool(prep_ctx.include_boundary)
    cycle_len = int(prep_ctx.cycle_len)
    y0_raw = prep_ctx.y0_raw
    gt_seq = prep_ctx.gt_seq
    device = prep_ctx.device
    time_base = None
    if isinstance(batch, dict):
        base = batch.get("start", None)
        if base is not None:
            time_base = base.to(device=device) if torch.is_tensor(base) else base
    time_index_mode = str(runtime_ctx["time_index_mode"] or "global").strip().lower()
    if time_index_mode == "auto":
        time_index_mode = "global"
    if time_index_mode not in ("global", "cycle", "none"):
        time_index_mode = "global"

    def _yaw_gt_from_gt(idx_step: int) -> Optional[torch.Tensor]:
        try:
            if include_boundary and y0_raw is not None and int(idx_step) == (int(cycle_len) - 1):
                gt_raw_frame = y0_raw
            else:
                gt_idx = min(int(gt_seq.shape[1]) - 1, int(idx_step))
                gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
            return trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
        except _ROLLOUT_SOFT_FAIL_ERRORS:
            _record_posttrain_soft_fail(trainer, "run_unroll_yaw_gt")
            return None

    unroll_ctx = LambdaRolloutStepContext(
        runtime=LambdaRolloutRuntimeContext(
            trainer=trainer,
            model=model,
            state=prep_ctx.state,
            total_steps=int(prep_ctx.total_steps),
            cycle_len=int(prep_ctx.cycle_len),
            include_boundary=bool(prep_ctx.include_boundary),
            steps=int(prep_ctx.steps),
            offset=int(prep_ctx.offset),
            time_index_mode=str(time_index_mode),
            time_base=time_base,
            enable_reprojection=bool(runtime_ctx["enable_reprojection"]),
            detach_rollout_state=bool(runtime_ctx["detach_rollout_state"]),
            yaw_gt_fn=_yaw_gt_from_gt,
            columns=runtime_ctx["columns"],
            B=int(prep_ctx.B),
            J=int(prep_ctx.J),
            objective=str(runtime_ctx["objective"]),
            y0_raw=prep_ctx.y0_raw,
            gt_seq=prep_ctx.gt_seq,
            device=prep_ctx.device,
            dtype=prep_ctx.dtype,
            rot_len=int(prep_ctx.rot_len),
        ),
        data=LambdaRolloutDataContext(
            cond_seq=prep_ctx.cond_seq,
            cond_raw_tgt=prep_ctx.cond_raw_tgt,
            cond_norm_mu=prep_ctx.cond_norm_mu,
            cond_norm_std=prep_ctx.cond_norm_std,
            angvel_seq=prep_ctx.angvel_seq,
            pose_hist_seq=prep_ctx.pose_hist_seq,
            contacts_seq=prep_ctx.contacts_seq,
            step_weights=prep_ctx.step_weights,
            std_y=prep_ctx.std_y,
            rot_slice=prep_ctx.rot_slice,
        ),
        weights=weights_ctx,
        accum=accum_ctx,
        state_vars=state_vars,
    )
    total_steps = int(unroll_ctx.runtime.total_steps)
    for t in range(total_steps):
        _lambda_rollout_unroll_single_step(t=int(t), ctx=unroll_ctx)
    return bool(state_vars.meas_used_logits), float(state_vars.direct_nonleg_focus_applied)


def _finalize_direct_group_norm(*, finalize_ctx: LambdaFusionFinalizeContext, trainer: Trainer, blend_loss_total: torch.Tensor, objective: str, dir_geo: torch.Tensor, dir_leg_base_terms: list[torch.Tensor], dir_nonleg_base_terms: list[torch.Tensor], dir_leg_base: torch.Tensor, dir_nonleg_base: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, Any]]]:
    nan_value, zero_value = blend_loss_total.new_tensor(float("nan")), blend_loss_total.new_tensor(0.0)
    leg = nonleg = leg_ema = nonleg_ema = leg_raw = nonleg_raw = leg_clamped = nonleg_clamped = nan_value
    leg_hit_min = leg_hit_max = nonleg_hit_min = nonleg_hit_max = leg_hit_any = nonleg_hit_any = zero_value
    used, ema_update_payload = 0.0, None
    if objective == "direct" and bool(finalize_ctx.direct_group_norm_enable) and dir_leg_base_terms and dir_nonleg_base_terms:
        try:
            ema_state = getattr(trainer, "_direct_pose_group_norm_ema", None)
            if not isinstance(ema_state, dict):
                ema_state = {}
            ema_leg_prev, ema_non_prev = ema_state.get("leg", None), ema_state.get("nonleg", None)
            leg_ema_ok = bool(torch.is_tensor(ema_leg_prev))
            if leg_ema_ok:
                try:
                    leg_ema_ok = bool(torch.isfinite(ema_leg_prev).all().detach().cpu().item())
                except (RuntimeError, ValueError, TypeError):
                    _record_posttrain_soft_fail(trainer, "finalize_group_norm_ema_leg_finite")
                    leg_ema_ok = False
            ema_leg_prev = dir_leg_base.detach() if not leg_ema_ok else ema_leg_prev.to(device=dir_leg_base.device, dtype=dir_leg_base.dtype)
            non_ema_ok = bool(torch.is_tensor(ema_non_prev))
            if non_ema_ok:
                try:
                    non_ema_ok = bool(torch.isfinite(ema_non_prev).all().detach().cpu().item())
                except (RuntimeError, ValueError, TypeError):
                    _record_posttrain_soft_fail(trainer, "finalize_group_norm_ema_nonleg_finite")
                    non_ema_ok = False
            ema_non_prev = dir_nonleg_base.detach() if not non_ema_ok else ema_non_prev.to(device=dir_nonleg_base.device, dtype=dir_nonleg_base.dtype)
            leg_raw = dir_leg_base / ema_leg_prev.clamp_min(float(finalize_ctx.direct_group_eps))
            nonleg_raw = dir_nonleg_base / ema_non_prev.clamp_min(float(finalize_ctx.direct_group_eps))
            leg = leg_clamped = leg_raw.clamp(float(finalize_ctx.direct_group_ratio_min), float(finalize_ctx.direct_group_ratio_max))
            nonleg = nonleg_clamped = nonleg_raw.clamp(float(finalize_ctx.direct_group_ratio_min), float(finalize_ctx.direct_group_ratio_max))
            leg_hit_min = (leg_raw <= float(finalize_ctx.direct_group_ratio_min)).to(dtype=dir_leg_base.dtype)
            leg_hit_max = (leg_raw >= float(finalize_ctx.direct_group_ratio_max)).to(dtype=dir_leg_base.dtype)
            nonleg_hit_min = (nonleg_raw <= float(finalize_ctx.direct_group_ratio_min)).to(dtype=dir_nonleg_base.dtype)
            nonleg_hit_max = (nonleg_raw >= float(finalize_ctx.direct_group_ratio_max)).to(dtype=dir_nonleg_base.dtype)
            leg_hit_any, nonleg_hit_any = torch.maximum(leg_hit_min, leg_hit_max), torch.maximum(nonleg_hit_min, nonleg_hit_max)
            leg_ema, nonleg_ema = ema_leg_prev, ema_non_prev
            dir_geo = float(finalize_ctx.direct_group_w_leg) * leg + float(finalize_ctx.direct_group_w_nonleg) * nonleg
            used = 1.0
            with torch.no_grad():
                beta = float(finalize_ctx.direct_group_beta)
                ema_update_payload = dict(ema_state, leg=(beta * ema_leg_prev + (1.0 - beta) * dir_leg_base.detach()).detach(), nonleg=(beta * ema_non_prev + (1.0 - beta) * dir_nonleg_base.detach()).detach())
        except _ROLLOUT_SOFT_FAIL_ERRORS:
            _record_posttrain_soft_fail(trainer, "finalize_group_norm_main")
            used = 0.0
    def _metric(v: Any, default: float = float("nan")) -> float:
        return float(v.detach().cpu()) if torch.is_tensor(v) else default

    return dir_geo, {
        "dir_group_norm_used": float(used),
        "dir_group_norm_leg_raw": _metric(leg_raw),
        "dir_group_norm_nonleg_raw": _metric(nonleg_raw),
        "dir_group_norm_leg_clamped": _metric(leg_clamped),
        "dir_group_norm_nonleg_clamped": _metric(nonleg_clamped),
        "dir_group_norm_leg": _metric(leg),
        "dir_group_norm_nonleg": _metric(nonleg),
        "dir_group_norm_leg_ema": _metric(leg_ema),
        "dir_group_norm_nonleg_ema": _metric(nonleg_ema),
        "dir_group_norm_leg_hit_min": _metric(leg_hit_min, 0.0),
        "dir_group_norm_leg_hit_max": _metric(leg_hit_max, 0.0),
        "dir_group_norm_nonleg_hit_min": _metric(nonleg_hit_min, 0.0),
        "dir_group_norm_nonleg_hit_max": _metric(nonleg_hit_max, 0.0),
        "dir_group_norm_leg_hit_any": _metric(leg_hit_any, 0.0),
        "dir_group_norm_nonleg_hit_any": _metric(nonleg_hit_any, 0.0),
        "dir_group_norm_w_leg": float(finalize_ctx.direct_group_w_leg),
        "dir_group_norm_w_nonleg": float(finalize_ctx.direct_group_w_nonleg),
    }, ema_update_payload


def _finalize_leg_align_joint_stats(*, trainer: Trainer, model: EventMotionModel, leg_align_joint_num_terms: list[torch.Tensor], leg_align_joint_den_terms: list[torch.Tensor], leg_align_joint_frac_terms: list[torch.Tensor]) -> Dict[str, float]:
    if not (leg_align_joint_num_terms and leg_align_joint_den_terms):
        return {}
    try:
        joint_num = torch.stack(leg_align_joint_num_terms).sum(dim=0)
        joint_den = torch.stack(leg_align_joint_den_terms).sum(dim=0)
        joint_frac = torch.stack(leg_align_joint_frac_terms).sum(dim=0) if leg_align_joint_frac_terms else None
        joint_names = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
        if len(joint_names) != int(joint_num.shape[0]):
            joint_names = [f"j{i}" for i in range(int(joint_num.shape[0]))]
        stats: Dict[str, float] = {}
        for joint_idx, joint_name in enumerate(joint_names):
            key_suffix = _sanitize_metric_key_suffix(str(joint_name), default=f"j{joint_idx}")
            den_value = float(joint_den[joint_idx].detach().cpu())
            stats[f"leg_align_joint_loss_{key_suffix}"] = float((joint_num[joint_idx] / joint_den[joint_idx].clamp_min(1e-6)).detach().cpu()) if den_value > 0.0 else float("nan")
            if joint_frac is not None:
                stats[f"leg_align_joint_frac_{key_suffix}"] = float(joint_frac[joint_idx].detach().cpu())
        return stats
    except (RuntimeError, ValueError, TypeError, IndexError):
        _record_posttrain_soft_fail(trainer, "finalize_leg_align_joint_stats")
        return {}


def _summarize_lambda_finalize_stats(*, trainer: Trainer, lam_vals: list[torch.Tensor], lam_eff_vals: list[torch.Tensor], lam_rel_vals: list[torch.Tensor]) -> Dict[str, float]:
    lam_mean = lam_std = lam_eff_mean = lam_eff_std = lam_rel_mean = None
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_vals], dim=0)
        lam_mean, lam_std = float(flat.mean().detach().cpu()), float(flat.std(unbiased=False).detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_raw")
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_eff_vals], dim=0)
        lam_eff_mean, lam_eff_std = float(flat.mean().detach().cpu()), float(flat.std(unbiased=False).detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_eff")
    try:
        if lam_rel_vals:
            lam_rel_mean = float(torch.cat([x.reshape(-1) for x in lam_rel_vals], dim=0).mean().detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_rel")
    return {
        "lambda_mean": float(lam_mean) if lam_mean is not None else float("nan"),
        "lambda_std": float(lam_std) if lam_std is not None else float("nan"),
        "lambda_eff_mean": float(lam_eff_mean) if lam_eff_mean is not None else float("nan"),
        "lambda_eff_std": float(lam_eff_std) if lam_eff_std is not None else float("nan"),
        "lambda_rel_mean": float(lam_rel_mean) if lam_rel_mean is not None else float("nan"),
    }


def _lambda_fusion_finalize(
    *,
    finalize_ctx: LambdaFusionFinalizeContext,
    accum_ctx: LambdaFusionAccum,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, Any]]]:
    trainer = finalize_ctx.trainer
    model = finalize_ctx.model
    objective = str(finalize_ctx.objective)
    include_boundary = bool(finalize_ctx.include_boundary)
    random_offset = bool(finalize_ctx.random_offset)
    meas_used_logits = bool(finalize_ctx.meas_used_logits)
    offset = int(finalize_ctx.offset)
    boundary_steps = int(finalize_ctx.boundary_steps)
    direct_nonleg_focus_requested = int(finalize_ctx.direct_nonleg_focus_requested)
    direct_nonleg_focus_resolved = int(finalize_ctx.direct_nonleg_focus_resolved)
    gate_sup_weight = float(finalize_ctx.gate_sup_weight)
    direct_pose_leg_gate_sup_weight = float(finalize_ctx.direct_pose_leg_gate_sup_weight)
    direct_pose_leg_align_weight = float(finalize_ctx.direct_pose_leg_align_weight)
    lambda_entropy_weight = float(finalize_ctx.lambda_entropy_weight)
    lambda_smooth_weight = float(finalize_ctx.lambda_smooth_weight)
    lambda_early_weight = float(finalize_ctx.lambda_early_weight)
    lambda_monotonic_weight = float(finalize_ctx.lambda_monotonic_weight)
    lambda_plan_entropy_weight = float(finalize_ctx.lambda_plan_entropy_weight)
    lambda_plan_dyn_weight = float(finalize_ctx.lambda_plan_dyn_weight)
    contact_meas_weight = float(finalize_ctx.contact_meas_weight)
    boundary_weight = float(finalize_ctx.boundary_weight)
    boundary_weighted_sum = float(finalize_ctx.boundary_weighted_sum)
    direct_nonleg_focus_weight_use = float(finalize_ctx.direct_nonleg_focus_weight_use)
    direct_nonleg_focus_applied = float(finalize_ctx.direct_nonleg_focus_applied)

    blend_loss_total = torch.stack(accum_ctx.loss_terms).sum()
    zero = blend_loss_total.new_tensor(0.0)

    def _sum_terms(terms: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(terms).sum() if terms else zero

    def _to_float(value: Any, default: float = float("nan")) -> float:
        return float(value.detach().cpu()) if torch.is_tensor(value) else default

    term_totals = {
        name: _sum_terms(terms)
        for name, terms in (
            ("inc_geo", accum_ctx.inc_terms),
            ("dir_geo", accum_ctx.dir_terms),
            ("dir_base", accum_ctx.dir_base_terms),
            ("dir_leg_base", accum_ctx.dir_leg_base_terms),
            ("dir_nonleg_base", accum_ctx.dir_nonleg_base_terms),
            ("dir_nonleg_plain", accum_ctx.dir_nonleg_plain_terms),
            ("gate_sup_loss", accum_ctx.gate_sup_terms),
            ("gate_sup_frac", accum_ctx.gate_sup_frac_terms),
            ("leg_gate_sup_loss", accum_ctx.leg_gate_sup_terms),
            ("leg_gate_sup_tgt_frac", accum_ctx.leg_gate_sup_tgt_frac_terms),
            ("leg_gate_sup_pred_mean", accum_ctx.leg_gate_sup_pred_mean_terms),
            ("leg_align_loss", accum_ctx.leg_align_terms),
            ("leg_align_frac", accum_ctx.leg_align_frac_terms),
            ("leg_align_distal_loss", accum_ctx.leg_align_distal_terms),
            ("leg_align_distal_frac", accum_ctx.leg_align_distal_frac_terms),
            ("leg_align_proximal_loss", accum_ctx.leg_align_proximal_terms),
            ("leg_align_proximal_frac", accum_ctx.leg_align_proximal_frac_terms),
            ("leg_align_anchor_loss", accum_ctx.leg_align_anchor_terms),
            ("leg_align_anchor_frac", accum_ctx.leg_align_anchor_frac_terms),
            ("entropy_loss", accum_ctx.ent_terms),
            ("smooth_loss", accum_ctx.smooth_terms),
            ("early_loss", accum_ctx.early_terms),
            ("mono_loss", accum_ctx.mono_terms),
            ("plan_ent_loss", accum_ctx.plan_ent_terms),
            ("plan_dyn_loss", accum_ctx.plan_dyn_terms),
            ("contact_meas_loss", accum_ctx.meas_terms),
        )
    }
    inc_geo = term_totals["inc_geo"]
    dir_geo = term_totals["dir_geo"]
    dir_base = term_totals["dir_base"]
    dir_leg_base = term_totals["dir_leg_base"]
    dir_nonleg_base = term_totals["dir_nonleg_base"]
    dir_nonleg_plain = term_totals["dir_nonleg_plain"]
    dir_geo, dir_group_norm_stats, ema_update_payload = _finalize_direct_group_norm(
        finalize_ctx=finalize_ctx,
        trainer=trainer,
        blend_loss_total=blend_loss_total,
        objective=objective,
        dir_geo=dir_geo,
        dir_leg_base_terms=accum_ctx.dir_leg_base_terms,
        dir_nonleg_base_terms=accum_ctx.dir_nonleg_base_terms,
        dir_leg_base=dir_leg_base,
        dir_nonleg_base=dir_nonleg_base,
    )
    total = dir_geo if objective == "direct" else inc_geo if objective == "inc" else blend_loss_total

    gate_sup_loss = term_totals["gate_sup_loss"]
    gate_sup_frac = term_totals["gate_sup_frac"]
    gate_sup_acc = None
    if accum_ctx.gate_sup_terms:
        total = total + float(gate_sup_weight) * gate_sup_loss
    if accum_ctx.gate_sup_acc_num_terms and accum_ctx.gate_sup_acc_den_terms:
        try:
            num = torch.stack(accum_ctx.gate_sup_acc_num_terms).sum()
            den = torch.stack(accum_ctx.gate_sup_acc_den_terms).sum()
            gate_sup_acc = torch.where(den > 0.0, num / den.clamp_min(1e-6), den.new_tensor(float("nan"))).detach()
        except (RuntimeError, ValueError, TypeError):
            _record_posttrain_soft_fail(trainer, "finalize_gate_sup_acc")
            gate_sup_acc = None

    leg_gate_sup_loss = term_totals["leg_gate_sup_loss"]
    leg_gate_sup_tgt_frac = term_totals["leg_gate_sup_tgt_frac"]
    leg_gate_sup_pred_mean = term_totals["leg_gate_sup_pred_mean"]
    if accum_ctx.leg_gate_sup_terms:
        total = total + float(direct_pose_leg_gate_sup_weight or 0.0) * leg_gate_sup_loss

    leg_align_loss = term_totals["leg_align_loss"]
    leg_align_frac = term_totals["leg_align_frac"]
    leg_align_distal_loss = term_totals["leg_align_distal_loss"]
    leg_align_distal_frac = term_totals["leg_align_distal_frac"]
    leg_align_proximal_loss = term_totals["leg_align_proximal_loss"]
    leg_align_proximal_frac = term_totals["leg_align_proximal_frac"]
    leg_align_anchor_loss = term_totals["leg_align_anchor_loss"]
    leg_align_anchor_frac = term_totals["leg_align_anchor_frac"]
    if accum_ctx.leg_align_terms:
        total = total + float(direct_pose_leg_align_weight or 0.0) * leg_align_loss
    leg_align_joint_stats = _finalize_leg_align_joint_stats(
        trainer=trainer,
        model=model,
        leg_align_joint_num_terms=accum_ctx.leg_align_joint_num_terms,
        leg_align_joint_den_terms=accum_ctx.leg_align_joint_den_terms,
        leg_align_joint_frac_terms=accum_ctx.leg_align_joint_frac_terms,
    )
    for terms, weight, key in (
        (accum_ctx.ent_terms, lambda_entropy_weight, "entropy_loss"),
        (accum_ctx.smooth_terms, lambda_smooth_weight, "smooth_loss"),
        (accum_ctx.early_terms, lambda_early_weight, "early_loss"),
        (accum_ctx.mono_terms, lambda_monotonic_weight, "mono_loss"),
        (accum_ctx.plan_ent_terms, lambda_plan_entropy_weight, "plan_ent_loss"),
        (accum_ctx.plan_dyn_terms, lambda_plan_dyn_weight, "plan_dyn_loss"),
        (accum_ctx.meas_terms, contact_meas_weight, "contact_meas_loss"),
    ):
        if terms:
            total = total + float(weight or 0.0) * term_totals[key]
    entropy_loss = term_totals["entropy_loss"]
    smooth_loss = term_totals["smooth_loss"]
    early_loss = term_totals["early_loss"]
    mono_loss = term_totals["mono_loss"]
    plan_ent_loss = term_totals["plan_ent_loss"]
    plan_dyn_loss = term_totals["plan_dyn_loss"]
    contact_meas_loss = term_totals["contact_meas_loss"] if accum_ctx.meas_terms else None

    lambda_stats = _summarize_lambda_finalize_stats(
        trainer=trainer,
        lam_vals=accum_ctx.lam_vals,
        lam_eff_vals=accum_ctx.lam_eff_vals,
        lam_rel_vals=accum_ctx.lam_rel_vals,
    )
    anchor_weight = float(finalize_ctx.direct_pose_leg_align_anchor_weight or 0.0)

    stats = {
        "dir_nonleg_focus_requested": float(direct_nonleg_focus_requested),
        "dir_nonleg_focus_resolved": float(direct_nonleg_focus_resolved),
        "dir_nonleg_focus_weight": float(direct_nonleg_focus_weight_use),
        "dir_nonleg_focus_applied": float(direct_nonleg_focus_applied),
        "gate_sup_acc@0.5": _to_float(gate_sup_acc),
        "leg_align_anchor_weight": anchor_weight,
        "leg_align_weight": float(direct_pose_leg_align_weight or 0.0),
    }
    for key, value in (
        ("blend_loss", blend_loss_total),
        ("gate_sup_loss", gate_sup_loss),
        ("gate_sup_frac", gate_sup_frac),
        ("leg_gate_sup_loss", leg_gate_sup_loss),
        ("leg_gate_sup_tgt_frac", leg_gate_sup_tgt_frac),
        ("leg_gate_sup_pred_mean", leg_gate_sup_pred_mean),
        ("leg_gate_sup_weighted", float(direct_pose_leg_gate_sup_weight or 0.0) * leg_gate_sup_loss),
        ("leg_align_loss", leg_align_loss),
        ("leg_align_frac", leg_align_frac),
        ("leg_align_distal_loss", leg_align_distal_loss),
        ("leg_align_distal_frac", leg_align_distal_frac),
        ("leg_align_proximal_loss", leg_align_proximal_loss),
        ("leg_align_proximal_frac", leg_align_proximal_frac),
        ("leg_align_anchor_loss", leg_align_anchor_loss),
        ("leg_align_anchor_frac", leg_align_anchor_frac),
        ("leg_align_anchor_weighted", float(direct_pose_leg_align_weight or 0.0) * anchor_weight * leg_align_anchor_loss),
        ("leg_align_weighted", float(direct_pose_leg_align_weight or 0.0) * leg_align_loss),
        ("inc_geo", inc_geo),
        ("dir_geo", dir_geo),
        ("dir_base", dir_base),
        ("dir_leg_base", dir_leg_base),
        ("dir_nonleg_base", dir_nonleg_base),
        ("dir_nonleg_plain", dir_nonleg_plain),
        ("entropy_loss", entropy_loss),
        ("smooth_loss", smooth_loss),
        ("early_loss", early_loss),
        ("mono_loss", mono_loss),
        ("plan_entropy_loss", plan_ent_loss),
        ("plan_dyn_loss", plan_dyn_loss),
        ("total", total),
    ):
        stats[key] = _to_float(value)
    stats.update(lambda_stats)
    stats.update(dir_group_norm_stats)
    stats.update(leg_align_joint_stats)
    if include_boundary:
        stats["rollout_include_boundary"] = 1.0
        stats["rollout_random_offset"] = 1.0 if bool(random_offset) else 0.0
        stats["rollout_offset"] = float(offset)
        stats["lambda_boundary_weight"] = float(boundary_weight or 0.0)
        stats["boundary_steps"] = float(boundary_steps or 0)
        stats["boundary_weighted_sum"] = float(boundary_weighted_sum or 0.0)
        for key, terms in (
            ("boundary_blend_geo", accum_ctx.boundary_blend_terms), ("boundary_inc_geo", accum_ctx.boundary_inc_terms),
            ("boundary_dir_geo", accum_ctx.boundary_dir_terms), ("boundary_lambda_mean", accum_ctx.boundary_lam_terms),
            ("boundary_lambda_eff_mean", accum_ctx.boundary_lam_eff_terms), ("boundary_r_mean", accum_ctx.boundary_r_terms),
        ):
            if not terms:
                continue
            try:
                stats[key] = float(torch.stack(terms).mean().detach().cpu())
            except (RuntimeError, ValueError, TypeError):
                _record_posttrain_soft_fail(trainer, f"finalize_boundary_{key}")
    for key, terms in (("plan_entropy_mean", accum_ctx.plan_ent_stat_terms), ("plan_dyn_mean", accum_ctx.plan_dyn_stat_terms)):
        if not terms:
            continue
        try:
            stats[key] = float(torch.stack(terms).sum().detach().cpu())
        except (RuntimeError, ValueError, TypeError):
            _record_posttrain_soft_fail(trainer, f"finalize_plan_{key}")
    if contact_meas_loss is not None:
        stats["contact_meas_bce" if bool(meas_used_logits) else "contact_meas_mse"] = _to_float(contact_meas_loss)
        stats["contact_meas_weighted"] = _to_float(float(contact_meas_weight or 0.0) * contact_meas_loss)
    aux_payload: Dict[str, Any] = {"ema_update_payload": ema_update_payload}
    if objective == "direct":
        aux_payload["leg_align_grad_probe"] = {
            "total": leg_align_loss,
            "distal": leg_align_distal_loss,
            "proximal": leg_align_proximal_loss,
        }
    return total, stats, aux_payload


def _lambda_fusion_loss_rollout(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    columns: Tuple[str, str],
    rollout_steps: int,
    rollout_cycles: int,
    include_boundary: bool,
    boundary_weight: float,
    random_offset: bool,
    time_index_mode: str,
    time_weight_max: float,
    time_weight_mode: str,
    detach_rollout_state: bool,
    lambda_entropy_weight: float,
    lambda_smooth_weight: float,
    lambda_early_steps: int = 0,
    lambda_early_weight: float = 0.0,
    lambda_monotonic_weight: float = 0.0,
    lambda_plan_entropy_weight: float = 0.0,
    lambda_plan_dyn_weight: float = 0.0,
    lambda_gate_sup_weight: float = 0.0,
    lambda_gate_sup_tau_deg: float = 2.5,
    lambda_gate_sup_margin_deg: float = 1.0,
    lambda_gate_sup_start_step: int = -1,
    contact_meas_weight: float = 0.0,
    objective: str = "blend",  # blend|direct|inc
    direct_pose_leg_gate_sup_weight: float = 0.0,
    direct_pose_leg_align_weight: float = 0.0,
    direct_pose_leg_align_oracle_min_deg: float = 0.0,
    direct_pose_leg_align_oracle_weight_deg: float = 0.0,
    direct_pose_leg_align_mode: str = "cos",
    direct_pose_leg_align_mag_weight: float = 1.0,
    direct_pose_leg_align_res_weight: float = 1.0,
    direct_pose_leg_align_sign_weight: float = 0.0,
    direct_pose_leg_align_cos_thresh: float = 0.0,
    direct_pose_leg_align_target_joints: Optional[str] = None,
    direct_pose_leg_align_anchor_joints: Optional[str] = None,
    direct_pose_leg_align_anchor_weight: float = 0.0,
    direct_pose_loss_leg_split: bool = False,
    direct_pose_loss_group_norm_enable: bool = False,
    direct_pose_loss_group_norm_w_leg: float = 1.0,
    direct_pose_loss_group_norm_w_nonleg: float = 1.0,
    direct_pose_loss_group_norm_ema_beta: float = 0.95,
    direct_pose_loss_group_norm_ratio_min: float = 0.2,
    direct_pose_loss_group_norm_ratio_max: float = 5.0,
    direct_pose_loss_group_norm_eps: float = 1e-6,
    direct_pose_nonleg_focus_bones: str = "",
    direct_pose_nonleg_focus_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, Any]]]:
    objective = str(objective or "blend").strip().lower()
    if objective not in ("blend", "direct", "inc"):
        raise ValueError(f"Unknown objective={objective!r} (expected blend|direct|inc)")
    if objective != "blend":
        lambda_entropy_weight = lambda_smooth_weight = 0.0
        lambda_early_steps = 0
        lambda_early_weight = lambda_monotonic_weight = 0.0
        lambda_plan_entropy_weight = lambda_plan_dyn_weight = 0.0
        lambda_gate_sup_weight = 0.0
    # ---- Prepare rollout context ----
    prep_ctx = _lambda_rollout_prepare_context(
        trainer,
        model,
        batch,
        rollout_steps=rollout_steps,
        rollout_cycles=rollout_cycles,
        include_boundary=include_boundary,
        boundary_weight=boundary_weight,
        random_offset=random_offset,
        time_weight_mode=time_weight_mode,
        time_weight_max=time_weight_max,
    )
    device = prep_ctx.device
    include_boundary = bool(prep_ctx.include_boundary)
    offset = int(prep_ctx.offset)
    J = int(prep_ctx.J)
    boundary_steps = int(prep_ctx.boundary_steps)
    boundary_weighted_sum = float(prep_ctx.boundary_weighted_sum)
    # ---- Resolve rollout-side weighting / regularization knobs ----
    nonleg_focus_ctx = _lambda_rollout_resolve_nonleg_focus(
        trainer,
        objective=objective,
        direct_pose_nonleg_focus_bones=direct_pose_nonleg_focus_bones,
        direct_pose_nonleg_focus_weight=direct_pose_nonleg_focus_weight,
        J=J,
        device=device,
    )
    direct_nonleg_focus_mask_j = nonleg_focus_ctx.direct_nonleg_focus_mask_j
    direct_nonleg_focus_requested = int(nonleg_focus_ctx.direct_nonleg_focus_requested)
    direct_nonleg_focus_resolved = int(nonleg_focus_ctx.direct_nonleg_focus_resolved)
    direct_nonleg_focus_weight_use = float(nonleg_focus_ctx.direct_nonleg_focus_weight_use)
    direct_nonleg_focus_applied = float(nonleg_focus_ctx.direct_nonleg_focus_applied)
    accum_ctx = LambdaFusionAccum()
    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))
    reg_ctx = _lambda_rollout_build_reg_params(
        trainer,
        objective=objective,
        lambda_gate_sup_weight=lambda_gate_sup_weight,
        lambda_gate_sup_start_step=lambda_gate_sup_start_step,
        lambda_gate_sup_tau_deg=lambda_gate_sup_tau_deg,
        lambda_gate_sup_margin_deg=lambda_gate_sup_margin_deg,
        direct_pose_loss_group_norm_enable=direct_pose_loss_group_norm_enable,
        direct_pose_loss_group_norm_w_leg=direct_pose_loss_group_norm_w_leg,
        direct_pose_loss_group_norm_w_nonleg=direct_pose_loss_group_norm_w_nonleg,
        direct_pose_loss_group_norm_ema_beta=direct_pose_loss_group_norm_ema_beta,
        direct_pose_loss_group_norm_ratio_min=direct_pose_loss_group_norm_ratio_min,
        direct_pose_loss_group_norm_ratio_max=direct_pose_loss_group_norm_ratio_max,
        direct_pose_loss_group_norm_eps=direct_pose_loss_group_norm_eps,
    )
    weights_ctx = LambdaRolloutWeights(
        contact_meas_weight=float(contact_meas_weight),
        direct_pose_leg_align_weight=float(direct_pose_leg_align_weight),
        direct_pose_leg_align_oracle_min_deg=float(direct_pose_leg_align_oracle_min_deg),
        direct_pose_leg_align_oracle_weight_deg=float(direct_pose_leg_align_oracle_weight_deg),
        direct_pose_leg_align_mode=str(direct_pose_leg_align_mode),
        direct_pose_leg_align_mag_weight=float(direct_pose_leg_align_mag_weight),
        direct_pose_leg_align_res_weight=float(direct_pose_leg_align_res_weight),
        direct_pose_leg_align_sign_weight=float(direct_pose_leg_align_sign_weight),
        direct_pose_leg_align_cos_thresh=float(direct_pose_leg_align_cos_thresh),
        direct_pose_leg_align_target_joints=direct_pose_leg_align_target_joints,
        direct_pose_leg_align_anchor_joints=direct_pose_leg_align_anchor_joints,
        direct_pose_leg_align_anchor_weight=float(direct_pose_leg_align_anchor_weight),
        direct_pose_leg_gate_sup_weight=float(direct_pose_leg_gate_sup_weight),
        direct_pose_loss_leg_split=bool(direct_pose_loss_leg_split),
        direct_nonleg_focus_mask_j=direct_nonleg_focus_mask_j,
        direct_nonleg_focus_resolved=int(direct_nonleg_focus_resolved),
        direct_nonleg_focus_weight_use=float(direct_nonleg_focus_weight_use),
        gate_sup_weight=float(reg_ctx.gate_sup_weight),
        gate_sup_start=int(reg_ctx.gate_sup_start),
        tau_rad=float(reg_ctx.tau_rad),
        margin_rad=float(reg_ctx.margin_rad),
        lambda_plan_entropy_weight=float(lambda_plan_entropy_weight),
        lambda_plan_dyn_weight=float(lambda_plan_dyn_weight),
        lambda_early_weight=float(lambda_early_weight),
        lambda_early_steps=int(lambda_early_steps),
        lambda_entropy_weight=float(lambda_entropy_weight),
        lambda_smooth_weight=float(lambda_smooth_weight),
        lambda_monotonic_weight=float(lambda_monotonic_weight),
    )
    state_vars = LambdaRolloutStepState(
        meas_used_logits=False,
        direct_nonleg_focus_applied=float(direct_nonleg_focus_applied),
        lam_prev=None,
        lam_prev_monot=None,
        plan_prev=None,
    )
    runtime_ctx = {
        "trainer": trainer, "model": model, "batch": batch, "prep_ctx": prep_ctx, "time_index_mode": time_index_mode,
        "enable_reprojection": enable_reprojection, "detach_rollout_state": detach_rollout_state, "columns": columns, "objective": objective,
    }
    meas_used_logits, direct_nonleg_focus_applied = _lambda_fusion_run_unroll(runtime_ctx=runtime_ctx, weights_ctx=weights_ctx, accum_ctx=accum_ctx, state_vars=state_vars)
    # ---- Finalize aggregated losses / stats ----
    finalize_ctx = LambdaFusionFinalizeContext(
        trainer=trainer,
        model=model,
        objective=objective,
        direct_pose_leg_gate_sup_weight=float(direct_pose_leg_gate_sup_weight),
        direct_pose_leg_align_weight=float(direct_pose_leg_align_weight),
        direct_pose_leg_align_anchor_weight=float(direct_pose_leg_align_anchor_weight),
        lambda_entropy_weight=float(lambda_entropy_weight),
        lambda_smooth_weight=float(lambda_smooth_weight),
        lambda_early_weight=float(lambda_early_weight),
        lambda_monotonic_weight=float(lambda_monotonic_weight),
        lambda_plan_entropy_weight=float(lambda_plan_entropy_weight),
        lambda_plan_dyn_weight=float(lambda_plan_dyn_weight),
        contact_meas_weight=float(contact_meas_weight),
        include_boundary=bool(include_boundary),
        random_offset=bool(random_offset),
        offset=int(offset),
        boundary_weight=float(boundary_weight),
        boundary_steps=int(boundary_steps),
        boundary_weighted_sum=float(boundary_weighted_sum),
        direct_nonleg_focus_requested=int(direct_nonleg_focus_requested),
        direct_nonleg_focus_resolved=int(direct_nonleg_focus_resolved),
        direct_nonleg_focus_weight_use=float(direct_nonleg_focus_weight_use),
        direct_nonleg_focus_applied=float(direct_nonleg_focus_applied),
        meas_used_logits=bool(meas_used_logits),
        gate_sup_weight=float(reg_ctx.gate_sup_weight),
        direct_group_norm_enable=bool(reg_ctx.direct_group_norm_enable),
        direct_group_w_leg=float(reg_ctx.direct_group_w_leg),
        direct_group_w_nonleg=float(reg_ctx.direct_group_w_nonleg),
        direct_group_beta=float(reg_ctx.direct_group_beta),
        direct_group_ratio_min=float(reg_ctx.direct_group_ratio_min),
        direct_group_ratio_max=float(reg_ctx.direct_group_ratio_max),
        direct_group_eps=float(reg_ctx.direct_group_eps),
    )
    return _lambda_fusion_finalize(finalize_ctx=finalize_ctx, accum_ctx=accum_ctx)


def _resolve_leg_align_grad_probe_named_params(
    model: EventMotionModel,
) -> tuple[str, list[tuple[str, torch.nn.Parameter]]]:
    prefixes: list[str] = []
    if getattr(model, "direct_pose_leg_head_shared", None) is not None:
        prefixes.append("direct_pose_leg_head_shared")
    if getattr(model, "direct_pose_leg_head", None) is not None:
        prefixes.append("direct_pose_leg_head")
    for prefix in prefixes:
        named: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            if str(name) == prefix or str(name).startswith(f"{prefix}."):
                named.append((str(name), param))
        if named:
            return prefix, named
    return "", []


def _run_leg_align_grad_probe(
    *,
    cfg: PostTrainConfig,
    model: EventMotionModel,
    stats: Dict[str, float],
    aux_payload: Optional[Dict[str, Any]],
) -> None:
    probe_steps = max(0, int(getattr(cfg, "direct_pose_leg_align_grad_probe_steps", 0) or 0))
    if (not bool(getattr(cfg, "direct_pose_leg_align_grad_probe_enable", False))) or probe_steps <= 0:
        return
    stats["leg_align_gradprobe_enabled"] = 1.0
    stats["leg_align_gradprobe_steps"] = float(probe_steps)
    if not isinstance(aux_payload, dict):
        stats["leg_align_gradprobe_ready"] = 0.0
        return
    probe_loss_payload = aux_payload.get("leg_align_grad_probe", None)
    if not isinstance(probe_loss_payload, dict):
        stats["leg_align_gradprobe_ready"] = 0.0
        return
    prefix, named_params = _resolve_leg_align_grad_probe_named_params(model)
    stats["leg_align_gradprobe_ready"] = 1.0 if named_params else 0.0
    stats["leg_align_gradprobe_target_shared"] = 1.0 if prefix == "direct_pose_leg_head_shared" else 0.0
    stats["leg_align_gradprobe_target_plain"] = 1.0 if prefix == "direct_pose_leg_head" else 0.0
    stats["leg_align_gradprobe_param_count"] = float(len(named_params))
    if not named_params:
        return

    params = [param for _, param in named_params]

    def _grad_for(key: str) -> Optional[tuple[Optional[torch.Tensor], ...]]:
        loss_t = probe_loss_payload.get(key, None)
        if not torch.is_tensor(loss_t) or (not bool(getattr(loss_t, "requires_grad", False))):
            return None
        try:
            grads_t = torch.autograd.grad(
                loss_t,
                params,
                allow_unused=True,
                retain_graph=True,
                create_graph=False,
            )
        except Exception:
            return None
        return tuple(grads_t)

    grads_distal = _grad_for("distal")
    grads_proximal = _grad_for("proximal")
    grads_total = _grad_for("total")

    stats["leg_align_gradprobe_loss_distal"] = float(
        probe_loss_payload["distal"].detach().cpu()
    ) if torch.is_tensor(probe_loss_payload.get("distal", None)) else float("nan")
    stats["leg_align_gradprobe_loss_proximal"] = float(
        probe_loss_payload["proximal"].detach().cpu()
    ) if torch.is_tensor(probe_loss_payload.get("proximal", None)) else float("nan")
    stats["leg_align_gradprobe_loss_total"] = float(
        probe_loss_payload["total"].detach().cpu()
    ) if torch.is_tensor(probe_loss_payload.get("total", None)) else float("nan")
    stats["leg_align_gradprobe_norm_distal"] = _grad_list_norm(grads_distal or ())
    stats["leg_align_gradprobe_norm_proximal"] = _grad_list_norm(grads_proximal or ())
    stats["leg_align_gradprobe_norm_total"] = _grad_list_norm(grads_total or ())
    cos_dp = _grad_list_cosine(grads_distal or (), grads_proximal or ())
    cos_dt = _grad_list_cosine(grads_distal or (), grads_total or ())
    cos_pt = _grad_list_cosine(grads_proximal or (), grads_total or ())
    stats["leg_align_gradprobe_cos_distal_proximal"] = float(cos_dp)
    stats["leg_align_gradprobe_cos_distal_total"] = float(cos_dt)
    stats["leg_align_gradprobe_cos_proximal_total"] = float(cos_pt)
    stats["leg_align_gradprobe_neg"] = 1.0 if math.isfinite(cos_dp) and cos_dp < 0.0 else 0.0


def _resolve_train_mode(cfg: PostTrainConfig) -> str:
    selected = int(bool(cfg.train_direct_pose)) + int(bool(cfg.train_lambda_head))
    if selected != 1:
        raise SystemExit("[FATAL] Choose exactly one: train_direct_pose | train_lambda_head.")
    return "direct" if bool(cfg.train_direct_pose) else "lambda"


def _build_rollout_mode_kwargs(cfg: PostTrainConfig, train_mode: str) -> Dict[str, Any]:
    if train_mode not in ("direct", "lambda"):
        raise ValueError(f"Unknown train_mode={train_mode!r}")
    if train_mode == "direct":
        return {
            "objective": "direct",
            "lambda_entropy_weight": 0.0,
            "lambda_smooth_weight": 0.0,
            "lambda_early_steps": 0,
            "lambda_early_weight": 0.0,
            "lambda_monotonic_weight": 0.0,
            "lambda_plan_entropy_weight": 0.0,
            "lambda_plan_dyn_weight": 0.0,
            "direct_pose_leg_gate_sup_weight": float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0),
            "direct_pose_leg_align_weight": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0),
            "direct_pose_leg_align_oracle_min_deg": float(
                getattr(cfg, "direct_pose_leg_align_oracle_min_deg", 0.0) or 0.0
            ),
            "direct_pose_leg_align_oracle_weight_deg": float(
                getattr(cfg, "direct_pose_leg_align_oracle_weight_deg", 0.0) or 0.0
            ),
            "direct_pose_leg_align_mode": str(getattr(cfg, "direct_pose_leg_align_mode", "cos") or "cos"),
            "direct_pose_leg_align_mag_weight": float(getattr(cfg, "direct_pose_leg_align_mag_weight", 1.0) or 1.0),
            "direct_pose_leg_align_res_weight": float(getattr(cfg, "direct_pose_leg_align_res_weight", 1.0) or 1.0),
            "direct_pose_leg_align_sign_weight": float(getattr(cfg, "direct_pose_leg_align_sign_weight", 0.0) or 0.0),
            "direct_pose_leg_align_cos_thresh": float(getattr(cfg, "direct_pose_leg_align_cos_thresh", 0.0) or 0.0),
            "direct_pose_leg_align_target_joints": getattr(cfg, "direct_pose_leg_align_target_joints", None),
            "direct_pose_leg_align_anchor_joints": getattr(cfg, "direct_pose_leg_align_anchor_joints", None),
            "direct_pose_leg_align_anchor_weight": float(getattr(cfg, "direct_pose_leg_align_anchor_weight", 0.0) or 0.0),
            "direct_pose_loss_leg_split": bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
            "direct_pose_loss_group_norm_enable": bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
            "direct_pose_loss_group_norm_w_leg": float(getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0) or 1.0),
            "direct_pose_loss_group_norm_w_nonleg": float(
                getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0) or 1.0
            ),
            "direct_pose_loss_group_norm_ema_beta": float(
                getattr(cfg, "direct_pose_loss_group_norm_ema_beta", 0.95) or 0.95
            ),
            "direct_pose_loss_group_norm_ratio_min": float(
                getattr(cfg, "direct_pose_loss_group_norm_ratio_min", 0.2) or 0.2
            ),
            "direct_pose_loss_group_norm_ratio_max": float(
                getattr(cfg, "direct_pose_loss_group_norm_ratio_max", 5.0) or 5.0
            ),
            "direct_pose_loss_group_norm_eps": float(getattr(cfg, "direct_pose_loss_group_norm_eps", 1e-6) or 1e-6),
            "direct_pose_nonleg_focus_bones": str(getattr(cfg, "direct_pose_nonleg_focus_bones", "") or ""),
            "direct_pose_nonleg_focus_weight": float(getattr(cfg, "direct_pose_nonleg_focus_weight", 1.0) or 1.0),
        }
    return {
        "objective": "blend",
        "lambda_entropy_weight": cfg.lambda_fusion_entropy_weight,
        "lambda_smooth_weight": cfg.lambda_fusion_smooth_weight,
        "lambda_early_steps": cfg.lambda_fusion_early_steps,
        "lambda_early_weight": cfg.lambda_fusion_early_weight,
        "lambda_monotonic_weight": cfg.lambda_fusion_monotonic_weight,
        "lambda_plan_entropy_weight": cfg.lambda_plan_entropy_weight,
        "lambda_plan_dyn_weight": cfg.lambda_plan_dyn_weight,
        "lambda_gate_sup_weight": cfg.lambda_gate_sup_weight,
        "lambda_gate_sup_tau_deg": cfg.lambda_gate_sup_tau_deg,
        "lambda_gate_sup_margin_deg": cfg.lambda_gate_sup_margin_deg,
        "lambda_gate_sup_start_step": cfg.lambda_gate_sup_start_step,
    }


def _resolve_direct_pose_leg_align_weight(cfg: PostTrainConfig, global_step: int) -> float:
    target_weight = float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0)
    if target_weight <= 0.0:
        return 0.0

    schedule = str(getattr(cfg, "direct_pose_leg_align_schedule", "none") or "none").strip().lower()
    if schedule in ("", "none", "off", "false", "0"):
        return target_weight

    start_weight = float(getattr(cfg, "direct_pose_leg_align_start_weight", 0.0) or 0.0)
    warmup_steps = max(0, int(getattr(cfg, "direct_pose_leg_align_warmup_steps", 0) or 0))
    ramp_steps = max(0, int(getattr(cfg, "direct_pose_leg_align_ramp_steps", 0) or 0))
    step_idx = max(0, int(global_step))

    if schedule == "linear":
        if step_idx < warmup_steps:
            return start_weight
        if ramp_steps <= 0:
            return target_weight
        progress = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(ramp_steps)))
        return float(start_weight + (target_weight - start_weight) * progress)

    return target_weight


def _format_posttrain_step_msg(
    *,
    train_mode: str,
    cfg: PostTrainConfig,
    stats: Dict[str, float],
    epoch: int,
    it: int,
    steps_per_epoch: int,
    l2sp_weight: float,
) -> str:
    if train_mode == "direct":
        leg_align_weight = float(
            stats.get("leg_align_weight", getattr(cfg, "direct_pose_leg_align_weight", 0.0)) or 0.0
        )
        msg = (
            f"[posttrain][e{epoch} i{it}/{steps_per_epoch}] "
            f"total={stats['total']:.6f} dir={stats['dir_geo']:.6f} "
            f"blend={stats['blend_loss']:.6f} inc={stats['inc_geo']:.6f} "
            f"λ={stats['lambda_mean']:.3f}±{stats['lambda_std']:.3f}"
        )
        if float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0) > 0.0:
            msg += (
                f" leg_gate={stats.get('leg_gate_sup_loss', float('nan')):.3e}"
                f" tgt={stats.get('leg_gate_sup_tgt_frac', float('nan')):.3f}"
                f" pred={stats.get('leg_gate_sup_pred_mean', float('nan')):.3f}"
            )
        if leg_align_weight > 0.0:
            msg += (
                f" leg_align={stats.get('leg_align_loss', float('nan')):.3e}"
                f" w={leg_align_weight:.3f}"
                f" frac={stats.get('leg_align_frac', float('nan')):.3f}"
            )
            leg_align_anchor_weight = float(stats.get("leg_align_anchor_weight", getattr(cfg, "direct_pose_leg_align_anchor_weight", 0.0)) or 0.0)
            if leg_align_anchor_weight > 0.0:
                msg += (
                    f" anchor={stats.get('leg_align_anchor_loss', float('nan')):.3e}"
                    f" aw={leg_align_anchor_weight:.3f}"
                )
        if float(stats.get("dir_group_norm_used", 0.0) or 0.0) > 0.0:
            msg += (
                f" gnorm(L/N)={stats.get('dir_group_norm_leg', float('nan')):.3f}/"
                f"{stats.get('dir_group_norm_nonleg', float('nan')):.3f}"
                f" raw={stats.get('dir_group_norm_leg_raw', float('nan')):.3f}/"
                f"{stats.get('dir_group_norm_nonleg_raw', float('nan')):.3f}"
                f" ema={stats.get('dir_group_norm_leg_ema', float('nan')):.3f}/"
                f"{stats.get('dir_group_norm_nonleg_ema', float('nan')):.3f}"
                f" hit={stats.get('dir_group_norm_leg_hit_any', 0.0):.0f}/"
                f"{stats.get('dir_group_norm_nonleg_hit_any', 0.0):.0f}"
            )
        if str(cfg.lambda_reliability_mode or "none").strip().lower() not in ("none", "off", "false", "0", ""):
            msg += (
                f" λ_eff={stats.get('lambda_eff_mean', float('nan')):.3f}±{stats.get('lambda_eff_std', float('nan')):.3f}"
                f" r={stats.get('lambda_rel_mean', float('nan')):.3f}"
            )
        if bool(getattr(cfg, "direct_pose_grad_monitor_enable", False)):
            msg += (
                f" g(trunk/leg/non)={stats.get('direct_grad_norm_trunk', float('nan')):.3e}/"
                f"{stats.get('direct_grad_norm_out_leg', float('nan')):.3e}/"
                f"{stats.get('direct_grad_norm_out_nonleg', float('nan')):.3e}"
                f" legω={stats.get('direct_grad_norm_leg_branch', float('nan')):.3e}"
                f" gr={stats.get('direct_grad_ratio_nonleg_over_leg', float('nan')):.3f}"
                f" grω={stats.get('direct_grad_ratio_nonleg_over_leg_branch', float('nan')):.3f}"
            )
            if float(stats.get("direct_grad_ratio_alert", 0.0) or 0.0) > 0.0:
                msg += " !grad_ratio_low"
            if float(stats.get("direct_grad_ratio_alert_leg_branch", 0.0) or 0.0) > 0.0:
                msg += " !grad_ratio_low_omega"
        if float(stats.get("leg_align_gradprobe_enabled", 0.0) or 0.0) > 0.0:
            msg += (
                f" gp(dp/dt/pt)={stats.get('leg_align_gradprobe_cos_distal_proximal', float('nan')):.3f}/"
                f"{stats.get('leg_align_gradprobe_cos_distal_total', float('nan')):.3f}/"
                f"{stats.get('leg_align_gradprobe_cos_proximal_total', float('nan')):.3f}"
                f" gn(d/p/t)={stats.get('leg_align_gradprobe_norm_distal', float('nan')):.3e}/"
                f"{stats.get('leg_align_gradprobe_norm_proximal', float('nan')):.3e}/"
                f"{stats.get('leg_align_gradprobe_norm_total', float('nan')):.3e}"
            )
            if float(stats.get("leg_align_gradprobe_neg", 0.0) or 0.0) > 0.0:
                msg += " !leg_conflict"
        return msg
    if train_mode == "lambda":
        msg = (
            f"[posttrain][e{epoch} i{it}/{steps_per_epoch}] "
            f"total={stats['total']:.6f} blend={stats['blend_loss']:.6f} "
            f"λ={stats['lambda_mean']:.3f}±{stats['lambda_std']:.3f} "
            f"inc={stats['inc_geo']:.6f} dir={stats['dir_geo']:.6f}"
        )
        if str(cfg.lambda_reliability_mode or "none").strip().lower() not in ("none", "off", "false", "0", ""):
            msg += (
                f" λ_eff={stats.get('lambda_eff_mean', float('nan')):.3f}±{stats.get('lambda_eff_std', float('nan')):.3f}"
                f" r={stats.get('lambda_rel_mean', float('nan')):.3f}"
            )
        if float(cfg.lambda_gate_sup_weight or 0.0) > 0.0:
            msg += (
                f" gate_sup={stats.get('gate_sup_loss', float('nan')):.3e}"
                f" acc@0.5={stats.get('gate_sup_acc@0.5', float('nan')):.3f}"
                f" frac={stats.get('gate_sup_frac', float('nan')):.3f}"
            )
        if float(cfg.lambda_fusion_early_weight or 0.0) > 0.0:
            msg += f" early={stats.get('early_loss', float('nan')):.3e}"
        if float(cfg.lambda_fusion_monotonic_weight or 0.0) > 0.0:
            msg += f" mono={stats.get('mono_loss', float('nan')):.3e}"
        if float(cfg.lambda_plan_entropy_weight or 0.0) > 0.0:
            msg += f" planH={stats.get('plan_entropy_loss', float('nan')):.3e}"
            if "plan_entropy_mean" in stats:
                msg += f" H={stats.get('plan_entropy_mean', float('nan')):.3f}"
        if float(cfg.lambda_plan_dyn_weight or 0.0) > 0.0:
            msg += f" planDyn={stats.get('plan_dyn_loss', float('nan')):.3e}"
            if "plan_dyn_mean" in stats:
                msg += f" dPlan={stats.get('plan_dyn_mean', float('nan')):.3f}"
        if float(l2sp_weight or 0.0) > 0.0:
            msg += f" l2sp={stats.get('l2sp_loss', float('nan')):.3e}"
        return msg
    raise ValueError(f"Unknown train_mode={train_mode!r}")


def _apply_posttrain_cli_overrides(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.paths is not None:
        payload["paths"] = args.paths or None
    _apply_cli_overrides_shared(
        payload,
        args,
        bool_keys=_CLI_BOOL_OVERRIDE_KEYS,
        optional_float_keys=_CLI_OPTIONAL_FLOAT_OVERRIDE_KEYS,
        skip_keys=_CLI_OVERRIDE_SPECIAL_KEYS,
    )


def _run_training_loop(*, cfg: PostTrainConfig, train_mode: str, model: EventMotionModel, params: list[torch.nn.Parameter], opt: torch.optim.Optimizer, batch_iter: Any, rollout_common_kwargs: Dict[str, Any], rollout_mode_kwargs: Dict[str, Any], l2sp_pairs: list[tuple[torch.nn.Parameter, torch.Tensor]], l2sp_weight: float) -> list[dict[str, Any]]:
    log_rows: list[dict[str, Any]] = []
    trainer = rollout_common_kwargs["trainer"]
    direct_mode = train_mode == "direct"
    lambda_mode = train_mode == "lambda"
    epochs = int(cfg.epochs)
    steps_per_epoch = int(cfg.steps_per_epoch)
    direct_grad_monitor_enable = direct_mode and bool(getattr(cfg, "direct_pose_grad_monitor_enable", False))
    leg_align_grad_probe_steps = max(0, int(getattr(cfg, "direct_pose_leg_align_grad_probe_steps", 0) or 0))
    global_step = 0
    save_step_set = _parse_int_set_spec(getattr(cfg, "save_step_ckpts", None))

    def _save_step_snapshot(step_idx: int) -> None:
        if int(step_idx) < 0:
            return
        ckpt_step_out = cfg.out_dir / f"ckpt_step_{int(step_idx):06d}_{cfg.run_name}.pth"
        torch.save({"model": model.state_dict(), "posttrain_cfg": _cfg_to_jsonable(cfg)}, ckpt_step_out)

    if 0 in save_step_set:
        _save_step_snapshot(0)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        ok_steps = 0
        bad_steps = 0
        for it in range(steps_per_epoch):
            batch = next(batch_iter)
            opt.zero_grad(set_to_none=True)
            if direct_mode:
                rollout_mode_kwargs_step = dict(rollout_mode_kwargs)
                rollout_mode_kwargs_step["direct_pose_leg_align_weight"] = _resolve_direct_pose_leg_align_weight(
                    cfg, global_step
                )
            else:
                rollout_mode_kwargs_step = rollout_mode_kwargs
            loss, stats, aux_payload = _lambda_fusion_loss_rollout(
                batch=batch,
                **rollout_common_kwargs,
                **rollout_mode_kwargs_step,
            )
            ema_update_payload = aux_payload.get("ema_update_payload", None) if isinstance(aux_payload, dict) else None
            if direct_mode and isinstance(ema_update_payload, dict):
                try:
                    leg_ema = ema_update_payload.get("leg", None)
                    nonleg_ema = ema_update_payload.get("nonleg", None)
                    valid_ema = torch.is_tensor(leg_ema) and torch.is_tensor(nonleg_ema)
                    valid_ema = valid_ema and bool(torch.isfinite(leg_ema).all().detach().cpu().item())
                    valid_ema = valid_ema and bool(torch.isfinite(nonleg_ema).all().detach().cpu().item())
                    if valid_ema:
                        trainer._direct_pose_group_norm_ema = dict(
                            ema_update_payload,
                            leg=leg_ema.detach(),
                            nonleg=nonleg_ema.detach(),
                        )
                    else:
                        _record_posttrain_soft_fail(trainer, "apply_ema_update_invalid_payload")
                except _ROLLOUT_SOFT_FAIL_ERRORS:
                    _record_posttrain_soft_fail(trainer, "apply_ema_update_setattr")
            elif ema_update_payload is not None:
                _record_posttrain_soft_fail(trainer, "apply_ema_update_nontrain_or_bad_payload")
            if lambda_mode and l2sp_pairs and l2sp_weight > 0.0:
                try:
                    terms = [(p.float() - p0.float()).pow(2).mean() for p, p0 in l2sp_pairs]
                    l2sp_loss = torch.stack(terms).mean() if terms else None
                except Exception:
                    l2sp_loss = None
                if torch.is_tensor(l2sp_loss):
                    loss = loss + (l2sp_loss * float(l2sp_weight))
                    stats["l2sp_loss"] = float(l2sp_loss.detach().item())
                    stats["total"] = float(loss.detach().item())
            if not torch.isfinite(loss):
                bad_steps += 1
                global_step += 1
                print(f"[posttrain][WARN] non-finite loss at step={global_step} (skipped). stats={stats}")
                continue
            if (
                direct_mode
                and leg_align_grad_probe_steps > 0
                and bool(getattr(cfg, "direct_pose_leg_align_grad_probe_enable", False))
                and int(global_step) < leg_align_grad_probe_steps
            ):
                try:
                    _run_leg_align_grad_probe(cfg=cfg, model=model, stats=stats, aux_payload=aux_payload)
                except _ROLLOUT_SOFT_FAIL_ERRORS:
                    _record_posttrain_soft_fail(trainer, "leg_align_grad_probe")
            loss.backward()
            if direct_grad_monitor_enable:
                grad_norms = {
                    "trunk": _grad_norm_of_module(getattr(model, "direct_pose_head", None)),
                    "out_leg": _grad_norm_of_module(getattr(model, "direct_pose_leg_terminal", None)),
                    "out_nonleg_head": _grad_norm_of_module(getattr(model, "direct_pose_out_nonleg", None)),
                    "out_arm": _grad_norm_of_module(getattr(model, "direct_pose_out_arm", None)),
                    "out_else": _grad_norm_of_module(getattr(model, "direct_pose_out_else", None)),
                    "leg_head": _grad_norm_of_module(getattr(model, "direct_pose_leg_head", None)),
                    "leg_head_shared": _grad_norm_of_module(getattr(model, "direct_pose_leg_head_shared", None)),
                }
                for merged_key, source_keys in (
                    ("out_nonleg", ("out_nonleg_head", "out_arm", "out_else")),
                    ("leg_branch", ("leg_head", "leg_head_shared")),
                ):
                    total_sq = 0.0
                    has_grad = False
                    for source_key in source_keys:
                        try:
                            grad_val = float(grad_norms[source_key])
                        except Exception:
                            continue
                        if not math.isfinite(grad_val):
                            continue
                        total_sq += float(grad_val * grad_val)
                        has_grad = True
                    grad_norms[merged_key] = float(math.sqrt(max(0.0, total_sq))) if has_grad else float("nan")
                ratio = float("nan")
                if math.isfinite(grad_norms["out_leg"]) and math.isfinite(grad_norms["out_nonleg"]):
                    ratio = float(grad_norms["out_nonleg"] / max(1e-12, grad_norms["out_leg"]))
                ratio_leg_branch = float("nan")
                if math.isfinite(grad_norms["leg_branch"]) and math.isfinite(grad_norms["out_nonleg"]):
                    ratio_leg_branch = float(grad_norms["out_nonleg"] / max(1e-12, grad_norms["leg_branch"]))
                stats.update(
                    direct_grad_norm_trunk=float(grad_norms["trunk"]),
                    direct_grad_norm_out_leg=float(grad_norms["out_leg"]),
                    direct_grad_norm_out_nonleg=float(grad_norms["out_nonleg"]),
                    direct_grad_norm_out_arm=float(grad_norms["out_arm"]),
                    direct_grad_norm_out_else=float(grad_norms["out_else"]),
                    direct_grad_norm_leg_head=float(grad_norms["leg_head"]),
                    direct_grad_norm_leg_head_shared=float(grad_norms["leg_head_shared"]),
                    direct_grad_norm_leg_branch=float(grad_norms["leg_branch"]),
                    direct_grad_ratio_nonleg_over_leg=float(ratio),
                    direct_grad_ratio_nonleg_over_leg_branch=float(ratio_leg_branch),
                )
                gate_thr = float(getattr(cfg, "direct_pose_grad_ratio_gate", 0.35) or 0.35)
                if math.isfinite(ratio) and math.isfinite(gate_thr) and gate_thr > 0.0:
                    stats["direct_grad_ratio_gate"] = float(gate_thr)
                    stats["direct_grad_ratio_alert"] = 1.0 if ratio < gate_thr else 0.0
                if math.isfinite(ratio_leg_branch) and math.isfinite(gate_thr) and gate_thr > 0.0:
                    stats["direct_grad_ratio_gate_branch"] = float(gate_thr)
                    stats["direct_grad_ratio_alert_leg_branch"] = 1.0 if ratio_leg_branch < gate_thr else 0.0
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            ok_steps += 1
            epoch_loss += float(stats["total"])
            global_step += 1
            if global_step in save_step_set:
                _save_step_snapshot(global_step)
            if (it % 20) == 0:
                msg = _format_posttrain_step_msg(
                    train_mode=train_mode,
                    cfg=cfg,
                    stats=stats,
                    epoch=epoch,
                    it=it,
                    steps_per_epoch=steps_per_epoch,
                    l2sp_weight=l2sp_weight,
                )
                metric_key = "contact_meas_bce" if "contact_meas_bce" in stats else ("contact_meas_mse" if "contact_meas_mse" in stats else "")
                if metric_key:
                    msg += f" {metric_key}={stats[metric_key]:.4f}"
                print(msg)
            row = dict(stats)
            row.update({"gate_mode": train_mode, "epoch": float(epoch), "iter": float(it), "step": float(global_step)})
            log_rows.append(row)

        avg = epoch_loss / max(1, int(ok_steps))
        print(f"[posttrain][epoch {epoch}] avg_total={avg:.6f} ok_steps={ok_steps} skipped={bad_steps}")
    return log_rows

@dataclass(frozen=True)
class PosttrainLocalRuntimeOverlay:
    contact_meas_gate_by_hit_override: Optional[bool]
    contact_meas_vxy_mode: str
    contact_meas_ground_z_mode: str
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_max_up_m: float
    contact_meas_ground_z_max_down_m: float
    contacts_pretrain: ContactPretrainRuntime
    lambda_reliability_mode: str
    lambda_reliability_warmup_steps: int
    lambda_reliability_contact_err_max: float
    lambda_reliability_warmup_joint_scales: Any


def _resolve_posttrain_contact_meas_gate_override(raw_value: Any) -> Optional[bool]:
    gate_raw = str(raw_value or "auto").strip().lower()
    if gate_raw in ("true", "1", "yes", "y"):
        return True
    if gate_raw in ("false", "0", "no", "n"):
        return False
    return None


def _resolve_posttrain_ground_z_slew_limits_m(cfg: PostTrainConfig) -> tuple[float, float]:
    try:
        up_cm = float(getattr(cfg, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0)
        down_cm = float(getattr(cfg, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0)
    except Exception:
        up_cm, down_cm = 0.0, 0.0
    return max(0.0, up_cm) / 100.0, max(0.0, down_cm) / 100.0


def _resolve_posttrain_local_runtime_overlay(cfg: PostTrainConfig) -> PosttrainLocalRuntimeOverlay:
    gate_override = _resolve_posttrain_contact_meas_gate_override(getattr(cfg, "contact_meas_gate_by_hit", "auto"))
    ground_z_max_up_m, ground_z_max_down_m = _resolve_posttrain_ground_z_slew_limits_m(cfg)

    contact_pretrain_runtime = resolve_contact_pretrain_runtime(
        clamp_raw=getattr(cfg, "posttrain_contacts_pretrain_clamp", 1.0),
        affine_stats_raw=getattr(cfg, "posttrain_contacts_pretrain_affine_stats", None),
        warn=False,
    )
    return PosttrainLocalRuntimeOverlay(
        contact_meas_gate_by_hit_override=gate_override,
        contact_meas_vxy_mode=str(getattr(cfg, "contact_meas_vxy_mode", "abs") or "abs").strip().lower(),
        contact_meas_ground_z_mode=str(getattr(cfg, "contact_meas_ground_z_mode", "window") or "window").strip().lower(),
        contact_meas_ground_z_beta=float(getattr(cfg, "contact_meas_ground_z_beta", 0.05) or 0.05),
        contact_meas_ground_z_window=int(getattr(cfg, "contact_meas_ground_z_window", 5) or 5),
        contact_meas_ground_z_quantile=float(getattr(cfg, "contact_meas_ground_z_quantile", 0.2) or 0.2),
        contact_meas_ground_z_max_up_m=float(ground_z_max_up_m),
        contact_meas_ground_z_max_down_m=float(ground_z_max_down_m),
        contacts_pretrain=contact_pretrain_runtime,
        lambda_reliability_mode=str(cfg.lambda_reliability_mode or "none"),
        lambda_reliability_warmup_steps=int(cfg.lambda_reliability_warmup_steps or 0),
        lambda_reliability_contact_err_max=float(cfg.lambda_reliability_contact_err_max or 1.0),
        lambda_reliability_warmup_joint_scales=cfg.lambda_reliability_warmup_joint_scales,
    )


def _apply_posttrain_local_runtime_overlay(
    trainer: Trainer,
    overlay: PosttrainLocalRuntimeOverlay,
) -> None:
    apply_contacts_pretrain_runtime(
        trainer,
        owner_prefix="posttrain",
        runtime=overlay.contacts_pretrain,
    )
    for overlay_field in fields(overlay):
        field_name = str(overlay_field.name)
        if field_name == "contacts_pretrain":
            continue
        setattr(trainer, field_name, getattr(overlay, field_name))


def _save_posttrain_outputs(
    *,
    cfg: PostTrainConfig,
    artifacts: _posttrain_build_shell.PostTrainModelArtifacts,
    log_rows: list[dict[str, Any]],
) -> Path:
    model = artifacts.model
    cfg_jsonable = _cfg_to_jsonable(cfg)
    cfg_jsonable["direct_pose_feat_source"] = str(artifacts.direct_pose_feat_source)
    cfg_jsonable["direct_pose_time_pe_dim"] = int(artifacts.direct_pose_time_pe_dim)
    cfg_jsonable["direct_pose_time_pe_base"] = float(artifacts.direct_pose_time_pe_base)
    cfg_jsonable["direct_pose_use_phase_z"] = bool(artifacts.direct_pose_use_phase_z)
    cfg_jsonable["direct_pose_phase_z_mode"] = str(artifacts.direct_pose_phase_z_mode)
    cfg_jsonable["direct_pose_split_enable"] = bool(artifacts.direct_pose_split_enable)
    cfg_jsonable["direct_pose_nonleg_proj_dim"] = int(artifacts.direct_pose_nonleg_proj_dim)
    cfg_jsonable["direct_pose_arm_split_enable"] = bool(getattr(model, "direct_pose_arm_split_enable", False))
    cfg_jsonable["direct_pose_arm_bones"] = getattr(model, "direct_pose_arm_bones", None)
    cfg_jsonable["direct_pose_nonleg_train_only"] = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
    cfg_jsonable["direct_pose_leg_gate_mode"] = str(artifacts.direct_pose_leg_gate_mode_model)
    cfg_jsonable["direct_pose_leg_gate_power"] = float(artifacts.direct_pose_leg_gate_power_model)
    ckpt_out = cfg.out_dir / f"ckpt_last_{cfg.run_name}.pth"
    torch.save({"model": model.state_dict(), "posttrain_cfg": cfg_jsonable}, ckpt_out)
    dump_json(cfg.out_dir / f"posttrain_log_{cfg.run_name}.json", {"config": cfg_jsonable, "log": log_rows})
    return ckpt_out


def _build_dataset_and_loader(cfg: PostTrainConfig) -> tuple[dict[str, Any], MotionEventDataset, Any]:
    norm_spec = merge_norm_spec(
        cfg.bundle_json.expanduser().resolve(),
        cfg.pretrain_template,
        pretrain_keys=None,
        strict=True,
    )
    ds = build_motion_dataset(
        data_dir=str(cfg.data.expanduser().resolve()),
        seq_len=max(2, int(cfg.seq_len)),
        paths=[str(p.expanduser().resolve()) for p in cfg.paths] if cfg.paths else None,
        norm_spec=norm_spec,
        index_mode=str(getattr(cfg, "dataset_index_mode", "sliding") or "sliding"),
        is_train=True,
    )
    _assert_posttrain_dataset_has_samples(ds=ds, seq_len=int(cfg.seq_len))
    loader = build_motion_dataloader(
        ds,
        batch_size=int(cfg.batch),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    _assert_posttrain_loader_has_batches(loader=loader, ds=ds, batch=int(cfg.batch))
    return norm_spec, ds, _iter_infinite(loader)


def _collect_posttrain_clip_lengths(ds: MotionEventDataset) -> list[tuple[str, int]]:
    clip_lens: list[tuple[str, int]] = []
    for clip in getattr(ds, "clips", []) or []:
        try:
            clip_lens.append((str(getattr(clip, "npz_path", "?")), int(getattr(clip, "X", np.zeros((0,))).shape[0])))
        except Exception:
            pass
    clip_lens.sort(key=lambda x: x[1])
    return clip_lens


def _format_posttrain_empty_dataset_hint(ds: MotionEventDataset) -> str:
    clip_lens = _collect_posttrain_clip_lengths(ds)
    return f" Smallest clips: {', '.join([f'{Path(p).name}:{n}' for p, n in clip_lens[:5]])}." if clip_lens else ""


def _assert_posttrain_dataset_has_samples(*, ds: MotionEventDataset, seq_len: int) -> None:
    if len(ds) > 0:
        return
    hint = _format_posttrain_empty_dataset_hint(ds)
    raise SystemExit(f"[FATAL] posttrain dataset has 0 samples. seq_len={seq_len} is likely too large or paths/data are wrong." + hint + " Try lowering --seq_len or passing --paths to restrict to longer clips.")


def _assert_posttrain_loader_has_batches(*, loader: Any, ds: MotionEventDataset, batch: int) -> None:
    if len(loader) > 0:
        return
    raise SystemExit(f"[FATAL] posttrain DataLoader has 0 batches (len(dataset)={len(ds)}, batch={batch}, drop_last=True). Lower --batch or use more/longer --paths (or reduce --seq_len).")


def _sync_posttrain_bone_names(*, trainer: Trainer, loss_fn: MotionJointLoss, ds: MotionEventDataset) -> None:
    try:
        bone_names_ds = list(getattr(ds, "bone_names", []) or [])
        if bone_names_ds:
            setattr(trainer, "bone_names", bone_names_ds)
            if hasattr(loss_fn, "set_bone_names"):
                loss_fn.set_bone_names(bone_names_ds)
            else:
                setattr(loss_fn, "bone_names", bone_names_ds)
    except Exception:
        pass


def _build_posttrain_loss_and_trainer(*, cfg: PostTrainConfig, ds: MotionEventDataset, model: EventMotionModel) -> Trainer:
    loss_fn = MotionJointLoss(output_layout=getattr(ds, "output_layout", None), fps=float(getattr(ds, "fps", 60.0) or 60.0), rot6d_spec=getattr(ds, "rot6d_spec", None) or {}, meta=getattr(ds, "meta", None) or {})
    trainer = Trainer(model=model, loss_fn=loss_fn, lr=float(getattr(cfg, "lr", 1e-4) or 1e-4), grad_clip=0.0, weight_decay=float(getattr(cfg, "weight_decay", 0.0) or 0.0), use_amp=False, accum_steps=1, pin_memory=False)
    _sync_posttrain_bone_names(trainer=trainer, loss_fn=loss_fn, ds=ds)
    return trainer


def _attach_posttrain_trainer_runtime(*, cfg: PostTrainConfig, ds: MotionEventDataset, trainer: Trainer, norm_spec: dict[str, Any]) -> None:
    dataset_artifacts = build_and_attach_dataset_runtime(
        trainer,
        ds,
        bundle_path=str(cfg.bundle_json.expanduser().resolve()),
        norm_spec=norm_spec,
    )
    apply_shared_trainer_runtime(
        trainer,
        resolve_shared_trainer_runtime(
            dataset_artifacts=dataset_artifacts,
            trainer_default_yaw_forward_axis=int(getattr(trainer, "yaw_forward_axis", 2)),
            bundle_json_path=cfg.bundle_json.expanduser().resolve(),
            out_dir=cfg.out_dir,
            full_config=_cfg_to_jsonable(cfg),
            current_run_name=cfg.run_name,
        ),
    )
    apply_loss_runtime_from_trainer(trainer.loss_fn, trainer)
    _apply_posttrain_local_runtime_overlay(trainer, _resolve_posttrain_local_runtime_overlay(cfg))


def _build_model_and_trainer(*, cfg: PostTrainConfig, ds: MotionEventDataset, model: EventMotionModel, norm_spec: dict[str, Any]) -> Trainer:
    trainer = _build_posttrain_loss_and_trainer(cfg=cfg, ds=ds, model=model)
    _attach_posttrain_trainer_runtime(cfg=cfg, ds=ds, trainer=trainer, norm_spec=norm_spec)
    return trainer

_POSTTRAIN_ARG_SPECS_PATHS_AND_DATA = (
    (("--config",), dict(default="config/posttrain.json", help="Path to post-train JSON config (optional).")),
    (("--ckpt_in",), dict(help="Input checkpoint path (overrides config).")),
    (("--out_dir",), dict(help="Output directory (overrides config).")),
    (("--run_name",), dict(help="Run name (overrides config).")),
    (("--data",), dict(help="Dataset root (processed .npz directory).")),
    (("--paths",), dict(nargs="*", help="Optional explicit .npz paths (overrides config).")),
    (("--bundle_json",), dict(help="Bundle JSON (norm_template.json).")),
    (("--pretrain_template",), dict(help="Optional pretrain template JSON to merge norm spec.")),
    (("--encoder_bundle",), dict(help="Optional motion encoder bundle (.pt).")),
)


_POSTTRAIN_ARG_SPECS_RUNTIME_AND_TRAIN = (
    (("--device",), dict(help="auto|cpu|cuda|mps")),
    (("--batch",), dict(type=int)),
    (("--seq_len",), dict(type=int)),
    (
        ("--dataset_index_mode",),
        dict(type=str, help="Dataset window sampling: sliding|start0|clip_random (balanced per-clip random start)."),
    ),
    (
        ("--rollout_steps",),
        dict(type=int, help="Rollout horizon for loss unroll (<= seq_len-1). 0/None uses full window."),
    ),
    (
        ("--rollout_cycles",),
        dict(type=int, help="Unroll multiple cycles by repeating the (seq_len-1) transitions with modulo indexing."),
    ),
    (
        ("--rollout_include_boundary",),
        dict(type=str, help="true|false; include wrap boundary transitions when rollout_cycles>1 (aligns with freerun_cycles)."),
    ),
    (
        ("--rollout_random_offset",),
        dict(type=str, help="true|false; randomize cycle phase (start offset) per batch when rollout_cycles>1."),
    ),
    (
        ("--time_index_mode",),
        dict(type=str, help="global|cycle|auto|none (time_index feeding for contact_plan time-PE)."),
    ),
    (("--depth",), dict(type=int)),
    (("--num_heads",), dict(type=int)),
    (("--dropout",), dict(type=float)),
    (("--context_len",), dict(type=int)),
    # Train
    (("--epochs",), dict(type=int)),
    (("--steps_per_epoch",), dict(type=int)),
    (("--save_step_ckpts",), dict(type=str, help="Optional step checkpoints to save, e.g. '0,1,5,20,60'.")),
    (("--lr",), dict(type=float)),
    (("--weight_decay",), dict(type=float)),
    # Build / compat / runtime guards
    (
        ("--so3_corr_gate_logit_reset",),
        dict(help="Reset model.so3_corr_gate_logit to a float (e.g. -2.2)."),
    ),
    (("--detach_rollout_state",), dict(type=str, help="true|false")),
    (
        ("--train_direct_pose",),
        dict(type=str, help="true|false; whether to finetune direct_pose_head (direct expert) via rollout loss"),
    ),
    (("--contact_plan_init_mode",), dict(type=str, help=argparse.SUPPRESS)),
    (("--contact_plan_init_hidden",), dict(type=int, help=argparse.SUPPRESS)),
    (("--contact_plan_init_dropout",), dict(type=float, help=argparse.SUPPRESS)),
    (
        ("--event_clock",),
        dict(
            type=str,
            choices=("auto", "on", "off"),
            help="Event-Clock v3 mode: auto|on|off (auto keeps ckpt behavior; off drops weights on save).",
        ),
    ),
    (("--event_clock_max_delta",), dict(type=float, help="Event-Clock clamp for Δz residual magnitude.")),
    (("--event_clock_hidden_dim",), dict(type=int, help="Override Event-Clock corrector hidden dim (Δz MLP).")),
    (
        ("--event_clock_gate_hidden_dim",),
        dict(type=int, help="Override Event-Clock gate hidden dim (λ_corr MLP)."),
    ),
    (("--train_lambda_head",), dict(type=str, help="true|false; whether to train lambda_fusion_head (Stage2)")),
    (("--contact_meas_weight",), dict(type=float, help="Weight for contact_meas MSE vs GT soft contacts.")),
)


_POSTTRAIN_ARG_SPECS_DIRECT_POSE_BUILD = (
    (
        ("--direct_pose_split_enable",),
        dict(type=str, help="true|false; split direct output heads into leg/non-leg with shared trunk (B2)."),
    ),
    (
        ("--direct_pose_nonleg_proj_dim",),
        dict(type=int, help="Optional non-leg bottleneck dim for split head: h_nonleg=ReLU(Linear(hid,proj)); 0 disables."),
    ),
    (
        ("--direct_pose_arm_split_enable",),
        dict(type=str, help="true|false; split non-leg branch into arm/else heads (three-way: leg/arm/else)."),
    ),
    (
        ("--direct_pose_arm_bones",),
        dict(type=str, help="Comma-separated bone names/indices for arm branch when direct_pose_arm_split_enable=true."),
    ),
    (
        ("--direct_pose_nonleg_train_only",),
        dict(type=str, help="true|false; when train_direct_pose, freeze trunk/leg and train non-leg branch only."),
    ),
    (
        ("--direct_pose_leg_enable",),
        dict(type=str, help="true|false; enable leg-specific residual head for direct pose (extra lower-body capacity)."),
    ),
    (
        ("--direct_pose_leg_train_only",),
        dict(type=str, help="true|false; when train_direct_pose, freeze direct_pose_head and train leg head only."),
    ),
    (
        ("--direct_pose_leg_bones",),
        dict(type=str, help="Comma-separated bone names/indices for leg head (default: ball/foot/calf/thigh L+R)."),
    ),
    (
        ("--direct_pose_leg_mode",),
        dict(
            type=str,
            choices=("rot6d_add", "so3"),
            help="Leg residual mode: rot6d_add (compat) | so3 (on-manifold compose exp(omega)@R).",
        ),
    ),
    (
        ("--direct_pose_leg_stopgrad_main",),
        dict(type=str, help="true|false; when leg_mode=so3, stop-grad main head leg rotations in the composition."),
    ),
    (
        ("--direct_pose_leg_detach_feat",),
        dict(type=str, help="true|false; detach leg head inputs so leg loss won't update the backbone (strong decoupling)."),
    ),
    (("--direct_pose_leg_max_deg",), dict(type=float, help="Max ||omega|| in degrees for leg_mode=so3. 0 disables.")),
    (
        ("--direct_pose_leg_gate_mode",),
        dict(
            type=str,
            choices=("none", "learned", "scale"),
            help=(
                "Optional learned gate/scale for leg omega (SO(3) only): "
                "none | learned | scale."
            ),
        ),
    ),
    (
        ("--direct_pose_leg_gate_power",),
        dict(type=float, help="Gate power for leg omega (SO(3) only): omega_eff = sigmoid(gate_logits)**power * omega_raw."),
    ),
    (
        ("--direct_pose_leg_scale_clamp_k",),
        dict(type=float, help="Optional hard clamp on leg scale magnitude: k>1 => [1/k, k]. 0/1 disables."),
    ),
    (
        ("--direct_pose_leg_gate_sup_weight",),
        dict(
            dest="direct_pose_leg_gate_sup_weight",
            type=float,
            help="Optional supervised loss weight for learned leg gate (BCEWithLogits vs oracle ||omega_oracle|| thresholding). 0 disables.",
        ),
    ),
)


_POSTTRAIN_ARG_SPECS_DIRECT_POSE_LEG_ALIGN = (
    (
        ("--direct_pose_leg_align_weight",),
        dict(
            type=float,
            help="Optional direction alignment loss target weight for leg omega. When schedule=none it is constant; otherwise it is the ramp target.",
        ),
    ),
    (
        ("--direct_pose_leg_align_schedule",),
        dict(
            type=str,
            choices=("none", "linear"),
            help="Leg align weight schedule: none | linear (hold start_weight for warmup_steps, then ramp to target weight).",
        ),
    ),
    (
        ("--direct_pose_leg_align_start_weight",),
        dict(type=float, help="Leg align schedule start weight used before/at ramp start. Default 0."),
    ),
    (
        ("--direct_pose_leg_align_warmup_steps",),
        dict(type=int, help="Leg align schedule warmup steps that keep start_weight before the ramp begins."),
    ),
    (
        ("--direct_pose_leg_align_ramp_steps",),
        dict(type=int, help="Leg align schedule ramp length in optimizer steps. 0 means jump directly to target after warmup."),
    ),
    (
        ("--direct_pose_leg_align_oracle_min_deg",),
        dict(type=float, help="Oracle gate for leg omega alignment loss: only apply when ||omega_oracle|| >= this (deg)."),
    ),
    (
        ("--direct_pose_leg_align_oracle_weight_deg",),
        dict(type=float, help="Optional stop-grad weight ramp for leg omega alignment loss: w=clamp(||omega_oracle||/deg,0,1). 0 disables."),
    ),
    (
        ("--direct_pose_leg_align_mode",),
        dict(
            type=str,
            choices=("cos", "proj"),
            help="Leg omega alignment loss form: cos (relu(-cos), cheatable) | proj (mag+res, non-cheating).",
        ),
    ),
    (
        ("--direct_pose_leg_align_mag_weight",),
        dict(type=float, help="align_mode=proj: weight for projection magnitude term (proj-||oracle||)^2."),
    ),
    (
        ("--direct_pose_leg_align_res_weight",),
        dict(type=float, help="align_mode=proj: weight for orthogonal residual term ||res||^2."),
    ),
    (
        ("--direct_pose_leg_align_sign_weight",),
        dict(type=float, help="align_mode=proj: optional weight for relu(-proj)^2 sign penalty (rad^2)."),
    ),
    (
        ("--direct_pose_leg_align_cos_thresh",),
        dict(type=float, help="Optional hard-example mining: apply leg omega alignment only when cos(pred, oracle) < thresh. 0 disables."),
    ),
    (
        ("--direct_pose_leg_align_target_joints",),
        dict(type=str, help="Optional joint subset for the main leg_align objective. Supports presets like 'distal', 'proximal', 'calf', 'thigh', 'foot', 'ball'."),
    ),
    (
        ("--direct_pose_leg_align_anchor_joints",),
        dict(type=str, help="Optional auxiliary joint subset added on top of the main leg_align objective (e.g. 'calf')."),
    ),
    (
        ("--direct_pose_leg_align_anchor_weight",),
        dict(type=float, help="Relative weight for the auxiliary leg_align anchor subset. 0 disables the anchor."),
    ),
)


_POSTTRAIN_ARG_SPECS_DIRECT_POSE_OBJECTIVE_CORE = (
    (
        ("--direct_pose_loss_leg_split",),
        dict(type=str, help="Stage7 direct objective: true|false; split legs vs non-legs: L = mean(nonleg) + mean(leg)."),
    ),
    (
        ("--direct_pose_nonleg_focus_bones",),
        dict(type=str, help='Optional: comma-separated non-leg bones/indices to upweight inside L_nonleg (e.g. "upperarm_l,lowerarm_l,hand_l,pinky_01_l").'),
    ),
    (
        ("--direct_pose_nonleg_focus_weight",),
        dict(type=float, help="Only for --direct_pose_nonleg_focus_bones: multiplicative per-bone weight (>1 boosts selected bones; 1 disables)."),
    ),
)


_POSTTRAIN_ARG_SPECS_DIRECT_POSE_MONITORING = (
    (
        ("--direct_pose_loss_group_norm_enable",),
        dict(type=str, help="true|false; enable group-wise magnitude normalization for direct loss (leg vs non-leg)."),
    ),
    (("--direct_pose_loss_group_norm_w_leg",), dict(type=float, help="Weight for normalized leg group loss term.")),
    (("--direct_pose_loss_group_norm_w_nonleg",), dict(type=float, help="Weight for normalized non-leg group loss term.")),
    (
        ("--direct_pose_loss_group_norm_ema_beta",),
        dict(type=float, help="EMA beta for group-wise magnitude normalization (no warmup switch; first batch initializes EMA)."),
    ),
    (("--direct_pose_loss_group_norm_ratio_min",), dict(type=float, help="Lower clamp for normalized group ratio L/EMA.")),
    (("--direct_pose_loss_group_norm_ratio_max",), dict(type=float, help="Upper clamp for normalized group ratio L/EMA.")),
    (("--direct_pose_loss_group_norm_eps",), dict(type=float, help="Numerical epsilon for group-wise normalization denominator.")),
    (
        ("--direct_pose_grad_monitor_enable",),
        dict(type=str, help="true|false; log direct split-head grad norms (trunk/out_leg/out_nonleg)."),
    ),
    (
        ("--direct_pose_grad_ratio_gate",),
        dict(type=float, help="Alert threshold for grad_ratio = grad_nonleg / (grad_leg + eps)."),
    ),
)


_POSTTRAIN_ARG_SPECS_DIRECT_POSE_OBJECTIVE = (
    _POSTTRAIN_ARG_SPECS_DIRECT_POSE_LEG_ALIGN
    + _POSTTRAIN_ARG_SPECS_DIRECT_POSE_OBJECTIVE_CORE
    + _POSTTRAIN_ARG_SPECS_DIRECT_POSE_MONITORING
)


_POSTTRAIN_ARG_SPECS_CONTACT_MEAS = (
    (
        ("--contact_meas_gate_by_hit",),
        dict(
            type=str,
            choices=("auto", "true", "false"),
            help="Override white-box gate_by_hit used by validation diagnostics: auto|true|false.",
        ),
    ),
    (
        ("--contact_meas_vxy_mode",),
        dict(
            type=str,
            choices=("abs", "root_rel"),
            help="White-box vxy gate: abs uses ||v_foot_xy||, root_rel uses ||v_foot_xy - v_root_xy|| (more robust under translation).",
        ),
    ),
    (
        ("--contact_meas_ground_z_mode",),
        dict(type=str, choices=("ema", "window", "slew"), help="White-box ground_z update mode: ema|window|slew."),
    ),
    (("--contact_meas_ground_z_beta",), dict(type=float, help="EMA beta for ground_z when mode=ema.")),
    (("--contact_meas_ground_z_window",), dict(type=int, help="Window length when mode=window.")),
    (("--contact_meas_ground_z_quantile",), dict(type=float, help="Low-quantile (0..1) when mode=window.")),
    (
        ("--contact_meas_ground_z_slew_up_cm",),
        dict(type=float, help="Max upward change (cm/step) after ground_z update (0 disables)."),
    ),
    (
        ("--contact_meas_ground_z_slew_down_cm",),
        dict(type=float, help="Max downward change (cm/step) after ground_z update (0 disables)."),
    ),
    (
        ("--posttrain_contacts_pretrain_clamp",),
        dict(type=float, help="Clamp frozen encoder input to [-k,+k] for rollout contact resolution."),
    ),
    (
        ("--posttrain_contacts_pretrain_affine_stats",),
        dict(type=str, help="Optional affine stats JSON path or JSON string (scale/bias/eps) for pretrain contact calibration."),
    ),
)


_POSTTRAIN_ARG_SPECS_LAMBDA_FUSION = (
    (("--lambda_fusion_mode",), dict(type=str, help="global|per_joint")),
    (("--lambda_fusion_hidden",), dict(type=int)),
    (("--lambda_fusion_dropout",), dict(type=float)),
    (("--lambda_fusion_logit_init",), dict(type=float)),
    (("--lambda_fusion_use_rollout_step",), dict(type=str, help="true|false; concat rollout_step into lambda head input")),
    (("--lambda_fusion_entropy_weight",), dict(type=float)),
    (("--lambda_fusion_smooth_weight",), dict(type=float)),
    (("--lambda_fusion_early_steps",), dict(type=int, help="Penalize lambda_mean for the first K rollout steps (protect early).")),
    (("--lambda_fusion_early_weight",), dict(type=float, help="Weight for early-step lambda prior loss.")),
    (("--lambda_fusion_monotonic_weight",), dict(type=float, help="Weight for soft monotonic loss: sum(ReLU(lambda[t-1]-lambda[t])).")),
    (("--lambda_plan_entropy_weight",), dict(type=float, help="Penalty weight: lambda_mean * mean(H(contacts_plan)).")),
    (("--lambda_plan_dyn_weight",), dict(type=float, help="Penalty weight: lambda_mean * mean(|contacts_plan[t]-contacts_plan[t-1]|).")),
    (("--lambda_time_weight_mode",), dict(type=str, help="inv|linear|uniform (rollout step weights for lambda loss)")),
    (("--lambda_time_weight_max",), dict(type=float)),
    (
        ("--lambda_reliability_mode",),
        dict(type=str, help="none|warmup|contacts_err|warmup+contacts_err (deterministic r_t applied to λ for blend; shared in posttrain+freerun)."),
    ),
    (
        ("--lambda_reliability_warmup_steps",),
        dict(type=int, help="Warmup steps K for r_t ramp 0->1 when mode includes warmup."),
    ),
    (
        ("--lambda_reliability_contact_err_max",),
        dict(type=float, help="contacts_err_abs_mean scale for r_t=clamp(1-err/max,0,1) when mode includes contacts_err."),
    ),
    (
        ("--lambda_reliability_warmup_joint_scales",),
        dict(type=str, help="Optional per-joint warmup scales: JSON list (e.g. '[1,1,2,...]') or a JSON file path containing list/scales."),
    ),
    (("--lambda_l2sp_weight",), dict(type=float, help="Optional L2-SP weight to keep trainable head params close to init (improves generalization).")),
    (("--lambda_boundary_weight",), dict(type=float, help="Boundary loss weight multiplier when rollout_include_boundary=true (0 disables boundary supervision).")),
    (("--lambda_gate_sup_weight",), dict(type=float, help="Stage2: gate supervision weight (BCE on lambda_fusion_logits vs oracle soft label). 0 disables.")),
    (("--lambda_gate_sup_tau_deg",), dict(type=float, help="Stage2: τ (deg) for soft label: lambda*=sigmoid((err_inc-err_dir)/τ).")),
    (("--lambda_gate_sup_margin_deg",), dict(type=float, help="Stage2: margin δ (deg); supervise only when |err_inc-err_dir|>=δ. Default is 1°. Use 0 to disable.")),
    (("--lambda_gate_sup_start_step",), dict(type=int, help="Stage2: start rollout step for gate supervision. -1 auto uses lambda_reliability_warmup_steps when warmup enabled.")),
    (("--seed",), dict(type=int)),
)


_POSTTRAIN_ARG_SPECS = (
    _POSTTRAIN_ARG_SPECS_PATHS_AND_DATA
    + _POSTTRAIN_ARG_SPECS_RUNTIME_AND_TRAIN
    + _POSTTRAIN_ARG_SPECS_DIRECT_POSE_BUILD
    + _POSTTRAIN_ARG_SPECS_DIRECT_POSE_OBJECTIVE
    + _POSTTRAIN_ARG_SPECS_CONTACT_MEAS
    + _POSTTRAIN_ARG_SPECS_LAMBDA_FUSION
)


def _build_posttrain_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Post-train entry.\n"
            "Recommended newflow targets: train_direct_pose (Stage6/7) or train_lambda_head (lambda final).\n"
            "Legacy targets are retired and no longer supported."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    for flags, kwargs in _POSTTRAIN_ARG_SPECS:
        ap.add_argument(*flags, **kwargs)
    return ap


def main() -> None:
    ap = _build_posttrain_arg_parser()
    args = ap.parse_args()

    # ---- Config / mode selection ----
    base_cfg = load_json(Path(args.config).expanduser()) if args.config else {}
    payload: Dict[str, Any] = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    _apply_posttrain_cli_overrides(payload, args)

    cfg = _cfg_from_payload(payload)
    train_mode = _resolve_train_mode(cfg)
    seed = int(cfg.seed or 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)) and (not bool(getattr(cfg, "direct_pose_loss_leg_split", False))):
        print("[posttrain][WARN] direct_pose_loss_group_norm_enable=true but direct_pose_loss_leg_split=false; group norm will have no effect.")

    # ---- Dataset / model build ----
    device = _resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    norm_spec, ds, batch_iter = _build_dataset_and_loader(cfg)
    artifacts = _posttrain_build_shell._build_posttrain_model_from_ckpt(cfg=cfg, ds=ds, device=device)
    model = artifacts.model

    # ---- Runtime contracts / trainer ----
    if bool(getattr(model, "contact_plan_enable", False)):
        if cfg.encoder_bundle is None or (not cfg.encoder_bundle.expanduser().is_file()):
            raise SystemExit(
                "[FATAL] rollout contact resolution requires --encoder_bundle with frozen encoder/contact_head."
            )
        if getattr(model, "frozen_encoder", None) is None or getattr(model, "frozen_contact_head", None) is None:
            raise SystemExit(
                "[FATAL] rollout contact resolution requires bundle keys 'encoder' and 'contact_head'."
            )

    trainer = _build_model_and_trainer(cfg=cfg, ds=ds, model=model, norm_spec=norm_spec)
    print(f"[posttrain] mode={'train_direct_pose' if train_mode == 'direct' else 'train_lambda_head'}")

    # ---- Train runtime ----
    _freeze_all(model)
    _unfreeze_for_train_mode(
        model,
        train_mode=train_mode,
        direct_pose_leg_train_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
        direct_pose_leg_gate_train_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
        direct_pose_nonleg_train_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
    )
    model.train()

    params, names = _select_trainable_params(model)
    if not params:
        raise SystemExit("[FATAL] No trainable parameters selected for post-train.")
    print(f"[posttrain] trainable={len(params)} params: {', '.join(names[:8])}{' ...' if len(names)>8 else ''}")
    expected_prefix_map: Dict[str, Tuple[str, ...]] = {
        "lambda": ("lambda_fusion_head",),
        "direct": (
            "direct_pose_head",
            "direct_pose_leg_terminal",
            "direct_pose_out_nonleg",
            "direct_pose_nonleg_proj",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_leg_head",
            "direct_pose_leg_gate_head",
        ),
    }
    try:
        expected_prefixes = list(expected_prefix_map[train_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown train_mode={train_mode!r}") from exc
    if expected_prefixes:
        unexpected = [n for n in names if not any(n.startswith(p) for p in expected_prefixes)]
        if unexpected:
            print(f"[posttrain][WARN] unexpected trainable params (prefixes={expected_prefixes}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    l2sp_pairs: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    l2sp_weight = float(getattr(cfg, "lambda_l2sp_weight", 0.0) or 0.0)
    if l2sp_weight > 0.0:
        try:
            l2sp_pairs = [(p, p.detach().clone()) for p in params]
        except Exception:
            l2sp_pairs = []
        if l2sp_pairs:
            print(f"[posttrain] lambda_l2sp_weight={l2sp_weight:g} (anchor_tensors={len(l2sp_pairs)})")

    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # ---- Rollout runtime ----
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
    rollout_mode_kwargs = _build_rollout_mode_kwargs(cfg, train_mode)

    log_rows = _run_training_loop(cfg=cfg, train_mode=train_mode, model=model, params=params, opt=opt, batch_iter=batch_iter, rollout_common_kwargs=rollout_common_kwargs, rollout_mode_kwargs=rollout_mode_kwargs, l2sp_pairs=l2sp_pairs, l2sp_weight=l2sp_weight)
    ckpt_out = _save_posttrain_outputs(cfg=cfg, artifacts=artifacts, log_rows=log_rows)
    print(f"[posttrain][OK] saved: {ckpt_out}")


if __name__ == "__main__":
    main()
