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
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.configuration.io import dump_json, load_json
from train.dataset import MotionEventDataset
from train.geometry import (
    geodesic_R_safe as _geodesic_R_safe,
    matrix_to_rot6d,
    normalize_rot6d_delta,
    reproject_rot6d,
    rot6d_to_matrix,
    so3_exp_map,
    so3_log_map,
)
from train.history import (
    PoseHistState,
    advance_pose_hist_state,
    init_pose_hist_state,
    resolve_pose_hist_input,
)
from train.layout import DataNormalizer, parse_layout_entry
from train.models import EventMotionModel, MotionJointLoss
from train.rotvec_semantics import require_standard_rotvec_spec
from train.training_MPL import Trainer, validate_and_fix_model_


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
    # Internal phase reset source for contact_phase_state (train/infer consistency):
    # - contacts_meas: threshold-crossing event from contacts_meas
    # - none         : disable event resets
    phase_reset_source: str
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
    direct_pose_leg_gate_mode: str  # none|learned|scale (auto is kept as backward-compatible alias of "none")
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
    # Optional: per-side routing + shared leg head (see train/models.py).
    direct_pose_leg_side_routing: bool
    direct_pose_leg_contact_order: str  # "lr" (default) or "rl"
    direct_pose_leg_side_embed_dim: int
    # Optional: append other-side plan scalar to each routed shared omega head input.
    direct_pose_leg_side_plan_other: bool
    # Optional: append other-side phase (sin,cos) and/or explicit relative phase to each routed shared omega head input.
    direct_pose_leg_side_phase_other: bool
    direct_pose_leg_side_phase_rel: bool
    # Optional: extra per-side cue appended to routed shared leg head input (1 scalar per side).
    direct_pose_leg_side_cue: str
    direct_pose_leg_side_cue_tau: float
    # Optional: per-side sign gate for routed shared leg omega head (same scalar for all joints on that side).
    direct_pose_leg_side_sign_gate: bool
    # Optional: enforce rank-1 (shared direction + per-joint non-negative scale) structure for routed shared leg omega.
    direct_pose_leg_side_rank1: bool
    # Optional: regularizer weight to encourage |g_side| -> 1 (avoid collapsing to 0).
    direct_pose_leg_side_sign_gate_reg_weight: float
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
    # Optional: focus direct objective on selected step_in_cycle (sic) indices.
    # When set (e.g. "8,12,14,15,53,54,55,74"), losses in objective="direct" are computed only
    # on those steps (expanded across rollout_cycles). This is a debug-friendly way to finetune
    # phase-locked hotspots without globally reweighting the whole cycle.
    direct_pose_loss_sics: Optional[str]
    # Optional: when direct_pose_loss_sics is set, only apply it for rollout cycles >= this value.
    # (0 means "all cycles"; 1 matches the common eval mask "cycle>=1".)
    direct_pose_loss_cycle_gte: int
    # When direct_pose_loss_sics is set, how to use it:
    # - mask : compute direct objective loss ONLY on selected steps (hard focus)
    # - boost: compute loss on all steps, but upweight selected steps by a constant factor
    direct_pose_loss_sic_mode: str
    # Only used when direct_pose_loss_sic_mode="boost": multiplicative weight for selected steps.
    direct_pose_loss_sic_boost: float
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
    # Used when rollouts call Trainer._contact_meas_whitebox.
    contact_meas_gate_by_hit: str  # auto|true|false
    contact_meas_vxy_mode: str  # abs|root_rel
    contact_meas_ground_z_mode: str  # ema|window|slew
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_slew_up_cm: float
    contact_meas_ground_z_slew_down_cm: float

    # Posttrain rollout contacts source (fixed mainline contract).
    posttrain_contacts_source: str  # pretrain_contact
    posttrain_contacts_pretrain_clamp: float
    posttrain_contacts_pretrain_affine_stats: Optional[str]

    seed: int


_RETIRED_POSTTRAIN_TARGET_KEYS: tuple[str, ...] = (
    "train_so3_corrector",
    "train_contact_plan_init",
    "train_contact_plan",
    "train_contact_meas",
    "train_contact_td_hazard",
)
_RETIRED_POSTTRAIN_SHELL_KEY_PREFIXES: tuple[str, ...] = (
    "direct_pose_hinge_",
    "contact_td_hazard_",
    "contact_ttc_",
)
_RETIRED_POSTTRAIN_SHELL_EXACT_KEYS: tuple[str, ...] = (
    "train_contact_ttc",
    "direct_hinge_delta",
)

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
    "auto": "none",
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

_DIRECT_POSE_FEAT_SOURCE_CANONICAL: Tuple[str, ...] = ("cond", "hidden", "hidden_pre", "cond+hidden", "cond+hidden_pre")
_DIRECT_POSE_FEAT_SOURCE_ALIAS_MAP: Dict[str, str] = {
    "h": "hidden",
    "h_final": "hidden",
    "hidden_only": "hidden",
    "post": "hidden",
    "final": "hidden",
    "h_pre": "hidden_pre",
    "h_temporal": "hidden_pre",
    "pre": "hidden_pre",
    "temporal": "hidden_pre",
    "mid": "hidden_pre",
    "cond_hidden": "cond+hidden",
    "hidden_cond": "cond+hidden",
    "concat": "cond+hidden",
    "hidden+cond": "cond+hidden",
    "cond_hidden_pre": "cond+hidden_pre",
    "hidden_pre+cond": "cond+hidden_pre",
    "cond+pre": "cond+hidden_pre",
    "pre+cond": "cond+hidden_pre",
}


def _as_path(val: Any) -> Optional[Path]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    return Path(s).expanduser()


def _as_bool(val: Any, default: bool) -> bool:
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "on"):
        return True
    if s in ("0", "false", "no", "n", "f", "off", "none", "null", ""):
        return False
    return bool(val)


def _normalize_optional_csv(val: Any) -> Optional[str]:
    if isinstance(val, (list, tuple)):
        toks = [str(x).strip() for x in val if str(x).strip()]
        return ",".join(toks) if toks else None
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _parse_int_set_spec(spec: Any) -> set[int]:
    if spec is None:
        return set()
    out: set[int] = set()
    for tok in str(spec).replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t or ":" in t:
            sep = "-" if "-" in t else ":"
            a, b = [x.strip() for x in t.split(sep, 1)]
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                lo = int(a)
                hi = int(b)
                if lo > hi:
                    lo, hi = hi, lo
                for v in range(lo, hi + 1):
                    out.add(int(v))
            continue
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out


def _normalize_direct_pose_feat_source(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "auto"):
        return None
    s = _DIRECT_POSE_FEAT_SOURCE_ALIAS_MAP.get(s, s)
    if s in _DIRECT_POSE_FEAT_SOURCE_CANONICAL:
        return s
    return None


def _as_float_list(val: Any) -> Optional[list[float]]:
    if val is None:
        return None
    payload = val
    if isinstance(payload, Path):
        payload = str(payload)
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return None
        # 1) Allow passing a JSON file path
        try:
            p = Path(s).expanduser()
            if p.is_file():
                payload = load_json(p)
            else:
                payload = json.loads(s)
        except Exception:
            return None
    if isinstance(payload, dict):
        if "scales" in payload:
            payload = payload.get("scales")
        elif "values" in payload:
            payload = payload.get("values")
        else:
            return None
    if not isinstance(payload, (list, tuple)):
        return None
    out: list[float] = []
    for x in payload:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out if out else None


def _parse_pretrain_contact_affine_spec(spec: Any) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    raw = spec
    if isinstance(raw, Path):
        raw = str(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            p = Path(s).expanduser()
            if p.is_file():
                raw = load_json(p)
            else:
                raw = json.loads(s)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    scale = raw.get("scale", None)
    bias = raw.get("bias", None)
    if not isinstance(scale, (list, tuple)) or not isinstance(bias, (list, tuple)):
        return None
    try:
        scale_vals = [float(x) for x in scale]
        bias_vals = [float(x) for x in bias]
    except Exception:
        return None
    if len(scale_vals) <= 0 or len(scale_vals) != len(bias_vals):
        return None
    if not all(math.isfinite(float(x)) for x in scale_vals):
        return None
    if not all(math.isfinite(float(x)) for x in bias_vals):
        return None
    try:
        eps = float(raw.get("eps", 1e-4) or 1e-4)
    except Exception:
        eps = 1e-4
    if not math.isfinite(float(eps)):
        eps = 1e-4
    eps = float(min(1e-2, max(1e-8, eps)))
    return {"scale": scale_vals, "bias": bias_vals, "eps": eps}


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


def _canon_phase_reset_source(val: Any) -> str:
    s = str(val or "none").strip().lower()
    # External TTC-driven resets (handled in posttrain rollout loops, similar to run_freerun_cycles):
    # - ttc_gt  : use GT touchdown events (ttc_td_events) to reset phase_z to the anchor [0,1]
    if s in ("none", "null", "off", "disable", "disabled"):
        return "none"
    if s in ("ttc", "ttc_gt", "ttcgt"):
        return "ttc_gt"
    if s in ("contacts", "contacts_meas", "meas", "contact_meas"):
        return "contacts_meas"
    raise SystemExit(
        f"[FATAL] unsupported phase_reset_source={s!r}; allowed values: none | contacts_meas | ttc_gt."
    )


def _cfg_pick(payload: Dict[str, Any], key: str, *, aliases: Tuple[str, ...] = ()) -> Any:
    if key in payload:
        return payload.get(key)
    for alias in aliases:
        if alias in payload:
            return payload.get(alias)
    return None


def _cfg_get_bool(payload: Dict[str, Any], key: str, default: bool, *, aliases: Tuple[str, ...] = ()) -> bool:
    return _as_bool(_cfg_pick(payload, key, aliases=aliases), default)


def _clamp_int(val: int, *, min_value: Optional[int], max_value: Optional[int]) -> int:
    if min_value is not None:
        val = max(int(min_value), int(val))
    if max_value is not None:
        val = min(int(max_value), int(val))
    return int(val)


def _clamp_float(val: float, *, min_value: Optional[float], max_value: Optional[float]) -> float:
    if min_value is not None:
        val = max(float(min_value), float(val))
    if max_value is not None:
        val = min(float(max_value), float(val))
    return float(val)


def _cfg_get_int(
    payload: Dict[str, Any],
    key: str,
    default: Optional[int],
    *,
    aliases: Tuple[str, ...] = (),
    allow_none: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    raw = _cfg_pick(payload, key, aliases=aliases)
    if raw is None:
        val = None if allow_none else default
    else:
        try:
            val = int(raw)
        except Exception:
            val = None if allow_none else default
    if val is None:
        return None
    return _clamp_int(int(val), min_value=min_value, max_value=max_value)


def _cfg_get_float(
    payload: Dict[str, Any],
    key: str,
    default: Optional[float],
    *,
    aliases: Tuple[str, ...] = (),
    allow_none: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    require_finite: bool = True,
) -> Optional[float]:
    raw = _cfg_pick(payload, key, aliases=aliases)
    if raw is None:
        val = None if allow_none else default
    else:
        try:
            val = float(raw)
        except Exception:
            val = None if allow_none else default
    if val is None:
        return None
    if require_finite and (not math.isfinite(float(val))):
        val = None if allow_none else default
    if val is None:
        return None
    return _clamp_float(float(val), min_value=min_value, max_value=max_value)


def _cfg_get_enum(
    payload: Dict[str, Any],
    key: str,
    default: str,
    *,
    aliases: Tuple[str, ...] = (),
    alias_map: Optional[Dict[str, str]] = None,
    choices: Optional[Tuple[str, ...]] = None,
    lower: bool = True,
) -> str:
    raw = _cfg_pick(payload, key, aliases=aliases)
    s = str(default) if raw is None else str(raw)
    s = s.strip()
    s_cmp = s.lower() if lower else s
    if alias_map:
        s_cmp = alias_map.get(s_cmp, s_cmp)
    if choices and s_cmp not in choices:
        return str(default)
    return s_cmp


def _cfg_get_or(payload: Dict[str, Any], key: str, default: Any, *, aliases: Tuple[str, ...] = ()) -> Any:
    raw = _cfg_pick(payload, key, aliases=aliases)
    return raw or default


def _cfg_get_str_or(payload: Dict[str, Any], key: str, default: str, *, aliases: Tuple[str, ...] = ()) -> str:
    return str(_cfg_get_or(payload, key, default, aliases=aliases))


def _cfg_get_int_or(
    payload: Dict[str, Any],
    key: str,
    default: int,
    *,
    aliases: Tuple[str, ...] = (),
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    val = int(_cfg_get_or(payload, key, default, aliases=aliases))
    return _clamp_int(val, min_value=min_value, max_value=max_value)


def _cfg_get_float_or(
    payload: Dict[str, Any],
    key: str,
    default: float,
    *,
    aliases: Tuple[str, ...] = (),
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    val = float(_cfg_get_or(payload, key, default, aliases=aliases))
    return _clamp_float(val, min_value=min_value, max_value=max_value)


def _cfg_get_int_present(payload: Dict[str, Any], key: str, default: int, *, aliases: Tuple[str, ...] = ()) -> int:
    raw = _cfg_pick(payload, key, aliases=aliases)
    return int(default) if raw is None else int(raw)


def _cfg_get_float_present(payload: Dict[str, Any], key: str, default: float, *, aliases: Tuple[str, ...] = ()) -> float:
    raw = _cfg_pick(payload, key, aliases=aliases)
    return float(default) if raw is None else float(raw)


def _cfg_from_schema(payload: Dict[str, Any], schema: List[Tuple[str, Callable[..., Any], Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, getter, kwargs in schema:
        out[name] = getter(payload, **kwargs)
    return out


def _cfg_reject_retired_targets(payload: Dict[str, Any]) -> None:
    # 2026-02-07 Stage7 cleanup: TTC loss/training is removed end-to-end.
    # Keep configs that *mention* TTC keys but do not enable them (all weights==0, train_contact_ttc==false)
    # so old stage configs remain runnable, but fail-fast if a TTC run is requested.
    try:
        _ttc_train = _as_bool(payload.get("train_contact_ttc", False), False)
        _ttc_w = float(payload.get("contact_ttc_weight") or 0.0)
        _ttc_cw = float(payload.get("contact_ttc_consistency_weight") or 0.0)
        _ttc_evtw = float(payload.get("contact_ttc_event_weight") or 0.0)
        _ttc_smallw = float(payload.get("contact_ttc_small_weight") or 0.0)
        _ttc_rollw = float(payload.get("contact_ttc_rollout_weight") or 0.0)
    except Exception:
        _ttc_train = False
        _ttc_w = _ttc_cw = _ttc_evtw = _ttc_smallw = _ttc_rollw = 0.0
    if bool(_ttc_train) or any(float(x) > 0.0 for x in (_ttc_w, _ttc_cw, _ttc_evtw, _ttc_smallw, _ttc_rollw)):
        raise SystemExit(
            "[FATAL] TTC loss/training has been removed (Stage7 cleanup; 2026-02-07). "
            "Migrate to phase_reset_source=none (no-reset) or phase_reset_source=contacts_meas."
        )

    present_retired_keys = [k for k in _RETIRED_POSTTRAIN_TARGET_KEYS if k in payload]
    if present_retired_keys:
        keys_txt = ", ".join(present_retired_keys)
        raise SystemExit(
            "[FATAL][RETIRED_TARGET_KEY_PRESENT] active posttrain config must not contain retired target keys: "
            f"{keys_txt}. "
            "Use exactly one newflow target: train_direct_pose=true or train_lambda_head=true."
        )


def _cfg_reject_retired_shell_keys(payload: Dict[str, Any]) -> None:
    present_retired_shell_keys = [
        str(k)
        for k in payload.keys()
        if (
            str(k) in _RETIRED_POSTTRAIN_SHELL_EXACT_KEYS
            or any(str(k).startswith(prefix) for prefix in _RETIRED_POSTTRAIN_SHELL_KEY_PREFIXES)
        )
    ]
    if present_retired_shell_keys:
        keys_txt = ", ".join(sorted(present_retired_shell_keys))
        raise SystemExit(
            "[FATAL][RETIRED_SHELL_KEY_PRESENT] posttrain mainline config must not contain retired shell keys: "
            f"{keys_txt}. "
            "Remove hinge/contact_td_hazard/contact_ttc shells and keep only current newflow keys."
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
    parsed_scalars = _cfg_from_schema(
        payload,
        [
            ("event_clock_hidden_dim", _cfg_get_int, {"key": "event_clock_hidden_dim", "default": None, "allow_none": True}),
            ("event_clock_gate_hidden_dim", _cfg_get_int, {"key": "event_clock_gate_hidden_dim", "default": None, "allow_none": True}),
            ("gate_sup_weight", _cfg_get_float, {"key": "lambda_gate_sup_weight", "default": 0.0}),
            ("gate_sup_tau_deg", _cfg_get_float, {"key": "lambda_gate_sup_tau_deg", "default": 2.5}),
            ("gate_sup_margin_deg", _cfg_get_float, {"key": "lambda_gate_sup_margin_deg", "default": 1.0}),
            ("gate_sup_start_step", _cfg_get_int, {"key": "lambda_gate_sup_start_step", "default": -1}),
            ("direct_pose_time_pe_dim", _cfg_get_int, {"key": "direct_pose_time_pe_dim", "default": -1}),
            ("direct_pose_hidden_override", _cfg_get_int, {"key": "direct_pose_hidden_override", "default": None, "allow_none": True}),
            ("direct_pose_nonleg_proj_dim", _cfg_get_int, {"key": "direct_pose_nonleg_proj_dim", "default": 0, "min_value": 0}),
        ],
    )
    cfg: Dict[str, Any] = {
        "event_clock_hidden_dim": parsed_scalars["event_clock_hidden_dim"],
        "event_clock_gate_hidden_dim": parsed_scalars["event_clock_gate_hidden_dim"],
        "lambda_gate_sup_weight": float(parsed_scalars["gate_sup_weight"]),
        "lambda_gate_sup_tau_deg": float(parsed_scalars["gate_sup_tau_deg"]),
        "lambda_gate_sup_margin_deg": float(parsed_scalars["gate_sup_margin_deg"]),
        "lambda_gate_sup_start_step": int(parsed_scalars["gate_sup_start_step"]),
        "direct_pose_time_pe_dim": int(parsed_scalars["direct_pose_time_pe_dim"]),
        "direct_pose_hidden_override": parsed_scalars["direct_pose_hidden_override"],
        "direct_pose_nonleg_proj_dim": int(parsed_scalars["direct_pose_nonleg_proj_dim"]),
        "direct_pose_split_enable": _cfg_get_bool(payload, "direct_pose_split_enable", False),
        "direct_pose_arm_split_enable": _cfg_get_bool(payload, "direct_pose_arm_split_enable", False),
        "direct_pose_arm_bones": _normalize_optional_csv(payload.get("direct_pose_arm_bones", None)),
        "direct_pose_nonleg_train_only": _cfg_get_bool(payload, "direct_pose_nonleg_train_only", False),
        "direct_pose_leg_enable": _cfg_get_bool(payload, "direct_pose_leg_enable", False),
        "direct_pose_leg_bones": payload.get("direct_pose_leg_bones", None),
        "direct_pose_leg_train_only": _cfg_get_bool(payload, "direct_pose_leg_train_only", False),
        "direct_pose_leg_gate_train_only": _cfg_get_bool(payload, "direct_pose_leg_gate_train_only", False),
        "direct_pose_leg_mode": str(payload.get("direct_pose_leg_mode") or "rot6d_add"),
        "direct_pose_leg_stopgrad_main": _cfg_get_bool(payload, "direct_pose_leg_stopgrad_main", False),
        "direct_pose_leg_detach_feat": _cfg_get_bool(payload, "direct_pose_leg_detach_feat", False),
        "direct_pose_leg_max_deg": float(_cfg_get_float(payload, "direct_pose_leg_max_deg", 0.0, min_value=0.0)),
        "direct_pose_leg_gate_mode": _cfg_get_enum(
            payload,
            "direct_pose_leg_gate_mode",
            "none",
            alias_map=_DIRECT_POSE_LEG_GATE_ALIAS_MAP,
            choices=_DIRECT_POSE_LEG_GATE_CHOICES,
        ),
        "direct_pose_leg_gate_power": float(_cfg_get_float(payload, "direct_pose_leg_gate_power", 1.0, min_value=1e-8)),
        "direct_pose_leg_gate_sup_weight": float(
            _cfg_get_float(
                payload,
                "direct_pose_leg_gate_sup_weight",
                0.0,
                aliases=("direct_pose_leg_gate_loss_weight",),
                min_value=0.0,
            )
        ),
    }

    direct_pose_meas_mode_override = payload.get("direct_pose_meas_mode_override", None)
    if direct_pose_meas_mode_override is None:
        direct_pose_meas_mode_override = payload.get("direct_pose_meas_mode", None)
    if direct_pose_meas_mode_override is not None:
        s = str(direct_pose_meas_mode_override).strip()
        direct_pose_meas_mode_override = s if s else None
    cfg["direct_pose_meas_mode_override"] = direct_pose_meas_mode_override

    # Only used when direct_pose_leg_gate_mode='scale' (exp(log_mag)).
    direct_pose_leg_scale_log_clip = float(_cfg_get_float(payload, "direct_pose_leg_scale_log_clip", 4.0, min_value=1e-8))
    direct_pose_leg_scale_clamp_k = float(_cfg_get_float(payload, "direct_pose_leg_scale_clamp_k", 0.0))
    if direct_pose_leg_scale_clamp_k <= 1.0:
        direct_pose_leg_scale_clamp_k = 0.0
    cfg["direct_pose_leg_scale_log_clip"] = direct_pose_leg_scale_log_clip
    cfg["direct_pose_leg_scale_clamp_k"] = direct_pose_leg_scale_clamp_k

    # Optional: direction alignment loss for leg SO(3) residual omega.
    leg_align_cfg = _cfg_from_schema(
        payload,
        [
            ("direct_pose_leg_align_weight", _cfg_get_float, {"key": "direct_pose_leg_align_weight", "default": 0.0, "min_value": 0.0}),
            (
                "direct_pose_leg_align_oracle_min_deg",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_oracle_min_deg", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_oracle_weight_deg",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_oracle_weight_deg", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_mode",
                _cfg_get_enum,
                {
                    "key": "direct_pose_leg_align_mode",
                    "default": "cos",
                    "alias_map": {"": "cos", "none": "cos", "off": "cos", "disable": "cos", "disabled": "cos"},
                    "choices": ("cos", "proj"),
                },
            ),
            (
                "direct_pose_leg_align_mag_weight",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_mag_weight", "default": 1.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_res_weight",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_res_weight", "default": 1.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_sign_weight",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_sign_weight", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_cos_thresh",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_cos_thresh", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_target_joints",
                _cfg_get_or,
                {"key": "direct_pose_leg_align_target_joints", "default": None},
            ),
            (
                "direct_pose_leg_align_anchor_joints",
                _cfg_get_or,
                {"key": "direct_pose_leg_align_anchor_joints", "default": None},
            ),
            (
                "direct_pose_leg_align_anchor_weight",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_anchor_weight", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_schedule",
                _cfg_get_enum,
                {
                    "key": "direct_pose_leg_align_schedule",
                    "default": "none",
                    "alias_map": {
                        "": "none",
                        "off": "none",
                        "false": "none",
                        "0": "none",
                    },
                    "choices": ("none", "linear"),
                },
            ),
            (
                "direct_pose_leg_align_start_weight",
                _cfg_get_float,
                {"key": "direct_pose_leg_align_start_weight", "default": 0.0, "min_value": 0.0},
            ),
            (
                "direct_pose_leg_align_warmup_steps",
                _cfg_get_int,
                {"key": "direct_pose_leg_align_warmup_steps", "default": 0, "min_value": 0},
            ),
            (
                "direct_pose_leg_align_ramp_steps",
                _cfg_get_int,
                {"key": "direct_pose_leg_align_ramp_steps", "default": 0, "min_value": 0},
            ),
        ],
    )
    direct_pose_leg_align_target_joints = leg_align_cfg["direct_pose_leg_align_target_joints"]
    if direct_pose_leg_align_target_joints is not None:
        direct_pose_leg_align_target_joints = str(direct_pose_leg_align_target_joints).strip()
        if str(direct_pose_leg_align_target_joints).lower() in ("", "none", "null", "off", "disabled"):
            direct_pose_leg_align_target_joints = None
    direct_pose_leg_align_anchor_joints = leg_align_cfg["direct_pose_leg_align_anchor_joints"]
    if direct_pose_leg_align_anchor_joints is not None:
        direct_pose_leg_align_anchor_joints = str(direct_pose_leg_align_anchor_joints).strip()
        if str(direct_pose_leg_align_anchor_joints).lower() in ("", "none", "null", "off", "disabled"):
            direct_pose_leg_align_anchor_joints = None
    cfg.update(
        direct_pose_leg_align_weight=float(leg_align_cfg["direct_pose_leg_align_weight"]),
        direct_pose_leg_align_oracle_min_deg=float(leg_align_cfg["direct_pose_leg_align_oracle_min_deg"]),
        direct_pose_leg_align_oracle_weight_deg=float(leg_align_cfg["direct_pose_leg_align_oracle_weight_deg"]),
        direct_pose_leg_align_mode=str(leg_align_cfg["direct_pose_leg_align_mode"]),
        direct_pose_leg_align_mag_weight=float(leg_align_cfg["direct_pose_leg_align_mag_weight"]),
        direct_pose_leg_align_res_weight=float(leg_align_cfg["direct_pose_leg_align_res_weight"]),
        direct_pose_leg_align_sign_weight=float(leg_align_cfg["direct_pose_leg_align_sign_weight"]),
        direct_pose_leg_align_cos_thresh=float(leg_align_cfg["direct_pose_leg_align_cos_thresh"]),
        direct_pose_leg_align_target_joints=direct_pose_leg_align_target_joints,
        direct_pose_leg_align_anchor_joints=direct_pose_leg_align_anchor_joints,
        direct_pose_leg_align_anchor_weight=float(leg_align_cfg["direct_pose_leg_align_anchor_weight"]),
        direct_pose_leg_align_schedule=str(leg_align_cfg["direct_pose_leg_align_schedule"]),
        direct_pose_leg_align_start_weight=float(leg_align_cfg["direct_pose_leg_align_start_weight"]),
        direct_pose_leg_align_warmup_steps=int(leg_align_cfg["direct_pose_leg_align_warmup_steps"]),
        direct_pose_leg_align_ramp_steps=int(leg_align_cfg["direct_pose_leg_align_ramp_steps"]),
    )

    cfg.update(
        direct_pose_leg_side_routing=False,
        direct_pose_leg_contact_order="lr",
        direct_pose_leg_side_embed_dim=0,
        direct_pose_leg_side_plan_other=False,
        direct_pose_leg_side_phase_other=False,
        direct_pose_leg_side_phase_rel=False,
        direct_pose_leg_side_cue="none",
        direct_pose_leg_side_cue_tau=30.0,
        direct_pose_leg_side_sign_gate=False,
        direct_pose_leg_side_rank1=False,
        direct_pose_leg_side_sign_gate_reg_weight=0.0,
    )

    cfg["direct_pose_loss_leg_split"] = _cfg_get_bool(payload, "direct_pose_loss_leg_split", False)
    cfg["direct_pose_nonleg_focus_bones"] = _normalize_optional_csv(payload.get("direct_pose_nonleg_focus_bones", None))

    direct_pose_loss_cfg = _cfg_from_schema(
        payload,
        [
            ("direct_pose_loss_group_norm_enable", _cfg_get_bool, {"key": "direct_pose_loss_group_norm_enable", "default": False}),
            ("direct_pose_loss_group_norm_w_leg", _cfg_get_float, {"key": "direct_pose_loss_group_norm_w_leg", "default": 1.0, "require_finite": False}),
            (
                "direct_pose_loss_group_norm_w_nonleg",
                _cfg_get_float,
                {"key": "direct_pose_loss_group_norm_w_nonleg", "default": 1.0, "require_finite": False},
            ),
            ("direct_pose_loss_group_norm_ema_beta", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ema_beta", "default": 0.95}),
            ("direct_pose_loss_group_norm_ratio_min", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ratio_min", "default": 0.2}),
            ("direct_pose_loss_group_norm_ratio_max", _cfg_get_float, {"key": "direct_pose_loss_group_norm_ratio_max", "default": 5.0}),
            ("direct_pose_loss_group_norm_eps", _cfg_get_float, {"key": "direct_pose_loss_group_norm_eps", "default": 1e-6}),
            ("direct_pose_nonleg_focus_weight", _cfg_get_float, {"key": "direct_pose_nonleg_focus_weight", "default": 1.0}),
            ("direct_pose_grad_monitor_enable", _cfg_get_bool, {"key": "direct_pose_grad_monitor_enable", "default": False}),
            ("direct_pose_grad_ratio_gate", _cfg_get_float, {"key": "direct_pose_grad_ratio_gate", "default": 0.35}),
            (
                "direct_pose_leg_align_grad_probe_enable",
                _cfg_get_bool,
                {"key": "direct_pose_leg_align_grad_probe_enable", "default": False},
            ),
            (
                "direct_pose_leg_align_grad_probe_steps",
                _cfg_get_int,
                {"key": "direct_pose_leg_align_grad_probe_steps", "default": 0, "min_value": 0},
            ),
        ],
    )
    direct_pose_loss_group_norm_w_leg = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_w_leg"])
    direct_pose_loss_group_norm_w_nonleg = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_w_nonleg"])
    direct_pose_loss_group_norm_ema_beta = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ema_beta"])
    if (not math.isfinite(direct_pose_loss_group_norm_ema_beta)) or direct_pose_loss_group_norm_ema_beta < 0.0:
        direct_pose_loss_group_norm_ema_beta = 0.95
    direct_pose_loss_group_norm_ema_beta = max(0.0, min(0.9999, float(direct_pose_loss_group_norm_ema_beta)))
    direct_pose_loss_group_norm_ratio_min = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_min"])
    direct_pose_loss_group_norm_ratio_max = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_ratio_max"])
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_min)) or direct_pose_loss_group_norm_ratio_min <= 0.0:
        direct_pose_loss_group_norm_ratio_min = 0.2
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_max)) or direct_pose_loss_group_norm_ratio_max <= 0.0:
        direct_pose_loss_group_norm_ratio_max = 5.0
    if direct_pose_loss_group_norm_ratio_min > direct_pose_loss_group_norm_ratio_max:
        direct_pose_loss_group_norm_ratio_min, direct_pose_loss_group_norm_ratio_max = (
            direct_pose_loss_group_norm_ratio_max,
            direct_pose_loss_group_norm_ratio_min,
        )
    direct_pose_loss_group_norm_eps = float(direct_pose_loss_cfg["direct_pose_loss_group_norm_eps"])
    if (not math.isfinite(direct_pose_loss_group_norm_eps)) or direct_pose_loss_group_norm_eps <= 0.0:
        direct_pose_loss_group_norm_eps = 1e-6
    direct_pose_nonleg_focus_weight = float(direct_pose_loss_cfg["direct_pose_nonleg_focus_weight"])
    if (not math.isfinite(direct_pose_nonleg_focus_weight)) or direct_pose_nonleg_focus_weight <= 0.0:
        direct_pose_nonleg_focus_weight = 1.0
    direct_pose_grad_ratio_gate = float(direct_pose_loss_cfg["direct_pose_grad_ratio_gate"])
    if (not math.isfinite(direct_pose_grad_ratio_gate)) or direct_pose_grad_ratio_gate <= 0.0:
        direct_pose_grad_ratio_gate = 0.35
    direct_pose_leg_align_grad_probe_steps = int(direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_steps"])
    if direct_pose_leg_align_grad_probe_steps < 0:
        direct_pose_leg_align_grad_probe_steps = 0
    if bool(direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_enable"]) and direct_pose_leg_align_grad_probe_steps <= 0:
        direct_pose_leg_align_grad_probe_steps = 30
    cfg.update(
        direct_pose_loss_sics=None,
        direct_pose_loss_cycle_gte=0,
        direct_pose_loss_sic_mode="mask",
        direct_pose_loss_sic_boost=1.0,
        direct_pose_loss_group_norm_enable=bool(direct_pose_loss_cfg["direct_pose_loss_group_norm_enable"]),
        direct_pose_loss_group_norm_w_leg=float(direct_pose_loss_group_norm_w_leg),
        direct_pose_loss_group_norm_w_nonleg=float(direct_pose_loss_group_norm_w_nonleg),
        direct_pose_loss_group_norm_ema_beta=float(direct_pose_loss_group_norm_ema_beta),
        direct_pose_loss_group_norm_ratio_min=float(direct_pose_loss_group_norm_ratio_min),
        direct_pose_loss_group_norm_ratio_max=float(direct_pose_loss_group_norm_ratio_max),
        direct_pose_loss_group_norm_eps=float(direct_pose_loss_group_norm_eps),
        direct_pose_nonleg_focus_weight=float(direct_pose_nonleg_focus_weight),
        direct_pose_grad_monitor_enable=bool(direct_pose_loss_cfg["direct_pose_grad_monitor_enable"]),
        direct_pose_grad_ratio_gate=float(direct_pose_grad_ratio_gate),
        direct_pose_leg_align_grad_probe_enable=bool(direct_pose_loss_cfg["direct_pose_leg_align_grad_probe_enable"]),
        direct_pose_leg_align_grad_probe_steps=int(direct_pose_leg_align_grad_probe_steps),
    )
    return cfg


def _cfg_parse_lambda_rollout(payload: Dict[str, Any]) -> Dict[str, Any]:
    core_cfg = _cfg_from_schema(
        payload,
        [
            ("device", _cfg_get_str_or, {"key": "device", "default": "auto"}),
            ("batch", _cfg_get_int_or, {"key": "batch", "default": 8}),
            ("seq_len", _cfg_get_int_or, {"key": "seq_len", "default": 180}),
            ("dataset_index_mode", _cfg_get_str_or, {"key": "dataset_index_mode", "default": "sliding", "aliases": ("index_mode",)}),
            ("rollout_steps", _cfg_get_int_or, {"key": "rollout_steps", "default": 0}),
            ("rollout_cycles", _cfg_get_int_or, {"key": "rollout_cycles", "default": 1, "min_value": 1}),
            ("rollout_include_boundary_raw", _cfg_pick, {"key": "rollout_include_boundary"}),
            ("rollout_random_offset", _cfg_get_bool, {"key": "rollout_random_offset", "default": False}),
            ("time_index_mode", _cfg_get_str_or, {"key": "time_index_mode", "default": "global"}),
            ("phase_reset_source", _cfg_get_str_or, {"key": "phase_reset_source", "default": "none"}),
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
            (
                "posttrain_contacts_source",
                _cfg_get_str_or,
                {
                    "key": "posttrain_contacts_source",
                    "default": "pretrain_contact",
                },
            ),
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
    core_cfg["phase_reset_source"] = _canon_phase_reset_source(core_cfg["phase_reset_source"])
    core_cfg["lambda_reliability_warmup_joint_scales"] = _as_float_list(
        core_cfg.pop("lambda_reliability_warmup_joint_scales_raw")
    )
    src = str(core_cfg.get("posttrain_contacts_source", "pretrain_contact") or "pretrain_contact").strip().lower()
    if src != "pretrain_contact":
        raise SystemExit(
            f"[FATAL] unsupported posttrain_contacts_source={src!r}; only 'pretrain_contact' is allowed."
        )
    core_cfg["posttrain_contacts_source"] = "pretrain_contact"
    clamp_v = core_cfg.get("posttrain_contacts_pretrain_clamp", 1.0)
    try:
        clamp_f = float(clamp_v)
    except Exception:
        clamp_f = 1.0
    if not math.isfinite(clamp_f):
        clamp_f = 1.0
    core_cfg["posttrain_contacts_pretrain_clamp"] = max(0.0, float(clamp_f))
    affine_spec = core_cfg.get("posttrain_contacts_pretrain_affine_stats", None)
    if affine_spec is None:
        core_cfg["posttrain_contacts_pretrain_affine_stats"] = None
    else:
        s = str(affine_spec).strip()
        core_cfg["posttrain_contacts_pretrain_affine_stats"] = s if s else None
    return core_cfg


def _cfg_from_payload(payload: Dict[str, Any]) -> PostTrainConfig:
    if not isinstance(payload, dict):
        raise TypeError("posttrain config payload must be a dict")
    _cfg_reject_retired_targets(payload)
    _cfg_reject_retired_shell_keys(payload)
    _cfg_reject_retired_direct_pose_highorder(payload)
    cfg_kwargs: Dict[str, Any] = {}
    cfg_kwargs.update(_cfg_parse_path_basic(payload))
    cfg_kwargs.update(_cfg_parse_direct_pose(payload))
    cfg_kwargs.update(_cfg_parse_lambda_rollout(payload))
    return PostTrainConfig(**cfg_kwargs)


def _resolve_device(pref: str) -> torch.device:
    pref = str(pref or "auto").lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if has_mps else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _merge_norm_spec(bundle_path: Path, pretrain_path: Optional[Path]) -> Dict[str, Any]:
    try:
        with bundle_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as err:
        raise RuntimeError(f"[FATAL] failed to read bundle_json {bundle_path}: {err}") from err
    require_standard_rotvec_spec(spec, context=f"bundle_json {bundle_path}")
    if pretrain_path is not None and pretrain_path.is_file():
        with pretrain_path.open("r", encoding="utf-8") as f:
            extra = json.load(f)
        require_standard_rotvec_spec(extra, context=f"pretrain_template {pretrain_path}")
        if isinstance(extra, dict):
            spec = dict(extra, **spec)
    return spec


def _init_y_from_x(normalizer: DataNormalizer, x_raw: torch.Tensor, dy: int) -> torch.Tensor:
    y = x_raw.new_zeros((x_raw.shape[0], dy))
    mapping = getattr(normalizer, "y_to_x_map", None) or []
    if mapping:
        for item in mapping:
            xs, xk = int(item["x_start"]), int(item["x_size"])
            ys, yk = int(item["y_start"]), int(item["y_size"])
            if xk <= 0 or yk <= 0:
                continue
            y[..., ys : ys + yk] = x_raw[..., xs : xs + xk]
        return y
    # Fallback: assume the leading dims match.
    take = min(int(dy), int(x_raw.shape[-1]))
    y[..., :take] = x_raw[..., :take]
    return y


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


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


def _rollout_cond_raw_idx(*, idx: int, cond_raw_tgt: torch.Tensor, include_boundary: bool, cycle_len: int) -> int:
    if include_boundary:
        return int((int(idx) + 1) % max(1, int(cycle_len)))
    return min(int(cond_raw_tgt.shape[1]) - 1, int(idx) + 1)


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
    cond_t = cond_seq[:, idx] if (torch.is_tensor(cond_seq) and cond_seq.dim() == 3) else cond_seq
    cond_raw_step = None
    if torch.is_tensor(cond_raw_tgt):
        if cond_raw_tgt.dim() == 3:
            idx_raw = _rollout_cond_raw_idx(
                idx=int(idx),
                cond_raw_tgt=cond_raw_tgt,
                include_boundary=bool(include_boundary),
                cycle_len=int(cycle_len),
            )
            cond_raw_step = cond_raw_tgt[:, idx_raw]
        else:
            cond_raw_step = cond_raw_tgt

    cond_raw_for_model = cond_raw_step
    if enable_reprojection and int(t) > 0 and torch.is_tensor(cond_raw_step):
        yaw_gt = None
        if callable(yaw_gt_fn):
            try:
                yaw_gt = yaw_gt_fn(int(idx))
            except Exception:
                yaw_gt = None
        yaw_pred = None
        try:
            yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
        except Exception:
            yaw_pred = None
        if yaw_gt is not None and yaw_pred is not None:
            cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
            if cond_proj is not None:
                cond_raw_for_model = cond_proj

    if cond_raw_for_model is not None:
        try:
            cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
        except Exception:
            cond_override = None
        if cond_override is not None:
            cond_t = cond_override
    return cond_t, cond_raw_step


def _predict_pretrain_contacts_from_frozen(
    trainer: Trainer,
    model: EventMotionModel,
    *,
    motion_step_t: Optional[torch.Tensor],
    pose_hist_step_t: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    _ = model
    fn = getattr(trainer, '_predict_pretrain_contacts_from_frozen', None)
    if not callable(fn):
        return None
    try:
        return fn(motion_step_t=motion_step_t, pose_hist_step_t=pose_hist_step_t)
    except Exception:
        return None


def _prepare_rollout_contacts_input(
    trainer: Trainer,
    model: EventMotionModel,
    *,
    motion_t: torch.Tensor,
    motion_raw: torch.Tensor,
    pose_hist_t: Optional[torch.Tensor],
    plan_z: Optional[torch.Tensor],
    t: int,
    prev_foot_pos_meas: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    plan_enable = bool(getattr(model, "contact_plan_enable", False))
    contacts_source = str(getattr(trainer, "posttrain_contacts_source", "pretrain_contact") or "pretrain_contact").strip().lower()
    if contacts_source != "pretrain_contact":
        raise RuntimeError(
            f"[FATAL] unsupported posttrain_contacts_source={contacts_source!r}; only 'pretrain_contact' is allowed."
        )
    _ = motion_raw
    _ = plan_z
    _ = t

    contacts_in_t = None
    if plan_enable:
        contacts_in_t = _predict_pretrain_contacts_from_frozen(
            trainer,
            model,
            motion_step_t=motion_t,
            pose_hist_step_t=pose_hist_t,
        )
        if contacts_in_t is None:
            raise RuntimeError(
                "[FATAL] posttrain_contacts_source=pretrain_contact requires valid frozen encoder+contact_head "
                "and runtime-compatible encoder input dimensions."
            )
    return contacts_in_t, prev_foot_pos_meas


def _resolve_rollout_time_index(
    *,
    t: int,
    idx: int,
    time_index_mode: str,
    time_base: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if time_index_mode == "none":
        return None
    if time_index_mode == "cycle":
        time_index_t: Any = int(idx)
        if time_base is not None:
            try:
                time_index_t = time_base + int(idx)
            except Exception:
                pass
        return time_index_t

    time_index_t = int(idx)
    if time_base is not None:
        try:
            time_index_t = time_base + int(idx)
        except Exception:
            time_index_t = int(idx)
    else:
        time_index_t = int(t)
    return time_index_t


def _resolve_rollout_step_tensor(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    t: int,
    total_steps: int,
) -> Optional[torch.Tensor]:
    rollout_step_t = None
    try:
        if int(total_steps) > 1:
            step_norm = float(t) / float(int(total_steps) - 1)
        else:
            step_norm = 0.0
        rollout_step_t = torch.full((batch_size, 1, 1), step_norm, device=device, dtype=dtype)
    except Exception:
        rollout_step_t = None
    return rollout_step_t


def _update_rollout_recurrent_state(
    model: EventMotionModel,
    ret: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    if bool(getattr(model, "contact_plan_enable", False)):
        try:
            z_next = ret.get("plan_z_next", None)
            if torch.is_tensor(z_next):
                state["plan_z"] = z_next.detach()
            p_next = ret.get("phase_z_next", None)
            if torch.is_tensor(p_next):
                state["phase_z"] = p_next.detach()
            a_next = ret.get("phase_event_age_next", None)
            if torch.is_tensor(a_next):
                state["phase_event_age"] = a_next.detach()
        except Exception:
            pass
    try:
        mlog = ret.get("contacts_meas_logits", None)
        if torch.is_tensor(mlog):
            if mlog.dim() == 3:
                state["meas_logits_prev"] = mlog[:, -1].detach()
            elif mlog.dim() == 2:
                state["meas_logits_prev"] = mlog.detach()
    except Exception:
        pass


def _apply_rollout_carry_state(
    trainer: Trainer,
    state: Dict[str, Any],
    *,
    y_next_raw: torch.Tensor,
    cond_raw_step: Optional[torch.Tensor],
) -> None:
    cond_env = cond_raw_step if torch.is_tensor(cond_raw_step) else None
    motion_raw = trainer._apply_free_carry(state["motion_raw"], y_next_raw, cond_next_raw=cond_env)
    motion_raw = _finite(motion_raw)
    motion = trainer._diag_norm_x(motion_raw)
    state["motion_raw"] = motion_raw
    state["motion"] = motion

    pose_hist_state = state.get("pose_hist_state", None)
    if not isinstance(pose_hist_state, PoseHistState):
        pose_hist_state = PoseHistState(enabled=False, length=0, dim=0, stride=0)
    state["pose_hist_state"] = advance_pose_hist_state(
        pose_hist_state,
        y_next_raw=y_next_raw,
        rot_slice=state.get("rot_slice", None),
    )

    state["y_prev_raw"] = y_next_raw


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
    task_callback: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    motion = state["motion"]
    motion_raw = state["motion_raw"]
    y_prev_raw = state["y_prev_raw"]
    device = motion.device
    dtype = motion.dtype
    B = int(motion.shape[0])

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

    if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
        angvel_t = motion[..., trainer.angvel_x_slice].detach()
    else:
        angvel_t = angvel_seq[:, idx] if (torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3) else angvel_seq

    pose_hist_state = state.get("pose_hist_state", None)
    if not isinstance(pose_hist_state, PoseHistState):
        pose_hist_state = PoseHistState(enabled=False, length=0, dim=0, stride=0)
    pose_hist_t = resolve_pose_hist_input(
        state=pose_hist_state,
        pose_hist_seq=pose_hist_seq,
        idx=int(idx),
    )

    inp_motion = motion.unsqueeze(1)
    inp_cond = cond_t.unsqueeze(1) if torch.is_tensor(cond_t) and cond_t.dim() == 2 else cond_t
    inp_angvel = angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) and angvel_t.dim() == 2 else angvel_t
    inp_pose_hist = pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) and pose_hist_t.dim() == 2 else pose_hist_t

    contacts_in_t, prev_foot_pos_meas = _prepare_rollout_contacts_input(
        trainer,
        model,
        motion_t=motion,
        motion_raw=motion_raw,
        pose_hist_t=pose_hist_t,
        plan_z=state.get("plan_z", None),
        t=int(t),
        prev_foot_pos_meas=state.get("prev_foot_pos_meas", None),
    )
    state["prev_foot_pos_meas"] = prev_foot_pos_meas

    time_index_t = _resolve_rollout_time_index(
        t=int(t),
        idx=int(idx),
        time_index_mode=str(time_index_mode),
        time_base=time_base,
    )
    rollout_step_t = _resolve_rollout_step_tensor(
        batch_size=B,
        device=device,
        dtype=dtype,
        t=int(t),
        total_steps=int(total_steps),
    )

    ret = model(
        inp_motion,
        inp_cond,
        contacts=contacts_in_t,
        angvel=inp_angvel,
        pose_history=inp_pose_hist,
        plan_z=state.get("plan_z", None),
        phase_z=state.get("phase_z", None),
        phase_event_age=state.get("phase_event_age", None),
        meas_logits_prev=state.get("meas_logits_prev", None),
        time_index=time_index_t,
        rollout_step=rollout_step_t,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict.")
    _update_rollout_recurrent_state(model, ret, state)

    step_ctx: Dict[str, Any] = {
        "ret": ret,
        "t": int(t),
        "idx": int(idx),
        "contacts_in_t": contacts_in_t,
        "cond_raw_step": cond_raw_step,
        "time_index_t": time_index_t,
        "rollout_step_t": rollout_step_t,
        "state": state,
    }
    task_out = task_callback(step_ctx) if callable(task_callback) else None
    if not isinstance(task_out, dict):
        task_out = {}
    y_carry_raw = task_out.get("y_carry_raw", None)
    if torch.is_tensor(y_carry_raw):
        if detach_rollout_state:
            y_carry_raw = y_carry_raw.detach()
            task_out["y_carry_raw"] = y_carry_raw
        if int(t) < int(total_steps) - 1:
            _apply_rollout_carry_state(
                trainer,
                state,
                y_next_raw=y_carry_raw,
                cond_raw_step=cond_raw_step,
            )
    task_out.setdefault("ret", ret)
    task_out.setdefault("contacts_in_t", contacts_in_t)
    task_out.setdefault("cond_raw_step", cond_raw_step)
    task_out.setdefault("time_index_t", time_index_t)
    task_out.setdefault("rollout_step_t", rollout_step_t)
    return task_out


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


def _lambda_rollout_prepare_context(
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
    time_weight_mode: str,
    time_weight_max: float,
) -> Dict[str, Any]:
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
            y0_raw = _init_y_from_x(trainer.normalizer, motion0_raw, Dy)
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
    y_prev_raw = _init_y_from_x(trainer.normalizer, motion_raw, Dy)
    pose_hist_state = init_pose_hist_state(
        ref_tensor=motion_seq,
        pose_hist_seq=pose_hist_seq,
        y_prev_raw=y_prev_raw,
        rot_slice=rot_slice,
        pose_hist_len=int(getattr(trainer, "pose_hist_len", 0) or 0),
        pose_hist_dim=int(getattr(trainer, "pose_hist_dim", 0) or 0),
        params_fn=trainer._pose_hist_params,
        offset=int(offset),
    )
    state: Dict[str, Any] = {
        "motion": motion,
        "motion_raw": motion_raw,
        "y_prev_raw": y_prev_raw,
        "plan_z": None,
        "phase_z": None,
        "phase_event_age": None,
        "meas_logits_prev": None,
        "prev_foot_pos_meas": None,
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

    return {
        "device": device,
        "dtype": dtype,
        "motion_seq": motion_seq,
        "gt_seq": gt_seq,
        "cond_seq": cond_seq,
        "cond_raw_tgt": cond_raw_tgt,
        "cond_norm_mu": cond_norm_mu,
        "cond_norm_std": cond_norm_std,
        "contacts_seq": contacts_seq,
        "angvel_seq": angvel_seq,
        "pose_hist_seq": pose_hist_seq,
        "B": int(B),
        "T": int(T),
        "Dy": int(Dy),
        "steps": int(steps),
        "rollout_cycles": int(rollout_cycles),
        "include_boundary": bool(include_boundary),
        "cycle_len": int(cycle_len),
        "total_steps": int(total_steps),
        "offset": int(offset),
        "y0_raw": y0_raw,
        "rot_slice": rot_slice,
        "rot_len": int(rot_len),
        "J": int(J),
        "std_y": std_y,
        "state": state,
        "step_weights": step_weights,
        "boundary_steps": int(boundary_steps),
        "boundary_weighted_sum": float(boundary_weighted_sum),
    }


def _lambda_rollout_resolve_nonleg_focus(
    trainer: Trainer,
    *,
    objective: str,
    direct_pose_nonleg_focus_bones: str,
    direct_pose_nonleg_focus_weight: float,
    J: int,
    device: torch.device,
) -> Dict[str, Any]:
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

    return {
        "direct_nonleg_focus_mask_j": direct_nonleg_focus_mask_j,
        "direct_nonleg_focus_requested": int(direct_nonleg_focus_requested),
        "direct_nonleg_focus_resolved": int(direct_nonleg_focus_resolved),
        "direct_nonleg_focus_weight_use": float(direct_nonleg_focus_weight_use),
        "direct_nonleg_focus_applied": float(direct_nonleg_focus_applied),
    }


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
) -> Dict[str, Any]:
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

    return {
        "gate_sup_weight": float(gate_sup_weight),
        "gate_sup_start": int(gate_sup_start),
        "tau_rad": float(tau_rad),
        "margin_rad": float(margin_rad),
        "direct_group_norm_enable": bool(direct_group_norm_enable),
        "direct_group_w_leg": float(direct_group_w_leg),
        "direct_group_w_nonleg": float(direct_group_w_nonleg),
        "direct_group_beta": float(direct_group_beta),
        "direct_group_ratio_min": float(direct_group_ratio_min),
        "direct_group_ratio_max": float(direct_group_ratio_max),
        "direct_group_eps": float(direct_group_eps),
    }


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


def _append_leg_align_group_term(
    *,
    per: torch.Tensor,
    w: torch.Tensor,
    joint_names: list[str],
    target_joints: Iterable[str],
    step_weight: torch.Tensor,
    dtype: torch.dtype,
    loss_terms: list[torch.Tensor],
    frac_terms: list[torch.Tensor],
) -> None:
    loss, frac = _compute_leg_align_subset_term(
        per=per,
        w=w,
        joint_names=joint_names,
        target_joints=target_joints,
        dtype=dtype,
    )
    if torch.is_tensor(loss):
        loss_terms.append(loss * step_weight)
    if torch.is_tensor(frac):
        frac_terms.append(frac * step_weight)


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
        if torch.is_tensor(omega_leg):
            if omega_leg.dim() == 4 and omega_leg.size(1) == 1:
                omega_leg = omega_leg[:, 0]
            leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
            if torch.is_tensor(leg_idx) and int(leg_idx.numel()) > 0:
                idx_use = leg_idx.to(device=device)
                # Ensure omega has shape (B,K,3) matching idx_use.
                if omega_leg.dim() == 3 and omega_leg.shape[0] == B and omega_leg.shape[-1] == 3:
                    K = int(idx_use.numel())
                    if int(omega_leg.shape[1]) == K:
                        keep_mask = None
                        # Exclude root if present (keep omega aligned).
                        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
                        if 0 <= root_idx < J and bool((idx_use == int(root_idx)).any().detach().cpu().item()):
                            keep_mask = (idx_use != int(root_idx))
                            if bool(keep_mask.any().detach().cpu().item()):
                                idx_use = idx_use[keep_mask]
                                omega_leg = omega_leg[:, keep_mask, :]
                        if int(idx_use.numel()) > 0:
                            base6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J, 6)
                            R_base = rot6d_to_matrix(base6, columns=columns)  # (B,J,3,3)
                            R_leg_base = R_base[:, idx_use, :, :]
                            if bool(getattr(model, "direct_pose_leg_stopgrad_main", False)):
                                R_leg_base = R_leg_base.detach()

                            omega_oracle = None
                            oracle_norm = None

                            # Optional: direction alignment supervision for leg omega.
                            # Match run_freerun_cycles: omega_oracle = so3_log_map(R_gt @ R_base^T).
                            align_w = float(direct_pose_leg_align_weight or 0.0)
                            if align_w > 0.0:
                                try:
                                    with torch.no_grad():
                                        R_gt_leg = R_gt[:, idx_use, :, :]
                                        R_delta_oracle = torch.matmul(R_gt_leg, R_leg_base.transpose(-1, -2))
                                        omega_oracle = so3_log_map(R_delta_oracle)
                                        oracle_norm = omega_oracle.norm(dim=-1)  # (B,K)

                                        min_deg = float(direct_pose_leg_align_oracle_min_deg or 0.0)
                                        min_rad = float(min_deg) * (math.pi / 180.0)
                                        w = (oracle_norm > float(min_rad)).to(dtype=dtype)

                                        w_deg = float(direct_pose_leg_align_oracle_weight_deg or 0.0)
                                        if w_deg > 0.0 and math.isfinite(w_deg):
                                            w_rad = float(w_deg) * (math.pi / 180.0)
                                            w = w * (oracle_norm / float(w_rad)).clamp(0.0, 1.0)

                                        w = w.detach()

                                    p = omega_leg.to(device=device, dtype=dtype)

                                    # Optional: hard-example mining for "direction" issues.
                                    # Apply alignment only when cos(pred, oracle) is below a threshold,
                                    # so we don't perturb cases that are already direction-aligned and
                                    # mainly need amplitude (alpha) tuning.
                                    try:
                                        cos_thr = float(direct_pose_leg_align_cos_thresh or 0.0)
                                    except (TypeError, ValueError): _record_posttrain_soft_fail(trainer, "leg_align_cos_thresh"); cos_thr = 0.0
                                    if 0.0 < cos_thr < 1.0:
                                        with torch.no_grad():
                                            dot = (p * omega_oracle).sum(dim=-1)
                                            den = (p.norm(dim=-1) * omega_oracle.norm(dim=-1)).clamp_min(1e-8)
                                            cos = (dot / den).clamp(-1.0, 1.0)
                                            w = w * (cos < float(cos_thr)).to(dtype=dtype)
                                            w = w.detach()

                                    align_mode = str(direct_pose_leg_align_mode or "cos").strip().lower()
                                    if align_mode not in ("cos", "proj"):
                                        align_mode = "cos"

                                    if align_mode == "cos":
                                        # Cheatable by ||omega_pred||->0 (cos->0 => relu(-cos)=0).
                                        dot = (p * omega_oracle).sum(dim=-1)
                                        den = (p.norm(dim=-1) * omega_oracle.norm(dim=-1)).clamp_min(1e-8)
                                        cos = (dot / den).clamp(-1.0, 1.0)
                                        per = F.relu(-cos)
                                    else:
                                        # Non-cheating projection/magnitude + orthogonal residual supervision.
                                        # omega_oracle = oracle_norm * oracle_dir.
                                        eps = 1e-8
                                        oracle_norm_safe = oracle_norm.clamp_min(eps)
                                        oracle_dir = omega_oracle / oracle_norm_safe.unsqueeze(-1)  # (B,K,3)

                                        proj = (p * oracle_dir).sum(dim=-1)  # (B,K)
                                        res = p - proj.unsqueeze(-1) * oracle_dir
                                        l_mag = (proj - oracle_norm).pow(2)  # rad^2
                                        l_res = (res.pow(2).sum(dim=-1))  # rad^2

                                        mag_w = float(direct_pose_leg_align_mag_weight or 1.0)
                                        res_w = float(direct_pose_leg_align_res_weight or 1.0)
                                        sign_w = float(direct_pose_leg_align_sign_weight or 0.0)
                                        if (not math.isfinite(mag_w)) or mag_w < 0.0:
                                            mag_w = 1.0
                                        if (not math.isfinite(res_w)) or res_w < 0.0:
                                            res_w = 1.0
                                        if (not math.isfinite(sign_w)) or sign_w < 0.0:
                                            sign_w = 0.0

                                        per = (mag_w * l_mag) + (res_w * l_res)
                                        if sign_w > 0.0:
                                            per = per + (sign_w * F.relu(-proj).pow(2))
                                    denom = w.sum().clamp_min(1.0)
                                    leg_align = (per * w).sum() / denom
                                    leg_align_terms.append(leg_align * step_weight)
                                    leg_align_frac_terms.append((w > 0.0).to(dtype=dtype).mean() * step_weight)
                                    joint_names_use = _resolve_leg_align_joint_names(
                                        model=model,
                                        expected_count=int(per.shape[1]),
                                        keep_mask=keep_mask,
                                    )
                                    main_target_joints = _resolve_leg_align_selector_joints(
                                        direct_pose_leg_align_target_joints,
                                        joint_names=joint_names_use,
                                    )
                                    main_loss, main_frac = _compute_leg_align_subset_term(
                                        per=per,
                                        w=w,
                                        joint_names=joint_names_use,
                                        target_joints=main_target_joints,
                                        dtype=dtype,
                                    )
                                    if torch.is_tensor(main_loss):
                                        leg_align_terms.append(main_loss * step_weight)
                                    else:
                                        _record_posttrain_soft_fail(trainer, "leg_align_target_spec_empty")
                                    if torch.is_tensor(main_frac):
                                        leg_align_frac_terms.append(main_frac * step_weight)
                                    _append_leg_align_group_term(
                                        per=per,
                                        w=w,
                                        joint_names=joint_names_use,
                                        target_joints=_LEG_ALIGN_DISTAL_JOINTS,
                                        step_weight=step_weight,
                                        dtype=dtype,
                                        loss_terms=leg_align_distal_terms,
                                        frac_terms=leg_align_distal_frac_terms,
                                    )
                                    _append_leg_align_group_term(
                                        per=per,
                                        w=w,
                                        joint_names=joint_names_use,
                                        target_joints=_LEG_ALIGN_PROXIMAL_JOINTS,
                                        step_weight=step_weight,
                                        dtype=dtype,
                                        loss_terms=leg_align_proximal_terms,
                                        frac_terms=leg_align_proximal_frac_terms,
                                    )
                                    try:
                                        anchor_weight = float(direct_pose_leg_align_anchor_weight or 0.0)
                                    except (TypeError, ValueError):
                                        anchor_weight = 0.0
                                    if anchor_weight > 0.0 and direct_pose_leg_align_anchor_joints is not None:
                                        anchor_target_joints = _resolve_leg_align_selector_joints(
                                            direct_pose_leg_align_anchor_joints,
                                            joint_names=joint_names_use,
                                        )
                                        anchor_loss, anchor_frac = _compute_leg_align_subset_term(
                                            per=per,
                                            w=w,
                                            joint_names=joint_names_use,
                                            target_joints=anchor_target_joints,
                                            dtype=dtype,
                                        )
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

                            # Optional: supervise learned leg omega gate (apply vs no-op) using oracle magnitude.
                            # Target: gate=1 if ||omega_oracle|| >= direct_pose_leg_align_oracle_min_deg else 0.
                            leg_gate_sup_w = float(direct_pose_leg_gate_sup_weight or 0.0)
                            if leg_gate_sup_w > 0.0:
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
                                                # Compute oracle_norm if not already available from align supervision.
                                                if oracle_norm is None:
                                                    with torch.no_grad():
                                                        R_gt_leg = R_gt[:, idx_use, :, :]
                                                        R_delta_oracle = torch.matmul(R_gt_leg, R_leg_base.transpose(-1, -2))
                                                        omega_oracle = so3_log_map(R_delta_oracle)
                                                        oracle_norm = omega_oracle.norm(dim=-1)
                                                if torch.is_tensor(oracle_norm):
                                                    with torch.no_grad():
                                                        min_deg = float(direct_pose_leg_align_oracle_min_deg or 0.0)
                                                        min_rad = float(min_deg) * (math.pi / 180.0)
                                                        tgt = (oracle_norm >= float(min_rad)).to(device=device, dtype=dtype)
                                                    err = F.binary_cross_entropy_with_logits(gl, tgt, reduction="none")
                                                    leg_gate = err.mean()
                                                    leg_gate_sup_terms.append(leg_gate * step_weight)
                                                    leg_gate_sup_tgt_frac_terms.append(tgt.mean() * step_weight)
                                                    leg_gate_sup_pred_mean_terms.append(torch.sigmoid(gl).mean() * step_weight)
                                except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "leg_gate_supervision")

                            R_delta_leg = so3_exp_map(omega_leg)  # (B,K,3,3)
                            R_leg = torch.matmul(R_delta_leg, R_leg_base)
                            R_final = R_base.clone()
                            R_final[:, idx_use, :, :] = R_leg
                            rot6_final = matrix_to_rot6d(R_final, columns=columns).view(B, rot_len)
                            direct_raw_base = direct_raw_base.clone()
                            direct_raw_base[..., rot_slice] = rot6_final
    except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "leg_adjustment_main")

    return direct_raw_base


def _lambda_rollout_unroll_single_step(*, t: int, ctx: Dict[str, Any]) -> None:
    runtime = ctx["runtime"]
    data = ctx["data"]
    weights = ctx["weights"]
    accum = ctx["accum"]
    state_vars = ctx["state_vars"]

    trainer = runtime["trainer"]
    model = runtime["model"]
    state = runtime["state"]
    total_steps = int(runtime["total_steps"])
    cycle_len = int(runtime["cycle_len"])
    include_boundary = bool(runtime["include_boundary"])
    steps = int(runtime["steps"])
    offset = int(runtime["offset"])
    B = int(runtime["B"])
    J = int(runtime["J"])
    objective = str(runtime["objective"])
    rot_len = int(runtime["rot_len"])
    rot_slice = data["rot_slice"]
    step_weights = data["step_weights"]

    meas_used_logits = bool(state_vars["meas_used_logits"])
    direct_nonleg_focus_applied = float(state_vars["direct_nonleg_focus_applied"])
    lam_prev = state_vars["lam_prev"]
    lam_prev_monot = state_vars["lam_prev_monot"]
    plan_prev = state_vars["plan_prev"]

    denom = cycle_len if include_boundary else steps
    idx = int((offset + int(t)) % max(1, int(denom)))
    step_common = _rollout_step_common(
        trainer,
        model,
        state=state,
        t=int(t),
        idx=int(idx),
        total_steps=total_steps,
        cond_seq=data["cond_seq"],
        cond_raw_tgt=data["cond_raw_tgt"],
        cond_norm_mu=data["cond_norm_mu"],
        cond_norm_std=data["cond_norm_std"],
        angvel_seq=data["angvel_seq"],
        pose_hist_seq=data["pose_hist_seq"],
        time_index_mode=runtime["time_index_mode"],
        time_base=runtime["time_base"],
        enable_reprojection=bool(runtime["enable_reprojection"]),
        include_boundary=include_boundary,
        cycle_len=cycle_len,
        yaw_gt_fn=runtime["yaw_gt_fn"],
        detach_rollout_state=bool(runtime["detach_rollout_state"]),
        task_callback=None,
    )
    ret = step_common["ret"]
    contacts_in_t = step_common["contacts_in_t"]
    cond_raw_step = step_common["cond_raw_step"]
    rollout_step_t = step_common["rollout_step_t"]
    y_prev_raw = state["y_prev_raw"]

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

    if contacts_in_t is None and float(data["contact_meas_weight"] or 0.0) > 0.0 and torch.is_tensor(data["contacts_seq"]):
        try:
            contacts_seq = data["contacts_seq"]
            gt_c_t = contacts_seq[:, idx] if contacts_seq.dim() == 3 else contacts_seq
            meas_logits = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(meas_logits):
                if meas_logits.dim() == 3:
                    meas_logits = meas_logits[:, -1]
                if torch.is_tensor(gt_c_t) and gt_c_t.shape == meas_logits.shape:
                    meas_used_logits = True
                    gt = gt_c_t.clamp(0.0, 1.0)
                    accum["meas_terms"].append(F.binary_cross_entropy_with_logits(meas_logits, gt) * step_weights[t])
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_contact_meas_logits")

    delta_raw = delta_norm * data["std_y"]
    prev6 = reproject_rot6d(y_prev_raw[..., rot_slice]).view(B, J, 6)
    R_prev = rot6d_to_matrix(prev6, columns=runtime["columns"])

    if include_boundary and runtime["y0_raw"] is not None and int(idx) == (cycle_len - 1):
        gt_raw = runtime["y0_raw"]
    else:
        gt_raw = trainer._denorm(runtime["gt_seq"][:, idx])
    gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(B, J, 6)
    R_gt = rot6d_to_matrix(gt6, columns=runtime["columns"])

    delta6 = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=runtime["columns"])
    R_delta = rot6d_to_matrix(delta6, columns=runtime["columns"])
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
        device=runtime["device"],
        dtype=runtime["dtype"],
        columns=runtime["columns"],
        rot_slice=rot_slice,
        rot_len=rot_len,
        direct_pose_leg_align_weight=float(weights["direct_pose_leg_align_weight"]),
        direct_pose_leg_align_oracle_min_deg=float(weights["direct_pose_leg_align_oracle_min_deg"]),
        direct_pose_leg_align_oracle_weight_deg=float(weights["direct_pose_leg_align_oracle_weight_deg"]),
        direct_pose_leg_align_mode=str(weights["direct_pose_leg_align_mode"]),
        direct_pose_leg_align_mag_weight=float(weights["direct_pose_leg_align_mag_weight"]),
        direct_pose_leg_align_res_weight=float(weights["direct_pose_leg_align_res_weight"]),
        direct_pose_leg_align_sign_weight=float(weights["direct_pose_leg_align_sign_weight"]),
        direct_pose_leg_align_cos_thresh=float(weights["direct_pose_leg_align_cos_thresh"]),
        direct_pose_leg_align_target_joints=weights["direct_pose_leg_align_target_joints"],
        direct_pose_leg_align_anchor_joints=weights["direct_pose_leg_align_anchor_joints"],
        direct_pose_leg_align_anchor_weight=float(weights["direct_pose_leg_align_anchor_weight"]),
        direct_pose_leg_gate_sup_weight=float(weights["direct_pose_leg_gate_sup_weight"]),
        step_weight=step_weights[t],
        leg_align_terms=accum["leg_align_terms"],
        leg_align_frac_terms=accum["leg_align_frac_terms"],
        leg_align_joint_num_terms=accum["leg_align_joint_num_terms"],
        leg_align_joint_den_terms=accum["leg_align_joint_den_terms"],
        leg_align_joint_frac_terms=accum["leg_align_joint_frac_terms"],
        leg_align_distal_terms=accum["leg_align_distal_terms"],
        leg_align_distal_frac_terms=accum["leg_align_distal_frac_terms"],
        leg_align_proximal_terms=accum["leg_align_proximal_terms"],
        leg_align_proximal_frac_terms=accum["leg_align_proximal_frac_terms"],
        leg_align_anchor_terms=accum["leg_align_anchor_terms"],
        leg_align_anchor_frac_terms=accum["leg_align_anchor_frac_terms"],
        leg_gate_sup_terms=accum["leg_gate_sup_terms"],
        leg_gate_sup_tgt_frac_terms=accum["leg_gate_sup_tgt_frac_terms"],
        leg_gate_sup_pred_mean_terms=accum["leg_gate_sup_pred_mean_terms"],
    )
    dir6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J, 6)
    R_dir = rot6d_to_matrix(dir6, columns=runtime["columns"])

    lam = lam.to(device=runtime["device"], dtype=runtime["dtype"])
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

    accum["lam_vals"].append(lam_raw.detach())
    accum["lam_eff_vals"].append(lam_eff.detach())
    if torch.is_tensor(lam_rel):
        accum["lam_rel_vals"].append(lam_rel.detach())

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

    if (float(weights["lambda_plan_entropy_weight"] or 0.0) > 0.0 or float(weights["lambda_plan_dyn_weight"] or 0.0) > 0.0) and torch.is_tensor(plan_step):
        try:
            plan_det = plan_step.detach()
            ent = _lambda_entropy(plan_det).mean(dim=-1)
            accum["plan_ent_stat_terms"].append(ent.mean() * step_weights[t])
            if float(weights["lambda_plan_entropy_weight"] or 0.0) > 0.0:
                accum["plan_ent_terms"].append((lam_eff.mean(dim=-1) * ent).mean() * step_weights[t])
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_plan_entropy")
        try:
            plan_det = plan_step.detach()
            if plan_prev is not None and torch.is_tensor(plan_prev) and plan_prev.shape == plan_step.shape:
                dyn = (plan_det - plan_prev).abs().mean(dim=-1)
            else:
                dyn = plan_det.new_zeros((B,))
            accum["plan_dyn_stat_terms"].append(dyn.mean() * step_weights[t])
            if float(weights["lambda_plan_dyn_weight"] or 0.0) > 0.0:
                accum["plan_dyn_terms"].append((lam_eff.mean(dim=-1) * dyn).mean() * step_weights[t])
            plan_prev = plan_det
        except _ROLLOUT_SOFT_FAIL_ERRORS:
            _record_posttrain_soft_fail(trainer, "unroll_plan_dynamics")
            plan_prev = plan_step.detach()

    R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
    omega = so3_log_map(R_res)
    R_blend = torch.matmul(so3_exp_map(omega * lam_eff.unsqueeze(-1)), R_inc)

    e_blend = _geodesic_R_safe(R_blend, R_gt)
    e_inc = _geodesic_R_safe(R_inc, R_gt)
    e_dir = _geodesic_R_safe(R_dir, R_gt)
    w_step = step_weights[t]
    accum["loss_terms"].append(e_blend.mean() * w_step)
    accum["inc_terms"].append(e_inc.mean() * w_step)
    e_dir_use = e_dir.mean()

    if objective == "direct":
        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
        root_idx = root_idx if (0 <= root_idx < J) else 0
        nr_mask = None
        if J > 1 and 0 <= root_idx < J:
            nr_mask = torch.ones((J,), device=e_dir.device, dtype=torch.bool)
            nr_mask[root_idx] = False
            e = e_dir[:, nr_mask]
        else:
            e = e_dir
        e_dir_use = e.mean()
        L_leg_base, L_nonleg_base, L_nonleg_plain = None, None, None
        if bool(weights["direct_pose_loss_leg_split"]):
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
                        L_nonleg_plain = e_nonleg.mean()
                        L_nonleg_base = L_nonleg_plain
                        if torch.is_tensor(weights["direct_nonleg_focus_mask_j"]) and int(weights["direct_nonleg_focus_resolved"]) > 0 and abs(float(weights["direct_nonleg_focus_weight_use"]) - 1.0) > 1e-12:
                            focus_mask = weights["direct_nonleg_focus_mask_j"][nr_mask] if torch.is_tensor(nr_mask) and nr_mask.shape == weights["direct_nonleg_focus_mask_j"].shape else weights["direct_nonleg_focus_mask_j"]
                            if focus_mask.shape == leg_mask.shape:
                                focus_nonleg = focus_mask[~leg_mask]
                                if bool(focus_nonleg.any().detach().cpu().item()):
                                    w_non = torch.ones((int(e_nonleg.shape[-1]),), device=e_nonleg.device, dtype=e_nonleg.dtype)
                                    w_non = torch.where(focus_nonleg, w_non * w_non.new_tensor(float(weights["direct_nonleg_focus_weight_use"])), w_non)
                                    L_nonleg_base = ((e_nonleg * w_non.unsqueeze(0)).sum(dim=-1) / w_non.sum().clamp_min(1e-6)).mean()
                                    direct_nonleg_focus_applied = 1.0
                        L_leg_base = e_leg.mean()
                        e_dir_use = L_nonleg_base + L_leg_base
                except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_direct_leg_split")
        accum["dir_base_terms"].append(e_dir_use * w_step)
        if torch.is_tensor(L_leg_base):
            accum["dir_leg_base_terms"].append(L_leg_base * w_step)
        if torch.is_tensor(L_nonleg_base):
            accum["dir_nonleg_base_terms"].append(L_nonleg_base * w_step)
        if torch.is_tensor(L_nonleg_plain):
            accum["dir_nonleg_plain_terms"].append(L_nonleg_plain * w_step)

    accum["dir_terms"].append(e_dir_use * w_step)

    if float(weights["gate_sup_weight"]) > 0.0 and int(t) >= int(weights["gate_sup_start"]):
        lam_logits = ret.get("lambda_fusion_logits", None)
        if torch.is_tensor(lam_logits):
            if lam_logits.dim() == 3:
                lam_logits = lam_logits[:, -1]
            if lam_logits.dim() == 2 and lam_logits.shape[0] == B:
                try:
                    with torch.no_grad():
                        delta = (e_inc - e_dir).detach()
                        lam_star = torch.sigmoid(delta / float(weights["tau_rad"])).detach()
                        mask = (delta.abs() >= float(weights["margin_rad"])).to(dtype=lam_star.dtype) if float(weights["margin_rad"]) > 0.0 else torch.ones_like(lam_star)
                    if lam_logits.shape[-1] == 1:
                        lam_star, mask = lam_star.mean(dim=-1, keepdim=True), mask.mean(dim=-1, keepdim=True)
                    elif lam_logits.shape[-1] != J:
                        lam_star = None
                    if lam_star is not None:
                        lam_star = lam_star.to(device=lam_logits.device, dtype=lam_logits.dtype)
                        mask = mask.to(device=lam_logits.device, dtype=lam_logits.dtype)
                        bce = F.binary_cross_entropy_with_logits(lam_logits, lam_star, reduction="none")
                        mask_sum = mask.sum()
                        accum["gate_sup_terms"].append(((bce * mask).sum() / mask_sum.clamp_min(1e-6)) * w_step)
                        accum["gate_sup_frac_terms"].append(mask.mean() * w_step)
                        with torch.no_grad():
                            pred = (torch.sigmoid(lam_logits) > 0.5).to(dtype=mask.dtype)
                            tgt = (lam_star > 0.5).to(dtype=mask.dtype)
                            accum["gate_sup_acc_num_terms"].append(((pred == tgt).to(dtype=mask.dtype) * mask).sum() * w_step)
                            accum["gate_sup_acc_den_terms"].append(mask_sum * w_step)
                except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_gate_supervision")

    if include_boundary and int(idx) == (cycle_len - 1):
        try:
            accum["boundary_blend_terms"].append(_geodesic_R_safe(R_blend, R_gt).mean().detach())
            accum["boundary_inc_terms"].append(_geodesic_R_safe(R_inc, R_gt).mean().detach())
            accum["boundary_dir_terms"].append(_geodesic_R_safe(R_dir, R_gt).mean().detach())
            accum["boundary_lam_terms"].append(lam_raw.mean().detach())
            accum["boundary_lam_eff_terms"].append(lam_eff.mean().detach())
            if torch.is_tensor(lam_rel):
                accum["boundary_r_terms"].append(lam_rel.mean().detach())
        except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(trainer, "unroll_boundary_stats")

    if float(weights["lambda_early_weight"] or 0.0) > 0.0 and int(weights["lambda_early_steps"] or 0) > 0 and int(t) < int(weights["lambda_early_steps"]):
        accum["early_terms"].append(lam_eff.mean() * w_step)
    if float(weights["lambda_entropy_weight"] or 0.0) > 0.0:
        accum["ent_terms"].append((-_lambda_entropy(lam_eff).mean()) * w_step)
    if float(weights["lambda_smooth_weight"] or 0.0) > 0.0:
        if lam_prev is not None:
            accum["smooth_terms"].append(((lam_eff - lam_prev).pow(2).mean()) * w_step)
        lam_prev = lam_eff.detach()
    if float(weights["lambda_monotonic_weight"] or 0.0) > 0.0:
        if lam_prev_monot is not None:
            accum["mono_terms"].append(F.relu(lam_prev_monot - lam_eff).mean() * w_step)
        lam_prev_monot = lam_eff.detach()

    rot_next6d = matrix_to_rot6d(R_blend, columns=runtime["columns"]).view(B, rot_len)
    y_next_raw = y_prev_raw + delta_raw
    y_next_raw = y_next_raw.clone()
    y_next_raw[..., rot_slice] = rot_next6d
    if bool(runtime["detach_rollout_state"]):
        y_next_raw = y_next_raw.detach()
    if t < total_steps - 1:
        _apply_rollout_carry_state(trainer, state, y_next_raw=y_next_raw, cond_raw_step=cond_raw_step)

    state_vars["meas_used_logits"] = bool(meas_used_logits)
    state_vars["direct_nonleg_focus_applied"] = float(direct_nonleg_focus_applied)
    state_vars["lam_prev"] = lam_prev
    state_vars["lam_prev_monot"] = lam_prev_monot
    state_vars["plan_prev"] = plan_prev


def _build_rollout_unroll_ctx(
    *, trainer: Trainer, model: EventMotionModel, state: Dict[str, Any], prep_ctx: Dict[str, Any], time_index_mode: str,
    time_base: Optional[torch.Tensor], enable_reprojection: bool, detach_rollout_state: bool,
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]], columns: Tuple[str, str], objective: str,
    weights_ctx: Dict[str, Any], accum_ctx: Dict[str, Any], state_vars: Dict[str, Any],
) -> Dict[str, Any]:
    runtime = dict(
        trainer=trainer, model=model, state=state, total_steps=int(prep_ctx["total_steps"]), cycle_len=int(prep_ctx["cycle_len"]),
        include_boundary=bool(prep_ctx["include_boundary"]), steps=int(prep_ctx["steps"]), offset=int(prep_ctx["offset"]),
        time_index_mode=str(time_index_mode), time_base=time_base, enable_reprojection=bool(enable_reprojection),
        detach_rollout_state=bool(detach_rollout_state), yaw_gt_fn=yaw_gt_fn, columns=columns, B=int(prep_ctx["B"]),
        J=int(prep_ctx["J"]), objective=str(objective), y0_raw=prep_ctx["y0_raw"], gt_seq=prep_ctx["gt_seq"],
        device=prep_ctx["device"], dtype=prep_ctx["dtype"], rot_len=int(prep_ctx["rot_len"]),
    )
    data = dict(
        cond_seq=prep_ctx["cond_seq"], cond_raw_tgt=prep_ctx["cond_raw_tgt"], cond_norm_mu=prep_ctx["cond_norm_mu"],
        cond_norm_std=prep_ctx["cond_norm_std"], angvel_seq=prep_ctx["angvel_seq"], pose_hist_seq=prep_ctx["pose_hist_seq"],
        contacts_seq=prep_ctx["contacts_seq"], contact_meas_weight=float(weights_ctx["contact_meas_weight"]),
        step_weights=prep_ctx["step_weights"], std_y=prep_ctx["std_y"], rot_slice=prep_ctx["rot_slice"],
    )
    return {"runtime": runtime, "data": data, "weights": weights_ctx, "accum": accum_ctx, "state_vars": state_vars}


def _lambda_rollout_unroll_steps(ctx: Dict[str, Any]) -> Tuple[bool, float]:
    total_steps = int(ctx["runtime"]["total_steps"])
    for t in range(total_steps):
        _lambda_rollout_unroll_single_step(t=int(t), ctx=ctx)
    state_vars = ctx["state_vars"]
    return bool(state_vars["meas_used_logits"]), float(state_vars["direct_nonleg_focus_applied"])


def _lambda_fusion_run_unroll(
    *,
    runtime_ctx: Dict[str, Any],
    weights_ctx: Dict[str, Any],
    accum_ctx: Dict[str, Any],
    state_vars: Dict[str, Any],
) -> Tuple[bool, float]:
    trainer = runtime_ctx["trainer"]
    model = runtime_ctx["model"]
    batch = runtime_ctx["batch"]
    prep_ctx = runtime_ctx["prep_ctx"]
    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    y0_raw = prep_ctx["y0_raw"]
    gt_seq = prep_ctx["gt_seq"]
    device = prep_ctx["device"]
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

    unroll_ctx = _build_rollout_unroll_ctx(
        trainer=trainer,
        model=model,
        state=prep_ctx["state"],
        prep_ctx=prep_ctx,
        time_index_mode=time_index_mode,
        time_base=time_base,
        enable_reprojection=bool(runtime_ctx["enable_reprojection"]),
        detach_rollout_state=bool(runtime_ctx["detach_rollout_state"]),
        yaw_gt_fn=_yaw_gt_from_gt,
        columns=runtime_ctx["columns"],
        objective=str(runtime_ctx["objective"]),
        weights_ctx=weights_ctx,
        accum_ctx=accum_ctx,
        state_vars=state_vars,
    )
    return _lambda_rollout_unroll_steps(unroll_ctx)


def _lambda_fusion_init_accum_ctx() -> Dict[str, Any]:
    keys = (
        "loss_terms", "inc_terms", "dir_terms", "dir_base_terms", "dir_leg_base_terms", "dir_nonleg_base_terms",
        "dir_nonleg_plain_terms", "leg_gate_sup_terms", "leg_gate_sup_tgt_frac_terms",
        "leg_gate_sup_pred_mean_terms", "leg_align_terms", "leg_align_frac_terms", "leg_align_joint_num_terms",
        "leg_align_joint_den_terms", "leg_align_joint_frac_terms", "leg_align_distal_terms",
        "leg_align_distal_frac_terms", "leg_align_proximal_terms", "leg_align_proximal_frac_terms",
        "leg_align_anchor_terms", "leg_align_anchor_frac_terms", "ent_terms", "smooth_terms",
        "early_terms", "mono_terms", "plan_ent_terms", "plan_dyn_terms", "plan_ent_stat_terms", "plan_dyn_stat_terms",
        "meas_terms", "lam_vals", "lam_eff_vals", "lam_rel_vals", "boundary_blend_terms", "boundary_inc_terms", "boundary_dir_terms",
        "boundary_lam_terms", "boundary_lam_eff_terms", "boundary_r_terms", "gate_sup_terms", "gate_sup_frac_terms",
        "gate_sup_acc_num_terms", "gate_sup_acc_den_terms",
    )
    return {k: [] for k in keys}
def _lambda_fusion_finalize(
    *,
    finalize_ctx: Dict[str, Any],
    accum_ctx: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, Any]]]:
    f = finalize_ctx
    trainer = f["trainer"]
    model = f["model"]
    objective = str(f["objective"])
    direct_group_norm_enable = bool(f["direct_group_norm_enable"])
    include_boundary = bool(f["include_boundary"])
    random_offset = bool(f["random_offset"])
    meas_used_logits = bool(f["meas_used_logits"])
    offset = int(f["offset"])
    boundary_steps = int(f["boundary_steps"])
    direct_nonleg_focus_requested = int(f["direct_nonleg_focus_requested"])
    direct_nonleg_focus_resolved = int(f["direct_nonleg_focus_resolved"])
    (
        direct_group_w_leg, direct_group_w_nonleg, direct_group_beta, direct_group_ratio_min, direct_group_ratio_max,
        direct_group_eps, gate_sup_weight, direct_pose_leg_gate_sup_weight,
        direct_pose_leg_align_weight, lambda_entropy_weight, lambda_smooth_weight, lambda_early_weight,
        lambda_monotonic_weight, lambda_plan_entropy_weight, lambda_plan_dyn_weight, contact_meas_weight,
        boundary_weight, boundary_weighted_sum, direct_nonleg_focus_weight_use, direct_nonleg_focus_applied,
    ) = (
        float(f[k]) for k in (
            "direct_group_w_leg", "direct_group_w_nonleg", "direct_group_beta", "direct_group_ratio_min",
            "direct_group_ratio_max", "direct_group_eps", "gate_sup_weight",
            "direct_pose_leg_gate_sup_weight", "direct_pose_leg_align_weight", "lambda_entropy_weight",
            "lambda_smooth_weight", "lambda_early_weight", "lambda_monotonic_weight", "lambda_plan_entropy_weight",
            "lambda_plan_dyn_weight", "contact_meas_weight", "boundary_weight", "boundary_weighted_sum",
            "direct_nonleg_focus_weight_use", "direct_nonleg_focus_applied",
        )
    )
    (
        loss_terms, inc_terms, dir_terms, dir_base_terms, dir_leg_base_terms, dir_nonleg_base_terms, dir_nonleg_plain_terms,
        leg_gate_sup_terms, leg_gate_sup_tgt_frac_terms, leg_gate_sup_pred_mean_terms,
        leg_align_terms, leg_align_frac_terms, leg_align_joint_num_terms, leg_align_joint_den_terms,
        leg_align_joint_frac_terms, leg_align_distal_terms, leg_align_distal_frac_terms,
        leg_align_proximal_terms, leg_align_proximal_frac_terms, leg_align_anchor_terms, leg_align_anchor_frac_terms,
        ent_terms, smooth_terms, early_terms, mono_terms, plan_ent_terms,
        plan_dyn_terms, plan_ent_stat_terms, plan_dyn_stat_terms, meas_terms, lam_vals, lam_eff_vals, lam_rel_vals,
        boundary_blend_terms, boundary_inc_terms, boundary_dir_terms, boundary_lam_terms, boundary_lam_eff_terms,
        boundary_r_terms, gate_sup_terms, gate_sup_frac_terms, gate_sup_acc_num_terms, gate_sup_acc_den_terms,
    ) = (
        accum_ctx[k]
        for k in (
            "loss_terms", "inc_terms", "dir_terms", "dir_base_terms", "dir_leg_base_terms", "dir_nonleg_base_terms",
            "dir_nonleg_plain_terms", "leg_gate_sup_terms", "leg_gate_sup_tgt_frac_terms",
            "leg_gate_sup_pred_mean_terms", "leg_align_terms", "leg_align_frac_terms", "leg_align_joint_num_terms",
            "leg_align_joint_den_terms", "leg_align_joint_frac_terms", "leg_align_distal_terms",
            "leg_align_distal_frac_terms", "leg_align_proximal_terms", "leg_align_proximal_frac_terms",
            "leg_align_anchor_terms", "leg_align_anchor_frac_terms", "ent_terms", "smooth_terms",
            "early_terms", "mono_terms", "plan_ent_terms", "plan_dyn_terms", "plan_ent_stat_terms", "plan_dyn_stat_terms",
            "meas_terms", "lam_vals", "lam_eff_vals", "lam_rel_vals", "boundary_blend_terms", "boundary_inc_terms",
            "boundary_dir_terms", "boundary_lam_terms", "boundary_lam_eff_terms", "boundary_r_terms", "gate_sup_terms",
            "gate_sup_frac_terms", "gate_sup_acc_num_terms", "gate_sup_acc_den_terms",
        )
    )

    blend_loss_total = torch.stack(loss_terms).sum()
    inc_geo = torch.stack(inc_terms).sum() if inc_terms else blend_loss_total.new_tensor(0.0)
    dir_geo = torch.stack(dir_terms).sum() if dir_terms else blend_loss_total.new_tensor(0.0)
    dir_base = torch.stack(dir_base_terms).sum() if dir_base_terms else blend_loss_total.new_tensor(0.0)
    dir_leg_base = torch.stack(dir_leg_base_terms).sum() if dir_leg_base_terms else blend_loss_total.new_tensor(0.0)
    dir_nonleg_base = torch.stack(dir_nonleg_base_terms).sum() if dir_nonleg_base_terms else blend_loss_total.new_tensor(0.0)
    dir_nonleg_plain = torch.stack(dir_nonleg_plain_terms).sum() if dir_nonleg_plain_terms else blend_loss_total.new_tensor(0.0)
    dir_group_norm_leg = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_leg_ema = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg_ema = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_leg_raw = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg_raw = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_leg_clamped = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg_clamped = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_leg_hit_min = blend_loss_total.new_tensor(0.0)
    dir_group_norm_leg_hit_max = blend_loss_total.new_tensor(0.0)
    dir_group_norm_nonleg_hit_min = blend_loss_total.new_tensor(0.0)
    dir_group_norm_nonleg_hit_max = blend_loss_total.new_tensor(0.0)
    dir_group_norm_leg_hit_any = blend_loss_total.new_tensor(0.0)
    dir_group_norm_nonleg_hit_any = blend_loss_total.new_tensor(0.0)
    dir_group_norm_used = 0.0
    ema_update_payload: Optional[Dict[str, Any]] = None
    if (
        objective == "direct"
        and bool(direct_group_norm_enable)
        and dir_leg_base_terms
        and dir_nonleg_base_terms
    ):
        try:
            ema_state = getattr(trainer, "_direct_pose_group_norm_ema", None)
            if not isinstance(ema_state, dict):
                ema_state = {}
            ema_leg_prev = ema_state.get("leg", None)
            ema_non_prev = ema_state.get("nonleg", None)
            leg_ema_ok = bool(torch.is_tensor(ema_leg_prev))
            if leg_ema_ok:
                try:
                    leg_ema_ok = bool(torch.isfinite(ema_leg_prev).all().detach().cpu().item())
                except (RuntimeError, ValueError, TypeError):
                    _record_posttrain_soft_fail(trainer, "finalize_group_norm_ema_leg_finite")
                    leg_ema_ok = False
            if not leg_ema_ok:
                ema_leg_prev = dir_leg_base.detach()
            else:
                ema_leg_prev = ema_leg_prev.to(device=dir_leg_base.device, dtype=dir_leg_base.dtype)
            non_ema_ok = bool(torch.is_tensor(ema_non_prev))
            if non_ema_ok:
                try:
                    non_ema_ok = bool(torch.isfinite(ema_non_prev).all().detach().cpu().item())
                except (RuntimeError, ValueError, TypeError):
                    _record_posttrain_soft_fail(trainer, "finalize_group_norm_ema_nonleg_finite")
                    non_ema_ok = False
            if not non_ema_ok:
                ema_non_prev = dir_nonleg_base.detach()
            else:
                ema_non_prev = ema_non_prev.to(device=dir_nonleg_base.device, dtype=dir_nonleg_base.dtype)

            leg_ratio_raw = dir_leg_base / ema_leg_prev.clamp_min(float(direct_group_eps))
            non_ratio_raw = dir_nonleg_base / ema_non_prev.clamp_min(float(direct_group_eps))
            leg_ratio = leg_ratio_raw.clamp(
                float(direct_group_ratio_min), float(direct_group_ratio_max)
            )
            non_ratio = non_ratio_raw.clamp(
                float(direct_group_ratio_min), float(direct_group_ratio_max)
            )
            dir_group_norm_leg_raw = leg_ratio_raw
            dir_group_norm_nonleg_raw = non_ratio_raw
            dir_group_norm_leg_clamped = leg_ratio
            dir_group_norm_nonleg_clamped = non_ratio
            dir_group_norm_leg_hit_min = (leg_ratio_raw <= float(direct_group_ratio_min)).to(dtype=dir_leg_base.dtype)
            dir_group_norm_leg_hit_max = (leg_ratio_raw >= float(direct_group_ratio_max)).to(dtype=dir_leg_base.dtype)
            dir_group_norm_nonleg_hit_min = (non_ratio_raw <= float(direct_group_ratio_min)).to(dtype=dir_nonleg_base.dtype)
            dir_group_norm_nonleg_hit_max = (non_ratio_raw >= float(direct_group_ratio_max)).to(dtype=dir_nonleg_base.dtype)
            dir_group_norm_leg_hit_any = torch.maximum(dir_group_norm_leg_hit_min, dir_group_norm_leg_hit_max)
            dir_group_norm_nonleg_hit_any = torch.maximum(dir_group_norm_nonleg_hit_min, dir_group_norm_nonleg_hit_max)
            dir_group_norm_leg = leg_ratio
            dir_group_norm_nonleg = non_ratio
            dir_group_norm_leg_ema = ema_leg_prev
            dir_group_norm_nonleg_ema = ema_non_prev
            dir_geo = float(direct_group_w_leg) * leg_ratio + float(direct_group_w_nonleg) * non_ratio
            dir_group_norm_used = 1.0

            with torch.no_grad():
                beta = float(direct_group_beta)
                ema_update_payload = dict(ema_state, leg=(beta * ema_leg_prev + (1.0 - beta) * dir_leg_base.detach()).detach(), nonleg=(beta * ema_non_prev + (1.0 - beta) * dir_nonleg_base.detach()).detach())
        except _ROLLOUT_SOFT_FAIL_ERRORS:
            _record_posttrain_soft_fail(trainer, "finalize_group_norm_main")
            dir_group_norm_used = 0.0
    if objective == "direct":
        total = dir_geo
    elif objective == "inc":
        total = inc_geo
    else:
        total = blend_loss_total

    gate_sup_loss = blend_loss_total.new_tensor(0.0)
    gate_sup_frac = blend_loss_total.new_tensor(0.0)
    gate_sup_acc = None
    if gate_sup_terms:
        gate_sup_loss = torch.stack(gate_sup_terms).sum()
        total = total + float(gate_sup_weight) * gate_sup_loss
    if gate_sup_frac_terms:
        gate_sup_frac = torch.stack(gate_sup_frac_terms).sum()
    if gate_sup_acc_num_terms and gate_sup_acc_den_terms:
        try:
            num = torch.stack(gate_sup_acc_num_terms).sum()
            den = torch.stack(gate_sup_acc_den_terms).sum()
            gate_sup_acc = torch.where(den > 0.0, num / den.clamp_min(1e-6), den.new_tensor(float("nan"))).detach()
        except (RuntimeError, ValueError, TypeError):
            _record_posttrain_soft_fail(trainer, "finalize_gate_sup_acc")
            gate_sup_acc = None

    leg_gate_sup_loss = blend_loss_total.new_tensor(0.0)
    leg_gate_sup_tgt_frac = blend_loss_total.new_tensor(0.0)
    leg_gate_sup_pred_mean = blend_loss_total.new_tensor(0.0)
    if leg_gate_sup_terms:
        leg_gate_sup_loss = torch.stack(leg_gate_sup_terms).sum()
        total = total + float(direct_pose_leg_gate_sup_weight or 0.0) * leg_gate_sup_loss
    if leg_gate_sup_tgt_frac_terms:
        leg_gate_sup_tgt_frac = torch.stack(leg_gate_sup_tgt_frac_terms).sum()
    if leg_gate_sup_pred_mean_terms:
        leg_gate_sup_pred_mean = torch.stack(leg_gate_sup_pred_mean_terms).sum()

    leg_align_loss = blend_loss_total.new_tensor(0.0)
    leg_align_frac = blend_loss_total.new_tensor(0.0)
    leg_align_distal_loss = blend_loss_total.new_tensor(0.0)
    leg_align_distal_frac = blend_loss_total.new_tensor(0.0)
    leg_align_proximal_loss = blend_loss_total.new_tensor(0.0)
    leg_align_proximal_frac = blend_loss_total.new_tensor(0.0)
    leg_align_anchor_loss = blend_loss_total.new_tensor(0.0)
    leg_align_anchor_frac = blend_loss_total.new_tensor(0.0)
    if leg_align_terms:
        leg_align_loss = torch.stack(leg_align_terms).sum()
        total = total + float(direct_pose_leg_align_weight or 0.0) * leg_align_loss
    if leg_align_frac_terms:
        leg_align_frac = torch.stack(leg_align_frac_terms).sum()
    if leg_align_distal_terms:
        leg_align_distal_loss = torch.stack(leg_align_distal_terms).sum()
    if leg_align_distal_frac_terms:
        leg_align_distal_frac = torch.stack(leg_align_distal_frac_terms).sum()
    if leg_align_proximal_terms:
        leg_align_proximal_loss = torch.stack(leg_align_proximal_terms).sum()
    if leg_align_proximal_frac_terms:
        leg_align_proximal_frac = torch.stack(leg_align_proximal_frac_terms).sum()
    if leg_align_anchor_terms:
        leg_align_anchor_loss = torch.stack(leg_align_anchor_terms).sum()
    if leg_align_anchor_frac_terms:
        leg_align_anchor_frac = torch.stack(leg_align_anchor_frac_terms).sum()

    leg_align_joint_stats: Dict[str, float] = {}
    if leg_align_joint_num_terms and leg_align_joint_den_terms:
        try:
            joint_num = torch.stack(leg_align_joint_num_terms).sum(dim=0)
            joint_den = torch.stack(leg_align_joint_den_terms).sum(dim=0)
            joint_frac = torch.stack(leg_align_joint_frac_terms).sum(dim=0) if leg_align_joint_frac_terms else None
            joint_names = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
            if len(joint_names) != int(joint_num.shape[0]):
                joint_names = [f"j{i}" for i in range(int(joint_num.shape[0]))]
            for joint_idx, joint_name in enumerate(joint_names):
                key_suffix = _sanitize_metric_key_suffix(str(joint_name), default=f"j{joint_idx}")
                den_value = float(joint_den[joint_idx].detach().cpu())
                if den_value > 0.0:
                    joint_loss = float((joint_num[joint_idx] / joint_den[joint_idx].clamp_min(1e-6)).detach().cpu())
                else:
                    joint_loss = float("nan")
                leg_align_joint_stats[f"leg_align_joint_loss_{key_suffix}"] = joint_loss
                if joint_frac is not None:
                    leg_align_joint_stats[f"leg_align_joint_frac_{key_suffix}"] = float(joint_frac[joint_idx].detach().cpu())
        except (RuntimeError, ValueError, TypeError, IndexError):
            _record_posttrain_soft_fail(trainer, "finalize_leg_align_joint_stats")

    entropy_loss = blend_loss_total.new_tensor(0.0)
    if ent_terms:
        entropy_loss = torch.stack(ent_terms).sum()
        total = total + float(lambda_entropy_weight or 0.0) * entropy_loss

    smooth_loss = blend_loss_total.new_tensor(0.0)
    if smooth_terms:
        smooth_loss = torch.stack(smooth_terms).sum()
        total = total + float(lambda_smooth_weight or 0.0) * smooth_loss

    early_loss = blend_loss_total.new_tensor(0.0)
    if early_terms:
        early_loss = torch.stack(early_terms).sum()
        total = total + float(lambda_early_weight or 0.0) * early_loss

    mono_loss = blend_loss_total.new_tensor(0.0)
    if mono_terms:
        mono_loss = torch.stack(mono_terms).sum()
        total = total + float(lambda_monotonic_weight or 0.0) * mono_loss

    plan_ent_loss = blend_loss_total.new_tensor(0.0)
    if plan_ent_terms:
        plan_ent_loss = torch.stack(plan_ent_terms).sum()
        total = total + float(lambda_plan_entropy_weight or 0.0) * plan_ent_loss

    plan_dyn_loss = blend_loss_total.new_tensor(0.0)
    if plan_dyn_terms:
        plan_dyn_loss = torch.stack(plan_dyn_terms).sum()
        total = total + float(lambda_plan_dyn_weight or 0.0) * plan_dyn_loss

    contact_meas_loss = None
    if meas_terms:
        contact_meas_loss = torch.stack(meas_terms).sum()
        total = total + float(contact_meas_weight or 0.0) * contact_meas_loss

    lam_mean = lam_std = None
    lam_eff_mean = lam_eff_std = None
    lam_rel_mean = None
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_vals], dim=0)
        lam_mean = float(flat.mean().detach().cpu())
        lam_std = float(flat.std(unbiased=False).detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_raw")
        lam_mean = lam_std = None
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_eff_vals], dim=0)
        lam_eff_mean = float(flat.mean().detach().cpu())
        lam_eff_std = float(flat.std(unbiased=False).detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_eff")
        lam_eff_mean = lam_eff_std = None
    try:
        if lam_rel_vals:
            flat = torch.cat([x.reshape(-1) for x in lam_rel_vals], dim=0)
            lam_rel_mean = float(flat.mean().detach().cpu())
    except (RuntimeError, ValueError, TypeError):
        _record_posttrain_soft_fail(trainer, "finalize_lambda_stats_rel")
        lam_rel_mean = None

    stats = {
        "blend_loss": float(blend_loss_total.detach().cpu()),
        "gate_sup_loss": float(gate_sup_loss.detach().cpu()),
        "gate_sup_frac": float(gate_sup_frac.detach().cpu()),
        "gate_sup_acc@0.5": float(gate_sup_acc.detach().cpu()) if torch.is_tensor(gate_sup_acc) else float("nan"),
        "leg_gate_sup_loss": float(leg_gate_sup_loss.detach().cpu()),
        "leg_gate_sup_tgt_frac": float(leg_gate_sup_tgt_frac.detach().cpu()),
        "leg_gate_sup_pred_mean": float(leg_gate_sup_pred_mean.detach().cpu()),
        "leg_gate_sup_weighted": float((float(direct_pose_leg_gate_sup_weight or 0.0) * leg_gate_sup_loss).detach().cpu()),
        "leg_align_loss": float(leg_align_loss.detach().cpu()),
        "leg_align_frac": float(leg_align_frac.detach().cpu()),
        "leg_align_distal_loss": float(leg_align_distal_loss.detach().cpu()),
        "leg_align_distal_frac": float(leg_align_distal_frac.detach().cpu()),
        "leg_align_proximal_loss": float(leg_align_proximal_loss.detach().cpu()),
        "leg_align_proximal_frac": float(leg_align_proximal_frac.detach().cpu()),
        "leg_align_anchor_loss": float(leg_align_anchor_loss.detach().cpu()),
        "leg_align_anchor_frac": float(leg_align_anchor_frac.detach().cpu()),
        "leg_align_anchor_weight": float(f.get("direct_pose_leg_align_anchor_weight", 0.0) or 0.0),
        "leg_align_anchor_weighted": float(
            (
                float(direct_pose_leg_align_weight or 0.0)
                * float(f.get("direct_pose_leg_align_anchor_weight", 0.0) or 0.0)
                * leg_align_anchor_loss
            ).detach().cpu()
        ),
        "leg_align_weight": float(direct_pose_leg_align_weight or 0.0),
        "leg_align_weighted": float((float(direct_pose_leg_align_weight or 0.0) * leg_align_loss).detach().cpu()),
        "lambda_mean": float(lam_mean) if lam_mean is not None else float("nan"),
        "lambda_std": float(lam_std) if lam_std is not None else float("nan"),
        "lambda_eff_mean": float(lam_eff_mean) if lam_eff_mean is not None else float("nan"),
        "lambda_eff_std": float(lam_eff_std) if lam_eff_std is not None else float("nan"),
        "lambda_rel_mean": float(lam_rel_mean) if lam_rel_mean is not None else float("nan"),
        "inc_geo": float(inc_geo.detach().cpu()),
        "dir_geo": float(dir_geo.detach().cpu()),
        "dir_base": float(dir_base.detach().cpu()),
        "dir_leg_base": float(dir_leg_base.detach().cpu()),
        "dir_nonleg_base": float(dir_nonleg_base.detach().cpu()),
        "dir_nonleg_plain": float(dir_nonleg_plain.detach().cpu()),
        "dir_nonleg_focus_requested": float(direct_nonleg_focus_requested),
        "dir_nonleg_focus_resolved": float(direct_nonleg_focus_resolved),
        "dir_nonleg_focus_weight": float(direct_nonleg_focus_weight_use),
        "dir_nonleg_focus_applied": float(direct_nonleg_focus_applied),
        "dir_group_norm_used": float(dir_group_norm_used),
        "dir_group_norm_leg_raw": float(dir_group_norm_leg_raw.detach().cpu()) if torch.is_tensor(dir_group_norm_leg_raw) else float("nan"),
        "dir_group_norm_nonleg_raw": float(dir_group_norm_nonleg_raw.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg_raw) else float("nan"),
        "dir_group_norm_leg_clamped": float(dir_group_norm_leg_clamped.detach().cpu()) if torch.is_tensor(dir_group_norm_leg_clamped) else float("nan"),
        "dir_group_norm_nonleg_clamped": float(dir_group_norm_nonleg_clamped.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg_clamped) else float("nan"),
        "dir_group_norm_leg": float(dir_group_norm_leg.detach().cpu()) if torch.is_tensor(dir_group_norm_leg) else float("nan"),
        "dir_group_norm_nonleg": float(dir_group_norm_nonleg.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg) else float("nan"),
        "dir_group_norm_leg_ema": float(dir_group_norm_leg_ema.detach().cpu())
        if torch.is_tensor(dir_group_norm_leg_ema)
        else float("nan"),
        "dir_group_norm_nonleg_ema": float(dir_group_norm_nonleg_ema.detach().cpu())
        if torch.is_tensor(dir_group_norm_nonleg_ema)
        else float("nan"),
        "dir_group_norm_leg_hit_min": float(dir_group_norm_leg_hit_min.detach().cpu()) if torch.is_tensor(dir_group_norm_leg_hit_min) else 0.0,
        "dir_group_norm_leg_hit_max": float(dir_group_norm_leg_hit_max.detach().cpu()) if torch.is_tensor(dir_group_norm_leg_hit_max) else 0.0,
        "dir_group_norm_nonleg_hit_min": float(dir_group_norm_nonleg_hit_min.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg_hit_min) else 0.0,
        "dir_group_norm_nonleg_hit_max": float(dir_group_norm_nonleg_hit_max.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg_hit_max) else 0.0,
        "dir_group_norm_leg_hit_any": float(dir_group_norm_leg_hit_any.detach().cpu()) if torch.is_tensor(dir_group_norm_leg_hit_any) else 0.0,
        "dir_group_norm_nonleg_hit_any": float(dir_group_norm_nonleg_hit_any.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg_hit_any) else 0.0,
        "dir_group_norm_w_leg": float(direct_group_w_leg),
        "dir_group_norm_w_nonleg": float(direct_group_w_nonleg),
        "entropy_loss": float(entropy_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "early_loss": float(early_loss.detach().cpu()),
        "mono_loss": float(mono_loss.detach().cpu()),
        "plan_entropy_loss": float(plan_ent_loss.detach().cpu()),
        "plan_dyn_loss": float(plan_dyn_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    stats.update(leg_align_joint_stats)
    if include_boundary:
        stats["rollout_include_boundary"] = 1.0
        stats["rollout_random_offset"] = 1.0 if bool(random_offset) else 0.0
        stats["rollout_offset"] = float(offset)
        stats["lambda_boundary_weight"] = float(boundary_weight or 0.0)
        stats["boundary_steps"] = float(boundary_steps or 0)
        stats["boundary_weighted_sum"] = float(boundary_weighted_sum or 0.0)
        for key, terms in (
            ("boundary_blend_geo", boundary_blend_terms), ("boundary_inc_geo", boundary_inc_terms),
            ("boundary_dir_geo", boundary_dir_terms), ("boundary_lambda_mean", boundary_lam_terms),
            ("boundary_lambda_eff_mean", boundary_lam_eff_terms), ("boundary_r_mean", boundary_r_terms),
        ):
            if not terms:
                continue
            try:
                stats[key] = float(torch.stack(terms).mean().detach().cpu())
            except (RuntimeError, ValueError, TypeError):
                _record_posttrain_soft_fail(trainer, f"finalize_boundary_{key}")
    for key, terms in (("plan_entropy_mean", plan_ent_stat_terms), ("plan_dyn_mean", plan_dyn_stat_terms)):
        if not terms:
            continue
        try:
            stats[key] = float(torch.stack(terms).sum().detach().cpu())
        except (RuntimeError, ValueError, TypeError):
            _record_posttrain_soft_fail(trainer, f"finalize_plan_{key}")
    if contact_meas_loss is not None:
        if bool(meas_used_logits):
            stats["contact_meas_bce"] = float(contact_meas_loss.detach().cpu())
        else:
            stats["contact_meas_mse"] = float(contact_meas_loss.detach().cpu())
        stats["contact_meas_weighted"] = float((float(contact_meas_weight or 0.0) * contact_meas_loss).detach().cpu())
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
        # Guard against accidentally applying λ-regularizers when not training λ.
        lambda_entropy_weight = 0.0
        lambda_smooth_weight = 0.0
        lambda_early_steps = 0
        lambda_early_weight = 0.0
        lambda_monotonic_weight = 0.0
        lambda_plan_entropy_weight = 0.0
        lambda_plan_dyn_weight = 0.0
        lambda_gate_sup_weight = 0.0
    prep_ctx = _lambda_rollout_prepare_context(
        trainer,
        model,
        batch,
        columns=columns,
        rollout_steps=rollout_steps,
        rollout_cycles=rollout_cycles,
        include_boundary=include_boundary,
        boundary_weight=boundary_weight,
        random_offset=random_offset,
        time_weight_mode=time_weight_mode,
        time_weight_max=time_weight_max,
    )
    device = prep_ctx["device"]
    dtype = prep_ctx["dtype"]
    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    total_steps = int(prep_ctx["total_steps"])
    offset = int(prep_ctx["offset"])
    J = int(prep_ctx["J"])
    step_weights = prep_ctx["step_weights"]
    boundary_steps = int(prep_ctx["boundary_steps"])
    boundary_weighted_sum = float(prep_ctx["boundary_weighted_sum"])
    nonleg_focus_ctx = _lambda_rollout_resolve_nonleg_focus(
        trainer,
        objective=objective,
        direct_pose_nonleg_focus_bones=direct_pose_nonleg_focus_bones,
        direct_pose_nonleg_focus_weight=direct_pose_nonleg_focus_weight,
        J=J,
        device=device,
    )
    direct_nonleg_focus_mask_j = nonleg_focus_ctx["direct_nonleg_focus_mask_j"]
    direct_nonleg_focus_requested = int(nonleg_focus_ctx["direct_nonleg_focus_requested"])
    direct_nonleg_focus_resolved = int(nonleg_focus_ctx["direct_nonleg_focus_resolved"])
    direct_nonleg_focus_weight_use = float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"])
    direct_nonleg_focus_applied = float(nonleg_focus_ctx["direct_nonleg_focus_applied"])
    accum_ctx = _lambda_fusion_init_accum_ctx()
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
    weights_ctx = {
        "contact_meas_weight": contact_meas_weight, "direct_pose_leg_align_weight": direct_pose_leg_align_weight,
        "direct_pose_leg_align_oracle_min_deg": direct_pose_leg_align_oracle_min_deg,
        "direct_pose_leg_align_oracle_weight_deg": direct_pose_leg_align_oracle_weight_deg,
        "direct_pose_leg_align_mode": direct_pose_leg_align_mode, "direct_pose_leg_align_mag_weight": direct_pose_leg_align_mag_weight,
        "direct_pose_leg_align_res_weight": direct_pose_leg_align_res_weight, "direct_pose_leg_align_sign_weight": direct_pose_leg_align_sign_weight,
        "direct_pose_leg_align_cos_thresh": direct_pose_leg_align_cos_thresh,
        "direct_pose_leg_align_target_joints": direct_pose_leg_align_target_joints,
        "direct_pose_leg_align_anchor_joints": direct_pose_leg_align_anchor_joints,
        "direct_pose_leg_align_anchor_weight": direct_pose_leg_align_anchor_weight,
        "direct_pose_leg_gate_sup_weight": direct_pose_leg_gate_sup_weight,
        "direct_pose_loss_leg_split": direct_pose_loss_leg_split, "direct_nonleg_focus_mask_j": direct_nonleg_focus_mask_j,
        "direct_nonleg_focus_resolved": direct_nonleg_focus_resolved, "direct_nonleg_focus_weight_use": direct_nonleg_focus_weight_use,
        "gate_sup_weight": float(reg_ctx["gate_sup_weight"]), "gate_sup_start": int(reg_ctx["gate_sup_start"]),
        "tau_rad": float(reg_ctx["tau_rad"]), "margin_rad": float(reg_ctx["margin_rad"]),
        "lambda_plan_entropy_weight": lambda_plan_entropy_weight, "lambda_plan_dyn_weight": lambda_plan_dyn_weight,
        "lambda_early_weight": lambda_early_weight, "lambda_early_steps": lambda_early_steps,
        "lambda_entropy_weight": lambda_entropy_weight, "lambda_smooth_weight": lambda_smooth_weight,
        "lambda_monotonic_weight": lambda_monotonic_weight,
    }
    state_vars = {"meas_used_logits": False, "direct_nonleg_focus_applied": direct_nonleg_focus_applied, "lam_prev": None, "lam_prev_monot": None, "plan_prev": None}
    runtime_ctx = {
        "trainer": trainer, "model": model, "batch": batch, "prep_ctx": prep_ctx, "time_index_mode": time_index_mode,
        "enable_reprojection": enable_reprojection, "detach_rollout_state": detach_rollout_state, "columns": columns, "objective": objective,
    }
    meas_used_logits, direct_nonleg_focus_applied = _lambda_fusion_run_unroll(runtime_ctx=runtime_ctx, weights_ctx=weights_ctx, accum_ctx=accum_ctx, state_vars=state_vars)
    finalize_ctx = {
        "trainer": trainer, "model": model, "objective": objective,
        "direct_pose_leg_gate_sup_weight": direct_pose_leg_gate_sup_weight, "direct_pose_leg_align_weight": direct_pose_leg_align_weight,
        "direct_pose_leg_align_anchor_weight": direct_pose_leg_align_anchor_weight,
        "lambda_entropy_weight": lambda_entropy_weight, "lambda_smooth_weight": lambda_smooth_weight, "lambda_early_weight": lambda_early_weight,
        "lambda_monotonic_weight": lambda_monotonic_weight, "lambda_plan_entropy_weight": lambda_plan_entropy_weight,
        "lambda_plan_dyn_weight": lambda_plan_dyn_weight, "contact_meas_weight": contact_meas_weight, "include_boundary": include_boundary,
        "random_offset": random_offset, "offset": offset, "boundary_weight": boundary_weight, "boundary_steps": boundary_steps,
        "boundary_weighted_sum": boundary_weighted_sum, "direct_nonleg_focus_requested": direct_nonleg_focus_requested,
        "direct_nonleg_focus_resolved": direct_nonleg_focus_resolved, "direct_nonleg_focus_weight_use": direct_nonleg_focus_weight_use,
        "direct_nonleg_focus_applied": direct_nonleg_focus_applied,
        "meas_used_logits": meas_used_logits, **reg_ctx,
    }
    return _lambda_fusion_finalize(finalize_ctx=finalize_ctx, accum_ctx=accum_ctx)
def _set_seed(seed: int) -> None:
    seed = int(seed or 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def _iter_infinite(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _select_trainable_params(model: EventMotionModel) -> Tuple[list[torch.nn.Parameter], list[str]]:
    trainable: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable.append(param)
        names.append(name)
    return trainable, names


def _module_grad_norm(module: Optional[torch.nn.Module]) -> float:
    if module is None:
        return float("nan")
    total_sq = 0.0
    has_grad = False
    for p in module.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        try:
            total_sq += float(g.detach().float().pow(2).sum().item())
            has_grad = True
        except Exception:
            continue
    if not has_grad:
        return float("nan")
    return float(math.sqrt(max(0.0, total_sq)))


def _grad_list_norm(grads: Iterable[Optional[torch.Tensor]]) -> float:
    total_sq = 0.0
    has_grad = False
    for g in grads:
        if g is None:
            continue
        try:
            gg = g.detach()
            if not torch.isfinite(gg).all():
                gg = torch.nan_to_num(gg, nan=0.0, posinf=0.0, neginf=0.0)
            total_sq += float(gg.float().pow(2).sum().item())
            has_grad = True
        except Exception:
            continue
    if not has_grad:
        return float("nan")
    return float(math.sqrt(max(0.0, total_sq)))


def _grad_list_cosine(grads_a: Iterable[Optional[torch.Tensor]], grads_b: Iterable[Optional[torch.Tensor]]) -> float:
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    has_a = False
    has_b = False
    for ga, gb in zip(grads_a, grads_b):
        gg_a = None
        gg_b = None
        if ga is not None:
            try:
                gg_a = ga.detach()
                if not torch.isfinite(gg_a).all():
                    gg_a = torch.nan_to_num(gg_a, nan=0.0, posinf=0.0, neginf=0.0)
                na2 += float(gg_a.float().pow(2).sum().item())
                has_a = True
            except Exception:
                gg_a = None
        if gb is not None:
            try:
                gg_b = gb.detach()
                if not torch.isfinite(gg_b).all():
                    gg_b = torch.nan_to_num(gg_b, nan=0.0, posinf=0.0, neginf=0.0)
                nb2 += float(gg_b.float().pow(2).sum().item())
                has_b = True
            except Exception:
                gg_b = None
        if gg_a is not None and gg_b is not None:
            try:
                dot += float((gg_a.float() * gg_b.float()).sum().item())
            except Exception:
                continue
    if (not has_a) or (not has_b):
        return float("nan")
    na = math.sqrt(max(0.0, na2))
    nb = math.sqrt(max(0.0, nb2))
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(dot / (na * nb))


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


def _merge_grad_norm(*vals: float) -> float:
    total_sq = 0.0
    has = False
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        total_sq += float(fv * fv)
        has = True
    if not has:
        return float("nan")
    return float(math.sqrt(max(0.0, total_sq)))


def _resolve_train_mode(cfg: PostTrainConfig) -> str:
    selected = int(bool(cfg.train_direct_pose)) + int(bool(cfg.train_lambda_head))
    if selected != 1:
        raise SystemExit("[FATAL] Choose exactly one: train_direct_pose | train_lambda_head.")
    return "direct" if bool(cfg.train_direct_pose) else "lambda"


def _train_mode_display_name(train_mode: str) -> str:
    mapping = {
        "direct": "train_direct_pose",
        "lambda": "train_lambda_head",
    }
    try:
        return str(mapping[train_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown train_mode={train_mode!r}") from exc


def _expected_trainable_prefixes(train_mode: str) -> list[str]:
    by_mode: Dict[str, Tuple[str, ...]] = {
        "lambda": ("lambda_fusion_head",),
        "direct": (
            "direct_pose_head",
            "direct_pose_out_leg",
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
        return list(by_mode[train_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown train_mode={train_mode!r}") from exc


def _build_rollout_mode_kwargs(cfg: PostTrainConfig, train_mode: str) -> Dict[str, Any]:
    if train_mode == "direct":
        return {
            "lambda_entropy_weight": 0.0,
            "lambda_smooth_weight": 0.0,
            "lambda_early_steps": 0,
            "lambda_early_weight": 0.0,
            "lambda_monotonic_weight": 0.0,
            "lambda_plan_entropy_weight": 0.0,
            "lambda_plan_dyn_weight": 0.0,
            "objective": "direct",
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
    if train_mode == "lambda":
        return {
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
            "objective": "blend",
        }
    raise ValueError(f"Unknown train_mode={train_mode!r}")


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


def _freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _enable_modules(model: torch.nn.Module, names: Tuple[str, ...]) -> None:
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad_(True)


def _unfreeze_lambda_fusion(model: EventMotionModel) -> None:
    _enable_modules(model, ("lambda_fusion_head",))


def _unfreeze_direct_pose(
    model: EventMotionModel,
    *,
    leg_only: bool = False,
    leg_gate_only: bool = False,
    nonleg_only: bool = False,
) -> None:
    if bool(leg_gate_only):
        _enable_modules(model, ("direct_pose_leg_gate_head",))
        return
    if bool(leg_only):
        _enable_modules(model, ("direct_pose_leg_head", "direct_pose_leg_gate_head"))
        return
    if bool(nonleg_only):
        _enable_modules(
            model,
            (
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_nonleg_proj",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_out_nonleg",
            ),
        )
        return

    _enable_modules(
        model,
        (
            "direct_pose_head",
            "direct_pose_out_leg",
            "direct_pose_out_nonleg",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_leg_head",
            "direct_pose_leg_gate_head",
        ),
    )


def _unfreeze_for_train_mode(model: EventMotionModel, cfg: PostTrainConfig, train_mode: str) -> None:
    if train_mode == "direct":
        _unfreeze_direct_pose(
            model,
            leg_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
            leg_gate_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
            nonleg_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
        )
        return
    if train_mode == "lambda":
        _unfreeze_lambda_fusion(model)
        return
    raise ValueError(f"Unknown train_mode={train_mode!r}")


def _apply_cli_overrides(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    args_map = vars(args)
    if args.paths is not None:
        payload["paths"] = args.paths or None

    for key in _CLI_BOOL_OVERRIDE_KEYS:
        raw = args_map.get(key)
        if raw is None:
            continue
        payload[key] = str(raw).strip().lower() in ("1", "true", "yes", "y")

    for key in _CLI_OPTIONAL_FLOAT_OVERRIDE_KEYS:
        raw = args_map.get(key)
        if raw is None:
            continue
        s = str(raw).strip().lower()
        payload[key] = None if s in ("null", "none", "") else float(raw)

    for key, value in args_map.items():
        if key in _CLI_OVERRIDE_SPECIAL_KEYS or value is None:
            continue
        payload[key] = value


def _run_training_loop(*, cfg: PostTrainConfig, train_mode: str, model: EventMotionModel, params: list[torch.nn.Parameter], opt: torch.optim.Optimizer, batch_iter: Any, rollout_common_kwargs: Dict[str, Any], rollout_mode_kwargs: Dict[str, Any], l2sp_pairs: list[tuple[torch.nn.Parameter, torch.Tensor]], l2sp_weight: float) -> list[dict[str, Any]]:
    log_rows: list[dict[str, Any]] = []
    global_step = 0
    save_step_set = _parse_int_set_spec(getattr(cfg, "save_step_ckpts", None))

    def _save_step_snapshot(step_idx: int) -> None:
        if int(step_idx) < 0:
            return
        ckpt_step_out = cfg.out_dir / f"ckpt_step_{int(step_idx):06d}_{cfg.run_name}.pth"
        torch.save({"model": model.state_dict(), "posttrain_cfg": _cfg_to_jsonable(cfg)}, ckpt_step_out)

    if 0 in save_step_set:
        _save_step_snapshot(0)

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_loss = 0.0
        ok_steps = 0
        bad_steps = 0
        for it in range(int(cfg.steps_per_epoch)):
            batch = next(batch_iter)
            opt.zero_grad(set_to_none=True)
            rollout_mode_kwargs_step = rollout_mode_kwargs
            if train_mode == "direct":
                rollout_mode_kwargs_step = dict(rollout_mode_kwargs)
                rollout_mode_kwargs_step["direct_pose_leg_align_weight"] = _resolve_direct_pose_leg_align_weight(
                    cfg, global_step
                )
            loss, stats, aux_payload = _lambda_fusion_loss_rollout(
                batch=batch,
                **rollout_common_kwargs,
                **rollout_mode_kwargs_step,
            )
            ema_update_payload = aux_payload.get("ema_update_payload", None) if isinstance(aux_payload, dict) else None
            if train_mode == "direct" and isinstance(ema_update_payload, dict):
                try: setattr(rollout_common_kwargs["trainer"], "_direct_pose_group_norm_ema", dict(ema_update_payload, leg=ema_update_payload["leg"].detach(), nonleg=ema_update_payload["nonleg"].detach())) if (torch.is_tensor(ema_update_payload.get("leg", None)) and torch.is_tensor(ema_update_payload.get("nonleg", None)) and bool(torch.isfinite(ema_update_payload["leg"]).all().detach().cpu().item()) and bool(torch.isfinite(ema_update_payload["nonleg"]).all().detach().cpu().item())) else _record_posttrain_soft_fail(rollout_common_kwargs["trainer"], "apply_ema_update_invalid_payload")
                except _ROLLOUT_SOFT_FAIL_ERRORS: _record_posttrain_soft_fail(rollout_common_kwargs["trainer"], "apply_ema_update_setattr")
            elif ema_update_payload is not None: _record_posttrain_soft_fail(rollout_common_kwargs["trainer"], "apply_ema_update_nontrain_or_bad_payload")
            if train_mode == "lambda" and l2sp_pairs and l2sp_weight > 0.0:
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
                train_mode == "direct"
                and bool(getattr(cfg, "direct_pose_leg_align_grad_probe_enable", False))
                and int(global_step) < int(getattr(cfg, "direct_pose_leg_align_grad_probe_steps", 0) or 0)
            ):
                try:
                    _run_leg_align_grad_probe(cfg=cfg, model=model, stats=stats, aux_payload=aux_payload)
                except _ROLLOUT_SOFT_FAIL_ERRORS:
                    _record_posttrain_soft_fail(rollout_common_kwargs["trainer"], "leg_align_grad_probe")
            loss.backward()
            if train_mode == "direct" and bool(getattr(cfg, "direct_pose_grad_monitor_enable", False)):
                g_trunk = _module_grad_norm(getattr(model, "direct_pose_head", None))
                g_leg = _module_grad_norm(getattr(model, "direct_pose_out_leg", None))
                g_nonleg_head = _module_grad_norm(getattr(model, "direct_pose_out_nonleg", None))
                g_arm = _module_grad_norm(getattr(model, "direct_pose_out_arm", None))
                g_else = _module_grad_norm(getattr(model, "direct_pose_out_else", None))
                g_nonleg = _merge_grad_norm(g_nonleg_head, _merge_grad_norm(g_arm, g_else))
                g_leg_head = _module_grad_norm(getattr(model, "direct_pose_leg_head", None))
                g_leg_head_shared = _module_grad_norm(getattr(model, "direct_pose_leg_head_shared", None))
                g_leg_branch = _merge_grad_norm(g_leg_head, g_leg_head_shared)
                ratio = float("nan")
                if math.isfinite(g_leg) and math.isfinite(g_nonleg):
                    ratio = float(g_nonleg / max(1e-12, g_leg))
                ratio_leg_branch = float("nan")
                if math.isfinite(g_leg_branch) and math.isfinite(g_nonleg):
                    ratio_leg_branch = float(g_nonleg / max(1e-12, g_leg_branch))
                stats["direct_grad_norm_trunk"] = float(g_trunk)
                stats["direct_grad_norm_out_leg"] = float(g_leg)
                stats["direct_grad_norm_out_nonleg"] = float(g_nonleg)
                stats["direct_grad_norm_out_arm"] = float(g_arm)
                stats["direct_grad_norm_out_else"] = float(g_else)
                stats["direct_grad_norm_leg_head"] = float(g_leg_head)
                stats["direct_grad_norm_leg_head_shared"] = float(g_leg_head_shared)
                stats["direct_grad_norm_leg_branch"] = float(g_leg_branch)
                stats["direct_grad_ratio_nonleg_over_leg"] = float(ratio)
                stats["direct_grad_ratio_nonleg_over_leg_branch"] = float(ratio_leg_branch)
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
                    steps_per_epoch=int(cfg.steps_per_epoch),
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


def _cfg_to_jsonable(cfg: PostTrainConfig) -> dict[str, Any]:
    cfg_jsonable: dict[str, Any] = {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, tuple) and v and all(isinstance(p, Path) for p in v) else v) for k, v in cfg.__dict__.items()}
    return cfg_jsonable


def _save_posttrain_outputs(*, cfg: PostTrainConfig, model: EventMotionModel, log_rows: list[dict[str, Any]], direct_pose_feat_source: str, direct_pose_time_pe_dim: int, direct_pose_time_pe_base: float, direct_pose_use_phase_z: bool, direct_pose_phase_z_mode: str, direct_pose_split_enable: bool, direct_pose_nonleg_proj_dim: int, direct_pose_leg_gate_mode_model: str, direct_pose_leg_gate_power_model: float) -> Path:
    cfg_jsonable = _cfg_to_jsonable(cfg)
    cfg_jsonable["direct_pose_feat_source"] = str(direct_pose_feat_source)
    cfg_jsonable["direct_pose_time_pe_dim"] = int(direct_pose_time_pe_dim)
    cfg_jsonable["direct_pose_time_pe_base"] = float(direct_pose_time_pe_base)
    cfg_jsonable["direct_pose_use_phase_z"] = bool(direct_pose_use_phase_z)
    cfg_jsonable["direct_pose_phase_z_mode"] = str(direct_pose_phase_z_mode)
    cfg_jsonable["direct_pose_split_enable"] = bool(direct_pose_split_enable)
    cfg_jsonable["direct_pose_nonleg_proj_dim"] = int(direct_pose_nonleg_proj_dim)
    cfg_jsonable["direct_pose_arm_split_enable"] = bool(getattr(cfg, "direct_pose_arm_split_enable", False))
    cfg_jsonable["direct_pose_arm_bones"] = getattr(cfg, "direct_pose_arm_bones", None)
    cfg_jsonable["direct_pose_nonleg_train_only"] = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
    cfg_jsonable["direct_pose_leg_gate_mode"] = str(direct_pose_leg_gate_mode_model)
    cfg_jsonable["direct_pose_leg_gate_power"] = float(direct_pose_leg_gate_power_model)
    ckpt_out = cfg.out_dir / f"ckpt_last_{cfg.run_name}.pth"
    torch.save({"model": model.state_dict(), "posttrain_cfg": cfg_jsonable}, ckpt_out)
    dump_json(cfg.out_dir / f"posttrain_log_{cfg.run_name}.json", {"config": cfg_jsonable, "log": log_rows})
    return ckpt_out


def _build_dataset_and_loader(cfg: PostTrainConfig) -> tuple[dict[str, Any], MotionEventDataset, Any]:
    norm_spec = _merge_norm_spec(cfg.bundle_json.expanduser().resolve(), cfg.pretrain_template)
    ds = MotionEventDataset(data_dir=str(cfg.data.expanduser().resolve()), seq_len=max(2, int(cfg.seq_len)), paths=[str(p.expanduser().resolve()) for p in cfg.paths] if cfg.paths else None, pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0), norm_spec=norm_spec, index_mode=str(getattr(cfg, "dataset_index_mode", "sliding") or "sliding"))
    ds.is_train = True
    if len(ds) <= 0:
        clip_lens: list[tuple[str, int]] = []
        for clip in getattr(ds, "clips", []) or []:
            try:
                clip_lens.append((str(getattr(clip, "npz_path", "?")), int(getattr(clip, "X", np.zeros((0,))).shape[0])))
            except Exception:
                pass
        clip_lens.sort(key=lambda x: x[1])
        hint = f" Smallest clips: {', '.join([f'{Path(p).name}:{n}' for p, n in clip_lens[:5]])}." if clip_lens else ""
        raise SystemExit(f"[FATAL] posttrain dataset has 0 samples. seq_len={cfg.seq_len} is likely too large or paths/data are wrong." + hint + " Try lowering --seq_len or passing --paths to restrict to longer clips.")
    loader = DataLoader(ds, batch_size=int(cfg.batch), shuffle=True, drop_last=True, num_workers=0)
    if len(loader) <= 0:
        raise SystemExit(f"[FATAL] posttrain DataLoader has 0 batches (len(dataset)={len(ds)}, batch={int(cfg.batch)}, drop_last=True). Lower --batch or use more/longer --paths (or reduce --seq_len).")
    return norm_spec, ds, _iter_infinite(loader)


def _build_model_and_trainer(*, cfg: PostTrainConfig, ds: MotionEventDataset, model: EventMotionModel, norm_spec: dict[str, Any]) -> Trainer:
    loss_fn = MotionJointLoss(output_layout=getattr(ds, "output_layout", None), fps=float(getattr(ds, "fps", 60.0) or 60.0), rot6d_spec=getattr(ds, "rot6d_spec", None) or {}, meta=getattr(ds, "meta", None) or {})
    trainer = Trainer(model=model, loss_fn=loss_fn, lr=float(getattr(cfg, "lr", 1e-4) or 1e-4), grad_clip=0.0, weight_decay=float(getattr(cfg, "weight_decay", 0.0) or 0.0), tf_warmup_steps=0, tf_total_steps=0, augmentor=None, use_amp=False, accum_steps=1, pin_memory=False)
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
    try:
        trainer.fps = float(getattr(ds, "fps", 60.0) or 60.0)
        trainer.bone_hz = float(trainer.fps)
    except Exception:
        pass

    Dx, Dy = int(ds.Dx), int(ds.Dy)
    x_layout = getattr(ds, "state_layout", None) or {}
    y_layout = getattr(ds, "output_layout", None) or {}
    trainer._x_layout, trainer._y_layout = dict(x_layout), dict(y_layout)
    trainer.rootvel_slice = parse_layout_entry(y_layout.get("RootVelocity"), "RootVelocity", Dy)
    trainer.angvel_slice = parse_layout_entry(y_layout.get("BoneAngularVelocities"), "BoneAngularVelocities", Dy)
    trainer.rootvel_x_slice = parse_layout_entry(x_layout.get("RootVelocity"), "RootVelocity", Dx)
    trainer.angvel_x_slice = parse_layout_entry(x_layout.get("BoneAngularVelocities"), "BoneAngularVelocities", Dx)
    trainer.rootpos_x_slice = parse_layout_entry(x_layout.get("RootPosition"), "RootPosition", Dx)
    trainer.rot6d_x_slice = parse_layout_entry(x_layout.get("BoneRotations6D"), "BoneRotations6D", Dx)
    trainer.rot6d_y_slice = parse_layout_entry(y_layout.get("BoneRotations6D"), "BoneRotations6D", Dy)

    y_to_x_map = []
    for name in sorted(set(x_layout.keys()) & set(y_layout.keys())):
        xs = parse_layout_entry(x_layout.get(name), name, Dx)
        ys = parse_layout_entry(y_layout.get(name), name, Dy)
        if not (isinstance(xs, slice) and isinstance(ys, slice)):
            continue
        k = min(int(xs.stop - xs.start), int(ys.stop - ys.start))
        if k <= 0:
            continue
        y_to_x_map.append({"name": str(name), "x_start": int(xs.start), "x_size": k, "y_start": int(ys.start), "y_size": k})
    trainer.y_to_x_map = y_to_x_map

    mu_x = np.asarray(norm_spec.get("MuX"), dtype=np.float32)
    std_x = np.asarray(norm_spec.get("StdX"), dtype=np.float32)
    mu_y = np.asarray(norm_spec.get("MuY"), dtype=np.float32)
    std_y = np.asarray(norm_spec.get("StdY"), dtype=np.float32)
    try:
        setattr(loss_fn, "mu_y", mu_y)
        setattr(loss_fn, "std_y", std_y)
        setattr(trainer, "mu_y", mu_y)
        setattr(trainer, "std_y", std_y)
    except Exception:
        pass
    trainer.normalizer = DataNormalizer(mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y, y_to_x_map=y_to_x_map, rootvel_x_slice=trainer.rootvel_x_slice, rootvel_y_slice=trainer.rootvel_slice, angvel_x_slice=trainer.angvel_x_slice, angvel_y_slice=trainer.angvel_slice, tanh_scales_rootvel=norm_spec.get("tanh_scales_rootvel", None), tanh_scales_angvel=norm_spec.get("tanh_scales_angvel", None), angvel_mode=getattr(ds, "angvel_norm_mode", None), angvel_mu=getattr(ds, "angvel_mu", None), angvel_std=getattr(ds, "angvel_std", None))
    trainer.pose_hist_len = int(getattr(ds, "pose_hist_len", 0) or 0)
    trainer.pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0) or 0)

    gate_raw = str(getattr(cfg, "contact_meas_gate_by_hit", "auto") or "auto").strip().lower()
    trainer.contact_meas_gate_by_hit_override = True if gate_raw in ("true", "1", "yes", "y") else False if gate_raw in ("false", "0", "no", "n") else None
    trainer.contact_meas_vxy_mode = str(getattr(cfg, "contact_meas_vxy_mode", "abs") or "abs").strip().lower()
    trainer.contact_meas_ground_z_mode = str(getattr(cfg, "contact_meas_ground_z_mode", "window") or "window").strip().lower()
    trainer.contact_meas_ground_z_beta = float(getattr(cfg, "contact_meas_ground_z_beta", 0.05) or 0.05)
    trainer.contact_meas_ground_z_window = int(getattr(cfg, "contact_meas_ground_z_window", 5) or 5)
    trainer.contact_meas_ground_z_quantile = float(getattr(cfg, "contact_meas_ground_z_quantile", 0.2) or 0.2)
    try:
        up_cm, down_cm = float(getattr(cfg, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0), float(getattr(cfg, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0)
    except Exception:
        up_cm, down_cm = 0.0, 0.0
    trainer.contact_meas_ground_z_max_up_m = max(0.0, up_cm) / 100.0
    trainer.contact_meas_ground_z_max_down_m = max(0.0, down_cm) / 100.0
    trainer.posttrain_contacts_source = str(getattr(cfg, "posttrain_contacts_source", "pretrain_contact") or "pretrain_contact").strip().lower()
    if trainer.posttrain_contacts_source != "pretrain_contact":
        raise SystemExit(
            f"[FATAL] unsupported posttrain_contacts_source={trainer.posttrain_contacts_source!r}; "
            "only 'pretrain_contact' is allowed."
        )
    try:
        clamp_k = float(getattr(cfg, "posttrain_contacts_pretrain_clamp", 1.0) or 0.0)
    except Exception:
        clamp_k = 1.0
    if (not math.isfinite(float(clamp_k))) or float(clamp_k) < 0.0:
        clamp_k = 1.0
    trainer.posttrain_contacts_pretrain_clamp = float(clamp_k)
    trainer.posttrain_contacts_pretrain_affine_stats_spec = getattr(cfg, "posttrain_contacts_pretrain_affine_stats", None)
    trainer.posttrain_contacts_pretrain_affine = _parse_pretrain_contact_affine_spec(
        trainer.posttrain_contacts_pretrain_affine_stats_spec
    )
    trainer.lambda_reliability_mode = str(cfg.lambda_reliability_mode or "none")
    trainer.lambda_reliability_warmup_steps = int(cfg.lambda_reliability_warmup_steps or 0)
    trainer.lambda_reliability_contact_err_max = float(cfg.lambda_reliability_contact_err_max or 1.0)
    trainer.lambda_reliability_warmup_joint_scales = cfg.lambda_reliability_warmup_joint_scales
    return trainer




def _build_posttrain_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Post-train entry.\n"
            "Recommended newflow targets: train_direct_pose (Stage6/7) or train_lambda_head (lambda final).\n"
            "Legacy targets are retired and no longer supported."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", default="config/posttrain.json", help="Path to post-train JSON config (optional).")
    ap.add_argument("--ckpt_in", default=None, help="Input checkpoint path (overrides config).")
    ap.add_argument("--out_dir", default=None, help="Output directory (overrides config).")
    ap.add_argument("--run_name", default=None, help="Run name (overrides config).")

    ap.add_argument("--data", default=None, help="Dataset root (processed .npz directory).")
    ap.add_argument("--paths", nargs="*", default=None, help="Optional explicit .npz paths (overrides config).")
    ap.add_argument("--bundle_json", default=None, help="Bundle JSON (norm_template.json).")
    ap.add_argument("--pretrain_template", default=None, help="Optional pretrain template JSON to merge norm spec.")
    ap.add_argument("--encoder_bundle", default=None, help="Optional motion encoder bundle (.pt).")

    ap.add_argument("--device", default=None, help="auto|cpu|cuda|mps")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument(
        "--dataset_index_mode",
        type=str,
        default=None,
        help="Dataset window sampling: sliding|start0|clip_random (balanced per-clip random start).",
    )
    ap.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="Rollout horizon for loss unroll (<= seq_len-1). 0/None uses full window.",
    )
    ap.add_argument("--rollout_cycles", type=int, default=None, help="Unroll multiple cycles by repeating the (seq_len-1) transitions with modulo indexing.")
    ap.add_argument("--rollout_include_boundary", type=str, default=None, help="true|false; include wrap boundary transitions when rollout_cycles>1 (aligns with freerun_cycles).")
    ap.add_argument("--rollout_random_offset", type=str, default=None, help="true|false; randomize cycle phase (start offset) per batch when rollout_cycles>1.")
    ap.add_argument("--time_index_mode", type=str, default=None, help="global|cycle|auto|none (time_index feeding for contact_plan time-PE).")
    ap.add_argument(
        "--phase_reset_source",
        type=str,
        default=None,
        help=(
            "Phase reset / clock-anchor source. "
            "'contacts_meas'=internal threshold-crossing reset; "
            "'ttc_gt'=external reset from GT touchdown events (ttc_td_events; rollout-only); "
            "'none'=disable internal resets."
        ),
    )
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--context_len", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps_per_epoch", type=int, default=None)
    ap.add_argument("--save_step_ckpts", type=str, default=None, help="Optional step checkpoints to save, e.g. '0,1,5,20,60'.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)

    ap.add_argument("--so3_corr_gate_logit_reset", default=None, help="Reset model.so3_corr_gate_logit to a float (e.g. -2.2).")
    ap.add_argument("--detach_rollout_state", type=str, default=None, help="true|false")
    ap.add_argument("--train_direct_pose", type=str, default=None, help="true|false; whether to finetune direct_pose_head (direct expert) via rollout loss")
    ap.add_argument("--contact_plan_init_mode", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--contact_plan_init_hidden", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--contact_plan_init_dropout", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--event_clock",
        type=str,
        default=None,
        choices=("auto", "on", "off"),
        help="Event-Clock v3 mode: auto|on|off (auto keeps ckpt behavior; off drops weights on save).",
    )
    ap.add_argument("--event_clock_max_delta", type=float, default=None, help="Event-Clock clamp for Δz residual magnitude.")
    ap.add_argument("--event_clock_hidden_dim", type=int, default=None, help="Override Event-Clock corrector hidden dim (Δz MLP).")
    ap.add_argument("--event_clock_gate_hidden_dim", type=int, default=None, help="Override Event-Clock gate hidden dim (λ_corr MLP).")
    ap.add_argument("--train_lambda_head", type=str, default=None, help="true|false; whether to train lambda_fusion_head (Stage2)")
    ap.add_argument("--contact_meas_weight", type=float, default=None, help="Weight for contact_meas MSE vs GT soft contacts.")
    ap.add_argument(
        "--direct_pose_split_enable",
        type=str,
        default=None,
        help="true|false; split direct output heads into leg/non-leg with shared trunk (B2).",
    )
    ap.add_argument(
        "--direct_pose_nonleg_proj_dim",
        type=int,
        default=None,
        help="Optional non-leg bottleneck dim for split head: h_nonleg=ReLU(Linear(hid,proj)); 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_arm_split_enable",
        type=str,
        default=None,
        help="true|false; split non-leg branch into arm/else heads (three-way: leg/arm/else).",
    )
    ap.add_argument(
        "--direct_pose_arm_bones",
        type=str,
        default=None,
        help="Comma-separated bone names/indices for arm branch when direct_pose_arm_split_enable=true.",
    )
    ap.add_argument(
        "--direct_pose_nonleg_train_only",
        type=str,
        default=None,
        help="true|false; when train_direct_pose, freeze trunk/leg and train non-leg branch only.",
    )
    ap.add_argument(
        "--direct_pose_leg_enable",
        type=str,
        default=None,
        help="true|false; enable leg-specific residual head for direct pose (extra lower-body capacity).",
    )
    ap.add_argument(
        "--direct_pose_leg_train_only",
        type=str,
        default=None,
        help="true|false; when train_direct_pose, freeze direct_pose_head and train leg head only.",
    )
    ap.add_argument(
        "--direct_pose_leg_bones",
        type=str,
        default=None,
        help="Comma-separated bone names/indices for leg head (default: ball/foot/calf/thigh L+R).",
    )
    ap.add_argument(
        "--direct_pose_leg_mode",
        type=str,
        default=None,
        choices=("rot6d_add", "so3"),
        help="Leg residual mode: rot6d_add (compat) | so3 (on-manifold compose exp(omega)@R).",
    )
    ap.add_argument(
        "--direct_pose_leg_stopgrad_main",
        type=str,
        default=None,
        help="true|false; when leg_mode=so3, stop-grad main head leg rotations in the composition.",
    )
    ap.add_argument(
        "--direct_pose_leg_detach_feat",
        type=str,
        default=None,
        help="true|false; detach leg head inputs so leg loss won't update the backbone (strong decoupling).",
    )
    ap.add_argument(
        "--direct_pose_leg_max_deg",
        type=float,
        default=None,
        help="Max ||omega|| in degrees for leg_mode=so3. 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_gate_mode",
        type=str,
        default=None,
        choices=("auto", "none", "learned", "scale"),
        help=(
            "Optional learned gate/scale for leg omega (SO(3) only): "
            "none | learned | scale. 'auto' is deprecated and treated as 'none'."
        ),
    )
    ap.add_argument(
        "--direct_pose_leg_gate_power",
        type=float,
        default=None,
        help="Gate power for leg omega (SO(3) only): omega_eff = sigmoid(gate_logits)**power * omega_raw.",
    )
    ap.add_argument(
        "--direct_pose_leg_scale_clamp_k",
        type=float,
        default=None,
        help="Optional hard clamp on leg scale magnitude: k>1 => [1/k, k]. 0/1 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_gate_sup_weight",
        "--direct_pose_leg_gate_loss_weight",
        dest="direct_pose_leg_gate_sup_weight",
        type=float,
        default=None,
        help="Optional supervised loss weight for learned leg gate (BCEWithLogits vs oracle ||omega_oracle|| thresholding). 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_weight",
        type=float,
        default=None,
        help="Optional direction alignment loss target weight for leg omega. When schedule=none it is constant; otherwise it is the ramp target.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_schedule",
        type=str,
        default=None,
        choices=("none", "linear"),
        help="Leg align weight schedule: none | linear (hold start_weight for warmup_steps, then ramp to target weight).",
    )
    ap.add_argument(
        "--direct_pose_leg_align_start_weight",
        type=float,
        default=None,
        help="Leg align schedule start weight used before/at ramp start. Default 0.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_warmup_steps",
        type=int,
        default=None,
        help="Leg align schedule warmup steps that keep start_weight before the ramp begins.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_ramp_steps",
        type=int,
        default=None,
        help="Leg align schedule ramp length in optimizer steps. 0 means jump directly to target after warmup.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_oracle_min_deg",
        type=float,
        default=None,
        help="Oracle gate for leg omega alignment loss: only apply when ||omega_oracle|| >= this (deg).",
    )
    ap.add_argument(
        "--direct_pose_leg_align_oracle_weight_deg",
        type=float,
        default=None,
        help="Optional stop-grad weight ramp for leg omega alignment loss: w=clamp(||omega_oracle||/deg,0,1). 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_mode",
        type=str,
        default=None,
        choices=("cos", "proj"),
        help="Leg omega alignment loss form: cos (relu(-cos), cheatable) | proj (mag+res, non-cheating).",
    )
    ap.add_argument(
        "--direct_pose_leg_align_mag_weight",
        type=float,
        default=None,
        help="align_mode=proj: weight for projection magnitude term (proj-||oracle||)^2.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_res_weight",
        type=float,
        default=None,
        help="align_mode=proj: weight for orthogonal residual term ||res||^2.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_sign_weight",
        type=float,
        default=None,
        help="align_mode=proj: optional weight for relu(-proj)^2 sign penalty (rad^2).",
    )
    ap.add_argument(
        "--direct_pose_leg_align_cos_thresh",
        type=float,
        default=None,
        help="Optional hard-example mining: apply leg omega alignment only when cos(pred, oracle) < thresh. 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_target_joints",
        type=str,
        default=None,
        help="Optional joint subset for the main leg_align objective. Supports presets like 'distal', 'proximal', 'calf', 'thigh', 'foot', 'ball'.",
    )
    ap.add_argument(
        "--direct_pose_leg_align_anchor_joints",
        type=str,
        default=None,
        help="Optional auxiliary joint subset added on top of the main leg_align objective (e.g. 'calf').",
    )
    ap.add_argument(
        "--direct_pose_leg_align_anchor_weight",
        type=float,
        default=None,
        help="Relative weight for the auxiliary leg_align anchor subset. 0 disables the anchor.",
    )
    ap.add_argument(
        "--direct_pose_leg_side_routing",
        type=str,
        default=None,
        help="Retired in posttrain mainline; only inert default false is accepted.",
    )
    ap.add_argument(
        "--direct_pose_leg_contact_order",
        type=str,
        default=None,
        choices=("lr", "rl"),
        help="contacts/phase channel order for side routing: lr (ch0=left,ch1=right) | rl (ch0=right,ch1=left).",
    )
    ap.add_argument(
        "--direct_pose_leg_side_embed_dim",
        type=int,
        default=None,
        help="Optional tiny side embedding dim appended to the shared leg head input (0 disables).",
    )
    ap.add_argument(
        "--direct_pose_leg_side_sign_gate",
        type=str,
        default=None,
        help="Retired in posttrain mainline; only inert default false is accepted.",
    )
    ap.add_argument(
        "--direct_pose_leg_side_sign_gate_reg_weight",
        type=float,
        default=None,
        help="Retired in posttrain mainline; only inert default 0 is accepted.",
    )
    ap.add_argument(
        "--direct_pose_leg_side_rank1",
        type=str,
        default=None,
        help="Retired in posttrain mainline; only inert default false is accepted.",
    )
    ap.add_argument(
        "--direct_pose_loss_leg_split",
        type=str,
        default=None,
        help="Stage7 direct objective: true|false; split legs vs non-legs: L = mean(nonleg) + mean(leg).",
    )
    ap.add_argument(
        "--direct_pose_nonleg_focus_bones",
        type=str,
        default=None,
        help='Optional: comma-separated non-leg bones/indices to upweight inside L_nonleg (e.g. "upperarm_l,lowerarm_l,hand_l,pinky_01_l").',
    )
    ap.add_argument(
        "--direct_pose_nonleg_focus_weight",
        type=float,
        default=None,
        help="Only for --direct_pose_nonleg_focus_bones: multiplicative per-bone weight (>1 boosts selected bones; 1 disables).",
    )
    ap.add_argument(
        "--direct_pose_loss_sics",
        type=str,
        default=None,
        help="Retired in posttrain mainline; only inert default null/empty is accepted.",
    )
    ap.add_argument(
        "--direct_pose_loss_cycle_gte",
        type=int,
        default=None,
        help="Retired in posttrain mainline; only inert default 0 is accepted.",
    )
    ap.add_argument(
        "--direct_pose_loss_sic_mode",
        type=str,
        default=None,
        choices=("mask", "boost"),
        help="Retired in posttrain mainline; only inert default mask is accepted.",
    )
    ap.add_argument(
        "--direct_pose_loss_sic_boost",
        type=float,
        default=None,
        help="Retired in posttrain mainline; only inert default 1.0 is accepted.",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_enable",
        type=str,
        default=None,
        help="true|false; enable group-wise magnitude normalization for direct loss (leg vs non-leg).",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_w_leg",
        type=float,
        default=None,
        help="Weight for normalized leg group loss term.",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_w_nonleg",
        type=float,
        default=None,
        help="Weight for normalized non-leg group loss term.",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_ema_beta",
        type=float,
        default=None,
        help="EMA beta for group-wise magnitude normalization (no warmup switch; first batch initializes EMA).",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_ratio_min",
        type=float,
        default=None,
        help="Lower clamp for normalized group ratio L/EMA.",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_ratio_max",
        type=float,
        default=None,
        help="Upper clamp for normalized group ratio L/EMA.",
    )
    ap.add_argument(
        "--direct_pose_loss_group_norm_eps",
        type=float,
        default=None,
        help="Numerical epsilon for group-wise normalization denominator.",
    )
    ap.add_argument(
        "--direct_pose_grad_monitor_enable",
        type=str,
        default=None,
        help="true|false; log direct split-head grad norms (trunk/out_leg/out_nonleg).",
    )
    ap.add_argument(
        "--direct_pose_grad_ratio_gate",
        type=float,
        default=None,
        help="Alert threshold for grad_ratio = grad_nonleg / (grad_leg + eps).",
    )
    ap.add_argument(
        "--contact_meas_gate_by_hit",
        type=str,
        default=None,
        choices=("auto", "true", "false"),
        help="Override white-box gate_by_hit used by _contact_meas_whitebox: auto|true|false.",
    )
    ap.add_argument(
        "--contact_meas_vxy_mode",
        type=str,
        default=None,
        choices=("abs", "root_rel"),
        help="White-box vxy gate: abs uses ||v_foot_xy||, root_rel uses ||v_foot_xy - v_root_xy|| (more robust under translation).",
    )
    ap.add_argument(
        "--contact_meas_ground_z_mode",
        type=str,
        default=None,
        choices=("ema", "window", "slew"),
        help="White-box ground_z update mode: ema|window|slew.",
    )
    ap.add_argument("--contact_meas_ground_z_beta", type=float, default=None, help="EMA beta for ground_z when mode=ema.")
    ap.add_argument("--contact_meas_ground_z_window", type=int, default=None, help="Window length when mode=window.")
    ap.add_argument("--contact_meas_ground_z_quantile", type=float, default=None, help="Low-quantile (0..1) when mode=window.")
    ap.add_argument("--contact_meas_ground_z_slew_up_cm", type=float, default=None, help="Max upward change (cm/step) after ground_z update (0 disables).")
    ap.add_argument("--contact_meas_ground_z_slew_down_cm", type=float, default=None, help="Max downward change (cm/step) after ground_z update (0 disables).")
    ap.add_argument(
        "--posttrain_contacts_source",
        type=str,
        default=None,
        choices=("pretrain_contact",),
        help="Contacts source used during posttrain rollout (fixed): pretrain_contact.",
    )
    ap.add_argument(
        "--posttrain_contacts_pretrain_clamp",
        type=float,
        default=None,
        help="When posttrain_contacts_source=pretrain_contact, clamp frozen encoder input to [-k,+k].",
    )
    ap.add_argument(
        "--posttrain_contacts_pretrain_affine_stats",
        type=str,
        default=None,
        help="Optional affine stats JSON path or JSON string (scale/bias/eps) for pretrain contact calibration.",
    )
    ap.add_argument("--lambda_fusion_mode", type=str, default=None, help="global|per_joint")
    ap.add_argument("--lambda_fusion_hidden", type=int, default=None)
    ap.add_argument("--lambda_fusion_dropout", type=float, default=None)
    ap.add_argument("--lambda_fusion_logit_init", type=float, default=None)
    ap.add_argument("--lambda_fusion_use_rollout_step", type=str, default=None, help="true|false; concat rollout_step into lambda head input")
    ap.add_argument("--lambda_fusion_entropy_weight", type=float, default=None)
    ap.add_argument("--lambda_fusion_smooth_weight", type=float, default=None)
    ap.add_argument("--lambda_fusion_early_steps", type=int, default=None, help="Penalize lambda_mean for the first K rollout steps (protect early).")
    ap.add_argument("--lambda_fusion_early_weight", type=float, default=None, help="Weight for early-step lambda prior loss.")
    ap.add_argument("--lambda_fusion_monotonic_weight", type=float, default=None, help="Weight for soft monotonic loss: sum(ReLU(lambda[t-1]-lambda[t])).")
    ap.add_argument("--lambda_plan_entropy_weight", type=float, default=None, help="Penalty weight: lambda_mean * mean(H(contacts_plan)).")
    ap.add_argument("--lambda_plan_dyn_weight", type=float, default=None, help="Penalty weight: lambda_mean * mean(|contacts_plan[t]-contacts_plan[t-1]|).")
    ap.add_argument("--lambda_time_weight_mode", type=str, default=None, help="inv|linear|uniform (rollout step weights for lambda loss)")
    ap.add_argument("--lambda_time_weight_max", type=float, default=None)
    ap.add_argument(
        "--lambda_reliability_mode",
        type=str,
        default=None,
        help="none|warmup|contacts_err|warmup+contacts_err (deterministic r_t applied to λ for blend; shared in posttrain+freerun).",
    )
    ap.add_argument("--lambda_reliability_warmup_steps", type=int, default=None, help="Warmup steps K for r_t ramp 0->1 when mode includes warmup.")
    ap.add_argument("--lambda_reliability_contact_err_max", type=float, default=None, help="contacts_err_abs_mean scale for r_t=clamp(1-err/max,0,1) when mode includes contacts_err.")
    ap.add_argument(
        "--lambda_reliability_warmup_joint_scales",
        type=str,
        default=None,
        help="Optional per-joint warmup scales: JSON list (e.g. '[1,1,2,...]') or a JSON file path containing list/scales.",
    )
    ap.add_argument("--lambda_l2sp_weight", type=float, default=None, help="Optional L2-SP weight to keep trainable head params close to init (improves generalization).")
    ap.add_argument("--lambda_boundary_weight", type=float, default=None, help="Boundary loss weight multiplier when rollout_include_boundary=true (0 disables boundary supervision).")
    ap.add_argument("--lambda_gate_sup_weight", type=float, default=None, help="Stage2: gate supervision weight (BCE on lambda_fusion_logits vs oracle soft label). 0 disables.")
    ap.add_argument("--lambda_gate_sup_tau_deg", type=float, default=None, help="Stage2: τ (deg) for soft label: lambda*=sigmoid((err_inc-err_dir)/τ).")
    ap.add_argument("--lambda_gate_sup_margin_deg", type=float, default=None, help="Stage2: margin δ (deg); supervise only when |err_inc-err_dir|>=δ. Default is 1°. Use 0 to disable.")
    ap.add_argument("--lambda_gate_sup_start_step", type=int, default=None, help="Stage2: start rollout step for gate supervision. -1 auto uses lambda_reliability_warmup_steps when warmup enabled.")
    ap.add_argument("--seed", type=int, default=None)
    return ap


def _build_posttrain_model_from_ckpt(
    *,
    cfg: PostTrainConfig,
    ds: MotionEventDataset,
    device: torch.device,
) -> Tuple[EventMotionModel, str, int, float, bool, str, bool, int, str, float]:
    ckpt = torch.load(cfg.ckpt_in.expanduser(), map_location="cpu")
    pt = ckpt.get("posttrain_cfg", None) if isinstance(ckpt, dict) else None
    ckpt_posttrain_cfg: Optional[dict[str, Any]] = pt if isinstance(pt, dict) else None

    raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {}
    for k, v in raw_state.items():
        if k.startswith("frozen_encoder.") or k.startswith("frozen_period_head.") or k.startswith("contact_plan_input_proj."):
            continue
        state_dict[k] = v
    width = int(state_dict["shared_encoder.0.weight"].shape[0])
    period_dim = int(state_dict["period_encoder.weight"].shape[1]) if "period_encoder.weight" in state_dict else 0
    nin = int(state_dict["shared_encoder.0.weight"].shape[1])

    # ---- Infer plan/meas options from checkpoint (must match shared_encoder in_features) ----
    contact_dim = int(getattr(ds, "contact_dim", 0) or 0)
    angvel_dim = int(getattr(ds, "angvel_dim", 0) or 0)
    pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0) or 0)
    cond_dim = int(ds.Dc)
    base_in = int(ds.Dx) + int(ds.Dc)
    extra_in = int(max(0, nin - base_in))

    plan_has_weights = any(k.startswith("contact_plan_cell.") for k in state_dict.keys())
    plan_hidden = None
    if plan_has_weights and "contact_plan_cell.weight_ih" in state_dict:
        try:
            plan_hidden = int(state_dict["contact_plan_cell.weight_ih"].shape[0] // 3)
        except Exception:
            plan_hidden = None
    if plan_hidden is None:
        plan_hidden = int(extra_in) if extra_in > 0 else 64

    contact_plan_inject = "none"
    if extra_in == 0:
        contact_plan_inject = "none"
    elif contact_dim > 0 and extra_in == contact_dim:
        contact_plan_inject = "contacts"
    elif plan_hidden > 0 and extra_in == int(plan_hidden):
        contact_plan_inject = "plan_z"
    elif extra_in > 0 and plan_has_weights:
        # Fallback: assume plan_z injection with hidden==extra_in (should rarely happen).
        contact_plan_inject = "plan_z"
        plan_hidden = int(extra_in)

    # ---- Infer direct pose head (cond + contacts_plan -> absolute pose) ----
    # NOTE: If we don't instantiate this head, `load_state_dict(strict=False)` will silently drop its weights,
    # and the post-train output checkpoint would lose the direct branch. Keep it when present.
    direct_has_weights = bool(
        any(k.startswith("direct_pose_head.") for k in state_dict.keys())
        or any(k.startswith("direct_pose_out_leg.") for k in state_dict.keys())
        or any(k.startswith("direct_pose_out_nonleg.") for k in state_dict.keys())
        or any(k.startswith("direct_pose_out_arm.") for k in state_dict.keys())
        or any(k.startswith("direct_pose_out_else.") for k in state_dict.keys())
    )
    # Optional config overrides (useful when reinitializing the direct head).
    direct_pose_reinit = bool(getattr(cfg, "direct_pose_reinit", False))
    if direct_pose_reinit and (not bool(getattr(cfg, "train_direct_pose", False))):
        print("[posttrain][WARN] direct_pose_reinit=true but train_direct_pose=false; ignoring reinit.")
        direct_pose_reinit = False

    direct_pose_enable_infer = False
    direct_pose_hidden_infer = 256
    direct_pose_meas_mode_infer = "concat"
    direct_pose_feat_source_infer = "cond"
    direct_pose_time_pe_dim_infer = 0
    direct_pose_use_phase_z_infer = False
    direct_pose_phase_z_mode_infer = "concat"
    try:
        if isinstance(ckpt_posttrain_cfg, dict):
            direct_pose_use_phase_z_infer = bool(ckpt_posttrain_cfg.get("direct_pose_use_phase_z", False))
            v = ckpt_posttrain_cfg.get("direct_pose_phase_z_mode", None)
            if v is not None:
                direct_pose_phase_z_mode_infer = str(v).strip().lower() or "concat"
    except Exception:
        direct_pose_use_phase_z_infer = False
        direct_pose_phase_z_mode_infer = "concat"
    if direct_has_weights and contact_dim > 0 and (not direct_pose_reinit):
        w_in = state_dict.get("direct_pose_head.0.weight", None)
        w_out = state_dict.get("direct_pose_head.6.weight", None)
        w_out_leg = state_dict.get("direct_pose_out_leg.weight", None)
        w_out_nonleg = state_dict.get("direct_pose_out_nonleg.weight", None)
        w_out_arm = state_dict.get("direct_pose_out_arm.weight", None)
        w_out_else = state_dict.get("direct_pose_out_else.weight", None)
        try:
            if torch.is_tensor(w_in) and w_in.ndim == 2:
                in_dim = int(w_in.shape[1])
                hid = int(w_in.shape[0])
                out_dim = None
                if torch.is_tensor(w_out) and w_out.ndim == 2:
                    out_dim = int(w_out.shape[0])
                elif (
                    torch.is_tensor(w_out_leg)
                    and w_out_leg.ndim == 2
                    and torch.is_tensor(w_out_nonleg)
                    and w_out_nonleg.ndim == 2
                ):
                    out_dim = int(w_out_leg.shape[0] + w_out_nonleg.shape[0])
                    if int(w_out_leg.shape[1]) > 0:
                        hid = int(w_out_leg.shape[1])
                elif (
                    torch.is_tensor(w_out_leg)
                    and w_out_leg.ndim == 2
                    and torch.is_tensor(w_out_arm)
                    and w_out_arm.ndim == 2
                    and torch.is_tensor(w_out_else)
                    and w_out_else.ndim == 2
                ):
                    out_dim = int(w_out_leg.shape[0] + w_out_arm.shape[0] + w_out_else.shape[0])
                    if int(w_out_leg.shape[1]) > 0:
                        hid = int(w_out_leg.shape[1])
                if out_dim is None:
                    raise SystemExit("[FATAL] direct_pose_head weights found but output readout weights are missing.")
                expected_out = int(ds.Dy)
                expected_out_modes = int(ds.Dy) * 2
                base_candidates = [
                    (int(cond_dim), "cond"),
                    (int(width), "hidden"),
                    (int(cond_dim + width), "cond+hidden"),
                ]
                Cc = int(contact_dim)

                mode = None
                if out_dim == expected_out:
                    mode = "concat"
                    for base_dim, src in base_candidates:
                        phase_dim = int(2 * Cc) if bool(direct_pose_use_phase_z_infer) else 0
                        if str(direct_pose_phase_z_mode_infer or "concat").strip().lower() == "replace_contacts":
                            # input = base + time_pe + phase_z (no plan/meas)
                            tdim = int(in_dim - base_dim - phase_dim)
                        else:
                            # input = base + time_pe + plan + meas (+ phase_z)
                            tdim = int(in_dim - base_dim - (2 * Cc) - phase_dim)
                        if tdim >= 0 and tdim % 2 == 0:
                            direct_pose_enable_infer = True
                            direct_pose_hidden_infer = hid
                            direct_pose_meas_mode_infer = mode
                            direct_pose_feat_source_infer = src
                            direct_pose_time_pe_dim_infer = int(tdim)
                            break
                elif out_dim == expected_out_modes:
                    mode = "mode_select"
                    for base_dim, src in base_candidates:
                        phase_dim = int(2 * Cc) if bool(direct_pose_use_phase_z_infer) else 0
                        if str(direct_pose_phase_z_mode_infer or "concat").strip().lower() == "replace_contacts":
                            raise SystemExit(
                                "[FATAL] direct_pose_phase_z_mode='replace_contacts' is not supported for direct_pose_meas_mode='mode_select'."
                            )
                        tdim = int(in_dim - base_dim - Cc - phase_dim)
                        if tdim >= 0 and tdim % 2 == 0:
                            direct_pose_enable_infer = True
                            direct_pose_hidden_infer = hid
                            direct_pose_meas_mode_infer = mode
                            direct_pose_feat_source_infer = src
                            direct_pose_time_pe_dim_infer = int(tdim)
                            break
                else:
                    raise SystemExit(
                        f"[FATAL] Unrecognized direct_pose_head out_dim={out_dim} (expected {expected_out} or {expected_out_modes})."
                    )

                if not direct_pose_enable_infer:
                    raise SystemExit(
                        f"[FATAL] Unrecognized direct_pose_head shape: in_dim={in_dim} out_dim={out_dim} "
                        f"(cond_dim={cond_dim}, hidden_dim={width}, contact_dim={contact_dim})."
                    )
        except Exception:
            direct_pose_enable_infer = False

    # Resolve instantiation config (ckpt infer remains for compat direct-head layouts).
    # Split/non-leg-proj are now config-driven (no ckpt auto-override).
    direct_pose_enable = bool(direct_pose_enable_infer or direct_has_weights or bool(getattr(cfg, "train_direct_pose", False)) or direct_pose_reinit)
    direct_pose_hidden = int(getattr(cfg, "direct_pose_hidden_override", None) or direct_pose_hidden_infer)
    direct_pose_meas_mode = str(getattr(cfg, "direct_pose_meas_mode_override", None) or direct_pose_meas_mode_infer)
    direct_pose_feat_source = str(getattr(cfg, "direct_pose_feat_source", "auto") or "auto").lower().strip()
    direct_pose_time_pe_dim = int(getattr(cfg, "direct_pose_time_pe_dim", -1))
    direct_pose_time_pe_base = float(getattr(cfg, "direct_pose_time_pe_base", 10000.0) or 10000.0)
    direct_pose_use_phase_z = bool(getattr(cfg, "direct_pose_use_phase_z", False))
    direct_pose_phase_z_mode = str(getattr(cfg, "direct_pose_phase_z_mode", "concat") or "concat").strip().lower()
    direct_pose_split_enable_infer = False
    direct_pose_arm_split_enable_infer = False
    direct_pose_nonleg_proj_dim_infer = 0
    try:
        has_leg_out = any(str(k).startswith("direct_pose_out_leg.") for k in state_dict.keys())
        has_nonleg_out = any(str(k).startswith("direct_pose_out_nonleg.") for k in state_dict.keys())
        has_arm_out = any(str(k).startswith("direct_pose_out_arm.") for k in state_dict.keys())
        has_else_out = any(str(k).startswith("direct_pose_out_else.") for k in state_dict.keys())
        direct_pose_split_enable_infer = bool(
            has_leg_out and (has_nonleg_out or (has_arm_out and has_else_out))
        )
        direct_pose_arm_split_enable_infer = bool(has_leg_out and has_arm_out and has_else_out)
    except Exception:
        direct_pose_split_enable_infer = False
        direct_pose_arm_split_enable_infer = False
    try:
        if isinstance(ckpt_posttrain_cfg, dict):
            if "direct_pose_arm_split_enable" in ckpt_posttrain_cfg:
                direct_pose_arm_split_enable_infer = bool(ckpt_posttrain_cfg.get("direct_pose_arm_split_enable", False))
            if bool(direct_pose_arm_split_enable_infer):
                direct_pose_split_enable_infer = True
    except Exception:
        pass
    try:
        w_non = state_dict.get("direct_pose_out_nonleg.weight", None)
        w_proj = state_dict.get("direct_pose_nonleg_proj.0.weight", None)
        w_arm_proj = state_dict.get("direct_pose_arm_proj.0.weight", None)
        w_else_proj = state_dict.get("direct_pose_else_proj.0.weight", None)
        if torch.is_tensor(w_proj) and w_proj.ndim == 2 and int(w_proj.shape[0]) > 0:
            direct_pose_nonleg_proj_dim_infer = int(w_proj.shape[0])
        elif torch.is_tensor(w_arm_proj) and w_arm_proj.ndim == 2 and int(w_arm_proj.shape[0]) > 0:
            direct_pose_nonleg_proj_dim_infer = int(w_arm_proj.shape[0])
        elif torch.is_tensor(w_else_proj) and w_else_proj.ndim == 2 and int(w_else_proj.shape[0]) > 0:
            direct_pose_nonleg_proj_dim_infer = int(w_else_proj.shape[0])
        elif (
            torch.is_tensor(w_non)
            and w_non.ndim == 2
            and int(direct_pose_hidden_infer) > 0
            and int(w_non.shape[1]) > 0
            and int(w_non.shape[1]) != int(direct_pose_hidden_infer)
        ):
            direct_pose_nonleg_proj_dim_infer = int(w_non.shape[1])
    except Exception:
        pass
    direct_pose_split_enable = bool(getattr(cfg, "direct_pose_split_enable", False))
    direct_pose_arm_split_enable = bool(getattr(cfg, "direct_pose_arm_split_enable", False))
    if bool(direct_pose_arm_split_enable):
        direct_pose_split_enable = True
    direct_pose_arm_bones = getattr(cfg, "direct_pose_arm_bones", None)
    direct_pose_nonleg_proj_dim = int(getattr(cfg, "direct_pose_nonleg_proj_dim", 0) or 0)
    direct_pose_nonleg_proj_dim = max(0, int(direct_pose_nonleg_proj_dim))
    if direct_pose_phase_z_mode in ("", "auto"):
        direct_pose_phase_z_mode = str(direct_pose_phase_z_mode_infer or "concat").strip().lower() or "concat"
    if direct_pose_phase_z_mode in ("replace", "replace_contacts", "phase", "phase_only"):
        direct_pose_phase_z_mode = "replace_contacts"
    elif direct_pose_phase_z_mode in ("concat", "append", "add", "plus", "contacts+phase"):
        direct_pose_phase_z_mode = "concat"
    else:
        direct_pose_phase_z_mode = str(direct_pose_phase_z_mode_infer or "concat").strip().lower() or "concat"

    if direct_pose_meas_mode not in ("concat", "mode_select"):
        direct_pose_meas_mode = direct_pose_meas_mode_infer

    if direct_pose_feat_source == "auto":
        # Prefer checkpoint posttrain_cfg when present (cannot infer hidden_pre from tensor shapes).
        hint = None
        if isinstance(ckpt_posttrain_cfg, dict):
            hint = _normalize_direct_pose_feat_source(ckpt_posttrain_cfg.get("direct_pose_feat_source", None))
        direct_pose_feat_source = hint or (direct_pose_feat_source_infer if direct_pose_enable_infer else "cond")
    direct_pose_feat_source = _normalize_direct_pose_feat_source(direct_pose_feat_source) or "cond"
    if int(direct_pose_time_pe_dim) < 0:
        direct_pose_time_pe_dim = int(direct_pose_time_pe_dim_infer)
    if int(direct_pose_time_pe_dim) % 2 == 1:
        print(f"[posttrain][WARN] direct_pose_time_pe_dim={direct_pose_time_pe_dim} is odd; rounding up to even.")
        direct_pose_time_pe_dim = int(direct_pose_time_pe_dim) + 1

    # Decide whether to drop ckpt direct head weights (shape mismatch -> would cause load_state_dict error).
    drop_direct_pose_weights = False
    if direct_pose_reinit and direct_has_weights:
        drop_direct_pose_weights = True
    def _direct_pose_feat_shape_class(src: str) -> str:
        # hidden_pre is shape-compatible with hidden (both use hidden_dim).
        s = str(src or "cond").strip().lower()
        if s in ("hidden", "hidden_pre"):
            return "hidden"
        if s in ("cond+hidden", "cond+hidden_pre"):
            return "cond+hidden"
        return "cond"

    shape_override = bool(
        direct_has_weights
        and direct_pose_enable_infer
        and (
        direct_pose_hidden != int(direct_pose_hidden_infer)
        or direct_pose_meas_mode != str(direct_pose_meas_mode_infer)
        or _direct_pose_feat_shape_class(direct_pose_feat_source) != _direct_pose_feat_shape_class(str(direct_pose_feat_source_infer))
        or int(direct_pose_time_pe_dim) != int(direct_pose_time_pe_dim_infer)
        )
    )
    nonleg_proj_mismatch = bool(int(direct_pose_nonleg_proj_dim) != int(direct_pose_nonleg_proj_dim_infer))
    split_mismatch = bool(direct_pose_split_enable) != bool(direct_pose_split_enable_infer)
    arm_split_mismatch = bool(direct_pose_arm_split_enable) != bool(direct_pose_arm_split_enable_infer)
    if shape_override:
        if not bool(getattr(cfg, "train_direct_pose", False)):
            raise SystemExit(
                "[FATAL] direct_pose_* overrides change direct head tensor shapes, but train_direct_pose=false. "
                "Enable train_direct_pose (and optionally direct_pose_reinit=true) to reinitialize the head."
            )
        drop_direct_pose_weights = True
    if nonleg_proj_mismatch and (not bool(getattr(cfg, "train_direct_pose", False))):
        raise SystemExit(
            "[FATAL] direct_pose_nonleg_proj_dim differs from checkpoint but train_direct_pose=false. "
            "Enable train_direct_pose to adapt non-leg readout weights."
        )
    if split_mismatch:
        allow_compat_to_split = bool(direct_pose_split_enable) and (not bool(direct_pose_split_enable_infer))
        if not allow_compat_to_split:
            if not bool(getattr(cfg, "train_direct_pose", False)):
                raise SystemExit(
                    "[FATAL] direct_pose split mode differs from checkpoint but train_direct_pose=false. "
                    "Enable train_direct_pose (or match direct_pose_split_enable to checkpoint)."
                )
            drop_direct_pose_weights = True
    if arm_split_mismatch:
        allow_two_to_three = bool(direct_pose_arm_split_enable) and (not bool(direct_pose_arm_split_enable_infer))
        if not allow_two_to_three:
            if not bool(getattr(cfg, "train_direct_pose", False)):
                raise SystemExit(
                    "[FATAL] direct_pose arm-split mode differs from checkpoint but train_direct_pose=false. "
                    "Enable train_direct_pose (or match direct_pose_arm_split_enable to checkpoint)."
                )
            drop_direct_pose_weights = True

    contact_plan_enable = bool(
        plan_has_weights
        or direct_pose_enable
        or (extra_in > 0 and contact_dim > 0 and cond_dim > 0)
    )
    contact_plan_time_pe_dim = 0
    try:
        w_time = state_dict.get("contact_plan_time_head.weight", None)
        if torch.is_tensor(w_time) and w_time.ndim == 2:
            contact_plan_time_pe_dim = int(w_time.shape[1])
    except Exception:
        contact_plan_time_pe_dim = 0

    # ---- Infer obs-conditioned contact plan init head (plan_z0 = init_z + init_head(obs0)) ----
    init_has_weights = any(k.startswith("contact_plan_init_head.") for k in state_dict.keys())
    contact_plan_init_mode = str(getattr(cfg, "contact_plan_init_mode", "learnable") or "learnable")
    contact_plan_init_hidden = int(getattr(cfg, "contact_plan_init_hidden", 128) or 128)
    contact_plan_init_dropout = float(getattr(cfg, "contact_plan_init_dropout", 0.0) or 0.0)
    if init_has_weights:
        if str(contact_plan_init_mode).lower().strip() not in ("obs", "learnable+obs", "learnable_obs", "obs+learnable"):
            print("[posttrain][WARN] checkpoint has contact_plan_init_head weights; overriding contact_plan_init_mode -> learnable+obs.")
            contact_plan_init_mode = "learnable+obs"
        w_init = state_dict.get("contact_plan_init_head.1.weight", None)
        if torch.is_tensor(w_init) and w_init.ndim == 2:
            contact_plan_init_hidden = int(w_init.shape[0])

    # ---- Infer Event-Clock v3 (contact_plan residual correction) ----
    event_clock_has_weights = any(
        str(k).startswith("event_clock_gate.") or str(k).startswith("event_clock_corrector.")
        for k in state_dict.keys()
    )
    event_clock_mode = str(getattr(cfg, "event_clock", "auto") or "auto").strip().lower()
    use_event_clock = bool(event_clock_has_weights)
    if event_clock_mode == "on":
        if not event_clock_has_weights:
            print("[posttrain][WARN] --event_clock=on but ckpt has no event_clock_* weights; initializing Event-Clock randomly.")
        use_event_clock = True
    elif event_clock_mode == "off":
        if event_clock_has_weights:
            print("[posttrain][WARN] --event_clock=off will drop event_clock_* weights when saving the posttrain checkpoint.")
        use_event_clock = False

    event_clock_hidden_dim = 64
    event_clock_gate_hidden_dim = 32
    try:
        w_ec = state_dict.get("event_clock_corrector.correction_head.0.weight", None)
        if torch.is_tensor(w_ec) and w_ec.ndim == 2:
            event_clock_hidden_dim = int(w_ec.shape[0])
    except Exception:
        pass
    try:
        w_gate = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
        if torch.is_tensor(w_gate) and w_gate.ndim == 2:
            event_clock_gate_hidden_dim = int(w_gate.shape[0])
    except Exception:
        pass
    if getattr(cfg, "event_clock_hidden_dim", None) is not None:
        try:
            event_clock_hidden_dim = int(cfg.event_clock_hidden_dim)
        except Exception:
            pass
    if getattr(cfg, "event_clock_gate_hidden_dim", None) is not None:
        try:
            event_clock_gate_hidden_dim = int(cfg.event_clock_gate_hidden_dim)
        except Exception:
            pass
    event_clock_max_delta = float(getattr(cfg, "event_clock_max_delta", 0.5) or 0.5)

    # Important: EventMotionModel's period_dim can be mutated later by attach_motion_encoder().
    # Some checkpoints are trained with period_dim=0 at init (so Event-Clock ignores period_feat),
    # then period_dim is set by the attached encoder bundle (creating period_encoder weights).
    # To faithfully reconstruct such ckpts, we infer the Event-Clock period_feat_dim from its weight shapes
    # and use it as the model init period_dim (then attach encoder bundle BEFORE loading weights).
    event_clock_period_feat_dim = None
    try:
        w0 = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
        if torch.is_tensor(w0) and w0.ndim == 2:
            base = int(contact_dim) * 2 + 1
            event_clock_period_feat_dim = max(0, int(w0.shape[1]) - base)
    except Exception:
        event_clock_period_feat_dim = None
    period_dim_init = int(period_dim)
    try:
        if (
            bool(event_clock_has_weights)
            and event_clock_period_feat_dim is not None
            and int(event_clock_period_feat_dim) != int(period_dim)
        ):
            period_dim_init = int(event_clock_period_feat_dim)
    except Exception:
        period_dim_init = int(period_dim)
    if period_dim_init != int(period_dim) and bool(event_clock_has_weights):
        if cfg.encoder_bundle is None or (not cfg.encoder_bundle.expanduser().is_file()):
            print(
                f"[posttrain][WARN] ckpt period_dim={int(period_dim)} but Event-Clock was initialized with period_feat_dim={int(period_dim_init)}; "
                "no encoder_bundle provided so period_encoder weights may be dropped. "
                "Pass --encoder_bundle to fully reconstruct the model."
            )
        else:
            print(
                f"[posttrain][INFO] ckpt period_dim={int(period_dim)} but Event-Clock period_feat_dim={int(period_dim_init)}; "
                "initializing model with Event-Clock-compatible period_dim then attaching encoder bundle before loading weights."
            )

    # ---- Infer lambda fusion head (Stage2) ----
    lambda_has_weights = any(k.startswith("lambda_fusion_head.") for k in state_dict.keys())
    lambda_fusion_enable = bool(cfg.train_lambda_head or lambda_has_weights)
    lambda_fusion_mode = str(getattr(cfg, "lambda_fusion_mode", "per_joint") or "per_joint")
    lambda_fusion_hidden = int(getattr(cfg, "lambda_fusion_hidden", 128) or 128)
    lambda_fusion_dropout = float(getattr(cfg, "lambda_fusion_dropout", 0.0) or 0.0)
    lambda_fusion_logit_init = float(getattr(cfg, "lambda_fusion_logit_init", -2.0) or -2.0)
    lambda_fusion_use_rollout_step_cfg = bool(getattr(cfg, "lambda_fusion_use_rollout_step", False))
    lambda_fusion_use_rollout_step = bool(lambda_fusion_use_rollout_step_cfg)
    if lambda_has_weights:
        # Must match checkpoint shapes to avoid load_state_dict size mismatch.
        w_in = state_dict.get("lambda_fusion_head.1.weight", None)
        w_out = state_dict.get("lambda_fusion_head.4.weight", None)
        try:
            if torch.is_tensor(w_in) and w_in.ndim == 2:
                lambda_fusion_hidden = int(w_in.shape[0])
                base_in = int(width + (contact_dim if contact_plan_enable else 0))
                in_features = int(w_in.shape[1])
                inferred = None
                if in_features == base_in + 1:
                    inferred = True
                elif in_features == base_in:
                    inferred = False
                if inferred is not None and inferred != lambda_fusion_use_rollout_step_cfg:
                    print(
                        f"[posttrain][WARN] lambda_fusion_use_rollout_step={lambda_fusion_use_rollout_step_cfg} "
                        f"but ckpt expects {in_features} in_features (base={base_in}); overriding to {inferred}."
                    )
                if inferred is not None:
                    lambda_fusion_use_rollout_step = bool(inferred)
            if torch.is_tensor(w_out) and w_out.ndim == 2:
                out_dim = int(w_out.shape[0])
                lambda_fusion_mode = "global" if out_dim == 1 else "per_joint"
        except Exception:
            pass

    # ---- Infer contact phase state (prev_phase_vec) from checkpoint ----
    phase_state_enable = False
    phase_state_hidden = int(getattr(cfg, "contact_phase_state_hidden", 64) or 64)
    try:
        phase_state_enable = any(
            k == "contact_phase_state_init"
            or k.startswith("contact_phase_state_delta_head.")
            for k in state_dict.keys()
        )
        w_h = state_dict.get("contact_phase_state_delta_head.1.weight", None)
        if torch.is_tensor(w_h) and w_h.ndim == 2 and int(w_h.shape[0]) > 0:
            phase_state_hidden = int(w_h.shape[0])
        w_out = state_dict.get("contact_phase_state_delta_head.3.weight", None)
        if torch.is_tensor(w_out) and w_out.ndim == 2 and int(w_out.shape[1]) > 0:
            phase_state_hidden = int(w_out.shape[1])
    except Exception:
        phase_state_enable = False

    phase_reset_source_model = str(cfg.phase_reset_source or "none").strip().lower()
    # Consistent with run_freerun_cycles:
    # - contacts_meas: internal threshold-crossing resets inside the model (event_kind controls it)
    # - ttc_gt: resets are applied externally (posttrain rollout loops), so disable internal resets
    phase_event_kind_model = str(getattr(cfg, "contact_phase_state_event_kind", "touchdown") or "touchdown").strip().lower()
    phase_min_interval_model = int(getattr(cfg, "contact_phase_state_event_min_interval", 0) or 0)
    if phase_reset_source_model == "ttc_gt":
        phase_event_kind_model = "none"
        phase_min_interval_model = 0

    # ---- Resolve leg gate config from explicit runtime config (no ckpt auto-infer) ----
    direct_pose_leg_gate_mode_raw = str(getattr(cfg, "direct_pose_leg_gate_mode", "none") or "none").strip().lower()
    if direct_pose_leg_gate_mode_raw == "auto":
        print(
            "[posttrain][WARN] direct_pose_leg_gate_mode='auto' is deprecated and no longer inferred from checkpoint; "
            "using 'none'. Set explicit 'learned' or 'scale' when needed."
        )
    direct_pose_leg_gate_mode_model = _DIRECT_POSE_LEG_GATE_ALIAS_MAP.get(direct_pose_leg_gate_mode_raw, direct_pose_leg_gate_mode_raw)
    if direct_pose_leg_gate_mode_model not in _DIRECT_POSE_LEG_GATE_CHOICES:
        direct_pose_leg_gate_mode_model = "none"
        print(
            f"[posttrain][WARN] unrecognized direct_pose_leg_gate_mode={direct_pose_leg_gate_mode_raw!r}; using 'none'. "
            "Set explicit 'none'/'learned'/'scale'."
        )
    else:
        direct_pose_leg_gate_mode_model = str(direct_pose_leg_gate_mode_model)
    try:
        direct_pose_leg_gate_power_model = float(getattr(cfg, "direct_pose_leg_gate_power", 1.0) or 1.0)
    except Exception:
        direct_pose_leg_gate_power_model = 1.0
    if (not math.isfinite(direct_pose_leg_gate_power_model)) or direct_pose_leg_gate_power_model <= 0.0:
        direct_pose_leg_gate_power_model = 1.0

    model = EventMotionModel(
        in_state_dim=int(ds.Dx),
        out_motion_dim=int(ds.Dy),
        cond_dim=int(ds.Dc),
        period_dim=int(period_dim_init),
        hidden_dim=width,
        num_layers=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        dropout=float(cfg.dropout),
        context_len=int(cfg.context_len),
        contact_dim=contact_dim,
        angvel_dim=angvel_dim,
        pose_hist_dim=pose_hist_dim,
        state_layout=getattr(ds, "state_layout", None),
        bone_names=getattr(ds, "bone_names", None),
        output_layout=getattr(ds, "output_layout", None),
        contact_plan_enable=contact_plan_enable,
        contact_plan_hidden=int(plan_hidden or 64),
        contact_plan_dropout=0.0,
        contact_plan_inject=str(contact_plan_inject),
        contact_plan_inject_detach=True,
        contact_plan_time_pe_dim=int(contact_plan_time_pe_dim),
        contact_plan_init_mode=str(contact_plan_init_mode),
        contact_plan_init_hidden=int(contact_plan_init_hidden),
        contact_plan_init_dropout=float(contact_plan_init_dropout),
        contact_phase_state_enable=bool(phase_state_enable),
        contact_phase_state_init_mode=str(getattr(cfg, "contact_phase_state_init_mode", "obs") or "obs"),
        contact_phase_state_hidden=int(phase_state_hidden),
        contact_phase_state_delta_max=float(getattr(cfg, "contact_phase_state_delta_max", 0.5) or 0.5),
        contact_phase_state_delta_init=float(getattr(cfg, "contact_phase_state_delta_init", (6.283185307179586 / 80.0)) or (6.283185307179586 / 80.0)),
        contact_phase_state_event_kind=str(phase_event_kind_model),
        contact_phase_state_event_thr=float(getattr(cfg, "contact_phase_state_event_thr", 0.5) or 0.5),
        contact_phase_state_event_hyst=float(getattr(cfg, "contact_phase_state_event_hyst", 0.0) or 0.0),
        contact_phase_state_event_min_interval=int(phase_min_interval_model),
        phase_reset_source=str(phase_reset_source_model),
        use_event_clock=bool(use_event_clock),
        event_clock_max_delta=float(event_clock_max_delta),
        event_clock_hidden_dim=int(event_clock_hidden_dim),
        event_clock_gate_hidden_dim=int(event_clock_gate_hidden_dim),
        direct_pose_enable=bool(direct_pose_enable),
        direct_pose_hidden=int(direct_pose_hidden),
        direct_pose_dropout=0.0,
        direct_pose_detach_plan=True,
        direct_pose_meas_mode=str(direct_pose_meas_mode),
        direct_pose_meas_drop_prob=0.0,
        direct_pose_meas_noise_std=0.0,
        direct_pose_plan_drop_prob=0.0,
        direct_pose_feat_source=str(direct_pose_feat_source),
        direct_pose_time_pe_dim=int(direct_pose_time_pe_dim),
        direct_pose_time_pe_base=float(direct_pose_time_pe_base),
        direct_pose_use_phase_z=bool(direct_pose_use_phase_z),
        direct_pose_phase_z_mode=str(direct_pose_phase_z_mode),
        direct_pose_split_enable=bool(direct_pose_split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_nonleg_proj_dim),
        direct_pose_arm_split_enable=bool(getattr(cfg, "direct_pose_arm_split_enable", False)),
        direct_pose_arm_bones=getattr(cfg, "direct_pose_arm_bones", None),
        direct_pose_leg_enable=bool(getattr(cfg, "direct_pose_leg_enable", False)),
        direct_pose_leg_bones=getattr(cfg, "direct_pose_leg_bones", None),
        direct_pose_leg_mode=str(getattr(cfg, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add"),
        direct_pose_leg_stopgrad_main=bool(getattr(cfg, "direct_pose_leg_stopgrad_main", False)),
        direct_pose_leg_detach_feat=bool(getattr(cfg, "direct_pose_leg_detach_feat", False)),
        direct_pose_leg_max_deg=float(getattr(cfg, "direct_pose_leg_max_deg", 0.0) or 0.0),
        direct_pose_leg_gate_mode=str(direct_pose_leg_gate_mode_model),
        direct_pose_leg_gate_power=float(direct_pose_leg_gate_power_model),
        direct_pose_leg_scale_log_clip=float(getattr(cfg, "direct_pose_leg_scale_log_clip", 4.0) or 4.0),
        direct_pose_leg_scale_clamp_k=float(getattr(cfg, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0),
        lambda_fusion_enable=bool(lambda_fusion_enable),
        lambda_fusion_mode=str(lambda_fusion_mode),
        lambda_fusion_hidden=int(lambda_fusion_hidden),
        lambda_fusion_dropout=float(lambda_fusion_dropout),
        lambda_fusion_detach_err=True,
        lambda_fusion_logit_init=float(lambda_fusion_logit_init),
        lambda_fusion_use_rollout_step=bool(lambda_fusion_use_rollout_step),
    ).to(device)
    validate_and_fix_model_(model, int(ds.Dx), int(ds.Dc))
    # Attach frozen encoder BEFORE loading weights (period_dim/period_encoder may be created here).
    if cfg.encoder_bundle is not None and cfg.encoder_bundle.expanduser().is_file():
        model.attach_motion_encoder(torch.load(str(cfg.encoder_bundle.expanduser()), map_location="cpu"))

    if 'drop_direct_pose_weights' in locals() and bool(drop_direct_pose_weights):
        removed = [
            k
            for k in list(state_dict.keys())
            if str(k).startswith("direct_pose_head.")
            or str(k).startswith("direct_pose_out_leg.")
            or str(k).startswith("direct_pose_out_nonleg.")
            or str(k).startswith("direct_pose_out_arm.")
            or str(k).startswith("direct_pose_out_else.")
            or str(k).startswith("direct_pose_leg_head.")
            or str(k).startswith("direct_pose_leg_head_shared.")
            or str(k).startswith("direct_pose_arm_proj.")
            or str(k).startswith("direct_pose_else_proj.")
            or str(k).startswith("direct_pose_leg_gate_head.")
            or str(k).startswith("direct_pose_leg_gate_head_shared.")
            or str(k).startswith("direct_pose_leg_side_sign_gate_head.")
            or str(k).startswith("direct_pose_leg_side_embed.")
            or str(k) == "direct_pose_leg_joint_idx_tensor"
            or str(k) in ("direct_pose_leg_side_pos_r_tensor", "direct_pose_leg_side_pos_l_tensor")
            or str(k) in ("direct_pose_leg_out_idx", "direct_pose_nonleg_out_idx", "direct_pose_arm_out_idx", "direct_pose_else_out_idx")
        ]
        for k in removed:
            state_dict.pop(k, None)
        if removed:
            print(
                f"[posttrain][INFO] dropped {len(removed)} direct_pose_* tensors from checkpoint (reinit/override)."
            )

    # If we enable phase_z_in conditioning, older checkpoints may have a larger/smaller input dim
    # on first-layer weights. Adapt these tensors in-place to preserve warm-start behavior.
    #
    # Supported mappings:
    # - concat mode:   [base(+time), plan+meas] -> [base(+time), plan+meas, phase]
    # - replace mode:  [base(+time), plan+meas, phase] -> [base(+time), phase]
    #
    # We apply this to the direct trunk and leg/gate first-layer heads so skip-stage handoffs
    # (e.g., 70b->70R) don't silently drop leg weights due to shape mismatch.
    try:
        if (
            (not bool(locals().get("drop_direct_pose_weights", False)))
            and bool(locals().get("direct_pose_use_phase_z", False))
            and any(k.startswith("direct_pose_head.") for k in state_dict.keys())
        ):
            model_sd = model.state_dict()
            phase_mode = str(locals().get("direct_pose_phase_z_mode", "concat") or "concat").strip().lower()
            phase_dim = int(2 * int(contact_dim))

            def _adapt_phase_weight_tensor_(key: str) -> str:
                w0 = state_dict.get(key, None)
                w0_exp = model_sd.get(key, None)
                if not (torch.is_tensor(w0) and torch.is_tensor(w0_exp) and w0.ndim == 2 and w0_exp.ndim == 2):
                    return "skip"
                old_in = int(w0.shape[1])
                new_in = int(w0_exp.shape[1])
                if old_in == new_in:
                    return "skip"
                if int(w0.shape[0]) != int(w0_exp.shape[0]):
                    return "mismatch"
                if (old_in + phase_dim) == new_in:
                    new_w = torch.zeros((int(w0.shape[0]), int(new_in)), device=w0.device, dtype=w0.dtype)
                    new_w[:, :old_in] = w0
                    state_dict[key] = new_w
                    print(
                        f"[posttrain][INFO] expanded {key} in_dim {old_in} -> {new_in} "
                        f"(appended phase_z_in dim={phase_dim} as zeros)."
                    )
                    return "ok"
                if (
                    phase_mode == "replace_contacts"
                    and (old_in == (new_in + phase_dim))
                    and int(new_in) >= int(phase_dim)
                ):
                    # ckpt: [base(+time), plan+meas, phase] -> model(replace): [base(+time), phase]
                    base_in = int(new_in - phase_dim)
                    new_w = torch.zeros((int(w0.shape[0]), int(new_in)), device=w0.device, dtype=w0.dtype)
                    new_w[:, :base_in] = w0[:, :base_in]
                    new_w[:, base_in:] = w0[:, (old_in - phase_dim) :]
                    state_dict[key] = new_w
                    print(
                        f"[posttrain][INFO] adapted {key} for phase replace: in_dim {old_in} -> {new_in} "
                        f"(dropped plan+meas, kept phase tail dim={phase_dim})."
                    )
                    return "ok"
                return "mismatch"

            status_head = _adapt_phase_weight_tensor_("direct_pose_head.0.weight")
            phase_keys = (
                "direct_pose_leg_head.0.weight",
                "direct_pose_leg_head_shared.0.weight",
                "direct_pose_leg_gate_head.0.weight",
                "direct_pose_leg_gate_head_shared.0.weight",
            )
            for k in phase_keys:
                _adapt_phase_weight_tensor_(k)

            if status_head == "mismatch":
                w0 = state_dict.get("direct_pose_head.0.weight", None)
                w0_exp = model_sd.get("direct_pose_head.0.weight", None)
                old_in = int(w0.shape[1]) if torch.is_tensor(w0) and w0.ndim == 2 else -1
                new_in = int(w0_exp.shape[1]) if torch.is_tensor(w0_exp) and w0_exp.ndim == 2 else -1
                if bool(getattr(cfg, "train_direct_pose", False)):
                    # Fallback: drop and reinit trunk direct head if we're training direct.
                    removed = [k for k in list(state_dict.keys()) if str(k).startswith("direct_pose_head.")]
                    for k in removed:
                        state_dict.pop(k, None)
                    print(
                        f"[posttrain][WARN] direct_pose_use_phase_z=true but cannot adapt direct_pose_head shape "
                        f"(ckpt_in_dim={old_in}, model_in_dim={new_in}, phase_dim={phase_dim}); "
                        f"dropped {len(removed)} direct_pose_head.* tensors (will reinit)."
                    )
                else:
                    raise SystemExit(
                        f"[FATAL] direct_pose_use_phase_z=true but direct_pose_head.0.weight shape mismatch "
                        f"(ckpt_in_dim={old_in}, model_in_dim={new_in}). Enable train_direct_pose to reinit/adapt."
                    )
    except Exception:
        pass
    try:
        retired_direct_pose_prefixes = (
            "direct_pose_leg_head_shared.",
            "direct_pose_leg_gate_head_shared.",
            "direct_pose_leg_side_sign_gate_head.",
            "direct_pose_leg_side_embed.",
        )
        removed_highorder = []
        for k in list(state_dict.keys()):
            if any(str(k).startswith(prefix) for prefix in retired_direct_pose_prefixes):
                removed_highorder.append(str(k))
                state_dict.pop(k, None)
        for k in ("direct_pose_leg_side_pos_r_tensor", "direct_pose_leg_side_pos_l_tensor"):
            if k in state_dict:
                removed_highorder.append(str(k))
                state_dict.pop(k, None)
        if removed_highorder:
            print(
                f"[posttrain][INFO] dropped {len(removed_highorder)} retired direct_pose high-order ckpt tensor(s) "
                "(side-routing/sign-gate/rank1 compat shell)."
            )
    except Exception:
        pass
    # If leg bones/mode are overridden relative to the checkpoint, drop leg head tensors (and idx buffer)
    # to avoid shape mismatch or wrong joint mapping.
    try:
        def _norm_bones(v: Any) -> List[str]:
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                items = [str(x).strip() for x in v]
            else:
                items = [s.strip() for s in str(v).split(",") if s.strip()]
            return [x for x in items if x]

        leg_prefixes = (
            "direct_pose_leg_head.",
            "direct_pose_leg_head_shared.",
            "direct_pose_leg_side_sign_gate_head.",
            "direct_pose_leg_side_embed.",
        )
        # If user overrides bones, avoid silently loading wrong mapping even when shapes match.
        ckpt_leg_bones = []
        if isinstance(ckpt_posttrain_cfg, dict):
            ckpt_leg_bones = _norm_bones(ckpt_posttrain_cfg.get("direct_pose_leg_bones", None))
        tgt_leg_bones = _norm_bones(getattr(cfg, "direct_pose_leg_bones", None))

        removed = []
        if bool(getattr(cfg, "direct_pose_leg_enable", False)) and ckpt_leg_bones and tgt_leg_bones and (tgt_leg_bones != ckpt_leg_bones):
            for k in list(state_dict.keys()):
                if (
                    any(str(k).startswith(p) for p in leg_prefixes)
                    or str(k) == "direct_pose_leg_joint_idx_tensor"
                    or str(k) in ("direct_pose_leg_side_pos_r_tensor", "direct_pose_leg_side_pos_l_tensor")
                ):
                    removed.append(str(k))
                    state_dict.pop(k, None)
            if removed:
                print(
                    f"[posttrain][INFO] direct_pose_leg_bones override: ckpt={ckpt_leg_bones} cfg={tgt_leg_bones}; "
                    f"dropped {len(removed)} direct_pose_leg_* tensors (will re-init leg head / idx)."
                )

        # Drop any leg tensors whose shapes don't match the instantiated model (e.g. mode rot6d_add<->so3).
        model_sd = model.state_dict()
        removed_shape = []
        for k in list(state_dict.keys()):
            if not any(str(k).startswith(p) for p in leg_prefixes):
                continue
            v = state_dict.get(k, None)
            vv = model_sd.get(k, None)
            if torch.is_tensor(v) and torch.is_tensor(vv) and tuple(v.shape) != tuple(vv.shape):
                # Backward-compatible warm-start: allow adding a small number of new input columns to the
                # routed shared leg head (e.g., extra per-side cue). We pad/truncate the first layer weight
                # and keep other layers intact.
                try:
                    if (
                        str(k).endswith("direct_pose_leg_head_shared.0.weight")
                        and v.ndim == 2
                        and vv.ndim == 2
                        and int(v.shape[0]) == int(vv.shape[0])
                    ):
                        in_old = int(v.shape[1])
                        in_new = int(vv.shape[1])
                        if in_old < in_new and (in_new - in_old) <= 8:
                            pad = int(in_new - in_old)
                            state_dict[k] = torch.cat([v, v.new_zeros((int(v.shape[0]), pad))], dim=1)
                            continue
                        if in_old > in_new:
                            state_dict[k] = v[:, :in_new].contiguous()
                            continue
                except Exception:
                    pass
                removed_shape.append(str(k))
                state_dict.pop(k, None)
        if removed_shape:
            # Also drop idx buffer if we changed leg head shape; it often implies a mode/bones change.
            state_dict.pop("direct_pose_leg_joint_idx_tensor", None)
            state_dict.pop("direct_pose_leg_side_pos_r_tensor", None)
            state_dict.pop("direct_pose_leg_side_pos_l_tensor", None)
            print(
                f"[posttrain][INFO] dropped {len(removed_shape)} direct_pose_leg_head.* tensors due to shape mismatch "
                "(likely leg_mode or bone set changed)."
            )
    except Exception:
        pass
    model.load_state_dict(state_dict, strict=False)

    if cfg.train_direct_pose:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] direct_pose_head is not instantiated; cannot train direct pose expert.")
        # Sanity: avoid ambiguous "train_only" combinations.
        leg_only = bool(getattr(cfg, "direct_pose_leg_train_only", False))
        leg_gate_only = bool(getattr(cfg, "direct_pose_leg_gate_train_only", False))
        nonleg_only = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
        if nonleg_only and (leg_only or leg_gate_only):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true is incompatible with leg train_only modes. "
                "Pick exactly one train_only mode."
            )
        if (leg_only or leg_gate_only) and getattr(model, "direct_pose_leg_head", None) is None:
            raise SystemExit(
                "[FATAL] direct_pose_leg_*_train_only=true but no leg head is instantiated. "
                "Enable direct_pose_leg_enable and provide valid direct_pose_leg_bones."
            )
        has_nonleg_branch = (
            getattr(model, "direct_pose_out_nonleg", None) is not None
            or (
                getattr(model, "direct_pose_out_arm", None) is not None
                and getattr(model, "direct_pose_out_else", None) is not None
            )
        )
        if nonleg_only and (not has_nonleg_branch):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true but no non-leg branch is instantiated. "
                "Enable direct_pose_split_enable (optionally with direct_pose_arm_split_enable)."
            )
        if bool(leg_gate_only):
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_train_only=true but no leg gate/scale head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned'/'scale' and enable direct_pose_leg_enable with valid bones."
                )
        if float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0) > 0.0:
            leg_mode = str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").strip().lower()
            if leg_mode != "so3":
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 requires direct_pose_leg_mode='so3' "
                    f"(got {leg_mode!r})."
                )
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 but no learned leg gate head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned' and enable direct_pose_leg_enable with valid bones."
                )
    if cfg.train_lambda_head:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs direct_pose_head (out_direct), but checkpoint/model does not enable it.")
        if getattr(model, "lambda_fusion_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs lambda_fusion_head, but it is not instantiated.")

    # Optional: reset gate logit after loading ckpt (helps avoid near-zero gates that throttle gradients).
    if cfg.so3_corr_gate_logit_reset is not None:
        logit = getattr(model, "so3_corr_gate_logit", None)
        if torch.is_tensor(logit):
            with torch.no_grad():
                logit.fill_(float(cfg.so3_corr_gate_logit_reset))
            print(f"[posttrain] reset so3_corr_gate_logit={float(cfg.so3_corr_gate_logit_reset):.4f}")

    return (
        model,
        str(direct_pose_feat_source),
        int(direct_pose_time_pe_dim),
        float(direct_pose_time_pe_base),
        bool(direct_pose_use_phase_z),
        str(direct_pose_phase_z_mode),
        bool(direct_pose_split_enable),
        int(direct_pose_nonleg_proj_dim),
        str(direct_pose_leg_gate_mode_model),
        float(direct_pose_leg_gate_power_model),
    )


def main() -> None:
    ap = _build_posttrain_arg_parser()
    args = ap.parse_args()

    base_cfg = load_json(Path(args.config).expanduser()) if args.config else {}
    payload: Dict[str, Any] = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    _apply_cli_overrides(payload, args)

    cfg = _cfg_from_payload(payload)
    _set_seed(cfg.seed)

    if bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)) and (not bool(getattr(cfg, "direct_pose_loss_leg_split", False))):
        print("[posttrain][WARN] direct_pose_loss_group_norm_enable=true but direct_pose_loss_leg_split=false; group norm will have no effect.")

    device = _resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    norm_spec, ds, batch_iter = _build_dataset_and_loader(cfg)

    model, direct_pose_feat_source, direct_pose_time_pe_dim, direct_pose_time_pe_base, direct_pose_use_phase_z, direct_pose_phase_z_mode, direct_pose_split_enable, direct_pose_nonleg_proj_dim, direct_pose_leg_gate_mode_model, direct_pose_leg_gate_power_model = _build_posttrain_model_from_ckpt(
        cfg=cfg,
        ds=ds,
        device=device,
    )

    contacts_source_cfg = str(getattr(cfg, "posttrain_contacts_source", "pretrain_contact") or "pretrain_contact").strip().lower()
    if contacts_source_cfg != "pretrain_contact":
        raise SystemExit(
            f"[FATAL] unsupported posttrain_contacts_source={contacts_source_cfg!r}; only 'pretrain_contact' is allowed."
        )
    if bool(getattr(model, "contact_plan_enable", False)):
        if cfg.encoder_bundle is None or (not cfg.encoder_bundle.expanduser().is_file()):
            raise SystemExit(
                "[FATAL] posttrain_contacts_source=pretrain_contact requires --encoder_bundle with frozen encoder/contact_head."
            )
        if getattr(model, "frozen_encoder", None) is None or getattr(model, "frozen_contact_head", None) is None:
            raise SystemExit(
                "[FATAL] posttrain_contacts_source=pretrain_contact requires bundle keys 'encoder' and 'contact_head'."
            )

    trainer = _build_model_and_trainer(cfg=cfg, ds=ds, model=model, norm_spec=norm_spec)

    train_mode = _resolve_train_mode(cfg)
    print(f"[posttrain] mode={_train_mode_display_name(train_mode)}")

    _freeze_all(model)
    _unfreeze_for_train_mode(model, cfg, train_mode)
    model.train()

    params, names = _select_trainable_params(model)
    if not params:
        raise SystemExit("[FATAL] No trainable parameters selected for post-train.")
    print(f"[posttrain] trainable={len(params)} params: {', '.join(names[:8])}{' ...' if len(names)>8 else ''}")
    expected_prefixes = _expected_trainable_prefixes(train_mode)
    if expected_prefixes:
        unexpected = [n for n in names if not any(n.startswith(p) for p in expected_prefixes)]
        if unexpected:
            print(f"[posttrain][WARN] unexpected trainable params (prefixes={expected_prefixes}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    # Optional: L2-SP regularization to keep the trained head close to its initialization.
    # This is helpful when posttrain is used as a small "calibration" step and we want to
    # avoid harming long-horizon generalization by over-updating λ.
    l2sp_pairs: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    l2sp_weight = float(getattr(cfg, "lambda_l2sp_weight", 0.0) or 0.0)
    if l2sp_weight > 0.0:
        try:
            for p in params:
                l2sp_pairs.append((p, p.detach().clone()))
            print(f"[posttrain] lambda_l2sp_weight={l2sp_weight:g} (anchor_tensors={len(l2sp_pairs)})")
        except Exception:
            l2sp_pairs = []

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
    rollout_mode_kwargs = _build_rollout_mode_kwargs(cfg, train_mode)

    log_rows = _run_training_loop(cfg=cfg, train_mode=train_mode, model=model, params=params, opt=opt, batch_iter=batch_iter, rollout_common_kwargs=rollout_common_kwargs, rollout_mode_kwargs=rollout_mode_kwargs, l2sp_pairs=l2sp_pairs, l2sp_weight=l2sp_weight)
    ckpt_out = _save_posttrain_outputs(cfg=cfg, model=model, log_rows=log_rows, direct_pose_feat_source=direct_pose_feat_source, direct_pose_time_pe_dim=direct_pose_time_pe_dim, direct_pose_time_pe_base=direct_pose_time_pe_base, direct_pose_use_phase_z=direct_pose_use_phase_z, direct_pose_phase_z_mode=direct_pose_phase_z_mode, direct_pose_split_enable=direct_pose_split_enable, direct_pose_nonleg_proj_dim=direct_pose_nonleg_proj_dim, direct_pose_leg_gate_mode_model=direct_pose_leg_gate_mode_model, direct_pose_leg_gate_power_model=direct_pose_leg_gate_power_model)
    print(f"[posttrain][OK] saved: {ckpt_out}")


if __name__ == "__main__":
    main()
