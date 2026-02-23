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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from train.layout import DataNormalizer, parse_layout_entry
from train.models import EventMotionModel, MotionJointLoss
from train.training_MPL import Trainer, validate_and_fix_model_


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
    # - sliding    : enumerate all windows (legacy; can heavily overweight long clips)
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
    # - contacts_meas: threshold-crossing event from contacts_meas (legacy)
    # - td_hazard    : integrate-to-1 clock-anchor from contact_td_hazard_head (stable)
    # - none         : disable event resets
    phase_reset_source: str
    depth: int
    num_heads: int
    dropout: float
    context_len: int
    epochs: int
    steps_per_epoch: int
    lr: float
    weight_decay: float

    so3_corr_gate_force: Optional[float]
    so3_corr_gate_logit_reset: Optional[float]
    gate_warmup_steps: int
    gate_warmup_value: Optional[float]
    so3_corr_max_deg: float
    so3_corr_omega_l2_weight: float
    corr_time_weight_max: float
    detach_rollout_state: bool

    # Whether to train SO(3) corrector head (default True).
    train_so3_corrector: bool

    # Optional: fine-tune contact_plan_init_z only (improves contacts_plan cold-start).
    train_contact_plan_init: bool
    contact_plan_init_weight: float
    contact_plan_init_mode: str
    contact_plan_init_hidden: int
    contact_plan_init_dropout: float
    # Fine-tune full contacts_plan dynamics (GRU + heads) with teacher supervision.
    train_contact_plan: bool
    contact_plan_weight: float

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
    # Optional: when train_direct_pose=true, freeze direct_pose_head and train direct_pose_hinge_head only.
    # Useful when hinge is meant to be a residual correction without perturbing the base direct readout.
    direct_pose_hinge_train_only: bool
    # Optional: when train_direct_pose=true, freeze direct_pose_head + direct_pose_hinge_head and train gate head only.
    # Useful when hinge delta is already good (pose-conditioned) and we only need a stance/swing "safety valve".
    direct_pose_hinge_gate_train_only: bool
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
    # - signed_scale : omega_eff = (2*sigmoid(sign_logit)-1) * exp(softclip(log_mag, [-clip,+clip])) * omega_raw
    direct_pose_leg_gate_mode: str  # none|learned|scale|signed_scale|auto
    direct_pose_leg_gate_power: float
    # Only used when direct_pose_leg_gate_mode='scale'.
    direct_pose_leg_scale_log_clip: float
    # Optional hard clamp on leg scale magnitude: k>1 => [1/k, k]; 0/1 disables.
    direct_pose_leg_scale_clamp_k: float
    # Optional: supervise learned leg gate using oracle ||omega_oracle|| thresholding (BCEWithLogits).
    # Target: gate=1 if ||omega_oracle|| >= direct_pose_leg_align_oracle_min_deg else 0.
    direct_pose_leg_gate_sup_weight: float
    # Optional: supervise learned leg *scale* head from an offline alpha-sweep table.
    # Uses regression on the clamped log-scale:
    #   log_mag_target = log(max(best_alpha,0) + eps)
    #   loss = MSE(log_mag, log_mag_target)
    direct_pose_leg_scale_sup_weight: float
    direct_pose_leg_scale_sup_alpha_table_json: str
    direct_pose_leg_scale_sup_log_eps: float
    # Optional: direction alignment loss for leg SO(3) residual omega (see docs/Problems/... 8.10).
    # align_mode='cos':  L_align = relu(-cos(omega_pred, omega_oracle))  (cheatable by ||omega_pred||->0)
    # align_mode='proj': omega_oracle = log(R_gt @ R_base^T) * 2; L = w_mag*(proj-||oracle||)^2 + w_res*||res||^2 (+ w_sign*relu(-proj)^2)
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
    # - concat           : append phase_z_in as extra features (legacy "add phase")
    # - replace_contacts : use phase_z_in to replace (contacts_plan, contacts_meas) in direct concat mode
    direct_pose_phase_z_mode: str
    # Optional: split direct output into leg/non-leg heads with shared trunk.
    direct_pose_split_enable: bool
    # Optional: non-leg projection bottleneck dim for split head.
    # >0 => h_nonleg=ReLU(Linear(hid, proj)); out_nonleg=Linear(proj, D_nonleg)
    # 0 => legacy split (out_nonleg=Linear(hid, D_nonleg))
    direct_pose_nonleg_proj_dim: int
    # Optional: when train_direct_pose=true, freeze trunk/leg and train non-leg branch only.
    direct_pose_nonleg_train_only: bool
    direct_pose_reinit: bool
    direct_pose_hidden_override: Optional[int]
    direct_pose_meas_mode_override: Optional[str]
    direct_pose_hinge_enable: bool
    direct_pose_hinge_bones: Optional[str]
    direct_pose_hinge_axis: str
    direct_pose_hinge_max_deg: float
    direct_pose_hinge_hidden: Optional[int]
    # Optional: hinge-specific feature source (defaults to direct_pose_feat_source when set to "auto").
    direct_pose_hinge_feat_source: str
    # Optional: expose base direct prediction to hinge head input (see train/models.py).
    direct_pose_hinge_base_feat: str
    # Optional: clean hinge split (delta_base(nonhidden) + eps(hidden)).
    direct_pose_hinge_clean: bool
    # Eps branch hyperparams / regularizers (used only when direct_pose_hinge_clean=true).
    # - eps_max_deg: if >0, uses this absolute bound (deg) for eps; else uses eps_max_scale * hinge_max_deg
    # - eps_lr_scale: optimizer LR multiplier for eps head params (e.g. 0.1)
    # - eps_l2_weight: output penalty weight: mean(eps^2) (rad^2)
    direct_pose_hinge_eps_max_deg: float
    direct_pose_hinge_eps_max_scale: float
    direct_pose_hinge_eps_hidden: Optional[int]
    direct_pose_hinge_eps_dropout: float
    # Eps branch input source (shape-compatible; used only when direct_pose_hinge_clean=true).
    # - hidden: h_final (post-PASA) [default]
    # - hidden_pre: h_temporal (pre-PASA)
    direct_pose_hinge_eps_source: str
    direct_pose_hinge_eps_lr_scale: float
    direct_pose_hinge_eps_l2_weight: float
    # Optional: contact-based gating for hinge correction (inference-aligned).
    direct_pose_hinge_gate_mode: str
    direct_pose_hinge_gate_source: str
    direct_pose_hinge_gate_power: float
    # Optional: supervised hinge delta regression (delta_target) to stabilize sign/magnitude.
    direct_pose_hinge_sup_weight: float
    direct_pose_hinge_sup_kind: str
    direct_pose_hinge_sup_contact_source: str
    direct_pose_hinge_sup_contact_value: Optional[int]
    direct_pose_hinge_sup_contact_thresh: float
    direct_pose_hinge_sup_angle_thresh_deg: float
    # Optional: hard-example mining on hinge supervision using |delta_target| magnitude (deg).
    direct_pose_hinge_sup_delta_thresh_deg: float
    # Optional: weight hinge supervision by normalized |delta_target|^p (p>0).
    direct_pose_hinge_sup_delta_weight_power: float
    # Optional: delta magnitude weighting scale (deg). If >0, uses (|delta_target|/scale)^p without clamping to 1.
    # If 0, falls back to normalizing by hinge max_rad and clamping to [0,1] (legacy).
    direct_pose_hinge_sup_delta_weight_scale_deg: float
    # Optional: clamp max value of delta magnitude weight (only when delta_weight_power>0).
    # 0 disables the clamp.
    direct_pose_hinge_sup_delta_weight_max: float
    # Optional: supervise learned hinge gate (swing=1, stance=0) using contact thresholding.
    direct_pose_hinge_gate_sup_weight: float
    direct_pose_hinge_gate_sup_contact_source: str  # gt|plan|meas
    direct_pose_hinge_gate_sup_contact_thresh: float
    # Optional: suppress hinge corrections on stance frames (safety term when gate_mode=none).
    # Penalizes |delta_raw| when contact >= thresh (per hinge joint contact channel).
    direct_pose_hinge_stance_weight: float
    direct_pose_hinge_stance_kind: str  # smooth_l1|l2
    direct_pose_hinge_stance_contact_source: str  # gt|plan|meas
    direct_pose_hinge_stance_contact_thresh: float
    # Optional: contact-free hinge delta magnitude regularizer (encourages minimal corrections).
    # Applied to the *effective* hinge delta used to correct direct pose (after any gate/clamp).
    direct_pose_hinge_reg_weight: float
    direct_pose_hinge_reg_kind: str  # l1|l2|smooth_l1

    # Optional: de-dilution weighting for the Stage6/Stage7 direct objective (cond anchor).
    # These reweight per-step joint errors to give more gradient credit to worse joints / swing phases,
    # without introducing per-phase/bone LUTs.
    direct_pose_loss_tail_mix: float
    direct_pose_loss_tail_temp_deg: float
    direct_pose_loss_state_swing_boost: float
    direct_pose_loss_state_contact_source: str  # gt|plan|meas
    direct_pose_loss_state_scope: str  # legs|limbs|all
    # Stage7 direct objective: optionally decouple legs vs non-legs (see discussion in Jan 2026 notes).
    direct_pose_loss_leg_split: bool
    direct_pose_loss_leg_tail_scale: str  # center|mad|none
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
    # Optional: per-(sic,bone) hotspot weighting for direct objective.
    # Table schema follows alpha_by_sic_bone (same sic mask semantics as stage7 alpha tables),
    # but only non-neutral pairs (alpha != 1) are used as a binary hotspot mask.
    # This path does NOT supervise gate scale values; it only reweights direct pose geodesic loss.
    direct_pose_loss_pair_boost_table_json: str
    # Multiplicative weight for masked (sic,bone) pairs; all other pairs keep weight=1.
    direct_pose_loss_pair_boost: float
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

    # Optional: also train contact_meas_head (supervised by GT soft contacts, but NOT fed into model)
    train_contact_meas: bool
    contact_meas_weight: float
    # When train_contact_meas is the only enabled target, optionally supervise meas head under
    # closed-loop rollout (freerun-like) instead of teacher forcing to reduce drift/OOD.
    contact_meas_rollout: bool
    # Optional stability regularizers on contacts_meas logits (teacher and/or rollout):
    # - temporal smoothness: penalize high-frequency logit changes (|Δlogit| / L2 / smooth_l1)
    # - confidence margin: penalize logits near 0 (p≈0.5) to reduce threshold-crossing noise
    # Both can be masked out near GT transitions (contacts near 0.5).
    contact_meas_smooth_weight: float
    contact_meas_smooth_kind: str  # l1|l2|smooth_l1
    contact_meas_margin_weight: float
    contact_meas_margin_logit: float
    contact_meas_transition_band: float
    # Optional: mixed supervision for train_contact_meas_only.
    # When >0, adds a closed-loop rollout meas loss term on top of the teacher loss.
    contact_meas_rollout_weight: float

    # Optional: also train contact_td_hazard_head (pose-derived touchdown hazard; integrate-to-1 clock)
    train_contact_td_hazard: bool
    contact_td_hazard_bce_weight: float
    # Extra BCE weight on touchdown event frames (ttc_td_events==1). Helps avoid the trivial constant-rate solution.
    contact_td_hazard_event_weight: float
    contact_td_hazard_mass_weight: float
    contact_td_hazard_unimodal_weight: float
    # Optional: entropy penalty (softmax over time) to encourage a sharp single peak per cycle.
    contact_td_hazard_entropy_weight: float
    # Optional: clock-alignment loss to make integrate-to-1 trigger at the GT touchdown step.
    # Uses cumulative sum of sigmoid(logit) within a cycle to encourage:
    #   cum_prob[t_gt] ~= 1.0 and cum_prob[t_gt-1] ~= 0.0 (=> a sharp spike at touchdown).
    contact_td_hazard_clock_weight: float
    # When train_contact_td_hazard is the only enabled target, optionally supervise hazard head under
    # closed-loop rollout (freerun-like) instead of teacher forcing to reduce drift/OOD.
    contact_td_hazard_rollout: bool
    # Optional: mixed supervision for train_contact_td_hazard_only (extra rollout term on top of teacher loss).
    contact_td_hazard_rollout_weight: float
    # TD hazard head hyperparams (only used when initializing head without weights).
    contact_td_hazard_hidden: int
    contact_td_hazard_dropout: float

    # Direct<->meas ablations (training-time): keep values, change gradient/data flow only.
    # - direct_pose_meas_force_zero: direct head ignores contacts_meas (concat->zeros, mode_select->uniform)
    # - direct_pose_meas_detach: stop-grad from direct head into contacts_meas (prevents co-adaptation)
    direct_pose_meas_force_zero: bool
    direct_pose_meas_detach: bool

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

    seed: int


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


def _as_int_list(val: Any) -> Optional[list[int]]:
    """
    Parse a list of integers from:
      - list/tuple: [1,2,3]
      - csv-ish string: "1,2,3" (also supports ';' and whitespace)
      - range tokens: "49-55", "49..55", "49:55" (inclusive)
      - JSON list string: "[1,2,3]"
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        out: list[int] = []
        for x in val:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out if out else None
    s = str(val).strip()
    if not s:
        return None
    # Allow JSON list strings like "[1,2,3]".
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            payload = json.loads(s)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            payload = payload.get("values") or payload.get("sics") or payload.get("list")
        if isinstance(payload, (list, tuple)):
            out = []
            for x in payload:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out if out else None
        # Fallthrough: treat as a plain string spec.

    # Tokenize: allow commas, semicolons, and whitespace.
    spec = s.replace(";", ",").replace("\n", ",").replace("\t", ",").replace(" ", ",")
    toks = [t.strip() for t in spec.split(",") if t.strip()]
    if not toks:
        return None
    out_vals: list[int] = []
    for tok in toks:
        if not tok:
            continue
        # Inclusive range tokens: a-b, a..b, a:b
        sep = None
        if ".." in tok:
            sep = ".."
        elif ":" in tok:
            sep = ":"
        elif "-" in tok and not tok.startswith("-"):
            sep = "-"
        if sep is not None:
            try:
                a_str, b_str = tok.split(sep, 1)
                a = int(a_str.strip())
                b = int(b_str.strip())
            except Exception:
                continue
            if a <= b:
                out_vals.extend(list(range(a, b + 1)))
            else:
                out_vals.extend(list(range(a, b - 1, -1)))
            continue
        try:
            out_vals.append(int(tok))
        except Exception:
            continue

    # De-dup while preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for v in out_vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out if out else None


def _canon_phase_reset_source(val: Any) -> str:
    s = str(val or "contacts_meas").strip().lower()
    # External TTC-driven resets (handled in posttrain rollout loops, similar to run_freerun_cycles):
    # - ttc_gt  : use GT touchdown events (ttc_td_events) to reset phase_z to the anchor [0,1]
    if s in ("ttc", "ttc_gt", "ttcgt"):
        return "ttc_gt"
    if s in ("ttc_pred", "ttcpred"):
        raise SystemExit(
            "[FATAL] phase_reset_source=ttc_pred is no longer supported (TTC path removed; 2026-02-07). "
            "Use --phase_reset_source none (recommended) or td_hazard."
        )
    if s in ("hazard", "tdhazard", "td_hazard", "tdhaz"):
        return "td_hazard"
    if s in ("none", "null", "off", "disable", "disabled"):
        return "none"
    if s in ("contacts", "contacts_meas", "meas", "contact_meas"):
        return "contacts_meas"
    return "contacts_meas"


def _load_cfg(path: Path) -> PostTrainConfig:
    payload = load_json(path)
    if not payload:
        raise FileNotFoundError(f"Missing or empty config: {path}")
    return _cfg_from_payload(payload)


def _cfg_from_payload(payload: Dict[str, Any]) -> PostTrainConfig:
    if not isinstance(payload, dict):
        raise TypeError("posttrain config payload must be a dict")

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
            "Migrate to phase_reset_source=none (no-reset) or train_contact_td_hazard."
        )

    ckpt_in = _as_path(payload.get("ckpt_in"))
    out_dir = _as_path(payload.get("out_dir")) or Path("./models/posttrain")
    if ckpt_in is None:
        raise ValueError("Config must set 'ckpt_in'.")

    run_name = str(payload.get("run_name") or f"posttrain_{time.strftime('%Y%m%d-%H%M%S')}")

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
    pretrain = _as_path(payload.get("pretrain_template"))
    encoder = _as_path(payload.get("encoder_bundle") or payload.get("encoder_path"))

    event_clock_hidden_dim = payload.get("event_clock_hidden_dim", None)
    if event_clock_hidden_dim is not None:
        try:
            event_clock_hidden_dim = int(event_clock_hidden_dim)
        except Exception:
            event_clock_hidden_dim = None
    event_clock_gate_hidden_dim = payload.get("event_clock_gate_hidden_dim", None)
    if event_clock_gate_hidden_dim is not None:
        try:
            event_clock_gate_hidden_dim = int(event_clock_gate_hidden_dim)
        except Exception:
            event_clock_gate_hidden_dim = None

    gate_sup_weight = payload.get("lambda_gate_sup_weight", None)
    gate_sup_weight = float(gate_sup_weight) if gate_sup_weight is not None else 0.0
    gate_sup_tau_deg = payload.get("lambda_gate_sup_tau_deg", None)
    gate_sup_tau_deg = float(gate_sup_tau_deg) if gate_sup_tau_deg is not None else 2.5
    gate_sup_margin_deg = payload.get("lambda_gate_sup_margin_deg", None)
    gate_sup_margin_deg = float(gate_sup_margin_deg) if gate_sup_margin_deg is not None else 1.0
    gate_sup_start_step = payload.get("lambda_gate_sup_start_step", None)
    gate_sup_start_step = int(gate_sup_start_step) if gate_sup_start_step is not None else -1

    direct_pose_time_pe_dim = payload.get("direct_pose_time_pe_dim", None)
    if direct_pose_time_pe_dim is not None:
        try:
            direct_pose_time_pe_dim = int(direct_pose_time_pe_dim)
        except Exception:
            direct_pose_time_pe_dim = None
    if direct_pose_time_pe_dim is None:
        direct_pose_time_pe_dim = -1  # auto/infer

    direct_pose_hidden_override = payload.get("direct_pose_hidden_override", None)
    if direct_pose_hidden_override is not None:
        try:
            direct_pose_hidden_override = int(direct_pose_hidden_override)
        except Exception:
            direct_pose_hidden_override = None

    direct_pose_meas_mode_override = payload.get("direct_pose_meas_mode_override", None)
    if direct_pose_meas_mode_override is None:
        direct_pose_meas_mode_override = payload.get("direct_pose_meas_mode", None)
    if direct_pose_meas_mode_override is not None:
        s = str(direct_pose_meas_mode_override).strip()
        direct_pose_meas_mode_override = s if s else None
    direct_pose_split_enable = _as_bool(payload.get("direct_pose_split_enable", False), False)
    try:
        direct_pose_nonleg_proj_dim = int(payload.get("direct_pose_nonleg_proj_dim") or 0)
    except Exception:
        direct_pose_nonleg_proj_dim = 0
    if direct_pose_nonleg_proj_dim < 0:
        direct_pose_nonleg_proj_dim = 0
    direct_pose_nonleg_train_only = _as_bool(payload.get("direct_pose_nonleg_train_only", False), False)

    # Optional: main+leg split for direct pose (leg residual head).
    direct_pose_leg_enable = _as_bool(payload.get("direct_pose_leg_enable", False), False)
    direct_pose_leg_bones = payload.get("direct_pose_leg_bones", None)
    direct_pose_leg_train_only = _as_bool(payload.get("direct_pose_leg_train_only", False), False)
    direct_pose_leg_gate_train_only = _as_bool(payload.get("direct_pose_leg_gate_train_only", False), False)
    direct_pose_leg_mode = str(payload.get("direct_pose_leg_mode") or "rot6d_add")
    direct_pose_leg_stopgrad_main = _as_bool(payload.get("direct_pose_leg_stopgrad_main", False), False)
    direct_pose_leg_detach_feat = _as_bool(payload.get("direct_pose_leg_detach_feat", False), False)
    try:
        direct_pose_leg_max_deg = float(payload.get("direct_pose_leg_max_deg") or 0.0)
    except Exception:
        direct_pose_leg_max_deg = 0.0
    if (not math.isfinite(direct_pose_leg_max_deg)) or direct_pose_leg_max_deg < 0.0:
        direct_pose_leg_max_deg = 0.0

    # Optional: learned gate for leg omega (SO(3) only; see train/models.py).
    direct_pose_leg_gate_mode = str(payload.get("direct_pose_leg_gate_mode") or "auto").strip().lower()
    if direct_pose_leg_gate_mode in ("", "auto"):
        direct_pose_leg_gate_mode = "auto"
    elif direct_pose_leg_gate_mode in ("learned", "on", "true", "1", "yes", "y"):
        direct_pose_leg_gate_mode = "learned"
    elif direct_pose_leg_gate_mode in (
        "signed_scale",
        "signedscale",
        "signed",
        "signmag",
        "sign_mag",
        "signmagscale",
        "signedmag",
        "sscale",
    ):
        direct_pose_leg_gate_mode = "signed_scale"
    elif direct_pose_leg_gate_mode in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
        direct_pose_leg_gate_mode = "scale"
    elif direct_pose_leg_gate_mode in ("none", "off", "false", "0", "no", "n", "disable", "disabled"):
        direct_pose_leg_gate_mode = "none"
    else:
        direct_pose_leg_gate_mode = "auto"
    try:
        direct_pose_leg_gate_power = float(payload.get("direct_pose_leg_gate_power") or 1.0)
    except Exception:
        direct_pose_leg_gate_power = 1.0
    if (not math.isfinite(direct_pose_leg_gate_power)) or direct_pose_leg_gate_power <= 0.0:
        direct_pose_leg_gate_power = 1.0
    try:
        direct_pose_leg_gate_sup_weight = float(
            payload.get("direct_pose_leg_gate_sup_weight", payload.get("direct_pose_leg_gate_loss_weight", 0.0)) or 0.0
        )
    except Exception:
        direct_pose_leg_gate_sup_weight = 0.0
    if (not math.isfinite(direct_pose_leg_gate_sup_weight)) or direct_pose_leg_gate_sup_weight < 0.0:
        direct_pose_leg_gate_sup_weight = 0.0

    # Only used when direct_pose_leg_gate_mode='scale' (exp(log_mag)).
    try:
        direct_pose_leg_scale_log_clip = float(payload.get("direct_pose_leg_scale_log_clip") or 4.0)
    except Exception:
        direct_pose_leg_scale_log_clip = 4.0
    if (not math.isfinite(direct_pose_leg_scale_log_clip)) or direct_pose_leg_scale_log_clip <= 0.0:
        direct_pose_leg_scale_log_clip = 4.0
    try:
        direct_pose_leg_scale_clamp_k = float(payload.get("direct_pose_leg_scale_clamp_k") or 0.0)
    except Exception:
        direct_pose_leg_scale_clamp_k = 0.0
    if (not math.isfinite(direct_pose_leg_scale_clamp_k)) or direct_pose_leg_scale_clamp_k <= 1.0:
        direct_pose_leg_scale_clamp_k = 0.0

    # Optional: supervision for the scale head using offline alpha-sweep best_alpha table.
    try:
        direct_pose_leg_scale_sup_weight = float(payload.get("direct_pose_leg_scale_sup_weight") or 0.0)
    except Exception:
        direct_pose_leg_scale_sup_weight = 0.0
    if (not math.isfinite(direct_pose_leg_scale_sup_weight)) or direct_pose_leg_scale_sup_weight < 0.0:
        direct_pose_leg_scale_sup_weight = 0.0
    direct_pose_leg_scale_sup_alpha_table_json = str(payload.get("direct_pose_leg_scale_sup_alpha_table_json") or "").strip()
    try:
        direct_pose_leg_scale_sup_log_eps = float(payload.get("direct_pose_leg_scale_sup_log_eps") or 0.01)
    except Exception:
        direct_pose_leg_scale_sup_log_eps = 0.01
    if (not math.isfinite(direct_pose_leg_scale_sup_log_eps)) or direct_pose_leg_scale_sup_log_eps <= 0.0:
        direct_pose_leg_scale_sup_log_eps = 0.01

    # Optional: direction alignment loss for leg SO(3) residual omega.
    try:
        direct_pose_leg_align_weight = float(payload.get("direct_pose_leg_align_weight") or 0.0)
    except Exception:
        direct_pose_leg_align_weight = 0.0
    if (not math.isfinite(direct_pose_leg_align_weight)) or direct_pose_leg_align_weight < 0.0:
        direct_pose_leg_align_weight = 0.0
    try:
        direct_pose_leg_align_oracle_min_deg = float(payload.get("direct_pose_leg_align_oracle_min_deg") or 0.0)
    except Exception:
        direct_pose_leg_align_oracle_min_deg = 0.0
    if (not math.isfinite(direct_pose_leg_align_oracle_min_deg)) or direct_pose_leg_align_oracle_min_deg < 0.0:
        direct_pose_leg_align_oracle_min_deg = 0.0
    try:
        direct_pose_leg_align_oracle_weight_deg = float(payload.get("direct_pose_leg_align_oracle_weight_deg") or 0.0)
    except Exception:
        direct_pose_leg_align_oracle_weight_deg = 0.0
    if (not math.isfinite(direct_pose_leg_align_oracle_weight_deg)) or direct_pose_leg_align_oracle_weight_deg < 0.0:
        direct_pose_leg_align_oracle_weight_deg = 0.0
    direct_pose_leg_align_mode = str(payload.get("direct_pose_leg_align_mode") or "cos").strip().lower()
    if direct_pose_leg_align_mode in ("", "none", "off", "disable", "disabled"):
        direct_pose_leg_align_mode = "cos"
    if direct_pose_leg_align_mode not in ("cos", "proj"):
        direct_pose_leg_align_mode = "cos"
    try:
        direct_pose_leg_align_mag_weight = float(payload.get("direct_pose_leg_align_mag_weight") or 1.0)
    except Exception:
        direct_pose_leg_align_mag_weight = 1.0
    if (not math.isfinite(direct_pose_leg_align_mag_weight)) or direct_pose_leg_align_mag_weight < 0.0:
        direct_pose_leg_align_mag_weight = 1.0
    try:
        direct_pose_leg_align_res_weight = float(payload.get("direct_pose_leg_align_res_weight") or 1.0)
    except Exception:
        direct_pose_leg_align_res_weight = 1.0
    if (not math.isfinite(direct_pose_leg_align_res_weight)) or direct_pose_leg_align_res_weight < 0.0:
        direct_pose_leg_align_res_weight = 1.0
    try:
        direct_pose_leg_align_sign_weight = float(payload.get("direct_pose_leg_align_sign_weight") or 0.0)
    except Exception:
        direct_pose_leg_align_sign_weight = 0.0
    if (not math.isfinite(direct_pose_leg_align_sign_weight)) or direct_pose_leg_align_sign_weight < 0.0:
        direct_pose_leg_align_sign_weight = 0.0
    try:
        direct_pose_leg_align_cos_thresh = float(payload.get("direct_pose_leg_align_cos_thresh") or 0.0)
    except Exception:
        direct_pose_leg_align_cos_thresh = 0.0
    if (not math.isfinite(direct_pose_leg_align_cos_thresh)) or direct_pose_leg_align_cos_thresh < 0.0:
        direct_pose_leg_align_cos_thresh = 0.0

    # Optional: per-side routing + shared leg head (SO(3) only; see train/models.py).
    direct_pose_leg_side_routing = _as_bool(payload.get("direct_pose_leg_side_routing", False), False)
    direct_pose_leg_contact_order = str(payload.get("direct_pose_leg_contact_order") or "lr").strip().lower()
    if direct_pose_leg_contact_order in ("rl", "r,l", "r l"):
        direct_pose_leg_contact_order = "rl"
    else:
        direct_pose_leg_contact_order = "lr"
    try:
        direct_pose_leg_side_embed_dim = int(payload.get("direct_pose_leg_side_embed_dim") or 0)
    except Exception:
        direct_pose_leg_side_embed_dim = 0
    if direct_pose_leg_side_embed_dim < 0:
        direct_pose_leg_side_embed_dim = 0
    direct_pose_leg_side_plan_other = _as_bool(payload.get("direct_pose_leg_side_plan_other", False), False)
    direct_pose_leg_side_phase_other = _as_bool(payload.get("direct_pose_leg_side_phase_other", False), False)
    direct_pose_leg_side_phase_rel = _as_bool(payload.get("direct_pose_leg_side_phase_rel", False), False)
    direct_pose_leg_side_cue = str(payload.get("direct_pose_leg_side_cue") or "none").strip().lower()
    if direct_pose_leg_side_cue in ("", "none", "off", "disable", "disabled"):
        direct_pose_leg_side_cue = "none"
    try:
        direct_pose_leg_side_cue_tau = float(payload.get("direct_pose_leg_side_cue_tau") or 30.0)
    except Exception:
        direct_pose_leg_side_cue_tau = 30.0
    if (not math.isfinite(direct_pose_leg_side_cue_tau)) or direct_pose_leg_side_cue_tau <= 0.0:
        direct_pose_leg_side_cue_tau = 30.0

    # Optional: per-side sign gate for routed shared leg omega head.
    direct_pose_leg_side_sign_gate = _as_bool(payload.get("direct_pose_leg_side_sign_gate", False), False)
    # Optional: rank-1 coupling for routed shared leg omega head.
    direct_pose_leg_side_rank1 = _as_bool(payload.get("direct_pose_leg_side_rank1", False), False)
    try:
        direct_pose_leg_side_sign_gate_reg_weight = float(payload.get("direct_pose_leg_side_sign_gate_reg_weight") or 0.0)
    except Exception:
        direct_pose_leg_side_sign_gate_reg_weight = 0.0
    if (not math.isfinite(direct_pose_leg_side_sign_gate_reg_weight)) or direct_pose_leg_side_sign_gate_reg_weight < 0.0:
        direct_pose_leg_side_sign_gate_reg_weight = 0.0

    direct_pose_hinge_enable = _as_bool(payload.get("direct_pose_hinge_enable", False), False)
    direct_pose_hinge_bones = payload.get("direct_pose_hinge_bones", None)
    direct_pose_hinge_axis = str(payload.get("direct_pose_hinge_axis") or "z")
    try:
        direct_pose_hinge_max_deg = float(payload.get("direct_pose_hinge_max_deg") or 45.0)
    except Exception:
        direct_pose_hinge_max_deg = 45.0
    direct_pose_hinge_hidden = payload.get("direct_pose_hinge_hidden", None)
    if direct_pose_hinge_hidden is not None:
        try:
            direct_pose_hinge_hidden = int(direct_pose_hinge_hidden)
        except Exception:
            direct_pose_hinge_hidden = None

    direct_pose_hinge_feat_source = str(payload.get("direct_pose_hinge_feat_source") or "auto")
    direct_pose_hinge_base_feat = str(payload.get("direct_pose_hinge_base_feat") or "none")
    direct_pose_hinge_clean = _as_bool(payload.get("direct_pose_hinge_clean", False), False)
    try:
        direct_pose_hinge_eps_max_deg = float(payload.get("direct_pose_hinge_eps_max_deg") or 0.0)
    except Exception:
        direct_pose_hinge_eps_max_deg = 0.0
    if not math.isfinite(direct_pose_hinge_eps_max_deg) or direct_pose_hinge_eps_max_deg < 0.0:
        direct_pose_hinge_eps_max_deg = 0.0
    # NOTE: allow 0.0 to explicitly disable eps(hidden) in clean hinge mode.
    try:
        _raw = payload.get("direct_pose_hinge_eps_max_scale", 0.5)
        if _raw is None:
            _raw = 0.5
        direct_pose_hinge_eps_max_scale = float(_raw)
    except Exception:
        direct_pose_hinge_eps_max_scale = 0.5
    if (not math.isfinite(direct_pose_hinge_eps_max_scale)) or direct_pose_hinge_eps_max_scale < 0.0:
        direct_pose_hinge_eps_max_scale = 0.5
    direct_pose_hinge_eps_hidden = payload.get("direct_pose_hinge_eps_hidden", None)
    if direct_pose_hinge_eps_hidden is not None:
        try:
            direct_pose_hinge_eps_hidden = int(direct_pose_hinge_eps_hidden)
        except Exception:
            direct_pose_hinge_eps_hidden = None
    try:
        direct_pose_hinge_eps_dropout = float(payload.get("direct_pose_hinge_eps_dropout") or 0.0)
    except Exception:
        direct_pose_hinge_eps_dropout = 0.0
    if (not math.isfinite(direct_pose_hinge_eps_dropout)) or direct_pose_hinge_eps_dropout < 0.0:
        direct_pose_hinge_eps_dropout = 0.0
    direct_pose_hinge_eps_dropout = max(0.0, min(1.0, float(direct_pose_hinge_eps_dropout)))
    # Eps(hidden) input source routing for clean hinge split (semantic only; shape-compatible).
    try:
        direct_pose_hinge_eps_source = str(payload.get("direct_pose_hinge_eps_source") or "hidden").strip().lower()
    except Exception:
        direct_pose_hinge_eps_source = "hidden"
    if direct_pose_hinge_eps_source in ("h_pre", "h_temporal", "pre", "temporal", "mid", "hidden_pre"):
        direct_pose_hinge_eps_source = "hidden_pre"
    elif direct_pose_hinge_eps_source in ("h_final", "post", "final", "hidden"):
        direct_pose_hinge_eps_source = "hidden"
    if direct_pose_hinge_eps_source not in ("hidden", "hidden_pre"):
        direct_pose_hinge_eps_source = "hidden"
    try:
        direct_pose_hinge_eps_lr_scale = float(payload.get("direct_pose_hinge_eps_lr_scale") or 1.0)
    except Exception:
        direct_pose_hinge_eps_lr_scale = 1.0
    if (not math.isfinite(direct_pose_hinge_eps_lr_scale)) or direct_pose_hinge_eps_lr_scale <= 0.0:
        direct_pose_hinge_eps_lr_scale = 1.0
    try:
        direct_pose_hinge_eps_l2_weight = float(payload.get("direct_pose_hinge_eps_l2_weight") or 0.0)
    except Exception:
        direct_pose_hinge_eps_l2_weight = 0.0
    if (not math.isfinite(direct_pose_hinge_eps_l2_weight)) or direct_pose_hinge_eps_l2_weight < 0.0:
        direct_pose_hinge_eps_l2_weight = 0.0
    direct_pose_hinge_gate_mode = str(payload.get("direct_pose_hinge_gate_mode") or "none")
    direct_pose_hinge_gate_source = str(payload.get("direct_pose_hinge_gate_source") or "plan")
    try:
        direct_pose_hinge_gate_power = float(payload.get("direct_pose_hinge_gate_power") or 1.0)
    except Exception:
        direct_pose_hinge_gate_power = 1.0

    try:
        direct_pose_hinge_sup_weight = float(payload.get("direct_pose_hinge_sup_weight") or 0.0)
    except Exception:
        direct_pose_hinge_sup_weight = 0.0
    direct_pose_hinge_sup_kind = str(payload.get("direct_pose_hinge_sup_kind") or "smooth_l1")
    direct_pose_hinge_sup_contact_source = str(payload.get("direct_pose_hinge_sup_contact_source") or "gt")
    direct_pose_hinge_sup_contact_value = payload.get("direct_pose_hinge_sup_contact_value", None)
    if direct_pose_hinge_sup_contact_value is not None:
        try:
            direct_pose_hinge_sup_contact_value = int(direct_pose_hinge_sup_contact_value)
        except Exception:
            direct_pose_hinge_sup_contact_value = None
    if direct_pose_hinge_sup_contact_value not in (None, 0, 1):
        direct_pose_hinge_sup_contact_value = None
    try:
        direct_pose_hinge_sup_contact_thresh = float(payload.get("direct_pose_hinge_sup_contact_thresh") or 0.5)
    except Exception:
        direct_pose_hinge_sup_contact_thresh = 0.5
    try:
        direct_pose_hinge_sup_angle_thresh_deg = float(payload.get("direct_pose_hinge_sup_angle_thresh_deg") or 0.0)
    except Exception:
        direct_pose_hinge_sup_angle_thresh_deg = 0.0
    try:
        direct_pose_hinge_sup_delta_thresh_deg = float(payload.get("direct_pose_hinge_sup_delta_thresh_deg") or 0.0)
    except Exception:
        direct_pose_hinge_sup_delta_thresh_deg = 0.0
    try:
        direct_pose_hinge_sup_delta_weight_power = float(payload.get("direct_pose_hinge_sup_delta_weight_power") or 0.0)
    except Exception:
        direct_pose_hinge_sup_delta_weight_power = 0.0
    try:
        direct_pose_hinge_sup_delta_weight_scale_deg = float(payload.get("direct_pose_hinge_sup_delta_weight_scale_deg") or 0.0)
    except Exception:
        direct_pose_hinge_sup_delta_weight_scale_deg = 0.0
    try:
        direct_pose_hinge_sup_delta_weight_max = float(payload.get("direct_pose_hinge_sup_delta_weight_max") or 0.0)
    except Exception:
        direct_pose_hinge_sup_delta_weight_max = 0.0
    try:
        direct_pose_hinge_stance_weight = float(payload.get("direct_pose_hinge_stance_weight") or 0.0)
    except Exception:
        direct_pose_hinge_stance_weight = 0.0
    direct_pose_hinge_stance_kind = str(payload.get("direct_pose_hinge_stance_kind") or "l2")
    direct_pose_hinge_stance_contact_source = str(payload.get("direct_pose_hinge_stance_contact_source") or "gt")
    try:
        direct_pose_hinge_stance_contact_thresh = float(payload.get("direct_pose_hinge_stance_contact_thresh") or 0.5)
    except Exception:
        direct_pose_hinge_stance_contact_thresh = 0.5
    try:
        direct_pose_hinge_reg_weight = float(payload.get("direct_pose_hinge_reg_weight") or 0.0)
    except Exception:
        direct_pose_hinge_reg_weight = 0.0
    direct_pose_hinge_reg_kind = str(payload.get("direct_pose_hinge_reg_kind") or "l1")

    # ---- Stage6/Stage7: direct objective de-dilution weights ----
    try:
        direct_pose_loss_tail_mix = float(payload.get("direct_pose_loss_tail_mix") or 0.0)
    except Exception:
        direct_pose_loss_tail_mix = 0.0
    if (not math.isfinite(direct_pose_loss_tail_mix)) or direct_pose_loss_tail_mix <= 0.0:
        direct_pose_loss_tail_mix = 0.0
    direct_pose_loss_tail_mix = max(0.0, min(1.0, float(direct_pose_loss_tail_mix)))

    try:
        direct_pose_loss_tail_temp_deg = float(payload.get("direct_pose_loss_tail_temp_deg") or 0.0)
    except Exception:
        direct_pose_loss_tail_temp_deg = 0.0
    if (not math.isfinite(direct_pose_loss_tail_temp_deg)) or direct_pose_loss_tail_temp_deg <= 0.0:
        direct_pose_loss_tail_temp_deg = 0.0

    try:
        direct_pose_loss_state_swing_boost = float(payload.get("direct_pose_loss_state_swing_boost") or 0.0)
    except Exception:
        direct_pose_loss_state_swing_boost = 0.0
    if (not math.isfinite(direct_pose_loss_state_swing_boost)) or direct_pose_loss_state_swing_boost < 0.0:
        direct_pose_loss_state_swing_boost = 0.0

    direct_pose_loss_state_contact_source = str(payload.get("direct_pose_loss_state_contact_source") or "gt").strip().lower()
    if direct_pose_loss_state_contact_source not in ("gt", "plan", "meas"):
        direct_pose_loss_state_contact_source = "gt"
    direct_pose_loss_state_scope = str(payload.get("direct_pose_loss_state_scope") or "legs").strip().lower()
    if direct_pose_loss_state_scope not in ("legs", "limbs", "all"):
        direct_pose_loss_state_scope = "legs"
    direct_pose_loss_leg_split = _as_bool(payload.get("direct_pose_loss_leg_split", False), False)
    direct_pose_loss_leg_tail_scale = str(payload.get("direct_pose_loss_leg_tail_scale") or "center").strip().lower()
    if direct_pose_loss_leg_tail_scale in ("median", "med", "center", "c"):
        direct_pose_loss_leg_tail_scale = "center"
    elif direct_pose_loss_leg_tail_scale in ("mad", "median_abs_dev", "median_abs_deviation"):
        direct_pose_loss_leg_tail_scale = "mad"
    elif direct_pose_loss_leg_tail_scale in ("none", "off", "0"):
        direct_pose_loss_leg_tail_scale = "none"
    else:
        direct_pose_loss_leg_tail_scale = "center"

    # Optional: direct objective step_in_cycle (sic) focus list.
    direct_pose_loss_sics = payload.get("direct_pose_loss_sics", None)
    if isinstance(direct_pose_loss_sics, (list, tuple)):
        ints = _as_int_list(direct_pose_loss_sics) or []
        direct_pose_loss_sics = ",".join(str(int(x)) for x in ints) if ints else None
    elif direct_pose_loss_sics is not None:
        s = str(direct_pose_loss_sics).strip()
        direct_pose_loss_sics = s if s else None
    try:
        direct_pose_loss_cycle_gte = int(payload.get("direct_pose_loss_cycle_gte") or 0)
    except Exception:
        direct_pose_loss_cycle_gte = 0
    if direct_pose_loss_cycle_gte < 0:
        direct_pose_loss_cycle_gte = 0

    direct_pose_loss_sic_mode = str(payload.get("direct_pose_loss_sic_mode") or "mask").strip().lower()
    if direct_pose_loss_sic_mode in ("", "none", "off", "disable", "disabled"):
        direct_pose_loss_sic_mode = "mask"
    if direct_pose_loss_sic_mode not in ("mask", "boost"):
        direct_pose_loss_sic_mode = "mask"
    try:
        direct_pose_loss_sic_boost = float(payload.get("direct_pose_loss_sic_boost") or 1.0)
    except Exception:
        direct_pose_loss_sic_boost = 1.0
    if (not math.isfinite(direct_pose_loss_sic_boost)) or direct_pose_loss_sic_boost <= 0.0:
        direct_pose_loss_sic_boost = 1.0
    direct_pose_loss_pair_boost_table_json = str(payload.get("direct_pose_loss_pair_boost_table_json") or "").strip()
    try:
        direct_pose_loss_pair_boost = float(payload.get("direct_pose_loss_pair_boost") or 1.0)
    except Exception:
        direct_pose_loss_pair_boost = 1.0
    if (not math.isfinite(direct_pose_loss_pair_boost)) or direct_pose_loss_pair_boost <= 0.0:
        direct_pose_loss_pair_boost = 1.0
    direct_pose_loss_group_norm_enable = _as_bool(payload.get("direct_pose_loss_group_norm_enable", False), False)
    try:
        direct_pose_loss_group_norm_w_leg = float(payload.get("direct_pose_loss_group_norm_w_leg") or 1.0)
    except Exception:
        direct_pose_loss_group_norm_w_leg = 1.0
    try:
        direct_pose_loss_group_norm_w_nonleg = float(payload.get("direct_pose_loss_group_norm_w_nonleg") or 1.0)
    except Exception:
        direct_pose_loss_group_norm_w_nonleg = 1.0
    try:
        direct_pose_loss_group_norm_ema_beta = float(payload.get("direct_pose_loss_group_norm_ema_beta") or 0.95)
    except Exception:
        direct_pose_loss_group_norm_ema_beta = 0.95
    if (not math.isfinite(direct_pose_loss_group_norm_ema_beta)) or direct_pose_loss_group_norm_ema_beta < 0.0:
        direct_pose_loss_group_norm_ema_beta = 0.95
    direct_pose_loss_group_norm_ema_beta = max(0.0, min(0.9999, float(direct_pose_loss_group_norm_ema_beta)))
    try:
        direct_pose_loss_group_norm_ratio_min = float(payload.get("direct_pose_loss_group_norm_ratio_min") or 0.2)
    except Exception:
        direct_pose_loss_group_norm_ratio_min = 0.2
    try:
        direct_pose_loss_group_norm_ratio_max = float(payload.get("direct_pose_loss_group_norm_ratio_max") or 5.0)
    except Exception:
        direct_pose_loss_group_norm_ratio_max = 5.0
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_min)) or direct_pose_loss_group_norm_ratio_min <= 0.0:
        direct_pose_loss_group_norm_ratio_min = 0.2
    if (not math.isfinite(direct_pose_loss_group_norm_ratio_max)) or direct_pose_loss_group_norm_ratio_max <= 0.0:
        direct_pose_loss_group_norm_ratio_max = 5.0
    if direct_pose_loss_group_norm_ratio_min > direct_pose_loss_group_norm_ratio_max:
        direct_pose_loss_group_norm_ratio_min, direct_pose_loss_group_norm_ratio_max = (
            direct_pose_loss_group_norm_ratio_max,
            direct_pose_loss_group_norm_ratio_min,
        )
    try:
        direct_pose_loss_group_norm_eps = float(payload.get("direct_pose_loss_group_norm_eps") or 1e-6)
    except Exception:
        direct_pose_loss_group_norm_eps = 1e-6
    if (not math.isfinite(direct_pose_loss_group_norm_eps)) or direct_pose_loss_group_norm_eps <= 0.0:
        direct_pose_loss_group_norm_eps = 1e-6
    direct_pose_grad_monitor_enable = _as_bool(payload.get("direct_pose_grad_monitor_enable", False), False)
    try:
        direct_pose_grad_ratio_gate = float(payload.get("direct_pose_grad_ratio_gate") or 0.35)
    except Exception:
        direct_pose_grad_ratio_gate = 0.35
    if (not math.isfinite(direct_pose_grad_ratio_gate)) or direct_pose_grad_ratio_gate <= 0.0:
        direct_pose_grad_ratio_gate = 0.35

    return PostTrainConfig(
        ckpt_in=ckpt_in,
        out_dir=out_dir,
        run_name=run_name,
        data=data,
        paths=paths,
        bundle_json=bundle,
        pretrain_template=pretrain,
        encoder_bundle=encoder,
        device=str(payload.get("device") or "auto"),
        batch=int(payload.get("batch") or 8),
        seq_len=int(payload.get("seq_len") or 180),
        dataset_index_mode=str(payload.get("dataset_index_mode") or payload.get("index_mode") or "sliding"),
        rollout_steps=int(payload.get("rollout_steps") or 0),
        rollout_cycles=max(1, int(payload.get("rollout_cycles") or 1)),
        rollout_include_boundary=_as_bool(
            payload.get("rollout_include_boundary", None),
            default=(int(payload.get("rollout_cycles") or 1) > 1),
        ),
        rollout_random_offset=_as_bool(payload.get("rollout_random_offset", False), False),
        time_index_mode=str(payload.get("time_index_mode") or "global"),
        phase_reset_source=_canon_phase_reset_source(payload.get("phase_reset_source", "contacts_meas")),
        depth=int(payload.get("depth") or 3),
        num_heads=int(payload.get("num_heads") or 4),
        dropout=float(payload.get("dropout") or 0.1),
        context_len=int(payload.get("context_len") or 16),
        # NOTE: allow explicit 0 in configs (e.g. no-op posttrain stages with epochs=0).
        epochs=int(payload["epochs"]) if ("epochs" in payload and payload["epochs"] is not None) else 1,
        steps_per_epoch=int(payload["steps_per_epoch"]) if ("steps_per_epoch" in payload and payload["steps_per_epoch"] is not None) else 200,
        lr=float(payload["lr"]) if ("lr" in payload and payload["lr"] is not None) else 2e-4,
        weight_decay=float(payload.get("weight_decay") or 0.0),
        so3_corr_gate_force=payload.get("so3_corr_gate_force", None),
        so3_corr_gate_logit_reset=payload.get("so3_corr_gate_logit_reset", None),
        gate_warmup_steps=int(payload.get("gate_warmup_steps") or 0),
        gate_warmup_value=payload.get("gate_warmup_value", None),
        so3_corr_max_deg=float(payload.get("so3_corr_max_deg") or 20.0),
        so3_corr_omega_l2_weight=float(payload.get("so3_corr_omega_l2_weight") or 5e-4),
        corr_time_weight_max=float(payload.get("corr_time_weight_max") or 2.0),
        detach_rollout_state=_as_bool(payload.get("detach_rollout_state", True), True),
        train_so3_corrector=_as_bool(payload.get("train_so3_corrector", True), True),
        train_contact_plan_init=_as_bool(payload.get("train_contact_plan_init", False), False),
        contact_plan_init_weight=float(payload.get("contact_plan_init_weight") or 1.0),
        contact_plan_init_mode=str(payload.get("contact_plan_init_mode") or "learnable"),
        contact_plan_init_hidden=int(payload.get("contact_plan_init_hidden") or 128),
        contact_plan_init_dropout=float(payload.get("contact_plan_init_dropout") or 0.0),
        train_contact_plan=_as_bool(payload.get("train_contact_plan", False), False),
        contact_plan_weight=float(payload.get("contact_plan_weight") or 1.0),
        event_clock=str(payload.get("event_clock") or "auto"),
        event_clock_max_delta=float(payload.get("event_clock_max_delta") or 0.5),
        event_clock_hidden_dim=event_clock_hidden_dim,
        event_clock_gate_hidden_dim=event_clock_gate_hidden_dim,
        train_lambda_head=_as_bool(payload.get("train_lambda_head", False), False),
        train_direct_pose=_as_bool(payload.get("train_direct_pose", False), False),
        direct_pose_hinge_train_only=_as_bool(payload.get("direct_pose_hinge_train_only", False), False),
        direct_pose_hinge_gate_train_only=_as_bool(payload.get("direct_pose_hinge_gate_train_only", False), False),
        direct_pose_leg_enable=bool(direct_pose_leg_enable),
        direct_pose_leg_bones=direct_pose_leg_bones,
        direct_pose_leg_train_only=bool(direct_pose_leg_train_only),
        direct_pose_leg_gate_train_only=bool(direct_pose_leg_gate_train_only),
        direct_pose_leg_mode=str(direct_pose_leg_mode),
        direct_pose_leg_stopgrad_main=bool(direct_pose_leg_stopgrad_main),
        direct_pose_leg_detach_feat=bool(direct_pose_leg_detach_feat),
        direct_pose_leg_max_deg=float(direct_pose_leg_max_deg),
        direct_pose_leg_gate_mode=str(direct_pose_leg_gate_mode),
        direct_pose_leg_gate_power=float(direct_pose_leg_gate_power),
        direct_pose_leg_scale_log_clip=float(direct_pose_leg_scale_log_clip),
        direct_pose_leg_scale_clamp_k=float(direct_pose_leg_scale_clamp_k),
        direct_pose_leg_gate_sup_weight=float(direct_pose_leg_gate_sup_weight),
        direct_pose_leg_scale_sup_weight=float(direct_pose_leg_scale_sup_weight),
        direct_pose_leg_scale_sup_alpha_table_json=str(direct_pose_leg_scale_sup_alpha_table_json),
        direct_pose_leg_scale_sup_log_eps=float(direct_pose_leg_scale_sup_log_eps),
        direct_pose_leg_align_weight=float(direct_pose_leg_align_weight),
        direct_pose_leg_align_oracle_min_deg=float(direct_pose_leg_align_oracle_min_deg),
        direct_pose_leg_align_oracle_weight_deg=float(direct_pose_leg_align_oracle_weight_deg),
        direct_pose_leg_align_mode=str(direct_pose_leg_align_mode),
        direct_pose_leg_align_mag_weight=float(direct_pose_leg_align_mag_weight),
        direct_pose_leg_align_res_weight=float(direct_pose_leg_align_res_weight),
        direct_pose_leg_align_sign_weight=float(direct_pose_leg_align_sign_weight),
        direct_pose_leg_align_cos_thresh=float(direct_pose_leg_align_cos_thresh),
        direct_pose_leg_side_routing=bool(direct_pose_leg_side_routing),
        direct_pose_leg_contact_order=str(direct_pose_leg_contact_order),
        direct_pose_leg_side_embed_dim=int(direct_pose_leg_side_embed_dim),
        direct_pose_leg_side_plan_other=bool(direct_pose_leg_side_plan_other),
        direct_pose_leg_side_phase_other=bool(direct_pose_leg_side_phase_other),
        direct_pose_leg_side_phase_rel=bool(direct_pose_leg_side_phase_rel),
        direct_pose_leg_side_cue=str(direct_pose_leg_side_cue),
        direct_pose_leg_side_cue_tau=float(direct_pose_leg_side_cue_tau),
        direct_pose_leg_side_sign_gate=bool(direct_pose_leg_side_sign_gate),
        direct_pose_leg_side_rank1=bool(direct_pose_leg_side_rank1),
        direct_pose_leg_side_sign_gate_reg_weight=float(direct_pose_leg_side_sign_gate_reg_weight),
        direct_pose_feat_source=str(payload.get("direct_pose_feat_source") or "auto"),
        direct_pose_time_pe_dim=int(direct_pose_time_pe_dim),
        direct_pose_time_pe_base=float(payload.get("direct_pose_time_pe_base") or 10000.0),
        direct_pose_use_phase_z=_as_bool(payload.get("direct_pose_use_phase_z", False), False),
        direct_pose_phase_z_mode=str(payload.get("direct_pose_phase_z_mode") or "concat"),
        direct_pose_split_enable=bool(direct_pose_split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_nonleg_proj_dim),
        direct_pose_nonleg_train_only=bool(direct_pose_nonleg_train_only),
        direct_pose_reinit=_as_bool(payload.get("direct_pose_reinit", False), False),
        direct_pose_hidden_override=direct_pose_hidden_override,
        direct_pose_meas_mode_override=direct_pose_meas_mode_override,
        direct_pose_hinge_enable=direct_pose_hinge_enable,
        direct_pose_hinge_bones=direct_pose_hinge_bones,
        direct_pose_hinge_axis=str(direct_pose_hinge_axis),
        direct_pose_hinge_max_deg=float(direct_pose_hinge_max_deg),
        direct_pose_hinge_hidden=direct_pose_hinge_hidden,
        direct_pose_hinge_feat_source=str(direct_pose_hinge_feat_source),
        direct_pose_hinge_base_feat=str(direct_pose_hinge_base_feat),
        direct_pose_hinge_clean=bool(direct_pose_hinge_clean),
        direct_pose_hinge_eps_max_deg=float(direct_pose_hinge_eps_max_deg),
        direct_pose_hinge_eps_max_scale=float(direct_pose_hinge_eps_max_scale),
        direct_pose_hinge_eps_hidden=direct_pose_hinge_eps_hidden,
        direct_pose_hinge_eps_dropout=float(direct_pose_hinge_eps_dropout),
        direct_pose_hinge_eps_source=str(direct_pose_hinge_eps_source),
        direct_pose_hinge_eps_lr_scale=float(direct_pose_hinge_eps_lr_scale),
        direct_pose_hinge_eps_l2_weight=float(direct_pose_hinge_eps_l2_weight),
        direct_pose_hinge_gate_mode=str(direct_pose_hinge_gate_mode),
        direct_pose_hinge_gate_source=str(direct_pose_hinge_gate_source),
        direct_pose_hinge_gate_power=float(direct_pose_hinge_gate_power),
        direct_pose_hinge_sup_weight=float(direct_pose_hinge_sup_weight),
        direct_pose_hinge_sup_kind=str(direct_pose_hinge_sup_kind),
        direct_pose_hinge_sup_contact_source=str(direct_pose_hinge_sup_contact_source),
        direct_pose_hinge_sup_contact_value=direct_pose_hinge_sup_contact_value,
        direct_pose_hinge_sup_contact_thresh=float(direct_pose_hinge_sup_contact_thresh),
        direct_pose_hinge_sup_angle_thresh_deg=float(direct_pose_hinge_sup_angle_thresh_deg),
        direct_pose_hinge_sup_delta_thresh_deg=float(direct_pose_hinge_sup_delta_thresh_deg),
        direct_pose_hinge_sup_delta_weight_power=float(direct_pose_hinge_sup_delta_weight_power),
        direct_pose_hinge_sup_delta_weight_scale_deg=float(direct_pose_hinge_sup_delta_weight_scale_deg),
        direct_pose_hinge_sup_delta_weight_max=float(direct_pose_hinge_sup_delta_weight_max),
        direct_pose_hinge_gate_sup_weight=float(payload.get("direct_pose_hinge_gate_sup_weight") or 0.0),
        direct_pose_hinge_gate_sup_contact_source=str(payload.get("direct_pose_hinge_gate_sup_contact_source") or "gt"),
        direct_pose_hinge_gate_sup_contact_thresh=float(payload.get("direct_pose_hinge_gate_sup_contact_thresh") or 0.5),
        direct_pose_hinge_stance_weight=float(direct_pose_hinge_stance_weight),
        direct_pose_hinge_stance_kind=str(direct_pose_hinge_stance_kind),
        direct_pose_hinge_stance_contact_source=str(direct_pose_hinge_stance_contact_source),
        direct_pose_hinge_stance_contact_thresh=float(direct_pose_hinge_stance_contact_thresh),
        direct_pose_hinge_reg_weight=float(direct_pose_hinge_reg_weight),
        direct_pose_hinge_reg_kind=str(direct_pose_hinge_reg_kind),
        direct_pose_loss_tail_mix=float(direct_pose_loss_tail_mix),
        direct_pose_loss_tail_temp_deg=float(direct_pose_loss_tail_temp_deg),
        direct_pose_loss_state_swing_boost=float(direct_pose_loss_state_swing_boost),
        direct_pose_loss_state_contact_source=str(direct_pose_loss_state_contact_source),
        direct_pose_loss_state_scope=str(direct_pose_loss_state_scope),
        direct_pose_loss_leg_split=bool(direct_pose_loss_leg_split),
        direct_pose_loss_leg_tail_scale=str(direct_pose_loss_leg_tail_scale),
        direct_pose_loss_sics=direct_pose_loss_sics,
        direct_pose_loss_cycle_gte=int(direct_pose_loss_cycle_gte),
        direct_pose_loss_sic_mode=str(direct_pose_loss_sic_mode),
        direct_pose_loss_sic_boost=float(direct_pose_loss_sic_boost),
        direct_pose_loss_pair_boost_table_json=str(direct_pose_loss_pair_boost_table_json),
        direct_pose_loss_pair_boost=float(direct_pose_loss_pair_boost),
        direct_pose_loss_group_norm_enable=bool(direct_pose_loss_group_norm_enable),
        direct_pose_loss_group_norm_w_leg=float(direct_pose_loss_group_norm_w_leg),
        direct_pose_loss_group_norm_w_nonleg=float(direct_pose_loss_group_norm_w_nonleg),
        direct_pose_loss_group_norm_ema_beta=float(direct_pose_loss_group_norm_ema_beta),
        direct_pose_loss_group_norm_ratio_min=float(direct_pose_loss_group_norm_ratio_min),
        direct_pose_loss_group_norm_ratio_max=float(direct_pose_loss_group_norm_ratio_max),
        direct_pose_loss_group_norm_eps=float(direct_pose_loss_group_norm_eps),
        direct_pose_grad_monitor_enable=bool(direct_pose_grad_monitor_enable),
        direct_pose_grad_ratio_gate=float(direct_pose_grad_ratio_gate),
        lambda_fusion_mode=str(payload.get("lambda_fusion_mode") or "per_joint"),
        lambda_fusion_hidden=int(payload.get("lambda_fusion_hidden") or 128),
        lambda_fusion_dropout=float(payload.get("lambda_fusion_dropout") or 0.0),
        lambda_fusion_logit_init=float(payload.get("lambda_fusion_logit_init") or -2.0),
        lambda_fusion_use_rollout_step=_as_bool(payload.get("lambda_fusion_use_rollout_step", False), False),
        lambda_fusion_entropy_weight=float(payload.get("lambda_fusion_entropy_weight") or 0.0),
        lambda_fusion_smooth_weight=float(payload.get("lambda_fusion_smooth_weight") or 0.0),
        lambda_fusion_early_steps=int(payload.get("lambda_fusion_early_steps") or 0),
        lambda_fusion_early_weight=float(payload.get("lambda_fusion_early_weight") or 0.0),
        lambda_fusion_monotonic_weight=float(payload.get("lambda_fusion_monotonic_weight") or 0.0),
        lambda_plan_entropy_weight=float(payload.get("lambda_plan_entropy_weight") or 0.0),
        lambda_plan_dyn_weight=float(payload.get("lambda_plan_dyn_weight") or 0.0),
        lambda_time_weight_mode=str(payload.get("lambda_time_weight_mode") or "inv"),
        lambda_time_weight_max=float(payload.get("lambda_time_weight_max") or 2.0),
        lambda_reliability_mode=str(payload.get("lambda_reliability_mode") or "none"),
        lambda_reliability_warmup_steps=int(payload.get("lambda_reliability_warmup_steps") or 0),
        lambda_reliability_contact_err_max=float(payload.get("lambda_reliability_contact_err_max") or 1.0),
        lambda_reliability_warmup_joint_scales=_as_float_list(payload.get("lambda_reliability_warmup_joint_scales")),
        lambda_l2sp_weight=float(payload.get("lambda_l2sp_weight") or 0.0),
        lambda_boundary_weight=float(payload.get("lambda_boundary_weight") or 0.0),
        lambda_gate_sup_weight=gate_sup_weight,
        lambda_gate_sup_tau_deg=gate_sup_tau_deg,
        lambda_gate_sup_margin_deg=gate_sup_margin_deg,
        lambda_gate_sup_start_step=gate_sup_start_step,
        train_contact_meas=_as_bool(payload.get("train_contact_meas", False), False),
        contact_meas_weight=float(payload.get("contact_meas_weight") or 0.0),
        contact_meas_rollout=_as_bool(payload.get("contact_meas_rollout", False), False),
        contact_meas_smooth_weight=float(payload.get("contact_meas_smooth_weight") or 0.0),
        contact_meas_smooth_kind=str(payload.get("contact_meas_smooth_kind") or "l1"),
        contact_meas_margin_weight=float(payload.get("contact_meas_margin_weight") or 0.0),
        contact_meas_margin_logit=float(payload.get("contact_meas_margin_logit") or 0.0),
        contact_meas_transition_band=float(payload.get("contact_meas_transition_band") or 0.0),
        contact_meas_rollout_weight=float(payload.get("contact_meas_rollout_weight") or 0.0),
        train_contact_td_hazard=_as_bool(payload.get("train_contact_td_hazard", False), False),
        contact_td_hazard_bce_weight=float(payload.get("contact_td_hazard_bce_weight") or 0.0),
        contact_td_hazard_event_weight=float(payload.get("contact_td_hazard_event_weight") or 0.0),
        contact_td_hazard_mass_weight=float(payload.get("contact_td_hazard_mass_weight") or 0.0),
        contact_td_hazard_unimodal_weight=float(payload.get("contact_td_hazard_unimodal_weight") or 0.0),
        contact_td_hazard_entropy_weight=float(payload.get("contact_td_hazard_entropy_weight") or 0.0),
        contact_td_hazard_clock_weight=float(payload.get("contact_td_hazard_clock_weight") or 0.0),
        contact_td_hazard_rollout=_as_bool(payload.get("contact_td_hazard_rollout", True), True),
        contact_td_hazard_rollout_weight=float(payload.get("contact_td_hazard_rollout_weight") or 0.0),
        contact_td_hazard_hidden=int(payload.get("contact_td_hazard_hidden") or 64),
        contact_td_hazard_dropout=float(payload.get("contact_td_hazard_dropout") or 0.0),
        direct_pose_meas_force_zero=_as_bool(payload.get("direct_pose_meas_force_zero", False), False),
        direct_pose_meas_detach=_as_bool(payload.get("direct_pose_meas_detach", False), False),
        contact_meas_gate_by_hit=str(payload.get("contact_meas_gate_by_hit") or "auto"),
        contact_meas_vxy_mode=str(payload.get("contact_meas_vxy_mode") or "abs"),
        contact_meas_ground_z_mode=str(payload.get("contact_meas_ground_z_mode") or "window"),
        contact_meas_ground_z_beta=float(payload.get("contact_meas_ground_z_beta") or 0.05),
        contact_meas_ground_z_window=int(payload.get("contact_meas_ground_z_window") or 5),
        contact_meas_ground_z_quantile=float(payload.get("contact_meas_ground_z_quantile") or 0.2),
        contact_meas_ground_z_slew_up_cm=float(payload.get("contact_meas_ground_z_slew_up_cm") or 0.0),
        contact_meas_ground_z_slew_down_cm=float(payload.get("contact_meas_ground_z_slew_down_cm") or 0.0),
        seed=int(payload.get("seed") or 0),
    )


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
    spec = {}
    try:
        with bundle_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception:
        spec = {}
    if pretrain_path is not None and pretrain_path.is_file():
        try:
            with pretrain_path.open("r", encoding="utf-8") as f:
                extra = json.load(f)
            if isinstance(extra, dict):
                spec = dict(extra, **spec)
        except Exception:
            pass
    return spec


def _prepare_omega(
    model: EventMotionModel,
    omega_hat: torch.Tensor,
    *,
    gate_force: Optional[float],
    max_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if omega_hat.dim() == 4:
        omega_hat = omega_hat[:, -1]
    if omega_hat.dim() != 3 or omega_hat.shape[-1] != 3:
        raise ValueError(f"omega_hat must be (B,J,3) or (B,1,J,3); got {tuple(omega_hat.shape)}")

    if gate_force is not None:
        gate = omega_hat.new_tensor(float(gate_force))
    else:
        logit = getattr(model, "so3_corr_gate_logit", None)
        if logit is None or (not torch.is_tensor(logit)):
            gate = omega_hat.new_tensor(0.0)
        else:
            gate = torch.sigmoid(logit.to(device=omega_hat.device, dtype=omega_hat.dtype))
    omega = omega_hat * gate

    max_deg = float(max_deg or 0.0)
    if max_deg > 0.0:
        max_rad = omega.new_tensor(max_deg * (math.pi / 180.0))
        n = omega.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = (max_rad / n).clamp(max=1.0)
        omega = omega * s
    return omega, gate

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


def _corr_loss_rollout(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    columns: Tuple[str, str],
    gate_force: Optional[float],
    max_deg: float,
    omega_l2_weight: float,
    rollout_steps: int,
    rollout_cycles: int,
    time_index_mode: str,
    time_weight_max: float,
    detach_rollout_state: bool,
    contact_meas_weight: float = 0.0,
    objective: str = "blend",  # blend|direct|inc (kept for symmetry with lambda posttrain)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = trainer.device
    dtype = next(model.parameters()).dtype

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

    B, T, Dx = motion_seq.shape
    Dy = int(gt_seq.shape[-1])
    if T < 2:
        raise ValueError(f"seq_len must be >=2, got {T}")
    steps = _resolve_rollout_steps(T, rollout_steps)
    steps = max(1, int(steps))
    rollout_cycles = max(1, int(rollout_cycles or 1))
    total_steps = int(steps) * int(rollout_cycles)

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

    motion = motion_seq[:, 0]
    motion_raw = trainer.normalizer.denorm_x(motion)
    y_prev_raw = _init_y_from_x(trainer.normalizer, motion_raw, Dy)

    pose_hist_enabled = bool(getattr(trainer, "pose_hist_len", 0) or 0) > 0 and bool(getattr(trainer, "pose_hist_dim", 0) or 0) > 0
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
    scales = mu = std = None
    pose_hist_buffer_norm = None
    pose_hist_buffer_raw = None
    if pose_hist_enabled and pose_hist_stride > 0:
        try:
            scales, mu, std = trainer._pose_hist_params(motion_seq)
        except Exception:
            scales = mu = std = None
        if scales is None:
            pose_hist_enabled = False
        else:
            with torch.no_grad():
                if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0 and pose_hist_seq.dim() == 3:
                    pose_hist_buffer_norm = pose_hist_seq[:, 0]
                    pose_hist_buffer_raw = trainer._pose_hist_inverse_vec(pose_hist_buffer_norm, scales, mu, std)
                else:
                    base_rot = y_prev_raw[..., rot_slice]
                    pose_hist_buffer_raw = (
                        base_rot.unsqueeze(1)
                        .repeat(1, pose_hist_len, 1)
                        .reshape(B, pose_hist_dim)
                    )
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

    # Cache plan_z across steps to match inference loop (plan is an independent anchor).
    # NOTE: let the model decide the initial plan_z when plan_z is None.
    # This allows using a learnable contact_plan_init_z (or falling back to zeros).
    plan_z = None
    phase_z = None
    phase_event_age = None
    phase_z = None
    phase_event_age = None
    # Cache meas_logits_prev across forward calls so Event-Clock delta_meas is non-zero when unrolling with T=1.
    meas_logits_prev = None
    td_hazard_acc = None  # (B,C) integrate-to-1 accumulator for phase_reset_source=td_hazard

    loss_terms = []
    omega_l2_terms = []
    gate_vals = []
    meas_terms = []
    meas_used_logits = False

    time_weight_max = max(1.0, float(time_weight_max or 1.0))
    time_weights = torch.linspace(1.0, time_weight_max, steps=total_steps, device=device, dtype=dtype)
    prev_foot_pos_meas = None
    time_base = None
    try:
        if isinstance(batch, dict):
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_base = base
    except Exception:
        time_base = None

    time_index_mode = str(time_index_mode or "global").strip().lower()
    if time_index_mode == "auto":
        time_index_mode = "cycle" if rollout_cycles > 1 else "global"
    if time_index_mode not in ("global", "cycle", "none"):
        time_index_mode = "global"
    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))

    for t in range(total_steps):
        idx = int(t % steps)
        cond_t = cond_seq[:, idx] if (torch.is_tensor(cond_seq) and cond_seq.dim() == 3) else cond_seq
        cond_raw_step = None
        if torch.is_tensor(cond_raw_tgt):
            if cond_raw_tgt.dim() == 3:
                idx_raw = min(int(cond_raw_tgt.shape[1]) - 1, int(idx) + 1)
                cond_raw_step = cond_raw_tgt[:, idx_raw]
            else:
                cond_raw_step = cond_raw_tgt
        cond_raw_for_model = cond_raw_step
        if enable_reprojection and t > 0 and torch.is_tensor(cond_raw_step):
            yaw_gt = None
            try:
                gt_idx = min(int(gt_seq.shape[1]) - 1, int(idx))
                gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
                yaw_gt = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
            except Exception:
                yaw_gt = None
            yaw_pred = None
            try:
                yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
            except Exception:
                yaw_pred = None
            if yaw_gt is not None and yaw_pred is not None:
                try:
                    cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
                except Exception:
                    cond_proj = None
                if cond_proj is not None:
                    cond_raw_for_model = cond_proj
        if cond_raw_for_model is not None:
            try:
                cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
            except Exception:
                cond_override = None
            if cond_override is not None:
                cond_t = cond_override
        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, idx] if (torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3) else angvel_seq
        if pose_hist_enabled and pose_hist_buffer_norm is not None:
            pose_hist_t = pose_hist_buffer_norm
        else:
            pose_hist_t = pose_hist_seq[:, idx] if (torch.is_tensor(pose_hist_seq) and pose_hist_seq.dim() == 3) else pose_hist_seq

        inp_motion = motion.unsqueeze(1)
        inp_cond = cond_t.unsqueeze(1) if torch.is_tensor(cond_t) and cond_t.dim() == 2 else cond_t
        inp_angvel = angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) and angvel_t.dim() == 2 else angvel_t
        inp_pose_hist = pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) and pose_hist_t.dim() == 2 else pose_hist_t

        # Keep contacts semantics aligned with training_MPL / inference:
        # - If the model has a learned contact_meas_head, do NOT override it with white-box contacts every step,
        #   otherwise contacts_meas_head receives no gradients and train/infer diverges.
        # - For plan_z0 init in obs-based modes, we still feed a one-step contacts input at t=0 so the init head
        #   can see an observation signal (it uses `contacts_input` directly).
        plan_enable = bool(getattr(model, "contact_plan_enable", False))
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False))
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        contacts_wb_t = None
        if plan_enable and (
            (not use_learned_meas)
            or (init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0)
        ):
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if plan_enable:
            if not use_learned_meas:
                contacts_in_t = contacts_wb_t
            elif init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0:
                contacts_in_t = contacts_wb_t

        if time_index_mode == "none":
            time_index_t = None
        elif time_index_mode == "cycle":
            # Phase index within the sequence; when start is provided, align to absolute clip time.
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    pass
        else:
            # Global time index: prefer absolute clip frame (start + idx) when available.
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    time_index_t = int(idx)
            else:
                time_index_t = int(t)

        rollout_step_t = None
        try:
            if int(total_steps) > 1:
                step_norm = float(t) / float(int(total_steps) - 1)
            else:
                step_norm = 0.0
            rollout_step_t = torch.full((B, 1, 1), step_norm, device=device, dtype=dtype)
        except Exception:
            rollout_step_t = None

        ret = model(
            inp_motion,
            inp_cond,
            contacts=contacts_in_t,
            angvel=inp_angvel,
            pose_history=inp_pose_hist,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            td_hazard_acc=td_hazard_acc,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index_t,
            rollout_step=rollout_step_t,
        )
        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict with keys {out, omega_hat}.")
        delta_norm = ret.get("out")
        omega_hat = ret.get("omega_hat")
        if delta_norm is None or omega_hat is None:
            raise RuntimeError("Model dict output missing 'out' or 'omega_hat'.")
        if delta_norm.dim() == 3:
            delta_norm = delta_norm[:, -1]

        if bool(getattr(model, "contact_plan_enable", False)):
            try:
                z_next = ret.get("plan_z_next", None)
                if torch.is_tensor(z_next):
                    plan_z = z_next.detach()
                p_next = ret.get("phase_z_next", None)
                if torch.is_tensor(p_next):
                    phase_z = p_next.detach()
                a_next = ret.get("phase_event_age_next", None)
                if torch.is_tensor(a_next):
                    phase_event_age = a_next.detach()
            except Exception:
                pass
        try:
            hz_acc_next = ret.get("td_hazard_acc_next", None)
            if torch.is_tensor(hz_acc_next):
                td_hazard_acc = hz_acc_next.detach()
        except Exception:
            pass
        try:
            mlog = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(mlog):
                if mlog.dim() == 3:
                    meas_logits_prev = mlog[:, -1].detach()
                elif mlog.dim() == 2:
                    meas_logits_prev = mlog.detach()
        except Exception:
            pass

        # Optional: supervise contact_meas_head against GT soft contacts.
        # NOTE: We do NOT feed GT contacts into the model; this only trains the meas head to be a
        # pose-derived estimator, so e_t = plan - meas stays meaningful in inference.
        if contacts_in_t is None and float(contact_meas_weight or 0.0) > 0.0 and torch.is_tensor(contacts_seq):
            try:
                gt_c_t = contacts_seq[:, idx] if contacts_seq.dim() == 3 else contacts_seq
                meas_logits = ret.get("contacts_meas_logits", None)
                if torch.is_tensor(meas_logits):
                    if meas_logits.dim() == 3:
                        meas_logits = meas_logits[:, -1]
                    if torch.is_tensor(gt_c_t) and gt_c_t.shape == meas_logits.shape:
                        meas_used_logits = True
                        gt = gt_c_t.clamp(0.0, 1.0)
                        meas_terms.append(F.binary_cross_entropy_with_logits(meas_logits, gt) * time_weights[t])
                else:
                    # If logits are missing, skip (avoid supervising contacts_input/white-box fallback).
                    pass
            except Exception:
                pass

        delta_raw = delta_norm * std_y

        prev6 = reproject_rot6d(y_prev_raw[..., rot_slice]).view(B, J, 6)
        R_prev = rot6d_to_matrix(prev6, columns=columns)

        gt_raw = trainer._denorm(gt_seq[:, idx])
        gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(B, J, 6)
        R_gt = rot6d_to_matrix(gt6, columns=columns)

        dR_target = torch.matmul(R_gt, R_prev.transpose(-1, -2))

        delta6 = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=columns)  # (B,J,6)
        R_delta_pred = rot6d_to_matrix(delta6, columns=columns)

        R_err = torch.matmul(dR_target, R_delta_pred.transpose(-1, -2))
        omega_used, gate = _prepare_omega(
            model,
            omega_hat,
            gate_force=gate_force,
            max_deg=max_deg,
        )
        gate_vals.append(gate.detach())

        R_corr = so3_exp_map(omega_used)
        corr_geo = _geodesic_R_safe(R_corr, R_err)  # (B,J)
        corr_loss = corr_geo.mean()
        loss_terms.append(corr_loss * time_weights[t])

        omega_norm = omega_used.norm(dim=-1)
        omega_l2_terms.append((omega_norm * omega_norm).mean() * time_weights[t])

        R_delta_used = torch.matmul(R_corr, R_delta_pred)
        R_next = torch.matmul(R_delta_used, R_prev)
        rot_next6d = matrix_to_rot6d(R_next, columns=columns).view(B, rot_len)
        y_next_raw = y_prev_raw + delta_raw
        y_next_raw = y_next_raw.clone()
        y_next_raw[..., rot_slice] = rot_next6d

        if detach_rollout_state:
            y_next_raw = y_next_raw.detach()

        if t < total_steps - 1:
            cond_env = None
            if torch.is_tensor(cond_raw_step):
                cond_env = cond_raw_step
            motion_raw = trainer._apply_free_carry(motion_raw, y_next_raw, cond_next_raw=cond_env)
            motion_raw = _finite(motion_raw)
            motion = trainer._diag_norm_x(motion_raw)

            if pose_hist_enabled and pose_hist_buffer_raw is not None and pose_hist_stride > 0:
                with torch.no_grad():
                    pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_next_raw[..., rot_slice]
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

            y_prev_raw = y_next_raw

    denom = time_weights.sum().clamp_min(1e-6)
    corr_loss_total = torch.stack(loss_terms).sum() / denom
    omega_l2 = torch.stack(omega_l2_terms).sum() / denom if omega_l2_terms else corr_loss_total.new_tensor(0.0)
    total = corr_loss_total + float(omega_l2_weight or 0.0) * omega_l2
    contact_meas_loss = None
    if meas_terms:
        contact_meas_loss = torch.stack(meas_terms).sum() / denom
        total = total + float(contact_meas_weight or 0.0) * contact_meas_loss

    gate_mean = None
    try:
        gate_mean = float(torch.stack([g.reshape(-1) for g in gate_vals]).mean().detach().cpu())
    except Exception:
        gate_mean = None

    stats = {
        "corr_loss": float(corr_loss_total.detach().cpu()),
        "omega_l2": float(omega_l2.detach().cpu()),
        "gate_mean": float(gate_mean) if gate_mean is not None else float("nan"),
        "total": float(total.detach().cpu()),
    }
    if contact_meas_loss is not None:
        if bool(meas_used_logits):
            stats["contact_meas_bce"] = float(contact_meas_loss.detach().cpu())
        else:
            stats["contact_meas_mse"] = float(contact_meas_loss.detach().cpu())
        stats["contact_meas_weighted"] = float((float(contact_meas_weight or 0.0) * contact_meas_loss).detach().cpu())
    return total, stats


def _lambda_entropy(p: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def _contact_plan_init_loss_teacher(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    time_index_mode: str = "none",
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contacts_plan against GT soft contacts in teacher mode.

    Intended usage: fine-tune contact_plan_init_z only to reduce plan cold-start,
    without touching the GRU weights.
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion", None)
    cond_seq = batch.get("cond_in", None)
    gt_contacts = batch.get("contacts", None)
    angvel_seq = batch.get("angvel", None)
    pose_hist_seq = batch.get("pose_hist", None)
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_contacts)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / contacts.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_contacts = gt_contacts.to(device=device, dtype=dtype)
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    # If we use obs-conditioned init (plan_z0 = f(obs0)), feed a loop-closed contacts_meas signal
    # derived from the current pose (white-box meas) so init_head can see the same anchor used in inference.
    contacts_in = None
    try:
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").lower().strip()
        if init_mode in ("obs", "learnable+obs"):
            x0 = motion_seq[:, :1] if motion_seq.dim() == 3 else motion_seq  # (B,1,Dx) or (B,Dx)
            if x0.dim() == 3 and x0.size(1) == 1:
                x0 = x0[:, 0]
            x0_raw = trainer.normalizer.denorm_x(x0) if getattr(trainer, "normalizer", None) is not None else None
            if torch.is_tensor(x0_raw):
                contacts_in, _ = trainer._contact_meas_whitebox(x0_raw, None)
    except Exception:
        contacts_in = None

    time_index_mode = str(time_index_mode or "none").strip().lower()
    time_index = None
    # Default to "cycle-like" time (arange within window) to keep time-PE in-range.
    if time_index_mode == "global":
        try:
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_index = base
        except Exception:
            time_index = None
    elif time_index_mode in ("none", "cycle", "auto"):
        time_index = None

    ret = model(
        motion_seq,
        cond_seq,
        contacts=contacts_in,  # loop-closed white-box meas for obs-conditioned init (or None)
        angvel=angvel_seq,
        pose_history=pose_hist_seq,
        plan_z=None,
        time_index=time_index,
        rollout_step=None,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict.")
    plan = ret.get("contacts_plan", None)
    if not torch.is_tensor(plan):
        raise RuntimeError("Model output missing contacts_plan.")
    if plan.dim() == 2:
        plan = plan.unsqueeze(1)
    if gt_contacts.dim() == 2:
        gt_contacts = gt_contacts.unsqueeze(1)
    T = min(int(plan.shape[1]), int(gt_contacts.shape[1]))
    if T <= 0 or plan.shape[-1] != gt_contacts.shape[-1]:
        raise RuntimeError(f"contacts_plan shape {tuple(plan.shape)} vs GT {tuple(gt_contacts.shape)} mismatch.")

    # Metric (for continuity): MSE on (L,R) marginals.
    mse_all = F.mse_loss(plan[:, :T], gt_contacts[:, :T])
    mse_early = None
    try:
        k = min(10, T)
        if k > 0:
            mse_early = F.mse_loss(plan[:, :k], gt_contacts[:, :k])
    except Exception:
        mse_early = None

    # Training loss: prefer logits-space supervision (sharper / less "regress to mean").
    logits = ret.get("contacts_plan_logits", None)
    if not torch.is_tensor(logits):
        raise RuntimeError("Missing 'contacts_plan_logits' in model output; re-export/retrain with contact_plan logits.")
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    if logits.dim() == 3:
        logits = logits[:, :T]
    gt = gt_contacts[:, :T].clamp(0.0, 1.0)
    if logits.shape[-1] != gt.shape[-1]:
        raise RuntimeError(f"contacts_plan_logits dim mismatch: logits={tuple(logits.shape)} gt={tuple(gt.shape)}")
    # Independent per-foot Bernoulli (sigmoid head).
    plan_loss = F.binary_cross_entropy_with_logits(logits, gt)
    plan_loss_early = None
    try:
        k = min(10, T)
        if k > 0:
            plan_loss_early = F.binary_cross_entropy_with_logits(logits[:, :k], gt[:, :k])
    except Exception:
        plan_loss_early = None

    w = float(weight or 0.0)
    loss = plan_loss * w
    stats = {
        "contact_plan_mse": float(mse_all.detach().cpu()),
        "contact_plan_loss": float(plan_loss.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    if mse_early is not None:
        stats["contact_plan_mse_early10"] = float(mse_early.detach().cpu())
    if plan_loss_early is not None:
        stats["contact_plan_loss_early10"] = float(plan_loss_early.detach().cpu())
    return loss, stats


def _contact_plan_loss_teacher(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    time_index_mode: str = "none",
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contacts_plan against GT soft contacts in teacher mode.

    Intended usage: fine-tune contact_plan dynamics (GRU + heads) so contacts_plan becomes a
    useful phase/anchor proxy for direct head and Stage2 reliability, without relying on fixed phase labels.
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion", None)
    cond_seq = batch.get("cond_in", None)
    gt_contacts = batch.get("contacts", None)
    angvel_seq = batch.get("angvel", None)
    pose_hist_seq = batch.get("pose_hist", None)
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_contacts)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / contacts.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_contacts = gt_contacts.to(device=device, dtype=dtype)
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    # Align inference: if init uses obs features, feed loop-closed white-box meas at t=0
    # (do NOT leak GT contacts into init).
    contacts_in = None
    try:
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").lower().strip()
        if init_mode in ("obs", "learnable+obs"):
            x0 = motion_seq[:, :1] if motion_seq.dim() == 3 else motion_seq  # (B,1,Dx) or (B,Dx)
            if x0.dim() == 3 and x0.size(1) == 1:
                x0 = x0[:, 0]
            x0_raw = trainer.normalizer.denorm_x(x0) if getattr(trainer, "normalizer", None) is not None else None
            if torch.is_tensor(x0_raw):
                contacts_in, _ = trainer._contact_meas_whitebox(x0_raw, None)
    except Exception:
        contacts_in = None

    time_index_mode = str(time_index_mode or "none").strip().lower()
    time_index = None
    if time_index_mode == "global":
        try:
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_index = base
        except Exception:
            time_index = None
    elif time_index_mode in ("none", "cycle", "auto"):
        time_index = None

    ret = model(
        motion_seq,
        cond_seq,
        contacts=contacts_in,
        angvel=angvel_seq,
        pose_history=pose_hist_seq,
        plan_z=None,
        time_index=time_index,
        rollout_step=None,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict.")
    plan = ret.get("contacts_plan", None)
    if not torch.is_tensor(plan):
        raise RuntimeError("Model output missing contacts_plan.")
    if plan.dim() == 2:
        plan = plan.unsqueeze(1)
    if gt_contacts.dim() == 2:
        gt_contacts = gt_contacts.unsqueeze(1)

    T = min(int(plan.shape[1]), int(gt_contacts.shape[1]))
    if T <= 0 or plan.shape[-1] != gt_contacts.shape[-1]:
        raise RuntimeError(f"contacts_plan shape {tuple(plan.shape)} vs GT {tuple(gt_contacts.shape)} mismatch.")

    # Metric (for continuity): MSE on (L,R) marginals.
    mse_all = F.mse_loss(plan[:, :T], gt_contacts[:, :T])
    mse_early = None
    try:
        k = min(10, T)
        if k > 0:
            mse_early = F.mse_loss(plan[:, :k], gt_contacts[:, :k])
    except Exception:
        mse_early = None

    # Training loss: prefer logits-space supervision (sharper / less "regress to mean").
    logits = ret.get("contacts_plan_logits", None)
    if not torch.is_tensor(logits):
        raise RuntimeError("Missing 'contacts_plan_logits' in model output; re-export/retrain with contact_plan logits.")
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    if logits.dim() == 3:
        logits = logits[:, :T]
    gt = gt_contacts[:, :T].clamp(0.0, 1.0)
    if logits.shape[-1] != gt.shape[-1]:
        raise RuntimeError(f"contacts_plan_logits dim mismatch: logits={tuple(logits.shape)} gt={tuple(gt.shape)}")
    # Independent per-foot Bernoulli (sigmoid head).
    plan_loss = F.binary_cross_entropy_with_logits(logits, gt)
    plan_loss_early = None
    try:
        k = min(10, T)
        if k > 0:
            plan_loss_early = F.binary_cross_entropy_with_logits(logits[:, :k], gt[:, :k])
    except Exception:
        plan_loss_early = None

    w = float(weight or 0.0)
    loss = plan_loss * w
    stats = {
        "contact_plan_mse": float(mse_all.detach().cpu()),
        "contact_plan_loss": float(plan_loss.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    if mse_early is not None:
        stats["contact_plan_mse_early10"] = float(mse_early.detach().cpu())
    if plan_loss_early is not None:
        stats["contact_plan_loss_early10"] = float(plan_loss_early.detach().cpu())
    try:
        stats["contact_plan_mean"] = float(plan[:, :T].mean().detach().cpu())
        stats["contact_plan_std"] = float(plan[:, :T].std().detach().cpu())
    except Exception:
        pass
    return loss, stats


def _contact_meas_loss_teacher(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    time_index_mode: str = "none",
    weight: float = 1.0,
    smooth_weight: float = 0.0,
    smooth_kind: str = "l1",
    margin_weight: float = 0.0,
    margin_logit: float = 0.0,
    transition_band: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contacts_meas_head against GT soft contacts in teacher mode.

    Important: do NOT feed contacts into the model (no white-box override, no GT leakage).
    This trains the meas head as a pose-derived estimator so e_t = plan - meas remains meaningful in inference.
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion", None)
    cond_seq = batch.get("cond_in", None)
    gt_contacts = batch.get("contacts", None)
    angvel_seq = batch.get("angvel", None)
    pose_hist_seq = batch.get("pose_hist", None)
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_contacts)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / contacts.")

    if not bool(getattr(model, "contact_meas_enable", False)) or getattr(model, "contact_meas_head", None) is None:
        raise RuntimeError("Model has no contact_meas_head enabled; cannot train contact_meas.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_contacts = gt_contacts.to(device=device, dtype=dtype)
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    time_index_mode = str(time_index_mode or "none").strip().lower()
    time_index = None
    if time_index_mode == "global":
        try:
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_index = base
        except Exception:
            time_index = None
    elif time_index_mode in ("none", "cycle", "auto"):
        time_index = None

    ret = model(
        motion_seq,
        cond_seq,
        contacts=None,  # critical: train meas_head, do NOT override contacts_meas
        angvel=angvel_seq,
        pose_history=pose_hist_seq,
        plan_z=None,
        time_index=time_index,
        rollout_step=None,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict.")

    logits = ret.get("contacts_meas_logits", None)
    meas = ret.get("contacts_meas", None)
    if not torch.is_tensor(meas):
        raise RuntimeError("Model output missing contacts_meas.")
    if meas.dim() == 2:
        meas = meas.unsqueeze(1)
    if gt_contacts.dim() == 2:
        gt_contacts = gt_contacts.unsqueeze(1)
    T = min(int(meas.shape[1]), int(gt_contacts.shape[1]))
    if T <= 0 or meas.shape[-1] != gt_contacts.shape[-1]:
        raise RuntimeError(f"contacts_meas shape {tuple(meas.shape)} vs GT {tuple(gt_contacts.shape)} mismatch.")

    mse_all = F.mse_loss(meas[:, :T], gt_contacts[:, :T])

    w = float(weight or 0.0)
    if torch.is_tensor(logits):
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits[:, :T]
        gt = gt_contacts[:, :T].clamp(0.0, 1.0)
        if logits.shape[-1] != gt.shape[-1]:
            raise RuntimeError(f"contacts_meas_logits dim mismatch: logits={tuple(logits.shape)} gt={tuple(gt.shape)}")
        meas_bce = F.binary_cross_entropy_with_logits(logits, gt)
        loss_unweighted = meas_bce

        smooth_weight = float(smooth_weight or 0.0)
        margin_weight = float(margin_weight or 0.0)
        smooth = None
        margin = None
        non_transition_frac = None

        band = float(transition_band or 0.0)
        if not math.isfinite(band):
            band = 0.0
        band = max(0.0, min(0.49, band))
        if band <= 0.0:
            mask = torch.ones_like(gt, dtype=torch.bool)
        else:
            low = float(0.5 - band)
            high = float(0.5 + band)
            mask = (gt <= low) | (gt >= high)
        mask_f = mask.to(dtype=logits.dtype)
        try:
            non_transition_frac = float(mask_f.mean().detach().cpu())
        except Exception:
            non_transition_frac = None

        if margin_weight > 0.0:
            m = float(margin_logit or 0.0)
            if not math.isfinite(m):
                m = 0.0
            m = max(0.0, m)
            if m > 0.0:
                margin_elem = F.relu(m - logits.abs())
                denom = mask_f.sum().clamp_min(1.0)
                margin = (margin_elem * mask_f).sum() / denom
                loss_unweighted = loss_unweighted + margin_weight * margin

        if smooth_weight > 0.0 and logits.shape[1] >= 2:
            d = logits[:, 1:] - logits[:, :-1]
            kind = str(smooth_kind or "l1").strip().lower()
            if kind == "l2":
                d_elem = d.pow(2)
            elif kind in ("smooth_l1", "huber"):
                d_elem = F.smooth_l1_loss(d, torch.zeros_like(d), reduction="none")
            else:
                d_elem = d.abs()
            mask_pair = (mask[:, 1:] & mask[:, :-1]).to(dtype=logits.dtype)
            denom = mask_pair.sum().clamp_min(1.0)
            smooth = (d_elem * mask_pair).sum() / denom
            loss_unweighted = loss_unweighted + smooth_weight * smooth

        loss = loss_unweighted * w
        stats = {
            "contact_meas_mse": float(mse_all.detach().cpu()),
            "contact_meas_bce": float(meas_bce.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
        if smooth is not None:
            stats["contact_meas_smooth"] = float(smooth.detach().cpu())
        if margin is not None:
            stats["contact_meas_margin"] = float(margin.detach().cpu())
        if non_transition_frac is not None:
            stats["contact_meas_non_transition_frac"] = float(non_transition_frac)
    else:
        loss = mse_all * w
        stats = {
            "contact_meas_mse": float(mse_all.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
    return loss, stats


def _contact_meas_loss_rollout(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    rollout_steps: int,
    rollout_cycles: int,
    include_boundary: bool,
    boundary_weight: float,
    random_offset: bool,
    time_index_mode: str,
    time_weight_max: float,
    detach_rollout_state: bool,
    weight: float = 1.0,
    smooth_weight: float = 0.0,
    smooth_kind: str = "l1",
    margin_weight: float = 0.0,
    margin_logit: float = 0.0,
    transition_band: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contacts_meas_head against GT soft contacts while unrolling the model in a
    closed-loop rollout (freerun-like) to expose the meas head to drift/OOD inputs.

    Notes:
      - Does NOT feed GT contacts into the model (no leakage). If contact_meas_head is absent,
        this raises.
      - Uses stop-gradient state carry by default (detach_rollout_state) to keep training stable.
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion")
    cond_seq = batch.get("cond_in")
    gt_contacts = batch.get("contacts")
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_contacts)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / contacts.")

    if not bool(getattr(model, "contact_meas_enable", False)) or getattr(model, "contact_meas_head", None) is None:
        raise RuntimeError("Model has no contact_meas_head enabled; cannot train contact_meas.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)  # (B,T,Dx)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_contacts = gt_contacts.to(device=device, dtype=dtype)

    # Optional: GT touchdown events/valid mask (used only for phase_reset_source='ttc_gt' in rollout).
    ttc_valid_seq = batch.get("ttc_td_valid")
    if torch.is_tensor(ttc_valid_seq):
        ttc_valid_seq = ttc_valid_seq.to(device=device)
    ttc_events_seq = batch.get("ttc_td_events")
    if torch.is_tensor(ttc_events_seq):
        ttc_events_seq = ttc_events_seq.to(device=device)

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
    cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, motion_seq) if cond_norm_mu is not None else None
    cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, motion_seq) if cond_norm_std is not None else None

    angvel_seq = batch.get("angvel")
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    pose_hist_seq = batch.get("pose_hist")
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    B, T, _ = motion_seq.shape
    if T < 2:
        raise ValueError(f"seq_len must be >=2, got {T}")
    steps = _resolve_rollout_steps(T, rollout_steps)
    steps = max(1, int(steps))
    rollout_cycles = max(1, int(rollout_cycles or 1))
    include_boundary = bool(include_boundary) and int(rollout_cycles) > 1 and int(steps) == int(T - 1)
    cycle_len = int(T) if include_boundary else int(steps)
    total_steps = (int(rollout_cycles) * int(cycle_len) - 1) if include_boundary else (int(steps) * int(rollout_cycles))

    # Optional random phase offset (only meaningful when unrolling multiple cycles).
    offset = 0
    if bool(random_offset) and int(rollout_cycles) > 1 and int(cycle_len) > 1:
        try:
            offset = int(torch.randint(low=0, high=int(cycle_len), size=(1,), device="cpu").item())
        except Exception:
            offset = 0

    Dy = int(getattr(trainer, "Dy", 0) or 0)
    if Dy <= 0:
        # Fallback: infer from normalizer / output_layout in trainer when available.
        try:
            Dy = int(getattr(trainer.normalizer, "mu_y", np.zeros((0,), dtype=np.float32)).shape[-1])
        except Exception:
            Dy = 0
    if Dy <= 0:
        raise RuntimeError("Cannot infer Dy for rollout carry; Trainer.Dy is missing.")

    motion = motion_seq[:, int(offset)]
    motion_raw = trainer.normalizer.denorm_x(motion)
    y_prev_raw = _init_y_from_x(trainer.normalizer, motion_raw, Dy)

    pose_hist_enabled = bool(getattr(trainer, "pose_hist_len", 0) or 0) > 0 and bool(getattr(trainer, "pose_hist_dim", 0) or 0) > 0
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
    scales = mu = std = None
    pose_hist_buffer_norm = None
    pose_hist_buffer_raw = None
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if pose_hist_enabled and pose_hist_stride > 0:
        try:
            scales, mu, std = trainer._pose_hist_params(motion_seq)
        except Exception:
            scales = mu = std = None
        if scales is None:
            pose_hist_enabled = False
        else:
            with torch.no_grad():
                if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0 and pose_hist_seq.dim() == 3:
                    pose_hist_buffer_norm = pose_hist_seq[:, int(offset)]
                    pose_hist_buffer_raw = trainer._pose_hist_inverse_vec(pose_hist_buffer_norm, scales, mu, std)
                else:
                    if not isinstance(rot_slice, slice):
                        raise RuntimeError("pose_hist enabled but rot slice missing for init.")
                    base_rot = y_prev_raw[..., rot_slice]
                    pose_hist_buffer_raw = base_rot.unsqueeze(1).repeat(1, pose_hist_len, 1).reshape(B, pose_hist_dim)
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

    plan_z = None
    phase_z = None
    phase_event_age = None
    meas_logits_prev = None
    td_hazard_acc = None  # (B,C) stateful when phase_reset_source=td_hazard

    time_weight_max = max(1.0, float(time_weight_max or 1.0))
    time_weights = torch.linspace(1.0, time_weight_max, steps=total_steps, device=device, dtype=dtype)
    if include_boundary:
        bw = float(boundary_weight or 0.0)
        bw = max(0.0, bw)
        if abs(bw - 1.0) > 1e-12:
            try:
                idxs = (torch.arange(int(total_steps), device=device) + int(offset)) % int(cycle_len)
                boundary_mask = idxs == (int(cycle_len) - 1)
                factors = torch.ones_like(time_weights)
                factors = torch.where(boundary_mask, time_weights.new_tensor(bw), factors)
                time_weights = time_weights * factors
            except Exception:
                pass
    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))

    time_base = None
    try:
        base = batch.get("start", None)
        if base is not None:
            if torch.is_tensor(base):
                base = base.to(device=device)
            time_base = base
    except Exception:
        time_base = None

    time_index_mode = str(time_index_mode or "global").strip().lower()
    if time_index_mode == "auto":
        time_index_mode = "cycle" if rollout_cycles > 1 else "global"
    if time_index_mode not in ("global", "cycle", "none"):
        time_index_mode = "global"
    # Note: do not pre-add offset to time_base here; time_index uses idx (which already includes offset).

    meas_terms: list[torch.Tensor] = []
    smooth_terms: list[torch.Tensor] = []
    margin_terms: list[torch.Tensor] = []
    non_transition_fracs: list[torch.Tensor] = []
    prev_foot_pos_meas = None
    prev_logits = None
    prev_mask = None

    for t in range(total_steps):
        idx = int((int(offset) + int(t)) % int(cycle_len))

        cond_t = cond_seq[:, idx] if cond_seq.dim() == 3 else cond_seq
        cond_raw_step = None
        if torch.is_tensor(cond_raw_tgt):
            if cond_raw_tgt.dim() == 3:
                if include_boundary:
                    idx_raw = int((int(idx) + 1) % int(cycle_len))
                else:
                    idx_raw = min(int(cond_raw_tgt.shape[1]) - 1, int(idx) + 1)
                cond_raw_step = cond_raw_tgt[:, idx_raw]
            else:
                cond_raw_step = cond_raw_tgt
        cond_raw_for_model = cond_raw_step
        if enable_reprojection and t > 0 and torch.is_tensor(cond_raw_step):
            yaw_gt = None
            try:
                if include_boundary:
                    gt_idx = int((int(idx) + 1) % int(cycle_len))
                else:
                    gt_idx = min(int(motion_seq.shape[1]) - 1, int(idx))
                gt_raw_frame = trainer.normalizer.denorm_x(motion_seq[:, gt_idx])
                yaw_gt = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
            except Exception:
                yaw_gt = None
            yaw_pred = None
            try:
                yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
            except Exception:
                yaw_pred = None
            if yaw_gt is not None and yaw_pred is not None:
                try:
                    cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
                except Exception:
                    cond_proj = None
                if cond_proj is not None:
                    cond_raw_for_model = cond_proj
        if cond_raw_for_model is not None:
            try:
                cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
            except Exception:
                cond_override = None
            if cond_override is not None:
                cond_t = cond_override

        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, idx] if (torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3) else angvel_seq

        if pose_hist_enabled and pose_hist_buffer_norm is not None:
            pose_hist_t = pose_hist_buffer_norm
        else:
            pose_hist_t = pose_hist_seq[:, idx] if (torch.is_tensor(pose_hist_seq) and pose_hist_seq.dim() == 3) else pose_hist_seq

        inp_motion = motion.unsqueeze(1)
        inp_cond = cond_t.unsqueeze(1) if torch.is_tensor(cond_t) and cond_t.dim() == 2 else cond_t
        inp_angvel = angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) and angvel_t.dim() == 2 else angvel_t
        inp_pose_hist = pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) and pose_hist_t.dim() == 2 else pose_hist_t

        plan_enable = bool(getattr(model, "contact_plan_enable", False))
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False))
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        contacts_wb_t = None
        if plan_enable and (
            (not use_learned_meas)
            or (init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0)
        ):
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if plan_enable:
            if not use_learned_meas:
                contacts_in_t = contacts_wb_t
            elif init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0:
                contacts_in_t = contacts_wb_t

        if time_index_mode == "none":
            time_index_t = None
        elif time_index_mode == "cycle":
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    pass
        else:
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    time_index_t = int(idx)
            else:
                time_index_t = int(t)

        rollout_step_t = None
        try:
            denom = int(max(1, total_steps - 1))
            step_norm = float(int(t)) / float(denom)
            rollout_step_t = torch.full((B, 1, 1), step_norm, device=device, dtype=dtype)
        except Exception:
            rollout_step_t = None

        ret = model(
            inp_motion,
            inp_cond,
            contacts=contacts_in_t,
            angvel=inp_angvel,
            pose_history=inp_pose_hist,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            td_hazard_acc=td_hazard_acc,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index_t,
            rollout_step=rollout_step_t,
        )
        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict.")

        # Supervise meas head using logits (skip when contacts are overridden / fallback is used).
        if contacts_in_t is None and float(weight or 0.0) > 0.0:
            try:
                gt_c_t = gt_contacts[:, idx] if gt_contacts.dim() == 3 else gt_contacts
                meas_logits = ret.get("contacts_meas_logits", None)
                if torch.is_tensor(meas_logits):
                    if meas_logits.dim() == 3:
                        meas_logits = meas_logits[:, -1]
                    if torch.is_tensor(gt_c_t) and gt_c_t.shape == meas_logits.shape:
                        gt = gt_c_t.clamp(0.0, 1.0)
                        bce_t = F.binary_cross_entropy_with_logits(meas_logits, gt)
                        meas_terms.append(bce_t * time_weights[t])

                        band = float(transition_band or 0.0)
                        if not math.isfinite(band):
                            band = 0.0
                        band = max(0.0, min(0.49, band))
                        if band <= 0.0:
                            mask = torch.ones_like(gt, dtype=torch.bool)
                        else:
                            low = float(0.5 - band)
                            high = float(0.5 + band)
                            mask = (gt <= low) | (gt >= high)
                        mask_f = mask.to(dtype=meas_logits.dtype)
                        try:
                            non_transition_fracs.append(mask_f.mean().detach().cpu())
                        except Exception:
                            pass

                        mw = float(margin_weight or 0.0)
                        if mw > 0.0:
                            m = float(margin_logit or 0.0)
                            if not math.isfinite(m):
                                m = 0.0
                            m = max(0.0, m)
                            if m > 0.0:
                                # relu(m - |logit|), masked by non-transition GT.
                                margin_elem = F.relu(m - meas_logits.abs())
                                denom = mask_f.sum().clamp_min(1.0)
                                margin_t = (margin_elem * mask_f).sum() / denom
                                margin_terms.append(margin_t * time_weights[t])

                        sw = float(smooth_weight or 0.0)
                        if sw > 0.0 and prev_logits is not None and prev_mask is not None:
                            d = meas_logits - prev_logits
                            kind = str(smooth_kind or "l1").strip().lower()
                            if kind == "l2":
                                d_elem = d.pow(2)
                            elif kind in ("smooth_l1", "huber"):
                                d_elem = F.smooth_l1_loss(d, torch.zeros_like(d), reduction="none")
                            else:
                                d_elem = d.abs()
                            mask_pair = (mask & prev_mask).to(dtype=meas_logits.dtype)
                            denom = mask_pair.sum().clamp_min(1.0)
                            smooth_t = (d_elem * mask_pair).sum() / denom
                            smooth_terms.append(smooth_t * time_weights[t])

                        prev_logits = meas_logits.detach()
                        prev_mask = mask.detach()
            except Exception:
                pass

        # Cache recurrent states.
        if bool(getattr(model, "contact_plan_enable", False)):
            try:
                z_next = ret.get("plan_z_next", None)
                if torch.is_tensor(z_next):
                    plan_z = z_next.detach()
                p_next = ret.get("phase_z_next", None)
                if torch.is_tensor(p_next):
                    phase_z = p_next.detach()
                a_next = ret.get("phase_event_age_next", None)
                if torch.is_tensor(a_next):
                    phase_event_age = a_next.detach()
            except Exception:
                pass
        try:
            hz_acc_next = ret.get("td_hazard_acc_next", None)
            if torch.is_tensor(hz_acc_next):
                td_hazard_acc = hz_acc_next.detach()
        except Exception:
            pass
        try:
            mlog = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(mlog):
                if mlog.dim() == 3:
                    meas_logits_prev = mlog[:, -1].detach()
                elif mlog.dim() == 2:
                    meas_logits_prev = mlog.detach()
        except Exception:
            pass

        # External phase reset from TTC anchors (posttrain rollout).
        # This mirrors train/validate/run_freerun_cycles.py:
        # - for phase_reset_source=ttc_gt, reset phase_z to anchor [sin=0,cos=1] on GT touchdown events.
        # - phase_reset_source=td_hazard is handled inside the model (integrate-to-1), so we don't touch it here.
        phase_reset_src = str(getattr(model, "phase_reset_source", "contacts_meas") or "contacts_meas").strip().lower()
        if (
            bool(getattr(model, "contact_plan_enable", False))
            and phase_reset_src in ("ttc_gt", "ttc")
            and torch.is_tensor(phase_z)
            and torch.is_tensor(ttc_events_seq)
        ):
            gt_event_t = (
                ttc_events_seq[:, idx]
                if (torch.is_tensor(ttc_events_seq) and ttc_events_seq.dim() == 3)
                else ttc_events_seq
            )
            gt_valid_t = (
                ttc_valid_seq[:, idx]
                if (torch.is_tensor(ttc_valid_seq) and ttc_valid_seq.dim() == 3)
                else ttc_valid_seq
            )
            try:
                Cc = int(getattr(model, "contact_dim", 0) or 0)
            except Exception:
                Cc = 0
            if Cc > 0 and phase_z.ndim == 2 and int(phase_z.shape[-1]) == int(2 * Cc):
                try:
                    ev = gt_event_t
                    if ev.ndim == 3 and ev.size(1) == 1:
                        ev = ev[:, 0]
                    if ev.ndim == 1:
                        ev = ev.view(1, -1)
                    if ev.ndim != 2:
                        ev = ev.reshape(phase_z.shape[0], -1)
                    if ev.shape[0] == 1 and phase_z.shape[0] > 1:
                        ev = ev.expand(phase_z.shape[0], -1)
                    if int(ev.shape[-1]) != int(Cc):
                        if int(ev.shape[-1]) > int(Cc):
                            ev = ev[..., :Cc]
                        else:
                            pad = int(Cc) - int(ev.shape[-1])
                            ev = torch.cat([ev, ev.new_zeros(ev.shape[0], pad)], dim=-1)

                    # Gate by validity if provided.
                    if torch.is_tensor(gt_valid_t) and gt_valid_t.shape == ev.shape:
                        if ev.dtype == torch.bool:
                            ev = ev & gt_valid_t
                        else:
                            ev = ev * gt_valid_t.to(dtype=ev.dtype)

                    if ev.dtype != torch.bool:
                        ev = ev > 0.5

                    with torch.no_grad():
                        phase = phase_z.view(phase_z.shape[0], Cc, 2)
                        anchor = phase.new_zeros((phase.shape[0], Cc, 2))
                        anchor[..., 1] = 1.0
                        m = ev.to(dtype=phase.dtype).unsqueeze(-1)
                        phase = phase * (1.0 - m) + anchor * m
                        phase_z = phase.reshape(phase_z.shape[0], -1)

                        # Track/update phase_event_age externally (frames since last reset).
                        age = phase_event_age
                        if not torch.is_tensor(age):
                            age = torch.zeros((phase_z.shape[0], Cc), device=phase_z.device, dtype=phase_z.dtype)
                        else:
                            if age.ndim == 3 and age.size(1) == 1:
                                age = age[:, 0]
                            if age.ndim == 1:
                                age = age.view(1, -1)
                            if age.ndim != 2:
                                age = age.reshape(phase_z.shape[0], -1)
                            if age.shape[0] == 1 and phase_z.shape[0] > 1:
                                age = age.expand(phase_z.shape[0], -1)
                            if int(age.shape[-1]) != int(Cc):
                                if int(age.shape[-1]) > int(Cc):
                                    age = age[..., :Cc]
                                else:
                                    pad = int(Cc) - int(age.shape[-1])
                                    age = torch.cat([age, age.new_zeros(age.shape[0], pad)], dim=-1)
                        phase_event_age = torch.where(ev, torch.zeros_like(age), age + 1.0)
                except Exception:
                    pass

        delta_norm = ret.get("out")
        if delta_norm is None:
            raise RuntimeError("Model dict output missing required key: out.")
        if delta_norm.dim() == 3:
            delta_norm = delta_norm[:, -1]

        # Rollout carry (match run_freerun_cycles semantics as much as possible):
        #   1) compose incremental Δ to y_inc_raw (optionally with SO(3) corrector)
        #   2) optionally apply lambda fusion to get y_used_raw (incremental->direct blend)
        #   3) write y_used_raw back to x_raw via _apply_free_carry
        omega_hat = ret.get("omega_hat", None)
        try:
            y_inc_raw = trainer._compose_delta_to_raw(
                y_prev_raw,
                delta_norm,
                omega_hat=omega_hat,
                so3_gate=getattr(trainer, "so3_corr_gate_force", None),
                so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
                omega_detach=True,
            )
        except Exception:
            y_inc_raw = trainer._denorm(delta_norm)

        y_used_raw = y_inc_raw
        try:
            lam_step = ret.get("lambda_fusion", None)
            direct_norm_step = ret.get("out_direct", None)
            if torch.is_tensor(lam_step) and torch.is_tensor(direct_norm_step):
                lam_eff = lam_step
                try:
                    lam_eff, _lam_rel = trainer._lambda_fusion_apply_reliability(
                        lam_step,
                        step_idx=int(t),
                        total_steps=int(total_steps),
                        rollout_step=rollout_step_t,
                        ret=ret,
                    )
                except Exception:
                    lam_eff = lam_step
                y_used_raw = trainer._apply_lambda_fusion_to_raw(
                    y_inc_raw,
                    direct_norm=direct_norm_step,
                    direct_hinge_delta=ret.get("direct_hinge_delta", None),
                    lambda_fusion=lam_eff,
                )
        except Exception:
            y_used_raw = y_inc_raw

        if detach_rollout_state:
            y_used_raw = y_used_raw.detach()

        if t < total_steps - 1:
            cond_env = cond_raw_step if torch.is_tensor(cond_raw_step) else None
            motion_raw = trainer._apply_free_carry(motion_raw, y_used_raw, cond_next_raw=cond_env)
            motion_raw = _finite(motion_raw)
            motion = trainer._diag_norm_x(motion_raw)

            if pose_hist_enabled and pose_hist_buffer_raw is not None and pose_hist_stride > 0 and isinstance(rot_slice, slice):
                with torch.no_grad():
                    pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_used_raw[..., rot_slice]
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

            y_prev_raw = y_used_raw

    if not meas_terms:
        raise RuntimeError("No contact_meas logits were produced during rollout (did contacts get overridden?).")

    denom = time_weights.sum().clamp_min(1e-6)
    meas_bce = torch.stack(meas_terms).sum() / denom
    w = float(weight or 0.0)
    smooth_loss = None
    if smooth_terms:
        smooth_loss = torch.stack(smooth_terms).sum() / denom
    margin_loss = None
    if margin_terms:
        margin_loss = torch.stack(margin_terms).sum() / denom
    non_transition_frac = None
    if non_transition_fracs:
        try:
            non_transition_frac = float(torch.stack(non_transition_fracs).mean().detach().cpu())
        except Exception:
            non_transition_frac = None

    sw = float(smooth_weight or 0.0)
    mw = float(margin_weight or 0.0)
    loss_unweighted = meas_bce
    if smooth_loss is not None and sw > 0.0:
        loss_unweighted = loss_unweighted + sw * smooth_loss
    if margin_loss is not None and mw > 0.0:
        loss_unweighted = loss_unweighted + mw * margin_loss
    loss = loss_unweighted * w
    stats = {
        "contact_meas_bce": float(meas_bce.detach().cpu()),
        "contact_meas_weighted": float((w * meas_bce).detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    if smooth_loss is not None:
        stats["contact_meas_smooth"] = float(smooth_loss.detach().cpu())
    if margin_loss is not None:
        stats["contact_meas_margin"] = float(margin_loss.detach().cpu())
    if non_transition_frac is not None:
        stats["contact_meas_non_transition_frac"] = float(non_transition_frac)
    return loss, stats


def _contact_td_hazard_loss_rollout(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    rollout_steps: int,
    rollout_cycles: int,
    include_boundary: bool,
    boundary_weight: float,
    random_offset: bool,
    time_index_mode: str,
    time_weight_max: float,
    detach_rollout_state: bool,
    bce_weight: float = 1.0,
    event_weight: float = 0.0,
    mass_weight: float = 0.0,
    unimodal_weight: float = 0.0,
    entropy_weight: float = 0.0,
    clock_weight: float = 0.0,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contact_td_hazard_head against GT touchdown events while unrolling the model in a closed-loop rollout
    (freerun-like) to expose the hazard head to drift/OOD inputs.

    Loss terms follow _contact_td_hazard_loss_teacher, but applied on rollout states:
    - L_bce: BCEWithLogits(hazard_logit, ttc_td_events) (masked by ttc_td_valid when provided; optionally reweight events)
    - L_mass: per-cycle mass matching: (sum_t sigmoid(logit[t]) - sum_t events[t])^2
    - L_unimodal: per-cycle log-softmax concavity penalty on time axis to suppress multi-peaks
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion")
    cond_seq = batch.get("cond_in")
    gt_events = batch.get("ttc_td_events")
    gt_valid = batch.get("ttc_td_valid")
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_events)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / ttc_td_events.")

    if not bool(getattr(model, "contact_td_hazard_enable", False)) or getattr(model, "contact_td_hazard_head", None) is None:
        raise RuntimeError("Model has no contact_td_hazard_head enabled; cannot train contact_td_hazard.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)  # (B,T,Dx)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_events = gt_events.to(device=device)
    if torch.is_tensor(gt_valid):
        gt_valid = gt_valid.to(device=device)

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
    cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, motion_seq) if cond_norm_mu is not None else None
    cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, motion_seq) if cond_norm_std is not None else None

    angvel_seq = batch.get("angvel")
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    pose_hist_seq = batch.get("pose_hist")
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    B, T, _ = motion_seq.shape
    if T < 2:
        raise ValueError(f"seq_len must be >=2, got {T}")
    steps = _resolve_rollout_steps(T, rollout_steps)
    steps = max(1, int(steps))
    rollout_cycles = max(1, int(rollout_cycles or 1))
    include_boundary = bool(include_boundary) and int(rollout_cycles) > 1 and int(steps) == int(T - 1)
    cycle_len = int(T) if include_boundary else int(steps)
    # TD hazard supervision is per-frame (not per-transition). When including the synthetic wrap boundary,
    # include the last frame of the last cycle so per-cycle mass/unimodality terms see full cycles.
    total_steps = (int(rollout_cycles) * int(cycle_len)) if include_boundary else (int(steps) * int(rollout_cycles))

    # TD hazard supervision is per-frame (not per-transition). When unrolling a single cycle without wrap,
    # include the last frame so per-cycle mass matches `ttc_td_events` computed on the seq_len window.
    if (not include_boundary) and int(rollout_cycles) == 1 and int(steps) == int(T - 1):
        cycle_len = int(T)
        total_steps = int(T)

    # Optional random phase offset (only meaningful when unrolling multiple cycles).
    offset = 0
    if bool(random_offset) and int(rollout_cycles) > 1 and int(cycle_len) > 1:
        try:
            offset = int(torch.randint(low=0, high=int(cycle_len), size=(1,), device="cpu").item())
        except Exception:
            offset = 0

    Dy = int(getattr(trainer, "Dy", 0) or 0)
    if Dy <= 0:
        try:
            Dy = int(getattr(trainer.normalizer, "mu_y", np.zeros((0,), dtype=np.float32)).shape[-1])
        except Exception:
            Dy = 0
    if Dy <= 0:
        raise RuntimeError("Cannot infer Dy for rollout carry; Trainer.Dy is missing.")

    motion = motion_seq[:, int(offset)]
    motion_raw = trainer.normalizer.denorm_x(motion)
    y_prev_raw = _init_y_from_x(trainer.normalizer, motion_raw, Dy)

    pose_hist_enabled = bool(getattr(trainer, "pose_hist_len", 0) or 0) > 0 and bool(getattr(trainer, "pose_hist_dim", 0) or 0) > 0
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
    scales = mu = std = None
    pose_hist_buffer_norm = None
    pose_hist_buffer_raw = None
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if pose_hist_enabled and pose_hist_stride > 0:
        try:
            scales, mu, std = trainer._pose_hist_params(motion_seq)
        except Exception:
            scales = mu = std = None
        if scales is None:
            pose_hist_enabled = False
        else:
            with torch.no_grad():
                if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0 and pose_hist_seq.dim() == 3:
                    pose_hist_buffer_norm = pose_hist_seq[:, int(offset)]
                    pose_hist_buffer_raw = trainer._pose_hist_inverse_vec(pose_hist_buffer_norm, scales, mu, std)
                else:
                    if not isinstance(rot_slice, slice):
                        raise RuntimeError("pose_hist enabled but rot slice missing for init.")
                    base_rot = y_prev_raw[..., rot_slice]
                    pose_hist_buffer_raw = base_rot.unsqueeze(1).repeat(1, pose_hist_len, 1).reshape(B, pose_hist_dim)
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

    plan_z = None
    phase_z = None
    phase_event_age = None
    meas_logits_prev = None
    td_hazard_acc = None  # (B,C) stateful when phase_reset_source=td_hazard

    bw = float(bce_weight or 0.0)
    ew = float(event_weight or 0.0)
    mw = float(mass_weight or 0.0)
    uw = float(unimodal_weight or 0.0)
    hw = float(entropy_weight or 0.0)
    cw = float(clock_weight or 0.0)

    time_weight_max = max(1.0, float(time_weight_max or 1.0))
    time_weights = torch.linspace(1.0, time_weight_max, steps=total_steps, device=device, dtype=dtype)
    if include_boundary:
        bw_boundary = float(boundary_weight or 0.0)
        bw_boundary = max(0.0, bw_boundary)
        if abs(bw_boundary - 1.0) > 1e-12:
            try:
                idxs = (torch.arange(int(total_steps), device=device) + int(offset)) % int(cycle_len)
                boundary_mask = idxs == (int(cycle_len) - 1)
                factors = torch.ones_like(time_weights)
                factors = torch.where(boundary_mask, time_weights.new_tensor(bw_boundary), factors)
                time_weights = time_weights * factors
            except Exception:
                pass
    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))

    time_base = None
    try:
        base = batch.get("start", None)
        if base is not None:
            if torch.is_tensor(base):
                base = base.to(device=device)
            time_base = base
    except Exception:
        time_base = None

    time_index_mode = str(time_index_mode or "global").strip().lower()
    if time_index_mode == "auto":
        time_index_mode = "cycle" if rollout_cycles > 1 else "global"
    if time_index_mode not in ("global", "cycle", "none"):
        time_index_mode = "global"
    # Note: do not pre-add offset to time_base here; time_index uses idx (which already includes offset).

    bce_num = None
    bce_den = None
    valid_fracs: list[torch.Tensor] = []
    hz_logits: list[torch.Tensor] = []
    hz_gt: list[torch.Tensor] = []
    hz_mask: list[torch.Tensor] = []
    prev_foot_pos_meas = None

    for t in range(total_steps):
        idx = int((int(offset) + int(t)) % int(cycle_len))

        cond_t = cond_seq[:, idx] if cond_seq.dim() == 3 else cond_seq
        cond_raw_step = None
        if torch.is_tensor(cond_raw_tgt):
            if cond_raw_tgt.dim() == 3:
                if include_boundary:
                    idx_raw = int((int(idx) + 1) % int(cycle_len))
                else:
                    idx_raw = min(int(cond_raw_tgt.shape[1]) - 1, int(idx) + 1)
                cond_raw_step = cond_raw_tgt[:, idx_raw]
            else:
                cond_raw_step = cond_raw_tgt
        cond_raw_for_model = cond_raw_step
        if enable_reprojection and t > 0 and torch.is_tensor(cond_raw_step):
            yaw_gt = None
            try:
                if include_boundary:
                    gt_idx = int((int(idx) + 1) % int(cycle_len))
                else:
                    gt_idx = min(int(motion_seq.shape[1]) - 1, int(idx))
                gt_raw_frame = trainer.normalizer.denorm_x(motion_seq[:, gt_idx])
                yaw_gt = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
            except Exception:
                yaw_gt = None
            yaw_pred = None
            try:
                yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
            except Exception:
                yaw_pred = None
            if yaw_gt is not None and yaw_pred is not None:
                try:
                    cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
                except Exception:
                    cond_proj = None
                if cond_proj is not None:
                    cond_raw_for_model = cond_proj
        if cond_raw_for_model is not None:
            try:
                cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
            except Exception:
                cond_override = None
            if cond_override is not None:
                cond_t = cond_override

        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, idx] if (torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3) else angvel_seq

        if pose_hist_enabled and pose_hist_buffer_norm is not None:
            pose_hist_t = pose_hist_buffer_norm
        else:
            pose_hist_t = pose_hist_seq[:, idx] if (torch.is_tensor(pose_hist_seq) and pose_hist_seq.dim() == 3) else pose_hist_seq

        inp_motion = motion.unsqueeze(1)
        inp_cond = cond_t.unsqueeze(1) if torch.is_tensor(cond_t) and cond_t.dim() == 2 else cond_t
        inp_angvel = angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) and angvel_t.dim() == 2 else angvel_t
        inp_pose_hist = pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) and pose_hist_t.dim() == 2 else pose_hist_t

        plan_enable = bool(getattr(model, "contact_plan_enable", False))
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False))
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        contacts_wb_t = None
        if plan_enable and (
            (not use_learned_meas)
            or (init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0)
        ):
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if plan_enable:
            if not use_learned_meas:
                contacts_in_t = contacts_wb_t
            elif init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0:
                contacts_in_t = contacts_wb_t

        if time_index_mode == "none":
            time_index_t = None
        elif time_index_mode == "cycle":
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    pass
        else:
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    time_index_t = int(idx)
            else:
                time_index_t = int(t)

        rollout_step_t = None
        try:
            denom = int(max(1, total_steps - 1))
            step_norm = float(int(t)) / float(denom)
            rollout_step_t = torch.full((B, 1, 1), step_norm, device=device, dtype=dtype)
        except Exception:
            rollout_step_t = None

        ret = model(
            inp_motion,
            inp_cond,
            contacts=contacts_in_t,
            angvel=inp_angvel,
            pose_history=inp_pose_hist,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            td_hazard_acc=td_hazard_acc,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index_t,
            rollout_step=rollout_step_t,
        )
        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict.")

        pred_logit_t = ret.get("contacts_td_hazard_logit", None)
        if not torch.is_tensor(pred_logit_t):
            raise RuntimeError("Model output missing contacts_td_hazard_logit during rollout.")
        if pred_logit_t.dim() == 3:
            pred_logit_t = pred_logit_t[:, -1]

        gt_event_t = gt_events[:, idx] if (torch.is_tensor(gt_events) and gt_events.dim() == 3) else gt_events
        gt_valid_t = gt_valid[:, idx] if (torch.is_tensor(gt_valid) and gt_valid.dim() == 3) else gt_valid

        if not torch.is_tensor(gt_event_t):
            raise RuntimeError("Batch missing ttc_td_events during rollout.")
        if gt_event_t.shape != pred_logit_t.shape:
            raise RuntimeError(
                f"contacts_td_hazard_logit shape {tuple(pred_logit_t.shape)} vs GT {tuple(gt_event_t.shape)} mismatch."
            )

        gt_t = gt_event_t.to(device=pred_logit_t.device, dtype=pred_logit_t.dtype).clamp(0.0, 1.0)
        if torch.is_tensor(gt_valid_t) and gt_valid_t.shape == gt_t.shape:
            m = gt_valid_t
        else:
            m = torch.ones_like(gt_t, dtype=torch.bool, device=gt_t.device)
        mf = m.to(dtype=pred_logit_t.dtype)

        if bw > 0.0:
            err = F.binary_cross_entropy_with_logits(pred_logit_t, gt_t, reduction="none")
            wf = mf
            if ew > 0.0:
                wf = wf * (1.0 + ew * gt_t)
            w_t = time_weights[t]
            if bce_num is None:
                bce_num = (err * wf).sum() * w_t
                bce_den = wf.sum() * w_t
            else:
                bce_num = bce_num + (err * wf).sum() * w_t
                bce_den = bce_den + wf.sum() * w_t
            try:
                valid_fracs.append(mf.mean().detach().cpu())
            except Exception:
                pass

        if (mw > 0.0) or (uw > 0.0) or (hw > 0.0):
            hz_logits.append(pred_logit_t)
            hz_gt.append(gt_t)
            hz_mask.append(m)

        # Cache recurrent states.
        if bool(getattr(model, "contact_plan_enable", False)):
            try:
                z_next = ret.get("plan_z_next", None)
                if torch.is_tensor(z_next):
                    plan_z = z_next.detach()
                p_next = ret.get("phase_z_next", None)
                if torch.is_tensor(p_next):
                    phase_z = p_next.detach()
                a_next = ret.get("phase_event_age_next", None)
                if torch.is_tensor(a_next):
                    phase_event_age = a_next.detach()
            except Exception:
                pass
        try:
            hz_acc_next = ret.get("td_hazard_acc_next", None)
            if torch.is_tensor(hz_acc_next):
                td_hazard_acc = hz_acc_next.detach()
        except Exception:
            pass
        try:
            mlog = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(mlog):
                if mlog.dim() == 3:
                    meas_logits_prev = mlog[:, -1].detach()
                elif mlog.dim() == 2:
                    meas_logits_prev = mlog.detach()
        except Exception:
            pass

        delta_norm = ret.get("out")
        if delta_norm is None:
            raise RuntimeError("Model dict output missing required key: out.")
        if delta_norm.dim() == 3:
            delta_norm = delta_norm[:, -1]

        omega_hat = ret.get("omega_hat", None)
        try:
            y_inc_raw = trainer._compose_delta_to_raw(
                y_prev_raw,
                delta_norm,
                omega_hat=omega_hat,
                so3_gate=getattr(trainer, "so3_corr_gate_force", None),
                so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
                omega_detach=True,
            )
        except Exception:
            y_inc_raw = trainer._denorm(delta_norm)

        y_used_raw = y_inc_raw
        try:
            lam_step = ret.get("lambda_fusion", None)
            direct_norm_step = ret.get("out_direct", None)
            if torch.is_tensor(lam_step) and torch.is_tensor(direct_norm_step):
                lam_eff = lam_step
                try:
                    lam_eff, _lam_rel = trainer._lambda_fusion_apply_reliability(
                        lam_step,
                        step_idx=int(t),
                        total_steps=int(total_steps),
                        rollout_step=rollout_step_t,
                        ret=ret,
                    )
                except Exception:
                    lam_eff = lam_step
                y_used_raw = trainer._apply_lambda_fusion_to_raw(
                    y_inc_raw,
                    direct_norm=direct_norm_step,
                    direct_hinge_delta=ret.get("direct_hinge_delta", None),
                    lambda_fusion=lam_eff,
                )
        except Exception:
            y_used_raw = y_inc_raw

        if detach_rollout_state:
            y_used_raw = y_used_raw.detach()

        if t < total_steps - 1:
            cond_env = cond_raw_step if torch.is_tensor(cond_raw_step) else None
            motion_raw = trainer._apply_free_carry(motion_raw, y_used_raw, cond_next_raw=cond_env)
            motion_raw = _finite(motion_raw)
            motion = trainer._diag_norm_x(motion_raw)

            if pose_hist_enabled and pose_hist_buffer_raw is not None and pose_hist_stride > 0 and isinstance(rot_slice, slice):
                with torch.no_grad():
                    pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_used_raw[..., rot_slice]
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

            y_prev_raw = y_used_raw

    if bce_num is None or bce_den is None:
        l_bce = motion_seq.new_tensor(0.0)
    else:
        l_bce = bce_num / bce_den.clamp_min(1.0)

    l_mass = motion_seq.new_tensor(0.0)
    mass_pred_mean = None
    mass_tgt_mean = None
    if mw > 0.0 and hz_logits:
        logit_all = torch.stack(hz_logits, dim=1)  # (B, S, C)
        gt_all = torch.stack(hz_gt, dim=1)         # (B, S, C)
        mask_all = torch.stack(hz_mask, dim=1)      # (B, S, C) bool
        s_full = int((logit_all.shape[1] // int(cycle_len)) * int(cycle_len))
        if s_full >= int(cycle_len):
            C = int(logit_all.shape[-1])
            logit_c = logit_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            gt_c = gt_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            mask_c = mask_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            mf = mask_c.to(dtype=logit_all.dtype)
            p = torch.sigmoid(logit_c)
            mass_pred = (p * mf).sum(dim=2)    # (B, cycles, C)
            mass_tgt = (gt_c * mf).sum(dim=2)  # (B, cycles, C)
            l_mass = (mass_pred - mass_tgt).pow(2).mean()
            try:
                mass_pred_mean = float(mass_pred.detach().mean().cpu())
                mass_tgt_mean = float(mass_tgt.detach().mean().cpu())
            except Exception:
                mass_pred_mean = None
                mass_tgt_mean = None

    l_uni = motion_seq.new_tensor(0.0)
    if uw > 0.0 and hz_logits and int(cycle_len) >= 3:
        logit_all = torch.stack(hz_logits, dim=1)
        mask_all = torch.stack(hz_mask, dim=1)
        s_full = int((logit_all.shape[1] // int(cycle_len)) * int(cycle_len))
        if s_full >= int(cycle_len):
            C = int(logit_all.shape[-1])
            logit_c = logit_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            mask_c = mask_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            logit_masked = torch.where(mask_c, logit_c, logit_c.new_full((), -1e9))
            logp = F.log_softmax(logit_masked, dim=2)
            d2 = logp[:, :, :-2] - 2.0 * logp[:, :, 1:-1] + logp[:, :, 2:]
            pen = F.relu(d2)
            triplet = mask_c[:, :, :-2] & mask_c[:, :, 1:-1] & mask_c[:, :, 2:]
            tf = triplet.to(dtype=pen.dtype)
            denom_u = tf.sum().clamp_min(1.0)
            l_uni = (pen * tf).sum() / denom_u

    l_ent = motion_seq.new_tensor(0.0)
    if hw > 0.0 and hz_logits and int(cycle_len) >= 1:
        logit_all = torch.stack(hz_logits, dim=1)
        mask_all = torch.stack(hz_mask, dim=1)
        s_full = int((logit_all.shape[1] // int(cycle_len)) * int(cycle_len))
        if s_full >= int(cycle_len):
            C = int(logit_all.shape[-1])
            logit_c = logit_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            mask_c = mask_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            logit_masked = torch.where(mask_c, logit_c, logit_c.new_full((), -1e9))
            logp = F.log_softmax(logit_masked, dim=2)
            p = logp.exp()
            ent = -(p * logp).sum(dim=2)  # (B, cycles, C)
            l_ent = ent.mean()

    l_clock = motion_seq.new_tensor(0.0)
    if cw > 0.0 and hz_logits and int(cycle_len) >= 1:
        logit_all = torch.stack(hz_logits, dim=1)  # (B, S, C)
        gt_all = torch.stack(hz_gt, dim=1)
        mask_all = torch.stack(hz_mask, dim=1)
        s_full = int((logit_all.shape[1] // int(cycle_len)) * int(cycle_len))
        if s_full >= int(cycle_len):
            C = int(logit_all.shape[-1])
            logit_c = logit_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            gt_c = gt_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            mask_c = mask_all[:, :s_full].reshape(B, -1, int(cycle_len), C)
            # Flatten (B,cycles) -> N to reuse teacher-style clock loss.
            N = int(logit_c.shape[0] * logit_c.shape[1])
            l_clock = _td_hazard_clock_align_loss(
                logit_c.reshape(N, int(cycle_len), C),
                gt_c.reshape(N, int(cycle_len), C),
                mask_c.reshape(N, int(cycle_len), C),
            )

    scale = float(weight or 1.0)
    total = ((bw * l_bce) + (mw * l_mass) + (uw * l_uni) + (hw * l_ent) + (cw * l_clock)) * scale

    stats: Dict[str, float] = {"total": float(total.detach().cpu())}
    if bw > 0.0:
        stats["contact_td_hazard_bce"] = float(l_bce.detach().cpu())
        stats["contact_td_hazard_bce_weighted"] = float((scale * bw * l_bce).detach().cpu())
        if valid_fracs:
            try:
                stats["contact_td_hazard_valid_frac"] = float(torch.stack(valid_fracs).mean().detach().cpu())
            except Exception:
                pass
    if mw > 0.0:
        stats["contact_td_hazard_mass_l2"] = float(l_mass.detach().cpu())
        stats["contact_td_hazard_mass_weighted"] = float((scale * mw * l_mass).detach().cpu())
        if mass_pred_mean is not None:
            stats["contact_td_hazard_mass_pred_mean"] = float(mass_pred_mean)
        if mass_tgt_mean is not None:
            stats["contact_td_hazard_mass_tgt_mean"] = float(mass_tgt_mean)
    if uw > 0.0:
        stats["contact_td_hazard_unimodal"] = float(l_uni.detach().cpu())
        stats["contact_td_hazard_unimodal_weighted"] = float((scale * uw * l_uni).detach().cpu())
    if hw > 0.0:
        stats["contact_td_hazard_entropy"] = float(l_ent.detach().cpu())
        stats["contact_td_hazard_entropy_weighted"] = float((scale * hw * l_ent).detach().cpu())
    if cw > 0.0:
        stats["contact_td_hazard_clock"] = float(l_clock.detach().cpu())
        stats["contact_td_hazard_clock_weighted"] = float((scale * cw * l_clock).detach().cpu())

    return total, stats


def _contact_td_hazard_loss_teacher(
    trainer: Trainer,
    model: EventMotionModel,
    batch: Dict[str, torch.Tensor],
    *,
    time_index_mode: str = "none",
    bce_weight: float = 1.0,
    event_weight: float = 0.0,
    mass_weight: float = 0.0,
    unimodal_weight: float = 0.0,
    entropy_weight: float = 0.0,
    clock_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise contact_td_hazard_head against GT touchdown event targets in teacher mode.

    - L_bce: BCEWithLogits(hazard_logit, ttc_td_events) (mask by ttc_td_valid when provided; optionally reweight events)
    - L_mass: (sum_t sigmoid(logit[t]) - sum_t events[t])^2
    - L_unimodal: log-softmax concavity penalty on time axis to suppress multi-peaks:
        mean(relu(logp[t-1] - 2*logp[t] + logp[t+1]))
    """
    device = trainer.device
    dtype = next(model.parameters()).dtype

    motion_seq = batch.get("motion", None)
    cond_seq = batch.get("cond_in", None)
    gt_events = batch.get("ttc_td_events", None)
    gt_valid = batch.get("ttc_td_valid", None)
    angvel_seq = batch.get("angvel", None)
    pose_hist_seq = batch.get("pose_hist", None)
    if not (torch.is_tensor(motion_seq) and torch.is_tensor(cond_seq) and torch.is_tensor(gt_events)):
        raise RuntimeError("Batch missing required keys: motion / cond_in / ttc_td_events.")

    if not bool(getattr(model, "contact_td_hazard_enable", False)) or getattr(model, "contact_td_hazard_head", None) is None:
        raise RuntimeError("Model has no contact_td_hazard_head enabled; cannot train contact_td_hazard.")

    motion_seq = motion_seq.to(device=device, dtype=dtype)
    cond_seq = cond_seq.to(device=device, dtype=dtype)
    gt_events = gt_events.to(device=device)
    if torch.is_tensor(gt_valid):
        gt_valid = gt_valid.to(device=device)
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(device=device, dtype=dtype)
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(device=device, dtype=dtype)

    time_index_mode = str(time_index_mode or "none").strip().lower()
    time_index = None
    if time_index_mode == "global":
        try:
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_index = base
        except Exception:
            time_index = None
    elif time_index_mode in ("none", "cycle", "auto"):
        time_index = None

    ret = model(
        motion_seq,
        cond_seq,
        contacts=None,
        angvel=angvel_seq,
        pose_history=pose_hist_seq,
        plan_z=None,
        time_index=time_index,
        rollout_step=None,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict.")

    pred_logit = ret.get("contacts_td_hazard_logit", None)
    if not torch.is_tensor(pred_logit):
        raise RuntimeError("Model output missing contacts_td_hazard_logit.")

    logit = pred_logit
    gt = gt_events
    if logit.dim() == 2:
        logit = logit.unsqueeze(1)
    if gt.dim() == 2:
        gt = gt.unsqueeze(1)
    T = min(int(logit.shape[1]), int(gt.shape[1]))
    if T <= 0 or int(logit.shape[-1]) != int(gt.shape[-1]):
        raise RuntimeError(f"contacts_td_hazard_logit shape {tuple(logit.shape)} vs GT {tuple(gt.shape)} mismatch.")

    gt_t = gt[:, :T].to(device=logit.device, dtype=logit.dtype).clamp(0.0, 1.0)
    if torch.is_tensor(gt_valid):
        v = gt_valid
        if v.dim() == 2:
            v = v.unsqueeze(1)
        v = v[:, :T]
        m = v if v.shape == gt_t.shape else None
    else:
        m = None
    if m is None:
        m = torch.ones_like(gt_t, dtype=torch.bool, device=gt_t.device)
    mf = m.to(dtype=logit.dtype)

    bw = float(bce_weight or 0.0)
    ew = float(event_weight or 0.0)
    mw = float(mass_weight or 0.0)
    uw = float(unimodal_weight or 0.0)
    hw = float(entropy_weight or 0.0)
    cw = float(clock_weight or 0.0)

    l_bce = logit.new_tensor(0.0)
    if bw > 0.0:
        err = F.binary_cross_entropy_with_logits(logit[:, :T], gt_t, reduction="none")
        wf = mf
        if ew > 0.0:
            wf = wf * (1.0 + ew * gt_t)
        denom = wf.sum().clamp_min(1.0)
        l_bce = (err * wf).sum() / denom

    l_mass = logit.new_tensor(0.0)
    mass_pred = None
    mass_tgt = None
    if mw > 0.0:
        p = torch.sigmoid(logit[:, :T])
        mass_pred = (p * mf).sum(dim=1)    # (B,C)
        mass_tgt = (gt_t * mf).sum(dim=1)  # (B,C)
        l_mass = (mass_pred - mass_tgt).pow(2).mean()

    l_uni = logit.new_tensor(0.0)
    if uw > 0.0 and T >= 3:
        logit_masked = torch.where(m, logit[:, :T], logit.new_full((), -1e9))
        logp = F.log_softmax(logit_masked, dim=1)
        d2 = logp[:, :-2] - 2.0 * logp[:, 1:-1] + logp[:, 2:]
        pen = F.relu(d2)
        triplet = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        tf = triplet.to(dtype=pen.dtype)
        denom = tf.sum().clamp_min(1.0)
        l_uni = (pen * tf).sum() / denom

    l_ent = logit.new_tensor(0.0)
    if hw > 0.0 and T >= 1:
        logit_masked = torch.where(m, logit[:, :T], logit.new_full((), -1e9))
        logp = F.log_softmax(logit_masked, dim=1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=1)  # (B,C)
        l_ent = ent.mean()

    l_clock = logit.new_tensor(0.0)
    if cw > 0.0 and T >= 1:
        try:
            l_clock = _td_hazard_clock_align_loss(logit[:, :T], gt_t, m)
        except Exception:
            l_clock = logit.new_tensor(0.0)

    total = (bw * l_bce) + (mw * l_mass) + (uw * l_uni) + (hw * l_ent) + (cw * l_clock)

    stats: Dict[str, float] = {"total": float(total.detach().cpu())}
    if bw > 0.0:
        stats["contact_td_hazard_bce"] = float(l_bce.detach().cpu())
        stats["contact_td_hazard_bce_weighted"] = float((bw * l_bce).detach().cpu())
        try:
            stats["contact_td_hazard_valid_frac"] = float(mf.mean().detach().cpu())
        except Exception:
            pass
    if mw > 0.0:
        stats["contact_td_hazard_mass_l2"] = float(l_mass.detach().cpu())
        stats["contact_td_hazard_mass_weighted"] = float((mw * l_mass).detach().cpu())
        if torch.is_tensor(mass_pred) and torch.is_tensor(mass_tgt):
            try:
                stats["contact_td_hazard_mass_pred_mean"] = float(mass_pred.detach().mean().cpu())
                stats["contact_td_hazard_mass_tgt_mean"] = float(mass_tgt.detach().mean().cpu())
            except Exception:
                pass
    if uw > 0.0:
        stats["contact_td_hazard_unimodal"] = float(l_uni.detach().cpu())
        stats["contact_td_hazard_unimodal_weighted"] = float((uw * l_uni).detach().cpu())
    if hw > 0.0:
        stats["contact_td_hazard_entropy"] = float(l_ent.detach().cpu())
        stats["contact_td_hazard_entropy_weighted"] = float((hw * l_ent).detach().cpu())
    if cw > 0.0:
        stats["contact_td_hazard_clock"] = float(l_clock.detach().cpu())
        stats["contact_td_hazard_clock_weighted"] = float((cw * l_clock).detach().cpu())

    return total, stats


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
    # Optional: supervised hinge delta regression (delta_target) for direct_pose_hinge_head.
    direct_pose_hinge_sup_weight: float = 0.0,
    direct_pose_hinge_sup_kind: str = "smooth_l1",  # smooth_l1|l2|apply_geo
    direct_pose_hinge_sup_contact_source: str = "gt",  # gt|plan|meas
    direct_pose_hinge_sup_contact_value: Optional[int] = None,  # 0/1 or None to disable contact masking
    direct_pose_hinge_sup_contact_thresh: float = 0.5,
    direct_pose_hinge_sup_angle_thresh_deg: float = 0.0,
    direct_pose_hinge_sup_delta_thresh_deg: float = 0.0,
    direct_pose_hinge_sup_delta_weight_power: float = 0.0,
    direct_pose_hinge_sup_delta_weight_scale_deg: float = 0.0,
    direct_pose_hinge_sup_delta_weight_max: float = 0.0,
    # Optional: supervise learned hinge gate (swing=1, stance=0) using contact thresholding.
    direct_pose_hinge_gate_sup_weight: float = 0.0,
    direct_pose_hinge_gate_sup_contact_source: str = "gt",  # gt|plan|meas
    direct_pose_hinge_gate_sup_contact_thresh: float = 0.5,
    # Optional: suppress hinge corrections on stance frames (safety term when gate_mode=none).
    direct_pose_hinge_stance_weight: float = 0.0,
    direct_pose_hinge_stance_kind: str = "l2",  # smooth_l1|l2
    direct_pose_hinge_stance_contact_source: str = "gt",  # gt|plan|meas
    direct_pose_hinge_stance_contact_thresh: float = 0.5,
    # Optional: contact-free regularizer on the applied hinge delta magnitude.
    direct_pose_hinge_reg_weight: float = 0.0,
    direct_pose_hinge_reg_kind: str = "l1",  # l1|l2|smooth_l1
    # Optional: L2 penalty on eps(hidden) contribution when using clean hinge split.
    # Penalizes mean(eps^2) in rad^2 (before any contact gating; after eps_max clamp via tanh scaling).
    direct_pose_hinge_eps_l2_weight: float = 0.0,
    # Optional: regularizer on per-side sign gate for routed shared leg omega head.
    # Penalizes (1-|g|)^2 so g does not collapse to 0 (keeps it "sign-like").
    direct_pose_leg_side_sign_gate_reg_weight: float = 0.0,
    # Optional: supervise learned leg omega gate using oracle ||omega_oracle|| thresholding.
    # Target: gate=1 if ||omega_oracle|| >= direct_pose_leg_align_oracle_min_deg else 0.
    direct_pose_leg_gate_sup_weight: float = 0.0,
    # Optional: supervise learned leg scale head using offline alpha-sweep best_alpha table.
    # Expects model output key:
    #   - direct_leg_scale_log (clamped log_mag, shape (B,K) for the leg joints)
    # If direct_pose_leg_gate_mode == 'signed_scale' and the table provides sign targets, also expects:
    #   - direct_leg_scale_sign_logit (shape (B,K))
    direct_pose_leg_scale_sup_weight: float = 0.0,
    direct_pose_leg_scale_sup_table: Optional[Dict[str, Any]] = None,
    # Optional: direction alignment loss for leg omega (SO(3) residual).
    # align_mode='cos':  L_align = relu(-cos(omega_pred, omega_oracle))  (cheatable by ||omega_pred||->0)
    # align_mode='proj': omega_oracle = log(R_gt @ R_base^T) * 2; L = w_mag*(proj-||oracle||)^2 + w_res*||res||^2 (+ w_sign*relu(-proj)^2)
    direct_pose_leg_align_weight: float = 0.0,
    direct_pose_leg_align_oracle_min_deg: float = 0.0,
    direct_pose_leg_align_oracle_weight_deg: float = 0.0,
    direct_pose_leg_align_mode: str = "cos",
    direct_pose_leg_align_mag_weight: float = 1.0,
    direct_pose_leg_align_res_weight: float = 1.0,
    direct_pose_leg_align_sign_weight: float = 0.0,
    direct_pose_leg_align_cos_thresh: float = 0.0,
    # Optional: de-dilution weighting for the direct objective (Stage6/7).
    # Uses stop-grad soft tail weights + optional state-aware swing boost to give more gradient
    # credit to worse joints / swing phases without adding per-phase/bone LUTs.
    direct_pose_loss_tail_mix: float = 0.0,
    direct_pose_loss_tail_temp_deg: float = 0.0,
    direct_pose_loss_state_swing_boost: float = 0.0,
    direct_pose_loss_state_contact_source: str = "gt",  # gt|plan|meas
    direct_pose_loss_state_scope: str = "legs",  # legs|limbs|all
    # Stage7 direct objective: optional legs vs non-legs split (decouple gradients).
    direct_pose_loss_leg_split: bool = False,
    direct_pose_loss_leg_tail_scale: str = "center",  # center|mad|none
    # Optional: focus direct objective on selected step_in_cycle indices (phase-locked hotspots).
    direct_pose_loss_sics: str = "",
    direct_pose_loss_cycle_gte: int = 0,
    direct_pose_loss_sic_mode: str = "mask",
    direct_pose_loss_sic_boost: float = 1.0,
    direct_pose_loss_pair_boost_table: Optional[Dict[str, Any]] = None,
    direct_pose_loss_group_norm_enable: bool = False,
    direct_pose_loss_group_norm_w_leg: float = 1.0,
    direct_pose_loss_group_norm_w_nonleg: float = 1.0,
    direct_pose_loss_group_norm_ema_beta: float = 0.95,
    direct_pose_loss_group_norm_ratio_min: float = 0.2,
    direct_pose_loss_group_norm_ratio_max: float = 5.0,
    direct_pose_loss_group_norm_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = trainer.device
    dtype = next(model.parameters()).dtype

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
        except Exception:
            offset = 0

    # For wrap-boundary steps (idx==T-1), use y0 as the target pose (matches tiled freerun_cycles).
    y0_raw = None
    if include_boundary:
        try:
            motion0 = motion_seq[:, 0]
            motion0_raw = trainer.normalizer.denorm_x(motion0)
            y0_raw = _init_y_from_x(trainer.normalizer, motion0_raw, Dy)
        except Exception:
            y0_raw = None

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

    pose_hist_enabled = bool(getattr(trainer, "pose_hist_len", 0) or 0) > 0 and bool(getattr(trainer, "pose_hist_dim", 0) or 0) > 0
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_stride = pose_hist_dim // pose_hist_len if pose_hist_len > 0 else 0
    scales = mu = std = None
    pose_hist_buffer_norm = None
    pose_hist_buffer_raw = None
    if pose_hist_enabled and pose_hist_stride > 0:
        try:
            scales, mu, std = trainer._pose_hist_params(motion_seq)
        except Exception:
            scales = mu = std = None
        if scales is None:
            pose_hist_enabled = False
        else:
            with torch.no_grad():
                if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0 and pose_hist_seq.dim() == 3:
                    pose_hist_buffer_norm = pose_hist_seq[:, int(offset)]
                    pose_hist_buffer_raw = trainer._pose_hist_inverse_vec(pose_hist_buffer_norm, scales, mu, std)
                else:
                    base_rot = y_prev_raw[..., rot_slice]
                    pose_hist_buffer_raw = (
                        base_rot.unsqueeze(1)
                        .repeat(1, pose_hist_len, 1)
                        .reshape(B, pose_hist_dim)
                    )
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

    # NOTE: let the model decide the initial plan_z when plan_z is None.
    # This allows using a learnable contact_plan_init_z (or falling back to zeros).
    plan_z = None
    phase_z = None
    phase_event_age = None

    step_weights = _make_rollout_step_weights(
        total_steps,
        device=device,
        dtype=dtype,
        mode=str(time_weight_mode or "inv"),
        max_val=float(time_weight_max or 1.0),
    )
    boundary_mask = None
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
        except Exception:
            boundary_mask = None
            boundary_steps = 0
            boundary_weighted_sum = 0.0

    # ---- Optional: focus objective="direct" on selected step_in_cycle indices (phase-locked hotspots) ----
    direct_pose_sic_focus_steps = 0
    direct_pose_sic_focus_weight_sum_before = 0.0
    direct_pose_sic_focus_weight_sum_after = 0.0
    direct_pose_sic_focus_mode = "mask"
    direct_pose_sic_focus_boost = 1.0
    try:
        sic_spec = str(direct_pose_loss_sics or "").strip()
    except Exception:
        sic_spec = ""
    if objective == "direct" and sic_spec:
        sics = _as_int_list(sic_spec) or []
        # Only keep non-negative indices; step_in_cycle is defined as idx % cycle_len.
        sics = [int(x) for x in sics if int(x) >= 0]
        try:
            cycle_gte = int(direct_pose_loss_cycle_gte or 0)
        except Exception:
            cycle_gte = 0
        cycle_gte = max(0, int(cycle_gte))
        if sics:
            try:
                sic_mode = str(direct_pose_loss_sic_mode or "mask").strip().lower()
                if sic_mode in ("", "none", "off", "disable", "disabled"):
                    sic_mode = "mask"
                if sic_mode not in ("mask", "boost"):
                    sic_mode = "mask"
                direct_pose_sic_focus_mode = sic_mode
                try:
                    boost = float(direct_pose_loss_sic_boost or 1.0)
                except Exception:
                    boost = 1.0
                if (not math.isfinite(boost)) or boost <= 0.0:
                    boost = 1.0
                direct_pose_sic_focus_boost = float(boost)

                ar = torch.arange(int(total_steps), device=device)
                if int(cycle_len) > 0:
                    step_in_cycle = (ar + int(offset)) % int(cycle_len)
                    cycle_idx = (ar + int(offset)) // int(cycle_len)
                else:
                    step_in_cycle = ar * 0
                    cycle_idx = ar * 0
                sic_mask = torch.zeros_like(step_in_cycle, dtype=torch.bool)
                for sic in sics:
                    sic_mask = sic_mask | (step_in_cycle == int(sic))
                if cycle_gte > 0:
                    sic_mask = sic_mask & (cycle_idx >= int(cycle_gte))
                # Always drop wrap boundary step (matches common eval mask "drop_wrap").
                if bool(include_boundary) and int(cycle_len) > 0:
                    sic_mask = sic_mask & (step_in_cycle != (int(cycle_len) - 1))
                if bool(sic_mask.any().detach().cpu().item()):
                    direct_pose_sic_focus_steps = int(sic_mask.sum().detach().cpu().item())
                    direct_pose_sic_focus_weight_sum_before = float(step_weights[sic_mask].sum().detach().cpu().item())
                    if direct_pose_sic_focus_mode == "mask":
                        step_weights = step_weights * sic_mask.to(dtype=dtype)
                    elif abs(float(direct_pose_sic_focus_boost) - 1.0) > 1e-12:
                        b = step_weights.new_tensor(float(direct_pose_sic_focus_boost))
                        step_weights = torch.where(sic_mask, step_weights * b, step_weights)
                    step_weights = step_weights / step_weights.sum().clamp_min(1e-6)
                    direct_pose_sic_focus_weight_sum_after = float(step_weights[sic_mask].sum().detach().cpu().item())
            except Exception:
                pass

    # ---- Optional: per-(sic,bone) direct objective weighting from binary hotspot mask ----
    # Table schema: {"alpha_by_sic_bone": {sic: {bone: alpha}}, "mask": {...}}
    # We use only non-neutral pairs (alpha != 1) as the hotspot set and apply a
    # constant multiplicative boost to those pair losses.
    direct_pose_pair_boost_enabled = False
    direct_pose_pair_boost = 1.0
    direct_pose_pair_cycle_gte = 0
    direct_pose_pair_drop_wrap = True
    direct_pose_pair_joint_idx_by_sic: Dict[int, List[int]] = {}
    direct_pose_pair_focus_steps = 0
    direct_pose_pair_focus_pairs = 0
    if objective == "direct" and isinstance(direct_pose_loss_pair_boost_table, dict) and direct_pose_loss_pair_boost_table:
        try:
            direct_pose_pair_boost = float(direct_pose_loss_pair_boost_table.get("boost", 1.0) or 1.0)
        except Exception:
            direct_pose_pair_boost = 1.0
        if (not math.isfinite(direct_pose_pair_boost)) or direct_pose_pair_boost <= 1.0:
            direct_pose_pair_boost = 1.0
        try:
            direct_pose_pair_cycle_gte = int(direct_pose_loss_pair_boost_table.get("cycle_gte", 0) or 0)
        except Exception:
            direct_pose_pair_cycle_gte = 0
        direct_pose_pair_cycle_gte = max(0, int(direct_pose_pair_cycle_gte))
        try:
            direct_pose_pair_drop_wrap = bool(direct_pose_loss_pair_boost_table.get("drop_wrap", True))
        except Exception:
            direct_pose_pair_drop_wrap = True
        raw_map = direct_pose_loss_pair_boost_table.get("joint_idx_by_sic", None)
        if isinstance(raw_map, dict):
            for k, vals in raw_map.items():
                try:
                    sic = int(k)
                except Exception:
                    continue
                idxs: List[int] = []
                if isinstance(vals, (list, tuple)):
                    for v in vals:
                        try:
                            j = int(v)
                        except Exception:
                            continue
                        if 0 <= j < int(J) and j not in idxs:
                            idxs.append(j)
                if idxs:
                    direct_pose_pair_joint_idx_by_sic[int(sic)] = idxs
        if direct_pose_pair_boost > 1.0 and direct_pose_pair_joint_idx_by_sic:
            direct_pose_pair_boost_enabled = True

    loss_terms = []
    inc_terms = []
    dir_terms = []
    dir_base_terms = []
    dir_tail_terms = []
    dir_tail_raw_terms = []
    dir_tail_alpha_terms = []
    hinge_sup_terms = []
    hinge_sup_frac_terms = []
    hinge_sup_abs_delta_tgt_deg_num_terms = []
    hinge_sup_abs_delta_pred_deg_num_terms = []
    hinge_sup_abs_delta_deg_den_terms = []
    hinge_gate_sup_terms = []
    hinge_gate_sup_frac_terms = []
    hinge_stance_terms = []
    hinge_stance_frac_terms = []
    hinge_reg_terms = []
    hinge_eps_l2_terms = []
    leg_side_gate_reg_terms = []
    leg_gate_sup_terms = []
    leg_gate_sup_tgt_frac_terms = []
    leg_gate_sup_pred_mean_terms = []
    leg_scale_sup_terms = []
    leg_scale_sup_tgt_mean_terms = []
    leg_scale_sup_pred_mean_terms = []
    leg_scale_sup_sign_terms = []
    leg_scale_sup_sign_tgt_mean_terms = []
    leg_scale_sup_sign_pred_mean_terms = []
    leg_align_terms = []
    leg_align_frac_terms = []
    ent_terms = []
    smooth_terms = []
    early_terms = []
    mono_terms = []
    plan_ent_terms = []
    plan_dyn_terms = []
    plan_ent_stat_terms = []
    plan_dyn_stat_terms = []
    meas_terms = []
    lam_vals = []        # raw λ from head (after clamp)
    lam_eff_vals = []    # effective λ used for blend (after r_t)
    lam_rel_vals = []    # reliability r_t (B,) when enabled
    boundary_blend_terms = []
    boundary_inc_terms = []
    boundary_dir_terms = []
    boundary_lam_terms = []
    boundary_lam_eff_terms = []
    boundary_r_terms = []
    gate_sup_terms = []
    gate_sup_frac_terms = []
    gate_sup_acc_num_terms = []
    gate_sup_acc_den_terms = []

    prev_foot_pos_meas = None
    time_base = None
    try:
        if isinstance(batch, dict):
            base = batch.get("start", None)
            if base is not None:
                if torch.is_tensor(base):
                    base = base.to(device=device)
                time_base = base
    except Exception:
        time_base = None

    lam_prev = None
    lam_prev_monot = None
    plan_prev = None
    meas_logits_prev = None
    td_hazard_acc = None  # (B,C) stateful when phase_reset_source=td_hazard
    time_index_mode = str(time_index_mode or "global").strip().lower()
    if time_index_mode == "auto":
        # Default to 'global' to avoid hard-reset discontinuities across cycle boundaries.
        time_index_mode = "global"
    if time_index_mode not in ("global", "cycle", "none"):
        time_index_mode = "global"
    # Note: do not pre-add offset to time_base here; time_index uses idx (which already includes offset).

    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))

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

    # ---- Stage6/7 direct objective: de-dilution weights (soft tail + optional swing boost) ----
    direct_tail_mix = float(direct_pose_loss_tail_mix or 0.0)
    if (not math.isfinite(direct_tail_mix)) or direct_tail_mix <= 0.0:
        direct_tail_mix = 0.0
    direct_tail_mix = max(0.0, min(1.0, float(direct_tail_mix)))

    direct_tail_temp_deg = float(direct_pose_loss_tail_temp_deg or 0.0)
    if (not math.isfinite(direct_tail_temp_deg)) or direct_tail_temp_deg <= 0.0:
        direct_tail_temp_deg = 0.0
    direct_tail_temp_rad = max(1e-6, direct_tail_temp_deg * (math.pi / 180.0)) if direct_tail_temp_deg > 0.0 else 0.0

    direct_state_boost = float(direct_pose_loss_state_swing_boost or 0.0)
    if (not math.isfinite(direct_state_boost)) or direct_state_boost <= 0.0:
        direct_state_boost = 0.0

    direct_state_contact_source = str(direct_pose_loss_state_contact_source or "gt").strip().lower()
    if direct_state_contact_source not in ("gt", "plan", "meas"):
        direct_state_contact_source = "gt"
    direct_state_scope = str(direct_pose_loss_state_scope or "legs").strip().lower()
    if direct_state_scope not in ("legs", "limbs", "all"):
        direct_state_scope = "legs"

    direct_state_side = None  # (J,) long in {-1,0,1}
    direct_state_mask = None  # (J,) bool
    if objective == "direct" and direct_tail_mix > 0.0 and direct_state_boost > 0.0:
        bone_names_src = getattr(getattr(trainer, "loss_fn", None), "bone_names", None) or getattr(trainer, "_bone_names", None)
        if not bone_names_src:
            meta = getattr(getattr(trainer, "loss_fn", None), "meta", None)
            if isinstance(meta, dict):
                bone_names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
        bone_names = [str(b) for b in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []

        side_idx: list[int] = [-1 for _ in range(int(J))]
        scope_mask: list[bool] = [False for _ in range(int(J))]
        for j in range(int(J)):
            nm = str(bone_names[j]) if j < len(bone_names) else ""
            s = nm.lower()
            if "_l" in s or s.endswith("l"):
                side_idx[j] = 0
            elif "_r" in s or s.endswith("r"):
                side_idx[j] = 1

            if direct_state_scope == "all":
                scope_mask[j] = True
            else:
                # legs: thigh/calf/foot/ball (+twist via substring match)
                # limbs: legs + upperarm/lowerarm/hand (matches MotionJointLoss.limb_monitor_names)
                if direct_state_scope == "limbs":
                    tokens = ("thigh", "calf", "foot", "ball", "upperarm", "lowerarm", "hand")
                else:
                    tokens = ("thigh", "calf", "foot", "ball")
                scope_mask[j] = any(tok in s for tok in tokens)

        side_t = torch.as_tensor(side_idx, device=device, dtype=torch.long)
        scope_t = torch.as_tensor(scope_mask, device=device, dtype=torch.bool)
        mask_t = (side_t >= 0) & scope_t
        if bool(mask_t.any().detach().cpu().item()):
            direct_state_side = side_t
            direct_state_mask = mask_t

    # ---- Stage7/B2: optional legs vs non-legs group-wise magnitude normalization ----
    direct_group_norm_enable = bool(objective == "direct" and bool(direct_pose_loss_group_norm_enable))
    try:
        direct_group_w_leg = float(direct_pose_loss_group_norm_w_leg or 1.0)
    except Exception:
        direct_group_w_leg = 1.0
    try:
        direct_group_w_nonleg = float(direct_pose_loss_group_norm_w_nonleg or 1.0)
    except Exception:
        direct_group_w_nonleg = 1.0
    try:
        direct_group_beta = float(direct_pose_loss_group_norm_ema_beta or 0.95)
    except Exception:
        direct_group_beta = 0.95
    if (not math.isfinite(direct_group_beta)) or direct_group_beta < 0.0:
        direct_group_beta = 0.95
    direct_group_beta = max(0.0, min(0.9999, float(direct_group_beta)))
    try:
        direct_group_ratio_min = float(direct_pose_loss_group_norm_ratio_min or 0.2)
    except Exception:
        direct_group_ratio_min = 0.2
    try:
        direct_group_ratio_max = float(direct_pose_loss_group_norm_ratio_max or 5.0)
    except Exception:
        direct_group_ratio_max = 5.0
    if (not math.isfinite(direct_group_ratio_min)) or direct_group_ratio_min <= 0.0:
        direct_group_ratio_min = 0.2
    if (not math.isfinite(direct_group_ratio_max)) or direct_group_ratio_max <= 0.0:
        direct_group_ratio_max = 5.0
    if direct_group_ratio_min > direct_group_ratio_max:
        direct_group_ratio_min, direct_group_ratio_max = direct_group_ratio_max, direct_group_ratio_min
    try:
        direct_group_eps = float(direct_pose_loss_group_norm_eps or 1e-6)
    except Exception:
        direct_group_eps = 1e-6
    if (not math.isfinite(direct_group_eps)) or direct_group_eps <= 0.0:
        direct_group_eps = 1e-6
    dir_leg_base_terms: list[torch.Tensor] = []
    dir_nonleg_base_terms: list[torch.Tensor] = []

    for t in range(total_steps):
        denom = int(cycle_len) if include_boundary else int(steps)
        idx = int((int(offset) + int(t)) % max(1, denom))
        cond_t = cond_seq[:, idx] if (torch.is_tensor(cond_seq) and cond_seq.dim() == 3) else cond_seq
        # Align condition feeding with eval free-run:
        # - Use raw cond_tgt_raw at (idx+1) as the "next-step" command.
        # - Reproject it into model's local yaw frame under drift.
        # - Normalize with per-window (cond_norm_mu/std).
        cond_raw_step = None
        if torch.is_tensor(cond_raw_tgt):
            if cond_raw_tgt.dim() == 3:
                if include_boundary:
                    idx_raw = int((int(idx) + 1) % int(cycle_len))
                else:
                    idx_raw = min(int(cond_raw_tgt.shape[1]) - 1, int(idx) + 1)
                cond_raw_step = cond_raw_tgt[:, idx_raw]
            else:
                cond_raw_step = cond_raw_tgt
        cond_raw_for_model = cond_raw_step
        if enable_reprojection and t > 0 and torch.is_tensor(cond_raw_step):
            yaw_gt = None
            try:
                if include_boundary and y0_raw is not None and int(idx) == (int(cycle_len) - 1):
                    gt_raw_frame = y0_raw
                else:
                    gt_idx = min(int(gt_seq.shape[1]) - 1, int(idx))
                    gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
                yaw_gt = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
            except Exception:
                yaw_gt = None
            yaw_pred = None
            try:
                yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
            except Exception:
                yaw_pred = None
            if yaw_gt is not None and yaw_pred is not None:
                try:
                    cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
                except Exception:
                    cond_proj = None
                if cond_proj is not None:
                    cond_raw_for_model = cond_proj
        if cond_raw_for_model is not None:
            try:
                cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
            except Exception:
                cond_override = None
            if cond_override is not None:
                cond_t = cond_override
        if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
            angvel_t = motion[..., trainer.angvel_x_slice].detach()
        else:
            angvel_t = angvel_seq[:, idx] if (torch.is_tensor(angvel_seq) and angvel_seq.dim() == 3) else angvel_seq
        if pose_hist_enabled and pose_hist_buffer_norm is not None:
            pose_hist_t = pose_hist_buffer_norm
        else:
            pose_hist_t = pose_hist_seq[:, idx] if (torch.is_tensor(pose_hist_seq) and pose_hist_seq.dim() == 3) else pose_hist_seq

        inp_motion = motion.unsqueeze(1)
        inp_cond = cond_t.unsqueeze(1) if torch.is_tensor(cond_t) and cond_t.dim() == 2 else cond_t
        inp_angvel = angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) and angvel_t.dim() == 2 else angvel_t
        inp_pose_hist = pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) and pose_hist_t.dim() == 2 else pose_hist_t

        plan_enable = bool(getattr(model, "contact_plan_enable", False))
        use_learned_meas = bool(getattr(model, "contact_meas_enable", False))
        init_mode = str(getattr(model, "contact_plan_init_mode", "learnable") or "learnable").strip().lower()
        contacts_wb_t = None
        if plan_enable and (
            (not use_learned_meas)
            or (init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0)
        ):
            try:
                contacts_wb_t, prev_foot_pos_meas = trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)
            except Exception:
                contacts_wb_t = None

        contacts_in_t = None
        if plan_enable:
            if not use_learned_meas:
                contacts_in_t = contacts_wb_t
            elif init_mode in ("obs", "learnable+obs") and plan_z is None and t == 0:
                contacts_in_t = contacts_wb_t

        if time_index_mode == "none":
            time_index_t = None
        elif time_index_mode == "cycle":
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    pass
        else:
            time_index_t = int(idx)
            if time_base is not None:
                try:
                    time_index_t = time_base + int(idx)
                except Exception:
                    time_index_t = int(idx)
            else:
                time_index_t = int(t)

        rollout_step_t = None
        try:
            if int(total_steps) > 1:
                step_norm = float(t) / float(int(total_steps) - 1)
            else:
                step_norm = 0.0
            rollout_step_t = torch.full((B, 1, 1), step_norm, device=device, dtype=dtype)
        except Exception:
            rollout_step_t = None

        ret = model(
            inp_motion,
            inp_cond,
            contacts=contacts_in_t,
            angvel=inp_angvel,
            pose_history=inp_pose_hist,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            td_hazard_acc=td_hazard_acc,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index_t,
            rollout_step=rollout_step_t,
        )
        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict.")

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

        if bool(getattr(model, "contact_plan_enable", False)):
            try:
                z_next = ret.get("plan_z_next", None)
                if torch.is_tensor(z_next):
                    plan_z = z_next.detach()
                p_next = ret.get("phase_z_next", None)
                if torch.is_tensor(p_next):
                    phase_z = p_next.detach()
                a_next = ret.get("phase_event_age_next", None)
                if torch.is_tensor(a_next):
                    phase_event_age = a_next.detach()
            except Exception:
                pass
        try:
            hz_acc_next = ret.get("td_hazard_acc_next", None)
            if torch.is_tensor(hz_acc_next):
                td_hazard_acc = hz_acc_next.detach()
        except Exception:
            pass
        try:
            mlog = ret.get("contacts_meas_logits", None)
            if torch.is_tensor(mlog):
                if mlog.dim() == 3:
                    meas_logits_prev = mlog[:, -1].detach()
                elif mlog.dim() == 2:
                    meas_logits_prev = mlog.detach()
        except Exception:
            pass

        if contacts_in_t is None and float(contact_meas_weight or 0.0) > 0.0 and torch.is_tensor(contacts_seq):
            try:
                gt_c_t = contacts_seq[:, idx] if contacts_seq.dim() == 3 else contacts_seq
                meas_logits = ret.get("contacts_meas_logits", None)
                if torch.is_tensor(meas_logits):
                    if meas_logits.dim() == 3:
                        meas_logits = meas_logits[:, -1]
                    if torch.is_tensor(gt_c_t) and gt_c_t.shape == meas_logits.shape:
                        meas_used_logits = True
                        gt = gt_c_t.clamp(0.0, 1.0)
                        meas_terms.append(F.binary_cross_entropy_with_logits(meas_logits, gt) * step_weights[t])
                else:
                    # If logits are missing, skip (avoid supervising contacts_input/white-box fallback).
                    pass
            except Exception:
                pass

        delta_raw = delta_norm * std_y

        prev6 = reproject_rot6d(y_prev_raw[..., rot_slice]).view(B, J, 6)
        R_prev = rot6d_to_matrix(prev6, columns=columns)

        if include_boundary and y0_raw is not None and int(idx) == (int(cycle_len) - 1):
            gt_raw = y0_raw
        else:
            gt_raw = trainer._denorm(gt_seq[:, idx])
        gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(B, J, 6)
        R_gt = rot6d_to_matrix(gt6, columns=columns)

        delta6 = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=columns)  # (B,J,6)
        R_delta = rot6d_to_matrix(delta6, columns=columns)
        R_inc = torch.matmul(R_delta, R_prev)

        direct_raw_base = trainer._denorm(direct_norm)
        # Optional: apply leg-specific SO(3) residual to the direct branch in RAW space.
        # NOTE: out_direct is in normalized Y space; model only outputs omega (axis-angle).
        # We denorm -> compose -> keep direct_raw_base in RAW space for geodesic loss.
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
                                # Match run_freerun_cycles: omega_oracle = so3_log_map(R_gt @ R_base^T) * 2.
                                align_w = float(direct_pose_leg_align_weight or 0.0)
                                if align_w > 0.0:
                                    try:
                                        with torch.no_grad():
                                            R_gt_leg = R_gt[:, idx_use, :, :]
                                            R_delta_oracle = torch.matmul(R_gt_leg, R_leg_base.transpose(-1, -2))
                                            omega_oracle = so3_log_map(R_delta_oracle) * 2.0  # full axis-angle
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
                                        except Exception:
                                            cos_thr = 0.0
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
                                        leg_align_terms.append(leg_align * step_weights[t])
                                        leg_align_frac_terms.append((w > 0.0).to(dtype=dtype).mean() * step_weights[t])
                                    except Exception:
                                        pass

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
                                                            omega_oracle = so3_log_map(R_delta_oracle) * 2.0  # full axis-angle
                                                            oracle_norm = omega_oracle.norm(dim=-1)
                                                    if torch.is_tensor(oracle_norm):
                                                        with torch.no_grad():
                                                            min_deg = float(direct_pose_leg_align_oracle_min_deg or 0.0)
                                                            min_rad = float(min_deg) * (math.pi / 180.0)
                                                            tgt = (oracle_norm >= float(min_rad)).to(device=device, dtype=dtype)
                                                        err = F.binary_cross_entropy_with_logits(gl, tgt, reduction="none")
                                                        leg_gate = err.mean()
                                                        leg_gate_sup_terms.append(leg_gate * step_weights[t])
                                                        leg_gate_sup_tgt_frac_terms.append(tgt.mean() * step_weights[t])
                                                        leg_gate_sup_pred_mean_terms.append(torch.sigmoid(gl).mean() * step_weights[t])
                                    except Exception:
                                        pass

                                # Optional: supervise learned leg scale head (log_mag) using an offline alpha-sweep table.
                                # Target: log_mag_target = log(max(best_alpha,0) + eps)  (eps is baked into the table).
                                leg_scale_sup_w = float(direct_pose_leg_scale_sup_weight or 0.0)
                                if leg_scale_sup_w > 0.0 and isinstance(direct_pose_leg_scale_sup_table, dict) and direct_pose_leg_scale_sup_table:
                                    try:
                                        pred_log = ret.get("direct_leg_scale_log", None)
                                        if torch.is_tensor(pred_log):
                                            if pred_log.dim() == 3:
                                                pred_log = pred_log[:, -1]
                                            if pred_log.dim() == 2 and pred_log.shape[0] == B:
                                                if torch.is_tensor(keep_mask) and pred_log.shape[1] == keep_mask.shape[0]:
                                                    pred_log = pred_log[:, keep_mask]
                                                if int(pred_log.shape[1]) == int(idx_use.numel()):
                                                    pl = pred_log.to(device=device, dtype=dtype)
                                                    # Apply the same mask semantics as the alpha-table generation:
                                                    # cycle>=cycle_gte and optionally drop wrap boundary step.
                                                    try:
                                                        cycle_gte = int(direct_pose_leg_scale_sup_table.get("cycle_gte", 1) or 1)
                                                    except Exception:
                                                        cycle_gte = 1
                                                    drop_wrap = bool(direct_pose_leg_scale_sup_table.get("drop_wrap", True))
                                                    try:
                                                        cyc_len = int(cycle_len) if int(cycle_len) > 0 else 0
                                                    except Exception:
                                                        cyc_len = 0
                                                    if cyc_len > 0:
                                                        step_in_cycle = int((int(t) + int(offset)) % int(cyc_len))
                                                        cyc_idx = int((int(t) + int(offset)) // int(cyc_len))
                                                    else:
                                                        step_in_cycle = int(t)
                                                        cyc_idx = 0
                                                    if cyc_idx >= int(cycle_gte) and (not (drop_wrap and cyc_len > 0 and step_in_cycle == (cyc_len - 1))):
                                                        tgt_by_sic = direct_pose_leg_scale_sup_table.get("tgt_log_by_sic", None)
                                                        tgt_default = direct_pose_leg_scale_sup_table.get("tgt_log_default", None)
                                                        if not isinstance(tgt_default, (list, tuple)):
                                                            tgt_default = [0.0 for _ in range(int(idx_use.numel()))]
                                                        tgt_vec = None
                                                        if isinstance(tgt_by_sic, dict):
                                                            tgt_vec = tgt_by_sic.get(int(step_in_cycle), None)
                                                            if tgt_vec is None:
                                                                # tolerate JSON-like string keys
                                                                tgt_vec = tgt_by_sic.get(str(int(step_in_cycle)), None)
                                                        if (
                                                            not isinstance(tgt_vec, (list, tuple))
                                                            or int(len(tgt_vec)) != int(idx_use.numel())
                                                        ):
                                                            tgt_vec = tgt_default
                                                        tgt = pl.new_tensor([float(x) for x in tgt_vec])  # (K,)
                                                        mse = (pl - tgt.unsqueeze(0)).pow(2).mean()
                                                        leg_scale_sup_terms.append(mse * step_weights[t])
                                                        leg_scale_sup_tgt_mean_terms.append(tgt.mean() * step_weights[t])
                                                        leg_scale_sup_pred_mean_terms.append(pl.mean() * step_weights[t])

                                                        # Optional: signed_scale sign supervision (BCEWithLogits on sign logits).
                                                        # Uses the same per-sic mask (cycle_gte + optional drop_wrap).
                                                        try:
                                                            tgt_sign_by_sic = direct_pose_leg_scale_sup_table.get("tgt_sign01_by_sic", None)
                                                            tgt_sign_default = direct_pose_leg_scale_sup_table.get("tgt_sign01_default", None)
                                                            if isinstance(tgt_sign_default, (list, tuple)) and int(len(tgt_sign_default)) != int(idx_use.numel()):
                                                                tgt_sign_default = None
                                                            if isinstance(tgt_sign_default, (list, tuple)) or isinstance(tgt_sign_by_sic, dict):
                                                                pred_sign_logit = ret.get("direct_leg_scale_sign_logit", None)
                                                                if torch.is_tensor(pred_sign_logit):
                                                                    if pred_sign_logit.dim() == 3:
                                                                        pred_sign_logit = pred_sign_logit[:, -1]
                                                                    if pred_sign_logit.dim() == 2 and pred_sign_logit.shape[0] == B:
                                                                        if torch.is_tensor(keep_mask) and pred_sign_logit.shape[1] == keep_mask.shape[0]:
                                                                            pred_sign_logit = pred_sign_logit[:, keep_mask]
                                                                        if int(pred_sign_logit.shape[1]) == int(idx_use.numel()):
                                                                            sl = pred_sign_logit.to(device=device, dtype=dtype)
                                                                            tgt_sign_vec = None
                                                                            if isinstance(tgt_sign_by_sic, dict):
                                                                                tgt_sign_vec = tgt_sign_by_sic.get(int(step_in_cycle), None)
                                                                                if tgt_sign_vec is None:
                                                                                    tgt_sign_vec = tgt_sign_by_sic.get(str(int(step_in_cycle)), None)
                                                                            if (
                                                                                not isinstance(tgt_sign_vec, (list, tuple))
                                                                                or int(len(tgt_sign_vec)) != int(idx_use.numel())
                                                                            ):
                                                                                tgt_sign_vec = tgt_sign_default
                                                                            if not isinstance(tgt_sign_vec, (list, tuple)):
                                                                                tgt_sign_vec = [1.0 for _ in range(int(idx_use.numel()))]
                                                                            tgt_sign = sl.new_tensor([float(x) for x in tgt_sign_vec])  # (K,)
                                                                            bce = F.binary_cross_entropy_with_logits(
                                                                                sl, tgt_sign.unsqueeze(0).expand_as(sl)
                                                                            )
                                                                            leg_scale_sup_sign_terms.append(bce * step_weights[t])
                                                                            leg_scale_sup_sign_tgt_mean_terms.append(
                                                                                tgt_sign.mean() * step_weights[t]
                                                                            )
                                                                            leg_scale_sup_sign_pred_mean_terms.append(
                                                                                torch.sigmoid(sl).mean() * step_weights[t]
                                                                            )
                                                        except Exception:
                                                            pass
                                    except Exception:
                                        pass

                                R_delta_leg = so3_exp_map(omega_leg)  # (B,K,3,3)
                                R_leg = torch.matmul(R_delta_leg, R_leg_base)
                                R_final = R_base.clone()
                                R_final[:, idx_use, :, :] = R_leg
                                rot6_final = matrix_to_rot6d(R_final, columns=columns).view(B, rot_len)
                                direct_raw_base = direct_raw_base.clone()
                                direct_raw_base[..., rot_slice] = rot6_final
        except Exception:
            pass
        # Optional: regularize per-side sign gate (encourage |g|->1 so it acts like a sign, not a shrink-to-zero knob).
        gate_reg_w = float(direct_pose_leg_side_sign_gate_reg_weight or 0.0)
        if gate_reg_w > 0.0:
            try:
                g = ret.get("direct_leg_side_sign_gate", None)
                if torch.is_tensor(g):
                    # Expect shape (B,2) for single-step; tolerate (B,1,2).
                    if g.dim() == 3 and g.size(1) == 1:
                        g = g[:, 0]
                    if g.dim() == 2 and g.shape[0] == B and int(g.shape[-1]) == 2:
                        g = g.to(device=device, dtype=dtype).clamp(-1.0, 1.0)
                        # (1 - |g|)^2 encourages saturation at ±1 without preferring a sign.
                        reg = (1.0 - g.abs()).pow(2).mean()
                        leg_side_gate_reg_terms.append(reg * step_weights[t])
            except Exception:
                pass

        hinge_step = None
        try:
            hinge_step = ret.get("direct_hinge_delta", None)
            if torch.is_tensor(hinge_step):
                if hinge_step.dim() == 3:
                    hinge_step = hinge_step[:, -1]
                if hinge_step.dim() != 2 or hinge_step.shape[0] != B:
                    hinge_step = None
        except Exception:
            hinge_step = None

        # Raw hinge delta (before contact gating). Prefer supervising this to avoid vanishing gradients
        # when gate_power is large (hinge_step = raw * gate).
        hinge_step_raw = None
        try:
            hinge_step_raw = ret.get("direct_hinge_delta_raw", None)
            if torch.is_tensor(hinge_step_raw):
                if hinge_step_raw.dim() == 3:
                    hinge_step_raw = hinge_step_raw[:, -1]
                if hinge_step_raw.dim() != 2 or hinge_step_raw.shape[0] != B:
                    hinge_step_raw = None
        except Exception:
            hinge_step_raw = None

        # Optional: L2 penalty on eps(hidden) contribution (clean hinge split).
        hinge_eps_step_raw = None
        try:
            hinge_eps_step_raw = ret.get("direct_hinge_delta_eps_raw", None)
            if torch.is_tensor(hinge_eps_step_raw):
                if hinge_eps_step_raw.dim() == 3:
                    hinge_eps_step_raw = hinge_eps_step_raw[:, -1]
                if hinge_eps_step_raw.dim() != 2 or hinge_eps_step_raw.shape[0] != B:
                    hinge_eps_step_raw = None
        except Exception:
            hinge_eps_step_raw = None
        eps_l2_w = float(direct_pose_hinge_eps_l2_weight or 0.0)
        if eps_l2_w > 0.0 and torch.is_tensor(hinge_eps_step_raw):
            try:
                hinge_eps_l2 = hinge_eps_step_raw.pow(2).mean()
                hinge_eps_l2_terms.append(hinge_eps_l2 * step_weights[t])
            except Exception:
                pass

        hinge_gate_logits = None
        try:
            hinge_gate_logits = ret.get("direct_hinge_gate_logits", None)
            if torch.is_tensor(hinge_gate_logits):
                if hinge_gate_logits.dim() == 3:
                    hinge_gate_logits = hinge_gate_logits[:, -1]
                if hinge_gate_logits.dim() != 2 or hinge_gate_logits.shape[0] != B:
                    hinge_gate_logits = None
        except Exception:
            hinge_gate_logits = None

        # Optional: supervised hinge delta regression (delta_target) to stabilize sign/magnitude.
        # delta_target is computed from base direct error:
        #   R_err = R_base^T @ R_gt
        #   omega = log(R_err)  (axis-angle vector, rad)
        #   delta_target = omega[axis]
        sup_w = float(direct_pose_hinge_sup_weight or 0.0)
        if sup_w > 0.0 and torch.is_tensor(hinge_step_raw):
            try:
                hinge_idx = getattr(trainer, "direct_pose_hinge_joint_idx", None)
                if isinstance(hinge_idx, (list, tuple)) and hinge_idx:
                    hinge_idx = [int(i) for i in hinge_idx]
                else:
                    hinge_idx = []

                axis = str(getattr(trainer, "direct_pose_hinge_axis", "Z") or "Z").strip().upper()
                axis_i = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)
                max_rad = getattr(trainer, "direct_pose_hinge_max_rad", None)
                try:
                    max_rad = float(max_rad) if max_rad is not None else None
                except Exception:
                    max_rad = None

                if (
                    hinge_idx
                    and max(hinge_idx) < int(J)
                    and torch.is_tensor(hinge_step_raw)
                    and int(hinge_step_raw.shape[-1]) == int(len(hinge_idx))
                ):
                    with torch.no_grad():
                        base6 = reproject_rot6d(direct_raw_base[..., rot_slice]).view(B, J, 6)
                        R_base = rot6d_to_matrix(base6, columns=columns)
                        R_err = torch.matmul(R_base.transpose(-1, -2), R_gt)
                        # NOTE: train.geometry.so3_log_map returns a half-angle rotation vector historically, hence *2.
                        omega_err = so3_log_map(R_err) * 2.0  # (B,J,3)
                        omega_h = omega_err[:, hinge_idx]  # (B,K,3)
                        # Axis-oracle target: best axis-only correction angle (radians) that matches R_err.
                        # This is NOT the same as omega_h[..., axis] when swing is non-trivial.
                        R_h = R_err[:, hinge_idx]  # (B,K,3,3)
                        if int(axis_i) == 0:  # X
                            delta_tgt = torch.atan2(
                                R_h[..., 2, 1] - R_h[..., 1, 2],
                                R_h[..., 1, 1] + R_h[..., 2, 2],
                            )
                        elif int(axis_i) == 1:  # Y
                            delta_tgt = torch.atan2(
                                R_h[..., 0, 2] - R_h[..., 2, 0],
                                R_h[..., 0, 0] + R_h[..., 2, 2],
                            )
                        else:  # Z
                            delta_tgt = torch.atan2(
                                R_h[..., 1, 0] - R_h[..., 0, 1],
                                R_h[..., 0, 0] + R_h[..., 1, 1],
                            )
                        if max_rad is not None and max_rad > 0.0 and math.isfinite(max_rad):
                            delta_tgt = delta_tgt.clamp(-max_rad, max_rad)

                        # Build sample weights (B,K).
                        w_h = torch.ones_like(delta_tgt, dtype=dtype, device=device)
                        w_contact = None  # (B,K) after contact masking, before angle masking

                        # Contact mask (optional).
                        contact_val = direct_pose_hinge_sup_contact_value
                        if contact_val is not None:
                            try:
                                contact_val = int(contact_val)
                            except Exception:
                                contact_val = None
                        if contact_val in (0, 1):
                            c_src = str(direct_pose_hinge_sup_contact_source or "gt").strip().lower()
                            c_t = None  # (B,C)
                            if c_src == "gt":
                                cg = batch.get("contacts", None)
                                if torch.is_tensor(cg) and cg.dim() == 3:
                                    if int(idx) < int(cg.shape[1]):
                                        c_t = cg[:, int(idx)].to(device=device, dtype=dtype)
                            elif c_src == "plan":
                                cp = ret.get("contacts_plan", None)
                                if torch.is_tensor(cp):
                                    if cp.dim() == 3:
                                        c_t = cp[:, -1].to(device=device, dtype=dtype)
                                    elif cp.dim() == 2:
                                        c_t = cp.to(device=device, dtype=dtype)
                            elif c_src == "meas":
                                cm = ret.get("contacts_meas", None)
                                if torch.is_tensor(cm):
                                    if cm.dim() == 3:
                                        c_t = cm[:, -1].to(device=device, dtype=dtype)
                                    elif cm.dim() == 2:
                                        c_t = cm.to(device=device, dtype=dtype)

                            if torch.is_tensor(c_t) and c_t.dim() == 2 and int(c_t.shape[0]) == int(B):
                                cidxs = getattr(model, "direct_pose_hinge_gate_contact_idx", None)
                                if not (isinstance(cidxs, list) and len(cidxs) == len(hinge_idx)):
                                    cidxs = [1 if int(getattr(model, "contact_dim", 0) or 0) >= 2 else 0] * len(hinge_idx)
                                thr = float(direct_pose_hinge_sup_contact_thresh or 0.5)
                                thr = thr if thr > 0.0 else 0.5
                                for k, cidx in enumerate(cidxs):
                                    try:
                                        cidx = int(cidx)
                                    except Exception:
                                        cidx = -1
                                    if cidx < 0 or cidx >= int(c_t.shape[1]) or k >= int(w_h.shape[-1]):
                                        continue
                                    c = c_t[:, cidx]
                                    mk = (c >= thr) if int(contact_val) == 1 else (c < thr)
                                    w_h[:, k] = w_h[:, k] * mk.to(dtype=w_h.dtype)
                        w_contact = w_h.detach()

                        # Error threshold mask (optional, based on base direct error angle).
                        th_deg = float(direct_pose_hinge_sup_angle_thresh_deg or 0.0)
                        mk_ang = None
                        ang_deg = None
                        if th_deg > 0.0 and math.isfinite(th_deg):
                            ang_deg = omega_h.norm(dim=-1) * (180.0 / math.pi)
                            mk_ang = (ang_deg >= th_deg).to(dtype=w_h.dtype)
                            w_h = w_h * mk_ang
                        else:
                            mk_ang = torch.ones_like(w_h, dtype=w_h.dtype)

                        # Delta magnitude hard-mining / weighting (optional, based on axis-oracle delta_tgt).
                        dt_thr = float(direct_pose_hinge_sup_delta_thresh_deg or 0.0)
                        if dt_thr > 0.0 and math.isfinite(dt_thr):
                            dt_deg = delta_tgt.detach().abs() * (180.0 / math.pi)
                            w_h = w_h * (dt_deg >= dt_thr).to(dtype=w_h.dtype)
                        dt_pow = float(direct_pose_hinge_sup_delta_weight_power or 0.0)
                        if dt_pow > 0.0 and math.isfinite(dt_pow):
                            # NOTE: posttrain often runs with batch=1; we intentionally keep weights "absolute"
                            # (i.e., not normalized by sum(w)) so the weighting still affects gradients.
                            dt_scale_deg = float(direct_pose_hinge_sup_delta_weight_scale_deg or 0.0)
                            if dt_scale_deg > 0.0 and math.isfinite(dt_scale_deg):
                                denom = float(dt_scale_deg) * (math.pi / 180.0)
                                denom = denom if denom > 0.0 else float(math.pi)
                                w_delta = delta_tgt.detach().abs() / float(denom)
                                w_delta = w_delta.clamp_min(0.0)
                            else:
                                # Legacy: normalize by max_rad and clamp to [0,1].
                                denom = max_rad
                                if denom is None or (not math.isfinite(float(denom))) or float(denom) <= 0.0:
                                    denom = float(math.pi)
                                w_delta = (delta_tgt.detach().abs() / float(denom)).clamp(0.0, 1.0)

                            if abs(dt_pow - 1.0) > 1e-12:
                                w_delta = w_delta.pow(dt_pow)

                            dt_w_max = float(direct_pose_hinge_sup_delta_weight_max or 0.0)
                            if dt_w_max > 0.0 and math.isfinite(dt_w_max):
                                w_delta = w_delta.clamp_max(dt_w_max)

                            w_h = w_h * w_delta.to(dtype=w_h.dtype)

                        # Optional debug: accumulate hinge supervision mask/target stats by rollout phase idx.
                        # Enable with: DEBUG_HINGE_SUP_PHASE=1 python -m train.posttrain --config ...
                        try:
                            dbg_flag = str(os.environ.get("DEBUG_HINGE_SUP_PHASE", "0") or "0").strip().lower()
                            if dbg_flag in ("1", "true", "yes", "y", "on"):
                                K = int(w_h.shape[-1])
                                Bk_total = int(w_h.shape[0])

                                # Use absolute phase (start + idx) when available to avoid sliding-window phase mixing.
                                ph_b = None  # (B,) int64 on device
                                try:
                                    base = batch.get("start", None) if isinstance(batch, dict) else None
                                    if torch.is_tensor(base):
                                        base_t = base.to(device=device)
                                        if base_t.dim() == 0:
                                            base_t = base_t.view(1).expand(Bk_total)
                                        else:
                                            base_t = base_t.reshape(-1)
                                            if base_t.numel() == 1 and Bk_total > 1:
                                                base_t = base_t.expand(Bk_total)
                                            elif base_t.numel() != Bk_total:
                                                base_t = base_t[:Bk_total]
                                        if base_t.numel() == Bk_total:
                                            ph_b = (base_t.to(dtype=torch.int64) + int(idx)).to(device=device)
                                    elif base is not None:
                                        ph_b = torch.full(
                                            (Bk_total,),
                                            int(base) + int(idx),
                                            device=device,
                                            dtype=torch.int64,
                                        )
                                except Exception:
                                    ph_b = None
                                if ph_b is None:
                                    ph_b = torch.full((Bk_total,), int(idx), device=device, dtype=torch.int64)

                                dbg = getattr(trainer, "_debug_hinge_sup_phase", None)
                                if not isinstance(dbg, dict):
                                    dbg = {
                                        "note": "Per-phase hinge supervision stats (sums). Divide sums by n_total or n_w_final as needed.",
                                        "axis": axis,
                                        "hinge_joint_idx": [int(i) for i in hinge_idx],
                                        "contact_val": contact_val,
                                        "contact_source": str(direct_pose_hinge_sup_contact_source or "gt"),
                                        "contact_thresh": float(direct_pose_hinge_sup_contact_thresh or 0.5),
                                        "angle_thresh_deg": float(th_deg),
                                        "by_phase": {},
                                    }
                                    setattr(trainer, "_debug_hinge_sup_phase", dbg)

                                rad2deg = float(180.0 / math.pi)
                                by_phase = dbg.setdefault("by_phase", {})

                                # Aggregate per unique phase within the batch.
                                try:
                                    phases = ph_b.detach().unique(sorted=True).to(device="cpu").tolist()
                                except Exception:
                                    phases = [int(idx)]

                                for ph in phases:
                                    try:
                                        ph_i = int(ph)
                                    except Exception:
                                        ph_i = int(idx)
                                    m = (ph_b == int(ph_i))
                                    try:
                                        b_cnt = int(m.sum().detach().cpu().item())
                                    except Exception:
                                        b_cnt = int(Bk_total)
                                    if b_cnt <= 0:
                                        continue

                                    # Per-hinge sums over selected batch rows (b_cnt,K) -> (K,).
                                    w_sel = w_h[m]
                                    wc_src = w_contact if torch.is_tensor(w_contact) else w_h
                                    wc_sum = wc_src[m].sum(dim=0).detach().cpu().tolist()
                                    wa_src = mk_ang if torch.is_tensor(mk_ang) else torch.ones_like(w_h)
                                    wa_sum = wa_src[m].sum(dim=0).detach().cpu().tolist()
                                    wf_sum = w_sel.sum(dim=0).detach().cpu().tolist()

                                    dt_abs = (delta_tgt.detach().abs()[m] * w_sel).sum(dim=0) * rad2deg
                                    pr_abs = (hinge_step_raw.detach().abs()[m] * w_sel).sum(dim=0) * rad2deg
                                    dt_abs = dt_abs.detach().cpu().tolist()
                                    pr_abs = pr_abs.detach().cpu().tolist()

                                    # Error angle stats (already in degrees if available).
                                    ang_abs = None
                                    if torch.is_tensor(ang_deg):
                                        try:
                                            ang_abs = (ang_deg.detach().abs()[m] * w_sel).sum(dim=0).detach().cpu().tolist()
                                        except Exception:
                                            ang_abs = None

                                    rec = by_phase.get(str(ph_i))
                                    if not isinstance(rec, dict):
                                        rec = {
                                            "n_total": 0,  # total samples per hinge joint summed across batches: += B
                                            "w_contact_sum": [0.0] * K,
                                            "w_angle_sum": [0.0] * K,
                                            "w_final_sum": [0.0] * K,
                                            "abs_delta_tgt_deg_wsum": [0.0] * K,
                                            "abs_pred_deg_wsum": [0.0] * K,
                                            "abs_ang_deg_wsum": [0.0] * K,
                                        }
                                        by_phase[str(ph_i)] = rec

                                    rec["n_total"] = int(rec.get("n_total", 0)) + int(b_cnt)
                                    for k in range(K):
                                        rec["w_contact_sum"][k] += float(wc_sum[k]) if k < len(wc_sum) else 0.0
                                        rec["w_angle_sum"][k] += float(wa_sum[k]) if k < len(wa_sum) else 0.0
                                        rec["w_final_sum"][k] += float(wf_sum[k]) if k < len(wf_sum) else 0.0
                                        rec["abs_delta_tgt_deg_wsum"][k] += float(dt_abs[k]) if k < len(dt_abs) else 0.0
                                        rec["abs_pred_deg_wsum"][k] += float(pr_abs[k]) if k < len(pr_abs) else 0.0
                                        if ang_abs is not None and k < len(ang_abs):
                                            rec["abs_ang_deg_wsum"][k] += float(ang_abs[k])
                        except Exception:
                            pass

                    # Loss (differentiable only wrt hinge_step_raw).
                    kind = str(direct_pose_hinge_sup_kind or "smooth_l1").strip().lower()
                    if kind in ("apply_geo", "geo_apply", "apply_geodesic", "geodesic_apply"):
                        # Closed-loop hinge supervision: directly minimize the residual *after apply*.
                        # By left-invariance:
                        #   geodesic(R_base @ R_delta_pred, R_gt) == geodesic(R_delta_pred, R_base^T @ R_gt) == geodesic(R_delta_pred, R_err)
                        # Here R_err (== R_h) is computed under no_grad so the loss only trains hinge_step_raw.
                        try:
                            K = int(hinge_step_raw.shape[-1])
                            omega_pred = hinge_step_raw.new_zeros((B, K, 3))
                            omega_pred[..., axis_i] = hinge_step_raw
                            R_delta_pred = so3_exp_map(omega_pred)
                            per = _geodesic_R_safe(R_delta_pred, R_h).to(dtype=dtype)
                        except Exception:
                            # Fallback to angle regression (should be rare; keeps training running).
                            per = F.smooth_l1_loss(hinge_step_raw, delta_tgt, reduction="none")
                    elif kind in ("l2", "mse", "sq"):
                        per = (hinge_step_raw - delta_tgt).pow(2)
                    else:
                        per = F.smooth_l1_loss(hinge_step_raw, delta_tgt, reduction="none")

                    # Diagnostics: track mean |delta_target| and |delta_pred| (deg) on the supervised subset.
                    try:
                        # Use mask count (not sum(w)) so continuous weights (e.g. w_delta) don't cancel out in logs.
                        w_mask = (w_h > 0.0).to(dtype=w_h.dtype)
                        den = w_mask.sum()
                        if torch.is_tensor(den):
                            rad2deg = float(180.0 / math.pi)
                            hinge_sup_abs_delta_deg_den_terms.append(den.detach() * step_weights[t])
                            hinge_sup_abs_delta_tgt_deg_num_terms.append((delta_tgt.detach().abs() * w_mask).sum().detach() * rad2deg * step_weights[t])
                            hinge_sup_abs_delta_pred_deg_num_terms.append((hinge_step_raw.detach().abs() * w_mask).sum().detach() * rad2deg * step_weights[t])
                    except Exception:
                        pass

                    # IMPORTANT: denom must be based on mask count (not sum(w)), otherwise in the common
                    # batch=1,K=1 case any continuous weights cancel out: (per*w)/(w) -> per.
                    denom_h = (w_h > 0.0).to(dtype=w_h.dtype).sum().clamp_min(1.0)
                    hinge_sup = (per * w_h).sum() / denom_h
                    hinge_sup_terms.append(hinge_sup * step_weights[t])
                    # Optional debug: verify hinge_sup gradients reach hinge_step_raw (esp. apply_geo path).
                    # Enable with: DEBUG_HINGE_SUP_GRAD=1 python -m train.posttrain --config ...
                    try:
                        dbg_flag = str(os.environ.get("DEBUG_HINGE_SUP_GRAD", "0") or "0").strip().lower()
                        if dbg_flag in ("1", "true", "yes", "y", "on") and (not bool(getattr(trainer, "_debug_hinge_sup_grad_printed", False))):
                            g = None
                            try:
                                g = torch.autograd.grad(hinge_sup, hinge_step_raw, retain_graph=True, allow_unused=True)[0]
                            except Exception:
                                g = None
                            if g is None or (not torch.is_tensor(g)):
                                g_abs_mean = float("nan")
                                g_abs_max = float("nan")
                            else:
                                g_abs = g.detach().abs()
                                g_abs_mean = float(g_abs.mean().cpu())
                                g_abs_max = float(g_abs.max().cpu())
                            try:
                                w_mean = float(w_h.detach().mean().cpu())
                            except Exception:
                                w_mean = float("nan")
                            try:
                                per_mean = float(per.detach().mean().cpu())
                            except Exception:
                                per_mean = float("nan")
                            try:
                                dt_mean_deg = float((delta_tgt.detach().abs() * (180.0 / math.pi)).mean().cpu())
                            except Exception:
                                dt_mean_deg = float("nan")
                            try:
                                pr_mean_deg = float((hinge_step_raw.detach().abs() * (180.0 / math.pi)).mean().cpu())
                            except Exception:
                                pr_mean_deg = float("nan")
                            print(
                                f"[debug][hinge_sup_grad] kind={kind} hinge_sup={float(hinge_sup.detach().cpu()):.6e} "
                                f"grad|mean={g_abs_mean:.3e} grad|max={g_abs_max:.3e} "
                                f"per_mean={per_mean:.3e} w_mean={w_mean:.3e} "
                                f"|delta_tgt|={dt_mean_deg:.2f}deg |delta_pred|={pr_mean_deg:.2f}deg"
                            )
                            setattr(trainer, "_debug_hinge_sup_grad_printed", True)
                    except Exception:
                        pass
                    try:
                        hinge_sup_frac_terms.append((w_h > 0.0).to(dtype=dtype).mean() * step_weights[t])
                    except Exception:
                        pass
            except Exception:
                pass

        # Optional: supervise learned hinge gate (swing=1, stance=0) using contact thresholding.
        gate_sup_w = float(direct_pose_hinge_gate_sup_weight or 0.0)
        if gate_sup_w > 0.0 and torch.is_tensor(hinge_gate_logits):
            try:
                c_src = str(direct_pose_hinge_gate_sup_contact_source or "gt").strip().lower()
                c_t = None  # (B,C)
                if c_src == "gt":
                    cg = batch.get("contacts", None)
                    if torch.is_tensor(cg) and cg.dim() == 3:
                        if int(idx) < int(cg.shape[1]):
                            c_t = cg[:, int(idx)].to(device=device, dtype=dtype)
                elif c_src == "plan":
                    cp = ret.get("contacts_plan", None)
                    if torch.is_tensor(cp):
                        if cp.dim() == 3:
                            c_t = cp[:, -1].to(device=device, dtype=dtype)
                        elif cp.dim() == 2:
                            c_t = cp.to(device=device, dtype=dtype)
                elif c_src == "meas":
                    cm = ret.get("contacts_meas", None)
                    if torch.is_tensor(cm):
                        if cm.dim() == 3:
                            c_t = cm[:, -1].to(device=device, dtype=dtype)
                        elif cm.dim() == 2:
                            c_t = cm.to(device=device, dtype=dtype)

                if torch.is_tensor(c_t) and c_t.dim() == 2 and int(c_t.shape[0]) == int(B):
                    K = int(hinge_gate_logits.shape[-1])
                    cidxs = getattr(model, "direct_pose_hinge_gate_contact_idx", None)
                    if not (isinstance(cidxs, list) and len(cidxs) >= K):
                        cidxs = [1 if int(getattr(model, "contact_dim", 0) or 0) >= 2 else 0] * K
                    thr = float(direct_pose_hinge_gate_sup_contact_thresh or 0.5)
                    thr = thr if thr > 0.0 else 0.5

                    w_h = hinge_gate_logits.new_zeros((B, K))
                    tgt = hinge_gate_logits.new_zeros((B, K))
                    for k, cidx in enumerate(cidxs[:K]):
                        try:
                            cidx = int(cidx)
                        except Exception:
                            cidx = -1
                        if cidx < 0 or cidx >= int(c_t.shape[1]) or k >= int(w_h.shape[-1]):
                            continue
                        c = c_t[:, cidx]
                        w_h[:, k] = 1.0
                        tgt[:, k] = (c < thr).to(dtype=tgt.dtype)  # swing=1, stance=0

                    err = F.binary_cross_entropy_with_logits(hinge_gate_logits, tgt, reduction="none")
                    denom = w_h.sum().clamp_min(1.0)
                    gate_sup = (err * w_h).sum() / denom
                    hinge_gate_sup_terms.append(gate_sup * step_weights[t])
                    try:
                        hinge_gate_sup_frac_terms.append((w_h > 0.0).to(dtype=dtype).mean() * step_weights[t])
                    except Exception:
                        pass
            except Exception:
                pass

        # Optional: suppress hinge corrections on stance frames (safety term when gate_mode=none).
        stance_w = float(direct_pose_hinge_stance_weight or 0.0)
        if stance_w > 0.0 and torch.is_tensor(hinge_step_raw):
            try:
                c_src = str(direct_pose_hinge_stance_contact_source or "gt").strip().lower()
                c_t = None  # (B,C)
                if c_src == "gt":
                    cg = batch.get("contacts", None)
                    if torch.is_tensor(cg) and cg.dim() == 3:
                        if int(idx) < int(cg.shape[1]):
                            c_t = cg[:, int(idx)].to(device=device, dtype=dtype)
                elif c_src == "plan":
                    cp = ret.get("contacts_plan", None)
                    if torch.is_tensor(cp):
                        if cp.dim() == 3:
                            c_t = cp[:, -1].to(device=device, dtype=dtype)
                        elif cp.dim() == 2:
                            c_t = cp.to(device=device, dtype=dtype)
                elif c_src == "meas":
                    cm = ret.get("contacts_meas", None)
                    if torch.is_tensor(cm):
                        if cm.dim() == 3:
                            c_t = cm[:, -1].to(device=device, dtype=dtype)
                        elif cm.dim() == 2:
                            c_t = cm.to(device=device, dtype=dtype)

                if torch.is_tensor(c_t) and c_t.dim() == 2 and int(c_t.shape[0]) == int(B):
                    K = int(hinge_step_raw.shape[-1])
                    cidxs = getattr(model, "direct_pose_hinge_gate_contact_idx", None)
                    if not (isinstance(cidxs, list) and len(cidxs) >= K):
                        cidxs = [1 if int(getattr(model, "contact_dim", 0) or 0) >= 2 else 0] * K
                    thr = float(direct_pose_hinge_stance_contact_thresh or 0.5)
                    thr = thr if thr > 0.0 else 0.5

                    w_h = hinge_step_raw.new_ones((B, K))
                    for k, cidx in enumerate(cidxs[:K]):
                        try:
                            cidx = int(cidx)
                        except Exception:
                            cidx = -1
                        if cidx < 0 or cidx >= int(c_t.shape[1]) or k >= int(w_h.shape[-1]):
                            continue
                        c = c_t[:, cidx]
                        mk = (c >= thr)
                        w_h[:, k] = w_h[:, k] * mk.to(dtype=w_h.dtype)

                    kind = str(direct_pose_hinge_stance_kind or "l2").strip().lower()
                    if kind in ("smooth_l1", "huber"):
                        per = F.smooth_l1_loss(hinge_step_raw, hinge_step_raw.new_zeros(hinge_step_raw.shape), reduction="none")
                    else:
                        per = hinge_step_raw.pow(2)

                    denom_h = w_h.sum().clamp_min(1.0)
                    hinge_stance = (per * w_h).sum() / denom_h
                    hinge_stance_terms.append(hinge_stance * step_weights[t])
                    try:
                        hinge_stance_frac_terms.append((w_h > 0.0).to(dtype=dtype).mean() * step_weights[t])
                    except Exception:
                        pass
            except Exception:
                pass

        # Optional: contact-free regularizer on hinge delta magnitude.
        # This is intentionally independent of contact signals to avoid train/infer mismatch.
        reg_w = float(direct_pose_hinge_reg_weight or 0.0)
        if reg_w > 0.0:
            try:
                x = hinge_step if torch.is_tensor(hinge_step) else hinge_step_raw
                if torch.is_tensor(x):
                    kind = str(direct_pose_hinge_reg_kind or "l1").strip().lower()
                    if kind in ("l1", "abs"):
                        hinge_reg = x.abs().mean()
                    elif kind in ("smooth_l1", "huber"):
                        hinge_reg = F.smooth_l1_loss(x, x.new_zeros(x.shape), reduction="mean")
                    else:
                        hinge_reg = x.pow(2).mean()
                    hinge_reg_terms.append(hinge_reg * step_weights[t])
            except Exception:
                pass

        # Optional: hinge-style correction on the direct branch before computing errors / blend.
        # Without this, direct_pose_hinge_head never receives gradients under posttrain rollout loss.
        direct_raw = direct_raw_base
        if torch.is_tensor(hinge_step):
            try:
                direct_raw = trainer._apply_direct_hinge_correction_raw(direct_raw_base, hinge_step)
            except Exception:
                direct_raw = direct_raw_base

        dir6 = reproject_rot6d(direct_raw[..., rot_slice]).view(B, J, 6)
        R_dir = rot6d_to_matrix(dir6, columns=columns)

        lam = lam.to(device=device, dtype=dtype)
        if lam.ndim == 2 and lam.shape[-1] == 1:
            lam = lam.expand(B, J)
        if lam.shape[-1] != J:
            raise RuntimeError(f"lambda_fusion has wrong shape {tuple(lam.shape)} (expected (B,{J}))")
        lam = lam.clamp(0.0, 1.0)
        lam_raw = lam

        # Apply shared reliability r_t (deterministic, no grad) to avoid train/infer mismatch.
        lam_eff = lam_raw
        lam_rel = None
        try:
            lam_eff, lam_rel = trainer._lambda_fusion_apply_reliability(
                lam_raw,
                step_idx=int(t),
                total_steps=int(total_steps),
                rollout_step=rollout_step_t,
                ret=ret,
            )
        except Exception:
            lam_eff, lam_rel = lam_raw, None
        if lam_eff is None or (not torch.is_tensor(lam_eff)):
            lam_eff = lam_raw

        lam_vals.append(lam_raw.detach())
        lam_eff_vals.append(lam_eff.detach())
        if torch.is_tensor(lam_rel):
            lam_rel_vals.append(lam_rel.detach())

        # ---- Optional reliability terms from contact_plan stability ----
        plan_step = None
        try:
            plan_step = ret.get("contacts_plan", None)
            if torch.is_tensor(plan_step):
                if plan_step.dim() == 3:
                    plan_step = plan_step[:, -1]
                if plan_step.dim() != 2:
                    plan_step = None
        except Exception:
            plan_step = None

        if (float(lambda_plan_entropy_weight or 0.0) > 0.0 or float(lambda_plan_dyn_weight or 0.0) > 0.0) and torch.is_tensor(plan_step):
            try:
                plan_det = plan_step.detach()
                ent = _lambda_entropy(plan_det).mean(dim=-1)  # (B,)
                plan_ent_stat_terms.append(ent.mean() * step_weights[t])
                if float(lambda_plan_entropy_weight or 0.0) > 0.0:
                    plan_ent_terms.append((lam_eff.mean(dim=-1) * ent).mean() * step_weights[t])
            except Exception:
                pass
            try:
                dyn = None
                if plan_prev is not None and torch.is_tensor(plan_prev) and plan_prev.shape == plan_step.shape:
                    dyn = (plan_det - plan_prev).abs().mean(dim=-1)  # (B,)
                else:
                    dyn = plan_det.new_zeros((B,))
                plan_dyn_stat_terms.append(dyn.mean() * step_weights[t])
                if float(lambda_plan_dyn_weight or 0.0) > 0.0:
                    plan_dyn_terms.append((lam_eff.mean(dim=-1) * dyn).mean() * step_weights[t])
                plan_prev = plan_det
            except Exception:
                plan_prev = plan_step.detach()

        R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
        omega = so3_log_map(R_res)  # (B,J,3)
        R_blend = torch.matmul(so3_exp_map(omega * lam_eff.unsqueeze(-1)), R_inc)

        w = step_weights[t]
        e_blend = _geodesic_R_safe(R_blend, R_gt)  # (B,J) rad
        e_inc = _geodesic_R_safe(R_inc, R_gt)  # (B,J) rad
        e_dir = _geodesic_R_safe(R_dir, R_gt)  # (B,J) rad
        loss_terms.append(e_blend.mean() * w)
        inc_terms.append(e_inc.mean() * w)
        e_dir_mean = e_dir.mean()
        e_dir_use = e_dir_mean
        if objective == "direct":
            # --- Stage7 direct objective: base mean (optional leg/nonleg split; no tail reweight) ---
            # Exclude root to align with GeoLocal evaluation protocol.
            root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
            if not (0 <= root_idx < J):
                root_idx = 0
            if J > 1 and 0 <= root_idx < J:
                nr_mask = torch.ones((J,), device=e_dir.device, dtype=torch.bool)
                nr_mask[root_idx] = False
                e = e_dir[:, nr_mask]  # (B,J-1)
            else:
                e = e_dir  # (B,J)

            # Optional: per-(sic,bone) binary hotspot boost.
            # Build a per-joint weight vector (default all-ones) for the current step.
            pair_joint_w = None  # (J_nr,) or None
            if direct_pose_pair_boost_enabled:
                try:
                    if int(cycle_len) > 0:
                        step_in_cycle = int((int(t) + int(offset)) % int(cycle_len))
                        cyc_idx = int((int(t) + int(offset)) // int(cycle_len))
                    else:
                        step_in_cycle = int(t)
                        cyc_idx = 0
                    in_mask = bool(cyc_idx >= int(direct_pose_pair_cycle_gte))
                    if in_mask and bool(direct_pose_pair_drop_wrap) and int(cycle_len) > 0:
                        in_mask = bool(step_in_cycle != (int(cycle_len) - 1))
                    if in_mask:
                        idxs = direct_pose_pair_joint_idx_by_sic.get(int(step_in_cycle), [])
                        if idxs:
                            pair_joint_w_full = torch.ones((J,), device=e_dir.device, dtype=e_dir.dtype)
                            pair_joint_w_full[idxs] = float(direct_pose_pair_boost)
                            if J > 1 and 0 <= root_idx < J and "nr_mask" in locals():
                                pair_joint_w = pair_joint_w_full[nr_mask]
                            else:
                                pair_joint_w = pair_joint_w_full
                            direct_pose_pair_focus_steps += 1
                            direct_pose_pair_focus_pairs += int(len(idxs))
                except Exception:
                    pair_joint_w = None

            # Optional: decouple legs vs non-legs so legs don't get diluted by already-good joints.
            use_leg_split = bool(direct_pose_loss_leg_split)
            if use_leg_split:
                leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
                if torch.is_tensor(leg_idx) and int(leg_idx.numel()) > 0:
                    try:
                        leg_mask = torch.zeros((J,), device=e_dir.device, dtype=torch.bool)
                        leg_mask[leg_idx.to(device=e_dir.device)] = True
                        if J > 1 and 0 <= root_idx < J:
                            leg_mask[root_idx] = False
                        if J > 1 and 'nr_mask' in locals() and torch.is_tensor(nr_mask) and nr_mask.shape == leg_mask.shape:
                            leg_mask = leg_mask[nr_mask]
                        if bool(leg_mask.any().detach().cpu().item()) and bool((~leg_mask).any().detach().cpu().item()):
                            e_leg = e[:, leg_mask]
                            e_nonleg = e[:, ~leg_mask]

                            w_leg = pair_joint_w[leg_mask] if torch.is_tensor(pair_joint_w) else None
                            w_nonleg = pair_joint_w[~leg_mask] if torch.is_tensor(pair_joint_w) else None

                            # Base terms (rad); equal-weight the two groups.
                            if torch.is_tensor(w_nonleg) and int(w_nonleg.numel()) == int(e_nonleg.shape[1]):
                                den_nonleg = w_nonleg.sum().clamp_min(1e-6)
                                L_nonleg_base = ((e_nonleg * w_nonleg.unsqueeze(0)).sum(dim=-1) / den_nonleg).mean()
                            else:
                                L_nonleg_base = e_nonleg.mean()
                            if torch.is_tensor(w_leg) and int(w_leg.numel()) == int(e_leg.shape[1]):
                                den_leg = w_leg.sum().clamp_min(1e-6)
                                L_leg_base = ((e_leg * w_leg.unsqueeze(0)).sum(dim=-1) / den_leg).mean()
                            else:
                                L_leg_base = e_leg.mean()
                            e_dir_use = L_nonleg_base + L_leg_base

                            # Logging/debug.
                            dir_base_terms.append(e_dir_use * w)
                            dir_leg_base_terms.append(L_leg_base * w)
                            dir_nonleg_base_terms.append(L_nonleg_base * w)
                        else:
                            use_leg_split = False
                    except Exception:
                        use_leg_split = False
                else:
                    use_leg_split = False

            if not use_leg_split:
                # Base: keep all joints learning (no tail focus / no reweight).
                if torch.is_tensor(pair_joint_w) and int(pair_joint_w.numel()) == int(e.shape[1]):
                    den = pair_joint_w.sum().clamp_min(1e-6)
                    L_base = ((e * pair_joint_w.unsqueeze(0)).sum(dim=-1) / den).mean()
                else:
                    L_base = e.mean()
                e_dir_use = L_base
                dir_base_terms.append(L_base * w)

        dir_terms.append(e_dir_use * w)

        # Stage2: supervise gate logits to match which expert is better (oracle soft label).
        # Use detached per-joint geodesic errors to keep supervision stable and avoid extra gradients
        # to frozen experts.
        if gate_sup_weight > 0.0 and int(t) >= int(gate_sup_start):
            lam_logits = ret.get("lambda_fusion_logits", None)
            if torch.is_tensor(lam_logits):
                if lam_logits.dim() == 3:
                    lam_logits = lam_logits[:, -1]
                if lam_logits.dim() == 2 and lam_logits.shape[0] == B:
                    try:
                        with torch.no_grad():
                            delta = (e_inc - e_dir).detach()
                            lam_star = torch.sigmoid(delta / float(tau_rad)).detach()
                            if margin_rad > 0.0:
                                mask = (delta.abs() >= float(margin_rad)).to(dtype=lam_star.dtype)
                            else:
                                mask = torch.ones_like(lam_star)

                        lam_star_used = lam_star
                        mask_used = mask
                        if lam_logits.shape[-1] == 1:
                            lam_star_used = lam_star.mean(dim=-1, keepdim=True)
                            mask_used = mask.mean(dim=-1, keepdim=True)
                        elif lam_logits.shape[-1] != J:
                            lam_star_used = None
                            mask_used = None

                        if lam_star_used is not None and mask_used is not None:
                            lam_star_used = lam_star_used.to(device=lam_logits.device, dtype=lam_logits.dtype)
                            mask_used = mask_used.to(device=lam_logits.device, dtype=lam_logits.dtype)
                            bce = F.binary_cross_entropy_with_logits(lam_logits, lam_star_used, reduction="none")
                            mask_sum = mask_used.sum()
                            gate_loss = (bce * mask_used).sum() / mask_sum.clamp_min(1e-6)
                            gate_sup_terms.append(gate_loss * w)
                            gate_sup_frac_terms.append(mask_used.mean() * w)
                            with torch.no_grad():
                                pred = (torch.sigmoid(lam_logits) > 0.5).to(dtype=mask_used.dtype)
                                tgt = (lam_star_used > 0.5).to(dtype=mask_used.dtype)
                                corr = (pred == tgt).to(dtype=mask_used.dtype)
                                gate_sup_acc_num_terms.append((corr * mask_used).sum() * w)
                                gate_sup_acc_den_terms.append(mask_sum * w)
                    except Exception:
                        pass
        if include_boundary and int(idx) == (int(cycle_len) - 1):
            try:
                boundary_blend_terms.append(_geodesic_R_safe(R_blend, R_gt).mean().detach())
                boundary_inc_terms.append(_geodesic_R_safe(R_inc, R_gt).mean().detach())
                boundary_dir_terms.append(_geodesic_R_safe(R_dir, R_gt).mean().detach())
                boundary_lam_terms.append(lam_raw.mean().detach())
                boundary_lam_eff_terms.append(lam_eff.mean().detach())
                if torch.is_tensor(lam_rel):
                    boundary_r_terms.append(lam_rel.mean().detach())
            except Exception:
                pass

        if float(lambda_early_weight or 0.0) > 0.0 and int(lambda_early_steps or 0) > 0 and int(t) < int(lambda_early_steps):
            early_terms.append(lam_eff.mean() * w)

        if float(lambda_entropy_weight or 0.0) > 0.0:
            ent_terms.append((-_lambda_entropy(lam_eff).mean()) * w)  # maximize entropy => minimize -H

        if float(lambda_smooth_weight or 0.0) > 0.0:
            if lam_prev is not None:
                smooth_terms.append(((lam_eff - lam_prev).pow(2).mean()) * w)
            lam_prev = lam_eff.detach()

        if float(lambda_monotonic_weight or 0.0) > 0.0:
            if lam_prev_monot is not None:
                # Encourage λ to be non-decreasing over rollout age: penalize decreases only.
                mono_terms.append(F.relu(lam_prev_monot - lam_eff).mean() * w)
            lam_prev_monot = lam_eff.detach()

        rot_next6d = matrix_to_rot6d(R_blend, columns=columns).view(B, rot_len)
        y_next_raw = y_prev_raw + delta_raw
        y_next_raw = y_next_raw.clone()
        y_next_raw[..., rot_slice] = rot_next6d

        if detach_rollout_state:
            y_next_raw = y_next_raw.detach()

        if t < total_steps - 1:
            cond_env = None
            if torch.is_tensor(cond_raw_step):
                cond_env = cond_raw_step
            motion_raw = trainer._apply_free_carry(motion_raw, y_next_raw, cond_next_raw=cond_env)
            motion_raw = _finite(motion_raw)
            motion = trainer._diag_norm_x(motion_raw)

            if pose_hist_enabled and pose_hist_buffer_raw is not None and pose_hist_stride > 0:
                with torch.no_grad():
                    pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
                    pose_hist_buffer_raw[..., -pose_hist_stride:] = y_next_raw[..., rot_slice]
                    pose_hist_buffer_norm = trainer._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)

            y_prev_raw = y_next_raw

    blend_loss_total = torch.stack(loss_terms).sum()
    inc_geo = torch.stack(inc_terms).sum() if inc_terms else blend_loss_total.new_tensor(0.0)
    dir_geo = torch.stack(dir_terms).sum() if dir_terms else blend_loss_total.new_tensor(0.0)
    dir_base = torch.stack(dir_base_terms).sum() if dir_base_terms else blend_loss_total.new_tensor(0.0)
    dir_leg_base = torch.stack(dir_leg_base_terms).sum() if dir_leg_base_terms else blend_loss_total.new_tensor(0.0)
    dir_nonleg_base = torch.stack(dir_nonleg_base_terms).sum() if dir_nonleg_base_terms else blend_loss_total.new_tensor(0.0)
    dir_tail = torch.stack(dir_tail_terms).sum() if dir_tail_terms else blend_loss_total.new_tensor(0.0)
    dir_tail_raw = torch.stack(dir_tail_raw_terms).sum() if dir_tail_raw_terms else blend_loss_total.new_tensor(0.0)
    dir_tail_alpha = torch.stack(dir_tail_alpha_terms).sum() if dir_tail_alpha_terms else blend_loss_total.new_tensor(0.0)
    dir_group_norm_leg = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_leg_ema = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_nonleg_ema = blend_loss_total.new_tensor(float("nan"))
    dir_group_norm_used = 0.0
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
                except Exception:
                    leg_ema_ok = False
            if not leg_ema_ok:
                ema_leg_prev = dir_leg_base.detach()
            else:
                ema_leg_prev = ema_leg_prev.to(device=dir_leg_base.device, dtype=dir_leg_base.dtype)
            non_ema_ok = bool(torch.is_tensor(ema_non_prev))
            if non_ema_ok:
                try:
                    non_ema_ok = bool(torch.isfinite(ema_non_prev).all().detach().cpu().item())
                except Exception:
                    non_ema_ok = False
            if not non_ema_ok:
                ema_non_prev = dir_nonleg_base.detach()
            else:
                ema_non_prev = ema_non_prev.to(device=dir_nonleg_base.device, dtype=dir_nonleg_base.dtype)

            leg_ratio = (dir_leg_base / ema_leg_prev.clamp_min(float(direct_group_eps))).clamp(
                float(direct_group_ratio_min), float(direct_group_ratio_max)
            )
            non_ratio = (dir_nonleg_base / ema_non_prev.clamp_min(float(direct_group_eps))).clamp(
                float(direct_group_ratio_min), float(direct_group_ratio_max)
            )
            dir_group_norm_leg = leg_ratio
            dir_group_norm_nonleg = non_ratio
            dir_group_norm_leg_ema = ema_leg_prev
            dir_group_norm_nonleg_ema = ema_non_prev
            dir_geo = float(direct_group_w_leg) * leg_ratio + float(direct_group_w_nonleg) * non_ratio
            dir_group_norm_used = 1.0

            with torch.no_grad():
                beta = float(direct_group_beta)
                ema_leg_new = beta * ema_leg_prev + (1.0 - beta) * dir_leg_base.detach()
                ema_non_new = beta * ema_non_prev + (1.0 - beta) * dir_nonleg_base.detach()
                ema_state["leg"] = ema_leg_new.detach()
                ema_state["nonleg"] = ema_non_new.detach()
                setattr(trainer, "_direct_pose_group_norm_ema", ema_state)
        except Exception:
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
        except Exception:
            gate_sup_acc = None

    hinge_sup_loss = blend_loss_total.new_tensor(0.0)
    hinge_sup_frac = blend_loss_total.new_tensor(0.0)
    if hinge_sup_terms:
        hinge_sup_loss = torch.stack(hinge_sup_terms).sum()
        total = total + float(direct_pose_hinge_sup_weight or 0.0) * hinge_sup_loss
    if hinge_sup_frac_terms:
        hinge_sup_frac = torch.stack(hinge_sup_frac_terms).sum()
    hinge_sup_abs_delta_tgt_deg = blend_loss_total.new_tensor(float("nan"))
    hinge_sup_abs_delta_pred_deg = blend_loss_total.new_tensor(float("nan"))
    hinge_sup_abs_delta_ratio = blend_loss_total.new_tensor(float("nan"))
    if hinge_sup_abs_delta_deg_den_terms:
        try:
            den = torch.stack(hinge_sup_abs_delta_deg_den_terms).sum()
            num_tgt = torch.stack(hinge_sup_abs_delta_tgt_deg_num_terms).sum()
            num_pred = torch.stack(hinge_sup_abs_delta_pred_deg_num_terms).sum()
            tgt_mean = num_tgt / den.clamp_min(1e-6)
            pred_mean = num_pred / den.clamp_min(1e-6)
            hinge_sup_abs_delta_tgt_deg = torch.where(den > 0.0, tgt_mean, den.new_tensor(float("nan")))
            hinge_sup_abs_delta_pred_deg = torch.where(den > 0.0, pred_mean, den.new_tensor(float("nan")))
            hinge_sup_abs_delta_ratio = torch.where(
                (den > 0.0) & (hinge_sup_abs_delta_tgt_deg > 0.0),
                hinge_sup_abs_delta_pred_deg / hinge_sup_abs_delta_tgt_deg.clamp_min(1e-6),
                den.new_tensor(float("nan")),
            )
        except Exception:
            pass

    hinge_gate_sup_loss = blend_loss_total.new_tensor(0.0)
    hinge_gate_sup_frac = blend_loss_total.new_tensor(0.0)
    if hinge_gate_sup_terms:
        hinge_gate_sup_loss = torch.stack(hinge_gate_sup_terms).sum()
        total = total + float(direct_pose_hinge_gate_sup_weight or 0.0) * hinge_gate_sup_loss
    if hinge_gate_sup_frac_terms:
        hinge_gate_sup_frac = torch.stack(hinge_gate_sup_frac_terms).sum()

    hinge_stance_loss = blend_loss_total.new_tensor(0.0)
    hinge_stance_frac = blend_loss_total.new_tensor(0.0)
    if hinge_stance_terms:
        hinge_stance_loss = torch.stack(hinge_stance_terms).sum()
        total = total + float(direct_pose_hinge_stance_weight or 0.0) * hinge_stance_loss
    if hinge_stance_frac_terms:
        hinge_stance_frac = torch.stack(hinge_stance_frac_terms).sum()

    hinge_reg_loss = blend_loss_total.new_tensor(0.0)
    if hinge_reg_terms:
        hinge_reg_loss = torch.stack(hinge_reg_terms).sum()
        total = total + float(direct_pose_hinge_reg_weight or 0.0) * hinge_reg_loss

    hinge_eps_l2_loss = blend_loss_total.new_tensor(0.0)
    if hinge_eps_l2_terms:
        hinge_eps_l2_loss = torch.stack(hinge_eps_l2_terms).sum()
        total = total + float(direct_pose_hinge_eps_l2_weight or 0.0) * hinge_eps_l2_loss

    leg_side_gate_reg_loss = blend_loss_total.new_tensor(0.0)
    if leg_side_gate_reg_terms:
        leg_side_gate_reg_loss = torch.stack(leg_side_gate_reg_terms).sum()
        total = total + float(direct_pose_leg_side_sign_gate_reg_weight or 0.0) * leg_side_gate_reg_loss

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

    leg_scale_sup_loss = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_tgt_mean = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_pred_mean = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_sign_loss = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_sign_tgt_mean = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_sign_pred_mean = blend_loss_total.new_tensor(0.0)
    leg_scale_sup_total_loss = blend_loss_total.new_tensor(0.0)
    if leg_scale_sup_terms:
        leg_scale_sup_loss = torch.stack(leg_scale_sup_terms).sum()
    if leg_scale_sup_tgt_mean_terms:
        leg_scale_sup_tgt_mean = torch.stack(leg_scale_sup_tgt_mean_terms).sum()
    if leg_scale_sup_pred_mean_terms:
        leg_scale_sup_pred_mean = torch.stack(leg_scale_sup_pred_mean_terms).sum()
    if leg_scale_sup_sign_terms:
        leg_scale_sup_sign_loss = torch.stack(leg_scale_sup_sign_terms).sum()
    if leg_scale_sup_sign_tgt_mean_terms:
        leg_scale_sup_sign_tgt_mean = torch.stack(leg_scale_sup_sign_tgt_mean_terms).sum()
    if leg_scale_sup_sign_pred_mean_terms:
        leg_scale_sup_sign_pred_mean = torch.stack(leg_scale_sup_sign_pred_mean_terms).sum()
    if leg_scale_sup_terms or leg_scale_sup_sign_terms:
        leg_scale_sup_total_loss = leg_scale_sup_loss + leg_scale_sup_sign_loss
        total = total + float(direct_pose_leg_scale_sup_weight or 0.0) * leg_scale_sup_total_loss

    leg_align_loss = blend_loss_total.new_tensor(0.0)
    leg_align_frac = blend_loss_total.new_tensor(0.0)
    if leg_align_terms:
        leg_align_loss = torch.stack(leg_align_terms).sum()
        total = total + float(direct_pose_leg_align_weight or 0.0) * leg_align_loss
    if leg_align_frac_terms:
        leg_align_frac = torch.stack(leg_align_frac_terms).sum()

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
    except Exception:
        lam_mean = lam_std = None
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_eff_vals], dim=0)
        lam_eff_mean = float(flat.mean().detach().cpu())
        lam_eff_std = float(flat.std(unbiased=False).detach().cpu())
    except Exception:
        lam_eff_mean = lam_eff_std = None
    try:
        if lam_rel_vals:
            flat = torch.cat([x.reshape(-1) for x in lam_rel_vals], dim=0)
            lam_rel_mean = float(flat.mean().detach().cpu())
    except Exception:
        lam_rel_mean = None

    stats = {
        "blend_loss": float(blend_loss_total.detach().cpu()),
        "gate_sup_loss": float(gate_sup_loss.detach().cpu()),
        "gate_sup_frac": float(gate_sup_frac.detach().cpu()),
        "gate_sup_acc@0.5": float(gate_sup_acc.detach().cpu()) if torch.is_tensor(gate_sup_acc) else float("nan"),
        "hinge_sup_loss": float(hinge_sup_loss.detach().cpu()),
        "hinge_sup_frac": float(hinge_sup_frac.detach().cpu()),
        "hinge_sup_weighted": float((float(direct_pose_hinge_sup_weight or 0.0) * hinge_sup_loss).detach().cpu()),
        "hinge_sup_abs_delta_tgt_deg": float(hinge_sup_abs_delta_tgt_deg.detach().cpu())
        if torch.is_tensor(hinge_sup_abs_delta_tgt_deg)
        else float("nan"),
        "hinge_sup_abs_delta_pred_deg": float(hinge_sup_abs_delta_pred_deg.detach().cpu())
        if torch.is_tensor(hinge_sup_abs_delta_pred_deg)
        else float("nan"),
        "hinge_sup_abs_delta_ratio": float(hinge_sup_abs_delta_ratio.detach().cpu()) if torch.is_tensor(hinge_sup_abs_delta_ratio) else float("nan"),
        "hinge_gate_sup_loss": float(hinge_gate_sup_loss.detach().cpu()),
        "hinge_gate_sup_frac": float(hinge_gate_sup_frac.detach().cpu()),
        "hinge_gate_sup_weighted": float((float(direct_pose_hinge_gate_sup_weight or 0.0) * hinge_gate_sup_loss).detach().cpu()),
        "hinge_stance_loss": float(hinge_stance_loss.detach().cpu()),
        "hinge_stance_frac": float(hinge_stance_frac.detach().cpu()),
        "hinge_stance_weighted": float((float(direct_pose_hinge_stance_weight or 0.0) * hinge_stance_loss).detach().cpu()),
        "hinge_reg_loss": float(hinge_reg_loss.detach().cpu()),
        "hinge_reg_weighted": float((float(direct_pose_hinge_reg_weight or 0.0) * hinge_reg_loss).detach().cpu()),
        "hinge_eps_l2_loss": float(hinge_eps_l2_loss.detach().cpu()),
        "hinge_eps_l2_weighted": float((float(direct_pose_hinge_eps_l2_weight or 0.0) * hinge_eps_l2_loss).detach().cpu()),
        "leg_side_sign_gate_reg_loss": float(leg_side_gate_reg_loss.detach().cpu()),
        "leg_side_sign_gate_reg_weighted": float(
            (float(direct_pose_leg_side_sign_gate_reg_weight or 0.0) * leg_side_gate_reg_loss).detach().cpu()
        ),
        "leg_gate_sup_loss": float(leg_gate_sup_loss.detach().cpu()),
        "leg_gate_sup_tgt_frac": float(leg_gate_sup_tgt_frac.detach().cpu()),
        "leg_gate_sup_pred_mean": float(leg_gate_sup_pred_mean.detach().cpu()),
        "leg_gate_sup_weighted": float((float(direct_pose_leg_gate_sup_weight or 0.0) * leg_gate_sup_loss).detach().cpu()),
        "leg_scale_sup_loss": float(leg_scale_sup_loss.detach().cpu()),
        "leg_scale_sup_tgt_mean_log": float(leg_scale_sup_tgt_mean.detach().cpu()),
        "leg_scale_sup_pred_mean_log": float(leg_scale_sup_pred_mean.detach().cpu()),
        "leg_scale_sup_sign_loss": float(leg_scale_sup_sign_loss.detach().cpu()),
        "leg_scale_sup_sign_tgt_mean": float(leg_scale_sup_sign_tgt_mean.detach().cpu()),
        "leg_scale_sup_sign_pred_mean": float(leg_scale_sup_sign_pred_mean.detach().cpu()),
        "leg_scale_sup_total_loss": float(leg_scale_sup_total_loss.detach().cpu()),
        "leg_scale_sup_weighted": float(
            (float(direct_pose_leg_scale_sup_weight or 0.0) * leg_scale_sup_total_loss).detach().cpu()
        ),
        "leg_align_loss": float(leg_align_loss.detach().cpu()),
        "leg_align_frac": float(leg_align_frac.detach().cpu()),
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
        "dir_tail": float(dir_tail.detach().cpu()),
        "dir_tail_raw": float(dir_tail_raw.detach().cpu()),
        "dir_tail_alpha": float(dir_tail_alpha.detach().cpu()),
        "dir_group_norm_used": float(dir_group_norm_used),
        "dir_group_norm_leg": float(dir_group_norm_leg.detach().cpu()) if torch.is_tensor(dir_group_norm_leg) else float("nan"),
        "dir_group_norm_nonleg": float(dir_group_norm_nonleg.detach().cpu()) if torch.is_tensor(dir_group_norm_nonleg) else float("nan"),
        "dir_group_norm_leg_ema": float(dir_group_norm_leg_ema.detach().cpu())
        if torch.is_tensor(dir_group_norm_leg_ema)
        else float("nan"),
        "dir_group_norm_nonleg_ema": float(dir_group_norm_nonleg_ema.detach().cpu())
        if torch.is_tensor(dir_group_norm_nonleg_ema)
        else float("nan"),
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
    if objective == "direct" and sic_spec:
        stats["direct_pose_sic_focus_steps"] = float(direct_pose_sic_focus_steps)
        stats["direct_pose_sic_focus_weight_sum_before"] = float(direct_pose_sic_focus_weight_sum_before)
        stats["direct_pose_sic_focus_weight_sum_after"] = float(direct_pose_sic_focus_weight_sum_after)
        stats["direct_pose_sic_focus_cycle_gte"] = float(direct_pose_loss_cycle_gte or 0)
        stats["direct_pose_sic_focus_mode"] = str(direct_pose_sic_focus_mode)
        stats["direct_pose_sic_focus_boost"] = float(direct_pose_sic_focus_boost)
    if objective == "direct" and direct_pose_pair_boost_enabled:
        stats["direct_pose_pair_focus_steps"] = float(direct_pose_pair_focus_steps)
        stats["direct_pose_pair_focus_pairs"] = float(direct_pose_pair_focus_pairs)
        stats["direct_pose_pair_focus_sics"] = float(len(direct_pose_pair_joint_idx_by_sic))
        stats["direct_pose_pair_focus_cycle_gte"] = float(direct_pose_pair_cycle_gte)
        stats["direct_pose_pair_focus_drop_wrap"] = 1.0 if bool(direct_pose_pair_drop_wrap) else 0.0
        stats["direct_pose_pair_focus_boost"] = float(direct_pose_pair_boost)
    if include_boundary:
        stats["rollout_include_boundary"] = 1.0
        stats["rollout_random_offset"] = 1.0 if bool(random_offset) else 0.0
        stats["rollout_offset"] = float(offset)
        stats["lambda_boundary_weight"] = float(boundary_weight or 0.0)
        stats["boundary_steps"] = float(boundary_steps or 0)
        stats["boundary_weighted_sum"] = float(boundary_weighted_sum or 0.0)
        if boundary_blend_terms:
            try:
                stats["boundary_blend_geo"] = float(torch.stack(boundary_blend_terms).mean().detach().cpu())
            except Exception:
                pass
        if boundary_inc_terms:
            try:
                stats["boundary_inc_geo"] = float(torch.stack(boundary_inc_terms).mean().detach().cpu())
            except Exception:
                pass
        if boundary_dir_terms:
            try:
                stats["boundary_dir_geo"] = float(torch.stack(boundary_dir_terms).mean().detach().cpu())
            except Exception:
                pass
        if boundary_lam_terms:
            try:
                stats["boundary_lambda_mean"] = float(torch.stack(boundary_lam_terms).mean().detach().cpu())
            except Exception:
                pass
        if boundary_lam_eff_terms:
            try:
                stats["boundary_lambda_eff_mean"] = float(torch.stack(boundary_lam_eff_terms).mean().detach().cpu())
            except Exception:
                pass
        if boundary_r_terms:
            try:
                stats["boundary_r_mean"] = float(torch.stack(boundary_r_terms).mean().detach().cpu())
            except Exception:
                pass
    if plan_ent_stat_terms:
        try:
            stats["plan_entropy_mean"] = float(torch.stack(plan_ent_stat_terms).sum().detach().cpu())
        except Exception:
            pass
    if plan_dyn_stat_terms:
        try:
            stats["plan_dyn_mean"] = float(torch.stack(plan_dyn_stat_terms).sum().detach().cpu())
        except Exception:
            pass
    if contact_meas_loss is not None:
        if bool(meas_used_logits):
            stats["contact_meas_bce"] = float(contact_meas_loss.detach().cpu())
        else:
            stats["contact_meas_mse"] = float(contact_meas_loss.detach().cpu())
        stats["contact_meas_weighted"] = float((float(contact_meas_weight or 0.0) * contact_meas_loss).detach().cpu())
    return total, stats


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


def _freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_so3_corrector(model: EventMotionModel) -> None:
    head = getattr(model, "so3_delta_corrector", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad_(True)
    logit = getattr(model, "so3_corr_gate_logit", None)
    if torch.is_tensor(logit):
        logit.requires_grad_(True)


def _unfreeze_lambda_fusion(model: EventMotionModel) -> None:
    head = getattr(model, "lambda_fusion_head", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad_(True)


def _unfreeze_direct_pose(
    model: EventMotionModel,
    *,
    hinge_only: bool = False,
    gate_only: bool = False,
    leg_only: bool = False,
    leg_gate_only: bool = False,
    nonleg_only: bool = False,
) -> None:
    # By default, train both the base direct head and hinge head (if any).
    # When hinge_only=True, keep the base direct head frozen and train hinge as a residual corrector.
    # When gate_only=True, train only the learned gate head (keep base direct + hinge frozen).
    if bool(gate_only):
        for gate in (
            getattr(model, "direct_pose_hinge_gate_head", None),
            getattr(model, "direct_pose_hinge_gate_head_clean", None),
        ):
            if gate is not None:
                for p in gate.parameters():
                    p.requires_grad_(True)
        return
    if bool(leg_gate_only):
        # Train only the leg gate/scale head (keep base direct + leg omega + hinge frozen).
        if bool(getattr(model, "direct_pose_leg_side_routing", False)) and getattr(model, "direct_pose_leg_gate_head_shared", None) is not None:
            gate_leg = getattr(model, "direct_pose_leg_gate_head_shared", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        else:
            gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        return
    if bool(leg_only):
        # Prefer the routed shared head when enabled (legacy head is unused in forward in that mode).
        if bool(getattr(model, "direct_pose_leg_side_routing", False)) and getattr(model, "direct_pose_leg_head_shared", None) is not None:
            leg = getattr(model, "direct_pose_leg_head_shared", None)
            if leg is not None:
                for p in leg.parameters():
                    p.requires_grad_(True)
            gate_leg = getattr(model, "direct_pose_leg_gate_head_shared", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
            gate = getattr(model, "direct_pose_leg_side_sign_gate_head", None)
            if gate is not None:
                for p in gate.parameters():
                    p.requires_grad_(True)
            emb = getattr(model, "direct_pose_leg_side_embed", None)
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(True)
        else:
            leg = getattr(model, "direct_pose_leg_head", None)
            if leg is not None:
                for p in leg.parameters():
                    p.requires_grad_(True)
            gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        return
    if bool(nonleg_only):
        nonleg_proj = getattr(model, "direct_pose_nonleg_proj", None)
        if nonleg_proj is not None:
            for p in nonleg_proj.parameters():
                p.requires_grad_(True)
        out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
        if out_nonleg is not None:
            for p in out_nonleg.parameters():
                p.requires_grad_(True)
        return

    if not hinge_only:
        head = getattr(model, "direct_pose_head", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(True)
        out_leg = getattr(model, "direct_pose_out_leg", None)
        if out_leg is not None:
            for p in out_leg.parameters():
                p.requires_grad_(True)
        out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
        if out_nonleg is not None:
            for p in out_nonleg.parameters():
                p.requires_grad_(True)
        leg = getattr(model, "direct_pose_leg_head", None)
        if leg is not None:
            for p in leg.parameters():
                p.requires_grad_(True)
        leg_shared = getattr(model, "direct_pose_leg_head_shared", None)
        if leg_shared is not None:
            for p in leg_shared.parameters():
                p.requires_grad_(True)
        gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
        if gate_leg is not None:
            for p in gate_leg.parameters():
                p.requires_grad_(True)
        gate_leg_shared = getattr(model, "direct_pose_leg_gate_head_shared", None)
        if gate_leg_shared is not None:
            for p in gate_leg_shared.parameters():
                p.requires_grad_(True)
        gate = getattr(model, "direct_pose_leg_side_sign_gate_head", None)
        if gate is not None:
            for p in gate.parameters():
                p.requires_grad_(True)
        emb = getattr(model, "direct_pose_leg_side_embed", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad_(True)
    for hinge in (
        getattr(model, "direct_pose_hinge_head", None),
        getattr(model, "direct_pose_hinge_nonhidden_head", None),
        getattr(model, "direct_pose_hinge_eps_head", None),
    ):
        if hinge is not None:
            for p in hinge.parameters():
                p.requires_grad_(True)
    for gate in (
        getattr(model, "direct_pose_hinge_gate_head", None),
        getattr(model, "direct_pose_hinge_gate_head_clean", None),
    ):
        if gate is not None:
            for p in gate.parameters():
                p.requires_grad_(True)


def _unfreeze_contact_meas(model: EventMotionModel) -> None:
    head = getattr(model, "contact_meas_head", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad_(True)


def _unfreeze_contact_td_hazard(model: EventMotionModel) -> None:
    head = getattr(model, "contact_td_hazard_head", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad_(True)


def _warm_start_direct_pose_leg_head_shared_from_legacy(model: EventMotionModel) -> bool:
    """
    Warm-start the per-side shared leg head (direct_pose_leg_head_shared) from the legacy
    all-joints leg head (direct_pose_leg_head) weights.

    Assumptions (current Stage7 setup):
    - direct_pose_phase_z_mode == 'replace_contacts' so the legacy leg head input ends with phase_z_in
      of size 2*contact_dim (=4 when contact_dim=2).
    - shared head input is: [base(=direct_feat+time_pe), plan_side, meas_side, phase_side(2D)] (+optional side_emb).
      We initialize plan/meas columns to 0 and map phase columns from the matching channel.

    Returns:
        True if warm-start was applied, False otherwise.
    """
    if not bool(getattr(model, "direct_pose_leg_side_routing", False)):
        return False
    legacy = getattr(model, "direct_pose_leg_head", None)
    shared = getattr(model, "direct_pose_leg_head_shared", None)
    if legacy is None or shared is None:
        return False
    if int(getattr(model, "contact_dim", 0) or 0) != 2:
        return False
    if int(getattr(model, "direct_pose_leg_side_embed_dim", 0) or 0) > 0:
        # When side embedding is enabled, input dims differ; keep safe init and learn from scratch.
        return False
    try:
        legacy_linears = [m for m in legacy if isinstance(m, torch.nn.Linear)]
        shared_linears = [m for m in shared if isinstance(m, torch.nn.Linear)]
    except Exception:
        return False
    if len(legacy_linears) != 3 or len(shared_linears) != 3:
        return False
    l0, l1, l2 = legacy_linears
    s0, s1, s2 = shared_linears

    # Validate shapes.
    if l0.in_features < 4 or s0.in_features < 4:
        return False
    if l0.out_features != s0.out_features or l1.in_features != s1.in_features or l1.out_features != s1.out_features:
        return False
    if l2.in_features != s2.in_features:
        return False
    # Expect same hidden size; output dims differ (legacy is K*3, shared is K_side*3).
    hid = int(s0.out_features)
    if hid <= 0:
        return False

    # Determine phase column locations.
    # Legacy (replace_contacts): [... base ..., sin(c0),cos(c0), sin(c1),cos(c1)]  => last 4 dims.
    legacy_in = int(l0.in_features)
    shared_in = int(s0.in_features)
    if legacy_in != shared_in:
        # Only support the common Stage7 case where dims match (27).
        return False
    phase_off_legacy = legacy_in - 4
    # Shared input layout (no side-emb):
    #   [... base ..., plan_side, meas_side, sin_side, cos_side]
    phase_off_shared = shared_in - 2
    plan_off_shared = phase_off_shared - 2
    if plan_off_shared < 0 or phase_off_legacy < 0:
        return False

    # Side channel mapping (contact order).
    ch_r = int(getattr(model, "direct_pose_leg_contact_ch_r", 1) or 0)
    ch_l = int(getattr(model, "direct_pose_leg_contact_ch_l", 0) or 0)
    ch_r = max(0, min(1, ch_r))
    ch_l = max(0, min(1, ch_l))

    # Build side-specific first-layer weights (map phase columns, zero plan/meas).
    W0 = l0.weight.detach().clone()
    b0 = l0.bias.detach().clone() if l0.bias is not None else None
    W0_r = torch.zeros_like(W0)
    W0_l = torch.zeros_like(W0)
    # Copy base features as-is.
    W0_r[:, :phase_off_legacy] = W0[:, :phase_off_legacy]
    W0_l[:, :phase_off_legacy] = W0[:, :phase_off_legacy]
    # plan/meas cols are left at 0 (learnable after finetune).
    # Map phase(c_r) -> shared phase dims.
    idx_sin_r = phase_off_legacy + 2 * ch_r + 0
    idx_cos_r = phase_off_legacy + 2 * ch_r + 1
    idx_sin_l = phase_off_legacy + 2 * ch_l + 0
    idx_cos_l = phase_off_legacy + 2 * ch_l + 1
    W0_r[:, phase_off_shared + 0] = W0[:, idx_sin_r]
    W0_r[:, phase_off_shared + 1] = W0[:, idx_cos_r]
    W0_l[:, phase_off_shared + 0] = W0[:, idx_sin_l]
    W0_l[:, phase_off_shared + 1] = W0[:, idx_cos_l]
    W0_shared = 0.5 * (W0_r + W0_l)

    # Second layer can be copied directly (shared trunk).
    W1_shared = l1.weight.detach().clone()
    b1_shared = l1.bias.detach().clone() if l1.bias is not None else None

    # Third layer: select per-side joint rows, then average to enforce symmetry.
    # legacy out layout: (K,3) in K-order (direct_pose_leg_joint_idx order).
    # shared out layout: (K_side,3) in side-order (pos_r / aligned pos_l order).
    try:
        pos_r = getattr(model, "direct_pose_leg_side_pos_r", None) or []
        pos_l = getattr(model, "direct_pose_leg_side_pos_l", None) or []
        names = [str(n).lower() for n in (getattr(model, "direct_pose_leg_joint_names", None) or [])]
    except Exception:
        pos_r, pos_l, names = [], [], []
    K = int(len(pos_r) + len(pos_l))
    K_side = int(len(pos_r))
    if K_side <= 0 or len(pos_l) != K_side:
        return False
    if int(l2.out_features) != 3 * int(K):
        return False
    # Shared head output:
    # - default: (K_side * 3) omega vectors
    # - rank1  : (3 + K_side) => v(3) + s(K_side)
    shared_out = int(s2.out_features)
    if shared_out not in (3 * int(K_side), 3 + int(K_side)):
        return False
    want_rank1 = shared_out == (3 + int(K_side))

    # Optional: align L joints to R order by name (thigh_r -> thigh_l, etc).
    pos_l_by_name = {}
    try:
        for p in pos_l:
            if 0 <= int(p) < len(names):
                pos_l_by_name[names[int(p)]] = int(p)
    except Exception:
        pos_l_by_name = {}
    aligned_pos_l = []
    if names and pos_l_by_name:
        for p in pos_r:
            pn = names[int(p)] if 0 <= int(p) < len(names) else ""
            cand = pn.replace("_r", "_l").replace("right", "left")
            if cand in pos_l_by_name:
                aligned_pos_l.append(pos_l_by_name[cand])
        if len(aligned_pos_l) != K_side:
            aligned_pos_l = list(pos_l)
    else:
        aligned_pos_l = list(pos_l)

    W2_shared = None
    b2_shared = None
    if not want_rank1:
        W2 = l2.weight.detach().clone()
        b2 = l2.bias.detach().clone() if l2.bias is not None else None
        W2_r = torch.zeros((3 * K_side, hid), device=W2.device, dtype=W2.dtype)
        W2_l = torch.zeros((3 * K_side, hid), device=W2.device, dtype=W2.dtype)
        b2_r = torch.zeros((3 * K_side,), device=W2.device, dtype=W2.dtype) if b2 is not None else None
        b2_l = torch.zeros((3 * K_side,), device=W2.device, dtype=W2.dtype) if b2 is not None else None
        for i, p in enumerate(pos_r):
            p = int(p)
            if p < 0:
                continue
            src = slice(3 * p, 3 * p + 3)
            dst = slice(3 * i, 3 * i + 3)
            W2_r[dst, :] = W2[src, :]
            if b2 is not None and b2_r is not None:
                b2_r[dst] = b2[src]
        for i, p in enumerate(aligned_pos_l):
            p = int(p)
            if p < 0:
                continue
            src = slice(3 * p, 3 * p + 3)
            dst = slice(3 * i, 3 * i + 3)
            W2_l[dst, :] = W2[src, :]
            if b2 is not None and b2_l is not None:
                b2_l[dst] = b2[src]
        W2_shared = 0.5 * (W2_r + W2_l)
        b2_shared = 0.5 * (b2_r + b2_l) if (b2 is not None and b2_r is not None and b2_l is not None) else None

    # Apply.
    with torch.no_grad():
        s0.weight.copy_(W0_shared)
        if b0 is not None and s0.bias is not None:
            s0.bias.copy_(b0)
        s1.weight.copy_(W1_shared)
        if b1_shared is not None and s1.bias is not None:
            s1.bias.copy_(b1_shared)
        if W2_shared is not None:
            s2.weight.copy_(W2_shared)
        if b2_shared is not None and s2.bias is not None:
            s2.bias.copy_(b2_shared)
        if want_rank1:
            # Keep the last layer at safe init (zeros) so omega starts near 0 and learns under the new constraint.
            s2.weight.zero_()
            if s2.bias is not None:
                s2.bias.zero_()
    return True


def _unfreeze_contact_plan_init(model: EventMotionModel) -> None:
    p = getattr(model, "contact_plan_init_z", None)
    if torch.is_tensor(p):
        p.requires_grad_(True)
    head = getattr(model, "contact_plan_init_head", None)
    if head is not None:
        for pp in head.parameters():
            pp.requires_grad_(True)


def _unfreeze_contact_plan(model: EventMotionModel) -> None:
    cell = getattr(model, "contact_plan_cell", None)
    head = getattr(model, "contact_plan_head", None)
    time_head = getattr(model, "contact_plan_time_head", None)
    init_z = getattr(model, "contact_plan_init_z", None)
    init_head = getattr(model, "contact_plan_init_head", None)
    if cell is not None:
        for pp in cell.parameters():
            pp.requires_grad_(True)
    if head is not None:
        for pp in head.parameters():
            pp.requires_grad_(True)
    if time_head is not None:
        for pp in time_head.parameters():
            pp.requires_grad_(True)
    if torch.is_tensor(init_z):
        init_z.requires_grad_(True)
    if init_head is not None:
        for pp in init_head.parameters():
            pp.requires_grad_(True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-train entry (SO(3) corrector / Stage2 lambda fusion).")
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
            "'td_hazard'=integrate-to-1 reset from contact_td_hazard_head; "
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
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)

    ap.add_argument("--so3_corr_gate_force", default=None, help="Force gate scalar (float) or 'null' to learn.")
    ap.add_argument("--so3_corr_gate_logit_reset", default=None, help="Reset model.so3_corr_gate_logit to a float (e.g. -2.2).")
    ap.add_argument("--gate_warmup_steps", type=int, default=None, help="First N steps force gate_warmup_value.")
    ap.add_argument("--gate_warmup_value", type=float, default=None, help="Warmup gate force value (e.g. 0.1).")
    ap.add_argument("--so3_corr_max_deg", type=float, default=None)
    ap.add_argument("--so3_corr_omega_l2_weight", type=float, default=None)
    ap.add_argument("--corr_time_weight_max", type=float, default=None)
    ap.add_argument("--detach_rollout_state", type=str, default=None, help="true|false")
    ap.add_argument("--train_so3_corrector", type=str, default=None, help="true|false; whether to finetune so3_delta_corrector")
    ap.add_argument("--train_contact_plan_init", type=str, default=None, help="true|false; whether to finetune contact plan init params (init_z / init_head)")
    ap.add_argument("--train_contact_plan", type=str, default=None, help="true|false; whether to finetune contact_plan dynamics (GRU + heads) via teacher supervision")
    ap.add_argument("--train_direct_pose", type=str, default=None, help="true|false; whether to finetune direct_pose_head (direct expert) via rollout loss")
    ap.add_argument("--contact_plan_init_weight", type=float, default=None, help="Weight for contacts_plan MSE vs GT soft contacts when training init_z.")
    ap.add_argument("--contact_plan_init_mode", type=str, default=None, help="zeros|learnable|obs|learnable+obs (how to init plan_z when plan_z is None)")
    ap.add_argument("--contact_plan_init_hidden", type=int, default=None, help="Hidden dim for contact_plan_init_head (obs init MLP)")
    ap.add_argument("--contact_plan_init_dropout", type=float, default=None, help="Dropout for contact_plan_init_head (obs init MLP)")
    ap.add_argument("--contact_plan_weight", type=float, default=None, help="Weight for contacts_plan MSE vs GT soft contacts when training full contact plan.")
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
    ap.add_argument("--train_contact_meas", type=str, default=None, help="true|false; also finetune contact_meas_head")
    ap.add_argument("--train_contact_td_hazard", type=str, default=None, help="true|false; also finetune contact_td_hazard_head (touchdown hazard/intensity)")
    ap.add_argument(
        "--contact_meas_rollout",
        type=str,
        default=None,
        help="true|false; when train_contact_meas_only, supervise contact_meas under rollout instead of teacher forcing.",
    )
    ap.add_argument("--contact_meas_weight", type=float, default=None, help="Weight for contact_meas MSE vs GT soft contacts.")
    ap.add_argument(
        "--contact_meas_rollout_weight",
        type=float,
        default=None,
        help="If >0 (and train_contact_meas_only), add an extra rollout meas loss term on top of teacher loss (mixed supervision).",
    )
    ap.add_argument(
        "--contact_meas_smooth_weight",
        type=float,
        default=None,
        help="Stability reg: weight for temporal smoothness on contacts_meas logits (penalize high-frequency Δlogit).",
    )
    ap.add_argument(
        "--contact_meas_smooth_kind",
        type=str,
        default=None,
        choices=("l1", "l2", "smooth_l1"),
        help="Stability reg: smoothness penalty type for contacts_meas logits.",
    )
    ap.add_argument(
        "--contact_meas_margin_weight",
        type=float,
        default=None,
        help="Stability reg: weight for confidence margin on contacts_meas logits (push away from p=0.5 / logit=0).",
    )
    ap.add_argument(
        "--contact_meas_margin_logit",
        type=float,
        default=None,
        help="Stability reg: logit margin m (penalize relu(m-|logit|)); m=0 disables.",
    )
    ap.add_argument(
        "--contact_meas_transition_band",
        type=float,
        default=None,
        help="Mask out GT transition region for stability regs: treat GT contacts in [0.5-band, 0.5+band] as transitions.",
    )
    ap.add_argument(
        "--contact_td_hazard_rollout",
        type=str,
        default=None,
        help="true|false; when train_contact_td_hazard_only, supervise contact_td_hazard under rollout instead of teacher forcing.",
    )
    ap.add_argument(
        "--contact_td_hazard_rollout_weight",
        type=float,
        default=None,
        help="If >0 (and train_contact_td_hazard_only), add an extra rollout TD hazard loss term on top of teacher loss (mixed supervision).",
    )
    ap.add_argument("--contact_td_hazard_bce_weight", type=float, default=None, help="Weight for TD hazard BCE (logits vs ttc_td_events).")
    ap.add_argument(
        "--contact_td_hazard_event_weight",
        type=float,
        default=None,
        help="Extra weight factor on event frames (ttc_td_events==1) for TD hazard BCE.",
    )
    ap.add_argument("--contact_td_hazard_mass_weight", type=float, default=None, help="Weight for TD hazard mass regularizer (sum(sigmoid(logit)) vs sum(events)).")
    ap.add_argument("--contact_td_hazard_unimodal_weight", type=float, default=None, help="Weight for TD hazard unimodality prior (log-softmax concavity penalty).")
    ap.add_argument(
        "--contact_td_hazard_entropy_weight",
        type=float,
        default=None,
        help="Weight for TD hazard entropy penalty (softmax over time; encourages a sharp single peak).",
    )
    ap.add_argument(
        "--contact_td_hazard_clock_weight",
        type=float,
        default=None,
        help="Weight for TD hazard clock-alignment loss (encourage integrate-to-1 event at GT touchdown step).",
    )
    ap.add_argument("--contact_td_hazard_hidden", type=int, default=None, help="Hidden dim for contact_td_hazard_head (only used when initializing head without weights).")
    ap.add_argument("--contact_td_hazard_dropout", type=float, default=None, help="Dropout for contact_td_hazard_head (only used when initializing head without weights).")
    ap.add_argument(
        "--direct_pose_meas_force_zero",
        type=str,
        default=None,
        help="true|false; ablation: force direct head to ignore contacts_meas (concat->zeros, mode_select->uniform).",
    )
    ap.add_argument(
        "--direct_pose_meas_detach",
        type=str,
        default=None,
        help="true|false; ablation: stop-grad from direct head into contacts_meas (prevents co-adaptation).",
    )
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
        help="Leg residual mode: rot6d_add (legacy) | so3 (on-manifold compose exp(omega)@R).",
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
        choices=("auto", "none", "learned", "scale", "signed_scale"),
        help=(
            "Optional learned gate/scale for leg omega (SO(3) only): "
            "auto (enable iff ckpt has weights or gate/scale sup enabled) | none | learned | scale | signed_scale."
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
        help="Optional direction alignment loss weight for leg omega: relu(-cos_pred_oracle). 0 disables.",
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
        "--direct_pose_leg_side_routing",
        type=str,
        default=None,
        help="true|false; use explicit per-side routing + shared omega head for leg residuals (SO(3) only).",
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
        help="true|false; predict a per-side sign gate g∈[-1,1] (tanh) and apply omega*=g*omega_raw (couples sign across same-side joints).",
    )
    ap.add_argument(
        "--direct_pose_leg_side_sign_gate_reg_weight",
        type=float,
        default=None,
        help="Optional regularizer weight to encourage |g|->1 (avoid collapsing to 0). 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_leg_side_rank1",
        type=str,
        default=None,
        help="true|false; enforce rank-1 same-side coupling: omega_j = softplus(s_j)*normalize(v_side). "
        "Incompatible with direct_pose_leg_side_sign_gate.",
    )
    ap.add_argument(
        "--direct_pose_leg_side_rank1_bones",
        type=str,
        default=None,
        help="Optional: comma-separated *base* bone names (without _l/_r) to include in rank-1 coupling "
        "(e.g. 'calf,foot,ball'). Empty/None couples all side joints.",
    )
    ap.add_argument(
        "--direct_pose_hinge_enable",
        type=str,
        default=None,
        help="true|false; enable hinge-style 1D correction for direct head (joint-local axis twist).",
    )
    ap.add_argument(
        "--direct_pose_hinge_train_only",
        type=str,
        default=None,
        help="true|false; when train_direct_pose, freeze direct_pose_head and train hinge head only (residual correction).",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_train_only",
        type=str,
        default=None,
        help="true|false; when train_direct_pose, freeze direct_pose_head+hinge head and train learned gate only.",
    )
    ap.add_argument(
        "--direct_pose_hinge_bones",
        type=str,
        default=None,
        help="Comma-separated bone names/indices for hinge correction (default: calf_r).",
    )
    ap.add_argument(
        "--direct_pose_hinge_axis",
        type=str,
        default=None,
        choices=("x", "y", "z"),
        help="Local axis for hinge correction (default: z).",
    )
    ap.add_argument(
        "--direct_pose_hinge_max_deg",
        type=float,
        default=None,
        help="Max hinge correction magnitude in degrees (tanh-scaled).",
    )
    ap.add_argument(
        "--direct_pose_hinge_hidden",
        type=int,
        default=None,
        help="Hidden dim for hinge head (0=auto).",
    )
    ap.add_argument(
        "--direct_pose_hinge_feat_source",
        type=str,
        default=None,
        choices=("auto", "cond", "hidden", "cond+hidden", "none"),
        help="Feature source for hinge head input (default: auto -> follow direct_pose_feat_source).",
    )
    ap.add_argument(
        "--direct_pose_hinge_clean",
        type=str,
        default=None,
        help="true|false; use clean split hinge: delta = base(nonhidden) + eps(hidden).",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_max_deg",
        type=float,
        default=None,
        help="Max |eps(hidden)| in degrees. 0/None uses eps_max_scale * hinge_max_deg.",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_max_scale",
        type=float,
        default=None,
        help="If eps_max_deg<=0, eps_max = eps_max_scale * hinge_max_deg (default 0.5).",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_hidden",
        type=int,
        default=None,
        help="Hidden dim for eps head (0/None=auto).",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_dropout",
        type=float,
        default=None,
        help="Dropout for eps head (default 0).",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_lr_scale",
        type=float,
        default=None,
        help="LR scale for eps head params (e.g. 0.1).",
    )
    ap.add_argument(
        "--direct_pose_hinge_eps_l2_weight",
        type=float,
        default=None,
        help="L2 penalty weight on eps output: mean(eps^2) in rad^2.",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_mode",
        type=str,
        default=None,
        choices=("none", "contact", "learned"),
        help="Optional gating for hinge delta: none|contact (delta *= (1-contact)^power) | learned (delta *= sigmoid(g_logit)).",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_source",
        type=str,
        default=None,
        choices=("plan", "meas", "plan_or_meas"),
        help="Contact source for hinge gating: plan|meas|plan_or_meas (default: plan).",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_power",
        type=float,
        default=None,
        help="Power for hinge contact gating (gate=(1-contact)^power).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_weight",
        type=float,
        default=None,
        help="If >0, add supervised hinge delta regression (delta_target) loss term.",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_kind",
        type=str,
        default=None,
        choices=("smooth_l1", "l2", "apply_geo"),
        help="Loss type for supervised hinge: smooth_l1|l2 (angle regression) or apply_geo (closed-loop geodesic residual).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_contact_source",
        type=str,
        default=None,
        choices=("gt", "plan", "meas"),
        help="Contact source for masking supervised hinge loss (gt|plan|meas).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_contact_value",
        type=int,
        default=None,
        choices=(0, 1),
        help="If set (0/1), only supervise hinge on frames where contact matches this value (thr-based).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_contact_thresh",
        type=float,
        default=None,
        help="Threshold for contact_value masking (default 0.5).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_angle_thresh_deg",
        type=float,
        default=None,
        help="Only supervise hinge when base direct angular error >= this threshold (deg).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_delta_thresh_deg",
        type=float,
        default=None,
        help="Only supervise hinge when |delta_target| >= this threshold (deg).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_delta_weight_power",
        type=float,
        default=None,
        help="Weight hinge supervision by normalized |delta_target|^p (p>0). 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_delta_weight_scale_deg",
        type=float,
        default=None,
        help="If >0, delta weighting uses (|delta_target| / scale)^p (scale in degrees). 0 uses max_rad normalization (<=1).",
    )
    ap.add_argument(
        "--direct_pose_hinge_sup_delta_weight_max",
        type=float,
        default=None,
        help="Optional clamp for delta weight w. 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_sup_weight",
        type=float,
        default=None,
        help="If >0, supervise learned gate with BCE (swing=1, stance=0) from contact thresholding.",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_sup_contact_source",
        type=str,
        default=None,
        choices=("gt", "plan", "meas"),
        help="Contact source for learned gate supervision (gt|plan|meas).",
    )
    ap.add_argument(
        "--direct_pose_hinge_gate_sup_contact_thresh",
        type=float,
        default=None,
        help="Threshold for learned gate supervision (swing if contact<thr; default 0.5).",
    )
    ap.add_argument(
        "--direct_pose_hinge_stance_weight",
        type=float,
        default=None,
        help="If >0, suppress hinge delta on stance frames (penalize delta_raw when contact>=thr).",
    )
    ap.add_argument(
        "--direct_pose_hinge_stance_kind",
        type=str,
        default=None,
        choices=("smooth_l1", "l2"),
        help="Loss type for stance suppression on hinge delta.",
    )
    ap.add_argument(
        "--direct_pose_hinge_stance_contact_source",
        type=str,
        default=None,
        choices=("gt", "plan", "meas"),
        help="Contact source for stance suppression masking (gt|plan|meas).",
    )
    ap.add_argument(
        "--direct_pose_hinge_stance_contact_thresh",
        type=float,
        default=None,
        help="Threshold for stance masking (contact>=thr; default 0.5).",
    )
    ap.add_argument(
        "--direct_pose_hinge_reg_weight",
        type=float,
        default=None,
        help="If >0, add a contact-free regularizer on the applied hinge delta magnitude.",
    )
    ap.add_argument(
        "--direct_pose_hinge_reg_kind",
        type=str,
        default=None,
        choices=("l1", "l2", "smooth_l1"),
        help="Regularizer type for hinge delta magnitude (default: l1).",
    )
    ap.add_argument(
        "--direct_pose_loss_tail_mix",
        type=float,
        default=None,
        help="Stage6/7 direct objective: convex mix λ for tail/state weighting. 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_loss_tail_temp_deg",
        type=float,
        default=None,
        help="Stage6/7 direct objective: softmax temperature (deg) for stop-grad tail weights. 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_loss_state_swing_boost",
        type=float,
        default=None,
        help="Stage6/7 direct objective: swing boost a for state-aware weighting: w=1+a*(1-contact). 0 disables.",
    )
    ap.add_argument(
        "--direct_pose_loss_state_contact_source",
        type=str,
        default=None,
        choices=("gt", "plan", "meas"),
        help="Stage6/7 direct objective: contact source for swing boost (gt|plan|meas).",
    )
    ap.add_argument(
        "--direct_pose_loss_state_scope",
        type=str,
        default=None,
        choices=("legs", "limbs", "all"),
        help="Stage6/7 direct objective: joint scope for swing boost (legs|limbs|all).",
    )
    ap.add_argument(
        "--direct_pose_loss_leg_split",
        type=str,
        default=None,
        help="Stage7 direct objective: true|false; split legs vs non-legs: L = mean(nonleg) + mean(leg).",
    )
    ap.add_argument(
        "--direct_pose_loss_leg_tail_scale",
        type=str,
        default=None,
        choices=("center", "mad", "none"),
        help="DEPRECATED (no-op): tail term removed; kept for backward-compatible configs. center|mad|none.",
    )
    ap.add_argument(
        "--direct_pose_loss_sics",
        type=str,
        default=None,
        help='Optional: focus objective="direct" losses on selected step_in_cycle (sic). Example: "8,12,14,15,53,54,55,74". '
        'Supports ranges like "49-55" (inclusive).',
    )
    ap.add_argument(
        "--direct_pose_loss_cycle_gte",
        type=int,
        default=None,
        help='When --direct_pose_loss_sics is set, apply it only for rollout cycles >= N (0=all; 1 matches eval mask "cycle>=1").',
    )
    ap.add_argument(
        "--direct_pose_loss_sic_mode",
        type=str,
        default=None,
        choices=("mask", "boost"),
        help='When --direct_pose_loss_sics is set: "mask" trains only those steps; "boost" upweights them but keeps all steps.',
    )
    ap.add_argument(
        "--direct_pose_loss_sic_boost",
        type=float,
        default=None,
        help='Only for --direct_pose_loss_sic_mode=boost: multiplicative weight for selected steps (e.g. 10.0).',
    )
    ap.add_argument(
        "--direct_pose_loss_pair_boost_table_json",
        type=str,
        default=None,
        help=(
            "Optional: JSON table (alpha_by_sic_bone schema) used as a binary hotspot mask for "
            "direct objective pair weighting. Non-neutral pairs (alpha!=1) are boosted."
        ),
    )
    ap.add_argument(
        "--direct_pose_loss_pair_boost",
        type=float,
        default=None,
        help=(
            "Only used with --direct_pose_loss_pair_boost_table_json: multiplicative weight for selected "
            "(sic,bone) pairs in direct pose loss."
        ),
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
    args = ap.parse_args()

    base_cfg = load_json(Path(args.config).expanduser()) if args.config else {}
    payload: Dict[str, Any] = dict(base_cfg) if isinstance(base_cfg, dict) else {}

    def _set_if(name: str, value: Any) -> None:
        if value is None:
            return
        payload[name] = value

    _set_if("ckpt_in", args.ckpt_in)
    _set_if("out_dir", args.out_dir)
    _set_if("run_name", args.run_name)
    _set_if("data", args.data)
    if args.paths is not None:
        payload["paths"] = args.paths if args.paths else None
    _set_if("bundle_json", args.bundle_json)
    _set_if("pretrain_template", args.pretrain_template)
    _set_if("encoder_bundle", args.encoder_bundle)

    _set_if("device", args.device)
    _set_if("batch", args.batch)
    _set_if("seq_len", args.seq_len)
    _set_if("dataset_index_mode", args.dataset_index_mode)
    _set_if("rollout_steps", args.rollout_steps)
    _set_if("rollout_cycles", args.rollout_cycles)
    if args.rollout_include_boundary is not None:
        payload["rollout_include_boundary"] = str(args.rollout_include_boundary).strip().lower() in ("1", "true", "yes", "y")
    if args.rollout_random_offset is not None:
        payload["rollout_random_offset"] = str(args.rollout_random_offset).strip().lower() in ("1", "true", "yes", "y")
    _set_if("time_index_mode", args.time_index_mode)
    _set_if("phase_reset_source", args.phase_reset_source)
    _set_if("depth", args.depth)
    _set_if("num_heads", args.num_heads)
    _set_if("dropout", args.dropout)
    _set_if("context_len", args.context_len)

    _set_if("epochs", args.epochs)
    _set_if("steps_per_epoch", args.steps_per_epoch)
    _set_if("lr", args.lr)
    _set_if("weight_decay", args.weight_decay)

    if args.so3_corr_gate_force is not None:
        if str(args.so3_corr_gate_force).strip().lower() in ("null", "none", ""):
            payload["so3_corr_gate_force"] = None
        else:
            payload["so3_corr_gate_force"] = float(args.so3_corr_gate_force)
    if args.so3_corr_gate_logit_reset is not None:
        if str(args.so3_corr_gate_logit_reset).strip().lower() in ("null", "none", ""):
            payload["so3_corr_gate_logit_reset"] = None
        else:
            payload["so3_corr_gate_logit_reset"] = float(args.so3_corr_gate_logit_reset)
    _set_if("gate_warmup_steps", args.gate_warmup_steps)
    _set_if("gate_warmup_value", args.gate_warmup_value)
    _set_if("so3_corr_max_deg", args.so3_corr_max_deg)
    _set_if("so3_corr_omega_l2_weight", args.so3_corr_omega_l2_weight)
    _set_if("corr_time_weight_max", args.corr_time_weight_max)
    if args.detach_rollout_state is not None:
        payload["detach_rollout_state"] = str(args.detach_rollout_state).strip().lower() in ("1", "true", "yes", "y")
    if args.train_so3_corrector is not None:
        payload["train_so3_corrector"] = str(args.train_so3_corrector).strip().lower() in ("1", "true", "yes", "y")
    if args.train_contact_plan_init is not None:
        payload["train_contact_plan_init"] = str(args.train_contact_plan_init).strip().lower() in ("1", "true", "yes", "y")
    if args.train_contact_plan is not None:
        payload["train_contact_plan"] = str(args.train_contact_plan).strip().lower() in ("1", "true", "yes", "y")
    if args.train_direct_pose is not None:
        payload["train_direct_pose"] = str(args.train_direct_pose).strip().lower() in ("1", "true", "yes", "y")
    if args.train_lambda_head is not None:
        payload["train_lambda_head"] = str(args.train_lambda_head).strip().lower() in ("1", "true", "yes", "y")
    if args.train_contact_meas is not None:
        payload["train_contact_meas"] = str(args.train_contact_meas).strip().lower() in ("1", "true", "yes", "y")
    if args.train_contact_td_hazard is not None:
        payload["train_contact_td_hazard"] = str(args.train_contact_td_hazard).strip().lower() in ("1", "true", "yes", "y")
    if args.contact_meas_rollout is not None:
        payload["contact_meas_rollout"] = str(args.contact_meas_rollout).strip().lower() in ("1", "true", "yes", "y")
    if args.contact_td_hazard_rollout is not None:
        payload["contact_td_hazard_rollout"] = str(args.contact_td_hazard_rollout).strip().lower() in ("1", "true", "yes", "y")
    _set_if("contact_plan_init_weight", args.contact_plan_init_weight)
    _set_if("contact_plan_init_mode", args.contact_plan_init_mode)
    _set_if("contact_plan_init_hidden", args.contact_plan_init_hidden)
    _set_if("contact_plan_init_dropout", args.contact_plan_init_dropout)
    _set_if("contact_plan_weight", args.contact_plan_weight)
    _set_if("event_clock", args.event_clock)
    _set_if("event_clock_max_delta", args.event_clock_max_delta)
    _set_if("event_clock_hidden_dim", args.event_clock_hidden_dim)
    _set_if("event_clock_gate_hidden_dim", args.event_clock_gate_hidden_dim)
    _set_if("contact_meas_weight", args.contact_meas_weight)
    _set_if("contact_meas_rollout_weight", args.contact_meas_rollout_weight)
    _set_if("contact_meas_smooth_weight", args.contact_meas_smooth_weight)
    _set_if("contact_meas_smooth_kind", args.contact_meas_smooth_kind)
    _set_if("contact_meas_margin_weight", args.contact_meas_margin_weight)
    _set_if("contact_meas_margin_logit", args.contact_meas_margin_logit)
    _set_if("contact_meas_transition_band", args.contact_meas_transition_band)
    _set_if("contact_td_hazard_rollout_weight", args.contact_td_hazard_rollout_weight)
    _set_if("contact_td_hazard_bce_weight", args.contact_td_hazard_bce_weight)
    _set_if("contact_td_hazard_event_weight", args.contact_td_hazard_event_weight)
    _set_if("contact_td_hazard_mass_weight", args.contact_td_hazard_mass_weight)
    _set_if("contact_td_hazard_unimodal_weight", args.contact_td_hazard_unimodal_weight)
    _set_if("contact_td_hazard_entropy_weight", args.contact_td_hazard_entropy_weight)
    _set_if("contact_td_hazard_clock_weight", args.contact_td_hazard_clock_weight)
    _set_if("contact_td_hazard_hidden", args.contact_td_hazard_hidden)
    _set_if("contact_td_hazard_dropout", args.contact_td_hazard_dropout)
    _set_if("direct_pose_meas_force_zero", args.direct_pose_meas_force_zero)
    _set_if("direct_pose_meas_detach", args.direct_pose_meas_detach)
    _set_if("direct_pose_split_enable", args.direct_pose_split_enable)
    _set_if("direct_pose_nonleg_proj_dim", args.direct_pose_nonleg_proj_dim)
    _set_if("direct_pose_nonleg_train_only", args.direct_pose_nonleg_train_only)
    _set_if("direct_pose_leg_enable", args.direct_pose_leg_enable)
    _set_if("direct_pose_leg_train_only", args.direct_pose_leg_train_only)
    _set_if("direct_pose_leg_bones", args.direct_pose_leg_bones)
    _set_if("direct_pose_leg_mode", args.direct_pose_leg_mode)
    _set_if("direct_pose_leg_stopgrad_main", args.direct_pose_leg_stopgrad_main)
    _set_if("direct_pose_leg_detach_feat", args.direct_pose_leg_detach_feat)
    _set_if("direct_pose_leg_max_deg", args.direct_pose_leg_max_deg)
    _set_if("direct_pose_leg_gate_mode", args.direct_pose_leg_gate_mode)
    _set_if("direct_pose_leg_gate_power", args.direct_pose_leg_gate_power)
    _set_if("direct_pose_leg_scale_clamp_k", args.direct_pose_leg_scale_clamp_k)
    _set_if("direct_pose_leg_gate_sup_weight", args.direct_pose_leg_gate_sup_weight)
    _set_if("direct_pose_leg_align_weight", args.direct_pose_leg_align_weight)
    _set_if("direct_pose_leg_align_oracle_min_deg", args.direct_pose_leg_align_oracle_min_deg)
    _set_if("direct_pose_leg_align_oracle_weight_deg", args.direct_pose_leg_align_oracle_weight_deg)
    _set_if("direct_pose_leg_align_mode", args.direct_pose_leg_align_mode)
    _set_if("direct_pose_leg_align_mag_weight", args.direct_pose_leg_align_mag_weight)
    _set_if("direct_pose_leg_align_res_weight", args.direct_pose_leg_align_res_weight)
    _set_if("direct_pose_leg_align_sign_weight", args.direct_pose_leg_align_sign_weight)
    _set_if("direct_pose_leg_align_cos_thresh", args.direct_pose_leg_align_cos_thresh)
    _set_if("direct_pose_leg_side_routing", args.direct_pose_leg_side_routing)
    _set_if("direct_pose_leg_contact_order", args.direct_pose_leg_contact_order)
    _set_if("direct_pose_leg_side_embed_dim", args.direct_pose_leg_side_embed_dim)
    _set_if("direct_pose_leg_side_sign_gate", args.direct_pose_leg_side_sign_gate)
    _set_if("direct_pose_leg_side_sign_gate_reg_weight", args.direct_pose_leg_side_sign_gate_reg_weight)
    _set_if("direct_pose_leg_side_rank1", args.direct_pose_leg_side_rank1)
    _set_if("direct_pose_leg_side_rank1_bones", args.direct_pose_leg_side_rank1_bones)
    _set_if("direct_pose_hinge_enable", args.direct_pose_hinge_enable)
    _set_if("direct_pose_hinge_train_only", args.direct_pose_hinge_train_only)
    _set_if("direct_pose_hinge_gate_train_only", args.direct_pose_hinge_gate_train_only)
    _set_if("direct_pose_hinge_bones", args.direct_pose_hinge_bones)
    _set_if("direct_pose_hinge_axis", args.direct_pose_hinge_axis)
    _set_if("direct_pose_hinge_max_deg", args.direct_pose_hinge_max_deg)
    _set_if("direct_pose_hinge_hidden", args.direct_pose_hinge_hidden)
    _set_if("direct_pose_hinge_feat_source", args.direct_pose_hinge_feat_source)
    _set_if("direct_pose_hinge_clean", args.direct_pose_hinge_clean)
    _set_if("direct_pose_hinge_eps_max_deg", args.direct_pose_hinge_eps_max_deg)
    _set_if("direct_pose_hinge_eps_max_scale", args.direct_pose_hinge_eps_max_scale)
    _set_if("direct_pose_hinge_eps_hidden", args.direct_pose_hinge_eps_hidden)
    _set_if("direct_pose_hinge_eps_dropout", args.direct_pose_hinge_eps_dropout)
    _set_if("direct_pose_hinge_eps_lr_scale", args.direct_pose_hinge_eps_lr_scale)
    _set_if("direct_pose_hinge_eps_l2_weight", args.direct_pose_hinge_eps_l2_weight)
    _set_if("direct_pose_hinge_gate_mode", args.direct_pose_hinge_gate_mode)
    _set_if("direct_pose_hinge_gate_source", args.direct_pose_hinge_gate_source)
    _set_if("direct_pose_hinge_gate_power", args.direct_pose_hinge_gate_power)
    _set_if("direct_pose_hinge_sup_weight", args.direct_pose_hinge_sup_weight)
    _set_if("direct_pose_hinge_sup_kind", args.direct_pose_hinge_sup_kind)
    _set_if("direct_pose_hinge_sup_contact_source", args.direct_pose_hinge_sup_contact_source)
    _set_if("direct_pose_hinge_sup_contact_value", args.direct_pose_hinge_sup_contact_value)
    _set_if("direct_pose_hinge_sup_contact_thresh", args.direct_pose_hinge_sup_contact_thresh)
    _set_if("direct_pose_hinge_sup_angle_thresh_deg", args.direct_pose_hinge_sup_angle_thresh_deg)
    _set_if("direct_pose_hinge_sup_delta_thresh_deg", args.direct_pose_hinge_sup_delta_thresh_deg)
    _set_if("direct_pose_hinge_sup_delta_weight_power", args.direct_pose_hinge_sup_delta_weight_power)
    _set_if("direct_pose_hinge_sup_delta_weight_scale_deg", args.direct_pose_hinge_sup_delta_weight_scale_deg)
    _set_if("direct_pose_hinge_sup_delta_weight_max", args.direct_pose_hinge_sup_delta_weight_max)
    _set_if("direct_pose_hinge_gate_sup_weight", args.direct_pose_hinge_gate_sup_weight)
    _set_if("direct_pose_hinge_gate_sup_contact_source", args.direct_pose_hinge_gate_sup_contact_source)
    _set_if("direct_pose_hinge_gate_sup_contact_thresh", args.direct_pose_hinge_gate_sup_contact_thresh)
    _set_if("direct_pose_hinge_stance_weight", args.direct_pose_hinge_stance_weight)
    _set_if("direct_pose_hinge_stance_kind", args.direct_pose_hinge_stance_kind)
    _set_if("direct_pose_hinge_stance_contact_source", args.direct_pose_hinge_stance_contact_source)
    _set_if("direct_pose_hinge_stance_contact_thresh", args.direct_pose_hinge_stance_contact_thresh)
    _set_if("direct_pose_hinge_reg_weight", args.direct_pose_hinge_reg_weight)
    _set_if("direct_pose_hinge_reg_kind", args.direct_pose_hinge_reg_kind)
    _set_if("direct_pose_loss_tail_mix", args.direct_pose_loss_tail_mix)
    _set_if("direct_pose_loss_tail_temp_deg", args.direct_pose_loss_tail_temp_deg)
    _set_if("direct_pose_loss_state_swing_boost", args.direct_pose_loss_state_swing_boost)
    _set_if("direct_pose_loss_state_contact_source", args.direct_pose_loss_state_contact_source)
    _set_if("direct_pose_loss_state_scope", args.direct_pose_loss_state_scope)
    _set_if("direct_pose_loss_leg_split", args.direct_pose_loss_leg_split)
    _set_if("direct_pose_loss_leg_tail_scale", args.direct_pose_loss_leg_tail_scale)
    _set_if("direct_pose_loss_sics", args.direct_pose_loss_sics)
    _set_if("direct_pose_loss_cycle_gte", args.direct_pose_loss_cycle_gte)
    _set_if("direct_pose_loss_sic_mode", args.direct_pose_loss_sic_mode)
    _set_if("direct_pose_loss_sic_boost", args.direct_pose_loss_sic_boost)
    _set_if("direct_pose_loss_pair_boost_table_json", args.direct_pose_loss_pair_boost_table_json)
    _set_if("direct_pose_loss_pair_boost", args.direct_pose_loss_pair_boost)
    _set_if("direct_pose_loss_group_norm_enable", args.direct_pose_loss_group_norm_enable)
    _set_if("direct_pose_loss_group_norm_w_leg", args.direct_pose_loss_group_norm_w_leg)
    _set_if("direct_pose_loss_group_norm_w_nonleg", args.direct_pose_loss_group_norm_w_nonleg)
    _set_if("direct_pose_loss_group_norm_ema_beta", args.direct_pose_loss_group_norm_ema_beta)
    _set_if("direct_pose_loss_group_norm_ratio_min", args.direct_pose_loss_group_norm_ratio_min)
    _set_if("direct_pose_loss_group_norm_ratio_max", args.direct_pose_loss_group_norm_ratio_max)
    _set_if("direct_pose_loss_group_norm_eps", args.direct_pose_loss_group_norm_eps)
    _set_if("direct_pose_grad_monitor_enable", args.direct_pose_grad_monitor_enable)
    _set_if("direct_pose_grad_ratio_gate", args.direct_pose_grad_ratio_gate)
    _set_if("contact_meas_gate_by_hit", args.contact_meas_gate_by_hit)
    _set_if("contact_meas_vxy_mode", args.contact_meas_vxy_mode)
    _set_if("contact_meas_ground_z_mode", args.contact_meas_ground_z_mode)
    _set_if("contact_meas_ground_z_beta", args.contact_meas_ground_z_beta)
    _set_if("contact_meas_ground_z_window", args.contact_meas_ground_z_window)
    _set_if("contact_meas_ground_z_quantile", args.contact_meas_ground_z_quantile)
    _set_if("contact_meas_ground_z_slew_up_cm", args.contact_meas_ground_z_slew_up_cm)
    _set_if("contact_meas_ground_z_slew_down_cm", args.contact_meas_ground_z_slew_down_cm)
    _set_if("lambda_fusion_mode", args.lambda_fusion_mode)
    _set_if("lambda_fusion_hidden", args.lambda_fusion_hidden)
    _set_if("lambda_fusion_dropout", args.lambda_fusion_dropout)
    _set_if("lambda_fusion_logit_init", args.lambda_fusion_logit_init)
    _set_if("lambda_fusion_use_rollout_step", args.lambda_fusion_use_rollout_step)
    _set_if("lambda_fusion_entropy_weight", args.lambda_fusion_entropy_weight)
    _set_if("lambda_fusion_smooth_weight", args.lambda_fusion_smooth_weight)
    _set_if("lambda_fusion_early_steps", args.lambda_fusion_early_steps)
    _set_if("lambda_fusion_early_weight", args.lambda_fusion_early_weight)
    _set_if("lambda_fusion_monotonic_weight", args.lambda_fusion_monotonic_weight)
    _set_if("lambda_plan_entropy_weight", args.lambda_plan_entropy_weight)
    _set_if("lambda_plan_dyn_weight", args.lambda_plan_dyn_weight)
    _set_if("lambda_time_weight_mode", args.lambda_time_weight_mode)
    _set_if("lambda_time_weight_max", args.lambda_time_weight_max)
    _set_if("lambda_reliability_mode", args.lambda_reliability_mode)
    _set_if("lambda_reliability_warmup_steps", args.lambda_reliability_warmup_steps)
    _set_if("lambda_reliability_contact_err_max", args.lambda_reliability_contact_err_max)
    _set_if("lambda_reliability_warmup_joint_scales", args.lambda_reliability_warmup_joint_scales)
    _set_if("lambda_l2sp_weight", args.lambda_l2sp_weight)
    _set_if("lambda_boundary_weight", args.lambda_boundary_weight)
    _set_if("lambda_gate_sup_weight", args.lambda_gate_sup_weight)
    _set_if("lambda_gate_sup_tau_deg", args.lambda_gate_sup_tau_deg)
    _set_if("lambda_gate_sup_margin_deg", args.lambda_gate_sup_margin_deg)
    _set_if("lambda_gate_sup_start_step", args.lambda_gate_sup_start_step)
    _set_if("time_index_mode", args.time_index_mode)
    _set_if("seed", args.seed)

    cfg = _cfg_from_payload(payload)
    _set_seed(cfg.seed)

    if cfg.train_contact_meas and float(cfg.contact_meas_weight or 0.0) <= 0.0:
        print("[posttrain][WARN] train_contact_meas=true but contact_meas_weight<=0; meas head will not be supervised.")
    if (not cfg.train_contact_meas) and float(cfg.contact_meas_weight or 0.0) > 0.0:
        print("[posttrain][WARN] contact_meas_weight>0 but train_contact_meas=false; meas head is frozen so this term is ignored.")
    if (
        cfg.train_contact_td_hazard
        and float(cfg.contact_td_hazard_bce_weight or 0.0) <= 0.0
        and float(cfg.contact_td_hazard_mass_weight or 0.0) <= 0.0
        and float(cfg.contact_td_hazard_unimodal_weight or 0.0) <= 0.0
        and float(cfg.contact_td_hazard_entropy_weight or 0.0) <= 0.0
        and float(cfg.contact_td_hazard_clock_weight or 0.0) <= 0.0
    ):
        print(
            "[posttrain][WARN] train_contact_td_hazard=true but all TD hazard weights are <=0; hazard head will not be supervised."
        )
    if (not cfg.train_contact_td_hazard) and (
        float(cfg.contact_td_hazard_bce_weight or 0.0) > 0.0
        or float(cfg.contact_td_hazard_mass_weight or 0.0) > 0.0
        or float(cfg.contact_td_hazard_unimodal_weight or 0.0) > 0.0
        or float(cfg.contact_td_hazard_entropy_weight or 0.0) > 0.0
        or float(cfg.contact_td_hazard_clock_weight or 0.0) > 0.0
    ):
        print(
            "[posttrain][WARN] contact_td_hazard_*_weight>0 but train_contact_td_hazard=false; TD hazard head is frozen so these terms are ignored."
        )
    if bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)) and (not bool(getattr(cfg, "direct_pose_loss_leg_split", False))):
        print("[posttrain][WARN] direct_pose_loss_group_norm_enable=true but direct_pose_loss_leg_split=false; group norm will have no effect.")

    device = _resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    norm_spec = _merge_norm_spec(cfg.bundle_json.expanduser().resolve(), cfg.pretrain_template)

    ds = MotionEventDataset(
        data_dir=str(cfg.data.expanduser().resolve()),
        seq_len=max(2, int(cfg.seq_len)),
        paths=[str(p.expanduser().resolve()) for p in cfg.paths] if cfg.paths else None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode=str(getattr(cfg, "dataset_index_mode", "sliding") or "sliding"),
    )
    ds.is_train = True
    if len(ds) <= 0:
        clip_lens: list[tuple[str, int]] = []
        for clip in getattr(ds, "clips", []) or []:
            try:
                clip_lens.append((str(getattr(clip, "npz_path", "?")), int(getattr(clip, "X", np.zeros((0,))).shape[0])))
            except Exception:
                pass
        clip_lens.sort(key=lambda x: x[1])
        hint = ""
        if clip_lens:
            smallest = ", ".join([f"{Path(p).name}:{n}" for p, n in clip_lens[:5]])
            hint = f" Smallest clips: {smallest}."
        raise SystemExit(
            f"[FATAL] posttrain dataset has 0 samples. seq_len={cfg.seq_len} is likely too large or paths/data are wrong."
            + hint
            + " Try lowering --seq_len or passing --paths to restrict to longer clips."
        )
    loader = DataLoader(ds, batch_size=int(cfg.batch), shuffle=True, drop_last=True, num_workers=0)
    if len(loader) <= 0:
        raise SystemExit(
            f"[FATAL] posttrain DataLoader has 0 batches (len(dataset)={len(ds)}, batch={int(cfg.batch)}, drop_last=True). "
            "Lower --batch or use more/longer --paths (or reduce --seq_len)."
        )
    batch_iter = _iter_infinite(loader)

    ckpt = torch.load(cfg.ckpt_in.expanduser(), map_location="cpu")
    ckpt_posttrain_cfg: Optional[dict[str, Any]] = None
    try:
        if isinstance(ckpt, dict):
            pt = ckpt.get("posttrain_cfg", None)
            if isinstance(pt, dict):
                ckpt_posttrain_cfg = pt
    except Exception:
        ckpt_posttrain_cfg = None

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

    # Resolve instantiation config (auto -> infer from ckpt when possible).
    direct_pose_enable = bool(direct_pose_enable_infer or direct_has_weights or bool(getattr(cfg, "train_direct_pose", False)) or direct_pose_reinit)
    direct_pose_hidden = int(getattr(cfg, "direct_pose_hidden_override", None) or direct_pose_hidden_infer)
    direct_pose_meas_mode = str(getattr(cfg, "direct_pose_meas_mode_override", None) or direct_pose_meas_mode_infer)
    direct_pose_feat_source = str(getattr(cfg, "direct_pose_feat_source", "auto") or "auto").lower().strip()
    direct_pose_time_pe_dim = int(getattr(cfg, "direct_pose_time_pe_dim", -1))
    direct_pose_time_pe_base = float(getattr(cfg, "direct_pose_time_pe_base", 10000.0) or 10000.0)
    direct_pose_use_phase_z = bool(getattr(cfg, "direct_pose_use_phase_z", False))
    direct_pose_phase_z_mode = str(getattr(cfg, "direct_pose_phase_z_mode", "concat") or "concat").strip().lower()
    direct_pose_split_enable_infer = False
    direct_pose_nonleg_proj_dim_infer = 0
    try:
        direct_pose_split_enable_infer = bool(
            any(str(k).startswith("direct_pose_out_leg.") for k in state_dict.keys())
            and any(str(k).startswith("direct_pose_out_nonleg.") for k in state_dict.keys())
        )
    except Exception:
        direct_pose_split_enable_infer = False
    if (not direct_pose_split_enable_infer) and isinstance(ckpt_posttrain_cfg, dict):
        try:
            direct_pose_split_enable_infer = bool(ckpt_posttrain_cfg.get("direct_pose_split_enable", False))
        except Exception:
            direct_pose_split_enable_infer = False
    try:
        if isinstance(ckpt_posttrain_cfg, dict):
            v = int(ckpt_posttrain_cfg.get("direct_pose_nonleg_proj_dim", 0) or 0)
            if v > 0:
                direct_pose_nonleg_proj_dim_infer = int(v)
    except Exception:
        direct_pose_nonleg_proj_dim_infer = 0
    try:
        w_non = state_dict.get("direct_pose_out_nonleg.weight", None)
        w_proj = state_dict.get("direct_pose_nonleg_proj.0.weight", None)
        if torch.is_tensor(w_proj) and w_proj.ndim == 2 and int(w_proj.shape[0]) > 0:
            direct_pose_nonleg_proj_dim_infer = int(w_proj.shape[0])
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
    split_cfg_explicit = bool("direct_pose_split_enable" in payload)
    direct_pose_split_enable = bool(getattr(cfg, "direct_pose_split_enable", False))
    if (not split_cfg_explicit) and (not direct_pose_split_enable) and bool(direct_pose_split_enable_infer):
        # Keep split mode by default for split checkpoints unless explicitly overridden.
        direct_pose_split_enable = True
    nonleg_proj_cfg_explicit = bool("direct_pose_nonleg_proj_dim" in payload)
    direct_pose_nonleg_proj_dim = int(getattr(cfg, "direct_pose_nonleg_proj_dim", 0) or 0)
    if (not nonleg_proj_cfg_explicit) and int(direct_pose_nonleg_proj_dim) <= 0 and int(direct_pose_nonleg_proj_dim_infer) > 0:
        direct_pose_nonleg_proj_dim = int(direct_pose_nonleg_proj_dim_infer)
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
    def _normalize_direct_pose_feat_source(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in ("", "auto"):
            return None
        if s in ("h", "h_final", "hidden_only", "post", "final"):
            s = "hidden"
        if s in ("h_pre", "h_temporal", "hidden_pre", "pre", "temporal", "mid"):
            s = "hidden_pre"
        if s in ("cond_hidden", "hidden_cond", "concat", "cond+hidden", "hidden+cond"):
            s = "cond+hidden"
        if s in ("cond+hidden_pre", "cond_hidden_pre", "hidden_pre+cond", "cond+pre", "pre+cond"):
            s = "cond+hidden_pre"
        if s in ("cond", "hidden", "hidden_pre", "cond+hidden", "cond+hidden_pre"):
            return s
        return None

    if direct_pose_feat_source == "auto":
        # Prefer checkpoint posttrain_cfg when present (cannot infer hidden_pre from tensor shapes).
        hint = None
        if isinstance(ckpt_posttrain_cfg, dict):
            hint = _normalize_direct_pose_feat_source(ckpt_posttrain_cfg.get("direct_pose_feat_source", None))
        direct_pose_feat_source = hint or (direct_pose_feat_source_infer if direct_pose_enable_infer else "cond")
    if direct_pose_feat_source in ("h", "h_final", "hidden_only"):
        direct_pose_feat_source = "hidden"
    if direct_pose_feat_source in ("h_pre", "h_temporal", "hidden_pre", "pre", "temporal", "mid"):
        direct_pose_feat_source = "hidden_pre"
    if direct_pose_feat_source in ("cond_hidden", "hidden_cond", "concat", "cond+hidden", "hidden+cond"):
        direct_pose_feat_source = "cond+hidden"
    if direct_pose_feat_source in ("cond+hidden_pre", "cond_hidden_pre", "hidden_pre+cond", "cond+pre", "pre+cond"):
        direct_pose_feat_source = "cond+hidden_pre"
    if direct_pose_feat_source not in ("cond", "hidden", "hidden_pre", "cond+hidden", "cond+hidden_pre"):
        direct_pose_feat_source = "cond"
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
        allow_legacy_to_split = bool(direct_pose_split_enable) and (not bool(direct_pose_split_enable_infer))
        if not allow_legacy_to_split:
            if not bool(getattr(cfg, "train_direct_pose", False)):
                raise SystemExit(
                    "[FATAL] direct_pose split mode differs from checkpoint but train_direct_pose=false. "
                    "Enable train_direct_pose (or match direct_pose_split_enable to checkpoint)."
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

    meas_has_weights = any(k.startswith("contact_meas_head.") for k in state_dict.keys())
    contact_meas_enable = bool(meas_has_weights)
    contact_meas_hidden = 64
    if meas_has_weights:
        w0 = state_dict.get("contact_meas_head.mlp.0.weight", None)
        if not (torch.is_tensor(w0) and w0.ndim == 2):
            raise SystemExit(
                "[FATAL] This repo now only supports contact_meas_head v1 (lowerbody_nohist_v1). "
                "The provided checkpoint seems to contain a legacy contact_meas_head; please retrain."
            )
        contact_meas_hidden = int(w0.shape[0])

    # ---- Infer contact TD hazard head (touchdown event intensity; integrate-to-1 clock) ----
    td_hazard_has_weights = any(k.startswith("contact_td_hazard_head.") for k in state_dict.keys())
    contact_td_hazard_enable = bool(td_hazard_has_weights or bool(cfg.train_contact_td_hazard))
    contact_td_hazard_hidden = int(getattr(cfg, "contact_td_hazard_hidden", 64) or 64)
    contact_td_hazard_dropout = float(getattr(cfg, "contact_td_hazard_dropout", 0.0) or 0.0)
    if td_hazard_has_weights:
        w0 = state_dict.get("contact_td_hazard_head.mlp.0.weight", None)
        if torch.is_tensor(w0) and w0.ndim == 2:
            contact_td_hazard_hidden = int(w0.shape[0])

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

    phase_reset_source_model = str(cfg.phase_reset_source or "contacts_meas").strip().lower()
    # Consistent with run_freerun_cycles:
    # - contacts_meas: internal threshold-crossing resets inside the model (event_kind controls it)
    # - ttc_gt: resets are applied externally (posttrain rollout loops), so disable internal resets
    # - td_hazard: phase resets are handled by integrate-to-1 inside the model; disable contacts_meas resets
    phase_event_kind_model = str(getattr(cfg, "contact_phase_state_event_kind", "touchdown") or "touchdown").strip().lower()
    phase_min_interval_model = int(getattr(cfg, "contact_phase_state_event_min_interval", 0) or 0)
    if phase_reset_source_model in ("ttc_gt", "td_hazard"):
        phase_event_kind_model = "none"
        phase_min_interval_model = 0

    # ---- Resolve direct hinge config (preserve checkpoint behavior for Stage5) ----
    #
    # posttrain_lambda_fusion.json intentionally focuses on λ and historically omitted direct_pose_hinge_*.
    # If we don't reconstruct hinge config from the input checkpoint, we would:
    #   - fail to instantiate the hinge heads (dropping hinge weights), and/or
    #   - change direct expert behavior during λ training (mismatch).
    #
    # We only do this reconstruction for lambda-only posttrain to avoid surprising behavior when
    # users are explicitly training the direct expert (Stage4/hinge-only).
    hinge_has_weights = any(
        str(k).startswith("direct_pose_hinge_head.")
        or str(k).startswith("direct_pose_hinge_nonhidden_head.")
        or str(k).startswith("direct_pose_hinge_eps_head.")
        or str(k).startswith("direct_pose_hinge_gate_head.")
        or str(k).startswith("direct_pose_hinge_gate_head_clean.")
        for k in state_dict.keys()
    )

    direct_pose_hinge_enable_model = bool(getattr(cfg, "direct_pose_hinge_enable", False))
    direct_pose_hinge_bones_model = getattr(cfg, "direct_pose_hinge_bones", None)
    direct_pose_hinge_axis_model = str(getattr(cfg, "direct_pose_hinge_axis", "z") or "z")
    direct_pose_hinge_max_deg_model = float(getattr(cfg, "direct_pose_hinge_max_deg", 45.0) or 45.0)
    direct_pose_hinge_hidden_model = getattr(cfg, "direct_pose_hinge_hidden", None)
    direct_pose_hinge_feat_source_model: Optional[str] = (
        None
        if str(getattr(cfg, "direct_pose_hinge_feat_source", "auto") or "auto").strip().lower() == "auto"
        else str(getattr(cfg, "direct_pose_hinge_feat_source", "auto") or "auto")
    )
    direct_pose_hinge_base_feat_model = str(getattr(cfg, "direct_pose_hinge_base_feat", "none") or "none")
    direct_pose_hinge_clean_model = bool(getattr(cfg, "direct_pose_hinge_clean", False))

    direct_pose_hinge_eps_max_deg_model: Optional[float] = (
        None
        if float(getattr(cfg, "direct_pose_hinge_eps_max_deg", 0.0) or 0.0) <= 0.0
        else float(getattr(cfg, "direct_pose_hinge_eps_max_deg", 0.0) or 0.0)
    )
    # Allow 0.0 to explicitly disable eps(hidden) in clean hinge mode.
    direct_pose_hinge_eps_max_scale_model = float(
        getattr(cfg, "direct_pose_hinge_eps_max_scale", 0.5)
        if getattr(cfg, "direct_pose_hinge_eps_max_scale", None) is not None
        else 0.5
    )
    direct_pose_hinge_eps_hidden_model = getattr(cfg, "direct_pose_hinge_eps_hidden", None)
    direct_pose_hinge_eps_dropout_model = float(getattr(cfg, "direct_pose_hinge_eps_dropout", 0.0) or 0.0)
    direct_pose_hinge_eps_source_model = str(getattr(cfg, "direct_pose_hinge_eps_source", "hidden") or "hidden")

    direct_pose_hinge_gate_mode_model = str(getattr(cfg, "direct_pose_hinge_gate_mode", "none") or "none")
    direct_pose_hinge_gate_source_model = str(getattr(cfg, "direct_pose_hinge_gate_source", "plan") or "plan")
    direct_pose_hinge_gate_power_model = float(getattr(cfg, "direct_pose_hinge_gate_power", 1.0) or 1.0)

    # NOTE(2026-01-24): hinge is treated as a debugging tool and is not part of the default posttrain pipeline.
    # We intentionally do NOT auto-enable/auto-reconstruct direct_pose_hinge_* from checkpoint weights during
    # lambda-only posttrain, to avoid silently coupling the posttrain flow to hinge behavior.
    if False and bool(getattr(cfg, "train_lambda_head", False)) and (not bool(getattr(cfg, "train_direct_pose", False))) and hinge_has_weights:
        # (Deprecated) Preserve the checkpoint's hinge behavior unless the user explicitly overrides it.
        direct_pose_hinge_enable_model = True

        if isinstance(ckpt_posttrain_cfg, dict):
            # NOTE: keep parsing permissive; posttrain_cfg is user-provided and may contain strings.
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_bones", None)
                if v is not None:
                    direct_pose_hinge_bones_model = v
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_axis", None)
                if v is not None:
                    direct_pose_hinge_axis_model = str(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_max_deg", None)
                if v is not None:
                    direct_pose_hinge_max_deg_model = float(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_base_feat", None)
                if v is not None:
                    direct_pose_hinge_base_feat_model = str(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_clean", None)
                if v is not None:
                    direct_pose_hinge_clean_model = bool(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_eps_max_deg", None)
                if v is not None:
                    vv = float(v)
                    direct_pose_hinge_eps_max_deg_model = None if (not math.isfinite(vv) or vv <= 0.0) else vv
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_eps_max_scale", None)
                if v is not None:
                    vv = float(v)
                    if (not math.isfinite(vv)) or vv < 0.0:
                        vv = 0.5
                    direct_pose_hinge_eps_max_scale_model = float(vv)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_eps_hidden", None)
                direct_pose_hinge_eps_hidden_model = int(v) if v is not None else None
            except Exception:
                direct_pose_hinge_eps_hidden_model = None
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_eps_dropout", None)
                if v is not None:
                    vv = float(v)
                    if (not math.isfinite(vv)) or vv < 0.0:
                        vv = 0.0
                    direct_pose_hinge_eps_dropout_model = float(max(0.0, min(1.0, vv)))
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_eps_source", None)
                if v is not None:
                    s = str(v).strip().lower()
                    if s in ("h_pre", "h_temporal", "pre", "temporal", "mid", "hidden_pre"):
                        direct_pose_hinge_eps_source_model = "hidden_pre"
                    elif s in ("h_final", "post", "final", "hidden"):
                        direct_pose_hinge_eps_source_model = "hidden"
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_gate_mode", None)
                if v is not None:
                    direct_pose_hinge_gate_mode_model = str(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_gate_source", None)
                if v is not None:
                    direct_pose_hinge_gate_source_model = str(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_gate_power", None)
                if v is not None:
                    direct_pose_hinge_gate_power_model = float(v)
            except Exception:
                pass
            try:
                v = ckpt_posttrain_cfg.get("direct_pose_hinge_feat_source", None)
                if v is not None:
                    s = str(v).strip().lower()
                    if s not in ("", "auto"):
                        direct_pose_hinge_feat_source_model = str(v)
            except Exception:
                pass

        # If not explicitly set, infer clean/legacy hinge mode from weights.
        if not bool(direct_pose_hinge_clean_model):
            try:
                if ("direct_pose_hinge_nonhidden_head.0.weight" in state_dict) or ("direct_pose_hinge_eps_head.1.weight" in state_dict):
                    direct_pose_hinge_clean_model = True
            except Exception:
                pass

        # If a learned gate exists in ckpt, reconstruct it (otherwise load_state_dict will drop it).
        try:
            gate_has_weights = ("direct_pose_hinge_gate_head.0.weight" in state_dict) or ("direct_pose_hinge_gate_head_clean.0.weight" in state_dict)
            if gate_has_weights and str(direct_pose_hinge_gate_mode_model or "none").strip().lower() != "learned":
                direct_pose_hinge_gate_mode_model = "learned"
        except Exception:
            pass

        # Infer hinge hidden sizes from weights to avoid load_state_dict size mismatches.
        try:
            if bool(direct_pose_hinge_clean_model):
                w0 = state_dict.get("direct_pose_hinge_nonhidden_head.0.weight", None)
                if torch.is_tensor(w0) and w0.ndim == 2 and int(w0.shape[0]) > 0:
                    direct_pose_hinge_hidden_model = int(w0.shape[0])
                if direct_pose_hinge_eps_hidden_model is None:
                    w_eps = state_dict.get("direct_pose_hinge_eps_head.1.weight", None)
                    if torch.is_tensor(w_eps) and w_eps.ndim == 2 and int(w_eps.shape[0]) > 0:
                        direct_pose_hinge_eps_hidden_model = int(w_eps.shape[0])
            else:
                w0 = state_dict.get("direct_pose_hinge_head.0.weight", None)
                if torch.is_tensor(w0) and w0.ndim == 2 and int(w0.shape[0]) > 0:
                    direct_pose_hinge_hidden_model = int(w0.shape[0])
        except Exception:
            pass

    # ---- Resolve leg gate config (preserve ckpt behavior; allow cfg 'auto') ----
    direct_pose_leg_gate_mode_model = str(getattr(cfg, "direct_pose_leg_gate_mode", "auto") or "auto").strip().lower()
    if direct_pose_leg_gate_mode_model in ("", "auto"):
        direct_pose_leg_gate_mode_model = "auto"
    elif direct_pose_leg_gate_mode_model in ("learned", "on", "true", "1", "yes", "y"):
        direct_pose_leg_gate_mode_model = "learned"
    elif direct_pose_leg_gate_mode_model in (
        "signed_scale",
        "signedscale",
        "signed",
        "signmag",
        "sign_mag",
        "signmagscale",
        "signedmag",
        "sscale",
    ):
        direct_pose_leg_gate_mode_model = "signed_scale"
    elif direct_pose_leg_gate_mode_model in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
        direct_pose_leg_gate_mode_model = "scale"
    elif direct_pose_leg_gate_mode_model in ("none", "off", "false", "0", "no", "n", "disable", "disabled"):
        direct_pose_leg_gate_mode_model = "none"
    else:
        direct_pose_leg_gate_mode_model = "auto"
    try:
        direct_pose_leg_gate_power_model = float(getattr(cfg, "direct_pose_leg_gate_power", 1.0) or 1.0)
    except Exception:
        direct_pose_leg_gate_power_model = 1.0
    if (not math.isfinite(direct_pose_leg_gate_power_model)) or direct_pose_leg_gate_power_model <= 0.0:
        direct_pose_leg_gate_power_model = 1.0

    # If cfg doesn't explicitly set gate mode (auto), preserve the input ckpt behavior.
    if direct_pose_leg_gate_mode_model == "auto" and isinstance(ckpt_posttrain_cfg, dict):
        try:
            v = ckpt_posttrain_cfg.get("direct_pose_leg_gate_mode", None)
            if v is not None:
                s = str(v).strip().lower()
                if s in ("", "auto"):
                    direct_pose_leg_gate_mode_model = "auto"
                elif s in ("learned", "on", "true", "1", "yes", "y"):
                    direct_pose_leg_gate_mode_model = "learned"
                elif s in (
                    "signed_scale",
                    "signedscale",
                    "signed",
                    "signmag",
                    "sign_mag",
                    "signmagscale",
                    "signedmag",
                    "sscale",
                ):
                    direct_pose_leg_gate_mode_model = "signed_scale"
                elif s in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
                    direct_pose_leg_gate_mode_model = "scale"
                elif s in ("none", "off", "false", "0", "no", "n", "disable", "disabled"):
                    direct_pose_leg_gate_mode_model = "none"
        except Exception:
            pass
        try:
            v = ckpt_posttrain_cfg.get("direct_pose_leg_gate_power", None)
            if v is not None:
                vv = float(v)
                if math.isfinite(vv) and vv > 0.0:
                    direct_pose_leg_gate_power_model = float(vv)
        except Exception:
            pass

    leg_gate_has_weights = any(
        str(k).startswith("direct_pose_leg_gate_head.")
        or str(k).startswith("direct_pose_leg_gate_head_shared.")
        for k in state_dict.keys()
    )
    # If a leg gate/scale head exists in ckpt, reconstruct it (otherwise load_state_dict will drop it).
    # Note: without an explicit cfg/ckpt mode, this is ambiguous (same module name for learned/scale),
    # so we default to "learned" for backward compatibility unless scale supervision is enabled.
    if leg_gate_has_weights and direct_pose_leg_gate_mode_model == "auto":
        try:
            scale_sup_w = float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0)
        except Exception:
            scale_sup_w = 0.0
        if scale_sup_w > 0.0:
            direct_pose_leg_gate_mode_model = "scale"
        else:
            direct_pose_leg_gate_mode_model = "learned"
    if direct_pose_leg_gate_mode_model == "auto":
        try:
            leg_gate_sup_w = float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0)
        except Exception:
            leg_gate_sup_w = 0.0
        try:
            scale_sup_w = float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0)
        except Exception:
            scale_sup_w = 0.0
        if scale_sup_w > 0.0:
            direct_pose_leg_gate_mode_model = "scale"
        elif leg_gate_has_weights or (leg_gate_sup_w > 0.0):
            direct_pose_leg_gate_mode_model = "learned"
        else:
            direct_pose_leg_gate_mode_model = "none"

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
        direct_pose_leg_side_routing=bool(getattr(cfg, "direct_pose_leg_side_routing", False)),
        direct_pose_leg_contact_order=str(getattr(cfg, "direct_pose_leg_contact_order", "lr") or "lr"),
        direct_pose_leg_side_embed_dim=int(getattr(cfg, "direct_pose_leg_side_embed_dim", 0) or 0),
        direct_pose_leg_side_plan_other=bool(getattr(cfg, "direct_pose_leg_side_plan_other", False)),
        direct_pose_leg_side_phase_other=bool(getattr(cfg, "direct_pose_leg_side_phase_other", False)),
        direct_pose_leg_side_phase_rel=bool(getattr(cfg, "direct_pose_leg_side_phase_rel", False)),
        direct_pose_leg_side_cue=str(getattr(cfg, "direct_pose_leg_side_cue", "none") or "none"),
        direct_pose_leg_side_cue_tau=float(getattr(cfg, "direct_pose_leg_side_cue_tau", 30.0) or 30.0),
        direct_pose_leg_side_sign_gate=bool(getattr(cfg, "direct_pose_leg_side_sign_gate", False)),
        direct_pose_leg_side_rank1=bool(getattr(cfg, "direct_pose_leg_side_rank1", False)),
        direct_pose_hinge_enable=bool(direct_pose_hinge_enable_model),
        direct_pose_hinge_bones=direct_pose_hinge_bones_model,
        direct_pose_hinge_axis=str(direct_pose_hinge_axis_model),
        direct_pose_hinge_max_deg=float(direct_pose_hinge_max_deg_model),
        direct_pose_hinge_hidden=direct_pose_hinge_hidden_model,
        direct_pose_hinge_feat_source=direct_pose_hinge_feat_source_model,
        direct_pose_hinge_base_feat=str(direct_pose_hinge_base_feat_model),
        direct_pose_hinge_clean=bool(direct_pose_hinge_clean_model),
        direct_pose_hinge_eps_max_deg=direct_pose_hinge_eps_max_deg_model,
        direct_pose_hinge_eps_max_scale=float(direct_pose_hinge_eps_max_scale_model),
        direct_pose_hinge_eps_hidden=direct_pose_hinge_eps_hidden_model,
        direct_pose_hinge_eps_dropout=float(direct_pose_hinge_eps_dropout_model),
        direct_pose_hinge_eps_source=str(direct_pose_hinge_eps_source_model),
        direct_pose_hinge_gate_mode=str(direct_pose_hinge_gate_mode_model),
        direct_pose_hinge_gate_source=str(direct_pose_hinge_gate_source_model),
        direct_pose_hinge_gate_power=float(direct_pose_hinge_gate_power_model),
        lambda_fusion_enable=bool(lambda_fusion_enable),
        lambda_fusion_mode=str(lambda_fusion_mode),
        lambda_fusion_hidden=int(lambda_fusion_hidden),
        lambda_fusion_dropout=float(lambda_fusion_dropout),
        lambda_fusion_detach_err=True,
        lambda_fusion_logit_init=float(lambda_fusion_logit_init),
        lambda_fusion_use_rollout_step=bool(lambda_fusion_use_rollout_step),
        contact_meas_enable=contact_meas_enable,
        contact_meas_hidden=int(contact_meas_hidden),
        contact_meas_dropout=0.0,
        contact_td_hazard_enable=bool(contact_td_hazard_enable),
        contact_td_hazard_hidden=int(contact_td_hazard_hidden),
        contact_td_hazard_dropout=float(contact_td_hazard_dropout),
    ).to(device)
    validate_and_fix_model_(model, int(ds.Dx), int(ds.Dc))
    # Attach frozen encoder BEFORE loading weights (period_dim/period_encoder may be created here).
    if cfg.encoder_bundle is not None and cfg.encoder_bundle.expanduser().is_file():
        model.attach_motion_encoder(torch.load(str(cfg.encoder_bundle.expanduser()), map_location="cpu"))

    # When training TD hazard as the phase reset source, initialize the head to a low baseline
    # (sum p ≈ 1 per cycle) to avoid unstable over-triggering at step0.
    if (not td_hazard_has_weights) and bool(cfg.train_contact_td_hazard) and str(cfg.phase_reset_source) == "td_hazard":
        head = getattr(model, "contact_td_hazard_head", None)
        try:
            mlp = getattr(head, "mlp", None)
            last = mlp[-1] if isinstance(mlp, torch.nn.Sequential) and len(mlp) > 0 else None
            if isinstance(last, torch.nn.Linear):
                init_p = 1.0 / float(max(1, int(cfg.seq_len)))
                init_p = float(max(1e-4, min(0.25, init_p)))
                init_logit = float(math.log(init_p / max(1e-8, (1.0 - init_p))))
                with torch.no_grad():
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.fill_(init_logit)
                print(
                    f"[posttrain][INFO] init contact_td_hazard_head last bias to logit(p={init_p:.6f})={init_logit:.3f} "
                    f"(seq_len={int(cfg.seq_len)}) for phase_reset_source=td_hazard."
                )
        except Exception:
            pass
    if 'drop_direct_pose_weights' in locals() and bool(drop_direct_pose_weights):
        removed = [
            k
            for k in list(state_dict.keys())
            if str(k).startswith("direct_pose_head.")
            or str(k).startswith("direct_pose_out_leg.")
            or str(k).startswith("direct_pose_out_nonleg.")
            or str(k).startswith("direct_pose_leg_head.")
            or str(k).startswith("direct_pose_leg_head_shared.")
            or str(k).startswith("direct_pose_leg_gate_head.")
            or str(k).startswith("direct_pose_leg_gate_head_shared.")
            or str(k).startswith("direct_pose_leg_side_sign_gate_head.")
            or str(k).startswith("direct_pose_leg_side_embed.")
            or str(k) == "direct_pose_leg_joint_idx_tensor"
            or str(k) in ("direct_pose_leg_side_pos_r_tensor", "direct_pose_leg_side_pos_l_tensor")
            or str(k).startswith("direct_pose_hinge_head.")
            or str(k).startswith("direct_pose_hinge_nonhidden_head.")
            or str(k).startswith("direct_pose_hinge_eps_head.")
            or str(k).startswith("direct_pose_hinge_gate_head.")
            or str(k).startswith("direct_pose_hinge_gate_head_clean.")
        ]
        for k in removed:
            state_dict.pop(k, None)
        if removed:
            print(
                f"[posttrain][INFO] dropped {len(removed)} direct_pose_* hinge tensors from checkpoint (reinit/override)."
            )

    # If we enable phase_z_in conditioning for direct head, older checkpoints will have a smaller
    # input dimension on direct_pose_head.0.weight. Expand it by appending zero columns so the
    # initial behavior is identical (phase weights start at 0) and the head can learn to use phase.
    try:
        if (
            (not bool(locals().get("drop_direct_pose_weights", False)))
            and bool(locals().get("direct_pose_use_phase_z", False))
            and any(k.startswith("direct_pose_head.") for k in state_dict.keys())
        ):
            w0 = state_dict.get("direct_pose_head.0.weight", None)
            w0_exp = model.state_dict().get("direct_pose_head.0.weight", None)
            if torch.is_tensor(w0) and torch.is_tensor(w0_exp) and w0.ndim == 2 and w0_exp.ndim == 2:
                old_in = int(w0.shape[1])
                new_in = int(w0_exp.shape[1])
                phase_dim = int(2 * int(contact_dim))
                if old_in != new_in:
                    phase_mode = str(locals().get("direct_pose_phase_z_mode", "concat") or "concat").strip().lower()
                    if (old_in + phase_dim) == new_in and int(w0.shape[0]) == int(w0_exp.shape[0]):
                        new_w = torch.zeros((int(w0.shape[0]), int(new_in)), device=w0.device, dtype=w0.dtype)
                        new_w[:, :old_in] = w0
                        state_dict["direct_pose_head.0.weight"] = new_w
                        print(
                            f"[posttrain][INFO] expanded direct_pose_head.0.weight in_dim {old_in} -> {new_in} "
                            f"(appended phase_z_in dim={phase_dim} as zeros)."
                        )
                    elif (
                        phase_mode == "replace_contacts"
                        and (old_in == (new_in + phase_dim))
                        and int(w0.shape[0]) == int(w0_exp.shape[0])
                        and int(new_in) >= int(phase_dim)
                    ):
                        # ckpt: [base(+time), plan+meas, phase]  -> model(replace): [base(+time), phase]
                        base_in = int(new_in - phase_dim)
                        new_w = torch.zeros((int(w0.shape[0]), int(new_in)), device=w0.device, dtype=w0.dtype)
                        new_w[:, :base_in] = w0[:, :base_in]
                        new_w[:, base_in:] = w0[:, (old_in - phase_dim) :]
                        state_dict["direct_pose_head.0.weight"] = new_w
                        print(
                            f"[posttrain][INFO] adapted direct_pose_head.0.weight for phase replace: "
                            f"in_dim {old_in} -> {new_in} (dropped plan+meas, kept phase tail dim={phase_dim})."
                        )
                    elif bool(getattr(cfg, "train_direct_pose", False)):
                        # Fallback: drop and reinit if we're going to train direct anyway.
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
    # If hinge bones are overridden relative to the checkpoint, hinge head tensors may either mismatch in shape
    # (K changes) or become semantically wrong (same K but different bone order). In both cases, drop hinge tensors
    # and rely on safe-zero initialization before hinge-only finetune / oracle runs.
    try:
        def _norm_bones(v: Any) -> List[str]:
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                items = [str(x).strip() for x in v]
            else:
                items = [s.strip() for s in str(v).split(",") if s.strip()]
            return [x for x in items if x]

        hinge_prefixes = (
            "direct_pose_hinge_head.",
            "direct_pose_hinge_nonhidden_head.",
            "direct_pose_hinge_eps_head.",
            "direct_pose_hinge_gate_head.",
            "direct_pose_hinge_gate_head_clean.",
        )

        ckpt_bones = []
        if isinstance(ckpt_posttrain_cfg, dict):
            ckpt_bones = _norm_bones(ckpt_posttrain_cfg.get("direct_pose_hinge_bones", None))
        tgt_bones = _norm_bones(direct_pose_hinge_bones_model)

        # Drop on explicit bone mismatch (avoids wrong mapping even when shapes match).
        removed = []
        if bool(direct_pose_hinge_enable_model) and ckpt_bones and tgt_bones and (tgt_bones != ckpt_bones):
            for k in list(state_dict.keys()):
                if any(str(k).startswith(p) for p in hinge_prefixes):
                    removed.append(str(k))
                    state_dict.pop(k, None)
            if removed:
                print(
                    f"[posttrain][INFO] direct_pose_hinge_bones override: ckpt={ckpt_bones} cfg={tgt_bones}; "
                    f"dropped {len(removed)} direct_pose_hinge_* tensors (will re-init hinge heads)."
                )

        # Fallback: drop any hinge tensors whose shapes don't match the instantiated model.
        model_sd = model.state_dict()
        removed_shape = []
        for k in list(state_dict.keys()):
            if not any(str(k).startswith(p) for p in hinge_prefixes):
                continue
            v = state_dict.get(k, None)
            vv = model_sd.get(k, None)
            if torch.is_tensor(v) and torch.is_tensor(vv) and tuple(v.shape) != tuple(vv.shape):
                removed_shape.append(str(k))
                state_dict.pop(k, None)
        if removed_shape:
            print(
                f"[posttrain][INFO] dropped {len(removed_shape)} direct_pose_hinge_* tensors due to shape mismatch "
                "(likely hinge_bones K changed)."
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
    try:
        if bool(model.adapt_legacy_state_dict_(state_dict)):
            print("[posttrain][INFO] migrated legacy direct_pose_head.6 weights to split direct_pose_out_{leg,nonleg}.")
    except Exception:
        pass
    model.load_state_dict(state_dict, strict=False)

    # Warm-start: if the checkpoint has the legacy leg head but not the new shared routed head,
    # initialize the shared head from the legacy weights (helps convergence, avoids cold-start).
    try:
        if bool(getattr(cfg, "direct_pose_leg_side_routing", False)):
            if getattr(model, "direct_pose_leg_head_shared", None) is None:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_side_routing=true but direct_pose_leg_head_shared is not instantiated "
                    "(check leg bones naming/_r/_l symmetry and contact_dim==2)."
                )
            sd_has_shared = any(str(k).startswith("direct_pose_leg_head_shared.") for k in state_dict.keys())
            sd_has_legacy = any(str(k).startswith("direct_pose_leg_head.") for k in state_dict.keys())
            if (not sd_has_shared) and sd_has_legacy:
                ok = _warm_start_direct_pose_leg_head_shared_from_legacy(model)
                if ok:
                    print("[posttrain][INFO] warm-started direct_pose_leg_head_shared from legacy direct_pose_leg_head.")
                else:
                    print("[posttrain][WARN] failed to warm-start direct_pose_leg_head_shared; using safe init (zeros).")
    except SystemExit:
        raise
    except Exception:
        pass

    if cfg.train_so3_corrector:
        if getattr(model, "so3_delta_corrector", None) is None or int(getattr(model, "so3_corr_joint_count", 0) or 0) <= 0:
            raise SystemExit("[FATAL] Model has no SO(3) corrector head (so3_corr_joint_count<=0).")
    if cfg.train_contact_plan_init:
        if not bool(getattr(model, "contact_plan_enable", False)):
            raise SystemExit("[FATAL] contact_plan_enable=false; cannot train contact_plan_init_z.")
        init_z = getattr(model, "contact_plan_init_z", None)
        if init_z is None or (not torch.is_tensor(init_z)):
            raise SystemExit("[FATAL] Model has no contact_plan_init_z parameter to train.")
    if cfg.train_contact_plan:
        if not bool(getattr(model, "contact_plan_enable", False)):
            raise SystemExit("[FATAL] contact_plan_enable=false; cannot train contact_plan.")
        if getattr(model, "contact_plan_cell", None) is None or getattr(model, "contact_plan_head", None) is None:
            raise SystemExit("[FATAL] Model has no contact_plan_cell/head to train.")
    if bool(cfg.train_contact_td_hazard) and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
        if (not bool(getattr(model, "contact_td_hazard_enable", False))) or getattr(model, "contact_td_hazard_head", None) is None:
            raise SystemExit("[FATAL] contact_td_hazard_head is not instantiated/enabled; cannot train contact_td_hazard.")
    if cfg.train_direct_pose:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] direct_pose_head is not instantiated; cannot train direct pose expert.")
        # Sanity: avoid ambiguous "train_only" combinations.
        leg_only = bool(getattr(cfg, "direct_pose_leg_train_only", False))
        leg_gate_only = bool(getattr(cfg, "direct_pose_leg_gate_train_only", False))
        hinge_only = bool(getattr(cfg, "direct_pose_hinge_train_only", False))
        hinge_gate_only = bool(getattr(cfg, "direct_pose_hinge_gate_train_only", False))
        nonleg_only = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
        if (leg_only or leg_gate_only) and (hinge_only or hinge_gate_only):
            raise SystemExit(
                "[FATAL] direct_pose_leg_*_train_only=true is incompatible with direct_pose_hinge_*_train_only. "
                "Pick exactly one train_only mode."
            )
        if nonleg_only and (leg_only or leg_gate_only or hinge_only or hinge_gate_only):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true is incompatible with leg/hinge train_only modes. "
                "Pick exactly one train_only mode."
            )
        if (leg_only or leg_gate_only) and getattr(model, "direct_pose_leg_head", None) is None:
            raise SystemExit(
                "[FATAL] direct_pose_leg_*_train_only=true but no leg head is instantiated. "
                "Enable direct_pose_leg_enable and provide valid direct_pose_leg_bones."
            )
        if nonleg_only and getattr(model, "direct_pose_out_nonleg", None) is None:
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true but no direct_pose_out_nonleg head is instantiated. "
                "Enable direct_pose_split_enable."
            )
        if bool(leg_gate_only):
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_train_only=true but no leg gate/scale head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned'/'scale'/'signed_scale' and enable direct_pose_leg_enable with valid bones."
                )
        if bool(getattr(cfg, "direct_pose_hinge_train_only", False)):
            has_legacy = getattr(model, "direct_pose_hinge_head", None) is not None
            has_clean = (getattr(model, "direct_pose_hinge_nonhidden_head", None) is not None) and (
                getattr(model, "direct_pose_hinge_eps_head", None) is not None
            )
            if not (has_legacy or has_clean):
                raise SystemExit(
                    "[FATAL] direct_pose_hinge_train_only=true but no hinge head is instantiated "
                    "(expected direct_pose_hinge_head or direct_pose_hinge_nonhidden_head+direct_pose_hinge_eps_head). "
                    "Enable direct_pose_hinge_enable and provide valid direct_pose_hinge_bones."
                )
        if bool(getattr(cfg, "direct_pose_hinge_gate_train_only", False)):
            has_gate = (getattr(model, "direct_pose_hinge_gate_head", None) is not None) or (
                getattr(model, "direct_pose_hinge_gate_head_clean", None) is not None
            )
            if not has_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_hinge_gate_train_only=true but no learned gate head is instantiated. "
                    "Set direct_pose_hinge_gate_mode='learned' and enable direct_pose_hinge_enable with valid bones."
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
        if float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0) > 0.0:
            leg_mode = str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").strip().lower()
            if leg_mode != "so3":
                raise SystemExit(
                    "[FATAL] direct_pose_leg_scale_sup_weight>0 requires direct_pose_leg_mode='so3' "
                    f"(got {leg_mode!r})."
                )
            has_leg_scale = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_scale:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_scale_sup_weight>0 but no leg scale head is instantiated. "
                    "Set direct_pose_leg_gate_mode='scale' or 'signed_scale' and enable direct_pose_leg_enable with valid bones."
                )
    if cfg.train_lambda_head:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs direct_pose_head (out_direct), but checkpoint/model does not enable it.")
        if getattr(model, "lambda_fusion_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs lambda_fusion_head, but it is not instantiated.")

    # Optional ablations: decouple direct head from (potentially biased) contacts_meas during training.
    if bool(getattr(cfg, "direct_pose_meas_force_zero", False)):
        try:
            setattr(model, "direct_pose_meas_force_zero", True)
            print("[posttrain] direct_pose_meas_force_zero=true (direct ignores contacts_meas)")
        except Exception:
            pass
    if bool(getattr(cfg, "direct_pose_meas_detach", False)):
        try:
            setattr(model, "direct_pose_meas_detach", True)
            print("[posttrain] direct_pose_meas_detach=true (stop-grad direct->contacts_meas)")
        except Exception:
            pass

    # Optional: reset gate logit after loading ckpt (helps avoid near-zero gates that throttle gradients).
    if cfg.so3_corr_gate_logit_reset is not None:
        logit = getattr(model, "so3_corr_gate_logit", None)
        if torch.is_tensor(logit):
            with torch.no_grad():
                logit.fill_(float(cfg.so3_corr_gate_logit_reset))
            print(f"[posttrain] reset so3_corr_gate_logit={float(cfg.so3_corr_gate_logit_reset):.4f}")

    loss_fn = MotionJointLoss(
        output_layout=getattr(ds, "output_layout", None),
        fps=float(getattr(ds, "fps", 60.0) or 60.0),
        rot6d_spec=getattr(ds, "rot6d_spec", None) or {},
        meta=getattr(ds, "meta", None) or {},
    )
    # Respect cfg.lr / cfg.weight_decay; posttrain often needs tuned LR for small heads.
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=float(getattr(cfg, "lr", 1e-4) or 1e-4),
        grad_clip=0.0,
        weight_decay=float(getattr(cfg, "weight_decay", 0.0) or 0.0),
        tf_warmup_steps=0,
        tf_total_steps=0,
        augmentor=None,
        use_amp=False,
        accum_steps=1,
        pin_memory=False,
    )
    try:
        hinge_idx = getattr(model, "direct_pose_hinge_joint_idx", None)
        if hinge_idx:
            trainer.direct_pose_hinge_joint_idx = list(hinge_idx)
            trainer.direct_pose_hinge_axis = str(getattr(model, "direct_pose_hinge_axis", "Z") or "Z")
            trainer.direct_pose_hinge_max_rad = getattr(model, "direct_pose_hinge_max_rad", None)
            loss_fn.direct_pose_hinge_joint_idx = list(hinge_idx)
            loss_fn.direct_pose_hinge_axis = str(getattr(model, "direct_pose_hinge_axis", "Z") or "Z")
            loss_fn.direct_pose_hinge_max_rad = getattr(model, "direct_pose_hinge_max_rad", None)
    except Exception:
        pass
    try:
        trainer.fps = float(getattr(ds, "fps", 60.0) or 60.0)
        trainer.bone_hz = float(trainer.fps)
    except Exception:
        pass

    # ---- Inject dataset-derived slices & normalizer (avoid relying on LayoutCenter meta) ----
    Dx = int(ds.Dx)
    Dy = int(ds.Dy)
    x_layout = getattr(ds, "state_layout", None) or {}
    y_layout = getattr(ds, "output_layout", None) or {}
    trainer._x_layout = dict(x_layout)
    trainer._y_layout = dict(y_layout)

    trainer.rootvel_slice = parse_layout_entry(y_layout.get("RootVelocity"), "RootVelocity", Dy)
    trainer.angvel_slice = parse_layout_entry(y_layout.get("BoneAngularVelocities"), "BoneAngularVelocities", Dy)

    trainer.rootvel_x_slice = parse_layout_entry(x_layout.get("RootVelocity"), "RootVelocity", Dx)
    trainer.angvel_x_slice = parse_layout_entry(x_layout.get("BoneAngularVelocities"), "BoneAngularVelocities", Dx)
    trainer.rootpos_x_slice = parse_layout_entry(x_layout.get("RootPosition"), "RootPosition", Dx)

    trainer.rot6d_x_slice = parse_layout_entry(x_layout.get("BoneRotations6D"), "BoneRotations6D", Dx)
    trainer.rot6d_y_slice = parse_layout_entry(y_layout.get("BoneRotations6D"), "BoneRotations6D", Dy)

    # y->x map (for state carry), derived from common groups
    y_to_x_map = []
    common = sorted(set(x_layout.keys()) & set(y_layout.keys()))
    for name in common:
        xs = parse_layout_entry(x_layout.get(name), name, Dx)
        ys = parse_layout_entry(y_layout.get(name), name, Dy)
        if not (isinstance(xs, slice) and isinstance(ys, slice)):
            continue
        xk = int(xs.stop - xs.start)
        yk = int(ys.stop - ys.start)
        k = min(xk, yk)
        if k <= 0:
            continue
        y_to_x_map.append({"name": str(name), "x_start": int(xs.start), "x_size": k, "y_start": int(ys.start), "y_size": k})
    trainer.y_to_x_map = y_to_x_map

    mu_x = np.asarray(norm_spec.get("MuX"), dtype=np.float32)
    std_x = np.asarray(norm_spec.get("StdX"), dtype=np.float32)
    mu_y = np.asarray(norm_spec.get("MuY"), dtype=np.float32)
    std_y = np.asarray(norm_spec.get("StdY"), dtype=np.float32)
    # Make denormalization available to geodesic losses (rot6d must be computed in raw space).
    try:
        setattr(loss_fn, "mu_y", mu_y)
        setattr(loss_fn, "std_y", std_y)
        setattr(trainer, "mu_y", mu_y)
        setattr(trainer, "std_y", std_y)
    except Exception:
        pass
    trainer.normalizer = DataNormalizer(
        mu_x=mu_x,
        std_x=std_x,
        mu_y=mu_y,
        std_y=std_y,
        y_to_x_map=y_to_x_map,
        rootvel_x_slice=trainer.rootvel_x_slice,
        rootvel_y_slice=trainer.rootvel_slice,
        angvel_x_slice=trainer.angvel_x_slice,
        angvel_y_slice=trainer.angvel_slice,
        tanh_scales_rootvel=norm_spec.get("tanh_scales_rootvel", None),
        tanh_scales_angvel=norm_spec.get("tanh_scales_angvel", None),
        angvel_mode=getattr(ds, "angvel_norm_mode", None),
        angvel_mu=getattr(ds, "angvel_mu", None),
        angvel_std=getattr(ds, "angvel_std", None),
    )
    trainer.pose_hist_len = int(getattr(ds, "pose_hist_len", 0) or 0)
    trainer.pose_hist_dim = int(getattr(ds, "pose_hist_dim", 0) or 0)
    # ---- White-box contacts_meas knobs (P2 ground_z stability / ablations) ----
    gate_raw = str(getattr(cfg, "contact_meas_gate_by_hit", "auto") or "auto").strip().lower()
    if gate_raw in ("true", "1", "yes", "y"):
        trainer.contact_meas_gate_by_hit_override = True
    elif gate_raw in ("false", "0", "no", "n"):
        trainer.contact_meas_gate_by_hit_override = False
    else:
        trainer.contact_meas_gate_by_hit_override = None
    trainer.contact_meas_vxy_mode = str(getattr(cfg, "contact_meas_vxy_mode", "abs") or "abs").strip().lower()
    trainer.contact_meas_ground_z_mode = str(getattr(cfg, "contact_meas_ground_z_mode", "window") or "window").strip().lower()
    trainer.contact_meas_ground_z_beta = float(getattr(cfg, "contact_meas_ground_z_beta", 0.05) or 0.05)
    trainer.contact_meas_ground_z_window = int(getattr(cfg, "contact_meas_ground_z_window", 5) or 5)
    trainer.contact_meas_ground_z_quantile = float(getattr(cfg, "contact_meas_ground_z_quantile", 0.2) or 0.2)
    try:
        up_cm = float(getattr(cfg, "contact_meas_ground_z_slew_up_cm", 0.0) or 0.0)
    except Exception:
        up_cm = 0.0
    try:
        down_cm = float(getattr(cfg, "contact_meas_ground_z_slew_down_cm", 0.0) or 0.0)
    except Exception:
        down_cm = 0.0
    trainer.contact_meas_ground_z_max_up_m = max(0.0, up_cm) / 100.0
    trainer.contact_meas_ground_z_max_down_m = max(0.0, down_cm) / 100.0
    # Stage2: deterministic reliability r_t applied to λ for on-manifold blend (shared with freerun).
    trainer.lambda_reliability_mode = str(cfg.lambda_reliability_mode or "none")
    trainer.lambda_reliability_warmup_steps = int(cfg.lambda_reliability_warmup_steps or 0)
    trainer.lambda_reliability_contact_err_max = float(cfg.lambda_reliability_contact_err_max or 1.0)
    trainer.lambda_reliability_warmup_joint_scales = cfg.lambda_reliability_warmup_joint_scales

    selected = (
        int(bool(cfg.train_so3_corrector))
        + int(bool(cfg.train_contact_plan_init))
        + int(bool(cfg.train_contact_plan))
        + int(bool(cfg.train_direct_pose))
        + int(bool(cfg.train_lambda_head))
    )
    if selected != 1:
        # Allow lightweight calibration modes (no so3/plan/lambda heads updated).
        if selected == 0:
            calib_selected = (
                int(bool(cfg.train_contact_meas))
                + int(bool(cfg.train_contact_td_hazard))
            )
            if calib_selected != 1:
                raise SystemExit(
                    "[FATAL] Choose exactly one: train_contact_meas | train_contact_td_hazard "
                    "(when no other target is selected)."
                )
            if bool(cfg.train_contact_meas) and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
                print("[posttrain] mode=train_contact_meas_only (no so3/plan/lambda heads updated)")
            elif bool(cfg.train_contact_td_hazard) and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
                print("[posttrain] mode=train_contact_td_hazard_only (no so3/plan/lambda heads updated)")
        else:
            raise SystemExit(
                "[FATAL] Choose exactly one: train_so3_corrector | train_contact_plan_init | train_contact_plan | train_direct_pose | train_lambda_head."
            )
    if cfg.train_contact_plan_init and cfg.train_contact_meas:
        print("[posttrain][WARN] train_contact_meas=true is ignored for train_contact_plan_init mode.")
    if cfg.train_contact_plan and cfg.train_contact_meas:
        print("[posttrain][WARN] train_contact_meas=true is ignored for train_contact_plan mode.")
    if cfg.train_contact_plan_init and bool(cfg.train_contact_td_hazard):
        print("[posttrain][WARN] train_contact_td_hazard=true is ignored for train_contact_plan_init mode.")
    if cfg.train_contact_plan and bool(cfg.train_contact_td_hazard):
        print("[posttrain][WARN] train_contact_td_hazard=true is ignored for train_contact_plan mode.")

    _freeze_all(model)
    if cfg.train_so3_corrector:
        _unfreeze_so3_corrector(model)
    if cfg.train_contact_plan_init:
        _unfreeze_contact_plan_init(model)
    if cfg.train_contact_plan:
        _unfreeze_contact_plan(model)
    if cfg.train_direct_pose:
        _unfreeze_direct_pose(
            model,
            hinge_only=bool(getattr(cfg, "direct_pose_hinge_train_only", False)),
            gate_only=bool(getattr(cfg, "direct_pose_hinge_gate_train_only", False)),
            leg_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
            leg_gate_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
            nonleg_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
        )
    if cfg.train_lambda_head:
        _unfreeze_lambda_fusion(model)
    if cfg.train_contact_meas and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
        _unfreeze_contact_meas(model)
    if bool(cfg.train_contact_td_hazard) and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
        _unfreeze_contact_td_hazard(model)
    model.train()

    params, names = _select_trainable_params(model)
    if not params:
        raise SystemExit("[FATAL] No trainable parameters selected for post-train.")
    print(f"[posttrain] trainable={len(params)} params: {', '.join(names[:8])}{' ...' if len(names)>8 else ''}")
    expected_prefixes: list[str] = []
    if cfg.train_so3_corrector:
        expected_prefixes.extend(["so3_delta_corrector", "so3_corr_gate_logit"])
    if cfg.train_contact_plan_init:
        expected_prefixes.extend(["contact_plan_init_z", "contact_plan_init_head"])
    if cfg.train_contact_plan:
        expected_prefixes.extend(
            [
                "contact_plan_cell",
                "contact_plan_head",
                "contact_plan_time_head",
                "contact_plan_init_z",
                "contact_plan_init_head",
            ]
        )
    if cfg.train_lambda_head:
        expected_prefixes.append("lambda_fusion_head")
    if cfg.train_direct_pose:
        expected_prefixes.append("direct_pose_head")
        expected_prefixes.append("direct_pose_out_leg")
        expected_prefixes.append("direct_pose_out_nonleg")
        expected_prefixes.append("direct_pose_nonleg_proj")
        expected_prefixes.append("direct_pose_leg_head")
        expected_prefixes.append("direct_pose_leg_head_shared")
        expected_prefixes.append("direct_pose_leg_gate_head")
        expected_prefixes.append("direct_pose_leg_gate_head_shared")
        expected_prefixes.append("direct_pose_leg_side_sign_gate_head")
        expected_prefixes.append("direct_pose_leg_side_embed")
        expected_prefixes.append("direct_pose_hinge_head")
        expected_prefixes.append("direct_pose_hinge_nonhidden_head")
        expected_prefixes.append("direct_pose_hinge_eps_head")
        expected_prefixes.append("direct_pose_hinge_gate_head")
        expected_prefixes.append("direct_pose_hinge_gate_head_clean")
    if cfg.train_contact_meas and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
        expected_prefixes.append("contact_meas_head")
    if bool(cfg.train_contact_td_hazard) and (not cfg.train_contact_plan_init) and (not cfg.train_contact_plan):
        expected_prefixes.append("contact_td_hazard_head")
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

    # Optional: smaller LR for eps(hidden) hinge branch in clean split.
    eps_lr_scale = float(getattr(cfg, "direct_pose_hinge_eps_lr_scale", 1.0) or 1.0)
    if (not math.isfinite(eps_lr_scale)) or eps_lr_scale <= 0.0:
        eps_lr_scale = 1.0
    eps_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    if abs(eps_lr_scale - 1.0) > 1e-12:
        for n, p in zip(names, params):
            if str(n).startswith("direct_pose_hinge_eps_head."):
                eps_params.append(p)
            else:
                base_params.append(p)
    if eps_params and base_params and abs(eps_lr_scale - 1.0) > 1e-12:
        print(f"[posttrain] direct_pose_hinge_eps_lr_scale={eps_lr_scale:g} (eps_params={len(eps_params)})")
        opt = torch.optim.AdamW(
            [
                {"params": base_params, "lr": float(cfg.lr), "weight_decay": float(cfg.weight_decay)},
                {"params": eps_params, "lr": float(cfg.lr) * float(eps_lr_scale), "weight_decay": float(cfg.weight_decay)},
            ],
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    else:
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

    # Optional: offline alpha-table supervision for the leg scale head.
    # We precompute per-sic target vectors in log space (clamped in the model).
    direct_leg_scale_sup_table: Optional[Dict[str, Any]] = None
    try:
        scale_sup_w = float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0)
    except Exception:
        scale_sup_w = 0.0
    if scale_sup_w > 0.0:
        table_path = str(getattr(cfg, "direct_pose_leg_scale_sup_alpha_table_json", "") or "").strip()
        if not table_path:
            raise SystemExit(
                "[FATAL] direct_pose_leg_scale_sup_weight>0 but direct_pose_leg_scale_sup_alpha_table_json is empty."
            )
        p_table = Path(table_path).expanduser()
        if not p_table.is_file():
            raise SystemExit(f"[FATAL] direct_pose_leg_scale_sup_alpha_table_json not found: {p_table}")
        try:
            eps = float(getattr(cfg, "direct_pose_leg_scale_sup_log_eps", 0.01) or 0.01)
        except Exception:
            eps = 0.01
        if (not math.isfinite(eps)) or eps <= 0.0:
            eps = 0.01

        obj = json.loads(p_table.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise SystemExit(
                "[FATAL] direct_pose_leg_scale_sup_alpha_table_json: unsupported schema. "
                "Expected dict with key 'alpha_by_sic_bone'."
            )
        # Reject phase-bin migrated schemas (we keep sic-only tables for this supervision path).
        keys_norm = {str(k).replace("_", "").replace("-", "").strip().lower() for k in obj.keys()}
        if any(("phasebin" in k) or ("phaseanchor" in k) for k in keys_norm) or (
            ("bins" in keys_norm) and ("alpha_by_sic_bone" not in obj)
        ):
            raise SystemExit(
                "[FATAL] direct_pose_leg_scale_sup_alpha_table_json: phase-bin table schema is no longer supported. "
                "Use a sic-keyed table via 'alpha_by_sic_bone'."
            )
        alpha_by_key_bone = obj.get("alpha_by_sic_bone", None)
        if not isinstance(alpha_by_key_bone, dict):
            raise SystemExit(
                "[FATAL] direct_pose_leg_scale_sup_alpha_table_json: unsupported schema. "
                "Expected dict with key 'alpha_by_sic_bone'."
            )

        mask = obj.get("mask", None)
        cycle_gte = 1
        drop_wrap = True
        if isinstance(mask, dict):
            try:
                cycle_gte = int(mask.get("cycle_gte", cycle_gte) or cycle_gte)
            except Exception:
                cycle_gte = 1
            try:
                drop_wrap = bool(mask.get("drop_wrap", drop_wrap))
            except Exception:
                drop_wrap = True
        cycle_gte = max(0, int(cycle_gte))

        try:
            leg_names = list(getattr(model, "direct_pose_leg_joint_names", None) or [])
        except Exception:
            leg_names = []
        if not leg_names:
            raise SystemExit(
                "[FATAL] direct_pose_leg_scale_sup_weight>0 but model has no direct_pose_leg_joint_names (leg head disabled?)."
            )
        name_to_i = {str(n): int(i) for i, n in enumerate(leg_names)}
        k_is_right = [str(n).strip().lower().endswith(("_r", "right")) for n in leg_names]
        K = int(len(leg_names))

        # If the model is in signed_scale mode, interpret alpha values as signed:
        #   alpha = 0   => sign_target=0 (y=0.5), mag_target=1 (log=0)
        #   alpha < 0   => sign_target=-1 (y=0),  mag_target=|alpha|
        #   alpha > 0   => sign_target=+1 (y=1),  mag_target=|alpha|
        try:
            gm_leg = str(getattr(model, "direct_pose_leg_gate_mode", "none") or "none").strip().lower()
        except Exception:
            gm_leg = "none"
        signed = bool(gm_leg == "signed_scale")
        try:
            clip = float(getattr(model, "direct_pose_leg_scale_log_clip", 4.0) or 4.0)
        except Exception:
            clip = 4.0
        if (not math.isfinite(clip)) or clip <= 0.0:
            clip = 4.0

        if signed:
            tgt_log_default = [0.0 for _ in range(K)]          # log_mag target (mag=1)
            tgt_sign01_default = [1.0 for _ in range(K)]       # y in [0,1] for BCEWithLogits; +1 => y=1
            tgt_log_by_sic: Dict[int, List[float]] = {}
            tgt_sign01_by_sic: Dict[int, List[float]] = {}
            for key_k, bone_map in alpha_by_key_bone.items():
                try:
                    key_i = int(key_k)
                except Exception:
                    continue
                if not isinstance(bone_map, dict):
                    continue
                vec_log = list(tgt_log_default)
                vec_sign = list(tgt_sign01_default)
                for bone, a0 in bone_map.items():
                    b = str(bone)
                    if b not in name_to_i:
                        continue
                    try:
                        a = float(a0)
                    except Exception:
                        continue
                    if not math.isfinite(a):
                        continue
                    ii = name_to_i[b]
                    if abs(float(a)) <= 1e-12:
                        # Off: sign=0, mag=1.
                        vec_sign[ii] = 0.5
                        vec_log[ii] = 0.0
                    else:
                        vec_sign[ii] = 1.0 if float(a) > 0.0 else 0.0
                        # Target is clamped to match model's soft-clip range.
                        vec_log[ii] = float(max(-float(clip), min(float(clip), math.log(abs(float(a))))))
                tgt_log_by_sic[int(key_i)] = vec_log
                tgt_sign01_by_sic[int(key_i)] = vec_sign
            direct_leg_scale_sup_table = {
                "mode": "signed_scale",
                "coord": "sic",
                "path": str(p_table.resolve()),
                "mask": {"cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
                "cycle_gte": int(cycle_gte),
                "drop_wrap": bool(drop_wrap),
                "eps": float(eps),  # kept for provenance; not used for signed_scale targets
                "clip": float(clip),
                "tgt_log_default": tgt_log_default,
                "tgt_log_by_sic": tgt_log_by_sic,
                "tgt_sign01_default": tgt_sign01_default,
                "tgt_sign01_by_sic": tgt_sign01_by_sic,
                "k_is_right": k_is_right,
                "bones": [str(n) for n in leg_names],
            }
            print(
                f"[posttrain] direct_pose_leg_scale_sup_weight={scale_sup_w:g} table={p_table} "
                f"(mode=signed_scale coord=sic keys={len(alpha_by_key_bone)}) clip={clip:g} "
                f"mask: cycle>={cycle_gte} drop_wrap={drop_wrap}"
            )
        else:
            default_log = float(math.log(1.0 + float(eps)))
            tgt_default = [default_log for _ in range(K)]
            tgt_by_sic: Dict[int, List[float]] = {}
            for key_k, bone_map in alpha_by_key_bone.items():
                try:
                    key_i = int(key_k)
                except Exception:
                    continue
                if not isinstance(bone_map, dict):
                    continue
                vec = list(tgt_default)
                for bone, a0 in bone_map.items():
                    b = str(bone)
                    if b not in name_to_i:
                        continue
                    try:
                        a = float(a0)
                    except Exception:
                        continue
                    if not math.isfinite(a):
                        continue
                    a = max(0.0, float(a))  # mag-only legacy mode; sign handled separately
                    vec[name_to_i[b]] = float(math.log(a + float(eps)))
                tgt_by_sic[int(key_i)] = vec
            direct_leg_scale_sup_table = {
                "mode": "scale",
                "coord": "sic",
                "path": str(p_table.resolve()),
                "mask": {"cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
                "cycle_gte": int(cycle_gte),
                "drop_wrap": bool(drop_wrap),
                "eps": float(eps),
                "clip": float(clip),
                "tgt_log_default": tgt_default,
                "tgt_log_by_sic": tgt_by_sic,
                "k_is_right": k_is_right,
                "bones": [str(n) for n in leg_names],
            }
            print(
                f"[posttrain] direct_pose_leg_scale_sup_weight={scale_sup_w:g} table={p_table} "
                f"(mode=scale coord=sic keys={len(alpha_by_key_bone)}) eps={eps:g} clip={clip:g} "
                f"mask: cycle>={cycle_gte} drop_wrap={drop_wrap}"
            )

    # Optional: binary hotspot weighting table for direct pose loss (no scale-target supervision).
    direct_pose_pair_boost_table: Optional[Dict[str, Any]] = None
    pair_table_path = str(getattr(cfg, "direct_pose_loss_pair_boost_table_json", "") or "").strip()
    if pair_table_path:
        p_table = Path(pair_table_path).expanduser()
        if not p_table.is_file():
            raise SystemExit(f"[FATAL] direct_pose_loss_pair_boost_table_json not found: {p_table}")
        try:
            pair_boost = float(getattr(cfg, "direct_pose_loss_pair_boost", 1.0) or 1.0)
        except Exception:
            pair_boost = 1.0
        if (not math.isfinite(pair_boost)) or pair_boost <= 1.0:
            pair_boost = 1.0

        obj = json.loads(p_table.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise SystemExit(
                "[FATAL] direct_pose_loss_pair_boost_table_json: unsupported schema. "
                "Expected dict with key 'alpha_by_sic_bone'."
            )
        alpha_by_key_bone = obj.get("alpha_by_sic_bone", None)
        if not isinstance(alpha_by_key_bone, dict):
            raise SystemExit(
                "[FATAL] direct_pose_loss_pair_boost_table_json: unsupported schema. "
                "Expected dict with key 'alpha_by_sic_bone'."
            )

        mask = obj.get("mask", None)
        cycle_gte = 1
        drop_wrap = True
        if isinstance(mask, dict):
            try:
                cycle_gte = int(mask.get("cycle_gte", cycle_gte) or cycle_gte)
            except Exception:
                cycle_gte = 1
            try:
                drop_wrap = bool(mask.get("drop_wrap", drop_wrap))
            except Exception:
                drop_wrap = True
        cycle_gte = max(0, int(cycle_gte))

        bone_names_src = getattr(ds, "bone_names", None)
        if not bone_names_src:
            bone_names_src = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
        if not bone_names_src:
            meta = getattr(getattr(trainer, "loss_fn", None), "meta", None)
            if isinstance(meta, dict):
                bone_names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
        bone_names = [str(b) for b in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []
        if not bone_names:
            raise SystemExit(
                "[FATAL] direct_pose_loss_pair_boost_table_json is set but failed to resolve bone_names for mapping."
            )
        name_to_idx: Dict[str, int] = {}
        for i, nm in enumerate(bone_names):
            s = str(nm)
            name_to_idx.setdefault(s, int(i))
            name_to_idx.setdefault(s.lower(), int(i))

        joint_idx_by_sic: Dict[int, List[int]] = {}
        pair_count = 0
        missing_bones: Dict[str, int] = {}
        for key_k, bone_map in alpha_by_key_bone.items():
            try:
                key_i = int(key_k)
            except Exception:
                continue
            if not isinstance(bone_map, dict):
                continue
            idxs: List[int] = []
            for bone, a0 in bone_map.items():
                try:
                    a = float(a0)
                except Exception:
                    continue
                if not math.isfinite(a):
                    continue
                # Binary mask: keep only non-neutral entries (alpha != 1).
                if abs(float(a) - 1.0) <= 1e-12:
                    continue
                b = str(bone)
                j = name_to_idx.get(b, name_to_idx.get(b.lower(), None))
                if j is None:
                    missing_bones[b] = int(missing_bones.get(b, 0)) + 1
                    continue
                jj = int(j)
                if jj not in idxs:
                    idxs.append(jj)
                    pair_count += 1
            if idxs:
                joint_idx_by_sic[int(key_i)] = idxs

        direct_pose_pair_boost_table = {
            "path": str(p_table.resolve()),
            "boost": float(pair_boost),
            "cycle_gte": int(cycle_gte),
            "drop_wrap": bool(drop_wrap),
            "joint_idx_by_sic": joint_idx_by_sic,
            "num_pairs": int(pair_count),
            "num_sics": int(len(joint_idx_by_sic)),
            "num_missing_bones": int(sum(missing_bones.values())),
        }
        msg = (
            f"[posttrain] direct_pose_loss_pair_boost_table={p_table} boost={pair_boost:g} "
            f"pairs={pair_count} sics={len(joint_idx_by_sic)} mask: cycle>={cycle_gte} drop_wrap={drop_wrap}"
        )
        if missing_bones:
            miss_preview = ", ".join(sorted(missing_bones.keys())[:4])
            msg += f" (unmatched bones: {len(missing_bones)}; e.g. {miss_preview})"
        print(msg)

    log_rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_loss = 0.0
        ok_steps = 0
        bad_steps = 0
        for it in range(int(cfg.steps_per_epoch)):
            batch = next(batch_iter)
            opt.zero_grad(set_to_none=True)
            gate_mode = "lambda"
            if cfg.train_so3_corrector:
                gate_force = cfg.so3_corr_gate_force
                gate_mode = "learned" if gate_force is None else "forced"
                if int(cfg.gate_warmup_steps or 0) > 0 and cfg.gate_warmup_value is not None and global_step < int(cfg.gate_warmup_steps):
                    gate_force = float(cfg.gate_warmup_value)
                    gate_mode = "warmup"
                loss, stats = _corr_loss_rollout(
                    trainer,
                    model,
                    batch,
                    columns=columns,
                    gate_force=gate_force,
                    max_deg=cfg.so3_corr_max_deg,
                    omega_l2_weight=cfg.so3_corr_omega_l2_weight,
                    rollout_steps=cfg.rollout_steps,
                    rollout_cycles=cfg.rollout_cycles,
                    time_index_mode=cfg.time_index_mode,
                    time_weight_max=cfg.corr_time_weight_max,
                    detach_rollout_state=cfg.detach_rollout_state,
                    contact_meas_weight=cfg.contact_meas_weight,
                )
            elif cfg.train_contact_plan_init:
                gate_mode = "plan_init"
                loss, stats = _contact_plan_init_loss_teacher(
                    trainer,
                    model,
                    batch,
                    time_index_mode=cfg.time_index_mode,
                    weight=cfg.contact_plan_init_weight,
                )
            elif cfg.train_contact_plan:
                gate_mode = "plan"
                loss, stats = _contact_plan_loss_teacher(
                    trainer,
                    model,
                    batch,
                    time_index_mode=cfg.time_index_mode,
                    weight=cfg.contact_plan_weight,
                )
            elif (not cfg.train_lambda_head) and (not cfg.train_direct_pose) and bool(cfg.train_contact_meas):
                gate_mode = "meas"
                smooth_w = float(getattr(cfg, "contact_meas_smooth_weight", 0.0) or 0.0)
                smooth_kind = str(getattr(cfg, "contact_meas_smooth_kind", "l1") or "l1")
                margin_w = float(getattr(cfg, "contact_meas_margin_weight", 0.0) or 0.0)
                margin_logit = float(getattr(cfg, "contact_meas_margin_logit", 0.0) or 0.0)
                band = float(getattr(cfg, "contact_meas_transition_band", 0.0) or 0.0)
                rollout_w = float(getattr(cfg, "contact_meas_rollout_weight", 0.0) or 0.0)

                if rollout_w > 0.0:
                    # Mixed: teacher (main) + short closed-loop rollout (OOD exposure).
                    loss_t, stats_t = _contact_meas_loss_teacher(
                        trainer,
                        model,
                        batch,
                        time_index_mode=cfg.time_index_mode,
                        weight=cfg.contact_meas_weight,
                        smooth_weight=smooth_w,
                        smooth_kind=smooth_kind,
                        margin_weight=margin_w,
                        margin_logit=margin_logit,
                        transition_band=band,
                    )
                    loss_r, stats_r = _contact_meas_loss_rollout(
                        trainer,
                        model,
                        batch,
                        rollout_steps=cfg.rollout_steps,
                        rollout_cycles=cfg.rollout_cycles,
                        include_boundary=cfg.rollout_include_boundary,
                        boundary_weight=cfg.lambda_boundary_weight,
                        random_offset=cfg.rollout_random_offset,
                        time_index_mode=cfg.time_index_mode,
                        time_weight_max=cfg.corr_time_weight_max,
                        detach_rollout_state=cfg.detach_rollout_state,
                        weight=rollout_w,
                        smooth_weight=smooth_w,
                        smooth_kind=smooth_kind,
                        margin_weight=margin_w,
                        margin_logit=margin_logit,
                        transition_band=band,
                    )
                    loss = loss_t + loss_r
                    stats = {"total": float(loss.detach().cpu())}
                    for k, v in stats_t.items():
                        stats[f"teacher_{k}"] = v
                    for k, v in stats_r.items():
                        stats[f"rollout_{k}"] = v
                elif bool(getattr(cfg, "contact_meas_rollout", False)):
                    loss, stats = _contact_meas_loss_rollout(
                        trainer,
                        model,
                        batch,
                        rollout_steps=cfg.rollout_steps,
                        rollout_cycles=cfg.rollout_cycles,
                        include_boundary=cfg.rollout_include_boundary,
                        boundary_weight=cfg.lambda_boundary_weight,
                        random_offset=cfg.rollout_random_offset,
                        time_index_mode=cfg.time_index_mode,
                        time_weight_max=cfg.corr_time_weight_max,
                        detach_rollout_state=cfg.detach_rollout_state,
                        weight=cfg.contact_meas_weight,
                        smooth_weight=smooth_w,
                        smooth_kind=smooth_kind,
                        margin_weight=margin_w,
                        margin_logit=margin_logit,
                        transition_band=band,
                    )
                else:
                    loss, stats = _contact_meas_loss_teacher(
                        trainer,
                        model,
                        batch,
                        time_index_mode=cfg.time_index_mode,
                        weight=cfg.contact_meas_weight,
                        smooth_weight=smooth_w,
                        smooth_kind=smooth_kind,
                        margin_weight=margin_w,
                        margin_logit=margin_logit,
                        transition_band=band,
                    )
            elif (not cfg.train_lambda_head) and (not cfg.train_direct_pose) and bool(cfg.train_contact_td_hazard):
                gate_mode = "td_hazard"
                rollout_w = float(getattr(cfg, "contact_td_hazard_rollout_weight", 0.0) or 0.0)

                if rollout_w > 0.0:
                    # Mixed: teacher (main) + short closed-loop rollout (OOD exposure).
                    loss_t, stats_t = _contact_td_hazard_loss_teacher(
                        trainer,
                        model,
                        batch,
                        time_index_mode=cfg.time_index_mode,
                        bce_weight=cfg.contact_td_hazard_bce_weight,
                        event_weight=cfg.contact_td_hazard_event_weight,
                        mass_weight=cfg.contact_td_hazard_mass_weight,
                        unimodal_weight=cfg.contact_td_hazard_unimodal_weight,
                        entropy_weight=cfg.contact_td_hazard_entropy_weight,
                        clock_weight=cfg.contact_td_hazard_clock_weight,
                    )
                    loss_r, stats_r = _contact_td_hazard_loss_rollout(
                        trainer,
                        model,
                        batch,
                        rollout_steps=cfg.rollout_steps,
                        rollout_cycles=cfg.rollout_cycles,
                        include_boundary=cfg.rollout_include_boundary,
                        boundary_weight=cfg.lambda_boundary_weight,
                        random_offset=cfg.rollout_random_offset,
                        time_index_mode=cfg.time_index_mode,
                        time_weight_max=cfg.corr_time_weight_max,
                        detach_rollout_state=cfg.detach_rollout_state,
                        bce_weight=cfg.contact_td_hazard_bce_weight,
                        event_weight=cfg.contact_td_hazard_event_weight,
                        mass_weight=cfg.contact_td_hazard_mass_weight,
                        unimodal_weight=cfg.contact_td_hazard_unimodal_weight,
                        entropy_weight=cfg.contact_td_hazard_entropy_weight,
                        clock_weight=cfg.contact_td_hazard_clock_weight,
                        weight=rollout_w,
                    )
                    loss = loss_t + loss_r
                    stats = {"total": float(loss.detach().cpu())}
                    for k, v in stats_t.items():
                        stats[f"teacher_{k}"] = v
                    for k, v in stats_r.items():
                        stats[f"rollout_{k}"] = v
                elif bool(getattr(cfg, "contact_td_hazard_rollout", False)):
                    loss, stats = _contact_td_hazard_loss_rollout(
                        trainer,
                        model,
                        batch,
                        rollout_steps=cfg.rollout_steps,
                        rollout_cycles=cfg.rollout_cycles,
                        include_boundary=cfg.rollout_include_boundary,
                        boundary_weight=cfg.lambda_boundary_weight,
                        random_offset=cfg.rollout_random_offset,
                        time_index_mode=cfg.time_index_mode,
                        time_weight_max=cfg.corr_time_weight_max,
                        detach_rollout_state=cfg.detach_rollout_state,
                        bce_weight=cfg.contact_td_hazard_bce_weight,
                        event_weight=cfg.contact_td_hazard_event_weight,
                        mass_weight=cfg.contact_td_hazard_mass_weight,
                        unimodal_weight=cfg.contact_td_hazard_unimodal_weight,
                        entropy_weight=cfg.contact_td_hazard_entropy_weight,
                        clock_weight=cfg.contact_td_hazard_clock_weight,
                        weight=1.0,
                    )
                else:
                    loss, stats = _contact_td_hazard_loss_teacher(
                        trainer,
                        model,
                        batch,
                        time_index_mode=cfg.time_index_mode,
                        bce_weight=cfg.contact_td_hazard_bce_weight,
                        event_weight=cfg.contact_td_hazard_event_weight,
                        mass_weight=cfg.contact_td_hazard_mass_weight,
                        unimodal_weight=cfg.contact_td_hazard_unimodal_weight,
                        entropy_weight=cfg.contact_td_hazard_entropy_weight,
                        clock_weight=cfg.contact_td_hazard_clock_weight,
                    )
            elif cfg.train_direct_pose:
                gate_mode = "direct"
                loss, stats = _lambda_fusion_loss_rollout(
                    trainer,
                    model,
                    batch,
                    columns=columns,
                    rollout_steps=cfg.rollout_steps,
                    rollout_cycles=cfg.rollout_cycles,
                    include_boundary=cfg.rollout_include_boundary,
                    boundary_weight=cfg.lambda_boundary_weight,
                    random_offset=cfg.rollout_random_offset,
                    time_index_mode=cfg.time_index_mode,
                    time_weight_max=cfg.lambda_time_weight_max,
                    time_weight_mode=cfg.lambda_time_weight_mode,
                    detach_rollout_state=cfg.detach_rollout_state,
                    lambda_entropy_weight=0.0,
                    lambda_smooth_weight=0.0,
                    lambda_early_steps=0,
                    lambda_early_weight=0.0,
                    lambda_monotonic_weight=0.0,
                    lambda_plan_entropy_weight=0.0,
                    lambda_plan_dyn_weight=0.0,
                    contact_meas_weight=cfg.contact_meas_weight,
                    objective="direct",
                    direct_pose_hinge_sup_weight=float(getattr(cfg, "direct_pose_hinge_sup_weight", 0.0) or 0.0),
                    direct_pose_hinge_sup_kind=str(getattr(cfg, "direct_pose_hinge_sup_kind", "smooth_l1") or "smooth_l1"),
                    direct_pose_hinge_sup_contact_source=str(getattr(cfg, "direct_pose_hinge_sup_contact_source", "gt") or "gt"),
                    direct_pose_hinge_sup_contact_value=getattr(cfg, "direct_pose_hinge_sup_contact_value", None),
                    direct_pose_hinge_sup_contact_thresh=float(getattr(cfg, "direct_pose_hinge_sup_contact_thresh", 0.5) or 0.5),
                    direct_pose_hinge_sup_angle_thresh_deg=float(getattr(cfg, "direct_pose_hinge_sup_angle_thresh_deg", 0.0) or 0.0),
                    direct_pose_hinge_sup_delta_thresh_deg=float(getattr(cfg, "direct_pose_hinge_sup_delta_thresh_deg", 0.0) or 0.0),
                    direct_pose_hinge_sup_delta_weight_power=float(getattr(cfg, "direct_pose_hinge_sup_delta_weight_power", 0.0) or 0.0),
                    direct_pose_hinge_sup_delta_weight_scale_deg=float(getattr(cfg, "direct_pose_hinge_sup_delta_weight_scale_deg", 0.0) or 0.0),
                    direct_pose_hinge_sup_delta_weight_max=float(getattr(cfg, "direct_pose_hinge_sup_delta_weight_max", 0.0) or 0.0),
                    direct_pose_hinge_gate_sup_weight=float(getattr(cfg, "direct_pose_hinge_gate_sup_weight", 0.0) or 0.0),
                    direct_pose_hinge_gate_sup_contact_source=str(getattr(cfg, "direct_pose_hinge_gate_sup_contact_source", "gt") or "gt"),
                    direct_pose_hinge_gate_sup_contact_thresh=float(getattr(cfg, "direct_pose_hinge_gate_sup_contact_thresh", 0.5) or 0.5),
                    direct_pose_hinge_stance_weight=float(getattr(cfg, "direct_pose_hinge_stance_weight", 0.0) or 0.0),
                    direct_pose_hinge_stance_kind=str(getattr(cfg, "direct_pose_hinge_stance_kind", "l2") or "l2"),
                    direct_pose_hinge_stance_contact_source=str(getattr(cfg, "direct_pose_hinge_stance_contact_source", "gt") or "gt"),
                    direct_pose_hinge_stance_contact_thresh=float(getattr(cfg, "direct_pose_hinge_stance_contact_thresh", 0.5) or 0.5),
                    direct_pose_hinge_reg_weight=float(getattr(cfg, "direct_pose_hinge_reg_weight", 0.0) or 0.0),
                    direct_pose_hinge_reg_kind=str(getattr(cfg, "direct_pose_hinge_reg_kind", "l1") or "l1"),
                    direct_pose_hinge_eps_l2_weight=float(getattr(cfg, "direct_pose_hinge_eps_l2_weight", 0.0) or 0.0),
                    direct_pose_leg_side_sign_gate_reg_weight=float(
                        getattr(cfg, "direct_pose_leg_side_sign_gate_reg_weight", 0.0) or 0.0
                    ),
                    direct_pose_leg_gate_sup_weight=float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0),
                    direct_pose_leg_scale_sup_weight=float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0),
                    direct_pose_leg_scale_sup_table=direct_leg_scale_sup_table,
                    direct_pose_leg_align_weight=float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0),
                    direct_pose_leg_align_oracle_min_deg=float(getattr(cfg, "direct_pose_leg_align_oracle_min_deg", 0.0) or 0.0),
                    direct_pose_leg_align_oracle_weight_deg=float(getattr(cfg, "direct_pose_leg_align_oracle_weight_deg", 0.0) or 0.0),
                    direct_pose_leg_align_mode=str(getattr(cfg, "direct_pose_leg_align_mode", "cos") or "cos"),
                    direct_pose_leg_align_mag_weight=float(getattr(cfg, "direct_pose_leg_align_mag_weight", 1.0) or 1.0),
                    direct_pose_leg_align_res_weight=float(getattr(cfg, "direct_pose_leg_align_res_weight", 1.0) or 1.0),
                    direct_pose_leg_align_sign_weight=float(getattr(cfg, "direct_pose_leg_align_sign_weight", 0.0) or 0.0),
                    direct_pose_leg_align_cos_thresh=float(getattr(cfg, "direct_pose_leg_align_cos_thresh", 0.0) or 0.0),
                    direct_pose_loss_tail_mix=float(getattr(cfg, "direct_pose_loss_tail_mix", 0.0) or 0.0),
                    direct_pose_loss_tail_temp_deg=float(getattr(cfg, "direct_pose_loss_tail_temp_deg", 0.0) or 0.0),
                    direct_pose_loss_state_swing_boost=float(getattr(cfg, "direct_pose_loss_state_swing_boost", 0.0) or 0.0),
                    direct_pose_loss_state_contact_source=str(getattr(cfg, "direct_pose_loss_state_contact_source", "gt") or "gt"),
                    direct_pose_loss_state_scope=str(getattr(cfg, "direct_pose_loss_state_scope", "legs") or "legs"),
                    direct_pose_loss_leg_split=bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
                    direct_pose_loss_leg_tail_scale=str(getattr(cfg, "direct_pose_loss_leg_tail_scale", "center") or "center"),
                    direct_pose_loss_sics=str(getattr(cfg, "direct_pose_loss_sics", "") or ""),
                    direct_pose_loss_cycle_gte=int(getattr(cfg, "direct_pose_loss_cycle_gte", 0) or 0),
                    direct_pose_loss_sic_mode=str(getattr(cfg, "direct_pose_loss_sic_mode", "mask") or "mask"),
                    direct_pose_loss_sic_boost=float(getattr(cfg, "direct_pose_loss_sic_boost", 1.0) or 1.0),
                    direct_pose_loss_pair_boost_table=direct_pose_pair_boost_table,
                    direct_pose_loss_group_norm_enable=bool(
                        getattr(cfg, "direct_pose_loss_group_norm_enable", False)
                    ),
                    direct_pose_loss_group_norm_w_leg=float(
                        getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0) or 1.0
                    ),
                    direct_pose_loss_group_norm_w_nonleg=float(
                        getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0) or 1.0
                    ),
                    direct_pose_loss_group_norm_ema_beta=float(
                        getattr(cfg, "direct_pose_loss_group_norm_ema_beta", 0.95) or 0.95
                    ),
                    direct_pose_loss_group_norm_ratio_min=float(
                        getattr(cfg, "direct_pose_loss_group_norm_ratio_min", 0.2) or 0.2
                    ),
                    direct_pose_loss_group_norm_ratio_max=float(
                        getattr(cfg, "direct_pose_loss_group_norm_ratio_max", 5.0) or 5.0
                    ),
                    direct_pose_loss_group_norm_eps=float(
                        getattr(cfg, "direct_pose_loss_group_norm_eps", 1e-6) or 1e-6
                    ),
                )
            else:
                loss, stats = _lambda_fusion_loss_rollout(
                    trainer,
                    model,
                    batch,
                    columns=columns,
                    rollout_steps=cfg.rollout_steps,
                    rollout_cycles=cfg.rollout_cycles,
                    include_boundary=cfg.rollout_include_boundary,
                    boundary_weight=cfg.lambda_boundary_weight,
                    random_offset=cfg.rollout_random_offset,
                    time_index_mode=cfg.time_index_mode,
                    time_weight_max=cfg.lambda_time_weight_max,
                    time_weight_mode=cfg.lambda_time_weight_mode,
                    detach_rollout_state=cfg.detach_rollout_state,
                    lambda_entropy_weight=cfg.lambda_fusion_entropy_weight,
                    lambda_smooth_weight=cfg.lambda_fusion_smooth_weight,
                    lambda_early_steps=cfg.lambda_fusion_early_steps,
                    lambda_early_weight=cfg.lambda_fusion_early_weight,
                    lambda_monotonic_weight=cfg.lambda_fusion_monotonic_weight,
                    lambda_plan_entropy_weight=cfg.lambda_plan_entropy_weight,
                    lambda_plan_dyn_weight=cfg.lambda_plan_dyn_weight,
                    lambda_gate_sup_weight=(cfg.lambda_gate_sup_weight if cfg.train_lambda_head else 0.0),
                    lambda_gate_sup_tau_deg=cfg.lambda_gate_sup_tau_deg,
                    lambda_gate_sup_margin_deg=cfg.lambda_gate_sup_margin_deg,
                    lambda_gate_sup_start_step=cfg.lambda_gate_sup_start_step,
                    contact_meas_weight=cfg.contact_meas_weight,
                    objective="blend",
                )
                if l2sp_pairs and l2sp_weight > 0.0:
                    try:
                        terms = []
                        for p, p0 in l2sp_pairs:
                            # Use per-tensor mean to keep scale stable across shapes.
                            terms.append((p.float() - p0.float()).pow(2).mean())
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
            loss.backward()
            if cfg.train_direct_pose and bool(getattr(cfg, "direct_pose_grad_monitor_enable", False)):
                g_trunk = _module_grad_norm(getattr(model, "direct_pose_head", None))
                g_leg = _module_grad_norm(getattr(model, "direct_pose_out_leg", None))
                g_nonleg = _module_grad_norm(getattr(model, "direct_pose_out_nonleg", None))
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
            if (it % 20) == 0:
                if cfg.train_so3_corrector:
                    msg = (
                        f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] "
                        f"total={stats['total']:.6f} corr={stats['corr_loss']:.6f} "
                        f"omega_l2={stats['omega_l2']:.3e} gate={stats['gate_mean']:.4f} ({gate_mode})"
                    )
                elif cfg.train_contact_plan_init:
                    msg = (
                        f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] "
                        f"total={stats['total']:.6f} contact_plan_mse={stats.get('contact_plan_mse', float('nan')):.6f}"
                    )
                    if "contact_plan_loss" in stats:
                        msg += f" plan_loss={stats.get('contact_plan_loss', float('nan')):.6f}"
                    if "contact_plan_mse_early10" in stats:
                        msg += f" early10={stats.get('contact_plan_mse_early10', float('nan')):.6f}"
                    if "contact_plan_loss_early10" in stats:
                        msg += f" loss_early10={stats.get('contact_plan_loss_early10', float('nan')):.6f}"
                elif cfg.train_contact_plan:
                    msg = (
                        f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] "
                        f"total={stats['total']:.6f} contact_plan_mse={stats.get('contact_plan_mse', float('nan')):.6f}"
                    )
                    if "contact_plan_loss" in stats:
                        msg += f" plan_loss={stats.get('contact_plan_loss', float('nan')):.6f}"
                    if "contact_plan_mse_early10" in stats:
                        msg += f" early10={stats.get('contact_plan_mse_early10', float('nan')):.6f}"
                    if "contact_plan_loss_early10" in stats:
                        msg += f" loss_early10={stats.get('contact_plan_loss_early10', float('nan')):.6f}"
                    if "contact_plan_mean" in stats and "contact_plan_std" in stats:
                        msg += f" plan={stats.get('contact_plan_mean', float('nan')):.3f}±{stats.get('contact_plan_std', float('nan')):.3f}"
                elif cfg.train_direct_pose:
                    msg = (
                        f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] "
                        f"total={stats['total']:.6f} dir={stats['dir_geo']:.6f} "
                        f"blend={stats['blend_loss']:.6f} inc={stats['inc_geo']:.6f} "
                        f"λ={stats['lambda_mean']:.3f}±{stats['lambda_std']:.3f}"
                    )
                    # Helpful when tuning hinge supervision (smooth_l1 vs apply_geo, w_delta scaling, etc.).
                    if float(getattr(cfg, "direct_pose_hinge_sup_weight", 0.0) or 0.0) > 0.0:
                        msg += (
                            f" hinge_sup={stats.get('hinge_sup_loss', float('nan')):.3e}"
                            f" ratio={stats.get('hinge_sup_abs_delta_ratio', float('nan')):.3f}"
                            f" |pred|={stats.get('hinge_sup_abs_delta_pred_deg', float('nan')):.2f}deg"
                            f" |tgt|={stats.get('hinge_sup_abs_delta_tgt_deg', float('nan')):.2f}deg"
                        )
                    if float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0) > 0.0:
                        msg += (
                            f" leg_gate={stats.get('leg_gate_sup_loss', float('nan')):.3e}"
                            f" tgt={stats.get('leg_gate_sup_tgt_frac', float('nan')):.3f}"
                            f" pred={stats.get('leg_gate_sup_pred_mean', float('nan')):.3f}"
                        )
                    if float(getattr(cfg, "direct_pose_leg_scale_sup_weight", 0.0) or 0.0) > 0.0:
                        msg += (
                            f" leg_scale={stats.get('leg_scale_sup_total_loss', float('nan')):.3e}"
                            f" logμ(tgt/pred)={stats.get('leg_scale_sup_tgt_mean_log', float('nan')):.3f}"
                            f"/{stats.get('leg_scale_sup_pred_mean_log', float('nan')):.3f}"
                        )
                        # Signed-scale tables optionally provide sign targets; print when in signed_scale mode.
                        try:
                            gm = str(getattr(cfg, "direct_pose_leg_gate_mode", "") or "").strip().lower()
                        except Exception:
                            gm = ""
                        if gm in (
                            "signed_scale",
                            "signedscale",
                            "signed",
                            "signmag",
                            "sign_mag",
                            "signmagscale",
                            "signedmag",
                            "sscale",
                        ):
                            msg += (
                                f" sign={stats.get('leg_scale_sup_sign_loss', float('nan')):.3e}"
                                f" tgt={stats.get('leg_scale_sup_sign_tgt_mean', float('nan')):.3f}"
                                f" pred={stats.get('leg_scale_sup_sign_pred_mean', float('nan')):.3f}"
                            )
                    if float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0) > 0.0:
                        msg += (
                            f" leg_align={stats.get('leg_align_loss', float('nan')):.3e}"
                            f" frac={stats.get('leg_align_frac', float('nan')):.3f}"
                        )
                    if float(stats.get("dir_group_norm_used", 0.0) or 0.0) > 0.0:
                        msg += (
                            f" gnorm(L/N)={stats.get('dir_group_norm_leg', float('nan')):.3f}/"
                            f"{stats.get('dir_group_norm_nonleg', float('nan')):.3f}"
                            f" ema={stats.get('dir_group_norm_leg_ema', float('nan')):.3f}/"
                            f"{stats.get('dir_group_norm_nonleg_ema', float('nan')):.3f}"
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
                elif (not cfg.train_lambda_head) and (not cfg.train_direct_pose) and bool(cfg.train_contact_meas):
                    msg = f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] total={stats['total']:.6f}"
                elif (not cfg.train_lambda_head) and (not cfg.train_direct_pose) and bool(cfg.train_contact_td_hazard):
                    msg = f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] total={stats['total']:.6f}"
                else:
                    msg = (
                        f"[posttrain][e{epoch} i{it}/{cfg.steps_per_epoch}] "
                        f"total={stats['total']:.6f} blend={stats['blend_loss']:.6f} "
                        f"λ={stats['lambda_mean']:.3f}±{stats['lambda_std']:.3f} "
                        f"inc={stats['inc_geo']:.6f} dir={stats['dir_geo']:.6f}"
                    )
                    if str(cfg.lambda_reliability_mode or "none").strip().lower() not in ("none", "off", "false", "0", ""):
                        msg += (
                            f" λ_eff={stats.get('lambda_eff_mean', float('nan')):.3f}±{stats.get('lambda_eff_std', float('nan')):.3f}"
                            f" r={stats.get('lambda_rel_mean', float('nan')):.3f}"
                        )
                    if bool(cfg.train_lambda_head) and float(cfg.lambda_gate_sup_weight or 0.0) > 0.0:
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
                if "contact_meas_bce" in stats:
                    msg += f" contact_meas_bce={stats['contact_meas_bce']:.4f}"
                elif "contact_meas_mse" in stats:
                    msg += f" contact_meas_mse={stats['contact_meas_mse']:.4f}"
                if "contact_td_hazard_bce" in stats:
                    msg += f" td_hz_bce={stats['contact_td_hazard_bce']:.3f}"
                if "contact_td_hazard_mass_l2" in stats:
                    msg += f" td_hz_mass={stats['contact_td_hazard_mass_l2']:.3f}"
                if "contact_td_hazard_unimodal" in stats:
                    msg += f" td_hz_uni={stats['contact_td_hazard_unimodal']:.3e}"
                print(msg)
            row = dict(stats)
            row["gate_mode"] = gate_mode
            row["epoch"] = float(epoch)
            row["iter"] = float(it)
            row["step"] = float(global_step)
            log_rows.append(row)

        denom = max(1, int(ok_steps))
        avg = epoch_loss / denom
        print(f"[posttrain][epoch {epoch}] avg_total={avg:.6f} ok_steps={ok_steps} skipped={bad_steps}")

    cfg_jsonable: dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            cfg_jsonable[k] = str(v)
        elif isinstance(v, tuple) and v and all(isinstance(p, Path) for p in v):
            cfg_jsonable[k] = [str(p) for p in v]
        else:
            cfg_jsonable[k] = v

    # Persist the *effective* semantic routing used to instantiate the model.
    # These options can be shape-compatible (e.g. hidden vs hidden_pre) and thus not inferable from tensors alone.
    cfg_jsonable["direct_pose_feat_source"] = str(direct_pose_feat_source)
    cfg_jsonable["direct_pose_time_pe_dim"] = int(direct_pose_time_pe_dim)
    cfg_jsonable["direct_pose_time_pe_base"] = float(direct_pose_time_pe_base)
    cfg_jsonable["direct_pose_use_phase_z"] = bool(direct_pose_use_phase_z)
    cfg_jsonable["direct_pose_phase_z_mode"] = str(direct_pose_phase_z_mode)
    cfg_jsonable["direct_pose_split_enable"] = bool(direct_pose_split_enable)
    cfg_jsonable["direct_pose_nonleg_proj_dim"] = int(direct_pose_nonleg_proj_dim)
    cfg_jsonable["direct_pose_nonleg_train_only"] = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
    cfg_jsonable["direct_pose_leg_gate_mode"] = str(direct_pose_leg_gate_mode_model)
    cfg_jsonable["direct_pose_leg_gate_power"] = float(direct_pose_leg_gate_power_model)
    cfg_jsonable["direct_pose_hinge_enable"] = bool(direct_pose_hinge_enable_model)
    cfg_jsonable["direct_pose_hinge_bones"] = direct_pose_hinge_bones_model
    cfg_jsonable["direct_pose_hinge_axis"] = str(direct_pose_hinge_axis_model)
    cfg_jsonable["direct_pose_hinge_max_deg"] = float(direct_pose_hinge_max_deg_model)
    cfg_jsonable["direct_pose_hinge_hidden"] = direct_pose_hinge_hidden_model
    cfg_jsonable["direct_pose_hinge_feat_source"] = (
        "auto" if direct_pose_hinge_feat_source_model is None else str(direct_pose_hinge_feat_source_model)
    )
    cfg_jsonable["direct_pose_hinge_base_feat"] = str(direct_pose_hinge_base_feat_model)
    cfg_jsonable["direct_pose_hinge_clean"] = bool(direct_pose_hinge_clean_model)
    cfg_jsonable["direct_pose_hinge_eps_max_deg"] = 0.0 if direct_pose_hinge_eps_max_deg_model is None else float(direct_pose_hinge_eps_max_deg_model)
    cfg_jsonable["direct_pose_hinge_eps_max_scale"] = float(direct_pose_hinge_eps_max_scale_model)
    cfg_jsonable["direct_pose_hinge_eps_hidden"] = direct_pose_hinge_eps_hidden_model
    cfg_jsonable["direct_pose_hinge_eps_dropout"] = float(direct_pose_hinge_eps_dropout_model)
    cfg_jsonable["direct_pose_hinge_eps_source"] = str(direct_pose_hinge_eps_source_model)
    cfg_jsonable["direct_pose_hinge_gate_mode"] = str(direct_pose_hinge_gate_mode_model)
    cfg_jsonable["direct_pose_hinge_gate_source"] = str(direct_pose_hinge_gate_source_model)
    cfg_jsonable["direct_pose_hinge_gate_power"] = float(direct_pose_hinge_gate_power_model)
    ckpt_out = cfg.out_dir / f"ckpt_last_{cfg.run_name}.pth"
    torch.save({"model": model.state_dict(), "posttrain_cfg": cfg_jsonable}, ckpt_out)
    out_log: dict[str, Any] = {"config": cfg_jsonable, "log": log_rows}
    try:
        dbg = getattr(trainer, "_debug_hinge_sup_phase", None)
        if isinstance(dbg, dict) and isinstance(dbg.get("by_phase", None), dict) and dbg["by_phase"]:
            out_log["debug_hinge_sup_phase"] = dbg
    except Exception:
        pass
    dump_json(cfg.out_dir / f"posttrain_log_{cfg.run_name}.json", out_log)
    print(f"[posttrain][OK] saved: {ckpt_out}")


if __name__ == "__main__":
    main()
