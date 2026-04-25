from __future__ import annotations

"""
Unified model definitions for training and inference.
"""

import math as _math
import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    _resolve_joint_spec_indices,
    build_mlp,
)
from .history import AdaptiveHistoryModule
from .geometry import (
    rot6d_to_matrix,
    geodesic_R,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    parent_relative_matrices,
)
from .data.layout import infer_rot_joint_count, parse_layout_entry, resolve_rot6d_slice
from .checkpoint.compat import (
    maybe_upgrade_direct_pose_split_state_dict,
)

__all__ = [
    'MotionEncoder',
    'PeriodHead',
    'EventMotionModel',
    'MotionJointLoss',
    'DEFAULT_DIRECT_POSE_LEG_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES_CSV',
]

DEFAULT_DIRECT_POSE_LEG_BONES: tuple[str, ...] = (
    'thigh_r', 'calf_r', 'foot_r', 'ball_r',
    'thigh_l', 'calf_l', 'foot_l', 'ball_l',
)

STAGE6_3WAY_ARMCHAIN_BONES: tuple[str, ...] = (
    'clavicle_l', 'upperarm_l', 'RUpArmTwist_l_01', 'RUpArmTwist_l_02',
    'lowerarm_l', 'L_ForeTwist_01', 'L_ForeTwist_02', 'hand_l',
    'index_01_l', 'middle_01_l', 'ring_01_l', 'pinky_01_l', 'thumb_01_l',
    'clavicle_r', 'upperarm_r', 'RUpArmTwist_r_01', 'RUpArmTwist_r_02',
    'lowerarm_r', 'R_ForeTwist_01', 'R_ForeTwist_02', 'hand_r',
    'index_01_r', 'middle_01_r', 'ring_01_r', 'pinky_01_r', 'thumb_01_r',
)

STAGE6_3WAY_ARMCHAIN_BONES_CSV = ','.join(STAGE6_3WAY_ARMCHAIN_BONES)

_DIRECT_POSE_DEFAULT_STAT_KEYS: tuple[str, ...] = (
    'direct_pose_geo',
    'direct_pose_geo_deg',
    'direct_pose_objective',
    'direct_pose_weighted',
    'direct_pose_split_active',
    'direct_pose_arm_split_active',
    'dir_base',
    'dir_leg_base',
    'dir_nonleg_base',
    'dir_nonleg_effective_base',
    'dir_arm_base',
    'dir_else_base',
    'leg_over_nonleg',
    'leg_over_nonleg_effective',
    'arm_over_else',
    'direct_pose_arm_else_balance_active',
    'direct_pose_loss_arm_weight',
    'direct_pose_loss_else_weight',
    'dir_group_norm_used',
    'dir_group_norm_leg_raw',
    'dir_group_norm_nonleg_raw',
    'dir_group_norm_leg_clamped',
    'dir_group_norm_nonleg_clamped',
    'dir_group_norm_leg',
    'dir_group_norm_nonleg',
    'dir_group_norm_leg_ema',
    'dir_group_norm_nonleg_ema',
    'dir_group_norm_leg_hit_min',
    'dir_group_norm_leg_hit_max',
    'dir_group_norm_nonleg_hit_min',
    'dir_group_norm_nonleg_hit_max',
    'dir_group_norm_leg_hit_any',
    'dir_group_norm_nonleg_hit_any',
)

_DIRECT_POSE_COMPONENT_STAT_KEYS: tuple[str, ...] = (
    *_DIRECT_POSE_DEFAULT_STAT_KEYS,
    'dir_group_norm_w_leg',
    'dir_group_norm_w_nonleg',
)


def _torch_dynamo_is_compiling_safe() -> bool:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        return False
    probe = getattr(dynamo, "is_compiling", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except (AttributeError, RuntimeError):
        return False


def _torch_onnx_is_in_export_safe() -> bool:
    onnx_mod = getattr(torch, "onnx", None)
    if onnx_mod is None:
        return False
    probe = getattr(onnx_mod, "is_in_onnx_export", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except (AttributeError, RuntimeError):
        return False


def _normalize_eval_runtime_ablate_mode(raw: Any) -> str:
    mode = str(raw or "none").strip().lower()
    if mode in ("", "none", "off", "disable", "disabled"):
        return "none"
    if mode in ("0", "zero", "zeros"):
        return "zero"
    if mode in ("roll", "roll_batch", "shift", "shift_batch"):
        return "roll_batch"
    if mode in ("roll_time", "shift_time"):
        return "roll_time"
    return "none"


@dataclass(frozen=True, slots=True)
class _EvalRuntimeControls:
    direct_pose_plan_override: Any = None
    direct_pose_meas_override: Any = None
    direct_pose_leg_side_plan_other_ablate_mode: str = "none"
    direct_pose_leg_cross_leg_ablate_mode: str = "none"
    contact_plan_inject_scale: float = 1.0
    contact_plan_time_bias_scale: float = 1.0
    debug_contact_plan_logits_decomp: bool = False


_DEFAULT_EVAL_RUNTIME_CONTROLS = _EvalRuntimeControls()


@dataclass(slots=True)
class _ContactPlanDebugBuffers:
    contacts_plan_logits_base: list[torch.Tensor]
    contacts_plan_logits_phase: list[torch.Tensor]
    contacts_plan_logits_time: list[torch.Tensor]
    contacts_plan_logits_raw: list[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _ContactPlanDebugLogits:
    contacts_plan_logits_base: Optional[torch.Tensor] = None
    contacts_plan_logits_phase: Optional[torch.Tensor] = None
    contacts_plan_logits_time: Optional[torch.Tensor] = None
    contacts_plan_logits_raw: Optional[torch.Tensor] = None


@dataclass(frozen=True, slots=True)
class _EventMotionForwardInputPrep:
    state: torch.Tensor
    cond: Optional[torch.Tensor]
    contacts: Any
    angvel: Optional[torch.Tensor]
    pose_history: Optional[torch.Tensor]
    plan_z: Optional[torch.Tensor]
    phase_z: Optional[torch.Tensor]
    phase_event_age: Optional[torch.Tensor]
    is_single: bool
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    query_steps: int
    runtime_controls: _EvalRuntimeControls
    contacts_input: Any
    contacts_enc: Any


@dataclass(frozen=True, slots=True)
class _ContactClockForwardDefaults:
    soft_period: Optional[torch.Tensor]
    contacts_meas: Optional[torch.Tensor]
    event_clock_delta_meas: Optional[torch.Tensor]
    event_clock_lr_diff: Optional[torch.Tensor]
    event_clock_lambda_corr: Optional[torch.Tensor]
    event_clock_lambda_logit: Optional[torch.Tensor]
    event_clock_dynamic_prior: Optional[torch.Tensor]
    event_clock_delta_z: Optional[torch.Tensor]
    pose_hist_processed: bool
    contacts_plan: Optional[torch.Tensor]
    plan_z_next: Optional[torch.Tensor]
    plan_feat_for_inject: Optional[torch.Tensor]
    contacts_plan_logits: Optional[torch.Tensor]
    contact_plan_debug_logits: _ContactPlanDebugLogits
    time_pe_direct: Optional[torch.Tensor]
    phase_z_in_direct: Optional[torch.Tensor]
    leg_side_cue_in: Optional[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _ContactPlanForwardFinal:
    contacts_plan: torch.Tensor
    phase_z_in_direct: Optional[torch.Tensor]
    leg_side_cue_in: Optional[torch.Tensor]
    contacts_plan_logits: Optional[torch.Tensor]
    contact_plan_debug_logits: _ContactPlanDebugLogits
    plan_z_next: Optional[torch.Tensor]
    plan_feat_for_inject: Optional[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _DirectPoseForwardRuntime:
    plan_override: Any
    meas_override: Any
    leg_side_plan_other_ablate_mode: str
    leg_cross_leg_ablate_mode: str


@dataclass(frozen=True, slots=True)
class _DirectPoseLegGateOutputs:
    omega_eff: torch.Tensor
    direct_leg_gate: Optional[torch.Tensor] = None
    direct_leg_gate_logits: Optional[torch.Tensor] = None
    direct_leg_scale: Optional[torch.Tensor] = None
    direct_leg_scale_log: Optional[torch.Tensor] = None
    direct_leg_scale_log_raw: Optional[torch.Tensor] = None


@dataclass(frozen=True, slots=True)
class _DirectPoseSideLegAssembly:
    joint_count: int
    branch_joint_count: int
    pos_r: torch.Tensor
    pos_l: torch.Tensor
    leg_flat_r: torch.Tensor
    leg_flat_l: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DirectPoseSideLegOmegaOutputs:
    omega_r: Optional[torch.Tensor]
    omega_l: Optional[torch.Tensor]
    direct_leg_side_sign_gate: Optional[torch.Tensor] = None


_DEFAULT_CONTACT_PLAN_DEBUG_LOGITS = _ContactPlanDebugLogits()
_CONTACT_PLAN_DEBUG_LOGIT_KEYS = (
    "contacts_plan_logits_base",
    "contacts_plan_logits_phase",
    "contacts_plan_logits_time",
    "contacts_plan_logits_raw",
)


class MotionEncoder(nn.Module):
    """
    Stateless per-frame encoder that mirrors the Plan-A pretraining MLP.
    - Uses shared MLP over frames, outputs [B, T, H]
    - Optional summary head for global pooling
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        z_dim: int = 0,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)

        self.mlp = build_mlp(
            input_dim,
            self.hidden_dim,
            num_layers=max(1, int(num_layers)),
            activation=nn.GELU,
            dropout=float(dropout),
        )
        self.summary_head = nn.Linear(self.hidden_dim, self.z_dim) if self.z_dim > 0 else None

    def forward(self, x: torch.Tensor, return_summary: bool | None = None):
        """
        x: [B, T, D] or [T, D]; returns per-frame hidden states [B, T, H].
        When return_summary=True (or summary_head exists) also returns a pooled summary.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        enc = self.mlp(flat).reshape(B, T, self.hidden_dim)

        need_summary = return_summary if return_summary is not None else (self.summary_head is not None)
        if not need_summary:
            return enc

        summary_vec = enc.mean(dim=1)
        if self.summary_head is not None:
            summary_vec = self.summary_head(summary_vec)
        return summary_vec, enc


class PeriodHead(nn.Module):
    """Lightweight linear head used during pretraining to predict a soft hint embedding (contact-hint in first dims)."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)



class _CondFiLM(nn.Module):

    def __init__(self, cond_dim: int, hidden_dim: int, film_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, film_dim * 2)

    def forward(self, cond: torch.Tensor):
        h = torch.nn.functional.gelu(self.fc1(cond))
        g, b = self.fc2(h).chunk(2, dim=-1)
        g = 1.0 + 0.5 * torch.tanh(g)
        b = 0.5 * torch.tanh(b)
        return (g, b)


class _ResidualMLPBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim)
        self.act = activation()
        self.drop1 = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        if zero_init_last:
            nn.init.zeros_(self.fc2.weight)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class _BoneSliceResidualAdapter(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        alpha_mode: str = "tanh",  # "tanh" | "linear"
        alpha_init: float = 0.05,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            in_dim,
            int(hidden_dim),
            num_layers=1,
            activation=activation,
            dropout=float(dropout),
            final_dim=int(out_dim),
        )
        # NOTE:
        # - alpha=0 makes the adapter output exactly 0, but also blocks gradients to `net`.
        # - We keep initial behavior == baseline by zero-initializing the last Linear layer
        #   while setting alpha to a small non-zero so the adapter can start learning.
        self.alpha = nn.Parameter(torch.as_tensor(float(alpha_init)).reshape(()))
        self.alpha_mode = str(alpha_mode)
        if zero_init_last:
            last_linear = None
            for mod in reversed(self.net):
                if isinstance(mod, nn.Linear):
                    last_linear = mod
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        if self.alpha_mode == "tanh":
            a = torch.tanh(a)
        return a * self.net(x)


class PlanZCorrector(nn.Module):
    """
    Event-Clock v3: residual correction on contact_plan GRU hidden state.

    plan_z_corrected = plan_z_raw + lambda_corr * clip(LN(delta_z))
    """

    def __init__(
        self,
        *,
        plan_z_dim: int,
        contact_dim: int,
        period_feat_dim: int,
        hidden_dim: int = 64,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.plan_z_dim = int(plan_z_dim)
        self.contact_dim = int(contact_dim)
        self.period_feat_dim = max(0, int(period_feat_dim))
        self.max_delta = float(max_delta)

        in_dim = self.plan_z_dim + self.contact_dim * 3 + self.period_feat_dim
        h = max(8, int(hidden_dim))
        self.correction_head = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.plan_z_dim),
        )
        self.layer_norm = nn.LayerNorm(self.plan_z_dim)

    def forward(
        self,
        *,
        plan_z_raw: torch.Tensor,  # (B, H)
        contacts_meas: torch.Tensor,  # (B, C)
        delta_meas: torch.Tensor,  # (B, C)
        err_raw: torch.Tensor,  # (B, C) plan_raw - meas
        period_feat: Optional[torch.Tensor],  # (B, P) or None
        lambda_corr: torch.Tensor,  # (B, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = [plan_z_raw, contacts_meas, delta_meas, err_raw]
        if self.period_feat_dim > 0:
            if period_feat is None:
                inputs.append(torch.zeros(plan_z_raw.shape[0], self.period_feat_dim, device=plan_z_raw.device, dtype=plan_z_raw.dtype))
            else:
                inputs.append(period_feat)
        corr_in = torch.cat(inputs, dim=-1)

        delta_z_raw = self.correction_head(corr_in)
        delta_z = self.layer_norm(delta_z_raw)
        if self.max_delta > 0:
            delta_z = delta_z.clamp(-self.max_delta, self.max_delta)

        plan_z_corr = plan_z_raw + lambda_corr * delta_z
        return plan_z_corr, delta_z


class PeriodicityGate(nn.Module):
    """
    Event-Clock v3: confidence/periodicity gating for correction strength.

    Inputs:
      - err_raw      : (B, C)   plan/meas consistency
      - delta_meas   : (B, C)   direction / change signal (logits diff if available)
      - lr_diff      : (B, 1)   L/R separation (feature, not multiplicative mask)
      - period_feat  : (B, P)   independent periodicity feature (optional)
    Outputs:
      - lambda_corr  : (B, 1)
      - lambda_logit : (B, 1)
      - dynamic_prior: (B, 1)  prior = f(|delta_meas|, period_feat)
    """

    def __init__(
        self,
        *,
        contact_dim: int,
        period_feat_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.contact_dim = int(contact_dim)
        self.period_feat_dim = max(0, int(period_feat_dim))

        in_dim = self.contact_dim * 2 + 1 + self.period_feat_dim
        h = max(8, int(hidden_dim))
        self.confidence_head = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

        prior_in = self.contact_dim + self.period_feat_dim
        self.prior_head = nn.Sequential(
            nn.Linear(prior_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        *,
        err_raw: torch.Tensor,
        delta_meas: torch.Tensor,
        lr_diff: torch.Tensor,
        period_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.period_feat_dim > 0:
            if period_feat is None:
                period_feat = torch.zeros(err_raw.shape[0], self.period_feat_dim, device=err_raw.device, dtype=err_raw.dtype)
        else:
            period_feat = None

        parts = [err_raw, delta_meas, lr_diff]
        if period_feat is not None:
            parts.append(period_feat)
        gate_in = torch.cat(parts, dim=-1)

        lambda_logit = self.confidence_head(gate_in)
        lambda_corr = torch.sigmoid(lambda_logit)

        prior_parts = [delta_meas.abs()]
        if period_feat is not None:
            prior_parts.append(period_feat)
        prior_in = torch.cat(prior_parts, dim=-1)
        dynamic_prior = self.prior_head(prior_in)
        return lambda_corr, lambda_logit, dynamic_prior

class EventMotionModel(nn.Module):
    """
    无状态动作生成模型：通过显式传入的历史缓冲而非隐式 hidden_state 建模。
    """

    def __init__(
        self,
        in_state_dim: int,
        out_motion_dim: int,
        cond_dim: int = 0,
        period_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        context_len: int = 32,
        use_layer_norm: bool = True,
        *,
        contact_dim: int = 0,
        angvel_dim: int = 0,
        pose_hist_dim: int = 0,
        state_layout: Optional[Dict[str, Any]] = None,
        bone_names: Optional[Sequence[str]] = None,
        output_layout: Optional[Dict[str, Any]] = None,
        residual_adapter_bones: Optional[Sequence[str]] = None,
        residual_adapter_hidden: int = 128,
        residual_adapter_dropout: Optional[float] = None,
        # ===== Contact Plan (independent anchor) =====
        contact_plan_enable: bool = False,
        contact_plan_hidden: int = 64,
        contact_plan_dropout: float = 0.0,
        contact_plan_inject: str = "none",  # 'none' | 'contacts' | 'plan_z'
        contact_plan_inject_detach: bool = True,
        # Optional: time embedding for contact_plan (helps when cond has little/no phase info).
        # Implemented as an additive bias on contact_plan logits: logits += time_head(PE(t)).
        contact_plan_time_pe_dim: int = 0,  # 0 disables; recommended 8/16
        contact_plan_time_pe_base: float = 10000.0,
        # Contact-plan init (cold-start): how to initialize plan_z when plan_z is None.
        # - zeros: init plan_z as zeros
        # - learnable: use learnable contact_plan_init_z (default)
        # - obs: init from observation features (contacts/angvel/pose_history) via a small MLP
        # - learnable+obs: use learnable init_z + obs-conditioned delta (recommended for phase/anchor disambiguation)
        contact_plan_init_mode: str = "learnable",
        contact_plan_init_hidden: int = 128,
        contact_plan_init_dropout: float = 0.0,
        # ===== Event-Clock v3 (contact_plan residual correction) =====
        use_event_clock: bool = False,
        event_clock_max_delta: float = 0.5,
        event_clock_hidden_dim: int = 64,
        event_clock_gate_hidden_dim: int = 32,
        # ===== Direct Pose Head (cond + contacts_plan -> absolute pose in Y space) =====
        direct_pose_enable: bool = False,
        direct_pose_hidden: int = 256,
        direct_pose_dropout: float = 0.0,
        direct_pose_detach_plan: bool = True,
        # Optional: phase-hint bridge for direct head (see docs/phase_disambiguation_bridge.md)
        # - concat: direct uses (cond, contacts_plan, contacts_meas) (D0)
        #           (or, when direct_pose_use_phase_z=true and direct_pose_phase_z_mode='replace_contacts',
        #            uses phase_z_in to replace the 2D contact hint in direct conditioning)
        # - mode_select: predict 2 modes from (cond, contacts_plan) and blend by contacts_meas (D1)
        direct_pose_meas_mode: str = "concat",  # 'concat' | 'mode_select'
        # Train-time corruption for robustness / avoid shortcutting (D2)
        direct_pose_meas_drop_prob: float = 0.0,  # drop (zero) meas hint per-step
        direct_pose_meas_noise_std: float = 0.0,  # add noise then clamp to [0,1]
        direct_pose_plan_drop_prob: float = 0.0,  # drop (zero) plan input per-step
        # Direct pose head feature selection:
        # - cond: only use external conditioning C (legacy)
        # - hidden: use internal shared representation h_final (includes PASA temporal aggregation)
        # - hidden_pre: use pre-PASA representation h_temporal (per-step; no temporal aggregation)
        # - cond+hidden: concatenate [cond, h_final]
        # - cond+hidden_pre: concatenate [cond, h_temporal]
        direct_pose_feat_source: str = "cond",  # 'cond' | 'hidden' | 'hidden_pre' | 'cond+hidden' | 'cond+hidden_pre'
        # Optional: explicit time/clock embedding concatenated into direct head input (uses time_index in forward).
        direct_pose_time_pe_dim: int = 0,  # 0 disables; recommended 8/16
        direct_pose_time_pe_base: float = 10000.0,
        # Optional: also concatenate explicit phase state (phase_z_in) into direct head input.
        # This is higher-bandwidth than 2D contact probs and is already maintained as a step-stateful clock.
        # Shape: phase_z_in = flatten([sinφ_L, cosφ_L, sinφ_R, cosφ_R, ...]) => dim = 2*contact_dim.
        direct_pose_use_phase_z: bool = False,
        # How to route phase_z_in into direct conditioning:
        # - concat           : append phase_z_in as extra features (legacy "add phase" behavior)
        # - replace_contacts : use phase_z_in as the *only* phase hint, replacing (contacts_plan, contacts_meas)
        #                      in concat mode. This keeps the direct head input dim unchanged (2*C).
        direct_pose_phase_z_mode: str = "concat",
        # Optional: split direct head output into leg/non-leg heads while keeping a shared trunk.
        # This reallocates output capacity without changing downstream output semantics.
        direct_pose_split_enable: bool = False,
        # Optional: add a non-leg projection bottleneck before direct_pose_out_nonleg.
        # When >0: h_nonleg = ReLU(Linear(hid, proj)); out_nonleg = Linear(proj, D_nonleg)
        # When <=0: out_nonleg = Linear(hid, D_nonleg) (legacy split behavior).
        direct_pose_nonleg_proj_dim: int = 0,
        # Optional: split the non-leg branch into arm/else readouts (three-way: leg/arm/else).
        # Arm/else branches share the same trunk as leg branch, but use independent readouts (and optional
        # independent proj bottlenecks) to reduce gradient competition in a single non-leg head.
        direct_pose_arm_split_enable: bool = False,
        direct_pose_arm_bones: Optional[Sequence[str]] = None,
        # Optional: leg-specific residual head for direct pose (extra capacity for lower-body joints).
        # This head predicts a 6D residual that is added only on selected leg joints' BoneRotations6D slice.
        direct_pose_leg_enable: bool = False,
        direct_pose_leg_bones: Optional[Sequence[str]] = None,
        # Leg residual mode:
        # - rot6d_add: predict a 6D delta and add it directly on the rot6d parameters (legacy; off-manifold)
        # - so3: predict an on-manifold so(3) delta (omega) and compose: R_final = exp(omega) @ R_main
        direct_pose_leg_mode: str = "rot6d_add",
        # If true (recommended for decoupling), stop-grad the main head leg rotations in the composition:
        #   R_leg = exp(omega_leg) @ detach(R_main_leg)
        direct_pose_leg_stopgrad_main: bool = False,
        # If true (stronger decoupling), detach the feature input to the leg head so leg loss won't update the backbone.
        direct_pose_leg_detach_feat: bool = False,
        # Optional clamp on ||omega|| in degrees when direct_pose_leg_mode='so3'. 0 disables.
        direct_pose_leg_max_deg: float = 0.0,
        # Optional: learned gate/scale for leg omega (SO(3) mode only).
        # Mode options:
        # - none        : omega_eff = omega_raw
        # - learned     : omega_eff = sigmoid(gate_logits) ** gate_power * omega_raw
        #                (per-joint gate in [0,1], attenuation only)
        # - scale       : omega_eff = exp(clamp(log_mag, [-clip,+clip])) * omega_raw
        #                (per-joint positive scale in [exp(-clip),exp(+clip)])
        direct_pose_leg_gate_mode: str = "none",  # 'none' | 'learned' | 'scale'
        direct_pose_leg_gate_power: float = 1.0,
        # Only used when direct_pose_leg_gate_mode='scale' (exp(log_mag)).
        direct_pose_leg_scale_log_clip: float = 4.0,
        # Optional hard clamp on leg scale magnitude (k>1 => [1/k, k], 0/1 disables).
        # Applied to positive scale in mode='scale'.
        direct_pose_leg_scale_clamp_k: float = 0.0,
        # Optional: explicit per-side routing + shared omega head for leg residuals.
        # Motivation: avoid implicit joint->side routing failures during contact transitions / double-support.
        # When enabled, we run a shared head twice (R/L), feeding only that side's phase/contact hints, then
        # scatter back into the original K-joint ordering (direct_pose_leg_joint_idx order).
        direct_pose_leg_side_routing: bool = False,
        # contacts/phase channel order to map to (left/right) sides:
        # - "lr": channel0=left, channel1=right  (dataset default; see train/io.py load_soft_contacts_from_json)
        # - "rl": channel0=right, channel1=left
        direct_pose_leg_contact_order: str = "lr",
        # Optional tiny side embedding appended to leg head input (0 disables).
        direct_pose_leg_side_embed_dim: int = 0,
        # Optional: also append the other side's plan scalar to each routed shared omega head input.
        # This provides a cheap cross-leg context signal for double-support / transition disambiguation.
        direct_pose_leg_side_plan_other: bool = False,
        # Optional: also append the other side's phase (sin,cos) to each routed shared omega head input.
        # This is particularly important when direct_pose_phase_z_mode='replace_contacts' (phase_z is the main hint).
        direct_pose_leg_side_phase_other: bool = False,
        # Optional: append an explicit relative phase feature per side:
        #   phase_rel = (sin(phi_other - phi_self), cos(phi_other - phi_self))
        # This makes cross-leg phase differences linearly accessible instead of relying on the MLP to learn it.
        direct_pose_leg_side_phase_rel: bool = False,
        # Optional: extra per-side stateful cue appended to the routed shared leg head input (1 scalar per side).
        # - none: disabled (default; backward-compatible)
        # - phase_event_age: frames since last accepted phase reset event per contact channel (normalized by tau, clipped to [0,1])
        direct_pose_leg_side_cue: str = "none",
        direct_pose_leg_side_cue_tau: float = 30.0,
        # Optional: per-side sign gate for the routed shared leg omega head.
        # Predict g_side in [-1,1] (tanh) and apply: omega_side,j = g_side * omega_raw_side,j.
        # This couples the sign across joints on the same side, targeting "same-side co-flip" failures.
        direct_pose_leg_side_sign_gate: bool = False,
        # Optional: enforce a rank-1 (shared direction + per-joint non-negative scale) structure on routed
        # leg omega outputs. This "hard-couples" same-side joints and removes per-joint sign freedom:
        #   omega_side,j = softplus(s_side,j) * normalize(v_side)
        # Only applies when direct_pose_leg_side_routing=true (SO3-only).
        direct_pose_leg_side_rank1: bool = False,
        # ===== Stage2: λ Fusion Gate (incremental vs direct) =====
        lambda_fusion_enable: bool = False,
        lambda_fusion_mode: str = "per_joint",  # "global" | "per_joint"
        lambda_fusion_hidden: int = 128,
        lambda_fusion_dropout: float = 0.0,
        lambda_fusion_detach_err: bool = True,
        lambda_fusion_logit_init: float = -2.0,
        lambda_fusion_use_rollout_step: bool = False,
        # ===== SO(3) Delta Corrector (post-train friendly) =====
        so3_corr_hidden: int = 128,
        so3_corr_dropout: float = 0.0,
        so3_corr_gate_logit_init: float = -5.0,
    ):
        super().__init__()
        self.in_state_dim = int(in_state_dim)
        self.out_motion_dim = int(out_motion_dim)
        self.cond_dim = int(cond_dim)
        self.period_dim = int(period_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.context_len = int(context_len)
        self.contact_dim = max(0, int(contact_dim))
        self.angvel_dim = max(0, int(angvel_dim))
        self.pose_hist_dim = max(0, int(pose_hist_dim))
        self.state_layout: Dict[str, Any] = dict(state_layout) if isinstance(state_layout, dict) else {}
        self.encoder_input_dim = self.contact_dim + self.angvel_dim + self.pose_hist_dim
        self.contact_plan_enable = bool(contact_plan_enable and self.contact_dim > 0 and self.cond_dim > 0)
        self.contact_plan_hidden = int(contact_plan_hidden)
        if self.contact_plan_enable:
            self.contact_plan_hidden = max(8, int(self.contact_plan_hidden))
        self._contact_plan_dropout = float(contact_plan_dropout)
        self.contact_plan_inject = str(contact_plan_inject or "none").lower().strip()
        if self.contact_plan_inject not in ("none", "contacts", "plan_z"):
            self.contact_plan_inject = "none"
        self.contact_plan_inject_detach = bool(contact_plan_inject_detach)
        self._contact_plan_logits_dim = self.contact_dim
        self.adaptive_history_module: Optional[AdaptiveHistoryModule] = None
        self.pose_hist_len: int = 0
        self._adaptive_history_device: Optional[torch.device] = None
        self.contact_plan_time_pe_dim = int(contact_plan_time_pe_dim or 0)
        if self.contact_plan_time_pe_dim % 2 == 1:
            self.contact_plan_time_pe_dim += 1
        self._contact_plan_time_pe_base = float(contact_plan_time_pe_base or 10000.0)
        self._reset_eval_runtime_controls()
        self.contact_plan_init_mode = str(contact_plan_init_mode or "learnable").lower().strip()
        if self.contact_plan_init_mode in ("learnable_obs", "obs+learnable", "learnable+obs"):
            self.contact_plan_init_mode = "learnable+obs"
        if self.contact_plan_init_mode not in ("zeros", "learnable", "obs", "learnable+obs"):
            self.contact_plan_init_mode = "learnable"
        self.contact_plan_init_hidden = max(8, int(contact_plan_init_hidden or 0))
        self._contact_plan_init_dropout = float(contact_plan_init_dropout or 0.0)
        # Event-Clock v3: residual correction inside contact_plan loop
        self.use_event_clock = bool(use_event_clock and self.contact_plan_enable)
        self.event_clock_max_delta = float(event_clock_max_delta or 0.0)
        self.event_clock_hidden_dim = max(8, int(event_clock_hidden_dim or 0))
        self.event_clock_gate_hidden_dim = max(8, int(event_clock_gate_hidden_dim or 0))
        self.direct_pose_enable = bool(direct_pose_enable)
        self.direct_pose_hidden = max(8, int(direct_pose_hidden or 0))
        self._direct_pose_dropout = float(direct_pose_dropout)
        self.direct_pose_detach_plan = bool(direct_pose_detach_plan)
        self.direct_pose_meas_mode = str(direct_pose_meas_mode or "concat").lower().strip()
        if self.direct_pose_enable and self.direct_pose_meas_mode not in ("concat", "mode_select"):
            raise ValueError(
                f"Unsupported direct_pose_meas_mode={self.direct_pose_meas_mode!r}; expected 'concat' or 'mode_select'."
            )
        self.direct_pose_meas_drop_prob = float(direct_pose_meas_drop_prob or 0.0)
        self.direct_pose_meas_noise_std = float(direct_pose_meas_noise_std or 0.0)
        self.direct_pose_plan_drop_prob = float(direct_pose_plan_drop_prob or 0.0)
        self.direct_pose_feat_source = str(direct_pose_feat_source or "cond").lower().strip()
        if self.direct_pose_feat_source in ("h", "h_final", "hidden_only"):
            self.direct_pose_feat_source = "hidden"
        if self.direct_pose_feat_source in ("h_pre", "h_temporal", "hidden_pre", "pre", "temporal", "mid"):
            self.direct_pose_feat_source = "hidden_pre"
        if self.direct_pose_feat_source in ("cond_hidden", "hidden_cond", "concat", "cond+hidden", "hidden+cond"):
            self.direct_pose_feat_source = "cond+hidden"
        if self.direct_pose_feat_source in ("cond+hidden_pre", "cond_hidden_pre", "hidden_pre+cond", "cond+pre", "pre+cond"):
            self.direct_pose_feat_source = "cond+hidden_pre"
        if self.direct_pose_feat_source not in ("cond", "hidden", "hidden_pre", "cond+hidden", "cond+hidden_pre"):
            raise ValueError(
                "direct_pose_feat_source must be one of "
                "{'cond', 'hidden', 'hidden_pre', 'cond+hidden', 'cond+hidden_pre'} "
                f"after alias normalization; got {direct_pose_feat_source!r}."
            )
        self.direct_pose_time_pe_dim = int(direct_pose_time_pe_dim or 0)
        if self.direct_pose_time_pe_dim % 2 == 1:
            # sin/cos pairs
            self.direct_pose_time_pe_dim += 1
        self._direct_pose_time_pe_base = float(direct_pose_time_pe_base or 10000.0)
        # Optional: feed the explicit phase state into the direct head (dim = 2*contact_dim).
        self.direct_pose_use_phase_z = bool(direct_pose_use_phase_z) and int(self.contact_dim) > 0
        # How phase_z is used in the direct head input (append vs replace contact hints).
        if direct_pose_phase_z_mode is None:
            m = "concat"
        elif not isinstance(direct_pose_phase_z_mode, str):
            raise TypeError(
                "direct_pose_phase_z_mode must be a string or None; "
                "expected aliases for {'concat', 'replace_contacts'}; "
                f"got actual_type={type(direct_pose_phase_z_mode).__name__}."
            )
        else:
            m = direct_pose_phase_z_mode.strip().lower()
        if m in ("replace", "replace_contacts", "replace_contact", "phase", "phase_only", "phase_only_hint"):
            m = "replace_contacts"
        elif m in ("concat", "append", "add", "plus", "contacts+phase"):
            m = "concat"
        else:
            m = "concat"
        self.direct_pose_phase_z_mode = m
        self._direct_pose_phase_dim = int(2 * self.contact_dim) if self.direct_pose_use_phase_z else 0
        if self.direct_pose_phase_z_mode == "replace_contacts":
            if not self.direct_pose_use_phase_z:
                raise ValueError("direct_pose_phase_z_mode='replace_contacts' requires direct_pose_use_phase_z=True.")
            if self.direct_pose_meas_mode != "concat":
                raise ValueError(
                    "direct_pose_phase_z_mode='replace_contacts' is only supported with direct_pose_meas_mode='concat' "
                    "(it replaces plan+meas phase hint)."
                )
        self.direct_pose_split_enable = bool(direct_pose_split_enable) and bool(self.direct_pose_enable)
        self.direct_pose_arm_split_enable = bool(direct_pose_arm_split_enable) and bool(self.direct_pose_split_enable)
        self.direct_pose_arm_bones = direct_pose_arm_bones
        self.direct_pose_leg_terminal: Optional[nn.Module] = None
        self.direct_pose_out_nonleg: Optional[nn.Module] = None
        self.direct_pose_nonleg_proj: Optional[nn.Module] = None
        self.direct_pose_out_arm: Optional[nn.Module] = None
        self.direct_pose_out_else: Optional[nn.Module] = None
        self.direct_pose_arm_proj: Optional[nn.Module] = None
        self.direct_pose_else_proj: Optional[nn.Module] = None
        self.direct_pose_nonleg_proj_dim = max(0, int(direct_pose_nonleg_proj_dim or 0))
        self._direct_pose_init_base_seed = int(torch.initial_seed())
        self.register_buffer("direct_pose_leg_out_idx", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("direct_pose_nonleg_out_idx", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("direct_pose_arm_out_idx", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("direct_pose_else_out_idx", torch.empty(0, dtype=torch.long), persistent=True)

        # NOTE: these attributes are needed by the leg-bone parsing below. They must exist before we
        # potentially append to `direct_pose_leg_joint_idx` when direct_pose_leg_enable=True.
        self.direct_pose_leg_head: Optional[nn.Module] = None
        # New (optional): per-side routed, shared-weight leg head.
        self.direct_pose_leg_head_shared: Optional[nn.Module] = None
        # Optional: learned gate head(s) for leg omega.
        self.direct_pose_leg_gate_head: Optional[nn.Module] = None
        self.direct_pose_leg_gate_head_shared: Optional[nn.Module] = None
        self.direct_pose_leg_side_embed: Optional[nn.Module] = None
        self.direct_pose_leg_side_sign_gate_head: Optional[nn.Module] = None
        self.direct_pose_leg_rot6d_slice: Optional[slice] = None
        self.direct_pose_leg_joint_idx: list[int] = []
        self.direct_pose_leg_joint_names: list[str] = []
        # Side routing metadata (positions in the K-leg list, and contact-channel mapping).
        self.direct_pose_leg_side_routing: bool = False
        self.direct_pose_leg_contact_order: str = "lr"
        self.direct_pose_leg_contact_ch_r: int = 1
        self.direct_pose_leg_contact_ch_l: int = 0
        self.direct_pose_leg_side_k: int = 0
        self.direct_pose_leg_side_pos_r: list[int] = []
        self.direct_pose_leg_side_pos_l: list[int] = []
        self.direct_pose_leg_side_sign_gate: bool = False

        # Optional: leg-specific residual head for direct pose.
        # This provides extra capacity for lower-body joints without forcing per-joint loss tricks.
        self.direct_pose_leg_enable = bool(direct_pose_leg_enable) and bool(self.direct_pose_enable)
        self.direct_pose_leg_bones = direct_pose_leg_bones
        # How to apply the leg residual (add in 6D space vs on-manifold SO(3) composition).
        if direct_pose_leg_mode is None:
            m = "rot6d_add"
        elif not isinstance(direct_pose_leg_mode, str):
            raise TypeError(
                "direct_pose_leg_mode must be a string or None; "
                "expected aliases for {'rot6d_add', 'so3'}; "
                f"got actual_type={type(direct_pose_leg_mode).__name__}."
            )
        else:
            m = direct_pose_leg_mode.strip().lower()
        if m in ("so3", "omega", "so3_compose", "compose", "exp", "expmap", "log", "axisangle", "axis_angle"):
            m = "so3"
        elif m in ("", "rot6d_add"):
            m = "rot6d_add"
        else:
            m = "rot6d_add"
        self.direct_pose_leg_mode = m
        self.direct_pose_leg_stopgrad_main = bool(direct_pose_leg_stopgrad_main)
        self.direct_pose_leg_detach_feat = bool(direct_pose_leg_detach_feat)
        if direct_pose_leg_max_deg is None:
            mx = 0.0
        else:
            try:
                mx = float(direct_pose_leg_max_deg)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_max_deg must be a finite scalar in range [0, inf); "
                    f"got value={direct_pose_leg_max_deg!r} actual_type={type(direct_pose_leg_max_deg).__name__}."
                ) from exc
        if (not _math.isfinite(mx)) or mx < 0.0:
            raise ValueError(
                "direct_pose_leg_max_deg must be a finite scalar in range [0, inf); "
                f"got value={mx!r} actual_type={type(direct_pose_leg_max_deg).__name__}."
            )
        self.direct_pose_leg_max_rad: float = max(0.0, mx) * (_math.pi / 180.0)

        # Optional: learned gate (only meaningful for SO(3) leg mode).
        if direct_pose_leg_gate_mode is None:
            gm = "none"
        elif not isinstance(direct_pose_leg_gate_mode, str):
            raise TypeError(
                "direct_pose_leg_gate_mode must be a string or None; "
                "expected aliases for {'none', 'learned', 'scale'}; "
                f"got actual_type={type(direct_pose_leg_gate_mode).__name__}."
            )
        else:
            gm = direct_pose_leg_gate_mode.strip().lower()
        if gm in ("mlp", "net", "nn", "learn", "learned", "gate"):
            gm = "learned"
        if gm in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
            gm = "scale"
        if gm in ("", "none", "off", "disable", "disabled", "0"):
            gm = "none"
        if gm not in ("none", "learned", "scale"):
            gm = "none"
        self.direct_pose_leg_gate_mode: str = str(gm)
        if direct_pose_leg_gate_power is None:
            gp = 1.0
        else:
            try:
                gp = float(direct_pose_leg_gate_power)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_gate_power must be a finite scalar in range (0, inf); "
                    f"got value={direct_pose_leg_gate_power!r} actual_type={type(direct_pose_leg_gate_power).__name__}."
                ) from exc
        if (not _math.isfinite(gp)) or gp <= 0.0:
            raise ValueError(
                "direct_pose_leg_gate_power must be a finite scalar in range (0, inf); "
                f"got value={gp!r} actual_type={type(direct_pose_leg_gate_power).__name__}."
            )
        self.direct_pose_leg_gate_power: float = float(gp)
        if direct_pose_leg_scale_log_clip is None:
            lc = 4.0
        else:
            try:
                lc = float(direct_pose_leg_scale_log_clip)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_scale_log_clip must be a finite scalar in range (0, inf); "
                    f"got value={direct_pose_leg_scale_log_clip!r} actual_type={type(direct_pose_leg_scale_log_clip).__name__}."
                ) from exc
        if (not _math.isfinite(lc)) or lc <= 0.0:
            raise ValueError(
                "direct_pose_leg_scale_log_clip must be a finite scalar in range (0, inf); "
                f"got value={lc!r} actual_type={type(direct_pose_leg_scale_log_clip).__name__}."
            )
        self.direct_pose_leg_scale_log_clip: float = float(lc)
        if direct_pose_leg_scale_clamp_k is None:
            sk = 0.0
        else:
            try:
                sk = float(direct_pose_leg_scale_clamp_k)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_scale_clamp_k must be a finite scalar; "
                    "values <= 1 disable the clamp. "
                    f"got value={direct_pose_leg_scale_clamp_k!r} actual_type={type(direct_pose_leg_scale_clamp_k).__name__}."
                ) from exc
        if not _math.isfinite(sk):
            raise ValueError(
                "direct_pose_leg_scale_clamp_k must be a finite scalar; "
                "values <= 1 disable the clamp. "
                f"got value={sk!r} actual_type={type(direct_pose_leg_scale_clamp_k).__name__}."
            )
        if sk <= 1.0:
            sk = 0.0
        self.direct_pose_leg_scale_clamp_k: float = float(sk)
        if (not bool(self.direct_pose_leg_enable)) or (str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add") != "so3"):
            # Gate only applies to on-manifold omega composition path.
            self.direct_pose_leg_gate_mode = "none"

        # Optional: per-side routing + shared head (applied only when leg head is enabled).
        self.direct_pose_leg_side_routing = bool(direct_pose_leg_side_routing) and bool(self.direct_pose_leg_enable)
        if direct_pose_leg_contact_order is None:
            order = "lr"
        elif not isinstance(direct_pose_leg_contact_order, str):
            raise TypeError(
                "direct_pose_leg_contact_order must be a string or None; "
                "expected aliases for {'lr', 'rl'}; "
                f"got actual_type={type(direct_pose_leg_contact_order).__name__}."
            )
        else:
            order = direct_pose_leg_contact_order.strip().lower()
        if order in ("rl", "r,l", "r l"):
            self.direct_pose_leg_contact_order = "rl"
            self.direct_pose_leg_contact_ch_r, self.direct_pose_leg_contact_ch_l = 0, 1
        else:
            # Default: dataset contact channels are [L, R] (see train/io.py: load_soft_contacts_from_json).
            self.direct_pose_leg_contact_order = "lr"
            self.direct_pose_leg_contact_ch_l, self.direct_pose_leg_contact_ch_r = 0, 1
        if direct_pose_leg_side_embed_dim is None:
            side_emb = 0
        else:
            try:
                side_emb = int(direct_pose_leg_side_embed_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_side_embed_dim must be an integer scalar in range [0, inf); "
                    f"got value={direct_pose_leg_side_embed_dim!r} actual_type={type(direct_pose_leg_side_embed_dim).__name__}."
                ) from exc
        self.direct_pose_leg_side_embed_dim: int = max(0, int(side_emb))
        # Optional: cross-leg context via plan_other appended per side.
        self.direct_pose_leg_side_plan_other: bool = bool(direct_pose_leg_side_plan_other) and bool(self.direct_pose_leg_side_routing)
        self.direct_pose_leg_side_plan_other_dim: int = 1 if bool(self.direct_pose_leg_side_plan_other) else 0
        # Optional: cross-leg phase context appended per side (2D sin/cos).
        self.direct_pose_leg_side_phase_other: bool = (
            bool(direct_pose_leg_side_phase_other)
            and bool(self.direct_pose_leg_side_routing)
            and bool(self.direct_pose_use_phase_z)
        )
        self.direct_pose_leg_side_phase_other_dim: int = 2 if bool(self.direct_pose_leg_side_phase_other) else 0
        # Optional: explicit relative phase per side (2D sin/cos).
        self.direct_pose_leg_side_phase_rel: bool = (
            bool(direct_pose_leg_side_phase_rel)
            and bool(self.direct_pose_leg_side_routing)
            and bool(self.direct_pose_use_phase_z)
        )
        self.direct_pose_leg_side_phase_rel_dim: int = 2 if bool(self.direct_pose_leg_side_phase_rel) else 0
        # Optional: extra per-side cue for the routed shared leg head.
        if direct_pose_leg_side_cue is None:
            cue = "none"
        elif not isinstance(direct_pose_leg_side_cue, str):
            raise TypeError(
                "direct_pose_leg_side_cue must be a string or None; "
                "expected aliases for {'none', 'phase_event_age'}; "
                f"got actual_type={type(direct_pose_leg_side_cue).__name__}."
            )
        else:
            cue = direct_pose_leg_side_cue.strip().lower()
        if cue in ("", "none", "off", "disable", "disabled"):
            cue = "none"
        elif cue in ("age", "event_age", "eventage", "phase_age", "phase_event_age", "phaseeventage"):
            cue = "phase_event_age"
        elif cue in ("hazard", "td_hazard", "tdhazard", "hazard_acc", "td_hazard_acc", "tdhazard_acc", "hzacc"):
            raise ValueError("direct_pose_leg_side_cue='td_hazard_acc' has been retired; use 'none' or 'phase_event_age'.")
        else:
            cue = "none"
        self.direct_pose_leg_side_cue: str = str(cue)
        if direct_pose_leg_side_cue_tau is None:
            tau = 30.0
        else:
            try:
                tau = float(direct_pose_leg_side_cue_tau)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_side_cue_tau must be a finite scalar in range (0, inf); "
                    f"got value={direct_pose_leg_side_cue_tau!r} actual_type={type(direct_pose_leg_side_cue_tau).__name__}."
                ) from exc
        if (not _math.isfinite(tau)) or tau <= 0.0:
            raise ValueError(
                "direct_pose_leg_side_cue_tau must be a finite scalar in range (0, inf); "
                f"got value={tau!r} actual_type={type(direct_pose_leg_side_cue_tau).__name__}."
            )
        self.direct_pose_leg_side_cue_tau: float = float(tau)
        self.direct_pose_leg_side_cue_dim: int = 1 if cue != "none" else 0
        # Optional: per-side sign gate is only meaningful when side routing is enabled.
        self.direct_pose_leg_side_sign_gate = bool(direct_pose_leg_side_sign_gate) and bool(self.direct_pose_leg_side_routing)
        # Optional: rank-1 coupling is only meaningful when side routing is enabled.
        self.direct_pose_leg_side_rank1 = bool(direct_pose_leg_side_rank1) and bool(self.direct_pose_leg_side_routing)
        if bool(self.direct_pose_leg_side_rank1) and bool(self.direct_pose_leg_side_sign_gate):
            raise ValueError("direct_pose_leg_side_rank1 is incompatible with direct_pose_leg_side_sign_gate (pick one).")
        self._init_direct_pose_routing_metadata(
            bone_names=bone_names,
            output_layout=output_layout,
        )
        self.lambda_fusion_enable = bool(lambda_fusion_enable)
        self.lambda_fusion_mode = str(lambda_fusion_mode or "per_joint").lower().strip()
        if self.lambda_fusion_mode not in ("global", "per_joint"):
            self.lambda_fusion_mode = "per_joint"
        self.lambda_fusion_hidden = max(8, int(lambda_fusion_hidden or 0))
        self._lambda_fusion_dropout = float(lambda_fusion_dropout)
        self.lambda_fusion_detach_err = bool(lambda_fusion_detach_err)
        self._lambda_fusion_logit_init = float(lambda_fusion_logit_init)
        self.lambda_fusion_use_rollout_step = bool(lambda_fusion_use_rollout_step)

        plan_inject_dim = 0
        if self.contact_plan_enable:
            if self.contact_plan_inject == "contacts":
                plan_inject_dim = int(self.contact_dim)
            elif self.contact_plan_inject == "plan_z":
                plan_inject_dim = int(self.contact_plan_hidden)
        input_dim = self.in_state_dim + self.cond_dim + plan_inject_dim
        enc_depth = max(1, int(num_layers))
        self._encoder_residual = bool(enc_depth > 2)
        if not self._encoder_residual:
            self.shared_encoder = build_mlp(
                input_dim,
                hidden_dim,
                num_layers=max(1, enc_depth),
                activation=nn.ReLU,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            stem = build_mlp(
                input_dim,
                hidden_dim,
                num_layers=2,  # keep baseline encoder as the stem
                activation=nn.ReLU,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            enc_layers = list(stem)
            for _ in range(max(1, enc_depth - 2)):
                enc_layers.append(
                    _ResidualMLPBlock(
                        hidden_dim,
                        activation=nn.ReLU,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm,
                        zero_init_last=True,
                    )
                )
            self.shared_encoder = nn.Sequential(*enc_layers)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self._pasa_heads = max(1, int(num_heads))
        if hidden_dim % self._pasa_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {self._pasa_heads}.")
        self._pasa_dhead = hidden_dim // self._pasa_heads
        self._pasa_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_lnq = nn.LayerNorm(hidden_dim)
        self._pasa_film = _CondFiLM(cond_dim=self.cond_dim, hidden_dim=128, film_dim=hidden_dim)
        self.coupling_norm = nn.LayerNorm(hidden_dim)
        self.input_clip = 16.0

        self.motion_head = build_mlp(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            activation=nn.ReLU,
            dropout=dropout,
            final_dim=out_motion_dim,
        )
        self.period_encoder = nn.Linear(self.period_dim, hidden_dim) if self.period_dim > 0 else None
        self._build_contact_plan_modules()
        self._build_direct_pose_modules()
        self._build_lambda_and_so3_aux_heads(
            output_layout=output_layout,
            bone_names=bone_names,
            so3_corr_hidden=so3_corr_hidden,
            so3_corr_dropout=so3_corr_dropout,
            so3_corr_gate_logit_init=so3_corr_gate_logit_init,
        )

        # Low-risk per-bone residual adapters:
        # - initial adapter output == 0 (zero-init last Linear) so behavior matches baseline;
        # - α starts small-but-nonzero so the adapter branch can receive gradients immediately.
        default_bones = ('thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r')
        adapter_bones = list(residual_adapter_bones) if residual_adapter_bones is not None else list(default_bones)
        self._bone_adapter_slices: list[slice] = []
        self._bone_adapter_names: list[str] = []
        self._bone_adapters = nn.ModuleList()
        try:
            self._init_bone_residual_adapters(
                bone_names=bone_names,
                output_layout=output_layout,
                target_bones=adapter_bones,
                hidden_dim=int(residual_adapter_hidden),
                dropout=float(dropout if residual_adapter_dropout is None else residual_adapter_dropout),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            # Keep adapters disabled if metadata is missing/mismatched.
            self._bone_adapter_slices = []
            self._bone_adapter_names = []
            self._bone_adapters = nn.ModuleList()

        # Optional frozen encoder from预训练，用于提供 soft hint（接触提示 embedding）
        self.frozen_encoder: Optional['MotionEncoder'] = None
        self.frozen_period_head: Optional['PeriodHead'] = None
        self.frozen_contact_head: Optional[nn.Module] = None

    def _init_direct_pose_routing_metadata(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
    ) -> None:
        if self.direct_pose_leg_enable:
            rot_sl = resolve_rot6d_slice(
                output_layout,
                total_dim=self.out_motion_dim,
            )
            if isinstance(rot_sl, slice):
                self.direct_pose_leg_rot6d_slice = rot_sl
            joint_count = infer_rot_joint_count(rot_sl)
            leg_idx, leg_names = _resolve_joint_spec_indices(
                getattr(self, "direct_pose_leg_bones", None),
                default_items=("ball_r", "ball_l", "foot_r", "foot_l", "calf_r", "calf_l", "thigh_r", "thigh_l"),
                bone_names=bone_names,
                joint_count=joint_count,
                collect_names=True,
            )
            self.direct_pose_leg_joint_idx.extend(int(i) for i in leg_idx)
            self.direct_pose_leg_joint_names.extend(str(name) for name in leg_names)
            if self.direct_pose_leg_joint_idx:
                leg_idx_tensor = torch.as_tensor(self.direct_pose_leg_joint_idx, dtype=torch.long)
                try:
                    self.register_buffer(
                        "direct_pose_leg_joint_idx_tensor",
                        leg_idx_tensor,
                        persistent=True,
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "direct_pose leg routing metadata registration failed: "
                        "field='direct_pose_leg_joint_idx_tensor' expected a persistent torch.LongTensor with "
                        f"shape=(K_leg,) and dtype=torch.int64; got joint_indices={self.direct_pose_leg_joint_idx!r}, "
                        f"tensor_shape={tuple(int(v) for v in leg_idx_tensor.shape)!r}, "
                        f"tensor_dtype={leg_idx_tensor.dtype}."
                    ) from exc

        if bool(self.direct_pose_split_enable):
            if str(getattr(self, "direct_pose_meas_mode", "concat") or "concat").strip().lower() != "concat":
                raise ValueError("direct_pose_split_enable currently supports direct_pose_meas_mode='concat' only.")

            split_leg_joint_idx = list(getattr(self, "direct_pose_leg_joint_idx", None) or [])
            if not split_leg_joint_idx:
                rot_sl_split = resolve_rot6d_slice(
                    output_layout,
                    total_dim=self.out_motion_dim,
                )
                joint_count_split = infer_rot_joint_count(rot_sl_split)
                split_leg_joint_idx, _ = _resolve_joint_spec_indices(
                    getattr(self, "direct_pose_leg_bones", None),
                    default_items=DEFAULT_DIRECT_POSE_LEG_BONES,
                    bone_names=bone_names,
                    joint_count=joint_count_split,
                )

            if split_leg_joint_idx:
                self.direct_pose_leg_joint_idx = [int(i) for i in split_leg_joint_idx]
                if not self.direct_pose_leg_joint_names:
                    try:
                        if bone_names is not None:
                            self.direct_pose_leg_joint_names = [
                                str(bone_names[int(i)])
                                for i in self.direct_pose_leg_joint_idx
                                if 0 <= int(i) < len(bone_names)
                            ]
                    except (TypeError, IndexError):
                        pass
                leg_idx_tensor = torch.as_tensor(self.direct_pose_leg_joint_idx, dtype=torch.long)
                try:
                    if hasattr(self, "direct_pose_leg_joint_idx_tensor"):
                        self.direct_pose_leg_joint_idx_tensor = leg_idx_tensor
                    else:
                        self.register_buffer("direct_pose_leg_joint_idx_tensor", leg_idx_tensor, persistent=True)
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    action = "update" if hasattr(self, "direct_pose_leg_joint_idx_tensor") else "register"
                    raise RuntimeError(
                        "direct_pose split leg routing metadata registration failed: "
                        "field='direct_pose_leg_joint_idx_tensor' expected a torch.LongTensor with "
                        f"shape=(K_leg,) during split metadata {action}; got joint_indices={self.direct_pose_leg_joint_idx!r}, "
                        f"tensor_shape={tuple(int(v) for v in leg_idx_tensor.shape)!r}, "
                        f"tensor_dtype={leg_idx_tensor.dtype}."
                    ) from exc

            rot_sl = resolve_rot6d_slice(
                output_layout,
                total_dim=self.out_motion_dim,
            )
            if (not isinstance(rot_sl, slice)) or rot_sl.start is None or rot_sl.stop is None:
                raise ValueError("direct_pose_split_enable requires a valid BoneRotations6D output slice.")

            out_dim_total = int(self.out_motion_dim)
            rot_start = int(rot_sl.start)
            rot_stop = int(rot_sl.stop)
            rot_len = max(0, rot_stop - rot_start)
            if rot_len <= 0 or (rot_len % 6) != 0:
                raise ValueError(
                    f"direct_pose_split_enable requires BoneRotations6D dim to be a positive multiple of 6 (got {rot_len})."
                )
            joint_count_rot = int(rot_len // 6)

            def build_split_out_index(
                joint_indices: Sequence[int],
                *,
                base_mask: Optional[torch.Tensor] = None,
                empty_error: str,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                dim_mask = torch.zeros((out_dim_total,), dtype=torch.bool)
                for j_idx in joint_indices:
                    jj = int(j_idx)
                    if 0 <= jj < joint_count_rot:
                        dim_start = int(rot_start + jj * 6)
                        dim_stop = int(dim_start + 6)
                        if 0 <= dim_start and dim_stop <= out_dim_total:
                            dim_mask[dim_start:dim_stop] = True
                if torch.is_tensor(base_mask):
                    dim_mask = dim_mask & base_mask
                if not bool(dim_mask.any().item()):
                    raise ValueError(empty_error)
                out_idx = torch.nonzero(dim_mask, as_tuple=False).flatten().to(dtype=torch.long)
                return dim_mask, out_idx

            leg_dim_mask, leg_out_idx = build_split_out_index(
                split_leg_joint_idx,
                empty_error="direct_pose_split_enable resolved empty leg output dims; check direct_pose_leg_bones mapping.",
            )
            nonleg_dim_mask = ~leg_dim_mask
            if not bool(nonleg_dim_mask.any().item()):
                raise ValueError("direct_pose_split_enable resolved empty non-leg output dims.")
            nonleg_out_idx = torch.nonzero(nonleg_dim_mask, as_tuple=False).flatten().to(dtype=torch.long)
            if int(leg_out_idx.numel() + nonleg_out_idx.numel()) != int(out_dim_total):
                raise ValueError("direct split head index coverage mismatch (D_leg + D_nonleg != out_motion_dim).")
            self.direct_pose_leg_out_idx = leg_out_idx
            self.direct_pose_nonleg_out_idx = nonleg_out_idx
            if bool(getattr(self, "direct_pose_arm_split_enable", False)):
                arm_joint_idx, _ = _resolve_joint_spec_indices(
                    getattr(self, "direct_pose_arm_bones", None),
                    default_items=STAGE6_3WAY_ARMCHAIN_BONES,
                    bone_names=bone_names,
                    joint_count=joint_count_rot,
                )
                arm_dim_mask, arm_out_idx = build_split_out_index(
                    arm_joint_idx,
                    base_mask=nonleg_dim_mask,
                    empty_error=(
                        "direct_pose_arm_split_enable resolved empty arm output dims; "
                        "check direct_pose_arm_bones mapping."
                    ),
                )
                else_dim_mask = nonleg_dim_mask & (~arm_dim_mask)
                if not bool(else_dim_mask.any().item()):
                    raise ValueError("direct_pose_arm_split_enable resolved empty else output dims.")
                else_out_idx = torch.nonzero(else_dim_mask, as_tuple=False).flatten().to(dtype=torch.long)
                if int(arm_out_idx.numel() + else_out_idx.numel()) != int(nonleg_out_idx.numel()):
                    raise ValueError("direct arm/else split index coverage mismatch (D_arm + D_else != D_nonleg).")
                self.direct_pose_arm_out_idx = arm_out_idx
                self.direct_pose_else_out_idx = else_out_idx

        if not bool(self.direct_pose_leg_side_routing):
            return
        if str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip() != "so3":
            raise ValueError("direct_pose_leg_side_routing currently supports only direct_pose_leg_mode='so3'.")
        if int(getattr(self, "contact_dim", 0) or 0) != 2:
            raise ValueError(
                f"direct_pose_leg_side_routing requires contact_dim==2 (got contact_dim={int(getattr(self, 'contact_dim', 0) or 0)})."
            )
        if not self.direct_pose_leg_joint_names:
            raise ValueError(
                "direct_pose_leg_side_routing requires leg joint names (direct_pose_leg_bones should be names, not indices)."
            )
        joint_names_lower = [str(name).lower() for name in self.direct_pose_leg_joint_names]
        pos_r = [idx for idx, name in enumerate(joint_names_lower) if name.endswith(("_r", "right"))]
        pos_l = [idx for idx, name in enumerate(joint_names_lower) if name.endswith(("_l", "left"))]
        if not pos_r or not pos_l:
            raise ValueError(
                f"direct_pose_leg_side_routing expects both _r and _l joints; got names={self.direct_pose_leg_joint_names}."
            )
        if len(pos_r) != len(pos_l):
            raise ValueError(
                f"direct_pose_leg_side_routing expects symmetric joint counts per side; got n_r={len(pos_r)} n_l={len(pos_l)} "
                f"(names={self.direct_pose_leg_joint_names})."
            )
        if (len(pos_r) + len(pos_l)) != len(joint_names_lower):
            unknown = [
                self.direct_pose_leg_joint_names[idx]
                for idx in range(len(joint_names_lower))
                if idx not in pos_r and idx not in pos_l
            ]
            raise ValueError(
                "direct_pose_leg_side_routing expects all leg joints to be side-tagged with _r/_l; "
                f"unknown={unknown} (names={self.direct_pose_leg_joint_names})."
            )
        self.direct_pose_leg_side_k = int(len(pos_r))
        self.direct_pose_leg_side_pos_r = list(pos_r)
        self.direct_pose_leg_side_pos_l = list(pos_l)
        pos_r_tensor = torch.as_tensor(self.direct_pose_leg_side_pos_r, dtype=torch.long)
        pos_l_tensor = torch.as_tensor(self.direct_pose_leg_side_pos_l, dtype=torch.long)
        try:
            self.register_buffer(
                "direct_pose_leg_side_pos_r_tensor",
                pos_r_tensor,
                persistent=True,
            )
            self.register_buffer(
                "direct_pose_leg_side_pos_l_tensor",
                pos_l_tensor,
                persistent=True,
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "direct_pose side routing position buffer registration failed: "
                "fields=('direct_pose_leg_side_pos_r_tensor', 'direct_pose_leg_side_pos_l_tensor') expected "
                "persistent torch.LongTensor buffers with shape=(K_side,); "
                f"got pos_r={self.direct_pose_leg_side_pos_r!r}, pos_l={self.direct_pose_leg_side_pos_l!r}, "
                f"pos_r_shape={tuple(int(v) for v in pos_r_tensor.shape)!r}, "
                f"pos_l_shape={tuple(int(v) for v in pos_l_tensor.shape)!r}."
            ) from exc
        if int(getattr(self, "direct_pose_leg_side_embed_dim", 0) or 0) > 0:
            self.direct_pose_leg_side_embed = nn.Embedding(2, int(self.direct_pose_leg_side_embed_dim))
            weight = getattr(self.direct_pose_leg_side_embed, "weight", None)
            if not torch.is_tensor(weight):
                raise TypeError(
                    "direct_pose side embedding deterministic init failed: "
                    "field='direct_pose_leg_side_embed.weight' expected a torch.Tensor with shape=(2, side_embed_dim); "
                    f"got module_type={type(self.direct_pose_leg_side_embed).__name__}, weight_type={type(weight).__name__}."
                )
            try:
                with torch.no_grad():
                    weight.zero_()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "direct_pose side embedding deterministic init failed: "
                    "field='direct_pose_leg_side_embed.weight' expected zero init with shape=(2, side_embed_dim); "
                    f"got side_embed_dim={int(self.direct_pose_leg_side_embed_dim)}, "
                    f"weight_shape={tuple(int(v) for v in weight.shape)!r}, weight_dtype={weight.dtype}."
                ) from exc

    def _build_contact_plan_modules(self) -> None:
        self.contact_plan_cell: Optional[nn.GRUCell] = None
        self.contact_plan_head: Optional[nn.Module] = None
        self.contact_plan_time_head: Optional[nn.Module] = None
        self.contact_plan_phase_head: Optional[nn.Module] = None
        self.contact_plan_init_z: Optional[nn.Parameter] = None
        self.contact_plan_init_head: Optional[nn.Module] = None
        self._contact_plan_init_obs_dim = 0
        self.event_clock_gate: Optional[PeriodicityGate] = None
        self.event_clock_corrector: Optional[PlanZCorrector] = None
        if not self.contact_plan_enable:
            return

        h_plan = int(self.contact_plan_hidden)
        self.contact_plan_cell = nn.GRUCell(self.cond_dim, h_plan)
        self.contact_plan_init_z = nn.Parameter(torch.zeros(h_plan), requires_grad=True)
        if self.contact_plan_init_mode in ("obs", "learnable+obs"):
            obs_dim = int(self.contact_dim + self.angvel_dim + self.pose_hist_dim)
            self._contact_plan_init_obs_dim = obs_dim
            if obs_dim > 0:
                h_init = int(self.contact_plan_init_hidden or h_plan)
                drop = float(self._contact_plan_init_dropout)
                self.contact_plan_init_head = nn.Sequential(
                    nn.LayerNorm(obs_dim),
                    nn.Linear(obs_dim, h_init),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    nn.Linear(h_init, h_plan),
                )
                last = self.contact_plan_init_head[-1]
                if not isinstance(last, nn.Linear):
                    raise TypeError(
                        "contact_plan init head deterministic init failed: "
                        "field='contact_plan_init_head[-1]' expected nn.Linear final layer for zero init; "
                        f"got module_type={type(last).__name__}."
                    )
                bias_shape = tuple(int(v) for v in last.bias.shape) if torch.is_tensor(last.bias) else None
                try:
                    with torch.no_grad():
                        last.weight.zero_()
                        if last.bias is not None:
                            last.bias.zero_()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "contact_plan init head deterministic init failed: "
                        "field='contact_plan_init_head[-1]' expected final weight/bias tensors that can be zero-initialized; "
                        f"weight_shape={tuple(int(v) for v in last.weight.shape)!r}, "
                        f"bias_shape={bias_shape!r}."
                    ) from exc
        self.contact_plan_head = nn.Sequential(
            nn.LayerNorm(h_plan),
            nn.Linear(h_plan, h_plan),
            nn.ReLU(),
            nn.Dropout(self._contact_plan_dropout),
            nn.Linear(h_plan, int(self._contact_plan_logits_dim)),
        )
        if self.contact_plan_time_pe_dim > 0:
            self.contact_plan_time_head = nn.Linear(self.contact_plan_time_pe_dim, int(self._contact_plan_logits_dim))
            bias = getattr(self.contact_plan_time_head, "bias", None)
            bias_shape = tuple(int(v) for v in bias.shape) if torch.is_tensor(bias) else None
            try:
                with torch.no_grad():
                    self.contact_plan_time_head.weight.zero_()
                    if bias is not None:
                        self.contact_plan_time_head.bias.zero_()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "contact_plan time head deterministic init failed: "
                    "field='contact_plan_time_head' expected zero-initializable weight/bias tensors for time bias construction; "
                    f"weight_shape={tuple(int(v) for v in self.contact_plan_time_head.weight.shape)!r}, "
                    f"bias_shape={bias_shape!r}, "
                    f"time_pe_dim={int(self.contact_plan_time_pe_dim)}, logits_dim={int(self._contact_plan_logits_dim)}."
                ) from exc
        if self.use_event_clock:
            self.event_clock_gate = PeriodicityGate(
                contact_dim=int(self.contact_dim),
                period_feat_dim=int(self.period_dim),
                hidden_dim=int(self.event_clock_gate_hidden_dim),
            )
            self.event_clock_corrector = PlanZCorrector(
                plan_z_dim=int(h_plan),
                contact_dim=int(self.contact_dim),
                period_feat_dim=int(self.period_dim),
                hidden_dim=int(self.event_clock_hidden_dim),
                max_delta=float(self.event_clock_max_delta),
            )

    def _build_direct_pose_modules(self) -> None:
        self.direct_pose_head: Optional[nn.Module] = None
        if not (self.contact_plan_enable and self.direct_pose_enable):
            return
        direct_pose_stream_state = torch.random.get_rng_state()

        want_meas = self.direct_pose_meas_mode == "concat"
        base_dim = int(self.cond_dim)
        if self.direct_pose_feat_source in ("hidden", "hidden_pre"):
            base_dim = int(self.hidden_dim)
        elif self.direct_pose_feat_source in ("cond+hidden", "cond+hidden_pre"):
            base_dim = int(self.cond_dim + self.hidden_dim)
        time_dim = int(getattr(self, "direct_pose_time_pe_dim", 0) or 0)
        phase_dim = int(getattr(self, "_direct_pose_phase_dim", 0) or 0)
        if self.direct_pose_phase_z_mode == "replace_contacts":
            in_dim = int(base_dim + time_dim + phase_dim)
        else:
            in_dim = int(base_dim + self.contact_dim + (self.contact_dim if want_meas else 0) + time_dim + phase_dim)
        hid = int(self.direct_pose_hidden)
        drop = float(self._direct_pose_dropout)

        split_leg_terminal_out_dim: Optional[int] = None
        split_stream_gen: Optional[torch.Generator] = None
        if bool(getattr(self, "direct_pose_split_enable", False)):
            split_state = self._direct_pose_split_state()
            if split_state is None:
                raise ValueError("direct_pose_split_enable requires split output index buffers.")
            leg_out_dim = int(split_state["idx_leg"].numel())
            nonleg_out_dim = int(split_state["idx_nonleg"].numel())
            if leg_out_dim <= 0 or nonleg_out_dim <= 0:
                raise ValueError("direct_pose_split_enable requires non-empty leg/non-leg output indices.")
            if int(leg_out_dim + nonleg_out_dim) != int(self.out_motion_dim):
                raise ValueError(
                    f"direct_pose_split_enable index mismatch: D_leg={leg_out_dim} D_nonleg={nonleg_out_dim} "
                    f"out_motion_dim={int(self.out_motion_dim)}"
                )
            self.direct_pose_head = nn.Sequential(
                self._build_seeded_linear(
                    in_dim,
                    hid,
                    branch_name="direct_pose_head.fc0",
                ),
                nn.ReLU(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                self._build_seeded_linear(
                    hid,
                    hid,
                    branch_name="direct_pose_head.fc1",
                ),
                nn.ReLU(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
            )
            split_leg_terminal_out_dim = int(leg_out_dim)
            split_stream_gen = self._new_generator_from_state(direct_pose_stream_state)
            self._advance_linear_stream_(split_stream_gen, in_dim, hid)
            self._advance_linear_stream_(split_stream_gen, hid, hid)
            self._advance_linear_stream_(split_stream_gen, hid, leg_out_dim)
            if bool(split_state["arm_split"]):
                arm_out_dim = int(split_state["idx_arm"].numel())
                else_out_dim = int(split_state["idx_else"].numel())
                if arm_out_dim <= 0 or else_out_dim <= 0:
                    raise ValueError("direct_pose_arm_split_enable requires non-empty arm/else output indices.")
                if int(arm_out_dim + else_out_dim) != int(nonleg_out_dim):
                    raise ValueError(
                        f"direct arm/else split index mismatch: D_arm={arm_out_dim} D_else={else_out_dim} "
                        f"D_nonleg={nonleg_out_dim}"
                    )
                proj_dim = int(getattr(self, "direct_pose_nonleg_proj_dim", 0) or 0)
                self.direct_pose_out_arm, self.direct_pose_arm_proj = self._build_split_head_branch(
                    trunk_dim=hid,
                    out_dim=arm_out_dim,
                    proj_dim=proj_dim,
                    out_name="direct_pose_out_arm",
                    proj_name="direct_pose_arm_proj",
                    generator=split_stream_gen,
                )
                self.direct_pose_out_else, self.direct_pose_else_proj = self._build_split_head_branch(
                    trunk_dim=hid,
                    out_dim=else_out_dim,
                    proj_dim=proj_dim,
                    out_name="direct_pose_out_else",
                    proj_name="direct_pose_else_proj",
                    generator=split_stream_gen,
                )
            else:
                proj_dim = int(getattr(self, "direct_pose_nonleg_proj_dim", 0) or 0)
                self.direct_pose_out_nonleg, self.direct_pose_nonleg_proj = self._build_split_head_branch(
                    trunk_dim=hid,
                    out_dim=nonleg_out_dim,
                    proj_dim=proj_dim,
                    out_name="direct_pose_out_nonleg",
                    proj_name="direct_pose_nonleg_proj",
                    generator=split_stream_gen,
                )
        else:
            out_dim = int(self.out_motion_dim)
            if self.direct_pose_meas_mode == "mode_select":
                out_dim = int(out_dim) * 2
            self.direct_pose_head = nn.Sequential(
                self._build_seeded_linear(
                    in_dim,
                    hid,
                    branch_name="direct_pose_head.fc0",
                ),
                nn.ReLU(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                self._build_seeded_linear(
                    hid,
                    hid,
                    branch_name="direct_pose_head.fc1",
                ),
                nn.ReLU(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                self._build_seeded_linear(
                    hid,
                    int(out_dim),
                    branch_name="direct_pose_head.out",
                ),
            )

        if bool(getattr(self, "direct_pose_leg_enable", False)) and getattr(self, "direct_pose_leg_joint_idx", None):
            leg_k = int(len(self.direct_pose_leg_joint_idx))
            leg_mode = str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip()
            leg_out = (3 if leg_mode == "so3" else 6) * int(leg_k)
            if leg_out > 0:
                self.direct_pose_leg_head = nn.Sequential(
                    self._build_linear_from_generator(
                        in_dim,
                        hid,
                        generator=split_stream_gen,
                    ) if split_stream_gen is not None else self._build_seeded_linear(
                        in_dim,
                        hid,
                        branch_name="direct_pose_leg_head.fc0",
                    ),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    self._build_linear_from_generator(
                        hid,
                        hid,
                        generator=split_stream_gen,
                    ) if split_stream_gen is not None else self._build_seeded_linear(
                        hid,
                        hid,
                        branch_name="direct_pose_leg_head.fc1",
                    ),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    self._build_linear_from_generator(
                        hid,
                        int(leg_out),
                        generator=split_stream_gen,
                        init_fn=self._init_zero_linear_,
                    ) if split_stream_gen is not None else self._build_seeded_linear(
                        hid,
                        int(leg_out),
                        branch_name="direct_pose_leg_head.out",
                        init_fn=self._init_zero_linear_,
                    ),
                )
                gate_mode = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                if gate_mode in ("learned", "scale"):
                    gate_out = int(leg_k)
                    self.direct_pose_leg_gate_head = nn.Sequential(
                        self._build_linear_from_generator(
                            in_dim,
                            hid,
                            generator=split_stream_gen,
                        ) if split_stream_gen is not None else self._build_seeded_linear(
                            in_dim,
                            hid,
                            branch_name="direct_pose_leg_gate_head.fc0",
                        ),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        self._build_linear_from_generator(
                            hid,
                            hid,
                            generator=split_stream_gen,
                        ) if split_stream_gen is not None else self._build_seeded_linear(
                            hid,
                            hid,
                            branch_name="direct_pose_leg_gate_head.fc1",
                        ),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        self._build_linear_from_generator(
                            hid,
                            int(gate_out),
                            generator=split_stream_gen,
                            init_fn=self._make_zero_linear_init_(bias_value=(2.0 if gate_mode == "learned" else 0.0)),
                        ) if split_stream_gen is not None else self._build_seeded_linear(
                            hid,
                            int(gate_out),
                            branch_name="direct_pose_leg_gate_head.out",
                            init_fn=self._make_zero_linear_init_(bias_value=(2.0 if gate_mode == "learned" else 0.0)),
                        ),
                    )

            if bool(getattr(self, "direct_pose_leg_side_routing", False)) and int(getattr(self, "direct_pose_leg_side_k", 0) or 0) > 0:
                leg_side_k = int(getattr(self, "direct_pose_leg_side_k", 0) or 0)
                base_leg_dim = int(base_dim + time_dim)
                side_emb_dim = int(getattr(self, "direct_pose_leg_side_embed_dim", 0) or 0)
                phase_side_dim = 2 if bool(getattr(self, "direct_pose_use_phase_z", False)) else 0
                meas_side_dim = 1 if want_meas else 0
                plan_other_side_dim = int(getattr(self, "direct_pose_leg_side_plan_other_dim", 0) or 0)
                phase_other_side_dim = int(getattr(self, "direct_pose_leg_side_phase_other_dim", 0) or 0)
                phase_rel_side_dim = int(getattr(self, "direct_pose_leg_side_phase_rel_dim", 0) or 0)
                cue_side_dim = int(getattr(self, "direct_pose_leg_side_cue_dim", 0) or 0)
                leg_in_dim = int(
                    base_leg_dim
                    + 1
                    + meas_side_dim
                    + phase_side_dim
                    + plan_other_side_dim
                    + phase_other_side_dim
                    + phase_rel_side_dim
                    + cue_side_dim
                    + side_emb_dim
                )
                leg_out_side = 3 + int(leg_side_k) if bool(getattr(self, "direct_pose_leg_side_rank1", False)) else 3 * int(leg_side_k)
                if leg_in_dim > 0 and leg_out_side > 0:
                    self.direct_pose_leg_head_shared = nn.Sequential(
                        self._build_seeded_linear(
                            leg_in_dim,
                            hid,
                            branch_name="direct_pose_leg_head_shared.fc0",
                        ),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        self._build_seeded_linear(
                            hid,
                            hid,
                            branch_name="direct_pose_leg_head_shared.fc1",
                        ),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        self._build_seeded_linear(
                            hid,
                            int(leg_out_side),
                            branch_name="direct_pose_leg_head_shared.out",
                            init_fn=self._init_zero_linear_,
                        ),
                    )
                    gate_mode = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                    if gate_mode in ("learned", "scale"):
                        gate_out = int(leg_side_k)
                        self.direct_pose_leg_gate_head_shared = nn.Sequential(
                            self._build_seeded_linear(
                                leg_in_dim,
                                hid,
                                branch_name="direct_pose_leg_gate_head_shared.fc0",
                            ),
                            nn.ReLU(),
                            nn.Dropout(drop) if drop > 0 else nn.Identity(),
                            self._build_seeded_linear(
                                hid,
                                hid,
                                branch_name="direct_pose_leg_gate_head_shared.fc1",
                            ),
                            nn.ReLU(),
                            nn.Dropout(drop) if drop > 0 else nn.Identity(),
                            self._build_seeded_linear(
                                hid,
                                int(gate_out),
                                branch_name="direct_pose_leg_gate_head_shared.out",
                                init_fn=self._make_zero_linear_init_(bias_value=(2.0 if gate_mode == "learned" else 0.0)),
                            ),
                        )
                    if bool(getattr(self, "direct_pose_leg_side_sign_gate", False)):
                        h_gate = max(8, int(hid // 4))
                        self.direct_pose_leg_side_sign_gate_head = nn.Sequential(
                            self._build_seeded_linear(
                                leg_in_dim,
                                h_gate,
                                branch_name="direct_pose_leg_side_sign_gate_head.fc0",
                            ),
                            nn.ReLU(),
                            self._build_seeded_linear(
                                h_gate,
                                1,
                                branch_name="direct_pose_leg_side_sign_gate_head.out",
                                init_fn=self._make_zero_linear_init_(bias_value=2.0),
                            ),
                        )

        if split_leg_terminal_out_dim is not None:
            self.direct_pose_leg_terminal = self._build_direct_pose_terminal_block(
                trunk_dim=hid,
                out_dim=int(split_leg_terminal_out_dim),
                drop=drop,
            )

    def _build_lambda_and_so3_aux_heads(
        self,
        *,
        output_layout: Optional[Dict[str, Any]],
        bone_names: Optional[Sequence[str]],
        so3_corr_hidden: int,
        so3_corr_dropout: float,
        so3_corr_gate_logit_init: float,
    ) -> None:
        self.lambda_fusion_joint_count = 0
        self.lambda_fusion_head: Optional[nn.Module] = None
        if self.lambda_fusion_enable:
            try:
                rot_sl = resolve_rot6d_slice(
                    output_layout,
                    total_dim=self.out_motion_dim,
                )
                self.lambda_fusion_joint_count = infer_rot_joint_count(rot_sl)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "lambda_fusion rot6d layout resolution failed: "
                    "field='output_layout' expected a valid BoneRotations6D slice so lambda_fusion_joint_count can be inferred; "
                    f"out_motion_dim={int(self.out_motion_dim)}, output_layout_type={type(output_layout).__name__}, "
                    f"output_layout={output_layout!r}."
                ) from exc

            out_dim = 1 if self.lambda_fusion_mode == "global" else int(self.lambda_fusion_joint_count)
            if int(out_dim) > 0:
                use_rollout_step = bool(getattr(self, "lambda_fusion_use_rollout_step", False))
                in_dim = int(self.hidden_dim + (self.contact_dim if self.contact_plan_enable else 0) + (1 if use_rollout_step else 0))
                h_mid = max(8, int(self.lambda_fusion_hidden))
                drop = float(self._lambda_fusion_dropout)
                self.lambda_fusion_head = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, h_mid),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    nn.Linear(h_mid, int(out_dim)),
                )
                last = self.lambda_fusion_head[-1]
                if not isinstance(last, nn.Linear):
                    raise TypeError(
                        "lambda_fusion deterministic init failed: "
                        "field='lambda_fusion_head[-1]' expected nn.Linear final layer; "
                        f"got module_type={type(last).__name__}."
                    )
                bias_shape = tuple(int(v) for v in last.bias.shape) if torch.is_tensor(last.bias) else None
                try:
                    with torch.no_grad():
                        last.weight.zero_()
                        if last.bias is not None:
                            last.bias.fill_(float(self._lambda_fusion_logit_init))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "lambda_fusion deterministic init failed: "
                        "field='lambda_fusion_head[-1]' expected zero-initializable weight and optional bias logit init; "
                        f"weight_shape={tuple(int(v) for v in last.weight.shape)!r}, "
                        f"bias_shape={bias_shape!r}, "
                        f"logit_init={float(self._lambda_fusion_logit_init)}."
                    ) from exc

        self.so3_corr_joint_count = 0
        self.so3_delta_corrector: Optional[nn.Module] = None
        self.so3_corr_gate_logit: Optional[nn.Parameter] = None
        try:
            rot_sl = resolve_rot6d_slice(
                output_layout,
                total_dim=self.out_motion_dim,
            )
            self.so3_corr_joint_count = infer_rot_joint_count(rot_sl)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "so3 corrector rot6d layout resolution failed: "
                "field='output_layout' expected a valid BoneRotations6D slice so so3_corr_joint_count can be inferred; "
                f"out_motion_dim={int(self.out_motion_dim)}, output_layout_type={type(output_layout).__name__}, "
                f"output_layout={output_layout!r}."
            ) from exc
        if self.so3_corr_joint_count > 0:
            self.so3_corr_gate_logit = nn.Parameter(torch.tensor(float(so3_corr_gate_logit_init)))
            h_mid = max(8, int(so3_corr_hidden))
            corr_in_dim = int(self.hidden_dim + (self.contact_dim if self.contact_plan_enable else 0))
            self.so3_delta_corrector = nn.Sequential(
                nn.LayerNorm(corr_in_dim),
                nn.Linear(corr_in_dim, h_mid),
                nn.ReLU(),
                nn.Dropout(float(so3_corr_dropout)),
                nn.Linear(h_mid, int(self.so3_corr_joint_count) * 3),
            )
            last = self.so3_delta_corrector[-1]
            if not isinstance(last, nn.Linear):
                raise TypeError(
                    "so3 corrector deterministic init failed: "
                    "field='so3_delta_corrector[-1]' expected nn.Linear final layer; "
                    f"got module_type={type(last).__name__}."
                )
            bias_shape = tuple(int(v) for v in last.bias.shape) if torch.is_tensor(last.bias) else None
            try:
                with torch.no_grad():
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "so3 corrector deterministic init failed: "
                    "field='so3_delta_corrector[-1]' expected zero-initializable weight/bias tensors; "
                    f"weight_shape={tuple(int(v) for v in last.weight.shape)!r}, "
                    f"bias_shape={bias_shape!r}, "
                    f"joint_count={int(self.so3_corr_joint_count)}."
                ) from exc

    def _direct_pose_split_state(self) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "direct_pose_split_enable", False)):
            return None
        leg_terminal = getattr(self, "direct_pose_leg_terminal", None)
        state = {
            "arm_split": bool(getattr(self, "direct_pose_arm_split_enable", False)),
            "head": getattr(self, "direct_pose_head", None),
            "unified_leg_terminal": True,
            "leg_head": leg_terminal,
            "nonleg_head": getattr(self, "direct_pose_out_nonleg", None),
            "arm_head": getattr(self, "direct_pose_out_arm", None),
            "else_head": getattr(self, "direct_pose_out_else", None),
            "nonleg_proj": getattr(self, "direct_pose_nonleg_proj", None),
            "arm_proj": getattr(self, "direct_pose_arm_proj", None),
            "else_proj": getattr(self, "direct_pose_else_proj", None),
            "idx_leg": getattr(self, "direct_pose_leg_out_idx", None),
            "idx_nonleg": getattr(self, "direct_pose_nonleg_out_idx", None),
            "idx_arm": getattr(self, "direct_pose_arm_out_idx", None),
            "idx_else": getattr(self, "direct_pose_else_out_idx", None),
        }
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

        def _idx_contract_error(field_name: str, value: Any, expected: str) -> RuntimeError:
            actual_shape = tuple(int(dim) for dim in value.shape) if torch.is_tensor(value) else None
            actual_dtype = value.dtype if torch.is_tensor(value) else None
            return RuntimeError(
                "direct_pose_split_enable split state contract failed: "
                f"field='{field_name}' expected {expected}; "
                f"got actual_type={type(value).__name__}, actual_shape={actual_shape!r}, actual_dtype={actual_dtype}."
            )

        def _require_index_tensor(field_name: str) -> torch.Tensor:
            idx = state.get(field_name, None)
            expected = "a non-empty 1D integer torch.Tensor with indices in [0, out_motion_dim)"
            if not torch.is_tensor(idx):
                raise _idx_contract_error(field_name, idx, expected)
            if idx.ndim != 1:
                raise _idx_contract_error(field_name, idx, expected)
            if idx.dtype not in integer_dtypes:
                raise _idx_contract_error(field_name, idx, expected)
            if int(idx.numel()) <= 0:
                raise _idx_contract_error(field_name, idx, expected)
            idx_cpu = idx.detach().cpu()
            min_value = int(idx_cpu.min().item())
            max_value = int(idx_cpu.max().item())
            if min_value < 0 or max_value >= int(self.out_motion_dim):
                raise RuntimeError(
                    "direct_pose_split_enable split state contract failed: "
                    f"field='{field_name}' expected indices in [0, out_motion_dim={int(self.out_motion_dim)}); "
                    f"got min={min_value}, max={max_value}, actual_shape={tuple(int(dim) for dim in idx.shape)!r}."
                )
            return idx

        idx_leg = _require_index_tensor("idx_leg")
        idx_nonleg = _require_index_tensor("idx_nonleg")
        coverage = torch.cat([idx_leg.detach().cpu().to(dtype=torch.long), idx_nonleg.detach().cpu().to(dtype=torch.long)])
        expected_dim = int(self.out_motion_dim)
        coverage_unique = torch.unique(coverage)
        if int(coverage.numel()) != expected_dim or int(coverage_unique.numel()) != expected_dim:
            raise RuntimeError(
                "direct_pose_split_enable split state contract failed: "
                "fields=('direct_pose_leg_out_idx', 'direct_pose_nonleg_out_idx') expected full disjoint coverage "
                f"of out_motion_dim={expected_dim}; got leg_shape={tuple(int(dim) for dim in idx_leg.shape)!r}, "
                f"nonleg_shape={tuple(int(dim) for dim in idx_nonleg.shape)!r}, coverage_numel={int(coverage.numel())}, "
                f"unique_numel={int(coverage_unique.numel())}."
            )
        if state["arm_split"]:
            idx_arm = _require_index_tensor("idx_arm")
            idx_else = _require_index_tensor("idx_else")
            arm_else = torch.cat([idx_arm.detach().cpu().to(dtype=torch.long), idx_else.detach().cpu().to(dtype=torch.long)])
            arm_else_sorted = torch.sort(arm_else).values
            nonleg_sorted = torch.sort(idx_nonleg.detach().cpu().to(dtype=torch.long)).values
            if int(arm_else.numel()) != int(idx_nonleg.numel()) or not bool(torch.equal(arm_else_sorted, nonleg_sorted)):
                raise RuntimeError(
                    "direct_pose_split_enable split state contract failed: "
                    "fields=('direct_pose_arm_out_idx', 'direct_pose_else_out_idx') expected full disjoint coverage "
                    "of direct_pose_nonleg_out_idx; "
                    f"arm_shape={tuple(int(dim) for dim in idx_arm.shape)!r}, "
                    f"else_shape={tuple(int(dim) for dim in idx_else.shape)!r}, "
                    f"nonleg_shape={tuple(int(dim) for dim in idx_nonleg.shape)!r}."
                )
        return state

    @staticmethod
    def _init_square_identity_linear_(module: Any) -> None:
        if not isinstance(module, nn.Linear):
            return
        if int(module.in_features) != int(module.out_features):
            return
        with torch.no_grad():
            module.weight.zero_()
            module.weight.add_(
                torch.eye(int(module.out_features), device=module.weight.device, dtype=module.weight.dtype)
            )
            if module.bias is not None:
                module.bias.zero_()

    @staticmethod
    def _make_zero_linear_init_(*, bias_value: float = 0.0):
        def _init(module: Any) -> None:
            EventMotionModel._init_zero_linear_(module, bias_value=bias_value)

        return _init

    @staticmethod
    def _init_zero_linear_(module: Any, *, bias_value: float = 0.0) -> None:
        if not isinstance(module, nn.Linear):
            return
        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.fill_(float(bias_value))

    def _derive_deterministic_branch_seed(self, branch_name: str, *shape_parts: Any) -> int:
        base_seed = int(getattr(self, "_direct_pose_init_base_seed", torch.initial_seed()))
        parts = [f"base_seed={base_seed}", f"branch={branch_name}"]
        for part in shape_parts:
            if isinstance(part, (tuple, list)):
                parts.append(",".join(str(item) for item in part))
            else:
                parts.append(str(part))
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)

    @contextmanager
    def _with_branch_seed(self, branch_name: str, *shape_parts: Any):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._derive_deterministic_branch_seed(branch_name, *shape_parts))
            yield

    def _build_seeded_linear(
        self,
        in_features: int,
        out_features: int,
        *,
        branch_name: str,
        bias: bool = True,
        init_fn: Optional[Any] = None,
    ) -> nn.Linear:
        shape_sig = ("linear", int(in_features), int(out_features), int(bool(bias)))
        with self._with_branch_seed(branch_name, shape_sig):
            layer = nn.Linear(int(in_features), int(out_features), bias=bool(bias))
        if init_fn is not None:
            init_fn(layer)
        return layer

    @staticmethod
    def _new_generator_from_state(state: torch.Tensor) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.set_state(state.clone())
        return generator

    @staticmethod
    def _init_linear_with_generator_(module: nn.Linear, generator: torch.Generator) -> None:
        nn.init.kaiming_uniform_(module.weight, a=_math.sqrt(5), generator=generator)
        if module.bias is None:
            return
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1.0 / _math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(module.bias, -bound, bound, generator=generator)

    @staticmethod
    def _advance_linear_stream_(
        generator: torch.Generator,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        weight = torch.empty((int(out_features), int(in_features)), dtype=torch.float32)
        nn.init.kaiming_uniform_(weight, a=_math.sqrt(5), generator=generator)
        if bool(bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1.0 / _math.sqrt(fan_in) if fan_in > 0 else 0.0
            bias_tensor = torch.empty(int(out_features), dtype=torch.float32)
            nn.init.uniform_(bias_tensor, -bound, bound, generator=generator)

    def _build_linear_from_generator(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: Optional[torch.Generator],
        bias: bool = True,
        init_fn: Optional[Any] = None,
    ) -> nn.Linear:
        if generator is None:
            raise ValueError("generator is required for _build_linear_from_generator.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            layer = nn.Linear(int(in_features), int(out_features), bias=bool(bias))
        self._init_linear_with_generator_(layer, generator)
        if init_fn is not None:
            init_fn(layer)
        return layer

    def _build_split_head_branch(
        self,
        *,
        trunk_dim: int,
        out_dim: int,
        proj_dim: int = 0,
        out_name: str,
        proj_name: str,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[nn.Linear, Optional[nn.Module]]:
        proj_dim = int(proj_dim or 0)
        linear_builder = (
            lambda in_features, out_features, *, branch_name, init_fn=None: self._build_linear_from_generator(
                in_features,
                out_features,
                generator=generator,
                init_fn=init_fn,
            )
            if generator is not None
            else self._build_seeded_linear(
                in_features,
                out_features,
                branch_name=branch_name,
                init_fn=init_fn,
            )
        )
        if proj_dim > 0:
            proj = nn.Sequential(
                linear_builder(
                    int(trunk_dim),
                    proj_dim,
                    branch_name=f"{proj_name}.fc0",
                ),
                nn.ReLU(),
            )
            return (
                linear_builder(
                    proj_dim,
                    int(out_dim),
                    branch_name=out_name,
                ),
                proj,
            )
        return (
            linear_builder(
                int(trunk_dim),
                int(out_dim),
                branch_name=out_name,
            ),
            None,
        )

    def _build_direct_pose_terminal_block(self, *, trunk_dim: int, out_dim: int, drop: float) -> nn.Sequential:
        block = nn.Sequential(
            self._build_seeded_linear(
                int(trunk_dim),
                int(trunk_dim),
                branch_name="direct_pose_leg_terminal.fc0",
                init_fn=self._init_square_identity_linear_,
            ),
            nn.ReLU(),
            nn.Dropout(float(drop)) if float(drop) > 0 else nn.Identity(),
            self._build_seeded_linear(
                int(trunk_dim),
                int(trunk_dim),
                branch_name="direct_pose_leg_terminal.fc1",
                init_fn=self._init_square_identity_linear_,
            ),
            nn.ReLU(),
            nn.Dropout(float(drop)) if float(drop) > 0 else nn.Identity(),
            self._build_seeded_linear(
                int(trunk_dim),
                int(out_dim),
                branch_name="direct_pose_leg_terminal.out",
            ),
        )
        return block

    def _forward_direct_pose_readout(self, direct_flat: torch.Tensor, *, B: int, Tq: int) -> torch.Tensor:
        if not bool(getattr(self, "direct_pose_split_enable", False)):
            return self.direct_pose_head(direct_flat).view(B, Tq, -1)

        split_state = self._direct_pose_split_state()
        if split_state is None or split_state["head"] is None or split_state["leg_head"] is None:
            raise RuntimeError("direct_pose_split_enable=true but split trunk/leg head modules are missing.")
        idx_leg = split_state["idx_leg"]
        idx_nonleg = split_state["idx_nonleg"]
        hidden = split_state["head"](direct_flat)
        leg_out = split_state["leg_head"](hidden)
        out_flat = hidden.new_zeros((hidden.shape[0], int(self.out_motion_dim)))
        idx_leg_use = idx_leg.to(device=out_flat.device)
        out_flat = out_flat.index_copy(1, idx_leg_use, leg_out)

        if bool(split_state["arm_split"]):
            idx_arm = split_state["idx_arm"]
            idx_else = split_state["idx_else"]
            if split_state["arm_head"] is None or split_state["else_head"] is None:
                raise RuntimeError("direct_pose_arm_split_enable=true but arm/else head modules are missing.")
            arm_in = hidden
            else_in = hidden
            arm_proj = split_state["arm_proj"]
            else_proj = split_state["else_proj"]
            if arm_proj is not None:
                arm_in = arm_proj(arm_in)
            if else_proj is not None:
                else_in = else_proj(else_in)
            arm_out = split_state["arm_head"](arm_in)
            else_out = split_state["else_head"](else_in)
            if (
                int(leg_out.shape[-1]) != int(idx_leg.numel())
                or int(arm_out.shape[-1]) != int(idx_arm.numel())
                or int(else_out.shape[-1]) != int(idx_else.numel())
                or int(idx_arm.numel() + idx_else.numel()) != int(idx_nonleg.numel())
            ):
                raise RuntimeError("direct split-head (leg/arm/else) output dim mismatch with index buffers.")
            out_flat = out_flat.index_copy(1, idx_arm.to(device=out_flat.device), arm_out)
            out_flat = out_flat.index_copy(1, idx_else.to(device=out_flat.device), else_out)
            return out_flat.view(B, Tq, -1)

        if split_state["nonleg_head"] is None:
            raise RuntimeError("direct_pose_split_enable=true but non-leg head module is missing.")
        nonleg_in = hidden
        nonleg_proj = split_state["nonleg_proj"]
        if nonleg_proj is not None:
            nonleg_in = nonleg_proj(nonleg_in)
        nonleg_out = split_state["nonleg_head"](nonleg_in)
        if int(leg_out.shape[-1]) != int(idx_leg.numel()) or int(nonleg_out.shape[-1]) != int(idx_nonleg.numel()):
            raise RuntimeError("direct split-head output dim mismatch with index buffers.")
        idx_nonleg_use = idx_nonleg.to(device=out_flat.device)
        out_flat = out_flat.index_copy(1, idx_nonleg_use, nonleg_out)
        return out_flat.view(B, Tq, -1)

    def load_state_dict(self, state_dict, strict: bool = True):
        if isinstance(state_dict, dict):
            maybe_upgrade_direct_pose_split_state_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def _init_bone_residual_adapters(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
        target_bones: Sequence[str],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        if not bone_names or not isinstance(bone_names, (list, tuple)) or not target_bones:
            return
        if not isinstance(output_layout, dict):
            output_layout = {}
        rot_sl = resolve_rot6d_slice(
            output_layout,
            total_dim=self.out_motion_dim,
        )
        if (not isinstance(rot_sl, slice)) or rot_sl.start is None or rot_sl.stop is None:
            return

        name_to_idx = {str(n): int(i) for i, n in enumerate(bone_names)}
        for name in target_bones:
            if name not in name_to_idx:
                continue
            j = name_to_idx[name]
            st = int(rot_sl.start) + j * 6
            ed = st + 6
            if ed > int(rot_sl.stop) or ed > self.out_motion_dim:
                continue
            self._bone_adapter_slices.append(slice(st, ed))
            self._bone_adapter_names.append(str(name))
            self._bone_adapters.append(
                _BoneSliceResidualAdapter(
                    in_dim=self.hidden_dim,
                    out_dim=6,
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    activation=nn.ReLU,
                    alpha_mode="tanh",
                )
            )

    def _target_device(self) -> torch.device:
        try:
            return next(self.motion_head.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def set_eval_runtime_controls(
        self,
        *,
        direct_pose_plan_override: Any = None,
        direct_pose_meas_override: Any = None,
        direct_pose_leg_side_plan_other_ablate_mode: Optional[str] = None,
        direct_pose_leg_cross_leg_ablate_mode: Optional[str] = None,
        contact_plan_inject_scale: float = 1.0,
        contact_plan_time_bias_scale: float = 1.0,
        debug_contact_plan_logits_decomp: bool = False,
    ) -> None:
        self._eval_runtime_controls = self._normalize_eval_runtime_controls(
            direct_pose_plan_override=direct_pose_plan_override,
            direct_pose_meas_override=direct_pose_meas_override,
            direct_pose_leg_side_plan_other_ablate_mode=direct_pose_leg_side_plan_other_ablate_mode,
            direct_pose_leg_cross_leg_ablate_mode=direct_pose_leg_cross_leg_ablate_mode,
            contact_plan_inject_scale=contact_plan_inject_scale,
            contact_plan_time_bias_scale=contact_plan_time_bias_scale,
            debug_contact_plan_logits_decomp=debug_contact_plan_logits_decomp,
        )

    def _reset_eval_runtime_controls(self) -> None:
        self._eval_runtime_controls = _DEFAULT_EVAL_RUNTIME_CONTROLS

    def _normalize_eval_runtime_controls(
        self,
        *,
        direct_pose_plan_override: Any = None,
        direct_pose_meas_override: Any = None,
        direct_pose_leg_side_plan_other_ablate_mode: Optional[str] = None,
        direct_pose_leg_cross_leg_ablate_mode: Optional[str] = None,
        contact_plan_inject_scale: float = 1.0,
        contact_plan_time_bias_scale: float = 1.0,
        debug_contact_plan_logits_decomp: bool = False,
    ) -> _EvalRuntimeControls:
        if contact_plan_inject_scale is None:
            inject_scale = 1.0
        else:
            try:
                inject_scale = float(contact_plan_inject_scale)
            except (TypeError, ValueError):
                inject_scale = 1.0
        if contact_plan_time_bias_scale is None:
            time_bias_scale = 1.0
        else:
            try:
                time_bias_scale = float(contact_plan_time_bias_scale)
            except (TypeError, ValueError):
                time_bias_scale = 1.0
        leg_side_plan_other_ablate_mode = _normalize_eval_runtime_ablate_mode(
            direct_pose_leg_side_plan_other_ablate_mode
        )
        leg_cross_leg_ablate_mode = _normalize_eval_runtime_ablate_mode(
            direct_pose_leg_cross_leg_ablate_mode
        )
        debug_contact_plan_logits_decomp = bool(debug_contact_plan_logits_decomp)
        if (
            direct_pose_plan_override is None
            and direct_pose_meas_override is None
            and leg_side_plan_other_ablate_mode == "none"
            and leg_cross_leg_ablate_mode == "none"
            and inject_scale == 1.0
            and time_bias_scale == 1.0
            and not debug_contact_plan_logits_decomp
        ):
            return _DEFAULT_EVAL_RUNTIME_CONTROLS
        return _EvalRuntimeControls(
            direct_pose_plan_override=direct_pose_plan_override,
            direct_pose_meas_override=direct_pose_meas_override,
            direct_pose_leg_side_plan_other_ablate_mode=leg_side_plan_other_ablate_mode,
            direct_pose_leg_cross_leg_ablate_mode=leg_cross_leg_ablate_mode,
            contact_plan_inject_scale=inject_scale,
            contact_plan_time_bias_scale=time_bias_scale,
            debug_contact_plan_logits_decomp=debug_contact_plan_logits_decomp,
        )

    def _eval_runtime_controls_bundle(self) -> _EvalRuntimeControls:
        runtime = getattr(self, "_eval_runtime_controls", None)
        return runtime if isinstance(runtime, _EvalRuntimeControls) else _DEFAULT_EVAL_RUNTIME_CONTROLS

    # === future: train/models/event_forward.py ===
    # Forward input prep / runtime-control shell.

    @staticmethod
    def _forward_input_shape_error(field_name: str, value: Any, expected: str, reason: str) -> str:
        actual_shape = tuple(int(dim) for dim in value.shape) if torch.is_tensor(value) else None
        actual_ndim = int(value.ndim) if torch.is_tensor(value) else None
        return (
            f"{field_name} input shape contract failed in EventMotionModel.forward: "
            f"expected {expected}; got actual_type={type(value).__name__}, "
            f"actual_ndim={actual_ndim}, actual_shape={actual_shape!r}. {reason}"
        )

    def _prepare_forward_inputs(
        self,
        *,
        state: torch.Tensor,
        cond: Optional[torch.Tensor],
        contacts: Optional[torch.Tensor],
        angvel: Optional[torch.Tensor],
        pose_history: Optional[torch.Tensor],
        plan_z: Optional[torch.Tensor],
        phase_z: Optional[torch.Tensor],
        phase_event_age: Optional[torch.Tensor],
    ) -> _EventMotionForwardInputPrep:
        if not torch.is_tensor(state):
            raise TypeError(
                self._forward_input_shape_error(
                    "state",
                    state,
                    f"a torch.Tensor with shape (B, Tq, in_state_dim={int(self.in_state_dim)}) or (B, in_state_dim)",
                    "state must be a Tensor before any forward normalization.",
                )
            )
        if state.ndim not in (2, 3):
            raise RuntimeError(
                self._forward_input_shape_error(
                    "state",
                    state,
                    f"rank 2 or 3 with final dim in_state_dim={int(self.in_state_dim)}",
                    f"state rank must be 2 or 3, got {int(state.ndim)}.",
                )
            )
        if cond is not None and not torch.is_tensor(cond):
            raise TypeError(
                self._forward_input_shape_error(
                    "cond",
                    cond,
                    f"a torch.Tensor with shape (B, Tq, cond_dim={int(self.cond_dim)}) or (B, cond_dim)",
                    "cond must be a Tensor when provided.",
                )
            )
        is_single = state.ndim == 2
        if is_single:
            state = state.unsqueeze(1)
            if cond is not None and cond.ndim == 2:
                cond = cond.unsqueeze(1)
            if contacts is not None and contacts.ndim == 2:
                contacts = contacts.unsqueeze(1)
            if angvel is not None and angvel.ndim == 2:
                angvel = angvel.unsqueeze(1)
        if pose_history is not None and pose_history.ndim == 2:
            pose_history = pose_history.unsqueeze(1)
        if plan_z is not None and plan_z.ndim == 1:
            plan_z = plan_z.unsqueeze(0)
        if phase_z is not None and phase_z.ndim == 1:
            phase_z = phase_z.unsqueeze(0)
        if phase_event_age is not None and phase_event_age.ndim == 1:
            phase_event_age = phase_event_age.unsqueeze(0)
        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(state.shape[:-1] + (self.cond_dim,), device=state.device, dtype=state.dtype)
        if cond is not None and cond.ndim == 2 and state.ndim == 3:
            cond = cond.unsqueeze(1)
        if state.ndim != 3 or int(state.shape[-1]) != int(self.in_state_dim):
            raise RuntimeError(
                self._forward_input_shape_error(
                    "state",
                    state,
                    f"shape (B, Tq, in_state_dim={int(self.in_state_dim)}) after 2D input normalization",
                    f"state feature dim must equal in_state_dim={int(self.in_state_dim)}.",
                )
            )
        if cond is not None:
            expected_cond = (
                f"shape (B={int(state.shape[0])}, Tq={int(state.shape[1])}, cond_dim={int(self.cond_dim)}) "
                "after 2D input normalization"
            )
            if (
                cond.ndim != 3
                or int(cond.shape[0]) != int(state.shape[0])
                or int(cond.shape[1]) != int(state.shape[1])
                or int(cond.shape[-1]) != int(self.cond_dim)
            ):
                raise RuntimeError(
                    self._forward_input_shape_error(
                        "cond",
                        cond,
                        expected_cond,
                        "cond batch/time axes and feature dim must match state and model cond_dim.",
                    )
                )

        device = state.device
        dtype = state.dtype
        batch_size, query_steps, _ = state.shape
        runtime_controls = self._eval_runtime_controls_bundle()
        contacts_input = contacts
        contacts_enc = contacts_input
        return _EventMotionForwardInputPrep(
            state=state,
            cond=cond,
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_history,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            is_single=is_single,
            device=device,
            dtype=dtype,
            batch_size=int(batch_size),
            query_steps=int(query_steps),
            runtime_controls=runtime_controls,
            contacts_input=contacts_input,
            contacts_enc=contacts_enc,
        )

    # Forward output finalize shell.

    def _build_forward_base_result(
        self,
        *,
        out: torch.Tensor,
        hidden_out: torch.Tensor,
        attn: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            'out': out,
            'delta': out,
            'attn': attn.mean(dim=1),
            'h_final': hidden_out,
        }

    def _write_forward_direct_pose_outputs(
        self,
        result: Dict[str, torch.Tensor],
        *,
        direct_out: torch.Tensor,
        direct_leg_omega: Optional[torch.Tensor],
        direct_leg_omega_raw: Optional[torch.Tensor],
        direct_leg_gate: Optional[torch.Tensor],
        direct_leg_gate_logits: Optional[torch.Tensor],
        direct_leg_scale: Optional[torch.Tensor],
        direct_leg_scale_log: Optional[torch.Tensor],
        direct_leg_scale_log_raw: Optional[torch.Tensor],
        direct_leg_side_sign_gate: Optional[torch.Tensor],
        is_single: bool,
    ) -> None:
        if is_single:
            direct_out = direct_out.squeeze(1)
        result['out_direct'] = direct_out
        if torch.is_tensor(direct_leg_omega):
            result["direct_leg_omega"] = direct_leg_omega.squeeze(1) if is_single else direct_leg_omega
        if torch.is_tensor(direct_leg_omega_raw):
            result["direct_leg_omega_raw"] = direct_leg_omega_raw.squeeze(1) if is_single else direct_leg_omega_raw
        if torch.is_tensor(direct_leg_gate):
            result["direct_leg_gate"] = direct_leg_gate.squeeze(1) if is_single else direct_leg_gate
        if torch.is_tensor(direct_leg_gate_logits):
            result["direct_leg_gate_logits"] = direct_leg_gate_logits.squeeze(1) if is_single else direct_leg_gate_logits
        if torch.is_tensor(direct_leg_scale):
            result["direct_leg_scale"] = direct_leg_scale.squeeze(1) if is_single else direct_leg_scale
        if torch.is_tensor(direct_leg_scale_log):
            result["direct_leg_scale_log"] = direct_leg_scale_log.squeeze(1) if is_single else direct_leg_scale_log
        if torch.is_tensor(direct_leg_scale_log_raw):
            result["direct_leg_scale_log_raw"] = (
                direct_leg_scale_log_raw.squeeze(1) if is_single else direct_leg_scale_log_raw
            )
        if torch.is_tensor(direct_leg_side_sign_gate):
            result["direct_leg_side_sign_gate"] = (
                direct_leg_side_sign_gate.squeeze(1) if is_single else direct_leg_side_sign_gate
            )

    def _lambda_fusion_rollout_step_feature(
        self,
        *,
        rollout_step: Optional[torch.Tensor | int | float],
        batch_size: int,
        query_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        like: torch.Tensor,
    ) -> torch.Tensor:
        B = int(batch_size)
        Tq = int(query_steps)
        try:
            if rollout_step is None:
                step_feat = like.new_zeros((B, Tq, 1))
            elif torch.is_tensor(rollout_step):
                s = rollout_step.to(device=device, dtype=dtype)
                if s.dim() == 0:
                    step_feat = s.view(1, 1, 1).expand(B, Tq, 1)
                elif s.dim() == 1:
                    if s.numel() == 1:
                        step_feat = s.view(1, 1, 1).expand(B, Tq, 1)
                    elif s.shape[0] == B:
                        step_feat = s.view(B, 1, 1).expand(B, Tq, 1)
                    else:
                        raise ValueError(f"1D rollout_step must have 1 or B={B} elements, got {int(s.numel())}.")
                elif s.dim() == 2:
                    if s.shape[0] not in (1, B):
                        raise ValueError(f"2D rollout_step batch axis must be 1 or B={B}, got {int(s.shape[0])}.")
                    if s.shape[1] < 1:
                        raise ValueError("2D rollout_step time axis must have at least one step.")
                    if s.shape[0] == 1 and B > 1:
                        s = s.expand(B, -1)
                    if s.shape[1] == 1:
                        step_feat = s[:, :1].reshape(B, 1, 1).expand(B, Tq, 1)
                    else:
                        step_feat = s[:, :Tq].unsqueeze(-1)
                        if step_feat.shape[1] < Tq:
                            raise ValueError(
                                f"2D rollout_step time axis must have at least Tq={Tq} steps or exactly 1, got {int(s.shape[1])}."
                            )
                else:
                    if s.dim() != 3:
                        raise ValueError(f"rollout_step tensor rank must be 0, 1, 2, or 3, got {int(s.dim())}.")
                    if s.shape[0] not in (1, B):
                        raise ValueError(f"3D rollout_step batch axis must be 1 or B={B}, got {int(s.shape[0])}.")
                    if s.shape[1] not in (1, Tq):
                        raise ValueError(f"3D rollout_step time axis must be 1 or Tq={Tq}, got {int(s.shape[1])}.")
                    if s.shape[-1] != 1:
                        raise ValueError(f"3D rollout_step feature axis must be 1, got {int(s.shape[-1])}.")
                    if s.shape[0] == 1 and B > 1:
                        s = s.expand(B, -1, -1)
                    if s.shape[1] == 1 and Tq > 1:
                        s = s.expand(-1, Tq, -1)
                    step_feat = s
            else:
                step_feat = torch.full((B, Tq, 1), float(rollout_step), device=device, dtype=dtype)
            if tuple(int(dim) for dim in step_feat.shape) != (B, Tq, 1):
                raise RuntimeError(
                    f"normalized rollout_step feature must have shape (B={B}, Tq={Tq}, 1), "
                    f"got {tuple(int(dim) for dim in step_feat.shape)}."
                )
        except (RuntimeError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise RuntimeError(
                "lambda_fusion rollout_step contract failed "
                f"(B={B}, Tq={Tq}, rollout_step_type={type(rollout_step).__name__}, "
                f"rollout_step.shape={tuple(int(dim) for dim in rollout_step.shape) if torch.is_tensor(rollout_step) else None})"
            ) from exc
        return step_feat

    def _write_forward_lambda_fusion_outputs(
        self,
        result: Dict[str, torch.Tensor],
        *,
        h_final: torch.Tensor,
        contact_error: Optional[torch.Tensor],
        rollout_step: Optional[torch.Tensor | int | float],
        device: torch.device,
        dtype: torch.dtype,
        is_single: bool,
        batch_size: int,
        query_steps: int,
    ) -> None:
        if self.lambda_fusion_head is None:
            return

        lam_in: Optional[torch.Tensor] = None
        B = int(batch_size)
        Tq = int(query_steps)
        try:
            lam_in = h_final
            if lam_in.ndim == 2:
                lam_in = lam_in.unsqueeze(1)

            if self.contact_plan_enable and contact_error is not None:
                err_in = contact_error.detach() if self.lambda_fusion_detach_err else contact_error
                if err_in.ndim == 2:
                    err_in = err_in.unsqueeze(1)
                lam_in = torch.cat([lam_in, err_in.to(device=device, dtype=dtype)], dim=-1)

            if bool(getattr(self, "lambda_fusion_use_rollout_step", False)):
                step_feat = self._lambda_fusion_rollout_step_feature(
                    rollout_step=rollout_step,
                    batch_size=B,
                    query_steps=Tq,
                    device=device,
                    dtype=dtype,
                    like=lam_in,
                )
                lam_in = torch.cat([lam_in, step_feat], dim=-1)

            flat = lam_in.reshape(-1, lam_in.shape[-1])
            logits = self.lambda_fusion_head(flat).view(lam_in.shape[0], lam_in.shape[1], -1)
            lam = torch.sigmoid(logits)
            if self.lambda_fusion_mode == "global" and int(self.lambda_fusion_joint_count) > 0:
                lam = lam.expand(lam.shape[0], lam.shape[1], int(self.lambda_fusion_joint_count))
            if is_single:
                logits = logits.squeeze(1)
                lam = lam.squeeze(1)
            result["lambda_fusion_logits"] = logits
            result["lambda_fusion"] = lam
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            raise RuntimeError(
                "lambda_fusion forward failed "
                f"(B={B}, Tq={Tq}, lam_in.shape={tuple(int(dim) for dim in lam_in.shape) if torch.is_tensor(lam_in) else None}, "
                f"mode={getattr(self, 'lambda_fusion_mode', None)!r}, "
                f"joint_count={int(getattr(self, 'lambda_fusion_joint_count', 0) or 0)})"
            ) from exc

    def _write_forward_so3_delta_outputs(
        self,
        result: Dict[str, torch.Tensor],
        *,
        h_final: torch.Tensor,
        contact_error: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        is_single: bool,
    ) -> None:
        if self.so3_delta_corrector is None or self.so3_corr_joint_count <= 0:
            return
        corr_in = h_final
        if self.contact_plan_enable and contact_error is not None:
            if contact_error.ndim == 2:
                contact_error = contact_error.unsqueeze(1)
            corr_in = torch.cat([h_final, contact_error.to(device=device, dtype=dtype)], dim=-1)
        omega = self.so3_delta_corrector(corr_in)
        omega = omega.view(omega.shape[0], omega.shape[1], self.so3_corr_joint_count, 3)
        if is_single:
            omega = omega.squeeze(1)
        result["omega_hat"] = omega

    @staticmethod
    def _write_forward_period_output(
        result: Dict[str, torch.Tensor],
        *,
        soft_period: Optional[torch.Tensor],
    ) -> None:
        if soft_period is not None:
            result["period_pred"] = soft_period

    # Contact-plan / Event-Clock shell.

    def _init_contact_clock_forward_defaults(self) -> _ContactClockForwardDefaults:
        return _ContactClockForwardDefaults(
            soft_period=None,
            contacts_meas=None,
            event_clock_delta_meas=None,
            event_clock_lr_diff=None,
            event_clock_lambda_corr=None,
            event_clock_lambda_logit=None,
            event_clock_dynamic_prior=None,
            event_clock_delta_z=None,
            pose_hist_processed=False,
            contacts_plan=None,
            plan_z_next=None,
            plan_feat_for_inject=None,
            contacts_plan_logits=None,
            contact_plan_debug_logits=_DEFAULT_CONTACT_PLAN_DEBUG_LOGITS,
            time_pe_direct=None,
            phase_z_in_direct=None,
            leg_side_cue_in=None,
        )

    def _finalize_contact_plan_outputs(
        self,
        *,
        plan_probs: list[torch.Tensor],
        plan_logits: list[torch.Tensor],
        phase_in_direct_seq: Optional[list[torch.Tensor]],
        leg_side_cue_seq: Optional[list[torch.Tensor]],
        contact_plan_debug_buffers: Optional[_ContactPlanDebugBuffers],
        plan_z_t: torch.Tensor,
        plan_z_seq: Optional[list[torch.Tensor]],
        batch_size: int,
        query_steps: int,
        phase_in_direct_dim: int,
        leg_side_cue_mode: str,
    ) -> _ContactPlanForwardFinal:
        B = int(batch_size)
        Tq = int(query_steps)
        contacts_plan = torch.stack(plan_probs, dim=1)  # (B,T,C)
        phase_z_in_direct = None
        leg_side_cue_in = None
        if phase_in_direct_seq is not None:
            if len(phase_in_direct_seq) != Tq:
                raise RuntimeError(
                    "direct_pose phase sequence stack failed "
                    f"(expected {Tq} steps, got {len(phase_in_direct_seq)}, "
                    f"phase_dim={phase_in_direct_dim}, event_clock={bool(self.use_event_clock)})"
                )
            try:
                phase_z_in_direct = torch.stack(phase_in_direct_seq, dim=1)  # (B,Tq,2*C)
            except (RuntimeError, ValueError, TypeError) as exc:
                elem_shapes = [
                    tuple(int(dim) for dim in item.shape) if torch.is_tensor(item) else None
                    for item in phase_in_direct_seq
                ]
                raise RuntimeError(
                    "direct_pose phase sequence stack failed "
                    f"(B={B}, Tq={Tq}, phase_dim={phase_in_direct_dim}, element_shapes={elem_shapes})"
                ) from exc
        if leg_side_cue_seq is not None:
            if len(leg_side_cue_seq) != Tq:
                raise RuntimeError(
                    "direct_pose side cue sequence stack failed "
                    f"(expected {Tq} steps, got {len(leg_side_cue_seq)}, "
                    f"cue_mode={leg_side_cue_mode!r}, contact_dim={int(self.contact_dim)})"
                )
            try:
                leg_side_cue_in = torch.stack(leg_side_cue_seq, dim=1)  # (B,Tq,C)
            except (RuntimeError, ValueError, TypeError) as exc:
                elem_shapes = [
                    tuple(int(dim) for dim in item.shape) if torch.is_tensor(item) else None
                    for item in leg_side_cue_seq
                ]
                raise RuntimeError(
                    "direct_pose side cue sequence stack failed "
                    f"(B={B}, Tq={Tq}, cue_mode={leg_side_cue_mode!r}, "
                    f"contact_dim={int(self.contact_dim)}, element_shapes={elem_shapes})"
                ) from exc
        try:
            contacts_plan_logits = torch.stack(plan_logits, dim=1) if plan_logits else None  # (B,T,logits_dim)
        except (RuntimeError, ValueError, TypeError) as exc:
            elem_shapes = [
                tuple(int(dim) for dim in item.shape) if torch.is_tensor(item) else None
                for item in plan_logits
            ]
            raise RuntimeError(
                "contacts_plan_logits stack failed "
                f"(B={B}, Tq={Tq}, num_steps={len(plan_logits)}, element_shapes={elem_shapes})"
            ) from exc
        contact_plan_debug_logits = self._finalize_contact_plan_debug_logits(contact_plan_debug_buffers)
        plan_z_next = plan_z_t
        plan_feat_for_inject = None
        if self.contact_plan_inject == "contacts":
            plan_feat_for_inject = contacts_plan
        elif self.contact_plan_inject == "plan_z" and plan_z_seq is not None:
            plan_feat_for_inject = torch.stack(plan_z_seq, dim=1)  # (B,T,H)
        return _ContactPlanForwardFinal(
            contacts_plan=contacts_plan,
            phase_z_in_direct=phase_z_in_direct,
            leg_side_cue_in=leg_side_cue_in,
            contacts_plan_logits=contacts_plan_logits,
            contact_plan_debug_logits=contact_plan_debug_logits,
            plan_z_next=plan_z_next,
            plan_feat_for_inject=plan_feat_for_inject,
        )

    # Direct-pose shell.

    def _should_run_direct_pose_forward(self, contacts_plan: Optional[torch.Tensor]) -> bool:
        return self.direct_pose_head is not None and contacts_plan is not None

    def _init_direct_pose_forward_runtime(
        self,
        runtime_controls: _EvalRuntimeControls,
    ) -> _DirectPoseForwardRuntime:
        return _DirectPoseForwardRuntime(
            plan_override=runtime_controls.direct_pose_plan_override,
            meas_override=runtime_controls.direct_pose_meas_override,
            leg_side_plan_other_ablate_mode=runtime_controls.direct_pose_leg_side_plan_other_ablate_mode,
            leg_cross_leg_ablate_mode=runtime_controls.direct_pose_leg_cross_leg_ablate_mode,
        )

    def _prepare_direct_pose_leg_omega(
        self,
        *,
        omega_parts: tuple[torch.Tensor, ...],
        batch_size: int,
        query_steps: int,
        joint_count: int,
        error_prefix: str,
        side_positions: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = int(batch_size)
        Tq = int(query_steps)
        K = int(joint_count)
        if side_positions is None:
            if len(omega_parts) != 1:
                raise RuntimeError(
                    f"{error_prefix} omega prep contract failed: expected a single omega tensor, "
                    f"got num_parts={len(omega_parts)}."
                )
            omega_leg = omega_parts[0]
        else:
            if len(omega_parts) != 2:
                raise RuntimeError(
                    f"{error_prefix} omega prep contract failed: expected right/left omega tensors, "
                    f"got num_parts={len(omega_parts)}."
                )
            pos_r_use, pos_l_use = side_positions
            omega_r, omega_l = omega_parts
            omega_leg = omega_r.new_zeros((B, Tq, K, 3))
            omega_leg = omega_leg.index_copy(2, pos_r_use, omega_r)
            omega_leg = omega_leg.index_copy(2, pos_l_use, omega_l)

        max_rad = float(getattr(self, "direct_pose_leg_max_rad", 0.0) or 0.0)
        if _math.isfinite(max_rad) and max_rad > 0.0:
            theta = omega_leg.norm(dim=-1, keepdim=True)
            denom = theta + 1e-8
            scale = (max_rad * torch.tanh(theta / max_rad)) / denom
            scale = torch.where(theta > 1e-8, scale, torch.ones_like(scale))
            omega_leg = omega_leg * scale
        return omega_leg

    def _resolve_direct_pose_side_leg_omegas(
        self,
        *,
        out_r: torch.Tensor,
        out_l: torch.Tensor,
        leg_flat_r: torch.Tensor,
        leg_flat_l: torch.Tensor,
        batch_size: int,
        query_steps: int,
        branch_joint_count: int,
    ) -> _DirectPoseSideLegOmegaOutputs:
        B = int(batch_size)
        Tq = int(query_steps)
        K_side = int(branch_joint_count)
        if bool(getattr(self, "direct_pose_leg_side_rank1", False)):
            if out_r.shape[-1] == (3 + K_side) and out_l.shape[-1] == (3 + K_side):
                v_r = out_r[..., :3]
                v_l = out_l[..., :3]
                s_r = F.softplus(out_r[..., 3:])
                s_l = F.softplus(out_l[..., 3:])
                dir_r = F.normalize(v_r, dim=-1, eps=1e-8)
                dir_l = F.normalize(v_l, dim=-1, eps=1e-8)
                return _DirectPoseSideLegOmegaOutputs(
                    omega_r=dir_r.unsqueeze(-2) * s_r.unsqueeze(-1),
                    omega_l=dir_l.unsqueeze(-2) * s_l.unsqueeze(-1),
                )
            return _DirectPoseSideLegOmegaOutputs(omega_r=None, omega_l=None)

        if out_r.shape[-1] != 3 * K_side or out_l.shape[-1] != 3 * K_side:
            return _DirectPoseSideLegOmegaOutputs(omega_r=None, omega_l=None)

        omega_r = out_r.view(B, Tq, K_side, 3)
        omega_l = out_l.view(B, Tq, K_side, 3)
        gate = getattr(self, "direct_pose_leg_side_sign_gate_head", None)
        if gate is None or not bool(getattr(self, "direct_pose_leg_side_sign_gate", False)):
            return _DirectPoseSideLegOmegaOutputs(omega_r=omega_r, omega_l=omega_l)

        try:
            g_r = torch.tanh(gate(leg_flat_r)).view(B, Tq, 1, 1)
            g_l = torch.tanh(gate(leg_flat_l)).view(B, Tq, 1, 1)
            return _DirectPoseSideLegOmegaOutputs(
                omega_r=omega_r * g_r,
                omega_l=omega_l * g_l,
                direct_leg_side_sign_gate=torch.cat([g_r.view(B, Tq, 1), g_l.view(B, Tq, 1)], dim=-1),
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            raise RuntimeError(
                "direct_pose side-routed leg sign gate forward failed "
                f"(B={B}, Tq={Tq}, K_side={K_side})"
            ) from exc

    def _resolve_direct_pose_non_side_leg_delta(
        self,
        *,
        leg_in: torch.Tensor,
        direct_feat: torch.Tensor,
        plan_in: Any,
        meas_in: Any,
        phase_in_direct: Optional[torch.Tensor],
        batch_size: int,
        query_steps: int,
        joint_count: int,
        ablation_mode: str,
    ) -> torch.Tensor:
        B = int(batch_size)
        Tq = int(query_steps)
        K = int(joint_count)
        leg_delta = self._compute_direct_pose_leg_cross_leg_ablation(
            leg_in=leg_in,
            direct_feat=direct_feat,
            plan_in=plan_in,
            meas_in=meas_in,
            phase_in_direct=phase_in_direct,
            batch_size=B,
            seq_len=Tq,
            joint_count=K,
            ablation_mode=ablation_mode,
        )
        if leg_delta is None:
            if self.direct_pose_leg_head is None:
                raise RuntimeError("direct_pose leg head is missing")
            leg_delta = self.direct_pose_leg_head(leg_in).view(B, Tq, -1)
        if leg_delta.ndim != 3 or int(leg_delta.shape[0]) != B or int(leg_delta.shape[1]) != Tq:
            raise RuntimeError(
                "direct_pose non-side leg delta contract failed: expected shape "
                f"(B={B}, Tq={Tq}, C), got {tuple(int(dim) for dim in leg_delta.shape)}."
            )
        return leg_delta

    def _assemble_direct_pose_side_leg_features(
        self,
        *,
        direct_feat: torch.Tensor,
        plan_in: Any,
        meas_in: Any,
        phase_in_direct: Optional[torch.Tensor],
        leg_side_cue_in: Optional[torch.Tensor],
        batch_size: int,
        query_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        leg_side_plan_other_ablate_mode: str,
    ) -> Optional[_DirectPoseSideLegAssembly]:
        B = int(batch_size)
        Tq = int(query_steps)
        idx = getattr(self, "direct_pose_leg_joint_idx_tensor", None)
        rot_sl = getattr(self, "direct_pose_leg_rot6d_slice", None)
        pos_r = getattr(self, "direct_pose_leg_side_pos_r_tensor", None)
        pos_l = getattr(self, "direct_pose_leg_side_pos_l_tensor", None)
        if (
            (not torch.is_tensor(idx))
            or (not torch.is_tensor(pos_r))
            or (not torch.is_tensor(pos_l))
            or (not isinstance(rot_sl, slice))
            or rot_sl.start is None
            or rot_sl.stop is None
        ):
            return None

        K = int(idx.numel())
        K_side = int(pos_r.numel())
        if K <= 0 or K_side <= 0 or int(pos_l.numel()) != K_side:
            return None

        rot_dim = int(rot_sl.stop - rot_sl.start)
        if rot_dim <= 0 or (rot_dim % 6) != 0:
            return None
        J = int(rot_dim // 6)
        idx_use = idx.to(device=device)
        if not bool(torch.all((idx_use >= 0) & (idx_use < J)).detach().cpu().item()):
            return None

        plan_bt = plan_in.to(device=device, dtype=dtype) if torch.is_tensor(plan_in) else None
        if plan_bt is None:
            plan_bt = torch.zeros((B, Tq, int(self.contact_dim)), device=device, dtype=dtype)
        elif plan_bt.ndim == 2:
            plan_bt = plan_bt.unsqueeze(1)
        if plan_bt.ndim == 3 and plan_bt.shape[1] == 1 and Tq > 1:
            plan_bt = plan_bt.expand(-1, Tq, -1)

        meas_bt = meas_in.to(device=device, dtype=dtype) if torch.is_tensor(meas_in) else None
        if meas_bt is not None and meas_bt.ndim == 2:
            meas_bt = meas_bt.unsqueeze(1)
        if meas_bt is not None and meas_bt.ndim == 3 and meas_bt.shape[1] == 1 and Tq > 1:
            meas_bt = meas_bt.expand(-1, Tq, -1)

        ch_r = int(getattr(self, "direct_pose_leg_contact_ch_r", 1) or 0)
        ch_l = int(getattr(self, "direct_pose_leg_contact_ch_l", 0) or 0)
        Cc = int(plan_bt.shape[-1])
        if Cc > 0:
            ch_r = max(0, min(int(Cc - 1), ch_r))
            ch_l = max(0, min(int(Cc - 1), ch_l))

        plan_r = plan_bt[..., ch_r : ch_r + 1]
        plan_l = plan_bt[..., ch_l : ch_l + 1]

        if meas_bt is None or int(meas_bt.shape[-1]) <= 0:
            meas_r = plan_r.new_zeros(plan_r.shape)
            meas_l = plan_l.new_zeros(plan_l.shape)
        else:
            meas_r = meas_bt[..., ch_r : ch_r + 1]
            meas_l = meas_bt[..., ch_l : ch_l + 1]

        if torch.is_tensor(phase_in_direct) and bool(getattr(self, "direct_pose_use_phase_z", False)):
            phase_bt = phase_in_direct
            if phase_bt.ndim == 2:
                phase_bt = phase_bt.unsqueeze(1)
            if phase_bt.ndim == 3 and phase_bt.shape[1] == 1 and Tq > 1:
                phase_bt = phase_bt.expand(-1, Tq, -1)
            phase_shape = tuple(int(dim) for dim in phase_bt.shape)
            expected_phase_dim = int(2 * Cc)
            if (
                phase_bt.ndim != 3
                or int(phase_bt.shape[0]) != B
                or int(phase_bt.shape[1]) != Tq
                or int(phase_bt.shape[-1]) != expected_phase_dim
            ):
                raise RuntimeError(
                    "direct_pose side-routed phase_z contract failed: "
                    f"expected phase_z_in_direct to resolve to "
                    f"(B={B}, Tq={Tq}, 2*contact_channels={expected_phase_dim}) "
                    f"before per-side view `(B, Tq, contact_channels={Cc}, 2)`, "
                    f"got ndim={int(phase_bt.ndim)}, shape={phase_shape}, "
                    f"ch_r={ch_r}, ch_l={ch_l}, contact_dim={int(getattr(self, 'contact_dim', 0) or 0)}."
                )
            phase_view = phase_bt.contiguous().view(B, Tq, Cc, 2)
            phase_r = phase_view[..., ch_r, :].to(device=device, dtype=dtype)
            phase_l = phase_view[..., ch_l, :].to(device=device, dtype=dtype)
        else:
            phase_r = plan_r.new_zeros((B, Tq, 0))
            phase_l = plan_l.new_zeros((B, Tq, 0))

        plan_other_r = plan_r.new_zeros((B, Tq, 0))
        plan_other_l = plan_l.new_zeros((B, Tq, 0))
        if bool(getattr(self, "direct_pose_leg_side_plan_other", False)):
            plan_other_r = plan_l
            plan_other_l = plan_r
            plan_other_r, plan_other_l = self._apply_direct_pose_leg_side_plan_other_ablation(
                plan_other_r,
                plan_other_l,
                ablation_mode=leg_side_plan_other_ablate_mode,
                batch_size=B,
                seq_len=Tq,
            )

        phase_other_r = plan_r.new_zeros((B, Tq, 0))
        phase_other_l = plan_l.new_zeros((B, Tq, 0))
        phase_rel_r = plan_r.new_zeros((B, Tq, 0))
        phase_rel_l = plan_l.new_zeros((B, Tq, 0))
        if bool(getattr(self, "direct_pose_leg_side_phase_other", False)) or bool(
            getattr(self, "direct_pose_leg_side_phase_rel", False)
        ):
            if phase_r.shape[-1] != 2 or phase_l.shape[-1] != 2:
                phase_r = plan_r.new_zeros((B, Tq, 2))
                phase_l = plan_l.new_zeros((B, Tq, 2))

            if bool(getattr(self, "direct_pose_leg_side_phase_other", False)):
                phase_other_r = phase_l
                phase_other_l = phase_r

            if bool(getattr(self, "direct_pose_leg_side_phase_rel", False)):
                sin_r = phase_r[..., 0:1]
                cos_r = phase_r[..., 1:2]
                sin_l = phase_l[..., 0:1]
                cos_l = phase_l[..., 1:2]
                sin_rel_r = sin_l * cos_r - cos_l * sin_r
                cos_rel_r = cos_l * cos_r + sin_l * sin_r
                sin_rel_l = sin_r * cos_l - cos_r * sin_l
                cos_rel_l = cos_r * cos_l + sin_r * sin_l
                phase_rel_r = torch.cat([sin_rel_r, cos_rel_r], dim=-1)
                phase_rel_l = torch.cat([sin_rel_l, cos_rel_l], dim=-1)

        cue_r = plan_bt.new_zeros((B, Tq, 0))
        cue_l = plan_bt.new_zeros((B, Tq, 0))
        if int(getattr(self, "direct_pose_leg_side_cue_dim", 0) or 0) > 0:
            cue_bt = leg_side_cue_in
            if cue_bt is None:
                cue_bt = plan_bt.new_zeros((B, Tq, int(self.contact_dim)))
            elif cue_bt.ndim == 2:
                cue_bt = cue_bt.unsqueeze(1)
            if cue_bt.ndim == 3 and cue_bt.shape[1] == 1 and Tq > 1:
                cue_bt = cue_bt.expand(-1, Tq, -1)
            if cue_bt.ndim == 3 and int(cue_bt.shape[-1]) >= int(self.contact_dim):
                cue_r = cue_bt[..., ch_r : ch_r + 1]
                cue_l = cue_bt[..., ch_l : ch_l + 1]
                cue_mode = str(getattr(self, "direct_pose_leg_side_cue", "none") or "none").strip().lower()
                if cue_mode == "phase_event_age":
                    tau = float(getattr(self, "direct_pose_leg_side_cue_tau", 30.0) or 30.0)
                    if (not _math.isfinite(tau)) or tau <= 1e-6:
                        tau = 30.0
                    cue_r = (cue_r / tau).clamp(0.0, 1.0)
                    cue_l = (cue_l / tau).clamp(0.0, 1.0)
                else:
                    cue_r = cue_r.clamp(0.0, 1.0)
                    cue_l = cue_l.clamp(0.0, 1.0)

        emb_r = None
        emb_l = None
        if getattr(self, "direct_pose_leg_side_embed", None) is not None:
            emb_weight_shape = None
            try:
                side_embed = self.direct_pose_leg_side_embed
                emb_w = side_embed.weight  # type: ignore[union-attr]
                emb_weight_shape = tuple(int(dim) for dim in emb_w.shape)
                emb_r_idx = emb_w.new_zeros((1,), dtype=torch.long)
                emb_l_idx = emb_w.new_ones((1,), dtype=torch.long)
                emb_r = side_embed(emb_r_idx.to(device=device)).view(1, 1, -1)  # type: ignore[operator]
                emb_l = side_embed(emb_l_idx.to(device=device)).view(1, 1, -1)  # type: ignore[operator]
                if emb_r.ndim != 3 or emb_l.ndim != 3:
                    raise RuntimeError(
                        "side embedding module must produce rank-3 tensors after canonical view "
                        f"but got emb_r.ndim={int(emb_r.ndim)}, emb_l.ndim={int(emb_l.ndim)}."
                    )
                if emb_r.shape[0] != 1 or emb_r.shape[1] != 1 or emb_l.shape[0] != 1 or emb_l.shape[1] != 1:
                    raise RuntimeError(
                        "side embedding module must produce a single embedding per side before broadcast "
                        f"but got emb_r.shape={tuple(int(dim) for dim in emb_r.shape)}, "
                        f"emb_l.shape={tuple(int(dim) for dim in emb_l.shape)}."
                    )
                if emb_r.shape[-1] != emb_l.shape[-1]:
                    raise RuntimeError(
                        "side embedding module returned inconsistent right/left feature dims "
                        f"(emb_r.shape={tuple(int(dim) for dim in emb_r.shape)}, "
                        f"emb_l.shape={tuple(int(dim) for dim in emb_l.shape)})."
                    )
                emb_r = emb_r.expand(B, Tq, -1)
                emb_l = emb_l.expand(B, Tq, -1)
            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                emb_r_shape = tuple(int(dim) for dim in emb_r.shape) if torch.is_tensor(emb_r) else None
                emb_l_shape = tuple(int(dim) for dim in emb_l.shape) if torch.is_tensor(emb_l) else None
                raise RuntimeError(
                    "direct_pose side-routed side embedding forward failed "
                    f"(B={B}, Tq={Tq}, expected emb_r/emb_l broadcast to (B, Tq, D), "
                    f"embed_weight_shape={emb_weight_shape}, emb_r.shape={emb_r_shape}, emb_l.shape={emb_l_shape})"
                ) from exc

        parts_r = [direct_feat, plan_r, meas_r, phase_r, plan_other_r, phase_other_r, phase_rel_r, cue_r]
        parts_l = [direct_feat, plan_l, meas_l, phase_l, plan_other_l, phase_other_l, phase_rel_l, cue_l]
        if torch.is_tensor(emb_r) and torch.is_tensor(emb_l):
            parts_r.append(emb_r)
            parts_l.append(emb_l)
        leg_in_r = torch.cat([part for part in parts_r if torch.is_tensor(part) and part.numel() > 0], dim=-1)
        leg_in_l = torch.cat([part for part in parts_l if torch.is_tensor(part) and part.numel() > 0], dim=-1)
        leg_shape_r = tuple(int(dim) for dim in leg_in_r.shape)
        leg_shape_l = tuple(int(dim) for dim in leg_in_l.shape)
        if leg_in_r.ndim != 3 or leg_shape_r[:2] != (B, Tq):
            raise RuntimeError(
                "direct_pose side-routed leg feature assembly contract failed: expected shape "
                f"(B={B}, Tq={Tq}, C) for right branch, got {leg_shape_r}."
            )
        if leg_in_l.ndim != 3 or leg_shape_l[:2] != (B, Tq):
            raise RuntimeError(
                "direct_pose side-routed leg feature assembly contract failed: expected shape "
                f"(B={B}, Tq={Tq}, C) for left branch, got {leg_shape_l}."
            )
        leg_flat_r = leg_in_r.reshape(-1, leg_in_r.shape[-1])
        leg_flat_l = leg_in_l.reshape(-1, leg_in_l.shape[-1])
        if bool(getattr(self, "direct_pose_leg_detach_feat", False)):
            leg_flat_r = leg_flat_r.detach()
            leg_flat_l = leg_flat_l.detach()
        return _DirectPoseSideLegAssembly(
            joint_count=K,
            branch_joint_count=K_side,
            pos_r=pos_r.to(device=device),
            pos_l=pos_l.to(device=device),
            leg_flat_r=leg_flat_r,
            leg_flat_l=leg_flat_l,
        )

    def _apply_direct_pose_leg_gate_outputs(
        self,
        *,
        omega_leg: torch.Tensor,
        gate_head: Optional[nn.Module],
        head_inputs: tuple[torch.Tensor, ...],
        batch_size: int,
        query_steps: int,
        joint_count: int,
        branch_joint_count: int,
        error_prefix: str,
        gate_head_name: str,
        side_positions: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> _DirectPoseLegGateOutputs:
        B = int(batch_size)
        Tq = int(query_steps)
        K = int(joint_count)
        K_branch = int(branch_joint_count)
        use_side_positions = side_positions is not None
        context = f"(B={B}, Tq={Tq}, K={K}"
        if use_side_positions:
            context = f"{context}, K_side={K_branch}"
        context = f"{context})"

        omega_eff = omega_leg
        gm_leg = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
        clamp_k = float(getattr(self, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0)
        use_scale_clamp = bool(_math.isfinite(clamp_k) and clamp_k > 1.0)
        if use_scale_clamp:
            scale_min = float(1.0 / clamp_k)
            scale_max = float(clamp_k)

        if gm_leg == "learned":
            try:
                if gate_head is None:
                    raise RuntimeError(f"learned leg gate enabled but {gate_head_name} is missing")
                gate_logits_parts = [gate_head(inp).view(B, Tq, K_branch) for inp in head_inputs]
                gate_parts = [torch.sigmoid(part) for part in gate_logits_parts]
                power = float(getattr(self, "direct_pose_leg_gate_power", 1.0) or 1.0)
                if (not _math.isfinite(power)) or power <= 0.0:
                    power = 1.0
                if abs(power - 1.0) > 1e-12:
                    gate_parts = [part.pow(power) for part in gate_parts]

                if use_side_positions:
                    if len(gate_parts) != 2 or side_positions is None:
                        raise RuntimeError("side-routed learned gate expects right/left gate parts and side positions")
                    pos_r_use, pos_l_use = side_positions
                    direct_leg_gate = omega_leg.new_zeros((B, Tq, K))
                    direct_leg_gate = direct_leg_gate.index_copy(2, pos_r_use, gate_parts[0])
                    direct_leg_gate = direct_leg_gate.index_copy(2, pos_l_use, gate_parts[1])
                    direct_leg_gate_logits = omega_leg.new_zeros((B, Tq, K))
                    direct_leg_gate_logits = direct_leg_gate_logits.index_copy(2, pos_r_use, gate_logits_parts[0])
                    direct_leg_gate_logits = direct_leg_gate_logits.index_copy(2, pos_l_use, gate_logits_parts[1])
                else:
                    if len(gate_parts) != 1:
                        raise RuntimeError("non-side learned gate expects a single gate part")
                    direct_leg_gate = gate_parts[0]
                    direct_leg_gate_logits = gate_logits_parts[0]

                omega_eff = omega_leg * direct_leg_gate.unsqueeze(-1)
                return _DirectPoseLegGateOutputs(
                    omega_eff=omega_eff,
                    direct_leg_gate=direct_leg_gate,
                    direct_leg_gate_logits=direct_leg_gate_logits,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                raise RuntimeError(f"{error_prefix} learned gate forward failed {context}") from exc

        if gm_leg == "scale":
            try:
                if gate_head is None:
                    raise RuntimeError(f"leg scale enabled but {gate_head_name} is missing")
                scale_log_raw_parts = [gate_head(inp).view(B, Tq, K_branch) for inp in head_inputs]
                clip = float(getattr(self, "direct_pose_leg_scale_log_clip", 4.0) or 4.0)
                if (not _math.isfinite(clip)) or clip <= 0.0:
                    clip = 4.0
                scale_log_parts = [part.clamp(-float(clip), float(clip)) for part in scale_log_raw_parts]
                scale_parts = [torch.exp(part) for part in scale_log_parts]
                if use_scale_clamp:
                    scale_parts = [part.clamp(scale_min, scale_max) for part in scale_parts]
                    scale_log_parts = [torch.log(part) for part in scale_parts]

                if use_side_positions:
                    if len(scale_parts) != 2 or side_positions is None:
                        raise RuntimeError("side-routed scale gate expects right/left scale parts and side positions")
                    pos_r_use, pos_l_use = side_positions
                    direct_leg_scale = omega_leg.new_zeros((B, Tq, K))
                    direct_leg_scale = direct_leg_scale.index_copy(2, pos_r_use, scale_parts[0])
                    direct_leg_scale = direct_leg_scale.index_copy(2, pos_l_use, scale_parts[1])
                    direct_leg_scale_log = omega_leg.new_zeros((B, Tq, K))
                    direct_leg_scale_log = direct_leg_scale_log.index_copy(2, pos_r_use, scale_log_parts[0])
                    direct_leg_scale_log = direct_leg_scale_log.index_copy(2, pos_l_use, scale_log_parts[1])
                    direct_leg_scale_log_raw = omega_leg.new_zeros((B, Tq, K))
                    direct_leg_scale_log_raw = direct_leg_scale_log_raw.index_copy(2, pos_r_use, scale_log_raw_parts[0])
                    direct_leg_scale_log_raw = direct_leg_scale_log_raw.index_copy(2, pos_l_use, scale_log_raw_parts[1])
                else:
                    if len(scale_parts) != 1:
                        raise RuntimeError("non-side scale gate expects a single scale part")
                    direct_leg_scale = scale_parts[0]
                    direct_leg_scale_log = scale_log_parts[0]
                    direct_leg_scale_log_raw = scale_log_raw_parts[0]

                omega_eff = omega_leg * direct_leg_scale.unsqueeze(-1)
                return _DirectPoseLegGateOutputs(
                    omega_eff=omega_eff,
                    direct_leg_scale=direct_leg_scale,
                    direct_leg_scale_log=direct_leg_scale_log,
                    direct_leg_scale_log_raw=direct_leg_scale_log_raw,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                raise RuntimeError(f"{error_prefix} scale gate forward failed {context}") from exc

        return _DirectPoseLegGateOutputs(omega_eff=omega_eff)

    def _forward_side_routed_leg_residual(
        self,
        *,
        direct_out: torch.Tensor,
        direct_feat: torch.Tensor,
        plan_in: Any,
        meas_in: Any,
        phase_in_direct: Optional[torch.Tensor],
        leg_side_cue_in: Optional[torch.Tensor],
        batch_size: int,
        query_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        leg_side_plan_other_ablate_mode: str,
    ):
        direct_leg_omega = None
        direct_leg_omega_raw = None
        direct_leg_gate = None
        direct_leg_gate_logits = None
        direct_leg_scale = None
        direct_leg_scale_log = None
        direct_leg_scale_log_raw = None
        direct_leg_side_sign_gate = None

        assembly = self._assemble_direct_pose_side_leg_features(
            direct_feat=direct_feat,
            plan_in=plan_in,
            meas_in=meas_in,
            phase_in_direct=phase_in_direct,
            leg_side_cue_in=leg_side_cue_in,
            batch_size=batch_size,
            query_steps=query_steps,
            device=device,
            dtype=dtype,
            leg_side_plan_other_ablate_mode=leg_side_plan_other_ablate_mode,
        )
        if assembly is None:
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        B = int(batch_size)
        Tq = int(query_steps)
        K = assembly.joint_count
        K_side = assembly.branch_joint_count
        pos_r_use = assembly.pos_r
        pos_l_use = assembly.pos_l
        out_r = self.direct_pose_leg_head_shared(assembly.leg_flat_r).view(B, Tq, -1)
        out_l = self.direct_pose_leg_head_shared(assembly.leg_flat_l).view(B, Tq, -1)

        omega_outputs = self._resolve_direct_pose_side_leg_omegas(
            out_r=out_r,
            out_l=out_l,
            leg_flat_r=assembly.leg_flat_r,
            leg_flat_l=assembly.leg_flat_l,
            batch_size=B,
            query_steps=Tq,
            branch_joint_count=K_side,
        )
        direct_leg_side_sign_gate = omega_outputs.direct_leg_side_sign_gate
        if (not torch.is_tensor(omega_outputs.omega_r)) or (not torch.is_tensor(omega_outputs.omega_l)):
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        omega_leg = self._prepare_direct_pose_leg_omega(
            omega_parts=(omega_outputs.omega_r, omega_outputs.omega_l),
            batch_size=B,
            query_steps=Tq,
            joint_count=K,
            error_prefix="direct_pose side-routed leg",
            side_positions=(pos_r_use, pos_l_use),
        )
        gate_outputs = self._apply_direct_pose_leg_gate_outputs(
            omega_leg=omega_leg,
            gate_head=getattr(self, "direct_pose_leg_gate_head_shared", None),
            head_inputs=(assembly.leg_flat_r, assembly.leg_flat_l),
            batch_size=B,
            query_steps=Tq,
            joint_count=K,
            branch_joint_count=K_side,
            error_prefix="direct_pose side-routed leg",
            gate_head_name="direct_pose_leg_gate_head_shared",
            side_positions=(pos_r_use, pos_l_use),
        )
        direct_leg_omega_raw = omega_leg
        direct_leg_omega = gate_outputs.omega_eff
        direct_leg_gate = gate_outputs.direct_leg_gate
        direct_leg_gate_logits = gate_outputs.direct_leg_gate_logits
        direct_leg_scale = gate_outputs.direct_leg_scale
        direct_leg_scale_log = gate_outputs.direct_leg_scale_log
        direct_leg_scale_log_raw = gate_outputs.direct_leg_scale_log_raw
        return (
            direct_out,
            direct_leg_omega,
            direct_leg_omega_raw,
            direct_leg_gate,
            direct_leg_gate_logits,
            direct_leg_scale,
            direct_leg_scale_log,
            direct_leg_scale_log_raw,
            direct_leg_side_sign_gate,
        )

    def _forward_non_side_leg_residual(
        self,
        *,
        direct_out: torch.Tensor,
        direct_flat: torch.Tensor,
        direct_feat: torch.Tensor,
        plan_in: Any,
        meas_in: Any,
        phase_in_direct: Optional[torch.Tensor],
        batch_size: int,
        query_steps: int,
        device: torch.device,
        leg_cross_leg_ablate_mode: str,
    ):
        direct_leg_omega = None
        direct_leg_omega_raw = None
        direct_leg_gate = None
        direct_leg_gate_logits = None
        direct_leg_scale = None
        direct_leg_scale_log = None
        direct_leg_scale_log_raw = None
        direct_leg_side_sign_gate = None

        idx = getattr(self, "direct_pose_leg_joint_idx_tensor", None)
        rot_sl = getattr(self, "direct_pose_leg_rot6d_slice", None)
        if (
            (not torch.is_tensor(idx))
            or (not isinstance(rot_sl, slice))
            or rot_sl.start is None
            or rot_sl.stop is None
        ):
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        K = int(idx.numel())
        if K <= 0:
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        B = int(batch_size)
        Tq = int(query_steps)
        leg_in = direct_flat
        leg_shape = tuple(int(dim) for dim in leg_in.shape)
        if leg_in.ndim != 2 or int(leg_in.shape[0]) != B * Tq:
            raise RuntimeError(
                "direct_pose leg flat feature contract failed: expected leading dim "
                f"B*Tq={B * Tq}, got shape={leg_shape}."
            )
        if bool(getattr(self, "direct_pose_leg_detach_feat", False)):
            leg_in = leg_in.detach()

        rot_dim = int(rot_sl.stop - rot_sl.start)
        if rot_dim <= 0 or (rot_dim % 6) != 0:
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        J = int(rot_dim // 6)
        idx_use = idx.to(device=device)
        if not bool(torch.all((idx_use >= 0) & (idx_use < J)).detach().cpu().item()):
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        leg_mode = str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip()
        leg_delta = self._resolve_direct_pose_non_side_leg_delta(
            leg_in=leg_in,
            direct_feat=direct_feat,
            plan_in=plan_in,
            meas_in=meas_in,
            phase_in_direct=phase_in_direct,
            batch_size=B,
            query_steps=Tq,
            joint_count=K,
            ablation_mode=leg_cross_leg_ablate_mode,
        )

        if leg_mode == "so3":
            if leg_delta.shape[-1] != 3 * K:
                return (
                    direct_out,
                    direct_leg_omega,
                    direct_leg_omega_raw,
                    direct_leg_gate,
                    direct_leg_gate_logits,
                    direct_leg_scale,
                    direct_leg_scale_log,
                    direct_leg_scale_log_raw,
                    direct_leg_side_sign_gate,
                )
            omega_leg = self._prepare_direct_pose_leg_omega(
                omega_parts=(leg_delta.view(B, Tq, K, 3),),
                batch_size=B,
                query_steps=Tq,
                joint_count=K,
                error_prefix="direct_pose leg",
            )
            gate_outputs = self._apply_direct_pose_leg_gate_outputs(
                omega_leg=omega_leg,
                gate_head=getattr(self, "direct_pose_leg_gate_head", None),
                head_inputs=(leg_in,),
                batch_size=B,
                query_steps=Tq,
                joint_count=K,
                branch_joint_count=K,
                error_prefix="direct_pose leg",
                gate_head_name="direct_pose_leg_gate_head",
            )
            direct_leg_omega_raw = omega_leg
            direct_leg_omega = gate_outputs.omega_eff
            direct_leg_gate = gate_outputs.direct_leg_gate
            direct_leg_gate_logits = gate_outputs.direct_leg_gate_logits
            direct_leg_scale = gate_outputs.direct_leg_scale
            direct_leg_scale_log = gate_outputs.direct_leg_scale_log
            direct_leg_scale_log_raw = gate_outputs.direct_leg_scale_log_raw
            return (
                direct_out,
                direct_leg_omega,
                direct_leg_omega_raw,
                direct_leg_gate,
                direct_leg_gate_logits,
                direct_leg_scale,
                direct_leg_scale_log,
                direct_leg_scale_log_raw,
                direct_leg_side_sign_gate,
            )

        if (
            leg_delta.ndim == 3
            and int(leg_delta.shape[0]) == B
            and int(leg_delta.shape[1]) == Tq
            and int(leg_delta.shape[-1]) == 6 * K
        ):
            delta_rot = direct_out.new_zeros((B, Tq, J, 6))
            delta_rot[:, :, idx_use, :] = leg_delta.view(B, Tq, K, 6)
            delta_full = torch.zeros_like(direct_out)
            delta_full[..., rot_sl] = delta_rot.view(B, Tq, -1)
            direct_out = direct_out + delta_full

        return (
            direct_out,
            direct_leg_omega,
            direct_leg_omega_raw,
            direct_leg_gate,
            direct_leg_gate_logits,
            direct_leg_scale,
            direct_leg_scale_log,
            direct_leg_scale_log_raw,
            direct_leg_side_sign_gate,
        )

    def _init_contact_plan_debug_buffers(
        self,
        enabled: bool,
    ) -> Optional[_ContactPlanDebugBuffers]:
        if not enabled:
            return None
        return _ContactPlanDebugBuffers(
            contacts_plan_logits_base=[],
            contacts_plan_logits_phase=[],
            contacts_plan_logits_time=[],
            contacts_plan_logits_raw=[],
        )

    def _append_contact_plan_debug_logits(
        self,
        buffers: Optional[_ContactPlanDebugBuffers],
        *,
        logits_raw: torch.Tensor,
        logits_base: torch.Tensor,
        logits_phase: Optional[torch.Tensor] = None,
        logits_time: Optional[torch.Tensor] = None,
    ) -> None:
        if buffers is None:
            return
        zero_term = logits_base.new_zeros(logits_base.shape)
        buffers.contacts_plan_logits_raw.append(logits_raw)
        buffers.contacts_plan_logits_base.append(logits_base)
        buffers.contacts_plan_logits_phase.append(zero_term if logits_phase is None else logits_phase)
        buffers.contacts_plan_logits_time.append(zero_term if logits_time is None else logits_time)

    def _finalize_contact_plan_debug_logits(
        self,
        buffers: Optional[_ContactPlanDebugBuffers],
    ) -> _ContactPlanDebugLogits:
        if buffers is None:
            return _DEFAULT_CONTACT_PLAN_DEBUG_LOGITS

        def _stack(seq: list[torch.Tensor]) -> Optional[torch.Tensor]:
            try:
                return torch.stack(seq, dim=1) if seq else None
            except RuntimeError:
                return None

        return _ContactPlanDebugLogits(
            contacts_plan_logits_base=_stack(buffers.contacts_plan_logits_base),
            contacts_plan_logits_phase=_stack(buffers.contacts_plan_logits_phase),
            contacts_plan_logits_time=_stack(buffers.contacts_plan_logits_time),
            contacts_plan_logits_raw=_stack(buffers.contacts_plan_logits_raw),
        )

    def _write_contact_plan_debug_logits(
        self,
        result: Dict[str, Any],
        debug_logits: _ContactPlanDebugLogits,
        *,
        is_single: bool,
        keys: Optional[tuple[str, ...]] = None,
    ) -> None:
        if keys is None:
            keys = _CONTACT_PLAN_DEBUG_LOGIT_KEYS
        else:
            keys = tuple(keys)
        for key in keys:
            value = getattr(debug_logits, key, None)
            if value is not None and torch.is_tensor(value):
                result[key] = value.squeeze(1) if is_single else value

    def _canonicalize_direct_hint_override(
        self, override: Any, *, batch_size: int, seq_len: int, target_c: int, device: torch.device, dtype: torch.dtype, detach: bool
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(override):
            return None
        ov = override.detach() if detach else override
        if ov.ndim == 1:
            ov = ov.view(1, 1, -1)
        elif ov.ndim == 2:
            ov = ov.unsqueeze(1)
        if ov.ndim != 3:
            return None
        if ov.shape[0] == 1 and batch_size > 1:
            ov = ov.expand(batch_size, -1, -1)
        if ov.shape[1] == 1 and seq_len > 1:
            ov = ov.expand(-1, seq_len, -1)
        if target_c > 0 and ov.shape[-1] != target_c:
            if ov.shape[-1] > target_c:
                ov = ov[..., :target_c]
            else:
                ov = F.pad(ov, (0, target_c - ov.shape[-1]))
        return ov.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _apply_direct_hint_override(
        self,
        base: Optional[torch.Tensor],
        *,
        override: Any,
        fallback_like: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        detach: bool,
    ) -> Optional[torch.Tensor]:
        if override is None:
            return base
        target_c = int(fallback_like.shape[-1])
        ov = self._canonicalize_direct_hint_override(
            override,
            batch_size=batch_size,
            seq_len=seq_len,
            target_c=target_c,
            device=device,
            dtype=dtype,
            detach=detach,
        )
        return ov if torch.is_tensor(ov) else base

    def _canonicalize_contacts_meas_inputs(
        self,
        contacts_input: Any,
        meas_logits_prev: Any,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        contact_dim = max(0, int(self.contact_dim))

        def _pad_or_truncate_channels(value: torch.Tensor) -> torch.Tensor:
            if int(value.shape[-1]) > contact_dim:
                value = value[..., :contact_dim]
            elif int(value.shape[-1]) < contact_dim:
                value = F.pad(value, (0, contact_dim - int(value.shape[-1])))
            return value.contiguous()

        def _canonicalize_contacts_seq(value: Any) -> torch.Tensor:
            if not torch.is_tensor(value):
                raise TypeError("contacts_input must be a tensor")
            meas = value.to(device=device, dtype=dtype)
            if meas.ndim == 1:
                meas = meas.view(1, 1, -1)
            elif meas.ndim == 2:
                if int(meas.shape[0]) not in (1, int(batch_size)):
                    raise ValueError(
                        f"contacts batch mismatch: got {int(meas.shape[0])}, expected 1 or {int(batch_size)}"
                    )
                meas = meas.unsqueeze(1)
            elif meas.ndim == 3:
                if int(meas.shape[0]) not in (1, int(batch_size)):
                    raise ValueError(
                        f"contacts batch mismatch: got {int(meas.shape[0])}, expected 1 or {int(batch_size)}"
                    )
                if int(meas.shape[1]) not in (1, int(seq_len)):
                    raise ValueError(
                        f"contacts time mismatch: got {int(meas.shape[1])}, expected 1 or {int(seq_len)}"
                    )
            else:
                raise ValueError(
                    f"contacts expects shape (C,), (B,C), or (B,T,C), got {tuple(meas.shape)}"
                )
            if int(meas.shape[0]) == 1 and int(batch_size) > 1:
                meas = meas.expand(int(batch_size), -1, -1)
            if int(meas.shape[1]) == 1 and int(seq_len) > 1:
                meas = meas.expand(-1, int(seq_len), -1)
            return _pad_or_truncate_channels(meas)

        def _canonicalize_meas_prev(value: Any) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if not torch.is_tensor(value):
                raise TypeError("meas_logits_prev must be a tensor")
            prev = value.to(device=device, dtype=dtype)
            if prev.ndim == 1:
                prev = prev.view(1, -1)
            elif prev.ndim == 2:
                pass
            elif prev.ndim == 3:
                if int(prev.shape[0]) not in (1, int(batch_size)):
                    raise ValueError(
                        f"meas_logits_prev batch mismatch: got {int(prev.shape[0])}, expected 1 or {int(batch_size)}"
                    )
                if int(prev.shape[1]) != 1:
                    raise ValueError(
                        f"meas_logits_prev time mismatch: got {int(prev.shape[1])}, expected 1"
                    )
                prev = prev[:, 0]
            else:
                raise ValueError(
                    f"meas_logits_prev expects shape (C,), (B,C), or (B,1,C), got {tuple(prev.shape)}"
                )
            if int(prev.shape[0]) not in (1, int(batch_size)):
                raise ValueError(
                    f"meas_logits_prev batch mismatch: got {int(prev.shape[0])}, expected 1 or {int(batch_size)}"
                )
            if int(prev.shape[0]) == 1 and int(batch_size) > 1:
                prev = prev.expand(int(batch_size), -1)
            return _pad_or_truncate_channels(prev)

        contacts_meas: Optional[torch.Tensor] = None
        if contacts_input is not None:
            try:
                contacts_meas = _canonicalize_contacts_seq(contacts_input)
            except (AttributeError, RuntimeError, TypeError):
                contacts_meas = None
        if contacts_meas is None:
            contacts_meas = torch.zeros((int(batch_size), int(seq_len), contact_dim), device=device, dtype=dtype)

        meas_prev_t: Optional[torch.Tensor] = None
        if meas_logits_prev is not None:
            try:
                meas_prev_t = _canonicalize_meas_prev(meas_logits_prev)
            except (AttributeError, RuntimeError, TypeError):
                meas_prev_t = None

        delta_meas = torch.zeros_like(contacts_meas)
        if int(seq_len) > 1:
            delta_meas[:, 1:] = contacts_meas[:, 1:] - contacts_meas[:, :-1]
        if meas_prev_t is not None and int(seq_len) > 0:
            delta_meas[:, 0] = contacts_meas[:, 0] - meas_prev_t
        return contacts_meas, delta_meas, meas_prev_t

    def _apply_direct_pose_leg_side_plan_other_ablation(
        self,
        plan_other_r: torch.Tensor,
        plan_other_l: torch.Tensor,
        *,
        ablation_mode: Optional[str],
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ab = str(ablation_mode or "none").strip().lower()
        if ab == "none":
            return plan_other_r, plan_other_l
        if ab == "zero":
            return plan_other_r * 0.0, plan_other_l * 0.0
        if ab == "roll_batch" and int(batch_size) > 1:
            return plan_other_r.roll(shifts=1, dims=0), plan_other_l.roll(shifts=1, dims=0)
        if ab == "roll_time" and int(seq_len) > 1:
            return plan_other_r.roll(shifts=1, dims=1), plan_other_l.roll(shifts=1, dims=1)
        return plan_other_r, plan_other_l

    def _compute_direct_pose_leg_cross_leg_ablation(
        self, *, leg_in: torch.Tensor, direct_feat: Optional[torch.Tensor], plan_in: Optional[torch.Tensor],
        meas_in: Optional[torch.Tensor], phase_in_direct: Optional[torch.Tensor], batch_size: int, seq_len: int,
        joint_count: int, ablation_mode: Optional[str],
    ) -> Optional[torch.Tensor]:
        ab = str(ablation_mode or "none").strip().lower()
        if ab == "none":
            return None
        if getattr(self, "direct_pose_leg_head", None) is None:
            return None
        try:
            contact_dim = int(getattr(self, "contact_dim", 0) or 0)
        except (TypeError, ValueError):
            contact_dim = 0
        try:
            joint_names = list(getattr(self, "direct_pose_leg_joint_names", []) or [])
        except (TypeError, ValueError):
            joint_names = []
        if not (torch.is_tensor(leg_in) and contact_dim >= 2 and isinstance(joint_names, list) and len(joint_names) == joint_count):
            return None
        joint_names_lower = [str(name).lower() for name in joint_names]
        pos_r = [idx for idx, name in enumerate(joint_names_lower) if name.endswith(("_r", "right"))]
        pos_l = [idx for idx, name in enumerate(joint_names_lower) if name.endswith(("_l", "left"))]
        if not (pos_r and pos_l and (len(pos_r) + len(pos_l) == joint_count)):
            return None

        x = leg_in.reshape(batch_size, seq_len, -1)
        total_dim = int(x.shape[-1])
        direct_dim = int(direct_feat.shape[-1]) if torch.is_tensor(direct_feat) else 0
        plan_dim = int(plan_in.shape[-1]) if torch.is_tensor(plan_in) else 0
        phase_dim = int(phase_in_direct.shape[-1]) if torch.is_tensor(phase_in_direct) else 0
        meas_dim_raw = int(meas_in.shape[-1]) if torch.is_tensor(meas_in) else 0
        meas_dim = 0
        if total_dim == (direct_dim + plan_dim + meas_dim_raw + phase_dim):
            meas_dim = meas_dim_raw
        elif total_dim == (direct_dim + plan_dim + phase_dim):
            meas_dim = 0

        ch_r = int(getattr(self, "direct_pose_leg_contact_ch_r", 1) or 0)
        ch_l = int(getattr(self, "direct_pose_leg_contact_ch_l", 0) or 0)
        ch_r = max(0, min(contact_dim - 1, ch_r))
        ch_l = max(0, min(contact_dim - 1, ch_l))

        x_r = x.clone()
        x_l = x.clone()
        if direct_dim > 0 and plan_dim == contact_dim and total_dim == (direct_dim + plan_dim + meas_dim + phase_dim):
            off_plan = direct_dim
            off_meas = off_plan + plan_dim
            off_phase = off_meas + meas_dim

            def _ablate(xx: torch.Tensor, ch: int) -> None:
                if ab == "zero":
                    xx[..., off_plan + ch] = 0.0
                    if meas_dim > 0:
                        xx[..., off_meas + ch] = 0.0
                    if phase_dim == 2 * contact_dim:
                        start = off_phase + 2 * ch
                        xx[..., start:start + 2] = 0.0
                elif ab == "roll_batch" and int(batch_size) > 1:
                    xx[..., off_plan + ch] = xx[..., off_plan + ch].roll(shifts=1, dims=0)
                    if meas_dim > 0:
                        xx[..., off_meas + ch] = xx[..., off_meas + ch].roll(shifts=1, dims=0)
                    if phase_dim == 2 * contact_dim:
                        start = off_phase + 2 * ch
                        xx[..., start:start + 2] = xx[..., start:start + 2].roll(shifts=1, dims=0)
                elif ab == "roll_time" and int(seq_len) > 1:
                    xx[..., off_plan + ch] = xx[..., off_plan + ch].roll(shifts=1, dims=1)
                    if meas_dim > 0:
                        xx[..., off_meas + ch] = xx[..., off_meas + ch].roll(shifts=1, dims=1)
                    if phase_dim == 2 * contact_dim:
                        start = off_phase + 2 * ch
                        xx[..., start:start + 2] = xx[..., start:start + 2].roll(shifts=1, dims=1)
        elif direct_dim > 0 and phase_dim == 2 * contact_dim and total_dim == (direct_dim + phase_dim):
            off_phase = direct_dim

            def _ablate(xx: torch.Tensor, ch: int) -> None:
                start = off_phase + 2 * ch
                if ab == "zero":
                    xx[..., start:start + 2] = 0.0
                elif ab == "roll_batch" and int(batch_size) > 1:
                    xx[..., start:start + 2] = xx[..., start:start + 2].roll(shifts=1, dims=0)
                elif ab == "roll_time" and int(seq_len) > 1:
                    xx[..., start:start + 2] = xx[..., start:start + 2].roll(shifts=1, dims=1)
        else:
            return None

        _ablate(x_r, ch_l)
        _ablate(x_l, ch_r)
        flat_r = x_r.reshape(-1, x_r.shape[-1])
        flat_l = x_l.reshape(-1, x_l.shape[-1])
        if bool(getattr(self, "direct_pose_leg_detach_feat", False)):
            flat_r = flat_r.detach()
            flat_l = flat_l.detach()
        out_r = self.direct_pose_leg_head(flat_r).view(batch_size, seq_len, -1)
        out_l = self.direct_pose_leg_head(flat_l).view(batch_size, seq_len, -1)
        if out_r.shape != out_l.shape or out_r.shape[-1] not in (3 * joint_count, 6 * joint_count):
            return None
        value_dim = int(out_r.shape[-1] // joint_count)
        merged = out_r.view(batch_size, seq_len, joint_count, value_dim).clone()
        merged[:, :, pos_l, :] = out_l.view(batch_size, seq_len, joint_count, value_dim)[:, :, pos_l, :]
        return merged.view(batch_size, seq_len, joint_count * value_dim)

    def enable_adaptive_history(self, module: AdaptiveHistoryModule, *, pose_hist_len: Optional[int] = None) -> None:
        self.adaptive_history_module = module
        try:
            self._adaptive_history_device = next(module.parameters()).device
        except StopIteration:
            self._adaptive_history_device = torch.device('cpu')
        if pose_hist_len is not None:
            self.pose_hist_len = int(pose_hist_len)

    def forward(
        self,
        state: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        contacts: Optional[torch.Tensor] = None,
        angvel: Optional[torch.Tensor] = None,
        pose_history: Optional[torch.Tensor] = None,
        plan_z: Optional[torch.Tensor] = None,
        phase_z: Optional[torch.Tensor] = None,
        phase_event_age: Optional[torch.Tensor] = None,
        meas_logits_prev: Optional[torch.Tensor] = None,
        time_index: Optional[torch.Tensor | int | float] = None,
        rollout_step: Optional[torch.Tensor | int | float] = None,
    ) -> dict[str, torch.Tensor]:
        forward_inputs = self._prepare_forward_inputs(
            state=state,
            cond=cond,
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_history,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
        )
        state = forward_inputs.state
        cond = forward_inputs.cond
        contacts = forward_inputs.contacts
        angvel = forward_inputs.angvel
        pose_history = forward_inputs.pose_history
        plan_z = forward_inputs.plan_z
        phase_z = forward_inputs.phase_z
        phase_event_age = forward_inputs.phase_event_age
        is_single = forward_inputs.is_single
        device = forward_inputs.device
        dtype = forward_inputs.dtype
        B = forward_inputs.batch_size
        Tq = forward_inputs.query_steps
        runtime_controls = forward_inputs.runtime_controls
        contacts_input = forward_inputs.contacts_input
        contacts_enc = forward_inputs.contacts_enc
        def _state_sequence_contract_error(
            field_name: str,
            value: Optional[torch.Tensor],
            feat_dim: int,
            *,
            reason: str,
            actual_ndim: Optional[int] = None,
            actual_shape: Optional[tuple[int, ...]] = None,
        ) -> str:
            if actual_ndim is None:
                ndim_value = getattr(value, "ndim", None)
                actual_ndim = int(ndim_value) if ndim_value is not None else None
            if actual_shape is None:
                shape_value = getattr(value, "shape", None)
                if shape_value is not None:
                    try:
                        actual_shape = tuple(int(dim) for dim in shape_value)
                    except TypeError:
                        actual_shape = tuple(shape_value)
            return (
                f"{field_name} sequence contract failed in EventMotionModel.forward: "
                f"expected {field_name} to be broadcastable to (B={B}, Tq={Tq}, feat_dim={feat_dim}). "
                f"Accepted shapes are 1D `(feat_dim,)`, 2D `(batch_or_1, feat_dim)` broadcast over time, "
                f"or 3D `(batch_or_1, time_or_1, feat_dim)` with batch_or_1 in {{1, {B}}} and "
                f"time_or_1 in {{1, {Tq}}}; rank>3 inputs must reshape cleanly to `(B, Tq, feat_dim)`. "
                f"2D inputs are interpreted as batch-major, not time-major. "
                f"Got type={type(value).__name__}, ndim={actual_ndim}, shape={actual_shape}. "
                f"{reason}"
            )

        def _expand_state_sequence(
            value: Optional[torch.Tensor],
            feat_dim: int,
            *,
            field_name: str,
        ) -> Optional[torch.Tensor]:
            if value is None or feat_dim <= 0:
                return None
            try:
                seq = value.to(device=device, dtype=dtype)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    _state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        reason="Tensor conversion to the forward device/dtype failed; no compatibility fallback exists.",
                    )
                ) from exc

            input_ndim = int(seq.ndim)
            input_shape = tuple(int(dim) for dim in seq.shape)
            if seq.ndim == 1:
                if seq.shape[0] != feat_dim:
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                f"1D input must carry exactly feat_dim={feat_dim} features, "
                                f"but got {int(seq.shape[0])}."
                            ),
                        )
                    )
                seq = seq.view(1, 1, feat_dim)
            elif seq.ndim == 2:
                if seq.shape[-1] != feat_dim:
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                f"2D input must carry exactly feat_dim={feat_dim} features, "
                                f"but got {int(seq.shape[-1])}."
                            ),
                        )
                    )
                if seq.shape[0] not in (1, B):
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                f"2D input leading axis must be 1 or B={B}; got {int(seq.shape[0])}. "
                                "Pass a 3D tensor for explicit per-step inputs."
                            ),
                        )
                    )
                seq = seq.unsqueeze(1)
            elif seq.ndim == 3:
                if seq.shape[-1] != feat_dim:
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                f"3D input must carry exactly feat_dim={feat_dim} features, "
                                f"but got {int(seq.shape[-1])}."
                            ),
                        )
                    )
                if seq.shape[0] not in (1, B):
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=f"3D input batch axis must be 1 or B={B}; got {int(seq.shape[0])}.",
                        )
                    )
                if seq.shape[1] not in (1, Tq):
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=f"3D input time axis must be 1 or Tq={Tq}; got {int(seq.shape[1])}.",
                        )
                    )
            else:
                try:
                    seq = seq.reshape(B, Tq, -1)
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                "rank>3 input is not reshape-compatible with `(B, Tq, feat_dim)`; "
                                "no compatibility fallback exists."
                            ),
                        )
                    ) from exc
                if seq.shape[-1] != feat_dim:
                    raise RuntimeError(
                        _state_sequence_contract_error(
                            field_name,
                            value,
                            feat_dim,
                            actual_ndim=input_ndim,
                            actual_shape=input_shape,
                            reason=(
                                f"reshaped input must carry exactly feat_dim={feat_dim} features, "
                                f"but got {int(seq.shape[-1])}."
                            ),
                        )
                    )
            if seq.shape[0] == 1 and B > 1:
                seq = seq.expand(B, -1, -1)
            if seq.shape[1] == 1 and Tq > 1:
                seq = seq.expand(-1, Tq, -1)
            return seq.contiguous()
        contact_clock_defaults = self._init_contact_clock_forward_defaults()
        soft_period = contact_clock_defaults.soft_period
        contacts_meas = contact_clock_defaults.contacts_meas
        event_clock_delta_meas = contact_clock_defaults.event_clock_delta_meas
        event_clock_lr_diff = contact_clock_defaults.event_clock_lr_diff
        event_clock_lambda_corr = contact_clock_defaults.event_clock_lambda_corr
        event_clock_lambda_logit = contact_clock_defaults.event_clock_lambda_logit
        event_clock_dynamic_prior = contact_clock_defaults.event_clock_dynamic_prior
        event_clock_delta_z = contact_clock_defaults.event_clock_delta_z
        _pose_hist_processed = contact_clock_defaults.pose_hist_processed
        contacts_plan = contact_clock_defaults.contacts_plan
        plan_z_next = contact_clock_defaults.plan_z_next
        plan_feat_for_inject = contact_clock_defaults.plan_feat_for_inject
        contacts_plan_logits = contact_clock_defaults.contacts_plan_logits
        contact_plan_debug_logits = contact_clock_defaults.contact_plan_debug_logits
        time_pe_direct = contact_clock_defaults.time_pe_direct
        phase_z_in_direct = contact_clock_defaults.phase_z_in_direct
        leg_side_cue_in = contact_clock_defaults.leg_side_cue_in
        h_temporal = None
        h_final = None
        result = None
        e_t = None

        def _run_contact_plan_stage() -> None:
            nonlocal angvel
            nonlocal pose_history
            nonlocal soft_period
            nonlocal contacts_meas
            nonlocal e_t
            nonlocal event_clock_delta_meas
            nonlocal event_clock_lr_diff
            nonlocal event_clock_lambda_corr
            nonlocal event_clock_lambda_logit
            nonlocal event_clock_dynamic_prior
            nonlocal event_clock_delta_z
            nonlocal _pose_hist_processed
            nonlocal contacts_plan
            nonlocal plan_z_next
            nonlocal plan_feat_for_inject
            nonlocal contacts_plan_logits
            nonlocal contact_plan_debug_logits
            nonlocal time_pe_direct
            nonlocal phase_z_in_direct
            nonlocal leg_side_cue_in
            nonlocal contacts_enc

            if angvel is None and self.angvel_dim > 0:
                angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
            if pose_history is None and self.pose_hist_dim > 0:
                pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

            # ---- Stage B: Contact-plan / Event-Clock state machine ----
            # - contacts_plan is produced from cond history via a GRUCell and stays independent of pose.
            # - plan_z is the only cached state needed at inference.
            if self.contact_plan_enable and self.contact_plan_cell is not None and self.contact_plan_head is not None:
                B, Tq, _ = state.shape
                h_plan = int(self.contact_plan_hidden)
                if plan_z is None:
                    init_mode = str(getattr(self, "contact_plan_init_mode", "learnable") or "learnable").lower().strip()
                    plan_z_t = None
                    if init_mode == "zeros":
                        plan_z_t = torch.zeros((B, h_plan), device=device, dtype=dtype)
                    elif init_mode in ("obs", "learnable+obs"):
                        init_head = getattr(self, "contact_plan_init_head", None)
                        if init_head is not None and int(getattr(self, "_contact_plan_init_obs_dim", 0) or 0) > 0:
                            obs0 = None
                            try:
                                obs_feats = []
                                # Contacts (meas) at t=0
                                if int(self.contact_dim) > 0:
                                    if contacts_input is None:
                                        c0 = torch.zeros((B, int(self.contact_dim)), device=device, dtype=dtype)
                                    else:
                                        c_in = contacts_input.to(device=device, dtype=dtype)
                                        if c_in.ndim == 3:
                                            c0 = c_in[:, 0]
                                        elif c_in.ndim == 2:
                                            c0 = c_in
                                        elif c_in.ndim == 1:
                                            c0 = c_in.view(1, -1).expand(B, -1)
                                        else:
                                            c0 = c_in.reshape(B, -1)
                                        if c0.shape[-1] != int(self.contact_dim):
                                            if c0.shape[-1] > int(self.contact_dim):
                                                c0 = c0[..., : int(self.contact_dim)]
                                            else:
                                                c0 = F.pad(c0, (0, int(self.contact_dim) - c0.shape[-1]))
                                    obs_feats.append(c0)
                                # Angvel at t=0
                                if int(self.angvel_dim) > 0:
                                    av_in = angvel
                                    if av_in is None:
                                        av0 = torch.zeros((B, int(self.angvel_dim)), device=device, dtype=dtype)
                                    else:
                                        av = av_in.to(device=device, dtype=dtype)
                                        if av.ndim == 3:
                                            av0 = av[:, 0]
                                        elif av.ndim == 2:
                                            av0 = av
                                        elif av.ndim == 1:
                                            av0 = av.view(1, -1).expand(B, -1)
                                        else:
                                            av0 = av.reshape(B, -1)
                                        if av0.shape[-1] != int(self.angvel_dim):
                                            if av0.shape[-1] > int(self.angvel_dim):
                                                av0 = av0[..., : int(self.angvel_dim)]
                                            else:
                                                av0 = F.pad(av0, (0, int(self.angvel_dim) - av0.shape[-1]))
                                    obs_feats.append(av0)
                                # Pose history at t=0
                                if int(self.pose_hist_dim) > 0:
                                    ph_in = pose_history
                                    if ph_in is None:
                                        ph0 = torch.zeros((B, int(self.pose_hist_dim)), device=device, dtype=dtype)
                                    else:
                                        ph = ph_in.to(device=device, dtype=dtype)
                                        if ph.ndim == 3:
                                            ph0 = ph[:, 0]
                                        elif ph.ndim == 2:
                                            ph0 = ph
                                        elif ph.ndim == 1:
                                            ph0 = ph.view(1, -1).expand(B, -1)
                                        else:
                                            ph0 = ph.reshape(B, -1)
                                        if ph0.shape[-1] != int(self.pose_hist_dim):
                                            if ph0.shape[-1] > int(self.pose_hist_dim):
                                                ph0 = ph0[..., : int(self.pose_hist_dim)]
                                            else:
                                                ph0 = F.pad(ph0, (0, int(self.pose_hist_dim) - ph0.shape[-1]))
                                    obs_feats.append(ph0)
                                if obs_feats:
                                    obs0 = torch.cat(obs_feats, dim=-1)
                                    plan_z_t = init_head(obs0)
                            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                                raise RuntimeError(
                                    "contact_plan observed init failed "
                                    f"(init_mode={init_mode!r}, B={B}, h_plan={h_plan}, "
                                    f"obs_dim={int(getattr(self, '_contact_plan_init_obs_dim', 0) or 0)}, "
                                    f"obs0.shape={tuple(int(dim) for dim in obs0.shape) if torch.is_tensor(obs0) else None})"
                                ) from exc

                        if init_mode == "learnable+obs":
                            init_z = getattr(self, "contact_plan_init_z", None)
                            if torch.is_tensor(init_z) and init_z.numel() == h_plan:
                                init_z_t = init_z.to(device=device, dtype=dtype).view(1, h_plan).expand(B, h_plan)
                            else:
                                init_z_t = torch.zeros((B, h_plan), device=device, dtype=dtype)
                            plan_z_t = init_z_t if plan_z_t is None else (plan_z_t + init_z_t)

                    if plan_z_t is None:
                        init_z = getattr(self, "contact_plan_init_z", None)
                        if torch.is_tensor(init_z) and init_z.numel() == h_plan:
                            plan_z_t = init_z.to(device=device, dtype=dtype).view(1, h_plan).expand(B, h_plan)
                        else:
                            plan_z_t = torch.zeros((B, h_plan), device=device, dtype=dtype)
                else:
                    plan_z_t = plan_z.to(device=device, dtype=dtype)
                    if plan_z_t.ndim == 3 and plan_z_t.size(1) == 1:
                        plan_z_t = plan_z_t[:, 0]
                    if plan_z_t.ndim != 2:
                        plan_z_t = plan_z_t.reshape(B, h_plan)
                cond_seq = cond if cond is not None else torch.zeros((B, Tq, self.cond_dim), device=device, dtype=dtype)

                # ---- Optional: time/clock embeddings (shared time index, different dims per consumer) ----
                t_grid = None
                want_time_grid = bool(
                    (self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0)
                    or int(getattr(self, "direct_pose_time_pe_dim", 0) or 0) > 0
                )
                if want_time_grid:
                    # Build per-step time index: either provided directly as (B,T) / (B,) / scalar, or default to arange(Tq).
                    try:
                        if time_index is None:
                            t_grid = torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0).expand(B, Tq)
                        elif isinstance(time_index, (int, float)):
                            base = torch.full((B, 1), float(time_index), device=device, dtype=dtype)
                            t_grid = base + torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0)
                        elif torch.is_tensor(time_index):
                            t_in = time_index.to(device=device, dtype=dtype)
                            if t_in.dim() == 0:
                                base = t_in.view(1, 1).expand(B, 1)
                                t_grid = base + torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0)
                            elif t_in.dim() == 1:
                                if t_in.numel() == 1:
                                    base = t_in.view(1, 1).expand(B, 1)
                                elif t_in.numel() == B:
                                    base = t_in.view(B, 1)
                                else:
                                    raise ValueError(f"1D tensor must have 1 or B={B} elements, got {int(t_in.numel())}.")
                                t_grid = base + torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0)
                            elif t_in.dim() == 2:
                                # Either (B,Tq) or broadcastable; treat it as explicit time per step.
                                if t_in.shape[0] not in (1, B):
                                    raise ValueError(f"2D tensor batch axis must be 1 or B={B}, got {int(t_in.shape[0])}.")
                                if t_in.shape[1] < Tq:
                                    raise ValueError(f"2D tensor time axis must have at least Tq={Tq} steps, got {int(t_in.shape[1])}.")
                                if t_in.shape[0] == 1 and B > 1:
                                    t_in = t_in.expand(B, -1)
                                t_grid = t_in[:, :Tq]
                            else:
                                raise ValueError(f"time_index tensor rank must be 0, 1, or 2, got {int(t_in.dim())}.")
                        else:
                            raise TypeError(f"time_index must be None, a scalar, or a torch.Tensor; got {type(time_index).__name__}.")
                        if t_grid is None or t_grid.shape != (B, Tq):
                            shape = tuple(int(dim) for dim in t_grid.shape) if torch.is_tensor(t_grid) else None
                            raise RuntimeError(f"normalized time grid must have shape (B={B}, Tq={Tq}), got {shape}.")
                    except (RuntimeError, ValueError, TypeError, AttributeError, OverflowError) as exc:
                        actual_ndim = int(time_index.dim()) if torch.is_tensor(time_index) else None
                        actual_shape = tuple(int(dim) for dim in time_index.shape) if torch.is_tensor(time_index) else None
                        raise RuntimeError(
                            "time_index contract failed in EventMotionModel.forward: "
                            f"expected None, scalar, 0D tensor, 1D tensor with length 1 or B={B}, "
                            f"or 2D tensor with shape (1|B, >=Tq={Tq}) broadcastable to `(B, Tq)`. "
                            f"Got type={type(time_index).__name__}, ndim={actual_ndim}, shape={actual_shape}, "
                            f"contact_plan_time_pe_dim={int(getattr(self, 'contact_plan_time_pe_dim', 0) or 0)}, "
                            f"direct_pose_time_pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)}."
                        ) from exc

                time_pe = None
                if self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0 and t_grid is not None:
                    base_raw = getattr(self, "_contact_plan_time_pe_base", 10000.0)
                    try:
                        pe_dim = int(self.contact_plan_time_pe_dim)
                        if pe_dim <= 0 or (pe_dim % 2) != 0:
                            raise ValueError(f"contact_plan_time_pe_dim must be a positive even integer, got {pe_dim}.")
                        if t_grid.ndim != 2 or t_grid.shape != (B, Tq):
                            raise ValueError(
                                f"t_grid must have shape (B={B}, Tq={Tq}), got {tuple(int(dim) for dim in t_grid.shape)}."
                            )
                        half = pe_dim // 2
                        idx = torch.arange(0, pe_dim, 2, device=device, dtype=dtype)
                        base = float(base_raw or 10000.0)
                        freqs = 1.0 / torch.pow(torch.full((half,), base, device=device, dtype=dtype), idx / float(pe_dim))
                        angles = t_grid.unsqueeze(-1) * freqs.view(1, 1, half)
                        time_pe = torch.zeros((B, Tq, pe_dim), device=device, dtype=dtype)
                        time_pe[..., 0::2] = torch.sin(angles)
                        time_pe[..., 1::2] = torch.cos(angles)
                    except (RuntimeError, ValueError, TypeError, AttributeError, OverflowError) as exc:
                        raise RuntimeError(
                            "contact_plan time PE construction failed "
                            f"(B={B}, Tq={Tq}, pe_dim={int(getattr(self, 'contact_plan_time_pe_dim', 0) or 0)}, "
                            f"half={int(getattr(self, 'contact_plan_time_pe_dim', 0) or 0) // 2}, "
                            f"t_grid.shape={tuple(int(dim) for dim in t_grid.shape) if torch.is_tensor(t_grid) else None}, "
                            f"base={base_raw!r})"
                        ) from exc

                if int(getattr(self, "direct_pose_time_pe_dim", 0) or 0) > 0 and t_grid is not None:
                    base_raw = getattr(self, "_direct_pose_time_pe_base", 10000.0)
                    try:
                        pe_dim = int(getattr(self, "direct_pose_time_pe_dim", 0) or 0)
                        if pe_dim <= 0 or (pe_dim % 2) != 0:
                            raise ValueError(f"direct_pose_time_pe_dim must be a positive even integer, got {pe_dim}.")
                        if t_grid.ndim != 2 or t_grid.shape != (B, Tq):
                            raise ValueError(
                                f"t_grid must have shape (B={B}, Tq={Tq}), got {tuple(int(dim) for dim in t_grid.shape)}."
                            )
                        half = pe_dim // 2
                        idx = torch.arange(0, pe_dim, 2, device=device, dtype=dtype)
                        base = float(base_raw or 10000.0)
                        freqs = 1.0 / torch.pow(torch.full((half,), base, device=device, dtype=dtype), idx / float(pe_dim))
                        angles = t_grid.unsqueeze(-1) * freqs.view(1, 1, half)
                        time_pe_direct = torch.zeros((B, Tq, pe_dim), device=device, dtype=dtype)
                        time_pe_direct[..., 0::2] = torch.sin(angles)
                        time_pe_direct[..., 1::2] = torch.cos(angles)
                    except (RuntimeError, ValueError, TypeError, AttributeError, OverflowError) as exc:
                        raise RuntimeError(
                            "direct_pose time PE construction failed "
                            f"(B={B}, Tq={Tq}, pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)}, "
                            f"half={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0) // 2}, "
                            f"t_grid.shape={tuple(int(dim) for dim in t_grid.shape) if torch.is_tensor(t_grid) else None}, "
                            f"base={base_raw!r})"
                        ) from exc

                plan_probs: list[torch.Tensor] = []
                plan_logits: list[torch.Tensor] = []
                # Optional debug: decompose contacts_plan logits into:
                #   raw = head(plan_z_raw)               (pre Event-Clock correction; ==base when Event-Clock is off)
                #   base = head(plan_z_t)                (post correction, pre time-PE)
                #   phase = phase/TTA residual term      (optional; added directly on logits)
                #   time = time term added to logits     (scaled by lambda_corr when Event-Clock is on)
                contact_plan_debug_buffers = self._init_contact_plan_debug_buffers(
                    runtime_controls.debug_contact_plan_logits_decomp
                )
                time_bias_scale = runtime_controls.contact_plan_time_bias_scale
                plan_z_seq: Optional[list[torch.Tensor]] = [] if self.contact_plan_inject == "plan_z" else None

                phase_input_seq = _expand_state_sequence(
                    phase_z,
                    int(getattr(self, "_direct_pose_phase_dim", 0) or 0),
                    field_name="phase_z",
                )
                phase_age_seq = _expand_state_sequence(
                    phase_event_age,
                    int(self.contact_dim),
                    field_name="phase_event_age",
                )
                contacts_meas_obs: Optional[torch.Tensor] = None
                delta_meas_obs: Optional[torch.Tensor] = None
                phase_in_direct_dim = int(getattr(self, "_direct_pose_phase_dim", 0) or 0)
                phase_in_direct_seq: Optional[list[torch.Tensor]] = (
                    [] if (bool(getattr(self, "direct_pose_use_phase_z", False)) and phase_in_direct_dim > 0) else None
                )
                phase_in_direct_zero = (
                    torch.zeros((B, phase_in_direct_dim), device=device, dtype=dtype) if phase_in_direct_seq is not None else None
                )
                # Optional: capture per-step stateful cue for the routed shared leg head.
                leg_side_cue_mode = str(getattr(self, "direct_pose_leg_side_cue", "none") or "none").strip().lower()
                leg_side_cue_seq: Optional[list[torch.Tensor]] = None
                leg_side_cue_zero: Optional[torch.Tensor] = None
                if (
                    leg_side_cue_mode not in ("", "none")
                    and bool(getattr(self, "direct_pose_leg_side_routing", False))
                    and getattr(self, "direct_pose_leg_head_shared", None) is not None
                    and int(getattr(self, "contact_dim", 0) or 0) > 0
                ):
                    leg_side_cue_seq = []
                    leg_side_cue_zero = torch.zeros((B, int(self.contact_dim)), device=device, dtype=dtype)

                def _append_contact_plan_direct_step_inputs(step_idx: int, *, event_clock_on: bool) -> None:
                    event_clock_label = "on" if event_clock_on else "off"
                    if phase_in_direct_seq is not None:
                        try:
                            phase_step = phase_in_direct_zero if phase_input_seq is None else phase_input_seq[:, step_idx]
                            if not torch.is_tensor(phase_step) or tuple(int(dim) for dim in phase_step.shape) != (B, phase_in_direct_dim):
                                raise RuntimeError(
                                    f"phase step must have shape (B={B}, phase_dim={phase_in_direct_dim}), "
                                    f"got {tuple(int(dim) for dim in phase_step.shape) if torch.is_tensor(phase_step) else None}."
                                )
                            phase_in_direct_seq.append(phase_step)
                        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                            raise RuntimeError(
                                "direct_pose phase append failed "
                                f"(event_clock={event_clock_label}, step={step_idx}, B={B}, Tq={Tq}, "
                                f"phase_dim={phase_in_direct_dim}, "
                                f"phase_input_seq.shape={tuple(int(dim) for dim in phase_input_seq.shape) if torch.is_tensor(phase_input_seq) else None})"
                            ) from exc
                    if leg_side_cue_seq is not None and leg_side_cue_zero is not None:
                        cue_dim = int(self.contact_dim)
                        try:
                            cue_step = leg_side_cue_zero
                            if leg_side_cue_mode == "phase_event_age" and phase_age_seq is not None:
                                cue_step = phase_age_seq[:, step_idx]
                            if not torch.is_tensor(cue_step) or tuple(int(dim) for dim in cue_step.shape) != (B, cue_dim):
                                raise RuntimeError(
                                    f"cue step must have shape (B={B}, contact_dim={cue_dim}), "
                                    f"got {tuple(int(dim) for dim in cue_step.shape) if torch.is_tensor(cue_step) else None}."
                                )
                            leg_side_cue_seq.append(cue_step)
                        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                            raise RuntimeError(
                                "direct_pose side cue append failed "
                                f"(event_clock={event_clock_label}, step={step_idx}, cue_mode={leg_side_cue_mode!r}, "
                                f"B={B}, Tq={Tq}, contact_dim={cue_dim}, "
                                f"phase_age_seq.shape={tuple(int(dim) for dim in phase_age_seq.shape) if torch.is_tensor(phase_age_seq) else None})"
                            ) from exc

                if self.use_event_clock and self.event_clock_gate is not None and self.event_clock_corrector is not None:
                    # ---- Layer1: contacts_meas + delta_meas + lr_diff (computed before GRU loop) ----
                    # Apply adaptive-history once so meas/period use the same pose_history as downstream heads.
                    if (not _pose_hist_processed) and self.adaptive_history_module is not None and pose_history is not None and pose_history.size(-1) > 0:
                        try:
                            pose_hist_for_module = pose_history
                            if pose_hist_for_module.dim() == 3 and pose_hist_for_module.size(1) == 1:
                                pose_hist_for_module = pose_hist_for_module[:, 0]
                            hist_device = self._adaptive_history_device or pose_hist_for_module.device
                            context_feat = state.mean(dim=1).to(hist_device)
                            pose_hist_for_module = pose_hist_for_module.to(hist_device)
                            pose_hist_flat, _ = self.adaptive_history_module(
                                pose_hist_for_module,
                                context=context_feat,
                            )
                            pose_history = pose_hist_flat.to(device).unsqueeze(1)
                            _pose_hist_processed = True
                        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                            raise RuntimeError(
                                "adaptive history module forward failed "
                                f"(event_clock=on, B={B}, Tq={Tq}, "
                                f"pose_history.shape={tuple(int(dim) for dim in pose_history.shape) if torch.is_tensor(pose_history) else None}, "
                                f"context.shape={tuple(int(dim) for dim in context_feat.shape) if torch.is_tensor(context_feat) else None})"
                            ) from exc

                    contacts_meas, delta_meas, _ = self._canonicalize_contacts_meas_inputs(
                        contacts_input,
                        meas_logits_prev,
                        batch_size=B,
                        seq_len=Tq,
                        device=device,
                        dtype=dtype,
                    )

                    lr_diff = torch.zeros((B, Tq, 1), device=device, dtype=dtype)
                    if int(self.contact_dim) >= 2:
                        lr_diff = (contacts_meas[..., 0:1] - contacts_meas[..., 1:2]).abs()

                    # Detach observation signals to avoid co-adaptation between meas head and correction/gate.
                    contacts_meas_obs = contacts_meas.detach()
                    delta_meas_obs = delta_meas.detach()
                    lr_diff_obs = lr_diff.detach()

                    event_clock_delta_meas = delta_meas_obs
                    event_clock_lr_diff = lr_diff_obs

                    # ---- Layer1b: independent period feature (meas/pose_hist/angvel, not plan) ----
                    period_feat = None
                    if self.period_dim > 0 and self.frozen_encoder is not None and self.frozen_period_head is not None:
                        enc_in = None
                        enc_hidden = None
                        try:
                            enc_feats = []
                            if contacts_meas_obs is not None and contacts_meas_obs.size(-1) > 0:
                                enc_feats.append(contacts_meas_obs)
                            if angvel is not None and angvel.size(-1) > 0:
                                enc_feats.append(angvel)
                            if pose_history is not None and pose_history.size(-1) > 0:
                                enc_feats.append(pose_history)
                            enc_in = torch.cat(enc_feats, dim=-1) if enc_feats else None
                            if enc_in is not None and enc_in.size(-1) == self.encoder_input_dim:
                                with torch.no_grad():
                                    enc_hidden = self.frozen_encoder(enc_in, return_summary=False)
                                    if isinstance(enc_hidden, tuple):
                                        enc_hidden = enc_hidden[-1]
                                    if enc_hidden is not None:
                                        period_feat = torch.tanh(self.frozen_period_head(enc_hidden))
                                        soft_period = period_feat
                        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                            raise RuntimeError(
                                "frozen period feature forward failed "
                                f"(event_clock=on, B={B}, Tq={Tq}, period_dim={int(self.period_dim)}, "
                                f"encoder_input_dim={int(getattr(self, 'encoder_input_dim', 0) or 0)}, "
                                f"enc_in.shape={tuple(int(dim) for dim in enc_in.shape) if torch.is_tensor(enc_in) else None}, "
                                f"enc_hidden.shape={tuple(int(dim) for dim in enc_hidden.shape) if torch.is_tensor(enc_hidden) else None})"
                            ) from exc

                    # ---- Layer2+3: gated residual correction inside GRU loop ----
                    lambda_corr_seq: list[torch.Tensor] = []
                    lambda_logit_seq: list[torch.Tensor] = []
                    dyn_prior_seq: list[torch.Tensor] = []
                    delta_z_seq: list[torch.Tensor] = []

                    def _step_contact_plan_event_clock(step_idx: int) -> None:
                        nonlocal plan_z_t

                        _append_contact_plan_direct_step_inputs(step_idx, event_clock_on=True)

                        plan_in_t = cond_seq[:, step_idx]
                        plan_z_raw = self.contact_plan_cell(plan_in_t, plan_z_t)

                        logits_raw = self.contact_plan_head(plan_z_raw)
                        plan_raw = torch.sigmoid(logits_raw)
                        meas_t = contacts_meas_obs[:, step_idx]
                        err_raw = plan_raw - meas_t

                        delta_meas_t = delta_meas_obs[:, step_idx]
                        lr_diff_t = lr_diff_obs[:, step_idx]
                        period_t = period_feat[:, step_idx] if (period_feat is not None and period_feat.ndim == 3) else None

                        lam_corr_t, lam_logit_t, dyn_prior_t = self.event_clock_gate(
                            err_raw=err_raw,
                            delta_meas=delta_meas_t,
                            lr_diff=lr_diff_t,
                            period_feat=period_t,
                        )
                        plan_z_t, delta_z_t = self.event_clock_corrector(
                            plan_z_raw=plan_z_raw,
                            contacts_meas=meas_t,
                            delta_meas=delta_meas_t,
                            err_raw=err_raw,
                            period_feat=period_t,
                            lambda_corr=lam_corr_t,
                        )
                        if plan_z_seq is not None:
                            plan_z_seq.append(plan_z_t)

                        logits_base = self.contact_plan_head(plan_z_t)
                        time_term = None
                        if time_pe is not None and self.contact_plan_time_head is not None:
                            try:
                                time_bias = self.contact_plan_time_head(time_pe[:, step_idx])
                                time_term = lam_corr_t * (time_bias * time_bias_scale)
                            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                                raise RuntimeError(
                                    "contact_plan time bias forward failed "
                                    f"(event_clock=on, step={step_idx}, B={B}, Tq={Tq}, "
                                    f"time_pe.shape={tuple(int(dim) for dim in time_pe.shape)}, "
                                    f"time_bias_scale={time_bias_scale})"
                                ) from exc
                        self._append_contact_plan_debug_logits(
                            contact_plan_debug_buffers,
                            logits_raw=logits_raw,
                            logits_base=logits_base,
                            logits_time=time_term,
                        )
                        logits = logits_base
                        if time_term is not None:
                            logits = logits + time_term
                        plan_logits.append(logits)
                        plan_probs.append(torch.sigmoid(logits))

                        lambda_corr_seq.append(lam_corr_t)
                        lambda_logit_seq.append(lam_logit_t)
                        dyn_prior_seq.append(dyn_prior_t)
                        delta_z_seq.append(delta_z_t)

                    for _t in range(Tq):
                        _step_contact_plan_event_clock(_t)

                    if lambda_corr_seq:
                        event_clock_lambda_corr = torch.stack(lambda_corr_seq, dim=1)
                        event_clock_lambda_logit = torch.stack(lambda_logit_seq, dim=1)
                        event_clock_dynamic_prior = torch.stack(dyn_prior_seq, dim=1)
                        event_clock_delta_z = torch.stack(delta_z_seq, dim=1)
                else:
                    # Event-Clock off: still resolve measurement hints for direct/contact diagnostics.
                    contacts_meas, delta_meas, _ = self._canonicalize_contacts_meas_inputs(
                        contacts_input,
                        meas_logits_prev,
                        batch_size=B,
                        seq_len=Tq,
                        device=device,
                        dtype=dtype,
                    )

                    contacts_meas_obs = contacts_meas.detach()
                    delta_meas_obs = delta_meas.detach()

                    for _t in range(Tq):
                        _append_contact_plan_direct_step_inputs(_t, event_clock_on=False)
                        plan_in_t = cond_seq[:, _t]
                        plan_z_t = self.contact_plan_cell(plan_in_t, plan_z_t)
                        if plan_z_seq is not None:
                            plan_z_seq.append(plan_z_t)
                        logits_base = self.contact_plan_head(plan_z_t)
                        time_term = None
                        if time_pe is not None and self.contact_plan_time_head is not None:
                            try:
                                time_term = self.contact_plan_time_head(time_pe[:, _t]) * time_bias_scale
                            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                                raise RuntimeError(
                                    "contact_plan time bias forward failed "
                                    f"(event_clock=off, step={_t}, B={B}, Tq={Tq}, "
                                    f"time_pe.shape={tuple(int(dim) for dim in time_pe.shape)}, "
                                    f"time_bias_scale={time_bias_scale})"
                                ) from exc
                        self._append_contact_plan_debug_logits(
                            contact_plan_debug_buffers,
                            logits_raw=logits_base,
                            logits_base=logits_base,
                            logits_time=time_term,
                        )
                        logits = logits_base
                        if time_term is not None:
                            logits = logits + time_term
                        plan_logits.append(logits)
                        plan_probs.append(torch.sigmoid(logits))
                contact_plan_final = self._finalize_contact_plan_outputs(
                    plan_probs=plan_probs,
                    plan_logits=plan_logits,
                    phase_in_direct_seq=phase_in_direct_seq,
                    leg_side_cue_seq=leg_side_cue_seq,
                    contact_plan_debug_buffers=contact_plan_debug_buffers,
                    plan_z_t=plan_z_t,
                    plan_z_seq=plan_z_seq,
                    batch_size=B,
                    query_steps=Tq,
                    phase_in_direct_dim=phase_in_direct_dim,
                    leg_side_cue_mode=leg_side_cue_mode,
                )
                contacts_plan = contact_plan_final.contacts_plan
                phase_z_in_direct = contact_plan_final.phase_z_in_direct
                leg_side_cue_in = contact_plan_final.leg_side_cue_in
                contacts_plan_logits = contact_plan_final.contacts_plan_logits
                contact_plan_debug_logits = contact_plan_final.contact_plan_debug_logits
                plan_z_next = contact_plan_final.plan_z_next
                plan_feat_for_inject = contact_plan_final.plan_feat_for_inject

            if contacts_meas is None:
                if contacts_input is not None:
                    contacts_meas = contacts_input.to(device=device, dtype=dtype)
                    if contacts_meas.ndim == 2:
                        contacts_meas = contacts_meas.unsqueeze(1)
                    elif contacts_meas.ndim != 3:
                        raise ValueError(f"contacts expects shape (B,C) or (B,T,C), got {tuple(contacts_meas.shape)}")
            if contacts_meas is None:
                if contacts_plan is not None:
                    contacts_meas = torch.zeros_like(contacts_plan)
                elif self.contact_dim > 0:
                    contacts_meas = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)
            if contacts_plan is not None and contacts_meas is not None:
                if contacts_meas.ndim == 2:
                    contacts_meas = contacts_meas.unsqueeze(1)
                e_t = contacts_plan - contacts_meas.to(device=device, dtype=dtype)

            # Frozen-encoder input for the period hint:
            # - Event-Clock v3 prefers an *independent* meas-derived contact signal (avoids co-drift with plan).
            # - Otherwise, keep the prior train/infer-consistent behavior (prefer plan).
            if self.use_event_clock and contacts_meas is not None:
                contacts_enc = contacts_meas.detach()
            elif contacts_plan is not None:
                contacts_enc = contacts_plan.detach()
            if contacts_enc is None and self.contact_dim > 0:
                contacts_enc = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)

        def _run_motion_core_stage() -> None:
            nonlocal pose_history
            nonlocal _pose_hist_processed
            nonlocal soft_period
            nonlocal h_temporal
            nonlocal h_final
            nonlocal result

            encoder_feats = []
            if contacts_enc is not None and contacts_enc.size(-1) > 0:
                encoder_feats.append(contacts_enc)
            if angvel is not None and angvel.size(-1) > 0:
                encoder_feats.append(angvel)
            if pose_history is not None and pose_history.size(-1) > 0:
                pose_hist_for_module = pose_history
                if pose_hist_for_module.dim() == 3 and pose_hist_for_module.size(1) == 1:
                    pose_hist_for_module = pose_hist_for_module[:, 0]
                if (not _pose_hist_processed) and self.adaptive_history_module is not None:
                    hist_device = self._adaptive_history_device or pose_hist_for_module.device
                    context_feat = state.mean(dim=1).to(hist_device)
                    pose_hist_for_module = pose_hist_for_module.to(hist_device)
                    pose_hist_flat, _ = self.adaptive_history_module(
                        pose_hist_for_module,
                        context=context_feat,
                    )
                    pose_history = pose_hist_flat.to(device).unsqueeze(1)
                    _pose_hist_processed = True
                encoder_feats.append(pose_history)
            encoder_input = torch.cat(encoder_feats, dim=-1) if encoder_feats else None

            x_inputs = [state]
            if cond is not None:
                x_inputs.append(cond)
            if plan_feat_for_inject is not None:
                feat = plan_feat_for_inject.to(device=device, dtype=dtype)
                if self.contact_plan_inject_detach:
                    feat = feat.detach()
                inject_scale = runtime_controls.contact_plan_inject_scale
                if inject_scale != 1.0:
                    feat = feat * inject_scale
                x_inputs.append(feat)
            x = torch.cat(x_inputs, dim=-1)
            # 导出/编译时跳过数据依赖的 guard，避免 torch.export 的 GuardOnDataDependentSymNode
            _skip_guard = _torch_dynamo_is_compiling_safe()
            _skip_guard = _skip_guard or _torch_onnx_is_in_export_safe()
            if not _skip_guard:
                torch._assert(torch.isfinite(x).all(), "[Guard] x to shared_encoder must be finite")
            lin0 = self.shared_encoder[0]
            if not _skip_guard:
                torch._assert(torch.isfinite(lin0.weight).all(), "[Guard] shared_encoder.0 weight must be finite")
                if lin0.bias is not None:
                    torch._assert(torch.isfinite(lin0.bias).all(), "[Guard] shared_encoder.0 bias must be finite")
            x = torch.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
            clip_val = float(getattr(self, 'input_clip', 16.0) or 16.0)
            x = x.clamp(-clip_val, clip_val)
            if not _skip_guard:
                with torch.no_grad():
                    for _idx, _mod in enumerate(self.shared_encoder):
                        if isinstance(_mod, torch.nn.Linear):
                            torch._assert(torch.isfinite(_mod.weight).all(), f"[Guard] shared_encoder.{_idx} weight must be finite")
                            if _mod.bias is not None:
                                torch._assert(torch.isfinite(_mod.bias).all(), f"[Guard] shared_encoder.{_idx} bias must be finite")

            act1 = self.shared_encoder[1]
            z0 = lin0(x)
            y1 = act1(z0)

            # Inject soft hint embedding from frozen encoder (if available)
            enc_hidden = None
            if soft_period is None:
                if (
                    encoder_input is not None
                    and self.frozen_encoder is not None
                    and encoder_input.size(-1) == self.encoder_input_dim
                ):
                    enc_hidden = self.frozen_encoder(encoder_input, return_summary=False)
                    if isinstance(enc_hidden, tuple):
                        enc_hidden = enc_hidden[-1]
                if enc_hidden is not None and self.frozen_period_head is not None:
                    soft_period = torch.tanh(self.frozen_period_head(enc_hidden))
            if self.period_dim > 0 and self.period_encoder is not None and soft_period is not None:
                period_emb = self.period_encoder(soft_period)
                y1 = y1 + period_emb

            h = self.shared_encoder[2:](y1)
            h_temporal = h + self.residual_proj(x)
            h_temporal = torch.nan_to_num(h_temporal, nan=0.0, posinf=1000000.0, neginf=-1000000.0).clamp(-100.0, 100.0)

            B, Tq, H = h_temporal.shape
            Dh = self._pasa_dhead
            scale = 1.0 / _math.sqrt(max(1, Dh))
            cond_for_film = cond
            if cond_for_film is not None and cond_for_film.ndim == 3 and cond_for_film.size(1) > 1:
                cond_for_film = cond_for_film[:, -1]
            if cond_for_film is None or cond_for_film.ndim != 2:
                cond_in = torch.zeros(B, self.cond_dim, device=device, dtype=dtype)
            else:
                cond_in = cond_for_film
            g, b = self._pasa_film(cond_in)
            q_in = self._pasa_lnq(h_temporal)
            Q = self._pasa_q(q_in).view(B, Tq, self._pasa_heads, Dh).transpose(1, 2)
            K = self._pasa_k(h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
            V = self._pasa_v(h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
            attn = torch.softmax(Q * scale @ K.transpose(-1, -2), dim=-1)
            ctx = (attn @ V).transpose(1, 2).contiguous().view(B, Tq, -1)
            attn_out = self._pasa_o(ctx)
            h_final = self.coupling_norm((h_temporal + attn_out) * (1 + g).unsqueeze(1) + b.unsqueeze(1))

            hidden_out = h_final
            out = self.motion_head(h_final)
            if self._bone_adapters and self._bone_adapter_slices:
                delta_full = torch.zeros_like(out)
                for sl, adapter in zip(self._bone_adapter_slices, self._bone_adapters):
                    delta_full[..., sl] = adapter(h_final)
                out = out + delta_full
            if is_single:
                out = out.squeeze(1)
                hidden_out = hidden_out.squeeze(1)
                if soft_period is not None:
                    soft_period = soft_period.squeeze(1)
            result = self._build_forward_base_result(out=out, hidden_out=hidden_out, attn=attn)

        def _run_direct_pose_stage() -> None:
            if not self._should_run_direct_pose_forward(contacts_plan):
                return
            try:
                direct_pose_runtime = self._init_direct_pose_forward_runtime(runtime_controls)
                plan_override = direct_pose_runtime.plan_override
                meas_override = direct_pose_runtime.meas_override
                leg_side_plan_other_ablate_mode = direct_pose_runtime.leg_side_plan_other_ablate_mode
                leg_cross_leg_ablate_mode = direct_pose_runtime.leg_cross_leg_ablate_mode
                plan_in = contacts_plan.detach() if self.direct_pose_detach_plan else contacts_plan
                if self.training and float(getattr(self, "direct_pose_plan_drop_prob", 0.0) or 0.0) > 0.0:
                    p = float(getattr(self, "direct_pose_plan_drop_prob", 0.0) or 0.0)
                    p = max(0.0, min(1.0, p))
                    if p > 0.0:
                        m = (torch.rand(plan_in.shape[:-1] + (1,), device=plan_in.device) < p).to(plan_in.dtype)
                        plan_in = plan_in * (1.0 - m)
                plan_in = self._apply_direct_hint_override(
                    plan_in,
                    override=plan_override,
                    fallback_like=plan_in,
                    batch_size=B,
                    seq_len=Tq,
                    device=device,
                    dtype=dtype,
                    detach=True,
                )

                mode = str(getattr(self, "direct_pose_meas_mode", "concat") or "concat").lower().strip()
                meas_in = None
                if mode in ("concat", "mode_select"):
                    meas_in = contacts_meas
                    if meas_in is None and mode == "concat" and int(self.contact_dim) > 0:
                        meas_in = torch.zeros_like(contacts_plan)
                    if meas_in is not None and meas_in.ndim == 2:
                        meas_in = meas_in.unsqueeze(1)
                    if meas_in is not None:
                        meas_in = meas_in.to(device=device, dtype=dtype)
                        if self.training:
                            drop_p = float(getattr(self, "direct_pose_meas_drop_prob", 0.0) or 0.0)
                            drop_p = max(0.0, min(1.0, drop_p))
                            if drop_p > 0.0:
                                m = (torch.rand(meas_in.shape[:-1] + (1,), device=meas_in.device) < drop_p).to(
                                    meas_in.dtype
                                )
                                meas_in = meas_in * (1.0 - m)
                            noise_std = float(getattr(self, "direct_pose_meas_noise_std", 0.0) or 0.0)
                            if noise_std > 0.0 and _math.isfinite(noise_std):
                                meas_in = meas_in + torch.randn_like(meas_in) * noise_std
                        meas_in = meas_in.clamp(0.0, 1.0)
                    meas_in = self._apply_direct_hint_override(
                        meas_in,
                        override=meas_override,
                        fallback_like=plan_in,
                        batch_size=B,
                        seq_len=Tq,
                        device=device,
                        dtype=dtype,
                        detach=False,
                    )

                # Choose which features feed the direct head.
                direct_feat = cond
                raw_src = getattr(self, "direct_pose_feat_source", "cond")
                try:
                    src = str(raw_src or "cond").lower().strip()
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        "direct_pose_feat_source contract failed: "
                        "expected one of {'cond', 'hidden', 'hidden_pre', 'cond+hidden', 'cond+hidden_pre'} "
                        f"or their documented aliases; got type={type(raw_src).__name__}, value={raw_src!r}."
                    ) from exc
                if src in ("h", "h_final", "hidden_only"):
                    src = "hidden"
                elif src in ("h_pre", "h_temporal", "hidden_pre", "pre", "temporal", "mid"):
                    src = "hidden_pre"
                elif src in ("cond_hidden", "hidden_cond", "concat", "cond+hidden", "hidden+cond"):
                    src = "cond+hidden"
                elif src in ("cond+hidden_pre", "cond_hidden_pre", "hidden_pre+cond", "cond+pre", "pre+cond"):
                    src = "cond+hidden_pre"
                elif src != "cond":
                    raise RuntimeError(
                        "direct_pose_feat_source contract failed: "
                        "expected one of {'cond', 'hidden', 'hidden_pre', 'cond+hidden', 'cond+hidden_pre'} "
                        f"after alias normalization, got {src!r} from value={raw_src!r}."
                    )
                if src == "hidden":
                    direct_feat = h_final
                elif src == "hidden_pre":
                    direct_feat = h_temporal
                elif src == "cond+hidden":
                    direct_feat = torch.cat([cond, h_final], dim=-1)
                elif src == "cond+hidden_pre":
                    direct_feat = torch.cat([cond, h_temporal], dim=-1)
                else:
                    direct_feat = cond
                if torch.is_tensor(time_pe_direct):
                    if direct_feat.ndim != 3:
                        raise RuntimeError(
                            "direct_pose time PE concat failed: "
                            f"expected direct_feat to be rank-3 `(B, Tq, F)` before time concat, "
                            f"got ndim={int(direct_feat.ndim)}, shape={tuple(int(dim) for dim in direct_feat.shape)}."
                        )
                    if int(direct_feat.shape[0]) != B or int(direct_feat.shape[1]) != Tq:
                        raise RuntimeError(
                            "direct_pose time PE concat failed: "
                            f"expected direct_feat prefix `(B={B}, Tq={Tq})`, "
                            f"got shape={tuple(int(dim) for dim in direct_feat.shape)}."
                        )
                    if time_pe_direct.ndim != 3:
                        raise RuntimeError(
                            "direct_pose time PE concat failed: "
                            f"expected time_pe_direct to be rank-3 `(B, Tq, time_pe_dim)`, "
                            f"got ndim={int(time_pe_direct.ndim)}, "
                            f"shape={tuple(int(dim) for dim in time_pe_direct.shape)}."
                        )
                    if int(time_pe_direct.shape[0]) != B or int(time_pe_direct.shape[1]) != Tq:
                        raise RuntimeError(
                            "direct_pose time PE concat failed: "
                            f"expected time_pe_direct prefix `(B={B}, Tq={Tq})`, "
                            f"got shape={tuple(int(dim) for dim in time_pe_direct.shape)}, "
                            f"time_pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)}."
                        )
                    try:
                        time_pe_direct_feat = time_pe_direct.to(device=device, dtype=dtype)
                        direct_feat = torch.cat([direct_feat, time_pe_direct_feat], dim=-1)
                    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                        raise RuntimeError(
                            "direct_pose time PE concat failed "
                            f"(B={B}, Tq={Tq}, direct_feat.shape={tuple(int(dim) for dim in direct_feat.shape)}, "
                            f"time_pe_direct.shape={tuple(int(dim) for dim in time_pe_direct.shape)}, "
                            f"time_pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)})"
                        ) from exc
                phase_in_direct = None
                if bool(getattr(self, "direct_pose_use_phase_z", False)) and int(getattr(self, "_direct_pose_phase_dim", 0) or 0) > 0:
                    phase_in_direct = phase_z_in_direct
                    if phase_in_direct is None:
                        phase_in_direct = torch.zeros(
                            (B, Tq, int(getattr(self, "_direct_pose_phase_dim", 0) or 0)), device=device, dtype=dtype
                        )
                    elif phase_in_direct.ndim == 2:
                        phase_in_direct = phase_in_direct.unsqueeze(1)
                    if phase_in_direct.ndim == 3 and phase_in_direct.shape[1] == 1 and Tq > 1:
                        phase_in_direct = phase_in_direct.expand(-1, Tq, -1)
                    if torch.is_tensor(phase_in_direct):
                        phase_in_direct = phase_in_direct.to(device=device, dtype=dtype)

                if mode == "concat":
                    if str(getattr(self, "direct_pose_phase_z_mode", "concat") or "concat").strip().lower() == "replace_contacts":
                        # Replace the low-bandwidth 2D contact hint (plan+meas) with the higher-bandwidth phase_z_in.
                        if torch.is_tensor(phase_in_direct):
                            hint = phase_in_direct
                        else:
                            hint = torch.zeros(
                                (B, Tq, int(getattr(self, "_direct_pose_phase_dim", 0) or 0)), device=device, dtype=dtype
                            )
                        direct_in = torch.cat([direct_feat, hint], dim=-1)
                    else:
                        if torch.is_tensor(phase_in_direct):
                            direct_in = torch.cat(
                                [direct_feat, plan_in.to(device=device, dtype=dtype), meas_in, phase_in_direct], dim=-1
                            )
                        else:
                            direct_in = torch.cat([direct_feat, plan_in.to(device=device, dtype=dtype), meas_in], dim=-1)
                    direct_flat = direct_in.reshape(-1, direct_in.shape[-1])
                    direct_out = self._forward_direct_pose_readout(direct_flat, B=B, Tq=Tq)
                elif mode == "mode_select":
                    if torch.is_tensor(phase_in_direct):
                        direct_in = torch.cat([direct_feat, plan_in.to(device=device, dtype=dtype), phase_in_direct], dim=-1)
                    else:
                        direct_in = torch.cat([direct_feat, plan_in.to(device=device, dtype=dtype)], dim=-1)
                    direct_flat = direct_in.reshape(-1, direct_in.shape[-1])
                    modes = self.direct_pose_head(direct_flat).view(B, Tq, -1)
                    Dy = int(self.out_motion_dim)
                    if modes.shape[-1] == Dy * 2:
                        y_left, y_right = modes[..., :Dy], modes[..., Dy:]
                        if meas_in is None:
                            w_left = w_right = None
                        elif meas_in.shape[-1] >= 2:
                            w_left = meas_in[..., :1]
                            w_right = meas_in[..., 1:2]
                        elif meas_in.shape[-1] == 1:
                            w_left = meas_in[..., :1]
                            w_right = 1.0 - w_left
                        else:
                            w_left = w_right = None
                        if w_left is None or w_right is None:
                            base = meas_in if torch.is_tensor(meas_in) else y_left
                            w_left = base.new_full(y_left.shape[:-1] + (1,), 0.5)
                            w_right = 1.0 - w_left
                        denom = (w_left + w_right).clamp_min(1e-6)
                        w_left = (w_left / denom).clamp(0.0, 1.0)
                        w_right = (w_right / denom).clamp(0.0, 1.0)
                        direct_out = w_left * y_left + w_right * y_right
                    else:
                        direct_out = modes
                else:
                    if torch.is_tensor(phase_in_direct):
                        direct_in = torch.cat([direct_feat, plan_in.to(device=device, dtype=dtype), meas_in, phase_in_direct], dim=-1)
                    else:
                        direct_in = torch.cat([direct_feat, plan_in.to(device=device, dtype=dtype), meas_in], dim=-1)
                    direct_flat = direct_in.reshape(-1, direct_in.shape[-1])
                    direct_out = self._forward_direct_pose_readout(direct_flat, B=B, Tq=Tq)

                # Optional: leg-specific residual head output (kept separate from out_direct).
                # NOTE: out_direct is in normalized Y space; do NOT apply SO(3) composition here.
                # Any on-manifold leg correction must be applied in RAW space (denorm -> compose -> renorm)
                # by the training/eval harness after denorm/compose/renorm.
                direct_leg_omega = None
                direct_leg_omega_raw = None
                direct_leg_gate = None
                direct_leg_gate_logits = None
                direct_leg_scale = None
                direct_leg_scale_log = None
                direct_leg_scale_log_raw = None
                direct_leg_side_sign_gate = None

                # Optional: add a leg-specific residual on BoneRotations6D only.
                # This keeps non-leg joints unchanged while giving extra capacity to legs.
                if (self.direct_pose_leg_head_shared is not None) and bool(getattr(self, "direct_pose_leg_side_routing", False)):
                    try:
                        (
                            direct_out,
                            direct_leg_omega,
                            direct_leg_omega_raw,
                            direct_leg_gate,
                            direct_leg_gate_logits,
                            direct_leg_scale,
                            direct_leg_scale_log,
                            direct_leg_scale_log_raw,
                            direct_leg_side_sign_gate,
                        ) = self._forward_side_routed_leg_residual(
                            direct_out=direct_out,
                            direct_feat=direct_feat,
                            plan_in=plan_in,
                            meas_in=meas_in,
                            phase_in_direct=phase_in_direct,
                            leg_side_cue_in=leg_side_cue_in,
                            batch_size=B,
                            query_steps=Tq,
                            device=device,
                            dtype=dtype,
                            leg_side_plan_other_ablate_mode=leg_side_plan_other_ablate_mode,
                        )
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                        raise RuntimeError(
                            "direct_pose side-routed leg residual forward failed "
                            f"(B={B}, Tq={Tq}, leg_mode={getattr(self, 'direct_pose_leg_mode', None)!r}, "
                            f"side_routing={getattr(self, 'direct_pose_leg_side_routing', None)!r})"
                        ) from exc
                elif self.direct_pose_leg_head is not None:
                    try:
                        (
                            direct_out,
                            direct_leg_omega,
                            direct_leg_omega_raw,
                            direct_leg_gate,
                            direct_leg_gate_logits,
                            direct_leg_scale,
                            direct_leg_scale_log,
                            direct_leg_scale_log_raw,
                            direct_leg_side_sign_gate,
                        ) = self._forward_non_side_leg_residual(
                            direct_out=direct_out,
                            direct_flat=direct_flat,
                            direct_feat=direct_feat,
                            plan_in=plan_in,
                            meas_in=meas_in,
                            phase_in_direct=phase_in_direct,
                            batch_size=B,
                            query_steps=Tq,
                            device=device,
                            leg_cross_leg_ablate_mode=leg_cross_leg_ablate_mode,
                        )
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                        raise RuntimeError(
                            "direct_pose leg residual forward failed "
                            f"(B={B}, Tq={Tq}, leg_mode={getattr(self, 'direct_pose_leg_mode', None)!r}, "
                            f"side_routing={getattr(self, 'direct_pose_leg_side_routing', None)!r})"
                        ) from exc

                self._write_forward_direct_pose_outputs(
                    result,
                    direct_out=direct_out,
                    direct_leg_omega=direct_leg_omega,
                    direct_leg_omega_raw=direct_leg_omega_raw,
                    direct_leg_gate=direct_leg_gate,
                    direct_leg_gate_logits=direct_leg_gate_logits,
                    direct_leg_scale=direct_leg_scale,
                    direct_leg_scale_log=direct_leg_scale_log,
                    direct_leg_scale_log_raw=direct_leg_scale_log_raw,
                    direct_leg_side_sign_gate=direct_leg_side_sign_gate,
                    is_single=is_single,
                )

            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError("direct_pose forward failed") from exc

        def _run_output_writeback_stage() -> None:
            if contacts_meas is not None:
                result['contacts_meas'] = contacts_meas.squeeze(1) if is_single else contacts_meas

            if contacts_plan is not None:
                result['contacts_plan'] = contacts_plan.squeeze(1) if is_single else contacts_plan
                if contacts_plan_logits is not None and torch.is_tensor(contacts_plan_logits):
                    result['contacts_plan_logits'] = contacts_plan_logits.squeeze(1) if is_single else contacts_plan_logits
                    self._write_contact_plan_debug_logits(
                        result,
                        contact_plan_debug_logits,
                        is_single=is_single,
                        keys=(
                            "contacts_plan_logits_base",
                            "contacts_plan_logits_phase",
                            "contacts_plan_logits_time",
                        ),
                    )
                self._write_contact_plan_debug_logits(
                    result,
                    contact_plan_debug_logits,
                    is_single=is_single,
                    keys=("contacts_plan_logits_raw",),
                )
                if plan_z_next is not None:
                    result['plan_z_next'] = plan_z_next
                if event_clock_lambda_corr is not None:
                    result['event_clock_lambda_corr'] = event_clock_lambda_corr.squeeze(1) if is_single else event_clock_lambda_corr
                if event_clock_lambda_logit is not None:
                    result['event_clock_lambda_logit'] = event_clock_lambda_logit.squeeze(1) if is_single else event_clock_lambda_logit
                if event_clock_dynamic_prior is not None:
                    result['event_clock_dynamic_prior'] = (
                        event_clock_dynamic_prior.squeeze(1) if is_single else event_clock_dynamic_prior
                    )
                if event_clock_delta_z is not None:
                    result['event_clock_delta_z'] = event_clock_delta_z.squeeze(1) if is_single else event_clock_delta_z
                if event_clock_delta_meas is not None:
                    result['event_clock_delta_meas'] = event_clock_delta_meas.squeeze(1) if is_single else event_clock_delta_meas
                if event_clock_lr_diff is not None:
                    result['event_clock_lr_diff'] = event_clock_lr_diff.squeeze(1) if is_single else event_clock_lr_diff
                if e_t is not None:
                    result['contacts_err'] = e_t.squeeze(1) if is_single else e_t

            self._write_forward_lambda_fusion_outputs(
                result,
                h_final=h_final,
                contact_error=e_t,
                rollout_step=rollout_step,
                device=device,
                dtype=dtype,
                is_single=is_single,
                batch_size=B,
                query_steps=Tq,
            )
            self._write_forward_so3_delta_outputs(
                result,
                h_final=h_final,
                contact_error=e_t,
                device=device,
                dtype=dtype,
                is_single=is_single,
            )
            self._write_forward_period_output(result, soft_period=soft_period)

        _run_contact_plan_stage()
        _run_motion_core_stage()
        _run_direct_pose_stage()
        _run_output_writeback_stage()
        if result is None:
            raise RuntimeError("EventMotionModel.forward motion core stage did not produce a result.")
        return result


def _masked_group_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return _masked_group_weighted_mean(values, mask, joint_weights=None)


def _masked_group_weighted_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    joint_weights: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if values is None or mask is None or values.numel() == 0:
        return None
    if mask.numel() != values.shape[-1]:
        return None
    if not bool(mask.any().detach().cpu().item()):
        return None
    group_values = values[..., mask]
    if joint_weights is None:
        return group_values.mean()
    if joint_weights.ndim != 1 or joint_weights.numel() != values.shape[-1]:
        return None
    group_weights = joint_weights.to(device=group_values.device, dtype=group_values.dtype)[mask]
    if group_weights.numel() != group_values.shape[-1]:
        return None
    group_weight_sum = group_weights.sum()
    if not bool((group_weight_sum > 0.0).detach().cpu().item()):
        return None
    weighted = (group_values * group_weights).sum(dim=-1) / group_weight_sum.clamp_min(1e-6)
    return weighted.mean()


def _stats_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _stats_float_or(value: Any, default: float = 0.0) -> float:
    try:
        return _stats_float(value)
    except (RuntimeError, TypeError, ValueError):
        return float(default)


def _ensure_temporal_axis(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    return tensor


def _setdefault_stats(stats: Dict[str, float], defaults: Dict[str, float]) -> None:
    for key, value in defaults.items():
        stats.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class _DirectPosePair:
    direct_seq: torch.Tensor
    gt_direct: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupBaseRequest:
    geo_theta: Optional[torch.Tensor] = None
    dir_base: Optional[torch.Tensor] = None
    dir_leg_base: Optional[torch.Tensor] = None
    dir_nonleg_base: Optional[torch.Tensor] = None
    dir_arm_base: Optional[torch.Tensor] = None
    dir_else_base: Optional[torch.Tensor] = None
    arm_split_enable: Optional[bool] = None
    arm_else_balance_enable: Optional[bool] = None
    arm_weight: Optional[float] = None
    else_weight: Optional[float] = None
    eps: Optional[float] = None


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormRequest:
    dir_leg_base: torch.Tensor
    dir_nonleg_base: torch.Tensor
    dir_nonleg_effective_base: torch.Tensor
    direct_group_w_leg: Optional[float] = None
    direct_group_w_nonleg: Optional[float] = None
    direct_group_beta: Optional[float] = None
    direct_group_ratio_min: Optional[float] = None
    direct_group_ratio_max: Optional[float] = None
    direct_group_eps: Optional[float] = None
    update_ema_state: bool = True


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormResult:
    objective: torch.Tensor
    stats: Dict[str, float]
    ema_update: Dict[str, Any]

    def as_tuple(self) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        return self.objective, self.stats, self.ema_update


@dataclass(frozen=True, slots=True)
class _DirectPosePayloadRequest:
    direct: torch.Tensor
    gt_motion: torch.Tensor
    deg_per_rad: float


@dataclass(frozen=True, slots=True)
class _DirectPosePayloadResult:
    objective: torch.Tensor
    extra: Dict[str, Any]

    def as_tuple(self) -> tuple[torch.Tensor, Dict[str, Any]]:
        return self.objective, self.extra


class MotionJointLoss(nn.Module):
    _masked_group_mean = staticmethod(_masked_group_mean)
    _masked_group_weighted_mean = staticmethod(_masked_group_weighted_mean)
    _stats_float = staticmethod(_stats_float)
    _stats_float_or = staticmethod(_stats_float_or)
    _ensure_temporal_axis = staticmethod(_ensure_temporal_axis)
    _setdefault_stats = staticmethod(_setdefault_stats)

    # === future: train/loss/init.py ===
    # Constructor / config bootstrap.
    def __init__(
        self,
        w_attn_reg: float = 0.01,
        output_layout: Dict[str, Any] = None,
        fps: float = 60.0,
        rot6d_spec: Dict[str, Any] = None,
        w_rot_ortho: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        w_rot_local: float = 0.0,
        w_root_vel: float = 0.0,
        w_root_speed: float = 0.0,
        # Contact plan (cond-only anchor) supervision
        w_contact_plan: float = 0.0,
        # Optional: supervise meas head (pose-derived contacts)
        w_contact_meas: float = 0.0,
        # Optional: supervise direct pose head (cond + contacts_plan -> absolute pose)
        w_direct_pose: float = 0.0,
        direct_pose_loss_leg_split: bool = False,
        direct_pose_leg_bones: Optional[Sequence[str] | str] = None,
        direct_pose_arm_split_enable: bool = False,
        direct_pose_arm_bones: Optional[Sequence[str] | str] = None,
        direct_pose_loss_arm_else_balance_enable: bool = False,
        direct_pose_loss_arm_weight: float = 1.0,
        direct_pose_loss_else_weight: float = 1.0,
        direct_pose_loss_group_norm_enable: bool = False,
        direct_pose_loss_group_norm_w_leg: float = 1.0,
        direct_pose_loss_group_norm_w_nonleg: float = 1.0,
        direct_pose_loss_group_norm_ema_beta: float = 0.9,
        direct_pose_loss_group_norm_ratio_min: float = 0.2,
        direct_pose_loss_group_norm_ratio_max: float = 5.0,
        direct_pose_loss_group_norm_eps: float = 1e-6,
        # Optional: regularize omega_hat magnitude (prevents aggressive corrections)
        w_omega_l2: float = 0.0,
        # Event-Clock v3 regularization (only active when model returns corresponding tensors)
        event_clock_lambda_entropy_weight: float = 0.0,
        event_clock_lambda_prior_weight: float = 0.0,
        event_clock_delta_z_l2_weight: float = 0.0,
        adaptive_bone_weights: bool = False,
        **legacy_kwargs: Any,
    ):
        super().__init__()

        # Fail-fast retired-key boundary.
        legacy_loss_keys = (
            "ignore_motion_groups",
            "bone_prior_stds",
            "use_hierarchy_weights",
            "hierarchy_mode",
            "hierarchy_alpha",
            "max_weight_ratio",
            "weight_gamma",
        )
        if legacy_kwargs:
            legacy_hits = sorted(k for k in legacy_kwargs.keys() if k in legacy_loss_keys)
            if legacy_hits:
                joined = ", ".join(legacy_hits)
                raise ValueError(
                    f"MotionJointLoss deprecated loss keys are no longer supported: {joined}. "
                    "Please remove them from config/CLI and use unified bone weighting knobs only."
                )
            unknown = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"MotionJointLoss got unexpected keyword arguments: {unknown}")

        # Local scalar normalization.
        self.meta = dict(meta) if isinstance(meta, dict) else {}

        def _resolve_positive_scalar(field_name: str, raw_value: Any, *, default_value: float) -> float:
            candidate = default_value if raw_value is None else raw_value
            try:
                resolved = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a finite scalar in range (0, inf); "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            if (not _math.isfinite(resolved)) or resolved <= 0.0:
                raise ValueError(
                    f"{field_name} must be a finite scalar in range (0, inf); "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        def _resolve_bounded_scalar(
            field_name: str,
            raw_value: Any,
            *,
            default_value: float,
            min_value: float,
            max_value: float,
            min_inclusive: bool,
            max_inclusive: bool,
            expected_range: str,
        ) -> float:
            candidate = default_value if raw_value is None else raw_value
            try:
                resolved = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a finite scalar in range {expected_range}; "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            lower_ok = resolved >= float(min_value) if min_inclusive else resolved > float(min_value)
            upper_ok = resolved <= float(max_value) if max_inclusive else resolved < float(max_value)
            if (not _math.isfinite(resolved)) or (not lower_ok) or (not upper_ok):
                raise ValueError(
                    f"{field_name} must be a finite scalar in range {expected_range}; "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        # Core loss weights / direct-pose config.
        self.w_attn_reg = float(w_attn_reg)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_local = float(w_rot_local)
        self.w_root_vel = float(w_root_vel)
        self.w_root_speed = float(w_root_speed)
        self.w_contact_plan = float(w_contact_plan)
        self.w_contact_meas = float(w_contact_meas)
        self.w_direct_pose = float(w_direct_pose)
        self.direct_pose_loss_leg_split = bool(direct_pose_loss_leg_split)
        self.direct_pose_leg_bones = direct_pose_leg_bones
        self.direct_pose_arm_split_enable = bool(direct_pose_arm_split_enable)
        self.direct_pose_arm_bones = direct_pose_arm_bones
        self.direct_pose_loss_arm_else_balance_enable = bool(direct_pose_loss_arm_else_balance_enable)
        self.direct_pose_loss_arm_weight = _resolve_positive_scalar(
            "direct_pose_loss_arm_weight",
            direct_pose_loss_arm_weight,
            default_value=1.0,
        )
        self.direct_pose_loss_else_weight = _resolve_positive_scalar(
            "direct_pose_loss_else_weight",
            direct_pose_loss_else_weight,
            default_value=1.0,
        )
        self.direct_pose_loss_group_norm_enable = bool(direct_pose_loss_group_norm_enable)
        self.direct_pose_loss_group_norm_w_leg = float(direct_pose_loss_group_norm_w_leg or 1.0)
        self.direct_pose_loss_group_norm_w_nonleg = float(direct_pose_loss_group_norm_w_nonleg or 1.0)
        self.direct_pose_loss_group_norm_ema_beta = _resolve_bounded_scalar(
            "direct_pose_loss_group_norm_ema_beta",
            direct_pose_loss_group_norm_ema_beta,
            default_value=0.9,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=False,
            max_inclusive=True,
            expected_range="(0.0, 0.9999]",
        )
        self.direct_pose_loss_group_norm_ratio_min = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_ratio_min",
            direct_pose_loss_group_norm_ratio_min,
            default_value=0.2,
        )
        self.direct_pose_loss_group_norm_ratio_max = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_ratio_max",
            direct_pose_loss_group_norm_ratio_max,
            default_value=5.0,
        )
        if self.direct_pose_loss_group_norm_ratio_min > self.direct_pose_loss_group_norm_ratio_max:
            raise ValueError(
                "direct_pose_loss_group_norm_ratio_min/direct_pose_loss_group_norm_ratio_max must satisfy "
                "ratio_min <= ratio_max with both finite scalars in range (0, inf); "
                f"got direct_pose_loss_group_norm_ratio_min={self.direct_pose_loss_group_norm_ratio_min!r}, "
                f"direct_pose_loss_group_norm_ratio_max={self.direct_pose_loss_group_norm_ratio_max!r}."
            )
        self.direct_pose_loss_group_norm_eps = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_eps",
            direct_pose_loss_group_norm_eps,
            default_value=1e-6,
        )
        self._direct_pose_group_norm_ema: Dict[str, torch.Tensor] = {}
        self.w_omega_l2 = float(w_omega_l2)
        self.event_clock_lambda_entropy_weight = float(event_clock_lambda_entropy_weight or 0.0)
        self.event_clock_lambda_prior_weight = float(event_clock_lambda_prior_weight or 0.0)
        self.event_clock_delta_z_l2_weight = float(event_clock_delta_z_l2_weight or 0.0)
        self.use_adaptive_weights = bool(adaptive_bone_weights)

        # Layout / rot6d contract.
        self.fps = float(fps)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        self._rot6d_columns = self._resolve_rot6d_columns(self.rot6d_spec)
        layout = self.output_layout or {}
        slices_layout = layout.get('slices')
        inner = slices_layout if isinstance(slices_layout, dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.attn_lambda_local = getattr(self, 'attn_lambda_local', 0.02)
        self.attn_lambda_entropy = getattr(self, 'attn_lambda_entropy', 0.0)

        # Skeleton / cache bootstrap.
        self._warned_messages: set[str] = set()
        self._warned_bad_rot6d = False
        # Tail-risk regularization for per-bone rotation errors (CVaR / top-k style).
        # When enabled, adds an extra term on the worst bones (by mean GeoLocalDeg),
        # which reduces whack-a-mole without requiring explicit per-bone weight tables.
        self.rot_local_tail_weight = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        self.rot_local_tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        self.rot_local_tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all')
        self.rot_local_tail_select = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch')
        self.rot_local_tail_ema_beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        # 缓存几何骨骼权重（按 device/dtype）
        self._joint_weight_cache: dict[tuple, torch.Tensor] = {}
        # Tail-loss 辅助缓存（candidate pool 与选择打分）
        self._tail_candidate_cache: dict[tuple, torch.Tensor] = {}
        self._tail_score_cache: dict[tuple, torch.Tensor] = {}
        self.root_idx = 0
        self.bone_names: list[str] = []
        self._bone_name_to_idx: dict[str, int] = {}
        self.limb_monitor_names: list[str] = [
            'upperarm_l', 'lowerarm_l', 'hand_l',
            'upperarm_r', 'lowerarm_r', 'hand_r',
            'thigh_l', 'calf_l', 'foot_l',
            'thigh_r', 'calf_r', 'foot_r',
        ]
        self._limb_mask_cache: Optional[torch.Tensor] = None
        self._torso_mask_cache: Optional[torch.Tensor] = None
        self._limb_mask_joint_count: Optional[int] = None
        skeleton = self.meta.get('skeleton') if isinstance(self.meta, dict) else None
        self.parents: list[int] = []
        self._parents_tensor: Optional[torch.Tensor] = None
        self.bone_offsets: Optional[torch.Tensor] = None
        if isinstance(skeleton, dict):
            parents = skeleton.get('parents')
            if isinstance(parents, (list, tuple)):
                self.parents = [int(p) for p in parents]
                try:
                    self.root_idx = max(0, self.parents.index(-1))
                except ValueError:
                    self.root_idx = 0
            if 'ref_local_offsets_m' in skeleton:
                offsets = skeleton.get('ref_local_offsets_m')
                if offsets is not None:
                    try:
                        offsets_tensor = torch.as_tensor(offsets, dtype=torch.float32)
                    except (RuntimeError, TypeError, ValueError) as exc:
                        raise TypeError(
                            "skeleton.ref_local_offsets_m must be convertible to a float32 tensor with shape=(num_joints, 3); "
                            f"got value={offsets!r} (type={type(offsets).__name__})."
                        ) from exc
                    actual_shape = tuple(int(v) for v in offsets_tensor.shape)
                    if offsets_tensor.ndim != 2 or offsets_tensor.shape[-1] != 3:
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must have shape=(num_joints, 3); "
                            f"got actual_shape={actual_shape!r} (type={type(offsets).__name__})."
                        )
                    if self.parents and offsets_tensor.shape[0] != len(self.parents):
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must have shape=(num_joints, 3) with num_joints matching "
                            f"len(parents)={len(self.parents)}; got actual_shape={actual_shape!r}."
                        )
                    if not bool(torch.isfinite(offsets_tensor).all().detach().cpu().item()):
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must contain only finite values; "
                            f"got actual_shape={actual_shape!r}."
                        )
                    self.bone_offsets = offsets_tensor

        # Orchestration tracker defaults.
        self._loss_group_totals: Dict[str, float] = {}
        self._loss_group_alias = {
            'attn': 'aux',
            'rot_ortho': 'core',
            'rot_local': 'core',
            'root_vel': 'core',
            'root_speed': 'core',
            'direct_pose': 'core',
        }

        # skeleton parents may be set later via set_skeleton; avoid early fallback here

    def _warn_once(
        self,
        key: str,
        message: str,
        *,
        category: type[Warning] = RuntimeWarning,
    ) -> None:
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        warnings.warn(message, category=category, stacklevel=2)

    @staticmethod
    def _resolve_rot6d_columns(spec: Optional[Dict[str, Any]]) -> tuple[str, str]:
        if isinstance(spec, dict):
            cols = spec.get('columns')
            if isinstance(cols, (list, tuple)) and len(cols) >= 2:
                a = str(cols[0]).strip().upper()
                b = str(cols[1]).strip().upper()
                if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                    return a, b
        return ("X", "Z")

    # === future: train/loss/skeleton_weights.py ===
    # Skeleton state / cache invalidation.
    def set_bone_names(self, names: Optional[Sequence[str]]) -> None:
        self.bone_names = [str(n) for n in (names or [])]
        self._bone_name_to_idx = {name: idx for idx, name in enumerate(self.bone_names)}
        self._limb_mask_cache = None
        self._torso_mask_cache = None
        self._limb_mask_joint_count = None
        self._tail_candidate_cache = {}
        self._tail_score_cache = {}
        # reset fk caches when bone count changes
        self._parents_tensor = None

    def set_skeleton(self, parents: Optional[Sequence[int]], offsets: Optional[Sequence[Sequence[float]]]) -> None:
        next_parents = self.parents
        if parents is not None:
            next_parents = [int(p) for p in parents]
        next_offsets = self.bone_offsets
        if offsets is not None:
            try:
                offsets_tensor = torch.as_tensor(offsets, dtype=torch.float32)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise TypeError(
                    "offsets must be convertible to a float32 tensor with shape=(num_joints, 3); "
                    f"got value={offsets!r} (type={type(offsets).__name__})."
                ) from exc
            actual_shape = tuple(int(v) for v in offsets_tensor.shape)
            if offsets_tensor.ndim != 2 or offsets_tensor.shape[-1] != 3:
                raise ValueError(
                    "offsets must have shape=(num_joints, 3); "
                    f"got actual_shape={actual_shape!r} (type={type(offsets).__name__})."
                )
            if next_parents and offsets_tensor.shape[0] != len(next_parents):
                raise ValueError(
                    "offsets must have shape=(num_joints, 3) with num_joints matching "
                    f"len(parents)={len(next_parents)}; got actual_shape={actual_shape!r}."
                )
            if not bool(torch.isfinite(offsets_tensor).all().detach().cpu().item()):
                raise ValueError(
                    "offsets must contain only finite values; "
                    f"got actual_shape={actual_shape!r}."
                )
            next_offsets = offsets_tensor
        if parents is not None:
            self.parents = next_parents
            self._parents_tensor = None
        if offsets is not None:
            self.bone_offsets = next_offsets
        self._tail_candidate_cache = {}

    def _invalidate_weight_cache(self) -> None:
        self._joint_weight_cache = {}

    # Mask resolution / skeleton-derived stats.
    def _resolve_named_joint_indices(
        self,
        *,
        joint_count: int,
        spec: Optional[Sequence[str] | str],
        default_items: Sequence[str],
    ) -> list[int]:
        if joint_count <= 0 or not self.bone_names or joint_count > len(self.bone_names):
            return []
        indices, _ = _resolve_joint_spec_indices(
            spec,
            default_items=default_items,
            bone_names=self.bone_names[:joint_count],
            joint_count=joint_count,
        )
        return indices

    def _resolve_limb_masks(self, joint_count: int, device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        import torch
        if joint_count <= 0:
            return None
        names = self.bone_names
        if not names or joint_count > len(names):
            return None
        monitor = getattr(self, 'limb_monitor_names', None) or []
        if not monitor:
            return None
        idx_map = {name: idx for idx, name in enumerate(names[:joint_count])}
        limb_indices = [idx_map[name] for name in monitor if name in idx_map]
        if not limb_indices:
            return None
        if self._limb_mask_joint_count != joint_count or self._limb_mask_cache is None:
            mask = torch.zeros(joint_count, dtype=torch.bool)
            mask[limb_indices] = True
            self._limb_mask_cache = mask
            self._torso_mask_cache = (~mask).clone()
            self._limb_mask_joint_count = joint_count
        limb_mask = self._limb_mask_cache.to(device=device)
        torso_mask = self._torso_mask_cache.to(device=device)
        if not torso_mask.any():
            torso_mask = (~limb_mask).clone()
            if torso_mask.numel() == 0 or not torso_mask.any():
                return None
        return limb_mask, torso_mask

    def _resolve_direct_group_masks(self, joint_count: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        import torch

        if joint_count <= 0 or not self.bone_names or joint_count > len(self.bone_names):
            return None
        leg_idx = self._resolve_named_joint_indices(
            joint_count=joint_count,
            spec=self.direct_pose_leg_bones,
            default_items=DEFAULT_DIRECT_POSE_LEG_BONES,
        )
        arm_idx = self._resolve_named_joint_indices(
            joint_count=joint_count,
            spec=self.direct_pose_arm_bones,
            default_items=STAGE6_3WAY_ARMCHAIN_BONES,
        )
        leg_mask = torch.zeros(joint_count, dtype=torch.bool, device=device)
        arm_mask = torch.zeros(joint_count, dtype=torch.bool, device=device)
        if leg_idx:
            leg_mask[torch.as_tensor(leg_idx, dtype=torch.long, device=device)] = True
        if arm_idx:
            arm_mask[torch.as_tensor(arm_idx, dtype=torch.long, device=device)] = True
        nonleg_mask = ~leg_mask
        arm_mask = arm_mask & nonleg_mask
        else_mask = nonleg_mask & (~arm_mask)
        all_ex_root = torch.ones(joint_count, dtype=torch.bool, device=device)
        root_idx = int(getattr(self, 'root_idx', 0) or 0)
        if 0 <= root_idx < joint_count:
            all_ex_root[root_idx] = False
            leg_mask[root_idx] = False
            nonleg_mask[root_idx] = False
            arm_mask[root_idx] = False
            else_mask[root_idx] = False
        return {
            'all_ex_root': all_ex_root,
            'leg': leg_mask,
            'nonleg': nonleg_mask,
            'arm': arm_mask,
            'else': else_mask,
            'trunk': else_mask,
        }

    def _collect_limb_local_stats(self, geo_tensor: torch.Tensor) -> Dict[str, float]:
        import torch, math
        if geo_tensor is None or geo_tensor.numel() == 0:
            return {}
        J = geo_tensor.shape[-1]
        masks = self._resolve_limb_masks(J, geo_tensor.device)
        if not masks:
            return {}
        limb_mask, torso_mask = masks
        flat = geo_tensor.reshape(-1, J)
        joint_mean = torch.nanmean(flat, dim=0)
        stats: Dict[str, float] = {}
        deg = 180.0 / math.pi
        limb_val = torso_val = None
        if limb_mask.any():
            limb_val = joint_mean[limb_mask].mean()
            stats['rot_local_limb_deg'] = float((limb_val * deg).detach().cpu())
            stats['rot_local_limb_count'] = int(limb_mask.sum().item())
        if torso_mask.any():
            torso_val = joint_mean[torso_mask].mean()
            stats['rot_local_torso_deg'] = float((torso_val * deg).detach().cpu())
            stats['rot_local_torso_count'] = int(torso_mask.sum().item())
        if limb_val is not None and torso_val is not None:
            ratio = limb_val / torso_val.clamp_min(1e-6)
            stats['rot_local_limb_over_torso'] = float(ratio.detach().cpu())
        return stats

    # Weight computation.
    def _joint_weight_vector(self, device, dtype, joint_count: int) -> torch.Tensor:
        """
        influence = self_scale * |offset| + (sum lever_arm_to_descendants) ** power
        clamp to min_w, optional visual importance; normalize mean=1.
        """
        key = (str(device), str(dtype), int(joint_count))
        cache = getattr(self, '_joint_weight_cache', None)
        if cache is None:
            cache = {}
            self._joint_weight_cache = cache
        if key in cache:
            return cache[key]

        # 基于几何的统一权重
        weights_cpu = self._compute_unified_weights_cpu(joint_count)

        weights = weights_cpu.to(device=device, dtype=dtype)
        cache[key] = weights
        if not hasattr(self, '_weight_vector_logged'):
            self._weight_vector_logged = True
            self._warn_once(
                "unified_weight_vector",
                f"[Loss][UnifiedWeights] range=[{weights.min():.3f}, {weights.max():.3f}] "
                f"mean={weights.mean():.3f} std={weights.std():.3f} "
                f"power={getattr(self, 'unified_downstream_power', 0.6)} "
                f"self_scale={getattr(self, 'unified_self_scale', 1.5)} "
                f"min_w={getattr(self, 'unified_min_weight', 0.05)} "
                f"visual=False",
                category=UserWarning,
            )
        return weights

    def _compute_unified_weights_cpu(self, joint_count: int) -> torch.Tensor:
        power = float(getattr(self, 'unified_downstream_power', 0.6))
        self_scale = float(getattr(self, 'unified_self_scale', 1.5))
        min_w = float(getattr(self, 'unified_min_weight', 0.05))

        if not self.parents or self.bone_offsets is None or len(self.parents) < joint_count:
            return torch.ones(joint_count, dtype=torch.float32)

        J = joint_count
        parents = self.parents[:J]
        offsets = self.bone_offsets[:J].detach().cpu().float()

        global_pos = torch.zeros(J, 3, dtype=torch.float32)
        for j in range(J):
            p = parents[j]
            if p < 0:
                global_pos[j] = offsets[j]
            else:
                global_pos[j] = global_pos[p] + offsets[j]

        children = [[] for _ in range(J)]
        for j, p in enumerate(parents):
            if p >= 0 and p < J:
                children[p].append(j)

        def _descendants(idx: int):
            qs = [idx]
            seen = set()
            out = []
            while qs:
                cur = qs.pop(0)
                for c in children[cur]:
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
                        qs.append(c)
            return out

        weights = torch.zeros(J, dtype=torch.float32)
        for i in range(J):
            self_len = torch.norm(offsets[i])
            bone_i = global_pos[i]
            down = 0.0
            for j in _descendants(i):
                end = global_pos[j] + offsets[j]
                lever = torch.norm(end - bone_i)
                down += float(lever)
            down_scaled = (down ** power) if down > 0 else 0.0
            weights[i] = self_scale * self_len + down_scaled

        weights = weights / weights.mean().clamp_min(1e-6)
        weights = torch.clamp(weights, min=min_w)
        weights = weights / weights.mean().clamp_min(1e-6)

        return weights

    # Tail-risk selection helpers.
    def _rot_local_tail_scores(self, per_bone: torch.Tensor) -> torch.Tensor:
        """
        Selection scores for tail top-k:
        - batch: current batch mean (default)
        - ema:   exponential moving average across batches for smoother selection
        """
        mode = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch').lower()
        if mode != 'ema':
            return per_bone
        beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        beta = max(0.0, min(0.999, beta))
        key = (str(per_bone.device), int(per_bone.numel()))
        cache = getattr(self, '_tail_score_cache', None)
        if cache is None:
            cache = {}
            self._tail_score_cache = cache
        prev = cache.get(key)
        if prev is None or (not torch.is_tensor(prev)) or prev.shape != per_bone.shape or prev.device != per_bone.device:
            score = per_bone.detach().clone()
        else:
            score = prev
            score.mul_(beta).add_(per_bone.detach(), alpha=(1.0 - beta))
        cache[key] = score
        return score

    def _rot_local_tail_candidates(self, scope: str, joint_count: int, device: torch.device, *, k: int) -> Optional[torch.Tensor]:
        """
        Candidate joint indices for tail selection. Uses:
        - explicit name list (limb_monitor_names / pelvis) when bone_names are known
        - skeleton leaves (end-effectors) as a dynamic fallback/augmentation
        """
        scope_norm = str(scope or 'all').lower()
        if scope_norm in ('all', '*'):
            return None
        if scope_norm not in ('limbs', 'limb', 'keybones', 'key_bones'):
            return None
        cache = getattr(self, '_tail_candidate_cache', None)
        if cache is None:
            cache = {}
            self._tail_candidate_cache = cache
        key = (scope_norm, int(joint_count), str(device))
        cached = cache.get(key)
        if cached is not None and torch.is_tensor(cached) and cached.device == device and cached.numel() > 0:
            return cached

        J = int(joint_count)
        idxs: list[int] = []

        # 1) Name-driven candidates (keeps old behavior when names match)
        name_to_idx = getattr(self, '_bone_name_to_idx', None)
        if isinstance(name_to_idx, dict) and name_to_idx:
            monitor = list(getattr(self, 'limb_monitor_names', None) or [])
            if scope_norm in ('keybones', 'key_bones'):
                monitor = ['pelvis'] + monitor
            for nm in monitor:
                idx = name_to_idx.get(str(nm))
                if isinstance(idx, int) and 0 <= idx < J:
                    idxs.append(int(idx))

        # 2) Skeleton-driven candidates: leaf joints (end-effectors) + root for keybones
        parents = getattr(self, 'parents', None)
        if isinstance(parents, list) and len(parents) >= J:
            child_counts = [0] * J
            for j, p in enumerate(parents[:J]):
                if isinstance(p, int) and 0 <= p < J:
                    child_counts[p] += 1
            leaves = [j for j in range(J) if child_counts[j] == 0]

            # Avoid selecting too many tiny leaves (e.g. fingers) as candidates.
            max_leaf = int(max(16, 4 * max(1, int(k))))
            if len(leaves) > max_leaf:
                w = self._compute_unified_weights_cpu(J)
                vals = w.index_select(0, torch.as_tensor(leaves, dtype=torch.long))
                keep = min(max_leaf, int(vals.numel()))
                if keep > 0:
                    _, sel = torch.topk(vals, k=keep, largest=True, sorted=False)
                    leaves = [leaves[int(i)] for i in sel.tolist()]
                else:
                    leaves = []

            if scope_norm in ('keybones', 'key_bones'):
                root_idx = int(getattr(self, 'root_idx', 0))
                if 0 <= root_idx < J:
                    idxs.append(root_idx)
            idxs.extend(leaves)

        # De-dup preserving order
        if not idxs:
            return None
        seen = set()
        idxs = [i for i in idxs if 0 <= i < J and not (i in seen or seen.add(i))]
        if not idxs:
            return None
        out = torch.as_tensor(idxs, device=device, dtype=torch.long)
        cache[key] = out
        return out

    # FK-relative rotation views.
    def _parent_relative_matrices(self, R: torch.Tensor) -> torch.Tensor:
        parents = getattr(self, 'parents', None)
        if not parents:
            return R
        J = int(R.shape[-3])
        if len(parents) < J or J <= 0:
            return R
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != R.device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(parents[:J], device=R.device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        return parent_relative_matrices(R, parents_tensor)

    def _root_relative(self, R: torch.Tensor) -> torch.Tensor:
        root_idx = int(getattr(self, 'root_idx', 0))
        return root_relative_matrices(R, root_idx)

    def _forward_base_inner(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor, attn_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            pred_motion: [B,T,D] or [T,D] or [B,D]
            gt_motion:   同形状
            attn_weights: None 或 [B,H,T,T]/[L,B,H,T,T] 或 list/tuple/dict 的任意嵌套
        返回:
            loss 标量, 分项 dict (float)
        """
        pm, gm = (pred_motion, gt_motion)
        assert pm.shape == gm.shape, f'pred/gt shape mismatch: {pm.shape} vs {gm.shape}'

        if attn_weights is not None:
            l_attn = self.compute_attention_regularization(attn_weights, geomask=None)
        else:
            l_attn = gm.new_zeros(())

        loss = self.w_attn_reg * l_attn
        self._accumulate_loss_contrib('attn', l_attn, self.w_attn_reg, group='aux')

        stats: Dict[str, float] = {
            'attn': float(l_attn.detach().cpu()),
            'rot_ortho': 0.0,
            'rot_ortho_raw': 0.0,
        }
        return loss, stats

    def _slice_if_exists(self, name: str, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        从 self.group_slices 中获取预先解析好的 slice，并应用于张量。
        """
        sl = self.group_slices.get(name)

        # 因为 self.group_slices 只包含 slice 对象，所以只需做一次类型检查即可。
        if isinstance(sl, slice):
            return X[..., sl]

        return None

    def compute_attention_regularization(self, attn_weights, geomask=None):
        """
        返回一个标量 loss：
        - 支持 Tensor: [B,H,T,T] 或 [L,B,H,T,T]
        - 支持 list/tuple/dict: 递归展开后逐个累加
        - geomask: None 或可广播到 [..., T, T] 的掩码（1=允许区域，0=不鼓励区域）
        """
        if attn_weights is None:
            if geomask is not None and torch.is_tensor(geomask):
                return torch.zeros((), device=geomask.device, dtype=geomask.dtype)
            return torch.tensor(0.0)

        def _flatten_items(x):
            if torch.is_tensor(x):
                return [x]
            if isinstance(x, (list, tuple)):
                items = []
                for y in x:
                    items.extend(_flatten_items(y))
                return items
            if isinstance(x, dict):
                items = []
                for y in x.values():
                    items.extend(_flatten_items(y))
                return items
            return []
        items = _flatten_items(attn_weights)
        if len(items) == 0:
            return torch.tensor(0.0, device=geomask.device if geomask is not None and torch.is_tensor(geomask) else None)
        loss_total = None
        count = 0
        for A in items:
            if A.dim() == 5:
                L, B, H, T, _ = A.shape
                A = A.reshape(L * B * H, T, T)
            elif A.dim() == 4:
                B, H, T, _ = A.shape
                A = A.reshape(B * H, T, T)
            elif A.dim() == 3:
                T = A.shape[-1]
            else:
                continue
            A = A.float()
            A = A / A.sum(-1, keepdim=True).clamp_min(1e-06)
            device = A.device
            if geomask is not None:
                if torch.is_tensor(geomask):
                    gm = geomask
                    gm_dim = int(gm.dim())
                    if gm_dim not in (2, 3, 4):
                        raise ValueError(
                            "geomask for attention regularization must be a tensor with rank 2, 3, or 4 "
                            "broadcastable to attention weights shaped (..., T, T); "
                            f"got actual_shape={tuple(int(v) for v in gm.shape)!r}, actual_ndim={gm_dim}, T={int(T)}."
                        )
                    try:
                        M = 1.0 - gm
                        loss_local = (A * M).mean()
                    except RuntimeError as direct_exc:
                        if gm_dim == 2:
                            try:
                                gm = gm.view(1, T, T)
                            except RuntimeError as reshape_exc:
                                raise RuntimeError(
                                    "geomask rank-2 fallback reshape failed in attention regularization: "
                                    "expected geomask shape (T, T) before view(1, T, T); "
                                    f"got geomask_shape={tuple(int(v) for v in geomask.shape)!r}, "
                                    f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                                ) from reshape_exc
                        elif gm_dim == 4:
                            gm = gm.mean(0)
                        else:
                            raise RuntimeError(
                                "geomask broadcast failed in attention regularization: "
                                "expected rank-3 geomask broadcastable to attention weights shaped (N, T, T); "
                                f"got geomask_shape={tuple(int(v) for v in gm.shape)!r}, "
                                f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                            ) from direct_exc
                        try:
                            M = 1.0 - gm
                            loss_local = (A * M).mean()
                        except RuntimeError as fallback_exc:
                            raise RuntimeError(
                                "geomask fallback broadcast failed in attention regularization: "
                                "expected rank-2 shape (T, T), rank-3 shape broadcastable to (N, T, T), "
                                "or rank-4 reducible via mean(0); "
                                f"got original_geomask_shape={tuple(int(v) for v in geomask.shape)!r}, "
                                f"fallback_geomask_shape={tuple(int(v) for v in gm.shape)!r}, "
                                f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                            ) from fallback_exc
                else:
                    idx = torch.arange(T, device=device)
                    Dmat = (idx[None, :] - idx[:, None]).abs().float()
                    Dmat = Dmat / Dmat.max().clamp_min(1.0)
                    loss_local = (A * Dmat).mean()
            else:
                idx = torch.arange(T, device=device)
                Dmat = (idx[None, :] - idx[:, None]).abs().float()
                Dmat = Dmat / Dmat.max().clamp_min(1.0)
                loss_local = (A * Dmat).mean()
            Aeps = A.clamp_min(1e-06)
            entropy = -(Aeps * Aeps.log()).sum(-1).mean()
            loss_attn = self.attn_lambda_local * loss_local + self.attn_lambda_entropy * -entropy
            loss_total = loss_attn if loss_total is None else loss_total + loss_attn
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=items[0].device)
        return loss_total / count

    # === future: train/loss_rot6d.py ===
    # Rot6D slice / denorm / matrix helpers.
    def _maybe_get_rot6d(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        若存在 "BoneRotations6D" 切片，则返回该切片；否则 None。
        """
        rot = self._slice_if_exists('BoneRotations6D', X)
        return rot

    def _denorm_rot6d_flat(self, rot6d_flat: torch.Tensor) -> tuple[torch.Tensor, bool]:
        sl = self.group_slices.get('BoneRotations6D', None)
        if (
            not isinstance(sl, slice)
            or getattr(self, 'mu_y', None) is None
            or getattr(self, 'std_y', None) is None
        ):
            return rot6d_flat, False
        start = int(sl.start or 0)
        stop = int(sl.stop or start)
        width = max(0, stop - start)
        if width != int(rot6d_flat.shape[-1]):
            return rot6d_flat, False
        mu = torch.as_tensor(self.mu_y, device=rot6d_flat.device, dtype=rot6d_flat.dtype)[..., start:stop]
        std = torch.as_tensor(self.std_y, device=rot6d_flat.device, dtype=rot6d_flat.dtype)[..., start:stop]
        std = std.clamp(min=1e-6)
        while mu.dim() < rot6d_flat.dim():
            mu = mu.unsqueeze(0)
            std = std.unsqueeze(0)
        return rot6d_flat * std + mu, True

    def _extract_rot6d_flat(
        self,
        X: torch.Tensor,
        *,
        denorm: bool = True,
        reproject: bool = False,
        sanitize: bool = False,
    ) -> Optional[torch.Tensor]:
        rot6d = self._maybe_get_rot6d(X)
        if rot6d is None:
            return None
        if int(rot6d.shape[-1]) % 6 != 0:
            return None
        if denorm:
            rot6d, hit = self._denorm_rot6d_flat(rot6d)
            if hit and not hasattr(self, "_train_denorm_hit"):
                self._train_denorm_hit = True
                self._warn_once(
                    "train_denorm_rot6d_flat",
                    "[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.",
                    category=UserWarning,
                )
        if sanitize:
            rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=1.0, neginf=-1.0)
        if reproject:
            rot6d = reproject_rot6d(rot6d)
        return rot6d

    def _extract_rot6d_mats(
        self,
        X: torch.Tensor,
        *,
        denorm: bool = True,
        reproject: bool = True,
        sanitize: bool = True,
    ) -> Optional[torch.Tensor]:
        rot6d = self._extract_rot6d_flat(
            X,
            denorm=denorm,
            reproject=reproject,
            sanitize=sanitize,
        )
        if rot6d is None:
            return None
        J = int(rot6d.shape[-1]) // 6
        rot6d = rot6d.view(*rot6d.shape[:-1], J, 6)
        return rot6d_to_matrix(rot6d)

    def _rot6d_matrices(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        return self._extract_rot6d_mats(X, denorm=True, reproject=True, sanitize=True)

    # Rot6D objective helpers.
    def compute_rot6d_ortho_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Ortho penalty on **raw 6D** (pre-GS):
        encourage unit-length columns and mutual orthogonality.
        This must NOT use rot6d_to_matrix (which orthonormalizes and would yield ~0 loss).
        """
        Z = lambda v: pred.new_tensor(float(v))
        pr = self._maybe_get_rot6d(pred)  # (..., D) or None
        if pr is None:
            return Z(0.0)
        D = pr.shape[-1]
        if D % 6 != 0:
            if not getattr(self, '_warned_bad_rot6d_ortho', False):
                self._warned_bad_rot6d_ortho = True
                self._warn_once(
                    "bad_rot6d_ortho_dim",
                    f"[Loss][WARN] BoneRotations6D slice dim={D} not multiple of 6. Skip rot6d_ortho.",
                )
            return Z(0.0)
        J = D // 6
        a6 = pr.view(*pr.shape[:-1], J, 6)  # (..., J, 6) raw 6D (no GS)
        v1 = a6[..., 0:3]
        v2 = a6[..., 3:6]
        len_p = (v1.norm(dim=-1) - 1.0).pow(2) + (v2.norm(dim=-1) - 1.0).pow(2)
        ortho_p = (v1.mul(v2).sum(dim=-1)).pow(2)
        return (len_p + ortho_p).mean()

    def compute_rot6d_geo_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        return_per_joint: bool = False,
    ):
        Z = lambda v: gt.new_tensor(float(v))
        sl = self.group_slices.get('BoneRotations6D', None)

        pr_raw = self._maybe_get_rot6d(pred)
        gr_raw = self._maybe_get_rot6d(gt)
        if pr_raw is None or gr_raw is None:
            return Z(0.0)
        D = int(pr_raw.shape[-1])
        if D % 6 != 0:
            if not self._warned_bad_rot6d:
                self._warned_bad_rot6d = True
                self._warn_once(
                    "bad_rot6d_geo_dim",
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). "
                    f"slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.",
                )
            return Z(0.0)
        if int(gr_raw.shape[-1]) != D:
            return Z(0.0)
        J = D // 6

        pr = self._extract_rot6d_flat(pred, denorm=True, reproject=True, sanitize=False)
        gr = self._extract_rot6d_flat(gt, denorm=True, reproject=True, sanitize=False)
        if pr is None or gr is None:
            return Z(0.0)
        pr = pr.view(*pr.shape[:-1], J, 6)  # (…, J, 6)
        gr = gr.view(*gr.shape[:-1], J, 6)  # (…, J, 6)

        Rp = rot6d_to_matrix(pr)
        Rg = rot6d_to_matrix(gr)
        theta = geodesic_R(Rp, Rg)

        weights = self._joint_weight_vector(theta.device, theta.dtype, J)
        view_shape = (1,) * (theta.dim() - 1) + (J,)
        w = weights.view(*view_shape)
        theta_weighted = theta * w
        loss_val = theta_weighted.mean()
        if return_per_joint:
            return loss_val, theta, weights
        return loss_val

    def compute_rot6d_log_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._rot6d_matrices(pred)
        Rg = self._rot6d_matrices(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 5:
            return Z(0.0)
        if int(Rp.shape[-4]) < 2:
            return Z(0.0)
        log_p = angvel_vec_from_R_seq(Rp, fps=1.0)
        log_g = angvel_vec_from_R_seq(Rg, fps=1.0)
        return torch.nn.functional.smooth_l1_loss(log_p, log_g)


    # === future: train/loss/components.py ===
    # Motion applicators.
    def _apply_rot_ortho_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
    ) -> torch.Tensor:
        if self.w_rot_ortho > 0 and not delta_fallback:
            target_for_ortho = delta_motion if delta_motion is not None else pred_motion
            l_ortho = self.compute_rot6d_ortho_loss(target_for_ortho)
            return self._submit_component_loss(
                total_loss,
                stats=stats,
                name='rot_ortho',
                tensor=l_ortho,
                weight=self.w_rot_ortho,
                group='core',
                raw_key='rot_ortho',
                weighted_key='rot_ortho_weighted',
                extra={'rot_ortho_raw': l_ortho},
            )

        _setdefault_stats(stats, {
            'rot_ortho': 0.0,
            'rot_ortho_weighted': 0.0,
            'rot_ortho_raw': 0.0,
        })
        if delta_fallback and self.w_rot_ortho > 0 and delta_motion is not None:
            l_ortho = self.compute_rot6d_ortho_loss(delta_motion)
            stats['rot_ortho_fallback'] = _stats_float_or(l_ortho, default=float('nan'))
        return total_loss

    def _apply_rot_local_tail_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        geo_local: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        tail_w = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all').lower()
        joint_count = int(geo_local.shape[-1])
        if tail_w <= 0.0 or tail_k <= 0 or joint_count <= 0:
            return total_loss

        k = min(max(1, tail_k), joint_count)
        cand_idx = self._rot_local_tail_candidates(tail_scope, joint_count, geo_local.device, k=k)
        per_bone = torch.nanmean(geo_local.detach(), dim=tuple(range(geo_local.dim() - 1)))
        scores = self._rot_local_tail_scores(per_bone)
        select_mode = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch').lower()
        score_count = int(scores.numel())
        if score_count <= 0:
            return total_loss
        if cand_idx is not None and cand_idx.numel() > 0:
            valid_idx = cand_idx[(cand_idx >= 0) & (cand_idx < score_count)]
            if valid_idx.numel() <= 0:
                return total_loss
            k_eff = min(k, int(valid_idx.numel()))
            vals = scores.index_select(0, valid_idx)
            _, sel = torch.topk(vals, k=k_eff, largest=True, sorted=False)
            idx = valid_idx.index_select(0, sel)
        else:
            k_eff = min(k, score_count)
            _, idx = torch.topk(scores, k=k_eff, largest=True, sorted=False)
        tail_loss = torch.nanmean(geo_local.index_select(-1, idx))
        total_loss = self._submit_component_loss(
            total_loss,
            stats=stats,
            name='rot_local_tail',
            tensor=tail_loss,
            weight=tail_w,
            group='core',
            extra={
                'rot_local_tail_deg': tail_loss * deg_per_rad,
                'rot_local_tail_k': float(k_eff),
                'rot_local_tail_scope': float({'all': 0.0, 'limbs': 1.0, 'keybones': 2.0}.get(tail_scope, 0.0)),
                'rot_local_tail_select': float({'batch': 0.0, 'ema': 1.0}.get(select_mode, 0.0)),
            },
        )
        return total_loss

    def _apply_rot_local_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        if self.w_rot_local <= 0.0:
            stats.setdefault('rot_local_deg', 0.0)
            return total_loss

        Rp_world = self._rot6d_matrices(pred_motion)
        Rg_world = self._rot6d_matrices(gt_motion)
        if Rp_world is None or Rg_world is None:
            return total_loss

        Rp_root = self._root_relative(Rp_world)
        Rg_root = self._root_relative(Rg_world)
        Rp_local = self._parent_relative_matrices(Rp_root)
        Rg_local = self._parent_relative_matrices(Rg_root)
        geo_local = geodesic_R(Rp_local, Rg_local)
        weights = self._joint_weight_vector(Rp_local.device, Rp_local.dtype, Rp_local.shape[-3])
        local_loss = (geo_local * weights.view(1, 1, -1)).mean()
        total_loss = self._submit_component_loss(
            total_loss,
            stats=stats,
            name='rot_local',
            tensor=local_loss,
            weight=self.w_rot_local,
            group='core',
            extra={'rot_local_deg': local_loss * deg_per_rad},
        )
        return self._apply_rot_local_tail_component(total_loss, stats, geo_local, deg_per_rad)

    def _apply_root_velocity_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> torch.Tensor:
        rv_slice = self.group_slices.get('RootVelocity')
        if rv_slice is None or (self.w_root_vel <= 0.0 and self.w_root_speed <= 0.0):
            _setdefault_stats(stats, {
                'root_vel_mse': 0.0,
                'root_speed_mae': 0.0,
            })
            return total_loss

        pred_vel = pred_motion[..., rv_slice]
        gt_vel = gt_motion[..., rv_slice]
        if self.w_root_vel > 0.0:
            vel_mse = F.mse_loss(pred_vel, gt_vel)
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='root_vel',
                tensor=vel_mse,
                weight=self.w_root_vel,
                group='core',
                raw_key='root_vel_mse',
            )
        else:
            stats.setdefault('root_vel_mse', 0.0)

        if self.w_root_speed > 0.0:
            pred_speed = torch.norm(pred_vel, dim=-1)
            gt_speed = torch.norm(gt_vel, dim=-1)
            speed_mae = F.l1_loss(pred_speed, gt_speed)
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='root_speed',
                tensor=speed_mae,
                weight=self.w_root_speed,
                group='core',
                raw_key='root_speed_mae',
            )
        else:
            stats.setdefault('root_speed_mae', 0.0)
        return total_loss

    # Motion component dispatch.
    def _apply_motion_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
        deg_per_rad: float,
    ) -> torch.Tensor:
        total_loss = self._apply_rot_ortho_component(
            total_loss,
            stats,
            pred_motion,
            delta_motion,
            delta_fallback,
        )
        total_loss = self._apply_rot_local_component(
            total_loss,
            stats,
            pred_motion,
            gt_motion,
            deg_per_rad,
        )
        return self._apply_root_velocity_components(total_loss, stats, pred_motion, gt_motion)

    # === future: train/loss/direct_pose.py ===
    # Stats contract / pair normalization.
    def _direct_pose_default_stats(self) -> Dict[str, float]:
        return {
            'direct_pose_geo': 0.0,
            'direct_pose_geo_deg': 0.0,
            'direct_pose_objective': 0.0,
            'direct_pose_weighted': 0.0,
            'direct_pose_split_active': 0.0,
            'direct_pose_arm_split_active': 0.0,
            'dir_base': float('nan'),
            'dir_leg_base': float('nan'),
            'dir_nonleg_base': float('nan'),
            'dir_nonleg_effective_base': float('nan'),
            'dir_arm_base': float('nan'),
            'dir_else_base': float('nan'),
            'leg_over_nonleg': float('nan'),
            'leg_over_nonleg_effective': float('nan'),
            'arm_over_else': float('nan'),
            'direct_pose_arm_else_balance_active': 0.0,
            'direct_pose_loss_arm_weight': float(getattr(self, 'direct_pose_loss_arm_weight', 1.0) or 1.0),
            'direct_pose_loss_else_weight': float(getattr(self, 'direct_pose_loss_else_weight', 1.0) or 1.0),
            'dir_group_norm_used': 0.0,
            'dir_group_norm_leg_raw': float('nan'),
            'dir_group_norm_nonleg_raw': float('nan'),
            'dir_group_norm_leg_clamped': float('nan'),
            'dir_group_norm_nonleg_clamped': float('nan'),
            'dir_group_norm_leg': float('nan'),
            'dir_group_norm_nonleg': float('nan'),
            'dir_group_norm_leg_ema': float('nan'),
            'dir_group_norm_nonleg_ema': float('nan'),
            'dir_group_norm_leg_hit_min': 0.0,
            'dir_group_norm_leg_hit_max': 0.0,
            'dir_group_norm_nonleg_hit_min': 0.0,
            'dir_group_norm_nonleg_hit_max': 0.0,
            'dir_group_norm_leg_hit_any': 0.0,
            'dir_group_norm_nonleg_hit_any': 0.0,
        }

    def _direct_pose_default_stat_keys(self) -> tuple[str, ...]:
        return _DIRECT_POSE_DEFAULT_STAT_KEYS

    def _direct_pose_component_stat_keys(self) -> tuple[str, ...]:
        return _DIRECT_POSE_COMPONENT_STAT_KEYS

    def _direct_pose_extra_defaults(self) -> Dict[str, float]:
        defaults = self._direct_pose_default_stats()
        defaults.pop('direct_pose_objective', None)
        defaults.pop('direct_pose_weighted', None)
        return defaults

    def _prepare_direct_pose_pair(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> Optional[_DirectPosePair]:
        if direct.dim() == 2 and gt_motion.dim() == 3:
            direct = direct.unsqueeze(1)
        gt_direct = gt_motion
        if gt_direct.dim() == 2 and direct.dim() == 3:
            gt_direct = gt_direct.unsqueeze(1)
        if direct.dim() != 3 or gt_direct.dim() != 3:
            return None
        steps = min(int(direct.shape[1]), int(gt_direct.shape[1]))
        if steps <= 0:
            return None
        return _DirectPosePair(direct_seq=direct[:, :steps], gt_direct=gt_direct[:, :steps])

    # Group base payload.
    def _compute_direct_pose_group_base_payload(
        self,
        *,
        geo_theta: Optional[torch.Tensor] = None,
        dir_base: Optional[torch.Tensor] = None,
        dir_leg_base: Optional[torch.Tensor] = None,
        dir_nonleg_base: Optional[torch.Tensor] = None,
        dir_arm_base: Optional[torch.Tensor] = None,
        dir_else_base: Optional[torch.Tensor] = None,
        arm_split_enable: Optional[bool] = None,
        arm_else_balance_enable: Optional[bool] = None,
        arm_weight: Optional[float] = None,
        else_weight: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        request = _DirectPoseGroupBaseRequest(
            geo_theta=geo_theta,
            dir_base=dir_base,
            dir_leg_base=dir_leg_base,
            dir_nonleg_base=dir_nonleg_base,
            dir_arm_base=dir_arm_base,
            dir_else_base=dir_else_base,
            arm_split_enable=arm_split_enable,
            arm_else_balance_enable=arm_else_balance_enable,
            arm_weight=arm_weight,
            else_weight=else_weight,
            eps=eps,
        )
        dir_base = request.dir_base
        dir_leg_base = request.dir_leg_base
        dir_nonleg_base = request.dir_nonleg_base
        dir_arm_base = request.dir_arm_base
        dir_else_base = request.dir_else_base

        if torch.is_tensor(request.geo_theta):
            split_masks = self._resolve_direct_group_masks(int(request.geo_theta.shape[-1]), request.geo_theta.device)
            if split_masks is None:
                return None
            if not torch.is_tensor(dir_base):
                dir_base = _masked_group_mean(request.geo_theta, split_masks.get('all_ex_root'))
            if not torch.is_tensor(dir_leg_base):
                dir_leg_base = _masked_group_mean(request.geo_theta, split_masks.get('leg'))
            if not torch.is_tensor(dir_nonleg_base):
                dir_nonleg_base = _masked_group_mean(request.geo_theta, split_masks.get('nonleg'))
            if not torch.is_tensor(dir_arm_base):
                dir_arm_base = _masked_group_mean(request.geo_theta, split_masks.get('arm'))
            if not torch.is_tensor(dir_else_base):
                dir_else_base = _masked_group_mean(request.geo_theta, split_masks.get('else'))
        if not any(torch.is_tensor(value) for value in (dir_base, dir_leg_base, dir_nonleg_base, dir_arm_base, dir_else_base)):
            return None

        eps_value = float(self.direct_pose_loss_group_norm_eps if request.eps is None else request.eps)
        if (not _math.isfinite(eps_value)) or eps_value <= 0.0:
            eps_value = 1e-6

        arm_split_active = (
            bool(self.direct_pose_arm_split_enable)
            if request.arm_split_enable is None
            else bool(request.arm_split_enable)
        )
        arm_else_balance_flag = (
            bool(self.direct_pose_loss_arm_else_balance_enable)
            if request.arm_else_balance_enable is None
            else bool(request.arm_else_balance_enable)
        )
        def _resolve_payload_weight(field_name: str, raw_value: Optional[float], default_value: float) -> float:
            candidate = default_value if raw_value is None else raw_value
            try:
                resolved = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a finite scalar in range (0, inf) for direct-pose group payload; "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            if (not _math.isfinite(resolved)) or resolved <= 0.0:
                raise ValueError(
                    f"{field_name} must be a finite scalar in range (0, inf) for direct-pose group payload; "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        arm_w = _resolve_payload_weight("arm_weight", request.arm_weight, self.direct_pose_loss_arm_weight)
        else_w = _resolve_payload_weight("else_weight", request.else_weight, self.direct_pose_loss_else_weight)

        dir_nonleg_effective_base = dir_nonleg_base
        arm_else_balance_active = 0.0
        if (
            arm_else_balance_flag
            and arm_split_active
            and torch.is_tensor(dir_arm_base)
            and torch.is_tensor(dir_else_base)
        ):
            denom = max(eps_value, arm_w + else_w)
            dir_nonleg_effective_base = (dir_arm_base * arm_w + dir_else_base * else_w) / denom
            arm_else_balance_active = 1.0

        payload: Dict[str, Any] = {
            'dir_base': dir_base if torch.is_tensor(dir_base) else float('nan'),
            'dir_leg_base': dir_leg_base if torch.is_tensor(dir_leg_base) else float('nan'),
            'dir_nonleg_base': dir_nonleg_base if torch.is_tensor(dir_nonleg_base) else float('nan'),
            'dir_nonleg_effective_base': (
                dir_nonleg_effective_base if torch.is_tensor(dir_nonleg_effective_base) else float('nan')
            ),
            'dir_arm_base': dir_arm_base if torch.is_tensor(dir_arm_base) else float('nan'),
            'dir_else_base': dir_else_base if torch.is_tensor(dir_else_base) else float('nan'),
            'leg_over_nonleg': float('nan'),
            'leg_over_nonleg_effective': float('nan'),
            'arm_over_else': float('nan'),
            'direct_pose_arm_else_balance_active': float(arm_else_balance_active),
            'direct_pose_loss_arm_weight': float(arm_w),
            'direct_pose_loss_else_weight': float(else_w),
        }
        if torch.is_tensor(dir_leg_base) and torch.is_tensor(dir_nonleg_base):
            payload['leg_over_nonleg'] = float(
                (dir_leg_base / dir_nonleg_base.clamp_min(eps_value)).detach().cpu()
            )
        if torch.is_tensor(dir_leg_base) and torch.is_tensor(dir_nonleg_effective_base):
            payload['leg_over_nonleg_effective'] = float(
                (dir_leg_base / dir_nonleg_effective_base.clamp_min(eps_value)).detach().cpu()
            )
        if torch.is_tensor(dir_arm_base) and torch.is_tensor(dir_else_base):
            payload['arm_over_else'] = float(
                (dir_arm_base / dir_else_base.clamp_min(eps_value)).detach().cpu()
            )
        return payload

    # Group norm public wrapper / EMA helpers.
    def _compute_direct_pose_group_norm_payload(
        self,
        dir_leg_base: torch.Tensor,
        dir_nonleg_base: torch.Tensor,
        dir_nonleg_effective_base: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        result = self._compute_direct_pose_group_norm_from_request(_DirectPoseGroupNormRequest(
            dir_leg_base,
            dir_nonleg_base,
            dir_nonleg_effective_base,
            update_ema_state=True,
        ))
        return result.objective, result.stats

    def _direct_pose_group_norm_ema_snapshot(self) -> Dict[str, Any]:
        ema_state = getattr(self, '_direct_pose_group_norm_ema', None)
        return dict(ema_state) if isinstance(ema_state, dict) else {}

    def _direct_pose_group_norm_ema_value(
        self,
        ema_state: Dict[str, Any],
        key: str,
        default_value: torch.Tensor,
    ) -> torch.Tensor:
        candidate = ema_state.get(key, None)
        if torch.is_tensor(candidate):
            candidate = candidate.to(device=default_value.device, dtype=default_value.dtype)
            try:
                if bool(torch.isfinite(candidate).all().detach().cpu().item()):
                    return candidate
            except (RuntimeError, ValueError, TypeError):
                return default_value.detach()
        return default_value.detach()

    def _store_direct_pose_group_norm_ema(self, ema_update_payload: Dict[str, Any]) -> None:
        self._direct_pose_group_norm_ema = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in ema_update_payload.items()
        }

    # Group norm typed implementation.
    def _compute_direct_pose_group_norm_shared(
        self,
        dir_leg_base: torch.Tensor,
        dir_nonleg_base: torch.Tensor,
        dir_nonleg_effective_base: torch.Tensor,
        *,
        direct_group_w_leg: Optional[float] = None,
        direct_group_w_nonleg: Optional[float] = None,
        direct_group_beta: Optional[float] = None,
        direct_group_ratio_min: Optional[float] = None,
        direct_group_ratio_max: Optional[float] = None,
        direct_group_eps: Optional[float] = None,
        update_ema_state: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        return self._compute_direct_pose_group_norm_from_request(_DirectPoseGroupNormRequest(
            dir_leg_base=dir_leg_base,
            dir_nonleg_base=dir_nonleg_base,
            dir_nonleg_effective_base=dir_nonleg_effective_base,
            direct_group_w_leg=direct_group_w_leg,
            direct_group_w_nonleg=direct_group_w_nonleg,
            direct_group_beta=direct_group_beta,
            direct_group_ratio_min=direct_group_ratio_min,
            direct_group_ratio_max=direct_group_ratio_max,
            direct_group_eps=direct_group_eps,
            update_ema_state=update_ema_state,
        )).as_tuple()

    def _compute_direct_pose_group_norm_from_request(
        self,
        request: _DirectPoseGroupNormRequest,
    ) -> _DirectPoseGroupNormResult:
        def _resolve_scalar(
            field_name: str,
            value: Optional[float],
            fallback: float,
            *,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            min_inclusive: bool = True,
            max_inclusive: bool = True,
            expected_range: Optional[str] = None,
        ) -> float:
            raw_value = fallback if value is None else value
            try:
                resolved = float(raw_value)
            except (TypeError, ValueError) as exc:
                range_text = f" in range {expected_range}" if expected_range is not None else ""
                raise TypeError(
                    f"{field_name} must be a finite scalar{range_text}; "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            lower_ok = True
            upper_ok = True
            if min_value is not None:
                lower_ok = resolved >= float(min_value) if min_inclusive else resolved > float(min_value)
            if max_value is not None:
                upper_ok = resolved <= float(max_value) if max_inclusive else resolved < float(max_value)
            if (not _math.isfinite(resolved)) or (not lower_ok) or (not upper_ok):
                range_text = f" in range {expected_range}" if expected_range is not None else ""
                raise ValueError(
                    f"{field_name} must be a finite scalar{range_text}; "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        eps_value = _resolve_scalar(
            "direct_group_eps",
            request.direct_group_eps,
            self.direct_pose_loss_group_norm_eps,
            min_value=0.0,
            min_inclusive=False,
            expected_range="(0, inf)",
        )
        ratio_min = _resolve_scalar(
            "direct_group_ratio_min",
            request.direct_group_ratio_min,
            self.direct_pose_loss_group_norm_ratio_min,
            min_value=eps_value,
            expected_range=f"[direct_group_eps={eps_value}, inf)",
        )
        ratio_max = _resolve_scalar(
            "direct_group_ratio_max",
            request.direct_group_ratio_max,
            self.direct_pose_loss_group_norm_ratio_max,
            min_value=eps_value,
            expected_range=f"[direct_group_eps={eps_value}, inf)",
        )
        if ratio_min > ratio_max:
            raise ValueError(
                "direct_group_ratio_min/direct_group_ratio_max must satisfy direct_group_ratio_min <= direct_group_ratio_max "
                f"with both finite scalars >= direct_group_eps={eps_value}; got direct_group_ratio_min={ratio_min!r}, "
                f"direct_group_ratio_max={ratio_max!r}."
            )
        beta = _resolve_scalar(
            "direct_group_beta",
            request.direct_group_beta,
            self.direct_pose_loss_group_norm_ema_beta,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=False,
            expected_range="(0.0, 0.9999]",
        )
        w_leg = _resolve_scalar(
            "direct_group_w_leg",
            request.direct_group_w_leg,
            self.direct_pose_loss_group_norm_w_leg,
        )
        w_nonleg = _resolve_scalar(
            "direct_group_w_nonleg",
            request.direct_group_w_nonleg,
            self.direct_pose_loss_group_norm_w_nonleg,
        )

        dir_leg_base = request.dir_leg_base
        dir_nonleg_base = request.dir_nonleg_base
        dir_nonleg_effective_base = request.dir_nonleg_effective_base
        ema_state = self._direct_pose_group_norm_ema_snapshot()
        ema_leg_prev = self._direct_pose_group_norm_ema_value(ema_state, 'leg', dir_leg_base)
        ema_non_prev = self._direct_pose_group_norm_ema_value(ema_state, 'nonleg', dir_nonleg_effective_base)

        leg_ratio_raw_t = dir_leg_base / ema_leg_prev.clamp_min(eps_value)
        nonleg_ratio_raw_t = dir_nonleg_effective_base / ema_non_prev.clamp_min(eps_value)
        leg_ratio_t = leg_ratio_raw_t.clamp(ratio_min, ratio_max)
        nonleg_ratio_t = nonleg_ratio_raw_t.clamp(ratio_min, ratio_max)
        leg_hit_min_t = (leg_ratio_raw_t <= ratio_min).to(dtype=dir_leg_base.dtype)
        leg_hit_max_t = (leg_ratio_raw_t >= ratio_max).to(dtype=dir_leg_base.dtype)
        nonleg_hit_min_t = (nonleg_ratio_raw_t <= ratio_min).to(dtype=dir_nonleg_base.dtype)
        nonleg_hit_max_t = (nonleg_ratio_raw_t >= ratio_max).to(dtype=dir_nonleg_base.dtype)
        direct_objective = w_leg * leg_ratio_t + w_nonleg * nonleg_ratio_t
        payload = {
            'dir_group_norm_used': 1.0,
            'dir_group_norm_leg_raw': float(leg_ratio_raw_t.detach().cpu()),
            'dir_group_norm_nonleg_raw': float(nonleg_ratio_raw_t.detach().cpu()),
            'dir_group_norm_leg_clamped': float(leg_ratio_t.detach().cpu()),
            'dir_group_norm_nonleg_clamped': float(nonleg_ratio_t.detach().cpu()),
            'dir_group_norm_leg': float(leg_ratio_t.detach().cpu()),
            'dir_group_norm_nonleg': float(nonleg_ratio_t.detach().cpu()),
            'dir_group_norm_leg_ema': float(ema_leg_prev.detach().cpu()),
            'dir_group_norm_nonleg_ema': float(ema_non_prev.detach().cpu()),
            'dir_group_norm_leg_hit_min': float(leg_hit_min_t.detach().cpu()),
            'dir_group_norm_leg_hit_max': float(leg_hit_max_t.detach().cpu()),
            'dir_group_norm_nonleg_hit_min': float(nonleg_hit_min_t.detach().cpu()),
            'dir_group_norm_nonleg_hit_max': float(nonleg_hit_max_t.detach().cpu()),
            'dir_group_norm_leg_hit_any': float(torch.maximum(leg_hit_min_t, leg_hit_max_t).detach().cpu()),
            'dir_group_norm_nonleg_hit_any': float(torch.maximum(nonleg_hit_min_t, nonleg_hit_max_t).detach().cpu()),
            'dir_group_norm_w_leg': float(w_leg),
            'dir_group_norm_w_nonleg': float(w_nonleg),
        }
        ema_update_payload = dict(
            ema_state,
            leg=(beta * ema_leg_prev + (1.0 - beta) * dir_leg_base.detach()).detach(),
            nonleg=(beta * ema_non_prev + (1.0 - beta) * dir_nonleg_effective_base.detach()).detach(),
        )
        if request.update_ema_state:
            self._store_direct_pose_group_norm_ema(ema_update_payload)
        return _DirectPoseGroupNormResult(direct_objective, payload, ema_update_payload)

    # Direct-pose payload public wrapper / typed assembly.
    def _compute_direct_pose_payload(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> Optional[tuple[torch.Tensor, Dict[str, Any]]]:
        result = self._compute_direct_pose_payload_from_request(_DirectPosePayloadRequest(
            direct=direct,
            gt_motion=gt_motion,
            deg_per_rad=float(deg_per_rad),
        ))
        return result.as_tuple() if result is not None else None

    def _compute_direct_pose_payload_from_request(
        self,
        request: _DirectPosePayloadRequest,
    ) -> Optional[_DirectPosePayloadResult]:
        pair = self._prepare_direct_pose_pair(request.direct, request.gt_motion)
        if pair is None:
            return None

        geo_payload = self.compute_rot6d_geo_loss(pair.direct_seq, pair.gt_direct, return_per_joint=True)
        if isinstance(geo_payload, tuple):
            geo_direct = geo_payload[0]
            geo_theta = geo_payload[1] if len(geo_payload) > 1 else None
        else:
            geo_direct = geo_payload
            geo_theta = None

        direct_objective = geo_direct
        extra: Dict[str, Any] = self._direct_pose_extra_defaults()
        extra.update({
            'direct_pose_geo': geo_direct,
            'direct_pose_geo_deg': geo_direct * request.deg_per_rad,
            'direct_pose_split_active': 1.0 if bool(self.direct_pose_loss_leg_split) else 0.0,
            'direct_pose_arm_split_active': 1.0 if bool(self.direct_pose_arm_split_enable) else 0.0,
        })

        if torch.is_tensor(geo_theta) and geo_theta.ndim >= 3:
            group_payload = self._compute_direct_pose_group_base_payload(geo_theta=geo_theta)
            if group_payload is not None:
                extra.update(group_payload)
                dir_leg_base = group_payload.get('dir_leg_base', None)
                dir_nonleg_base = group_payload.get('dir_nonleg_base', None)
                dir_nonleg_effective_base = group_payload.get('dir_nonleg_effective_base', None)
                if (
                    bool(self.direct_pose_loss_leg_split)
                    and torch.is_tensor(dir_leg_base)
                    and torch.is_tensor(dir_nonleg_base)
                    and torch.is_tensor(dir_nonleg_effective_base)
                ):
                    direct_objective = dir_leg_base + dir_nonleg_effective_base
                    if bool(self.direct_pose_loss_group_norm_enable):
                        direct_objective, norm_payload = self._compute_direct_pose_group_norm_payload(
                            dir_leg_base,
                            dir_nonleg_base,
                            dir_nonleg_effective_base,
                        )
                        extra.update(norm_payload)

        return _DirectPosePayloadResult(direct_objective, extra)

    # Direct-pose applicator.
    def _apply_direct_pose_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        if self.w_direct_pose <= 0.0 or not isinstance(pred_motion, dict):
            _setdefault_stats(stats, self._direct_pose_default_stats())
            return total_loss

        direct = pred_motion.get('out_direct', None)
        if torch.is_tensor(direct):
            payload = self._compute_direct_pose_payload(direct, gt_motion, deg_per_rad)
            if payload is not None:
                direct_objective, extra = payload
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='direct_pose',
                    tensor=direct_objective,
                    weight=self.w_direct_pose,
                    group='core',
                    raw_key='direct_pose_objective',
                    weighted_key='direct_pose_weighted',
                    extra=extra,
                )
        return total_loss

    # Auxiliary applicators.
    # Contact-plan applicator.
    def _contact_plan_extra_stats(
        self,
        plan: Any,
        logits: torch.Tensor,
        gt: torch.Tensor,
        steps: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        probs = plan if torch.is_tensor(plan) else torch.sigmoid(logits)
        probs = _ensure_temporal_axis(probs)
        if probs.dim() < 3 or probs.shape[-1] != gt.shape[-1] or int(probs.shape[1]) < steps:
            return None
        l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
        return {'contact_plan_mse': l_mse}

    def _apply_contact_plan_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if self.w_contact_plan <= 0.0 or not isinstance(pred_motion, dict) or not isinstance(batch, dict):
            return total_loss

        plan_logits = pred_motion.get('contacts_plan_logits', None)
        plan = pred_motion.get('contacts_plan', None)
        gt_contacts = batch.get('contacts', None)
        if torch.is_tensor(plan_logits) and torch.is_tensor(gt_contacts):
            logits, gt, steps = self._prepare_aux_supervision_pair(plan_logits, gt_contacts)
            if steps > 0 and logits.shape[-1] == gt.shape[-1]:
                gt_t = gt[:, :steps].clamp(0.0, 1.0)
                l_bce = F.binary_cross_entropy_with_logits(logits[:, :steps], gt_t)
                extra_stats = self._contact_plan_extra_stats(plan, logits, gt, steps)
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_plan',
                    tensor=l_bce,
                    weight=self.w_contact_plan,
                    group='aux',
                    raw_key='contact_plan_bce',
                    weighted_key='contact_plan_weighted',
                    extra=extra_stats,
                )
        elif torch.is_tensor(plan) and torch.is_tensor(gt_contacts):
            probs, gt, steps = self._prepare_aux_supervision_pair(plan, gt_contacts)
            if steps > 0 and probs.shape[-1] == gt.shape[-1]:
                l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_plan',
                    tensor=l_mse,
                    weight=self.w_contact_plan,
                    group='aux',
                    raw_key='contact_plan_mse',
                    weighted_key='contact_plan_weighted',
                )
        return total_loss

    # Event-clock applicators.
    def _apply_event_clock_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
    ) -> torch.Tensor:
        if not isinstance(pred_motion, dict):
            return total_loss
        if (
            self.event_clock_lambda_entropy_weight <= 0.0
            and self.event_clock_lambda_prior_weight <= 0.0
            and self.event_clock_delta_z_l2_weight <= 0.0
        ):
            return total_loss

        lam_logit = pred_motion.get('event_clock_lambda_logit', None)
        dyn_prior = pred_motion.get('event_clock_dynamic_prior', None)
        delta_z = pred_motion.get('event_clock_delta_z', None)
        if torch.is_tensor(lam_logit):
            logits = lam_logit.unsqueeze(1) if lam_logit.dim() == 2 else lam_logit
            p = torch.sigmoid(logits)
            eps = 1e-6
            ent = -p * torch.log(p + eps) - (1.0 - p) * torch.log(1.0 - p + eps)
            if self.event_clock_lambda_entropy_weight > 0.0:
                l_ent = -ent.mean()
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='event_clock_lambda_entropy',
                    tensor=l_ent,
                    weight=self.event_clock_lambda_entropy_weight,
                    group='aux',
                    raw_key='event_clock_lambda_entropy',
                    weighted_key='event_clock_lambda_entropy_weighted',
                )
            if self.event_clock_lambda_prior_weight > 0.0 and torch.is_tensor(dyn_prior):
                prior = dyn_prior.unsqueeze(1) if dyn_prior.dim() == 2 else dyn_prior
                if prior.shape == p.shape:
                    l_prior = (p - prior.detach()).pow(2).mean()
                    total_loss = self._submit_component_loss(
                        total_loss,
                        stats=stats,
                        name='event_clock_lambda_prior',
                        tensor=l_prior,
                        weight=self.event_clock_lambda_prior_weight,
                        group='aux',
                        raw_key='event_clock_lambda_prior',
                        weighted_key='event_clock_lambda_prior_weighted',
                    )
            stats['event_clock_lambda_mean'] = _stats_float_or(p.detach().mean(), default=float('nan'))

        if self.event_clock_delta_z_l2_weight > 0.0 and torch.is_tensor(delta_z) and delta_z.numel() > 0:
            dz = delta_z.unsqueeze(1) if delta_z.dim() == 2 else delta_z
            l2 = (dz * dz).mean()
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='event_clock_delta_z_l2',
                tensor=l2,
                weight=self.event_clock_delta_z_l2_weight,
                group='aux',
                raw_key='event_clock_delta_z_l2',
                weighted_key='event_clock_delta_z_l2_weighted',
            )
        return total_loss

    # Contact-measurement applicator.
    def _apply_contact_meas_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if self.w_contact_meas <= 0.0 or not isinstance(pred_motion, dict) or not isinstance(batch, dict):
            return total_loss

        meas = pred_motion.get('contacts_meas', None)
        gt_contacts = batch.get('contacts', None)
        if torch.is_tensor(meas) and torch.is_tensor(gt_contacts):
            probs, gt, steps = self._prepare_aux_supervision_pair(meas, gt_contacts)
            if steps > 0 and probs.shape[-1] == gt.shape[-1]:
                l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_meas',
                    tensor=l_mse,
                    weight=self.w_contact_meas,
                    group='aux',
                    raw_key='contact_meas_mse',
                    weighted_key='contact_meas_weighted',
                )
        return total_loss

    # Omega regularization applicator.
    def _apply_omega_l2_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
    ) -> torch.Tensor:
        if self.w_omega_l2 <= 0.0 or not isinstance(pred_motion, dict):
            return total_loss

        omega = pred_motion.get('omega_hat', None)
        if torch.is_tensor(omega) and omega.numel() > 0:
            if omega.dim() == 3:
                omega = omega.unsqueeze(1)
            l2 = (omega * omega).mean()
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='omega_l2',
                tensor=l2,
                weight=self.w_omega_l2,
                group='aux',
                raw_key='omega_l2',
                weighted_key='omega_l2_weighted',
            )
        return total_loss

    # Auxiliary component dispatch.
    def _apply_aux_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        total_loss = self._apply_contact_plan_component(total_loss, stats, pred_motion, batch)
        total_loss = self._apply_event_clock_components(total_loss, stats, pred_motion)
        total_loss = self._apply_contact_meas_component(total_loss, stats, pred_motion, batch)
        return self._apply_omega_l2_component(total_loss, stats, pred_motion)

    # === future: train/loss/orchestration.py ===
    # Forward input prep / base loss.
    def _prepare_forward_inputs(
        self,
        pred_motion: Any,
        gt_motion: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, Optional[torch.Tensor], bool]:
        delta_fallback = False
        if isinstance(pred_motion, dict):
            delta_fallback = bool(pred_motion.get('_delta_fallback', False))
            pred_core_motion = pred_motion.get('out')
            delta_motion = pred_motion.get('delta')
        else:
            pred_core_motion = pred_motion
            delta_motion = None
        return pred_core_motion, gt_motion, delta_motion, delta_fallback

    def _coerce_forward_base_output(self, base_out: Any) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(base_out, tuple):
            loss, stats = base_out
        else:
            loss, stats = base_out, {}
        if isinstance(stats, dict):
            return loss, dict(stats)
        return loss, {}

    def _run_forward_base(
        self,
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        attn_weights=None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base_out = self._forward_base_inner(pred_motion, gt_motion, attn_weights=attn_weights)  # type: ignore[arg-type]
        return self._coerce_forward_base_output(base_out)

    # Loss tracker / stats finalize.
    def _init_loss_group_tracker(self):
        self._loss_group_totals = {key: 0.0 for key in ('core', 'aux', 'long')}

    def _accumulate_loss_contrib(self, name: str, tensor: Optional[torch.Tensor], weight: float, group: Optional[str] = None):
        if tensor is None:
            return
        try:
            w = float(weight)
        except (RuntimeError, TypeError, ValueError):
            w = float(weight.item()) if hasattr(weight, 'item') else 0.0
        if not _math.isfinite(w) or abs(w) < 1e-9:
            return
        if group is None:
            group = self._loss_group_alias.get(name, 'core')
        if group not in self._loss_group_totals:
            self._loss_group_totals[group] = 0.0
        contrib = _stats_float_or(tensor.detach() * w, default=0.0)
        if _math.isfinite(contrib):
            self._loss_group_totals[group] += contrib

    def _loss_group_stats(self) -> Dict[str, float]:
        return {f'loss_group/{k}': float(v) for k, v in self._loss_group_totals.items()}

    def _prepare_aux_supervision_pair(
        self,
        pred_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        pred = _ensure_temporal_axis(pred_tensor)
        target = _ensure_temporal_axis(target_tensor.to(device=pred.device, dtype=pred.dtype))
        steps = min(int(pred.shape[1]), int(target.shape[1]))
        return pred, target, steps

    def _submit_component_loss(
        self,
        total_loss: torch.Tensor,
        *,
        stats: Dict[str, float],
        name: str,
        tensor: Optional[torch.Tensor],
        weight: float,
        group: Optional[str] = None,
        raw_key: Optional[str] = None,
        weighted_key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if tensor is None:
            return total_loss
        total_loss = total_loss + tensor * float(weight)
        self._accumulate_loss_contrib(name, tensor, weight, group=group)
        payload: Dict[str, float] = {}
        if raw_key is not None:
            payload[raw_key] = _stats_float(tensor)
        if weighted_key is not None:
            payload[weighted_key] = _stats_float(tensor * float(weight))
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload[key] = _stats_float(value)
        if payload:
            stats.update(payload)
        return total_loss

    # Applicator dispatch shell.
    def _dispatch_forward_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        *,
        pred_motion: Any,
        pred_core_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
        deg_per_rad: float,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        total_loss = self._apply_motion_components(
            total_loss,
            stats,
            pred_core_motion,
            gt_motion,
            delta_motion,
            delta_fallback,
            deg_per_rad,
        )
        total_loss = self._apply_direct_pose_component(total_loss, stats, pred_motion, gt_motion, deg_per_rad)
        return self._apply_aux_components(total_loss, stats, pred_motion, batch)

    def _finalize_forward_outputs(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        stats.update(self._loss_group_stats())
        return total_loss, stats

    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        self._init_loss_group_tracker()
        pred_core_motion, gt_core_motion, delta_motion, delta_fallback = self._prepare_forward_inputs(pred_motion, gt_motion)
        loss, stats = self._run_forward_base(pred_core_motion, gt_core_motion, attn_weights=attn_weights)
        deg_per_rad = 180.0 / _math.pi
        loss = self._dispatch_forward_components(
            loss,
            stats,
            pred_motion=pred_motion,
            pred_core_motion=pred_core_motion,
            gt_motion=gt_core_motion,
            delta_motion=delta_motion,
            delta_fallback=delta_fallback,
            deg_per_rad=deg_per_rad,
            batch=batch,
        )
        return self._finalize_forward_outputs(loss, stats)
