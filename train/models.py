from __future__ import annotations

"""
Unified model definitions for training and inference.
"""

import math as _math
import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    _resolve_joint_spec_indices,
    build_mlp,
)
from .history import AdaptiveHistoryModule
from .data.layout import infer_rot_joint_count, resolve_rot6d_slice
from .checkpoint.load_schema import (
    normalize_direct_pose_split_state_dict_schema,
)
from .checkpoint.contract import (
    normalize_contact_plan_init_mode,
    normalize_direct_pose_leg_gate_mode,
    normalize_direct_pose_leg_mode,
    normalize_direct_pose_phase_z_mode,
)
from .losses import (
    MotionJointLoss,
    DEFAULT_DIRECT_POSE_LEG_BONES,
    STAGE6_3WAY_ARMCHAIN_BONES,
    STAGE6_3WAY_ARMCHAIN_BONES_CSV,
    _DIRECT_POSE_DEFAULT_STAT_KEYS,
    _DIRECT_POSE_COMPONENT_STAT_KEYS,
    _DirectPosePair,
    _DirectPoseGroupBaseRequest,
    _DirectPoseGroupNormRequest,
    _DirectPoseGroupNormResult,
    _DirectPosePayloadResult,
    _DirectPosePayloadRequest,
    _ensure_temporal_axis,
    _masked_group_mean,
    _masked_group_weighted_mean,
    _setdefault_stats,
    _stats_float,
    _stats_float_or,
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


def _normalize_contact_plan_inject(value: Any) -> str:
    inject = str(value or "none").strip().lower()
    if inject in ("none", "contacts", "plan_z"):
        return inject
    raise ValueError(
        f"Unsupported contact_plan_inject={inject!r}; expected 'none', 'contacts', or 'plan_z'."
    )


def _normalize_contact_plan_init_mode(value: Any) -> str:
    normalized_value = None if isinstance(value, str) and value.strip() == "" else value
    try:
        return normalize_contact_plan_init_mode(normalized_value, default="learnable", strict=True)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def _normalize_direct_pose_phase_z_mode(value: Any) -> str:
    if not isinstance(value, str):
        if value is None:
            return "concat"
        raise TypeError(
            "direct_pose_phase_z_mode must be a string or None; "
            "expected aliases for {'concat', 'replace_contacts'}; "
            f"got actual_type={type(value).__name__}."
        )
    normalized_value = None if value.strip() == "" else value
    try:
        return normalize_direct_pose_phase_z_mode(normalized_value, default="concat", strict=True)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def _normalize_direct_pose_leg_mode(value: Any) -> str:
    if not isinstance(value, str):
        if value is None:
            return "rot6d_add"
        raise TypeError(
            "direct_pose_leg_mode must be a string or None; "
            "expected aliases for {'rot6d_add', 'so3'}; "
            f"got actual_type={type(value).__name__}."
        )
    normalized_value = None if value.strip() == "" else value
    try:
        return normalize_direct_pose_leg_mode(normalized_value, default="rot6d_add", strict=True)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def _normalize_direct_pose_leg_gate_mode(value: Any) -> str:
    if not isinstance(value, str):
        if value is None:
            return "none"
        raise TypeError(
            "direct_pose_leg_gate_mode must be a string or None; "
            "expected aliases for {'none', 'learned', 'scale'}; "
            f"got actual_type={type(value).__name__}."
        )
    try:
        return normalize_direct_pose_leg_gate_mode(value, default="none", strict=True)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def _normalize_direct_pose_leg_contact_order(value: Any) -> str:
    if value is None:
        return "lr"
    if not isinstance(value, str):
        raise TypeError(
            "direct_pose_leg_contact_order must be a string or None; "
            "expected aliases for {'lr', 'rl'}; "
            f"got actual_type={type(value).__name__}."
        )
    order = value.strip().lower()
    if order in ("rl", "r,l", "r l"):
        return "rl"
    if order in ("lr", "l,r", "l r"):
        return "lr"
    raise ValueError(
        "Unsupported direct_pose_leg_contact_order="
        f"{order!r}; expected 'lr' or 'rl' after alias normalization."
    )


def _normalize_direct_pose_leg_side_cue(value: Any) -> str:
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise TypeError(
            "direct_pose_leg_side_cue must be a string or None; "
            "expected aliases for {'none', 'phase_event_age'}; "
            f"got actual_type={type(value).__name__}."
        )
    cue = value.strip().lower()
    if cue in ("", "none", "off", "disable", "disabled"):
        return "none"
    if cue in ("age", "event_age", "eventage", "phase_age", "phase_event_age", "phaseeventage"):
        return "phase_event_age"
    if cue in ("hazard", "td_hazard", "tdhazard", "hazard_acc", "td_hazard_acc", "tdhazard_acc", "hzacc"):
        raise ValueError("direct_pose_leg_side_cue='td_hazard_acc' has been retired; use 'none' or 'phase_event_age'.")
    raise ValueError(
        "Unsupported direct_pose_leg_side_cue="
        f"{cue!r}; expected 'none' or 'phase_event_age' after alias normalization."
    )


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
class _ContactPlanForwardFinal:
    contacts_plan: torch.Tensor
    phase_z_in_direct: Optional[torch.Tensor]
    leg_side_cue_in: Optional[torch.Tensor]
    contacts_plan_logits: Optional[torch.Tensor]
    contact_plan_debug_logits: _ContactPlanDebugLogits
    plan_z_next: Optional[torch.Tensor]
    plan_feat_for_inject: Optional[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _EventClockStepOutputs:
    plan_z: torch.Tensor
    logits_raw: torch.Tensor
    gate_factor: torch.Tensor
    lambda_logit: torch.Tensor
    dynamic_prior: torch.Tensor
    delta_z: torch.Tensor


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


@dataclass(slots=True)
class _ForwardState:
    state: torch.Tensor
    cond: Optional[torch.Tensor]
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
    meas_logits_prev: Optional[torch.Tensor]
    time_index: Optional[torch.Tensor | int | float]
    rollout_step: Optional[torch.Tensor | int | float]
    soft_period: Optional[torch.Tensor] = None
    contacts_meas: Optional[torch.Tensor] = None
    event_clock_delta_meas: Optional[torch.Tensor] = None
    event_clock_lr_diff: Optional[torch.Tensor] = None
    event_clock_lambda_corr: Optional[torch.Tensor] = None
    event_clock_lambda_logit: Optional[torch.Tensor] = None
    event_clock_dynamic_prior: Optional[torch.Tensor] = None
    event_clock_delta_z: Optional[torch.Tensor] = None
    pose_hist_processed: bool = False
    contacts_plan: Optional[torch.Tensor] = None
    plan_z_next: Optional[torch.Tensor] = None
    plan_feat_for_inject: Optional[torch.Tensor] = None
    contacts_plan_logits: Optional[torch.Tensor] = None
    contact_plan_debug_logits: _ContactPlanDebugLogits = field(default_factory=_ContactPlanDebugLogits)
    time_pe_direct: Optional[torch.Tensor] = None
    phase_z_in_direct: Optional[torch.Tensor] = None
    leg_side_cue_in: Optional[torch.Tensor] = None
    h_temporal: Optional[torch.Tensor] = None
    h_final: Optional[torch.Tensor] = None
    result: Optional[Dict[str, torch.Tensor]] = None
    e_t: Optional[torch.Tensor] = None


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

    @classmethod
    def from_config(cls, cfg: "ModelBuildConfig") -> "EventMotionModel":
        return cls(**cfg.to_model_kwargs())

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
        self.contact_plan_inject = _normalize_contact_plan_inject(contact_plan_inject)
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
        self.contact_plan_init_mode = _normalize_contact_plan_init_mode(contact_plan_init_mode)
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
        self.direct_pose_phase_z_mode = _normalize_direct_pose_phase_z_mode(direct_pose_phase_z_mode)
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

        # Optional: leg-specific residual head for direct pose.
        # This provides extra capacity for lower-body joints without forcing per-joint loss tricks.
        self.direct_pose_leg_enable = bool(direct_pose_leg_enable) and bool(self.direct_pose_enable)
        self.direct_pose_leg_bones = direct_pose_leg_bones
        # How to apply the leg residual (add in 6D space vs on-manifold SO(3) composition).
        self.direct_pose_leg_mode = _normalize_direct_pose_leg_mode(direct_pose_leg_mode)
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
        self.direct_pose_leg_gate_mode: str = _normalize_direct_pose_leg_gate_mode(direct_pose_leg_gate_mode)
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
        self._build_event_clock_modules()
        self._build_direct_pose_modules(
            bone_names=bone_names,
            output_layout=output_layout,
            side_routing_requested=direct_pose_leg_side_routing,
            contact_order=direct_pose_leg_contact_order,
            side_embed_dim=direct_pose_leg_side_embed_dim,
            side_plan_other_requested=direct_pose_leg_side_plan_other,
            side_phase_other_requested=direct_pose_leg_side_phase_other,
            side_phase_rel_requested=direct_pose_leg_side_phase_rel,
            side_cue=direct_pose_leg_side_cue,
            side_cue_tau=direct_pose_leg_side_cue_tau,
            side_sign_gate_requested=direct_pose_leg_side_sign_gate,
            side_rank1_requested=direct_pose_leg_side_rank1,
        )
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

    def _resolve_direct_pose_leg_joint_indices(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
    ) -> None:
        if not (bool(self.direct_pose_leg_enable) or bool(self.direct_pose_split_enable)):
            return

        rot_sl = resolve_rot6d_slice(
            output_layout,
            total_dim=self.out_motion_dim,
        )
        if isinstance(rot_sl, slice):
            self.direct_pose_leg_rot6d_slice = rot_sl
        joint_count = infer_rot_joint_count(rot_sl)

        leg_idx = list(getattr(self, "direct_pose_leg_joint_idx", None) or [])
        leg_names = list(getattr(self, "direct_pose_leg_joint_names", None) or [])

        if self.direct_pose_leg_enable:
            leg_idx_raw, leg_names_raw = _resolve_joint_spec_indices(
                getattr(self, "direct_pose_leg_bones", None),
                default_items=("ball_r", "ball_l", "foot_r", "foot_l", "calf_r", "calf_l", "thigh_r", "thigh_l"),
                bone_names=bone_names,
                joint_count=joint_count,
                collect_names=True,
            )
            leg_idx = [int(i) for i in leg_idx_raw]
            leg_names = [str(name) for name in leg_names_raw]
        elif bool(self.direct_pose_split_enable):
            if not leg_idx:
                leg_idx_raw, _ = _resolve_joint_spec_indices(
                    getattr(self, "direct_pose_leg_bones", None),
                    default_items=DEFAULT_DIRECT_POSE_LEG_BONES,
                    bone_names=bone_names,
                    joint_count=joint_count,
                )
                leg_idx = [int(i) for i in leg_idx_raw]
            if leg_idx and not leg_names:
                try:
                    if bone_names is not None:
                        leg_names = [
                            str(bone_names[int(i)])
                            for i in leg_idx
                            if 0 <= int(i) < len(bone_names)
                        ]
                except (TypeError, IndexError):
                    pass

        self.direct_pose_leg_joint_idx = [int(i) for i in leg_idx]
        self.direct_pose_leg_joint_names = [str(name) for name in leg_names]
        if not self.direct_pose_leg_joint_idx:
            return

        leg_idx_tensor = torch.as_tensor(self.direct_pose_leg_joint_idx, dtype=torch.long)
        try:
            if hasattr(self, "direct_pose_leg_joint_idx_tensor"):
                self.direct_pose_leg_joint_idx_tensor = leg_idx_tensor
            else:
                self.register_buffer("direct_pose_leg_joint_idx_tensor", leg_idx_tensor, persistent=True)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            action = "update" if hasattr(self, "direct_pose_leg_joint_idx_tensor") else "register"
            raise RuntimeError(
                "direct_pose leg routing metadata registration failed: "
                "field='direct_pose_leg_joint_idx_tensor' expected a torch.LongTensor with "
                f"shape=(K_leg,) during {action}; got joint_indices={self.direct_pose_leg_joint_idx!r}, "
                f"tensor_shape={tuple(int(v) for v in leg_idx_tensor.shape)!r}, "
                f"tensor_dtype={leg_idx_tensor.dtype}."
            ) from exc

    def _build_direct_pose_split_output_indices(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
    ) -> None:
        if not bool(self.direct_pose_split_enable):
            return
        if str(getattr(self, "direct_pose_meas_mode", "concat") or "concat").strip().lower() != "concat":
            raise ValueError("direct_pose_split_enable currently supports direct_pose_meas_mode='concat' only.")

        split_leg_joint_idx = list(getattr(self, "direct_pose_leg_joint_idx", None) or [])
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

    def _resolve_direct_pose_leg_side_routing_positions(self) -> None:
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
            if hasattr(self, "direct_pose_leg_side_pos_r_tensor"):
                self.direct_pose_leg_side_pos_r_tensor = pos_r_tensor
            else:
                self.register_buffer(
                    "direct_pose_leg_side_pos_r_tensor",
                    pos_r_tensor,
                    persistent=True,
                )
            if hasattr(self, "direct_pose_leg_side_pos_l_tensor"):
                self.direct_pose_leg_side_pos_l_tensor = pos_l_tensor
            else:
                self.register_buffer(
                    "direct_pose_leg_side_pos_l_tensor",
                    pos_l_tensor,
                    persistent=True,
                )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            action = (
                "update"
                if hasattr(self, "direct_pose_leg_side_pos_r_tensor") or hasattr(self, "direct_pose_leg_side_pos_l_tensor")
                else "register"
            )
            raise RuntimeError(
                "direct_pose side routing position buffer registration failed: "
                "fields=('direct_pose_leg_side_pos_r_tensor', 'direct_pose_leg_side_pos_l_tensor') expected "
                f"persistent torch.LongTensor buffers with shape=(K_side,) during {action}; "
                f"got pos_r={self.direct_pose_leg_side_pos_r!r}, pos_l={self.direct_pose_leg_side_pos_l!r}, "
                f"pos_r_shape={tuple(int(v) for v in pos_r_tensor.shape)!r}, "
                f"pos_l_shape={tuple(int(v) for v in pos_l_tensor.shape)!r}."
            ) from exc

    def _init_direct_pose_routing_metadata(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
    ) -> None:
        self._resolve_direct_pose_leg_joint_indices(
            bone_names=bone_names,
            output_layout=output_layout,
        )
        self._build_direct_pose_split_output_indices(
            bone_names=bone_names,
            output_layout=output_layout,
        )
        self._resolve_direct_pose_leg_side_routing_positions()

    def _build_contact_plan_modules(self) -> None:
        self.contact_plan_cell: Optional[nn.GRUCell] = None
        self.contact_plan_head: Optional[nn.Module] = None
        self.contact_plan_time_head: Optional[nn.Module] = None
        self.contact_plan_phase_head: Optional[nn.Module] = None
        self.contact_plan_init_z: Optional[nn.Parameter] = None
        self.contact_plan_init_head: Optional[nn.Module] = None
        self._contact_plan_init_obs_dim = 0
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
    def _build_event_clock_modules(self) -> None:
        self.event_clock_gate: Optional[PeriodicityGate] = None
        self.event_clock_corrector: Optional[PlanZCorrector] = None
        if not (self.contact_plan_enable and self.use_event_clock):
            return

        h_plan = int(self.contact_plan_hidden)
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

    def _prepare_direct_pose_leg_build_state(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
        side_routing_requested: Any,
        contact_order: Any,
        side_embed_dim: Any,
        side_plan_other_requested: Any,
        side_phase_other_requested: Any,
        side_phase_rel_requested: Any,
        side_cue: Any,
        side_cue_tau: Any,
        side_sign_gate_requested: Any,
        side_rank1_requested: Any,
    ) -> None:
        self.direct_pose_leg_side_routing = False
        self.direct_pose_leg_contact_order = "lr"
        self.direct_pose_leg_contact_ch_r = 1
        self.direct_pose_leg_contact_ch_l = 0
        self.direct_pose_leg_side_k = 0
        self.direct_pose_leg_side_pos_r = []
        self.direct_pose_leg_side_pos_l = []
        self.direct_pose_leg_side_embed_dim = 0
        self.direct_pose_leg_side_plan_other = False
        self.direct_pose_leg_side_plan_other_dim = 0
        self.direct_pose_leg_side_phase_other = False
        self.direct_pose_leg_side_phase_other_dim = 0
        self.direct_pose_leg_side_phase_rel = False
        self.direct_pose_leg_side_phase_rel_dim = 0
        self.direct_pose_leg_side_cue = "none"
        self.direct_pose_leg_side_cue_tau = 30.0
        self.direct_pose_leg_side_cue_dim = 0
        self.direct_pose_leg_side_sign_gate = False
        self.direct_pose_leg_side_rank1 = False
        if (not bool(self.direct_pose_leg_enable)) or (
            str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add") != "so3"
        ):
            self.direct_pose_leg_gate_mode = "none"

        self.direct_pose_leg_side_routing = bool(side_routing_requested) and bool(self.direct_pose_leg_enable)

        order = _normalize_direct_pose_leg_contact_order(contact_order)
        if order == "rl":
            self.direct_pose_leg_contact_order = "rl"
            self.direct_pose_leg_contact_ch_r, self.direct_pose_leg_contact_ch_l = 0, 1
        else:
            self.direct_pose_leg_contact_order = "lr"
            self.direct_pose_leg_contact_ch_l, self.direct_pose_leg_contact_ch_r = 0, 1

        if side_embed_dim is None:
            side_emb = 0
        else:
            try:
                side_emb = int(side_embed_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_side_embed_dim must be an integer scalar in range [0, inf); "
                    f"got value={side_embed_dim!r} actual_type={type(side_embed_dim).__name__}."
                ) from exc
        self.direct_pose_leg_side_embed_dim = max(0, int(side_emb))
        self.direct_pose_leg_side_plan_other = bool(side_plan_other_requested) and bool(self.direct_pose_leg_side_routing)
        self.direct_pose_leg_side_plan_other_dim = 1 if bool(self.direct_pose_leg_side_plan_other) else 0
        self.direct_pose_leg_side_phase_other = (
            bool(side_phase_other_requested)
            and bool(self.direct_pose_leg_side_routing)
            and bool(self.direct_pose_use_phase_z)
        )
        self.direct_pose_leg_side_phase_other_dim = 2 if bool(self.direct_pose_leg_side_phase_other) else 0
        self.direct_pose_leg_side_phase_rel = (
            bool(side_phase_rel_requested)
            and bool(self.direct_pose_leg_side_routing)
            and bool(self.direct_pose_use_phase_z)
        )
        self.direct_pose_leg_side_phase_rel_dim = 2 if bool(self.direct_pose_leg_side_phase_rel) else 0

        cue = _normalize_direct_pose_leg_side_cue(side_cue)
        self.direct_pose_leg_side_cue = str(cue)
        if side_cue_tau is None:
            tau = 30.0
        else:
            try:
                tau = float(side_cue_tau)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "direct_pose_leg_side_cue_tau must be a finite scalar in range (0, inf); "
                    f"got value={side_cue_tau!r} actual_type={type(side_cue_tau).__name__}."
                ) from exc
        if (not _math.isfinite(tau)) or tau <= 0.0:
            raise ValueError(
                "direct_pose_leg_side_cue_tau must be a finite scalar in range (0, inf); "
                f"got value={tau!r} actual_type={type(side_cue_tau).__name__}."
            )
        self.direct_pose_leg_side_cue_tau = float(tau)
        self.direct_pose_leg_side_cue_dim = 1 if cue != "none" else 0
        self.direct_pose_leg_side_sign_gate = bool(side_sign_gate_requested) and bool(self.direct_pose_leg_side_routing)
        self.direct_pose_leg_side_rank1 = bool(side_rank1_requested) and bool(self.direct_pose_leg_side_routing)
        if bool(self.direct_pose_leg_side_rank1) and bool(self.direct_pose_leg_side_sign_gate):
            raise ValueError("direct_pose_leg_side_rank1 is incompatible with direct_pose_leg_side_sign_gate (pick one).")

        self._init_direct_pose_routing_metadata(
            bone_names=bone_names,
            output_layout=output_layout,
        )

    def _build_direct_pose_leg_modules(
        self,
        *,
        in_dim: int,
        base_dim: int,
        time_dim: int,
        hid: int,
        drop: float,
        want_meas: bool,
        split_stream_gen: Optional[torch.Generator],
        split_leg_terminal_out_dim: Optional[int],
    ) -> None:
        if not (bool(getattr(self, "direct_pose_leg_enable", False)) and getattr(self, "direct_pose_leg_joint_idx", None)):
            if split_leg_terminal_out_dim is not None:
                self.direct_pose_leg_terminal = self._build_direct_pose_terminal_block(
                    trunk_dim=hid,
                    out_dim=int(split_leg_terminal_out_dim),
                    drop=drop,
                )
            return

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
            if side_emb_dim > 0:
                self.direct_pose_leg_side_embed = nn.Embedding(2, side_emb_dim)
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

    def _build_direct_pose_arm_split_modules(
        self,
        *,
        split_state: Dict[str, Any],
        hid: int,
        nonleg_out_dim: int,
        split_stream_gen: Optional[torch.Generator],
    ) -> None:
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

    def _build_direct_pose_split_modules(
        self,
        *,
        split_state: Optional[Dict[str, Any]],
        in_dim: int,
        hid: int,
        split_stream_gen: Optional[torch.Generator],
    ) -> int:
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
        if split_stream_gen is not None:
            self._advance_linear_stream_(split_stream_gen, in_dim, hid)
            self._advance_linear_stream_(split_stream_gen, hid, hid)
            self._advance_linear_stream_(split_stream_gen, hid, leg_out_dim)
        if bool(split_state["arm_split"]):
            self._build_direct_pose_arm_split_modules(
                split_state=split_state,
                hid=hid,
                nonleg_out_dim=nonleg_out_dim,
                split_stream_gen=split_stream_gen,
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
        return int(leg_out_dim)

    def _build_direct_pose_modules(
        self,
        *,
        bone_names: Optional[Sequence[str]],
        output_layout: Optional[Dict[str, Any]],
        side_routing_requested: Any,
        contact_order: Any,
        side_embed_dim: Any,
        side_plan_other_requested: Any,
        side_phase_other_requested: Any,
        side_phase_rel_requested: Any,
        side_cue: Any,
        side_cue_tau: Any,
        side_sign_gate_requested: Any,
        side_rank1_requested: Any,
    ) -> None:
        self.direct_pose_head: Optional[nn.Module] = None
        self._prepare_direct_pose_leg_build_state(
            bone_names=bone_names,
            output_layout=output_layout,
            side_routing_requested=side_routing_requested,
            contact_order=contact_order,
            side_embed_dim=side_embed_dim,
            side_plan_other_requested=side_plan_other_requested,
            side_phase_other_requested=side_phase_other_requested,
            side_phase_rel_requested=side_phase_rel_requested,
            side_cue=side_cue,
            side_cue_tau=side_cue_tau,
            side_sign_gate_requested=side_sign_gate_requested,
            side_rank1_requested=side_rank1_requested,
        )
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
            split_stream_gen = self._new_generator_from_state(direct_pose_stream_state)
            split_leg_terminal_out_dim = self._build_direct_pose_split_modules(
                split_state=split_state,
                in_dim=in_dim,
                hid=hid,
                split_stream_gen=split_stream_gen,
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

        self._build_direct_pose_leg_modules(
            in_dim=in_dim,
            base_dim=base_dim,
            time_dim=time_dim,
            hid=hid,
            drop=drop,
            want_meas=want_meas,
            split_stream_gen=split_stream_gen,
            split_leg_terminal_out_dim=split_leg_terminal_out_dim,
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
            normalize_direct_pose_split_state_dict_schema(self, state_dict)
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

    def _init_forward_state(
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
        meas_logits_prev: Optional[torch.Tensor],
        time_index: Optional[torch.Tensor | int | float],
        rollout_step: Optional[torch.Tensor | int | float],
    ) -> _ForwardState:
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
        return _ForwardState(
            state=forward_inputs.state,
            cond=forward_inputs.cond,
            angvel=forward_inputs.angvel,
            pose_history=forward_inputs.pose_history,
            plan_z=forward_inputs.plan_z,
            phase_z=forward_inputs.phase_z,
            phase_event_age=forward_inputs.phase_event_age,
            is_single=forward_inputs.is_single,
            device=forward_inputs.device,
            dtype=forward_inputs.dtype,
            batch_size=forward_inputs.batch_size,
            query_steps=forward_inputs.query_steps,
            runtime_controls=forward_inputs.runtime_controls,
            contacts_input=forward_inputs.contacts_input,
            contacts_enc=forward_inputs.contacts_enc,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index,
            rollout_step=rollout_step,
        )

    @staticmethod
    def _state_sequence_contract_error(
        field_name: str,
        value: Optional[torch.Tensor],
        feat_dim: int,
        *,
        batch_size: int,
        query_steps: int,
        reason: str,
        actual_ndim: Optional[int] = None,
        actual_shape: Optional[tuple[int, ...]] = None,
    ) -> str:
        B = int(batch_size)
        Tq = int(query_steps)
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
        self,
        value: Optional[torch.Tensor],
        feat_dim: int,
        *,
        field_name: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        query_steps: int,
    ) -> Optional[torch.Tensor]:
        B = int(batch_size)
        Tq = int(query_steps)
        if value is None or feat_dim <= 0:
            return None
        try:
            seq = value.to(device=device, dtype=dtype)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                self._state_sequence_contract_error(
                    field_name,
                    value,
                    feat_dim,
                    batch_size=B,
                    query_steps=Tq,
                    reason="Tensor conversion to the forward device/dtype failed; no compatibility fallback exists.",
                )
            ) from exc

        input_ndim = int(seq.ndim)
        input_shape = tuple(int(dim) for dim in seq.shape)
        if seq.ndim == 1:
            if seq.shape[0] != feat_dim:
                raise RuntimeError(
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
                        actual_ndim=input_ndim,
                        actual_shape=input_shape,
                        reason=f"3D input batch axis must be 1 or B={B}; got {int(seq.shape[0])}.",
                    )
                )
            if seq.shape[1] not in (1, Tq):
                raise RuntimeError(
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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
                    self._state_sequence_contract_error(
                        field_name,
                        value,
                        feat_dim,
                        batch_size=B,
                        query_steps=Tq,
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

    def _build_forward_time_grid(
        self,
        *,
        time_index: Optional[torch.Tensor | int | float],
        batch_size: int,
        query_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B = int(batch_size)
        Tq = int(query_steps)
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
            if t_grid.shape != (B, Tq):
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
        return t_grid

    def _build_forward_time_pe(
        self,
        *,
        t_grid: torch.Tensor,
        pe_dim: int,
        base_raw: float,
        batch_size: int,
        query_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        error_prefix: str,
    ) -> torch.Tensor:
        B = int(batch_size)
        Tq = int(query_steps)
        try:
            if pe_dim <= 0 or (pe_dim % 2) != 0:
                raise ValueError(f"{error_prefix}_time_pe_dim must be a positive even integer, got {pe_dim}.")
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
                f"{error_prefix} time PE construction failed "
                f"(B={B}, Tq={Tq}, pe_dim={int(pe_dim)}, "
                f"half={int(pe_dim) // 2}, "
                f"t_grid.shape={tuple(int(dim) for dim in t_grid.shape) if torch.is_tensor(t_grid) else None}, "
                f"base={base_raw!r})"
            ) from exc
        return time_pe

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

    def _apply_event_clock_correction(
        self,
        *,
        plan_z_raw: torch.Tensor,
        contacts_meas: torch.Tensor,
        delta_meas: torch.Tensor,
        lr_diff: torch.Tensor,
        period_feat: Optional[torch.Tensor],
    ) -> _EventClockStepOutputs:
        logits_raw = self.contact_plan_head(plan_z_raw)
        plan_raw = torch.sigmoid(logits_raw)
        err_raw = plan_raw - contacts_meas
        gate_factor, lambda_logit, dynamic_prior = self.event_clock_gate(
            err_raw=err_raw,
            delta_meas=delta_meas,
            lr_diff=lr_diff,
            period_feat=period_feat,
        )
        plan_z, delta_z = self.event_clock_corrector(
            plan_z_raw=plan_z_raw,
            contacts_meas=contacts_meas,
            delta_meas=delta_meas,
            err_raw=err_raw,
            period_feat=period_feat,
            lambda_corr=gate_factor,
        )
        return _EventClockStepOutputs(
            plan_z=plan_z,
            logits_raw=logits_raw,
            gate_factor=gate_factor,
            lambda_logit=lambda_logit,
            dynamic_prior=dynamic_prior,
            delta_z=delta_z,
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

    def _run_contact_plan_stage(self, fw: _ForwardState) -> _ForwardState:
        state = fw.state
        cond = fw.cond
        device = fw.device
        dtype = fw.dtype
        B = int(fw.batch_size)
        Tq = int(fw.query_steps)
        runtime_controls = fw.runtime_controls

        if fw.angvel is None and self.angvel_dim > 0:
            fw.angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
        if fw.pose_history is None and self.pose_hist_dim > 0:
            fw.pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

        if self.contact_plan_enable and self.contact_plan_cell is not None and self.contact_plan_head is not None:
            h_plan = int(self.contact_plan_hidden)
            if fw.plan_z is None:
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
                            if int(self.contact_dim) > 0:
                                if fw.contacts_input is None:
                                    c0 = torch.zeros((B, int(self.contact_dim)), device=device, dtype=dtype)
                                else:
                                    c_in = fw.contacts_input.to(device=device, dtype=dtype)
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
                            if int(self.angvel_dim) > 0:
                                av_in = fw.angvel
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
                            if int(self.pose_hist_dim) > 0:
                                ph_in = fw.pose_history
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
                plan_z_t = fw.plan_z.to(device=device, dtype=dtype)
                if plan_z_t.ndim == 3 and plan_z_t.size(1) == 1:
                    plan_z_t = plan_z_t[:, 0]
                if plan_z_t.ndim != 2:
                    plan_z_t = plan_z_t.reshape(B, h_plan)
            cond_seq = cond if cond is not None else torch.zeros((B, Tq, self.cond_dim), device=device, dtype=dtype)

            t_grid = None
            want_time_grid = bool(
                (self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0)
                or int(getattr(self, "direct_pose_time_pe_dim", 0) or 0) > 0
            )
            if want_time_grid:
                t_grid = self._build_forward_time_grid(
                    time_index=fw.time_index,
                    batch_size=B,
                    query_steps=Tq,
                    device=device,
                    dtype=dtype,
                )

            time_pe = None
            if self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0 and t_grid is not None:
                time_pe = self._build_forward_time_pe(
                    t_grid=t_grid,
                    pe_dim=int(self.contact_plan_time_pe_dim),
                    base_raw=getattr(self, "_contact_plan_time_pe_base", 10000.0),
                    batch_size=B,
                    query_steps=Tq,
                    device=device,
                    dtype=dtype,
                    error_prefix="contact_plan",
                )

            if int(getattr(self, "direct_pose_time_pe_dim", 0) or 0) > 0 and t_grid is not None:
                fw.time_pe_direct = self._build_forward_time_pe(
                    t_grid=t_grid,
                    pe_dim=int(getattr(self, "direct_pose_time_pe_dim", 0) or 0),
                    base_raw=getattr(self, "_direct_pose_time_pe_base", 10000.0),
                    batch_size=B,
                    query_steps=Tq,
                    device=device,
                    dtype=dtype,
                    error_prefix="direct_pose",
                )

            plan_probs: list[torch.Tensor] = []
            plan_logits: list[torch.Tensor] = []
            contact_plan_debug_buffers = self._init_contact_plan_debug_buffers(
                runtime_controls.debug_contact_plan_logits_decomp
            )
            time_bias_scale = runtime_controls.contact_plan_time_bias_scale
            plan_z_seq: Optional[list[torch.Tensor]] = [] if self.contact_plan_inject == "plan_z" else None

            phase_input_seq = self._expand_state_sequence(
                fw.phase_z,
                int(getattr(self, "_direct_pose_phase_dim", 0) or 0),
                field_name="phase_z",
                device=device,
                dtype=dtype,
                batch_size=B,
                query_steps=Tq,
            )
            phase_age_seq = self._expand_state_sequence(
                fw.phase_event_age,
                int(self.contact_dim),
                field_name="phase_event_age",
                device=device,
                dtype=dtype,
                batch_size=B,
                query_steps=Tq,
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

            event_clock_on = bool(
                self.use_event_clock and self.event_clock_gate is not None and self.event_clock_corrector is not None
            )
            period_feat: Optional[torch.Tensor] = None
            lr_diff_obs: Optional[torch.Tensor] = None
            lambda_corr_seq: list[torch.Tensor] = []
            lambda_logit_seq: list[torch.Tensor] = []
            dyn_prior_seq: list[torch.Tensor] = []
            delta_z_seq: list[torch.Tensor] = []

            if event_clock_on:
                if (
                    (not fw.pose_hist_processed)
                    and self.adaptive_history_module is not None
                    and fw.pose_history is not None
                    and fw.pose_history.size(-1) > 0
                ):
                    try:
                        pose_hist_for_module = fw.pose_history
                        if pose_hist_for_module.dim() == 3 and pose_hist_for_module.size(1) == 1:
                            pose_hist_for_module = pose_hist_for_module[:, 0]
                        hist_device = self._adaptive_history_device or pose_hist_for_module.device
                        context_feat = state.mean(dim=1).to(hist_device)
                        pose_hist_for_module = pose_hist_for_module.to(hist_device)
                        pose_hist_flat, _ = self.adaptive_history_module(
                            pose_hist_for_module,
                            context=context_feat,
                        )
                        fw.pose_history = pose_hist_flat.to(device).unsqueeze(1)
                        fw.pose_hist_processed = True
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                        raise RuntimeError(
                            "adaptive history module forward failed "
                            f"(event_clock=on, B={B}, Tq={Tq}, "
                            f"pose_history.shape={tuple(int(dim) for dim in fw.pose_history.shape) if torch.is_tensor(fw.pose_history) else None}, "
                            f"context.shape={tuple(int(dim) for dim in context_feat.shape) if torch.is_tensor(context_feat) else None})"
                        ) from exc

                fw.contacts_meas, delta_meas, _ = self._canonicalize_contacts_meas_inputs(
                    fw.contacts_input,
                    fw.meas_logits_prev,
                    batch_size=B,
                    seq_len=Tq,
                    device=device,
                    dtype=dtype,
                )

                lr_diff = torch.zeros((B, Tq, 1), device=device, dtype=dtype)
                if int(self.contact_dim) >= 2:
                    lr_diff = (fw.contacts_meas[..., 0:1] - fw.contacts_meas[..., 1:2]).abs()

                contacts_meas_obs = fw.contacts_meas.detach()
                delta_meas_obs = delta_meas.detach()
                lr_diff_obs = lr_diff.detach()

                fw.event_clock_delta_meas = delta_meas_obs
                fw.event_clock_lr_diff = lr_diff_obs

                if self.period_dim > 0 and self.frozen_encoder is not None and self.frozen_period_head is not None:
                    enc_in = None
                    enc_hidden = None
                    try:
                        enc_feats = []
                        if contacts_meas_obs is not None and contacts_meas_obs.size(-1) > 0:
                            enc_feats.append(contacts_meas_obs)
                        if fw.angvel is not None and fw.angvel.size(-1) > 0:
                            enc_feats.append(fw.angvel)
                        if fw.pose_history is not None and fw.pose_history.size(-1) > 0:
                            enc_feats.append(fw.pose_history)
                        enc_in = torch.cat(enc_feats, dim=-1) if enc_feats else None
                        if enc_in is not None and enc_in.size(-1) == self.encoder_input_dim:
                            with torch.no_grad():
                                enc_hidden = self.frozen_encoder(enc_in, return_summary=False)
                                if isinstance(enc_hidden, tuple):
                                    enc_hidden = enc_hidden[-1]
                                if enc_hidden is not None:
                                    period_feat = torch.tanh(self.frozen_period_head(enc_hidden))
                                    fw.soft_period = period_feat
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                        raise RuntimeError(
                            "frozen period feature forward failed "
                            f"(event_clock=on, B={B}, Tq={Tq}, period_dim={int(self.period_dim)}, "
                            f"encoder_input_dim={int(getattr(self, 'encoder_input_dim', 0) or 0)}, "
                            f"enc_in.shape={tuple(int(dim) for dim in enc_in.shape) if torch.is_tensor(enc_in) else None}, "
                            f"enc_hidden.shape={tuple(int(dim) for dim in enc_hidden.shape) if torch.is_tensor(enc_hidden) else None})"
                        ) from exc
            else:
                fw.contacts_meas, delta_meas, _ = self._canonicalize_contacts_meas_inputs(
                    fw.contacts_input,
                    fw.meas_logits_prev,
                    batch_size=B,
                    seq_len=Tq,
                    device=device,
                    dtype=dtype,
                )
                contacts_meas_obs = fw.contacts_meas.detach()
                delta_meas_obs = delta_meas.detach()

            def _step_contact_plan(step_idx: int) -> None:
                nonlocal plan_z_t

                event_clock_label = "on" if event_clock_on else "off"
                _append_contact_plan_direct_step_inputs(step_idx, event_clock_on=event_clock_on)

                plan_in_t = cond_seq[:, step_idx]
                plan_z_raw = self.contact_plan_cell(plan_in_t, plan_z_t)

                if event_clock_on:
                    period_t = period_feat[:, step_idx] if (period_feat is not None and period_feat.ndim == 3) else None
                    event_clock_out = self._apply_event_clock_correction(
                        plan_z_raw=plan_z_raw,
                        contacts_meas=contacts_meas_obs[:, step_idx],
                        delta_meas=delta_meas_obs[:, step_idx],
                        lr_diff=lr_diff_obs[:, step_idx],
                        period_feat=period_t,
                    )
                    plan_z_t = event_clock_out.plan_z
                    logits_raw = event_clock_out.logits_raw
                    gate_factor = event_clock_out.gate_factor
                else:
                    plan_z_t = plan_z_raw

                if plan_z_seq is not None:
                    plan_z_seq.append(plan_z_t)

                logits_base = self.contact_plan_head(plan_z_t)
                if not event_clock_on:
                    logits_raw = logits_base

                time_term = None
                if time_pe is not None and self.contact_plan_time_head is not None:
                    try:
                        time_bias = self.contact_plan_time_head(time_pe[:, step_idx])
                        if event_clock_on:
                            time_term = gate_factor * (time_bias * time_bias_scale)
                        else:
                            time_term = time_bias * time_bias_scale
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
                        raise RuntimeError(
                            "contact_plan time bias forward failed "
                            f"(event_clock={event_clock_label}, step={step_idx}, B={B}, Tq={Tq}, "
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

                if event_clock_on:
                    lambda_corr_seq.append(event_clock_out.gate_factor)
                    lambda_logit_seq.append(event_clock_out.lambda_logit)
                    dyn_prior_seq.append(event_clock_out.dynamic_prior)
                    delta_z_seq.append(event_clock_out.delta_z)

            for _t in range(Tq):
                _step_contact_plan(_t)

            if lambda_corr_seq:
                fw.event_clock_lambda_corr = torch.stack(lambda_corr_seq, dim=1)
                fw.event_clock_lambda_logit = torch.stack(lambda_logit_seq, dim=1)
                fw.event_clock_dynamic_prior = torch.stack(dyn_prior_seq, dim=1)
                fw.event_clock_delta_z = torch.stack(delta_z_seq, dim=1)
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
            fw.contacts_plan = contact_plan_final.contacts_plan
            fw.phase_z_in_direct = contact_plan_final.phase_z_in_direct
            fw.leg_side_cue_in = contact_plan_final.leg_side_cue_in
            fw.contacts_plan_logits = contact_plan_final.contacts_plan_logits
            fw.contact_plan_debug_logits = contact_plan_final.contact_plan_debug_logits
            fw.plan_z_next = contact_plan_final.plan_z_next
            fw.plan_feat_for_inject = contact_plan_final.plan_feat_for_inject

        if fw.contacts_meas is None:
            if fw.contacts_input is not None:
                fw.contacts_meas = fw.contacts_input.to(device=device, dtype=dtype)
                if fw.contacts_meas.ndim == 2:
                    fw.contacts_meas = fw.contacts_meas.unsqueeze(1)
                elif fw.contacts_meas.ndim != 3:
                    raise ValueError(f"contacts expects shape (B,C) or (B,T,C), got {tuple(fw.contacts_meas.shape)}")
        if fw.contacts_meas is None:
            if fw.contacts_plan is not None:
                fw.contacts_meas = torch.zeros_like(fw.contacts_plan)
            elif self.contact_dim > 0:
                fw.contacts_meas = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)
        if fw.contacts_plan is not None and fw.contacts_meas is not None:
            if fw.contacts_meas.ndim == 2:
                fw.contacts_meas = fw.contacts_meas.unsqueeze(1)
            fw.e_t = fw.contacts_plan - fw.contacts_meas.to(device=device, dtype=dtype)

        if self.use_event_clock and fw.contacts_meas is not None:
            fw.contacts_enc = fw.contacts_meas.detach()
        elif fw.contacts_plan is not None:
            fw.contacts_enc = fw.contacts_plan.detach()
        if fw.contacts_enc is None and self.contact_dim > 0:
            fw.contacts_enc = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)
        return fw

    def _run_motion_core_stage(self, fw: _ForwardState) -> _ForwardState:
        state = fw.state
        cond = fw.cond
        device = fw.device
        dtype = fw.dtype

        encoder_feats = []
        if fw.contacts_enc is not None and fw.contacts_enc.size(-1) > 0:
            encoder_feats.append(fw.contacts_enc)
        if fw.angvel is not None and fw.angvel.size(-1) > 0:
            encoder_feats.append(fw.angvel)
        if fw.pose_history is not None and fw.pose_history.size(-1) > 0:
            pose_hist_for_module = fw.pose_history
            if pose_hist_for_module.dim() == 3 and pose_hist_for_module.size(1) == 1:
                pose_hist_for_module = pose_hist_for_module[:, 0]
            if (not fw.pose_hist_processed) and self.adaptive_history_module is not None:
                hist_device = self._adaptive_history_device or pose_hist_for_module.device
                context_feat = state.mean(dim=1).to(hist_device)
                pose_hist_for_module = pose_hist_for_module.to(hist_device)
                pose_hist_flat, _ = self.adaptive_history_module(
                    pose_hist_for_module,
                    context=context_feat,
                )
                fw.pose_history = pose_hist_flat.to(device).unsqueeze(1)
                fw.pose_hist_processed = True
            encoder_feats.append(fw.pose_history)
        encoder_input = torch.cat(encoder_feats, dim=-1) if encoder_feats else None

        x_inputs = [state]
        if cond is not None:
            x_inputs.append(cond)
        if fw.plan_feat_for_inject is not None:
            feat = fw.plan_feat_for_inject.to(device=device, dtype=dtype)
            if self.contact_plan_inject_detach:
                feat = feat.detach()
            inject_scale = fw.runtime_controls.contact_plan_inject_scale
            if inject_scale != 1.0:
                feat = feat * inject_scale
            x_inputs.append(feat)
        x = torch.cat(x_inputs, dim=-1)
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

        enc_hidden = None
        if fw.soft_period is None:
            if (
                encoder_input is not None
                and self.frozen_encoder is not None
                and encoder_input.size(-1) == self.encoder_input_dim
            ):
                enc_hidden = self.frozen_encoder(encoder_input, return_summary=False)
                if isinstance(enc_hidden, tuple):
                    enc_hidden = enc_hidden[-1]
            if enc_hidden is not None and self.frozen_period_head is not None:
                fw.soft_period = torch.tanh(self.frozen_period_head(enc_hidden))
        if self.period_dim > 0 and self.period_encoder is not None and fw.soft_period is not None:
            period_emb = self.period_encoder(fw.soft_period)
            y1 = y1 + period_emb

        h = self.shared_encoder[2:](y1)
        fw.h_temporal = h + self.residual_proj(x)
        fw.h_temporal = torch.nan_to_num(fw.h_temporal, nan=0.0, posinf=1000000.0, neginf=-1000000.0).clamp(-100.0, 100.0)

        B = int(fw.batch_size)
        Tq = int(fw.query_steps)
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
        q_in = self._pasa_lnq(fw.h_temporal)
        Q = self._pasa_q(q_in).view(B, Tq, self._pasa_heads, Dh).transpose(1, 2)
        K = self._pasa_k(fw.h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        V = self._pasa_v(fw.h_temporal).view(B, Tq, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        attn = torch.softmax(Q * scale @ K.transpose(-1, -2), dim=-1)
        ctx = (attn @ V).transpose(1, 2).contiguous().view(B, Tq, -1)
        attn_out = self._pasa_o(ctx)
        fw.h_final = self.coupling_norm((fw.h_temporal + attn_out) * (1 + g).unsqueeze(1) + b.unsqueeze(1))

        hidden_out = fw.h_final
        out = self.motion_head(fw.h_final)
        if self._bone_adapters and self._bone_adapter_slices:
            delta_full = torch.zeros_like(out)
            for sl, adapter in zip(self._bone_adapter_slices, self._bone_adapters):
                delta_full[..., sl] = adapter(fw.h_final)
            out = out + delta_full
        if fw.is_single:
            out = out.squeeze(1)
            hidden_out = hidden_out.squeeze(1)
            if fw.soft_period is not None:
                fw.soft_period = fw.soft_period.squeeze(1)
        fw.result = self._build_forward_base_result(out=out, hidden_out=hidden_out, attn=attn)
        return fw

    def _run_direct_pose_stage(self, fw: _ForwardState) -> _ForwardState:
        if not self._should_run_direct_pose_forward(fw.contacts_plan):
            return fw

        B = int(fw.batch_size)
        Tq = int(fw.query_steps)
        device = fw.device
        dtype = fw.dtype
        try:
            direct_pose_runtime = self._init_direct_pose_forward_runtime(fw.runtime_controls)
            plan_override = direct_pose_runtime.plan_override
            meas_override = direct_pose_runtime.meas_override
            leg_side_plan_other_ablate_mode = direct_pose_runtime.leg_side_plan_other_ablate_mode
            leg_cross_leg_ablate_mode = direct_pose_runtime.leg_cross_leg_ablate_mode
            plan_in = fw.contacts_plan.detach() if self.direct_pose_detach_plan else fw.contacts_plan
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
                meas_in = fw.contacts_meas
                if meas_in is None and mode == "concat" and int(self.contact_dim) > 0:
                    meas_in = torch.zeros_like(fw.contacts_plan)
                if meas_in is not None and meas_in.ndim == 2:
                    meas_in = meas_in.unsqueeze(1)
                if meas_in is not None:
                    meas_in = meas_in.to(device=device, dtype=dtype)
                    if self.training:
                        drop_p = float(getattr(self, "direct_pose_meas_drop_prob", 0.0) or 0.0)
                        drop_p = max(0.0, min(1.0, drop_p))
                        if drop_p > 0.0:
                            m = (torch.rand(meas_in.shape[:-1] + (1,), device=meas_in.device) < drop_p).to(meas_in.dtype)
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

            direct_feat = fw.cond
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
                direct_feat = fw.h_final
            elif src == "hidden_pre":
                direct_feat = fw.h_temporal
            elif src == "cond+hidden":
                direct_feat = torch.cat([fw.cond, fw.h_final], dim=-1)
            elif src == "cond+hidden_pre":
                direct_feat = torch.cat([fw.cond, fw.h_temporal], dim=-1)
            else:
                direct_feat = fw.cond
            if torch.is_tensor(fw.time_pe_direct):
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
                if fw.time_pe_direct.ndim != 3:
                    raise RuntimeError(
                        "direct_pose time PE concat failed: "
                        f"expected time_pe_direct to be rank-3 `(B, Tq, time_pe_dim)`, "
                        f"got ndim={int(fw.time_pe_direct.ndim)}, "
                        f"shape={tuple(int(dim) for dim in fw.time_pe_direct.shape)}."
                    )
                if int(fw.time_pe_direct.shape[0]) != B or int(fw.time_pe_direct.shape[1]) != Tq:
                    raise RuntimeError(
                        "direct_pose time PE concat failed: "
                        f"expected time_pe_direct prefix `(B={B}, Tq={Tq})`, "
                        f"got shape={tuple(int(dim) for dim in fw.time_pe_direct.shape)}, "
                        f"time_pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)}."
                    )
                try:
                    time_pe_direct_feat = fw.time_pe_direct.to(device=device, dtype=dtype)
                    direct_feat = torch.cat([direct_feat, time_pe_direct_feat], dim=-1)
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        "direct_pose time PE concat failed "
                        f"(B={B}, Tq={Tq}, direct_feat.shape={tuple(int(dim) for dim in direct_feat.shape)}, "
                        f"time_pe_direct.shape={tuple(int(dim) for dim in fw.time_pe_direct.shape)}, "
                        f"time_pe_dim={int(getattr(self, 'direct_pose_time_pe_dim', 0) or 0)})"
                    ) from exc
            phase_in_direct = None
            if bool(getattr(self, "direct_pose_use_phase_z", False)) and int(getattr(self, "_direct_pose_phase_dim", 0) or 0) > 0:
                phase_in_direct = fw.phase_z_in_direct
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

            direct_leg_omega = None
            direct_leg_omega_raw = None
            direct_leg_gate = None
            direct_leg_gate_logits = None
            direct_leg_scale = None
            direct_leg_scale_log = None
            direct_leg_scale_log_raw = None
            direct_leg_side_sign_gate = None

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
                        leg_side_cue_in=fw.leg_side_cue_in,
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
                fw.result,
                direct_out=direct_out,
                direct_leg_omega=direct_leg_omega,
                direct_leg_omega_raw=direct_leg_omega_raw,
                direct_leg_gate=direct_leg_gate,
                direct_leg_gate_logits=direct_leg_gate_logits,
                direct_leg_scale=direct_leg_scale,
                direct_leg_scale_log=direct_leg_scale_log,
                direct_leg_scale_log_raw=direct_leg_scale_log_raw,
                direct_leg_side_sign_gate=direct_leg_side_sign_gate,
                is_single=fw.is_single,
            )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("direct_pose forward failed") from exc
        return fw

    def _run_output_writeback_stage(self, fw: _ForwardState) -> _ForwardState:
        if fw.contacts_meas is not None:
            fw.result['contacts_meas'] = fw.contacts_meas.squeeze(1) if fw.is_single else fw.contacts_meas

        if fw.contacts_plan is not None:
            fw.result['contacts_plan'] = fw.contacts_plan.squeeze(1) if fw.is_single else fw.contacts_plan
            if fw.contacts_plan_logits is not None and torch.is_tensor(fw.contacts_plan_logits):
                fw.result['contacts_plan_logits'] = fw.contacts_plan_logits.squeeze(1) if fw.is_single else fw.contacts_plan_logits
                self._write_contact_plan_debug_logits(
                    fw.result,
                    fw.contact_plan_debug_logits,
                    is_single=fw.is_single,
                    keys=(
                        "contacts_plan_logits_base",
                        "contacts_plan_logits_phase",
                        "contacts_plan_logits_time",
                    ),
                )
            self._write_contact_plan_debug_logits(
                fw.result,
                fw.contact_plan_debug_logits,
                is_single=fw.is_single,
                keys=("contacts_plan_logits_raw",),
            )
            if fw.plan_z_next is not None:
                fw.result['plan_z_next'] = fw.plan_z_next
            if fw.event_clock_lambda_corr is not None:
                fw.result['event_clock_lambda_corr'] = (
                    fw.event_clock_lambda_corr.squeeze(1) if fw.is_single else fw.event_clock_lambda_corr
                )
            if fw.event_clock_lambda_logit is not None:
                fw.result['event_clock_lambda_logit'] = (
                    fw.event_clock_lambda_logit.squeeze(1) if fw.is_single else fw.event_clock_lambda_logit
                )
            if fw.event_clock_dynamic_prior is not None:
                fw.result['event_clock_dynamic_prior'] = (
                    fw.event_clock_dynamic_prior.squeeze(1) if fw.is_single else fw.event_clock_dynamic_prior
                )
            if fw.event_clock_delta_z is not None:
                fw.result['event_clock_delta_z'] = fw.event_clock_delta_z.squeeze(1) if fw.is_single else fw.event_clock_delta_z
            if fw.event_clock_delta_meas is not None:
                fw.result['event_clock_delta_meas'] = (
                    fw.event_clock_delta_meas.squeeze(1) if fw.is_single else fw.event_clock_delta_meas
                )
            if fw.event_clock_lr_diff is not None:
                fw.result['event_clock_lr_diff'] = fw.event_clock_lr_diff.squeeze(1) if fw.is_single else fw.event_clock_lr_diff
            if fw.e_t is not None:
                fw.result['contacts_err'] = fw.e_t.squeeze(1) if fw.is_single else fw.e_t

        self._write_forward_lambda_fusion_outputs(
            fw.result,
            h_final=fw.h_final,
            contact_error=fw.e_t,
            rollout_step=fw.rollout_step,
            device=fw.device,
            dtype=fw.dtype,
            is_single=fw.is_single,
            batch_size=fw.batch_size,
            query_steps=fw.query_steps,
        )
        self._write_forward_so3_delta_outputs(
            fw.result,
            h_final=fw.h_final,
            contact_error=fw.e_t,
            device=fw.device,
            dtype=fw.dtype,
            is_single=fw.is_single,
        )
        self._write_forward_period_output(fw.result, soft_period=fw.soft_period)
        return fw

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
        fw = self._init_forward_state(
            state=state,
            cond=cond,
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_history,
            plan_z=plan_z,
            phase_z=phase_z,
            phase_event_age=phase_event_age,
            meas_logits_prev=meas_logits_prev,
            time_index=time_index,
            rollout_step=rollout_step,
        )
        fw = self._run_contact_plan_stage(fw)
        fw = self._run_motion_core_stage(fw)
        fw = self._run_direct_pose_stage(fw)
        fw = self._run_output_writeback_stage(fw)
        if fw.result is None:
            raise RuntimeError("EventMotionModel.forward motion core stage did not produce a result.")
        return fw.result
