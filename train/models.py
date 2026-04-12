from __future__ import annotations

"""
Unified model definitions for training and inference.
"""

import math as _math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_mlp
from .history import AdaptiveHistoryModule
from .geometry import (
    rot6d_to_matrix,
    geodesic_R,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    parent_relative_matrices,
)
from .layout import parse_layout_entry
from .rotvec_semantics import require_standard_rotvec_bundle

__all__ = [
    'MotionEncoder',
    'PeriodHead',
    '_CondFiLM',
    'EventMotionModel',
    'MotionJointLoss',
    'DEFAULT_DIRECT_POSE_LEG_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES_CSV',
]

# Lower-body joint indices for our default 46-bone skeleton.
# pelvis + thigh/calf/twist/foot/ball (L+R), total 15 joints.
LOWER_BODY_INDICES_V1: tuple[int, ...] = (
    0,
    32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45,
)

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


def _normalize_joint_spec_items(
    spec: Optional[Sequence[Any] | str],
    *,
    default_items: Sequence[Any],
) -> list[Any]:
    raw_items: Any = default_items if spec is None else spec
    if isinstance(raw_items, str):
        candidates = raw_items.split(',')
    elif isinstance(raw_items, (list, tuple)):
        candidates = list(raw_items)
    else:
        candidates = [raw_items]

    items: list[Any] = []
    for item in candidates:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
        elif isinstance(item, (int, np.integer)):
            items.append(int(item))
        elif item is not None:
            text = str(item).strip()
            if text:
                items.append(text)
    return items


def _resolve_joint_spec_indices(
    spec: Optional[Sequence[Any] | str],
    *,
    default_items: Sequence[Any],
    bone_names: Optional[Sequence[str]],
    joint_count: int,
    collect_names: bool = False,
) -> tuple[list[int], list[str]]:
    items = _normalize_joint_spec_items(spec, default_items=default_items)
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(bone_names or [])}
    indices: list[int] = []
    names: list[str] = []
    seen: set[int] = set()
    for item in items:
        idx = None
        name = None
        if isinstance(item, (int, np.integer)):
            idx = int(item)
        else:
            text = str(item).strip()
            if text.isdigit() or (text.startswith('-') and text[1:].isdigit()):
                try:
                    idx = int(text)
                except Exception:
                    idx = None
            else:
                name = text
                idx = name_to_idx.get(text, None)
        if idx is None or idx < 0 or (joint_count > 0 and idx >= joint_count) or idx in seen:
            continue
        seen.add(int(idx))
        indices.append(int(idx))
        if collect_names:
            if name is None and bone_names is not None and int(idx) < len(bone_names):
                name = str(bone_names[int(idx)])
            if name is not None:
                names.append(str(name))
    return indices, names


def _resolve_rot6d_joint_count(rot_slice: Optional[slice], bone_names: Optional[Sequence[str]]) -> int:
    joint_count = 0
    try:
        if isinstance(rot_slice, slice) and rot_slice.start is not None and rot_slice.stop is not None:
            rot_len = int(rot_slice.stop - rot_slice.start)
            if rot_len > 0 and (rot_len % 6) == 0:
                joint_count = int(rot_len // 6)
    except Exception:
        joint_count = 0
    if joint_count <= 0 and bone_names is not None:
        try:
            joint_count = int(len(bone_names))
        except Exception:
            joint_count = 0
    return joint_count


class ContactMeasHeadLowerBodyNoHistV1(nn.Module):
    """
    ContactMeas head v1 (docs/contact_meas_head_redesign_lowerbody_nohist.md):
    - Input: lower-body pose (6D) + lower-body angvel (3D), current frame only.
    - Branch LayerNorm: LN_pose, LN_angvel, then concat -> MLP -> logits.
    """

    def __init__(
        self,
        *,
        pose_dim: int,
        angvel_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pose_dim = int(pose_dim)
        angvel_dim = int(angvel_dim)
        in_dim = max(0, pose_dim) + max(0, angvel_dim)
        hidden_dim = max(8, int(hidden_dim))
        self.pose_dim = pose_dim
        self.angvel_dim = angvel_dim
        self.in_dim = int(in_dim)

        self.ln_pose = nn.LayerNorm(pose_dim) if pose_dim > 0 else nn.Identity()
        self.ln_angvel = nn.LayerNorm(angvel_dim) if angvel_dim > 0 else nn.Identity()

        drop = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Linear(hidden_dim, int(out_dim)),
        )
        try:
            last = self.mlp[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()
        except Exception:
            pass

    def forward(self, pose_lower: torch.Tensor, angvel_lower: torch.Tensor) -> torch.Tensor:
        # pose_lower: (B,T,Dp)  angvel_lower: (B,T,Dw) -> logits: (B,T,C)
        z_pose = self.ln_pose(pose_lower) if self.pose_dim > 0 else None
        z_w = self.ln_angvel(angvel_lower) if self.angvel_dim > 0 else None
        if z_pose is None:
            x = z_w
        elif z_w is None:
            x = z_pose
        else:
            x = torch.cat([z_pose, z_w], dim=-1)
        flat = x.reshape(-1, x.shape[-1])
        logits = self.mlp(flat).view(x.shape[0], x.shape[1], -1)
        return logits


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


import os, json, math, glob, time, argparse

from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable


def _legacy_phase_state_name(suffix: str = "") -> str:
    prefix = "contact" + "_phase" + "_state"
    return prefix if not suffix else prefix + "_" + str(suffix)




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
        # Legacy phase-clock state is retired from mainline; `phase_reset_source` remains only
        # for external validation/post-train coordination.
        phase_reset_source: str = "contacts_meas",
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
        # Step C: keep the shared trunk + grouped readout, but retire the legacy
        # direct_pose_out_leg boundary in favor of a single leg terminal block.
        direct_pose_stepc_unified_leg_terminal: bool = False,
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
        **legacy_kwargs: Any,
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
        legacy_phase_suffixes = (
            "enable",
            "init_mode",
            "hidden",
            "delta_max",
            "delta_init",
            "event_kind",
            "event_thr",
            "event_hyst",
            "event_min_interval",
        )
        for suffix in legacy_phase_suffixes:
            legacy_kwargs.pop(_legacy_phase_state_name(suffix), None)
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(k) for k in legacy_kwargs.keys()))
            raise TypeError(f"EventMotionModel got unexpected keyword arguments: {unknown}")
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
        self.contact_plan_init_mode = str(contact_plan_init_mode or "learnable").lower().strip()
        if self.contact_plan_init_mode in ("learnable_obs", "obs+learnable", "learnable+obs"):
            self.contact_plan_init_mode = "learnable+obs"
        if self.contact_plan_init_mode not in ("zeros", "learnable", "obs", "learnable+obs"):
            self.contact_plan_init_mode = "learnable"
        self.contact_plan_init_hidden = max(8, int(contact_plan_init_hidden or 0))
        self._contact_plan_init_dropout = float(contact_plan_init_dropout or 0.0)
        # Phase reset source is only consumed by external validate/post-train tooling.
        self.phase_reset_source = str(phase_reset_source or "contacts_meas").strip().lower()
        if self.phase_reset_source in ("contacts", "contact", "meas", "contacts_meas"):
            self.phase_reset_source = "contacts_meas"
        elif self.phase_reset_source in ("none", "off", "disable", "disabled"):
            self.phase_reset_source = "none"
        elif self.phase_reset_source in ("hazard", "tdhazard", "td_hazard", "tdhaz"):
            raise ValueError("phase_reset_source='td_hazard' has been retired; use 'contacts_meas' or 'none'.")
        else:
            self.phase_reset_source = "contacts_meas"
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
            self.direct_pose_feat_source = "cond"
        self.direct_pose_time_pe_dim = int(direct_pose_time_pe_dim or 0)
        if self.direct_pose_time_pe_dim % 2 == 1:
            # sin/cos pairs
            self.direct_pose_time_pe_dim += 1
        self._direct_pose_time_pe_base = float(direct_pose_time_pe_base or 10000.0)
        # Optional: feed the explicit phase state into the direct head (dim = 2*contact_dim).
        self.direct_pose_use_phase_z = bool(direct_pose_use_phase_z) and int(self.contact_dim) > 0
        # How phase_z is used in the direct head input (append vs replace contact hints).
        try:
            m = str(direct_pose_phase_z_mode or "concat").strip().lower()
        except Exception:
            m = "concat"
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
        self.direct_pose_stepc_unified_leg_terminal = bool(direct_pose_stepc_unified_leg_terminal) and bool(
            self.direct_pose_split_enable
        )
        self.direct_pose_arm_split_enable = bool(direct_pose_arm_split_enable) and bool(self.direct_pose_split_enable)
        self.direct_pose_arm_bones = direct_pose_arm_bones
        self.direct_pose_out_leg: Optional[nn.Module] = None
        self.direct_pose_leg_terminal: Optional[nn.Module] = None
        self.direct_pose_out_nonleg: Optional[nn.Module] = None
        self.direct_pose_nonleg_proj: Optional[nn.Module] = None
        self.direct_pose_out_arm: Optional[nn.Module] = None
        self.direct_pose_out_else: Optional[nn.Module] = None
        self.direct_pose_arm_proj: Optional[nn.Module] = None
        self.direct_pose_else_proj: Optional[nn.Module] = None
        self.direct_pose_nonleg_proj_dim = max(0, int(direct_pose_nonleg_proj_dim or 0))
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
        try:
            m = str(direct_pose_leg_mode or "rot6d_add").strip().lower()
        except Exception:
            m = "rot6d_add"
        if m in ("so3", "omega", "so3_compose", "compose", "exp", "expmap", "log", "axisangle", "axis_angle"):
            m = "so3"
        else:
            m = "rot6d_add"
        self.direct_pose_leg_mode = m
        self.direct_pose_leg_stopgrad_main = bool(direct_pose_leg_stopgrad_main)
        self.direct_pose_leg_detach_feat = bool(direct_pose_leg_detach_feat)
        try:
            mx = float(direct_pose_leg_max_deg or 0.0)
        except Exception:
            mx = 0.0
        self.direct_pose_leg_max_rad: float = max(0.0, mx) * (_math.pi / 180.0)

        # Optional: learned gate (only meaningful for SO(3) leg mode).
        try:
            gm = str(direct_pose_leg_gate_mode or "none").strip().lower()
        except Exception:
            gm = "none"
        if gm in ("mlp", "net", "nn", "learn", "learned", "gate"):
            gm = "learned"
        if gm in ("scale", "mag", "magnitude", "logmag", "log_mag", "exp", "alpha"):
            gm = "scale"
        if gm in ("", "none", "off", "disable", "disabled", "0"):
            gm = "none"
        if gm not in ("none", "learned", "scale"):
            gm = "none"
        self.direct_pose_leg_gate_mode: str = str(gm)
        try:
            gp = float(direct_pose_leg_gate_power or 1.0)
        except Exception:
            gp = 1.0
        if (not _math.isfinite(gp)) or gp <= 0.0:
            gp = 1.0
        self.direct_pose_leg_gate_power: float = float(gp)
        try:
            lc = float(direct_pose_leg_scale_log_clip or 4.0)
        except Exception:
            lc = 4.0
        if (not _math.isfinite(lc)) or lc <= 0.0:
            lc = 4.0
        self.direct_pose_leg_scale_log_clip: float = float(lc)
        try:
            sk = float(direct_pose_leg_scale_clamp_k or 0.0)
        except Exception:
            sk = 0.0
        if (not _math.isfinite(sk)) or sk <= 1.0:
            sk = 0.0
        self.direct_pose_leg_scale_clamp_k: float = float(sk)
        if (not bool(self.direct_pose_leg_enable)) or (str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add") != "so3"):
            # Gate only applies to on-manifold omega composition path.
            self.direct_pose_leg_gate_mode = "none"

        # Optional: per-side routing + shared head (applied only when leg head is enabled).
        self.direct_pose_leg_side_routing = bool(direct_pose_leg_side_routing) and bool(self.direct_pose_leg_enable)
        try:
            order = str(direct_pose_leg_contact_order or "lr").strip().lower()
        except Exception:
            order = "lr"
        if order in ("rl", "r,l", "r l"):
            self.direct_pose_leg_contact_order = "rl"
            self.direct_pose_leg_contact_ch_r, self.direct_pose_leg_contact_ch_l = 0, 1
        else:
            # Default: dataset contact channels are [L, R] (see train/io.py: load_soft_contacts_from_json).
            self.direct_pose_leg_contact_order = "lr"
            self.direct_pose_leg_contact_ch_l, self.direct_pose_leg_contact_ch_r = 0, 1
        try:
            side_emb = int(direct_pose_leg_side_embed_dim or 0)
        except Exception:
            side_emb = 0
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
        try:
            cue = str(direct_pose_leg_side_cue or "none").strip().lower()
        except Exception:
            cue = "none"
        if cue in ("", "none", "off", "disable", "disabled"):
            cue = "none"
        elif cue in ("age", "event_age", "eventage", "phase_age", "phase_event_age", "phaseeventage"):
            cue = "phase_event_age"
        elif cue in ("hazard", "td_hazard", "tdhazard", "hazard_acc", "td_hazard_acc", "tdhazard_acc", "hzacc"):
            raise ValueError("direct_pose_leg_side_cue='td_hazard_acc' has been retired; use 'none' or 'phase_event_age'.")
        else:
            cue = "none"
        self.direct_pose_leg_side_cue: str = str(cue)
        try:
            tau = float(direct_pose_leg_side_cue_tau or 0.0)
        except Exception:
            tau = 0.0
        if (not _math.isfinite(tau)) or tau <= 0.0:
            tau = 30.0
        self.direct_pose_leg_side_cue_tau: float = float(tau)
        self.direct_pose_leg_side_cue_dim: int = 1 if cue != "none" else 0
        # Optional: per-side sign gate is only meaningful when side routing is enabled.
        self.direct_pose_leg_side_sign_gate = bool(direct_pose_leg_side_sign_gate) and bool(self.direct_pose_leg_side_routing)
        # Optional: rank-1 coupling is only meaningful when side routing is enabled.
        self.direct_pose_leg_side_rank1 = bool(direct_pose_leg_side_rank1) and bool(self.direct_pose_leg_side_routing)
        if bool(self.direct_pose_leg_side_rank1) and bool(self.direct_pose_leg_side_sign_gate):
            raise ValueError("direct_pose_leg_side_rank1 is incompatible with direct_pose_leg_side_sign_gate (pick one).")
        if self.direct_pose_leg_enable:
            # Determine BoneRotations6D slice and joint count.
            rot_sl = None
            if isinstance(output_layout, dict):
                rot_sl = parse_layout_entry(output_layout.get("BoneRotations6D"), "BoneRotations6D", self.out_motion_dim)
            if rot_sl is None and bone_names is not None and len(bone_names) > 0:
                rot_sl = slice(0, min(self.out_motion_dim, int(len(bone_names) * 6)))
            if isinstance(rot_sl, slice):
                self.direct_pose_leg_rot6d_slice = rot_sl

            J = _resolve_rot6d_joint_count(rot_sl, bone_names)
            leg_idx, leg_names = _resolve_joint_spec_indices(
                direct_pose_leg_bones,
                default_items=("ball_r", "ball_l", "foot_r", "foot_l", "calf_r", "calf_l", "thigh_r", "thigh_l"),
                bone_names=bone_names,
                joint_count=J,
                collect_names=True,
            )
            self.direct_pose_leg_joint_idx.extend(int(i) for i in leg_idx)
            self.direct_pose_leg_joint_names.extend(str(name) for name in leg_names)

            # Persist joint indices into state_dict so evaluation scripts can reconstruct the head.
            if self.direct_pose_leg_joint_idx:
                try:
                    self.register_buffer(
                        "direct_pose_leg_joint_idx_tensor",
                        torch.as_tensor(self.direct_pose_leg_joint_idx, dtype=torch.long),
                        persistent=True,
                    )
                except Exception:
                    pass

        # Build leg/non-leg output index mapping once (persisted in buffers) for direct split heads.
        if bool(self.direct_pose_split_enable):
            if str(getattr(self, "direct_pose_meas_mode", "concat") or "concat").strip().lower() != "concat":
                raise ValueError("direct_pose_split_enable currently supports direct_pose_meas_mode='concat' only.")

            split_leg_joint_idx = list(getattr(self, "direct_pose_leg_joint_idx", None) or [])
            if not split_leg_joint_idx:
                rot_sl_split = None
                if isinstance(output_layout, dict):
                    rot_sl_split = parse_layout_entry(output_layout.get("BoneRotations6D"), "BoneRotations6D", self.out_motion_dim)
                if rot_sl_split is None and bone_names is not None and len(bone_names) > 0:
                    rot_sl_split = slice(0, min(self.out_motion_dim, int(len(bone_names) * 6)))

                J_split = _resolve_rot6d_joint_count(rot_sl_split, bone_names)
                split_leg_joint_idx, _ = _resolve_joint_spec_indices(
                    direct_pose_leg_bones,
                    default_items=DEFAULT_DIRECT_POSE_LEG_BONES,
                    bone_names=bone_names,
                    joint_count=J_split,
                )

            # Keep split leg joint indices available to posttrain loss even when
            # direct_pose_leg_enable=false (split-head-only training).
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
                    except Exception:
                        pass
                try:
                    leg_idx_tensor = torch.as_tensor(self.direct_pose_leg_joint_idx, dtype=torch.long)
                    if hasattr(self, "direct_pose_leg_joint_idx_tensor"):
                        self.direct_pose_leg_joint_idx_tensor = leg_idx_tensor
                    else:
                        self.register_buffer("direct_pose_leg_joint_idx_tensor", leg_idx_tensor, persistent=True)
                except Exception:
                    pass

            rot_sl = None
            if isinstance(output_layout, dict):
                rot_sl = parse_layout_entry(output_layout.get("BoneRotations6D"), "BoneRotations6D", self.out_motion_dim)
            if rot_sl is None and bone_names is not None and len(bone_names) > 0:
                rot_sl = slice(0, min(self.out_motion_dim, int(len(bone_names) * 6)))
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
            J_rot = int(rot_len // 6)

            def build_split_out_index(
                joint_indices: Sequence[int],
                *,
                base_mask: Optional[torch.Tensor] = None,
                empty_error: str,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                dim_mask = torch.zeros((out_dim_total,), dtype=torch.bool)
                for j_idx in joint_indices:
                    jj = int(j_idx)
                    if 0 <= jj < J_rot:
                        d0 = int(rot_start + jj * 6)
                        d1 = int(d0 + 6)
                        if 0 <= d0 and d1 <= out_dim_total:
                            dim_mask[d0:d1] = True
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
                    joint_count=J_rot,
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

        # Derive per-side leg joint positions for explicit routing (optional).
        # Positions are in the K-leg list order (direct_pose_leg_joint_idx order).
        if bool(self.direct_pose_leg_side_routing):
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
            names_l = [str(n).lower() for n in self.direct_pose_leg_joint_names]
            pos_r = [i for i, n in enumerate(names_l) if n.endswith(("_r", "right"))]
            pos_l = [i for i, n in enumerate(names_l) if n.endswith(("_l", "left"))]
            if not pos_r or not pos_l:
                raise ValueError(
                    f"direct_pose_leg_side_routing expects both _r and _l joints; got names={self.direct_pose_leg_joint_names}."
                )
            if len(pos_r) != len(pos_l):
                raise ValueError(
                    f"direct_pose_leg_side_routing expects symmetric joint counts per side; got n_r={len(pos_r)} n_l={len(pos_l)} "
                    f"(names={self.direct_pose_leg_joint_names})."
                )
            if (len(pos_r) + len(pos_l)) != len(names_l):
                unknown = [self.direct_pose_leg_joint_names[i] for i in range(len(names_l)) if (i not in pos_r and i not in pos_l)]
                raise ValueError(
                    "direct_pose_leg_side_routing expects all leg joints to be side-tagged with _r/_l; "
                    f"unknown={unknown} (names={self.direct_pose_leg_joint_names})."
                )
            self.direct_pose_leg_side_k = int(len(pos_r))
            self.direct_pose_leg_side_pos_r = list(pos_r)
            self.direct_pose_leg_side_pos_l = list(pos_l)
            # Persist for eval harnesses (tiny, but helpful for reproducibility).
            try:
                self.register_buffer(
                    "direct_pose_leg_side_pos_r_tensor",
                    torch.as_tensor(self.direct_pose_leg_side_pos_r, dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "direct_pose_leg_side_pos_l_tensor",
                    torch.as_tensor(self.direct_pose_leg_side_pos_l, dtype=torch.long),
                    persistent=True,
                )
            except Exception:
                pass
            # Optional side embedding (tiny asymmetry adapter).
            if int(getattr(self, "direct_pose_leg_side_embed_dim", 0) or 0) > 0:
                self.direct_pose_leg_side_embed = nn.Embedding(2, int(self.direct_pose_leg_side_embed_dim))
                # Safe init: start identical for both sides (zeros), allow learning asymmetry if needed.
                try:
                    with torch.no_grad():
                        self.direct_pose_leg_side_embed.weight.zero_()
                except Exception:
                    pass
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

        # ===== Contact Plan (cond-only GRUCell) =====
        # Purpose:
        #   - produce contacts_plan as an *independent anchor* (only sees cond + its own hidden state),
        #     so e_t = contacts_plan - contacts_meas stays informative when pose drifts.
        self.contact_plan_cell: Optional[nn.GRUCell] = None
        self.contact_plan_head: Optional[nn.Module] = None
        self.contact_plan_time_head: Optional[nn.Module] = None
        self.contact_plan_phase_head: Optional[nn.Module] = None
        self.contact_plan_init_z: Optional[nn.Parameter] = None
        self.contact_plan_init_head: Optional[nn.Module] = None
        self._contact_plan_init_obs_dim: int = 0
        self.event_clock_gate: Optional[PeriodicityGate] = None
        self.event_clock_corrector: Optional[PlanZCorrector] = None
        self.direct_pose_head: Optional[nn.Module] = None
        self.lambda_fusion_joint_count: int = 0
        self.lambda_fusion_head: Optional[nn.Module] = None
        if self.contact_plan_enable:
            h_plan = int(self.contact_plan_hidden)
            self.contact_plan_cell = nn.GRUCell(self.cond_dim, h_plan)
            # NOTE: keep GRU input strictly cond-only. Phase/TTA is injected as an additive residual on logits.

            # Learnable initial hidden state for contact-plan GRU (mitigates plan_z cold-start).
            # If older checkpoints don't have this parameter, strict=False loading keeps it at zeros.
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
                    # Safe init: keep obs-conditioned delta near 0 before training.
                    try:
                        last = self.contact_plan_init_head[-1]
                        if isinstance(last, nn.Linear):
                            with torch.no_grad():
                                last.weight.zero_()
                                if last.bias is not None:
                                    last.bias.zero_()
                    except Exception:
                        pass
            self.contact_plan_head = nn.Sequential(
                nn.LayerNorm(h_plan),
                nn.Linear(h_plan, h_plan),
                nn.ReLU(),
                nn.Dropout(self._contact_plan_dropout),
                nn.Linear(h_plan, int(self._contact_plan_logits_dim)),
            )
            if self.contact_plan_time_pe_dim > 0:
                self.contact_plan_time_head = nn.Linear(self.contact_plan_time_pe_dim, int(self._contact_plan_logits_dim))
                try:
                    with torch.no_grad():
                        self.contact_plan_time_head.weight.zero_()
                        if getattr(self.contact_plan_time_head, "bias", None) is not None:
                            self.contact_plan_time_head.bias.zero_()
                except Exception:
                    pass


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

            if self.direct_pose_enable:
                want_meas = (self.direct_pose_meas_mode == "concat")
                base_dim = int(self.cond_dim)
                if self.direct_pose_feat_source in ("hidden", "hidden_pre"):
                    base_dim = int(self.hidden_dim)
                elif self.direct_pose_feat_source in ("cond+hidden", "cond+hidden_pre"):
                    base_dim = int(self.cond_dim + self.hidden_dim)
                time_dim = int(getattr(self, "direct_pose_time_pe_dim", 0) or 0)
                phase_dim = int(getattr(self, "_direct_pose_phase_dim", 0) or 0)
                if self.direct_pose_phase_z_mode == "replace_contacts":
                    # direct_in = [direct_feat(+time_pe), phase_z_in]  (no plan/meas)
                    in_dim = int(base_dim + time_dim + phase_dim)
                else:
                    in_dim = int(
                        base_dim
                        + self.contact_dim
                        + (self.contact_dim if want_meas else 0)
                        + time_dim
                        + phase_dim
                    )
                hid = int(self.direct_pose_hidden)
                drop = float(self._direct_pose_dropout)
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
                    # Shared trunk stays compatible with the legacy head's first two Linear layers.
                    self.direct_pose_head = nn.Sequential(
                        nn.Linear(in_dim, hid),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        nn.Linear(hid, hid),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    )
                    if bool(getattr(self, "direct_pose_stepc_unified_leg_terminal", False)):
                        self.direct_pose_leg_terminal = self._build_direct_pose_terminal_block(
                            trunk_dim=hid,
                            out_dim=leg_out_dim,
                            drop=drop,
                        )
                    else:
                        self.direct_pose_out_leg, _ = self._build_split_head_branch(trunk_dim=hid, out_dim=leg_out_dim)
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
                            trunk_dim=hid, out_dim=arm_out_dim, proj_dim=proj_dim
                        )
                        self.direct_pose_out_else, self.direct_pose_else_proj = self._build_split_head_branch(
                            trunk_dim=hid, out_dim=else_out_dim, proj_dim=proj_dim
                        )
                    else:
                        proj_dim = int(getattr(self, "direct_pose_nonleg_proj_dim", 0) or 0)
                        self.direct_pose_out_nonleg, self.direct_pose_nonleg_proj = self._build_split_head_branch(
                            trunk_dim=hid, out_dim=nonleg_out_dim, proj_dim=proj_dim
                        )
                else:
                    out_dim = int(self.out_motion_dim)
                    if self.direct_pose_meas_mode == "mode_select":
                        out_dim = int(out_dim) * 2
                    self.direct_pose_head = nn.Sequential(
                        nn.Linear(in_dim, hid),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        nn.Linear(hid, hid),
                        nn.ReLU(),
                        nn.Dropout(drop) if drop > 0 else nn.Identity(),
                        nn.Linear(hid, int(out_dim)),
                    )
                # Optional: leg residual head (extra capacity for selected joints only).
                # - rot6d_add: predicts 6D residuals (added in parameter space; legacy)
                # - so3: predicts omega in so(3) (composed on-manifold: exp(omega) @ R_main)
                if bool(getattr(self, "direct_pose_leg_enable", False)) and getattr(self, "direct_pose_leg_joint_idx", None):
                    leg_k = int(len(self.direct_pose_leg_joint_idx))
                    leg_mode = str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip()
                    leg_out = (3 if leg_mode == "so3" else 6) * int(leg_k)
                    if leg_out > 0:
                        self.direct_pose_leg_head = nn.Sequential(
                            nn.Linear(in_dim, hid),
                            nn.ReLU(),
                            nn.Dropout(drop) if drop > 0 else nn.Identity(),
                            nn.Linear(hid, hid),
                            nn.ReLU(),
                            nn.Dropout(drop) if drop > 0 else nn.Identity(),
                            nn.Linear(hid, int(leg_out)),
                        )
                        # Safe init: start with zero residual so behavior matches baseline ckpts.
                        try:
                            last = self.direct_pose_leg_head[-1]
                            if isinstance(last, nn.Linear):
                                with torch.no_grad():
                                    last.weight.zero_()
                                    if last.bias is not None:
                                        last.bias.zero_()
                        except Exception:
                            pass
                        # Optional: learned gate/scale head (predicts per-joint logits; applied in forward).
                        gm_leg = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                        if gm_leg in ("learned", "scale"):
                            gate_out = int(leg_k)
                            self.direct_pose_leg_gate_head = nn.Sequential(
                                nn.Linear(in_dim, hid),
                                nn.ReLU(),
                                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                nn.Linear(hid, hid),
                                nn.ReLU(),
                                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                nn.Linear(hid, int(gate_out)),
                            )
                            # Safe init:
                            # - learned gate: start mostly "open" but not fully saturated (sigmoid(2)≈0.881)
                            # - scale: start as identity scaling (exp(0)=1)
                            try:
                                last = self.direct_pose_leg_gate_head[-1]
                                if isinstance(last, nn.Linear):
                                    with torch.no_grad():
                                        last.weight.zero_()
                                        if last.bias is not None:
                                            if gm_leg == "learned":
                                                last.bias.fill_(2.0)
                                            else:
                                                last.bias.zero_()
                            except Exception:
                                pass
                    # Optional: shared-weight, per-side routed leg head (SO(3) only).
                    # - Input: [direct_feat(+time_pe), plan_side, meas_side, phase_side(2D), (optional side_emb)]
                    # - Output: omega for one side only, then scatter back to K joints.
                    if bool(getattr(self, "direct_pose_leg_side_routing", False)) and int(getattr(self, "direct_pose_leg_side_k", 0) or 0) > 0:
                        leg_side_k = int(getattr(self, "direct_pose_leg_side_k", 0) or 0)
                        # Base feature dim = direct_feat(+time_pe) dim.
                        base_leg_dim = int(base_dim + time_dim)
                        side_emb_dim = int(getattr(self, "direct_pose_leg_side_embed_dim", 0) or 0)
                        phase_side_dim = 2 if bool(getattr(self, "direct_pose_use_phase_z", False)) else 0
                        meas_side_dim = 1 if want_meas else 0
                        plan_other_side_dim = int(getattr(self, "direct_pose_leg_side_plan_other_dim", 0) or 0)
                        phase_other_side_dim = int(getattr(self, "direct_pose_leg_side_phase_other_dim", 0) or 0)
                        phase_rel_side_dim = int(getattr(self, "direct_pose_leg_side_phase_rel_dim", 0) or 0)
                        cue_side_dim = int(getattr(self, "direct_pose_leg_side_cue_dim", 0) or 0)
                        # Input: [direct_feat(+time_pe), plan_side, meas_side, phase_side(2D), (optional plan_other), (optional cue), (optional side_emb)]
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
                        # Output parameterization:
                        # - default: per-joint omega vectors => (K_side * 3)
                        # - rank1  : shared direction (3) + per-joint non-negative scale (K_side) => (3 + K_side)
                        if bool(getattr(self, "direct_pose_leg_side_rank1", False)):
                            leg_out_side = 3 + int(leg_side_k)
                        else:
                            leg_out_side = 3 * int(leg_side_k)  # so3 only
                        if leg_in_dim > 0 and leg_out_side > 0:
                            self.direct_pose_leg_head_shared = nn.Sequential(
                                nn.Linear(leg_in_dim, hid),
                                nn.ReLU(),
                                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                nn.Linear(hid, hid),
                                nn.ReLU(),
                                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                nn.Linear(hid, int(leg_out_side)),
                            )
                            # Safe init: start with zero residual so behavior is controllable for finetune.
                            try:
                                last = self.direct_pose_leg_head_shared[-1]
                                if isinstance(last, nn.Linear):
                                    with torch.no_grad():
                                        last.weight.zero_()
                                        if last.bias is not None:
                                            last.bias.zero_()
                            except Exception:
                                pass
                            # Optional: learned gate/scale head for routed shared leg omega (per-joint logits per side).
                            gm_leg = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                            if gm_leg in ("learned", "scale"):
                                gate_out = int(leg_side_k)
                                self.direct_pose_leg_gate_head_shared = nn.Sequential(
                                    nn.Linear(leg_in_dim, hid),
                                    nn.ReLU(),
                                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                    nn.Linear(hid, hid),
                                    nn.ReLU(),
                                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                                    nn.Linear(hid, int(gate_out)),
                                )
                                # Safe init: same semantics as direct_pose_leg_gate_head.
                                try:
                                    last = self.direct_pose_leg_gate_head_shared[-1]
                                    if isinstance(last, nn.Linear):
                                        with torch.no_grad():
                                            last.weight.zero_()
                                            if last.bias is not None:
                                                if gm_leg == "learned":
                                                    last.bias.fill_(2.0)
                                                else:
                                                    last.bias.zero_()
                                except Exception:
                                    pass
                            # Optional: per-side sign gate head (shared weights; run twice for R/L).
                            if bool(getattr(self, "direct_pose_leg_side_sign_gate", False)):
                                h_gate = max(8, int(hid // 4))
                                self.direct_pose_leg_side_sign_gate_head = nn.Sequential(
                                    nn.Linear(leg_in_dim, h_gate),
                                    nn.ReLU(),
                                    nn.Linear(h_gate, 1),
                                )
                                # Safe init: start near +1 (identity) so existing ckpts behave the same.
                                # (omega head is zero-initialized anyway; this just avoids weird scaling when it learns.)
                                try:
                                    last = self.direct_pose_leg_side_sign_gate_head[-1]
                                    if isinstance(last, nn.Linear):
                                        with torch.no_grad():
                                            last.weight.zero_()
                                            if last.bias is not None:
                                                last.bias.fill_(2.0)  # tanh(2)≈0.964
                                except Exception:
                                    pass

        if self.lambda_fusion_enable:
            try:
                rot_sl = None
                if isinstance(output_layout, dict):
                    rot_sl = parse_layout_entry(output_layout.get("BoneRotations6D"), "BoneRotations6D", self.out_motion_dim)
                if rot_sl is not None:
                    rot_dim = int(rot_sl.stop - rot_sl.start)
                    if rot_dim > 0 and rot_dim % 6 == 0:
                        self.lambda_fusion_joint_count = int(rot_dim // 6)
                elif bone_names is not None and len(bone_names) > 0 and (self.out_motion_dim // 6) > 0:
                    self.lambda_fusion_joint_count = int(len(bone_names))
            except Exception:
                self.lambda_fusion_joint_count = 0

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
                try:
                    last = self.lambda_fusion_head[-1]
                    if isinstance(last, nn.Linear):
                        with torch.no_grad():
                            last.weight.zero_()
                            if last.bias is not None:
                                last.bias.fill_(float(self._lambda_fusion_logit_init))
                except Exception:
                    pass

        # Contacts are provided externally via `contacts_input`; internal meas/hazard heads are retired.

        # ===== SO(3) Delta Corrector (lightweight head) =====
        # - Predicts omega_hat in so(3) to correct ΔR on-manifold.
        # - Does NOT change the main motion output; baseline behavior remains identical
        #   unless the caller explicitly uses omega_hat.
        self.so3_corr_joint_count: int = 0
        self.so3_delta_corrector: Optional[nn.Module] = None
        self.so3_corr_gate_logit: Optional[nn.Parameter] = None
        try:
            rot_sl = None
            if isinstance(output_layout, dict):
                rot_sl = parse_layout_entry(output_layout.get("BoneRotations6D"), "BoneRotations6D", self.out_motion_dim)
            if rot_sl is not None:
                rot_dim = int(rot_sl.stop - rot_sl.start)
                if rot_dim > 0 and rot_dim % 6 == 0:
                    self.so3_corr_joint_count = int(rot_dim // 6)
            elif bone_names is not None and len(bone_names) > 0 and (self.out_motion_dim // 6) > 0:
                self.so3_corr_joint_count = int(len(bone_names))
        except Exception:
            self.so3_corr_joint_count = 0
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
            try:
                last = self.so3_delta_corrector[-1]
                if isinstance(last, nn.Linear):
                    with torch.no_grad():
                        last.weight.zero_()
                        if last.bias is not None:
                            last.bias.zero_()
            except Exception:
                pass

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
        except Exception:
            # Keep adapters disabled if metadata is missing/mismatched.
            self._bone_adapter_slices = []
            self._bone_adapter_names = []
            self._bone_adapters = nn.ModuleList()

        # Optional frozen encoder from预训练，用于提供 soft hint（接触提示 embedding）
        self.frozen_encoder: Optional['MotionEncoder'] = None
        self.frozen_period_head: Optional['PeriodHead'] = None
        self.frozen_contact_head: Optional[nn.Module] = None

    def _direct_pose_split_state(self) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "direct_pose_split_enable", False)):
            return None
        leg_terminal = getattr(self, "direct_pose_leg_terminal", None)
        state = {
            "arm_split": bool(getattr(self, "direct_pose_arm_split_enable", False)),
            "head": getattr(self, "direct_pose_head", None),
            "unified_leg_terminal": bool(leg_terminal is not None),
            "leg_head": leg_terminal if leg_terminal is not None else getattr(self, "direct_pose_out_leg", None),
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
        if (not torch.is_tensor(state["idx_leg"])) or (not torch.is_tensor(state["idx_nonleg"])):
            return None
        if state["arm_split"] and ((not torch.is_tensor(state["idx_arm"])) or (not torch.is_tensor(state["idx_else"]))):
            return None
        return state

    @staticmethod
    def _direct_pose_first_linear(module: Any) -> Optional[nn.Linear]:
        if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], nn.Linear):
            return module[0]
        if isinstance(module, nn.Linear):
            return module
        return None

    @staticmethod
    def _direct_pose_last_linear(module: Any) -> Optional[nn.Linear]:
        if isinstance(module, nn.Linear):
            return module
        if not isinstance(module, nn.Module):
            return None
        last_linear = None
        for mm in module.modules():
            if isinstance(mm, nn.Linear):
                last_linear = mm
        return last_linear

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
    def _build_split_head_branch(
        *,
        trunk_dim: int,
        out_dim: int,
        proj_dim: int = 0,
    ) -> tuple[nn.Linear, Optional[nn.Module]]:
        proj_dim = int(proj_dim or 0)
        if proj_dim > 0:
            proj = nn.Sequential(nn.Linear(int(trunk_dim), proj_dim), nn.ReLU())
            return nn.Linear(proj_dim, int(out_dim)), proj
        return nn.Linear(int(trunk_dim), int(out_dim)), None

    @staticmethod
    def _build_direct_pose_terminal_block(*, trunk_dim: int, out_dim: int, drop: float) -> nn.Sequential:
        block = nn.Sequential(
            nn.Linear(int(trunk_dim), int(trunk_dim)),
            nn.ReLU(),
            nn.Dropout(float(drop)) if float(drop) > 0 else nn.Identity(),
            nn.Linear(int(trunk_dim), int(trunk_dim)),
            nn.ReLU(),
            nn.Dropout(float(drop)) if float(drop) > 0 else nn.Identity(),
            nn.Linear(int(trunk_dim), int(out_dim)),
        )
        try:
            EventMotionModel._init_square_identity_linear_(block[0])
            EventMotionModel._init_square_identity_linear_(block[3])
        except Exception:
            pass
        return block

    @staticmethod
    def _direct_pose_local_index(parent_idx: torch.Tensor, child_idx: torch.Tensor, *, device: torch.device) -> Optional[torch.Tensor]:
        try:
            pos_map = {int(v): i for i, v in enumerate(parent_idx.detach().cpu().tolist())}
            local_idx = [int(pos_map[int(v)]) for v in child_idx.detach().cpu().tolist()]
        except Exception:
            return None
        local_tensor = torch.as_tensor(local_idx, dtype=torch.long, device=device)
        if int(local_tensor.numel()) != int(child_idx.numel()):
            return None
        return local_tensor

    @staticmethod
    def _normalize_split_index_buffer(state_dict: Dict[str, Any], key: str, target_idx: torch.Tensor) -> bool:
        value = state_dict.get(key, None)
        if not torch.is_tensor(value):
            return False
        if tuple(value.shape) != tuple(target_idx.shape):
            state_dict.pop(key, None)
            return True
        if value.dtype == target_idx.dtype:
            return False
        try:
            state_dict[key] = value.to(dtype=target_idx.dtype)
        except Exception:
            state_dict.pop(key, None)
        return True

    @staticmethod
    def _copy_tensor_if_compatible(
        state_dict: Dict[str, Any],
        *,
        target_key: str,
        target_tensor: Optional[torch.Tensor],
        source_tensor: Optional[torch.Tensor],
    ) -> bool:
        if (not torch.is_tensor(target_tensor)) or (not torch.is_tensor(source_tensor)):
            return False
        current = state_dict.get(target_key, None)
        if torch.is_tensor(current) and tuple(current.shape) == tuple(target_tensor.shape):
            return False
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            return False
        state_dict[target_key] = source_tensor
        return True

    @staticmethod
    def _copy_indexed_tensor_if_needed(
        state_dict: Dict[str, Any],
        *,
        target_key: str,
        target_tensor: Optional[torch.Tensor],
        source_tensor: Optional[torch.Tensor],
        index_tensor: Optional[torch.Tensor],
    ) -> bool:
        if (
            (not torch.is_tensor(target_tensor))
            or (not torch.is_tensor(source_tensor))
            or (not torch.is_tensor(index_tensor))
        ):
            return False
        current = state_dict.get(target_key, None)
        if torch.is_tensor(current) and tuple(current.shape) == tuple(target_tensor.shape):
            return False
        try:
            copied = source_tensor.index_select(0, index_tensor.to(device=source_tensor.device, dtype=torch.long))
        except Exception:
            return False
        if tuple(copied.shape) != tuple(target_tensor.shape):
            return False
        state_dict[target_key] = copied
        return True

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

    def _maybe_upgrade_direct_pose_split_state_dict(self, state_dict: Dict[str, Any]) -> bool:
        split_state = self._direct_pose_split_state()
        if (not isinstance(state_dict, dict)) or split_state is None:
            return False
        arm_split = bool(split_state["arm_split"])
        leg_head = split_state["leg_head"]
        nonleg_head = split_state["nonleg_head"]
        arm_head = split_state["arm_head"]
        else_head = split_state["else_head"]
        idx_leg = split_state["idx_leg"]
        idx_nonleg = split_state["idx_nonleg"]
        idx_arm = split_state["idx_arm"]
        idx_else = split_state["idx_else"]
        if leg_head is None:
            return False
        leg_last = self._direct_pose_last_linear(leg_head)
        if leg_last is None:
            return False
        if arm_split:
            if arm_head is None or else_head is None:
                return False
        elif nonleg_head is None:
            return False

        old_w = state_dict.get("direct_pose_head.6.weight", None)
        old_b = state_dict.get("direct_pose_head.6.bias", None)
        has_old = bool(torch.is_tensor(old_w) and old_w.ndim == 2 and int(old_w.shape[0]) == int(self.out_motion_dim))
        ref_device = old_w.device if has_old else leg_last.weight.device
        idx_leg_use = idx_leg.to(device=ref_device, dtype=torch.long)
        idx_nonleg_use = idx_nonleg.to(device=ref_device, dtype=torch.long)
        idx_arm_use = None
        idx_else_use = None
        arm_nonleg_local = None
        else_nonleg_local = None
        if arm_split:
            idx_arm_use = idx_arm.to(device=ref_device, dtype=torch.long)
            idx_else_use = idx_else.to(device=ref_device, dtype=torch.long)
            arm_nonleg_local = self._direct_pose_local_index(idx_nonleg_use, idx_arm_use, device=ref_device)
            else_nonleg_local = self._direct_pose_local_index(idx_nonleg_use, idx_else_use, device=ref_device)
            if arm_nonleg_local is None or else_nonleg_local is None:
                return False
        converted = False
        unified_leg_terminal = bool(split_state.get("unified_leg_terminal", False))
        leg_weight_key = "direct_pose_leg_terminal.6.weight" if unified_leg_terminal else "direct_pose_out_leg.weight"
        leg_bias_key = "direct_pose_leg_terminal.6.bias" if unified_leg_terminal else "direct_pose_out_leg.bias"

        if unified_leg_terminal:
            model_sd = self.state_dict()
            for key in (
                "direct_pose_leg_terminal.0.weight",
                "direct_pose_leg_terminal.0.bias",
                "direct_pose_leg_terminal.3.weight",
                "direct_pose_leg_terminal.3.bias",
            ):
                target_tensor = model_sd.get(key, None)
                converted = self._copy_tensor_if_compatible(
                    state_dict,
                    target_key=key,
                    target_tensor=target_tensor,
                    source_tensor=target_tensor,
                ) or converted

        # Legacy/non-split checkpoints may persist empty split-index buffers.
        # Keep the model-computed split mapping by dropping incompatible ckpt buffers.
        idx_pairs = [
            ("direct_pose_leg_out_idx", idx_leg),
            ("direct_pose_nonleg_out_idx", idx_nonleg),
        ]
        if arm_split:
            idx_pairs.append(("direct_pose_arm_out_idx", idx_arm))
            idx_pairs.append(("direct_pose_else_out_idx", idx_else))
        for key, idx_tgt in idx_pairs:
            converted = self._normalize_split_index_buffer(state_dict, key, idx_tgt) or converted

        if has_old:
            copy_specs = [
                (leg_weight_key, leg_last.weight, idx_leg_use),
            ]
            if arm_split:
                copy_specs.extend([
                    ("direct_pose_out_arm.weight", arm_head.weight, idx_arm_use),
                    ("direct_pose_out_else.weight", else_head.weight, idx_else_use),
                ])
            else:
                copy_specs.append(("direct_pose_out_nonleg.weight", nonleg_head.weight, idx_nonleg_use))
            for target_key, target_tensor, index_tensor in copy_specs:
                converted = self._copy_indexed_tensor_if_needed(
                    state_dict,
                    target_key=target_key,
                    target_tensor=target_tensor,
                    source_tensor=old_w,
                    index_tensor=index_tensor,
                ) or converted

        if has_old and torch.is_tensor(old_b):
            copy_specs = [
                (leg_bias_key, leg_last.bias, idx_leg_use),
            ]
            if arm_split:
                copy_specs.extend([
                    ("direct_pose_out_arm.bias", arm_head.bias, idx_arm_use),
                    ("direct_pose_out_else.bias", else_head.bias, idx_else_use),
                ])
            else:
                copy_specs.append(("direct_pose_out_nonleg.bias", nonleg_head.bias, idx_nonleg_use))
            for target_key, target_tensor, index_tensor in copy_specs:
                converted = self._copy_indexed_tensor_if_needed(
                    state_dict,
                    target_key=target_key,
                    target_tensor=target_tensor,
                    source_tensor=old_b,
                    index_tensor=index_tensor,
                ) or converted

        src_nonleg_w = None
        src_nonleg_b = None
        if has_old:
            try:
                src_nonleg_w = old_w.index_select(0, idx_nonleg_use)
            except Exception:
                src_nonleg_w = None
            if torch.is_tensor(old_b):
                try:
                    src_nonleg_b = old_b.index_select(0, idx_nonleg_use)
                except Exception:
                    src_nonleg_b = None
        if src_nonleg_w is None:
            w_ckpt_nonleg = state_dict.get("direct_pose_out_nonleg.weight", None)
            if torch.is_tensor(w_ckpt_nonleg) and w_ckpt_nonleg.ndim == 2:
                src_nonleg_w = w_ckpt_nonleg
        if src_nonleg_b is None:
            b_ckpt_nonleg = state_dict.get("direct_pose_out_nonleg.bias", None)
            if torch.is_tensor(b_ckpt_nonleg) and b_ckpt_nonleg.ndim == 1:
                src_nonleg_b = b_ckpt_nonleg

        if arm_split:
            for source_tensor, targets in (
                (
                    src_nonleg_w,
                    (
                        ("direct_pose_out_arm.weight", arm_head.weight, arm_nonleg_local),
                        ("direct_pose_out_else.weight", else_head.weight, else_nonleg_local),
                    ),
                ),
                (
                    src_nonleg_b,
                    (
                        ("direct_pose_out_arm.bias", arm_head.bias, arm_nonleg_local),
                        ("direct_pose_out_else.bias", else_head.bias, else_nonleg_local),
                    ),
                ),
            ):
                if not torch.is_tensor(source_tensor):
                    continue
                if int(source_tensor.shape[0]) != int(idx_nonleg_use.numel()):
                    continue
                for target_key, target_tensor, local_idx in targets:
                    converted = self._copy_indexed_tensor_if_needed(
                        state_dict,
                        target_key=target_key,
                        target_tensor=target_tensor,
                        source_tensor=source_tensor,
                        index_tensor=local_idx,
                    ) or converted

        if not arm_split:
            # Optional projection bottleneck for non-leg branch:
            # if checkpoint has legacy non-leg readout W_old: (D_nonleg, hid),
            # factorize it to W_old ~= W_out @ W_proj with rank=proj_dim.
            proj_linear = self._direct_pose_first_linear(split_state["nonleg_proj"])
            if proj_linear is not None:
                tgt_proj_w = proj_linear.weight
                tgt_nonleg_w = nonleg_head.weight
                cur_proj_w = state_dict.get("direct_pose_nonleg_proj.0.weight", None)
                cur_nonleg_w = state_dict.get("direct_pose_out_nonleg.weight", None)
                need_proj = (not torch.is_tensor(cur_proj_w)) or tuple(cur_proj_w.shape) != tuple(tgt_proj_w.shape)
                need_nonleg = (not torch.is_tensor(cur_nonleg_w)) or tuple(cur_nonleg_w.shape) != tuple(tgt_nonleg_w.shape)

                if (
                    (need_proj or need_nonleg)
                    and torch.is_tensor(src_nonleg_w)
                    and src_nonleg_w.ndim == 2
                    and int(src_nonleg_w.shape[0]) == int(tgt_nonleg_w.shape[0])
                    and int(src_nonleg_w.shape[1]) == int(tgt_proj_w.shape[1])
                ):
                    try:
                        src = src_nonleg_w.detach().to(dtype=torch.float32)
                        u, s, vh = torch.linalg.svd(src, full_matrices=False)
                        rank = int(min(int(tgt_proj_w.shape[0]), int(s.numel())))
                        proj_w = torch.zeros(
                            tuple(tgt_proj_w.shape),
                            dtype=src_nonleg_w.dtype,
                            device=src_nonleg_w.device,
                        )
                        out_w = torch.zeros(
                            tuple(tgt_nonleg_w.shape),
                            dtype=src_nonleg_w.dtype,
                            device=src_nonleg_w.device,
                        )
                        if rank > 0:
                            out_w[:, :rank] = (u[:, :rank] * s[:rank].unsqueeze(0)).to(dtype=out_w.dtype, device=out_w.device)
                            proj_w[:rank, :] = vh[:rank, :].to(dtype=proj_w.dtype, device=proj_w.device)
                        state_dict["direct_pose_nonleg_proj.0.weight"] = proj_w
                        state_dict["direct_pose_nonleg_proj.0.bias"] = torch.zeros(
                            (int(tgt_proj_w.shape[0]),),
                            dtype=src_nonleg_w.dtype,
                            device=src_nonleg_w.device,
                        )
                        state_dict["direct_pose_out_nonleg.weight"] = out_w
                        converted = True
                    except Exception:
                        pass
        else:
            # Warm-start arm/else projection layers by copying legacy non-leg projection when available.
            src_proj_w = state_dict.get("direct_pose_nonleg_proj.0.weight", None)
            src_proj_b = state_dict.get("direct_pose_nonleg_proj.0.bias", None)
            for branch, key in (
                (split_state["arm_proj"], "direct_pose_arm_proj"),
                (split_state["else_proj"], "direct_pose_else_proj"),
            ):
                lin = self._direct_pose_first_linear(branch)
                if lin is None:
                    continue
                converted = self._copy_tensor_if_compatible(
                    state_dict,
                    target_key=f"{key}.0.weight",
                    target_tensor=lin.weight,
                    source_tensor=src_proj_w if torch.is_tensor(src_proj_w) and src_proj_w.ndim == 2 else None,
                ) or converted
                converted = self._copy_tensor_if_compatible(
                    state_dict,
                    target_key=f"{key}.0.bias",
                    target_tensor=lin.bias,
                    source_tensor=src_proj_b if torch.is_tensor(src_proj_b) and src_proj_b.ndim == 1 else None,
                ) or converted
            # Remove stale two-way branch tensors when loading into three-way model.
            for k in (
                "direct_pose_out_nonleg.weight",
                "direct_pose_out_nonleg.bias",
                "direct_pose_nonleg_proj.0.weight",
                "direct_pose_nonleg_proj.0.bias",
            ):
                if k in state_dict:
                    state_dict.pop(k, None)
                    converted = True

        if converted:
            state_dict.pop("direct_pose_head.6.weight", None)
            state_dict.pop("direct_pose_head.6.bias", None)
        return converted

    def _maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict(self, state_dict: Dict[str, Any]) -> bool:
        if (not isinstance(state_dict, dict)) or getattr(self, "direct_pose_leg_terminal", None) is None:
            return False
        if not any(str(key).startswith("direct_pose_out_leg.") for key in state_dict.keys()):
            return False

        model_sd = self.state_dict()
        converted = False
        for key in (
            "direct_pose_leg_terminal.0.weight",
            "direct_pose_leg_terminal.0.bias",
            "direct_pose_leg_terminal.3.weight",
            "direct_pose_leg_terminal.3.bias",
        ):
            target_tensor = model_sd.get(key, None)
            converted = self._copy_tensor_if_compatible(
                state_dict,
                target_key=key,
                target_tensor=target_tensor,
                source_tensor=target_tensor,
            ) or converted

        converted = self._copy_tensor_if_compatible(
            state_dict,
            target_key="direct_pose_leg_terminal.6.weight",
            target_tensor=model_sd.get("direct_pose_leg_terminal.6.weight", None),
            source_tensor=state_dict.get("direct_pose_out_leg.weight", None),
        ) or converted
        converted = self._copy_tensor_if_compatible(
            state_dict,
            target_key="direct_pose_leg_terminal.6.bias",
            target_tensor=model_sd.get("direct_pose_leg_terminal.6.bias", None),
            source_tensor=state_dict.get("direct_pose_out_leg.bias", None),
        ) or converted

        removed_legacy = False
        for key in list(state_dict.keys()):
            if str(key).startswith("direct_pose_out_leg."):
                state_dict.pop(key, None)
                removed_legacy = True
        return bool(converted or removed_legacy)

    def adapt_legacy_state_dict_(self, state_dict: Dict[str, Any]) -> bool:
        try:
            converted_stepc = bool(self._maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict(state_dict))
            converted_split = bool(self._maybe_upgrade_direct_pose_split_state_dict(state_dict))
            return bool(converted_stepc or converted_split)
        except Exception:
            return False

    def load_state_dict(self, state_dict, strict: bool = True):
        if isinstance(state_dict, dict):
            try:
                self._maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict(state_dict)
            except Exception:
                pass
            try:
                self._maybe_upgrade_direct_pose_split_state_dict(state_dict)
            except Exception:
                pass
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
        rot_sl = parse_layout_entry(output_layout.get('BoneRotations6D'), 'BoneRotations6D', self.out_motion_dim)
        if rot_sl is None:
            rot_sl = slice(0, min(self.out_motion_dim, int(len(bone_names) * 6)))
        if rot_sl.start is None or rot_sl.stop is None:
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

        device = state.device
        dtype = state.dtype
        B, Tq, _ = state.shape
        contacts_input = contacts
        contacts_enc = contacts_input
        def _expand_state_sequence(
            value: Optional[torch.Tensor],
            feat_dim: int,
        ) -> Optional[torch.Tensor]:
            if value is None or feat_dim <= 0:
                return None
            try:
                seq = value.to(device=device, dtype=dtype)
                if seq.ndim == 1:
                    seq = seq.view(1, 1, -1)
                elif seq.ndim == 2:
                    seq = seq.unsqueeze(1)
                elif seq.ndim != 3:
                    seq = seq.reshape(B, Tq, -1)
                if seq.shape[0] == 1 and B > 1:
                    seq = seq.expand(B, -1, -1)
                if seq.shape[1] == 1 and Tq > 1:
                    seq = seq.expand(-1, Tq, -1)
                if seq.shape[-1] > feat_dim:
                    seq = seq[..., :feat_dim]
                elif seq.shape[-1] < feat_dim:
                    seq = F.pad(seq, (0, feat_dim - seq.shape[-1]))
                return seq.contiguous()
            except Exception:
                return None
        if angvel is None and self.angvel_dim > 0:
            angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
        if pose_history is None and self.pose_hist_dim > 0:
            pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

        # Pre-computed signals for Event-Clock (v3). Populated only when enabled.
        soft_period: Optional[torch.Tensor] = None
        contacts_meas: Optional[torch.Tensor] = None
        event_clock_delta_meas: Optional[torch.Tensor] = None
        event_clock_lr_diff: Optional[torch.Tensor] = None
        event_clock_lambda_corr: Optional[torch.Tensor] = None
        event_clock_lambda_logit: Optional[torch.Tensor] = None
        event_clock_dynamic_prior: Optional[torch.Tensor] = None
        event_clock_delta_z: Optional[torch.Tensor] = None
        _pose_hist_processed = False

        # ---- Contact plan (independent anchor) ----
        # - contacts_plan is produced from cond history via a GRUCell and stays independent of pose.
        # - plan_z is the only cached state needed at inference.
        contacts_plan = None
        plan_z_next = None
        plan_feat_for_inject = None
        contacts_plan_logits = None
        contacts_plan_logits_base = None
        contacts_plan_logits_time = None
        contacts_plan_logits_phase = None
        contacts_plan_logits_raw = None
        time_pe_direct = None
        phase_z_in_direct = None  # (B,Tq,2*C) when direct_pose_use_phase_z=True
        leg_side_cue_in = None  # (B,Tq,C) per-step side cue (raw, before per-side select)
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
                        except Exception:
                            plan_z_t = None

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
                            else:
                                base = t_in.view(B, 1)
                            t_grid = base + torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0)
                        elif t_in.dim() == 2:
                            # Either (B,Tq) or broadcastable; treat it as explicit time per step.
                            if t_in.shape[0] == 1 and B > 1:
                                t_in = t_in.expand(B, -1)
                            t_grid = t_in[:, :Tq]
                        else:
                            t_grid = torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0).expand(B, Tq)
                    else:
                        t_grid = torch.arange(Tq, device=device, dtype=dtype).unsqueeze(0).expand(B, Tq)
                except Exception:
                    t_grid = None

            time_pe = None
            if self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0 and t_grid is not None:
                try:
                    pe_dim = int(self.contact_plan_time_pe_dim)
                    half = pe_dim // 2
                    idx = torch.arange(0, pe_dim, 2, device=device, dtype=dtype)
                    base = float(getattr(self, "_contact_plan_time_pe_base", 10000.0) or 10000.0)
                    freqs = 1.0 / torch.pow(torch.full((half,), base, device=device, dtype=dtype), idx / float(pe_dim))
                    angles = t_grid.unsqueeze(-1) * freqs.view(1, 1, half)
                    time_pe = torch.zeros((B, Tq, pe_dim), device=device, dtype=dtype)
                    time_pe[..., 0::2] = torch.sin(angles)
                    time_pe[..., 1::2] = torch.cos(angles)
                except Exception:
                    time_pe = None

            if int(getattr(self, "direct_pose_time_pe_dim", 0) or 0) > 0 and t_grid is not None:
                try:
                    pe_dim = int(getattr(self, "direct_pose_time_pe_dim", 0) or 0)
                    half = pe_dim // 2
                    idx = torch.arange(0, pe_dim, 2, device=device, dtype=dtype)
                    base = float(getattr(self, "_direct_pose_time_pe_base", 10000.0) or 10000.0)
                    freqs = 1.0 / torch.pow(torch.full((half,), base, device=device, dtype=dtype), idx / float(pe_dim))
                    angles = t_grid.unsqueeze(-1) * freqs.view(1, 1, half)
                    time_pe_direct = torch.zeros((B, Tq, pe_dim), device=device, dtype=dtype)
                    time_pe_direct[..., 0::2] = torch.sin(angles)
                    time_pe_direct[..., 1::2] = torch.cos(angles)
                except Exception:
                    time_pe_direct = None

            plan_probs: list[torch.Tensor] = []
            plan_logits: list[torch.Tensor] = []
            # Optional debug: decompose contacts_plan logits into:
            #   raw = head(plan_z_raw)               (pre Event-Clock correction; ==base when Event-Clock is off)
            #   base = head(plan_z_t)                (post correction, pre time-PE)
            #   phase = phase/TTA residual term      (optional; added directly on logits)
            #   time = time term added to logits     (scaled by lambda_corr when Event-Clock is on)
            debug_plan_logit_decomp = bool(getattr(self, "debug_contact_plan_logits_decomp", False))
            plan_logits_raw_seq: Optional[list[torch.Tensor]] = [] if debug_plan_logit_decomp else None
            plan_logits_base_seq: Optional[list[torch.Tensor]] = [] if debug_plan_logit_decomp else None
            plan_logits_phase_seq: Optional[list[torch.Tensor]] = [] if debug_plan_logit_decomp else None
            plan_logits_time_seq: Optional[list[torch.Tensor]] = [] if debug_plan_logit_decomp else None
            try:
                time_bias_scale = float(getattr(self, "contact_plan_time_bias_scale", 1.0))
            except Exception:
                time_bias_scale = 1.0
            plan_z_seq: Optional[list[torch.Tensor]] = [] if self.contact_plan_inject == "plan_z" else None

            phase_input_seq = _expand_state_sequence(
                phase_z,
                int(getattr(self, "_direct_pose_phase_dim", 0) or 0),
            )
            phase_age_seq = _expand_state_sequence(
                phase_event_age,
                int(self.contact_dim),
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
                    except Exception:
                        pass

                # contacts_meas: external override only.
                if contacts_input is not None:
                    try:
                        meas = contacts_input.to(device=device, dtype=dtype)
                        if meas.ndim == 1:
                            meas = meas.view(1, 1, -1)
                        elif meas.ndim == 2:
                            meas = meas.unsqueeze(1)
                        if meas.ndim == 3:
                            if meas.shape[0] == 1 and B > 1:
                                meas = meas.expand(B, -1, -1)
                            if meas.shape[1] == 1 and Tq > 1:
                                meas = meas.expand(-1, Tq, -1)
                            if meas.shape[-1] != int(self.contact_dim):
                                if meas.shape[-1] > int(self.contact_dim):
                                    meas = meas[..., : int(self.contact_dim)]
                                else:
                                    meas = F.pad(meas, (0, int(self.contact_dim) - meas.shape[-1]))
                            contacts_meas = meas
                    except Exception:
                        contacts_meas = None

                if contacts_meas is None:
                    contacts_meas = torch.zeros((B, Tq, int(self.contact_dim)), device=device, dtype=dtype)

                # Canonicalize meas_prev to (B, C) (logits if available else probs).
                meas_prev_t = None
                if torch.is_tensor(meas_logits_prev):
                    try:
                        prev = meas_logits_prev.to(device=device, dtype=dtype)
                        if prev.ndim == 3 and prev.size(1) == 1:
                            prev = prev[:, 0]
                        if prev.ndim == 2:
                            pass
                        elif prev.ndim == 1:
                            prev = prev.view(1, -1)
                        else:
                            prev = prev.reshape(B, -1)
                        if prev.shape[0] == 1 and B > 1:
                            prev = prev.expand(B, -1)
                        if prev.shape[-1] != int(self.contact_dim):
                            if prev.shape[-1] > int(self.contact_dim):
                                prev = prev[..., : int(self.contact_dim)]
                            else:
                                prev = F.pad(prev, (0, int(self.contact_dim) - prev.shape[-1]))
                        meas_prev_t = prev
                    except Exception:
                        meas_prev_t = None

                delta_meas = torch.zeros_like(contacts_meas)
                if Tq > 1:
                    delta_meas[:, 1:] = contacts_meas[:, 1:] - contacts_meas[:, :-1]
                if meas_prev_t is not None and Tq > 0:
                    delta_meas[:, 0] = contacts_meas[:, 0] - meas_prev_t

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
                    except Exception:
                        period_feat = None

                # ---- Layer2+3: gated residual correction inside GRU loop ----
                lambda_corr_seq: list[torch.Tensor] = []
                lambda_logit_seq: list[torch.Tensor] = []
                dyn_prior_seq: list[torch.Tensor] = []
                delta_z_seq: list[torch.Tensor] = []
                for _t in range(Tq):
                    if phase_in_direct_seq is not None:
                        try:
                            phase_step = phase_in_direct_zero if phase_input_seq is None else phase_input_seq[:, _t]
                            phase_in_direct_seq.append(phase_step)
                        except Exception:
                            pass
                    if leg_side_cue_seq is not None and leg_side_cue_zero is not None:
                        try:
                            if leg_side_cue_mode == "phase_event_age":
                                cue_step = leg_side_cue_zero if phase_age_seq is None else phase_age_seq[:, _t]
                                leg_side_cue_seq.append(cue_step)
                            else:
                                leg_side_cue_seq.append(leg_side_cue_zero)
                        except Exception:
                            pass
                    plan_in_t = cond_seq[:, _t]
                    plan_z_raw = self.contact_plan_cell(plan_in_t, plan_z_t)

                    logits_raw = self.contact_plan_head(plan_z_raw)
                    if plan_logits_raw_seq is not None:
                        plan_logits_raw_seq.append(logits_raw)
                    plan_raw = torch.sigmoid(logits_raw)
                    meas_t = contacts_meas_obs[:, _t]
                    err_raw = plan_raw - meas_t

                    delta_meas_t = delta_meas_obs[:, _t]
                    lr_diff_t = lr_diff_obs[:, _t]
                    period_t = period_feat[:, _t] if (period_feat is not None and period_feat.ndim == 3) else None

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
                    if plan_logits_phase_seq is not None:
                        plan_logits_phase_seq.append(logits_base.new_zeros(logits_base.shape))
                    time_term = None
                    if time_pe is not None and self.contact_plan_time_head is not None:
                        try:
                            time_bias = self.contact_plan_time_head(time_pe[:, _t])
                            time_term = lam_corr_t * (time_bias * time_bias_scale)
                        except Exception:
                            time_term = None
                    logits = logits_base
                    if time_term is None:
                        if plan_logits_time_seq is not None:
                            plan_logits_time_seq.append(logits_base.new_zeros(logits_base.shape))
                    else:
                        logits = logits + time_term
                        if plan_logits_time_seq is not None:
                            plan_logits_time_seq.append(time_term)
                    if plan_logits_base_seq is not None:
                        plan_logits_base_seq.append(logits_base)
                    plan_logits.append(logits)
                    plan_probs.append(torch.sigmoid(logits))

                    lambda_corr_seq.append(lam_corr_t)
                    lambda_logit_seq.append(lam_logit_t)
                    dyn_prior_seq.append(dyn_prior_t)
                    delta_z_seq.append(delta_z_t)

                if lambda_corr_seq:
                    event_clock_lambda_corr = torch.stack(lambda_corr_seq, dim=1)
                    event_clock_lambda_logit = torch.stack(lambda_logit_seq, dim=1)
                    event_clock_dynamic_prior = torch.stack(dyn_prior_seq, dim=1)
                    event_clock_delta_z = torch.stack(delta_z_seq, dim=1)
            else:
                # Event-Clock off: still resolve measurement hints for direct/contact diagnostics.
                Cc = int(self.contact_dim)
                if contacts_input is not None:
                    try:
                        meas = contacts_input.to(device=device, dtype=dtype)
                        if meas.ndim == 1:
                            meas = meas.view(1, 1, -1)
                        elif meas.ndim == 2:
                            meas = meas.unsqueeze(1)
                        if meas.ndim == 3:
                            if meas.shape[0] == 1 and B > 1:
                                meas = meas.expand(B, -1, -1)
                            if meas.shape[1] == 1 and Tq > 1:
                                meas = meas.expand(-1, Tq, -1)
                            if meas.shape[-1] != Cc:
                                if meas.shape[-1] > Cc:
                                    meas = meas[..., :Cc]
                                else:
                                    meas = F.pad(meas, (0, Cc - meas.shape[-1]))
                            contacts_meas = meas
                    except Exception:
                        contacts_meas = None

                if contacts_meas is None:
                    contacts_meas = torch.zeros((B, Tq, Cc), device=device, dtype=dtype)

                meas_prev_t = None
                if torch.is_tensor(meas_logits_prev):
                    try:
                        prev = meas_logits_prev.to(device=device, dtype=dtype)
                        if prev.ndim == 3 and prev.size(1) == 1:
                            prev = prev[:, 0]
                        if prev.ndim == 2:
                            pass
                        elif prev.ndim == 1:
                            prev = prev.view(1, -1)
                        else:
                            prev = prev.reshape(B, -1)
                        if prev.shape[0] == 1 and B > 1:
                            prev = prev.expand(B, -1)
                        if prev.shape[-1] != Cc:
                            if prev.shape[-1] > Cc:
                                prev = prev[..., :Cc]
                            else:
                                prev = F.pad(prev, (0, Cc - prev.shape[-1]))
                        meas_prev_t = prev
                    except Exception:
                        meas_prev_t = None

                delta_meas = torch.zeros_like(contacts_meas)
                if Tq > 1:
                    delta_meas[:, 1:] = contacts_meas[:, 1:] - contacts_meas[:, :-1]
                if meas_prev_t is not None and Tq > 0:
                    delta_meas[:, 0] = contacts_meas[:, 0] - meas_prev_t

                contacts_meas_obs = contacts_meas.detach()
                delta_meas_obs = delta_meas.detach()

                for _t in range(Tq):
                    if phase_in_direct_seq is not None:
                        try:
                            phase_step = phase_in_direct_zero if phase_input_seq is None else phase_input_seq[:, _t]
                            phase_in_direct_seq.append(phase_step)
                        except Exception:
                            pass
                    if leg_side_cue_seq is not None and leg_side_cue_zero is not None:
                        try:
                            if leg_side_cue_mode == "phase_event_age":
                                cue_step = leg_side_cue_zero if phase_age_seq is None else phase_age_seq[:, _t]
                                leg_side_cue_seq.append(cue_step)
                            else:
                                leg_side_cue_seq.append(leg_side_cue_zero)
                        except Exception:
                            pass
                    plan_in_t = cond_seq[:, _t]
                    plan_z_t = self.contact_plan_cell(plan_in_t, plan_z_t)
                    if plan_z_seq is not None:
                        plan_z_seq.append(plan_z_t)
                    logits_base = self.contact_plan_head(plan_z_t)
                    if plan_logits_raw_seq is not None:
                        plan_logits_raw_seq.append(logits_base)
                    if plan_logits_phase_seq is not None:
                        plan_logits_phase_seq.append(logits_base.new_zeros(logits_base.shape))
                    time_term = None
                    if time_pe is not None and self.contact_plan_time_head is not None:
                        try:
                            time_term = self.contact_plan_time_head(time_pe[:, _t]) * time_bias_scale
                        except Exception:
                            time_term = None
                    logits = logits_base
                    if time_term is None:
                        if plan_logits_time_seq is not None:
                            plan_logits_time_seq.append(logits_base.new_zeros(logits_base.shape))
                    else:
                        logits = logits + time_term
                        if plan_logits_time_seq is not None:
                            plan_logits_time_seq.append(time_term)
                    if plan_logits_base_seq is not None:
                        plan_logits_base_seq.append(logits_base)
                    plan_logits.append(logits)
                    plan_probs.append(torch.sigmoid(logits))
            contacts_plan = torch.stack(plan_probs, dim=1)  # (B,T,C)
            if phase_in_direct_seq is not None:
                try:
                    phase_z_in_direct = torch.stack(phase_in_direct_seq, dim=1)  # (B,Tq,2*C)
                except Exception:
                    phase_z_in_direct = None
            if leg_side_cue_seq is not None:
                try:
                    leg_side_cue_in = torch.stack(leg_side_cue_seq, dim=1)  # (B,Tq,C)
                except Exception:
                    leg_side_cue_in = None
            try:
                contacts_plan_logits = torch.stack(plan_logits, dim=1) if plan_logits else None  # (B,T,logits_dim)
            except Exception:
                contacts_plan_logits = None
            if plan_logits_base_seq is not None:
                try:
                    contacts_plan_logits_base = (
                        torch.stack(plan_logits_base_seq, dim=1) if plan_logits_base_seq else None
                    )  # (B,T,logits_dim)
                except Exception:
                    contacts_plan_logits_base = None
            if plan_logits_phase_seq is not None:
                try:
                    contacts_plan_logits_phase = (
                        torch.stack(plan_logits_phase_seq, dim=1) if plan_logits_phase_seq else None
                    )  # (B,T,logits_dim)
                except Exception:
                    contacts_plan_logits_phase = None
            if plan_logits_time_seq is not None:
                try:
                    contacts_plan_logits_time = (
                        torch.stack(plan_logits_time_seq, dim=1) if plan_logits_time_seq else None
                    )  # (B,T,logits_dim)
                except Exception:
                    contacts_plan_logits_time = None
            if plan_logits_raw_seq is not None:
                try:
                    contacts_plan_logits_raw = (
                        torch.stack(plan_logits_raw_seq, dim=1) if plan_logits_raw_seq else None
                    )  # (B,T,logits_dim)
                except Exception:
                    contacts_plan_logits_raw = None
            plan_z_next = plan_z_t
            if self.contact_plan_inject == "contacts":
                plan_feat_for_inject = contacts_plan
            elif self.contact_plan_inject == "plan_z" and plan_z_seq is not None:
                plan_feat_for_inject = torch.stack(plan_z_seq, dim=1)  # (B,T,H)

        # Frozen-encoder input for the period hint:
        # - Event-Clock v3 prefers an *independent* meas-derived contact signal (avoids co-drift with plan).
        # - Otherwise, keep the legacy behavior (prefer plan for train/infer consistency).
        if self.use_event_clock and contacts_meas is not None:
            contacts_enc = contacts_meas.detach()
        elif contacts_plan is not None:
            contacts_enc = contacts_plan.detach()
        if contacts_enc is None and self.contact_dim > 0:
            contacts_enc = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)

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
            try:
                inject_scale = float(getattr(self, "contact_plan_inject_scale", 1.0))
            except Exception:
                inject_scale = 1.0
            if inject_scale != 1.0:
                feat = feat * inject_scale
            x_inputs.append(feat)
        x = torch.cat(x_inputs, dim=-1)
        # 导出/编译时跳过数据依赖的 guard，避免 torch.export 的 GuardOnDataDependentSymNode
        _skip_guard = False
        try:
            _skip_guard = torch._dynamo.is_compiling()
        except Exception:
            pass
        try:
            _skip_guard = _skip_guard or torch.onnx.is_in_onnx_export()
        except Exception:
            pass
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
        setattr(self, '_last_hidden_seq', hidden_out.detach())
        result = {
            'out': out,
            'delta': out,
            'attn': attn.mean(dim=1),
        }
        result['h_final'] = hidden_out

        # ---- Contact meas (pose-derived) + error signal ----
        e_t = None
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

        if contacts_meas is not None:
            result['contacts_meas'] = contacts_meas.squeeze(1) if is_single else contacts_meas

        if contacts_plan is not None:
            result['contacts_plan'] = contacts_plan.squeeze(1) if is_single else contacts_plan
            if contacts_plan_logits is not None and torch.is_tensor(contacts_plan_logits):
                result['contacts_plan_logits'] = contacts_plan_logits.squeeze(1) if is_single else contacts_plan_logits
                if contacts_plan_logits_base is not None and torch.is_tensor(contacts_plan_logits_base):
                    result['contacts_plan_logits_base'] = (
                        contacts_plan_logits_base.squeeze(1) if is_single else contacts_plan_logits_base
                    )
                if contacts_plan_logits_phase is not None and torch.is_tensor(contacts_plan_logits_phase):
                    result['contacts_plan_logits_phase'] = (
                        contacts_plan_logits_phase.squeeze(1) if is_single else contacts_plan_logits_phase
                    )
                if contacts_plan_logits_time is not None and torch.is_tensor(contacts_plan_logits_time):
                    result['contacts_plan_logits_time'] = (
                        contacts_plan_logits_time.squeeze(1) if is_single else contacts_plan_logits_time
                    )
            if contacts_plan_logits_raw is not None and torch.is_tensor(contacts_plan_logits_raw):
                result['contacts_plan_logits_raw'] = (
                    contacts_plan_logits_raw.squeeze(1) if is_single else contacts_plan_logits_raw
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
            if contacts_meas is not None:
                # Ensure meas shape aligns with (B,T,C)
                if contacts_meas.ndim == 2:
                    contacts_meas = contacts_meas.unsqueeze(1)
                e_t = contacts_plan - contacts_meas.to(device=device, dtype=dtype)
                result['contacts_err'] = e_t.squeeze(1) if is_single else e_t

        # ---- Direct pose head (bridge: add phase-hint contacts_meas) ----
        if self.direct_pose_head is not None and contacts_plan is not None:
            try:
                plan_in = contacts_plan.detach() if self.direct_pose_detach_plan else contacts_plan
                if self.training and float(getattr(self, "direct_pose_plan_drop_prob", 0.0) or 0.0) > 0.0:
                    p = float(getattr(self, "direct_pose_plan_drop_prob", 0.0) or 0.0)
                    p = max(0.0, min(1.0, p))
                    if p > 0.0:
                        m = (torch.rand(plan_in.shape[:-1] + (1,), device=plan_in.device) < p).to(plan_in.dtype)
                        plan_in = plan_in * (1.0 - m)
                # Optional per-call override (debug): force a specific plan source for *direct* only.
                # - tensor: used as plan_in (after clamp); supports (B,C) / (B,T,C) / (C,)
                # - "ignore"/"zero": replace with zeros (keeps shape)
                try:
                    override = getattr(self, "direct_pose_plan_override", None)
                    if isinstance(override, str):
                        s = override.strip().lower()
                        if s in ("ignore", "zero", "none", "null"):
                            plan_in = torch.zeros_like(plan_in)
                            override = None
                    if torch.is_tensor(override):
                        ov = override.detach()
                        # Canonicalize to (B,T,C)
                        if ov.ndim == 1:
                            ov = ov.view(1, 1, -1)
                        elif ov.ndim == 2:
                            ov = ov.unsqueeze(1)
                        if ov.ndim == 3:
                            if ov.shape[0] == 1 and B > 1:
                                ov = ov.expand(B, -1, -1)
                            if ov.shape[1] == 1 and Tq > 1:
                                ov = ov.expand(-1, Tq, -1)
                            # Match contact dim (pad/trim) to avoid concat shape mismatch.
                            target_c = int(plan_in.shape[-1]) if torch.is_tensor(plan_in) else int(self.contact_dim)
                            if target_c > 0 and ov.shape[-1] != target_c:
                                if ov.shape[-1] > target_c:
                                    ov = ov[..., :target_c]
                                else:
                                    ov = F.pad(ov, (0, target_c - ov.shape[-1]))
                            plan_in = ov.to(device=device, dtype=dtype).clamp(0.0, 1.0)
                except Exception:
                    pass

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
                    # Optional per-call override (debug): force a specific meas hint source for *direct* only.
                    # - tensor: used as meas_in (after clamp); supports (B,C) / (B,T,C) / (C,)
                    # - "ignore"/"zero": treat as missing (concat->zeros, mode_select->uniform)
                    try:
                        override = getattr(self, "direct_pose_meas_override", None)
                        if isinstance(override, str):
                            s = override.strip().lower()
                            if s in ("ignore", "zero", "none", "null"):
                                if mode == "concat":
                                    meas_in = torch.zeros_like(plan_in)
                                elif mode == "mode_select":
                                    meas_in = None
                            else:
                                override = None
                        if torch.is_tensor(override):
                            ov = override
                            # Canonicalize to (B,T,C)
                            if ov.ndim == 1:
                                ov = ov.view(1, 1, -1)
                            elif ov.ndim == 2:
                                ov = ov.unsqueeze(1)
                            if ov.ndim == 3:
                                if ov.shape[0] == 1 and B > 1:
                                    ov = ov.expand(B, -1, -1)
                                if ov.shape[1] == 1 and Tq > 1:
                                    ov = ov.expand(-1, Tq, -1)
                                # Match contact dim (pad/trim) to avoid concat shape mismatch.
                                target_c = int(plan_in.shape[-1]) if torch.is_tensor(plan_in) else int(self.contact_dim)
                                if target_c > 0 and ov.shape[-1] != target_c:
                                    if ov.shape[-1] > target_c:
                                        ov = ov[..., :target_c]
                                    else:
                                        ov = F.pad(ov, (0, target_c - ov.shape[-1]))
                                meas_in = ov.to(device=device, dtype=dtype).clamp(0.0, 1.0)
                    except Exception:
                        pass

                # Choose which features feed the direct head.
                direct_feat = cond
                try:
                    src = str(getattr(self, "direct_pose_feat_source", "cond") or "cond").lower().strip()
                except Exception:
                    src = "cond"
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
                    try:
                        direct_feat = torch.cat([direct_feat, time_pe_direct.to(device=device, dtype=dtype)], dim=-1)
                    except Exception:
                        pass
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
                    # New path: explicit per-side routing + shared weights.
                    # We predict omega for each side separately, then scatter into the original K ordering.
                    try:
                        idx = getattr(self, "direct_pose_leg_joint_idx_tensor", None)
                        rot_sl = getattr(self, "direct_pose_leg_rot6d_slice", None)
                        pos_r = getattr(self, "direct_pose_leg_side_pos_r_tensor", None)
                        pos_l = getattr(self, "direct_pose_leg_side_pos_l_tensor", None)
                        if (
                            torch.is_tensor(idx)
                            and torch.is_tensor(pos_r)
                            and torch.is_tensor(pos_l)
                            and isinstance(rot_sl, slice)
                            and rot_sl.start is not None
                            and rot_sl.stop is not None
                        ):
                            K = int(idx.numel())
                            K_side = int(pos_r.numel())
                            if K > 0 and K_side > 0 and int(pos_l.numel()) == K_side:
                                rot_dim = int(rot_sl.stop - rot_sl.start)
                                if rot_dim > 0 and (rot_dim % 6) == 0:
                                    J = int(rot_dim // 6)
                                    idx_use = idx.to(device=device)
                                    if bool(torch.all((idx_use >= 0) & (idx_use < J)).detach().cpu().item()):
                                        # Canonicalize plan/meas to (B,T,C).
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

                                        # Per-side channel mapping (default: contacts=[L,R]).
                                        ch_r = int(getattr(self, "direct_pose_leg_contact_ch_r", 1) or 0)
                                        ch_l = int(getattr(self, "direct_pose_leg_contact_ch_l", 0) or 0)
                                        Cc = int(plan_bt.shape[-1])
                                        if Cc > 0:
                                            ch_r = max(0, min(int(Cc - 1), ch_r))
                                            ch_l = max(0, min(int(Cc - 1), ch_l))

                                        plan_r = plan_bt[..., ch_r : ch_r + 1]
                                        plan_l = plan_bt[..., ch_l : ch_l + 1]

                                        # Optional meas scalar per side (zeros if missing).
                                        if meas_bt is None or int(meas_bt.shape[-1]) <= 0:
                                            meas_r = plan_r.new_zeros(plan_r.shape)
                                            meas_l = plan_l.new_zeros(plan_l.shape)
                                        else:
                                            meas_r = meas_bt[..., ch_r : ch_r + 1]
                                            meas_l = meas_bt[..., ch_l : ch_l + 1]

                                        # Optional phase sin/cos per side.
                                        if torch.is_tensor(phase_in_direct) and bool(getattr(self, "direct_pose_use_phase_z", False)):
                                            phase_bt = phase_in_direct
                                            if phase_bt.ndim == 2:
                                                phase_bt = phase_bt.unsqueeze(1)
                                            if phase_bt.ndim == 3 and phase_bt.shape[1] == 1 and Tq > 1:
                                                phase_bt = phase_bt.expand(-1, Tq, -1)
                                            try:
                                                phase_view = phase_bt.view(B, Tq, Cc, 2)
                                                phase_r = phase_view[..., ch_r, :].to(device=device, dtype=dtype)
                                                phase_l = phase_view[..., ch_l, :].to(device=device, dtype=dtype)
                                            except Exception:
                                                phase_r = plan_r.new_zeros((B, Tq, 2))
                                                phase_l = plan_l.new_zeros((B, Tq, 2))
                                        else:
                                            phase_r = plan_r.new_zeros((B, Tq, 0))
                                            phase_l = plan_l.new_zeros((B, Tq, 0))

                                        # Optional: cross-leg context via plan_other scalar.
                                        plan_other_r = plan_r.new_zeros((B, Tq, 0))
                                        plan_other_l = plan_l.new_zeros((B, Tq, 0))
                                        if bool(getattr(self, "direct_pose_leg_side_plan_other", False)):
                                            plan_other_r = plan_l
                                            plan_other_l = plan_r
                                            # Runtime ablation (eval-only): drop or decorrelate cross-leg plan_other.
                                            # This is useful to diagnose whether direction prediction relies on cross-leg context.
                                            try:
                                                ab = str(
                                                    getattr(self, "direct_pose_leg_side_plan_other_ablate", "none") or "none"
                                                ).strip().lower()
                                            except Exception:
                                                ab = "none"
                                            if ab not in ("", "none", "off", "disable", "disabled"):
                                                if ab in ("0", "zero", "zeros"):
                                                    plan_other_r = plan_other_r * 0.0
                                                    plan_other_l = plan_other_l * 0.0
                                                elif ab in ("roll", "roll_batch", "shift", "shift_batch"):
                                                    # Deterministic shuffle: roll along batch dimension.
                                                    if int(B) > 1:
                                                        plan_other_r = plan_other_r.roll(shifts=1, dims=0)
                                                        plan_other_l = plan_other_l.roll(shifts=1, dims=0)
                                                elif ab in ("roll_time", "shift_time"):
                                                    # Roll along time (only meaningful when Tq>1).
                                                    if int(Tq) > 1:
                                                        plan_other_r = plan_other_r.roll(shifts=1, dims=1)
                                                        plan_other_l = plan_other_l.roll(shifts=1, dims=1)

                                        # Optional: cross-leg phase context (sin/cos) and explicit relative phase.
                                        phase_other_r = plan_r.new_zeros((B, Tq, 0))
                                        phase_other_l = plan_l.new_zeros((B, Tq, 0))
                                        phase_rel_r = plan_r.new_zeros((B, Tq, 0))
                                        phase_rel_l = plan_l.new_zeros((B, Tq, 0))
                                        if bool(getattr(self, "direct_pose_leg_side_phase_other", False)) or bool(
                                            getattr(self, "direct_pose_leg_side_phase_rel", False)
                                        ):
                                            # Ensure stable feature dims even if phase parsing fails.
                                            if phase_r.shape[-1] != 2 or phase_l.shape[-1] != 2:
                                                phase_r = plan_r.new_zeros((B, Tq, 2))
                                                phase_l = plan_l.new_zeros((B, Tq, 2))

                                            if bool(getattr(self, "direct_pose_leg_side_phase_other", False)):
                                                phase_other_r = phase_l
                                                phase_other_l = phase_r

                                            if bool(getattr(self, "direct_pose_leg_side_phase_rel", False)):
                                                # Given sin/cos pairs, compute:
                                                #   sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
                                                #   cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
                                                sin_r = phase_r[..., 0:1]
                                                cos_r = phase_r[..., 1:2]
                                                sin_l = phase_l[..., 0:1]
                                                cos_l = phase_l[..., 1:2]
                                                # Right: other-self = L - R
                                                sin_rel_r = sin_l * cos_r - cos_l * sin_r
                                                cos_rel_r = cos_l * cos_r + sin_l * sin_r
                                                # Left: other-self = R - L
                                                sin_rel_l = sin_r * cos_l - cos_r * sin_l
                                                cos_rel_l = cos_r * cos_l + sin_r * sin_l
                                                phase_rel_r = torch.cat([sin_rel_r, cos_rel_r], dim=-1)
                                                phase_rel_l = torch.cat([sin_rel_l, cos_rel_l], dim=-1)

                                        # Optional extra stateful cue per side (1 scalar).
                                        cue_r = plan_r.new_zeros((B, Tq, 0))
                                        cue_l = plan_l.new_zeros((B, Tq, 0))
                                        if int(getattr(self, "direct_pose_leg_side_cue_dim", 0) or 0) > 0:
                                            cue_bt = leg_side_cue_in
                                            if cue_bt is None:
                                                cue_bt = plan_bt.new_zeros((B, Tq, int(self.contact_dim)))
                                            elif cue_bt.ndim == 2:
                                                cue_bt = cue_bt.unsqueeze(1)
                                            if cue_bt.ndim == 3 and cue_bt.shape[1] == 1 and Tq > 1:
                                                cue_bt = cue_bt.expand(-1, Tq, -1)
                                            if cue_bt.ndim == 3 and cue_bt.shape[-1] >= int(self.contact_dim):
                                                cue_r = cue_bt[..., ch_r : ch_r + 1]
                                                cue_l = cue_bt[..., ch_l : ch_l + 1]
                                                cm = str(getattr(self, "direct_pose_leg_side_cue", "none") or "none").strip().lower()
                                                if cm == "phase_event_age":
                                                    tau = float(getattr(self, "direct_pose_leg_side_cue_tau", 30.0) or 30.0)
                                                    if (not _math.isfinite(tau)) or tau <= 1e-6:
                                                        tau = 30.0
                                                    cue_r = (cue_r / tau).clamp(0.0, 1.0)
                                                    cue_l = (cue_l / tau).clamp(0.0, 1.0)
                                                else:
                                                    cue_r = cue_r.clamp(0.0, 1.0)
                                                    cue_l = cue_l.clamp(0.0, 1.0)

                                        # Optional side embedding (tiny adapter).
                                        emb_r = None
                                        emb_l = None
                                        if getattr(self, "direct_pose_leg_side_embed", None) is not None:
                                            try:
                                                # Convention: id=0 => right, id=1 => left.
                                                emb_w = self.direct_pose_leg_side_embed.weight  # type: ignore[union-attr]
                                                emb_r = emb_w.new_zeros((1,), dtype=torch.long)
                                                emb_l = emb_w.new_ones((1,), dtype=torch.long)
                                                emb_r = self.direct_pose_leg_side_embed(emb_r.to(device=device)).view(1, 1, -1).expand(B, Tq, -1)  # type: ignore[operator]
                                                emb_l = self.direct_pose_leg_side_embed(emb_l.to(device=device)).view(1, 1, -1).expand(B, Tq, -1)  # type: ignore[operator]
                                            except Exception:
                                                emb_r = emb_l = None

                                        # Build per-side leg head inputs.
                                        parts_r = [
                                            direct_feat,
                                            plan_r,
                                            meas_r,
                                            phase_r,
                                            plan_other_r,
                                            phase_other_r,
                                            phase_rel_r,
                                            cue_r,
                                        ]
                                        parts_l = [
                                            direct_feat,
                                            plan_l,
                                            meas_l,
                                            phase_l,
                                            plan_other_l,
                                            phase_other_l,
                                            phase_rel_l,
                                            cue_l,
                                        ]
                                        if torch.is_tensor(emb_r) and torch.is_tensor(emb_l):
                                            parts_r.append(emb_r)
                                            parts_l.append(emb_l)
                                        leg_in_r = torch.cat([p for p in parts_r if torch.is_tensor(p) and p.numel() > 0], dim=-1)
                                        leg_in_l = torch.cat([p for p in parts_l if torch.is_tensor(p) and p.numel() > 0], dim=-1)
                                        leg_flat_r = leg_in_r.reshape(-1, leg_in_r.shape[-1])
                                        leg_flat_l = leg_in_l.reshape(-1, leg_in_l.shape[-1])
                                        if bool(getattr(self, "direct_pose_leg_detach_feat", False)):
                                            leg_flat_r = leg_flat_r.detach()
                                            leg_flat_l = leg_flat_l.detach()

                                        out_r = self.direct_pose_leg_head_shared(leg_flat_r).view(B, Tq, -1)
                                        out_l = self.direct_pose_leg_head_shared(leg_flat_l).view(B, Tq, -1)

                                        omega_r = None
                                        omega_l = None
                                        if bool(getattr(self, "direct_pose_leg_side_rank1", False)):
                                            # Rank-1 parameterization: omega_j = softplus(s_j) * normalize(v)
                                            if out_r.shape[-1] == (3 + K_side) and out_l.shape[-1] == (3 + K_side):
                                                v_r = out_r[..., :3]
                                                v_l = out_l[..., :3]
                                                s_r = F.softplus(out_r[..., 3:])
                                                s_l = F.softplus(out_l[..., 3:])
                                                dir_r = F.normalize(v_r, dim=-1, eps=1e-8)
                                                dir_l = F.normalize(v_l, dim=-1, eps=1e-8)
                                                omega_r = dir_r.unsqueeze(-2) * s_r.unsqueeze(-1)
                                                omega_l = dir_l.unsqueeze(-2) * s_l.unsqueeze(-1)
                                        else:
                                            # Default: per-joint omega vectors.
                                            if out_r.shape[-1] == 3 * K_side and out_l.shape[-1] == 3 * K_side:
                                                omega_r = out_r.view(B, Tq, K_side, 3)
                                                omega_l = out_l.view(B, Tq, K_side, 3)
                                                # Optional: per-side sign gate (same scalar for all joints on that side).
                                                gate = getattr(self, "direct_pose_leg_side_sign_gate_head", None)
                                                if gate is not None and bool(getattr(self, "direct_pose_leg_side_sign_gate", False)):
                                                    try:
                                                        g_r = torch.tanh(gate(leg_flat_r)).view(B, Tq, 1, 1)
                                                        g_l = torch.tanh(gate(leg_flat_l)).view(B, Tq, 1, 1)
                                                        omega_r = omega_r * g_r
                                                        omega_l = omega_l * g_l
                                                        direct_leg_side_sign_gate = torch.cat(
                                                            [g_r.view(B, Tq, 1), g_l.view(B, Tq, 1)], dim=-1
                                                        )
                                                    except Exception:
                                                        direct_leg_side_sign_gate = None

                                        if torch.is_tensor(omega_r) and torch.is_tensor(omega_l):
                                            # Scatter to original K ordering (positions in K list).
                                            pos_r_use = pos_r.to(device=device)
                                            pos_l_use = pos_l.to(device=device)
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
                                            direct_leg_omega_raw = omega_leg
                                            omega_eff = omega_leg

                                            # Optional learned gate / learned scale (per-joint, per-side; shared head).
                                            gm_leg = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                                            clamp_k = float(getattr(self, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0)
                                            use_scale_clamp = bool(_math.isfinite(clamp_k) and clamp_k > 1.0)
                                            if use_scale_clamp:
                                                scale_min = float(1.0 / clamp_k)
                                                scale_max = float(clamp_k)
                                            if gm_leg == "learned":
                                                try:
                                                    gate = getattr(self, "direct_pose_leg_gate_head_shared", None)
                                                    if gate is None:
                                                        raise RuntimeError("learned leg gate enabled but direct_pose_leg_gate_head_shared is missing")
                                                    gl_r = gate(leg_flat_r).view(B, Tq, K_side)
                                                    gl_l = gate(leg_flat_l).view(B, Tq, K_side)
                                                    g_r = torch.sigmoid(gl_r)
                                                    g_l = torch.sigmoid(gl_l)
                                                    power = float(getattr(self, "direct_pose_leg_gate_power", 1.0) or 1.0)
                                                    if (not _math.isfinite(power)) or power <= 0.0:
                                                        power = 1.0
                                                    if abs(power - 1.0) > 1e-12:
                                                        g_r = g_r.pow(power)
                                                        g_l = g_l.pow(power)
                                                    g_leg = omega_leg.new_zeros((B, Tq, K))
                                                    g_leg = g_leg.index_copy(2, pos_r_use, g_r)
                                                    g_leg = g_leg.index_copy(2, pos_l_use, g_l)
                                                    gl_leg = omega_leg.new_zeros((B, Tq, K))
                                                    gl_leg = gl_leg.index_copy(2, pos_r_use, gl_r)
                                                    gl_leg = gl_leg.index_copy(2, pos_l_use, gl_l)
                                                    direct_leg_gate = g_leg
                                                    direct_leg_gate_logits = gl_leg
                                                    omega_eff = omega_leg * g_leg.unsqueeze(-1)
                                                except Exception:
                                                    direct_leg_gate = None
                                                    direct_leg_gate_logits = None
                                                    omega_eff = omega_leg
                                            elif gm_leg == "scale":
                                                try:
                                                    scale_head = getattr(self, "direct_pose_leg_gate_head_shared", None)
                                                    if scale_head is None:
                                                        raise RuntimeError("leg scale enabled but direct_pose_leg_gate_head_shared is missing")
                                                    log_mag_raw_r = scale_head(leg_flat_r).view(B, Tq, K_side)
                                                    log_mag_raw_l = scale_head(leg_flat_l).view(B, Tq, K_side)
                                                    clip = float(getattr(self, "direct_pose_leg_scale_log_clip", 4.0) or 4.0)
                                                    if (not _math.isfinite(clip)) or clip <= 0.0:
                                                        clip = 4.0
                                                    log_mag_r = log_mag_raw_r.clamp(-float(clip), float(clip))
                                                    log_mag_l = log_mag_raw_l.clamp(-float(clip), float(clip))
                                                    scale_r = torch.exp(log_mag_r)
                                                    scale_l = torch.exp(log_mag_l)
                                                    if use_scale_clamp:
                                                        scale_r = scale_r.clamp(scale_min, scale_max)
                                                        scale_l = scale_l.clamp(scale_min, scale_max)
                                                        log_mag_r = torch.log(scale_r)
                                                        log_mag_l = torch.log(scale_l)

                                                    scale_leg = omega_leg.new_zeros((B, Tq, K))
                                                    scale_leg = scale_leg.index_copy(2, pos_r_use, scale_r)
                                                    scale_leg = scale_leg.index_copy(2, pos_l_use, scale_l)
                                                    log_leg = omega_leg.new_zeros((B, Tq, K))
                                                    log_leg = log_leg.index_copy(2, pos_r_use, log_mag_r)
                                                    log_leg = log_leg.index_copy(2, pos_l_use, log_mag_l)
                                                    log_raw_leg = omega_leg.new_zeros((B, Tq, K))
                                                    log_raw_leg = log_raw_leg.index_copy(2, pos_r_use, log_mag_raw_r)
                                                    log_raw_leg = log_raw_leg.index_copy(2, pos_l_use, log_mag_raw_l)

                                                    direct_leg_scale = scale_leg
                                                    direct_leg_scale_log = log_leg
                                                    direct_leg_scale_log_raw = log_raw_leg
                                                    omega_eff = omega_leg * scale_leg.unsqueeze(-1)
                                                except Exception:
                                                    direct_leg_scale = None
                                                    direct_leg_scale_log = None
                                                    direct_leg_scale_log_raw = None
                                                    omega_eff = omega_leg

                                            direct_leg_omega = omega_eff
                    except Exception:
                        pass
                elif self.direct_pose_leg_head is not None:
                    try:
                        idx = getattr(self, "direct_pose_leg_joint_idx_tensor", None)
                        rot_sl = getattr(self, "direct_pose_leg_rot6d_slice", None)
                        if torch.is_tensor(idx) and isinstance(rot_sl, slice) and rot_sl.start is not None and rot_sl.stop is not None:
                            K = int(idx.numel())
                            if K > 0:
                                leg_in = direct_flat.detach() if bool(getattr(self, "direct_pose_leg_detach_feat", False)) else direct_flat

                                # Optional: cross-leg ablation for the *non-routed* leg head.
                                # Goal: for right-side joints, ablate left-channel contact features (plan/meas/phase) in leg head input,
                                # and vice versa, then merge the per-side outputs back into the original K ordering.
                                #
                                # This keeps the main direct_out path unchanged and only probes whether leg omega relies on cross-leg context.
                                try:
                                    ab = str(getattr(self, "direct_pose_leg_cross_leg_ablate", "none") or "none").strip().lower()
                                except Exception:
                                    ab = "none"
                                leg_delta = None
                                if ab in ("", "none", "off", "disable", "disabled"):
                                    leg_delta = self.direct_pose_leg_head(leg_in).view(B, Tq, -1)
                                else:
                                    try:
                                        # Need both sides present in contact features.
                                        Cc = int(getattr(self, "contact_dim", 0) or 0)
                                    except Exception:
                                        Cc = 0
                                    # Require side-tagged joint names to route outputs.
                                    try:
                                        names = list(getattr(self, "direct_pose_leg_joint_names", []) or [])
                                    except Exception:
                                        names = []
                                    # Only supported for symmetric _r/_l sets.
                                    if (
                                        torch.is_tensor(direct_flat)
                                        and Cc >= 2
                                        and isinstance(names, list)
                                        and len(names) == K
                                    ):
                                        names_l = [str(n).lower() for n in names]
                                        pos_r = [i for i, n in enumerate(names_l) if n.endswith(("_r", "right"))]
                                        pos_l = [i for i, n in enumerate(names_l) if n.endswith(("_l", "left"))]
                                        if pos_r and pos_l and (len(pos_r) + len(pos_l) == K):
                                            # Determine where per-channel contact features live inside the already-built direct_flat input.
                                            #
                                            # Supported layouts for direct_flat (depends on direct_pose_meas_mode / phase_z_mode):
                                            #   A) [direct_feat, plan(C), meas(C?), phase(2C?)]
                                            #   B) [direct_feat, phase_hint(2C)]   (direct_pose_phase_z_mode='replace_contacts')
                                            x = direct_flat.reshape(B, Tq, -1)
                                            d_total = int(x.shape[-1])
                                            d_direct = int(direct_feat.shape[-1]) if torch.is_tensor(direct_feat) else 0
                                            d_plan = int(plan_in.shape[-1]) if torch.is_tensor(plan_in) else 0
                                            d_phase = int(phase_in_direct.shape[-1]) if torch.is_tensor(phase_in_direct) else 0
                                            d_meas_raw = int(meas_in.shape[-1]) if torch.is_tensor(meas_in) else 0
                                            d_meas = 0
                                            if d_total == (d_direct + d_plan + d_meas_raw + d_phase):
                                                d_meas = d_meas_raw
                                            elif d_total == (d_direct + d_plan + d_phase):
                                                d_meas = 0
                                            # Channel mapping (default dataset order is [L,R]).
                                            ch_r = int(getattr(self, "direct_pose_leg_contact_ch_r", 1) or 0)
                                            ch_l = int(getattr(self, "direct_pose_leg_contact_ch_l", 0) or 0)
                                            ch_r = max(0, min(Cc - 1, ch_r))
                                            ch_l = max(0, min(Cc - 1, ch_l))

                                            # Layout A: plan+meas(+phase) present in direct_flat.
                                            if d_direct > 0 and d_plan == Cc and d_total == (d_direct + d_plan + d_meas + d_phase):
                                                off_plan = d_direct
                                                off_meas = off_plan + d_plan
                                                off_phase = off_meas + d_meas

                                                x_r = x.clone()
                                                x_l = x.clone()

                                                def _ablate(xx: torch.Tensor, ch: int) -> None:
                                                    if ab in ("0", "zero", "zeros"):
                                                        xx[..., off_plan + ch] = 0.0
                                                        if d_meas > 0:
                                                            xx[..., off_meas + ch] = 0.0
                                                        # phase layout: [sin0,cos0,sin1,cos1,...] (dim=2*C)
                                                        if d_phase == 2 * Cc:
                                                            s = off_phase + 2 * ch
                                                            xx[..., s : s + 2] = 0.0
                                                    elif ab in ("roll", "roll_batch", "shift", "shift_batch"):
                                                        if int(B) > 1:
                                                            xx[..., off_plan + ch] = xx[..., off_plan + ch].roll(shifts=1, dims=0)
                                                            if d_meas > 0:
                                                                xx[..., off_meas + ch] = xx[..., off_meas + ch].roll(
                                                                    shifts=1, dims=0
                                                                )
                                                            if d_phase == 2 * Cc:
                                                                s = off_phase + 2 * ch
                                                                xx[..., s : s + 2] = xx[..., s : s + 2].roll(
                                                                    shifts=1, dims=0
                                                                )
                                                    elif ab in ("roll_time", "shift_time"):
                                                        if int(Tq) > 1:
                                                            xx[..., off_plan + ch] = xx[..., off_plan + ch].roll(shifts=1, dims=1)
                                                            if d_meas > 0:
                                                                xx[..., off_meas + ch] = xx[..., off_meas + ch].roll(
                                                                    shifts=1, dims=1
                                                                )
                                                            if d_phase == 2 * Cc:
                                                                s = off_phase + 2 * ch
                                                                xx[..., s : s + 2] = xx[..., s : s + 2].roll(
                                                                    shifts=1, dims=1
                                                                )

                                            # Layout B: phase-hint-only (replace_contacts): direct_flat = [direct_feat, phase(2C)].
                                            elif d_direct > 0 and d_phase == 2 * Cc and d_total == (d_direct + d_phase):
                                                off_phase = d_direct

                                                x_r = x.clone()
                                                x_l = x.clone()

                                                def _ablate(xx: torch.Tensor, ch: int) -> None:
                                                    # Only phase exists here.
                                                    if d_phase != 2 * Cc:
                                                        return
                                                    s = off_phase + 2 * ch
                                                    if ab in ("0", "zero", "zeros"):
                                                        xx[..., s : s + 2] = 0.0
                                                    elif ab in ("roll", "roll_batch", "shift", "shift_batch"):
                                                        if int(B) > 1:
                                                            xx[..., s : s + 2] = xx[..., s : s + 2].roll(shifts=1, dims=0)
                                                    elif ab in ("roll_time", "shift_time"):
                                                        if int(Tq) > 1:
                                                            xx[..., s : s + 2] = xx[..., s : s + 2].roll(shifts=1, dims=1)
                                            else:
                                                x_r = x_l = None

                                            if torch.is_tensor(x_r) and torch.is_tensor(x_l):
                                                # For right-side outputs, ablate left-channel features; for left-side, ablate right-channel.
                                                _ablate(x_r, ch_l)
                                                _ablate(x_l, ch_r)

                                                flat_r = x_r.reshape(-1, x_r.shape[-1])
                                                flat_l = x_l.reshape(-1, x_l.shape[-1])
                                                if bool(getattr(self, "direct_pose_leg_detach_feat", False)):
                                                    flat_r = flat_r.detach()
                                                    flat_l = flat_l.detach()
                                                out_r = self.direct_pose_leg_head(flat_r).view(B, Tq, -1)
                                                out_l = self.direct_pose_leg_head(flat_l).view(B, Tq, -1)

                                                # Merge per-side outputs in K ordering (supports 3*K or 6*K layouts).
                                                if out_r.shape == out_l.shape and out_r.shape[-1] in (3 * K, 6 * K):
                                                    L = int(out_r.shape[-1] // K)
                                                    v_r = out_r.view(B, Tq, K, L)
                                                    v_l = out_l.view(B, Tq, K, L)
                                                    merged = v_r.clone()
                                                    merged[:, :, pos_l, :] = v_l[:, :, pos_l, :]
                                                    leg_delta = merged.view(B, Tq, K * L)
                                if leg_delta is None:
                                    leg_delta = self.direct_pose_leg_head(leg_in).view(B, Tq, -1)
                                rot_dim = int(rot_sl.stop - rot_sl.start)
                                if rot_dim > 0 and (rot_dim % 6) == 0:
                                    J = int(rot_dim // 6)
                                    idx_use = idx.to(device=device)
                                    if bool(torch.all((idx_use >= 0) & (idx_use < J)).detach().cpu().item()):
                                        leg_mode = str(getattr(self, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").lower().strip()
                                        if leg_mode == "so3":
                                            # Output omega (axis-angle) for external RAW-space composition.
                                            if leg_delta.shape[-1] == 3 * K:
                                                omega_leg = leg_delta.view(B, Tq, K, 3)
                                                max_rad = float(getattr(self, "direct_pose_leg_max_rad", 0.0) or 0.0)
                                                if _math.isfinite(max_rad) and max_rad > 0.0:
                                                    # Bound ||omega|| <= max_rad with smooth scaling.
                                                    theta = omega_leg.norm(dim=-1, keepdim=True)
                                                    # Use a smooth clamp that preserves the limit scale->1 at theta->0,
                                                    # otherwise a zero-initialized head would get zero gradients.
                                                    denom = theta + 1e-8
                                                    scale = (max_rad * torch.tanh(theta / max_rad)) / denom
                                                    scale = torch.where(theta > 1e-8, scale, torch.ones_like(scale))
                                                    omega_leg = omega_leg * scale
                                                direct_leg_omega_raw = omega_leg
                                                omega_eff = omega_leg

                                                # Optional learned gate / learned scale (per-joint logits).
                                                gm_leg = str(getattr(self, "direct_pose_leg_gate_mode", "none") or "none").lower().strip()
                                                clamp_k = float(getattr(self, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0)
                                                use_scale_clamp = bool(_math.isfinite(clamp_k) and clamp_k > 1.0)
                                                if use_scale_clamp:
                                                    scale_min = float(1.0 / clamp_k)
                                                    scale_max = float(clamp_k)
                                                if gm_leg == "learned":
                                                    try:
                                                        gate = getattr(self, "direct_pose_leg_gate_head", None)
                                                        if gate is None:
                                                            raise RuntimeError("learned leg gate enabled but direct_pose_leg_gate_head is missing")
                                                        gl = gate(leg_in).view(B, Tq, K)  # type: ignore[arg-type]
                                                        g = torch.sigmoid(gl)
                                                        power = float(getattr(self, "direct_pose_leg_gate_power", 1.0) or 1.0)
                                                        if (not _math.isfinite(power)) or power <= 0.0:
                                                            power = 1.0
                                                        if abs(power - 1.0) > 1e-12:
                                                            g = g.pow(power)
                                                        direct_leg_gate = g
                                                        direct_leg_gate_logits = gl
                                                        omega_eff = omega_leg * g.unsqueeze(-1)
                                                    except Exception:
                                                        direct_leg_gate = None
                                                        direct_leg_gate_logits = None
                                                        omega_eff = omega_leg
                                                elif gm_leg == "scale":
                                                    # Scale can both attenuate and amplify omega (targets under-correct: best_alpha >> 1).
                                                    try:
                                                        scale_head = getattr(self, "direct_pose_leg_gate_head", None)
                                                        if scale_head is None:
                                                            raise RuntimeError("leg scale enabled but direct_pose_leg_gate_head is missing")
                                                        log_mag_raw = scale_head(leg_in).view(B, Tq, K)  # type: ignore[arg-type]
                                                        clip = float(getattr(self, "direct_pose_leg_scale_log_clip", 4.0) or 4.0)
                                                        if (not _math.isfinite(clip)) or clip <= 0.0:
                                                            clip = 4.0
                                                        log_mag = log_mag_raw.clamp(-float(clip), float(clip))
                                                        scale = torch.exp(log_mag)
                                                        if use_scale_clamp:
                                                            scale = scale.clamp(scale_min, scale_max)
                                                            log_mag = torch.log(scale)
                                                        direct_leg_scale = scale
                                                        direct_leg_scale_log = log_mag
                                                        direct_leg_scale_log_raw = log_mag_raw
                                                        omega_eff = omega_leg * scale.unsqueeze(-1)
                                                    except Exception:
                                                        direct_leg_scale = None
                                                        direct_leg_scale_log = None
                                                        direct_leg_scale_log_raw = None
                                                        omega_eff = omega_leg

                                                direct_leg_omega = omega_eff
                                            # else: unexpected shape; skip (keeps direct_leg_omega=None)
                                        else:
                                            # Legacy: additive residual in 6D parameter space (off-manifold).
                                            if leg_delta.shape[-1] == 6 * K:
                                                delta_rot = direct_out.new_zeros((B, Tq, J, 6))
                                                delta_rot[:, :, idx_use, :] = leg_delta.view(B, Tq, K, 6)
                                                delta_full = torch.zeros_like(direct_out)
                                                delta_full[..., rot_sl] = delta_rot.view(B, Tq, -1)
                                                direct_out = direct_out + delta_full
                    except Exception:
                        pass


                if is_single:
                    direct_out = direct_out.squeeze(1)
                result['out_direct'] = direct_out
                if torch.is_tensor(direct_leg_omega):
                    dlo = direct_leg_omega.squeeze(1) if is_single else direct_leg_omega
                    result["direct_leg_omega"] = dlo
                if torch.is_tensor(direct_leg_omega_raw):
                    dlr = direct_leg_omega_raw.squeeze(1) if is_single else direct_leg_omega_raw
                    result["direct_leg_omega_raw"] = dlr
                if torch.is_tensor(direct_leg_gate):
                    dlg = direct_leg_gate.squeeze(1) if is_single else direct_leg_gate
                    result["direct_leg_gate"] = dlg
                if torch.is_tensor(direct_leg_gate_logits):
                    dlg = direct_leg_gate_logits.squeeze(1) if is_single else direct_leg_gate_logits
                    result["direct_leg_gate_logits"] = dlg
                if torch.is_tensor(direct_leg_scale):
                    dls = direct_leg_scale.squeeze(1) if is_single else direct_leg_scale
                    result["direct_leg_scale"] = dls
                if torch.is_tensor(direct_leg_scale_log):
                    dls = direct_leg_scale_log.squeeze(1) if is_single else direct_leg_scale_log
                    result["direct_leg_scale_log"] = dls
                if torch.is_tensor(direct_leg_scale_log_raw):
                    dls = direct_leg_scale_log_raw.squeeze(1) if is_single else direct_leg_scale_log_raw
                    result["direct_leg_scale_log_raw"] = dls
                if torch.is_tensor(direct_leg_side_sign_gate):
                    dsg = direct_leg_side_sign_gate.squeeze(1) if is_single else direct_leg_side_sign_gate
                    result["direct_leg_side_sign_gate"] = dsg

            except Exception as exc:
                raise RuntimeError("direct_pose forward failed") from exc

        if self.lambda_fusion_head is not None:
            try:
                lam_in = h_final
                if lam_in.ndim == 2:
                    lam_in = lam_in.unsqueeze(1)

                if self.contact_plan_enable and e_t is not None:
                    err_in = e_t.detach() if self.lambda_fusion_detach_err else e_t
                    if err_in.ndim == 2:
                        err_in = err_in.unsqueeze(1)
                    lam_in = torch.cat([lam_in, err_in.to(device=device, dtype=dtype)], dim=-1)

                if bool(getattr(self, "lambda_fusion_use_rollout_step", False)):
                    B, Tq, _ = lam_in.shape
                    try:
                        if rollout_step is None:
                            step_feat = lam_in.new_zeros((B, Tq, 1))
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
                                    step_feat = s[:1].view(1, 1, 1).expand(B, Tq, 1)
                            elif s.dim() == 2:
                                if s.shape[0] == 1 and B > 1:
                                    s = s.expand(B, -1)
                                if s.shape[1] == 1:
                                    step_feat = s[:, :1].reshape(B, 1, 1).expand(B, Tq, 1)
                                else:
                                    step_feat = s[:, :Tq].unsqueeze(-1)
                                    if step_feat.shape[1] < Tq:
                                        pad = step_feat[:, -1:, :].expand(B, Tq - step_feat.shape[1], 1)
                                        step_feat = torch.cat([step_feat, pad], dim=1)
                            else:
                                if s.shape[0] == 1 and B > 1:
                                    s = s.expand(B, *s.shape[1:])
                                if s.shape[-1] != 1:
                                    s = s[..., :1]
                                if s.shape[1] == 1:
                                    step_feat = s[:, :1, :].expand(B, Tq, 1)
                                else:
                                    step_feat = s[:, :Tq, :].reshape(B, -1, 1)
                                    if step_feat.shape[1] < Tq:
                                        pad = step_feat[:, -1:, :].expand(B, Tq - step_feat.shape[1], 1)
                                        step_feat = torch.cat([step_feat, pad], dim=1)
                        else:
                            step_feat = torch.full((B, Tq, 1), float(rollout_step), device=device, dtype=dtype)
                    except Exception:
                        step_feat = lam_in.new_zeros((B, Tq, 1))
                    lam_in = torch.cat([lam_in, step_feat], dim=-1)

                flat = lam_in.reshape(-1, lam_in.shape[-1])
                logits = self.lambda_fusion_head(flat).view(lam_in.shape[0], lam_in.shape[1], -1)
                lam = torch.sigmoid(logits)
                # Normalize output to per-joint (B,T,J) even in global mode.
                if self.lambda_fusion_mode == "global" and int(self.lambda_fusion_joint_count) > 0:
                    lam = lam.expand(lam.shape[0], lam.shape[1], int(self.lambda_fusion_joint_count))
                if is_single:
                    logits = logits.squeeze(1)
                    lam = lam.squeeze(1)
                result["lambda_fusion_logits"] = logits
                result["lambda_fusion"] = lam
            except Exception:
                pass

        if self.so3_delta_corrector is not None and self.so3_corr_joint_count > 0:
            corr_in = h_final
            if self.contact_plan_enable and e_t is not None:
                # Ensure e_t aligns with (B,T,C)
                if e_t.ndim == 2:
                    e_t = e_t.unsqueeze(1)
                corr_in = torch.cat([h_final, e_t.to(device=device, dtype=dtype)], dim=-1)
            omega = self.so3_delta_corrector(corr_in)  # (B, T, 3J)
            omega = omega.view(omega.shape[0], omega.shape[1], self.so3_corr_joint_count, 3)
            if is_single:
                omega = omega.squeeze(1)
            result['omega_hat'] = omega
        if soft_period is not None:
            result['period_pred'] = soft_period
        return result

    def attach_motion_encoder(self, bundle, *, map_location: str | torch.device = 'cpu'):
        """
        加载并冻结预训练的 MotionEncoder + PeriodHead（以及可选 contact_head），
        用于提供 soft hint / frozen contact logits。
        """
        if isinstance(bundle, (str, os.PathLike)):
            payload = torch.load(bundle, map_location=map_location)
        else:
            payload = bundle
        if not isinstance(payload, dict):
            raise TypeError("MotionEncoder bundle must be a dict or path to a dict.")
        require_standard_rotvec_bundle(payload, context="MotionEncoder bundle")

        encoder_state = payload.get('encoder')
        period_state = payload.get('period_head')
        contact_state = payload.get('contact_head')
        if encoder_state is None or period_state is None:
            raise KeyError("Bundle missing 'encoder' or 'period_head' state_dict.")

        meta = dict(payload.get('meta', {}))
        hint_mode = str(meta.get("period_hint_mode") or "").strip()
        if not hint_mode:
            # Backward-compatible default for older bundles that didn't record this field.
            hint_mode = "contacts_tanh"
        if hint_mode != "contacts_tanh":
            raise ValueError(f"Unsupported MotionEncoder bundle period_hint_mode={hint_mode!r} (expected 'contacts_tanh').")
        weight0 = encoder_state.get('mlp.0.weight')
        if weight0 is None:
            for key, val in encoder_state.items():
                if key.endswith('weight') and val.ndim == 2:
                    weight0 = val
                    break
        if weight0 is None:
            raise ValueError("Unable to infer MotionEncoder dimensions from state_dict.")

        input_dim = int(meta.get('input_dim', weight0.shape[1]))
        hidden_dim = int(meta.get('hidden_dim', weight0.shape[0]))
        z_dim = int(meta.get('z_dim', 0))
        num_layers = int(meta.get('mlp_layers', 3))
        dropout = float(meta.get('mlp_dropout', 0.0))

        encoder = MotionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        encoder.load_state_dict(encoder_state)
        encoder.eval().requires_grad_(False)

        period_dim = int(period_state['fc.weight'].shape[0])
        period_head = PeriodHead(hidden_dim, period_dim)
        period_head.load_state_dict(period_state)
        period_head.eval().requires_grad_(False)

        frozen_contact_head: Optional[nn.Module] = None
        if isinstance(contact_state, dict):
            try:
                w = contact_state.get("fc.weight")
                b = contact_state.get("fc.bias")
                if torch.is_tensor(w):
                    out_dim = int(w.shape[0])
                    linear = nn.Linear(hidden_dim, out_dim)
                    linear_state = {"weight": w}
                    if torch.is_tensor(b):
                        linear_state["bias"] = b
                    linear.load_state_dict(linear_state, strict=False)
                    linear.eval().requires_grad_(False)
                    frozen_contact_head = linear
            except Exception:
                frozen_contact_head = None

        if self.encoder_input_dim and self.encoder_input_dim != input_dim:
            raise ValueError(f"Encoder input dim mismatch: dataset={self.encoder_input_dim} vs bundle={input_dim}")
        self.encoder_input_dim = input_dim

        device = self._target_device()
        self.frozen_encoder = encoder.to(device)
        self.frozen_period_head = period_head.to(device)
        self.frozen_contact_head = frozen_contact_head.to(device) if frozen_contact_head is not None else None

        if self.period_dim != period_dim or self.period_encoder is None:
            self.period_dim = period_dim
            self.period_encoder = nn.Linear(self.period_dim, self.hidden_dim).to(device)

        return meta

class MotionJointLoss(nn.Module):
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

        # ===== Adaptive bone weighting =====
        adaptive_bone_weights: bool = False,
        **legacy_kwargs: Any,
    ):
        super().__init__()
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
        self.meta = dict(meta) if isinstance(meta, dict) else {}
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
        try:
            self.direct_pose_loss_arm_weight = float(direct_pose_loss_arm_weight or 1.0)
        except Exception:
            self.direct_pose_loss_arm_weight = 1.0
        try:
            self.direct_pose_loss_else_weight = float(direct_pose_loss_else_weight or 1.0)
        except Exception:
            self.direct_pose_loss_else_weight = 1.0
        if self.direct_pose_loss_arm_weight < 0.0:
            self.direct_pose_loss_arm_weight = 0.0
        if self.direct_pose_loss_else_weight < 0.0:
            self.direct_pose_loss_else_weight = 0.0
        if (self.direct_pose_loss_arm_weight + self.direct_pose_loss_else_weight) <= 0.0:
            self.direct_pose_loss_arm_weight = 1.0
            self.direct_pose_loss_else_weight = 1.0
        self.direct_pose_loss_group_norm_enable = bool(direct_pose_loss_group_norm_enable)
        self.direct_pose_loss_group_norm_w_leg = float(direct_pose_loss_group_norm_w_leg or 1.0)
        self.direct_pose_loss_group_norm_w_nonleg = float(direct_pose_loss_group_norm_w_nonleg or 1.0)
        try:
            self.direct_pose_loss_group_norm_ema_beta = float(direct_pose_loss_group_norm_ema_beta or 0.9)
        except Exception:
            self.direct_pose_loss_group_norm_ema_beta = 0.9
        try:
            self.direct_pose_loss_group_norm_ratio_min = float(direct_pose_loss_group_norm_ratio_min or 0.2)
        except Exception:
            self.direct_pose_loss_group_norm_ratio_min = 0.2
        try:
            self.direct_pose_loss_group_norm_ratio_max = float(direct_pose_loss_group_norm_ratio_max or 5.0)
        except Exception:
            self.direct_pose_loss_group_norm_ratio_max = 5.0
        if self.direct_pose_loss_group_norm_ratio_min > self.direct_pose_loss_group_norm_ratio_max:
            self.direct_pose_loss_group_norm_ratio_min, self.direct_pose_loss_group_norm_ratio_max = (
                self.direct_pose_loss_group_norm_ratio_max,
                self.direct_pose_loss_group_norm_ratio_min,
            )
        try:
            self.direct_pose_loss_group_norm_eps = float(direct_pose_loss_group_norm_eps or 1e-6)
        except Exception:
            self.direct_pose_loss_group_norm_eps = 1e-6
        if self.direct_pose_loss_group_norm_eps <= 0.0:
            self.direct_pose_loss_group_norm_eps = 1e-6
        self._direct_pose_group_norm_ema: Dict[str, torch.Tensor] = {}
        self.w_omega_l2 = float(w_omega_l2)
        self.event_clock_lambda_entropy_weight = float(event_clock_lambda_entropy_weight or 0.0)
        self.event_clock_lambda_prior_weight = float(event_clock_lambda_prior_weight or 0.0)
        self.event_clock_delta_z_l2_weight = float(event_clock_delta_z_l2_weight or 0.0)
        # Tail-risk regularization for per-bone rotation errors (CVaR / top-k style).
        # When enabled, adds an extra term on the worst bones (by mean GeoLocalDeg),
        # which reduces whack-a-mole without requiring explicit per-bone weight tables.
        self.rot_local_tail_weight = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        self.rot_local_tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        self.rot_local_tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all')
        self.rot_local_tail_select = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch')
        self.rot_local_tail_ema_beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        self.fps = float(fps)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        self._rot6d_columns = self._resolve_rot6d_columns(self.rot6d_spec)
        layout = self.output_layout or {}
        inner = layout.get('slices') if isinstance(layout.get('slices'), dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.attn_lambda_local = getattr(self, 'attn_lambda_local', 0.02)
        self.attn_lambda_entropy = getattr(self, 'attn_lambda_entropy', 0.0)
        self._warned_bad_rot6d = False
        self.template_hint: Optional[str] = None
        self.bundle_hint: Optional[str] = None
        # 缓存几何骨骼权重（按 device/dtype）
        self._joint_weight_cache: dict[tuple, torch.Tensor] = {}
        # Tail-loss 辅助缓存（candidate pool 与选择打分）
        self._tail_candidate_cache: dict[tuple, torch.Tensor] = {}
        self._tail_score_cache: dict[tuple, torch.Tensor] = {}
        self.root_idx = 0
        self.bone_names: list[str] = []
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
            offsets = skeleton.get('ref_local_offsets_m')
            if isinstance(offsets, (list, tuple)):
                try:
                    self.bone_offsets = torch.as_tensor(offsets, dtype=torch.float32)
                except Exception:
                    self.bone_offsets = None
        # Loss component tracking (reserved for optional adaptive weighting; currently disabled).
        self._adaptive_loss_terms: Tuple[str, ...] = (
            "rot_local",
        )
        self._reset_adaptive_tracking()
        self._loss_group_totals: Dict[str, float] = {}
        self._loss_group_alias = {
            'attn': 'aux',
            'rot_ortho': 'core',
            'rot_local': 'core',
            'root_vel': 'core',
            'root_speed': 'core',
            'direct_pose': 'core',
        }

        # === adaptive bone weight params ===
        self.use_adaptive_weights = bool(adaptive_bone_weights)

        # skeleton parents may be set later via set_skeleton; avoid early fallback here

    def _format_template_hint(self, prefix: str) -> str:
        hints: list[str] = []
        if isinstance(self.template_hint, str) and self.template_hint:
            hints.append(f"norm_template={self.template_hint}")
        if isinstance(self.bundle_hint, str) and self.bundle_hint:
            hints.append(f"bundle_json={self.bundle_hint}")
        if hints:
            return f"{prefix} ({', '.join(hints)})"
        return prefix

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

    @staticmethod
    def _masked_group_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if values is None or mask is None or values.numel() == 0:
            return None
        if mask.numel() != values.shape[-1]:
            return None
        if not bool(mask.any().detach().cpu().item()):
            return None
        return values[..., mask].mean()

    def set_skeleton(self, parents: Optional[Sequence[int]], offsets: Optional[Sequence[Sequence[float]]]) -> None:
        if parents is not None:
            self.parents = [int(p) for p in parents]
            self._parents_tensor = None
        if offsets is not None:
            try:
                self.bone_offsets = torch.as_tensor(offsets, dtype=torch.float32)
            except Exception:
                self.bone_offsets = None
        self._tail_candidate_cache = {}

    def _invalidate_weight_cache(self) -> None:
        """Drop cached joint weights when configuration changes."""
        self._joint_weight_cache = {}

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

    # === Unified bone weight computation (replaces adaptive + hierarchy) ===
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
            print(
                f"[Loss][UnifiedWeights] range=[{weights.min():.3f}, {weights.max():.3f}] "
                f"mean={weights.mean():.3f} std={weights.std():.3f} "
                f"power={getattr(self, 'unified_downstream_power', 0.6)} "
                f"self_scale={getattr(self, 'unified_self_scale', 1.5)} "
                f"min_w={getattr(self, 'unified_min_weight', 0.05)} "
                f"visual=False"
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
                try:
                    M = 1.0 - geomask
                    loss_local = (A * M).mean()
                except Exception:
                    gm = geomask
                    if torch.is_tensor(gm):
                        if gm.dim() == 2:
                            gm = gm.view(1, T, T)
                        elif gm.dim() == 4:
                            gm = gm.mean(0)
                        M = 1.0 - gm
                        loss_local = (A * M).mean()
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
                print("[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.")
                self._train_denorm_hit = True
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
                print(
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.")
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

        # Geodesic kernel is centralized in train.geometry.
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
                print(f"[Loss][WARN] BoneRotations6D slice dim={D} not multiple of 6. Skip rot6d_ortho.")
            return Z(0.0)
        J = D // 6
        a6 = pr.view(*pr.shape[:-1], J, 6)  # (..., J, 6) raw 6D (no GS)
        v1 = a6[..., 0:3]
        v2 = a6[..., 3:6]
        len_p = (v1.norm(dim=-1) - 1.0).pow(2) + (v2.norm(dim=-1) - 1.0).pow(2)
        ortho_p = (v1.mul(v2).sum(dim=-1)).pow(2)
        return (len_p + ortho_p).mean()

    def _rot6d_matrices(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        return self._extract_rot6d_mats(X, denorm=True, reproject=True, sanitize=True)

    def _angvel_hz(self) -> float:
        hz = float(getattr(self, 'bone_hz', 0.0) or 0.0)
        if hz <= 0.0:
            dt = float(getattr(self, 'dt_bone', 0.0) or 0.0)
            if dt > 0.0:
                hz = 1.0 / max(dt, 1e-6)
        if hz <= 0.0:
            hz = float(getattr(self, 'fps', 60.0) or 60.0)
        return max(hz, 1e-6)

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


    def _prepare_forward_inputs(
        self,
        pred_motion: Any,
        gt_motion: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, Optional[torch.Tensor], bool]:
        delta_fallback = False
        if isinstance(pred_motion, dict):
            delta_fallback = bool(pred_motion.get('_delta_fallback', False))
            pm = pred_motion.get('out')
            delta_pm = pred_motion.get('delta')
        else:
            pm = pred_motion
            delta_pm = None
        return pm, gt_motion, delta_pm, delta_fallback

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

        self._setdefault_stats(stats, {
            'rot_ortho': 0.0,
            'rot_ortho_weighted': 0.0,
            'rot_ortho_raw': 0.0,
        })
        if delta_fallback and self.w_rot_ortho > 0 and delta_motion is not None:
            l_ortho = self.compute_rot6d_ortho_loss(delta_motion)
            stats['rot_ortho_fallback'] = self._stats_float_or(l_ortho, default=float('nan'))
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
            self._setdefault_stats(stats, {
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

    def _direct_pose_extra_defaults(self) -> Dict[str, float]:
        defaults = self._direct_pose_default_stats()
        defaults.pop('direct_pose_objective', None)
        defaults.pop('direct_pose_weighted', None)
        return defaults

    def _prepare_direct_pose_pair(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
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
        return direct[:, :steps], gt_direct[:, :steps]

    def _compute_direct_pose_group_norm_payload(
        self,
        dir_leg_base: torch.Tensor,
        dir_nonleg_base: torch.Tensor,
        dir_nonleg_effective_base: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        ema_state = getattr(self, '_direct_pose_group_norm_ema', None)
        if not isinstance(ema_state, dict):
            ema_state = {}
        ema_leg_prev = ema_state.get('leg', None)
        ema_non_prev = ema_state.get('nonleg', None)
        if not torch.is_tensor(ema_leg_prev):
            ema_leg_prev = dir_leg_base.detach()
        else:
            ema_leg_prev = ema_leg_prev.to(device=dir_leg_base.device, dtype=dir_leg_base.dtype)
        if not torch.is_tensor(ema_non_prev):
            ema_non_prev = dir_nonleg_effective_base.detach()
        else:
            ema_non_prev = ema_non_prev.to(device=dir_nonleg_effective_base.device, dtype=dir_nonleg_effective_base.dtype)

        leg_ratio_raw_t = dir_leg_base / ema_leg_prev.clamp_min(self.direct_pose_loss_group_norm_eps)
        nonleg_ratio_raw_t = dir_nonleg_effective_base / ema_non_prev.clamp_min(self.direct_pose_loss_group_norm_eps)
        leg_ratio_t = leg_ratio_raw_t.clamp(
            self.direct_pose_loss_group_norm_ratio_min,
            self.direct_pose_loss_group_norm_ratio_max,
        )
        nonleg_ratio_t = nonleg_ratio_raw_t.clamp(
            self.direct_pose_loss_group_norm_ratio_min,
            self.direct_pose_loss_group_norm_ratio_max,
        )
        leg_hit_min_t = (leg_ratio_raw_t <= self.direct_pose_loss_group_norm_ratio_min).to(dtype=dir_leg_base.dtype)
        leg_hit_max_t = (leg_ratio_raw_t >= self.direct_pose_loss_group_norm_ratio_max).to(dtype=dir_leg_base.dtype)
        nonleg_hit_min_t = (nonleg_ratio_raw_t <= self.direct_pose_loss_group_norm_ratio_min).to(dtype=dir_nonleg_base.dtype)
        nonleg_hit_max_t = (nonleg_ratio_raw_t >= self.direct_pose_loss_group_norm_ratio_max).to(dtype=dir_nonleg_base.dtype)
        direct_objective = (
            self.direct_pose_loss_group_norm_w_leg * leg_ratio_t
            + self.direct_pose_loss_group_norm_w_nonleg * nonleg_ratio_t
        )
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
        }
        with torch.no_grad():
            beta = float(self.direct_pose_loss_group_norm_ema_beta)
            self._direct_pose_group_norm_ema = {
                'leg': (beta * ema_leg_prev + (1.0 - beta) * dir_leg_base.detach()).detach(),
                'nonleg': (beta * ema_non_prev + (1.0 - beta) * dir_nonleg_effective_base.detach()).detach(),
            }
        return direct_objective, payload

    def _compute_direct_pose_payload(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> Optional[tuple[torch.Tensor, Dict[str, Any]]]:
        pair = self._prepare_direct_pose_pair(direct, gt_motion)
        if pair is None:
            return None

        direct_seq, gt_direct = pair
        geo_payload = self.compute_rot6d_geo_loss(direct_seq, gt_direct, return_per_joint=True)
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
            'direct_pose_geo_deg': geo_direct * deg_per_rad,
            'direct_pose_split_active': 1.0 if bool(self.direct_pose_loss_leg_split) else 0.0,
            'direct_pose_arm_split_active': 1.0 if bool(self.direct_pose_arm_split_enable) else 0.0,
        })

        if torch.is_tensor(geo_theta) and geo_theta.ndim >= 3:
            split_masks = self._resolve_direct_group_masks(int(geo_theta.shape[-1]), geo_theta.device)
            if split_masks is not None:
                dir_base = self._masked_group_mean(geo_theta, split_masks.get('all_ex_root'))
                dir_leg_base = self._masked_group_mean(geo_theta, split_masks.get('leg'))
                dir_nonleg_base = self._masked_group_mean(geo_theta, split_masks.get('nonleg'))
                dir_arm_base = self._masked_group_mean(geo_theta, split_masks.get('arm'))
                dir_else_base = self._masked_group_mean(geo_theta, split_masks.get('else'))
                dir_nonleg_effective_base = dir_nonleg_base
                arm_else_balance_active = 0.0
                if (
                    bool(self.direct_pose_loss_arm_else_balance_enable)
                    and bool(self.direct_pose_arm_split_enable)
                    and torch.is_tensor(dir_arm_base)
                    and torch.is_tensor(dir_else_base)
                ):
                    arm_w = max(0.0, float(getattr(self, 'direct_pose_loss_arm_weight', 1.0) or 0.0))
                    else_w = max(0.0, float(getattr(self, 'direct_pose_loss_else_weight', 1.0) or 0.0))
                    denom = max(self.direct_pose_loss_group_norm_eps, arm_w + else_w)
                    dir_nonleg_effective_base = (dir_arm_base * arm_w + dir_else_base * else_w) / denom
                    arm_else_balance_active = 1.0

                extra.update({
                    'dir_base': dir_base if torch.is_tensor(dir_base) else float('nan'),
                    'dir_leg_base': dir_leg_base if torch.is_tensor(dir_leg_base) else float('nan'),
                    'dir_nonleg_base': dir_nonleg_base if torch.is_tensor(dir_nonleg_base) else float('nan'),
                    'dir_nonleg_effective_base': dir_nonleg_effective_base if torch.is_tensor(dir_nonleg_effective_base) else float('nan'),
                    'dir_arm_base': dir_arm_base if torch.is_tensor(dir_arm_base) else float('nan'),
                    'dir_else_base': dir_else_base if torch.is_tensor(dir_else_base) else float('nan'),
                    'direct_pose_arm_else_balance_active': arm_else_balance_active,
                })
                if torch.is_tensor(dir_leg_base) and torch.is_tensor(dir_nonleg_base):
                    extra['leg_over_nonleg'] = float(
                        (dir_leg_base / dir_nonleg_base.clamp_min(self.direct_pose_loss_group_norm_eps)).detach().cpu()
                    )
                if torch.is_tensor(dir_arm_base) and torch.is_tensor(dir_else_base):
                    extra['arm_over_else'] = float(
                        (dir_arm_base / dir_else_base.clamp_min(self.direct_pose_loss_group_norm_eps)).detach().cpu()
                    )
                if (
                    bool(self.direct_pose_loss_leg_split)
                    and torch.is_tensor(dir_leg_base)
                    and torch.is_tensor(dir_nonleg_base)
                    and torch.is_tensor(dir_nonleg_effective_base)
                ):
                    extra['leg_over_nonleg_effective'] = float(
                        (dir_leg_base / dir_nonleg_effective_base.clamp_min(self.direct_pose_loss_group_norm_eps)).detach().cpu()
                    )
                    direct_objective = dir_leg_base + dir_nonleg_effective_base
                    if bool(self.direct_pose_loss_group_norm_enable):
                        direct_objective, norm_payload = self._compute_direct_pose_group_norm_payload(
                            dir_leg_base,
                            dir_nonleg_base,
                            dir_nonleg_effective_base,
                        )
                        extra.update(norm_payload)

        return direct_objective, extra

    def _apply_direct_pose_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        if self.w_direct_pose <= 0.0 or not isinstance(pred_motion, dict):
            self._setdefault_stats(stats, self._direct_pose_default_stats())
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

    def _contact_plan_extra_stats(
        self,
        plan: Any,
        logits: torch.Tensor,
        gt: torch.Tensor,
        steps: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        probs = plan if torch.is_tensor(plan) else torch.sigmoid(logits)
        probs = self._ensure_temporal_axis(probs)
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
            stats['event_clock_lambda_mean'] = self._stats_float_or(p.detach().mean(), default=float('nan'))

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

    def _finalize_forward_outputs(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        self._finalize_adaptive_payload(total_loss)
        stats.update(self._loss_group_stats())
        return total_loss, stats

    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        self._init_loss_group_tracker()
        pm, gm, delta_pm, delta_fallback = self._prepare_forward_inputs(pred_motion, gt_motion)
        self._reset_adaptive_tracking()
        loss, stats = self._run_forward_base(pm, gm, attn_weights=attn_weights)
        deg_per_rad = 180.0 / _math.pi
        loss = self._apply_motion_components(loss, stats, pm, gm, delta_pm, delta_fallback, deg_per_rad)
        loss = self._apply_direct_pose_component(loss, stats, pred_motion, gm, deg_per_rad)
        loss = self._apply_aux_components(loss, stats, pred_motion, batch)
        return self._finalize_forward_outputs(loss, stats)

    def _reset_adaptive_tracking(self):
        self._last_component_losses: Dict[str, torch.Tensor] = {}
        self._last_component_weights: Dict[str, float] = {}
        self._last_component_total_weight: float = 0.0
        self._last_core_loss: Optional[torch.Tensor] = None

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
        contrib = self._stats_float_or(tensor.detach() * w, default=0.0)
        if _math.isfinite(contrib):
            self._loss_group_totals[group] += contrib

    def _loss_group_stats(self) -> Dict[str, float]:
        return {f'loss_group/{k}': float(v) for k, v in self._loss_group_totals.items()}

    @staticmethod
    def _stats_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _stats_float_or(value: Any, default: float = 0.0) -> float:
        try:
            return MotionJointLoss._stats_float(value)
        except (RuntimeError, TypeError, ValueError):
            return float(default)

    @staticmethod
    def _ensure_temporal_axis(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        return tensor

    def _prepare_aux_supervision_pair(
        self,
        pred_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        pred = self._ensure_temporal_axis(pred_tensor)
        target = self._ensure_temporal_axis(target_tensor.to(device=pred.device, dtype=pred.dtype))
        steps = min(int(pred.shape[1]), int(target.shape[1]))
        return pred, target, steps

    @staticmethod
    def _setdefault_stats(stats: Dict[str, float], defaults: Dict[str, float]) -> None:
        for key, value in defaults.items():
            stats.setdefault(key, value)

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
            payload[raw_key] = self._stats_float(tensor)
        if weighted_key is not None:
            payload[weighted_key] = self._stats_float(tensor * float(weight))
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload[key] = self._stats_float(value)
        if payload:
            stats.update(payload)
        self._register_component_loss(name, tensor, weight)
        return total_loss

    def _register_component_loss(self, name: str, tensor: Optional[torch.Tensor], weight: float):
        if tensor is None or weight <= 0:
            return
        if name not in self._adaptive_loss_terms:
            return
        self._last_component_losses[name] = tensor
        self._last_component_weights[name] = float(weight)

    def _finalize_adaptive_payload(self, total_loss: torch.Tensor):
        if not self._last_component_losses:
            self._last_core_loss = total_loss
            self._last_component_total_weight = 0.0
            return
        contrib = None
        for name, tensor in self._last_component_losses.items():
            weight = self._last_component_weights.get(name, 0.0)
            if weight <= 0:
                continue
            term = tensor * weight
            contrib = term if contrib is None else contrib + term
        if contrib is None:
            self._last_core_loss = total_loss
            self._last_component_total_weight = 0.0
        else:
            self._last_core_loss = total_loss - contrib
            self._last_component_total_weight = float(
                sum(w for w in self._last_component_weights.values() if w > 0.0)
            )

    def adaptive_loss_payload(self) -> Optional[Dict[str, Any]]:
        if not self._last_component_losses:
            return None
        payload = {
            'losses': dict(self._last_component_losses),
            'weights': dict(self._last_component_weights),
            'total_weight': float(self._last_component_total_weight),
            'core_loss': self._last_core_loss,
        }
        return payload
