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
    matrix_to_rot6d,
    compose_rot6d_delta,
    infer_rot6d_delta_from_abs,
    axis_angle_to_matrix,
    geodesic_R,
    so3_log_map,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    _root_relative_matrices,
    _matrix_log_map,
    normalize_rot6d_delta,
    _rot6d_identity_like,
    wrap_to_pi_np,
    gram_schmidt_renorm_np,
)
from .layout import parse_layout_entry

__all__ = [
    'MotionEncoder',
    'PeriodHead',
    '_CondFiLM',
    'EventMotionModel',
    'MotionJointLoss',
]

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
        bidirectional: bool = False,  # kept for backward compatibility
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

    def __init__(self, hidden_dim: int, out_dim: int, bidirectional: bool = False):
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


import os, json, math, glob, time, argparse

from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable


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
        # ===== Direct Pose Head (cond + contacts_plan -> absolute pose in Y space) =====
        direct_pose_enable: bool = False,
        direct_pose_hidden: int = 256,
        direct_pose_dropout: float = 0.0,
        direct_pose_detach_plan: bool = True,
        # ===== Contact Meas (pose-derived, no physics) =====
        contact_meas_enable: bool = False,
        contact_meas_hidden: int = 64,
        contact_meas_dropout: float = 0.0,
        contact_meas_use_pose_hist: bool = True,
        contact_meas_use_angvel: bool = True,
        # If True, treat `contacts` input as contacts_meas override (debug/legacy).
        # Default False: safe for inference even if caller passes zeros.
        contacts_as_meas_override: bool = False,
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
        self.contacts_as_meas_override = bool(contacts_as_meas_override)
        self.adaptive_history_module: Optional[AdaptiveHistoryModule] = None
        self._adaptive_history_diag: Optional[dict[str, torch.Tensor | float]] = None
        self.pose_hist_len: int = 0
        self._adaptive_history_device: Optional[torch.device] = None
        self.contact_plan_time_pe_dim = int(contact_plan_time_pe_dim or 0)
        if self.contact_plan_time_pe_dim % 2 == 1:
            self.contact_plan_time_pe_dim += 1
        self._contact_plan_time_pe_base = float(contact_plan_time_pe_base or 10000.0)
        self.direct_pose_enable = bool(direct_pose_enable)
        self.direct_pose_hidden = max(8, int(direct_pose_hidden or 0))
        self._direct_pose_dropout = float(direct_pose_dropout)
        self.direct_pose_detach_plan = bool(direct_pose_detach_plan)

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
        self.direct_pose_head: Optional[nn.Module] = None
        if self.contact_plan_enable:
            h_plan = int(self.contact_plan_hidden)
            self.contact_plan_cell = nn.GRUCell(self.cond_dim, h_plan)
            self.contact_plan_head = nn.Sequential(
                nn.LayerNorm(h_plan),
                nn.Linear(h_plan, h_plan),
                nn.ReLU(),
                nn.Dropout(self._contact_plan_dropout),
                nn.Linear(h_plan, self.contact_dim),
            )
            if self.contact_plan_time_pe_dim > 0:
                self.contact_plan_time_head = nn.Linear(self.contact_plan_time_pe_dim, self.contact_dim)
                try:
                    with torch.no_grad():
                        self.contact_plan_time_head.weight.zero_()
                        if getattr(self.contact_plan_time_head, "bias", None) is not None:
                            self.contact_plan_time_head.bias.zero_()
                except Exception:
                    pass

            if self.direct_pose_enable:
                in_dim = int(self.cond_dim + self.contact_dim)
                hid = int(self.direct_pose_hidden)
                drop = float(self._direct_pose_dropout)
                self.direct_pose_head = nn.Sequential(
                    nn.Linear(in_dim, hid),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Dropout(drop) if drop > 0 else nn.Identity(),
                    nn.Linear(hid, int(self.out_motion_dim)),
                )

        # ===== Contact Meas (pose-derived, cheap; no physics) =====
        self.contact_meas_enable = bool(contact_meas_enable and self.contact_dim > 0)
        self.contact_meas_head: Optional[nn.Module] = None
        self._contact_meas_in_dim = 0
        self._contact_meas_dropout = float(contact_meas_dropout)
        if self.contact_meas_enable:
            use_pose_hist = bool(contact_meas_use_pose_hist and self.pose_hist_dim > 0)
            use_angvel = bool(contact_meas_use_angvel and self.angvel_dim > 0)
            meas_in_dim = (self.pose_hist_dim if use_pose_hist else 0) + (self.angvel_dim if use_angvel else 0)
            self._contact_meas_in_dim = int(meas_in_dim)
            if self._contact_meas_in_dim > 0:
                h_meas = max(8, int(contact_meas_hidden))
                self.contact_meas_head = nn.Sequential(
                    nn.LayerNorm(self._contact_meas_in_dim),
                    nn.Linear(self._contact_meas_in_dim, h_meas),
                    nn.ReLU(),
                    nn.Dropout(self._contact_meas_dropout),
                    nn.Linear(h_meas, self.contact_dim),
                )
                try:
                    last = self.contact_meas_head[-1]
                    if isinstance(last, nn.Linear):
                        with torch.no_grad():
                            last.weight.zero_()
                            if last.bias is not None:
                                last.bias.zero_()
                except Exception:
                    pass
            else:
                self.contact_meas_enable = False

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
        self._encoder_meta: dict[str, Any] = {}
        self._frozen_hidden_dim: Optional[int] = None

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
        time_index: Optional[torch.Tensor | int | float] = None,
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

        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(state.shape[:-1] + (self.cond_dim,), device=state.device, dtype=state.dtype)
        if cond is not None and cond.ndim == 2 and state.ndim == 3:
            cond = cond.unsqueeze(1)

        device = state.device
        dtype = state.dtype
        contacts_input = contacts
        contacts_enc = contacts_input
        if angvel is None and self.angvel_dim > 0:
            angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
        if pose_history is None and self.pose_hist_dim > 0:
            pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

        # ---- Contact plan (independent anchor) ----
        # - contacts_plan is produced from cond history via a GRUCell and stays independent of pose.
        # - plan_z is the only cached state needed at inference.
        contacts_plan = None
        plan_z_next = None
        plan_feat_for_inject = None
        if self.contact_plan_enable and self.contact_plan_cell is not None and self.contact_plan_head is not None:
            B, Tq, _ = state.shape
            h_plan = int(self.contact_plan_hidden)
            if plan_z is None:
                plan_z_t = torch.zeros((B, h_plan), device=device, dtype=dtype)
            else:
                plan_z_t = plan_z.to(device=device, dtype=dtype)
                if plan_z_t.ndim == 3 and plan_z_t.size(1) == 1:
                    plan_z_t = plan_z_t[:, 0]
                if plan_z_t.ndim != 2:
                    plan_z_t = plan_z_t.reshape(B, h_plan)
            cond_seq = cond if cond is not None else torch.zeros((B, Tq, self.cond_dim), device=device, dtype=dtype)

            time_pe = None
            if self.contact_plan_time_head is not None and int(self.contact_plan_time_pe_dim) > 0:
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

            plan_probs = []
            plan_z_seq = [] if self.contact_plan_inject == "plan_z" else None
            for _t in range(Tq):
                plan_z_t = self.contact_plan_cell(cond_seq[:, _t], plan_z_t)
                if plan_z_seq is not None:
                    plan_z_seq.append(plan_z_t)
                logits = self.contact_plan_head(plan_z_t)
                if time_pe is not None and self.contact_plan_time_head is not None:
                    try:
                        logits = logits + self.contact_plan_time_head(time_pe[:, _t])
                    except Exception:
                        pass
                plan_probs.append(torch.sigmoid(logits))
            contacts_plan = torch.stack(plan_probs, dim=1)  # (B,T,C)
            plan_z_next = plan_z_t
            if self.contact_plan_inject == "contacts":
                plan_feat_for_inject = contacts_plan
            elif self.contact_plan_inject == "plan_z" and plan_z_seq is not None:
                plan_feat_for_inject = torch.stack(plan_z_seq, dim=1)  # (B,T,H)

        # Use predicted contact plan as a proxy input to the frozen encoder (contact-hint embedding),
        # so we keep train/infer consistent without feeding GT contacts into forward.
        # NOTE: prefer plan whenever available, because deployment may pass zero/unknown contacts.
        if contacts_plan is not None:
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
            if self.adaptive_history_module is not None:
                hist_device = self._adaptive_history_device or pose_hist_for_module.device
                context_feat = state.mean(dim=1).to(hist_device)
                pose_hist_for_module = pose_hist_for_module.to(hist_device)
                pose_hist_flat, diag = self.adaptive_history_module(
                    pose_hist_for_module,
                    context=context_feat,
                )
                pose_history = pose_hist_flat.to(device).unsqueeze(1)
                self._adaptive_history_diag = diag
            encoder_feats.append(pose_history)
        encoder_input = torch.cat(encoder_feats, dim=-1) if encoder_feats else None

        x_inputs = [state]
        if cond is not None:
            x_inputs.append(cond)
        if plan_feat_for_inject is not None:
            feat = plan_feat_for_inject.to(device=device, dtype=dtype)
            if self.contact_plan_inject_detach:
                feat = feat.detach()
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
        soft_period = None
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

        # ---- Direct pose head (cond + contacts_plan -> absolute Y) ----
        if self.direct_pose_head is not None and contacts_plan is not None:
            try:
                plan_in = contacts_plan.detach() if self.direct_pose_detach_plan else contacts_plan
                direct_in = torch.cat([cond, plan_in.to(device=device, dtype=dtype)], dim=-1)
                direct_flat = direct_in.reshape(-1, direct_in.shape[-1])
                direct_out = self.direct_pose_head(direct_flat).view(B, Tq, -1)
                if is_single:
                    direct_out = direct_out.squeeze(1)
                result['out_direct'] = direct_out
            except Exception:
                pass

        # ---- Contact meas (pose-derived) + error signal ----
        e_t = None
        contacts_meas = None
        contacts_override = contacts_input if (self.contacts_as_meas_override and contacts_input is not None) else None
        if contacts_override is not None:
            contacts_meas = contacts_override.to(device=device, dtype=dtype)
            if contacts_meas.ndim == 2:
                contacts_meas = contacts_meas.unsqueeze(1)
        elif self.contact_meas_enable and self.contact_meas_head is not None and self._contact_meas_in_dim > 0:
            meas_feats = []
            if self.pose_hist_dim > 0 and pose_history is not None and pose_history.size(-1) == self.pose_hist_dim:
                meas_feats.append(pose_history)
            if self.angvel_dim > 0 and angvel is not None and angvel.size(-1) == self.angvel_dim:
                meas_feats.append(angvel)
            if meas_feats:
                meas_in = torch.cat(meas_feats, dim=-1)
                flat = meas_in.reshape(-1, meas_in.shape[-1])
                logits = self.contact_meas_head(flat).view(meas_in.shape[0], meas_in.shape[1], -1)
                contacts_meas = torch.sigmoid(logits)
        if contacts_meas is None:
            if contacts_plan is not None:
                contacts_meas = torch.zeros_like(contacts_plan)
            elif self.contact_dim > 0:
                contacts_meas = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)

        if contacts_meas is not None:
            result['contacts_meas'] = contacts_meas.squeeze(1) if is_single else contacts_meas

        if contacts_plan is not None:
            result['contacts_plan'] = contacts_plan.squeeze(1) if is_single else contacts_plan
            if plan_z_next is not None:
                result['plan_z_next'] = plan_z_next
            if contacts_meas is not None:
                # Ensure meas shape aligns with (B,T,C)
                if contacts_meas.ndim == 2:
                    contacts_meas = contacts_meas.unsqueeze(1)
                e_t = contacts_plan - contacts_meas.to(device=device, dtype=dtype)
                result['contacts_err'] = e_t.squeeze(1) if is_single else e_t

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
        加载并冻结预训练的 MotionEncoder + PeriodHead，用于提供 soft hint（接触提示 embedding）。
        """
        if isinstance(bundle, (str, os.PathLike)):
            payload = torch.load(bundle, map_location=map_location)
        else:
            payload = bundle
        if not isinstance(payload, dict):
            raise TypeError("MotionEncoder bundle must be a dict or path to a dict.")

        encoder_state = payload.get('encoder')
        period_state = payload.get('period_head')
        if encoder_state is None or period_state is None:
            raise KeyError("Bundle missing 'encoder' or 'period_head' state_dict.")

        meta = dict(payload.get('meta', {}))
        hint_mode = meta.get("period_hint_mode")
        if hint_mode is None:
            print("[WARN] MotionEncoder bundle meta missing 'period_hint_mode' (expected 'contacts_tanh'); bundle may be legacy.")
        elif str(hint_mode) != "contacts_tanh":
            print(f"[WARN] MotionEncoder bundle period_hint_mode={hint_mode!r} (expected 'contacts_tanh').")
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
            bidirectional=bool(meta.get('bidirectional', False)),
        )
        encoder.load_state_dict(encoder_state)
        encoder.eval().requires_grad_(False)

        period_dim = int(period_state['fc.weight'].shape[0])
        period_head = PeriodHead(hidden_dim, period_dim, bidirectional=bool(meta.get('bidirectional', False)))
        period_head.load_state_dict(period_state)
        period_head.eval().requires_grad_(False)

        if self.encoder_input_dim and self.encoder_input_dim != input_dim:
            raise ValueError(f"Encoder input dim mismatch: dataset={self.encoder_input_dim} vs bundle={input_dim}")
        self.encoder_input_dim = input_dim

        device = self._target_device()
        self.frozen_encoder = encoder.to(device)
        self.frozen_period_head = period_head.to(device)
        self._frozen_hidden_dim = hidden_dim
        self._encoder_meta = meta

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
        ignore_motion_groups: str = '',
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
        # Optional: regularize omega_hat magnitude (prevents aggressive corrections)
        w_omega_l2: float = 0.0,

        # ===== Adaptive bone weighting =====
        adaptive_bone_weights: bool = False,
        bone_prior_stds: Optional[Sequence[float]] = None,
        use_hierarchy_weights: bool = False,
        hierarchy_mode: str = 'multiply',  # 'multiply' | 'add' | 'none'
        hierarchy_alpha: float = 0.5,       # log-space blend between motion / hierarchy
        combine_space: str = 'log',         # 'log' | 'linear'
        max_weight_ratio: float = 50.0,
        weight_gamma: float = 0.7,
    ):
        super().__init__()
        self.meta = dict(meta) if isinstance(meta, dict) else {}
        self.w_attn_reg = float(w_attn_reg)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_local = float(w_rot_local)
        self.w_root_vel = float(w_root_vel)
        self.w_root_speed = float(w_root_speed)
        self.w_contact_plan = float(w_contact_plan)
        self.w_contact_meas = float(w_contact_meas)
        self.w_direct_pose = float(w_direct_pose)
        self.w_omega_l2 = float(w_omega_l2)
        # Tail-risk regularization for per-bone rotation errors (CVaR / top-k style).
        # When enabled, adds an extra term on the worst bones (by mean GeoLocalDeg),
        # which reduces whack-a-mole without requiring explicit per-bone weight tables.
        self.rot_local_tail_weight = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        self.rot_local_tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        self.rot_local_tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all')
        self.rot_local_tail_select = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch')
        self.rot_local_tail_ema_beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        self.angvel_eps = 1e-6
        self.fps = float(fps)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        self._rot6d_columns = self._resolve_rot6d_columns(self.rot6d_spec)
        layout = self.output_layout or {}
        inner = layout.get('slices') if isinstance(layout.get('slices'), dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.ignore_groups = [g.strip() for g in (ignore_motion_groups or '').split(',') if g.strip()]
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
        # 参与 AdaptiveLossWeighting 的 component 项。
        # 注意：
        #   - rot_ortho 仅作为小正则，不参与自适应缩放；
        #   - 目前仅对 rot_local 使用 uncertainty/gradnorm 等策略。
        self._adaptive_loss_terms: Tuple[str, ...] = (
            "rot_local",
        )
        self._reset_adaptive_tracking()
        self._loss_group_totals: Dict[str, float] = {}
        self._loss_group_alias = {
            'attn': 'aux',
            'rot_geo': 'core',
            'rot_ortho': 'core',
            'rot_local': 'core',
            'root_vel': 'core',
            'root_speed': 'core',
            'direct_pose': 'core',
        }

        # === adaptive bone weight params ===
        self.use_adaptive_weights = bool(adaptive_bone_weights)
        self.bone_prior_stds: Optional[torch.Tensor] = None
        if bone_prior_stds is not None:
            try:
                self.bone_prior_stds = torch.as_tensor(bone_prior_stds, dtype=torch.float32)
            except Exception:
                self.bone_prior_stds = None
        self.use_hierarchy_weights = bool(use_hierarchy_weights)
        self.hierarchy_mode = str(hierarchy_mode)
        self.hierarchy_alpha = float(hierarchy_alpha)
        self.combine_space = str(combine_space)
        self.max_weight_ratio = float(max_weight_ratio)
        self.weight_gamma = float(weight_gamma)

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
        """Drop cached joint weights / hierarchy weights when configuration changes."""
        self._joint_weight_cache = {}
        if hasattr(self, '_hierarchy_weights_cache'):
            delattr(self, '_hierarchy_weights_cache')

    def _compute_hierarchy_weights(self) -> Optional[torch.Tensor]:
        """Compute log-scaled descendant counts for each bone.

        Returns:
            Tensor[J] or None if parents are missing.
        """
        if not self.parents:
            return None
        J = len(self.parents)
        counts = torch.zeros(J, dtype=torch.float32)
        for j in range(J):
            ancestor = j
            visited: set[int] = set()
            while 0 <= ancestor < J and ancestor not in visited:
                visited.add(ancestor)
                counts[ancestor] += 1.0
                parent_idx = self.parents[ancestor]
                if not isinstance(parent_idx, int):
                    break
                ancestor = parent_idx
            # root sentinel reached
        # log smoothing, minimum 1.0 for leaves
        weights = torch.log(counts.clamp(min=1.0)) + 1.0
        return weights

    def _load_hierarchy_weights(self) -> Optional[torch.Tensor]:
        cached = getattr(self, '_hierarchy_weights_cache', None)
        if cached is not None:
            return cached
        w = self._compute_hierarchy_weights()
        if w is not None:
            self._hierarchy_weights_cache = w
        return w

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

    def _collect_limb_geo_stats(self, geo_tensor: torch.Tensor) -> Dict[str, float]:
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
            stats['rot_geo_limb_deg'] = float((limb_val * deg).detach().cpu())
            stats['rot_geo_limb_count'] = int(limb_mask.sum().item())
        if torso_mask.any():
            torso_val = joint_mean[torso_mask].mean()
            stats['rot_geo_torso_deg'] = float((torso_val * deg).detach().cpu())
            stats['rot_geo_torso_count'] = int(torso_mask.sum().item())
        if limb_val is not None and torso_val is not None:
            ratio = limb_val / torso_val.clamp_min(1e-6)
            stats['rot_geo_limb_over_torso'] = float(ratio.detach().cpu())
        return stats

    def _geo_weight_monitor(self, geo_tensor: torch.Tensor, weights: torch.Tensor) -> Dict[str, float]:
        """Monitor whether low-weight bones are being ignored.

        Reports mean geodesic error (deg) for low/high weight quartiles and the
        current weight spread. Computed on detached tensors to keep overhead low.
        """
        import torch
        stats: Dict[str, float] = {}
        if geo_tensor is None or weights is None:
            return stats
        if geo_tensor.numel() == 0 or weights.numel() == 0:
            return stats
        # geo_tensor: (..., J), weights: (J,)
        try:
            joint_mean = torch.nanmean(geo_tensor.detach(), dim=tuple(range(geo_tensor.dim() - 1)))
        except Exception:
            return stats
        w = weights.detach()
        if w.numel() != joint_mean.numel():
            return stats
        try:
            low_q = torch.quantile(w, 0.25)
            high_q = torch.quantile(w, 0.75)
        except Exception:
            return stats
        deg = 180.0 / _math.pi
        low_mask = w <= low_q
        high_mask = w >= high_q
        if low_mask.any():
            stats['rot_geo_lowW_deg'] = float((joint_mean[low_mask].mean() * deg).cpu())
        if high_mask.any():
            stats['rot_geo_highW_deg'] = float((joint_mean[high_mask].mean() * deg).cpu())
        spread = (w.max() / w.min().clamp_min(1e-6)).item()
        stats['rot_geo_weight_spread'] = float(spread)
        return stats

    def _parent_relative_matrices(self, R: torch.Tensor) -> torch.Tensor:
        import torch
        parents = getattr(self, 'parents', None)
        if not parents:
            return R
        if not torch.is_tensor(R):
            return R
        J = R.shape[-3]
        if len(parents) < J:
            return R
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != R.device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(parents[:J], device=R.device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        R_rel = torch.empty_like(R)
        for j in range(J):
            p = int(parents_tensor[j].item())
            if p < 0 or p >= J:
                R_rel[..., j, :, :] = R[..., j, :, :]
                continue
            parent = R[..., p, :, :]
            child = R[..., j, :, :]
            R_rel[..., j, :, :] = torch.matmul(parent.transpose(-1, -2), child)
        return R_rel

    def _root_relative(self, R: torch.Tensor) -> torch.Tensor:
        root_idx = int(getattr(self, 'root_idx', 0))
        return _root_relative_matrices(R, root_idx)

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
        - batch: current batch mean (default; backward compatible)
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
                try:
                    w = self._compute_unified_weights_cpu(J)
                    vals = w[torch.as_tensor(leaves, dtype=torch.long)]
                    _, sel = torch.topk(vals, k=min(max_leaf, int(vals.numel())), largest=True, sorted=False)
                    leaves = [leaves[int(i)] for i in sel.tolist()]
                except Exception:
                    leaves = leaves[:max_leaf]

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

        geo_payload = self.compute_rot6d_geo_loss(pm, gm, return_per_joint=True)
        if isinstance(geo_payload, tuple):
            l_geo = geo_payload[0]
            geo_details = geo_payload[1] if len(geo_payload) > 1 else None
            geo_weights = geo_payload[2] if len(geo_payload) > 2 else None
        else:
            l_geo = geo_payload
            geo_details = None
            geo_weights = None

        loss = self.w_attn_reg * l_attn
        self._accumulate_loss_contrib('attn', l_attn, self.w_attn_reg, group='aux')

        stats: Dict[str, float] = {
            'attn': float(l_attn.detach().cpu()),
            'rot_geo': float(l_geo.detach().cpu()),
            'rot_ortho': 0.0,
            'rot_ortho_raw': 0.0,
        }
        if geo_details is not None:
            limb_stats = self._collect_limb_geo_stats(geo_details.detach())
            if limb_stats:
                stats.update(limb_stats)
            if geo_weights is not None:
                weight_stats = self._geo_weight_monitor(geo_details.detach(), geo_weights.detach())
                if weight_stats:
                    stats.update(weight_stats)
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

    @staticmethod
    def _build_ignore_mask(D: int, group_slices: Dict[str, slice], ignore_groups: list, device) -> torch.Tensor:
        """
        返回一个布尔 mask，True=参与计算，False=忽略。
        """
        mask = torch.ones(D, dtype=torch.bool, device=device)
        for g in ignore_groups:
            sl = group_slices.get(g, None)
            if sl is not None:
                mask[sl] = False
        return mask

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

    def compute_rot6d_geo_loss(self, pred: torch.Tensor, gt: torch.Tensor, *, return_per_joint: bool = False):
        Z = lambda v: gt.new_tensor(float(v))
        sl = self.group_slices.get('BoneRotations6D', None)

        # 1) 只取 rot6d 的扁平切片 (…, D)，不要先 reshape 到 (J,6)
        pr = self._maybe_get_rot6d(pred)  # (…, D) or None
        gr = self._maybe_get_rot6d(gt)  # (…, D) or None
        if pr is None or gr is None:
            return Z(0.0)

        D = pr.shape[-1]
        if D % 6 != 0:
            if not self._warned_bad_rot6d:
                self._warned_bad_rot6d = True
                print(
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.")
            return Z(0.0)
        J = D // 6

        # 2) 训练端反归一化：在扁平 (…, D) 上做 raw = z*StdY + MuY
        try:
            sl_b = self.group_slices.get('BoneRotations6D', None)
            if isinstance(sl_b, slice) and getattr(self, "mu_y", None) is not None and getattr(self, "std_y",
                                                                                               None) is not None:
                st = int(sl_b.start);
                ln = int(sl_b.stop - sl_b.start)
                if ln == D:  # 只有当这段就是完整 rot6d 段时才生效
                    mu = torch.as_tensor(self.mu_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln]
                    sd = torch.as_tensor(self.std_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln].clamp(min=1e-6)
                    while mu.dim() < pr.dim():
                        mu = mu.unsqueeze(0);
                        sd = sd.unsqueeze(0)
                    pr = pr * sd + mu
                    gr = gr * sd + mu
                    if not hasattr(self, "_train_denorm_hit"):
                        print("[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.")
                        self._train_denorm_hit = True

        except Exception:
            pass

        # 3) 先在扁平 (…, D) 上做 reproject，再 reshape 到 (…, J, 6)
        pr = reproject_rot6d(pr)  # (…, D)
        gr = reproject_rot6d(gr)  # (…, D)
        pr = pr.view(*pr.shape[:-1], J, 6)  # (…, J, 6)
        gr = gr.view(*gr.shape[:-1], J, 6)  # (…, J, 6)

        # 4) geodesic
        Rp = rot6d_to_matrix(pr)
        Rg = rot6d_to_matrix(gr)
        RtR = torch.matmul(Rp.transpose(-1, -2), Rg)
        tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
        cos = (tr - 1.0) * 0.5
        cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.arccos(cos)  # (..., J)

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
        rot6d = self._maybe_get_rot6d(X)
        if rot6d is None:
            return None
        D = rot6d.shape[-1]
        if D % 6 != 0:
            return None
        J = D // 6

        try:
            sl = self.group_slices.get('BoneRotations6D', None)
            if (
                isinstance(sl, slice)
                and getattr(self, 'mu_y', None) is not None
                and getattr(self, 'std_y', None) is not None
                and (sl.stop - sl.start) == D
            ):
                mu = torch.as_tensor(self.mu_y, device=rot6d.device, dtype=rot6d.dtype)[..., sl]
                std = torch.as_tensor(self.std_y, device=rot6d.device, dtype=rot6d.dtype)[..., sl].clamp(min=1e-6)
                while mu.dim() < rot6d.dim():
                    mu = mu.unsqueeze(0)
                    std = std.unsqueeze(0)
                rot6d = rot6d * std + mu
        except Exception:
            pass

        rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=1.0, neginf=-1.0)
        rot6d = reproject_rot6d(rot6d)
        rot6d = rot6d.view(*rot6d.shape[:-1], J, 6)
        return rot6d_to_matrix(rot6d)

    def _angvel_hz(self) -> float:
        hz = float(getattr(self, 'bone_hz', 0.0) or 0.0)
        if hz <= 0.0:
            dt = float(getattr(self, 'dt_bone', 0.0) or 0.0)
            if dt > 0.0:
                hz = 1.0 / max(dt, 1e-6)
        if hz <= 0.0:
            hz = float(getattr(self, 'fps', 60.0) or 60.0)
        return max(hz, 1e-6)

    def _angular_velocity_from_mats(self, R_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if R_seq is None or R_seq.dim() < 5:
            return None
        T = R_seq.shape[-4]
        if T < 2:
            return None
        lead = R_seq.shape[:-4]
        B = int(_math.prod(lead)) if lead else 1
        J = R_seq.shape[-3]
        mats = R_seq.reshape(B, T, J, 3, 3)
        dR = torch.matmul(mats[:, 1:], mats[:, :-1].transpose(-1, -2))
        vec = _matrix_log_map(dR)
        hz = self._angvel_hz()
        omega = vec * hz
        return omega.reshape(*lead, T - 1, J, 3)

    def _angular_velocity_from_delta(self, delta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if delta is None:
            return None
        rot_delta = self._maybe_get_rot6d(delta)
        if rot_delta is None:
            return None
        D = rot_delta.shape[-1]
        if D % 6 != 0:
            return None
        delta_proj = normalize_rot6d_delta(rot_delta, columns=self._rot6d_columns)
        dR = rot6d_to_matrix(delta_proj, columns=self._rot6d_columns)
        if dR.dim() < 5:
            return None
        lead = dR.shape[:-4]
        B = int(_math.prod(lead)) if lead else 1
        T = dR.shape[-4]
        J = dR.shape[-3]
        mats = dR.reshape(B, T, J, 3, 3)
        vec = _matrix_log_map(mats)
        hz = self._angvel_hz()
        omega = vec * hz
        return omega.reshape(*lead, T, J, 3)

    def _align_angvel_pair(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if pred.shape[:-3] != gt.shape[:-3]:
            return None
        Tp = pred.shape[-3]
        Tg = gt.shape[-3]
        if Tp == 0 or Tg == 0:
            return None
        if Tp == Tg:
            return pred, gt
        if Tp == Tg + 1:
            pred = pred[..., 1:, :, :]
        elif Tg == Tp + 1:
            gt = gt[..., 1:, :, :]
        else:
            L = min(Tp, Tg)
            if L <= 0:
                return None
            pred = pred[..., :L, :, :]
            gt = gt[..., :L, :, :]
        if pred.shape[-3] == 0:
            return None
        return pred, gt

    def _angular_direction_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[torch.Tensor]:
        eps = self.angvel_eps
        norm_p = pred.norm(dim=-1)
        norm_g = gt.norm(dim=-1)
        mask = (norm_p > eps) & (norm_g > eps)
        if not torch.any(mask):
            return None
        denom = (norm_p * norm_g).clamp_min(eps)
        cos = ((pred * gt).sum(dim=-1) / denom).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos)
        return ang[mask].mean()

    def _broadcast_param_slice(self, arr, sl: Optional[slice], ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if arr is None or not isinstance(sl, slice):
            return None
        tensor = torch.as_tensor(arr, device=ref_tensor.device, dtype=ref_tensor.dtype)
        sliced = tensor[..., sl]
        if sliced.numel() == 0:
            return None
        width = sliced.numel()
        view_shape = [1] * (ref_tensor.dim() - 1) + [width]
        return sliced.reshape(*view_shape)

    def _prepare_angvel_payload(
        self,
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        omega_gt = self._angular_velocity_from_mats(self._rot6d_matrices(gt_motion))
        if omega_gt is None:
            return None
        omega_pred = self._angular_velocity_from_delta(delta_motion)
        if omega_pred is None:
            omega_pred = self._angular_velocity_from_mats(self._rot6d_matrices(pred_motion))
        if omega_pred is None:
            return None
        aligned = self._align_angvel_pair(omega_pred, omega_gt)
        if aligned is None:
            return None
        return aligned

    def compute_rot6d_log_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._rot6d_matrices(pred)
        Rg = self._rot6d_matrices(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 4:
            return Z(0.0)
        T = Rp.shape[-3]
        J = Rp.shape[-2]
        Rp = Rp.reshape(-1, T, J, 3, 3)
        Rg = Rg.reshape(-1, T, J, 3, 3)
        dRp = torch.matmul(Rp[:, 1:], Rp[:, :-1].transpose(-1, -2))
        dRg = torch.matmul(Rg[:, 1:], Rg[:, :-1].transpose(-1, -2))
        log_p = _matrix_log_map(dRp)
        log_g = _matrix_log_map(dRg)
        return torch.nn.functional.smooth_l1_loss(log_p, log_g)


    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        # 统一拿出模型输出（可能是 dict 或 tensor）
        self._init_loss_group_tracker()
        delta_fallback = False
        if isinstance(pred_motion, dict):
            delta_fallback = bool(pred_motion.get('_delta_fallback', False))
        pm = pred_motion.get('out') if isinstance(pred_motion, dict) else pred_motion
        gm = gt_motion
        delta_pm = pred_motion.get('delta') if isinstance(pred_motion, dict) else None
        self._reset_adaptive_tracking()

        # _forward_base_inner 已包含核心动作损失与统计
        base_out = self._forward_base_inner(pm, gt_motion, attn_weights=attn_weights)  # type: ignore
        if isinstance(base_out, tuple):
            loss, stats = base_out
        else:
            loss, stats = base_out, {}

        if isinstance(stats, dict):
            stats = dict(stats)
        else:
            stats = {}

        if self.w_rot_ortho > 0 and not delta_fallback:
            target_for_ortho = delta_pm if delta_pm is not None else pm
            l_ortho = self.compute_rot6d_ortho_loss(target_for_ortho)
            weighted_ortho = self.w_rot_ortho * l_ortho
            loss = loss + weighted_ortho
            self._accumulate_loss_contrib('rot_ortho', l_ortho, self.w_rot_ortho, group='core')
            stats['rot_ortho'] = float(l_ortho.detach().cpu())
            stats['rot_ortho_weighted'] = float(weighted_ortho.detach().cpu())
            stats['rot_ortho_raw'] = float(l_ortho.detach().cpu())
            self._register_component_loss('rot_ortho', l_ortho, self.w_rot_ortho)
        else:
            stats.setdefault('rot_ortho', 0.0)
            stats.setdefault('rot_ortho_weighted', 0.0)
            stats.setdefault('rot_ortho_raw', 0.0)

        if delta_fallback and self.w_rot_ortho > 0 and delta_pm is not None:
            # 即使跳过 rot_ortho，也在 stats 中记录原生值方便诊断
            try:
                l_ortho = self.compute_rot6d_ortho_loss(delta_pm)
                stats['rot_ortho_fallback'] = float(l_ortho.detach().cpu())
            except Exception:
                stats['rot_ortho_fallback'] = float('nan')


        Rp_world = Rg_world = None
        Rp_root = Rg_root = None
        if self.w_rot_local > 0.0:
            Rp_world = self._rot6d_matrices(pm)
            Rg_world = self._rot6d_matrices(gm)
            if Rp_world is not None and Rg_world is not None:
                Rp_root = self._root_relative(Rp_world)
                Rg_root = self._root_relative(Rg_world)

        if self.w_rot_local > 0.0:
            if Rp_root is not None and Rg_root is not None:
                Rp_local = self._parent_relative_matrices(Rp_root)
                Rg_local = self._parent_relative_matrices(Rg_root)
                geo_local = geodesic_R(Rp_local, Rg_local)
                weights = self._joint_weight_vector(Rp_local.device, Rp_local.dtype, Rp_local.shape[-3])
                w = weights.view(1, 1, -1)
                local_loss = (geo_local * w).mean()
                loss = loss + self.w_rot_local * local_loss
                self._accumulate_loss_contrib('rot_local', local_loss, self.w_rot_local, group='core')
                stats['rot_local_deg'] = float((local_loss * (180.0 / math.pi)).detach().cpu())
                self._register_component_loss('rot_local', local_loss, self.w_rot_local)

                # Optional tail loss on worst bones (unweighted selection; gradients flow to selected bones).
                tail_w = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
                tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
                tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all').lower()
                J = int(geo_local.shape[-1])
                if tail_w > 0.0 and tail_k > 0 and J > 0:
                    k = min(max(1, tail_k), J)

                    cand_idx = self._rot_local_tail_candidates(tail_scope, J, geo_local.device, k=k)
                    per_bone = torch.nanmean(geo_local.detach(), dim=tuple(range(geo_local.dim() - 1)))  # (J,)
                    scores = self._rot_local_tail_scores(per_bone)
                    select_mode = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch').lower()
                    try:
                        if cand_idx is not None and cand_idx.numel() > 0:
                            vals = scores.index_select(0, cand_idx)
                            _, sel = torch.topk(vals, k=k, largest=True, sorted=False)
                            idx = cand_idx.index_select(0, sel)
                        else:
                            _, idx = torch.topk(scores, k=k, largest=True, sorted=False)
                        tail_loss = torch.nanmean(geo_local.index_select(-1, idx))
                        loss = loss + tail_w * tail_loss
                        self._accumulate_loss_contrib('rot_local_tail', tail_loss, tail_w, group='core')
                        stats['rot_local_tail_deg'] = float((tail_loss * (180.0 / math.pi)).detach().cpu())
                        stats['rot_local_tail_k'] = float(k)
                        stats['rot_local_tail_scope'] = float({'all': 0.0, 'limbs': 1.0, 'keybones': 2.0}.get(tail_scope, 0.0))
                        stats['rot_local_tail_select'] = float({'batch': 0.0, 'ema': 1.0}.get(select_mode, 0.0))
                        self._register_component_loss('rot_local_tail', tail_loss, tail_w)
                    except Exception:
                        pass
        else:
            stats.setdefault('rot_local_deg', 0.0)

        # Root velocity losses（向量 + 模长），对应输出布局中的 RootVelocity 切片
        rv_slice = self.group_slices.get('RootVelocity')
        if rv_slice is not None and (self.w_root_vel > 0.0 or self.w_root_speed > 0.0):
            pred_vel = pm[..., rv_slice]
            gt_vel = gm[..., rv_slice]
            if self.w_root_vel > 0.0:
                vel_mse = F.mse_loss(pred_vel, gt_vel)
                loss = loss + self.w_root_vel * vel_mse
                self._accumulate_loss_contrib('root_vel', vel_mse, self.w_root_vel, group='core')
                stats['root_vel_mse'] = float(vel_mse.detach().cpu())
                self._register_component_loss('root_vel', vel_mse, self.w_root_vel)
            else:
                stats.setdefault('root_vel_mse', 0.0)

            if self.w_root_speed > 0.0:
                pred_speed = torch.norm(pred_vel, dim=-1)
                gt_speed = torch.norm(gt_vel, dim=-1)
                speed_mae = F.l1_loss(pred_speed, gt_speed)
                loss = loss + self.w_root_speed * speed_mae
                self._accumulate_loss_contrib('root_speed', speed_mae, self.w_root_speed, group='core')
                stats['root_speed_mae'] = float(speed_mae.detach().cpu())
                self._register_component_loss('root_speed', speed_mae, self.w_root_speed)
            else:
                stats.setdefault('root_speed_mae', 0.0)
        else:
            stats.setdefault('root_vel_mse', 0.0)
            stats.setdefault('root_speed_mae', 0.0)

        # Direct pose supervision (cond + contacts_plan -> absolute pose)
        if self.w_direct_pose > 0.0 and isinstance(pred_motion, dict):
            try:
                direct = pred_motion.get('out_direct', None)
                if torch.is_tensor(direct):
                    if direct.dim() == 2 and gm.dim() == 3:
                        direct = direct.unsqueeze(1)
                    gm_direct = gm
                    if gm_direct.dim() == 2 and direct.dim() == 3:
                        gm_direct = gm_direct.unsqueeze(1)
                    if direct.dim() == 3 and gm_direct.dim() == 3:
                        T = min(int(direct.shape[1]), int(gm_direct.shape[1]))
                        if T > 0:
                            l_direct = self.compute_rot6d_geo_loss(direct[:, :T], gm_direct[:, :T])
                            loss = loss + self.w_direct_pose * l_direct
                            self._accumulate_loss_contrib('direct_pose', l_direct, self.w_direct_pose, group='core')
                            stats['direct_pose_geo'] = float(l_direct.detach().cpu())
                            stats['direct_pose_geo_deg'] = float((l_direct * (180.0 / math.pi)).detach().cpu())
                            stats['direct_pose_weighted'] = float((self.w_direct_pose * l_direct).detach().cpu())
                            self._register_component_loss('direct_pose', l_direct, self.w_direct_pose)
            except Exception:
                pass
        else:
            stats.setdefault('direct_pose_geo', 0.0)
            stats.setdefault('direct_pose_geo_deg', 0.0)
            stats.setdefault('direct_pose_weighted', 0.0)

        # Contact plan supervision (soft targets in [0,1])
        if self.w_contact_plan > 0.0 and isinstance(pred_motion, dict) and isinstance(batch, dict):
            try:
                plan = pred_motion.get('contacts_plan', None)
                gt_c = batch.get('contacts', None)
                if torch.is_tensor(plan) and torch.is_tensor(gt_c):
                    gt_c = gt_c.to(device=plan.device, dtype=plan.dtype)
                    if plan.dim() == 2:
                        plan = plan.unsqueeze(1)
                    if gt_c.dim() == 2:
                        gt_c = gt_c.unsqueeze(1)
                    T = min(int(plan.shape[1]), int(gt_c.shape[1]))
                    if T > 0 and plan.shape[-1] == gt_c.shape[-1]:
                        l_contact = F.mse_loss(plan[:, :T], gt_c[:, :T])
                        loss = loss + self.w_contact_plan * l_contact
                        stats['contact_plan_mse'] = float(l_contact.detach().cpu())
                        stats['contact_plan_weighted'] = float((self.w_contact_plan * l_contact).detach().cpu())
            except Exception:
                pass

        # Contact meas supervision (optional, small weight; keeps meas head sane)
        if self.w_contact_meas > 0.0 and isinstance(pred_motion, dict) and isinstance(batch, dict):
            try:
                meas = pred_motion.get('contacts_meas', None)
                gt_c = batch.get('contacts', None)
                if torch.is_tensor(meas) and torch.is_tensor(gt_c):
                    gt_c = gt_c.to(device=meas.device, dtype=meas.dtype)
                    if meas.dim() == 2:
                        meas = meas.unsqueeze(1)
                    if gt_c.dim() == 2:
                        gt_c = gt_c.unsqueeze(1)
                    T = min(int(meas.shape[1]), int(gt_c.shape[1]))
                    if T > 0 and meas.shape[-1] == gt_c.shape[-1]:
                        l_meas = F.mse_loss(meas[:, :T], gt_c[:, :T])
                        loss = loss + self.w_contact_meas * l_meas
                        stats['contact_meas_mse'] = float(l_meas.detach().cpu())
                        stats['contact_meas_weighted'] = float((self.w_contact_meas * l_meas).detach().cpu())
            except Exception:
                pass

        # Omega regularization (optional)
        if self.w_omega_l2 > 0.0 and isinstance(pred_motion, dict):
            try:
                omega = pred_motion.get('omega_hat', None)
                if torch.is_tensor(omega) and omega.numel() > 0:
                    if omega.dim() == 3:  # (B,J,3)
                        omega = omega.unsqueeze(1)
                    l2 = (omega * omega).mean()
                    loss = loss + self.w_omega_l2 * l2
                    stats['omega_l2'] = float(l2.detach().cpu())
                    stats['omega_l2_weighted'] = float((self.w_omega_l2 * l2).detach().cpu())
            except Exception:
                pass

        self._finalize_adaptive_payload(loss)
        stats.update(self._loss_group_stats())
        return loss, stats

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
        except Exception:
            w = float(weight.item()) if hasattr(weight, 'item') else 0.0
        if not _math.isfinite(w) or abs(w) < 1e-9:
            return
        if group is None:
            group = self._loss_group_alias.get(name, 'core')
        if group not in self._loss_group_totals:
            self._loss_group_totals[group] = 0.0
        try:
            contrib = float((tensor.detach().cpu()) * w)
        except Exception:
            contrib = 0.0
        if _math.isfinite(contrib):
            self._loss_group_totals[group] += contrib

    def _loss_group_stats(self) -> Dict[str, float]:
        return {f'loss_group/{k}': float(v) for k, v in self._loss_group_totals.items()}

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
