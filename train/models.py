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
    """Lightweight linear head used during pretraining to predict soft period hints."""

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
        self.adaptive_history_module: Optional[AdaptiveHistoryModule] = None
        self._adaptive_history_diag: Optional[dict[str, torch.Tensor | float]] = None
        self.pose_hist_len: int = 0
        self._adaptive_history_device: Optional[torch.device] = None

        input_dim = self.in_state_dim + self.cond_dim
        self.shared_encoder = build_mlp(
            input_dim,
            hidden_dim,
            num_layers=2,
            activation=nn.ReLU,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
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

        # Optional frozen encoder from预训练，用于提供 soft period 提示
        self.frozen_encoder: Optional['MotionEncoder'] = None
        self.frozen_period_head: Optional['PeriodHead'] = None
        self._encoder_meta: dict[str, Any] = {}
        self._frozen_hidden_dim: Optional[int] = None

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

        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(state.shape[:-1] + (self.cond_dim,), device=state.device, dtype=state.dtype)
        if cond is not None and cond.ndim == 2 and state.ndim == 3:
            cond = cond.unsqueeze(1)

        device = state.device
        dtype = state.dtype
        if contacts is None and self.contact_dim > 0:
            contacts = torch.zeros(state.shape[:-1] + (self.contact_dim,), device=device, dtype=dtype)
        if angvel is None and self.angvel_dim > 0:
            angvel = torch.zeros(state.shape[:-1] + (self.angvel_dim,), device=device, dtype=dtype)
        if pose_history is None and self.pose_hist_dim > 0:
            pose_history = torch.zeros(state.shape[:-1] + (self.pose_hist_dim,), device=device, dtype=dtype)

        encoder_feats = []
        if contacts is not None and contacts.size(-1) > 0:
            encoder_feats.append(contacts)
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
        x = torch.cat(x_inputs, dim=-1)
        if not torch.isfinite(x).all():
            bad_mask = (~torch.isfinite(x))
            _nz = bad_mask.nonzero(as_tuple=False)
            bad = _nz[0].tolist() if _nz.numel() > 0 else []
            s_ok = bool(torch.isfinite(state).all())
            c_ok = True
            if cond is not None and cond.is_floating_point():
                c_ok = bool(torch.isfinite(cond).all())
            raise RuntimeError(f'[Guard] x to shared_encoder has non-finite at {bad} | state_finite={s_ok} cond_finite={c_ok}')
        _lin0 = self.shared_encoder[0]
        _w_ok = bool(torch.isfinite(_lin0.weight).all())
        _b_ok = _lin0.bias is None or bool(torch.isfinite(_lin0.bias).all())
        if not _w_ok or not _b_ok:
            raise RuntimeError('[Guard] shared_encoder.0 has non-finite params; refusing to auto-reinit in forward')
        x = torch.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        clip_val = float(getattr(self, 'input_clip', 16.0) or 16.0)
        x = x.clamp(-clip_val, clip_val)
        with torch.no_grad():
            for _idx, _mod in enumerate(self.shared_encoder):
                if isinstance(_mod, torch.nn.Linear):
                    _ok_w = bool(torch.isfinite(_mod.weight).all())
                    _ok_b = _mod.bias is None or bool(torch.isfinite(_mod.bias).all())
                    if not _ok_w or not _ok_b:
                        raise RuntimeError(f"[Guard] shared_encoder.{_idx} has non-finite params; refusing to auto-reinit in forward")

        lin0 = self.shared_encoder[0]
        act1 = self.shared_encoder[1]
        z0 = lin0(x)
        y1 = act1(z0)

        # Inject soft period embedding from frozen encoder (if available)
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
        if soft_period is not None:
            result['period_pred'] = soft_period
        return result

    def attach_motion_encoder(self, bundle, *, map_location: str | torch.device = 'cpu'):
        """
        加载并冻结预训练的 MotionEncoder + PeriodHead，用于提供 soft period 提示。
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
        w_rot_delta: float = 1.0,
        w_rot_delta_root: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        w_fk_pos: float = 0.0,
        w_rot_local: float = 0.0,
    ):
        super().__init__()
        self.meta = dict(meta) if isinstance(meta, dict) else {}
        self.w_attn_reg = float(w_attn_reg)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_delta = float(w_rot_delta)
        self.w_rot_delta_root = float(w_rot_delta_root)
        self.w_fk_pos = float(w_fk_pos)
        self.w_rot_local = float(w_rot_local)
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
        self._joint_weight_cache: dict[tuple[str, str, int], torch.Tensor] = {}
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
        self.has_fk = bool(self.parents and self.bone_offsets is not None)
        self._adaptive_loss_terms: Tuple[str, ...] = (
            "fk_pos",
            "rot_local",
            "rot_delta",
            "rot_ortho",
        )
        self._reset_adaptive_tracking()
        self._loss_group_totals: Dict[str, float] = {}
        self._loss_group_alias = {
            'attn': 'aux',
            'rot_geo': 'core',
            'rot_delta': 'core',
            'rot_delta_root': 'aux',
            'rot_ortho': 'core',
            'fk_pos': 'core',
            'rot_local': 'core',
        }

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
        self._limb_mask_cache = None
        self._torso_mask_cache = None
        self._limb_mask_joint_count = None
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
        self.has_fk = bool(self.parents and self.bone_offsets is not None)

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

    def _joint_weight_vector(self, device, dtype, joint_count: int) -> torch.Tensor:
        key = (str(device), str(dtype), int(joint_count))
        cache = getattr(self, '_joint_weight_cache', None)
        if cache is None:
            cache = {}
            self._joint_weight_cache = cache
        if key in cache:
            return cache[key]
        import torch
        weights = torch.ones(joint_count, device=device, dtype=dtype)
        cache[key] = weights
        return weights

    def _fk_positions(self, R_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if R_seq is None:
            return None
        if not getattr(self, 'has_fk', False):
            return None
        import torch
        J = R_seq.shape[-3]
        if len(self.parents) < J or self.bone_offsets.shape[0] < J:
            return None
        device = R_seq.device
        dtype = R_seq.dtype
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(self.parents[:J], device=device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        offsets = self.bone_offsets.to(device=device, dtype=dtype)
        B, T = R_seq.shape[:2]
        world_rot = torch.empty_like(R_seq)
        world_pos = torch.zeros(B, T, J, 3, device=device, dtype=dtype)
        for j in range(J):
            parent = int(parents_tensor[j].item())
            rot_j = R_seq[..., j, :, :]
            if parent < 0 or parent >= J:
                world_rot[..., j, :, :] = rot_j
                continue
            parent_rot = world_rot[..., parent, :, :].clone()
            world_rot[..., j, :, :] = torch.matmul(parent_rot, rot_j)
            offset = offsets[j].view(1, 1, 3, 1)
            delta = torch.matmul(parent_rot, offset).squeeze(-1)
            parent_pos = world_pos[..., parent, :].clone()
            world_pos[..., j, :] = parent_pos + delta
        return world_pos

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
            l_geo, geo_details = geo_payload
        else:
            l_geo = geo_payload
            geo_details = None

        loss = self.w_attn_reg * l_attn
        self._accumulate_loss_contrib('attn', l_attn, self.w_attn_reg, group='aux')

        stats: Dict[str, float] = {
            'attn': float(l_attn.detach().cpu()),
            'rot_geo': float(l_geo.detach().cpu()),
            'rot_delta': 0.0,
            'rot_ortho': 0.0,
            'rot_ortho_raw': 0.0,
        }
        if geo_details is not None:
            limb_stats = self._collect_limb_geo_stats(geo_details.detach())
            if limb_stats:
                stats.update(limb_stats)
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
        theta = torch.arccos(cos)
        if return_per_joint:
            return theta.mean(), theta
        return theta.mean()


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

    def compute_rot6d_delta_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        rot_delta = self._maybe_get_rot6d(pred)
        if rot_delta is None:
            return Z(0.0)
        D = rot_delta.shape[-1]
        if D % 6 != 0:
            return Z(0.0)
        J = D // 6

        delta_proj = normalize_rot6d_delta(rot_delta, columns=("X", "Z"))
        dRp = rot6d_to_matrix(delta_proj, columns=("X", "Z"))
        if dRp.dim() < 5:
            return Z(0.0)

        Rg = self._rot6d_matrices(gt)
        if Rg is None or Rg.dim() < 5:
            return Z(0.0)

        T_pred = dRp.shape[-4]
        T_gt = Rg.shape[-4]
        if T_gt < 2 or T_pred < 1:
            return Z(0.0)

        lead_pred = dRp.shape[:-4]
        lead_gt = Rg.shape[:-4]
        Bp = int(_math.prod(lead_pred)) if lead_pred else 1
        Bg = int(_math.prod(lead_gt)) if lead_gt else 1
        dRp = dRp.reshape(Bp, T_pred, J, 3, 3)
        Rg = Rg.reshape(Bg, T_gt, J, 3, 3)

        dRg = torch.matmul(Rg[:, 1:], Rg[:, :-1].transpose(-1, -2))

        if dRp.shape[1] == dRg.shape[1] + 1:
            dRp = dRp[:, 1:]
        elif dRp.shape[1] != dRg.shape[1]:
            L = min(dRp.shape[1], dRg.shape[1])
            dRp = dRp[:, :L]
            dRg = dRg[:, :L]

        if dRp.shape[1] == 0:
            return Z(0.0)

        theta = geodesic_R(dRp, dRg)
        theta = torch.nan_to_num(theta, nan=0.0, posinf=_math.pi, neginf=0.0)
        return theta.mean()

    def compute_root_geodesic_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._rot6d_matrices(pred)
        Rg = self._rot6d_matrices(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 4:
            return Z(0.0)
        if Rp.dim() < 5:
            return Z(0.0)
        T = int(Rp.shape[-4])
        J = int(Rp.shape[-3])
        if T <= 0 or J <= 0:
            return Z(0.0)
        Rp = Rp.reshape(-1, T, J, 3, 3)
        Rg = Rg.reshape(-1, T, J, 3, 3)
        root_idx = int(getattr(self, 'root_idx', 0))
        root_idx = max(0, min(J - 1, root_idx))
        Rp_root = Rp[:, :, root_idx]
        Rg_root = Rg[:, :, root_idx]
        theta = geodesic_R(Rp_root, Rg_root)
        theta = torch.nan_to_num(theta, nan=0.0, posinf=_math.pi, neginf=0.0)
        return theta.mean()

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

        if self.w_rot_delta > 0 and delta_pm is not None:
            l_delta = self.compute_rot6d_delta_loss(delta_pm, gt_motion)
            loss = loss + self.w_rot_delta * l_delta
            self._accumulate_loss_contrib('rot_delta', l_delta, self.w_rot_delta, group='core')
            stats['rot_delta'] = float(l_delta.detach().cpu())
            self._register_component_loss('rot_delta', l_delta, self.w_rot_delta)
        else:
            stats.setdefault('rot_delta', 0.0)

        if self.w_rot_delta_root > 0:
            l_root_geo = self.compute_root_geodesic_loss(pm, gt_motion)
            loss = loss + self.w_rot_delta_root * l_root_geo
            self._accumulate_loss_contrib('rot_delta_root', l_root_geo, self.w_rot_delta_root, group='aux')
            stats['rot_delta_root'] = float(l_root_geo.detach().cpu())
        else:
            stats.setdefault('rot_delta_root', 0.0)

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
        if self.w_fk_pos > 0.0 or self.w_rot_local > 0.0:
            Rp_world = self._rot6d_matrices(pm)
            Rg_world = self._rot6d_matrices(gm)
            if Rp_world is not None and Rg_world is not None:
                Rp_root = self._root_relative(Rp_world)
                Rg_root = self._root_relative(Rg_world)

        if self.w_fk_pos > 0.0 and self.has_fk:
            if Rp_root is not None and Rg_root is not None:
                pos_pred = self._fk_positions(Rp_root)
                pos_gt = self._fk_positions(Rg_root)
                if pos_pred is not None and pos_gt is not None:
                    fk_res = F.smooth_l1_loss(pos_pred, pos_gt, reduction='none').mean(dim=-1)
                    weights = self._joint_weight_vector(pos_pred.device, pos_pred.dtype, pos_pred.shape[-2])
                    w = weights.view(1, 1, -1)
                    fk_loss = (fk_res * w).mean()
                    loss = loss + self.w_fk_pos * fk_loss
                    self._accumulate_loss_contrib('fk_pos', fk_loss, self.w_fk_pos, group='core')
                    stats['fk_pos'] = float(fk_loss.detach().cpu())
                    self._register_component_loss('fk_pos', fk_loss, self.w_fk_pos)
        else:
            stats.setdefault('fk_pos', 0.0)

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
        else:
            stats.setdefault('rot_local_deg', 0.0)

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
