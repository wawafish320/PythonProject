from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalizers import VectorTanhNormalizerTorch


PoseHistParamsFn = Callable[
    [torch.Tensor],
    tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
]


@dataclass
class PoseHistState:
    enabled: bool
    length: int
    dim: int
    stride: int
    scales: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    buffer_norm: Optional[torch.Tensor] = None
    buffer_raw: Optional[torch.Tensor] = None


def pose_hist_transform_vec(
    raw_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if scales is None or raw_flat.numel() == 0:
        return raw_flat
    norm = VectorTanhNormalizerTorch(scales, mu, std)
    norm = norm.to(device=raw_flat.device, dtype=raw_flat.dtype)
    return norm(raw_flat)


def pose_hist_inverse_vec(
    norm_flat: torch.Tensor,
    scales: Optional[torch.Tensor],
    mu: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if scales is None or norm_flat.numel() == 0:
        return norm_flat
    norm = VectorTanhNormalizerTorch(scales, mu, std)
    norm = norm.to(device=norm_flat.device, dtype=norm_flat.dtype)
    return norm.inverse(norm_flat)


def init_pose_hist_state(
    *,
    ref_tensor: torch.Tensor,
    pose_hist_seq: Optional[torch.Tensor],
    y_prev_raw: Optional[torch.Tensor],
    rot_slice: Optional[slice],
    pose_hist_len: int,
    pose_hist_dim: int,
    params_fn: PoseHistParamsFn,
    offset: int = 0,
    force_disable: bool = False,
    require_rot_slice_for_fallback: bool = False,
) -> PoseHistState:
    pose_hist_len = int(pose_hist_len)
    pose_hist_dim = int(pose_hist_dim)
    if pose_hist_len <= 0 or pose_hist_dim <= 0:
        return PoseHistState(enabled=False, length=pose_hist_len, dim=pose_hist_dim, stride=0)
    if pose_hist_dim % pose_hist_len != 0:
        raise ValueError("pose_hist_dim must be divisible by pose_hist_len")

    state = PoseHistState(
        enabled=not bool(force_disable),
        length=pose_hist_len,
        dim=pose_hist_dim,
        stride=pose_hist_dim // pose_hist_len,
    )
    if not state.enabled:
        return state

    scales, mu, std = params_fn(ref_tensor)
    if scales is None:
        state.enabled = False
        return state
    state.scales = scales
    state.mu = mu
    state.std = std

    initial_norm = None
    if torch.is_tensor(pose_hist_seq) and pose_hist_seq.numel() > 0:
        seq = pose_hist_seq.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if seq.dim() == 3:
            idx = int(max(0, min(int(seq.shape[1]) - 1, int(offset))))
            initial_norm = seq[:, idx]
        else:
            initial_norm = seq

    with torch.no_grad():
        if initial_norm is not None:
            state.buffer_norm = initial_norm
            state.buffer_raw = pose_hist_inverse_vec(initial_norm, scales, mu, std)
            return state

        if require_rot_slice_for_fallback and not isinstance(rot_slice, slice):
            raise RuntimeError("pose_hist enabled but rot slice missing for fallback init")

        if (not torch.is_tensor(y_prev_raw)) or (not isinstance(rot_slice, slice)):
            state.enabled = False
            return state

        base_rot = y_prev_raw[..., rot_slice]
        state.buffer_raw = (
            base_rot.unsqueeze(1)
            .repeat(1, pose_hist_len, 1)
            .reshape(base_rot.shape[0], pose_hist_dim)
        )
        state.buffer_norm = pose_hist_transform_vec(
            state.buffer_raw,
            scales,
            mu,
            std,
        )
    return state


def resolve_pose_hist_input(
    *,
    state: PoseHistState,
    pose_hist_seq: Optional[torch.Tensor],
    idx: int,
) -> Optional[torch.Tensor]:
    if state.enabled and state.buffer_norm is not None:
        return state.buffer_norm
    if (not torch.is_tensor(pose_hist_seq)) or pose_hist_seq.numel() == 0:
        return None
    if pose_hist_seq.dim() == 3:
        step_idx = int(max(0, min(int(pose_hist_seq.shape[1]) - 1, int(idx))))
        return pose_hist_seq[:, step_idx]
    return pose_hist_seq


def advance_pose_hist_state(
    state: PoseHistState,
    *,
    y_next_raw: torch.Tensor,
    rot_slice: Optional[slice],
) -> PoseHistState:
    if (
        (not state.enabled)
        or state.stride <= 0
        or state.buffer_raw is None
        or (not torch.is_tensor(y_next_raw))
        or (not isinstance(rot_slice, slice))
    ):
        return state

    return advance_pose_hist_state_with_tail(
        state,
        rot_tail_raw=y_next_raw[..., rot_slice],
    )


def advance_pose_hist_state_with_tail(
    state: PoseHistState,
    *,
    rot_tail_raw: Optional[torch.Tensor],
) -> PoseHistState:
    if (
        (not state.enabled)
        or state.stride <= 0
        or state.buffer_raw is None
        or (not torch.is_tensor(rot_tail_raw))
    ):
        return state

    with torch.no_grad():
        next_buffer_raw = torch.roll(state.buffer_raw, shifts=-state.stride, dims=-1)
        next_buffer_raw[..., -state.stride:] = rot_tail_raw
        next_buffer_norm = pose_hist_transform_vec(
            next_buffer_raw,
            state.scales,
            state.mu,
            state.std,
        )
    return PoseHistState(
        enabled=state.enabled,
        length=state.length,
        dim=state.dim,
        stride=state.stride,
        scales=state.scales,
        mu=state.mu,
        std=state.std,
        buffer_norm=next_buffer_norm,
        buffer_raw=next_buffer_raw,
    )


class AdaptiveHistoryModule(nn.Module):
    """
    Attention-style adapter that lets the model observe a longer history window
    during training while exporting a fixed-length summary for deployment.
    """

    def __init__(
        self,
        pose_dim: int,
        hidden_dim: int,
        num_history_frames: int,
        *,
        max_history_frames: Optional[int] = None,
        cond_dim: int = 0,
        num_heads: int = 2,
        use_gate: bool = True,
        train_variable_history: bool = True,
        history_dropout_prob: float = 0.0,
        use_trend_features: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_history_frames <= 0:
            raise ValueError("num_history_frames must be > 0.")
        self.pose_dim = int(pose_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_history_frames)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_history_frames = int(max_history_frames or num_history_frames)
        self.train_variable_history = bool(train_variable_history)
        self.use_gate = bool(use_gate)
        self.history_dropout_prob = float(history_dropout_prob)
        self.use_trend_features = bool(use_trend_features)

        self.frame_proj = nn.Linear(self.pose_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.pose_dim)
        if self.use_trend_features:
            self.trend_proj = nn.Linear(self.pose_dim * 2, self.hidden_dim)
            self.trend_norm = nn.LayerNorm(self.hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(self.num_slots, self.hidden_dim))
        self.context_proj = nn.Linear(cond_dim, self.hidden_dim) if cond_dim > 0 else None
        self.gate_proj = nn.Linear(self.hidden_dim, self.hidden_dim) if self.use_gate else None
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self._last_diag: Dict[str, torch.Tensor | float] = {}

    def forward(
        self,
        pose_history: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | float]]:
        """
        Args:
            pose_history: Tensor shaped [B, P] or [B, L, pose_dim].
            context: optional tensor [B, D] used to modulate queries.
        Returns:
            aggregated pose history of shape [B, num_history_frames * pose_dim],
            diagnostics dictionary.
        """
        if pose_history is None:
            raise ValueError("pose_history tensor is required for AdaptiveHistoryModule.")

        # History Dropout: force model to rely on other cues (e.g., future cond) occasionally
        if self.training and self.history_dropout_prob > 0:
            if torch.rand(1, device=pose_history.device).item() < self.history_dropout_prob:
                B = pose_history.shape[0]
                zero_out = pose_history.new_zeros(B, self.num_slots * self.pose_dim)
                diag = {
                    "effective_frames": 0.0,
                    "dropout_applied": True,
                    "frame_importance": None,
                }
                self._last_diag = diag
                return zero_out, diag

        if pose_history.dim() == 2:
            total_dim = pose_history.shape[-1]
            if total_dim % self.pose_dim != 0:
                raise RuntimeError(
                    f"pose_history dim {total_dim} not divisible by pose_dim {self.pose_dim}"
                )
            L = total_dim // self.pose_dim
            hist = pose_history.view(pose_history.shape[0], L, self.pose_dim)
        elif pose_history.dim() == 3 and pose_history.shape[-1] == self.pose_dim:
            hist = pose_history
        else:
            raise RuntimeError(
                "pose_history must be [B, P] flattened or [B, L, pose_dim] tensor."
            )

        B, L, _ = hist.shape
        if L == 0:
            flat = hist.new_zeros(B, self.num_slots * self.pose_dim)
            diag = {"effective_frames": 0}
            self._last_diag = diag
            return flat, diag

        upper = min(self.max_history_frames, L)
        if self.training and self.train_variable_history and upper > self.num_slots:
            eff = int(
                torch.randint(
                    low=self.num_slots,
                    high=upper + 1,
                    size=(1,),
                    device=hist.device,
                ).item()
            )
        else:
            eff = min(self.num_slots, upper)
        eff = max(1, eff)
        hist_slice = hist[:, -eff:, :]  # [B, eff, pose_dim]

        frame_embed = self.frame_proj(hist_slice)  # [B, eff, hidden]
        trend_diag: Dict[str, torch.Tensor | float] = {}
        if self.use_trend_features:
            delta = hist_slice[:, 1:, :] - hist_slice[:, :-1, :]
            zero_pad = torch.zeros(delta.shape[0], 1, delta.shape[2], device=delta.device, dtype=delta.dtype)
            delta = torch.cat([zero_pad, delta], dim=1)
            drift = hist_slice - hist_slice[:, :1, :]
            trend_feat = torch.cat([delta, drift], dim=-1)
            trend_embed = self.trend_proj(trend_feat)
            frame_embed = frame_embed + self.trend_norm(trend_embed)
            with torch.no_grad():
                trend_diag = {
                    "trend_delta_rms": float(delta.detach().pow(2).mean().sqrt().cpu()),
                    "trend_drift_rms": float(drift.detach().pow(2).mean().sqrt().cpu()),
                }
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        if self.context_proj is not None and context is not None:
            ctx = context
            if ctx.dim() == 3:
                ctx = ctx.mean(dim=1)
            ctx_feat = self.context_proj(ctx)
            queries = queries + ctx_feat.unsqueeze(1)

        Q = self.q_proj(queries)
        K = self.k_proj(frame_embed)
        V = self.v_proj(frame_embed)

        Q = Q.view(B, self.num_slots, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, eff, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        V = V.view(B, eff, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K) * self.scale  # [B, H, num_slots, eff]
        attn = torch.softmax(scores, dim=-1)
        context_vec = torch.matmul(attn, V)  # [B, H, num_slots, head_dim]
        context_vec = context_vec.permute(0, 2, 1, 3).contiguous().view(B, self.num_slots, self.hidden_dim)

        if self.use_gate and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(queries))
            context_vec = gate * context_vec + (1.0 - gate) * queries

        out = self.out_proj(context_vec).reshape(B, self.num_slots * self.pose_dim)
        diag = {
            "effective_frames": float(eff),
            "frame_importance": attn.detach().mean(dim=1),  # [B, num_slots, eff]
            "dropout_applied": False,
        }
        if trend_diag:
            diag.update(trend_diag)
        self._last_diag = diag
        return out, diag

    def last_diagnostics(self) -> Dict[str, torch.Tensor | float]:
        return self._last_diag


def resolve_pose_hist_runtime_tensors(
    dataset: Any,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    pose_norm = getattr(dataset, "pose_hist_norm", None)
    pose_hist_scales = None
    pose_hist_mu = None
    pose_hist_std = None
    if pose_norm is not None:
        pose_hist_scales = torch.as_tensor(pose_norm.scales, dtype=torch.float32)
        pose_hist_mu = (
            torch.as_tensor(pose_norm.mu, dtype=torch.float32)
            if getattr(pose_norm, "mu", None) is not None
            else None
        )
        pose_hist_std = (
            torch.as_tensor(pose_norm.std, dtype=torch.float32)
            if getattr(pose_norm, "std", None) is not None
            else None
        )
    return pose_hist_scales, pose_hist_mu, pose_hist_std


def attach_adaptive_history_runtime(
    model: Any,
    *,
    history_export_frames: int,
    pose_hist_dim_raw: int,
    pose_hist_len_raw: int,
    history_frame_dim: int,
    history_hidden_dim: int,
    max_history_frames: Optional[int],
    history_heads: int,
    train_variable_history: bool,
    history_dropout_prob: float,
    use_trend_features: bool,
    device: torch.device,
) -> None:
    if int(history_export_frames) <= 0:
        return
    if int(pose_hist_dim_raw) <= 0 or int(pose_hist_len_raw) <= 0:
        print('[AdaptiveHistory][WARN] pose history not available; adaptive history disabled.')
        return
    if int(pose_hist_dim_raw) % int(pose_hist_len_raw) != 0:
        print('[AdaptiveHistory][WARN] pose history dim不整除帧数，跳过 adaptive history。')
        return

    max_frames = max_history_frames
    if max_frames is None:
        max_frames = int(pose_hist_len_raw)

    module_device = torch.device('cpu') if device.type == 'mps' else device
    history_module = AdaptiveHistoryModule(
        pose_dim=int(history_frame_dim),
        hidden_dim=int(history_hidden_dim),
        num_history_frames=int(history_export_frames),
        max_history_frames=int(max_frames),
        cond_dim=0,
        num_heads=int(history_heads),
        train_variable_history=bool(train_variable_history),
        history_dropout_prob=float(history_dropout_prob),
        use_trend_features=bool(use_trend_features),
    ).to(module_device)
    model.enable_adaptive_history(history_module, pose_hist_len=int(pose_hist_len_raw))


__all__ = [
    "AdaptiveHistoryModule",
    "PoseHistState",
    "pose_hist_transform_vec",
    "pose_hist_inverse_vec",
    "init_pose_hist_state",
    "resolve_pose_hist_input",
    "advance_pose_hist_state",
    "advance_pose_hist_state_with_tail",
    "resolve_pose_hist_runtime_tensors",
    "attach_adaptive_history_runtime",
]
