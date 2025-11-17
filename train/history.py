from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.frame_proj = nn.Linear(self.pose_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.pose_dim)
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
        }
        self._last_diag = diag
        return out, diag

    def last_diagnostics(self) -> Dict[str, torch.Tensor | float]:
        return self._last_diag


__all__ = ["AdaptiveHistoryModule"]
