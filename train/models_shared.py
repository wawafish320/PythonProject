"""Shared lightweight model components used by both pretrain and main training."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import build_mlp

__all__ = ["MotionEncoder", "PeriodHead"]


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

