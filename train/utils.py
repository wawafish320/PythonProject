from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    *,
    num_layers: int = 1,
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    use_layer_norm: bool = False,
    final_dim: Optional[int] = None,
    final_activation: bool = False,
    final_dropout: float = 0.0,
) -> nn.Sequential:
    """Reusable MLP builder shared by training and pretrain modules."""

    layers: list[nn.Module] = []
    d_in = int(input_dim)
    for _ in range(max(1, int(num_layers))):
        layers.append(nn.Linear(d_in, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if activation is not None:
            layers.append(activation())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        d_in = hidden_dim

    if final_dim is not None:
        layers.append(nn.Linear(d_in, final_dim))
        if final_activation and activation is not None:
            layers.append(activation())
        if float(final_dropout) > 0:
            layers.append(nn.Dropout(float(final_dropout)))
    return nn.Sequential(*layers)


def safe_set_slice(obj, attr, maybe_slice):
    """Assign attr only when maybe_slice is a valid slice."""
    if isinstance(maybe_slice, slice):
        setattr(obj, attr, maybe_slice)


__all__ = ["build_mlp", "safe_set_slice"]
