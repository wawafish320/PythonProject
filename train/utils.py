from __future__ import annotations

import glob
import math as _math
import os
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# MLP builder and slice helper
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# CLI helpers (formerly cli_utils.py)
# -----------------------------------------------------------------------------

def expand_paths_from_specs(specs: Optional[Iterable[str]]) -> List[str]:
    """
    Expand an iterable of path/glob specifications into a deduplicated list of .npz files.
    Accepts directories, glob patterns, plain file paths, or @file indirection (one per line).
    """
    if not specs:
        return []
    if isinstance(specs, str):
        specs = [specs]

    pending: List[str] = []
    for item in specs:
        if not item:
            continue
        tok = str(item).strip()
        if not tok:
            continue
        if tok.startswith("@") and os.path.isfile(tok[1:]):
            with open(tok[1:], "r", encoding="utf-8") as f:
                for line in f:
                    val = line.strip()
                    if val:
                        pending.append(val)
        else:
            pending.append(tok)

    files: List[str] = []
    for spec in pending:
        if os.path.isdir(spec):
            files.extend(sorted(glob.glob(os.path.join(spec, "*.npz"))))
        elif any(ch in spec for ch in "*?["):
            files.extend(sorted(glob.glob(spec)))
        elif os.path.isfile(spec):
            files.append(spec)

    out: List[str] = []
    seen = set()
    for path in files:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def get_flag_value_from_argv(argv: Iterable[str], flag: str, default=None):
    """
    Return the value that follows a given CLI flag.
    Supports '--key value' and '--key=value' forms.
    """
    for tok in argv:
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    argv_list = list(argv)
    for idx, tok in enumerate(argv_list):
        if tok == flag:
            nxt = idx + 1
            if nxt < len(argv_list) and not argv_list[nxt].startswith("-"):
                return argv_list[nxt]
    return default


def get_flag_values_from_argv(argv: Iterable[str], flag: str) -> List[str]:
    """
    Collect all occurrences of a flag that may accept multiple values.
    Supports repeated flags and comma-separated lists.
    """
    argv_list = list(argv)
    values: List[str] = []
    for idx, tok in enumerate(argv_list):
        if tok == flag:
            j = idx + 1
            while j < len(argv_list) and not argv_list[j].startswith("-"):
                values.append(argv_list[j])
                j += 1
    out: List[str] = []
    for val in values:
        if "," in val:
            out.extend([x for x in val.split(",") if x])
        else:
            out.append(val)
    return out


# -----------------------------------------------------------------------------
# Global arg helpers (used by training scripts)
# -----------------------------------------------------------------------------

_GLOBAL_ARGS = None


def set_global_args(namespace) -> None:
    """Register a namespace so get_global_arg can read defaults without tight coupling."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = namespace


def get_global_arg(name: str, default=None, namespace=None):
    """
    Fetch an attribute from the registered namespace (or explicit namespace override).
    """
    ns = namespace if namespace is not None else _GLOBAL_ARGS
    try:
        return getattr(ns, name)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Model sanity helpers (extracted from training_MPL.py)
# -----------------------------------------------------------------------------

def validate_and_fix_model_(m: nn.Module, Dx: int | None = None, Dc: int | None = None, *, reinit_on_nonfinite: bool = True) -> None:
    """Production-safe sanity check and optional re-init for models."""
    if Dx is not None and Dc is not None:
        first_linear_in = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                first_linear_in = mod.in_features
                break
        if first_linear_in is not None and first_linear_in != (Dx + Dc):
            raise RuntimeError(f"First Linear in_features={first_linear_in} != Dx+Dc={Dx+Dc}")

    def _reinit_module_(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, a=_math.sqrt(5))
            if getattr(mod, 'bias', None) is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mod.weight)
                bound = 1.0 / (_math.sqrt(max(fan_in, 1)))
                nn.init.uniform_(mod.bias, -bound, bound)
        elif hasattr(mod, 'reset_parameters'):
            try:
                mod.reset_parameters()
            except Exception:
                pass

    with torch.no_grad():
        for _, mod in m.named_modules():
            has_bad = False
            for _, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    has_bad = True
                    break
            if has_bad and reinit_on_nonfinite:
                _reinit_module_(mod)

        for name, mod in m.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"param still non-finite after reinit: {name}.{pname}")


def _first_linear_in_features(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def sanity_check_model_dims(model, Dx, Dy, Dc):
    nin = _first_linear_in_features(model)
    if nin is not None and nin != Dx + Dc:
        raise RuntimeError(f'[Guard] 模型第一层 in_features={nin}，但应为 Dx+Dc={Dx + Dc}；很可能构建时把 in_dim 设成 Dy+Dc={Dy + Dc} 了。')


__all__ = [
    "build_mlp",
    "safe_set_slice",
    "expand_paths_from_specs",
    "get_flag_value_from_argv",
    "get_flag_values_from_argv",
    "validate_and_fix_model_",
    "_first_linear_in_features",
    "sanity_check_model_dims",
    "set_global_args",
    "get_global_arg",
]
