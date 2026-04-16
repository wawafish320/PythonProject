from __future__ import annotations

import glob
import json
import math as _math
import os
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        elif isinstance(item, Integral):
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
        if isinstance(item, Integral):
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
def _build_pretrain_contact_encoder_input(
    motion_step_t: torch.Tensor,
    pose_hist_step_t: Optional[torch.Tensor],
    *,
    contact_dim: int,
    encoder_input_dim: int,
    angvel_slice: Optional[slice],
    clamp_val: float,
) -> torch.Tensor:
    batch_size = int(motion_step_t.shape[0])
    angvel_raw = None
    if isinstance(angvel_slice, slice):
        try:
            angvel_raw = motion_step_t[..., angvel_slice]
        except (IndexError, RuntimeError, TypeError, ValueError):
            angvel_raw = None
    features = [motion_step_t.new_zeros((batch_size, contact_dim))]
    for value in (angvel_raw, pose_hist_step_t):
        if not torch.is_tensor(value):
            features.append(motion_step_t.new_zeros((batch_size, 0)))
            continue
        feature = value.to(device=motion_step_t.device, dtype=motion_step_t.dtype)
        if feature.ndim == 3 and int(feature.size(1)) == 1:
            feature = feature[:, 0]
        elif feature.ndim != 2:
            feature = feature.reshape(batch_size, -1)
        features.append(feature)
    encoder_input = torch.cat(features, dim=-1)
    feature_dim = int(encoder_input.shape[-1])
    target_dim = int(encoder_input_dim)
    if feature_dim != target_dim:
        if feature_dim > target_dim:
            encoder_input = encoder_input[..., :target_dim]
        else:
            encoder_input = torch.cat([encoder_input, encoder_input.new_zeros((batch_size, target_dim - feature_dim))], dim=-1)
    if float(clamp_val) > 0.0:
        encoder_input = encoder_input.clamp(-float(clamp_val), float(clamp_val))
    return encoder_input


def safe_set_slice(obj, attr, maybe_slice):
    """Assign attr only when maybe_slice is a valid slice."""
    if isinstance(maybe_slice, slice):
        setattr(obj, attr, maybe_slice)


def warn_once(
    seen_keys: set[str],
    *,
    category: str,
    key: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> None:
    key_token = str(key)
    if key_token in seen_keys:
        return
    seen_keys.add(key_token)
    prefix = f"[{str(category)}][WARN]"
    if exc is None:
        print(f"{prefix} {message}")
    else:
        print(f"{prefix} {message}: {exc}")


def _grad_tensor_to_float(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grad is None:
        return None
    try:
        grad_tensor = grad.detach()
        if not torch.isfinite(grad_tensor).all():
            grad_tensor = torch.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_tensor.float()
    except Exception:
        return None


def grad_list_norm(grads: Iterable[Optional[torch.Tensor]]) -> float:
    total_sq = 0.0
    has_grad = False
    for grad in grads:
        grad_tensor = _grad_tensor_to_float(grad)
        if grad_tensor is None:
            continue
        total_sq += float(grad_tensor.pow(2).sum().item())
        has_grad = True
    if not has_grad:
        return float("nan")
    return float(_math.sqrt(max(0.0, total_sq)))


def grad_list_cosine(
    grads_a: Iterable[Optional[torch.Tensor]],
    grads_b: Iterable[Optional[torch.Tensor]],
) -> float:
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    has_a = False
    has_b = False
    for grad_a, grad_b in zip(grads_a, grads_b):
        grad_tensor_a = _grad_tensor_to_float(grad_a)
        grad_tensor_b = _grad_tensor_to_float(grad_b)
        if grad_tensor_a is not None:
            norm_a_sq += float(grad_tensor_a.pow(2).sum().item())
            has_a = True
        if grad_tensor_b is not None:
            norm_b_sq += float(grad_tensor_b.pow(2).sum().item())
            has_b = True
        if grad_tensor_a is not None and grad_tensor_b is not None:
            try:
                dot += float((grad_tensor_a * grad_tensor_b).sum().item())
            except Exception:
                continue
    if (not has_a) or (not has_b):
        return float("nan")
    norm_a = _math.sqrt(max(0.0, norm_a_sq))
    norm_b = _math.sqrt(max(0.0, norm_b_sq))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return float("nan")
    return float(dot / (norm_a * norm_b))


def grad_norm_of_module(module: Optional[nn.Module]) -> float:
    """Compute the global L2 norm of parameter gradients on a module."""
    if module is None:
        return float("nan")
    return grad_list_norm(getattr(param, "grad", None) for param in module.parameters(recurse=True))


def module_grad_norm(module: Optional[nn.Module]) -> float:
    """Deprecated alias for `grad_norm_of_module`; remove in the next refactor."""
    return grad_norm_of_module(module)


def resolve_device(pref: str) -> torch.device:
    """Resolve user device preference with CUDA/MPS fallback."""
    pref = str(pref or "auto").lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if has_mps else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def safe_int_scalar(value: Any) -> Optional[int]:
    """Best-effort scalar-to-int conversion that rejects non-scalar tensors."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    else:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def as_path(val: Any) -> Optional[Path]:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    return Path(text).expanduser()


def as_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    text = str(val).strip().lower()
    if text in ("1", "true", "yes", "y", "t", "on"):
        return True
    if text in ("0", "false", "no", "n", "f", "off", "none", "null", ""):
        return False
    return bool(val)


def normalize_optional_csv(val: Any) -> Optional[str]:
    if isinstance(val, (list, tuple)):
        toks = [str(x).strip() for x in val if str(x).strip()]
        return ",".join(toks) if toks else None
    if val is None:
        return None
    text = str(val).strip()
    return text if text else None


def parse_int_set_spec(spec: Any) -> set[int]:
    if spec is None:
        return set()
    out: set[int] = set()
    for tok in str(spec).replace(";", ",").split(","):
        text = tok.strip()
        if not text:
            continue
        if "-" in text or ":" in text:
            sep = "-" if "-" in text else ":"
            start, end = [x.strip() for x in text.split(sep, 1)]
            if start.lstrip("-").isdigit() and end.lstrip("-").isdigit():
                lo = int(start)
                hi = int(end)
                if lo > hi:
                    lo, hi = hi, lo
                for value in range(lo, hi + 1):
                    out.add(int(value))
            continue
        if text.lstrip("-").isdigit():
            out.add(int(text))
    return out


def iter_infinite(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def as_float_list(val: Any) -> Optional[list[float]]:
    if val is None:
        return None
    payload = str(val) if isinstance(val, Path) else val
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            path = Path(text).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    return None
            else:
                payload = json.loads(text)
        except Exception:
            return None
    if isinstance(payload, dict):
        if "scales" in payload:
            payload = payload.get("scales")
        elif "values" in payload:
            payload = payload.get("values")
        else:
            return None
    if not isinstance(payload, (list, tuple)):
        return None
    values: list[float] = []
    for item in payload:
        try:
            values.append(float(item))
        except Exception:
            return None
    return values or None


def pick_first_present(payload: Any, keys: Sequence[str]) -> Any:
    """Return the first non-None value from a dict-like payload for the given keys."""
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def clamp_int(val: int, *, min_value: Optional[int], max_value: Optional[int]) -> int:
    if min_value is not None:
        val = max(int(min_value), int(val))
    if max_value is not None:
        val = min(int(max_value), int(val))
    return int(val)


def clamp_float(val: float, *, min_value: Optional[float], max_value: Optional[float]) -> float:
    if min_value is not None:
        val = max(float(min_value), float(val))
    if max_value is not None:
        val = min(float(max_value), float(val))
    return float(val)


def cfg_pick(payload: Mapping[str, Any], key: str, *, aliases: Tuple[str, ...] = ()) -> Any:
    if key in payload:
        return payload.get(key)
    for alias in aliases:
        if alias in payload:
            return payload.get(alias)
    return None


def cfg_get_bool(payload: Mapping[str, Any], key: str, default: bool, *, aliases: Tuple[str, ...] = ()) -> bool:
    return as_bool(cfg_pick(payload, key, aliases=aliases), default)


def cfg_get_int(
    payload: Mapping[str, Any],
    key: str,
    default: Optional[int],
    *,
    aliases: Tuple[str, ...] = (),
    allow_none: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    raw = cfg_pick(payload, key, aliases=aliases)
    if raw is None:
        val = None if allow_none else default
    else:
        try:
            val = int(raw)
        except Exception:
            val = None if allow_none else default
    if val is None:
        return None
    return clamp_int(int(val), min_value=min_value, max_value=max_value)


def cfg_get_float(
    payload: Mapping[str, Any],
    key: str,
    default: Optional[float],
    *,
    aliases: Tuple[str, ...] = (),
    allow_none: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    require_finite: bool = True,
) -> Optional[float]:
    raw = cfg_pick(payload, key, aliases=aliases)
    if raw is None:
        val = None if allow_none else default
    else:
        try:
            val = float(raw)
        except Exception:
            val = None if allow_none else default
    if val is None:
        return None
    if require_finite and (not _math.isfinite(float(val))):
        val = None if allow_none else default
    if val is None:
        return None
    return clamp_float(float(val), min_value=min_value, max_value=max_value)


def cfg_get_enum(
    payload: Mapping[str, Any],
    key: str,
    default: str,
    *,
    aliases: Tuple[str, ...] = (),
    alias_map: Optional[Dict[str, str]] = None,
    choices: Optional[Tuple[str, ...]] = None,
    lower: bool = True,
) -> str:
    raw = cfg_pick(payload, key, aliases=aliases)
    value = str(default) if raw is None else str(raw)
    value = value.strip()
    value_cmp = value.lower() if lower else value
    if alias_map:
        value_cmp = alias_map.get(value_cmp, value_cmp)
    if choices and value_cmp not in choices:
        return str(default)
    return value_cmp


def cfg_get_or(payload: Mapping[str, Any], key: str, default: Any, *, aliases: Tuple[str, ...] = ()) -> Any:
    raw = cfg_pick(payload, key, aliases=aliases)
    return raw or default


def cfg_get_str_or(payload: Mapping[str, Any], key: str, default: str, *, aliases: Tuple[str, ...] = ()) -> str:
    return str(cfg_get_or(payload, key, default, aliases=aliases))


def cfg_get_int_or(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    *,
    aliases: Tuple[str, ...] = (),
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    val = int(cfg_get_or(payload, key, default, aliases=aliases))
    return clamp_int(val, min_value=min_value, max_value=max_value)


def cfg_get_float_or(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    *,
    aliases: Tuple[str, ...] = (),
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    val = float(cfg_get_or(payload, key, default, aliases=aliases))
    return clamp_float(val, min_value=min_value, max_value=max_value)


def cfg_get_int_present(payload: Mapping[str, Any], key: str, default: int, *, aliases: Tuple[str, ...] = ()) -> int:
    raw = cfg_pick(payload, key, aliases=aliases)
    return int(default) if raw is None else int(raw)


def cfg_get_float_present(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    *,
    aliases: Tuple[str, ...] = (),
) -> float:
    raw = cfg_pick(payload, key, aliases=aliases)
    return float(default) if raw is None else float(raw)


def cfg_from_schema(
    payload: Mapping[str, Any],
    schema: Sequence[Tuple[str, Callable[..., Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, getter, kwargs in schema:
        out[name] = getter(payload, **kwargs)
    return out


def apply_cli_overrides(
    payload: Dict[str, Any],
    args: Any,
    *,
    bool_keys: Sequence[str] = (),
    optional_float_keys: Sequence[str] = (),
    skip_keys: Sequence[str] = (),
) -> None:
    args_map = vars(args)
    bool_key_set = set(bool_keys)
    optional_float_key_set = set(optional_float_keys)
    skip_key_set = set(skip_keys) | bool_key_set | optional_float_key_set

    for key in bool_key_set:
        raw = args_map.get(key)
        if raw is None:
            continue
        payload[key] = str(raw).strip().lower() in ("1", "true", "yes", "y")

    for key in optional_float_key_set:
        raw = args_map.get(key)
        if raw is None:
            continue
        text = str(raw).strip().lower()
        payload[key] = None if text in ("null", "none", "") else float(raw)

    for key, value in args_map.items():
        if key in skip_key_set or value is None:
            continue
        payload[key] = value


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

def _expected_first_linear_in_features(model: nn.Module, Dx: int, Dc: int) -> int:
    """
    Compute the expected in_features for EventMotionModel's first Linear.

    Base input is [state, cond]. Some variants additionally inject plan features into the main trunk:
      - contact_plan_inject='contacts' -> +contact_dim
      - contact_plan_inject='plan_z'   -> +contact_plan_hidden
    """
    expected = int(Dx) + int(Dc)
    try:
        if bool(getattr(model, "contact_plan_enable", False)):
            inject = str(getattr(model, "contact_plan_inject", "none") or "none").lower().strip()
            if inject == "contacts":
                expected += int(getattr(model, "contact_dim", 0) or 0)
            elif inject == "plan_z":
                expected += int(getattr(model, "contact_plan_hidden", 0) or 0)
    except Exception:
        pass
    return int(expected)


def validate_and_fix_model_(m: nn.Module, Dx: int | None = None, Dc: int | None = None, *, reinit_on_nonfinite: bool = True) -> None:
    """Production-safe sanity check and optional re-init for models."""
    if Dx is not None and Dc is not None:
        first_linear_in = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                first_linear_in = mod.in_features
                break
        if first_linear_in is not None:
            expected = _expected_first_linear_in_features(m, int(Dx), int(Dc))
            if first_linear_in != expected:
                raise RuntimeError(f"First Linear in_features={first_linear_in} != expected={expected} (Dx={Dx}, Dc={Dc})")

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


def _guard_first_linear_finite_(model: nn.Module) -> None:
    with torch.no_grad():
        first_linear = model.shared_encoder[0]
        if not torch.isfinite(first_linear.weight).all() or (
            first_linear.bias is not None and (not torch.isfinite(first_linear.bias).all())
        ):
            print('[Guard] first-linear became non-finite post-sanitize, reinitializing')
            nn.init.kaiming_uniform_(first_linear.weight, a=_math.sqrt(5))
            if first_linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(first_linear.weight)
                bound = 1.0 / _math.sqrt(max(fan_in, 1))
                nn.init.uniform_(first_linear.bias, -bound, bound)
            assert torch.isfinite(first_linear.weight).all() and (
                first_linear.bias is None or torch.isfinite(first_linear.bias).all()
            )
    with torch.no_grad():
        first_linear = model.shared_encoder[0]
        assert torch.isfinite(first_linear.weight).all() and (
            first_linear.bias is None or torch.isfinite(first_linear.bias).all()
        ), '[PostCheck] shared_encoder.0 still not finite'


def _first_linear_in_features(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def sanity_check_model_dims(model, Dx, Dy, Dc):
    nin = _first_linear_in_features(model)
    if nin is not None:
        expected = _expected_first_linear_in_features(model, int(Dx), int(Dc))
        if nin != expected:
            raise RuntimeError(
                f'[Guard] 模型第一层 in_features={nin}，但应为 expected={expected} (Dx={Dx}, Dc={Dc})；'
                f'也可能是注入了 contact_plan_inject 导致输入维度增加，或构建时把 in_dim 设错了。'
            )


__all__ = [
    "build_mlp",
    "safe_set_slice",
    "warn_once",
    "grad_list_norm",
    "grad_list_cosine",
    "grad_norm_of_module",
    "module_grad_norm",
    "resolve_device",
    "safe_int_scalar",
    "as_path",
    "as_bool",
    "as_float_list",
    "expand_paths_from_specs",
    "get_flag_value_from_argv",
    "get_flag_values_from_argv",
    "validate_and_fix_model_",
    "_guard_first_linear_finite_",
    "_first_linear_in_features",
    "sanity_check_model_dims",
    "set_global_args",
    "get_global_arg",
]
