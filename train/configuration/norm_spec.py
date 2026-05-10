from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, overload

from ..contracts.asset_semantics import require_standard_rotvec_spec

NORM_SPEC_RUNTIME_PRETRAIN_KEYS: tuple[str, ...] = (
    "MuAngVel",
    "StdAngVel",
    "tanh_scales_angvel",
    "pose_hist_len",
    "pose_hist_dim",
    "tanh_scales_pose_hist",
    "MuPoseHist",
    "StdPoseHist",
)

__all__ = [
    "NORM_SPEC_RUNTIME_PRETRAIN_KEYS",
    "ContactPretrainRuntime",
    "merge_norm_spec",
    "parse_pretrain_contact_affine_spec",
    "resolve_contact_pretrain_runtime",
]


@dataclass(frozen=True)
class ContactPretrainRuntime:
    clamp: float
    affine_stats: Optional[str]
    affine: Optional[Dict[str, Any]]
    dropout_injection_mode: str = "off"
    dropout_prob: float = 0.0


def _warn(message: str, *, warn: bool, warn_prefix: str = "") -> None:
    if not warn:
        return
    prefix = str(warn_prefix or "").strip() or "[Spec]"
    print(f"{prefix}[WARN] {message}")


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as err:
        raise RuntimeError(f"[FATAL] failed to read {label} {path}: {err}") from err
    if not isinstance(payload, dict):
        raise RuntimeError(f"[FATAL] {label} {path} must contain a JSON object.")
    return payload


def _normalize_pretrain_keys(pretrain_keys: tuple[str, ...]) -> tuple[str, ...]:
    base_keys = tuple(pretrain_keys)
    ordered: list[str] = []
    seen: set[str] = set()
    for key in (*base_keys, "pose_hist_dim"):
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def _normalize_affine_stats_spec(spec: Any) -> Optional[str]:
    if spec is None:
        return None
    if isinstance(spec, Path):
        text = str(spec)
    elif isinstance(spec, str):
        text = spec.strip()
    else:
        try:
            text = json.dumps(spec, ensure_ascii=False)
        except Exception:
            text = str(spec).strip()
    text = text.strip()
    return text or None


def _load_checked_spec(
    path: Path,
    *,
    label: str,
    context: str,
    strict: bool,
    warn: bool,
    warn_prefix: str,
    warn_missing: bool,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        if strict:
            return _load_json_object(path, label=label)
        if warn_missing:
            _warn(f"{label} not found at {path}", warn=warn, warn_prefix=warn_prefix)
        return None
    try:
        payload = _load_json_object(path, label=label)
        require_standard_rotvec_spec(payload, context=context)
    except Exception as err:
        if strict:
            raise
        message = str(err).removeprefix("[FATAL] ").strip()
        if not message:
            message = f"invalid {label} {path}"
        _warn(message, warn=warn, warn_prefix=warn_prefix)
        return None
    return payload


def parse_pretrain_contact_affine_spec(spec: Any) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    raw = spec
    if isinstance(raw, Path):
        raw = str(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            path = Path(text).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
            else:
                raw = json.loads(text)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    scale = raw.get("scale", None)
    bias = raw.get("bias", None)
    if not isinstance(scale, (list, tuple)) or not isinstance(bias, (list, tuple)):
        return None
    try:
        scale_vals = [float(x) for x in scale]
        bias_vals = [float(x) for x in bias]
    except Exception:
        return None
    if len(scale_vals) <= 0 or len(scale_vals) != len(bias_vals):
        return None
    if not all(math.isfinite(float(x)) for x in scale_vals):
        return None
    if not all(math.isfinite(float(x)) for x in bias_vals):
        return None
    try:
        eps = float(raw.get("eps", 1e-4) or 1e-4)
    except Exception:
        eps = 1e-4
    if not math.isfinite(float(eps)):
        eps = 1e-4
    eps = float(min(1e-2, max(1e-8, eps)))
    return {"scale": scale_vals, "bias": bias_vals, "eps": eps}


def resolve_contact_pretrain_runtime(
    *,
    clamp_raw: Any,
    affine_stats_raw: Any,
    dropout_injection_mode_raw: Any = "off",
    dropout_prob_raw: Any = 0.0,
    warn: bool = False,
    warn_prefix: str = "",
) -> ContactPretrainRuntime:
    try:
        clamp = float(clamp_raw)
    except Exception:
        clamp = 1.0
    if (not math.isfinite(float(clamp))) or float(clamp) < 0.0:
        clamp = 1.0

    affine_stats = _normalize_affine_stats_spec(affine_stats_raw)
    affine = parse_pretrain_contact_affine_spec(affine_stats)
    if affine_stats is not None and affine is None:
        _warn(
            "failed to parse contact pretrain affine stats; continuing without affine calibration.",
            warn=warn,
            warn_prefix=warn_prefix,
        )
    mode_text = str(dropout_injection_mode_raw if dropout_injection_mode_raw is not None else "off").strip().lower()
    if mode_text == "":
        mode_text = "off"
    if mode_text not in ("off", "encoder_input", "hidden"):
        raise ValueError(
            "contacts_pretrain_dropout_injection_mode must be one of: off|encoder_input|hidden; "
            f"got {dropout_injection_mode_raw!r}."
        )
    try:
        dropout_prob = float(dropout_prob_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"contacts_pretrain_dropout_prob must be a finite float in [0,1); got {dropout_prob_raw!r}."
        ) from exc
    if (not math.isfinite(dropout_prob)) or dropout_prob < 0.0 or dropout_prob >= 1.0:
        raise ValueError(
            f"contacts_pretrain_dropout_prob must be in [0,1); got {dropout_prob}."
        )
    return ContactPretrainRuntime(
        clamp=float(clamp),
        affine_stats=affine_stats,
        affine=affine,
        dropout_injection_mode=mode_text,
        dropout_prob=float(dropout_prob),
    )


@overload
def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = NORM_SPEC_RUNTIME_PRETRAIN_KEYS,
    strict: Literal[True] = True,
    warn: bool = False,
    warn_prefix: str = "",
) -> Dict[str, Any]: ...


@overload
def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = NORM_SPEC_RUNTIME_PRETRAIN_KEYS,
    strict: Literal[False],
    warn: bool = False,
    warn_prefix: str = "",
) -> Optional[Dict[str, Any]]: ...


def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = NORM_SPEC_RUNTIME_PRETRAIN_KEYS,
    strict: bool = True,
    warn: bool = False,
    warn_prefix: str = "",
) -> Optional[Dict[str, Any]]:
    spec = _load_checked_spec(
        bundle_path,
        label="bundle_json",
        context=f"bundle {bundle_path}",
        strict=strict,
        warn=warn,
        warn_prefix=warn_prefix,
        warn_missing=True,
    )
    if spec is None:
        return None

    merged = dict(spec)
    if pretrain_path is None or not pretrain_path.is_file():
        return merged

    extra = _load_checked_spec(
        pretrain_path,
        label="pretrain_template",
        context=f"pretrain_template {pretrain_path}",
        strict=strict,
        warn=warn,
        warn_prefix=warn_prefix,
        warn_missing=False,
    )
    if extra is None:
        return merged

    if pretrain_keys is None:
        return dict(extra, **merged)

    for key in _normalize_pretrain_keys(pretrain_keys):
        if key in extra and extra[key] is not None:
            merged[key] = extra[key]
    return merged
