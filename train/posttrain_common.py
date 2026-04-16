#!/usr/bin/env python3
"""
Shared helpers extracted from train.posttrain.

This module intentionally contains lightweight utilities that are also used by
diagnostic tools, so those tools do not need to import the full posttrain entry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .rotvec_semantics import require_standard_rotvec_spec


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


def _parse_pretrain_contact_affine_spec(spec: Any) -> Optional[Dict[str, Any]]:
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


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as err:
        raise RuntimeError(f"[FATAL] failed to read {label} {path}: {err}") from err
    if not isinstance(payload, dict):
        raise RuntimeError(f"[FATAL] {label} {path} must contain a JSON object.")
    return payload


def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = None,
) -> Dict[str, Any]:
    spec = _load_json_object(bundle_path, label="bundle_json")
    require_standard_rotvec_spec(spec, context=f"bundle {bundle_path}")
    if pretrain_path is not None and pretrain_path.is_file():
        extra = _load_json_object(pretrain_path, label="pretrain_template")
        require_standard_rotvec_spec(extra, context=f"pretrain_template {pretrain_path}")
        if pretrain_keys is None:
            spec = dict(extra, **spec)
        else:
            spec = dict(spec)
            for key in pretrain_keys:
                if key in extra and extra[key] is not None:
                    spec[key] = extra[key]
    return spec


_merge_norm_spec = merge_norm_spec


def _select_trainable_params(model: torch.nn.Module) -> Tuple[list[torch.nn.Parameter], list[str]]:
    trainable: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable.append(param)
        names.append(name)
    return trainable, names


def _freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _enable_modules(model: torch.nn.Module, names: Tuple[str, ...]) -> None:
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad_(True)


def _unfreeze_direct_pose(
    model: torch.nn.Module,
    *,
    leg_only: bool = False,
    leg_gate_only: bool = False,
    nonleg_only: bool = False,
) -> None:
    if bool(leg_gate_only):
        _enable_modules(model, ("direct_pose_leg_gate_head",))
        return
    if bool(leg_only):
        _enable_modules(model, ("direct_pose_leg_terminal", "direct_pose_leg_head", "direct_pose_leg_gate_head"))
        return
    if bool(nonleg_only):
        _enable_modules(
            model,
            (
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_nonleg_proj",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_out_nonleg",
            ),
        )
        return

    _enable_modules(
        model,
        (
            "direct_pose_head",
            "direct_pose_leg_terminal",
            "direct_pose_out_leg",
            "direct_pose_out_nonleg",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_leg_head",
            "direct_pose_leg_gate_head",
        ),
    )
