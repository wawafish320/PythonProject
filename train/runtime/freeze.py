#!/usr/bin/env python3
"""
Shared freeze/unfreeze helpers extracted from posttrain runtime.

This module intentionally contains lightweight trainability utilities that are
also used by diagnostic tools, so those tools do not need to import the full
posttrain entry.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


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


def _enable_direct_pose_trunk(model: torch.nn.Module, mode: str) -> List[str]:
    mode = str(mode or "none").strip().lower()
    if mode in ("", "none", "off", "false", "0"):
        return []
    if mode not in ("last", "full"):
        raise ValueError(f"direct_pose_nonleg_trunk_mode must be one of: none | last | full (got {mode!r})")
    head = getattr(model, "direct_pose_head", None)
    if head is None:
        raise RuntimeError("model has no direct_pose_head")
    enabled: List[str] = []
    if mode == "full":
        for name, param in head.named_parameters():
            param.requires_grad_(True)
            enabled.append(f"direct_pose_head.{name}")
        return enabled

    target = None
    prefix = "direct_pose_head"
    if isinstance(head, nn.Sequential) and len(head) > 3 and isinstance(head[3], nn.Linear):
        target = head[3]
        prefix = "direct_pose_head.3"
    elif isinstance(head, nn.Sequential):
        for idx in range(len(head) - 1, -1, -1):
            if isinstance(head[idx], nn.Linear):
                target = head[idx]
                prefix = f"direct_pose_head.{idx}"
                break
    if target is None:
        raise RuntimeError("could not resolve last Linear in direct_pose_head")
    for name, param in target.named_parameters():
        param.requires_grad_(True)
        enabled.append(f"{prefix}.{name}")
    return enabled


def _unfreeze_direct_pose(
    model: torch.nn.Module,
    *,
    leg_only: bool = False,
    leg_gate_only: bool = False,
    nonleg_only: bool = False,
    nonleg_trunk_mode: str = "none",
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
        _enable_direct_pose_trunk(model, nonleg_trunk_mode)
        return

    _enable_modules(
        model,
        (
            "direct_pose_head",
            "direct_pose_leg_terminal",
            "direct_pose_out_nonleg",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_leg_head",
            "direct_pose_leg_gate_head",
        ),
    )


def _unfreeze_for_train_mode(
    model: torch.nn.Module,
    *,
    train_mode: str,
    direct_pose_leg_train_only: bool = False,
    direct_pose_leg_gate_train_only: bool = False,
    direct_pose_nonleg_train_only: bool = False,
    direct_pose_nonleg_trunk_mode: str = "none",
) -> None:
    if train_mode == "direct":
        _unfreeze_direct_pose(
            model,
            leg_only=bool(direct_pose_leg_train_only),
            leg_gate_only=bool(direct_pose_leg_gate_train_only),
            nonleg_only=bool(direct_pose_nonleg_train_only),
            nonleg_trunk_mode=str(direct_pose_nonleg_trunk_mode or "none"),
        )
        return
    if train_mode == "lambda":
        _enable_modules(model, ("lambda_fusion_head",))
        return
    raise ValueError(f"Unknown train_mode={train_mode!r}")
