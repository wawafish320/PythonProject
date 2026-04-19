#!/usr/bin/env python3
"""
Shared freeze/unfreeze helpers extracted from posttrain runtime.

This module intentionally contains lightweight trainability utilities that are
also used by diagnostic tools, so those tools do not need to import the full
posttrain entry.
"""

from __future__ import annotations

from typing import Tuple

import torch


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


def _unfreeze_for_train_mode(
    model: torch.nn.Module,
    *,
    train_mode: str,
    direct_pose_leg_train_only: bool = False,
    direct_pose_leg_gate_train_only: bool = False,
    direct_pose_nonleg_train_only: bool = False,
) -> None:
    if train_mode == "direct":
        _unfreeze_direct_pose(
            model,
            leg_only=bool(direct_pose_leg_train_only),
            leg_gate_only=bool(direct_pose_leg_gate_train_only),
            nonleg_only=bool(direct_pose_nonleg_train_only),
        )
        return
    if train_mode == "lambda":
        _enable_modules(model, ("lambda_fusion_head",))
        return
    raise ValueError(f"Unknown train_mode={train_mode!r}")
