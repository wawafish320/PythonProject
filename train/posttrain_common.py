#!/usr/bin/env python3
"""
Shared helpers extracted from train.posttrain.

This module intentionally contains lightweight utilities that are also used by
diagnostic tools, so those tools do not need to import the full posttrain entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _merge_norm_spec(bundle_path: Path, pretrain_path: Optional[Path]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {}
    try:
        with bundle_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception:
        spec = {}
    if pretrain_path is not None and pretrain_path.is_file():
        try:
            with pretrain_path.open("r", encoding="utf-8") as f:
                extra = json.load(f)
            if isinstance(extra, dict):
                spec = dict(extra, **spec)
        except Exception:
            pass
    return spec


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


def _unfreeze_direct_pose(
    model: torch.nn.Module,
    *,
    leg_only: bool = False,
    leg_gate_only: bool = False,
    nonleg_only: bool = False,
) -> None:
    if bool(leg_gate_only):
        # Train only the leg gate/scale head (keep base direct + leg omega frozen).
        if bool(getattr(model, "direct_pose_leg_side_routing", False)) and getattr(model, "direct_pose_leg_gate_head_shared", None) is not None:
            gate_leg = getattr(model, "direct_pose_leg_gate_head_shared", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        else:
            gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        return
    if bool(leg_only):
        # Prefer the routed shared head when enabled (legacy head is unused in forward in that mode).
        if bool(getattr(model, "direct_pose_leg_side_routing", False)) and getattr(model, "direct_pose_leg_head_shared", None) is not None:
            leg = getattr(model, "direct_pose_leg_head_shared", None)
            if leg is not None:
                for p in leg.parameters():
                    p.requires_grad_(True)
            gate_leg = getattr(model, "direct_pose_leg_gate_head_shared", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
            gate = getattr(model, "direct_pose_leg_side_sign_gate_head", None)
            if gate is not None:
                for p in gate.parameters():
                    p.requires_grad_(True)
            emb = getattr(model, "direct_pose_leg_side_embed", None)
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(True)
        else:
            leg = getattr(model, "direct_pose_leg_head", None)
            if leg is not None:
                for p in leg.parameters():
                    p.requires_grad_(True)
            gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
            if gate_leg is not None:
                for p in gate_leg.parameters():
                    p.requires_grad_(True)
        return
    if bool(nonleg_only):
        arm_proj = getattr(model, "direct_pose_arm_proj", None)
        if arm_proj is not None:
            for p in arm_proj.parameters():
                p.requires_grad_(True)
        else_proj = getattr(model, "direct_pose_else_proj", None)
        if else_proj is not None:
            for p in else_proj.parameters():
                p.requires_grad_(True)
        nonleg_proj = getattr(model, "direct_pose_nonleg_proj", None)
        if nonleg_proj is not None:
            for p in nonleg_proj.parameters():
                p.requires_grad_(True)
        out_arm = getattr(model, "direct_pose_out_arm", None)
        if out_arm is not None:
            for p in out_arm.parameters():
                p.requires_grad_(True)
        out_else = getattr(model, "direct_pose_out_else", None)
        if out_else is not None:
            for p in out_else.parameters():
                p.requires_grad_(True)
        out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
        if out_nonleg is not None:
            for p in out_nonleg.parameters():
                p.requires_grad_(True)
        return

    head = getattr(model, "direct_pose_head", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad_(True)
    out_leg = getattr(model, "direct_pose_out_leg", None)
    if out_leg is not None:
        for p in out_leg.parameters():
            p.requires_grad_(True)
    out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
    if out_nonleg is not None:
        for p in out_nonleg.parameters():
            p.requires_grad_(True)
    out_arm = getattr(model, "direct_pose_out_arm", None)
    if out_arm is not None:
        for p in out_arm.parameters():
            p.requires_grad_(True)
    out_else = getattr(model, "direct_pose_out_else", None)
    if out_else is not None:
        for p in out_else.parameters():
            p.requires_grad_(True)
    arm_proj = getattr(model, "direct_pose_arm_proj", None)
    if arm_proj is not None:
        for p in arm_proj.parameters():
            p.requires_grad_(True)
    else_proj = getattr(model, "direct_pose_else_proj", None)
    if else_proj is not None:
        for p in else_proj.parameters():
            p.requires_grad_(True)
    leg = getattr(model, "direct_pose_leg_head", None)
    if leg is not None:
        for p in leg.parameters():
            p.requires_grad_(True)
    leg_shared = getattr(model, "direct_pose_leg_head_shared", None)
    if leg_shared is not None:
        for p in leg_shared.parameters():
            p.requires_grad_(True)
    gate_leg = getattr(model, "direct_pose_leg_gate_head", None)
    if gate_leg is not None:
        for p in gate_leg.parameters():
            p.requires_grad_(True)
    gate_leg_shared = getattr(model, "direct_pose_leg_gate_head_shared", None)
    if gate_leg_shared is not None:
        for p in gate_leg_shared.parameters():
            p.requires_grad_(True)
    gate = getattr(model, "direct_pose_leg_side_sign_gate_head", None)
    if gate is not None:
        for p in gate.parameters():
            p.requires_grad_(True)
    emb = getattr(model, "direct_pose_leg_side_embed", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad_(True)

