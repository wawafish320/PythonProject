#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import _load_case  # noqa: E402
from train import posttrain  # noqa: E402


RUN_DATE = "20260407"
DEFAULT_TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
DEFAULT_BASELINE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "configs"
    / "posttrain_70b_replace_lowdrift_fromfresh_20260317.json"
)
DEFAULT_TAIL_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_from_70a_20260402"
    / "configs"
    / "posttrain_70b_replace_lowdrift_lr5e5_from_cp015_tailk7_70a_20260402.json"
)
DEFAULT_BASELINE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
DEFAULT_TAIL_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_from_70a_20260402"
    / "lr5e5"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_cp015_tailk7_70a_20260402.pth"
)
DEFAULT_BASELINE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)
DEFAULT_TAIL_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_from_70a_20260402"
    / "eval_model_source"
    / "lr5e5"
    / "Walk_F_freerun_cycles.json"
)
DEFAULT_OUT = (
    ROOT
    / "debug_output"
    / f"_tmp_cp015_tailk7_same_input_module_attribution_{RUN_DATE}"
    / "summary.json"
)

REQUIRED_ACTIVATION_MODULES: tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_leg_head",
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_out_leg",
    "direct_pose_out_arm",
    "direct_pose_out_else",
)
COUNTERFACTUAL_MODULES: tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_out_leg",
    "direct_pose_out_arm",
    "direct_pose_out_else",
)
STAGED_DIRECT_MODULES: tuple[str, ...] = REQUIRED_ACTIVATION_MODULES
NEAR_SUFFICIENT_THRESHOLD = 0.95


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _clone_tensor(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _clone_nested(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {str(k): _clone_nested(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_nested(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_nested(v) for v in x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return x


def _flatten(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    return x.detach().reshape(-1).to(dtype=torch.float32)


def _mean_abs(x: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(x):
        return float("nan")
    try:
        return float(x.detach().abs().mean().item())
    except Exception:
        return float("nan")


def _norm_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    va = _flatten(a)
    vb = _flatten(b)
    if va is None or vb is None or tuple(va.shape) != tuple(vb.shape) or int(va.numel()) <= 0:
        return float("nan")
    diff = va - vb
    return float(torch.linalg.vector_norm(diff).item() / math.sqrt(float(max(1, int(diff.numel())))))


def _cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    va = _flatten(a)
    vb = _flatten(b)
    if va is None or vb is None or tuple(va.shape) != tuple(vb.shape) or int(va.numel()) <= 0:
        return float("nan")
    na = float(torch.linalg.vector_norm(va).item())
    nb = float(torch.linalg.vector_norm(vb).item())
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(torch.dot(va, vb).item() / (na * nb))


def _tensor_shape(x: Any) -> Optional[List[int]]:
    if not torch.is_tensor(x):
        return None
    return [int(v) for v in x.shape]


def _clone_state_dict(state: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[str(key)] = value.detach().clone()
        else:
            out[str(key)] = copy.deepcopy(value)
    return out


def _normalize_time_index_mode(mode: str) -> str:
    out = str(mode or "global").strip().lower()
    if out == "auto":
        out = "global"
    if out not in ("global", "cycle", "none"):
        out = "global"
    return out


def _resolve_path_from_config(value: Any) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_config_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg_namespace(payload: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(payload))


def _case_bundle(
    *,
    case_name: str,
    ckpt_path: Path,
    eval_json_path: Path,
    teacher_path: Path,
    config_path: Path,
    device_pref: str,
) -> Dict[str, Any]:
    case = _load_case(
        case_name=case_name,
        ckpt_path=ckpt_path,
        eval_json_path=eval_json_path,
        teacher_path=teacher_path,
        device_pref=device_pref,
    )
    config_payload = _load_config_payload(config_path)
    cfg = _cfg_namespace(config_payload)
    train_mode = posttrain._resolve_train_mode(cfg)
    rollout_mode_kwargs = posttrain._build_rollout_mode_kwargs(cfg, train_mode)
    return {
        "case": case,
        "config_payload": config_payload,
        "cfg": cfg,
        "train_mode": str(train_mode),
        "rollout_mode_kwargs": dict(rollout_mode_kwargs),
    }


def _prepare_fixed_offset_context(bundle: Mapping[str, Any], *, offset: int) -> Dict[str, Any]:
    case = bundle["case"]
    cfg = bundle["cfg"]
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError(f"{case['case_name']}: missing model")
    batch = case["batched"]
    prep_ctx = posttrain._lambda_rollout_prepare_context(
        trainer,
        model,
        batch,
        columns=case["columns"],
        rollout_steps=int(getattr(cfg, "rollout_steps", 0) or 0),
        rollout_cycles=int(getattr(cfg, "rollout_cycles", 1) or 1),
        include_boundary=bool(getattr(cfg, "rollout_include_boundary", False)),
        boundary_weight=float(getattr(cfg, "lambda_boundary_weight", 1.0) or 0.0),
        random_offset=False,
        time_weight_mode=str(getattr(cfg, "lambda_time_weight_mode", "inv") or "inv"),
        time_weight_max=float(getattr(cfg, "lambda_time_weight_max", 1.0) or 1.0),
    )
    ctx = dict(prep_ctx)
    cycle_len = int(ctx["cycle_len"])
    total_steps = int(ctx["total_steps"])
    device = ctx["device"]
    dtype = ctx["dtype"]
    motion_seq = ctx["motion_seq"]
    pose_hist_seq = ctx["pose_hist_seq"]
    rot_slice = ctx["rot_slice"]
    Dy = int(ctx["Dy"])
    off = int(offset) % max(1, int(cycle_len))
    motion = motion_seq[:, off]
    motion_raw = trainer.normalizer.denorm_x(motion)
    y_prev_raw = posttrain._init_y_from_x(trainer.normalizer, motion_raw, Dy)
    pose_hist_state = posttrain.init_pose_hist_state(
        ref_tensor=motion_seq,
        pose_hist_seq=pose_hist_seq,
        y_prev_raw=y_prev_raw,
        rot_slice=rot_slice,
        pose_hist_len=int(getattr(trainer, "pose_hist_len", 0) or 0),
        pose_hist_dim=int(getattr(trainer, "pose_hist_dim", 0) or 0),
        params_fn=trainer._pose_hist_params,
        offset=int(off),
    )
    ctx["offset"] = int(off)
    ctx["state"] = {
        "motion": motion,
        "motion_raw": motion_raw,
        "y_prev_raw": y_prev_raw,
        "plan_z": None,
        "meas_logits_prev": None,
        "rot_slice": rot_slice,
        "pose_hist_state": pose_hist_state,
    }

    step_weights = posttrain._make_rollout_step_weights(
        total_steps,
        device=device,
        dtype=dtype,
        mode=str(getattr(bundle["cfg"], "lambda_time_weight_mode", "inv") or "inv"),
        max_val=float(getattr(bundle["cfg"], "lambda_time_weight_max", 1.0) or 1.0),
    )
    boundary_steps = 0
    boundary_weighted_sum = 0.0
    include_boundary = bool(ctx["include_boundary"])
    boundary_weight = float(getattr(bundle["cfg"], "lambda_boundary_weight", 1.0) or 0.0)
    if include_boundary:
        idxs = (torch.arange(int(total_steps), device=device) + int(off)) % int(cycle_len)
        boundary_mask = idxs == (int(cycle_len) - 1)
        boundary_steps = int(boundary_mask.sum().detach().cpu().item())
        bw = max(0.0, float(boundary_weight))
        if abs(bw - 1.0) > 1e-12:
            factors = torch.ones_like(step_weights)
            factors = torch.where(boundary_mask, step_weights.new_tensor(bw), factors)
            step_weights = step_weights * factors
            step_weights = step_weights / step_weights.sum().clamp_min(1e-6)
        boundary_weighted_sum = float(step_weights[boundary_mask].sum().detach().cpu().item())
    ctx["step_weights"] = step_weights
    ctx["boundary_steps"] = int(boundary_steps)
    ctx["boundary_weighted_sum"] = float(boundary_weighted_sum)
    return ctx


def _make_yaw_gt_fn(*, trainer: Any, prep_ctx: Mapping[str, Any]) -> Any:
    include_boundary = bool(prep_ctx["include_boundary"])
    cycle_len = int(prep_ctx["cycle_len"])
    y0_raw = prep_ctx["y0_raw"]
    gt_seq = prep_ctx["gt_seq"]

    def _yaw_gt(idx_step: int) -> Optional[torch.Tensor]:
        try:
            if include_boundary and y0_raw is not None and int(idx_step) == (cycle_len - 1):
                gt_raw_frame = y0_raw
            else:
                gt_idx = min(int(gt_seq.shape[1]) - 1, int(idx_step))
                gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
            return trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
        except Exception:
            return None

    return _yaw_gt


def _build_step_ctx(bundle: Mapping[str, Any], prep_ctx: Mapping[str, Any]) -> Dict[str, Any]:
    case = bundle["case"]
    cfg = bundle["cfg"]
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError(f"{case['case_name']}: missing model")
    objective = str(bundle["rollout_mode_kwargs"]["objective"])
    J = int(prep_ctx["J"])
    device = prep_ctx["device"]
    nonleg_focus_ctx = posttrain._lambda_rollout_resolve_nonleg_focus(
        trainer,
        objective=objective,
        direct_pose_nonleg_focus_bones=str(getattr(cfg, "direct_pose_nonleg_focus_bones", "") or ""),
        direct_pose_nonleg_focus_weight=float(getattr(cfg, "direct_pose_nonleg_focus_weight", 1.0) or 1.0),
        J=J,
        device=device,
    )
    reg_ctx = posttrain._lambda_rollout_build_reg_params(
        trainer,
        objective=objective,
        lambda_gate_sup_weight=float(getattr(cfg, "lambda_gate_sup_weight", 0.0) or 0.0),
        lambda_gate_sup_start_step=int(getattr(cfg, "lambda_gate_sup_start_step", -1) or -1),
        lambda_gate_sup_tau_deg=float(getattr(cfg, "lambda_gate_sup_tau_deg", 2.5) or 2.5),
        lambda_gate_sup_margin_deg=float(getattr(cfg, "lambda_gate_sup_margin_deg", 1.0) or 1.0),
        direct_pose_loss_group_norm_enable=bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
        direct_pose_loss_group_norm_w_leg=float(getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0) or 1.0),
        direct_pose_loss_group_norm_w_nonleg=float(getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0) or 1.0),
        direct_pose_loss_group_norm_ema_beta=float(
            getattr(cfg, "direct_pose_loss_group_norm_ema_beta", 0.95) or 0.95
        ),
        direct_pose_loss_group_norm_ratio_min=float(
            getattr(cfg, "direct_pose_loss_group_norm_ratio_min", 0.2) or 0.2
        ),
        direct_pose_loss_group_norm_ratio_max=float(
            getattr(cfg, "direct_pose_loss_group_norm_ratio_max", 5.0) or 5.0
        ),
        direct_pose_loss_group_norm_eps=float(getattr(cfg, "direct_pose_loss_group_norm_eps", 1e-6) or 1e-6),
        direct_pose_loss_3way_enable=bool(getattr(cfg, "direct_pose_loss_3way_enable", False)),
        direct_pose_loss_3way_w_leg=float(getattr(cfg, "direct_pose_loss_3way_w_leg", 1.0) or 1.0),
        direct_pose_loss_3way_w_arm=float(getattr(cfg, "direct_pose_loss_3way_w_arm", 1.0) or 1.0),
        direct_pose_loss_3way_w_else=float(getattr(cfg, "direct_pose_loss_3way_w_else", 1.0) or 1.0),
        direct_pose_loss_arm_else_balance_enable=bool(
            getattr(cfg, "direct_pose_loss_arm_else_balance_enable", False)
        ),
        direct_pose_loss_arm_weight=float(getattr(cfg, "direct_pose_loss_arm_weight", 1.0) or 1.0),
        direct_pose_loss_else_weight=float(getattr(cfg, "direct_pose_loss_else_weight", 1.0) or 1.0),
    )
    weights_ctx = {
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "direct_pose_leg_align_weight": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0) or 0.0),
        "direct_pose_leg_align_oracle_min_deg": float(
            getattr(cfg, "direct_pose_leg_align_oracle_min_deg", 0.0) or 0.0
        ),
        "direct_pose_leg_align_oracle_weight_deg": float(
            getattr(cfg, "direct_pose_leg_align_oracle_weight_deg", 0.0) or 0.0
        ),
        "direct_pose_leg_align_mode": str(getattr(cfg, "direct_pose_leg_align_mode", "cos") or "cos"),
        "direct_pose_leg_align_mag_weight": float(getattr(cfg, "direct_pose_leg_align_mag_weight", 1.0) or 1.0),
        "direct_pose_leg_align_res_weight": float(getattr(cfg, "direct_pose_leg_align_res_weight", 1.0) or 1.0),
        "direct_pose_leg_align_sign_weight": float(getattr(cfg, "direct_pose_leg_align_sign_weight", 0.0) or 0.0),
        "direct_pose_leg_align_cos_thresh": float(getattr(cfg, "direct_pose_leg_align_cos_thresh", 0.0) or 0.0),
        "direct_pose_leg_align_target_joints": getattr(cfg, "direct_pose_leg_align_target_joints", None),
        "direct_pose_leg_align_anchor_joints": getattr(cfg, "direct_pose_leg_align_anchor_joints", None),
        "direct_pose_leg_align_anchor_weight": float(getattr(cfg, "direct_pose_leg_align_anchor_weight", 0.0) or 0.0),
        "direct_pose_leg_gate_sup_weight": float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0),
        "direct_pose_loss_leg_split": bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
        "direct_nonleg_focus_mask_j": nonleg_focus_ctx["direct_nonleg_focus_mask_j"],
        "direct_nonleg_focus_resolved": int(nonleg_focus_ctx["direct_nonleg_focus_resolved"]),
        "direct_nonleg_focus_weight_use": float(nonleg_focus_ctx["direct_nonleg_focus_weight_use"]),
        "direct_pose_loss_3way_enable": bool(getattr(cfg, "direct_pose_loss_3way_enable", False)),
        "direct_pose_loss_3way_w_leg": float(getattr(cfg, "direct_pose_loss_3way_w_leg", 1.0) or 1.0),
        "direct_pose_loss_3way_w_arm": float(getattr(cfg, "direct_pose_loss_3way_w_arm", 1.0) or 1.0),
        "direct_pose_loss_3way_w_else": float(getattr(cfg, "direct_pose_loss_3way_w_else", 1.0) or 1.0),
        "direct_pose_loss_arm_else_balance_enable": bool(
            getattr(cfg, "direct_pose_loss_arm_else_balance_enable", False)
        ),
        "direct_pose_loss_arm_weight": float(getattr(cfg, "direct_pose_loss_arm_weight", 1.0) or 1.0),
        "direct_pose_loss_else_weight": float(getattr(cfg, "direct_pose_loss_else_weight", 1.0) or 1.0),
        "gate_sup_weight": float(reg_ctx["gate_sup_weight"]),
        "gate_sup_start": int(reg_ctx["gate_sup_start"]),
        "tau_rad": float(reg_ctx["tau_rad"]),
        "margin_rad": float(reg_ctx["margin_rad"]),
        "lambda_plan_entropy_weight": float(bundle["rollout_mode_kwargs"].get("lambda_plan_entropy_weight", 0.0) or 0.0),
        "lambda_plan_dyn_weight": float(bundle["rollout_mode_kwargs"].get("lambda_plan_dyn_weight", 0.0) or 0.0),
        "lambda_early_weight": float(bundle["rollout_mode_kwargs"].get("lambda_early_weight", 0.0) or 0.0),
        "lambda_early_steps": int(bundle["rollout_mode_kwargs"].get("lambda_early_steps", 0) or 0),
        "lambda_entropy_weight": float(bundle["rollout_mode_kwargs"].get("lambda_entropy_weight", 0.0) or 0.0),
        "lambda_smooth_weight": float(bundle["rollout_mode_kwargs"].get("lambda_smooth_weight", 0.0) or 0.0),
        "lambda_monotonic_weight": float(bundle["rollout_mode_kwargs"].get("lambda_monotonic_weight", 0.0) or 0.0),
    }
    state_vars = {
        "meas_used_logits": False,
        "direct_nonleg_focus_applied": float(nonleg_focus_ctx["direct_nonleg_focus_applied"]),
        "lam_prev": None,
        "lam_prev_monot": None,
        "plan_prev": None,
    }
    time_base = None
    batch = case["batched"]
    if isinstance(batch, dict):
        base = batch.get("start", None)
        if base is not None:
            time_base = base.to(device=prep_ctx["device"]) if torch.is_tensor(base) else base
    yaw_gt_fn = _make_yaw_gt_fn(trainer=trainer, prep_ctx=prep_ctx)
    accum_ctx = posttrain._lambda_fusion_init_accum_ctx()
    return posttrain._build_rollout_unroll_ctx(
        trainer=trainer,
        model=model,
        state=_clone_state_dict(prep_ctx["state"]),
        prep_ctx=dict(prep_ctx),
        time_index_mode=_normalize_time_index_mode(str(getattr(cfg, "time_index_mode", "global") or "global")),
        time_base=time_base,
        enable_reprojection=bool(getattr(trainer, "enable_cond_reprojection", True)),
        detach_rollout_state=bool(getattr(cfg, "detach_rollout_state", True)),
        yaw_gt_fn=yaw_gt_fn,
        columns=case["columns"],
        objective=objective,
        weights_ctx=weights_ctx,
        accum_ctx=accum_ctx,
        state_vars=state_vars,
    )


def _temporary_weight_swap(
    *,
    target_model: torch.nn.Module,
    donor_model: torch.nn.Module,
    module_names: Sequence[str],
) -> List[tuple[torch.nn.Module, Dict[str, Any]]]:
    backups: List[tuple[torch.nn.Module, Dict[str, Any]]] = []
    for name in module_names:
        target_module = getattr(target_model, str(name), None)
        donor_module = getattr(donor_model, str(name), None)
        if target_module is None or donor_module is None:
            continue
        backups.append((target_module, copy.deepcopy(target_module.state_dict())))
        target_module.load_state_dict(copy.deepcopy(donor_module.state_dict()))
    return backups


def _restore_weight_swap(backups: Sequence[tuple[torch.nn.Module, Dict[str, Any]]]) -> None:
    for module, state_dict in backups:
        module.load_state_dict(state_dict)


def _run_single_step(
    bundle: Mapping[str, Any],
    prep_ctx: Mapping[str, Any],
    *,
    fixed_contacts: Optional[torch.Tensor],
    plan_override: Optional[torch.Tensor] = None,
    activation_swap: Optional[Mapping[str, torch.Tensor]] = None,
    weight_swap_modules: Optional[Sequence[str]] = None,
    donor_bundle: Optional[Mapping[str, Any]] = None,
    capture_modules: Sequence[str] = REQUIRED_ACTIVATION_MODULES,
) -> Dict[str, Any]:
    case = bundle["case"]
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError(f"{case['case_name']}: missing model")
    donor_model = donor_bundle["case"]["trainer"].model if donor_bundle is not None else None
    if weight_swap_modules and donor_model is None:
        raise RuntimeError("weight swap requested but donor bundle/model missing")

    records: Dict[str, Any] = {
        "inputs": None,
        "ret": None,
        "activations": {},
    }
    handles: List[Any] = []
    activation_swap = dict(activation_swap or {})
    plan_override_orig = getattr(model, "direct_pose_plan_override", None)
    backups: List[tuple[torch.nn.Module, Dict[str, Any]]] = []
    orig_forward = model.forward
    orig_prepare_contacts = posttrain._prepare_rollout_contacts_input

    def _forward_wrapper(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        records["inputs"] = {
            "state": _clone_nested(args[0] if len(args) > 0 else kwargs.get("state", None)),
            "cond": _clone_nested(args[1] if len(args) > 1 else kwargs.get("cond", None)),
            "contacts": _clone_nested(kwargs.get("contacts", None)),
            "angvel": _clone_nested(kwargs.get("angvel", None)),
            "pose_history": _clone_nested(kwargs.get("pose_history", None)),
            "plan_z": _clone_nested(kwargs.get("plan_z", None)),
            "meas_logits_prev": _clone_nested(kwargs.get("meas_logits_prev", None)),
            "time_index": _clone_nested(kwargs.get("time_index", None)),
            "rollout_step": _clone_nested(kwargs.get("rollout_step", None)),
        }
        ret = orig_forward(*args, **kwargs)
        records["ret"] = _clone_nested(ret)
        return ret

    def _prepare_contacts_override(
        trainer_: Any,
        model_: Any,
        *,
        motion_t: torch.Tensor,
        pose_hist_t: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        _ = trainer_, model_, motion_t, pose_hist_t
        if fixed_contacts is None:
            return orig_prepare_contacts(trainer_, model_, motion_t=motion_t, pose_hist_t=pose_hist_t)
        return fixed_contacts.detach().clone().to(device=motion_t.device, dtype=motion_t.dtype)

    try:
        model.forward = types.MethodType(_forward_wrapper, model)
        posttrain._prepare_rollout_contacts_input = _prepare_contacts_override
        if plan_override is not None:
            setattr(model, "direct_pose_plan_override", plan_override.detach().clone())
        if weight_swap_modules:
            backups = _temporary_weight_swap(
                target_model=model,
                donor_model=donor_model,
                module_names=weight_swap_modules,
            )

        for name in capture_modules:
            module = getattr(model, str(name), None)
            if module is None:
                continue

            def _hook(_module: Any, _inputs: Any, output: Any, *, _name: str = str(name)) -> Any:
                if torch.is_tensor(output):
                    records["activations"][_name] = output.detach().clone()
                    replacement = activation_swap.get(_name, None)
                    if torch.is_tensor(replacement):
                        return replacement.detach().clone().to(device=output.device, dtype=output.dtype)
                return None

            handles.append(module.register_forward_hook(_hook))

        ctx = _build_step_ctx(bundle, prep_ctx)
        posttrain._lambda_rollout_unroll_single_step(t=0, ctx=ctx)
        accum = ctx["accum"]
        out = {
            "inputs": records["inputs"],
            "ret": records["ret"],
            "activations": records["activations"],
            "metrics": {
                "loss_terms_0": _safe_float(accum["loss_terms"][0] if accum["loss_terms"] else float("nan")),
                "inc_terms_0": _safe_float(accum["inc_terms"][0] if accum["inc_terms"] else float("nan")),
                "dir_base_terms_0": _safe_float(accum["dir_base_terms"][0] if accum["dir_base_terms"] else float("nan")),
                "dir_leg_base_terms_0": _safe_float(
                    accum["dir_leg_base_terms"][0] if accum["dir_leg_base_terms"] else float("nan")
                ),
                "dir_nonleg_base_terms_0": _safe_float(
                    accum["dir_nonleg_base_terms"][0] if accum["dir_nonleg_base_terms"] else float("nan")
                ),
            },
        }
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        model.forward = orig_forward
        posttrain._prepare_rollout_contacts_input = orig_prepare_contacts
        setattr(model, "direct_pose_plan_override", plan_override_orig)
        if backups:
            _restore_weight_swap(backups)
    return out


def _activation_row(module: str, base_tensor: Optional[torch.Tensor], tail_tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    return {
        "module": str(module),
        "shape": _tensor_shape(base_tensor) or _tensor_shape(tail_tensor),
        "baseline_mean_abs": _mean_abs(base_tensor),
        "tail_mean_abs": _mean_abs(tail_tensor),
        "delta_l2": _norm_l2(base_tensor, tail_tensor),
        "cosine": _cosine(base_tensor, tail_tensor),
    }


def _closure_ratio(orig_gap: float, swapped_gap: float) -> float:
    if (not math.isfinite(orig_gap)) or abs(orig_gap) <= 1e-12:
        return float("nan")
    return float(1.0 - (abs(swapped_gap) / abs(orig_gap)))


def _counterfactual_row(
    *,
    label: str,
    swap_type: str,
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    tail_result: Mapping[str, Any],
) -> Dict[str, Any]:
    base_ret = baseline_result["ret"]
    tail_ret = tail_result["ret"]
    cand_ret = result["ret"]
    orig_out_gap = _norm_l2(
        base_ret.get("out_direct") if isinstance(base_ret, Mapping) else None,
        tail_ret.get("out_direct") if isinstance(tail_ret, Mapping) else None,
    )
    cand_out_gap = _norm_l2(
        base_ret.get("out_direct") if isinstance(base_ret, Mapping) else None,
        cand_ret.get("out_direct") if isinstance(cand_ret, Mapping) else None,
    )
    orig_dir_gap = _safe_float(tail_result["metrics"]["dir_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_base_terms_0"]
    )
    cand_dir_gap = _safe_float(result["metrics"]["dir_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_base_terms_0"]
    )
    orig_leg_gap = _safe_float(tail_result["metrics"]["dir_leg_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_leg_base_terms_0"]
    )
    cand_leg_gap = _safe_float(result["metrics"]["dir_leg_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_leg_base_terms_0"]
    )
    orig_nonleg_gap = _safe_float(tail_result["metrics"]["dir_nonleg_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_nonleg_base_terms_0"]
    )
    cand_nonleg_gap = _safe_float(result["metrics"]["dir_nonleg_base_terms_0"]) - _safe_float(
        baseline_result["metrics"]["dir_nonleg_base_terms_0"]
    )
    return {
        "label": str(label),
        "swap_type": str(swap_type),
        "original_out_direct_gap": float(orig_out_gap),
        "swapped_out_direct_gap": float(cand_out_gap),
        "original_dir_base_gap": float(orig_dir_gap),
        "swapped_dir_base_gap": float(cand_dir_gap),
        "original_dir_leg_gap": float(orig_leg_gap),
        "swapped_dir_leg_gap": float(cand_leg_gap),
        "original_dir_nonleg_gap": float(orig_nonleg_gap),
        "swapped_dir_nonleg_gap": float(cand_nonleg_gap),
        "out_direct_closure_ratio": _closure_ratio(orig_out_gap, cand_out_gap),
        "dir_base_closure_ratio": _closure_ratio(orig_dir_gap, cand_dir_gap),
        "dir_leg_closure_ratio": _closure_ratio(orig_leg_gap, cand_leg_gap),
        "dir_nonleg_closure_ratio": _closure_ratio(orig_nonleg_gap, cand_nonleg_gap),
    }


def _importance_rank(rows: Sequence[MutableMapping[str, Any]]) -> List[Dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: _safe_float(row.get("delta_l2")), reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        rec = dict(row)
        rec["rank"] = int(rank)
        out.append(rec)
    return out


def _subset_label(modules: Sequence[str]) -> str:
    items = [str(name) for name in modules]
    return "+".join(items) if items else "none"


def _total_closure_score(row: Mapping[str, Any]) -> float:
    total = 0.0
    for key in (
        "out_direct_closure_ratio",
        "dir_base_closure_ratio",
        "dir_leg_closure_ratio",
        "dir_nonleg_closure_ratio",
    ):
        value = _safe_float(row.get(key))
        if math.isfinite(value):
            total += value
    return float(total)


def _is_near_sufficient(row: Mapping[str, Any], *, threshold: float = NEAR_SUFFICIENT_THRESHOLD) -> bool:
    return (
        _safe_float(row.get("out_direct_closure_ratio")) >= float(threshold)
        and _safe_float(row.get("dir_base_closure_ratio")) >= float(threshold)
    )


def _role_tags(*, swap_type: str, row: Mapping[str, Any]) -> List[str]:
    modules = {str(name) for name in row.get("modules", [])}
    tags: List[str] = []
    if "direct_pose_head" in modules:
        tags.append("head-anchor")
    if (
        _safe_float(row.get("dir_leg_closure_ratio")) >= 0.95
        and abs(_safe_float(row.get("dir_nonleg_closure_ratio"))) <= 0.25
    ):
        tags.append("leg-readout")
    if (
        _safe_float(row.get("dir_nonleg_closure_ratio")) >= 0.95
        and abs(_safe_float(row.get("dir_leg_closure_ratio"))) <= 0.25
    ):
        tags.append("nonleg-readout")
    if swap_type == "weight" and modules.intersection({"direct_pose_arm_proj", "direct_pose_else_proj"}):
        tags.append("branch-adapter")
    if swap_type == "weight" and "direct_pose_head" in modules and len(modules) >= 5:
        tags.append("joint-contract")
    if _is_near_sufficient(row):
        tags.append("near-sufficient")
    if not tags:
        tags.append("insufficient")
    return tags


def _subset_candidate_row(row: Mapping[str, Any], *, interpretation: str) -> Dict[str, Any]:
    return {
        "label": str(row.get("label")),
        "swap_type": str(row.get("swap_type")),
        "modules": [str(name) for name in row.get("modules", [])],
        "size": int(row.get("size", 0) or 0),
        "out_direct_gap": float(_safe_float(row.get("swapped_out_direct_gap"))),
        "dir_base_gap": float(_safe_float(row.get("swapped_dir_base_gap"))),
        "dir_leg_gap": float(_safe_float(row.get("swapped_dir_leg_gap"))),
        "dir_nonleg_gap": float(_safe_float(row.get("swapped_dir_nonleg_gap"))),
        "out_direct_closure_ratio": float(_safe_float(row.get("out_direct_closure_ratio"))),
        "dir_base_closure_ratio": float(_safe_float(row.get("dir_base_closure_ratio"))),
        "dir_leg_closure_ratio": float(_safe_float(row.get("dir_leg_closure_ratio"))),
        "dir_nonleg_closure_ratio": float(_safe_float(row.get("dir_nonleg_closure_ratio"))),
        "improves_over_best_parent": bool(row.get("improves_over_best_parent")),
        "best_parent_label": row.get("best_parent_label"),
        "role_tags": list(row.get("role_tags", [])),
        "staged_interpretation": str(interpretation),
    }


def _run_subset_search(
    *,
    swap_type: str,
    modules: Sequence[str],
    tail_bundle: Mapping[str, Any],
    prep_tail: Mapping[str, Any],
    baseline_bundle: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    tail_result: Mapping[str, Any],
    fixed_contacts: Optional[torch.Tensor],
) -> tuple[List[Dict[str, Any]], Dict[frozenset[str], Dict[str, Any]]]:
    if str(swap_type) not in ("weight", "activation"):
        raise RuntimeError(f"unsupported swap_type={swap_type}")

    rows: List[Dict[str, Any]] = []
    row_map: Dict[frozenset[str], Dict[str, Any]] = {}
    ordered_modules = [str(name) for name in modules]

    for size in range(len(ordered_modules) + 1):
        for subset in itertools.combinations(ordered_modules, size):
            subset_list = [str(name) for name in subset]
            if str(swap_type) == "weight":
                result = _run_single_step(
                    tail_bundle,
                    prep_tail,
                    fixed_contacts=fixed_contacts,
                    weight_swap_modules=subset_list or None,
                    donor_bundle=baseline_bundle if subset_list else None,
                )
            else:
                activation_swap = {
                    str(name): baseline_result["activations"].get(str(name))
                    for name in subset_list
                    if torch.is_tensor(baseline_result["activations"].get(str(name)))
                }
                result = _run_single_step(
                    tail_bundle,
                    prep_tail,
                    fixed_contacts=fixed_contacts,
                    activation_swap=activation_swap,
                )

            row = _counterfactual_row(
                label=f"{swap_type}_subset:{_subset_label(subset_list)}",
                swap_type=f"{swap_type}_subset",
                result=result,
                baseline_result=baseline_result,
                tail_result=tail_result,
            )
            row["modules"] = subset_list
            row["size"] = int(len(subset_list))
            row["subset_key"] = _subset_label(subset_list)
            row["role_tags"] = _role_tags(swap_type=str(swap_type), row=row)
            row["total_closure_score"] = _total_closure_score(row)
            row_map[frozenset(subset_list)] = row
            rows.append(row)

    for row in rows:
        modules_set = frozenset(str(name) for name in row.get("modules", []))
        if not modules_set:
            row["best_parent_label"] = None
            row["best_parent_total_closure_score"] = None
            row["improves_over_best_parent"] = False
            continue
        parents = [row_map[modules_set - {name}] for name in modules_set]
        best_parent = max(parents, key=_total_closure_score)
        row["best_parent_label"] = best_parent.get("label")
        row["best_parent_total_closure_score"] = float(_total_closure_score(best_parent))
        row["improves_over_best_parent"] = bool(
            _total_closure_score(row) > (_total_closure_score(best_parent) + 1e-12)
        )

    rows = sorted(
        rows,
        key=lambda row: (
            _is_near_sufficient(row),
            _safe_float(row.get("dir_base_closure_ratio")),
            _safe_float(row.get("out_direct_closure_ratio")),
            _safe_float(row.get("dir_leg_closure_ratio")),
            _safe_float(row.get("dir_nonleg_closure_ratio")),
            -int(row.get("size", 0) or 0),
        ),
        reverse=True,
    )
    return rows, row_map


def _pick_stage1_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_key: str,
    non_target_key: str,
) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if int(row.get("size", 0) or 0) > 0 and not _is_near_sufficient(row)
    ]
    if not candidates:
        raise RuntimeError("no valid stage1 candidates")
    best_target = max(_safe_float(row.get(target_key)) for row in candidates)
    target_filtered = [
        row
        for row in candidates
        if abs(_safe_float(row.get(target_key)) - best_target) <= 1e-9
    ]
    target_filtered = sorted(
        target_filtered,
        key=lambda row: (
            abs(_safe_float(row.get(non_target_key))),
            int(row.get("size", 0) or 0),
            -_safe_float(row.get("dir_base_closure_ratio")),
            -_safe_float(row.get("out_direct_closure_ratio")),
        ),
    )
    return dict(target_filtered[0])


def _pick_stage2_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage1_modules: Sequence[str],
    threshold: float = NEAR_SUFFICIENT_THRESHOLD,
) -> Dict[str, Any]:
    stage1_set = frozenset(str(name) for name in stage1_modules)
    candidates = [
        row
        for row in rows
        if stage1_set.issubset(frozenset(str(name) for name in row.get("modules", [])))
        and int(row.get("size", 0) or 0) > len(stage1_set)
        and _is_near_sufficient(row, threshold=float(threshold))
    ]
    if not candidates:
        raise RuntimeError(f"no stage2 candidates for stage1={sorted(stage1_set)}")
    candidates = sorted(
        candidates,
        key=lambda row: (
            len(set(str(name) for name in row.get("modules", [])) - set(stage1_set)),
            int(row.get("size", 0) or 0),
            -_safe_float(row.get("dir_leg_closure_ratio")),
            -_safe_float(row.get("dir_nonleg_closure_ratio")),
            -_safe_float(row.get("dir_base_closure_ratio")),
            -_safe_float(row.get("out_direct_closure_ratio")),
        ),
    )
    return dict(candidates[0])


def _lookup_subset(row_map: Mapping[frozenset[str], Mapping[str, Any]], modules: Sequence[str]) -> Dict[str, Any]:
    key = frozenset(str(name) for name in modules)
    row = row_map.get(key)
    if row is None:
        raise RuntimeError(f"missing subset row: {sorted(key)}")
    return dict(row)


def _path_summary(stage1: Mapping[str, Any], stage2: Mapping[str, Any]) -> Dict[str, Any]:
    stage1_modules = [str(name) for name in stage1.get("modules", [])]
    stage2_modules = [str(name) for name in stage2.get("modules", [])]
    add_modules = [name for name in stage2_modules if name not in set(stage1_modules)]
    return {
        "stage1_set": stage1_modules,
        "stage1_size": int(stage1.get("size", 0) or 0),
        "stage1_closure": {
            "out_direct": float(_safe_float(stage1.get("out_direct_closure_ratio"))),
            "dir_base": float(_safe_float(stage1.get("dir_base_closure_ratio"))),
            "dir_leg": float(_safe_float(stage1.get("dir_leg_closure_ratio"))),
            "dir_nonleg": float(_safe_float(stage1.get("dir_nonleg_closure_ratio"))),
        },
        "stage2_add_modules": add_modules,
        "stage2_set": stage2_modules,
        "stage2_size": int(stage2.get("size", 0) or 0),
        "final_closure": {
            "out_direct": float(_safe_float(stage2.get("out_direct_closure_ratio"))),
            "dir_base": float(_safe_float(stage2.get("dir_base_closure_ratio"))),
            "dir_leg": float(_safe_float(stage2.get("dir_leg_closure_ratio"))),
            "dir_nonleg": float(_safe_float(stage2.get("dir_nonleg_closure_ratio"))),
        },
    }


def _module_graph_rows(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = bundle["case"]["trainer"].model
    feat_source = str(getattr(model, "direct_pose_feat_source", "cond") or "cond")
    meas_mode = str(getattr(model, "direct_pose_meas_mode", "concat") or "concat")
    inject_mode = str(getattr(model, "contact_plan_inject", "none") or "none")
    time_pe_dim = int(getattr(model, "direct_pose_time_pe_dim", 0) or 0)
    return [
        {
            "module": "contact_plan_path",
            "input_source": "cond (+ contacts_meas/delta_meas/event_clock inside plan path)",
            "output_tensor": "ret.contacts_plan / ret.plan_z_next",
            "downstream_consumers": f"direct_pose_head plan_in; shared_encoder via contact_plan_inject={inject_mode}",
            "where_in_code": "train/models.py:2881",
        },
        {
            "module": "direct_pose_head",
            "input_source": f"direct_feat={feat_source} + contacts_plan + contacts_meas(mode={meas_mode}) + time_pe(dim={time_pe_dim})",
            "output_tensor": "shared direct trunk hidden",
            "downstream_consumers": "direct_pose_out_leg; direct_pose_arm_proj; direct_pose_else_proj",
            "where_in_code": "train/models.py:3634",
        },
        {
            "module": "direct_pose_arm_proj",
            "input_source": "direct_pose_head hidden",
            "output_tensor": "arm projected hidden",
            "downstream_consumers": "direct_pose_out_arm",
            "where_in_code": "train/models.py:2230",
        },
        {
            "module": "direct_pose_else_proj",
            "input_source": "direct_pose_head hidden",
            "output_tensor": "else projected hidden",
            "downstream_consumers": "direct_pose_out_else",
            "where_in_code": "train/models.py:2233",
        },
        {
            "module": "direct_pose_out_leg",
            "input_source": "direct_pose_head hidden",
            "output_tensor": "leg rot/output slice",
            "downstream_consumers": "scatter -> out_direct",
            "where_in_code": "train/models.py:2218",
        },
        {
            "module": "direct_pose_out_arm",
            "input_source": "direct_pose_arm_proj output",
            "output_tensor": "arm rot/output slice",
            "downstream_consumers": "scatter -> out_direct",
            "where_in_code": "train/models.py:2236",
        },
        {
            "module": "direct_pose_out_else",
            "input_source": "direct_pose_else_proj output",
            "output_tensor": "else rot/output slice",
            "downstream_consumers": "scatter -> out_direct",
            "where_in_code": "train/models.py:2237",
        },
        {
            "module": "direct_pose_leg_head",
            "input_source": f"same direct head input branch ({feat_source}+plan+meas+time_pe)",
            "output_tensor": "ret.direct_leg_omega(_raw)",
            "downstream_consumers": "posttrain leg adjustment -> direct_raw_base -> R_dir",
            "where_in_code": "train/models.py:4133",
        },
        {
            "module": "out_direct_to_loss",
            "input_source": "ret.out_direct (+ optional leg/arm residual adjustments)",
            "output_tensor": "R_dir -> e_dir -> L_leg_base / L_nonleg_base / dir_base_terms[0]",
            "downstream_consumers": "step-0 direct loss profile",
            "where_in_code": "train/posttrain.py:3017",
        },
    ]


def _support_row(*, hypothesis: str, support_level: str, strongest_evidence: str, weakest_point: str, next_best: str) -> Dict[str, Any]:
    return {
        "hypothesis": str(hypothesis),
        "support_level": str(support_level),
        "strongest_evidence": str(strongest_evidence),
        "weakest_point": str(weakest_point),
        "next_best_minimal_intervention": str(next_best),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Same-input single-step module attribution for cp015 tailk7 replace.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    ap.add_argument("--tail-config", type=Path, default=DEFAULT_TAIL_CONFIG)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--offset", type=int, default=45)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    teacher = args.teacher.expanduser().resolve()
    baseline_config = args.baseline_config.expanduser().resolve()
    tail_config = args.tail_config.expanduser().resolve()
    baseline_ckpt = args.baseline_ckpt.expanduser().resolve()
    tail_ckpt = args.tail_ckpt.expanduser().resolve()
    baseline_eval = args.baseline_eval.expanduser().resolve()
    tail_eval = args.tail_eval.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    for path in (teacher, baseline_config, tail_config, baseline_ckpt, tail_ckpt, baseline_eval, tail_eval):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    baseline = _case_bundle(
        case_name="baseline_replace",
        ckpt_path=baseline_ckpt,
        eval_json_path=baseline_eval,
        teacher_path=teacher,
        config_path=baseline_config,
        device_pref=str(args.device),
    )
    tail = _case_bundle(
        case_name="tailk7_full_replace",
        ckpt_path=tail_ckpt,
        eval_json_path=tail_eval,
        teacher_path=teacher,
        config_path=tail_config,
        device_pref=str(args.device),
    )

    prep_base = _prepare_fixed_offset_context(baseline, offset=int(args.offset))
    prep_tail = _prepare_fixed_offset_context(tail, offset=int(args.offset))

    baseline_native = _run_single_step(baseline, prep_base, fixed_contacts=None)
    baseline_contacts = (
        baseline_native["inputs"]["contacts"]
        if isinstance(baseline_native.get("inputs"), Mapping)
        else None
    )
    tail_same_input = _run_single_step(tail, prep_tail, fixed_contacts=baseline_contacts)

    baseline_plan = (
        baseline_native["ret"].get("contacts_plan")
        if isinstance(baseline_native.get("ret"), Mapping)
        else None
    )
    tail_plan = (
        tail_same_input["ret"].get("contacts_plan")
        if isinstance(tail_same_input.get("ret"), Mapping)
        else None
    )

    same_input_controls = {
        "offset": int(args.offset),
        "baseline_contacts_mean_abs": _mean_abs(baseline_contacts),
        "prepared_input_gaps": {
            "state": _norm_l2(baseline_native["inputs"]["state"], tail_same_input["inputs"]["state"]),
            "cond": _norm_l2(baseline_native["inputs"]["cond"], tail_same_input["inputs"]["cond"]),
            "contacts": _norm_l2(baseline_native["inputs"]["contacts"], tail_same_input["inputs"]["contacts"]),
            "angvel": _norm_l2(baseline_native["inputs"]["angvel"], tail_same_input["inputs"]["angvel"]),
            "pose_history": _norm_l2(
                baseline_native["inputs"]["pose_history"],
                tail_same_input["inputs"]["pose_history"],
            ),
            "rollout_step": _norm_l2(
                baseline_native["inputs"]["rollout_step"],
                tail_same_input["inputs"]["rollout_step"],
            ),
        },
        "ret_gaps": {
            "out_direct": _norm_l2(
                baseline_native["ret"].get("out_direct"),
                tail_same_input["ret"].get("out_direct"),
            ),
            "contacts_plan": _norm_l2(baseline_plan, tail_plan),
            "plan_z_next": _norm_l2(
                baseline_native["ret"].get("plan_z_next"),
                tail_same_input["ret"].get("plan_z_next"),
            ),
        },
    }

    activation_rows: List[Dict[str, Any]] = []
    for module_name in REQUIRED_ACTIVATION_MODULES:
        activation_rows.append(
            _activation_row(
                module_name,
                baseline_native["activations"].get(module_name),
                tail_same_input["activations"].get(module_name),
            )
        )
    control_rows = [
        _activation_row("contacts_plan(control)", baseline_plan, tail_plan),
        _activation_row(
            "plan_z_next(control)",
            baseline_native["ret"].get("plan_z_next"),
            tail_same_input["ret"].get("plan_z_next"),
        ),
    ]
    ranked_activation_rows = _importance_rank(activation_rows)

    counterfactual_specs: List[Dict[str, Any]] = [
        {"label": "plan_input_override", "swap_type": "input_override", "plan_override": baseline_plan},
        {"label": "weight_swap:direct_pose_head", "swap_type": "weight", "weight_swap_modules": ["direct_pose_head"]},
        {"label": "weight_swap:direct_pose_arm_proj", "swap_type": "weight", "weight_swap_modules": ["direct_pose_arm_proj"]},
        {"label": "weight_swap:direct_pose_else_proj", "swap_type": "weight", "weight_swap_modules": ["direct_pose_else_proj"]},
        {"label": "weight_swap:direct_pose_out_leg", "swap_type": "weight", "weight_swap_modules": ["direct_pose_out_leg"]},
        {"label": "weight_swap:direct_pose_out_arm", "swap_type": "weight", "weight_swap_modules": ["direct_pose_out_arm"]},
        {"label": "weight_swap:direct_pose_out_else", "swap_type": "weight", "weight_swap_modules": ["direct_pose_out_else"]},
        {"label": "weight_swap:direct_pose_leg_head", "swap_type": "weight", "weight_swap_modules": ["direct_pose_leg_head"]},
        {
            "label": "weight_swap:all_direct_modules_no_head",
            "swap_type": "weight",
            "weight_swap_modules": [
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_out_leg",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_leg_head",
            ],
        },
        {
            "label": "weight_swap:all_direct_modules_with_head",
            "swap_type": "weight",
            "weight_swap_modules": [
                "direct_pose_head",
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
                "direct_pose_out_leg",
                "direct_pose_out_arm",
                "direct_pose_out_else",
                "direct_pose_leg_head",
            ],
        },
        {"label": "activation:direct_pose_head", "swap_type": "activation", "modules": ["direct_pose_head"]},
        {"label": "activation:direct_pose_arm_proj", "swap_type": "activation", "modules": ["direct_pose_arm_proj"]},
        {"label": "activation:direct_pose_else_proj", "swap_type": "activation", "modules": ["direct_pose_else_proj"]},
        {"label": "activation:direct_pose_out_leg", "swap_type": "activation", "modules": ["direct_pose_out_leg"]},
        {"label": "activation:direct_pose_out_arm", "swap_type": "activation", "modules": ["direct_pose_out_arm"]},
        {"label": "activation:direct_pose_out_else", "swap_type": "activation", "modules": ["direct_pose_out_else"]},
        {"label": "activation:direct_pose_leg_head", "swap_type": "activation", "modules": ["direct_pose_leg_head"]},
        {
            "label": "activation:direct_pose_head+arm_proj+else_proj",
            "swap_type": "activation",
            "modules": ["direct_pose_head", "direct_pose_arm_proj", "direct_pose_else_proj"],
        },
        {
            "label": "activation:out_arm+out_else",
            "swap_type": "activation",
            "modules": ["direct_pose_out_arm", "direct_pose_out_else"],
        },
        {
            "label": "activation:out_leg+leg_head",
            "swap_type": "activation",
            "modules": ["direct_pose_out_leg", "direct_pose_leg_head"],
        },
        {
            "label": "activation:all_direct_readouts",
            "swap_type": "activation",
            "modules": ["direct_pose_out_leg", "direct_pose_out_arm", "direct_pose_out_else"],
        },
        {
            "label": "activation:all_direct_readouts+leg_head",
            "swap_type": "activation",
            "modules": ["direct_pose_out_leg", "direct_pose_out_arm", "direct_pose_out_else", "direct_pose_leg_head"],
        },
    ]

    counterfactual_rows: List[Dict[str, Any]] = []
    counterfactual_details: Dict[str, Any] = {}
    for spec in counterfactual_specs:
        activation_swap = None
        if "modules" in spec:
            activation_swap = {
                str(name): baseline_native["activations"].get(str(name))
                for name in spec["modules"]
                if torch.is_tensor(baseline_native["activations"].get(str(name)))
            }
        result = _run_single_step(
            tail,
            prep_tail,
            fixed_contacts=baseline_contacts,
            plan_override=spec.get("plan_override"),
            activation_swap=activation_swap,
            weight_swap_modules=spec.get("weight_swap_modules"),
            donor_bundle=baseline if spec.get("weight_swap_modules") else None,
        )
        row = _counterfactual_row(
            label=str(spec["label"]),
            swap_type=str(spec["swap_type"]),
            result=result,
            baseline_result=baseline_native,
            tail_result=tail_same_input,
        )
        counterfactual_rows.append(row)
        counterfactual_details[str(spec["label"])] = {
            "result": result,
            "row": row,
        }

    counterfactual_rows = sorted(
        counterfactual_rows,
        key=lambda row: (
            _safe_float(row.get("dir_base_closure_ratio")),
            _safe_float(row.get("out_direct_closure_ratio")),
        ),
        reverse=True,
    )

    weight_subset_rows, weight_subset_map = _run_subset_search(
        swap_type="weight",
        modules=STAGED_DIRECT_MODULES,
        tail_bundle=tail,
        prep_tail=prep_tail,
        baseline_bundle=baseline,
        baseline_result=baseline_native,
        tail_result=tail_same_input,
        fixed_contacts=baseline_contacts,
    )
    activation_subset_rows, activation_subset_map = _run_subset_search(
        swap_type="activation",
        modules=STAGED_DIRECT_MODULES,
        tail_bundle=tail,
        prep_tail=prep_tail,
        baseline_bundle=baseline,
        baseline_result=baseline_native,
        tail_result=tail_same_input,
        fixed_contacts=baseline_contacts,
    )

    leg_first_activation_stage1 = _pick_stage1_subset(
        activation_subset_rows,
        target_key="dir_leg_closure_ratio",
        non_target_key="dir_nonleg_closure_ratio",
    )
    leg_first_activation_stage2 = _pick_stage2_subset(
        activation_subset_rows,
        stage1_modules=leg_first_activation_stage1["modules"],
    )
    nonleg_first_activation_stage1 = _pick_stage1_subset(
        activation_subset_rows,
        target_key="dir_nonleg_closure_ratio",
        non_target_key="dir_leg_closure_ratio",
    )
    nonleg_first_activation_stage2 = _pick_stage2_subset(
        activation_subset_rows,
        stage1_modules=nonleg_first_activation_stage1["modules"],
    )

    head_only_weight = _lookup_subset(weight_subset_map, ["direct_pose_head"])
    head_plus_leg_weight = _lookup_subset(
        weight_subset_map,
        ["direct_pose_head", "direct_pose_out_leg", "direct_pose_leg_head"],
    )
    head_plus_nonleg_weight = _lookup_subset(
        weight_subset_map,
        [
            "direct_pose_head",
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_out_arm",
            "direct_pose_out_else",
        ],
    )
    head_plus_all_readouts_weight = _lookup_subset(
        weight_subset_map,
        [
            "direct_pose_head",
            "direct_pose_out_leg",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_leg_head",
        ],
    )
    all_direct_no_head_weight = _lookup_subset(
        weight_subset_map,
        [
            "direct_pose_arm_proj",
            "direct_pose_else_proj",
            "direct_pose_out_leg",
            "direct_pose_out_arm",
            "direct_pose_out_else",
            "direct_pose_leg_head",
        ],
    )
    all_direct_with_head_weight = _lookup_subset(weight_subset_map, STAGED_DIRECT_MODULES)
    leg_contract_stage1_weight = _lookup_subset(
        weight_subset_map,
        [
            "direct_pose_head",
            "direct_pose_arm_proj",
            "direct_pose_out_leg",
            "direct_pose_out_arm",
            "direct_pose_leg_head",
        ],
    )
    nonleg_readout_stage1_activation = _lookup_subset(
        activation_subset_map,
        ["direct_pose_out_arm", "direct_pose_out_else"],
    )
    leg_readout_stage1_activation = _lookup_subset(
        activation_subset_map,
        ["direct_pose_out_leg", "direct_pose_leg_head"],
    )
    all_readouts_activation = _lookup_subset(
        activation_subset_map,
        ["direct_pose_out_leg", "direct_pose_out_arm", "direct_pose_out_else", "direct_pose_leg_head"],
    )

    stage_candidates_table = [
        _subset_candidate_row(
            leg_readout_stage1_activation,
            interpretation="Clean leg-only readout intervention; closes the dominant leg slice with zero nonleg touch.",
        ),
        _subset_candidate_row(
            nonleg_readout_stage1_activation,
            interpretation="Clean nonleg-only readout intervention; closes the residual nonleg slice without touching leg.",
        ),
        _subset_candidate_row(
            all_readouts_activation,
            interpretation="Minimal pure-readout near-sufficient intervention; proves downstream readouts alone can close step-0 semantics at activation level.",
        ),
        _subset_candidate_row(
            head_only_weight,
            interpretation="Earliest boundary but anti-sufficient alone; swapping only the head contract amplifies the mismatch.",
        ),
        _subset_candidate_row(
            head_plus_leg_weight,
            interpretation="Head-anchored leg path closes leg almost completely, but leaves nonleg contract badly broken.",
        ),
        _subset_candidate_row(
            leg_contract_stage1_weight,
            interpretation="Best proper head-anchored leg-biased weight subset: preserves leg closure while reducing, but not eliminating, nonleg damage.",
        ),
        _subset_candidate_row(
            head_plus_nonleg_weight,
            interpretation="Explicit nonleg contract block: closes nonleg almost fully, but leaves leg path mismatched.",
        ),
        _subset_candidate_row(
            head_plus_all_readouts_weight,
            interpretation="Readouts without arm/else adapters still fail in weight space; adapter mismatch remains upstream of readouts.",
        ),
        _subset_candidate_row(
            all_direct_no_head_weight,
            interpretation="Joint downstream modules without the head anchor fail strongly; no-head is not a viable sufficiency path.",
        ),
        _subset_candidate_row(
            all_direct_with_head_weight,
            interpretation="Only weight-level near-sufficient set: the current minimal full direct-branch contract swap.",
        ),
    ]

    leg_first_best_path = {
        "semantic_readout_path": _path_summary(leg_first_activation_stage1, leg_first_activation_stage2),
        "semantic_readout_reason": (
            "Stage1 selects `direct_pose_out_leg + direct_pose_leg_head` because it reaches max leg closure (=1.0), keeps nonleg closure at 0.0, "
            "and uses the smallest 2-module leg block. Stage2 adds only `direct_pose_out_arm + direct_pose_out_else` to reach perfect total closure."
        ),
        "contract_weight_followup": {
            "stage1_set": list(leg_contract_stage1_weight["modules"]),
            "stage1_closure": {
                "out_direct": float(_safe_float(leg_contract_stage1_weight.get("out_direct_closure_ratio"))),
                "dir_base": float(_safe_float(leg_contract_stage1_weight.get("dir_base_closure_ratio"))),
                "dir_leg": float(_safe_float(leg_contract_stage1_weight.get("dir_leg_closure_ratio"))),
                "dir_nonleg": float(_safe_float(leg_contract_stage1_weight.get("dir_nonleg_closure_ratio"))),
            },
            "stage2_add_modules": ["direct_pose_else_proj", "direct_pose_out_else"],
            "stage2_set": list(all_direct_with_head_weight["modules"]),
            "final_closure": {
                "out_direct": float(_safe_float(all_direct_with_head_weight.get("out_direct_closure_ratio"))),
                "dir_base": float(_safe_float(all_direct_with_head_weight.get("dir_base_closure_ratio"))),
                "dir_leg": float(_safe_float(all_direct_with_head_weight.get("dir_leg_closure_ratio"))),
                "dir_nonleg": float(_safe_float(all_direct_with_head_weight.get("dir_nonleg_closure_ratio"))),
            },
            "note": "Weight-space leg-first remains collateral-heavy until the full 7-module contract is restored.",
        },
    }

    nonleg_first_best_path = {
        "semantic_readout_path": _path_summary(nonleg_first_activation_stage1, nonleg_first_activation_stage2),
        "semantic_readout_reason": (
            "Stage1 selects `direct_pose_out_arm + direct_pose_out_else` because it reaches max nonleg closure (=1.0), keeps leg closure at 0.0, "
            "and is the smallest 2-module nonleg readout block. Stage2 adds only `direct_pose_out_leg + direct_pose_leg_head` to reach perfect total closure."
        ),
        "contract_weight_followup": {
            "stage1_set": list(head_plus_nonleg_weight["modules"]),
            "stage1_closure": {
                "out_direct": float(_safe_float(head_plus_nonleg_weight.get("out_direct_closure_ratio"))),
                "dir_base": float(_safe_float(head_plus_nonleg_weight.get("dir_base_closure_ratio"))),
                "dir_leg": float(_safe_float(head_plus_nonleg_weight.get("dir_leg_closure_ratio"))),
                "dir_nonleg": float(_safe_float(head_plus_nonleg_weight.get("dir_nonleg_closure_ratio"))),
            },
            "stage2_add_modules": ["direct_pose_out_leg", "direct_pose_leg_head"],
            "stage2_set": list(all_direct_with_head_weight["modules"]),
            "final_closure": {
                "out_direct": float(_safe_float(all_direct_with_head_weight.get("out_direct_closure_ratio"))),
                "dir_base": float(_safe_float(all_direct_with_head_weight.get("dir_base_closure_ratio"))),
                "dir_leg": float(_safe_float(all_direct_with_head_weight.get("dir_leg_closure_ratio"))),
                "dir_nonleg": float(_safe_float(all_direct_with_head_weight.get("dir_nonleg_closure_ratio"))),
            },
            "note": "Weight-space nonleg-first identifies the arm/else adapter path cleanly, but still needs the leg block to close total gap.",
        },
    }

    head_anchor_analysis = {
        "head_only": _subset_candidate_row(
            head_only_weight,
            interpretation="Necessary anchor candidate, but not sufficient: isolated head swap makes every closure worse.",
        ),
        "head_plus_leg_path": _subset_candidate_row(
            head_plus_leg_weight,
            interpretation="Head+leg restores the dominant leg term, but leaves nonleg wildly mismatched.",
        ),
        "head_plus_nonleg_path": _subset_candidate_row(
            head_plus_nonleg_weight,
            interpretation="Head+nonleg path restores nonleg almost completely, but leaves the leg-dominant readout unresolved.",
        ),
        "head_plus_all_readouts": _subset_candidate_row(
            head_plus_all_readouts_weight,
            interpretation="Even all readouts plus head are not enough in weight space; missing arm/else adapters keep the contract broken.",
        ),
        "head_plus_all_direct_modules": _subset_candidate_row(
            all_direct_with_head_weight,
            interpretation="First genuine high-closure regime in weight space; full direct-branch contract restored.",
        ),
    }

    interaction_synergy = {
        "head_vs_readouts_weight": {
            "head_only_dir_base_closure": float(_safe_float(head_only_weight.get("dir_base_closure_ratio"))),
            "all_direct_no_head_dir_base_closure": float(_safe_float(all_direct_no_head_weight.get("dir_base_closure_ratio"))),
            "all_direct_with_head_dir_base_closure": float(_safe_float(all_direct_with_head_weight.get("dir_base_closure_ratio"))),
            "joint_gain_over_best_component": float(
                _safe_float(all_direct_with_head_weight.get("dir_base_closure_ratio"))
                - max(
                    _safe_float(head_only_weight.get("dir_base_closure_ratio")),
                    _safe_float(all_direct_no_head_weight.get("dir_base_closure_ratio")),
                )
            ),
            "judgement": "Strong positive synergy: head is a necessary anchor for any weight-level high-closure regime.",
        },
        "leg_block": {
            "activation_subset": list(leg_readout_stage1_activation["modules"]),
            "activation_dir_leg_closure": float(_safe_float(leg_readout_stage1_activation.get("dir_leg_closure_ratio"))),
            "activation_dir_base_closure": float(_safe_float(leg_readout_stage1_activation.get("dir_base_closure_ratio"))),
            "weight_subset": list(head_plus_leg_weight["modules"]),
            "weight_dir_leg_closure": float(_safe_float(head_plus_leg_weight.get("dir_leg_closure_ratio"))),
            "weight_dir_nonleg_closure": float(_safe_float(head_plus_leg_weight.get("dir_nonleg_closure_ratio"))),
            "judgement": "Interpretable leg-stage block exists at readout level; it is not a standalone contract-sufficient block in weight space.",
        },
        "nonleg_block": {
            "activation_subset": list(nonleg_readout_stage1_activation["modules"]),
            "activation_dir_nonleg_closure": float(_safe_float(nonleg_readout_stage1_activation.get("dir_nonleg_closure_ratio"))),
            "activation_dir_leg_closure": float(_safe_float(nonleg_readout_stage1_activation.get("dir_leg_closure_ratio"))),
            "weight_subset": list(head_plus_nonleg_weight["modules"]),
            "weight_dir_nonleg_closure": float(_safe_float(head_plus_nonleg_weight.get("dir_nonleg_closure_ratio"))),
            "weight_dir_leg_closure": float(_safe_float(head_plus_nonleg_weight.get("dir_leg_closure_ratio"))),
            "judgement": "Nonleg readout block is clean at activation level; arm/else adapters become necessary only when restoring weight-space contract.",
        },
        "all_direct_no_head_failure": {
            "subset": list(all_direct_no_head_weight["modules"]),
            "out_direct_closure_ratio": float(_safe_float(all_direct_no_head_weight.get("out_direct_closure_ratio"))),
            "dir_base_closure_ratio": float(_safe_float(all_direct_no_head_weight.get("dir_base_closure_ratio"))),
            "judgement": "Removing head leaves downstream modules attached to a tail-specific hidden contract, so swapping only non-head modules cannot enter a high-closure regime.",
        },
    }

    staged_final_judgement = [
        _support_row(
            hypothesis="The most explanatory staged decomposition is hybrid: readout-first at activation level, head-anchor + branch/readout closure at weight level.",
            support_level="strongly supported",
            strongest_evidence=(
                f"`activation_subset:direct_pose_out_leg+direct_pose_leg_head` gives dir-leg closure="
                f"{_safe_float(leg_readout_stage1_activation.get('dir_leg_closure_ratio')):.3f} with dir-nonleg closure="
                f"{_safe_float(leg_readout_stage1_activation.get('dir_nonleg_closure_ratio')):.3f}; "
                f"but weight high-closure appears only at full 7-module set with dir-base closure="
                f"{_safe_float(all_direct_with_head_weight.get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="A single swap grammar cannot express both semantic readout closure and contract-restoring sufficiency cleanly.",
            next_best="Use readout-level staging for explanation, then verify sufficiency with the full head-anchored weight set.",
        ),
        _support_row(
            hypothesis="`direct_pose_head` is a necessary anchor and earliest source boundary, but not a standalone sufficient explanation.",
            support_level="strongly supported",
            strongest_evidence=(
                f"`weight_subset:direct_pose_head` dir-base closure={_safe_float(head_only_weight.get('dir_base_closure_ratio')):.3f}, "
                f"whereas `weight_subset:all_direct_modules_with_head` dir-base closure={_safe_float(all_direct_with_head_weight.get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="Activation-level perfect closure is possible without touching head once downstream readouts are overridden directly.",
            next_best="Treat head as the anchor for weight-space interventions, not as a one-module root-cause patch.",
        ),
        _support_row(
            hypothesis="There is a strong `leg-only` explanation only at readout level, not at contract level.",
            support_level="partially supported",
            strongest_evidence=(
                f"`activation_subset:direct_pose_out_leg+direct_pose_leg_head` gives dir-base closure="
                f"{_safe_float(leg_readout_stage1_activation.get('dir_base_closure_ratio')):.3f} and dir-leg closure="
                f"{_safe_float(leg_readout_stage1_activation.get('dir_leg_closure_ratio')):.3f}; "
                f"`weight_subset:direct_pose_head+direct_pose_out_leg+direct_pose_leg_head` still has dir-nonleg closure="
                f"{_safe_float(head_plus_leg_weight.get('dir_nonleg_closure_ratio')):.3f}"
            ),
            weakest_point="Readout closure alone bypasses the earlier head-boundary contract break, so it does not localize the earliest source.",
            next_best="Use the leg block as the first semantic/readout stage, then add head-anchor contract analysis when moving to sufficiency.",
        ),
        _support_row(
            hypothesis="There is not a strong `nonleg-only` explanation for the earliest split; the nonleg path is a residual branch/readout slice plus adapter amplification.",
            support_level="partially supported",
            strongest_evidence=(
                f"`activation_subset:direct_pose_out_arm+direct_pose_out_else` gives dir-nonleg closure="
                f"{_safe_float(nonleg_readout_stage1_activation.get('dir_nonleg_closure_ratio')):.3f} with dir-leg closure="
                f"{_safe_float(nonleg_readout_stage1_activation.get('dir_leg_closure_ratio')):.3f}; "
                f"but `weight_subset:direct_pose_head+direct_pose_arm_proj+direct_pose_else_proj+direct_pose_out_arm+direct_pose_out_else` leaves dir-leg closure="
                f"{_safe_float(head_plus_nonleg_weight.get('dir_leg_closure_ratio')):.3f}"
            ),
            weakest_point="Because nonleg raw gap is only ~6.34% of dir-base, even perfect nonleg closure does not explain the main split on its own.",
            next_best="Keep nonleg-first as a residual branch diagnosis, not as the primary first-stage story.",
        ),
        _support_row(
            hypothesis="`direct_pose_arm_proj` / `direct_pose_else_proj` behave more like branch adapters / amplifiers than pure downstream readouts.",
            support_level="strongly supported",
            strongest_evidence=(
                f"Pure nonleg readout activation swap needs only `direct_pose_out_arm+direct_pose_out_else`, "
                f"while weight-space `head+all_readouts` still has dir-base closure={_safe_float(head_plus_all_readouts_weight.get('dir_base_closure_ratio')):.3f} "
                f"until arm/else adapters are added."
            ),
            weakest_point="They still sit on the earliest head→branch contract boundary, so they are not merely late observers.",
            next_best="Model them as adapter/amplifier pieces that must be paired with head in any contract-restoring intervention.",
        ),
    ]

    final_judgement = [
        _support_row(
            hypothesis="Earliest large same-input split sits at direct_pose_head boundary, not at contacts/meas preparation.",
            support_level="strongly supported",
            strongest_evidence=(
                f"same-input fixed-contacts `direct_pose_head` delta_l2="
                f"{_safe_float(next(row['delta_l2'] for row in ranked_activation_rows if row['module']=='direct_pose_head')):.6f}; "
                f"`plan_input_override` dir-base closure="
                f"{_safe_float((counterfactual_details['plan_input_override']['row'] or {}).get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="contacts_plan itself still diverges, so head input-side contribution is non-zero and must be controlled explicitly.",
            next_best="Minimal intervention: contract-preserving direct-branch swap starting from direct_pose_head plus matched downstream readouts.",
        ),
        _support_row(
            hypothesis="direct_pose_head alone is necessary and near-sufficient for most first-step raw-loss split.",
            support_level="not supported",
            strongest_evidence=(
                f"`activation:direct_pose_head` dir-base closure="
                f"{_safe_float((counterfactual_details['activation:direct_pose_head']['row'] or {}).get('dir_base_closure_ratio')):.3f}; "
                f"`weight_swap:direct_pose_head` dir-base closure="
                f"{_safe_float((counterfactual_details['weight_swap:direct_pose_head']['row'] or {}).get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="Both head-only activation swap and head-only weight swap increase the gap, so head is not a standalone sufficient intervention.",
            next_best="Do not touch head alone; pair it with matched downstream direct readouts.",
        ),
        _support_row(
            hypothesis="arm/else proj are branch adapters / downstream amplifiers, not standalone primary sources.",
            support_level="partially supported",
            strongest_evidence=(
                f"low-cosine activations appear at `direct_pose_arm_proj`/`direct_pose_else_proj`, but their single-module "
                f"activation closures are (`arm_proj`={_safe_float((counterfactual_details['activation:direct_pose_arm_proj']['row'] or {}).get('dir_base_closure_ratio')):.3f}, "
                f"`else_proj`={_safe_float((counterfactual_details['activation:direct_pose_else_proj']['row'] or {}).get('dir_base_closure_ratio')):.3f}); "
                f"weight-only closures are (`arm_proj`={_safe_float((counterfactual_details['weight_swap:direct_pose_arm_proj']['row'] or {}).get('dir_base_closure_ratio')):.3f}, "
                f"`else_proj`={_safe_float((counterfactual_details['weight_swap:direct_pose_else_proj']['row'] or {}).get('dir_base_closure_ratio')):.3f})"
            ),
            weakest_point="They sit exactly on the head→nonleg branch boundary, so they are still part of the earliest contract break rather than purely late readouts.",
            next_best="If head-side intervention leaves a non-leg residual, touch arm/else proj together with the corresponding readouts.",
        ),
        _support_row(
            hypothesis="leg readout / leg head are secondary readout paths, not the primary source.",
            support_level="partially supported",
            strongest_evidence=(
                f"`direct_pose_leg_head` activation delta_l2="
                f"{_safe_float(next(row['delta_l2'] for row in ranked_activation_rows if row['module']=='direct_pose_leg_head')):.6f}; "
                f"`activation:out_leg+leg_head` dir-base closure="
                f"{_safe_float((counterfactual_details['activation:out_leg+leg_head']['row'] or {}).get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="`step0` raw loss gap is 93.7% leg-dominated, so leg readouts are the dominant readout locus even if they are not the earliest semantic source.",
            next_best="Use leg modules as the first downstream follow-up after fixing the head-side contract.",
        ),
        _support_row(
            hypothesis="The first-step split is best explained by a whole direct-branch contract mismatch, not any single module alone.",
            support_level="strongly supported",
            strongest_evidence=(
                f"`weight_swap:all_direct_modules_no_head` dir-base closure="
                f"{_safe_float((counterfactual_details['weight_swap:all_direct_modules_no_head']['row'] or {}).get('dir_base_closure_ratio')):.3f}, "
                f"but `weight_swap:all_direct_modules_with_head` dir-base closure="
                f"{_safe_float((counterfactual_details['weight_swap:all_direct_modules_with_head']['row'] or {}).get('dir_base_closure_ratio')):.3f}"
            ),
            weakest_point="This still localizes only inside the direct branch; it does not separate which weights first deviated during training history.",
            next_best="Treat `direct_pose_head + arm/else proj + leg/arm/else readouts + leg_head` as the current minimal sufficient module set.",
        ),
        _support_row(
            hypothesis="contacts_in_t can now be downgraded to a side factor for this first-step split.",
            support_level="strongly supported",
            strongest_evidence=(
                "Inherited P1 result: contact-swap changes dir_base by only ~4.9e-06 vs native gap ~1.835e-03; "
                "current same-input module swaps operate with fixed baseline contacts and still show large closures."
            ),
            weakest_point="This judgement is specific to first-step semantic split, not a blanket claim about all long-horizon contact effects.",
            next_best="Keep contacts fixed while debugging head/readout modules; do not reopen contact path as mainline until head interventions fail.",
        ),
    ]

    payload = {
        "analysis": "cp015_tailk7_same_input_module_attribution",
        "operating_point": {
            "offset": int(args.offset),
            "fixed_contacts_source": "baseline native same-entry contacts_in_t",
            "same_input_contract": "baseline and tail share state/cond/contacts/angvel/pose_history/time_index/rollout_step at first forward",
            "baseline_case": {
                "config": str(baseline_config),
                "ckpt": str(baseline_ckpt),
                "eval": str(baseline_eval),
            },
            "tail_case": {
                "config": str(tail_config),
                "ckpt": str(tail_ckpt),
                "eval": str(tail_eval),
            },
            "prep_base": {
                "steps": int(prep_base["steps"]),
                "cycle_len": int(prep_base["cycle_len"]),
                "total_steps": int(prep_base["total_steps"]),
                "include_boundary": bool(prep_base["include_boundary"]),
                "boundary_weighted_sum": float(prep_base["boundary_weighted_sum"]),
            },
        },
        "code_facts": {
            "direct_pose_feat_source": str(getattr(baseline["case"]["trainer"].model, "direct_pose_feat_source", "unknown")),
            "direct_pose_meas_mode": str(getattr(baseline["case"]["trainer"].model, "direct_pose_meas_mode", "unknown")),
            "contact_plan_inject": str(getattr(baseline["case"]["trainer"].model, "contact_plan_inject", "unknown")),
            "direct_path_note": "Current ckpts use direct_pose_feat_source='cond', so out_direct does not consume shared_encoder/h_final; first-step direct split must come from contact_plan/meas/head/readout side.",
        },
        "module_graph": _module_graph_rows(baseline),
        "same_input_controls": same_input_controls,
        "baseline_step0_metrics": dict(baseline_native["metrics"]),
        "tail_step0_metrics": dict(tail_same_input["metrics"]),
        "activation_divergence_table": ranked_activation_rows,
        "activation_control_rows": control_rows,
        "counterfactual_table": counterfactual_rows,
        "final_judgement_table": final_judgement,
        "staged_search": {
            "modules": list(STAGED_DIRECT_MODULES),
            "near_sufficient_threshold": float(NEAR_SUFFICIENT_THRESHOLD),
            "weight_subset_table": weight_subset_rows,
            "activation_subset_table": activation_subset_rows,
        },
        "stage_candidates_table": stage_candidates_table,
        "leg_first_best_path": leg_first_best_path,
        "nonleg_first_best_path": nonleg_first_best_path,
        "head_anchor_analysis": head_anchor_analysis,
        "interaction_synergy": interaction_synergy,
        "staged_final_judgement_table": staged_final_judgement,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
