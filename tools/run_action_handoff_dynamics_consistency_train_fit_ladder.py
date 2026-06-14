#!/usr/bin/env python3
"""Dynamics-consistency train-fit ladder for action-handoff inbetweening.

Debug-only tool. It trains a tiny deterministic decoder under fixed oracle
support/command schedule and evaluates the output through the same reconstructed
state281 acceptance path used by the GT guard. It does not train production
Trainer/runtime/gate, does not mutate checkpoints, and does not attach a
production generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    FPS,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_SLICE,
    POSE_TOPK,
    RAW_COND_DIR_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)
from train.data.normalizers import VectorTanhNormalizerTorch  # noqa: E402
from train.geometry import compose_delta_raw_to_next  # noqa: E402
from tools.run_action_handoff_inbetween_b1_cond_baseline_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_PRETRAIN_TEMPLATE,
    _make_runner_args,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _fmt,
    _heading_error_rad,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
    _step_angvel_rms,
    _step_l2,
    _step_pose_l2,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    DecoderItem,
    Standardizer,
    TinyDeterministicDecoder,
    _apply_oracle_contact_passthrough,
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _dataset_arrays,
    _evaluate_split_predictions,
    _fit_standardizer,
    _fk_positions_from_rot6d_no_inplace,
    _guard_path_identity,
    _integrate_root_pos_torch,
    _loss_metrics,
    _predict_raw,
    _reshape_state_aux,
    _seq_from_prediction,
    _stack_seq,
    _summarize_rows,
    _world_root_vel_from_ego_torch,
)
from tools.run_action_handoff_adjusted_acceptance_guard import (  # noqa: E402
    DEFAULT_BONE_BRIDGE as DEFAULT_GUARD_BONE_BRIDGE,
    DEFAULT_COMMAND_DEMOTION_ROWS as DEFAULT_GUARD_COMMAND_DEMOTION_ROWS,
    DEFAULT_REGIME_BRIDGE as DEFAULT_GUARD_REGIME_BRIDGE,
    DEFAULT_TWO_FRAME as DEFAULT_GUARD_TWO_FRAME,
    run as _run_adjusted_acceptance_guard,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402,F401
    SUPPORT_SIDE_FEATURE_KEYS,
    _make_sequence,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402

import train.validate.run_freerun_cycles as freerun  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_dynamics_consistency_gt_residual_ladder_20260603")
DEFAULT_LOCALIZATION_OUT_DIR = Path("debug_output/_tmp_action_handoff_dynamics_consistency_localization_20260603")
DEFAULT_POSE_SWEEP_OUT_DIR = Path("debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_20260603")
DEFAULT_LOSS_REFACTOR_OUT_DIR = Path("debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_20260604")
EPS = 1e-8
CHANNELS = ("pose", "contact", "rootvel", "bone_angvel")
PAIR_KEYS = (
    ("pose", "contact"),
    ("pose", "rootvel"),
    ("pose", "bone_angvel"),
    ("contact", "rootvel"),
    ("contact", "bone_angvel"),
    ("rootvel", "bone_angvel"),
)
LOCALIZATION_METRICS = (
    ("pose_step_l2_p95", "pose_step_l2", "step", "pose_continuity", "pose_continuity_loss"),
    ("contact_step_l2_p95", "contact_step_l2", "step", "support_honesty", "contact_schedule"),
    ("angvel_step_rms_p95", "angvel_step_rms", "step", "rate_budget", "bone_angvel_rate_loss"),
    ("heading_error_p95_rad", "heading_error_rad", "frame", "command_response", "command_compatibility"),
)


@dataclass
class XNormCalibrator:
    mu_x: torch.Tensor
    std_x: torch.Tensor
    rootvel_scale: torch.Tensor
    angvel_scale: torch.Tensor


@dataclass
class BaseOperator:
    runner: Any
    model: nn.Module
    xcal: XNormCalibrator
    std_y: torch.Tensor
    eval_y_scale: torch.Tensor
    rot_y_slice: slice
    rootvel_y_slice: slice
    columns: Tuple[str, str]
    angvel_norm: Optional[VectorTanhNormalizerTorch]
    pose_hist_norm: Optional[VectorTanhNormalizerTorch]
    pose_hist_len: int
    cond_raw_by_clip: Dict[str, np.ndarray]
    raw_x_norm_max_abs_error: Dict[str, float]


def _jsonify(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonify(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _load_cond_raw_by_clip(npz_root: Path, clips: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for clip in clips:
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as z:
            out[str(clip)] = np.asarray(z["cond_in"], dtype=np.float32).copy()
    return out


def _robust_cond_norm(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    q1 = np.percentile(arr, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    safe = np.where((arr >= lo) & (arr <= hi), arr, np.nan)
    mu = np.nanmean(safe, axis=1, keepdims=True)
    std = np.nanstd(safe, axis=1, keepdims=True)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None).astype(np.float32)
    norm = (arr - mu) / std
    np.nan_to_num(norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(norm, -6.0, 6.0, out=norm)
    return norm.astype(np.float32, copy=False)


def _stack_cond_raw(base: BaseOperator, items: Sequence[DecoderItem], idxs: Sequence[int], horizon: int) -> np.ndarray:
    rows = []
    for item_idx in idxs:
        item = items[int(item_idx)]
        cond = base.cond_raw_by_clip[item.clip]
        s = int(item.start)
        rows.append(cond[s : s + int(horizon)])
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _fit_scale_from_x_norm(
    *,
    raw: np.ndarray,
    x_norm: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    comp = np.asarray(x_norm, dtype=np.float64) * np.asarray(std, dtype=np.float64).reshape(1, -1)
    comp = comp + np.asarray(mu, dtype=np.float64).reshape(1, -1)
    inv = np.arctanh(np.clip(comp, -1.0 + 1e-6, 1.0 - 1e-6))
    raw64 = np.asarray(raw, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = raw64 / inv
    scale = np.where(np.isfinite(scale) & (np.abs(inv) > 1e-8) & (np.abs(raw64) > 1e-8), scale, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(scale, axis=0)
    fb = np.asarray(fallback, dtype=np.float64).reshape(-1)
    med = np.where(np.isfinite(med) & (med > 1e-8), med, fb)
    return np.clip(med.astype(np.float32), 1e-6, None)


def _build_xnorm_calibrator(npz_root: Path, runner: Any, clips: Sequence[str], device: torch.device) -> Tuple[XNormCalibrator, Dict[str, float]]:
    normalizer = runner.normalizer
    mu_x = np.asarray(normalizer.mu_x, dtype=np.float32).reshape(-1)
    std_x = np.asarray(normalizer.std_x, dtype=np.float32).reshape(-1)
    root_sl = runner.trainer.rootvel_x_slice
    ang_sl = runner.trainer.angvel_x_slice
    if not isinstance(root_sl, slice) or not isinstance(ang_sl, slice):
        raise RuntimeError("base operator missing rootvel/angvel X slices")

    raw_root = []
    norm_root = []
    raw_ang = []
    norm_ang = []
    max_errors: Dict[str, float] = {}
    for clip in clips:
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as z:
            x_raw = np.asarray(z["x_in_features"], dtype=np.float32)
            x_norm = np.asarray(z["X_norm"], dtype=np.float32)
        n = min(int(x_raw.shape[0]), int(x_norm.shape[0]))
        x_raw = x_raw[:n]
        x_norm = x_norm[:n]
        raw_root.append(x_raw[:, root_sl])
        norm_root.append(x_norm[:, root_sl])
        raw_ang.append(x_raw[:, ang_sl])
        norm_ang.append(x_norm[:, ang_sl])

    root_fb = np.asarray(normalizer.tanh_scales_rootvel, dtype=np.float32).reshape(-1)
    ang_fb = np.asarray(normalizer.tanh_scales_angvel, dtype=np.float32).reshape(-1)
    root_scale = _fit_scale_from_x_norm(
        raw=np.concatenate(raw_root, axis=0),
        x_norm=np.concatenate(norm_root, axis=0),
        mu=mu_x[root_sl],
        std=std_x[root_sl],
        fallback=root_fb,
    )
    ang_scale = _fit_scale_from_x_norm(
        raw=np.concatenate(raw_ang, axis=0),
        x_norm=np.concatenate(norm_ang, axis=0),
        mu=mu_x[ang_sl],
        std=std_x[ang_sl],
        fallback=ang_fb,
    )

    xcal = XNormCalibrator(
        mu_x=torch.as_tensor(mu_x, dtype=torch.float32, device=device),
        std_x=torch.as_tensor(std_x, dtype=torch.float32, device=device).clamp_min(1e-6),
        rootvel_scale=torch.as_tensor(root_scale, dtype=torch.float32, device=device).clamp_min(1e-6),
        angvel_scale=torch.as_tensor(ang_scale, dtype=torch.float32, device=device).clamp_min(1e-6),
    )

    for clip in clips:
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as z:
            x_raw = torch.as_tensor(np.asarray(z["x_in_features"], dtype=np.float32), device=device)
            x_norm_np = np.asarray(z["X_norm"], dtype=np.float32)
        pred = _physical_x_to_model_x_norm(
            x_raw=x_raw,
            xcal=xcal,
            rootvel_x_slice=root_sl,
            angvel_x_slice=ang_sl,
        )
        n = min(int(pred.shape[0]), int(x_norm_np.shape[0]))
        err = pred[:n].detach().cpu().numpy() - x_norm_np[:n]
        max_errors[str(clip)] = float(np.max(np.abs(err))) if err.size else 0.0
    return xcal, max_errors


def _physical_x_to_model_x_norm(
    *,
    x_raw: torch.Tensor,
    xcal: XNormCalibrator,
    rootvel_x_slice: slice,
    angvel_x_slice: slice,
) -> torch.Tensor:
    x = x_raw.clone()
    x[..., rootvel_x_slice] = torch.tanh(x[..., rootvel_x_slice] / xcal.rootvel_scale)
    x[..., angvel_x_slice] = torch.tanh(x[..., angvel_x_slice] / xcal.angvel_scale)
    return (x - xcal.mu_x) / xcal.std_x


def _make_torch_vec_norm(obj: Any, device: torch.device) -> Optional[VectorTanhNormalizerTorch]:
    if obj is None or getattr(obj, "scales", None) is None:
        return None
    scales = torch.as_tensor(obj.scales, dtype=torch.float32, device=device)
    mu = None if getattr(obj, "mu", None) is None else torch.as_tensor(obj.mu, dtype=torch.float32, device=device)
    std = None if getattr(obj, "std", None) is None else torch.as_tensor(obj.std, dtype=torch.float32, device=device)
    return VectorTanhNormalizerTorch(scales, mu=mu, std=std).to(device)


def _build_base_operator(args: argparse.Namespace, npz_root: Path, device: torch.device) -> BaseOperator:
    runner_args = argparse.Namespace(
        checkpoint=str(args.checkpoint),
        bundle=str(args.bundle),
        pretrain_template=str(args.pretrain_template),
        encoder_bundle=str(args.encoder_bundle),
        device=str(device),
        context_len=int(args.context_len),
    )
    runner = freerun.FreeRunCycleRunner(_make_runner_args(runner_args))
    ds = runner._build_dataset(npz_root / "Walk_F.npz", seq_len=max(64, int(args.horizon)))
    runner._ensure_model_ready(ds)
    model = runner.model
    if model is None:
        raise RuntimeError("base EventMotionModel failed to initialize")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    xcal, raw_x_norm_error = _build_xnorm_calibrator(
        npz_root,
        runner,
        clips=tuple(MATCHED_TARGETS) + ("Walk_F",),
        device=device,
    )
    loss_fn = runner.loss_fn
    columns = tuple(getattr(loss_fn, "_rot6d_columns", ("X", "Z")) or ("X", "Z"))
    if len(columns) < 2:
        columns = ("X", "Z")
    std_y = torch.as_tensor(runner.normalizer.std_y, dtype=torch.float32, device=device).clamp_min(1e-6)
    return BaseOperator(
        runner=runner,
        model=model,
        xcal=xcal,
        std_y=std_y,
        eval_y_scale=std_y.clamp_min(float(args.dynamics_eval_scale_floor)),
        rot_y_slice=runner.trainer.rot6d_y_slice,
        rootvel_y_slice=runner.trainer.rootvel_slice,
        columns=(str(columns[0]), str(columns[1])),
        angvel_norm=_make_torch_vec_norm(getattr(ds, "angvel_norm", None), device),
        pose_hist_norm=_make_torch_vec_norm(getattr(ds, "pose_hist_norm", None), device),
        pose_hist_len=int(getattr(ds, "pose_hist_len", 0) or 0),
        cond_raw_by_clip=_load_cond_raw_by_clip(npz_root, tuple(MATCHED_TARGETS) + ("Walk_F",)),
        raw_x_norm_max_abs_error=raw_x_norm_error,
    )


def _checkpoint_overlap_report(base: BaseOperator) -> Dict[str, Any]:
    ckpt_sd = getattr(base.runner, "state_dict", {}) or {}
    model_sd = base.model.state_dict()
    ckpt_tensors = {str(k): v for k, v in ckpt_sd.items() if torch.is_tensor(v)}
    model_tensors = {str(k): v for k, v in model_sd.items() if torch.is_tensor(v)}
    matched = []
    shape_mismatch = []
    for key, val in model_tensors.items():
        src = ckpt_tensors.get(key)
        if src is None:
            continue
        if tuple(src.shape) == tuple(val.shape):
            matched.append(key)
        else:
            shape_mismatch.append(key)
    model_numel = int(sum(int(v.numel()) for v in model_tensors.values()))
    matched_numel = int(sum(int(model_tensors[k].numel()) for k in matched))
    missing_model = sorted(set(model_tensors.keys()) - set(ckpt_tensors.keys()))
    unexpected_ckpt = sorted(set(ckpt_tensors.keys()) - set(model_tensors.keys()))
    return {
        "note": "raw checkpoint/model exact-name overlap after FreeRun schema normalization path; not a strict load contract",
        "checkpoint_tensor_count": int(len(ckpt_tensors)),
        "model_tensor_count": int(len(model_tensors)),
        "exact_name_shape_match_count": int(len(matched)),
        "exact_name_shape_match_model_numel": matched_numel,
        "model_numel": model_numel,
        "exact_name_shape_match_model_numel_ratio": float(matched_numel / max(model_numel, 1)),
        "shape_mismatch_count": int(len(shape_mismatch)),
        "shape_mismatch_sample": shape_mismatch[:16],
        "model_keys_without_raw_ckpt_name_count": int(len(missing_model)),
        "model_keys_without_raw_ckpt_name_sample": missing_model[:16],
        "raw_ckpt_keys_without_model_name_count": int(len(unexpected_ckpt)),
        "raw_ckpt_keys_without_model_name_sample": unexpected_ckpt[:16],
    }


def _pose_history_from_rot6d(rot6d: torch.Tensor, base: BaseOperator) -> Optional[torch.Tensor]:
    hist_len = int(base.pose_hist_len)
    if hist_len <= 0:
        return None
    b, h, d = rot6d.shape
    rows = []
    for lag in range(hist_len, 0, -1):
        ids = torch.arange(h, device=rot6d.device) - lag
        ids = ids.clamp(min=0)
        rows.append(rot6d.index_select(1, ids))
    raw = torch.cat(rows, dim=-1).reshape(b, h, hist_len * d)
    return base.pose_hist_norm(raw) if base.pose_hist_norm is not None else raw


def _angvel_to_model_feature(aux: torch.Tensor, base: BaseOperator) -> Optional[torch.Tensor]:
    if base.angvel_norm is None:
        return aux
    return base.angvel_norm(aux)


def _state_aux_to_eval_tensors(
    *,
    state: torch.Tensor,
    aux: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, torch.Tensor]:
    state_for_eval = state
    if oracle_contact_passthrough:
        state_for_eval = state.clone()
        state_for_eval[:, :, CONTACT_SLICE] = true_contact
    root_vel = _world_root_vel_from_ego_torch(
        state_for_eval[:, :, EGO_VEL_SLICE],
        true_cond_dir,
        command_align_root_vel=command_align_root_vel,
    )
    root_pos = _integrate_root_pos_torch(root_vel, true_root_pos[:, 0])
    return {
        "state": state_for_eval,
        "rot6d": state_for_eval[:, :, POSE_SLICE],
        "contact": state_for_eval[:, :, CONTACT_SLICE],
        "ego": state_for_eval[:, :, EGO_VEL_SLICE],
        "yaw": state_for_eval[:, :, YAW_RATE_SLICE],
        "root_vel": root_vel,
        "root_pos": root_pos,
        "aux": aux,
    }


def _base_operator_next_y(
    *,
    base: BaseOperator,
    pred: Mapping[str, torch.Tensor],
    cond_norm: torch.Tensor,
    true_contact: torch.Tensor,
) -> torch.Tensor:
    rot = pred["rot6d"][:, :-1]
    root_pos = pred["root_pos"][:, :-1]
    root_vel = pred["root_vel"][:, :-1]
    aux = pred["aux"][:, :-1]
    x_raw = torch.cat([root_pos, root_vel, rot, aux], dim=-1)
    x_norm = _physical_x_to_model_x_norm(
        x_raw=x_raw,
        xcal=base.xcal,
        rootvel_x_slice=base.runner.trainer.rootvel_x_slice,
        angvel_x_slice=base.runner.trainer.angvel_x_slice,
    )
    angvel = _angvel_to_model_feature(aux, base)
    pose_hist = _pose_history_from_rot6d(pred["rot6d"], base)
    if pose_hist is not None:
        pose_hist = pose_hist[:, :-1]
    b, t, _ = x_norm.shape
    time_grid = torch.arange(t, device=x_norm.device, dtype=x_norm.dtype).view(1, t).expand(b, t)
    ret = base.model(
        x_norm,
        cond_norm[:, :-1],
        contacts=true_contact[:, :-1],
        angvel=angvel,
        pose_history=pose_hist,
        time_index=time_grid,
        rollout_step=time_grid,
    )
    if not isinstance(ret, dict) or not torch.is_tensor(ret.get("out")):
        raise RuntimeError("base EventMotionModel.forward did not return tensor ret['out']")
    delta_norm = ret["out"]
    if delta_norm.dim() == 2:
        delta_norm = delta_norm.unsqueeze(1)
    delta_raw = delta_norm * base.std_y.view(1, 1, -1)
    prev_y = torch.cat([rot, root_vel], dim=-1).reshape(b * t, -1)
    next_y = compose_delta_raw_to_next(
        prev_y,
        delta_raw.reshape(b * t, -1),
        rot_slice=base.rot_y_slice,
        columns=base.columns,
        omega_hat=None,
        gate_val=0.0,
        max_deg=0.0,
        omega_detach=True,
        reproject=False,
    )
    return next_y.reshape(b, t, -1)


def _dynamics_residual_from_state_aux(
    *,
    state: torch.Tensor,
    aux: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    cond_norm: torch.Tensor,
    base: BaseOperator,
    command_align_root_vel: bool,
    oracle_contact_passthrough: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _state_aux_to_eval_tensors(
        state=state,
        aux=aux,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
    )
    base_next = _base_operator_next_y(base=base, pred=pred, cond_norm=cond_norm, true_contact=true_contact)
    next_y = torch.cat([pred["rot6d"][:, 1:], pred["root_vel"][:, 1:]], dim=-1)
    resid = (next_y - base_next) / base.eval_y_scale.view(1, 1, -1)
    return pred, resid, next_y, base_next


def _gt_dynamics_residual_target(
    *,
    true_state: torch.Tensor,
    true_aux: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    cond_norm: torch.Tensor,
    base: BaseOperator,
    command_align_root_vel: bool,
    oracle_contact_passthrough: bool,
) -> torch.Tensor:
    with torch.no_grad():
        _, resid, _, _ = _dynamics_residual_from_state_aux(
            state=true_state,
            aux=true_aux,
            true_root_pos=true_root_pos,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            cond_norm=cond_norm,
            base=base,
            command_align_root_vel=command_align_root_vel,
            oracle_contact_passthrough=oracle_contact_passthrough,
        )
    return resid.detach()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    weight = mask.to(dtype=x.dtype)
    return (x * weight).sum() / weight.sum().clamp_min(1.0)


def _foot_velocity_loss(
    *,
    pred_rot6d: torch.Tensor,
    pred_root_pos: torch.Tensor,
    true_rot6d: torch.Tensor,
    true_root_pos: torch.Tensor,
    contact: torch.Tensor,
    skeleton: Any,
    offsets: torch.Tensor,
) -> torch.Tensor:
    b, h, pose_dim = pred_rot6d.shape
    joints = pose_dim // 6
    pred_pos = _fk_positions_from_rot6d_no_inplace(
        pred_rot6d.reshape(b, h, joints, 6),
        skeleton.parents,
        offsets,
        root_pos=pred_root_pos,
    )
    with torch.no_grad():
        true_pos = _fk_positions_from_rot6d_no_inplace(
            true_rot6d.reshape(b, h, joints, 6),
            skeleton.parents,
            offsets,
            root_pos=true_root_pos,
        )
    losses = []
    for ch_idx, joint_idx in ((0, skeleton.right_foot_idx), (1, skeleton.left_foot_idx)):
        if joint_idx is None:
            continue
        mask = (contact[:, :-1, ch_idx] > 0.5) & (contact[:, 1:, ch_idx] > 0.5)
        pred_speed = torch.linalg.norm(pred_pos[:, 1:, joint_idx] - pred_pos[:, :-1, joint_idx], dim=-1) * float(FPS)
        true_speed = torch.linalg.norm(true_pos[:, 1:, joint_idx] - true_pos[:, :-1, joint_idx], dim=-1) * float(FPS)
        losses.append(_masked_mean((pred_speed - true_speed).square(), mask))
    if not losses:
        return pred_rot6d.new_zeros(())
    return torch.stack(losses).sum()


def _label_has_side_for_loss(label: str, side: str) -> bool:
    return str(label) == "dual" or str(label) == str(side)


def _band_margin_loss(
    vals: torch.Tensor,
    bands: torch.Tensor,
    *,
    ignore_mask: Optional[torch.Tensor] = None,
    topk: int = 0,
) -> torch.Tensor:
    if vals.numel() == 0:
        return vals.new_zeros(())
    band = bands.to(device=vals.device, dtype=vals.dtype)
    while band.dim() < vals.dim():
        band = band.unsqueeze(-1)
    over = F.relu(vals / band.clamp_min(EPS) - 1.0).square()
    if ignore_mask is not None:
        mask = ignore_mask.to(device=vals.device, dtype=torch.bool)
        while mask.dim() < over.dim():
            mask = mask.unsqueeze(-1)
        over = torch.where(mask, torch.zeros_like(over), over)
    k = int(topk)
    if k > 0 and over.dim() >= 2 and over.shape[1] > k:
        over = torch.topk(over, k=k, dim=1).values
    return torch.mean(over)


def _interval_margin_loss(
    vals: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    scale_floor: float,
    power: float = 2.0,
    hard_gate_tolerance: bool = False,
    hard_gate_safety_margin: float = 0.0,
) -> torch.Tensor:
    if vals.numel() == 0:
        return vals.new_zeros(())
    lo_t = lo.to(device=vals.device, dtype=vals.dtype)
    hi_t = hi.to(device=vals.device, dtype=vals.dtype)
    while lo_t.dim() < vals.dim():
        lo_t = lo_t.unsqueeze(-1)
        hi_t = hi_t.unsqueeze(-1)
    if hard_gate_tolerance:
        tol = 1.0e-6 + 1.0e-5 * torch.maximum(torch.ones_like(lo_t), torch.maximum(lo_t.abs(), hi_t.abs()))
        scale = torch.maximum(tol, torch.full_like(tol, float(scale_floor))).clamp_min(EPS)
        safety = torch.full_like(tol, max(0.0, float(hard_gate_safety_margin)))
        low = F.relu((lo_t - tol + safety - vals) / scale)
        high = F.relu((vals - hi_t - tol + safety) / scale)
    else:
        scale = (hi_t - lo_t).abs().clamp_min(float(scale_floor))
        low = F.relu((lo_t - vals) / scale)
        high = F.relu((vals - hi_t) / scale)
    p = float(power)
    if abs(p - 1.0) <= 1.0e-12:
        return torch.mean(low + high)
    if abs(p - 2.0) <= 1.0e-12:
        return torch.mean(low.square() + high.square())
    return torch.mean(low.pow(p) + high.pow(p))


def _soft_max_violation(vals: Sequence[torch.Tensor], temperature: float) -> torch.Tensor:
    if not vals:
        return torch.zeros((), dtype=torch.float32)
    stacked = torch.stack([v.reshape(()) for v in vals])
    tau = float(temperature)
    if tau <= 0.0:
        return torch.max(stacked)
    return tau * torch.logsumexp(stacked / tau, dim=0) - tau * math.log(float(stacked.numel()))


def _masked_mean_batch(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=vals.device, dtype=vals.dtype)
    return (vals * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _masked_max_batch(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=vals.device, dtype=torch.bool)
    neg = torch.full_like(vals, -1.0e9)
    picked = torch.where(m, vals, neg)
    out = torch.max(picked, dim=1).values
    return torch.where(torch.any(m, dim=1), out, torch.zeros_like(out))


def _masked_quantile_batch(vals: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    m = mask.to(device=vals.device, dtype=torch.bool)
    for row, row_mask in zip(vals, m):
        picked = row[row_mask]
        if picked.numel() == 0:
            outs.append(row.new_zeros(()))
        else:
            outs.append(torch.quantile(picked, float(q)))
    return torch.stack(outs, dim=0) if outs else vals.new_zeros((0,))


def _heading_error_torch(root_vel: torch.Tensor, cond_dir: torch.Tensor) -> torch.Tensor:
    speed = torch.linalg.norm(root_vel, dim=-1)
    cmd_norm = torch.linalg.norm(cond_dir, dim=-1)
    valid = (speed > EPS) & (cmd_norm > EPS)
    rv = root_vel / speed.clamp_min(EPS).unsqueeze(-1)
    cd = cond_dir / cmd_norm.clamp_min(EPS).unsqueeze(-1)
    dot = torch.sum(rv * cd, dim=-1)
    cross = rv[..., 0] * cd[..., 1] - rv[..., 1] * cd[..., 0]
    err = torch.atan2(torch.abs(cross), dot)
    return torch.where(valid, err, torch.zeros_like(err))


def _loss_refactor_foot_positions(
    *,
    rot6d: torch.Tensor,
    root_pos: torch.Tensor,
    skeleton: Any,
    offsets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    b, h, pose_dim = rot6d.shape
    joints = pose_dim // 6
    pos = _fk_positions_from_rot6d_no_inplace(
        rot6d.reshape(b, h, joints, 6),
        skeleton.parents,
        offsets,
        root_pos=root_pos,
    )
    out: Dict[str, torch.Tensor] = {}
    if skeleton.right_foot_idx is not None:
        out["right"] = pos[:, :, skeleton.right_foot_idx]
    if skeleton.left_foot_idx is not None:
        out["left"] = pos[:, :, skeleton.left_foot_idx]
    return out


def _build_loss_refactor_context(
    *,
    items: Sequence[DecoderItem],
    idxs: Sequence[int],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    h = int(args.horizon)
    band_keys = (
        "pose_step_l2",
        "angvel_step_rms",
        "angvel_step_component_p95",
        "rootvel_step_l2",
        "yaw_rate_step_abs",
        "contact_step_l2",
        "heading_error_rad",
        "bone_angvel_level_rms",
        "foot_slip_contacted_speed_mps",
    )
    bands: Dict[str, List[float]] = {k: [] for k in band_keys}
    centers: List[np.ndarray] = []
    event_step_masks: List[np.ndarray] = []
    right_frame_masks: List[np.ndarray] = []
    left_frame_masks: List[np.ndarray] = []
    right_step_masks: List[np.ndarray] = []
    left_step_masks: List[np.ndarray] = []
    right_single_step_masks: List[np.ndarray] = []
    left_single_step_masks: List[np.ndarray] = []
    support_lo: Dict[str, List[float]] = {k: [] for k in SUPPORT_SIDE_FEATURE_KEYS}
    support_hi: Dict[str, List[float]] = {k: [] for k in SUPPORT_SIDE_FEATURE_KEYS}

    for item_idx in idxs:
        item = items[int(item_idx)]
        clip_bands = baseline_bands[item.clip]
        for key in band_keys:
            bands[key].append(float(clip_bands.get(key, 0.0)))
        centers.append(np.asarray(clip_bands["bone_angvel_level_center"], dtype=np.float32).reshape(ANGVEL_DIM))
        event = _oracle_event_masks(item, horizon=h, event_window=int(args.event_window))
        event_step_masks.append(np.asarray(event["step_mask"], dtype=bool).reshape(max(0, h - 1)))
        labels = [str(x) for x in event["labels"]]
        if len(labels) != h:
            labels = labels[:h] + ["flight_or_unknown"] * max(0, h - len(labels))
        right_frame_masks.append(np.asarray([_label_has_side_for_loss(x, "right") for x in labels], dtype=bool))
        left_frame_masks.append(np.asarray([_label_has_side_for_loss(x, "left") for x in labels], dtype=bool))
        right_step = np.zeros((max(0, h - 1),), dtype=bool)
        left_step = np.zeros_like(right_step)
        right_single = np.zeros_like(right_step)
        left_single = np.zeros_like(left_step)
        for t in range(max(0, h - 1)):
            r = _label_has_side_for_loss(labels[t], "right") and _label_has_side_for_loss(labels[t + 1], "right")
            l = _label_has_side_for_loss(labels[t], "left") and _label_has_side_for_loss(labels[t + 1], "left")
            right_step[t] = bool(r)
            left_step[t] = bool(l)
            right_single[t] = bool(r and not l)
            left_single[t] = bool(l and not r)
        right_step_masks.append(right_step)
        left_step_masks.append(left_step)
        right_single_step_masks.append(right_single)
        left_single_step_masks.append(left_single)

        feature_bands = support_bands[item.clip]["feature_bands"]
        for key in SUPPORT_SIDE_FEATURE_KEYS:
            band = feature_bands.get(key, {}) if isinstance(feature_bands, Mapping) else {}
            support_lo[key].append(float(band.get("min", 0.0)))
            support_hi[key].append(float(band.get("max", 0.0)))

    def t(arr: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return {
        "bands": {k: t(np.asarray(v, dtype=np.float32)).clamp_min(EPS) for k, v in bands.items()},
        "bone_angvel_level_center": t(np.stack(centers, axis=0) if centers else np.zeros((0, ANGVEL_DIM), dtype=np.float32)),
        "event_step_mask": t(np.stack(event_step_masks, axis=0) if event_step_masks else np.zeros((0, h - 1), dtype=bool), torch.bool),
        "right_frame_mask": t(np.stack(right_frame_masks, axis=0) if right_frame_masks else np.zeros((0, h), dtype=bool), torch.bool),
        "left_frame_mask": t(np.stack(left_frame_masks, axis=0) if left_frame_masks else np.zeros((0, h), dtype=bool), torch.bool),
        "right_step_mask": t(np.stack(right_step_masks, axis=0) if right_step_masks else np.zeros((0, h - 1), dtype=bool), torch.bool),
        "left_step_mask": t(np.stack(left_step_masks, axis=0) if left_step_masks else np.zeros((0, h - 1), dtype=bool), torch.bool),
        "right_single_step_mask": t(
            np.stack(right_single_step_masks, axis=0) if right_single_step_masks else np.zeros((0, h - 1), dtype=bool),
            torch.bool,
        ),
        "left_single_step_mask": t(
            np.stack(left_single_step_masks, axis=0) if left_single_step_masks else np.zeros((0, h - 1), dtype=bool),
            torch.bool,
        ),
        "support_lo": {k: t(np.asarray(v, dtype=np.float32)) for k, v in support_lo.items()},
        "support_hi": {k: t(np.asarray(v, dtype=np.float32)) for k, v in support_hi.items()},
        "rate_topk": int(args.loss_refactor_rate_topk),
        "pose_topk": int(args.loss_refactor_pose_topk),
        "heading_topk": int(args.loss_refactor_heading_topk),
        "support_feature_topk": int(args.loss_refactor_support_feature_topk),
        "support_scale_floor": float(args.loss_refactor_support_band_floor),
        "support_margin_power": float(args.loss_refactor_support_margin_power),
        "support_linear_feature_keys": str(args.loss_refactor_support_linear_feature_keys),
        "support_excluded_feature_keys": str(args.loss_refactor_support_excluded_feature_keys),
        "support_hard_gate_feature_keys": str(args.loss_refactor_support_hard_gate_feature_keys),
        "support_hard_gate_safety_margin": float(args.loss_refactor_support_hard_gate_safety_margin),
        "heading_tolerance_rad": float(args.heading_tolerance_rad),
        "dynamics_low_band": float(args.loss_refactor_dynamics_low_band),
        "minimax_temperature": float(args.loss_refactor_minimax_temperature),
        "anchor_weight": float(args.loss_refactor_anchor_weight),
    }


def _loss_refactor_support_side_terms(
    *,
    pred: Mapping[str, torch.Tensor],
    foot: Mapping[str, torch.Tensor],
    ctx: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    root_pos = pred["root_pos"]
    root_vel = pred["root_vel"]
    yaw = pred["yaw"].squeeze(-1)
    heading = _heading_error_torch(root_vel, ctx["cond_dir"])
    b, h = yaw.shape
    zeros = yaw.new_zeros((b,))

    if "right" in foot:
        right_speed = torch.linalg.norm(foot["right"][:, 1:] - foot["right"][:, :-1], dim=-1) * float(FPS)
    else:
        right_speed = yaw.new_zeros((b, max(0, h - 1)))
    if "left" in foot:
        left_speed = torch.linalg.norm(foot["left"][:, 1:] - foot["left"][:, :-1], dim=-1) * float(FPS)
    else:
        left_speed = yaw.new_zeros((b, max(0, h - 1)))

    right_claim = ctx["right_step_mask"].to(device=yaw.device)
    left_claim = ctx["left_step_mask"].to(device=yaw.device)
    right_single = ctx["right_single_step_mask"].to(device=yaw.device)
    left_single = ctx["left_single_step_mask"].to(device=yaw.device)
    claimed_vals = torch.cat([right_speed, left_speed], dim=1)
    claimed_mask = torch.cat([right_claim, left_claim], dim=1)
    diff_vals = torch.cat([right_speed - left_speed, left_speed - right_speed], dim=1)
    diff_mask = torch.cat([right_single, left_single], dim=1)
    ratio_vals = torch.cat(
        [right_speed / left_speed.clamp_min(1.0e-4), left_speed / right_speed.clamp_min(1.0e-4)],
        dim=1,
    )

    feats: Dict[str, torch.Tensor] = {
        "claimed_support_slip_mean_mps": _masked_mean_batch(claimed_vals, claimed_mask),
        "claimed_support_slip_p95_mps": _masked_quantile_batch(claimed_vals, claimed_mask, 0.95),
        "claimed_support_slip_max_mps": _masked_max_batch(claimed_vals, claimed_mask),
        "single_support_claimed_minus_opposite_mean_mps": _masked_mean_batch(diff_vals, diff_mask),
        "single_support_claimed_minus_opposite_p95_mps": _masked_quantile_batch(diff_vals, diff_mask, 0.95),
        "single_support_claimed_speed_ratio_p95": _masked_quantile_batch(ratio_vals, diff_mask, 0.95),
        "yaw_sum_rad": torch.sum(yaw, dim=1) / float(FPS),
        "yaw_abs_sum_rad": torch.sum(torch.abs(yaw), dim=1) / float(FPS),
        "heading_error_p95_rad": torch.quantile(heading, 0.95, dim=1) if heading.numel() else zeros,
        "root_speed_mean": torch.mean(torch.linalg.norm(root_vel, dim=-1), dim=1),
        "root_lateral_mean": torch.mean(root_vel[:, :, 1], dim=1),
    }

    for side in ("right", "left"):
        mask = ctx[f"{side}_frame_mask"].to(device=yaw.device)
        if side in foot:
            rel = foot[side] - root_pos
            rel_norm = torch.linalg.norm(rel, dim=-1)
            for dim, axis in enumerate(("x", "y", "z")):
                feats[f"{side}_rel_{axis}_mean"] = _masked_mean_batch(rel[:, :, dim], mask)
            feats[f"{side}_rel_norm_p95"] = _masked_quantile_batch(rel_norm, mask, 0.95)
        else:
            for axis in ("x", "y", "z"):
                feats[f"{side}_rel_{axis}_mean"] = zeros
            feats[f"{side}_rel_norm_p95"] = zeros

    balance = (
        ctx["right_frame_mask"].to(device=yaw.device, dtype=yaw.dtype).mean(dim=1)
        - ctx["left_frame_mask"].to(device=yaw.device, dtype=yaw.dtype).mean(dim=1)
    )
    feats["support_yaw_product"] = balance * feats["yaw_sum_rad"]
    feats["support_lateral_product"] = balance * feats["root_lateral_mean"]

    losses = []
    out: Dict[str, torch.Tensor] = {}
    scale_floor = float(ctx["support_scale_floor"])
    margin_power = float(ctx.get("support_margin_power", 2.0))
    linear_keys = set(str(ctx.get("support_linear_feature_keys", "")).split(","))
    linear_keys.discard("")
    excluded_keys = set(str(ctx.get("support_excluded_feature_keys", "")).split(","))
    excluded_keys.discard("")
    hard_gate_keys = set(str(ctx.get("support_hard_gate_feature_keys", "")).split(","))
    hard_gate_keys.discard("")
    hard_gate_safety = float(ctx.get("support_hard_gate_safety_margin", 0.0) or 0.0)
    for key, val in feats.items():
        if key in excluded_keys:
            continue
        loss = _interval_margin_loss(
            val,
            ctx["support_lo"][key],
            ctx["support_hi"][key],
            scale_floor=scale_floor,
            power=1.0 if key in linear_keys else margin_power,
            hard_gate_tolerance=key in hard_gate_keys,
            hard_gate_safety_margin=hard_gate_safety if key in hard_gate_keys else 0.0,
        )
        out[f"contact_support_side_{key}_margin"] = loss
        losses.append(loss)
    if not losses:
        return {"contact_support_side_margin": yaw.new_zeros(())}
    stacked = torch.stack(losses)
    topk = int(ctx.get("support_feature_topk", 0) or 0)
    if topk > 0 and stacked.numel() > topk:
        stacked = torch.topk(stacked, k=topk, dim=0).values
    out["contact_support_side_margin"] = stacked.mean()
    return out


def _loss_refactor_causal_terms(
    *,
    pred: Mapping[str, torch.Tensor],
    true_state: torch.Tensor,
    true_aux: torch.Tensor,
    true_root_vel: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    dyn_resid: torch.Tensor,
    gt_dynamics_resid: torch.Tensor,
    skeleton: Any,
    offsets: torch.Tensor,
    ctx: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    bands = ctx["bands"]
    event_mask = ctx["event_step_mask"].to(device=pred["rot6d"].device)
    pose_delta = pred["rot6d"][:, 1:] - pred["rot6d"][:, :-1]
    pose_step_l2 = torch.linalg.norm(pose_delta, dim=-1) / math.sqrt(max(1, pose_delta.shape[-1]))
    aux_delta = pred["aux"][:, 1:] - pred["aux"][:, :-1]
    angvel_rms = torch.sqrt(torch.mean(aux_delta.square(), dim=-1).clamp_min(0.0) + 1.0e-12)
    angvel_component = torch.quantile(torch.abs(aux_delta), 0.95, dim=-1) if aux_delta.numel() else angvel_rms
    rootvel_step = torch.linalg.norm(pred["root_vel"][:, 1:] - pred["root_vel"][:, :-1], dim=-1)
    yaw_step = torch.abs(pred["yaw"][:, 1:].squeeze(-1) - pred["yaw"][:, :-1].squeeze(-1))
    contact_step = torch.linalg.norm(pred["contact"][:, 1:] - pred["contact"][:, :-1], dim=-1)

    rot_width = POSE_SLICE.stop - POSE_SLICE.start
    articulation_low = F.mse_loss(dyn_resid[:, :, :rot_width], gt_dynamics_resid[:, :, :rot_width]) / max(
        float(ctx["dynamics_low_band"]) ** 2,
        EPS,
    )
    root_support_low = F.mse_loss(dyn_resid[:, :, rot_width:], gt_dynamics_resid[:, :, rot_width:]) / max(
        float(ctx["dynamics_low_band"]) ** 2,
        EPS,
    )
    articulation_angvel_rms = _band_margin_loss(
        angvel_rms,
        bands["angvel_step_rms"],
        ignore_mask=event_mask,
        topk=int(ctx["rate_topk"]),
    )
    articulation_angvel_component = _band_margin_loss(
        angvel_component,
        bands["angvel_step_component_p95"],
        topk=int(ctx["rate_topk"]),
    )
    articulation_rate = torch.stack([articulation_angvel_rms, articulation_angvel_component]).mean()
    rootvel_rate = _band_margin_loss(rootvel_step, bands["rootvel_step_l2"], topk=int(ctx["rate_topk"]))
    true_rootvel_delta = true_root_vel[:, 1:] - true_root_vel[:, :-1]
    rootvel_rate_anchor = torch.mean(
        (
            (pred["root_vel"][:, 1:] - pred["root_vel"][:, :-1] - true_rootvel_delta)
            / bands["rootvel_step_l2"].view(-1, 1, 1).clamp_min(EPS)
        ).square()
    )
    rootvel_path_anchor = torch.mean(
        ((pred["root_vel"] - true_root_vel) / bands["rootvel_step_l2"].view(-1, 1, 1).clamp_min(EPS)).square()
    )
    yaw_rate = _band_margin_loss(yaw_step, bands["yaw_rate_step_abs"], topk=int(ctx["rate_topk"]))
    pose_margin = _band_margin_loss(pose_step_l2, bands["pose_step_l2"], topk=int(ctx["pose_topk"]))

    foot = _loss_refactor_foot_positions(
        rot6d=pred["rot6d"],
        root_pos=pred["root_pos"],
        skeleton=skeleton,
        offsets=offsets,
    )
    foot_losses = []
    for ch_idx, side in ((0, "right"), (1, "left")):
        if side not in foot:
            continue
        mask = (true_contact[:, :-1, ch_idx] > 0.5) & (true_contact[:, 1:, ch_idx] > 0.5)
        speed = torch.linalg.norm(foot[side][:, 1:] - foot[side][:, :-1], dim=-1) * float(FPS)
        foot_losses.append(_band_margin_loss(speed, bands["foot_slip_contacted_speed_mps"], ignore_mask=~mask, topk=int(ctx["rate_topk"])))
    fk_slip = torch.stack(foot_losses).mean() if foot_losses else pred["rot6d"].new_zeros(())
    contact_schedule = _band_margin_loss(contact_step, bands["contact_step_l2"], ignore_mask=event_mask, topk=int(ctx["rate_topk"]))
    contact_anchor = torch.mean(((pred["contact"] - true_contact) / bands["contact_step_l2"].view(-1, 1, 1).clamp_min(EPS)).square())
    support_side_terms = _loss_refactor_support_side_terms(pred=pred, foot=foot, ctx={**ctx, "cond_dir": true_cond_dir})
    support_side = support_side_terms["contact_support_side_margin"]

    endpoint_pose = torch.mean(
        ((pred["rot6d"][:, -1] - true_state[:, -1, POSE_SLICE]) / bands["pose_step_l2"].view(-1, 1).clamp_min(EPS)).square()
    )
    endpoint_root = torch.mean(
        ((pred["root_vel"][:, -1] - true_root_vel[:, -1]) / bands["rootvel_step_l2"].view(-1, 1).clamp_min(EPS)).square()
    )
    endpoint_contact = torch.mean(
        ((pred["contact"][:, -1] - true_contact[:, -1]) / bands["contact_step_l2"].view(-1, 1).clamp_min(EPS)).square()
    )
    goal_endpoint = torch.stack([endpoint_pose, endpoint_root, endpoint_contact]).mean()
    heading_band = torch.maximum(
        bands["heading_error_rad"],
        torch.full_like(bands["heading_error_rad"], float(ctx["heading_tolerance_rad"])),
    )
    heading = _heading_error_torch(
        pred["root_vel"],
        true_cond_dir.to(device=pred["root_vel"].device, dtype=pred["root_vel"].dtype),
    )
    goal_heading = _band_margin_loss(heading, heading_band, topk=int(ctx["heading_topk"]))
    level_dist = torch.sqrt(torch.mean((pred["aux"][:, -1] - ctx["bone_angvel_level_center"].to(device=pred["aux"].device)) ** 2, dim=-1) + 1.0e-12)
    goal_regime = _band_margin_loss(level_dist, bands["bone_angvel_level_rms"], topk=0)

    l_articulation = torch.stack([articulation_low, articulation_rate, pose_margin]).mean()
    contact_honesty = torch.stack([contact_schedule, contact_anchor, fk_slip]).mean()
    l_root_support = torch.stack(
        [root_support_low, rootvel_rate, rootvel_rate_anchor, rootvel_path_anchor, contact_honesty, support_side]
    ).mean()
    l_goal = torch.stack([goal_endpoint, goal_heading, yaw_rate, goal_regime]).mean()
    gate_terms = [
        articulation_angvel_rms,
        articulation_angvel_component,
        pose_margin,
        rootvel_rate,
        contact_schedule,
        fk_slip,
        goal_heading,
        yaw_rate,
        goal_regime,
    ]
    support_feature_terms = [
        val
        for key, val in sorted(support_side_terms.items())
        if key.startswith("contact_support_side_") and key != "contact_support_side_margin"
    ]
    gate_terms.extend(support_feature_terms)
    gate_stack = torch.stack(gate_terms) if gate_terms else pred["rot6d"].new_zeros((1,))
    anchor_tiebreaker = torch.stack(
        [
            articulation_low,
            root_support_low,
            rootvel_rate_anchor,
            rootvel_path_anchor,
            contact_anchor,
            goal_endpoint,
        ]
    ).mean()
    soft_gate = _soft_max_violation(gate_terms, float(ctx["minimax_temperature"]))
    minimax_feasibility = soft_gate + float(ctx["anchor_weight"]) * anchor_tiebreaker
    out = {
        "L_articulation": l_articulation,
        "L_root_support": l_root_support,
        "L_goal": l_goal,
        "loss_refactor_minimax_feasibility": minimax_feasibility,
        "loss_refactor_softmax_gate_violation": soft_gate,
        "loss_refactor_hard_max_gate_violation": torch.max(gate_stack),
        "loss_refactor_mean_gate_violation": torch.mean(gate_stack),
        "loss_refactor_anchor_tiebreaker": anchor_tiebreaker,
        "articulation_low_anchor_loss": articulation_low,
        "articulation_angvel_rms_margin_loss": articulation_angvel_rms,
        "articulation_angvel_component_margin_loss": articulation_angvel_component,
        "articulation_angvel_rate_margin_loss": articulation_rate,
        "articulation_pose_step_margin_loss": pose_margin,
        "root_support_low_anchor_loss": root_support_low,
        "root_support_rootvel_rate_margin_loss": rootvel_rate,
        "root_support_rootvel_rate_anchor_loss": rootvel_rate_anchor,
        "root_support_rootvel_path_anchor_loss": rootvel_path_anchor,
        "root_support_contact_step_margin_loss": contact_schedule,
        "root_support_contact_anchor_loss": contact_anchor,
        "root_support_fk_slip_margin_loss": fk_slip,
        "root_support_contact_honesty_loss": contact_honesty,
        "root_support_side_margin_loss": support_side,
        "goal_endpoint_margin_loss": goal_endpoint,
        "goal_heading_margin_loss": goal_heading,
        "goal_yaw_rate_margin_loss": yaw_rate,
        "goal_regime_margin_loss": goal_regime,
    }
    for key, val in sorted(support_side_terms.items()):
        if key == "contact_support_side_margin":
            continue
        out[f"root_support_{key}_loss"] = val
    return out


def _objective(
    *,
    pred_std: torch.Tensor,
    ytr_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    true_raw: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_root_vel: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    cond_norm: torch.Tensor,
    gt_dynamics_resid: torch.Tensor,
    base: BaseOperator,
    skeleton: Any,
    offsets: torch.Tensor,
    horizon: int,
    weights: Mapping[str, float],
    command_align_root_vel: bool,
    oracle_contact_passthrough: bool,
    pose_gate_band: Optional[torch.Tensor] = None,
    pose_gate_topk: int = 0,
    loss_refactor_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    pred_raw = pred_std * y_std + y_mean
    state_width = int(horizon) * STATE_DIM
    pred_state = pred_raw[:, :state_width].reshape(-1, int(horizon), STATE_DIM)
    pred_aux = pred_raw[:, state_width:].reshape(-1, int(horizon), ANGVEL_DIM)
    true_state = true_raw[:, :state_width].reshape(-1, int(horizon), STATE_DIM)
    true_aux = true_raw[:, state_width:].reshape(-1, int(horizon), ANGVEL_DIM)
    pred_state.retain_grad()
    pred_aux.retain_grad()

    pred, dyn_resid, _, _ = _dynamics_residual_from_state_aux(
        state=pred_state,
        aux=pred_aux,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        cond_norm=cond_norm,
        base=base,
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
    )
    dynamics = F.mse_loss(dyn_resid, gt_dynamics_resid)
    dynamics_zero_residual = torch.mean(dyn_resid.square())

    endpoint_pose = F.mse_loss(pred["rot6d"][:, -1], true_state[:, -1, POSE_SLICE])
    endpoint_rootvel = F.mse_loss(pred["root_vel"][:, -1], true_root_vel[:, -1])
    endpoint_contact = F.mse_loss(pred["contact"][:, -1], true_contact[:, -1])
    endpoint = endpoint_pose + endpoint_rootvel + endpoint_contact
    regime = F.mse_loss(pred_aux[:, -1], true_aux[:, -1])

    lateral = pred["ego"][:, :, 1]
    speed = torch.linalg.norm(pred["ego"], dim=-1).clamp_min(EPS)
    command = torch.mean((lateral / speed).square()) + torch.mean(lateral.square())

    state_raw_mse = F.mse_loss(pred_state, true_state)
    flat = F.mse_loss(pred_std, ytr_std)
    pose = F.mse_loss(pred["rot6d"], true_state[:, :, POSE_SLICE])
    contact = F.mse_loss(pred["contact"], true_contact)
    root_vel = F.mse_loss(pred["root_vel"], true_root_vel)
    root_pos = F.mse_loss(pred["root_pos"], true_root_pos)
    aux = F.mse_loss(pred_aux, true_aux)
    pose_step = F.mse_loss(
        pred["rot6d"][:, 1:] - pred["rot6d"][:, :-1],
        true_state[:, 1:, POSE_SLICE] - true_state[:, :-1, POSE_SLICE],
    )
    pose_gate_margin = pred_state.new_zeros(())
    if pose_gate_band is not None:
        pose_delta = pred["rot6d"][:, 1:] - pred["rot6d"][:, :-1]
        pose_step_l2 = torch.linalg.norm(pose_delta, dim=-1) / math.sqrt(max(1, pose_delta.shape[-1]))
        margin = pose_step_l2 - pose_gate_band.reshape(-1, 1).to(device=pose_step_l2.device, dtype=pose_step_l2.dtype)
        over = F.relu(margin).square()
        topk = int(pose_gate_topk)
        if topk > 0 and over.shape[1] > topk:
            over = torch.topk(over, k=topk, dim=1).values
        pose_gate_margin = torch.mean(over)
    rootvel_step = F.mse_loss(
        pred["root_vel"][:, 1:] - pred["root_vel"][:, :-1],
        true_root_vel[:, 1:] - true_root_vel[:, :-1],
    )
    yaw_step = F.mse_loss(
        pred["yaw"][:, 1:] - pred["yaw"][:, :-1],
        true_state[:, 1:, YAW_RATE_SLICE] - true_state[:, :-1, YAW_RATE_SLICE],
    )
    aux_rate = F.mse_loss(pred_aux[:, 1:] - pred_aux[:, :-1], true_aux[:, 1:] - true_aux[:, :-1])
    foot_vel = _foot_velocity_loss(
        pred_rot6d=pred["rot6d"],
        pred_root_pos=pred["root_pos"],
        true_rot6d=true_state[:, :, POSE_SLICE],
        true_root_pos=true_root_pos,
        contact=true_contact,
        skeleton=skeleton,
        offsets=offsets,
    )

    terms = {
        "dynamics_consistency": dynamics,
        "dynamics_zero_residual_witness": dynamics_zero_residual,
        "command_compatibility": command,
        "endpoint_reaching": endpoint,
        "regime_reaching": regime,
        "state_raw_mse": state_raw_mse,
        "flat_standardized": flat,
        "pose_supervision": pose,
        "contact_schedule": contact,
        "root_vel_supervision": root_vel,
        "root_pos_supervision": root_pos,
        "bone_angvel_supervision": aux,
        "pose_continuity_loss": pose_step,
        "pose_gate_margin_loss": pose_gate_margin,
        "rootvel_rate_loss": rootvel_step,
        "yaw_rate_loss": yaw_step,
        "bone_angvel_rate_loss": aux_rate,
        "fk_foot_slip_loss": foot_vel,
    }
    if loss_refactor_context is not None:
        terms.update(
            _loss_refactor_causal_terms(
                pred=pred,
                true_state=true_state,
                true_aux=true_aux,
                true_root_vel=true_root_vel,
                true_cond_dir=true_cond_dir,
                true_contact=true_contact,
                dyn_resid=dyn_resid,
                gt_dynamics_resid=gt_dynamics_resid,
                skeleton=skeleton,
                offsets=offsets,
                ctx=loss_refactor_context,
            )
        )
    loss = pred_std.new_zeros(())
    details: Dict[str, float] = {}
    for key, val in terms.items():
        w = float(weights.get(key, 0.0))
        if w:
            loss = loss + w * val
        details[key] = float(val.detach().cpu().item())
    return loss, details, terms, pred_state, pred_aux


def _params_vector(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    chunks = []
    for grad in grads:
        if grad is None:
            continue
        chunks.append(grad.detach().reshape(-1).double().cpu())
    if not chunks:
        return torch.zeros((0,), dtype=torch.float64)
    return torch.cat(chunks, dim=0)


def _grad_norm(term: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> float:
    if not bool(getattr(term, "requires_grad", False)):
        return 0.0
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    vec = _params_vector(grads)
    return float(torch.linalg.norm(vec).item()) if vec.numel() else 0.0


def _channel_param_grad_vectors(
    *,
    total_loss: torch.Tensor,
    pred_state: torch.Tensor,
    pred_aux: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    grad_state, grad_aux = torch.autograd.grad(
        total_loss,
        (pred_state, pred_aux),
        retain_graph=True,
        allow_unused=True,
    )
    if grad_state is None:
        grad_state = torch.zeros_like(pred_state)
    if grad_aux is None:
        grad_aux = torch.zeros_like(pred_aux)
    specs = {
        "pose": (grad_state * _state_mask(pred_state, POSE_SLICE), torch.zeros_like(grad_aux)),
        "contact": (grad_state * _state_mask(pred_state, CONTACT_SLICE), torch.zeros_like(grad_aux)),
        "rootvel": (grad_state * _state_mask(pred_state, EGO_VEL_SLICE), torch.zeros_like(grad_aux)),
        "bone_angvel": (torch.zeros_like(grad_state), grad_aux),
    }
    out: Dict[str, torch.Tensor] = {}
    for name, (gs, ga) in specs.items():
        grads = torch.autograd.grad(
            (pred_state, pred_aux),
            params,
            grad_outputs=(gs, ga),
            retain_graph=True,
            allow_unused=True,
        )
        out[name] = _params_vector(grads)
    return out


def _state_mask(ref: torch.Tensor, sl: slice) -> torch.Tensor:
    mask = torch.zeros_like(ref)
    mask[:, :, sl] = 1.0
    return mask


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-20:
        return 0.0
    return float(torch.dot(a, b).item() / float(denom.item()))


def _instrument_step(
    *,
    stage: str,
    arm: str,
    epoch: int,
    total_loss: torch.Tensor,
    details: Mapping[str, float],
    terms: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    pred_state: torch.Tensor,
    pred_aux: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage": stage,
        "arm": arm,
        "epoch": int(epoch),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    for key in terms:
        term = terms[key]
        weight = float(weights.get(key, 0.0))
        row[f"{key}_loss"] = float(details.get(key, 0.0))
        row[f"{key}_weight"] = weight
        row[f"{key}_grad_norm"] = _grad_norm(term, params)
        row[f"{key}_weighted_grad_norm"] = _grad_norm(weight * term, params) if weight else 0.0
    channel_vecs = _channel_param_grad_vectors(
        total_loss=total_loss,
        pred_state=pred_state,
        pred_aux=pred_aux,
        params=params,
    )
    for ch in CHANNELS:
        vec = channel_vecs[ch]
        row[f"channel_{ch}_grad_norm"] = float(torch.linalg.norm(vec).item()) if vec.numel() else 0.0
    for a, b in PAIR_KEYS:
        row[f"channel_cos_{a}_vs_{b}"] = _cos(channel_vecs[a], channel_vecs[b])
    return row


def _weights_for_arm(args: argparse.Namespace, arm: str) -> Dict[str, float]:
    weights = {
        "dynamics_consistency": float(args.dynamics_loss_weight),
        "dynamics_zero_residual_witness": 0.0,
        "command_compatibility": float(args.command_loss_weight),
        "endpoint_reaching": float(args.endpoint_loss_weight),
        "regime_reaching": float(args.regime_loss_weight),
        "state_raw_mse": float(args.state_anchor_loss_weight),
        "flat_standardized": float(args.flat_standardized_loss_weight),
        "pose_supervision": 0.0,
        "contact_schedule": 0.0,
        "root_vel_supervision": 0.0,
        "root_pos_supervision": 0.0,
        "bone_angvel_supervision": 0.0,
        "pose_continuity_loss": 0.0,
        "rootvel_rate_loss": 0.0,
        "yaw_rate_loss": 0.0,
        "bone_angvel_rate_loss": 0.0,
        "fk_foot_slip_loss": 0.0,
        "L_articulation": 0.0,
        "L_root_support": 0.0,
        "L_goal": 0.0,
        "loss_refactor_minimax_feasibility": 0.0,
    }
    if arm == "symptom_ablation":
        weights["dynamics_consistency"] = 0.0
        weights.update(
            {
                "pose_supervision": float(args.symptom_pose_loss_weight),
                "contact_schedule": float(args.symptom_contact_loss_weight),
                "root_vel_supervision": float(args.symptom_rootvel_loss_weight),
                "root_pos_supervision": float(args.symptom_rootpos_loss_weight),
                "bone_angvel_supervision": float(args.symptom_aux_loss_weight),
                "pose_continuity_loss": float(args.symptom_pose_step_loss_weight),
                "rootvel_rate_loss": float(args.symptom_rootvel_rate_loss_weight),
                "yaw_rate_loss": float(args.symptom_yaw_rate_loss_weight),
                "bone_angvel_rate_loss": float(args.symptom_aux_rate_loss_weight),
                "fk_foot_slip_loss": float(args.symptom_fk_foot_loss_weight),
            }
        )
    elif arm == "dynamics_consistency":
        weights.update(
            {
                "contact_schedule": float(args.mechanism_contact_loss_weight),
                "pose_continuity_loss": float(args.mechanism_pose_step_loss_weight),
                "rootvel_rate_loss": float(args.mechanism_rootvel_rate_loss_weight),
                "yaw_rate_loss": float(args.mechanism_yaw_rate_loss_weight),
                "bone_angvel_rate_loss": float(args.mechanism_aux_rate_loss_weight),
                "fk_foot_slip_loss": float(args.mechanism_fk_foot_loss_weight),
            }
        )
    elif arm == "loss_refactor_causal3":
        for key in list(weights.keys()):
            weights[key] = 0.0
        objective = str(args.loss_refactor_objective)
        if objective == "weighted":
            weights.update(_loss_refactor_weighted_weights(args))
        elif objective == "minimax":
            weights["loss_refactor_minimax_feasibility"] = 1.0
        else:
            raise ValueError(f"unknown loss_refactor_objective: {objective}")
    else:
        raise ValueError(f"unknown arm: {arm}")
    return weights


def _loss_refactor_weighted_weights(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "L_articulation": float(args.loss_refactor_articulation_weight),
        "L_root_support": float(args.loss_refactor_root_support_weight),
        "L_goal": float(args.loss_refactor_goal_weight),
    }


def _train_stage_arm(
    *,
    stage: str,
    arm: str,
    idxs: Sequence[int],
    items: Sequence[DecoderItem],
    base: BaseOperator,
    skeleton: Any,
    args: argparse.Namespace,
    device: torch.device,
    pose_gate_bands: Optional[np.ndarray] = None,
    pose_gate_loss_weight: float = 0.0,
    pose_gate_topk: int = 0,
    loss_refactor_context: Optional[Mapping[str, Any]] = None,
    instrument_step_log: bool = True,
) -> Dict[str, Any]:
    train_x_raw, train_y_raw = _dataset_arrays(items, idxs)
    x_scaler = _fit_standardizer(train_x_raw)
    y_scaler = _fit_standardizer(train_y_raw)
    train_x = x_scaler.transform(train_x_raw)
    train_y = y_scaler.transform(train_y_raw)
    torch.manual_seed(int(args.seed) + len(idxs) * 1009 + (0 if arm == "dynamics_consistency" else 100_003))
    model = TinyDeterministicDecoder(train_x.shape[1], int(args.hidden_dim), train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    params = [p for p in model.parameters() if p.requires_grad]
    weights = _weights_for_arm(args, arm)
    weights["pose_gate_margin_loss"] = float(pose_gate_loss_weight)
    warmup_weights = dict(weights)
    warmup_epochs = 0
    if arm == "loss_refactor_causal3" and str(args.loss_refactor_objective) == "minimax":
        warmup_epochs = max(0, int(args.loss_refactor_minimax_warmup_epochs))
        if warmup_epochs > 0:
            for key in list(warmup_weights.keys()):
                warmup_weights[key] = 0.0
            warmup_mode = str(args.loss_refactor_minimax_warmup_mode)
            if warmup_mode == "weighted":
                warmup_weights.update(_loss_refactor_weighted_weights(args))
            elif warmup_mode == "supervised_flat":
                warmup_weights["flat_standardized"] = 1.0
            elif warmup_mode == "supervised_raw":
                warmup_weights["state_raw_mse"] = 1.0
                warmup_weights["bone_angvel_supervision"] = 1.0
            else:
                raise ValueError(f"unknown loss_refactor_minimax_warmup_mode: {warmup_mode}")
            warmup_weights["pose_gate_margin_loss"] = float(pose_gate_loss_weight)

    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    y_mean = torch.as_tensor(y_scaler.mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(y_scaler.std, dtype=torch.float32, device=device)
    true_raw = torch.as_tensor(train_y_raw, dtype=torch.float32, device=device)
    true_root_pos = torch.as_tensor(_stack_seq(items, idxs, "root_pos"), dtype=torch.float32, device=device)
    true_root_vel = torch.as_tensor(_stack_seq(items, idxs, "root_vel"), dtype=torch.float32, device=device)
    true_cond_dir = torch.as_tensor(_stack_seq(items, idxs, "cond_dir"), dtype=torch.float32, device=device)
    true_contact = torch.as_tensor(_stack_seq(items, idxs, "contact"), dtype=torch.float32, device=device)
    cond_raw = _stack_cond_raw(base, items, idxs, int(args.horizon))
    cond_norm = torch.as_tensor(_robust_cond_norm(cond_raw), dtype=torch.float32, device=device)
    offsets = torch.as_tensor(skeleton.offsets, dtype=torch.float32, device=device)
    state_width = int(args.horizon) * STATE_DIM
    true_state = true_raw[:, :state_width].reshape(-1, int(args.horizon), STATE_DIM)
    true_aux = true_raw[:, state_width:].reshape(-1, int(args.horizon), ANGVEL_DIM)
    pose_gate_band_t = None
    if pose_gate_bands is not None:
        pose_gate_band_t = torch.as_tensor(np.asarray(pose_gate_bands, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device)
    gt_dynamics_resid = _gt_dynamics_residual_target(
        true_state=true_state,
        true_aux=true_aux,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        cond_norm=cond_norm,
        base=base,
        command_align_root_vel=bool(args.command_align_root_vel),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
    )

    step_rows: List[Dict[str, Any]] = []
    final_loss = 0.0
    final_terms: Dict[str, float] = {}
    log_stride = max(1, int(args.instrument_step_log_stride))
    minimax_lr_applied = False
    for epoch in range(int(args.epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_std = model(xtr)
        epoch_weights = warmup_weights if epoch < warmup_epochs else weights
        if (
            warmup_epochs > 0
            and epoch >= warmup_epochs
            and not minimax_lr_applied
            and float(args.loss_refactor_minimax_tail_lr) > 0.0
        ):
            for group in opt.param_groups:
                group["lr"] = float(args.loss_refactor_minimax_tail_lr)
            minimax_lr_applied = True
        loss, final_terms, term_tensors, pred_state, pred_aux = _objective(
            pred_std=pred_std,
            ytr_std=ytr,
            y_mean=y_mean,
            y_std=y_std,
            true_raw=true_raw,
            true_root_pos=true_root_pos,
            true_root_vel=true_root_vel,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            cond_norm=cond_norm,
            gt_dynamics_resid=gt_dynamics_resid,
            base=base,
            skeleton=skeleton,
            offsets=offsets,
            horizon=int(args.horizon),
            weights=epoch_weights,
            command_align_root_vel=bool(args.command_align_root_vel),
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
            pose_gate_band=pose_gate_band_t,
            pose_gate_topk=int(pose_gate_topk),
            loss_refactor_context=loss_refactor_context,
        )
        if bool(instrument_step_log) and (epoch == 0 or epoch == int(args.epochs) - 1 or epoch % log_stride == 0):
            step_rows.append(
                _instrument_step(
                    stage=stage,
                    arm=arm,
                    epoch=epoch,
                    total_loss=loss,
                    details=final_terms,
                    terms=term_tensors,
                    weights=epoch_weights,
                    pred_state=pred_state,
                    pred_aux=pred_aux,
                    params=params,
                )
            )
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())

    pred_raw = _predict_raw(model, train_x_raw, x_scaler, y_scaler, device)
    if args.oracle_contact_passthrough:
        pred_raw = _apply_oracle_contact_passthrough(pred_raw, items, idxs, int(args.horizon))
    return {
        "stage": stage,
        "arm": arm,
        "idxs": [int(x) for x in idxs],
        "model": model,
        "pred_raw": pred_raw,
        "true_raw": train_y_raw,
        "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std, "constant_count": int(x_scaler.constant_count)},
        "y_scaler": {"mean": y_scaler.mean, "std": y_scaler.std, "constant_count": int(y_scaler.constant_count)},
        "train_loss_metrics": _loss_metrics(pred_raw, train_y_raw, int(args.horizon)),
        "final_train_objective": final_loss,
        "final_train_objective_terms": final_terms,
        "weights": weights,
        "warmup_epochs": int(warmup_epochs),
        "warmup_weights": warmup_weights if warmup_epochs > 0 else {},
        "step_rows": step_rows,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "input_dim": int(train_x_raw.shape[1]),
        "output_dim": int(train_y_raw.shape[1]),
    }


def _gt_dynamics_floor(
    *,
    idxs: Sequence[int],
    items: Sequence[DecoderItem],
    base: BaseOperator,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    _, y_raw = _dataset_arrays(items, idxs)
    state, aux = _reshape_state_aux(y_raw, int(args.horizon))
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    aux_t = torch.as_tensor(aux, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    true_root_pos = torch.as_tensor(_stack_seq(items, idxs, "root_pos"), dtype=torch.float32, device=device)
    true_root_vel = torch.as_tensor(_stack_seq(items, idxs, "root_vel"), dtype=torch.float32, device=device)
    true_cond_dir = torch.as_tensor(_stack_seq(items, idxs, "cond_dir"), dtype=torch.float32, device=device)
    true_contact = torch.as_tensor(_stack_seq(items, idxs, "contact"), dtype=torch.float32, device=device)
    cond_raw = _stack_cond_raw(base, items, idxs, int(args.horizon))
    cond_norm = torch.as_tensor(_robust_cond_norm(cond_raw), dtype=torch.float32, device=device)
    pred, resid, pred_next_y, base_next = _dynamics_residual_from_state_aux(
        state=state_t,
        aux=aux_t,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        cond_norm=cond_norm,
        base=base,
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    loss = torch.mean(resid.square())
    grad_state, grad_aux = torch.autograd.grad(loss, (state_t, aux_t), retain_graph=False, allow_unused=True)
    grad_state_norm = float(torch.linalg.norm(grad_state.detach()).cpu().item()) if grad_state is not None else 0.0
    grad_aux_norm = float(torch.linalg.norm(grad_aux.detach()).cpu().item()) if grad_aux is not None else 0.0
    frame_rms = torch.sqrt(torch.mean(resid.detach().square(), dim=-1)).reshape(-1)
    frame_rms_np = frame_rms.cpu().numpy()
    del true_root_vel
    return {
        "n": float(len(idxs)),
        "gt_self_anchored_dynamics_loss": 0.0,
        "gt_zero_residual_loss": float(loss.detach().cpu().item()),
        "residual_rms_scaled": float(torch.sqrt(torch.mean(resid.detach().square())).cpu().item()),
        "residual_frame_rms_scaled_p50": float(np.percentile(frame_rms_np, 50)) if frame_rms_np.size else 0.0,
        "residual_frame_rms_scaled_p95": float(np.percentile(frame_rms_np, 95)) if frame_rms_np.size else 0.0,
        "residual_frame_rms_scaled_max": float(np.max(frame_rms_np)) if frame_rms_np.size else 0.0,
        "residual_pose_rms_raw": float(
            torch.sqrt(torch.mean((pred_next_y[..., : POSE_SLICE.stop] - base_next[..., : POSE_SLICE.stop]).detach().square()))
            .cpu()
            .item()
        ),
        "residual_rootvel_rms_raw": float(
            torch.sqrt(torch.mean((pred_next_y[..., POSE_SLICE.stop :] - base_next[..., POSE_SLICE.stop :]).detach().square()))
            .cpu()
            .item()
        ),
        "output_grad_state_norm": grad_state_norm,
        "output_grad_aux_norm": grad_aux_norm,
    }


def _eval_result_rows(
    *,
    result: Mapping[str, Any],
    items: Sequence[DecoderItem],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    split_result = {
        "split": result["stage"],
        "split_kind": "dynamics_train_fit_ladder",
        "train_idx": tuple(result["idxs"]),
        "test_idx": tuple(result["idxs"]),
        "train_pred_raw": result["pred_raw"],
        "test_pred_raw": result["pred_raw"],
        "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
        "command_align_root_vel": bool(args.command_align_root_vel),
    }
    rows = _evaluate_split_predictions(
        split_result=split_result,
        all_items=items,
        horizon=int(args.horizon),
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        variant=str(result["arm"]),
        calibration_domain="reconstructed_state281",
    )
    for row in rows:
        row["arm"] = result["arm"]
        row["stage"] = result["stage"]
    return [r for r in rows if r.get("partition") == "train"]


def _oracle_event_masks(item: DecoderItem, *, horizon: int, event_window: int) -> Dict[str, Any]:
    h = int(horizon)
    contact = np.asarray(item.seq["contact"], dtype=np.float32).reshape(h, 2)
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    if len(labels) != h:
        labels = labels[:h] + ["unknown"] * max(0, h - len(labels))
    contact_bin = contact > 0.5
    switch_frames: List[int] = []
    for t in range(1, h):
        if labels[t] != labels[t - 1] or bool(np.any(contact_bin[t] != contact_bin[t - 1])):
            switch_frames.append(int(t))
    frame_mask = np.zeros((h,), dtype=bool)
    radius = max(0, int(event_window))
    for t in switch_frames:
        lo = max(0, t - radius)
        hi = min(h, t + radius + 1)
        frame_mask[lo:hi] = True
    step_mask = np.zeros((max(0, h - 1),), dtype=bool)
    if h > 1:
        step_mask = frame_mask[1:] | frame_mask[:-1]
    return {
        "labels": labels,
        "switch_frames": switch_frames,
        "frame_mask": frame_mask,
        "step_mask": step_mask,
    }


def _local_metric_values(seq: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_SLICE.stop - POSE_SLICE.start)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    return {
        "pose_step_l2_p95": _step_pose_l2(rot6d),
        "contact_step_l2_p95": _step_l2(contact),
        "angvel_step_rms_p95": _step_angvel_rms(bone_angvel),
        "heading_error_p95_rad": _heading_error_rad(root_vel, cond_dir),
    }


def _local_channel_values(seq: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_SLICE.stop - POSE_SLICE.start)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    out: Dict[str, np.ndarray] = {}
    out["pose_step_l2_p95"] = np.diff(rot6d, axis=0).astype(np.float64) if rot6d.shape[0] > 1 else np.zeros((0, rot6d.shape[1]))
    out["angvel_step_rms_p95"] = (
        np.diff(bone_angvel, axis=0).astype(np.float64) if bone_angvel.shape[0] > 1 else np.zeros((0, bone_angvel.shape[1]))
    )
    out["contact_step_l2_p95"] = (
        np.diff(contact, axis=0).astype(np.float64) if contact.shape[0] > 1 else np.zeros((0, contact.shape[1]))
    )
    return out


def _surrogate_frame_values(
    *,
    pred_seq: Mapping[str, np.ndarray],
    gt_seq: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    pred_rot = np.asarray(pred_seq["rot6d"], dtype=np.float64)
    gt_rot = np.asarray(gt_seq["rot6d"], dtype=np.float64)
    pred_aux = np.asarray(pred_seq["bone_angvel"], dtype=np.float64)
    gt_aux = np.asarray(gt_seq["bone_angvel"], dtype=np.float64)
    pred_contact = np.asarray(pred_seq["contact"], dtype=np.float64)
    gt_contact = np.asarray(gt_seq["contact"], dtype=np.float64)
    ego = np.asarray(pred_seq.get("ego", np.zeros((pred_contact.shape[0], 2), dtype=np.float64)), dtype=np.float64)
    if "ego" not in pred_seq:
        ego = np.zeros((pred_contact.shape[0], 2), dtype=np.float64)
    lateral = ego[:, 1] if ego.size else np.zeros((pred_contact.shape[0],), dtype=np.float64)
    speed = np.linalg.norm(ego.reshape(-1, 2), axis=1) if ego.size else np.zeros_like(lateral)
    command = (lateral / np.maximum(speed, EPS)) ** 2 + lateral**2
    return {
        "pose_step_l2_p95": (
            np.mean((np.diff(pred_rot, axis=0) - np.diff(gt_rot, axis=0)) ** 2, axis=1)
            if pred_rot.shape[0] > 1
            else np.zeros((0,), dtype=np.float64)
        ),
        "angvel_step_rms_p95": (
            np.mean((np.diff(pred_aux, axis=0) - np.diff(gt_aux, axis=0)) ** 2, axis=1)
            if pred_aux.shape[0] > 1
            else np.zeros((0,), dtype=np.float64)
        ),
        "contact_step_l2_p95": np.mean((pred_contact - gt_contact) ** 2, axis=1),
        "heading_error_p95_rad": command,
    }


def _seq_with_ego(
    item: DecoderItem,
    state: np.ndarray,
    aux: np.ndarray,
    *,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, np.ndarray]:
    seq = _seq_from_prediction(
        item,
        state,
        aux,
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
    )
    seq["ego"] = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)[:, EGO_VEL_SLICE].astype(np.float32, copy=False)
    return seq


def _dynamics_frame_witness(
    *,
    state: np.ndarray,
    aux: np.ndarray,
    item: DecoderItem,
    idxs: Sequence[int],
    items: Sequence[DecoderItem],
    base: BaseOperator,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    state_t = torch.as_tensor(state.reshape(1, int(args.horizon), STATE_DIM), dtype=torch.float32, device=device)
    aux_t = torch.as_tensor(aux.reshape(1, int(args.horizon), ANGVEL_DIM), dtype=torch.float32, device=device)
    true_state_t = torch.as_tensor(np.asarray(item.seq["state281"], dtype=np.float32).reshape(1, int(args.horizon), STATE_DIM), device=device)
    true_aux_t = torch.as_tensor(np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(1, int(args.horizon), ANGVEL_DIM), device=device)
    true_root_pos = torch.as_tensor(np.asarray(item.seq["root_pos"], dtype=np.float32).reshape(1, int(args.horizon), 3), device=device)
    true_cond_dir = torch.as_tensor(np.asarray(item.seq["cond_dir"], dtype=np.float32).reshape(1, int(args.horizon), 2), device=device)
    true_contact = torch.as_tensor(np.asarray(item.seq["contact"], dtype=np.float32).reshape(1, int(args.horizon), 2), device=device)
    cond_raw = _stack_cond_raw(base, items, idxs, int(args.horizon))
    cond_norm = torch.as_tensor(_robust_cond_norm(cond_raw), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, resid, _, _ = _dynamics_residual_from_state_aux(
            state=state_t,
            aux=aux_t,
            true_root_pos=true_root_pos,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            cond_norm=cond_norm,
            base=base,
            command_align_root_vel=bool(args.command_align_root_vel),
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        )
        _, gt_resid, _, _ = _dynamics_residual_from_state_aux(
            state=true_state_t,
            aux=true_aux_t,
            true_root_pos=true_root_pos,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            cond_norm=cond_norm,
            base=base,
            command_align_root_vel=bool(args.command_align_root_vel),
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        )
    zero = torch.sqrt(torch.mean(resid.square(), dim=-1)).reshape(-1).cpu().numpy()
    gt_zero = torch.sqrt(torch.mean(gt_resid.square(), dim=-1)).reshape(-1).cpu().numpy()
    anchored = torch.sqrt(torch.mean((resid - gt_resid).square(), dim=-1)).reshape(-1).cpu().numpy()
    return {
        "dynamics_zero_resid_frame_rms_scaled": zero.astype(np.float64),
        "gt_dynamics_zero_resid_frame_rms_scaled": gt_zero.astype(np.float64),
        "dynamics_anchor_frame_rms_scaled": anchored.astype(np.float64),
    }


def _set_ratio(lhs: Sequence[int], rhs: Sequence[int]) -> float:
    lhs_set = {int(x) for x in lhs}
    if not lhs_set:
        return 0.0
    rhs_set = {int(x) for x in rhs}
    return float(len(lhs_set & rhs_set) / len(lhs_set))


def _metric_bucket(metric: str, over_count: int, boundary_ratio: float, max_margin: float) -> str:
    if int(over_count) <= 0:
        return "pass"
    if float(boundary_ratio) >= 0.80:
        return "b_event_aware_band"
    if abs(float(max_margin)) <= 1e-4:
        return "a_surrogate_tail"
    if metric == "pose_step_l2_p95":
        return "c_representation_ceiling_candidate"
    return "a_surrogate_tail"


def _metric_next_step(bucket: str) -> str:
    if bucket == "b_event_aware_band":
        return "改 event-aware band，不动模型"
    if bucket == "c_representation_ceiling_candidate":
        return "先扩展非边界定位/GT 对照，再决定是否回查表征"
    if bucket == "pass":
        return "无 hard miss"
    return "换 tail-aware surrogate(p95/top-k)，不动表征"


def _top_channel(delta: np.ndarray) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None, None, None
    idx = int(np.argmax(np.abs(arr)))
    return idx, float(arr[idx]), float(abs(arr[idx]))


def _localize_result(
    *,
    result: Mapping[str, Any],
    items: Sequence[DecoderItem],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    base: BaseOperator,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    idxs = [int(x) for x in result["idxs"]]
    if len(idxs) != 1:
        raise RuntimeError(f"localization expects one window, got {len(idxs)}")
    item = items[idxs[0]]
    h = int(args.horizon)
    pred_state, pred_aux = _reshape_state_aux(np.asarray(result["pred_raw"], dtype=np.float32), h)
    gt_state, gt_aux = _reshape_state_aux(np.asarray(result["true_raw"], dtype=np.float32), h)
    pred_seq = _seq_with_ego(
        item,
        pred_state[0],
        pred_aux[0],
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    gt_seq = _seq_with_ego(
        item,
        gt_state[0],
        gt_aux[0],
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    event = _oracle_event_masks(item, horizon=h, event_window=int(args.event_window))
    pred_vals = _local_metric_values(pred_seq)
    gt_vals = _local_metric_values(gt_seq)
    surrogate_vals = _surrogate_frame_values(pred_seq=pred_seq, gt_seq=gt_seq)
    channel_vals = _local_channel_values(pred_seq)
    gt_channel_vals = _local_channel_values(gt_seq)
    dyn = _dynamics_frame_witness(
        state=pred_state[0],
        aux=pred_aux[0],
        item=item,
        idxs=idxs,
        items=items,
        base=base,
        args=args,
        device=device,
    )
    zero = dyn["dynamics_zero_resid_frame_rms_scaled"]
    zero_high_thr = _safe_percentile(zero, 95.0)
    zero_high_frames = [int(i + 1) for i, v in enumerate(zero) if float(v) >= zero_high_thr - EPS]

    per_frame_rows: List[Dict[str, Any]] = []
    for t in range(h):
        rec: Dict[str, Any] = {
            "stage": result["stage"],
            "arm": result["arm"],
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "local_frame": int(t),
            "global_frame": int(item.start + t),
            "transition_from_prev": bool(t > 0),
            "oracle_support_label": event["labels"][t],
            "oracle_contact_r": float(np.asarray(item.seq["contact"], dtype=np.float32)[t, 0]),
            "oracle_contact_l": float(np.asarray(item.seq["contact"], dtype=np.float32)[t, 1]),
            "event_switch_at_frame": bool(t in set(event["switch_frames"])),
            "is_event_boundary_frame": bool(event["frame_mask"][t]),
            "is_event_boundary_transition_from_prev": bool(t > 0 and event["step_mask"][t - 1]),
            "dynamics_zero_resid_frame_rms_scaled": float(zero[t - 1]) if t > 0 and t - 1 < zero.shape[0] else None,
            "gt_dynamics_zero_resid_frame_rms_scaled": float(dyn["gt_dynamics_zero_resid_frame_rms_scaled"][t - 1])
            if t > 0 and t - 1 < zero.shape[0]
            else None,
            "dynamics_anchor_frame_rms_scaled": float(dyn["dynamics_anchor_frame_rms_scaled"][t - 1])
            if t > 0 and t - 1 < zero.shape[0]
            else None,
            "dynamics_zero_resid_high_p95": bool(t in zero_high_frames),
            "dynamics_zero_resid_high_p95_threshold": float(zero_high_thr),
        }
        for metric, band_key, kind, _family, _surrogate_key in LOCALIZATION_METRICS:
            band = float(baseline_bands[item.clip][band_key])
            vals = pred_vals[metric]
            gvals = gt_vals[metric]
            svals = surrogate_vals[metric]
            if kind == "step":
                value = float(vals[t - 1]) if t > 0 and t - 1 < vals.shape[0] else None
                gt_value = float(gvals[t - 1]) if t > 0 and t - 1 < gvals.shape[0] else None
                surrogate_value = float(svals[t - 1]) if t > 0 and t - 1 < svals.shape[0] else None
            else:
                value = float(vals[t]) if t < vals.shape[0] else None
                gt_value = float(gvals[t]) if t < gvals.shape[0] else None
                surrogate_value = float(svals[t]) if t < svals.shape[0] else None
            margin = None if value is None else float(value - band)
            rec[f"{metric}_value"] = value
            rec[f"{metric}_gt_value"] = gt_value
            rec[f"{metric}_band"] = band
            rec[f"{metric}_margin"] = margin
            rec[f"{metric}_exceeds_band"] = bool(margin is not None and margin > EPS)
            rec[f"{metric}_surrogate_frame_value"] = surrogate_value
        per_frame_rows.append(rec)

    channel_rows: List[Dict[str, Any]] = []
    for metric in ("pose_step_l2_p95", "angvel_step_rms_p95", "contact_step_l2_p95"):
        arr = np.asarray(channel_vals[metric], dtype=np.float64)
        gt_arr = np.asarray(gt_channel_vals[metric], dtype=np.float64)
        for step_i in range(arr.shape[0]):
            t = int(step_i + 1)
            top_idx, top_val, top_abs = _top_channel(arr[step_i])
            gt_top_idx, gt_top_val, gt_top_abs = _top_channel(gt_arr[step_i])
            for ch in range(arr.shape[1]):
                channel_rows.append(
                    {
                        "stage": result["stage"],
                        "arm": result["arm"],
                        "clip": item.clip,
                        "start": int(item.start),
                        "local_frame": t,
                        "transition_from_prev": True,
                        "metric": metric,
                        "channel": int(ch),
                        "channel_delta": float(arr[step_i, ch]),
                        "channel_abs_delta": float(abs(arr[step_i, ch])),
                        "gt_channel_delta": float(gt_arr[step_i, ch]) if ch < gt_arr.shape[1] else None,
                        "frame_value": float(pred_vals[metric][step_i]),
                        "band": float(baseline_bands[item.clip][dict((m[0], m[1]) for m in LOCALIZATION_METRICS)[metric]]),
                        "frame_exceeds_band": bool(float(pred_vals[metric][step_i]) > float(baseline_bands[item.clip][dict((m[0], m[1]) for m in LOCALIZATION_METRICS)[metric]]) + EPS),
                        "is_event_boundary_transition_from_prev": bool(event["step_mask"][step_i]),
                        "is_top_abs_channel_for_frame": bool(top_idx == ch),
                        "top_channel": top_idx,
                        "top_channel_delta": top_val,
                        "top_channel_abs_delta": top_abs,
                        "gt_top_channel": gt_top_idx,
                        "gt_top_channel_delta": gt_top_val,
                        "gt_top_channel_abs_delta": gt_top_abs,
                    }
                )

    metric_summaries: List[Dict[str, Any]] = []
    over_frames_by_metric: Dict[str, List[int]] = {}
    for metric, band_key, kind, family, surrogate_key in LOCALIZATION_METRICS:
        vals = pred_vals[metric]
        gvals = gt_vals[metric]
        band = float(baseline_bands[item.clip][band_key])
        gate = _safe_percentile(vals, 95.0)
        gt_gate = _safe_percentile(gvals, 95.0)
        frame_offset = 1 if kind == "step" else 0
        over_frames = [int(i + frame_offset) for i, v in enumerate(vals) if float(v) > band + EPS]
        boundary_mask = event["step_mask"] if kind == "step" else event["frame_mask"]
        boundary_frames = [int(i + frame_offset) for i, v in enumerate(boundary_mask) if bool(v)]
        margins = np.asarray(vals, dtype=np.float64) - band
        max_margin = float(np.max(margins)) if margins.size else 0.0
        bucket = _metric_bucket(metric, len(over_frames), _set_ratio(over_frames, boundary_frames), max_margin)
        over_frames_by_metric[metric] = over_frames
        metric_summaries.append(
            {
                "metric": metric,
                "family": family,
                "kind": kind,
                "gate_p95": float(gate),
                "gt_same_window_p95": float(gt_gate),
                "band": band,
                "gate_margin": float(gate - band),
                "over_frame_count": int(len(over_frames)),
                "over_frames": over_frames,
                "max_sample_margin": max_margin,
                "mean_sample_margin": float(np.mean(margins)) if margins.size else 0.0,
                "event_boundary_overlap_ratio": _set_ratio(over_frames, boundary_frames),
                "dynamics_zero_high_overlap_ratio": _set_ratio(over_frames, zero_high_frames),
                "surrogate_key": surrogate_key,
                "surrogate_final_loss": float((result.get("final_train_objective_terms", {}) or {}).get(surrogate_key, 0.0)),
                "surrogate_frame_mean": float(np.mean(surrogate_vals[metric])) if surrogate_vals[metric].size else 0.0,
                "bucket": bucket,
                "next_step": _metric_next_step(bucket),
            }
        )

    collocation = {
        "pose_over_frames": over_frames_by_metric.get("pose_step_l2_p95", []),
        "angvel_over_frames": over_frames_by_metric.get("angvel_step_rms_p95", []),
        "contact_over_frames": over_frames_by_metric.get("contact_step_l2_p95", []),
        "heading_over_frames": over_frames_by_metric.get("heading_error_p95_rad", []),
        "dynamics_zero_resid_high_frames": zero_high_frames,
    }
    collocation.update(
        {
            "pose_over__angvel_over_ratio": _set_ratio(collocation["pose_over_frames"], collocation["angvel_over_frames"]),
            "angvel_over__pose_over_ratio": _set_ratio(collocation["angvel_over_frames"], collocation["pose_over_frames"]),
            "pose_over__dynamics_zero_high_ratio": _set_ratio(collocation["pose_over_frames"], zero_high_frames),
            "angvel_over__dynamics_zero_high_ratio": _set_ratio(collocation["angvel_over_frames"], zero_high_frames),
        }
    )
    return {
        "window": {"clip": item.clip, "start": int(item.start), "end": int(item.end), "horizon": h},
        "event": {
            "event_window": int(args.event_window),
            "switch_frames": [int(x) for x in event["switch_frames"]],
            "boundary_frames": [int(i) for i, v in enumerate(event["frame_mask"]) if bool(v)],
        },
        "metric_summaries": metric_summaries,
        "collocation": collocation,
        "per_frame_rows": per_frame_rows,
        "per_channel_rows": channel_rows,
    }


def _full_gt_event_control(
    *,
    items: Sequence[DecoderItem],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, band_key, kind, family, _surrogate_key in LOCALIZATION_METRICS:
        total = 0
        over = 0
        over_boundary = 0
        max_margin = -float("inf")
        for item in items:
            h = int(args.horizon)
            raw = np.asarray(item.seq["state281"], dtype=np.float32).reshape(1, h, STATE_DIM)
            aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(1, h, ANGVEL_DIM)
            seq = _seq_with_ego(
                item,
                raw[0],
                aux[0],
                oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
                command_align_root_vel=bool(args.command_align_root_vel),
            )
            vals = _local_metric_values(seq)[metric]
            event = _oracle_event_masks(item, horizon=h, event_window=int(args.event_window))
            mask = event["step_mask"] if kind == "step" else event["frame_mask"]
            band = float(baseline_bands[item.clip][band_key])
            for i, v in enumerate(vals):
                total += 1
                margin = float(v) - band
                max_margin = max(max_margin, margin)
                if margin > EPS:
                    over += 1
                    if bool(mask[i]):
                        over_boundary += 1
        rows.append(
            {
                "metric": metric,
                "family": family,
                "sample_count": int(total),
                "gt_sample_over_band_count": int(over),
                "gt_sample_over_band_rate": float(over / max(total, 1)),
                "gt_over_band_event_boundary_ratio": float(over_boundary / max(over, 1)) if over else 0.0,
                "gt_max_sample_margin": 0.0 if max_margin == -float("inf") else float(max_margin),
            }
        )
    return rows


def _write_generic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _term_summary(step_rows: Sequence[Mapping[str, Any]], terms: Sequence[str]) -> Dict[str, Any]:
    if not step_rows:
        return {"n": 0}
    first = step_rows[0]
    last = step_rows[-1]
    out: Dict[str, Any] = {
        "n": int(len(step_rows)),
        "initial_total_loss": float(first.get("total_loss", 0.0)),
        "final_total_loss": float(last.get("total_loss", 0.0)),
    }
    for key in terms:
        i = float(first.get(f"{key}_loss", 0.0))
        f = float(last.get(f"{key}_loss", 0.0))
        out[f"{key}_initial_loss"] = i
        out[f"{key}_final_loss"] = f
        out[f"{key}_loss_ratio"] = float(f / max(i, EPS))
        out[f"{key}_weighted_grad_norm_initial"] = float(first.get(f"{key}_weighted_grad_norm", 0.0))
        out[f"{key}_weighted_grad_norm_final"] = float(last.get(f"{key}_weighted_grad_norm", 0.0))
    for ch in CHANNELS:
        out[f"channel_{ch}_grad_norm_final"] = float(last.get(f"channel_{ch}_grad_norm", 0.0))
    for a, b in PAIR_KEYS:
        vals = [float(r.get(f"channel_cos_{a}_vs_{b}", 0.0)) for r in step_rows]
        out[f"channel_cos_{a}_vs_{b}_final"] = float(vals[-1]) if vals else 0.0
        out[f"channel_cos_{a}_vs_{b}_min"] = float(np.min(vals)) if vals else 0.0
        out[f"channel_cos_{a}_vs_{b}_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


def _classify(
    *,
    guard: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    term_summary: Mapping[str, Any],
    pass_rate_threshold: float,
) -> str:
    if not bool(guard.get("passed", False)):
        return "guard_path_identity_failed"
    acc = float(acceptance.get("acceptance_proxy_pass_rate", 0.0))
    if acc >= float(pass_rate_threshold):
        return "train_fit_acceptance_pass"
    dyn_anchor_final = float(term_summary.get("dynamics_consistency_final_loss", 0.0))
    dyn_zero_final = float(term_summary.get("dynamics_zero_residual_witness_final_loss", 0.0))
    regime_ratio = float(term_summary.get("regime_reaching_loss_ratio", 1.0))
    if dyn_zero_final < 0.0025 and dyn_anchor_final > 0.01:
        return "train_fit_fail_zero_residual_lowpass_basin_witness"
    if dyn_anchor_final < 0.01 and regime_ratio > 0.5:
        return "train_fit_fail_operator_consistency_blocks_regime_reaching"
    if dyn_anchor_final < 0.01:
        return "train_fit_fail_gt_residual_anchor_acceptance_fail"
    return "train_fit_fail_no_binding_signature_yet"


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "arm",
        "variant",
        "partition",
        "clip",
        "start",
        "end",
        "acceptance_proxy_pass",
        "failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
        "oracle_support_token_accuracy",
        "oracle_contact_mse",
        "foot_slip_p95_to_band_ratio",
        "old_pop_safe_diagnostic",
        "foot_slip_p95_mps",
        "heading_error_p95_rad",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rec = {k: row.get(k) for k in fields}
            metrics = row.get("metrics", {}) or {}
            rec["foot_slip_p95_mps"] = metrics.get("foot_slip_p95_mps")
            rec["heading_error_p95_rad"] = metrics.get("heading_error_p95_rad")
            writer.writerow(rec)


def _write_step_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = ["stage", "arm", "epoch", "total_loss"]
    term_keys = []
    for row in rows:
        for key in row:
            if key.endswith("_loss") and key not in ("total_loss",):
                term_keys.append(key[:-5])
    for key in sorted(set(term_keys)):
        fields.extend([f"{key}_loss", f"{key}_weight", f"{key}_grad_norm", f"{key}_weighted_grad_norm"])
    for ch in CHANNELS:
        fields.append(f"channel_{ch}_grad_norm")
    for a, b in PAIR_KEYS:
        fields.append(f"channel_cos_{a}_vs_{b}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GT-Residual-Anchored Dynamics-Consistency Train-Fit Ladder",
        "",
        "Debug-only ladder. No production Trainer/runtime/gate training or checkpoint mutation.",
        "",
        "## Guard",
        "",
    ]
    guard = payload["guard_path_identity"]
    lines.extend(
        [
            f"- passed: `{guard.get('passed')}`",
            f"- n: `{guard.get('n')}`",
            f"- max_abs_seq_delta: `{guard.get('max_abs_seq_delta')}`",
            f"- reconstructed_gt_acceptance_rate: `{_fmt(guard.get('reconstructed_gt_acceptance_rate', 0.0))}`",
            f"- decoder_path_from_gt_raw_acceptance_rate: `{_fmt(guard.get('decoder_path_from_gt_raw_acceptance_rate', 0.0))}`",
            "",
            "## Stage Results",
            "",
            "| stage | arm | windows | accept | support | side | command | pose | dyn anchor | zero-resid witness | min pose-root cos | diagnosis |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rec in payload.get("stage_results", []):
        acc = rec.get("train_acceptance", {}) or {}
        ts = rec.get("step_log_summary", {}) or {}
        lines.append(
            f"| {rec.get('stage')} | {rec.get('arm')} | {rec.get('train_n')} | "
            f"{_fmt(acc.get('acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_honesty_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_side_correctness_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('command_response_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('pose_continuity_pass_rate', 0.0))} | "
            f"{_fmt(ts.get('dynamics_consistency_final_loss', 0.0), 6)} | "
            f"{_fmt(ts.get('dynamics_zero_residual_witness_final_loss', 0.0), 6)} | "
            f"{_fmt(ts.get('channel_cos_pose_vs_rootvel_min', 0.0), 4)} | "
            f"{rec.get('diagnosis')} |"
        )
    decision = payload.get("decision", {})
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- stopped_at: `{decision.get('stopped_at')}`",
            f"- failure_signature: `{decision.get('failure_signature')}`",
            f"- ran_8window: `{decision.get('ran_8window')}`",
            f"- ran_full_188: `{decision.get('ran_full_188')}`",
            f"- interpretation: {decision.get('interpretation')}",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
            f"- step log csv: `{payload['artifacts']['step_log_csv']}`",
        ]
    )
    _dump_md(path, lines)


def _write_localization_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    loc = payload["localization"]
    window = loc["window"]
    lines = [
        "# Dynamics-Consistency Hard-Band Localization",
        "",
        "Debug-only read-only localization over the reproduced one-window decoder output. No production Trainer/runtime/gate training or checkpoint mutation.",
        "",
        "## Window",
        "",
        f"- clip/start/end: `{window['clip']}:{window['start']}-{window['end']}`",
        "- decoder output: `state281 [1,16,281] float32 cpu`; aux `bone_angvel [1,16,138] float32 cpu`",
        "- base operator witness input: `X_norm [1,15,419] float32 cpu`; residual frame RMS over `Y [1,15,278] float32 cpu`",
        f"- event switch frames: `{loc['event']['switch_frames']}`; boundary frames (±{loc['event']['event_window']}): `{loc['event']['boundary_frames']}`",
        "",
        "## Metric Localization",
        "",
        "| metric | family | p95 | band | p95 margin | over frames | max sample margin | boundary overlap | dyn high overlap | bucket | next |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for rec in loc["metric_summaries"]:
        lines.append(
            f"| `{rec['metric']}` | {rec['family']} | {_fmt(rec['gate_p95'], 8)} | {_fmt(rec['band'], 8)} | "
            f"{_fmt(rec['gate_margin'], 8)} | `{rec['over_frames']}` | {_fmt(rec['max_sample_margin'], 8)} | "
            f"{_fmt(rec['event_boundary_overlap_ratio'], 4)} | {_fmt(rec['dynamics_zero_high_overlap_ratio'], 4)} | "
            f"`{rec['bucket']}` | {rec['next_step']} |"
        )
    colloc = loc["collocation"]
    lines.extend(
        [
            "",
            "## Co-Location",
            "",
            f"- pose over frames: `{colloc['pose_over_frames']}`",
            f"- angvel over frames: `{colloc['angvel_over_frames']}`",
            f"- dynamics zero-residual high frames (pred frame RMS p95): `{colloc['dynamics_zero_resid_high_frames']}`",
            f"- pose-over ∩ angvel-over / pose-over: `{_fmt(colloc['pose_over__angvel_over_ratio'], 4)}`",
            f"- angvel-over ∩ pose-over / angvel-over: `{_fmt(colloc['angvel_over__pose_over_ratio'], 4)}`",
            f"- pose-over ∩ dynamics-high / pose-over: `{_fmt(colloc['pose_over__dynamics_zero_high_ratio'], 4)}`",
            f"- angvel-over ∩ dynamics-high / angvel-over: `{_fmt(colloc['angvel_over__dynamics_zero_high_ratio'], 4)}`",
            "",
            "## Surrogate vs Gate",
            "",
            "| metric | surrogate key | final surrogate loss | per-frame surrogate mean | gate p95 | band | gate margin |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rec in loc["metric_summaries"]:
        lines.append(
            f"| `{rec['metric']}` | `{rec['surrogate_key']}` | {_fmt(rec['surrogate_final_loss'], 10)} | "
            f"{_fmt(rec['surrogate_frame_mean'], 10)} | {_fmt(rec['gate_p95'], 8)} | "
            f"{_fmt(rec['band'], 8)} | {_fmt(rec['gate_margin'], 8)} |"
        )
    lines.extend(
        [
            "",
            "## Full-188 GT Control",
            "",
            "| metric | GT samples | GT sample > band | rate | event-boundary ratio among GT over-band samples | max sample margin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rec in payload.get("full_gt_control", []):
        lines.append(
            f"| `{rec['metric']}` | {rec['sample_count']} | {rec['gt_sample_over_band_count']} | "
            f"{_fmt(rec['gt_sample_over_band_rate'], 6)} | {_fmt(rec['gt_over_band_event_boundary_ratio'], 4)} | "
            f"{_fmt(rec['gt_max_sample_margin'], 8)} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- per-frame csv: `{payload['artifacts']['per_frame_csv']}`",
            f"- per-channel csv: `{payload['artifacts']['per_channel_csv']}`",
        ]
    )
    _dump_md(path, lines)


def run_localization(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")

    base = _build_base_operator(args, Path(args.npz_root), device)
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))

    start = int(args.overfit_start_index)
    one_idxs = tuple(range(start, start + 1))
    guard_idxs = tuple(range(len(main_items))) if bool(args.guard_all_windows) else one_idxs
    guard = _guard_path_identity(
        items=main_items,
        idxs=guard_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=float(args.pass_rate_threshold),
    )
    if not bool(guard.get("passed", False)):
        raise RuntimeError(f"guard_path_identity failed: {guard.get('reason')}")

    result = _train_stage_arm(
        stage="one_window",
        arm="dynamics_consistency",
        idxs=one_idxs,
        items=main_items,
        base=base,
        skeleton=skeleton,
        args=args,
        device=device,
    )
    pred_rows = _eval_result_rows(
        result=result,
        items=main_items,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        args=args,
    )
    localization = _localize_result(
        result=result,
        items=main_items,
        baseline_bands=reconstructed_baseline_bands,
        base=base,
        args=args,
        device=device,
    )
    full_gt_control = _full_gt_event_control(items=main_items, baseline_bands=reconstructed_baseline_bands, args=args)
    artifacts = {
        "summary_json": str(args.localization_out_dir / "localization_summary.json"),
        "summary_md": str(args.localization_out_dir / "summary.md"),
        "per_frame_csv": str(args.localization_out_dir / "per_frame.csv"),
        "per_channel_csv": str(args.localization_out_dir / "per_channel.csv"),
    }
    return {
        "task": "dynamics_consistency_hard_band_localization",
        "scope": "debug-only per-frame/per-channel localization; no production Trainer/runtime/gate training; no checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "checkpoint": str(args.checkpoint),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": "cpu",
            "dtype": "float32",
            "event_window": int(args.event_window),
            "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
            "command_align_root_vel": bool(args.command_align_root_vel),
            "note": "localization rehydrates the unsaved one-window debug decoder output with the same deterministic seed/objective; it does not train production code or mutate checkpoints",
        },
        "input_output_contract": {
            "decoder_input": {"shape": "[1,input_dim]", "dtype": "float32", "device": "cpu"},
            "middle_state_output": {"shape": [1, int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux_output": {"shape": [1, int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "per_frame_rows": {"shape": [int(args.horizon), "columns"], "dtype": "csv scalars", "device": "n/a"},
            "per_channel_rows": {"shape": [15 * (276 + 138 + 2), "columns"], "dtype": "csv scalars", "device": "n/a"},
        },
        "guard_path_identity": guard,
        "window_acceptance_row": pred_rows[0] if pred_rows else {},
        "localization": {k: v for k, v in localization.items() if k not in ("per_frame_rows", "per_channel_rows")},
        "full_gt_control": full_gt_control,
        "hard_constraint_confirmations": {
            "debug_only": True,
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "forwarded_base_event_motion_model_for_witness": True,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "changed_loss_weights_or_model": False,
            "used_residual_head": False,
            "used_diffusion_or_sampling": False,
            "yaw_or_cond_dir_prediction_target": False,
            "attached_to_runtime": False,
        },
        "artifacts": artifacts,
        "_per_frame_rows_for_csv": localization["per_frame_rows"],
        "_per_channel_rows_for_csv": localization["per_channel_rows"],
    }


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty float list")
    return out


def _adjusted_four_metric_status(localization: Mapping[str, Any], *, heading_tolerance_rad: float) -> Dict[str, Any]:
    by_metric = {str(r["metric"]): r for r in localization["metric_summaries"]}

    def rec(metric: str) -> Mapping[str, Any]:
        return by_metric.get(metric, {})

    pose = rec("pose_step_l2_p95")
    contact = rec("contact_step_l2_p95")
    angvel = rec("angvel_step_rms_p95")
    heading = rec("heading_error_p95_rad")
    pose_ok = float(pose.get("gate_p95", float("inf"))) <= float(pose.get("band", 0.0)) + EPS
    contact_event_ok = int(contact.get("over_frame_count", 0)) == 0 or float(contact.get("event_boundary_overlap_ratio", 0.0)) >= 0.999
    angvel_event_ok = int(angvel.get("over_frame_count", 0)) == 0 or float(angvel.get("event_boundary_overlap_ratio", 0.0)) >= 0.999
    heading_tol_ok = float(heading.get("gate_margin", float("inf"))) <= float(heading_tolerance_rad) + EPS
    return {
        "pose_ok": bool(pose_ok),
        "contact_event_aware_ok": bool(contact_event_ok),
        "angvel_event_aware_ok": bool(angvel_event_ok),
        "heading_tolerance_ok": bool(heading_tol_ok),
        "heading_tolerance_rad": float(heading_tolerance_rad),
        "adjusted_four_metric_pass": bool(pose_ok and contact_event_ok and angvel_event_ok and heading_tol_ok),
    }


def _save_sweep_artifacts(
    *,
    out_dir: Path,
    label: str,
    result: Mapping[str, Any],
    item: DecoderItem,
    localization: Mapping[str, Any],
) -> Dict[str, str]:
    safe = str(label).replace("/", "_").replace(" ", "_")
    npz_path = out_dir / f"{safe}_pred_raw.npz"
    weights_path = out_dir / f"{safe}_decoder_state.pt"
    np.savez(
        npz_path,
        pred_raw=np.asarray(result["pred_raw"], dtype=np.float32),
        true_raw=np.asarray(result["true_raw"], dtype=np.float32),
        train_indices=np.asarray(result["idxs"], dtype=np.int64),
        clip=np.asarray([item.clip]),
        start=np.asarray([int(item.start)], dtype=np.int64),
        end=np.asarray([int(item.end)], dtype=np.int64),
    )
    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "x_scaler": result.get("x_scaler"),
            "y_scaler": result.get("y_scaler"),
            "idxs": [int(x) for x in result["idxs"]],
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "localization": {k: v for k, v in localization.items() if k not in ("per_frame_rows", "per_channel_rows")},
        },
        weights_path,
    )
    return {"pred_raw_npz": str(npz_path), "decoder_state_pt": str(weights_path)}


def _write_pose_sweep_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Pose-Step C1/C2 Sweep",
        "",
        "Debug-only one-window sweep. Event-aware contact/angvel and heading tolerance are used only as an isolated decision view; production gate/runtime/checkpoint are unchanged.",
        "",
        "## Surrogate Alignment",
        "",
        "- existing `pose_continuity_loss`: mean MSE between predicted and GT pose-step deltas over `rot6d [1,16,276] float32 cpu`",
        "- gate `pose_step_l2_p95`: p95 of raw adjacent-frame `rot6d` L2/sqrt(276)",
        "- conclusion: definitions are not identical, so the sweep includes a gate-aligned `pose_gate_margin_loss = mean/top-k relu(pose_step_l2 - band)^2` arm.",
        "",
        "## Results",
        "",
        "| mode | weight | pose p95 | band | margin | pose over frames | dyn anchor | endpoint | adjusted pass | decision flag | pred artifact |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| `{row['mode']}` | {_fmt(row['weight'], 4)} | {_fmt(row['pose_gate_p95'], 8)} | "
            f"{_fmt(row['pose_band'], 8)} | {_fmt(row['pose_gate_margin'], 8)} | `{row['pose_over_frames']}` | "
            f"{_fmt(row['dynamics_consistency_final_loss'], 8)} | {_fmt(row['endpoint_reaching_final_loss'], 8)} | "
            f"{_fmt(float(row['adjusted_four_metric_pass']), 0)} | `{row['decision_flag']}` | `{row['pred_raw_npz']}` |"
        )
    verdict = payload.get("verdict", {})
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- result: `{verdict.get('result')}`",
            f"- best row: `{verdict.get('best_label')}`",
            f"- interpretation: {verdict.get('interpretation')}",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
        ]
    )
    _dump_md(path, lines)


def run_pose_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")

    base = _build_base_operator(args, Path(args.npz_root), device)
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    start = int(args.overfit_start_index)
    one_idxs = tuple(range(start, start + 1))
    guard = _guard_path_identity(
        items=main_items,
        idxs=tuple(range(len(main_items))) if bool(args.guard_all_windows) else one_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=float(args.pass_rate_threshold),
    )
    if not bool(guard.get("passed", False)):
        raise RuntimeError(f"guard_path_identity failed: {guard.get('reason')}")

    weights = _parse_float_list(str(args.pose_sweep_weights))
    modes = [m.strip() for m in str(args.pose_sweep_modes).split(",") if m.strip()]
    item = main_items[one_idxs[0]]
    pose_gate_band = float(reconstructed_baseline_bands[item.clip]["pose_step_l2"])
    rows: List[Dict[str, Any]] = []
    args.pose_sweep_out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode not in {"mean", "gate"}:
            raise ValueError(f"unknown pose sweep mode: {mode}")
        for weight in weights:
            run_args = argparse.Namespace(**vars(args))
            gate_loss_weight = 0.0
            gate_topk = 0
            if mode == "mean":
                run_args.mechanism_pose_step_loss_weight = float(weight)
            else:
                run_args.mechanism_pose_step_loss_weight = float(args.pose_sweep_gate_mode_mean_weight)
                gate_loss_weight = float(weight)
                gate_topk = int(args.pose_gate_topk)
            label = f"{mode}_w{float(weight):g}"
            result = _train_stage_arm(
                stage="one_window",
                arm="dynamics_consistency",
                idxs=one_idxs,
                items=main_items,
                base=base,
                skeleton=skeleton,
                args=run_args,
                device=device,
                pose_gate_bands=np.asarray([pose_gate_band], dtype=np.float32) if mode == "gate" else None,
                pose_gate_loss_weight=gate_loss_weight,
                pose_gate_topk=gate_topk,
                instrument_step_log=False,
            )
            localization = _localize_result(
                result=result,
                items=main_items,
                baseline_bands=reconstructed_baseline_bands,
                base=base,
                args=run_args,
                device=device,
            )
            adjusted = _adjusted_four_metric_status(
                localization,
                heading_tolerance_rad=float(args.heading_tolerance_rad),
            )
            metric_map = {str(r["metric"]): r for r in localization["metric_summaries"]}
            pose_rec = metric_map["pose_step_l2_p95"]
            artifacts = _save_sweep_artifacts(
                out_dir=args.pose_sweep_out_dir,
                label=label,
                result=result,
                item=item,
                localization=localization,
            )
            terms = result.get("final_train_objective_terms", {}) or {}
            decision_flag = "pose_pass_adjusted_four_metric_pass" if bool(adjusted["adjusted_four_metric_pass"]) else "pose_still_fails"
            rows.append(
                {
                    "label": label,
                    "mode": mode,
                    "weight": float(weight),
                    "pose_gate_p95": float(pose_rec["gate_p95"]),
                    "pose_band": float(pose_rec["band"]),
                    "pose_gate_margin": float(pose_rec["gate_margin"]),
                    "pose_over_frame_count": int(pose_rec["over_frame_count"]),
                    "pose_over_frames": ",".join(str(x) for x in pose_rec["over_frames"]),
                    "pose_gate_margin_loss_final": float(terms.get("pose_gate_margin_loss", 0.0)),
                    "pose_continuity_loss_final": float(terms.get("pose_continuity_loss", 0.0)),
                    "dynamics_consistency_final_loss": float(terms.get("dynamics_consistency", 0.0)),
                    "dynamics_zero_residual_witness_final_loss": float(terms.get("dynamics_zero_residual_witness", 0.0)),
                    "endpoint_reaching_final_loss": float(terms.get("endpoint_reaching", 0.0)),
                    "contact_schedule_final_loss": float(terms.get("contact_schedule", 0.0)),
                    "command_compatibility_final_loss": float(terms.get("command_compatibility", 0.0)),
                    **adjusted,
                    "decision_flag": decision_flag,
                    **artifacts,
                }
            )

    pass_rows = [r for r in rows if bool(r.get("adjusted_four_metric_pass", False))]
    if pass_rows:
        best = min(pass_rows, key=lambda r: (float(r["dynamics_consistency_final_loss"]), float(r["endpoint_reaching_final_loss"])))
        result_name = "c1_loss_balance_supported"
        interpretation = (
            "pose_step can be brought under the hard band under the adjusted event/heading view; this supports loss-balance/surrogate alignment before any representation claim."
        )
    else:
        best = min(rows, key=lambda r: float(r["pose_gate_margin"])) if rows else {}
        result_name = "c1_not_cleared_pose_still_fails"
        interpretation = (
            "no swept debug arm brought pose_step under band under the adjusted view; this does not prove representation conflict yet, but c1 remains uncleared."
        )
    artifacts = {
        "summary_json": str(args.pose_sweep_out_dir / "pose_step_c1c2_sweep_summary.json"),
        "summary_md": str(args.pose_sweep_out_dir / "summary.md"),
        "rows_csv": str(args.pose_sweep_out_dir / "rows.csv"),
    }
    return {
        "task": "pose_step_c1_c2_debug_sweep",
        "scope": "debug-only one-window pose loss-balance vs representation-conflict discriminator; no production Trainer/runtime/gate/checkpoint mutation",
        "config": {
            "weights": weights,
            "modes": modes,
            "pose_gate_topk": int(args.pose_gate_topk),
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "event_window": int(args.event_window),
            "gate_mode_mean_pose_weight": float(args.pose_sweep_gate_mode_mean_weight),
            "horizon": int(args.horizon),
            "dtype": "float32",
            "device": "cpu",
        },
        "surrogate_alignment": {
            "existing_pose_continuity_loss": "mean MSE(pred rot6d step delta - GT rot6d step delta), over [1,15,276] float32 cpu",
            "hard_gate_pose_step_l2_p95": "p95(raw adjacent-frame rot6d L2/sqrt(276)), over [15] float64 metric view",
            "aligned_gate_surrogate": "top-k/mean relu(pose_step_l2 - clip_band)^2, debug-only",
            "definitions_identical": False,
        },
        "guard_path_identity": guard,
        "window": {"clip": item.clip, "start": int(item.start), "end": int(item.end), "pose_band": pose_gate_band},
        "rows": rows,
        "verdict": {
            "result": result_name,
            "best_label": best.get("label"),
            "interpretation": interpretation,
        },
        "hard_constraint_confirmations": {
            "debug_only": True,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "changed_production_loss_or_model": False,
            "saved_debug_tiny_decoder_weights": True,
        },
        "artifacts": artifacts,
    }


def _loss_refactor_flat_row(
    *,
    base_row: Mapping[str, Any],
    adjusted_guard: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    out = {
        "stage": base_row.get("stage"),
        "arm": base_row.get("arm"),
        "clip": base_row.get("clip"),
        "start": base_row.get("start"),
        "end": base_row.get("end"),
        "base_acceptance_proxy_pass": base_row.get("acceptance_proxy_pass"),
        "base_failed_family": base_row.get("failed_family"),
        "base_support_side_correctness": base_row.get("support_side_correctness"),
        "base_support_side_failure_count": base_row.get("support_side_failure_count"),
        "pred_raw_npz": artifacts.get("pred_raw_npz"),
        "decoder_state_pt": artifacts.get("decoder_state_pt"),
    }
    adjusted = adjusted_guard.get("gate_w4096_adjusted", {}) or {}
    metrics = adjusted.get("metrics", {}) or {}
    for key in (
        "adjusted_pass",
        "adjusted_failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
    ):
        out[key] = adjusted.get(key)
    for key, val in metrics.items():
        out[f"metric_{key}"] = val
    verdict = adjusted_guard.get("verdict", {}) or {}
    out["shortcut_negative_controls_still_fail"] = verdict.get("shortcut_negative_controls_still_fail")
    out["command_demotion_negative_controls_still_fail"] = verdict.get("command_demotion_negative_controls_still_fail")
    out["adjusted_guard_decision"] = verdict.get("decision")
    return out


def _write_loss_refactor_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    guard = payload["guard_path_identity"]
    verdict = payload["verdict"]
    adjusted_guard = payload.get("adjusted_acceptance_guard", {}) or {}
    adjusted = adjusted_guard.get("gate_w4096_adjusted", {}) or {}
    metrics = adjusted.get("metrics", {}) or {}
    thresholds = adjusted.get("thresholds", {}) or {}
    config = payload.get("config", {}) or {}

    def _metric_le(metric_key: str, threshold_key: str, eps: float = 1e-6) -> Any:
        value = metrics.get(metric_key)
        threshold = thresholds.get(threshold_key)
        if value is None or threshold is None:
            return None
        return bool(float(value) <= float(threshold) + eps)

    rows = [
        "# Causal Loss Refactor",
        "",
        "Debug-only one-window train-fit. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## 9 -> 3 Mapping",
        "",
        "| witness / old symptom | causal item | debug surrogate |",
        "|---|---|---|",
        "| `dynamics_consistency` rot6d residual | `L_articulation` | GT-residual anchored `MSE(r_pred,r_gt)` over `rot6d [B,15,276]` |",
        "| `pose_continuity_loss` / `pose_step_l2_p95` | `L_articulation` | band-normalized rot6d step p95 margin |",
        "| `bone_angvel_rate_loss` / joint rate witnesses | `L_articulation` | band-normalized joint-angvel rate margins; RMS is event-aware |",
        "| `dynamics_consistency` root residual | `L_root_support` | GT-residual anchored `MSE(r_pred,r_gt)` over `root_vel [B,15,2]` |",
        "| `rootvel_rate_loss` / `rootvel_step_l2_p95` | `L_root_support` | band-normalized rootvel-rate margin |",
        "| `support_side_correctness` | `L_root_support` | FK foot-speed asymmetry, relative-foot, root-speed/lateral support-side feature margins |",
        "| `contact_schedule`, contact-step witness, `fk_foot_slip_loss` | `L_root_support` | event-aware contact-step, oracle schedule anchor, FK contacted-foot speed margin |",
        "| `endpoint_reaching`, `regime_reaching`, `command_compatibility`, `yaw_rate_step_abs` | `L_goal` | endpoint, final regime level, heading p95 tolerance, yaw-rate-step margin |",
        "",
        "## Objective",
        "",
        f"- mode: `{config.get('loss_refactor_objective', 'weighted')}`",
    ]
    if config.get("loss_refactor_objective") == "minimax":
        rows.extend(
            [
                f"- softmax temperature: `{_fmt(config.get('loss_refactor_minimax_temperature'), 6)}`",
                f"- anchor tie-breaker weight: `{_fmt(config.get('loss_refactor_anchor_weight'), 6)}`",
                f"- warmup mode: `{config.get('loss_refactor_minimax_warmup_mode', 'weighted')}`",
                f"- warmup epochs: `{config.get('loss_refactor_minimax_warmup_epochs', 0)}`",
                f"- minimax tail lr: `{_fmt(config.get('loss_refactor_minimax_tail_lr'), 8)}`",
                f"- support margin power: `{_fmt(config.get('loss_refactor_support_margin_power'), 6)}`",
                f"- support linear feature keys: `{config.get('loss_refactor_support_linear_feature_keys', '')}`",
                f"- support excluded feature keys: `{config.get('loss_refactor_support_excluded_feature_keys', '')}`",
                f"- support hard-gate feature keys: `{config.get('loss_refactor_support_hard_gate_feature_keys', '')}`",
                f"- support hard-gate safety margin: `{_fmt(config.get('loss_refactor_support_hard_gate_safety_margin'), 8)}`",
                f"- final softmax gate violation: `{_fmt(payload.get('final_train_objective_terms', {}).get('loss_refactor_softmax_gate_violation'), 8)}`",
                f"- final hard max gate violation: `{_fmt(payload.get('final_train_objective_terms', {}).get('loss_refactor_hard_max_gate_violation'), 8)}`",
                f"- final anchor tie-breaker: `{_fmt(payload.get('final_train_objective_terms', {}).get('loss_refactor_anchor_tiebreaker'), 8)}`",
            ]
        )
    rows.extend(
        [
            "",
            "## Legacy Three-Term Weights",
            "",
            "| item | weight |",
            "|---|---:|",
        ]
    )
    weights = payload.get("loss_weights", {}) or {}
    for key in ("L_articulation", "L_root_support", "L_goal"):
        rows.append(f"| `{key}` | {_fmt(weights.get(key, 0.0), 6)} |")
    rows.extend(
        [
            "",
            "## Guard Path Identity",
            "",
            f"- passed: `{guard.get('passed')}`",
            f"- reconstructed_gt_acceptance_rate: `{_fmt(guard.get('reconstructed_gt_acceptance_rate', 0.0))}`",
            f"- decoder_path_from_gt_raw_acceptance_rate: `{_fmt(guard.get('decoder_path_from_gt_raw_acceptance_rate', 0.0))}`",
            f"- max_abs_seq_delta: `{_fmt(guard.get('max_abs_seq_delta', 0.0), 8)}`",
            "",
            "## One-Window Adjusted Family",
            "",
            "| family/metric | value | band/tolerance | result |",
            "|---|---:|---:|---:|",
            f"| `regime_reached` | {_fmt(metrics.get('bone_angvel_level_rms_to_target'), 8)} | {_fmt(thresholds.get('bone_angvel_level_rms'), 8)} | `{adjusted.get('regime_reached')}` |",
            f"| `rate_budget.angvel_step_rms_p95` | {_fmt(metrics.get('angvel_step_rms_p95'), 8)} | event-aware `{_fmt(thresholds.get('angvel_step_rms'), 8)}` | `{_metric_le('angvel_step_rms_p95', 'angvel_step_rms')}` |",
            f"| `rate_budget.angvel_component_p95_p95` | {_fmt(metrics.get('angvel_component_p95_p95'), 8)} | {_fmt(thresholds.get('angvel_step_component_p95'), 8)} | `{_metric_le('angvel_component_p95_p95', 'angvel_step_component_p95')}` |",
            f"| `rate_budget.rootvel_step_l2_p95` | {_fmt(metrics.get('rootvel_step_l2_p95'), 8)} | {_fmt(thresholds.get('rootvel_step_l2'), 8)} | `{_metric_le('rootvel_step_l2_p95', 'rootvel_step_l2')}` |",
            f"| `rate_budget.yaw_rate_step_abs_p95` | {_fmt(metrics.get('yaw_rate_step_abs_p95'), 8)} | {_fmt(thresholds.get('yaw_rate_step_abs'), 8)} | `{_metric_le('yaw_rate_step_abs_p95', 'yaw_rate_step_abs')}` |",
            f"| `support_honesty.contact_step_l2_p95` | {_fmt(metrics.get('contact_step_l2_p95'), 8)} | event-aware `{_fmt(thresholds.get('contact_step_l2'), 8)}` | `{_metric_le('contact_step_l2_p95', 'contact_step_l2')}` |",
            f"| `support_honesty.foot_slip_p95_mps` | {_fmt(metrics.get('foot_slip_p95_mps'), 8)} | {_fmt(thresholds.get('foot_slip_contacted_speed_mps'), 8)} | `{_metric_le('foot_slip_p95_mps', 'foot_slip_contacted_speed_mps')}` |",
            f"| `support_side_correctness` | `{payload.get('base_row', {}).get('support_side_failure_count')}` failures | `0` | `{adjusted.get('support_side_correctness')}` |",
            f"| `command_response.heading_error_p95_rad` | {_fmt(metrics.get('heading_error_p95_rad'), 8)} | {_fmt(payload.get('config', {}).get('heading_tolerance_rad'), 8)} | `{adjusted.get('command_response')}` |",
            f"| `pose_continuity.pose_step_l2_p95` | {_fmt(metrics.get('pose_step_l2_p95'), 8)} | {_fmt(thresholds.get('pose_step_l2'), 8)} | `{adjusted.get('pose_continuity')}` |",
            f"| `endpoint_bridgeability` | `{adjusted.get('endpoint_details', {}).get('endpoint_bridgeability_proxy')}` | proxy | `{adjusted.get('endpoint_bridgeability')}` |",
            "",
            f"- adjusted_pass: `{adjusted.get('adjusted_pass')}`",
            f"- adjusted_failed_family: `{adjusted.get('adjusted_failed_family', '')}`",
            "",
            "## Negative Controls",
            "",
            f"- shortcut negative controls still fail: `{verdict.get('shortcut_negative_controls_still_fail')}`",
            f"- command demotion negative controls still fail: `{verdict.get('command_demotion_negative_controls_still_fail')}`",
            f"- adjusted guard decision: `{verdict.get('adjusted_guard_decision')}`",
            f"- authorize_8window: `{verdict.get('authorize_8window')}`",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
            f"- step log csv: `{payload['artifacts']['step_log_csv']}`",
            f"- pred raw: `{payload['artifacts'].get('pred_raw_npz')}`",
            f"- decoder state: `{payload['artifacts'].get('decoder_state_pt')}`",
            f"- adjusted guard summary: `{payload['artifacts'].get('adjusted_guard_summary_md')}`",
        ]
    )
    _dump_md(path, rows)


def run_loss_refactor(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")

    base = _build_base_operator(args, Path(args.npz_root), device)
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )

    start = int(args.overfit_start_index)
    one_idxs = tuple(range(start, start + 1))
    guard = _guard_path_identity(
        items=main_items,
        idxs=tuple(range(len(main_items))) if bool(args.guard_all_windows) else one_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=float(args.pass_rate_threshold),
    )
    if not bool(guard.get("passed", False)):
        raise RuntimeError(f"guard_path_identity failed: {guard.get('reason')}")

    loss_ctx = _build_loss_refactor_context(
        items=main_items,
        idxs=one_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        args=args,
        device=device,
    )
    result = _train_stage_arm(
        stage="one_window",
        arm="loss_refactor_causal3",
        idxs=one_idxs,
        items=main_items,
        base=base,
        skeleton=skeleton,
        args=args,
        device=device,
        loss_refactor_context=loss_ctx,
        instrument_step_log=True,
    )
    pred_rows = _eval_result_rows(
        result=result,
        items=main_items,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        args=args,
    )
    localization = _localize_result(
        result=result,
        items=main_items,
        baseline_bands=reconstructed_baseline_bands,
        base=base,
        args=args,
        device=device,
    )

    args.loss_refactor_out_dir.mkdir(parents=True, exist_ok=True)
    item = main_items[one_idxs[0]]
    artifact_label = f"causal3_{args.loss_refactor_objective}_one_window"
    saved = _save_sweep_artifacts(
        out_dir=args.loss_refactor_out_dir,
        label=artifact_label,
        result=result,
        item=item,
        localization=localization,
    )
    adjusted_out_dir = args.loss_refactor_out_dir / "adjusted_guard"
    adjusted_guard = _run_adjusted_acceptance_guard(
        argparse.Namespace(
            npz_root=args.npz_root,
            z_features=args.z_features,
            two_frame_summary=args.guard_two_frame_summary,
            bone_bridge_summary=args.guard_bone_bridge_summary,
            regime_bridge_summary=args.guard_regime_bridge_summary,
            command_demotion_rows=args.guard_command_demotion_rows,
            pose_sweep_pred_raw=Path(saved["pred_raw_npz"]),
            out_dir=adjusted_out_dir,
            horizon=int(args.horizon),
            context_len=int(args.context_len),
            stride=int(args.stride),
            min_run_frames=int(args.min_run_frames),
            baseline_quantile=float(args.guard_baseline_quantile),
            reconstructed_baseline_quantile=float(args.reconstructed_baseline_quantile),
            bridge_budget_quantile=float(args.guard_bridge_budget_quantile),
            pose_topk=int(args.guard_pose_topk),
            ground_contact_thr=float(args.guard_ground_contact_thr),
            ground_pose_thr=float(args.guard_ground_pose_thr),
            event_window=int(args.event_window),
            heading_tolerance_rad=float(args.heading_tolerance_rad),
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
            command_align_root_vel=bool(args.command_align_root_vel),
        )
    )

    top_terms = [
        "L_articulation",
        "L_root_support",
        "L_goal",
        "loss_refactor_minimax_feasibility",
        "loss_refactor_softmax_gate_violation",
        "loss_refactor_hard_max_gate_violation",
        "loss_refactor_mean_gate_violation",
        "loss_refactor_anchor_tiebreaker",
        "articulation_low_anchor_loss",
        "articulation_angvel_rms_margin_loss",
        "articulation_angvel_component_margin_loss",
        "articulation_angvel_rate_margin_loss",
        "articulation_pose_step_margin_loss",
        "root_support_low_anchor_loss",
        "root_support_rootvel_rate_margin_loss",
        "root_support_rootvel_rate_anchor_loss",
        "root_support_rootvel_path_anchor_loss",
        "root_support_contact_step_margin_loss",
        "root_support_contact_anchor_loss",
        "root_support_fk_slip_margin_loss",
        "root_support_contact_honesty_loss",
        "root_support_side_margin_loss",
        "goal_endpoint_margin_loss",
        "goal_heading_margin_loss",
        "goal_yaw_rate_margin_loss",
        "goal_regime_margin_loss",
    ]
    top_terms.extend(
        sorted(
            key
            for key in result.get("final_train_objective_terms", {})
            if key.startswith("root_support_contact_support_side_")
        )
    )
    step_summary = _term_summary(result["step_rows"], top_terms)
    adjusted_verdict = adjusted_guard.get("verdict", {}) or {}
    gate_full_pass = bool(adjusted_verdict.get("gate_w4096_full_family_pass", False))
    shortcut_fail = bool(adjusted_verdict.get("shortcut_negative_controls_still_fail", False))
    command_fail = bool(adjusted_verdict.get("command_demotion_negative_controls_still_fail", False))
    one_window_debug_ready = bool(gate_full_pass and shortcut_fail and command_fail)
    rootvel_zero_slack_hold = not bool(args.loss_refactor_allow_zero_slack_8window_authorization)
    authorize_8 = bool(one_window_debug_ready and not rootvel_zero_slack_hold)
    artifacts = {
        "summary_json": str(args.loss_refactor_out_dir / "loss_refactor_summary.json"),
        "summary_md": str(args.loss_refactor_out_dir / "summary.md"),
        "rows_csv": str(args.loss_refactor_out_dir / "rows.csv"),
        "step_log_csv": str(args.loss_refactor_out_dir / "step_log.csv"),
        "pred_raw_npz": saved["pred_raw_npz"],
        "decoder_state_pt": saved["decoder_state_pt"],
        "adjusted_guard_summary_json": str(adjusted_out_dir / "adjusted_acceptance_guard_summary.json"),
        "adjusted_guard_summary_md": str(adjusted_out_dir / "summary.md"),
        "adjusted_guard_rows_csv": str(adjusted_out_dir / "rows.csv"),
    }
    flat_rows = [
        _loss_refactor_flat_row(
            base_row=pred_rows[0] if pred_rows else {},
            adjusted_guard=adjusted_guard,
            artifacts=saved,
        )
    ]
    return {
        "task": "loss_refactor_causal3_debug_one_window",
        "scope": "debug-only causal loss refactor; no production Trainer/runtime/gate/checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.loss_refactor_out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": "cpu",
            "dtype": "float32",
            "event_window": int(args.event_window),
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
            "command_align_root_vel": bool(args.command_align_root_vel),
            "loss_refactor_objective": str(args.loss_refactor_objective),
            "loss_refactor_minimax_temperature": float(args.loss_refactor_minimax_temperature),
            "loss_refactor_anchor_weight": float(args.loss_refactor_anchor_weight),
            "loss_refactor_minimax_warmup_mode": str(args.loss_refactor_minimax_warmup_mode),
            "loss_refactor_minimax_warmup_epochs": int(args.loss_refactor_minimax_warmup_epochs),
            "loss_refactor_minimax_tail_lr": float(args.loss_refactor_minimax_tail_lr),
            "instrument_step_log_stride": int(args.instrument_step_log_stride),
            "loss_refactor_articulation_weight": float(args.loss_refactor_articulation_weight),
            "loss_refactor_root_support_weight": float(args.loss_refactor_root_support_weight),
            "loss_refactor_goal_weight": float(args.loss_refactor_goal_weight),
            "loss_refactor_rate_topk": int(args.loss_refactor_rate_topk),
            "loss_refactor_pose_topk": int(args.loss_refactor_pose_topk),
            "loss_refactor_heading_topk": int(args.loss_refactor_heading_topk),
            "loss_refactor_support_feature_topk": int(args.loss_refactor_support_feature_topk),
            "loss_refactor_support_band_floor": float(args.loss_refactor_support_band_floor),
            "loss_refactor_support_margin_power": float(args.loss_refactor_support_margin_power),
            "loss_refactor_support_linear_feature_keys": str(args.loss_refactor_support_linear_feature_keys),
            "loss_refactor_support_excluded_feature_keys": str(args.loss_refactor_support_excluded_feature_keys),
            "loss_refactor_support_hard_gate_feature_keys": str(args.loss_refactor_support_hard_gate_feature_keys),
            "loss_refactor_support_hard_gate_safety_margin": float(args.loss_refactor_support_hard_gate_safety_margin),
            "loss_refactor_allow_zero_slack_8window_authorization": bool(
                args.loss_refactor_allow_zero_slack_8window_authorization
            ),
        },
        "input_output_contract": {
            "decoder_input": {"shape": "[1,4957]", "dtype": "float32", "device": "cpu"},
            "middle_state_output": {"shape": [1, int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux_output": {"shape": [1, int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "saved_pred_raw": {"shape": [1, int(args.horizon) * (STATE_DIM + ANGVEL_DIM)], "dtype": "float32", "device": "cpu numpy"},
        },
        "guard_path_identity": guard,
        "base_row": pred_rows[0] if pred_rows else {},
        "localization": {k: v for k, v in localization.items() if k not in ("per_frame_rows", "per_channel_rows")},
        "adjusted_acceptance_guard": adjusted_guard,
        "loss_weights": result["weights"],
        "warmup_epochs": result.get("warmup_epochs", 0),
        "warmup_weights": result.get("warmup_weights", {}),
        "final_train_objective": result["final_train_objective"],
        "final_train_objective_terms": result["final_train_objective_terms"],
        "step_log_summary": step_summary,
        "verdict": {
            "one_window_adjusted_full_family_pass": gate_full_pass,
            "shortcut_negative_controls_still_fail": shortcut_fail,
            "command_demotion_negative_controls_still_fail": command_fail,
            "adjusted_guard_decision": adjusted_verdict.get("decision"),
            "one_window_debug_ready_for_8window_contract_retest": one_window_debug_ready,
            "rootvel_zero_slack_contract_hold": rootvel_zero_slack_hold,
            "authorize_8window": authorize_8,
        },
        "hard_constraint_confirmations": {
            "debug_only": True,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "changed_production_loss_or_model": False,
            "used_residual_head": False,
            "used_diffusion_or_sampling": False,
            "saved_debug_tiny_decoder_weights": True,
        },
        "rows": flat_rows,
        "artifacts": artifacts,
        "_step_rows_for_csv": result["step_rows"],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")

    base = _build_base_operator(args, Path(args.npz_root), device)
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))

    start = int(args.overfit_start_index)
    one_idxs = tuple(range(start, start + 1))
    eight_idxs = tuple(range(start, min(start + 8, len(main_items))))
    full_idxs = tuple(range(len(main_items)))
    gt_floor = {
        "one_window": _gt_dynamics_floor(idxs=one_idxs, items=main_items, base=base, args=args, device=device),
        "eight_window": _gt_dynamics_floor(idxs=eight_idxs, items=main_items, base=base, args=args, device=device),
        "full_188": _gt_dynamics_floor(idxs=full_idxs, items=main_items, base=base, args=args, device=device),
    }
    guard_idxs = full_idxs if bool(args.guard_all_windows) else eight_idxs
    guard = _guard_path_identity(
        items=main_items,
        idxs=guard_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=float(args.pass_rate_threshold),
    )

    rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    stage_results: List[Dict[str, Any]] = []
    terms = [
        "dynamics_consistency",
        "dynamics_zero_residual_witness",
        "command_compatibility",
        "endpoint_reaching",
        "regime_reaching",
        "state_raw_mse",
        "flat_standardized",
        "pose_supervision",
        "contact_schedule",
        "root_vel_supervision",
        "root_pos_supervision",
        "bone_angvel_supervision",
        "pose_continuity_loss",
        "rootvel_rate_loss",
        "yaw_rate_loss",
        "bone_angvel_rate_loss",
        "fk_foot_slip_loss",
    ]

    def run_stage(stage: str, idxs: Sequence[int]) -> bool:
        stage_pass = True
        for arm in ("dynamics_consistency", "symptom_ablation"):
            result = _train_stage_arm(
                stage=stage,
                arm=arm,
                idxs=idxs,
                items=main_items,
                base=base,
                skeleton=skeleton,
                args=args,
                device=device,
            )
            step_rows.extend(result["step_rows"])
            pred_rows = _eval_result_rows(
                result=result,
                items=main_items,
                baseline_bands=reconstructed_baseline_bands,
                support_bands=reconstructed_support_bands,
                skeleton=skeleton,
                args=args,
            )
            rows.extend(pred_rows)
            train_acceptance = _summarize_rows(pred_rows)
            step_summary = _term_summary(result["step_rows"], terms)
            diagnosis = _classify(
                guard=guard,
                acceptance=train_acceptance,
                term_summary=step_summary,
                pass_rate_threshold=float(args.pass_rate_threshold),
            )
            stage_results.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "train_n": int(len(idxs)),
                    "train_indices": [int(x) for x in idxs],
                    "train_windows": [
                        {"clip": main_items[int(i)].clip, "start": int(main_items[int(i)].start), "end": int(main_items[int(i)].end)}
                        for i in idxs[:16]
                    ],
                    "train_loss": result["train_loss_metrics"],
                    "train_acceptance": train_acceptance,
                    "step_log_summary": step_summary,
                    "diagnosis": diagnosis,
                    "loss_weights": result["weights"],
                    "final_train_objective": result["final_train_objective"],
                    "final_train_objective_terms": result["final_train_objective_terms"],
                    "parameter_count": result["parameter_count"],
                    "input_dim": result["input_dim"],
                    "output_dim": result["output_dim"],
                }
            )
            if arm == "dynamics_consistency":
                stage_pass = stage_pass and diagnosis == "train_fit_acceptance_pass"
        return stage_pass

    ran_8 = False
    ran_full = False
    if bool(guard.get("passed", False)):
        one_pass = run_stage("one_window", one_idxs)
        if one_pass and bool(args.run_8window_after_pass):
            ran_8 = True
            eight_pass = run_stage("eight_window", eight_idxs)
            if eight_pass and bool(args.run_full_after_pass):
                ran_full = True
                run_stage("full_188", full_idxs)

    dynamics_records = [r for r in stage_results if r.get("arm") == "dynamics_consistency"]
    last_dyn = dynamics_records[-1] if dynamics_records else {}
    failure_signature = str(last_dyn.get("diagnosis", "guard_path_identity_failed"))
    if failure_signature == "train_fit_fail_zero_residual_lowpass_basin_witness":
        interpretation = "zero-residual witness fell into a low-pass base-operator basin while GT-residual anchor remained unsatisfied."
    elif failure_signature == "train_fit_fail_operator_consistency_blocks_regime_reaching":
        interpretation = "operator-consistency signature: GT-residual anchored dynamics loss is low while regime reaching remains resistant."
    elif failure_signature == "train_fit_fail_gt_residual_anchor_acceptance_fail":
        interpretation = "GT-residual anchored dynamics fit is low, but acceptance witnesses still fail; keep witness constraints."
    elif failure_signature == "train_fit_acceptance_pass":
        interpretation = "current stage fit passed; no entanglement bottleneck from this stage."
    elif not bool(guard.get("passed", False)):
        interpretation = "guard failed; training result is not decision-eligible."
    else:
        interpretation = "train-fit failed without a decisive entanglement or operator-manifold signature."

    artifacts = {
        "summary_json": str(args.out_dir / "summary.json"),
        "summary_md": str(args.out_dir / "summary.md"),
        "rows_csv": str(args.out_dir / "rows.csv"),
        "step_log_csv": str(args.out_dir / "step_log.csv"),
    }
    return {
        "task": "dynamics_consistency_gt_residual_train_fit_ladder",
        "scope": "debug-only deterministic decoder train-fit ladder; no production Trainer/runtime/gate training; no checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": "cpu",
            "dtype": "float32",
            "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
            "command_align_root_vel": bool(args.command_align_root_vel),
            "dynamics_eval_scale_floor": float(args.dynamics_eval_scale_floor),
            "dynamics_anchor": "gt_residual",
            "mechanism_contact_loss_weight": float(args.mechanism_contact_loss_weight),
            "mechanism_pose_step_loss_weight": float(args.mechanism_pose_step_loss_weight),
            "mechanism_rootvel_rate_loss_weight": float(args.mechanism_rootvel_rate_loss_weight),
            "mechanism_yaw_rate_loss_weight": float(args.mechanism_yaw_rate_loss_weight),
            "mechanism_aux_rate_loss_weight": float(args.mechanism_aux_rate_loss_weight),
            "mechanism_fk_foot_loss_weight": float(args.mechanism_fk_foot_loss_weight),
            "base_operator_readout": "EventMotionModel ret['out'] main incremental delta; lambda/direct outputs not applied",
        },
        "input_output_contract": {
            "decoder_input": {"shape": "[B,input_dim]", "dtype": "float32", "device": "cpu"},
            "middle_state_output": {"shape": [None, int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux_output": {"shape": [None, int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "base_operator_input": {"shape": "[B,H-1,419]", "dtype": "float32", "device": "cpu"},
            "base_operator_eval_y": {"shape": "[B,H-1,278]", "dtype": "float32", "device": "cpu"},
            "channel_gradient_groups": {
                "pose": "[B,H,276]",
                "contact": "[B,H,2]",
                "rootvel": "[B,H,2] egocentric output channels; world rootvel used for dynamics eval",
                "bone_angvel": "[B,H,138]",
            },
        },
        "dataset": {
            "matched_window_count": int(len(main_items)),
            "per_clip_windows": dict(Counter(it.clip for it in main_items)),
        },
        "guard_path_identity": guard,
        "base_operator_preflight": {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_model_overlap": _checkpoint_overlap_report(base),
            "raw_x_norm_max_abs_error_by_clip": base.raw_x_norm_max_abs_error,
            "raw_x_norm_max_abs_error_max": float(max(base.raw_x_norm_max_abs_error.values())),
            "dynamics_eval_scale_floor": float(args.dynamics_eval_scale_floor),
            "gt_dynamics_floor": gt_floor,
            "angvel_feature_source": "decoder aux bone_angvel raw transformed through dataset VectorTanhNormalizerTorch",
            "pose_history_source": "decoder rot6d history [B,H,3*276] transformed through dataset VectorTanhNormalizerTorch",
        },
        "stage_results": stage_results,
        "decision": {
            "stopped_at": "full_188" if ran_full else ("eight_window" if ran_8 else "one_window"),
            "failure_signature": failure_signature,
            "ran_8window": bool(ran_8),
            "ran_full_188": bool(ran_full),
            "interpretation": interpretation,
        },
        "hard_constraint_confirmations": {
            "debug_only": True,
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "forwarded_base_event_motion_model": True,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "used_diffusion_or_sampling": False,
            "yaw_or_cond_dir_prediction_target": False,
            "attached_to_runtime": False,
            "production_ready_generator": False,
        },
        "rows": rows,
        "step_log_row_count": int(len(step_rows)),
        "artifacts": artifacts,
        "_step_rows_for_csv": step_rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=Path, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=Path, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--localize-only", action="store_true", default=False)
    p.add_argument("--localization-out-dir", type=Path, default=DEFAULT_LOCALIZATION_OUT_DIR)
    p.add_argument("--pose-sweep-only", action="store_true", default=False)
    p.add_argument("--pose-sweep-out-dir", type=Path, default=DEFAULT_POSE_SWEEP_OUT_DIR)
    p.add_argument("--loss-refactor-only", action="store_true", default=False)
    p.add_argument("--loss-refactor-out-dir", type=Path, default=DEFAULT_LOSS_REFACTOR_OUT_DIR)
    p.add_argument("--pose-sweep-weights", type=str, default="4,16,64,256")
    p.add_argument("--pose-sweep-modes", type=str, default="mean,gate")
    p.add_argument("--pose-sweep-gate-mode-mean-weight", type=float, default=0.0)
    p.add_argument("--pose-gate-topk", type=int, default=3)
    p.add_argument("--heading-tolerance-rad", type=float, default=1e-4)
    p.add_argument("--event-window", type=int, default=1)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--instrument-step-log-stride", type=int, default=1)
    p.add_argument("--baseline-quantile", type=float, default=95.0)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--pass-rate-threshold", type=float, default=1.0)
    p.add_argument("--overfit-start-index", type=int, default=0)
    p.add_argument("--guard-all-windows", action="store_true", default=True)
    p.add_argument("--run-8window-after-pass", action="store_true", default=True)
    p.add_argument("--run-full-after-pass", action="store_true", default=True)
    p.add_argument("--oracle-contact-passthrough", action="store_true", default=False)
    p.add_argument("--command-align-root-vel", action="store_true", default=False)
    p.add_argument("--dynamics-loss-weight", type=float, default=1.0)
    p.add_argument("--dynamics-eval-scale-floor", type=float, default=0.05)
    p.add_argument("--command-loss-weight", type=float, default=24.0)
    p.add_argument("--endpoint-loss-weight", type=float, default=4.0)
    p.add_argument("--regime-loss-weight", type=float, default=1.0)
    p.add_argument("--state-anchor-loss-weight", type=float, default=0.0)
    p.add_argument("--flat-standardized-loss-weight", type=float, default=0.0)
    p.add_argument("--mechanism-contact-loss-weight", type=float, default=4.0)
    p.add_argument("--mechanism-pose-step-loss-weight", type=float, default=4.0)
    p.add_argument("--mechanism-rootvel-rate-loss-weight", type=float, default=4.0)
    p.add_argument("--mechanism-yaw-rate-loss-weight", type=float, default=4.0)
    p.add_argument("--mechanism-aux-rate-loss-weight", type=float, default=1.0)
    p.add_argument("--mechanism-fk-foot-loss-weight", type=float, default=0.35)
    p.add_argument("--loss-refactor-objective", choices=("minimax", "weighted"), default="minimax")
    p.add_argument("--loss-refactor-minimax-temperature", type=float, default=0.05)
    p.add_argument("--loss-refactor-anchor-weight", type=float, default=0.05)
    p.add_argument("--loss-refactor-minimax-warmup-mode", choices=("weighted", "supervised_flat", "supervised_raw"), default="weighted")
    p.add_argument("--loss-refactor-minimax-warmup-epochs", type=int, default=0)
    p.add_argument("--loss-refactor-minimax-tail-lr", type=float, default=0.0)
    p.add_argument("--loss-refactor-articulation-weight", type=float, default=8.0)
    p.add_argument("--loss-refactor-root-support-weight", type=float, default=8.0)
    p.add_argument("--loss-refactor-goal-weight", type=float, default=1.0)
    p.add_argument("--loss-refactor-rate-topk", type=int, default=3)
    p.add_argument("--loss-refactor-pose-topk", type=int, default=3)
    p.add_argument("--loss-refactor-heading-topk", type=int, default=3)
    p.add_argument("--loss-refactor-support-feature-topk", type=int, default=0)
    p.add_argument("--loss-refactor-support-band-floor", type=float, default=1e-5)
    p.add_argument("--loss-refactor-support-margin-power", type=float, default=2.0)
    p.add_argument("--loss-refactor-support-linear-feature-keys", type=str, default="")
    p.add_argument("--loss-refactor-support-excluded-feature-keys", type=str, default="")
    p.add_argument("--loss-refactor-support-hard-gate-feature-keys", type=str, default="")
    p.add_argument("--loss-refactor-support-hard-gate-safety-margin", type=float, default=0.0)
    p.add_argument("--loss-refactor-allow-zero-slack-8window-authorization", action="store_true", default=False)
    p.add_argument("--loss-refactor-dynamics-low-band", type=float, default=1.0)
    p.add_argument("--guard-two-frame-summary", type=Path, default=DEFAULT_GUARD_TWO_FRAME)
    p.add_argument("--guard-bone-bridge-summary", type=Path, default=DEFAULT_GUARD_BONE_BRIDGE)
    p.add_argument("--guard-regime-bridge-summary", type=Path, default=DEFAULT_GUARD_REGIME_BRIDGE)
    p.add_argument("--guard-command-demotion-rows", type=Path, default=DEFAULT_GUARD_COMMAND_DEMOTION_ROWS)
    p.add_argument("--guard-baseline-quantile", type=float, default=99.5)
    p.add_argument("--guard-bridge-budget-quantile", type=float, default=95.0)
    p.add_argument("--guard-pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--guard-ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--guard-ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--symptom-pose-loss-weight", type=float, default=1.0)
    p.add_argument("--symptom-contact-loss-weight", type=float, default=4.0)
    p.add_argument("--symptom-rootvel-loss-weight", type=float, default=4.0)
    p.add_argument("--symptom-rootpos-loss-weight", type=float, default=8.0)
    p.add_argument("--symptom-aux-loss-weight", type=float, default=0.5)
    p.add_argument("--symptom-pose-step-loss-weight", type=float, default=16.0)
    p.add_argument("--symptom-rootvel-rate-loss-weight", type=float, default=4.0)
    p.add_argument("--symptom-yaw-rate-loss-weight", type=float, default=4.0)
    p.add_argument("--symptom-aux-rate-loss-weight", type=float, default=1.0)
    p.add_argument("--symptom-fk-foot-loss-weight", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.loss_refactor_only):
        args.loss_refactor_out_dir.mkdir(parents=True, exist_ok=True)
        payload = run_loss_refactor(args)
        step_rows = payload.pop("_step_rows_for_csv")
        _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
        _write_generic_csv(Path(payload["artifacts"]["rows_csv"]), payload["rows"])
        _write_step_csv(Path(payload["artifacts"]["step_log_csv"]), step_rows)
        _write_loss_refactor_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
        print(f"[OK] wrote {payload['artifacts']['summary_md']}")
        print(json.dumps(_jsonify(payload["verdict"]), ensure_ascii=False, indent=2))
        return
    if bool(args.pose_sweep_only):
        args.pose_sweep_out_dir.mkdir(parents=True, exist_ok=True)
        payload = run_pose_sweep(args)
        _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
        _write_generic_csv(Path(payload["artifacts"]["rows_csv"]), payload["rows"])
        _write_pose_sweep_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
        print(f"[OK] wrote {payload['artifacts']['summary_md']}")
        return
    if bool(args.localize_only):
        args.localization_out_dir.mkdir(parents=True, exist_ok=True)
        payload = run_localization(args)
        per_frame_rows = payload.pop("_per_frame_rows_for_csv")
        per_channel_rows = payload.pop("_per_channel_rows_for_csv")
        _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
        _write_generic_csv(Path(payload["artifacts"]["per_frame_csv"]), per_frame_rows)
        _write_generic_csv(Path(payload["artifacts"]["per_channel_csv"]), per_channel_rows)
        _write_localization_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
        print(f"[OK] wrote {payload['artifacts']['summary_md']}")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    step_rows = payload.pop("_step_rows_for_csv")
    _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
    _write_rows_csv(Path(payload["artifacts"]["rows_csv"]), payload["rows"])
    _write_step_csv(Path(payload["artifacts"]["step_log_csv"]), step_rows)
    _write_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")


if __name__ == "__main__":
    main()
