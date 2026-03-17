from __future__ import annotations


# ===== Common Helpers (extracted) =====

# ========== [Unified Geometry Utilities] ==========
import argparse
import ast
import glob
import json
import math as _math
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Mapping, Sequence, Callable, List, Tuple

from .eval_utils import FreeRunSettings, evaluate_teacher, evaluate_freerun
from .geometry import (
    rot6d_to_matrix,
    matrix_to_rot6d,
    compose_rot6d_delta,
    geodesic_R,
    so3_exp_map,
    so3_log_map,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    _root_relative_matrices,
    _matrix_log_map,
    normalize_rot6d_delta,
    _rot6d_identity_like,
    wrap_to_pi_np as _wrap_to_pi_np,
    gram_schmidt_renorm_np,
)
from .layout import (
    parse_layout_entry,
    normalize_layout as _normalize_layout,
    layout_span as _layout_span,
    LayoutCenter,
    DataNormalizer,
    apply_layout_center,
)
from .rotvec_semantics import require_standard_rotvec_spec
from .dataset import (
    MotionAugmentation,
    MotionEventDataset,
    ClipData,
    make_fixedlen_collate,
    _infer_forward_axis_from_clip,
)
from .diagnostics import _maybe_optimize_dataset_index, _norm_debug_once, _parse_stage_schedule
from .io import (
    load_soft_contacts_from_json as _load_soft_contacts_from_json,
    direction_yaw_from_array as _direction_yaw_from_array,
    velocity_yaw_from_array as _velocity_yaw_from_array,
    speed_from_X_layout as _speed_from_X_layout,
    npz_scalar_to_str as _npz_scalar_to_str,
)
from .utils import (
    build_mlp,
    safe_set_slice,
    expand_paths_from_specs,
    get_flag_value_from_argv,
    get_flag_values_from_argv,
    validate_and_fix_model_,
    sanity_check_model_dims,
    set_global_args,
    get_global_arg,
)
from .history import (
    PoseHistState,
    advance_pose_hist_state,
    init_pose_hist_state,
    resolve_pose_hist_input,
)

import torch.nn as nn

from .models import (
    MotionEncoder,
    PeriodHead,
    EventMotionModel,
    MotionJointLoss,
    DEFAULT_DIRECT_POSE_LEG_BONES,
    STAGE6_3WAY_ARMCHAIN_BONES,
    STAGE6_3WAY_ARMCHAIN_BONES_CSV,
)


_arg = get_global_arg
_STATE_UPDATE_UNSET = object()


@dataclass(frozen=True)
class RolloutPredictionBuffers:
    hidden_seq: Sequence[torch.Tensor]
    period_pred: Sequence[torch.Tensor]
    contacts_plan: Sequence[torch.Tensor]
    contacts_plan_logits: Sequence[torch.Tensor]
    out_direct: Sequence[torch.Tensor]
    contacts_meas: Sequence[torch.Tensor]
    contacts_err: Sequence[torch.Tensor]
    event_clock_lambda_logit: Sequence[torch.Tensor]
    event_clock_dynamic_prior: Sequence[torch.Tensor]
    event_clock_delta_z: Sequence[torch.Tensor]


@dataclass(frozen=True)
class ContactMeasRuntime:
    x_raw: torch.Tensor
    contact_dim: int
    rot_slice: slice
    rot_flat: torch.Tensor
    joint_count: int
    parents: Any
    offsets: torch.Tensor
    root_pos: torch.Tensor
    up_axis: int


def _parse_pretrain_contact_affine_spec(spec: Any) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    raw = spec
    if isinstance(raw, Path):
        raw = str(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            p = Path(s).expanduser()
            if p.is_file():
                with p.open('r', encoding='utf-8') as f:
                    raw = json.load(f)
            else:
                raw = json.loads(s)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    scale = raw.get('scale', None)
    bias = raw.get('bias', None)
    if not isinstance(scale, (list, tuple)) or not isinstance(bias, (list, tuple)):
        return None
    try:
        scale_vals = [float(x) for x in scale]
        bias_vals = [float(x) for x in bias]
    except Exception:
        return None
    if len(scale_vals) <= 0 or len(scale_vals) != len(bias_vals):
        return None
    if not all(_math.isfinite(float(x)) for x in scale_vals):
        return None
    if not all(_math.isfinite(float(x)) for x in bias_vals):
        return None
    try:
        eps = float(raw.get('eps', 1e-4) or 1e-4)
    except Exception:
        eps = 1e-4
    if not _math.isfinite(float(eps)):
        eps = 1e-4
    eps = float(min(1e-2, max(1e-8, eps)))
    return {'scale': scale_vals, 'bias': bias_vals, 'eps': eps}


@dataclass(frozen=True)
class RolloutSequenceInputs:
    state_seq: torch.Tensor
    cond_seq: Optional[torch.Tensor] = None
    cond_raw_seq: Optional[torch.Tensor] = None
    contacts_seq: Optional[torch.Tensor] = None
    angvel_seq: Optional[torch.Tensor] = None
    pose_hist_seq: Optional[torch.Tensor] = None
    gt_seq: Optional[torch.Tensor] = None


def _new_rollout_prediction_buffers() -> RolloutPredictionBuffers:
    return RolloutPredictionBuffers(
        hidden_seq=[],
        period_pred=[],
        contacts_plan=[],
        contacts_plan_logits=[],
        out_direct=[],
        contacts_meas=[],
        contacts_err=[],
        event_clock_lambda_logit=[],
        event_clock_dynamic_prior=[],
        event_clock_delta_z=[],
    )


@dataclass
class RolloutExecutionState:
    batch_size: int
    total_steps: int
    mode: str
    allow_grad: bool
    tf_ratio: float
    ss_chunk_len: int
    amp_enabled: bool
    rot6d_slice: slice
    rot6d_y_slice: slice
    has_time_dim: Mapping[str, bool]
    cond_norm_mu: Optional[torch.Tensor]
    cond_norm_std: Optional[torch.Tensor]
    enable_reprojection: bool
    plan_enable: bool
    time_base_local: Any
    motion: torch.Tensor
    motion_raw_local: Optional[torch.Tensor]
    y_raw_local: Optional[torch.Tensor]
    pose_hist_state: PoseHistState
    ss_sel_hold: Optional[torch.Tensor] = None
    plan_z: Optional[torch.Tensor] = None
    phase_z: Optional[torch.Tensor] = None
    phase_event_age: Optional[torch.Tensor] = None
    meas_prev_prob: Optional[torch.Tensor] = None
    prev_foot_pos_meas: Optional[torch.Tensor] = None
    reprojection_applied_count: int = 0
    last_attn: Optional[torch.Tensor] = None
    latest_y_raw: Optional[torch.Tensor] = None
    latest_cond_raw_for_env: Optional[torch.Tensor] = None
    outs: List[torch.Tensor] = field(default_factory=list)
    delta_preds: List[torch.Tensor] = field(default_factory=list)
    buffers: RolloutPredictionBuffers = field(default_factory=_new_rollout_prediction_buffers)


@dataclass(frozen=True)
class TrainEpochResult:
    avg_train: float
    train_metrics: Dict[str, Any]


@dataclass
class FitEpochValidationResult:
    metrics_for_json: Optional[Dict[str, Any]] = None
    metrics_tag: Optional[str] = None
    teacher_metrics_cached: Optional[Dict[str, Any]] = None
    forced_valfree_metrics: Optional[Dict[str, Any]] = None
    best_metrics_source: Optional[Dict[str, Any]] = None


@dataclass
class FitCheckpointState:
    best_teacher_val: float = float('inf')
    best_ckpt: Optional[str] = None
    best_teacher_ckpt: Optional[str] = None
    best_free_slope: float = float('inf')
    best_free_ckpt: Optional[str] = None
    best_teacher_payload: Optional[Dict[str, Any]] = None
    best_free_payload: Optional[Dict[str, Any]] = None
    last_payload: Optional[Dict[str, Any]] = None
    last_ckpt: Optional[str] = None
    latest_y_raw: Optional[torch.Tensor] = None
    latest_cond_raw_for_env: Optional[torch.Tensor] = None
    outs: List[torch.Tensor] = field(default_factory=list)
    delta_preds: List[torch.Tensor] = field(default_factory=list)
    buffers: RolloutPredictionBuffers = field(default_factory=_new_rollout_prediction_buffers)


def _finalize_rollout_prediction_buffers(
    preds: Dict[str, torch.Tensor],
    buffers: RolloutPredictionBuffers,
) -> None:
    if buffers.hidden_seq:
        preds["hidden_seq"] = torch.cat(list(buffers.hidden_seq), dim=1)
    if buffers.period_pred:
        preds["period_pred"] = torch.stack(
            [p if p.dim() == 2 else p.squeeze(1) for p in buffers.period_pred],
            dim=1,
        )
    for key, chunks in (
        ("contacts_plan", buffers.contacts_plan),
        ("contacts_plan_logits", buffers.contacts_plan_logits),
        ("out_direct", buffers.out_direct),
        ("contacts_meas", buffers.contacts_meas),
        ("contacts_err", buffers.contacts_err),
        ("event_clock_lambda_logit", buffers.event_clock_lambda_logit),
        ("event_clock_dynamic_prior", buffers.event_clock_dynamic_prior),
        ("event_clock_delta_z", buffers.event_clock_delta_z),
    ):
        if not chunks:
            continue
        try:
            preds[key] = torch.cat(list(chunks), dim=1)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[RolloutFinalize] failed to concat preds['{key}'] with {len(chunks)} chunks"
            ) from exc


def _record_optional_diag_curve(
    result: Dict[str, Any],
    *,
    metric_name: str,
    curve: Any,
    curve_max: Optional[Any] = None,
    curve_bones: Optional[Mapping[str, Sequence[float]]] = None,
    scope_alias: Optional[str] = None,
) -> None:
    def _curve_payload(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    curve_key = f"{metric_name}Curve"
    result[curve_key] = _curve_payload(curve)
    if curve_max is not None:
        curve_max_key = f"{metric_name}CurveMax"
        result[curve_max_key] = _curve_payload(curve_max)
    else:
        curve_max_key = None
    if curve_bones is not None:
        curve_bones_key = f"{metric_name}CurveBones"
        result[curve_bones_key] = curve_bones
    else:
        curve_bones_key = None
    if scope_alias:
        result[f"{scope_alias}/{curve_key}"] = result[curve_key]
        if curve_max_key is not None:
            result[f"{scope_alias}/{curve_max_key}"] = result[curve_max_key]
        if curve_bones_key is not None:
            result[f"{scope_alias}/{curve_bones_key}"] = result[curve_bones_key]


_PHASEC_WARN_ONCE_KEYS: set[str] = set()


def _phasec_warn_once(
    key: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> None:
    key_token = str(key)
    if key_token in _PHASEC_WARN_ONCE_KEYS:
        return
    _PHASEC_WARN_ONCE_KEYS.add(key_token)
    if exc is None:
        print(f"[PhaseC][WARN] {message}")
    else:
        print(f"[PhaseC][WARN] {message}: {exc}")


def _phasec_safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return None


LEGACY_LOSS_KEYS: tuple[str, ...] = (
    "ignore_motion_groups",
    "bone_prior_stds",
    "use_hierarchy_weights",
    "hierarchy_mode",
    "hierarchy_alpha",
    "max_weight_ratio",
    "weight_gamma",
)

LEGACY_LOSS_TOPLEVEL_KEYS: tuple[str, ...] = LEGACY_LOSS_KEYS + (
    "bone_prior_mode",
    "bone_prior_samples",
)

LEGACY_LOSS_CLI_FLAGS: dict[str, str] = {
    "--use_hierarchy_weights": "use_hierarchy_weights",
    "--hierarchy_mode": "hierarchy_mode",
    "--hierarchy_alpha": "hierarchy_alpha",
    "--max_weight_ratio": "max_weight_ratio",
    "--weight_gamma": "weight_gamma",
    "--bone_prior_mode": "bone_prior_mode",
    "--bone_prior_samples": "bone_prior_samples",
}

REMOVED_TRAINBASE_PHASE_RESET_KEYS: tuple[str, ...] = (
    "contact_phase_state_event_kind",
    "contact_phase_state_event_thr",
    "contact_phase_state_event_hyst",
    "contact_phase_state_event_min_interval",
    "phase_reset_source",
)

REMOVED_TRAINBASE_PHASE_RESET_CLI_FLAGS: dict[str, str] = {
    "--contact_phase_state_event_kind": "contact_phase_state_event_kind",
    "--contact_phase_state_event_thr": "contact_phase_state_event_thr",
    "--contact_phase_state_event_hyst": "contact_phase_state_event_hyst",
    "--contact_phase_state_event_min_interval": "contact_phase_state_event_min_interval",
    "--phase_reset_source": "phase_reset_source",
}


def _legacy_loss_keys_msg(keys: Sequence[str], *, context: str) -> str:
    keys_sorted = ", ".join(sorted({str(k) for k in keys}))
    return (
        f"[LegacyLossConfig] {context} contains removed keys: {keys_sorted}. "
        "Please remove them; MotionJointLoss now uses unified weights only "
        "(adaptive_bone_weights + unified_* knobs)."
    )


def _removed_trainbase_phase_reset_msg(keys: Sequence[str], *, context: str) -> str:
    keys_sorted = ", ".join(sorted({str(k) for k in keys}))
    return (
        f"[trainbase] {context} contains removed phase-reset keys: {keys_sorted}. "
        "train/training_MPL.py now hard-disables phase reset/event reset and always builds "
        "contact_phase_state with phase_reset_source='none' and contact_phase_state_event_kind='none'. "
        "Please remove these keys from trainbase configs/CLI. Posttrain/validate keep their own phase-reset controls."
    )


def _assert_no_legacy_loss_keys_in_schedule(schedule: Any, *, context: str) -> None:
    if not isinstance(schedule, Sequence):
        return
    for idx, stage in enumerate(schedule):
        if not isinstance(stage, Mapping):
            continue
        loss_cfg = stage.get("loss", {})
        if isinstance(loss_cfg, Mapping):
            hits = [k for k in LEGACY_LOSS_KEYS if k in loss_cfg]
            if hits:
                raise ValueError(_legacy_loss_keys_msg(hits, context=f"{context}.stage[{idx}].loss"))
        loss_groups = stage.get("loss_groups", {})
        if isinstance(loss_groups, Mapping):
            for group_name, group_cfg in loss_groups.items():
                if isinstance(group_cfg, Mapping):
                    hits = [k for k in LEGACY_LOSS_KEYS if k in group_cfg]
                    if hits:
                        raise ValueError(
                            _legacy_loss_keys_msg(
                                hits,
                                context=f"{context}.stage[{idx}].loss_groups[{group_name!r}]",
                            )
                        )


REMOVED_TRAINBASE_STAGE_PARAM_KEYS: tuple[str, ...] = (
    'freerun_horizon',
    'freerun_weight',
    'teacher_rot_noise_deg',
    'teacher_rot_noise_prob',
    'input_step_noise_prob',
    'input_noise_profile',
)

REMOVED_TRAINBASE_STAGE_ROOT_KEYS: tuple[str, ...] = (
    'teacher_rot_noise_deg_start',
    'teacher_rot_noise_deg_end',
    'teacher_rot_noise_prob_start',
    'teacher_rot_noise_prob_end',
)


def _assert_no_removed_trainbase_stage_keys(schedule: Any, *, context: str) -> None:
    if not isinstance(schedule, Sequence):
        return
    removed_hits: list[str] = []
    for idx, stage in enumerate(schedule):
        if not isinstance(stage, Mapping):
            continue
        for key in REMOVED_TRAINBASE_STAGE_ROOT_KEYS:
            if key in stage:
                removed_hits.append(f'{context}[{idx}].{key}')
        params = stage.get('params')
        if isinstance(params, Mapping):
            for key in REMOVED_TRAINBASE_STAGE_PARAM_KEYS:
                if key in params:
                    removed_hits.append(f'{context}[{idx}].params.{key}')
        trainer_cfg = stage.get('trainer')
        if isinstance(trainer_cfg, Mapping):
            for key in ('freerun_horizon', 'freerun_weight'):
                if key in trainer_cfg:
                    removed_hits.append(f'{context}[{idx}].trainer.{key}')
    if removed_hits:
        joined = ', '.join(removed_hits)
        raise ValueError(
            '[trainbase] the following stage-schedule keys were removed together with freerun-loss/noise branches: '
            f'{joined}. Keep `freerun_stage_schedule` only for TF/LR/history/direct-pose-trainability overrides.'
        )



import os, json, math, glob, time, argparse

from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable

class Trainer:
    def _parent_relative_matrices(self, R):
        fn = getattr(self.loss_fn, '_parent_relative_matrices', None)
        if callable(fn):
            try:
                return fn(R)
            except Exception:
                return R
        return R

    def _joint_weights(self, ref_tensor, joint_count):
        fn = getattr(self.loss_fn, '_joint_weight_vector', None)
        if callable(fn):
            try:
                return fn(ref_tensor.device, ref_tensor.dtype, joint_count)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _phasec_warn_once(
                    "joint_weights/fallback",
                    "loss_fn._joint_weight_vector failed; falling back to uniform weights",
                    exc,
                )
        import torch
        return torch.ones(joint_count, device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _resolve_contact_meas_runtime(self, x_raw) -> Optional[ContactMeasRuntime]:
        import torch

        if x_raw is None or (not torch.is_tensor(x_raw)):
            return None
        if x_raw.dim() == 3 and x_raw.size(1) == 1:
            x_raw = x_raw[:, 0]
        if x_raw.dim() != 2:
            return None

        model = getattr(self, "model", None)
        contact_dim = int(getattr(model, "contact_dim", 0) or 0) if model is not None else 0
        if contact_dim <= 0:
            return None

        x_layout = getattr(self, "_x_layout", None) or {}
        rot_slice = self._sl_from_layout(x_layout, "BoneRotations6D")
        if not isinstance(rot_slice, slice):
            return None
        rot_flat = x_raw[..., rot_slice]
        if rot_flat.numel() == 0 or (rot_flat.shape[-1] % 6 != 0):
            return None
        joint_count = int(rot_flat.shape[-1] // 6)

        parents = getattr(self.loss_fn, "parents", None)
        offsets = getattr(self.loss_fn, "bone_offsets", None)
        if not parents or offsets is None:
            return None
        if offsets.shape[0] < joint_count:
            return None

        root_slice = self._sl_from_layout(x_layout, "RootPosition")
        if isinstance(root_slice, slice) and (root_slice.stop - root_slice.start) == 3:
            root_pos = x_raw[..., root_slice]
        else:
            root_pos = x_raw.new_zeros((x_raw.shape[0], 3))

        up_axis = int(getattr(self, "eval_up_axis", getattr(self, "_up_axis", 2)))
        up_axis = 2 if up_axis not in (0, 1, 2) else up_axis
        return ContactMeasRuntime(
            x_raw=x_raw,
            contact_dim=contact_dim,
            rot_slice=rot_slice,
            rot_flat=rot_flat,
            joint_count=joint_count,
            parents=parents,
            offsets=offsets,
            root_pos=root_pos,
            up_axis=up_axis,
        )

    def _resolve_contact_meas_cfg(self) -> dict[str, Any]:
        cfg = getattr(self, "_contact_meas_cfg", None)
        if not isinstance(cfg, dict):
            meta = getattr(self.loss_fn, "meta", None)
            foot_evidence = (meta.get("foot_evidence") if isinstance(meta, dict) else {}) or {}
            sweep = (foot_evidence.get("sweep") if isinstance(foot_evidence, dict) else {}) or {}
            spec = (foot_evidence.get("soft_score_spec") if isinstance(foot_evidence, dict) else {}) or {}

            def _finite_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
                if not isinstance(mapping, Mapping):
                    return default
                try:
                    value = float(mapping.get(key, default))
                except Exception:
                    value = default
                return default if not math.isfinite(value) else value

            radius_cm = _finite_float(sweep, "sphere_radius_cm", 0.0)
            up_offset_cm = _finite_float(sweep, "up_offset_cm", 0.0)
            down_distance_cm = _finite_float(sweep, "down_distance_cm", 0.0)
            cfg = {
                "radius_m": max(0.0, radius_cm) / 100.0,
                "up_offset_m": max(0.0, up_offset_cm) / 100.0,
                "down_distance_m": max(0.0, down_distance_cm) / 100.0,
                "dist0_cm": max(0.0, _finite_float(spec, "dist0_cm", 0.5)),
                "alpha_dist": max(1e-6, _finite_float(spec, "alpha_dist", 2.0)),
                "vz0_cmps": max(1e-6, _finite_float(spec, "vz0_cmps", 40.0)),
                "alpha_vz": max(1e-6, _finite_float(spec, "alpha_vz", 0.5)),
                "vxy0_cmps": max(1e-6, _finite_float(spec, "vxy0_cmps", 96.0)),
                "alpha_vxy": max(1e-6, _finite_float(spec, "alpha_vxy", 0.2)),
                "gate_by_hit": bool(spec.get("gate_by_hit", True)) if isinstance(spec, dict) else True,
                "min_score": 1e-4,
                "max_score": 0.9,
                "scale": 0.92,
            }
            self._contact_meas_cfg = cfg

        gate_override = getattr(self, "contact_meas_gate_by_hit_override", None)
        if gate_override is not None:
            cfg["gate_by_hit"] = bool(gate_override)
        return cfg

    def _resolve_contact_meas_foot_indices(
        self,
        *,
        bone_names: Sequence[str],
        joint_count: int,
        contact_dim: int,
    ) -> Optional[list[int]]:
        foot_indices = getattr(self, "_contact_meas_foot_idxs", None)
        if not isinstance(foot_indices, (list, tuple)) or len(foot_indices) != contact_dim:
            foot_indices = None
        if foot_indices is not None:
            return [int(idx) for idx in foot_indices]

        resolved: list[int] = []
        name_to_idx = {name: idx for idx, name in enumerate(bone_names[:joint_count])} if bone_names else {}
        meta = getattr(self.loss_fn, "meta", None)
        if isinstance(meta, dict):
            markers = meta.get("foot_evidence", {}).get("markers")
            if isinstance(markers, str):
                for name in [item.strip() for item in markers.split(",") if item.strip()]:
                    idx = name_to_idx.get(name)
                    if isinstance(idx, int):
                        resolved.append(int(idx))
        for name in ("ball_l", "ball_r", "foot_l", "foot_r"):
            if len(resolved) >= contact_dim:
                break
            idx = name_to_idx.get(name)
            if isinstance(idx, int) and 0 <= idx < joint_count and idx not in resolved:
                resolved.append(int(idx))
        if len(resolved) != contact_dim:
            return None
        self._contact_meas_foot_idxs = list(resolved)
        return resolved

    def _contact_meas_whitebox(self, x_raw, prev_foot_pos=None):
        """
        White-box contacts_meas:
            pose (rot6d) -> FK -> foot height/velocity -> contact score

        Returns:
            contacts_meas: (B, C) or None
            foot_pos: (B, C, 3) detached for next-step velocity, or None
        """
        import torch
        log_wb = bool(getattr(self, "log_contacts_whitebox", False))
        debug_payload: Optional[Dict[str, Any]] = None
        runtime = self._resolve_contact_meas_runtime(x_raw)
        if runtime is None:
            if log_wb:
                self._contact_meas_whitebox_debug = None
            return None, prev_foot_pos
        bone_names_src = getattr(self.loss_fn, "bone_names", None) or getattr(self, "_bone_names", None)
        if not bone_names_src:
            meta = getattr(self.loss_fn, "meta", None)
            if isinstance(meta, dict):
                bone_names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
        bone_names = [str(name) for name in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []
        if runtime.joint_count > 0:
            bone_names = bone_names[:runtime.joint_count]
        foot_idxs = self._resolve_contact_meas_foot_indices(
            bone_names=bone_names,
            joint_count=runtime.joint_count,
            contact_dim=runtime.contact_dim,
        )
        if foot_idxs is None:
            if log_wb:
                self._contact_meas_whitebox_debug = None
            return None, prev_foot_pos
        cfg = self._resolve_contact_meas_cfg()

        x_raw = runtime.x_raw
        contact_dim = runtime.contact_dim
        rot_flat = runtime.rot_flat
        J = runtime.joint_count
        parents = runtime.parents
        offsets = runtime.offsets
        root_pos = runtime.root_pos
        up_axis = runtime.up_axis

        with torch.no_grad():
            from .geometry import fk_positions_from_rot6d, reproject_rot6d

            cols = getattr(self.loss_fn, "_rot6d_columns", ("X", "Z"))
            rot_proj = reproject_rot6d(rot_flat).view(x_raw.shape[0], J, 6)
            pos = fk_positions_from_rot6d(rot_proj, parents, offsets, root_pos=root_pos, columns=cols)  # (B,J,3)
            foot_pos = pos[:, torch.as_tensor(foot_idxs, device=pos.device, dtype=torch.long)]  # (B,C,3)

            has_prev = not (
                prev_foot_pos is None or (not torch.is_tensor(prev_foot_pos)) or prev_foot_pos.shape != foot_pos.shape
            )
            if not has_prev:
                vel = torch.zeros_like(foot_pos)
            else:
                vel = (foot_pos - prev_foot_pos.to(device=foot_pos.device, dtype=foot_pos.dtype)) * float(
                    getattr(self, "fps", 60.0) or 60.0
                )

            # Root velocity (for root-relative planar speed gating).
            # Important: only use it when prev_foot_pos is valid; otherwise treat as cold-start and reset.
            if not has_prev:
                root_vel = torch.zeros_like(root_pos)
            else:
                prev_root_pos = getattr(self, "_contact_meas_prev_root_pos", None)
                if prev_root_pos is None or (not torch.is_tensor(prev_root_pos)) or prev_root_pos.shape != root_pos.shape:
                    root_vel = torch.zeros_like(root_pos)
                else:
                    root_vel = (root_pos - prev_root_pos.to(device=root_pos.device, dtype=root_pos.dtype)) * float(
                        getattr(self, "fps", 60.0) or 60.0
                    )

            planar_axes = [0, 1, 2]
            planar_axes.remove(up_axis)
            vel_xy = vel[..., planar_axes]  # (B,C,2)
            root_vel_xy = root_vel[..., planar_axes]  # (B,2)
            vxy_abs_mps = vel_xy.norm(dim=-1)
            vxy_rel_mps = (vel_xy - root_vel_xy.unsqueeze(-2)).norm(dim=-1)
            vz_mps = vel[..., up_axis].abs()

            # Use abs speed by default; root-relative speed is more robust under global translation.
            vxy_mode = getattr(self, "contact_meas_vxy_mode", None)
            if vxy_mode is None and isinstance(cfg, dict):
                vxy_mode = cfg.get("vxy_mode", None)
            vxy_mode = str(vxy_mode or "abs").strip().lower()
            if vxy_mode in ("root", "root_rel", "root-relative", "rel", "relative"):
                vxy_mps_used = vxy_rel_mps
                vxy_mode = "root_rel"
            else:
                vxy_mps_used = vxy_abs_mps
                vxy_mode = "abs"

            # FootEvidence-style soft score uses cm / cmps.
            vz_cmps = vz_mps * 100.0
            vxy_abs_cmps = vxy_abs_mps * 100.0
            vxy_rel_cmps = vxy_rel_mps * 100.0
            vxy_cmps = vxy_mps_used * 100.0
            root_vxy_cmps = root_vel_xy.norm(dim=-1) * 100.0

            # Velocity scores (also used for stance-conditioned ground selection).
            vz0_cmps = float(cfg.get("vz0_cmps", 40.0) or 40.0)
            vz0_cmps = max(1e-6, vz0_cmps)
            alpha_vz = float(cfg.get("alpha_vz", 0.5) or 0.5)
            if (not math.isfinite(alpha_vz)) or alpha_vz <= 0.0:
                alpha_vz = 0.5
            denom_vz = max(1e-6, alpha_vz * vz0_cmps)
            vz_score = torch.exp(-torch.relu(vz_cmps - vz0_cmps) / denom_vz)

            vxy0_cmps = float(cfg.get("vxy0_cmps", 96.0) or 96.0)
            vxy0_cmps = max(1e-6, vxy0_cmps)
            alpha_vxy = float(cfg.get("alpha_vxy", 0.2) or 0.2)
            if (not math.isfinite(alpha_vxy)) or alpha_vxy <= 0.0:
                alpha_vxy = 0.2
            denom_vxy = max(1e-6, alpha_vxy * vxy0_cmps)
            vxy_score = torch.exp(-torch.relu(vxy_cmps - vxy0_cmps) / denom_vxy)

            radius_m = float(cfg.get("radius_m", 0.0) or 0.0)
            bottom_z = foot_pos[..., up_axis] - radius_m  # (B, C)
            # Choose the most stance-like foot (low planar+vertical speed) to estimate ground_z_now.
            stance_score = (vxy_score * vz_score).detach()  # higher => more likely stance
            idx = stance_score.argmax(dim=-1)  # (B,)
            ground_z_now = bottom_z.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (B,)
            ground_z_prev = getattr(self, "_contact_meas_ground_z", None)
            mode = str(getattr(self, "contact_meas_ground_z_mode", "window") or "window").strip().lower()
            if mode not in ("ema", "window", "slew"):
                mode = "window"
            prev_ok = (
                torch.is_tensor(ground_z_prev)
                and ground_z_prev.shape == ground_z_now.shape
                and (prev_foot_pos is not None)
            )
            if not prev_ok:
                ground_z = ground_z_now
                # Reset history buffer for window mode (avoid cross-clip leakage).
                if mode == "window":
                    win = int(getattr(self, "contact_meas_ground_z_window", 5) or 5)
                    win = max(1, win)
                    if win > 1:
                        self._contact_meas_ground_z_hist = ground_z_now.unsqueeze(-1).repeat(1, win).detach()
                    else:
                        self._contact_meas_ground_z_hist = None
            else:
                ground_z_prev = ground_z_prev.to(device=ground_z_now.device, dtype=ground_z_now.dtype)
                ground_z_cand = ground_z_now
                if mode == "ema":
                    beta = float(getattr(self, "contact_meas_ground_z_beta", 0.05) or 0.05)
                    if (not math.isfinite(beta)) or beta <= 0.0:
                        beta = 0.05
                    beta = min(1.0, beta)
                    ground_z_cand = ground_z_prev + beta * (ground_z_now - ground_z_prev)
                elif mode == "window":
                    win = int(getattr(self, "contact_meas_ground_z_window", 5) or 5)
                    win = max(1, win)
                    q = float(getattr(self, "contact_meas_ground_z_quantile", 0.2) or 0.2)
                    if (not math.isfinite(q)) or q < 0.0:
                        q = 0.0
                    q = min(1.0, q)
                    hist = getattr(self, "_contact_meas_ground_z_hist", None)
                    if (not torch.is_tensor(hist)) or hist.shape != (ground_z_now.shape[0], win):
                        hist = ground_z_prev.unsqueeze(-1).repeat(1, win)
                    else:
                        hist = hist.to(device=ground_z_now.device, dtype=ground_z_now.dtype)
                    if win > 1:
                        hist = torch.roll(hist, shifts=-1, dims=-1)
                        hist[..., -1] = ground_z_now
                        self._contact_meas_ground_z_hist = hist.detach()
                    try:
                        # Robust low-quantile over the last `win` observations (ignore single downward spikes).
                        vals = hist.sort(dim=-1).values
                        idx = int(math.ceil(q * float(win - 1)))
                        idx = max(0, min(win - 1, idx))
                        ground_z_cand = vals[..., idx]
                    except Exception:
                        ground_z_cand = ground_z_prev
                else:
                    # mode == "slew": cand is now, then apply rate limits below.
                    ground_z_cand = ground_z_now

                max_down = getattr(self, "contact_meas_ground_z_max_down_m", None)
                max_up = getattr(self, "contact_meas_ground_z_max_up_m", None)
                try:
                    max_down_m = float(max_down) if max_down is not None else 0.0
                except Exception:
                    max_down_m = 0.0
                try:
                    max_up_m = float(max_up) if max_up is not None else 0.0
                except Exception:
                    max_up_m = 0.0
                if mode == "slew" and max_down_m <= 0.0 and max_up_m <= 0.0:
                    # Reasonable defaults for a first ablation: allow recovery but prevent one-step explosions.
                    max_down_m = 0.01  # 1cm / frame
                    max_up_m = 0.002   # 0.2cm / frame
                max_down_m = max(0.0, max_down_m)
                max_up_m = max(0.0, max_up_m)
                if max_down_m > 0.0 or max_up_m > 0.0:
                    delta = (ground_z_cand - ground_z_prev).clamp(-max_down_m, max_up_m)
                    ground_z = ground_z_prev + delta
                else:
                    ground_z = ground_z_cand

            self._contact_meas_ground_z = ground_z.detach()

            dist_to_ground_m = (bottom_z - ground_z.unsqueeze(-1)).clamp_min(0.0)

            # Gate by sweep hit (matches JSON behavior: hit_flag=0 => contact=0).
            start_z = None
            sweep_target_z = None
            hit_flag = None
            if bool(cfg.get("gate_by_hit", True)):
                up_off = float(cfg.get("up_offset_m", 0.0) or 0.0)
                down_dist = float(cfg.get("down_distance_m", 0.0) or 0.0)
                start_z = foot_pos[..., up_axis] + up_off
                # Sphere-sweep hits the ground when the *center* crosses (ground_z + radius).
                sweep_target_z = ground_z.unsqueeze(-1) + radius_m
                hit_flag = (start_z >= sweep_target_z) & ((start_z - down_dist) <= sweep_target_z)

            # FootEvidence-style soft score in cm / cmps.
            # A robust, explainable sensor: contact is high when the foot is close to the estimated ground plane
            # and the foot is not moving too fast (esp. vertical velocity).
            #
            # We intentionally use a hinge-style velocity gate (no penalty below threshold) to keep the meas signal
            # sharp and interpretable; thresholds/softness are taken from meta['foot_evidence']['soft_score_spec'].
            dist_cm = dist_to_ground_m * 100.0

            dist0_cm = float(cfg.get("dist0_cm", 0.5) or 0.5)
            dist0_cm = max(1e-6, dist0_cm)
            alpha_dist = float(cfg.get("alpha_dist", 2.0) or 2.0)
            if (not math.isfinite(alpha_dist)) or alpha_dist <= 0.0:
                alpha_dist = 2.0
            # Normalize so dist_score==1 at dist=0 (i.e., on the ground) regardless of alpha_dist.
            # dist_raw in (0,1) with dist_raw(dist=0)=sigmoid(alpha_dist).
            dist_raw = torch.sigmoid((alpha_dist * (dist0_cm - dist_cm)) / dist0_cm)
            dist_raw_max = 1.0 / (1.0 + math.exp(-alpha_dist))  # sigmoid(alpha_dist)
            dist_score = (dist_raw / max(1e-6, float(dist_raw_max))).clamp(0.0, 1.0)

            contacts_meas = dist_score * vz_score * vxy_score
            scale = float(cfg.get("scale", 1.0) or 1.0)
            if math.isfinite(scale) and scale > 0.0:
                contacts_meas = contacts_meas * scale
            if hit_flag is not None:
                contacts_meas = contacts_meas * hit_flag.to(dtype=contacts_meas.dtype)

            contacts_meas = contacts_meas.clamp(0.0, float(cfg.get("max_score", 1.0) or 1.0))
            min_score = float(cfg.get("min_score", 0.0) or 0.0)
            if min_score > 0.0:
                if hit_flag is not None:
                    contacts_meas = torch.where(hit_flag, contacts_meas.clamp_min(min_score), contacts_meas)
                else:
                    contacts_meas = contacts_meas.clamp_min(min_score)

            if log_wb:
                try:
                    def _mean_list(x: torch.Tensor | None) -> list[float] | None:
                        if x is None or (not torch.is_tensor(x)):
                            return None
                        if x.ndim == 1:
                            return [float(x.detach().mean().item())]
                        if x.ndim == 2:
                            return x.detach().mean(dim=0).cpu().tolist()
                        # Fallback: flatten per-sample then mean over batch.
                        flat = x.detach().reshape(x.shape[0], -1)
                        return flat.mean(dim=0).cpu().tolist()

                    foot_pos_z = foot_pos[..., up_axis]
                    foot_names = None
                    if bone_names:
                        foot_names = [bone_names[int(i)] if int(i) < len(bone_names) else str(int(i)) for i in foot_idxs]
                    wb_cfg = {}
                    try:
                        for k, v in (cfg or {}).items():
                            if isinstance(v, bool):
                                wb_cfg[str(k)] = bool(v)
                            else:
                                wb_cfg[str(k)] = float(v)
                    except Exception:
                        wb_cfg = None

                    wb_debug = {
                        "UpAxis": int(up_axis),
                        "Batch": int(foot_pos.shape[0]),
                        "ContactDim": int(contact_dim),
                        "FootIdxs": [int(i) for i in foot_idxs],
                        "FootNames": foot_names,
                        "VxyMode": str(vxy_mode),
                        "GroundZSelect": "stance",
                        "Cfg": wb_cfg,
                        "GroundZNowMean": float(ground_z_now.detach().mean().item()) if torch.is_tensor(ground_z_now) else None,
                        "GroundZMean": float(ground_z.detach().mean().item()) if torch.is_tensor(ground_z) else None,
                        "GroundZPrevMean": float(ground_z_prev.detach().mean().item()) if torch.is_tensor(ground_z_prev) else None,
                        "FootPosZMean": _mean_list(foot_pos_z),
                        "BottomZMean": _mean_list(bottom_z),
                        "DistCmMean": _mean_list(dist_cm),
                        "VzCmpsMean": _mean_list(vz_cmps),
                        "RootVxyCmpsMean": _mean_list(root_vxy_cmps),
                        "VxyCmpsMean": _mean_list(vxy_cmps),
                        "VxyAbsCmpsMean": _mean_list(vxy_abs_cmps),
                        "VxyRelCmpsMean": _mean_list(vxy_rel_cmps),
                        "StartZMean": _mean_list(start_z),
                        "SweepTargetZMean": _mean_list(sweep_target_z),
                        "HitRate": _mean_list(hit_flag.to(dtype=contacts_meas.dtype)) if hit_flag is not None else None,
                        "DistScoreMean": _mean_list(dist_score),
                        "VzScoreMean": _mean_list(vz_score),
                        "VxyScoreMean": _mean_list(vxy_score),
                        "MeasMean": _mean_list(contacts_meas),
                    }
                    debug_payload = wb_debug
                except Exception:
                    debug_payload = None

        # Cache root_pos for the next-step velocity estimate (kept in Trainer state).
        self._contact_meas_prev_root_pos = root_pos.detach() if torch.is_tensor(root_pos) else root_pos
        if log_wb:
            self._contact_meas_whitebox_debug = debug_payload
        return contacts_meas, foot_pos.detach()

    def _predict_pretrain_contacts_from_frozen(
        self,
        *,
        motion_step_t: Optional[torch.Tensor],
        pose_hist_step_t: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        import torch

        if not torch.is_tensor(motion_step_t):
            return None
        model = getattr(self, 'model', None)
        if model is None:
            return None
        enc = getattr(model, 'frozen_encoder', None)
        head = getattr(model, 'frozen_contact_head', None)
        if enc is None or head is None:
            return None
        try:
            cdim = int(getattr(model, 'contact_dim', 0) or 0)
            in_dim = int(getattr(model, 'encoder_input_dim', 0) or 0)
            if cdim <= 0 or in_dim <= 0:
                return None

            batch_size = int(motion_step_t.shape[0])
            device = motion_step_t.device
            dtype = motion_step_t.dtype
            contacts_seed = torch.zeros((batch_size, cdim), device=device, dtype=dtype)

            angvel_slice = getattr(self, 'angvel_x_slice', None)
            if not isinstance(angvel_slice, slice):
                angvel_slice = getattr(model, '_contact_meas_state_angvel_slice', None)
            if isinstance(angvel_slice, slice):
                try:
                    angvel_t = motion_step_t[..., angvel_slice]
                except Exception:
                    angvel_t = None
            else:
                angvel_t = None
            if not torch.is_tensor(angvel_t):
                angvel_t = torch.zeros((batch_size, 0), device=device, dtype=dtype)
            elif angvel_t.ndim != 2:
                angvel_t = angvel_t.reshape(batch_size, -1)

            if torch.is_tensor(pose_hist_step_t):
                pose_hist_t = pose_hist_step_t.to(device=device, dtype=dtype)
                if pose_hist_t.ndim == 3 and int(pose_hist_t.size(1)) == 1:
                    pose_hist_t = pose_hist_t[:, 0]
                elif pose_hist_t.ndim != 2:
                    pose_hist_t = pose_hist_t.reshape(batch_size, -1)
            else:
                pose_hist_t = torch.zeros((batch_size, 0), device=device, dtype=dtype)

            encoder_input = torch.cat([contacts_seed, angvel_t, pose_hist_t], dim=-1)
            if int(encoder_input.shape[-1]) != int(in_dim):
                if int(encoder_input.shape[-1]) > int(in_dim):
                    encoder_input = encoder_input[..., : int(in_dim)]
                else:
                    encoder_input = F.pad(encoder_input, (0, int(in_dim) - int(encoder_input.shape[-1])))

            pre_clamp_raw = getattr(
                self,
                'trainbase_contacts_pretrain_clamp',
                getattr(self, 'posttrain_contacts_pretrain_clamp', 1.0),
            )
            try:
                pre_clamp = float(pre_clamp_raw or 0.0)
            except Exception:
                pre_clamp = 1.0
            if _math.isfinite(float(pre_clamp)) and float(pre_clamp) > 0.0:
                encoder_input = encoder_input.clamp(-float(pre_clamp), float(pre_clamp))

            affine_cfg = getattr(
                self,
                'trainbase_contacts_pretrain_affine',
                getattr(self, 'posttrain_contacts_pretrain_affine', None),
            )

            with torch.no_grad():
                hidden = enc(encoder_input.unsqueeze(1), return_summary=False)
                logits = head(hidden)
                if torch.is_tensor(logits) and logits.ndim == 3 and int(logits.size(1)) == 1:
                    logits = logits[:, 0]
                if (not torch.is_tensor(logits)) or logits.ndim != 2:
                    return None
                if isinstance(affine_cfg, dict):
                    scale = affine_cfg.get('scale', None)
                    bias = affine_cfg.get('bias', None)
                    try:
                        eps = float(affine_cfg.get('eps', 1e-4) or 1e-4)
                    except Exception:
                        eps = 1e-4
                    if not _math.isfinite(float(eps)):
                        eps = 1e-4
                    eps = float(min(1e-2, max(1e-8, eps)))
                    channels = int(logits.shape[-1])
                    if (
                        isinstance(scale, (list, tuple))
                        and isinstance(bias, (list, tuple))
                        and int(len(scale)) == channels
                        and int(len(bias)) == channels
                    ):
                        scale_t = torch.tensor([float(x) for x in scale], device=device, dtype=logits.dtype).view(1, channels)
                        bias_t = torch.tensor([float(x) for x in bias], device=device, dtype=logits.dtype).view(1, channels)
                        probs = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
                        logits = bias_t + scale_t * (torch.log(probs) - torch.log1p(-probs))
                probs = torch.sigmoid(logits)
                if int(probs.shape[-1]) != int(cdim):
                    if int(probs.shape[-1]) > int(cdim):
                        probs = probs[..., : int(cdim)]
                    else:
                        probs = F.pad(probs, (0, int(cdim) - int(probs.shape[-1])))
                return probs
        except Exception:
            return None

    def _prepare_pose_hist_state(
        self,
        state_seq: torch.Tensor,
        pose_hist_seq: Optional[torch.Tensor],
        y_raw_local: Optional[torch.Tensor],
        rot6d_y_slice: slice,
    ) -> PoseHistState:
        return init_pose_hist_state(
            ref_tensor=state_seq,
            pose_hist_seq=pose_hist_seq,
            y_prev_raw=y_raw_local,
            rot_slice=rot6d_y_slice,
            pose_hist_len=int(getattr(self, "pose_hist_len", 0) or 0),
            pose_hist_dim=int(getattr(self, "pose_hist_dim", 0) or 0),
            params_fn=self._pose_hist_params,
            force_disable=bool(getattr(self, "force_pose_hist_seq", False)),
        )

    def _resolve_rollout_step_inputs(self, context: Any) -> Any:
        import torch

        step_idx = int(context.step_idx)
        cond_input = context.cond_seq[:, step_idx] if context.has_time_dim['cond'] else context.cond_seq
        prev_foot_pos_meas = context.prev_foot_pos_meas

        angvel_slice = getattr(self, "angvel_x_slice", None)
        if bool(getattr(self, "use_freerun_state_sync", False)) and isinstance(angvel_slice, slice):
            angvel_t = context.motion[..., angvel_slice].detach()
        else:
            angvel_t = context.angvel_seq[:, step_idx] if context.has_time_dim['angvel'] else context.angvel_seq

        pose_history_t = resolve_pose_hist_input(
            state=context.pose_hist_state,
            pose_hist_seq=context.pose_hist_seq,
            idx=step_idx,
        )

        contacts_in_t = None
        if context.plan_enable:
            contacts_source = str(getattr(self, 'trainbase_contacts_source', 'whitebox') or 'whitebox').strip().lower()
            if contacts_source == 'pretrain_contact':
                contacts_in_t = self._predict_pretrain_contacts_from_frozen(
                    motion_step_t=context.motion,
                    pose_hist_step_t=pose_history_t,
                )
                if contacts_in_t is None:
                    raise RuntimeError(
                        '[FATAL] trainbase_contacts_source=pretrain_contact requires valid frozen encoder+contact_head '
                        'and runtime-compatible encoder input dimensions.'
                    )
            else:
                try:
                    contacts_in_t, prev_foot_pos_meas = self._contact_meas_whitebox(
                        context.motion_raw_local,
                        prev_foot_pos_meas,
                    )
                except Exception:
                    contacts_in_t = None

        cond_raw_t = None
        if context.cond_raw_seq is not None:
            if context.has_time_dim['cond_raw']:
                cond_idx = min(context.cond_raw_seq.shape[1] - 1, max(0, step_idx + 1))
                cond_raw_t = context.cond_raw_seq[:, cond_idx]
            elif torch.is_tensor(context.cond_raw_seq):
                cond_raw_t = context.cond_raw_seq
            else:
                cond_raw_t = context.cond_raw_seq

        cond_raw_for_env = cond_raw_t
        cond_raw_for_model = cond_raw_t
        reprojection_applied = False
        if context.enable_reprojection and step_idx > 0 and context.mode in ('free', 'train_free', 'mixed') and cond_raw_t is not None:
            gt_yaw = None
            if context.gt_seq is not None and context.has_time_dim.get('cond_raw'):
                gt_idx = min(context.gt_seq.shape[1] - 1, step_idx)
                gt_raw = self._denorm(context.gt_seq[:, gt_idx])
                gt_yaw = self._infer_root_yaw_from_rot6d(gt_raw)
            elif context.state_seq is not None:
                state_raw = self.normalizer.denorm_x(context.state_seq[:, step_idx], prev_raw=context.motion_raw_local)
                gt_yaw = self._infer_root_yaw_from_rot6d(state_raw)

            pred_yaw = self._infer_root_yaw_from_rot6d(context.y_raw_local) if context.y_raw_local is not None else None
            if gt_yaw is not None and pred_yaw is not None:
                cond_raw_t_reprojected = self._reproject_cond_to_local_frame(cond_raw_t, gt_yaw, pred_yaw)
                if cond_raw_t_reprojected is not None:
                    cond_raw_for_model = cond_raw_t_reprojected
                    reprojection_applied = True

        if cond_raw_for_model is not None:
            cond_override = self._normalize_cond_from_raw(
                cond_raw_for_model,
                context.cond_norm_mu,
                context.cond_norm_std,
            )
            if cond_override is not None:
                cond_input = cond_override
        if cond_input is None and context.cond_seq is not None:
            cond_input = context.cond_seq[:, step_idx] if context.has_time_dim['cond'] else context.cond_seq

        time_index_t = None
        if context.time_base_local is not None:
            try:
                time_index_t = context.time_base_local + step_idx
            except Exception:
                time_index_t = None

        rollout_step_t = None
        try:
            if int(context.total_steps) > 1:
                step_norm = float(step_idx) / float(int(context.total_steps) - 1)
            else:
                step_norm = 0.0
            rollout_step_t = torch.full(
                (context.motion.shape[0], 1, 1),
                step_norm,
                device=context.motion.device,
                dtype=context.motion.dtype,
            )
        except Exception:
            rollout_step_t = None

        return SimpleNamespace(
            cond_input=cond_input,
            contacts_in_t=contacts_in_t,
            angvel_t=angvel_t,
            pose_history_t=pose_history_t,
            cond_raw_for_env=cond_raw_for_env,
            prev_foot_pos_meas=prev_foot_pos_meas,
            time_index_t=time_index_t,
            rollout_step_t=rollout_step_t,
            reprojection_applied=reprojection_applied,
        )

    def _update_rollout_carry_state(self, request: Any) -> Any:
        import torch

        if request.motion_raw_local is None:
            self._raise_norm_error("rollout 更新需要 DataNormalizer 提供 RAW 状态写回。")
        free_raw = self._apply_free_carry(
            request.motion_raw_local,
            request.y_raw,
            cond_next_raw=request.cond_raw_for_env,
        )
        if request.allow_grad:
            free_raw = free_raw.clone()
        else:
            free_raw = free_raw.detach()
        free_z = self._diag_norm_x(free_raw)
        gt_next = request.state_seq[:, request.step_idx + 1]
        ss_sel_hold = request.ss_sel_hold
        if ss_sel_hold is None or request.ss_chunk_len <= 1 or (request.step_idx % request.ss_chunk_len == 0):
            ss_sel_hold = (torch.rand(request.batch_size, device=self.device) < float(request.tf_ratio)).float().unsqueeze(-1)
        sel = ss_sel_hold
        if sel.dtype != gt_next.dtype:
            sel = sel.to(gt_next.dtype)
        motion = sel * gt_next + (1.0 - sel) * free_z

        try:
            gt_raw_next = self.normalizer.denorm_x(gt_next, prev_raw=request.motion_raw_local)
            motion_raw_local = sel * gt_raw_next + (1.0 - sel) * free_raw
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在 rollout 更新时失败", exc)

        y_raw_local = request.y_raw_local
        if request.gt_seq is not None and request.gt_seq.dim() == 3:
            try:
                y_next = self._denorm(request.gt_seq[:, request.step_idx + 1]).detach()
                y_raw_local = sel * y_next + (1.0 - sel) * y_raw_local
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                _phasec_warn_once(
                    "rollout/carry_gt_denorm",
                    "failed to blend GT RAW carry in scheduled sampling; keeping model RAW carry",
                    exc,
                )

        pose_hist_state = advance_pose_hist_state(
            request.pose_hist_state,
            y_next_raw=y_raw_local,
            rot_slice=request.rot6d_y_slice,
        )
        return SimpleNamespace(
            motion=motion,
            motion_raw_local=motion_raw_local,
            y_raw_local=y_raw_local,
            ss_sel_hold=ss_sel_hold,
            pose_hist_state=pose_hist_state,
        )

    def _init_rollout_state(
        self,
        rollout_inputs: RolloutSequenceInputs,
        *,
        cond_norm_mu: Optional[torch.Tensor] = None,
        cond_norm_std: Optional[torch.Tensor] = None,
        mode: str = 'mixed',
        tf_ratio: float = 1.0,
        time_base: Any = None,
    ) -> RolloutExecutionState:
        state_seq = rollout_inputs.state_seq
        gt_seq = rollout_inputs.gt_seq
        batch_size, total_steps, _ = state_seq.shape
        allow_grad = mode == 'train_free'

        try:
            ss_chunk_len = int(getattr(self, 'ss_chunk_len', 1) or 1)
        except Exception:
            ss_chunk_len = 1
        ss_chunk_len = max(1, ss_chunk_len)

        motion = state_seq[:, 0]
        try:
            motion_raw_local = self.normalizer.denorm_x(motion)
        except Exception as exc:
            self._raise_norm_error('normalizer.denorm_x 在 roll-out 初始化时失败', exc)

        y_raw_local = None
        dy = None
        if gt_seq is not None and gt_seq.dim() == 3:
            y0 = gt_seq[:, 0]
            dy = y0.shape[-1]
            y_raw_local = self._denorm(y0)
        if dy is None:
            dy = int(getattr(self, 'Dy', 0) or (gt_seq.shape[-1] if gt_seq is not None else 0))

        rot6d_slice = getattr(self.train_loader, 'rot6d_x_slice', None) if hasattr(self, 'train_loader') else None
        if rot6d_slice is None:
            rot6d_slice = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(rot6d_slice, slice):
            rot6d_slice = slice(0, motion.size(-1))

        if y_raw_local is None and motion_raw_local is not None and dy:
            slice_len = rot6d_slice.stop - rot6d_slice.start
            if slice_len != dy:
                self._raise_norm_error('rollout 初始化 rot6d_x_slice 与 Dy 不匹配。')
            y_raw_local = motion_raw_local[..., rot6d_slice].clone()
        if y_raw_local is None and dy:
            joint_count = dy // 6
            if joint_count > 0:
                zeros = state_seq.new_zeros((batch_size, joint_count, 6))
                y_raw_local = _rot6d_identity_like(zeros).view(batch_size, dy)

        has_time_dim = {
            'cond': callable(getattr(rollout_inputs.cond_seq, 'dim', None)) and rollout_inputs.cond_seq.dim() == 3,
            'cond_raw': callable(getattr(rollout_inputs.cond_raw_seq, 'dim', None)) and getattr(rollout_inputs.cond_raw_seq, 'dim', lambda: 0)() == 3,
            'contacts': callable(getattr(rollout_inputs.contacts_seq, 'dim', None)) and rollout_inputs.contacts_seq.dim() == 3,
            'angvel': callable(getattr(rollout_inputs.angvel_seq, 'dim', None)) and rollout_inputs.angvel_seq.dim() == 3,
            'pose_hist': callable(getattr(rollout_inputs.pose_hist_seq, 'dim', None)) and rollout_inputs.pose_hist_seq.dim() == 3,
        }
        amp_enabled = bool(getattr(self, 'use_amp', False))
        rot6d_y_slice = getattr(self, 'rot6d_y_slice', None) or rot6d_slice
        pose_hist_state = self._prepare_pose_hist_state(
            state_seq,
            rollout_inputs.pose_hist_seq,
            y_raw_local,
            rot6d_y_slice,
        )

        cond_norm_mu = self._prepare_cond_stat(cond_norm_mu, state_seq) if cond_norm_mu is not None else None
        cond_norm_std = self._prepare_cond_stat(cond_norm_std, state_seq) if cond_norm_std is not None else None

        enable_reprojection = bool(getattr(self, 'enable_cond_reprojection', True))
        yaw_strategy = str(getattr(self, 'freerun_yaw_strategy', 'trajectory') or 'trajectory')
        if yaw_strategy == 'trajectory':
            enable_reprojection = False

        time_base_local = time_base
        if torch.is_tensor(time_base_local):
            try:
                time_base_local = time_base_local.to(device=state_seq.device)
            except Exception:
                time_base_local = time_base

        return RolloutExecutionState(
            batch_size=batch_size,
            total_steps=total_steps,
            mode=mode,
            allow_grad=allow_grad,
            tf_ratio=float(tf_ratio),
            ss_chunk_len=ss_chunk_len,
            amp_enabled=amp_enabled,
            rot6d_slice=rot6d_slice,
            rot6d_y_slice=rot6d_y_slice,
            has_time_dim=has_time_dim,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            enable_reprojection=enable_reprojection,
            plan_enable=bool(getattr(self.model, 'contact_plan_enable', False)),
            time_base_local=time_base_local,
            motion=motion,
            motion_raw_local=motion_raw_local,
            y_raw_local=y_raw_local,
            pose_hist_state=pose_hist_state,
        )

    def _get_rollout_step_tensor(
        self,
        ret: Mapping[str, Any],
        key: str,
    ) -> Optional[torch.Tensor]:
        value = ret.get(key, None)
        if not torch.is_tensor(value):
            return None
        if value.dim() == 2:
            return value.unsqueeze(1)
        if value.dim() >= 3 and value.size(1) != 1:
            return value[:, -1:, ...]
        return value

    def _update_rollout_plan_state(
        self,
        rollout: RolloutExecutionState,
        ret: Mapping[str, Any],
    ) -> None:
        if not rollout.plan_enable:
            return

        contacts_plan = self._get_rollout_step_tensor(ret, 'contacts_plan')
        if contacts_plan is not None:
            rollout.buffers.contacts_plan.append(contacts_plan)

        contacts_plan_logits = self._get_rollout_step_tensor(ret, 'contacts_plan_logits')
        if contacts_plan_logits is not None:
            rollout.buffers.contacts_plan_logits.append(contacts_plan_logits)

        direct_out = self._get_rollout_step_tensor(ret, 'out_direct')
        if direct_out is not None:
            rollout.buffers.out_direct.append(direct_out)

        for source_key, target_attr in (
            ('plan_z_next', 'plan_z'),
            ('phase_z_next', 'phase_z'),
            ('phase_event_age_next', 'phase_event_age'),
        ):
            value = ret.get(source_key, None)
            if value is None:
                continue
            if torch.is_tensor(value) and not rollout.allow_grad:
                value = value.detach()
            setattr(rollout, target_attr, value)

    def _record_rollout_step_outputs(
        self,
        rollout: RolloutExecutionState,
        *,
        y_norm: torch.Tensor,
        delta_out: torch.Tensor,
        period_pred: Optional[torch.Tensor],
        ret: Mapping[str, Any],
    ) -> None:
        rollout.outs.append(y_norm)
        rollout.delta_preds.append(delta_out)
        if period_pred is not None:
            rollout.buffers.period_pred.append(period_pred)

        hidden_step = ret.get('h_final', None)
        if torch.is_tensor(hidden_step):
            if hidden_step.dim() == 1:
                hidden_step = hidden_step.unsqueeze(0).unsqueeze(0)
            elif hidden_step.dim() == 2:
                hidden_step = hidden_step.unsqueeze(1)
            elif hidden_step.dim() >= 3 and hidden_step.size(1) != 1:
                hidden_step = hidden_step[:, -1:, ...]
            rollout.buffers.hidden_seq.append(hidden_step)

        for key, target in (
            ('contacts_meas', rollout.buffers.contacts_meas),
            ('contacts_err', rollout.buffers.contacts_err),
            ('event_clock_lambda_logit', rollout.buffers.event_clock_lambda_logit),
            ('event_clock_dynamic_prior', rollout.buffers.event_clock_dynamic_prior),
            ('event_clock_delta_z', rollout.buffers.event_clock_delta_z),
        ):
            value = self._get_rollout_step_tensor(ret, key)
            if value is not None:
                target.append(value)

    def _compute_rollout_step_debug_stats(
        self,
        *,
        delta_out: torch.Tensor,
        y_raw: torch.Tensor,
        prev_raw_snapshot: Optional[torch.Tensor],
        pred_raw_local: Optional[torch.Tensor],
        gt_seq: Optional[torch.Tensor],
        step_idx: int,
    ) -> Dict[str, Optional[float]]:
        debug_stats: Dict[str, Optional[float]] = {'rot6d_geo_deg': None}
        try:
            debug_stats['delta_norm_abs_mean'] = float(delta_out.abs().mean().item())
        except Exception:
            debug_stats['delta_norm_abs_mean'] = None
        try:
            if prev_raw_snapshot is not None:
                delta_raw = y_raw - prev_raw_snapshot
                debug_stats['delta_raw_abs_mean'] = float(delta_raw.abs().mean().item())
            else:
                debug_stats['delta_raw_abs_mean'] = None
        except Exception:
            debug_stats['delta_raw_abs_mean'] = None

        if torch.is_tensor(gt_seq) and gt_seq.dim() == 3 and pred_raw_local is not None:
            try:
                gt_frame = self._denorm(gt_seq[:, min(step_idx + 1, gt_seq.shape[1] - 1)])
                rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
                if isinstance(rot_slice, slice):
                    pred_block = pred_raw_local[:, rot_slice].reshape(pred_raw_local.shape[0], -1, 6)
                    gt_block = gt_frame[:, rot_slice].reshape(gt_frame.shape[0], -1, 6)
                    pred_m = rot6d_to_matrix(reproject_rot6d(pred_block))
                    gt_m = rot6d_to_matrix(reproject_rot6d(gt_block))
                    geo_diff = geodesic_R(pred_m, gt_m) * (180.0 / _math.pi)
                    debug_stats['rot6d_geo_deg'] = float(geo_diff.mean().item())
            except Exception:
                debug_stats['rot6d_geo_deg'] = None
        return debug_stats

    def _rollout_forward_step(
        self,
        rollout: RolloutExecutionState,
        rollout_inputs: RolloutSequenceInputs,
        *,
        step_idx: int,
    ) -> Dict[str, Optional[float]]:
        step_inputs = self._resolve_rollout_step_inputs(
            SimpleNamespace(
                step_idx=step_idx,
                total_steps=rollout.total_steps,
                motion=rollout.motion,
                motion_raw_local=rollout.motion_raw_local,
                y_raw_local=rollout.y_raw_local,
                state_seq=rollout_inputs.state_seq,
                gt_seq=rollout_inputs.gt_seq,
                cond_seq=rollout_inputs.cond_seq,
                cond_raw_seq=rollout_inputs.cond_raw_seq,
                contacts_seq=rollout_inputs.contacts_seq,
                angvel_seq=rollout_inputs.angvel_seq,
                pose_hist_seq=rollout_inputs.pose_hist_seq,
                cond_norm_mu=rollout.cond_norm_mu,
                cond_norm_std=rollout.cond_norm_std,
                has_time_dim=rollout.has_time_dim,
                pose_hist_state=rollout.pose_hist_state,
                plan_enable=rollout.plan_enable,
                mode=rollout.mode,
                enable_reprojection=rollout.enable_reprojection,
                time_base_local=rollout.time_base_local,
                prev_foot_pos_meas=rollout.prev_foot_pos_meas,
            )
        )
        rollout.prev_foot_pos_meas = step_inputs.prev_foot_pos_meas
        if step_inputs.reprojection_applied:
            rollout.reprojection_applied_count += 1

        with self._amp_context(rollout.amp_enabled):
            ret = self.model(
                rollout.motion,
                step_inputs.cond_input,
                contacts=step_inputs.contacts_in_t,
                angvel=step_inputs.angvel_t,
                pose_history=step_inputs.pose_history_t,
                plan_z=rollout.plan_z,
                phase_z=rollout.phase_z,
                phase_event_age=rollout.phase_event_age,
                meas_logits_prev=rollout.meas_prev_prob,
                time_index=step_inputs.time_index_t,
                rollout_step=step_inputs.rollout_step_t,
            )

        if not isinstance(ret, dict):
            raise RuntimeError("Model forward must return a dict with at least 'out'.")

        delta_out = ret.get('delta', ret.get('out', None))
        if delta_out is None:
            raise RuntimeError("Model forward must return 'delta' tensor.")

        period_pred = ret.get('period_pred', None)
        rollout.last_attn = ret.get('attn', rollout.last_attn)
        self._update_rollout_plan_state(rollout, ret)

        meas_prob_step = ret.get('contacts_meas', None)
        if torch.is_tensor(meas_prob_step):
            rollout.meas_prev_prob = meas_prob_step.detach()

        prev_raw_snapshot = rollout.y_raw_local.clone() if rollout.y_raw_local is not None else None
        y_raw = self._compose_delta_to_raw(rollout.y_raw_local, delta_out)
        if y_raw is None:
            self._raise_norm_error('compose_delta_to_raw 返回 None，缺少上一帧 RAW 数据。')

        apply_lambda = bool(getattr(self, 'lambda_fusion_apply', False))
        if apply_lambda and rollout.mode == 'mixed':
            try:
                apply_lambda = float(rollout.tf_ratio) < 0.999
            except Exception:
                apply_lambda = False
        if apply_lambda and rollout.mode in ('free', 'train_free', 'mixed'):
            try:
                lam_eff = ret.get('lambda_fusion', None)
                if lam_eff is not None:
                    try:
                        lam_eff, _ = self._lambda_fusion_apply_reliability(
                            lam_eff,
                            step_idx=int(step_idx),
                            total_steps=int(rollout.total_steps),
                            rollout_step=step_inputs.rollout_step_t,
                            ret=ret,
                        )
                    except Exception:
                        lam_eff = ret.get('lambda_fusion', None)
                y_raw = self._apply_lambda_fusion_to_raw(
                    y_raw,
                    direct_norm=ret.get('out_direct', None),
                    lambda_fusion=lam_eff,
                )
            except (RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                _phasec_warn_once(
                    "rollout/lambda_fusion",
                    "lambda-fusion postprocess failed; keeping incremental compose output",
                    exc,
                )

        rollout.y_raw_local = y_raw.clone() if rollout.allow_grad else y_raw.detach()
        y_norm = self._norm_y(y_raw)
        self._record_rollout_step_outputs(
            rollout,
            y_norm=y_norm,
            delta_out=delta_out,
            period_pred=period_pred,
            ret=ret,
        )
        step_debug_stats = self._compute_rollout_step_debug_stats(
            delta_out=delta_out,
            y_raw=y_raw,
            prev_raw_snapshot=prev_raw_snapshot,
            pred_raw_local=rollout.y_raw_local,
            gt_seq=rollout_inputs.gt_seq,
            step_idx=step_idx,
        )
        rollout.latest_y_raw = y_raw
        rollout.latest_cond_raw_for_env = step_inputs.cond_raw_for_env
        return step_debug_stats

    def _apply_scheduled_sampling_update(
        self,
        rollout: RolloutExecutionState,
        rollout_inputs: RolloutSequenceInputs,
        *,
        step_idx: int,
    ) -> None:
        if step_idx >= rollout.total_steps - 1:
            return

        carry_state = self._update_rollout_carry_state(
            SimpleNamespace(
                step_idx=step_idx,
                total_steps=rollout.total_steps,
                batch_size=rollout.batch_size,
                tf_ratio=float(rollout.tf_ratio),
                state_seq=rollout_inputs.state_seq,
                gt_seq=rollout_inputs.gt_seq,
                motion_raw_local=rollout.motion_raw_local,
                y_raw=rollout.latest_y_raw,
                y_raw_local=rollout.y_raw_local,
                allow_grad=rollout.allow_grad,
                cond_raw_for_env=rollout.latest_cond_raw_for_env,
                ss_chunk_len=rollout.ss_chunk_len,
                ss_sel_hold=rollout.ss_sel_hold,
                pose_hist_state=rollout.pose_hist_state,
                rot6d_y_slice=rollout.rot6d_y_slice,
            )
        )
        rollout.motion = carry_state.motion
        rollout.motion_raw_local = carry_state.motion_raw_local
        rollout.y_raw_local = carry_state.y_raw_local
        rollout.ss_sel_hold = carry_state.ss_sel_hold
        rollout.pose_hist_state = carry_state.pose_hist_state

    def _rollout_sequence(
        self,
        state_seq,
        cond_seq=None,
        cond_raw_seq=None,
        contacts_seq=None,
        angvel_seq=None,
        pose_hist_seq=None,
        *,
        gt_seq=None,
        cond_norm_mu=None,
        cond_norm_std=None,
        mode='mixed',
        tf_ratio=1.0,
        time_base=None,
    ):
        self._require_normalizer("Trainer._rollout_sequence")
        assert state_seq.dim() == 3, "state_seq expects [B,T,Dx]"
        mode = str(mode or 'mixed')
        valid_modes = {'mixed', 'train_free'}
        if mode not in valid_modes:
            raise ValueError(f"_rollout_sequence mode must be one of {valid_modes}, got {mode}")
        self._commit_rollout_diag_update(mode=mode)
        rollout_inputs = RolloutSequenceInputs(
            state_seq=state_seq,
            cond_seq=cond_seq,
            cond_raw_seq=cond_raw_seq,
            contacts_seq=contacts_seq,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            gt_seq=gt_seq,
        )
        rollout = self._init_rollout_state(
            rollout_inputs,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            mode=mode,
            tf_ratio=tf_ratio,
            time_base=time_base,
        )
        try:
            for step_idx in range(rollout.total_steps):
                self._commit_rollout_diag_update(step=int(step_idx))
                step_diag_update = self._rollout_forward_step(rollout, rollout_inputs, step_idx=step_idx)
                self._commit_rollout_diag_update(last_step_debug_stats=step_diag_update)
                self._apply_scheduled_sampling_update(rollout, rollout_inputs, step_idx=step_idx)

            y = torch.stack(rollout.outs, dim=1)
            preds = {'out': y, 'delta': torch.stack(rollout.delta_preds, dim=1)}
            _finalize_rollout_prediction_buffers(
                preds,
                rollout.buffers,
            )

            # 诊断：报告重投影应用情况
            if rollout.enable_reprojection and rollout.reprojection_applied_count > 0:
                diag_limit = int(getattr(self, '_reprojection_diag_limit', 3))
                if not hasattr(self, '_reprojection_diag_count'):
                    self._reprojection_diag_count = 0
                if self._reprojection_diag_count < diag_limit:
                    epoch = getattr(self, 'cur_epoch', -1)
                    print(
                        f"[CondReprojection] Epoch {epoch}, Mode '{rollout.mode}': "
                        f"Applied reprojection to {rollout.reprojection_applied_count}/{rollout.total_steps} steps"
                    )
                    self._reprojection_diag_count += 1

            return preds, rollout.last_attn
        finally:
            self._commit_rollout_diag_update(mode=None, step=-1)

    @staticmethod
    def _module_grad_norm(module: Optional[torch.nn.Module]) -> float:
        if module is None:
            return float('nan')
        total = None
        for param in module.parameters(recurse=True):
            if param.grad is None:
                continue
            g2 = param.grad.detach().float().pow(2).sum()
            total = g2 if total is None else total + g2
        if total is None:
            return float('nan')
        return float(total.sqrt().detach().cpu())

    @staticmethod
    def _merge_grad_norm(*vals: float) -> float:
        finite = [float(v) for v in vals if isinstance(v, (int, float)) and _math.isfinite(float(v))]
        if not finite:
            return float('nan')
        return float(_math.sqrt(sum(v * v for v in finite)))

    def _collect_direct_pose_grad_stats(self) -> Dict[str, float]:
        model = getattr(self, 'model', None)
        if model is None:
            return {}
        g_trunk = self._module_grad_norm(getattr(model, 'direct_pose_head', None))
        g_leg = self._module_grad_norm(getattr(model, 'direct_pose_out_leg', None))
        g_nonleg_head = self._module_grad_norm(getattr(model, 'direct_pose_out_nonleg', None))
        g_arm = self._module_grad_norm(getattr(model, 'direct_pose_out_arm', None))
        g_else = self._module_grad_norm(getattr(model, 'direct_pose_out_else', None))
        g_nonleg = self._merge_grad_norm(g_nonleg_head, g_arm, g_else)
        ratio_nonleg_leg = float('nan')
        if _math.isfinite(g_leg) and _math.isfinite(g_nonleg):
            ratio_nonleg_leg = float(g_nonleg / max(1e-12, g_leg))
        ratio_arm_else = float('nan')
        if _math.isfinite(g_arm) and _math.isfinite(g_else):
            ratio_arm_else = float(g_arm / max(1e-12, g_else))
        stats = {
            'direct_grad_norm_trunk': float(g_trunk),
            'direct_grad_norm_out_leg': float(g_leg),
            'direct_grad_norm_out_nonleg': float(g_nonleg),
            'direct_grad_norm_out_arm': float(g_arm),
            'direct_grad_norm_out_else': float(g_else),
            'direct_grad_ratio_nonleg_over_leg': float(ratio_nonleg_leg),
            'direct_grad_ratio_arm_over_else': float(ratio_arm_else),
        }
        gate_thr = float(getattr(self, 'direct_pose_grad_ratio_gate', 0.35) or 0.35)
        if _math.isfinite(ratio_nonleg_leg) and _math.isfinite(gate_thr) and gate_thr > 0.0:
            stats['direct_grad_ratio_gate'] = float(gate_thr)
            stats['direct_grad_ratio_alert'] = 1.0 if ratio_nonleg_leg < gate_thr else 0.0
        return stats

    def _history_drift_debug(
        self,
        state_seq,
        gt_seq,
        cond_seq,
        cond_raw_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        *,
        epoch: int,
        batch_idx: int,
        cond_norm_mu=None,
        cond_norm_std=None,
    ) -> None:
        steps = int(getattr(self, 'history_debug_steps', 0) or 0)
        if steps <= 1:
            return
        steps = min(steps, state_seq.shape[1])
        if steps <= 1:
            return
        with torch.no_grad():
            preds_free, _ = self._rollout_sequence(
                state_seq[:, :steps],
                cond_seq[:, :steps] if isinstance(cond_seq, torch.Tensor) and cond_seq.dim() == 3 else cond_seq,
                cond_raw_seq[:, :steps] if isinstance(cond_raw_seq, torch.Tensor) and cond_raw_seq.dim() == 3 else cond_raw_seq,
                contacts_seq=contacts_seq[:, :steps] if isinstance(contacts_seq, torch.Tensor) and contacts_seq.dim() == 3 else contacts_seq,
                angvel_seq=angvel_seq[:, :steps] if isinstance(angvel_seq, torch.Tensor) and angvel_seq.dim() == 3 else angvel_seq,
                pose_hist_seq=pose_hist_seq[:, :steps] if isinstance(pose_hist_seq, torch.Tensor) and pose_hist_seq.dim() == 3 else pose_hist_seq,
                gt_seq=gt_seq[:, :steps],
                mode='train_free',
                tf_ratio=0.0,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
            )
        pred_out = preds_free.get('out') if isinstance(preds_free, dict) else None
        if pred_out is None:
            return
        try:
            gt_raw = self._denorm(gt_seq[:, :steps])
            pred_raw = self._denorm(pred_out)
        except Exception as exc:
            print(f"[HistDrift][warn] denorm failed: {exc}")
            return
        rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        root_idx = int(getattr(self, 'eval_root_idx', getattr(self, 'root_idx', 0)))
        stats = {}
        if isinstance(rot_slice, slice):
            rot_width = rot_slice.stop - rot_slice.start
            if rot_width > 0 and rot_width % 6 == 0:
                B = gt_raw.shape[0]
                J = rot_width // 6
                gt_flat = gt_raw[:, :steps, rot_slice].reshape(B * steps, J, 6)
                pred_flat = pred_raw[:, :steps, rot_slice].reshape(B * steps, J, 6)
                gt_m = rot6d_to_matrix(reproject_rot6d(gt_flat)).view(B, steps, J, 3, 3)
                pred_m = rot6d_to_matrix(reproject_rot6d(pred_flat)).view(B, steps, J, 3, 3)
                pred_root = _root_relative_matrices(pred_m, root_idx)
                gt_root = _root_relative_matrices(gt_m, root_idx)
                joint_weights = self._joint_weights(pred_root, J)
                weights_sum = joint_weights.sum().clamp_min(1e-6)
                w = joint_weights.view(1, 1, -1)
                geo_local_rad = geodesic_R(pred_root, gt_root)
                geo_local = geo_local_rad * (180.0 / _math.pi)
                geo_local_mean = (geo_local * w).sum() / (weights_sum * geo_local.shape[0] * geo_local.shape[1])
                stats['rot_local_mean_deg'] = float(geo_local_mean.item())
                stats['rot_local_step_deg'] = ((geo_local * w).sum(dim=-1) / weights_sum).mean(dim=0).detach().cpu().tolist()
                stats['_geo_local_rad'] = geo_local_rad.detach()
        geo_local_tensor_rad = stats.get('_geo_local_rad')
        limb_summary = {}
        collect_fn = getattr(self.loss_fn, '_collect_rot_local_stats', None)
        if geo_local_tensor_rad is not None and callable(collect_fn):
            try:
                limb_summary = collect_fn(geo_local_tensor_rad)
            except Exception as exc:
                print(f"[HistDrift][ERR] limb summary failed: {exc}")
        try:
            if not stats:
                return
            local_val = stats.get('rot_local_mean_deg', float('nan'))
            extra = ""
            if limb_summary:
                limb_raw = limb_summary.get('rot_local_limb_deg', float('nan'))
                limb_weighted = limb_summary.get('rot_local_limb_over_torso', float('nan'))
                if _math.isfinite(limb_raw):
                    extra += f" limb={limb_raw:.2f}°"
                if _math.isfinite(limb_weighted):
                    extra += f" limb/torso={limb_weighted:.2f}"
            if not (isinstance(local_val, (float, int)) and _math.isfinite(local_val)):
                return
            print(
                "[HistDrift]"
                f"[ep {int(epoch):03d}]"
                f"[bi {int(batch_idx):04d}] "
                f"rot_local={local_val:.2f}° steps={steps}{extra}"
            )
            local_curve = stats.get('rot_local_step_deg')
            geo_local_tensor_rad = stats.get('_geo_local_rad')
            if isinstance(local_curve, list):
                for idx, local_val_step in enumerate(local_curve, start=1):
                    summary_txt = ""
                    if isinstance(geo_local_tensor_rad, torch.Tensor) and geo_local_tensor_rad.shape[1] >= idx and callable(collect_fn):
                        try:
                            step_tensor = geo_local_tensor_rad[:, idx - 1:idx]
                            limb_step = collect_fn(step_tensor)
                        except Exception as exc:
                            print(f"[HistDrift][ERR] limb step summary failed (step={idx}): {exc}")
                            limb_step = None
                        if limb_step:
                            limb_deg = limb_step.get('rot_local_limb_deg', float('nan'))
                            torso_deg = limb_step.get('rot_local_torso_deg', float('nan'))
                            if _math.isfinite(limb_deg):
                                summary_txt += f" limb={limb_deg:.2f}°"
                            if _math.isfinite(torso_deg):
                                summary_txt += f" torso={torso_deg:.2f}°"
                    if not _math.isnan(local_val_step):
                        print(
                            "[HistDrift]"
                            f"[ep {int(epoch):03d}]"
                            f"[bi {int(batch_idx):04d}]"
                            f"[step {idx:02d}] rot_local={local_val_step:.2f}°{summary_txt}"
                        )
        finally:
            stats.pop('_geo_local_rad', None)

    def _joint_group_masks(self, J: int, bone_names: Optional[Sequence[str]] = None):
        masks = {}
        if bone_names:
            torso_idx = []
            prox_idx = []
            dist_idx = []
            for idx, name in enumerate(bone_names):
                lname = str(name).lower()
                if any(key in lname for key in ('spine', 'pelvis', 'root', 'torso', 'chest', 'neck')):
                    torso_idx.append(idx)
                elif any(key in lname for key in ('upperarm', 'thigh', 'clavicle', 'shoulder', 'hip')):
                    prox_idx.append(idx)
                else:
                    dist_idx.append(idx)
        else:
            torso_count = min(5, J)
            prox_count = min(5, max(0, J - torso_count))
            torso_idx = list(range(torso_count))
            prox_idx = list(range(torso_count, torso_count + prox_count))
            dist_idx = list(range(torso_count + prox_count, J))
        def _mask(idxs):
            mask = torch.zeros(J, dtype=torch.bool, device=self.device)
            if idxs:
                valid = [i for i in idxs if 0 <= i < J]
                if valid:
                    mask[valid] = True
            return mask
        masks['torso'] = _mask(torso_idx)
        masks['proximal'] = _mask(prox_idx)
        masks['distal'] = _mask(dist_idx)
        return masks

    def _summarize_angvel_dir(
        self,
        pred_w: Optional[torch.Tensor],
        gt_w: Optional[torch.Tensor],
        *,
        bone_names: Optional[Sequence[str]] = None,
        magnitude_threshold: float = 0.1,
        smooth_window: int = 3,
    ) -> dict:
        if pred_w is None or gt_w is None:
            return {}
        if pred_w.numel() == 0 or gt_w.numel() == 0:
            return {}
        B, T, J, _ = pred_w.shape
        eps = 1e-6
        dot = (pred_w * gt_w).sum(dim=-1)
        norm = pred_w.norm(dim=-1) * gt_w.norm(dim=-1)
        cos = torch.clamp(dot / (norm + eps), -1.0 + 1e-7, 1.0 - 1e-7)
        angle_deg = torch.acos(cos) * (180.0 / _math.pi)
        raw = float(angle_deg.mean().item())
        mag = gt_w.norm(dim=-1)
        weight = (mag > magnitude_threshold).float()
        weighted = float((angle_deg * weight).sum().item() / (weight.sum().item() + eps))
        smooth = weighted
        if smooth_window >= 3 and T >= smooth_window:
            pad = smooth_window // 2
            pred_flat = pred_w.reshape(B, T, J * 3).transpose(1, 2)
            gt_flat = gt_w.reshape(B, T, J * 3).transpose(1, 2)
            pred_s = F.avg_pool1d(pred_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(B, T, J, 3)
            gt_s = F.avg_pool1d(gt_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(B, T, J, 3)
            dot_s = (pred_s * gt_s).sum(dim=-1)
            norm_s = pred_s.norm(dim=-1) * gt_s.norm(dim=-1)
            cos_s = torch.clamp(dot_s / (norm_s + eps), -1.0 + 1e-7, 1.0 - 1e-7)
            angle_s = torch.acos(cos_s) * (180.0 / _math.pi)
            smooth = float((angle_s * weight).sum().item() / (weight.sum().item() + eps))
        masks = self._joint_group_masks(J, bone_names)
        group_vals = {}
        for key, mask in masks.items():
            if mask.any():
                mask_f = mask.view(1, 1, J)
                grp_weight = weight * mask_f
                denom = grp_weight.sum().item()
                if denom > 0:
                    grp_val = float((angle_deg * grp_weight).sum().item() / (denom + eps))
                else:
                    grp_val = float('nan')
            else:
                grp_val = float('nan')
            group_vals[key] = grp_val
        return {
            'raw': raw,
            'weighted': weighted,
            'smooth': smooth,
            'torso': group_vals.get('torso', float('nan')),
            'proximal': group_vals.get('proximal', float('nan')),
            'distal': group_vals.get('distal', float('nan')),
        }

    def _set_direct_pose_trunk_trainable(self, enabled: bool) -> None:
        self.direct_pose_trunk_trainable = bool(enabled)
        model = getattr(self, 'model', None)
        head = getattr(model, 'direct_pose_head', None) if model is not None else None
        if head is None:
            print('[StageSched][WARN] direct_pose_head missing; cannot toggle trunk trainability.')
            return
        for param in head.parameters():
            param.requires_grad_(bool(enabled))

    def _apply_runtime_trainability_modes(self) -> None:
        if bool(getattr(self, 'direct_pose_trunk_trainable', True)):
            return
        model = getattr(self, 'model', None)
        head = getattr(model, 'direct_pose_head', None) if model is not None else None
        if head is not None:
            head.train(False)

    def _apply_stage_schedule(self, epoch: int):
        schedule = getattr(self, 'freerun_stage_schedule', None)
        overrides: Dict[str, Any] = {}
        if not schedule:
            return overrides

        if not hasattr(self, '_stage_active_idx') or self._stage_active_idx is None:
            idx = 0
            for i, stage in enumerate(schedule):
                try:
                    st = int(stage.get('start', 1))
                    ed = int(stage.get('end', st))
                except Exception:
                    continue
                if st <= epoch <= ed:
                    idx = i
                    break
                if epoch >= st:
                    idx = i
            self._activate_stage(idx, epoch)

        if getattr(self, '_stage_pending_advance', False):
            self._advance_stage(epoch)

        def _assign(key: str, value: Any) -> bool:
            target = self
            attr_name = key
            prefix = None
            if '.' in key:
                prefix, attr_name = key.split('.', 1)
                if prefix in ('loss', 'loss_fn'):
                    target = getattr(self, 'loss_fn', None)
                elif prefix in ('opt', 'optimizer'):
                    target = getattr(self, 'optimizer', None)
                elif prefix in ('trainer', 'self'):
                    target = self
                else:
                    target = getattr(self, prefix, None)
            elif not hasattr(target, attr_name):
                loss_candidate = getattr(self, 'loss_fn', None)
                if loss_candidate is not None and hasattr(loss_candidate, attr_name):
                    target = loss_candidate
                    prefix = 'loss'
                else:
                    target = None
            if target is None or not hasattr(target, attr_name):
                return False
            current = getattr(target, attr_name)
            coerced = value
            if current is not None:
                if isinstance(current, bool):
                    coerced = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    try:
                        coerced = int(round(float(value)))
                    except Exception:
                        coerced = current
                elif isinstance(current, float):
                    try:
                        coerced = float(value)
                    except Exception:
                        coerced = current
            # clamp history dropout if assigned via schedule
            if attr_name == 'history_dropout_prob':
                try:
                    lo = float(getattr(self, 'history_dropout_prob_min', 0.05))
                    hi = float(getattr(self, 'history_dropout_prob_max', 0.30))
                    coerced = max(lo, min(hi, float(coerced)))
                except (TypeError, ValueError, RuntimeError) as exc:
                    _phasec_warn_once(
                        "stage_schedule/history_dropout_prob",
                        "failed to clamp history_dropout_prob from stage schedule; keeping previous value",
                        exc,
                    )
            setattr(target, attr_name, coerced)
            key_name = key if prefix else attr_name
            overrides[key_name] = coerced
            return True

        selected = self._current_stage()
        if selected is None:
            return overrides

        while epoch > selected.get('end', epoch) and (self._stage_active_idx or 0) < len(schedule) - 1:
            self._advance_stage(epoch)
            selected = self._current_stage()
            if selected is None:
                return overrides

        params = dict(selected.get('params') or {})
        for key, value in params.items():
            # Special-case stage-wise optimizer LR scheduling.
            if key in ("opt_lr", "optimizer_lr"):
                try:
                    lr_val = float(value)
                except Exception:
                    lr_val = None
                if lr_val is not None and hasattr(self, "optimizer") and self.optimizer is not None:
                    try:
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr_val
                        overrides[key] = lr_val
                    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
                        _phasec_warn_once(
                            "stage_schedule/optimizer_lr",
                            "failed to set optimizer LR from stage schedule",
                            exc,
                        )
                continue
            if key in (
                "direct_pose_trunk_trainable",
                "trainer.direct_pose_trunk_trainable",
                "model.direct_pose_trunk_trainable",
            ):
                enabled = bool(value)
                self._set_direct_pose_trunk_trainable(enabled)
                overrides[key] = enabled
                continue
            _assign(key, value)

        label = selected.get('label')
        stage_tag = f"{selected.get('start', '?')}-{selected.get('end', '?')}"
        if label:
            stage_tag += f" {label}"
        if overrides:
            summary = ', '.join(f"{k}={overrides[k]}" for k in sorted(overrides))
        else:
            summary = 'no overrides'
        print(f"[StageSched][ep {epoch:03d}] stage={stage_tag} | {summary}")
        return overrides

    # === Adaptive metric-driven tuning helpers ===
    def _get_current_stage(self, config: Mapping[str, Any]):
        schedule = config.get("freerun_stage_schedule", []) if isinstance(config, Mapping) else []
        cur_ep = int(getattr(self, 'cur_epoch', -1))
        for stage in schedule:
            rng = stage.get("range")
            if rng:
                start = int(rng[0])
                end = int(rng[-1] if len(rng) > 1 else rng[0])
            else:
                start = int(stage.get("start", 1))
                end = int(stage.get("end", start))
            if start <= cur_ep <= end:
                return stage
        return None

    def _apply_config_changes(self, config: Mapping[str, Any]):
        stage = self._get_current_stage(config)
        if not stage:
            return
        trainer_cfg = stage.get("trainer", {})
        loss_cfg = stage.get("loss", {})
        if isinstance(loss_cfg, Mapping):
            legacy_in_loss = [k for k in LEGACY_LOSS_KEYS if k in loss_cfg]
            if legacy_in_loss:
                raise ValueError(_legacy_loss_keys_msg(legacy_in_loss, context="freerun_stage_schedule.loss"))

        if "eval_horizon" in trainer_cfg:
            if hasattr(self, 'eval_settings'):
                self.eval_settings.horizon = int(trainer_cfg["eval_horizon"])

        if hasattr(self, 'loss_fn') and self.loss_fn is not None:
            if "w_rot_local" in loss_cfg:
                self.loss_fn.w_rot_local = float(loss_cfg["w_rot_local"])
            if "adaptive_bone_weights" in loss_cfg:
                self.loss_fn.use_adaptive_weights = bool(loss_cfg["adaptive_bone_weights"])
                self.loss_fn._invalidate_weight_cache()
        # Handle loss_groups (e.g., "core" group weight overrides)
        loss_groups = stage.get("loss_groups", {})
        if hasattr(self, 'loss_fn') and self.loss_fn is not None:
            for group_name, group_weights in loss_groups.items():
                if isinstance(group_weights, dict):
                    legacy_in_group = [k for k in LEGACY_LOSS_KEYS if k in group_weights]
                    if legacy_in_group:
                        raise ValueError(
                            _legacy_loss_keys_msg(
                                legacy_in_group,
                                context=f"freerun_stage_schedule.loss_groups[{group_name!r}]",
                            )
                        )
                    for weight_name, weight_value in group_weights.items():
                        if weight_name == 'adaptive_bone_weights':
                            self.loss_fn.use_adaptive_weights = bool(weight_value)
                            self.loss_fn._invalidate_weight_cache()
                        elif hasattr(self.loss_fn, weight_name):
                            setattr(self.loss_fn, weight_name, float(weight_value))

        if hasattr(self, 'hyperparam_scheduler') and self.hyperparam_scheduler is not None:
            sched_params = self.hyperparam_scheduler.params
            if hasattr(self, 'teacher_forcing_ratio'):
                sched_params["teacher_forcing_ratio"] = float(getattr(self, 'teacher_forcing_ratio'))

    def _save_adjusted_config(self, epoch: int):
        out_dir = getattr(self, 'out_dir', None)
        cfg = getattr(self, 'full_config', None)
        if not out_dir or cfg is None:
            return
        try:
            out_path = Path(out_dir) / f"config_adjusted_ep{int(epoch):03d}.json"
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[AdaptiveTuning][WARN] failed to save adjusted config: {exc}")

    def _activate_stage(self, idx: int, epoch: int) -> None:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        if not schedule:
            return
        idx = max(0, min(idx, len(schedule) - 1))
        self._stage_active_idx = idx
        self._stage_epoch_entered = epoch
        self._stage_goal_history = {}
        stage = schedule[idx]
        stage.pop('_goal_state', None)

    def _advance_stage(self, epoch: int) -> None:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        if not schedule:
            self._stage_pending_advance = False
            return
        idx = (self._stage_active_idx or 0) + 1
        if idx >= len(schedule):
            self._stage_pending_advance = False
            return
        self._activate_stage(idx, epoch)
        self._stage_pending_advance = False

    def _current_stage(self) -> Optional[Dict[str, Any]]:
        schedule = getattr(self, 'freerun_stage_schedule', None)
        idx = getattr(self, '_stage_active_idx', None)
        if schedule and idx is not None and 0 <= idx < len(schedule):
            return schedule[idx]
        return None

    def _maybe_finish_stage(self, epoch: int, metrics: Dict[str, Any], *, tag: str) -> None:
        stage = self._current_stage()
        if stage is None:
            return
        goal = stage.get('goal')
        if not goal:
            return
        tags = goal.get('tags') or ['valfree']
        if tag not in tags:
            return
        window = int(goal.get('window', 3) or 3)
        if not isinstance(self._stage_goal_history, dict):
            self._stage_goal_history = {}
        history = self._stage_goal_history.get(tag)
        if history is None or history.maxlen != window:
            history = deque(maxlen=window)
            self._stage_goal_history[tag] = history
        history.append(dict(metrics))
        min_epochs = int(goal.get('min_epochs', 0) or 0)
        elapsed = 0
        if self._stage_epoch_entered is not None:
            elapsed = epoch - self._stage_epoch_entered + 1
        if elapsed < min_epochs:
            return
        if len(history) < window:
            return
        metrics_cfg = goal.get('metrics') or {}
        if not metrics_cfg:
            return
        for metric_name, cfg in metrics_cfg.items():
            values = [self._extract_metric_from_record(rec, metric_name) for rec in history]
            values = [v for v in values if v is not None]
            if not values:
                return
            avg_val = sum(values) / len(values)
            if not self._metric_within_goal(avg_val, cfg):
                return
        goal_state = stage.get('_goal_state')
        if not isinstance(goal_state, dict):
            goal_state = {}
            stage['_goal_state'] = goal_state
        if goal_state.get('met'):
            return
        goal_state['met'] = True
        goal_state['epoch'] = epoch
        label = stage.get('label') or f"{stage.get('start', '?')}-{stage.get('end', '?')}"
        print(f"[StageGoal] stage={label} met at epoch {epoch:03d}; scheduling advance")
        self._stage_pending_advance = True

    def _extract_metric_from_record(self, record: Mapping[str, Any], name: str) -> Optional[float]:
        target: Any = record
        for part in str(name).split('/'):
            if isinstance(target, Mapping) and part in target:
                target = target[part]
            else:
                return None
        try:
            return float(target)
        except Exception:
            return None

    def _metric_within_goal(self, value: float, cfg: Mapping[str, Any]) -> bool:
        ref = float(cfg.get('ref', 0.0) or 0.0)
        hi = cfg.get('hi')
        lo = cfg.get('lo')
        if hi is None and cfg.get('hi_ratio') is not None:
            hi = ref * float(cfg['hi_ratio'])
        if lo is None and cfg.get('lo_ratio') is not None:
            lo = ref * float(cfg['lo_ratio'])
        if hi is not None and value > float(hi):
            return False
        if lo is not None and value < float(lo):
            return False
        return True
    def test_gradient_connection(self, loader):
        if getattr(self, '_grad_connection_checked', False):
            return
        if not bool(getattr(self, 'enable_grad_connection_test', True)):
            self._grad_connection_checked = True
            return
        import torch
        sample_batch = None
        it = iter(loader)
        try:
            sample_batch = next(it)
        except StopIteration:
            print("[GradConn] skipped: empty loader.")
            self._grad_connection_checked = True
            return
        x_cand = self._pick_first(sample_batch, ('motion','X','x_in_features'))
        y_cand = self._pick_first(sample_batch, ('gt_motion','Y','y_out_features','y_out_seq'))
        if x_cand is None or y_cand is None:
            print("[GradConn] skipped: batch missing motion/gt.")
            self._grad_connection_checked = True
            return
        state_seq = x_cand.to(self.device).float()
        gt_seq = y_cand.to(self.device).float()
        window = min(int(getattr(self, 'grad_conn_window', 8) or 8), state_seq.shape[1])
        if window < 2:
            print("[GradConn] skipped: window < 2.")
            self._grad_connection_checked = True
            return
        state_seq = state_seq[:, :window]
        gt_seq = gt_seq[:, :window]

        def _slice_optional(key):
            val = sample_batch.get(key) if isinstance(sample_batch, dict) else None
            if val is None:
                return None
            tensor = val.to(self.device).float()
            if tensor.dim() == 3 and tensor.size(1) >= window:
                return tensor[:, :window]
            return tensor

        cond_seq = _slice_optional('cond_in')
        cond_raw_seq = _slice_optional('cond_tgt_raw')
        contacts_seq = _slice_optional('contacts')
        angvel_seq = _slice_optional('angvel')
        pose_hist_seq = _slice_optional('pose_hist')
        cond_norm_mu = sample_batch.get('cond_norm_mu') if isinstance(sample_batch, dict) else None
        cond_norm_std = sample_batch.get('cond_norm_std') if isinstance(sample_batch, dict) else None
        if cond_norm_mu is not None:
            cond_norm_mu = cond_norm_mu.to(self.device).float()
        if cond_norm_std is not None:
            cond_norm_std = cond_norm_std.to(self.device).float()
        time_base = None
        try:
            start_base = sample_batch.get("start") if isinstance(sample_batch, dict) else None
            if start_base is not None and torch.is_tensor(start_base):
                time_base = start_base.to(self.device).float()
        except Exception:
            time_base = None

        use_anomaly = bool(getattr(self, 'grad_conn_detect_anomaly', True))
        import contextlib
        anomaly_ctx = torch.autograd.set_detect_anomaly if use_anomaly else contextlib.nullcontext
        with anomaly_ctx(True if use_anomaly else False):
            preds, attn = self._rollout_sequence(
                state_seq,
                cond_seq,
                cond_raw_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
                gt_seq=gt_seq,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
                mode='train_free',
                tf_ratio=0.0,
                time_base=time_base,
            )
            with self._amp_context(self.use_amp):
                out = self.loss_fn(preds, gt_seq, attn_weights=attn, batch=sample_batch)
            loss = out[0] if isinstance(out, tuple) else out
            self.optimizer.zero_grad(set_to_none=True)
            try:
                loss.backward()
            except RuntimeError as exc:
                raise RuntimeError("[GradConn] backward failed; 检查 train_free 梯度链路。") from exc
        # loss/backward completed inside context at this point
        grad_hits = sum(
            1 for p in self.model.parameters()
            if p.grad is not None and torch.isfinite(p.grad).any()
        )
        if grad_hits == 0:
            raise RuntimeError("[GradConn] backward produced no gradients; 可能仍有 detach().")
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_connection_checked = True
        print(f"[GradConn] ok: window={window} grad_hits={grad_hits}.")

    def _maybe_apply_adaptive_loss(self, loss, stats):
        module = getattr(self, 'adaptive_loss_module', None)
        if module is None:
            return loss, stats
        payload_fn = getattr(self.loss_fn, 'adaptive_loss_payload', None)
        if not callable(payload_fn):
            return loss, stats
        payload = payload_fn()
        if not payload:
            return loss, stats
        raw_losses = payload.get('losses') or {}
        total_weight = float(payload.get('total_weight', 0.0))
        if total_weight <= 0:
            return loss, stats
        filtered = {
            name: raw_losses[name]
            for name in module.loss_names
            if name in raw_losses and raw_losses[name] is not None
        }
        # 如果未显式指定 loss_names，运行时自动使用 payload 中的条目
        if not filtered and not module.loss_names:
            filtered = {k: v for k, v in raw_losses.items() if v is not None}
        if not filtered:
            return loss, stats
        core_loss = payload.get('core_loss') or loss
        weighted_loss, rel_weights = module(
            filtered,
            model=self.model,
            epoch=getattr(self, 'cur_epoch', 0),
            scales_override=None,  # component weights不是尺度，避免放大/缩小loss
        )
        adapted = core_loss + weighted_loss * total_weight
        if not isinstance(stats, dict):
            stats = {} if stats is None else dict(stats)
        stats = dict(stats)
        stats['adaptive_loss/total_weight'] = float(total_weight)
        if hasattr(core_loss, 'detach'):
            stats['adaptive_loss/base'] = float(core_loss.detach().cpu())
        else:
            stats['adaptive_loss/base'] = float(core_loss)
        for name, rel in rel_weights.items():
            stats[f'adaptive_loss/weight/{name}'] = float(rel * total_weight)
        return adapted, stats

    def _step_hyperparam_scheduler(self, loss_tensor, grad_norm_value):
        scheduler = getattr(self, 'hyperparam_scheduler', None)
        if scheduler is None:
            return
        try:
            loss_val = float(loss_tensor.detach().cpu())
        except Exception:
            loss_val = float('nan')
        scheduler.step(loss_val, float(grad_norm_value))
        params = scheduler.get_params()
        if 'teacher_forcing_ratio' in params:
            self.teacher_forcing_ratio = float(params['teacher_forcing_ratio'])
    def __init__(self, model, loss_fn, lr=0.0001, grad_clip=0.0, weight_decay=0.01, tf_warmup_steps=0, tf_total_steps=0, augmentor=None, use_amp=None, accum_steps=1, *, pin_memory=False, args=None):
        import torch
        self.model = model
        self.loss_fn = loss_fn
        # Make MuY/StdY available on Trainer for _denorm()
        self.mu_y = getattr(loss_fn, 'mu_y', None)
        self.std_y = getattr(loss_fn, 'std_y', None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[LR-DBG:init] arg_lr={lr:.2e} opt_pg0={self.optimizer.param_groups[0]['lr']:.2e}")
        # Autoreg tuning knobs
        self.use_freerun_state_sync = True
        # Scheduled sampling (mixed mode): hold sel for N frames to reduce high-frequency switching.
        self.ss_chunk_len: int = 1
        self.history_debug_steps: int = 0
        self.freerun_stage_schedule = []
        self._stage_active_idx: Optional[int] = None
        self._stage_epoch_entered: Optional[int] = None
        self._stage_goal_history: Dict[str, deque] = {}
        self._stage_pending_advance: bool = False
        self.direct_pose_trunk_trainable: bool = True
        self.args = args

        self.grad_clip = float(grad_clip)
        self.tf_warmup_steps = int(tf_warmup_steps)
        self.tf_total_steps = int(tf_total_steps)
        self.augmentor = augmentor
        self.device = next(model.parameters()).device
        if use_amp is None:
            self.use_amp = getattr(self.device, 'type', '') in ('cuda',)
        else:
            self.use_amp = bool(use_amp)
        self.accum_steps = int(accum_steps)
        if getattr(self.device, 'type', None) == 'mps' and getattr(self, 'use_amp', False):
            # MPS autocast(fp16) 对部分算子（如 LU）不支持，会触发断言；强制关闭。
            print('[AMP] MPS backend detected; disabling AMP to avoid unsupported fp16 kernels.')
            self.use_amp = False
        dev_type = getattr(self.device, 'type', 'cpu')
        if dev_type == 'mps':
            import contextlib as _ctx
            self._amp_context = lambda enabled: _ctx.nullcontext()
        elif dev_type == 'cuda':
            self._amp_context = lambda enabled: torch.amp.autocast('cuda', enabled=enabled)
        else:
            import contextlib as _ctx
            self._amp_context = lambda enabled: _ctx.nullcontext()
        self._non_blocking = bool(pin_memory and dev_type != 'cpu')
        self._x_layout = {}
        self._y_layout = {}
        self.fps = 60.0
        self.y_to_x_map = []
        self.MuY = None
        self.StdY = None
        self._norm_cache = {}
        self._norm_template_path: Optional[str] = None
        self._bundle_json_path: Optional[str] = None
        # Stage2: optional reliability r_t to modulate lambda_fusion (avoid early direct cold-start).
        # NOTE: keep on Trainer so posttrain + freerun share identical logic (no train/infer mismatch).
        self.lambda_reliability_mode: str = "none"  # e.g. "warmup", "contacts_err", "warmup+contacts_err"
        self.lambda_reliability_warmup_steps: int = 0  # first K rollout steps: r ramps 0->1
        self.lambda_reliability_contact_err_max: float = 1.0  # r = clamp(1 - |contacts_err|/max, 0, 1)
        # Optional: per-joint scaling for warmup reliability (J,) to adapt warmup speed by bone.
        # When set, warmup r_t becomes (B,J) instead of (B,), and is broadcast against lambda_fusion.
        self.lambda_reliability_warmup_joint_scales = None
        # Pose history (explicit buffer) metadata injected from dataset/spec.
        self.pose_hist_len: int = 0
        self.pose_hist_dim: int = 0
        self.pose_hist_scales: Optional[torch.Tensor] = None
        self.pose_hist_mu: Optional[torch.Tensor] = None
        self.pose_hist_std: Optional[torch.Tensor] = None
        self.nan_grad_reports: int = 0
        self.nan_grad_report_limit: int = 5
        self.diag_input_stats: bool = False
        # yaw 诊断相关：仅在需要时打印有限次数的告警，避免刷屏
        self.yaw_diag_deg_threshold: float = 45.0
        self._yaw_diag_limit: int = 5
        self._yaw_diag_hits: int = 0
        self._diag_roll_mode: Optional[str] = None
        self._diag_roll_step: int = -1
        self._diag_roll_epoch: int = 0
        self.enable_grad_connection_test: bool = True
        self._grad_connection_checked: bool = False
        self.grad_conn_window: int = 8
        self.grad_conn_detect_anomaly: bool = True
        self.adaptive_loss_module = None
        self.hyperparam_scheduler: Optional[Any] = None
        self.teacher_forcing_ratio: float = 1.0
        # 自由运行时根部 yaw 的参考策略：
        #   - 'trajectory': 使用 cond_dir 定义世界/轨迹坐标系的 yaw（推荐）
        #   - 'skeleton' : 使用骨骼(pelvis)推断 yaw（旧行为，可能导致坐标系随误差漂移）
        self.freerun_yaw_strategy: str = str(
            getattr(args, 'freerun_yaw_strategy', _arg('freerun_yaw_strategy', 'trajectory')) or 'trajectory'
        )
        # ---- Metrics buffering for in-process consumers ----
        self.metric_history: list[dict[str, Any]] = []
        self.metric_history_maxlen: int = 256
        self.latest_metrics: dict[str, dict[str, Any]] = {}
        self._metric_callbacks: list[Callable[[dict[str, Any]], None]] = []

    def _diag_norm_x(self, x_raw, mu_x=None, std_x=None):
        # 仅使用 DataNormalizer；缺失即视为致命错误
        self._require_normalizer("Trainer._diag_norm_x")
        try:
            return self.normalizer.norm(x_raw)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm 在 _diag_norm_x 中失败", exc)

    def _pick_first(self, batch, keys):
        if batch is None:
            return None
        if isinstance(batch, dict):
            for k in keys:
                if k in batch and batch[k] is not None:
                    return batch[k]
        return None

    def _format_template_hint(self, prefix: str) -> str:
        hints: list[str] = []
        norm_tpl = getattr(self, '_norm_template_path', None)
        bundle = getattr(self, '_bundle_json_path', None)
        if isinstance(norm_tpl, str) and norm_tpl:
            hints.append(f"norm_template={norm_tpl}")
        if isinstance(bundle, str) and bundle:
            hints.append(f"bundle_json={bundle}")
        if hints:
            return f"{prefix} ({', '.join(hints)})"
        return prefix

    def _require_normalizer(self, context: str) -> None:
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            raise RuntimeError(self._format_template_hint(f"[FATAL] {context} 需要已注入的 DataNormalizer。"))

    def _raise_norm_error(self, context: str, exc: Optional[Exception] = None) -> None:
        msg = self._format_template_hint(f"[FATAL] {context}")
        raise RuntimeError(msg) from exc

    def _commit_rollout_diag_update(
        self,
        *,
        mode: Any = _STATE_UPDATE_UNSET,
        step: Any = _STATE_UPDATE_UNSET,
        last_step_debug_stats: Any = _STATE_UPDATE_UNSET,
    ) -> None:
        if mode is not _STATE_UPDATE_UNSET:
            self._diag_roll_mode = mode
        if step is not _STATE_UPDATE_UNSET:
            self._diag_roll_step = int(step)
        if isinstance(last_step_debug_stats, dict):
            self._last_step_debug_stats = last_step_debug_stats

    @torch.no_grad()
    def eval_epoch(self, loader, mode='mixed', max_batches=None):
        self.model.eval()
        return evaluate_teacher(self, loader, mode='mixed', max_batches=max_batches)

    def _compute_fit_drift_slope(self, free_metrics: Mapping[str, Any]) -> float:
        curve = free_metrics.get('GeoDegCurve')
        if not isinstance(curve, list) or not curve:
            start = float(free_metrics.get('GeoDegStart', free_metrics.get('GeoDeg', float('inf'))))
            end = float(free_metrics.get('GeoDegEnd', start))
            horizon = int(free_metrics.get('eval_horizon', 0) or 0)
            return (end - start) / max(1, horizon - 1)

        if isinstance(curve[0], (list, tuple)) and curve[0]:
            horizon = len(curve[0])
            mean_curve = []
            for step_idx in range(horizon):
                vals = []
                for batch_curve in curve:
                    if isinstance(batch_curve, (list, tuple)) and step_idx < len(batch_curve):
                        value = batch_curve[step_idx]
                        if isinstance(value, (int, float)) and value == value:
                            vals.append(float(value))
                mean_curve.append(sum(vals) / max(1, len(vals)))
        else:
            mean_curve = [float(v) for v in curve if isinstance(v, (int, float)) and v == v]

        if len(mean_curve) < 2:
            return float('inf')
        return (mean_curve[-1] - mean_curve[0]) / max(1, len(mean_curve) - 1)

    def _fit_checkpoint_payload(self) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {}
        for key, value in self.model.state_dict().items():
            if torch.is_tensor(value):
                model_state[str(key)] = value.detach().cpu().clone()
            else:
                model_state[str(key)] = value
        payload: Dict[str, Any] = {'model': model_state}
        full_config = getattr(self, 'full_config', None)
        if isinstance(full_config, Mapping):
            payload['config'] = dict(full_config)
        return payload

    def _save_fit_checkpoint_payload(
        self,
        *,
        out_dir: Optional[str],
        run_name: str,
        checkpoint_tag: str,
        payload: Optional[Mapping[str, Any]],
    ) -> Optional[str]:
        if not out_dir or payload is None:
            return None
        filename_map = {
            'best_teacher': 'ckpt_best_teacher_{run_name}.pth',
            'best_free': 'ckpt_best_free_{run_name}.pth',
            'last': 'ckpt_last_{run_name}.pth',
        }
        template = filename_map.get(str(checkpoint_tag))
        if template is None:
            raise ValueError(f'Unknown checkpoint tag: {checkpoint_tag}')
        out_dir_str = str(out_dir)
        os.makedirs(out_dir_str, exist_ok=True)
        ckpt_path = os.path.join(out_dir_str, template.format(run_name=str(run_name)))
        torch.save(dict(payload), ckpt_path)
        return ckpt_path

    def _prepare_fit_epoch(
        self,
        epoch: int,
        total_epochs: int,
        *,
        tf_mode: str,
        tf_start: int,
        tf_end: int,
        tf_max_base: float,
        tf_min_base: float,
    ) -> float:
        self.cur_epoch = int(epoch)
        self.current_epoch = int(epoch)
        self.total_epochs = int(total_epochs)
        self._diag_roll_epoch = int(epoch)
        self._yaw_diag_hits = 0
        self._diag_roll_step = -1
        self._diag_roll_mode = None
        if epoch == 1:
            print(f"[LR-DBG:fit-epoch{epoch:03d}-start] pg0={self.optimizer.param_groups[0]['lr']:.2e}")

        stage_overrides = self._apply_stage_schedule(epoch)
        tf_max_epoch = float(stage_overrides.get('tf_max', tf_max_base))
        tf_min_epoch = float(stage_overrides.get('tf_min', tf_min_base))
        if tf_mode == 'epoch_linear' and tf_end > tf_start:
            if epoch <= tf_start:
                tf_ratio = tf_max_epoch
            elif epoch >= tf_end:
                tf_ratio = tf_min_epoch
            else:
                ratio = (epoch - tf_start) / max(1, (tf_end - tf_start))
                tf_ratio = tf_max_epoch + (tf_min_epoch - tf_max_epoch) * ratio
        else:
            tf_ratio = tf_max_epoch

        self.teacher_forcing_ratio = float(tf_ratio)
        self._last_tf_ratio = float(tf_ratio)
        sched = getattr(self, 'hyperparam_scheduler', None)
        if sched is not None:
            sched.params['teacher_forcing_ratio'] = float(tf_ratio)

        self.model.train()
        self._apply_runtime_trainability_modes()
        self.optimizer.zero_grad(set_to_none=True)
        return float(tf_ratio)

    def _run_one_train_batch(self, batch, *, epoch: int, batch_idx: int, tf_ratio: float):
        if getattr(self, '_cached_train_batch', None) is None:
            def _cache_obj(obj):
                if torch.is_tensor(obj):
                    return obj.detach().cpu()
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_cache_obj(x) for x in obj)
                if isinstance(obj, dict):
                    return {k: _cache_obj(v) for k, v in obj.items()}
                return obj

            try:
                self._cached_train_batch = _cache_obj(batch)
            except Exception:
                self._cached_train_batch = None

        x_cand = self._pick_first(batch, ('motion', 'X', 'x_in_features'))
        y_cand = self._pick_first(batch, ('gt_motion', 'Y', 'y_out_features', 'y_out_seq'))
        if x_cand is None or y_cand is None:
            return None

        def _to_device(maybe_tensor):
            if maybe_tensor is None:
                return None
            try:
                tensor = maybe_tensor.to(self.device, non_blocking=self._non_blocking)
                return tensor if tensor.dtype == torch.float32 else tensor.float()
            except Exception:
                return None

        state_seq = _to_device(x_cand)
        gt_seq = _to_device(y_cand)
        if state_seq is None or gt_seq is None:
            return None
        # Teacher inputs stay clean; rollout noise injection is only applied in mixed/train_free paths.
        cond_seq = _to_device(batch.get('cond_in')) if isinstance(batch, dict) else None
        cond_raw_seq = _to_device(batch.get('cond_tgt_raw')) if isinstance(batch, dict) else None
        contacts_seq = _to_device(batch.get('contacts')) if isinstance(batch, dict) else None
        angvel_seq = _to_device(batch.get('angvel')) if isinstance(batch, dict) else None
        pose_hist_seq = _to_device(batch.get('pose_hist')) if isinstance(batch, dict) else None
        cond_norm_mu = _to_device(batch.get('cond_norm_mu')) if isinstance(batch, dict) else None
        cond_norm_std = _to_device(batch.get('cond_norm_std')) if isinstance(batch, dict) else None
        time_base = _to_device(batch.get('start')) if isinstance(batch, dict) else None
        current_tf_ratio = float(getattr(self, 'teacher_forcing_ratio', tf_ratio))
        preds_dict, last_attn = self._rollout_sequence(
            state_seq,
            cond_seq,
            cond_raw_seq,
            contacts_seq=contacts_seq,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            gt_seq=gt_seq,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            mode='mixed',
            tf_ratio=current_tf_ratio,
            time_base=time_base,
        )
        stats = {}
        with self._amp_context(self.use_amp):
            out = self.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
        if isinstance(out, tuple):
            loss, stats = out
        else:
            loss, stats = out, {}
        if not isinstance(stats, dict):
            stats = {} if stats is None else dict(stats)
        loss, stats = self._maybe_apply_adaptive_loss(loss, stats)
        if epoch == 1 and batch_idx == 1:
            if isinstance(stats, Mapping):
                cp_bce = stats.get('contact_plan_bce', None)
                if cp_bce is not None:
                    print(f'[Smoke] contact_plan_bce={cp_bce}')

        if getattr(self, 'history_debug_steps', 0) > 1 and batch_idx == 1:
            try:
                self._history_drift_debug(
                    state_seq,
                    gt_seq,
                    cond_seq,
                    cond_raw_seq,
                    contacts_seq,
                    angvel_seq,
                    pose_hist_seq,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    cond_norm_mu=cond_norm_mu,
                    cond_norm_std=cond_norm_std,
                )
            except Exception as exc:
                print(f'[HistDrift][warn] debug failed: {exc}')

        return loss, stats, preds_dict, state_seq, gt_seq, current_tf_ratio

    def _run_one_train_epoch(
        self,
        train_loader,
        *,
        epoch: int,
        log_every: int,
        scaler: torch.amp.GradScaler,
        accum_steps: int,
        tf_ratio: float,
    ) -> TrainEpochResult:
        running = 0.0
        count = 0
        epoch_sums: Dict[str, float] = {}
        epoch_counts: Dict[str, int] = {}
        tf_ratio_local = float(tf_ratio)

        for batch_idx, batch in enumerate(train_loader, start=1):
            train_batch = self._run_one_train_batch(batch, epoch=epoch, batch_idx=batch_idx, tf_ratio=tf_ratio_local)
            if train_batch is None:
                continue
            loss, stats, preds_dict, state_seq, gt_seq, tf_ratio_local = train_batch
            scaler.scale(loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(self.optimizer)

                if bool(getattr(self, 'direct_pose_grad_monitor_enable', False)) and isinstance(stats, dict):
                    try:
                        stats.update(self._collect_direct_pose_grad_stats())
                    except Exception as exc:
                        print(f'[DirectGrad][WARN] failed to collect grad stats: {exc}')

                any_bad_grad = False
                bad_names = []
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        any_bad_grad = True
                        if len(bad_names) < 3:
                            bad_names.append(name)
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                if any_bad_grad:
                    try:
                        loss_val = float(loss.detach().cpu())
                    except Exception:
                        loss_val = float('nan')
                    self._dump_nan_grad_report(epoch, batch_idx, batch, state_seq, gt_seq, preds_dict, loss_val, stats)
                    if log_every:
                        print(f"[Guard][Grad] non-finite grads on {', '.join(bad_names)} ... skip optimizer.step()")
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(getattr(self, 'grad_clip', 1.0)),
                )
                self._step_hyperparam_scheduler(loss, float(grad_norm))
                tf_ratio_local = float(getattr(self, 'teacher_forcing_ratio', tf_ratio_local))
                self._last_tf_ratio = float(tf_ratio_local)
                if log_every and (batch_idx % int(log_every or 50) == 0):
                    lr0 = float(self.optimizer.param_groups[0].get('lr', 0.0))
                    print(f'[Grad] ep={epoch:03d} bi={batch_idx:04d} gn={float(grad_norm):.3e} lr={lr0:.2e}')

                scaler.step(self.optimizer)
                if log_every and epoch == 1 and batch_idx == 1:
                    print(f"[LR-DBG:after-opt-step] pg0={self.optimizer.param_groups[0]['lr']:.2e}")
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                params_finite = True
                with torch.no_grad():
                    for _, param in self.model.named_parameters():
                        if not torch.isfinite(param).all():
                            params_finite = False
                            break
                if not params_finite:
                    if log_every:
                        print('[Guard][Param] non-finite parameters after step; try sanitize via validate_and_fix_model_')
                    try:
                        validate_and_fix_model_(self.model, reinit_on_nonfinite=True)
                    except Exception as sanitize_exc:
                        print('[Guard][Param] sanitize failed:', sanitize_exc)
                        raise

                lr_scheduler = getattr(self, 'lr_scheduler', None)
                if lr_scheduler is not None:
                    try:
                        lr_scheduler.step()
                    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
                        _phasec_warn_once(
                            "fit/lr_scheduler_step",
                            "lr_scheduler.step() failed; keeping optimizer state unchanged",
                            exc,
                        )

            if isinstance(stats, dict):
                for key, value in stats.items():
                    try:
                        if hasattr(value, 'detach'):
                            scalar = float(value.detach().cpu())
                        else:
                            scalar = float(value)
                    except (TypeError, ValueError, RuntimeError):
                        continue
                    if not _math.isfinite(scalar):
                        continue
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + scalar
                    epoch_counts[key] = epoch_counts.get(key, 0) + 1

            running += float(loss.detach().cpu())
            count += 1
            if log_every and (batch_idx % int(log_every) == 0):
                print('[Train][ep %03d][%04d/%d] loss=%.4f tf=%.3f' % (
                    epoch,
                    batch_idx,
                    len(train_loader),
                    running / max(1, count),
                    float(tf_ratio_local),
                ))

        avg_train = running / max(1, count)
        print('[Train][ep %03d] loss=%.4f' % (epoch, avg_train))
        train_metrics: Dict[str, Any] = {
            'loss': float(avg_train),
            'phase': 'train',
            'tf_ratio': float(getattr(self, '_last_tf_ratio', tf_ratio_local)),
        }
        for key, total in epoch_sums.items():
            metric_count = int(epoch_counts.get(key, 0) or 0)
            if metric_count > 0:
                train_metrics[key] = float(total / metric_count)
        return TrainEpochResult(avg_train=float(avg_train), train_metrics=train_metrics)

    def _run_epoch_validation(self, *, epoch: int) -> FitEpochValidationResult:
        result = FitEpochValidationResult()
        is_teacher_phase = float(getattr(self, '_last_tf_ratio', 1.0)) >= 0.999

        def _run_valfree_eval(log_prefix: str = 'ValFree') -> Optional[Dict[str, Any]]:
            if getattr(self, 'val_mode', 'none') != 'online' or bool(getattr(self, 'no_monitor', False)):
                return None
            val_loader = self.train_loader
            monitor_batches = getattr(self, 'monitor_batches', None)
            if monitor_batches is None:
                monitor_batches = 8
            try:
                monitor_batches = int(monitor_batches)
            except Exception:
                monitor_batches = 8
            if monitor_batches <= 0:
                return None
            free_metrics = dict(self.validate_autoreg_online(val_loader, max_batches=monitor_batches))
            free_metrics.setdefault('phase', 'freerun')
            free_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
            extra = ''
            key_geo = free_metrics.get('KeyBone/GeoDegMean', float('nan'))
            key_local = free_metrics.get('KeyBone/GeoLocalDegMean', float('nan'))
            free_ang_dir = free_metrics.get('AngVelDirDeg', float('nan'))
            if _math.isfinite(key_geo):
                extra += f' | LimbGeoDeg={key_geo:.3f}°'
            if _math.isfinite(key_local):
                extra += f' | LimbGeoLocalDeg={key_local:.3f}°'
            if _math.isfinite(free_ang_dir):
                extra += f' | AngVelDirDeg={free_ang_dir:.2f}'
            print(
                f'[{log_prefix}@ep {epoch:03d}] '
                f"GeoDeg={free_metrics.get('GeoDeg', float('nan')):.3f}° | "
                f"RootVelMAE={free_metrics.get('RootVelMAE', float('nan')):.5f} | "
                f"AngVelMAE={free_metrics.get('AngVelMAE', float('nan')):.5f} rad/s | "
                f"AngMagRel={free_metrics.get('AngVelMagRel', float('nan')):.3f}" + extra
            )
            return free_metrics

        try:
            if is_teacher_phase:
                max_teacher_batches = getattr(self, 'teacher_eval_max_batches', None)
                if max_teacher_batches is not None and int(max_teacher_batches) <= 0:
                    cached_batch = getattr(self, '_cached_train_batch', None)
                    if cached_batch is not None:
                        teacher_metrics = dict(self.eval_epoch([cached_batch], mode='teacher', max_batches=1) or {})
                        teacher_metrics.setdefault('phase', 'teacher')
                        teacher_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                        print(f'[ValTeacher@ep {epoch:03d}] cached-batch eval (no extra loader pass)')
                    else:
                        teacher_metrics = None
                        print(f'[ValTeacher@ep {epoch:03d}] skipped: no cached batch available (teacher_eval_max_batches<=0)')
                else:
                    teacher_metrics = dict(self.eval_epoch(self.train_loader, mode='teacher', max_batches=max_teacher_batches) or {})
                    teacher_metrics.setdefault('phase', 'teacher')
                    teacher_metrics['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                teacher_metrics.setdefault('phase', 'teacher')
                try:
                    plateau_scheduler = getattr(self, 'lr_plateau_scheduler', None)
                    if plateau_scheduler is not None:
                        keybone_mean = teacher_metrics.get('KeyBone/GeoLocalDegMean')
                        if keybone_mean is None:
                            keybone_mean = teacher_metrics.get('KeyBone/GeoDegMean')
                        if keybone_mean is None:
                            keybone_mean = teacher_metrics.get('GeoLocalDeg')
                        if keybone_mean is None:
                            keybone_mean = teacher_metrics.get('GeoDeg')
                        if keybone_mean is not None:
                            plateau_scheduler.step(float(keybone_mean))
                except Exception as plateau_exc:
                    print(f'[LR-Plateau][WARN] scheduler step failed: {plateau_exc}')

                base_debug_path = getattr(self, 'freerun_debug_path', None)
                if base_debug_path:
                    ep_tag = f'ep{epoch:03d}'
                    candidate = Path(base_debug_path)
                    if candidate.is_dir() or str(base_debug_path).endswith('/'):
                        candidate = candidate / f'teacher_diag_{ep_tag}.json'
                    else:
                        candidate = candidate.with_name(candidate.stem + f'_teacher_{ep_tag}.json')
                    try:
                        payload = {
                            'epoch': epoch,
                            'phase': 'teacher',
                            'tf_ratio': float(getattr(self, '_last_tf_ratio', 1.0)),
                            'metrics': teacher_metrics,
                        }
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        with open(candidate, 'w') as fw:
                            json.dump(payload, fw, indent=2)
                        print(f'[TeacherDiag] saved to {candidate}')
                    except Exception as teacher_diag_exc:
                        print(f'[TeacherDiag][WARN] failed to save: {teacher_diag_exc}')

                result.metrics_for_json = teacher_metrics
                result.metrics_tag = 'teacher'
                loss_val = teacher_metrics.get('loss', float('nan'))
                geo_deg = teacher_metrics.get('GeoDeg', float('nan'))
                geo_local_deg = teacher_metrics.get('GeoLocalDeg', float('nan'))
                ang_mae = teacher_metrics.get('AngVelMAE', float('nan'))
                ang_rel = teacher_metrics.get('AngVelMagRel', float('nan'))
                print(
                    f'[ValTeacher@ep {epoch:03d}] '
                    f'loss={loss_val:.6f} | '
                    f'GeoDeg={geo_deg:.3f}° | '
                    f'GeoLocalDeg={geo_local_deg:.3f}° | '
                    f'AngVelMAE={ang_mae:.5f} rad/s | '
                    f'AngMagRel={ang_rel:.3f}'
                )
            else:
                free_metrics = _run_valfree_eval()
                if free_metrics is not None:
                    result.metrics_for_json = free_metrics
                    result.metrics_tag = 'valfree'
                    result.best_metrics_source = free_metrics

                teacher_metrics_cached = dict(self.eval_epoch(self.train_loader, mode='teacher') or {})
                teacher_metrics_cached.setdefault('phase', 'teacher')
                teacher_metrics_cached['tf_ratio'] = float(getattr(self, '_last_tf_ratio', 1.0))
                result.teacher_metrics_cached = teacher_metrics_cached

                if result.metrics_for_json is not None:
                    gap_extra = ''
                    key_geo = result.metrics_for_json.get('KeyBone/GeoDegMean', float('nan'))
                    key_local = result.metrics_for_json.get('KeyBone/GeoLocalDegMean', float('nan'))
                    free_ang_dir = result.metrics_for_json.get('AngVelDirDeg', float('nan'))
                    if _math.isfinite(key_geo):
                        gap_extra += f' | LimbGeoDeg={key_geo:.3f}°'
                    if _math.isfinite(key_local):
                        gap_extra += f' | LimbGeoLocalDeg={key_local:.3f}°'
                    if _math.isfinite(free_ang_dir):
                        gap_extra += f' | AngVelDirDeg={free_ang_dir:.2f}'
                    print(
                        f'[Gap@ep {epoch:03d}] '
                        f"teach_loss={teacher_metrics_cached.get('loss', float('nan')):.6f} | "
                        f"GeoDeg={result.metrics_for_json.get('GeoDeg', float('nan')):.3f}° | "
                        f"AngVelMAE={result.metrics_for_json.get('AngVelMAE', float('nan')):.5f}" + gap_extra
                    )

            if getattr(self, 'force_valfree_eval', False):
                need_force = result.metrics_tag != 'valfree'
                if need_force:
                    forced_valfree_metrics = _run_valfree_eval('ValFreeForced')
                    if forced_valfree_metrics is not None:
                        result.forced_valfree_metrics = forced_valfree_metrics
                        result.best_metrics_source = forced_valfree_metrics
        except Exception as exc:
            phase_label = 'ValTeacher' if is_teacher_phase else 'ValFree'
            import traceback
            traceback.print_exc()
            print(f'[{phase_label}@ep {epoch:03d}] skipped due to error: {exc}')

        return result

    def _persist_epoch_validation_outputs(self, *, epoch: int, validation_result: FitEpochValidationResult) -> None:
        if validation_result.metrics_for_json is not None and validation_result.metrics_tag is not None:
            self._record_epoch_metrics(validation_result.metrics_for_json, tag=validation_result.metrics_tag, epoch=epoch)
            if validation_result.metrics_tag == 'valfree':
                self._save_val_metrics(epoch, validation_result.metrics_for_json)
            else:
                self._dump_metrics_json(validation_result.metrics_for_json, tag=validation_result.metrics_tag, epoch=epoch)
            self._maybe_finish_stage(epoch, validation_result.metrics_for_json, tag=str(validation_result.metrics_tag))

        if validation_result.forced_valfree_metrics is not None:
            self._record_epoch_metrics(validation_result.forced_valfree_metrics, tag='valfree', epoch=epoch)
            self._save_val_metrics(epoch, validation_result.forced_valfree_metrics)
            self._dump_metrics_json(validation_result.forced_valfree_metrics, tag='valfree', epoch=epoch)
            self._maybe_finish_stage(epoch, validation_result.forced_valfree_metrics, tag='valfree')

        if validation_result.teacher_metrics_cached is not None:
            self._record_epoch_metrics(validation_result.teacher_metrics_cached, tag='teacher', epoch=epoch)
            self._maybe_finish_stage(epoch, validation_result.teacher_metrics_cached, tag='teacher')
            self._dump_metrics_json(validation_result.teacher_metrics_cached, tag='teacher', epoch=epoch)

        try:
            self._write_basetrain_keybone_group_summary()
        except Exception as exc:
            print(f'[MetricsWrite][WARN] failed to update basetrain_keybone_group_summary.json: {exc}')

    def _update_best_ckpts(
        self,
        validation_result: FitEpochValidationResult,
        checkpoint_state: FitCheckpointState,
    ) -> FitCheckpointState:
        teacher_source = None
        if isinstance(validation_result.teacher_metrics_cached, dict) and validation_result.teacher_metrics_cached:
            teacher_source = validation_result.teacher_metrics_cached
        elif isinstance(validation_result.best_metrics_source, dict) and str(validation_result.best_metrics_source.get('phase', '')) == 'teacher':
            teacher_source = validation_result.best_metrics_source

        if teacher_source is not None:
            current_teacher = teacher_source.get('KeyBone/GeoLocalDegMean')
            if current_teacher is None:
                current_teacher = teacher_source.get('GeoLocalDeg')
            if current_teacher is None:
                current_teacher = teacher_source.get('KeyBone/GeoDegMean')
            if current_teacher is None:
                current_teacher = teacher_source.get('GeoDeg')
            current_teacher = float(current_teacher if current_teacher is not None else float('inf'))
            if current_teacher < checkpoint_state.best_teacher_val - 1e-9:
                checkpoint_state.best_teacher_val = current_teacher
                checkpoint_state.best_teacher_payload = self._fit_checkpoint_payload()

        if validation_result.metrics_tag == 'valfree' and isinstance(validation_result.metrics_for_json, dict) and validation_result.metrics_for_json:
            try:
                drift_slope = float(self._compute_fit_drift_slope(validation_result.metrics_for_json))
                validation_result.metrics_for_json['GeoDriftSlope'] = drift_slope
            except Exception:
                drift_slope = float('inf')
            if drift_slope < checkpoint_state.best_free_slope - 1e-9:
                checkpoint_state.best_free_slope = drift_slope
                checkpoint_state.best_free_payload = self._fit_checkpoint_payload()

        return checkpoint_state

    def fit(self, train_loader, epochs=10, log_every=50, out_dir=None, patience=10, run_name='run'):
        self.model.train()
        self.train_loader = train_loader
        device_type = getattr(self.device, 'type', 'cpu')
        scaler = torch.amp.GradScaler('cuda' if device_type=='cuda' else 'cpu', enabled=(getattr(self, 'use_amp', False) and device_type in ('cuda', 'mps')))
        accum_steps = int(getattr(self, 'accum_steps', 1) or 1)
        checkpoint_state = FitCheckpointState()
        history = {'train': [], 'val': []}
        tf_mode = getattr(self, 'tf_mode', 'epoch_linear')
        tf_start = int(getattr(self, 'tf_start_epoch', 0))
        tf_end = int(getattr(self, 'tf_end_epoch', 0))
        tf_max_base = float(getattr(self, 'tf_max', 1.0))
        tf_min_base = float(getattr(self, 'tf_min', 0.0))
        total_epochs = int(epochs)

        try:
            self.test_gradient_connection(train_loader)
        except Exception as _grad_exc:
            print(f"[GradConn] failed during warm-up: {_grad_exc}")
            raise

        for ep in range(1, total_epochs + 1):
            tf_ratio = self._prepare_fit_epoch(
                ep,
                total_epochs,
                tf_mode=tf_mode,
                tf_start=tf_start,
                tf_end=tf_end,
                tf_max_base=tf_max_base,
                tf_min_base=tf_min_base,
            )
            train_epoch_result = self._run_one_train_epoch(
                train_loader,
                epoch=ep,
                log_every=log_every,
                scaler=scaler,
                accum_steps=accum_steps,
                tf_ratio=tf_ratio,
            )
            history['train'].append(train_epoch_result.avg_train)
            self._record_epoch_metrics(train_epoch_result.train_metrics, tag='train', epoch=ep)
            self._dump_metrics_json(train_epoch_result.train_metrics, tag='train', epoch=ep)

            validation_result = self._run_epoch_validation(epoch=ep)
            self._persist_epoch_validation_outputs(epoch=ep, validation_result=validation_result)
            self._update_best_ckpts(
                validation_result,
                checkpoint_state,
            )

        checkpoint_state.last_payload = self._fit_checkpoint_payload()
        if out_dir:
            teacher_ckpt = self._save_fit_checkpoint_payload(
                out_dir=out_dir,
                run_name=run_name,
                checkpoint_tag='best_teacher',
                payload=checkpoint_state.best_teacher_payload,
            )
            if teacher_ckpt is not None:
                checkpoint_state.best_teacher_ckpt = teacher_ckpt
                checkpoint_state.best_ckpt = teacher_ckpt

            free_ckpt = self._save_fit_checkpoint_payload(
                out_dir=out_dir,
                run_name=run_name,
                checkpoint_tag='best_free',
                payload=checkpoint_state.best_free_payload,
            )
            if free_ckpt is not None:
                checkpoint_state.best_free_ckpt = free_ckpt

            checkpoint_state.last_ckpt = self._save_fit_checkpoint_payload(
                out_dir=out_dir,
                run_name=run_name,
                checkpoint_tag='last',
                payload=checkpoint_state.last_payload,
            )

        if checkpoint_state.best_teacher_ckpt is not None:
            print(f'[BestTeacher] ckpt={checkpoint_state.best_teacher_ckpt} GeoLocalDeg={checkpoint_state.best_teacher_val:.6f}°')
        if checkpoint_state.best_free_ckpt is not None and checkpoint_state.best_free_slope < float('inf'):
            print(f'[BestFree] ckpt={checkpoint_state.best_free_ckpt} GeoDriftSlope={checkpoint_state.best_free_slope:.6f} deg/step')

        return checkpoint_state.best_ckpt, history


    def _sl_from_layout(self, layout, key):
        if not isinstance(layout, dict) or key not in layout:
            return None
        st, ln = int(layout[key][0]), int(layout[key][1])
        return slice(st, st+ln) if ln > 0 else None

    # ===== autoregressive online validation (UE-shaped) =====
    @torch.no_grad()
    def validate_autoreg_online(self, loader, max_batches=8):
        settings = getattr(self, 'eval_settings', FreeRunSettings())
        if max_batches is not None:
            max_batches = int(max_batches)
        effective = FreeRunSettings(
            warmup_steps=settings.warmup_steps,
            horizon=settings.horizon,
            max_batches=max_batches if max_batches is not None else settings.max_batches,
        )
        return evaluate_freerun(self, loader, effective)

    def _denorm(self, y):
        # 仅使用 DataNormalizer；缺失或异常直接终止
        self._require_normalizer("Trainer._denorm")
        try:
            return self.normalizer.denorm(y)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm 在 _denorm 中失败", exc)

    def _cached_norm_param(self, key: str, value, ref_tensor):
        import torch
        if value is None:
            return None
        cache = self._norm_cache.setdefault(key, {})
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        cache_key = (device, dtype)
        tensor = cache.get(cache_key)
        if tensor is None:
            if torch.is_tensor(value):
                tensor = value.to(device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(value, device=device, dtype=dtype)
            cache[cache_key] = tensor
        return tensor

    def _norm_y(self, y_raw):
        self._require_normalizer("Trainer._norm_y")
        try:
            return self.normalizer.norm_y(y_raw)
        except AttributeError as exc:
            self._raise_norm_error("DataNormalizer 缺少 norm_y 方法", exc)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm_y 失败", exc)

    def _compose_delta_to_raw(
        self,
        y_prev_raw,
        delta_norm,
        *,
        omega_hat=None,
        so3_gate: Optional[float] = None,
        so3_max_deg: Optional[float] = None,
        omega_detach: bool = True,
    ):
        import torch
        if y_prev_raw is None:
            self._raise_norm_error("compose_delta_to_raw 需要上一帧 RAW，但收到 None。")
        if delta_norm is None:
            self._raise_norm_error("compose_delta_to_raw 收到 None delta。")
        if delta_norm.shape[-1] != y_prev_raw.shape[-1]:
            self._raise_norm_error("compose_delta_to_raw 维度不匹配。")
        std = getattr(self, 'StdY', None) or getattr(self, 'std_y', None)
        delta_raw = delta_norm
        if std is not None:
            try:
                std_t = self._cached_norm_param('std_y', std, delta_norm)
                if std_t is not None:
                    while std_t.dim() < delta_norm.dim():
                        std_t = std_t.unsqueeze(0)
                    delta_raw = delta_norm * std_t.clamp_min(1e-6)
            except (RuntimeError, TypeError, ValueError) as exc:
                _phasec_warn_once(
                    "compose_delta/std_scale",
                    "failed to apply StdY scaling for delta compose; using unscaled delta_norm",
                    exc,
                )
        # 仅对 rot6d 部分做增量合成，尾部附加通道（如 RootVelocity）直接做残差相加
        rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(rot_slice, slice):
            rot_slice = slice(0, y_prev_raw.shape[-1])
        rot_len = rot_slice.stop - rot_slice.start
        if rot_len % 6 != 0:
            self._raise_norm_error(f"compose_delta_to_raw: rot_slice 长度 {rot_len} 不是 6 的倍数。")

        # Optional SO(3) delta correction: ΔR_used = Exp(gate*omega) @ ΔR_pred
        if omega_hat is not None:
            try:
                J = rot_len // 6
                omega = omega_hat
                if torch.is_tensor(omega) and omega.dim() == 4 and omega.size(1) == 1:
                    omega = omega[:, 0]
                if torch.is_tensor(omega) and omega.shape[-2:] == (J, 3) and omega.shape[0] == delta_raw.shape[0]:
                    gate_val = so3_gate
                    if gate_val is None:
                        logit = getattr(self.model, 'so3_corr_gate_logit', None)
                        if torch.is_tensor(logit):
                            gate_val = float(torch.sigmoid(logit.detach()).item())
                        else:
                            gate_val = 0.0
                    gate_val = float(gate_val or 0.0)
                    if gate_val > 1e-6:
                        max_deg = so3_max_deg
                        if max_deg is None:
                            max_deg = float(getattr(self, 'so3_corr_max_deg', 20.0) or 20.0)
                        max_deg = float(max_deg or 0.0)
                        max_rad = (max_deg * (math.pi / 180.0)) if max_deg > 0.0 else None

                        omega_src = omega.detach() if bool(omega_detach) else omega
                        omega_eff = omega_src.to(device=delta_raw.device, dtype=delta_raw.dtype) * gate_val
                        if max_rad is not None:
                            n = omega_eff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                            s = (max_rad / n).clamp_max(1.0)
                            omega_eff = omega_eff * s

                        R_corr = so3_exp_map(omega_eff)  # (B,J,3,3)
                        # NOTE:
                        #   delta_raw[..., rot_slice] is a *residual* in 6D-space (around identity),
                        #   and compose_rot6d_delta expects the same residual convention.
                        #   So we must:
                        #     1) residual -> proper ΔR (near identity) via normalize_rot6d_delta
                        #     2) apply correction on-manifold
                        #     3) convert back to residual (Δrot6d_abs - rot6d_identity)
                        columns = getattr(self.loss_fn, '_rot6d_columns', ("X", "Z"))
                        cols = tuple(columns) if isinstance(columns, (list, tuple)) else ("X", "Z")

                        delta6_proj = normalize_rot6d_delta(delta_raw[..., rot_slice], columns=cols)  # (B,J,6) abs 6D
                        R_delta = rot6d_to_matrix(delta6_proj, columns=cols)  # (B,J,3,3)
                        R_used = torch.matmul(R_corr, R_delta)  # (B,J,3,3)

                        delta6_used_abs = matrix_to_rot6d(R_used, columns=cols)  # (B,J,6)
                        ident6 = _rot6d_identity_like(delta6_used_abs, columns=cols)
                        delta6_used = delta6_used_abs - ident6
                        delta_raw = delta_raw.clone()
                        delta_raw[..., rot_slice] = delta6_used.reshape(delta_raw.shape[0], J * 6)
            except (RuntimeError, TypeError, ValueError) as exc:
                _phasec_warn_once(
                    "compose_delta/so3_corr",
                    "SO(3) correction failed during compose; falling back to raw residual compose",
                    exc,
                )
        try:
            rot_next = compose_rot6d_delta(
                y_prev_raw[..., rot_slice],
                delta_raw[..., rot_slice],
                # 训练路径上关闭基于 SVD 的投影，以避免在 MPS 后端引入不稳定的梯度。
                # 自由运行诊断脚本（如 run_freerun_cycles）在 eval/no_grad 环境下
                # 仍可显式启用 reproject_result=True 以做数值分析。
                reproject_result=False,
            )
        except Exception as e:
            self._raise_norm_error("compose_rot6d_delta 失败", e)

        if rot_len == y_prev_raw.shape[-1]:
            return rot_next
        tail_prev = y_prev_raw[..., rot_slice.stop:]
        tail_delta = delta_raw[..., rot_slice.stop:]
        tail_next = tail_prev + tail_delta
        return torch.cat([rot_next, tail_next], dim=-1)

    def _lambda_fusion_apply_reliability(
        self,
        lambda_fusion,
        *,
        step_idx: Optional[int] = None,
        total_steps: Optional[int] = None,
        rollout_step=None,
        ret: Optional[dict] = None,
    ):
        """
        Stage2: apply a deterministic reliability factor r_t in [0,1] to λ (lambda_fusion).

        Motivation:
            - Reduce early direct cold-start damage (e.g. plan_z warm-up).
            - Eliminate train/infer mismatch by sharing the same r_t computation across:
                posttrain rollout loss + freerun (eval_utils / run_freerun_cycles / training rollout).

        Config (Trainer attributes):
            - lambda_reliability_mode: "none" | "warmup" | "contacts_err" | "warmup+contacts_err"
            - lambda_reliability_warmup_steps: int (K)
            - lambda_reliability_contact_err_max: float (err scale, usually 1.0)

        Returns:
            (lambda_eff, r_t) where:
                - lambda_eff has the same shape as lambda_fusion and is clamped to [0,1]
                - r_t is a detached tensor shaped (B,) or (B,J) in [0,1], or None when disabled.
        """
        import torch

        lam = lambda_fusion
        if lam is None or (not torch.is_tensor(lam)):
            return lambda_fusion, None

        mode = str(getattr(self, "lambda_reliability_mode", "none") or "none").strip().lower()
        if mode in ("", "none", "off", "false", "0", "disable", "disabled"):
            return lam, None

        tokens = [s.strip() for s in mode.replace(",", "+").split("+") if s.strip()]
        if not tokens or lam.dim() <= 0:
            return lam, None

        B = int(lam.shape[0])
        if B <= 0:
            return lam, None

        r: Optional[torch.Tensor] = None  # (B,) or (B,J)

        def _mul_r(a: Optional[torch.Tensor], b: torch.Tensor) -> torch.Tensor:
            if a is None:
                return b
            # Align dims for broadcast-safe multiply.
            if a.dim() == 1 and b.dim() == 2:
                a = a.unsqueeze(-1)
            elif a.dim() == 2 and b.dim() == 1:
                b = b.unsqueeze(-1)
            return a * b

        if "warmup" in tokens or "step_warmup" in tokens:
            warmup_steps = int(getattr(self, "lambda_reliability_warmup_steps", 0) or 0)
            if warmup_steps > 0:
                idx = int(step_idx or 0)
                idx = max(0, idx)
                denom = max(1, warmup_steps - 1)
                # NOTE:
                # - r_w_base is *not* clamped before applying per-joint scales, so scales < 1
                #   slow down warmup but still eventually reach 1 for long rollouts.
                # - When no per-joint scales are provided, we keep the historical behavior:
                #   clamp(idx/(K-1), 0, 1).
                r_w_base = float(idx) / float(denom)
                r_w = max(0.0, min(1.0, r_w_base))
                # Optionally scale warmup per joint (useful when different bones drift at different rates).
                r_w_t: torch.Tensor
                joint_scales = getattr(self, "lambda_reliability_warmup_joint_scales", None)
                J = int(lam.shape[-1]) if lam.dim() >= 2 else 0
                if joint_scales is not None and J > 0:
                    try:
                        if not torch.is_tensor(joint_scales):
                            joint_scales = torch.as_tensor(joint_scales, device=lam.device, dtype=lam.dtype)
                        joint_scales_t = joint_scales.to(device=lam.device, dtype=lam.dtype).reshape(-1)
                        if int(joint_scales_t.numel()) == J:
                            base = torch.full((B, 1), float(r_w_base), device=lam.device, dtype=lam.dtype)
                            r_w_t = (base * joint_scales_t.view(1, J)).clamp(0.0, 1.0)
                        else:
                            r_w_t = torch.full((B,), r_w, device=lam.device, dtype=lam.dtype)
                    except (RuntimeError, TypeError, ValueError) as exc:
                        _phasec_warn_once(
                            "lambda_reliability/warmup_joint_scales",
                            "invalid warmup joint scales; fallback to scalar warmup reliability",
                            exc,
                        )
                        r_w_t = torch.full((B,), r_w, device=lam.device, dtype=lam.dtype)
                else:
                    r_w_t = torch.full((B,), r_w, device=lam.device, dtype=lam.dtype)
                r = _mul_r(r, r_w_t)

        if "contacts_err" in tokens or "contact_err" in tokens:
            err = None
            if isinstance(ret, dict):
                err = ret.get("contacts_err", None)
            if torch.is_tensor(err):
                try:
                    if err.dim() == 3 and err.size(1) == 1:
                        err = err[:, 0]
                    elif err.dim() == 3 and err.size(1) > 1:
                        err = err[:, -1]
                    if err.dim() == 2 and err.shape[0] == B:
                        err_abs_mean = err.detach().abs().mean(dim=-1)  # (B,)
                        err_max = float(getattr(self, "lambda_reliability_contact_err_max", 1.0) or 1.0)
                        if err_max > 1e-8:
                            r_c = (1.0 - err_abs_mean / err_max).clamp(0.0, 1.0).to(dtype=lam.dtype)
                            r = _mul_r(r, r_c)
                except (RuntimeError, TypeError, ValueError) as exc:
                    _phasec_warn_once(
                        "lambda_reliability/contacts_err",
                        "contacts_err reliability term failed; using warmup-only reliability path",
                        exc,
                    )

        if r is None:
            return lam, None

        r = r.clamp(0.0, 1.0)
        # Broadcast r to match lam shape.
        try:
            if r.dim() == 1:
                view_shape = (B,) + (1,) * max(0, int(lam.dim()) - 1)
                r_view = r.view(*view_shape)
            elif r.dim() == 2 and lam.dim() >= 2:
                # (B,J) -> (B, 1, ..., 1, J) to broadcast across time if needed
                J = int(r.shape[1])
                view_shape = (B,) + (1,) * max(0, int(lam.dim()) - 2) + (J,)
                r_view = r.view(*view_shape)
            else:
                r_view = r
        except Exception:
            r_view = r

        lam_eff = (lam * r_view).clamp(0.0, 1.0)
        return lam_eff, r.detach()

    def _apply_lambda_fusion_to_raw(
        self,
        y_inc_raw,
        *,
        direct_norm=None,
        lambda_fusion=None,
    ):
        """
        Stage2: on-manifold blend between incremental rollout pose and direct pose prior.

        Given:
            - y_inc_raw: RAW next pose from incremental expert (Δ branch), shape (B, Dy)
            - direct_norm: normalized absolute Y from direct head, shape (B, Dy) or (B,1,Dy)
            - lambda_fusion: gate in [0,1], shape (B,J) / (B,1) / (B,1,J)

        Returns:
            - y_blend_raw: RAW pose where BoneRotations6D slice is replaced by SO(3) geodesic blend.
        """
        import torch

        if y_inc_raw is None or (not torch.is_tensor(y_inc_raw)):
            return y_inc_raw
        if direct_norm is None or lambda_fusion is None:
            return y_inc_raw
        if not torch.is_tensor(direct_norm) or not torch.is_tensor(lambda_fusion):
            return y_inc_raw
        if direct_norm.dim() == 3 and direct_norm.size(1) == 1:
            direct_norm = direct_norm[:, 0]
        if lambda_fusion.dim() == 3 and lambda_fusion.size(1) == 1:
            lambda_fusion = lambda_fusion[:, 0]
        if direct_norm.dim() != 2 or direct_norm.shape[0] != y_inc_raw.shape[0]:
            return y_inc_raw

        rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(rot_slice, slice):
            rot_slice = slice(0, y_inc_raw.shape[-1])
        rot_len = int(rot_slice.stop - rot_slice.start)
        if rot_len <= 0 or (rot_len % 6) != 0:
            return y_inc_raw
        J = rot_len // 6

        try:
            direct_raw = self._denorm(direct_norm)
        except Exception:
            return y_inc_raw
        if not torch.is_tensor(direct_raw) or direct_raw.shape != y_inc_raw.shape:
            return y_inc_raw
        columns = getattr(getattr(self, "loss_fn", None), '_rot6d_columns', ("X", "Z"))
        cols = tuple(columns) if isinstance(columns, (list, tuple)) and len(columns) >= 2 else ("X", "Z")

        try:
            inc6 = reproject_rot6d(y_inc_raw[..., rot_slice]).view(y_inc_raw.shape[0], J, 6)
            dir6 = reproject_rot6d(direct_raw[..., rot_slice]).view(y_inc_raw.shape[0], J, 6)
            R_inc = rot6d_to_matrix(inc6, columns=cols)
            R_dir = rot6d_to_matrix(dir6, columns=cols)
        except Exception:
            return y_inc_raw

        lam = lambda_fusion
        try:
            if lam.dim() == 1 and lam.shape[0] == y_inc_raw.shape[0]:
                lam = lam.unsqueeze(-1)
            if lam.dim() == 2 and lam.shape[-1] == 1:
                lam = lam.expand(lam.shape[0], J)
            if lam.dim() == 3 and lam.shape[-1] == 1 and lam.shape[-2] == J:
                lam = lam.squeeze(-1)
            if lam.dim() == 3 and lam.shape[-2] == 1 and lam.shape[-1] == J:
                lam = lam.squeeze(-2)
            if lam.dim() != 2 or lam.shape[0] != y_inc_raw.shape[0] or lam.shape[-1] != J:
                return y_inc_raw
            lam = lam.to(device=y_inc_raw.device, dtype=y_inc_raw.dtype).clamp(0.0, 1.0)
        except Exception:
            return y_inc_raw

        try:
            R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
            omega = so3_log_map(R_res)
            R_step = torch.matmul(so3_exp_map(omega * lam.unsqueeze(-1)), R_inc)
            rot_blend6 = matrix_to_rot6d(R_step, columns=cols).reshape(y_inc_raw.shape[0], rot_len)
            y_blend = y_inc_raw.clone()
            y_blend[..., rot_slice] = rot_blend6
            return y_blend
        except Exception:
            return y_inc_raw

    def _reproject_cond_to_local_frame(self, cond_raw, yaw_gt, yaw_pred):
        """
        将条件信息（目标方向/速度）重投影到模型预测的局部坐标系。

        参数:
            cond_raw: [B, cond_dim] 原始条件，格式: [..., dir_x, dir_y, speed]
            yaw_gt: [B] 或 [B, 1] GT的根骨朝向（世界坐标系）
            yaw_pred: [B] 或 [B, 1] 模型预测的根骨朝向（世界坐标系）

        返回:
            重投影后的 cond_raw，方向分量旋转到模型的局部坐标系
        """
        if cond_raw is None:
            return None

        import torch

        device = cond_raw.device
        dtype = cond_raw.dtype

        # 确保 yaw 是 [B] 形状
        if yaw_gt.dim() > 1:
            yaw_gt = yaw_gt.squeeze(-1)
        if yaw_pred.dim() > 1:
            yaw_pred = yaw_pred.squeeze(-1)

        # 计算朝向偏差：Δyaw = yaw_pred - yaw_gt
        delta_yaw = yaw_pred - yaw_gt
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))  # 归一化到 [-π, π]

        # 解析 cond_raw: [...action_dims, dir_x, dir_y, speed]
        cond_dim = cond_raw.shape[-1]
        if cond_dim < 3:
            self._raise_norm_error("_reproject_cond_to_local_frame cond_raw 最少需要 [dir_x, dir_y, speed]")

        action_dim = cond_dim - 3
        cond_reprojected = cond_raw.clone()

        # 提取方向分量
        dir_world = cond_raw[..., action_dim:action_dim + 2]  # [B, 2]

        # 将方向旋转 -Δyaw，转换到模型的局部坐标系
        # 旋转矩阵: [[cos(-θ), -sin(-θ)], [sin(-θ), cos(-θ)]]
        cos_delta = torch.cos(-delta_yaw)
        sin_delta = torch.sin(-delta_yaw)

        dir_local_x = dir_world[..., 0] * cos_delta - dir_world[..., 1] * sin_delta
        dir_local_y = dir_world[..., 0] * sin_delta + dir_world[..., 1] * cos_delta

        # 写回重投影后的方向
        cond_reprojected[..., action_dim] = dir_local_x
        cond_reprojected[..., action_dim + 1] = dir_local_y

        # 速度保持不变（标量，与朝向无关）

        return cond_reprojected

    def _freerun_traj_loss(self, state_sub, gt_sub, preds_free):
        """
        在 free-run 窗口内对根部世界位置施加“轨迹锚点”约束，并做时间加权。

        思路:
            - 使用 normalizer 将 state_sub / gt_sub / preds_free['out'] 反归一化到 RAW 空间。
            - 从 RAW 中提取 RootVelocity (Y 空间) 并在窗口内积分得到预测/GT 轨迹。
            - 计算两条轨迹在每个时间步的 L2 位置误差，并对时间维度做线性升权。

        要求:
            - DataNormalizer 已配置 rootpos_x_slice / rootvel_y_slice。
            - preds_free 必须是包含 'out' (Y-norm) 的 dict。
        """
        import torch, math

        norm = getattr(self, "normalizer", None)
        if norm is None:
            return None

        # RootPosition / RootVelocity 切片
        rootpos_x_sl = getattr(self, "rootpos_x_slice", None)
        rootvel_y_sl = getattr(self, "rootvel_slice", None)
        if not (isinstance(rootpos_x_sl, slice) and isinstance(rootvel_y_sl, slice)):
            return None

        if not isinstance(preds_free, dict):
            return None
        y_pred_norm = preds_free.get("out", None)
        if y_pred_norm is None:
            return None

        try:
            # 反归一化到 RAW 空间
            x_raw = norm.denorm_x(state_sub)
            y_pred_raw = norm.denorm_y(y_pred_norm)
            y_gt_raw = norm.denorm_y(gt_sub)
        except Exception:
            return None

        # 形状对齐: [B, T, ...]
        if x_raw.dim() != 3 or y_pred_raw.dim() != 3 or y_gt_raw.dim() != 3:
            return None

        B, T_x, _ = x_raw.shape
        _, T_pred, _ = y_pred_raw.shape
        _, T_gt, _ = y_gt_raw.shape
        # 轨迹长度由 root_vel 的有效长度决定
        T_vel = min(T_pred, T_gt)
        if T_vel <= 0 or T_x <= 0:
            return None

        # 起点位置: 使用当前窗口的第一帧 X(raw) 的 RootPosition
        try:
            pos0 = x_raw[:, 0, rootpos_x_sl]  # [B, P]
        except Exception:
            return None

        # RootVelocity (Y 空间)
        try:
            vel_pred = y_pred_raw[:, :T_vel, rootvel_y_sl]
            vel_gt = y_gt_raw[:, :T_vel, rootvel_y_sl]
        except Exception:
            return None

        if vel_pred.numel() == 0 or vel_gt.numel() == 0:
            return None

        # 仅使用与 RootPosition 维度匹配的前几个分量（通常是 XY 或 XYZ）
        P = pos0.shape[-1]
        vel_pred = vel_pred[..., :P]
        vel_gt = vel_gt[..., :P]

        dt = 1.0 / max(float(getattr(self, "bone_hz", 60.0) or 60.0), 1e-6)

        # 通过积分 root velocity 得到预测/GT 轨迹
        # pos_t = pos_{t-1} + v_t * dt
        pos_pred = []
        pos_gt = []
        pos_pred_t = pos0
        pos_gt_t = pos0
        for t in range(T_vel):
            pos_pred_t = pos_pred_t + vel_pred[:, t, :] * dt
            pos_gt_t = pos_gt_t + vel_gt[:, t, :] * dt
            pos_pred.append(pos_pred_t)
            pos_gt.append(pos_gt_t)

        pos_pred = torch.stack(pos_pred, dim=1)  # [B, T_vel, P]
        pos_gt = torch.stack(pos_gt, dim=1)      # [B, T_vel, P]

        # 每步位置误差范数
        traj_err = torch.norm(pos_pred - pos_gt, dim=-1)  # [B, T_vel]

        # 时间加权: 后期步数权重更大，强调长程一致性
        # 如: w_t = linspace(1.0, w_max, T_vel)
        w_max = float(getattr(self, "freerun_traj_time_weight_max", 2.0) or 2.0)
        w_max = max(1.0, w_max)
        time_weights = torch.linspace(1.0, w_max, steps=T_vel, device=traj_err.device, dtype=traj_err.dtype)
        traj_loss = (traj_err * time_weights.unsqueeze(0)).mean()

        stats = {
            "freerun_traj_loss": float(traj_loss.detach().cpu()),
            "freerun_traj_err_mean": float(traj_err.mean().detach().cpu()),
            "freerun_traj_err_last": float(traj_err[:, -1].mean().detach().cpu()),
        }
        return traj_loss, stats

    def _apply_free_carry(self, x_prev, y_denorm, cond_next_raw=None):
        """
        将模型预测的 Y(raw) 写回下一帧的 X(raw)，并根据 cond 信息更新根部位置/速度。
        """
        x_next = x_prev.clone()
        import torch, math
        device = x_prev.device
        dtype = x_prev.dtype

        # --- 1) 写回骨骼旋转 ---
        rx = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
        ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not (isinstance(rx, slice) and isinstance(ry, slice)):
            self._raise_norm_error("_apply_free_carry 缺少 rot6d 切片")
        if (rx.stop - rx.start) != (ry.stop - ry.start):
            self._raise_norm_error("_apply_free_carry rot6d 区间长度不一致")
        x_next[..., rx] = y_denorm[..., ry]

        # 预解析 cond 原始信息：动作维度 + dir(2) + speed(1) —— 必须存在
        if cond_next_raw is None:
            self._raise_norm_error("_apply_free_carry 缺少 cond_next_raw（应包含方向与速度信息）")
        cond_raw = torch.as_tensor(cond_next_raw, device=device, dtype=dtype)
        if cond_raw.dim() == 1:
            cond_raw = cond_raw.unsqueeze(0)
        if cond_raw.shape[0] != x_prev.shape[0]:
            cond_raw = cond_raw.expand(x_prev.shape[0], -1)
        cond_dim = cond_raw.shape[-1]
        if cond_dim < 3:
            self._raise_norm_error("_apply_free_carry cond_next_raw 最少需要 [dir_x, dir_y, speed]")
        action_dim = max(0, cond_dim - 3)
        cond_dir = cond_raw[..., action_dim:action_dim + 2]
        cond_speed = cond_raw[..., action_dim + 2]

        # cond_dir 已是世界系（转换脚本 convert_json_to_npz 已旋到 UE 世界坐标）
        cond_dir_world = cond_dir
        dir_norm = cond_dir_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit_world = cond_dir_world / dir_norm

        # --- 3) 衍生角速度 ---
        av_sl = getattr(self, 'angvel_x_slice', None)
        if isinstance(av_sl, slice):
            J = (rx.stop - rx.start) // 6
            if J <= 0:
                self._raise_norm_error("_apply_free_carry rot6d 切片无有效关节")
            prev6 = x_prev[..., rx].reshape(x_prev.shape[0], J, 6)
            curr6 = x_next[..., rx].reshape(x_prev.shape[0], J, 6)
            Rp = rot6d_to_matrix(prev6)
            Rc = rot6d_to_matrix(curr6)
            Rseq = torch.stack([Rp, Rc], dim=1)
            fps = float(getattr(self, 'bone_hz', 60.0) or 60.0)
            w = angvel_vec_from_R_seq(Rseq, fps=fps)[:, -1]
            x_next[..., av_sl] = w.reshape(x_prev.shape[0], J * 3)

        # --- 4) 根部速度/位置 ---
        rootvel_sl = getattr(self, 'rootvel_x_slice', None)
        rootpos_sl = getattr(self, 'rootpos_x_slice', None)
        if not isinstance(rootvel_sl, slice):
            self._raise_norm_error("_apply_free_carry 缺少 RootVelocity 切片")
        vel_world = dir_unit_world * cond_speed.unsqueeze(-1)
        vel_world = vel_world[..., : (rootvel_sl.stop - rootvel_sl.start)]
        x_next[..., rootvel_sl] = vel_world

        if not isinstance(rootpos_sl, slice):
            self._raise_norm_error("_apply_free_carry 缺少 RootPosition 切片")
        dt = 1.0 / max(float(getattr(self, 'bone_hz', 60.0) or 60.0), 1e-6)
        pos = x_prev[..., rootpos_sl].clone()
        step = vel_world[..., :min(2, vel_world.shape[-1])] * dt
        pos[..., :step.shape[-1]] = pos[..., :step.shape[-1]] + step
        x_next[..., rootpos_sl] = pos

        return x_next

    def _prepare_cond_stat(self, stat: Optional[torch.Tensor], ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if stat is None:
            return None
        import torch
        if not torch.is_tensor(stat):
            stat_t = torch.as_tensor(stat, device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            stat_t = stat.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if stat_t.dim() >= 3:
            stat_t = stat_t.view(stat_t.shape[0], -1)
        if stat_t.dim() == 1:
            stat_t = stat_t.unsqueeze(0)
        if stat_t.size(0) == 1 and ref_tensor.size(0) > 1:
            stat_t = stat_t.expand(ref_tensor.size(0), -1).contiguous()
        return stat_t

    def _normalize_cond_from_raw(
        self,
        cond_raw: Optional[torch.Tensor],
        cond_mu: Optional[torch.Tensor],
        cond_std: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        import torch
        if cond_raw is None or cond_mu is None or cond_std is None:
            return None
        if cond_mu.dim() == 3:
            cond_mu = cond_mu.squeeze(1)
        if cond_std.dim() == 3:
            cond_std = cond_std.squeeze(1)
        if cond_mu.shape != cond_raw.shape:
            # broadcast along batch if mu/std have single row
            if cond_mu.size(0) == 1 and cond_raw.size(0) > 1:
                cond_mu = cond_mu.expand(cond_raw.size(0), -1)
            if cond_std.size(0) == 1 and cond_raw.size(0) > 1:
                cond_std = cond_std.expand(cond_raw.size(0), -1)
        std = cond_std.clamp_min(1e-6)
        cond_norm = (cond_raw - cond_mu) / std
        clamp_val = float(getattr(self, 'cond_norm_clip', 6.0) or 0.0)
        if clamp_val > 0:
            cond_norm = cond_norm.clamp(-clamp_val, clamp_val)
        return cond_norm

    def _pose_hist_params(self, ref: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns pose-history normalizer tensors aligned with the reference tensor's device/dtype.
        """
        if self.pose_hist_scales is None or self.pose_hist_dim <= 0:
            return None, None, None
        device = ref.device
        dtype = ref.dtype
        scales = self.pose_hist_scales.to(device=device, dtype=dtype)
        mu = self.pose_hist_mu.to(device=device, dtype=dtype) if self.pose_hist_mu is not None else None
        std = self.pose_hist_std.to(device=device, dtype=dtype) if self.pose_hist_std is not None else None
        return scales, mu, std

    def _infer_root_yaw_from_rot6d(self, y_denorm: "torch.Tensor"):
        import torch
        ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(ry, slice):
            return None
        try:
            rot_flat = y_denorm[..., ry]
        except Exception:
            return None
        if rot_flat.numel() == 0:
            return None
        J = (ry.stop - ry.start) // 6
        if J <= 0:
            return None
        try:
            rot6d = reproject_rot6d(rot_flat).view(rot_flat.shape[0], J, 6)
            R = rot6d_to_matrix(rot6d)
        except Exception:
            return None
        root_idx = int(getattr(self, 'eval_root_idx', 0))
        root_idx = max(0, min(J - 1, root_idx))
        root_R = R[:, root_idx]
        up_axis = int(getattr(self, 'eval_up_axis', getattr(self, '_up_axis', 2)))
        up_axis = max(0, min(2, up_axis))
        forward_axis = int(getattr(self, 'yaw_forward_axis', 2))
        forward_axis = max(0, min(2, forward_axis))
        forward_vec = root_R[..., forward_axis]
        planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
        if len(planar_axes) != 2:
            return None
        ax0, ax1 = planar_axes
        yaw = torch.atan2(forward_vec[..., ax1], forward_vec[..., ax0])
        offset = float(getattr(self, 'yaw_forward_axis_offset', 0.0))
        if offset != 0.0:
            yaw = yaw - offset
        return torch.atan2(torch.sin(yaw), torch.cos(yaw))
    def _train_augment_if_needed(self, state_seq, gt_seq, cond_seq=None):
        """仅训练阶段使用的时序/噪声增强。"""
        import torch
        aug = getattr(self, 'augmentor', None)
        if aug is None:
            return state_seq, gt_seq, cond_seq

        prob = float(getattr(aug, 'time_warp_prob', 0.0) or 0.0)
        if prob > 0.0 and torch.rand(1, device=state_seq.device).item() < prob:
            scale = float(torch.empty(1, device=state_seq.device).uniform_(0.85, 1.15).item())
            state_seq = aug._time_warp(state_seq, scale)
            gt_seq = aug._time_warp(gt_seq, scale)
            if (cond_seq is not None) and (cond_seq.dim() == 3):
                cond_seq = aug._time_warp(cond_seq, scale)

        std = float(getattr(aug, 'noise_std', 0.0) or 0.0)
        if std > 0.0:
            def _n(sl):
                if isinstance(sl, slice):
                    state_seq[:, :, sl] = state_seq[:, :, sl] + torch.randn_like(state_seq[:, :, sl]) * std

            _n(getattr(self, 'rot6d_x_slice', None))
            _n(getattr(self, 'rootvel_x_slice', None))
            _n(getattr(self, 'angvel_x_slice', None))

        return state_seq, gt_seq, cond_seq

    def _metrics_json_safe(self, value):
        import math
        import torch
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if isinstance(value, dict):
            return {str(k): self._metrics_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._metrics_json_safe(v) for v in value]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                try:
                    return self._metrics_json_safe(value.item())
                except (RuntimeError, TypeError, ValueError) as exc:
                    _phasec_warn_once(
                        "metrics_json/tensor_item",
                        "failed to scalarize tensor metric via .item(); serializing as list instead",
                        exc,
                    )
            return [self._metrics_json_safe(v) for v in value.detach().cpu().tolist()]
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[arg-type]
            return [self._metrics_json_safe(v) for v in value.tolist()]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:
            if np is not None and isinstance(value, np.generic):  # type: ignore[arg-type]
                return self._metrics_json_safe(float(value))
        except (RuntimeError, TypeError, ValueError) as exc:
            _phasec_warn_once(
                "metrics_json/numpy_generic",
                "failed to convert numpy scalar metric; fallback to generic serialization",
                exc,
            )
        if hasattr(value, 'item') and not isinstance(value, (int, bool, str)):
            try:
                return self._metrics_json_safe(value.item())
            except (RuntimeError, TypeError, ValueError) as exc:
                _phasec_warn_once(
                    "metrics_json/generic_item",
                    "failed to scalarize metric via .item(); fallback to string serialization",
                    exc,
                )
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        return str(value)

    def register_metric_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """注册监听器，在每次记录指标时得到通知（运行在同一进程内）。"""
        if not callable(callback):
            return
        if callback not in self._metric_callbacks:
            self._metric_callbacks.append(callback)

    def unregister_metric_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if not callable(callback):
            return
        try:
            self._metric_callbacks.remove(callback)
        except ValueError:
            pass

    def get_metric_history(self, tag: Optional[str] = None, last: Optional[int] = None) -> list[dict[str, Any]]:
        """返回内存中的指标快照，用于训练过程内的策略决策。"""
        records = self.metric_history
        if tag is not None:
            records = [rec for rec in records if rec.get('tag') == tag]
        if last is not None and last > 0:
            records = records[-last:]
        return [dict(rec) for rec in records]

    def latest_epoch_metrics(self, tag: Optional[str] = None) -> Optional[dict[str, Any]]:
        """获取最近一次写入的指标（可按 tag 过滤）。"""
        if tag is not None:
            record = self.latest_metrics.get(str(tag))
            return None if record is None else dict(record)
        if not self.metric_history:
            return None
        return dict(self.metric_history[-1])

    def _record_epoch_metrics(self, metrics: Dict[str, Any], *, tag: str, epoch: int) -> None:
        if metrics is None:
            return
        payload: dict[str, Any] = {
            'epoch': int(epoch),
            'tag': str(tag),
            'metrics': self._metrics_json_safe(metrics),
        }
        tf_ratio = getattr(self, '_last_tf_ratio', None)
        if tf_ratio is not None:
            try:
                payload['tf_ratio'] = float(tf_ratio)
            except Exception:
                payload['tf_ratio'] = tf_ratio
        maxlen = max(1, int(getattr(self, 'metric_history_maxlen', 256) or 256))
        self.metric_history.append(payload)
        if len(self.metric_history) > maxlen:
            self.metric_history.pop(0)
        self.latest_metrics[str(tag)] = payload
        for callback in list(self._metric_callbacks):
            try:
                callback(payload)
            except Exception as exc:
                print(f"[MetricsCallback][WARN] {callback} raised: {exc}")

    def _dump_metrics_json(self, metrics: Dict[str, Any], *, tag: str, epoch: int) -> None:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return
        try:
            metrics_dir = os.path.join(out_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            payload = {
                'epoch': int(epoch),
                'tag': str(tag),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                'metrics': self._metrics_json_safe(metrics),
            }
            json_path = os.path.join(metrics_dir, f'{tag}_ep{int(epoch):03d}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[MetricsWrite][WARN] failed to write {tag} metrics @ep{epoch}: {exc}")

    @staticmethod
    def _panel_metric_scalar(metrics: Mapping[str, Any], key: str, *, fallback: float = float('nan')) -> float:
        val = metrics.get(key, fallback)
        try:
            return float(val)
        except Exception:
            return float(fallback)

    def _write_basetrain_keybone_group_summary(self) -> None:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return

        def _panel_record(entry: Mapping[str, Any], *, phase: str) -> Dict[str, Any]:
            metrics = entry.get('metrics', {}) if isinstance(entry.get('metrics', {}), Mapping) else {}
            keybone_summary = metrics.get('KeyBoneSummary', {}) if isinstance(metrics.get('KeyBoneSummary', {}), Mapping) else {}
            geo_local = self._panel_metric_scalar(metrics, 'GeoLocalDeg')
            key_geo_local = self._panel_metric_scalar(
                metrics,
                'KeyBone/GeoLocalDegMean',
                fallback=self._panel_metric_scalar(keybone_summary, 'GeoLocalDegMean'),
            )
            record: Dict[str, Any] = {
                'epoch': int(entry.get('epoch', -1) or -1),
                'GeoLocalDeg': geo_local,
                'KeyBoneGeoLocalDegMean': key_geo_local,
                'FreeRunGeoLocalDeg': geo_local if phase == 'freerun' else float('nan'),
                'FreeRunKeyBoneGeoLocalDegMean': key_geo_local if phase == 'freerun' else float('nan'),
                'group_mean': keybone_summary.get('group_mean', {}) if isinstance(keybone_summary.get('group_mean', {}), Mapping) else {},
                'GeoDriftSlopeProxy': self._panel_metric_scalar(
                    metrics,
                    'GeoDriftSlopeProxy',
                    fallback=self._panel_metric_scalar(metrics, 'GeoDriftSlope'),
                ),
            }
            return record

        def _train_direct_group_norm_record(entry: Mapping[str, Any]) -> Dict[str, Any]:
            metrics = entry.get('metrics', {}) if isinstance(entry.get('metrics', {}), Mapping) else {}
            return {
                'epoch': int(entry.get('epoch', -1) or -1),
                'dir_leg_base': self._panel_metric_scalar(metrics, 'dir_leg_base'),
                'dir_nonleg_base': self._panel_metric_scalar(metrics, 'dir_nonleg_base'),
                'dir_nonleg_effective_base': self._panel_metric_scalar(metrics, 'dir_nonleg_effective_base'),
                'dir_arm_base': self._panel_metric_scalar(metrics, 'dir_arm_base'),
                'dir_else_base': self._panel_metric_scalar(metrics, 'dir_else_base'),
                'leg_over_nonleg': self._panel_metric_scalar(metrics, 'leg_over_nonleg'),
                'leg_over_nonleg_effective': self._panel_metric_scalar(metrics, 'leg_over_nonleg_effective'),
                'arm_over_else': self._panel_metric_scalar(metrics, 'arm_over_else'),
                'direct_pose_arm_else_balance_active': self._panel_metric_scalar(metrics, 'direct_pose_arm_else_balance_active', fallback=0.0),
                'direct_pose_loss_arm_weight': self._panel_metric_scalar(metrics, 'direct_pose_loss_arm_weight', fallback=1.0),
                'direct_pose_loss_else_weight': self._panel_metric_scalar(metrics, 'direct_pose_loss_else_weight', fallback=1.0),
                'dir_group_norm_used': self._panel_metric_scalar(metrics, 'dir_group_norm_used', fallback=0.0),
                'dir_group_norm_leg_raw': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_raw'),
                'dir_group_norm_nonleg_raw': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_raw'),
                'dir_group_norm_leg_clamped': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_clamped'),
                'dir_group_norm_nonleg_clamped': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_clamped'),
                'dir_group_norm_leg_ema': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_ema'),
                'dir_group_norm_nonleg_ema': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_ema'),
                'dir_group_norm_leg_hit_min': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_hit_min', fallback=0.0),
                'dir_group_norm_leg_hit_max': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_hit_max', fallback=0.0),
                'dir_group_norm_nonleg_hit_min': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_hit_min', fallback=0.0),
                'dir_group_norm_nonleg_hit_max': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_hit_max', fallback=0.0),
                'GroupNormLegClampHitRate': self._panel_metric_scalar(metrics, 'dir_group_norm_leg_hit_any', fallback=0.0),
                'GroupNormNonlegClampHitRate': self._panel_metric_scalar(metrics, 'dir_group_norm_nonleg_hit_any', fallback=0.0),
                'direct_grad_norm_trunk': self._panel_metric_scalar(metrics, 'direct_grad_norm_trunk'),
                'direct_grad_norm_out_leg': self._panel_metric_scalar(metrics, 'direct_grad_norm_out_leg'),
                'direct_grad_norm_out_nonleg': self._panel_metric_scalar(metrics, 'direct_grad_norm_out_nonleg'),
                'direct_grad_norm_out_arm': self._panel_metric_scalar(metrics, 'direct_grad_norm_out_arm'),
                'direct_grad_norm_out_else': self._panel_metric_scalar(metrics, 'direct_grad_norm_out_else'),
                'direct_grad_ratio_nonleg_over_leg': self._panel_metric_scalar(metrics, 'direct_grad_ratio_nonleg_over_leg'),
                'direct_grad_ratio_arm_over_else': self._panel_metric_scalar(metrics, 'direct_grad_ratio_arm_over_else'),
            }

        teacher_rows = [_panel_record(rec, phase='teacher') for rec in self.metric_history if rec.get('tag') == 'teacher']
        freerun_rows = [_panel_record(rec, phase='freerun') for rec in self.metric_history if rec.get('tag') == 'valfree']
        train_direct_rows = [
            _train_direct_group_norm_record(rec)
            for rec in self.metric_history
            if rec.get('tag') == 'train'
        ]

        def _best_row(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
            best = None
            best_val = float('inf')
            for row in rows:
                try:
                    val = float(row.get(key, float('nan')))
                except Exception:
                    continue
                if not _math.isfinite(val):
                    continue
                if val < best_val:
                    best_val = val
                    best = dict(row)
            return best

        payload: Dict[str, Any] = {
            'teacher': teacher_rows,
            'freerun': freerun_rows,
            'train_direct_group_norm': train_direct_rows,
        }
        best_teacher = _best_row(teacher_rows, 'GeoLocalDeg')
        if best_teacher is not None:
            payload['best_teacher_by_GeoLocalDeg'] = best_teacher
        best_free = _best_row(freerun_rows, 'GeoDriftSlopeProxy')
        if best_free is not None:
            payload['best_free_by_GeoDriftSlopeProxy'] = best_free

        summary_path = Path(out_dir) / 'basetrain_keybone_group_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_val_metrics(self, epoch: int, metrics: Mapping[str, Any]) -> Optional[Path]:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return None
        try:
            self._dump_metrics_json(dict(metrics), tag='valfree', epoch=epoch)
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            _phasec_warn_once(
                "metrics/valfree_dump",
                "failed to dump valfree metrics JSON snapshot",
                exc,
            )
        metrics_dir = Path(out_dir) / 'metrics'
        json_path = metrics_dir / f'valfree_ep{int(epoch):03d}.json'
        return json_path if json_path.exists() else None

    @torch.no_grad()
    def _diagnose_free_run(
        self,
        batch,
        predY,
        gtY,
        predsX,
        period_seq_pred,
        motion_seq,
        y_seq,
        contacts_seq,
        angvel_seq,
        pose_hist_seq,
        angvel_raw_seq=None,
    ):
        diag = _diagnose_free_run_impl(
            self,
            batch,
            predY,
            gtY,
            predsX,
            period_seq_pred,
            motion_seq,
            y_seq,
            contacts_seq,
            angvel_seq,
            pose_hist_seq,
            angvel_raw_seq=angvel_raw_seq,
        )
        if diag is None:
            clip = None
            start = None
            if isinstance(batch, dict):
                clip = batch.get('clip_id')
                start = batch.get('start')
            msg = f"_diagnose_free_run returned None (clip={clip}, start={start})"
            print(f"[FreeRunDiag][WARN] {msg}")
        return diag


    def _dump_nan_grad_report(self, epoch, batch_idx, batch, state_seq, gt_seq, preds_dict, loss_value, stats):
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return
        limit = int(getattr(self, 'nan_grad_report_limit', 0) or 0)
        if self.nan_grad_reports >= limit:
            return
        import os, json

        def _tensor_stats(tensor):
            if tensor is None:
                return None
            try:
                t = tensor.detach()
                if t.numel() == 0:
                    return {'shape': list(t.shape), 'numel': 0}
                t = t.to(dtype=torch.float32, device='cpu')
                return {
                    'shape': list(t.shape),
                    'numel': int(t.numel()),
                    'min': float(t.min().item()),
                    'max': float(t.max().item()),
                    'mean': float(t.mean().item()),
                    'std': float(t.std().item()),
                }
            except Exception as exc:
                return {'error': str(exc)}

        try:
            os.makedirs(os.path.join(out_dir, 'nan_grad'), exist_ok=True)
            payload = {
                'epoch': int(epoch),
                'batch_idx': int(batch_idx),
                'tf_ratio': float(getattr(self, '_last_tf_ratio', 1.0)),
                'loss': float(loss_value),
                'loss_parts': dict(stats) if isinstance(stats, dict) else {},
                'state_stats': _tensor_stats(state_seq),
                'gt_stats': _tensor_stats(gt_seq),
                'pred_out_stats': _tensor_stats(preds_dict.get('out') if isinstance(preds_dict, dict) else None),
                'pred_delta_stats': _tensor_stats(preds_dict.get('delta') if isinstance(preds_dict, dict) else None),
                'batch_meta': {},
            }
            if isinstance(batch, dict):
                clip_id = batch.get('clip_id')
                start = batch.get('start')
                if clip_id is not None:
                    clip_id_int = _phasec_safe_int(clip_id)
                    if clip_id_int is not None:
                        payload['batch_meta']['clip_id'] = clip_id_int
                    else:
                        _phasec_warn_once(
                            "grad_nan_dump/clip_id",
                            f"failed to parse clip_id as int (type={type(clip_id).__name__}); omit from payload",
                        )
                if start is not None:
                    start_int = _phasec_safe_int(start)
                    if start_int is not None:
                        payload['batch_meta']['start'] = start_int
                    else:
                        _phasec_warn_once(
                            "grad_nan_dump/start",
                            f"failed to parse start as int (type={type(start).__name__}); omit from payload",
                        )
            fname = os.path.join(out_dir, 'nan_grad', f'ep{int(epoch):03d}_b{int(batch_idx):05d}.json')
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.nan_grad_reports += 1
            print(f"[GradNan] dumped diagnostic to {fname}")
        except Exception as exc:
            print(f"[GradNan][WARN] failed to dump diagnostic: {exc}")


@dataclass(frozen=True)
class FreeRunDiagSequences:
    predX_tensor: Optional[torch.Tensor]
    predX_raw: Optional[torch.Tensor]
    gtX_raw: Optional[torch.Tensor]
    gtX_raw_full: Optional[torch.Tensor]
    cond_raw_seq: Optional[torch.Tensor]
    model: Any


@dataclass
class FreeRunDiagKinematics:
    geo: Optional[torch.Tensor] = None
    geo_local: Optional[torch.Tensor] = None
    w_pred: Optional[torch.Tensor] = None
    w_gt: Optional[torch.Tensor] = None


def _record_diag_metric(
    result: Dict[str, Any],
    diag_scope: str,
    name: str,
    value: Any,
    *,
    extra_scope_aliases: Sequence[str] = (),
) -> None:
    result[name] = value
    scope_aliases: list[str] = []
    if diag_scope == 'free_run':
        scope_aliases.append('FreeRun')
    elif diag_scope == 'single_step':
        scope_aliases.append('SingleStep')
    for alias in extra_scope_aliases:
        if alias not in scope_aliases:
            scope_aliases.append(alias)
    for alias in scope_aliases:
        result[f'{alias}/{name}'] = value


def _collect_diag_sequences(self, *, predsX, motion_seq, batch) -> FreeRunDiagSequences:
    predX_tensor = torch.stack(predsX, dim=1) if predsX else None
    model = getattr(self, 'model', None)
    gtX_raw_full = None
    if motion_seq is not None:
        try:
            flat_motion = motion_seq.reshape(-1, motion_seq.shape[-1])
            gtX_raw_full = self.normalizer.denorm_x(flat_motion).view_as(motion_seq)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在诊断阶段还原 GT X 时失败", exc)

    if predX_tensor is not None:
        flat_pred = predX_tensor.reshape(-1, predX_tensor.shape[-1])
        try:
            predX_raw = self.normalizer.denorm_x(flat_pred).view_as(predX_tensor)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm_x 在诊断阶段还原预测 X 时失败", exc)
        if motion_seq is not None:
            if gtX_raw_full is None:
                self._raise_norm_error("诊断阶段缺少 GT RAW 序列。")
            gtX_raw = gtX_raw_full[:, :predX_tensor.shape[1]]
        else:
            gtX_raw = None
    else:
        predX_raw = None
        gtX_raw = None

    cond_raw_seq = None
    if isinstance(batch, dict):
        cond_raw_seq = batch.get("cond_tgt_raw")
        if cond_raw_seq is None:
            cond_raw_seq = batch.get("cond_in")

    return FreeRunDiagSequences(
        predX_tensor=predX_tensor,
        predX_raw=predX_raw,
        gtX_raw=gtX_raw,
        gtX_raw_full=gtX_raw_full,
        cond_raw_seq=cond_raw_seq,
        model=model,
    )


def _compute_input_drift_metrics(
    self,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    seqs: FreeRunDiagSequences,
    *,
    period_seq_pred,
) -> None:
    predX_raw = seqs.predX_raw
    gtX_raw = seqs.gtX_raw

    if isinstance(cfg.rv_x, slice) and predX_raw is not None and gtX_raw is not None:
        _record_diag_metric(
            result,
            cfg.diag_scope,
            'RootVelMAE',
            float((predX_raw[..., cfg.rv_x] - gtX_raw[..., cfg.rv_x]).abs().mean().item()),
        )
        if cfg.diag_input_stats:
            diff = (predX_raw[..., cfg.rv_x] - gtX_raw[..., cfg.rv_x]).abs()
            result['RootVelMAE_std'] = float(diff.std().item())
        if predX_raw.shape[1] > 0 and gtX_raw.shape[1] > 0:
            rv_end = (predX_raw[:, -1, cfg.rv_x] - gtX_raw[:, -1, cfg.rv_x]).abs().mean()
            _record_diag_metric(result, cfg.diag_scope, 'RootVelEndMAE', float(rv_end.item()))

    if isinstance(cfg.rot6d_x, slice) and predX_raw is not None and gtX_raw is not None:
        try:
            Bx, Tx, Dx = predX_raw.shape
            px = predX_raw[..., cfg.rot6d_x]
            gx = gtX_raw[..., cfg.rot6d_x]
            if Dx > 0 and px.shape[-1] % 6 == 0:
                Jx = px.shape[-1] // 6
                px6 = reproject_rot6d(px.reshape(-1, px.shape[-1]))
                gx6 = reproject_rot6d(gx.reshape(-1, gx.shape[-1]))
                Rp = rot6d_to_matrix(px6.view(-1, Jx, 6)).view(Bx, Tx, Jx, 3, 3)
                Rg = rot6d_to_matrix(gx6.view(-1, Jx, 6)).view(Bx, Tx, Jx, 3, 3)
                geo_in = geodesic_R(Rp, Rg)
                _record_diag_metric(
                    result,
                    cfg.diag_scope,
                    'InputRotErrorDeg',
                    float((geo_in.mean() * cfg.deg).item()),
                    extra_scope_aliases=('FreeRun',),
                )
                if cfg.diag_input_stats:
                    result['InputRotErrorDeg_max'] = float((geo_in.max() * cfg.deg).item())
                    result['InputRotErrorDeg_std'] = float((geo_in.std() * cfg.deg).item())
                try:
                    geo_in_deg = geo_in * cfg.deg
                    mean_curve = geo_in_deg.mean(dim=-1).mean(dim=0)
                    max_curve = geo_in_deg.max(dim=-1).values.max(dim=0).values
                    _record_optional_diag_curve(
                        result,
                        metric_name='InputRotErrorDeg',
                        curve=mean_curve,
                        curve_max=max_curve,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    _phasec_warn_once(
                        "diag/input_rot_error_curve",
                        "failed to record InputRotErrorDeg curve payload",
                        exc,
                    )
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _phasec_warn_once(
                "diag/input_rot_error",
                "failed to compute InputRotErrorDeg diagnostics",
                exc,
            )

    cond_raw_seq = seqs.cond_raw_seq
    if torch.is_tensor(cond_raw_seq):
        cond_raw_seq = cond_raw_seq.float()
        if cond_raw_seq.dim() == 2:
            cond_raw_seq = cond_raw_seq.unsqueeze(0)
        if cond_raw_seq.dim() == 3 and predX_raw is not None:
            B = predX_raw.shape[0]
            if cond_raw_seq.shape[0] == B:
                start_idx = 1
                horizon = predX_raw.shape[1]
                if cond_raw_seq.shape[1] >= start_idx + horizon:
                    cond_slice = cond_raw_seq[:, start_idx:start_idx + horizon]
                else:
                    cond_slice = cond_raw_seq[:, -horizon:]
                cond_dim = cond_slice.shape[-1]
                if cond_dim >= 2:
                    if cond_dim >= 3:
                        dir_slice = cond_slice[..., cond_dim - 3:cond_dim - 1]
                        speed_slice = cond_slice[..., -1]
                    else:
                        dir_slice = cond_slice[..., -2:]
                        speed_slice = dir_slice.norm(dim=-1)
                    L = min(cond_slice.shape[1], predX_raw.shape[1])
                    if L > 0:
                        device = predX_raw.device
                        dir_slice = dir_slice[:, :L].to(device)
                        speed_slice = speed_slice[:, :L].to(device)
                        dir_norm = dir_slice.norm(dim=-1).clamp_min(1e-6)
                        dir_unit = dir_slice / dir_norm.unsqueeze(-1)
                        yaw_cmd_world = torch.atan2(dir_unit[..., 1], dir_unit[..., 0])
                        yaw_cmd = torch.atan2(
                            torch.sin(yaw_cmd_world - cfg.yaw_forward_axis_offset),
                            torch.cos(yaw_cmd_world - cfg.yaw_forward_axis_offset),
                        )
                        _ = yaw_cmd
                        if isinstance(cfg.rv_x, slice):
                            cond_vel = dir_unit * speed_slice.unsqueeze(-1)
                            vel_pred = predX_raw[:, :L, cfg.rv_x]
                            _record_diag_metric(
                                result,
                                cfg.diag_scope,
                                'CondVelVsPredMAE',
                                float((vel_pred - cond_vel).abs().mean().item()),
                                extra_scope_aliases=('FreeRun',),
                            )
                            if gtX_raw is not None and gtX_raw.shape[1] >= start_idx + L:
                                vel_gt = gtX_raw[:, start_idx:start_idx + L, cfg.rv_x]
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    'CondVelVsGTMAE',
                                    float((vel_gt - cond_vel).abs().mean().item()),
                                    extra_scope_aliases=('FreeRun',),
                                )

    if period_seq_pred:
        try:
            norm_period = []
            for p in period_seq_pred:
                if p.dim() == 3 and p.size(1) == 1:
                    norm_period.append(p.squeeze(1))
                elif p.dim() == 2:
                    norm_period.append(p)
                else:
                    norm_period.append(p.reshape(p.shape[0], -1))
            if norm_period:
                period_tensor = torch.stack(norm_period, dim=1)
                result['period_abs_mean'] = float(period_tensor.abs().mean().item())
                result['period_abs_std'] = float(period_tensor.abs().std().item())
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _phasec_warn_once(
                "diag/period_abs_stats",
                "failed to compute period_abs_mean/std diagnostics",
                exc,
            )


def _compute_contact_and_angvel_metrics(
    self,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    seqs: FreeRunDiagSequences,
    *,
    predY,
    gtY,
    period_seq_pred,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
) -> FreeRunDiagKinematics:
    state = FreeRunDiagKinematics()
    if isinstance(cfg.rot6d_y, slice):
        predY_raw = self._denorm(predY)
        gtY_raw = self._denorm(gtY)
        py = predY_raw[..., cfg.rot6d_y]
        gy = gtY_raw[..., cfg.rot6d_y]
        if py.shape[-1] % 6 == 0:
            J = py.shape[-1] // 6
            py6 = reproject_rot6d(py).view(py.shape[0], py.shape[1], J, 6)
            gy6 = reproject_rot6d(gy).view(gy.shape[0], gy.shape[1], J, 6)
            Rp = rot6d_to_matrix(py6)
            Rg = rot6d_to_matrix(gy6)
            Rp_raw = Rp
            if cfg.eval_align_root and Rp.shape[1] > 0 and 0 <= cfg.root_idx < J:
                Rpr0 = Rp[:, 0, cfg.root_idx]
                Rgr0 = Rg[:, 0, cfg.root_idx]
                R_align = Rgr0 @ Rpr0.transpose(-1, -2)
                Rp = (R_align.view(Rp.shape[0], 1, 1, 3, 3).expand_as(Rp)) @ Rp
                state.geo = geodesic_R(Rp, Rg)
                _record_diag_metric(
                    result,
                    cfg.diag_scope,
                    'GeoDeg',
                    float((state.geo.mean() * cfg.deg).item()),
                    extra_scope_aliases=('SingleStep',),
                )
                try:
                    geo_deg = state.geo * cfg.deg
                    geo_curve = geo_deg.mean(dim=-1).mean(dim=0)
                    geo_curve_max = geo_deg.max(dim=-1).values.max(dim=0).values
                    _record_optional_diag_curve(
                        result,
                        metric_name='GeoDeg',
                        curve=geo_curve,
                        curve_max=geo_curve_max,
                    )
                    _record_diag_metric(result, cfg.diag_scope, 'GeoDegEnd', float(geo_curve[-1].item()))
                    try:
                        if cfg.bone_names:
                            geo_per_bone = {}
                            geo_mean_bone = geo_deg.mean(dim=0)
                            for j, name in enumerate(cfg.bone_names[:geo_mean_bone.shape[1]]):
                                geo_per_bone[name] = geo_mean_bone[:, j].detach().cpu().tolist()
                            _record_optional_diag_curve(
                                result,
                                metric_name='GeoDeg',
                                curve=result.get('GeoDegCurve', []),
                                curve_max=result.get('GeoDegCurveMax'),
                                curve_bones=geo_per_bone,
                            )
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        _phasec_warn_once(
                            "diag/geo_curve_bones",
                            "failed to build GeoDegCurveBones payload",
                            exc,
                        )
                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    _phasec_warn_once(
                        "diag/geo_curve",
                        "failed to compute GeoDeg curve diagnostics",
                        exc,
                    )
                state.geo_local = None
                try:
                    Rp_root = _root_relative_matrices(Rp, cfg.root_idx)
                    Rg_root = _root_relative_matrices(Rg, cfg.root_idx)
                    state.geo_local = geodesic_R(Rp_raw, Rg) * cfg.deg
                    joint_weights = self._joint_weights(Rp, J)
                    if 0 <= cfg.root_idx < joint_weights.numel():
                        joint_weights = joint_weights.clone()
                        joint_weights[cfg.root_idx] = 0.0
                    weights_sum = joint_weights.sum().clamp_min(1e-6)
                    w = joint_weights.view(1, 1, -1)
                    geo_local_mean = (state.geo_local * w).sum() / (
                        weights_sum * state.geo_local.shape[0] * state.geo_local.shape[1]
                    )
                    _record_diag_metric(
                        result,
                        cfg.diag_scope,
                        'GeoLocalDeg',
                        float(geo_local_mean.item()),
                        extra_scope_aliases=('SingleStep',),
                    )
                    step_vals = ((state.geo_local * w).sum(dim=-1) / weights_sum).mean(dim=0)
                    _record_optional_diag_curve(
                        result,
                        metric_name='GeoLocalDeg',
                        curve=step_vals,
                    )
                    if int(step_vals.numel()) >= 2:
                        drift_proxy = float(
                            (step_vals[-1] - step_vals[0]).detach().cpu() / max(1, int(step_vals.numel()) - 1)
                        )
                    else:
                        drift_proxy = float('nan')
                    _record_diag_metric(result, cfg.diag_scope, 'GeoDriftSlopeProxy', drift_proxy)

                    geo_for_max = state.geo_local
                    if 0 <= cfg.root_idx < geo_for_max.shape[-1]:
                        geo_for_max = state.geo_local.clone()
                        geo_for_max[..., cfg.root_idx] = -1e9
                    max_vals = geo_for_max.max(dim=-1).values.max(dim=0).values
                    _record_optional_diag_curve(
                        result,
                        metric_name='GeoLocalDeg',
                        curve=result.get('GeoLocalDegCurve', []),
                        curve_max=max_vals,
                    )
                    _record_diag_metric(result, cfg.diag_scope, 'GeoLocalDegEnd', float(step_vals[-1].item()))

                    try:
                        if cfg.bone_names:
                            geo_local_per_bone = {}
                            geo_local_mean_bone = state.geo_local.mean(dim=0)
                            if 0 <= cfg.root_idx < geo_local_mean_bone.shape[1]:
                                geo_local_mean_bone[:, cfg.root_idx] = 0.0
                            for j, name in enumerate(cfg.bone_names[:geo_local_mean_bone.shape[1]]):
                                geo_local_per_bone[name] = geo_local_mean_bone[:, j].detach().cpu().tolist()
                            _record_optional_diag_curve(
                                result,
                                metric_name='GeoLocalDeg',
                                curve=result.get('GeoLocalDegCurve', []),
                                curve_max=result.get('GeoLocalDegCurveMax'),
                                curve_bones=geo_local_per_bone,
                            )
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        _phasec_warn_once(
                            "diag/geo_local_curve_bones",
                            "failed to build GeoLocalDegCurveBones payload",
                            exc,
                        )
                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    _phasec_warn_once(
                        "diag/geo_local",
                        "failed to compute GeoLocalDeg diagnostics",
                        exc,
                    )
            try:
                Rp_parent = self._parent_relative_matrices(Rp_root)
                Rg_parent = self._parent_relative_matrices(Rg_root)
                state.w_pred = angvel_vec_from_R_seq(Rp_parent, cfg.fps_eval)
                state.w_gt = angvel_vec_from_R_seq(Rg_parent, cfg.fps_eval)
                _record_diag_metric(
                    result,
                    cfg.diag_scope,
                    'AngVelMAE',
                    float((state.w_pred - state.w_gt).abs().mean().item()),
                )
                mag_p = state.w_pred.norm(dim=-1)
                mag_g = state.w_gt.norm(dim=-1)
                mag_avg = 0.5 * (mag_p + mag_g)
                maskA = (mag_avg > cfg.mag_rel_threshold)
                mag_rel = (mag_p - mag_g).abs() / (mag_avg + cfg.mag_rel_beta)
                ang_mag_rel = (mag_rel * maskA).sum(dim=(0, 1)) / maskA.sum(dim=(0, 1)).clamp_min(1)
                _record_diag_metric(result, cfg.diag_scope, 'AngVelMagRel', float(torch.nanmedian(ang_mag_rel).item()))
                try:
                    ang_mae_full = (state.w_pred - state.w_gt).abs()
                    ang_mae_curve = ang_mae_full.mean(dim=(0, 2))
                    ang_mae_bone_curve = None
                    if cfg.bone_names and ang_mae_full.shape[2] == len(cfg.bone_names):
                        ang_mae_bones = ang_mae_full.mean(dim=0)
                        ang_mae_bone_curve = {
                            name: ang_mae_bones[:, j].norm(dim=-1).detach().cpu().tolist()
                            for j, name in enumerate(cfg.bone_names)
                        }
                    _record_optional_diag_curve(
                        result,
                        metric_name='AngVelMAE',
                        curve=ang_mae_curve,
                        curve_bones=ang_mae_bone_curve,
                        scope_alias='SingleStep' if cfg.diag_scope == 'single_step' else None,
                    )

                    dot_full = (state.w_pred * state.w_gt).sum(dim=-1)
                    ang_full = torch.zeros_like(dot_full)
                    valid_full = (mag_p > cfg.angvel_dir_threshold) & (mag_g > cfg.angvel_dir_threshold)
                    if valid_full.any():
                        norm_full = (mag_p * mag_g).clamp_min(cfg.angvel_eps)
                        cos = torch.clamp(dot_full / norm_full, -1.0 + 1e-6, 1.0 - 1e-6)
                        ang_full[valid_full] = torch.acos(cos[valid_full])
                    ang_full_deg = ang_full * cfg.deg
                    if valid_full.any():
                        valid_f = valid_full.float()
                        ang_sum = (ang_full_deg * valid_f).sum(dim=(0, 2))
                        valid_cnt = valid_f.sum(dim=(0, 2)).clamp_min(1.0)
                        ang_curve = ang_sum / valid_cnt
                        ang_curve_max = ang_full_deg.max(dim=2).values.max(dim=0).values
                        _record_optional_diag_curve(
                            result,
                            metric_name='AngVelDirDeg',
                            curve=ang_curve,
                            curve_max=ang_curve_max,
                        )
                        result['AngVelDirDegValidRatio'] = float(valid_f.mean().item())
                    else:
                        _record_optional_diag_curve(
                            result,
                            metric_name='AngVelDirDeg',
                            curve=[],
                            curve_max=[],
                        )
                        result['AngVelDirDegSkipped'] = True
                    _record_diag_metric(result, cfg.diag_scope, 'AngVelDirDegEnd', float(ang_curve[-1].item()))
                    if cfg.diag_scope == 'single_step':
                        result['SingleStep/AngVelDirDegCurve'] = result['AngVelDirDegCurve']
                        result['SingleStep/AngVelDirDegCurveMax'] = result['AngVelDirDegCurveMax']
                    try:
                        summary = self._summarize_angvel_dir(state.w_pred, state.w_gt, bone_names=cfg.bone_names)
                    except Exception as _angvel_exc:
                        print(f"[Diag][WARN] _summarize_angvel_dir failed: {_angvel_exc}")
                        import traceback
                        traceback.print_exc()
                        summary = {}
                    if summary:
                        _record_diag_metric(result, cfg.diag_scope, 'AngVelDirDegRaw', summary.get('raw', float('nan')))
                        _record_diag_metric(
                            result,
                            cfg.diag_scope,
                            'AngVelDirDegWeighted',
                            summary.get('weighted', float('nan')),
                        )
                        _record_diag_metric(result, cfg.diag_scope, 'AngVelDirDegSmooth', summary.get('smooth', float('nan')))
                        _record_diag_metric(result, cfg.diag_scope, 'AngVelDirDegTorso', summary.get('torso', float('nan')))
                        _record_diag_metric(
                            result,
                            cfg.diag_scope,
                            'AngVelDirDegProximal',
                            summary.get('proximal', float('nan')),
                        )
                        _record_diag_metric(
                            result,
                            cfg.diag_scope,
                            'AngVelDirDegDistal',
                            summary.get('distal', float('nan')),
                        )

                    foot_names = ('foot_l', 'foot_r')
                    idx_map = {name: idx for idx, name in enumerate(cfg.bone_names)} if cfg.bone_names else {}

                    def _masked_mean(val: torch.Tensor, mask: torch.Tensor):
                        mask_f = mask.to(val.dtype)
                        w_sum = mask_f.sum()
                        if w_sum < 1e-6:
                            return None
                        return (val * mask_f).sum() / w_sum

                    contacts_mask = None
                    if torch.is_tensor(contacts_seq) and contacts_seq.dim() >= 3:
                        contacts_mask = contacts_seq[:, : state.w_pred.shape[1]] > 0.5

                    for fname in foot_names:
                        j_idx = idx_map.get(fname, None)
                        if j_idx is None or j_idx >= state.w_pred.shape[2]:
                            continue
                        foot_idx = 0 if fname.endswith('_l') else 1
                        w_p = state.w_pred[..., j_idx, :]
                        w_g = state.w_gt[..., j_idx, :]
                        mag_p = w_p.norm(dim=-1)
                        mag_g = w_g.norm(dim=-1)

                        stance_mask = swing_mask = None
                        if contacts_mask is not None and foot_idx < contacts_mask.shape[-1]:
                            stance_mask = contacts_mask[..., foot_idx]
                            swing_mask = ~stance_mask

                        if stance_mask is not None and stance_mask.any():
                            mae_contact = _masked_mean((w_p - w_g).abs().norm(dim=-1), stance_mask)
                            mag_contact = _masked_mean(mag_p, stance_mask)
                            if mae_contact is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/ContactAngVelMAE',
                                    float(mae_contact.item()),
                                )
                            if mag_contact is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/ContactAngVelMag',
                                    float(mag_contact.item()),
                                )

                        dot = (w_p * w_g).sum(dim=-1)
                        norm_prod = mag_p * mag_g
                        ang = torch.zeros_like(dot)
                        valid = norm_prod > 1e-6
                        ang[valid] = torch.acos(torch.clamp(dot[valid] / norm_prod[valid], -1.0, 1.0)) * cfg.deg

                        if stance_mask is not None and stance_mask.any():
                            ang_stance = _masked_mean(ang, stance_mask)
                            if ang_stance is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/AngVelDirDegStance',
                                    float(ang_stance.item()),
                                )
                        if swing_mask is not None and swing_mask.any():
                            ang_swing = _masked_mean(ang, swing_mask)
                            if ang_swing is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/AngVelDirDegSwing',
                                    float(ang_swing.item()),
                                )

                        if stance_mask is not None and stance_mask.any():
                            mag_mae = _masked_mean((mag_p - mag_g).abs(), stance_mask)
                            if mag_mae is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/AngVelMagMAEStance',
                                    float(mag_mae.item()),
                                )
                        if swing_mask is not None and swing_mask.any():
                            mag_mae_sw = _masked_mean((mag_p - mag_g).abs(), swing_mask)
                            if mag_mae_sw is not None:
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    f'Foot/{fname}/AngVelMagMAESwing',
                                    float(mag_mae_sw.item()),
                                )

                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    _phasec_warn_once(
                        "diag/angvel",
                        "failed to compute angvel diagnostics",
                        exc,
                    )

                period_pred = None
                if period_seq_pred:
                    try:
                        pp = period_seq_pred[0]
                        if isinstance(pp, torch.Tensor):
                            period_pred = torch.stack(
                                [p if p.dim() == 3 else p.unsqueeze(1) for p in period_seq_pred],
                                dim=1,
                            )
                            if period_pred.dim() == 4 and period_pred.size(2) == 1:
                                period_pred = period_pred.squeeze(2)
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        _phasec_warn_once(
                            "diag/period_pred_pack",
                            "failed to stack period predictions for diagnostics",
                            exc,
                        )
                        period_pred = None

                period_gt = None
                model = seqs.model
                if (
                    model is not None
                    and getattr(model, 'frozen_encoder', None) is not None
                    and getattr(model, 'frozen_period_head', None) is not None
                ):
                    try:
                        enc_in_list = []
                        for tensor in (contacts_seq, angvel_seq, pose_hist_seq):
                            if torch.is_tensor(tensor):
                                enc_in_list.append(tensor)
                        if enc_in_list:
                            enc_input = torch.cat([t for t in enc_in_list if t is not None], dim=-1)
                            enc_hidden = model.frozen_encoder(enc_input, return_summary=False)
                            if isinstance(enc_hidden, tuple):
                                enc_hidden = enc_hidden[-1]
                            period_gt = torch.tanh(model.frozen_period_head(enc_hidden))
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        _phasec_warn_once(
                            "diag/period_gt_probe",
                            "failed to compute period_gt embedding probe",
                            exc,
                        )
                        period_gt = None

                embed_l2 = embed_cos = None
                if period_pred is not None and period_gt is not None and period_pred.shape == period_gt.shape:
                    try:
                        diff = period_pred - period_gt
                        embed_l2 = diff.norm(dim=-1).mean()
                        _record_diag_metric(result, cfg.diag_scope, 'Period/EmbedL2', float(embed_l2.item()))
                        eps = 1e-6
                        cos = ((period_pred * period_gt).sum(dim=-1)) / (
                            period_pred.norm(dim=-1) * period_gt.norm(dim=-1) + eps
                        )
                        embed_cos = cos.clamp(-1.0, 1.0).mean()
                        _record_diag_metric(result, cfg.diag_scope, 'Period/EmbedCos', float(embed_cos.item()))
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        _phasec_warn_once(
                            "diag/period_embed",
                            "failed to compute period embedding diagnostics",
                            exc,
                        )
                _ = embed_cos

                try:
                    tgt = contacts_seq if torch.is_tensor(contacts_seq) else None
                    if tgt is not None and tgt.shape[-1] >= 2:
                        tgt = tgt[..., :2]
                        ref = period_pred if period_pred is not None else (period_gt if period_gt is not None else tgt)
                        tgt = tgt.to(ref.device).to(ref.dtype)
                        tgt = tgt * 2.0 - 1.0
                        if period_pred is not None and period_pred.shape[:2] == tgt.shape[:2] and period_pred.shape[-1] >= 2:
                            pred_hint = period_pred[..., :2]
                            _record_diag_metric(
                                result,
                                cfg.diag_scope,
                                'Period/ContactHintMAE',
                                float((pred_hint - tgt).abs().mean().item()),
                            )
                        if period_gt is not None and period_gt.shape[:2] == tgt.shape[:2] and period_gt.shape[-1] >= 2:
                            gt_hint = period_gt[..., :2]
                            _record_diag_metric(
                                result,
                                cfg.diag_scope,
                                'Period/ContactHintGTMAE',
                                float((gt_hint - tgt).abs().mean().item()),
                            )
                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    _phasec_warn_once(
                        "diag/period_contact_hint",
                        "failed to compute period/contact hint diagnostics",
                        exc,
                    )

                result['Period/PhaseSkipped'] = True
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                _phasec_warn_once(
                    "diag/phase_block",
                    "period/contact diagnostic phase block failed",
                    exc,
                )

    return state


def _compute_keybone_metrics(
    self,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    state: FreeRunDiagKinematics,
) -> None:
    if not cfg.bone_names:
        return

    if state.w_pred is not None and state.w_gt is not None:
        for j_idx, name in enumerate(cfg.bone_names):
            if j_idx >= state.w_pred.shape[2]:
                continue
            w_pred_b = state.w_pred[..., j_idx, :]
            w_gt_b = state.w_gt[..., j_idx, :]
            mag_mae = float((w_pred_b.norm(dim=-1) - w_gt_b.norm(dim=-1)).abs().mean().item())
            ang_mae = float((w_pred_b - w_gt_b).abs().mean().item())
            result[f'Bone/{name}/AngVelMagMAE'] = mag_mae
            result[f'Bone/{name}/AngVelMAE'] = ang_mae

    key_bone_names = getattr(self, 'eval_key_bones', None)
    if not key_bone_names:
        key_bone_names = [
            'pelvis',
            'upperarm_l', 'lowerarm_l', 'hand_l',
            'upperarm_r', 'lowerarm_r', 'hand_r',
            'thigh_l', 'calf_l', 'foot_l',
            'thigh_r', 'calf_r', 'foot_r',
        ]

    idx_map = {name: idx for idx, name in enumerate(cfg.bone_names)}
    key_indices = [idx_map[name] for name in key_bone_names if name in idx_map]
    key_geo_vals: list[float] = []
    key_geo_local_vals: list[float] = []
    key_ang_mae_vals: list[float] = []
    key_ang_mag_mae_vals: list[float] = []
    key_ang_mag_rel_vals: list[float] = []
    key_ang_dir_vals: list[float] = []
    keybone_details: Dict[str, Dict[str, float]] = {}

    geo_local_tensor = state.geo_local if torch.is_tensor(state.geo_local) else None
    if geo_local_tensor is None:
        raise RuntimeError(
            "GeoLocalDeg metrics unavailable; ensure FK + geodesic computation succeeded before KeyBone diagnostics."
        )

    for name in key_bone_names:
        if name not in idx_map:
            continue
        j_idx = idx_map[name]
        prefix = f'KeyBone/{name}'
        if state.geo is not None and state.geo.shape[-1] > j_idx:
            geo_val = float((state.geo[..., j_idx].mean() * cfg.deg).item())
        else:
            geo_val = float('nan')
        result[f'{prefix}/GeoDeg'] = geo_val
        if _math.isfinite(geo_val):
            key_geo_vals.append(geo_val)

        geo_local_val = float('nan')
        if geo_local_tensor.shape[-1] > j_idx:
            geo_local_val = float(geo_local_tensor[..., j_idx].mean().item())
        result[f'{prefix}/GeoLocalDeg'] = geo_local_val
        if _math.isfinite(geo_local_val):
            key_geo_local_vals.append(geo_local_val)

        if state.w_pred is not None and state.w_gt is not None and state.w_pred.shape[2] > j_idx:
            w_pred_b = state.w_pred[..., j_idx, :]
            w_gt_b = state.w_gt[..., j_idx, :]
            ang_mae = float((w_pred_b - w_gt_b).abs().mean().item())
            result[f'{prefix}/AngVelMAE'] = ang_mae
            if _math.isfinite(ang_mae):
                key_ang_mae_vals.append(ang_mae)

            mag_p = w_pred_b.norm(dim=-1)
            mag_g = w_gt_b.norm(dim=-1)
            mag_avg = 0.5 * (mag_p + mag_g)
            mag_rel = (mag_p - mag_g).abs() / (mag_avg + cfg.mag_rel_beta)
            mag_mae = float((mag_p - mag_g).abs().mean().item())
            result[f'{prefix}/AngVelMagMAE'] = mag_mae
            if _math.isfinite(mag_mae):
                key_ang_mag_mae_vals.append(mag_mae)

            valid_mag = mag_avg > cfg.mag_rel_threshold
            mag_rel_val = float(torch.median(mag_rel[valid_mag]).item()) if valid_mag.any() else float('nan')
            result[f'{prefix}/AngVelMagRel'] = mag_rel_val
            if _math.isfinite(mag_rel_val):
                key_ang_mag_rel_vals.append(mag_rel_val)

            dir_val = geo_local_val
            if not _math.isfinite(dir_val):
                raise RuntimeError(
                    f"GeoLocalDeg for key bone '{name}' is NaN; ensure FK skeleton matches outputs."
                )
            result[f'{prefix}/AngVelDirDeg'] = dir_val
            key_ang_dir_vals.append(dir_val)
            keybone_details[name] = {
                'GeoDeg': geo_val,
                'GeoLocalDeg': geo_local_val,
                'AngVelMAE': ang_mae,
                'AngVelMagMAE': mag_mae,
                'AngVelMagRel': mag_rel_val,
                'AngVelDirDeg': dir_val,
            }
        else:
            result[f'{prefix}/AngVelMAE'] = float('nan')
            result[f'{prefix}/AngVelMagMAE'] = float('nan')
            result[f'{prefix}/AngVelMagRel'] = float('nan')
            dir_val = geo_local_val
            if not _math.isfinite(dir_val):
                raise RuntimeError(
                    f"GeoLocalDeg for key bone '{name}' is NaN; ensure FK skeleton matches outputs."
                )
            result[f'{prefix}/AngVelDirDeg'] = dir_val
            key_ang_dir_vals.append(dir_val)
            keybone_details[name] = {
                'GeoDeg': geo_val,
                'GeoLocalDeg': geo_local_val,
                'AngVelMAE': float('nan'),
                'AngVelMagMAE': float('nan'),
                'AngVelMagRel': float('nan'),
                'AngVelDirDeg': dir_val,
            }

    summary = {}
    try:
        geo_group_means = {}
        name_to_idx_full = {name: idx for idx, name in enumerate(cfg.bone_names[:geo_local_tensor.shape[-1]])}

        def _group_mean_from_names(names_group: Sequence[str]) -> Optional[float]:
            idxs = [name_to_idx_full[name] for name in names_group if name in name_to_idx_full]
            if not idxs:
                return None
            group_tensor = geo_local_tensor[..., idxs]
            return float(group_tensor.mean().item())

        leg_mean = _group_mean_from_names(DEFAULT_DIRECT_POSE_LEG_BONES)
        arm_mean = _group_mean_from_names(STAGE6_3WAY_ARMCHAIN_BONES)
        trunk_names = [
            name
            for name in cfg.bone_names[:geo_local_tensor.shape[-1]]
            if name not in set(DEFAULT_DIRECT_POSE_LEG_BONES)
            and name not in set(STAGE6_3WAY_ARMCHAIN_BONES)
            and name_to_idx_full.get(name, -1) != cfg.root_idx
        ]
        trunk_mean = _group_mean_from_names(trunk_names)
        if leg_mean is not None:
            geo_group_means['leg'] = leg_mean
        if arm_mean is not None:
            geo_group_means['arm'] = arm_mean
        if trunk_mean is not None:
            geo_group_means['trunk'] = trunk_mean
        if geo_group_means:
            summary['group_mean'] = geo_group_means
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _phasec_warn_once(
            "diag/keybone_group_mean",
            "failed to build KeyBone group_mean summary",
            exc,
        )

    if key_geo_vals:
        summary['GeoDegMean'] = float(sum(key_geo_vals) / len(key_geo_vals))
        _record_diag_metric(result, cfg.diag_scope, 'KeyBone/GeoDegMean', summary['GeoDegMean'])
    if key_ang_mae_vals:
        summary['AngVelMAE'] = float(sum(key_ang_mae_vals) / len(key_ang_mae_vals))
        _record_diag_metric(result, cfg.diag_scope, 'KeyBone/AngVelMAE', summary['AngVelMAE'])
    if key_ang_mag_mae_vals:
        summary['AngVelMagMAE'] = float(sum(key_ang_mag_mae_vals) / len(key_ang_mag_mae_vals))
        _record_diag_metric(result, cfg.diag_scope, 'KeyBone/AngVelMagMAE', summary['AngVelMagMAE'])
    if key_ang_mag_rel_vals:
        summary['AngVelMagRel'] = float(sum(key_ang_mag_rel_vals) / len(key_ang_mag_rel_vals))
        _record_diag_metric(result, cfg.diag_scope, 'KeyBone/AngVelMagRel', summary['AngVelMagRel'])
    if not key_geo_local_vals:
        raise RuntimeError("KeyBone GeoLocalDegMean is empty; diagnostics require valid limb geodesic values.")
    summary['GeoLocalDegMean'] = float(sum(key_geo_local_vals) / len(key_geo_local_vals))
    _record_diag_metric(result, cfg.diag_scope, 'KeyBone/GeoLocalDegMean', summary['GeoLocalDegMean'])
    if key_ang_dir_vals:
        summary['AngVelDirDeg'] = float(sum(key_ang_dir_vals) / len(key_ang_dir_vals))
        _record_diag_metric(result, cfg.diag_scope, 'KeyBone/AngVelDirDeg', summary['AngVelDirDeg'])
    if key_indices:
        kb_curve = geo_local_tensor[:, :, key_indices].mean(dim=(0, 2))
        result['KeyBone/AngVelDirDegCurve'] = kb_curve.detach().cpu().tolist()
    if keybone_details:
        _record_diag_metric(result, cfg.diag_scope, 'KeyBoneDetails', keybone_details)
    if summary:
        _record_diag_metric(result, cfg.diag_scope, 'KeyBoneSummary', summary)


def _diagnose_free_run_impl(
    self,
    batch,
    predY,
    gtY,
    predsX,
    period_seq_pred,
    motion_seq,
    y_seq,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
    angvel_raw_seq=None,
):
    self._require_normalizer("_diagnose_free_run_impl")
    _ = y_seq, angvel_raw_seq

    bone_names_src = getattr(self, '_bone_names', None)
    if not bone_names_src:
        bundle_meta = getattr(self, '_bundle_meta', None)
        if isinstance(bundle_meta, dict):
            bone_names_src = bundle_meta.get('bone_names') or bundle_meta.get('skeleton', {}).get('bone_names')
    bone_names = [str(b) for b in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []

    result: Dict[str, Any] = {}
    cfg = SimpleNamespace(
        diag_scope=str(getattr(self, '_diag_scope', 'free_run')),
        rot6d_y=self._sl_from_layout(getattr(self, '_y_layout', {}), 'BoneRotations6D'),
        rv_x=self._sl_from_layout(getattr(self, '_x_layout', {}), 'RootVelocity'),
        rot6d_x=self._sl_from_layout(getattr(self, '_x_layout', {}), 'BoneRotations6D'),
        eval_align_root=bool(getattr(self, 'eval_align_root0', True)),
        root_idx=int(getattr(self, 'eval_root_idx', 0)),
        up_axis=int(getattr(self, 'eval_up_axis', getattr(self, '_up_axis', 2))),
        fps_eval=float(getattr(self, 'bone_hz', 60.0)),
        contact_threshold=float(getattr(self, 'foot_contact_threshold', 1.5)),
        diag_input_stats=bool(getattr(self, 'diag_input_stats', False)),
        yaw_forward_axis_offset=float(getattr(self, 'yaw_forward_axis_offset', 0.0) or 0.0),
        mag_rel_beta=float(getattr(self, 'eval_angvel_beta', 0.25) or 0.25),
        mag_rel_threshold=float(getattr(self, 'eval_angvel_mag_threshold', 0.10) or 0.10),
        angvel_eps=float(getattr(self, 'angvel_eps', 1e-6) or 1e-6),
        angvel_dir_threshold=float(getattr(self, 'angvel_dir_threshold', 0.1) or 0.1),
        deg=180.0 / _math.pi,
        bone_names=bone_names,
        angvel_slice=getattr(self, 'angvel_x_slice', None),
    )
    seqs = _collect_diag_sequences(self, predsX=predsX, motion_seq=motion_seq, batch=batch)
    _compute_input_drift_metrics(
        self,
        result,
        cfg,
        seqs,
        period_seq_pred=period_seq_pred,
    )
    state = _compute_contact_and_angvel_metrics(
        self,
        result,
        cfg,
        seqs,
        predY=predY,
        gtY=gtY,
        period_seq_pred=period_seq_pred,
        contacts_seq=contacts_seq,
        angvel_seq=angvel_seq,
        pose_hist_seq=pose_hist_seq,
    )
    _compute_keybone_metrics(self, result, cfg, state)

    if state.w_gt is not None and seqs.gtX_raw_full is not None and isinstance(cfg.angvel_slice, slice):
        try:
            angvel_data = seqs.gtX_raw_full[:, :state.w_gt.shape[1] + 1, cfg.angvel_slice]
            J_ang = (cfg.angvel_slice.stop - cfg.angvel_slice.start) // 3
            if J_ang == state.w_gt.shape[2]:
                angvel_data = angvel_data[:, 1:state.w_gt.shape[1] + 1].reshape(
                    state.w_gt.shape[0], state.w_gt.shape[1], J_ang, 3
                )
                diff_gt = (state.w_gt - angvel_data).abs()
                result['AngVelGTReconMAE'] = float(diff_gt.mean().item())
                dot_gt = (state.w_gt * angvel_data).sum(dim=-1)
                norm_gt = state.w_gt.norm(dim=-1) * angvel_data.norm(dim=-1)
                mask_gt = norm_gt > 1e-6
                if mask_gt.any():
                    ang_dir = torch.acos(torch.clamp(dot_gt[mask_gt] / norm_gt[mask_gt], -1.0, 1.0)) * cfg.deg
                    result['AngVelGTReconDirDeg'] = float(ang_dir.mean().item())
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _phasec_warn_once(
                "diag/angvel_gt_recon",
                "failed to compute AngVelGTRecon diagnostics",
                exc,
            )

    if state.w_pred is not None:
        contact_pred = (state.w_pred.norm(dim=-1) < cfg.contact_threshold).float()
        result['FootContact'] = float(contact_pred.mean().item())
    return result

TRAIN_ENTRY_CONFIG_META_KEYS = {'dataset_profile', 'strategy_meta'}


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    norm_template_path: Optional[str]
    bundle_json_path: Optional[str]
    out_dir: str
    direct_pose_grad_monitor_enable: bool
    direct_pose_grad_ratio_gate: float
    full_config: Dict[str, Any]
    pose_hist_len: int
    pose_hist_dim: int
    pose_hist_scales: Optional[torch.Tensor]
    pose_hist_mu: Optional[torch.Tensor]
    pose_hist_std: Optional[torch.Tensor]
    foot_contact_threshold: float
    bone_hz: float
    fps: float
    trainbase_contacts_source: str
    trainbase_contacts_pretrain_clamp: float
    trainbase_contacts_pretrain_affine_stats: Optional[str]
    trainbase_contacts_pretrain_affine: Optional[Dict[str, Any]]
    contact_meas_gate_by_hit_override: Optional[bool]
    contact_meas_ground_z_mode: str
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_max_up_m: float
    contact_meas_ground_z_max_down_m: float
    diag_topk: int
    diag_thr: float
    yaw_forward_axis: int
    yaw_forward_axis_offset: float
    eval_angvel_dir_percentile: float
    diag_input_stats: bool
    val_mode: str
    no_monitor: bool
    monitor_batches: int
    teacher_eval_max_batches: Optional[int]
    force_valfree_eval: bool
    eval_settings: FreeRunSettings
    ss_chunk_len: int
    tf_mode: str
    tf_warmup_epochs: int
    tf_start_epoch: int
    tf_end_epoch: int
    tf_max: float
    tf_min: float
    history_debug_steps: int
    history_dropout_prob: float
    history_dropout_prob_min: float
    history_dropout_prob_max: float
    freerun_stage_schedule: list[Any]
    adaptive_loss_module: Any
    hyperparam_scheduler: Any
    freerun_debug_path: Optional[str]
    enable_grad_connection_test: bool
    current_run_name: str


def _load_train_entry_config_defaults(config_path: Optional[str], parser: argparse.ArgumentParser) -> Dict[str, Any]:
    if not config_path:
        return {}
    cfg_path = os.path.expanduser(config_path)
    if not os.path.isfile(cfg_path):
        parser.error(f"[config_json] 文件不存在: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        parser.error(f"[config_json] 根对象必须是 JSON dict，当前类型 {type(payload).__name__}")
    valid_dests = {action.dest for action in parser._actions if action.dest and action.dest != 'help'}
    unknown_keys = sorted(k for k in payload.keys() if k not in valid_dests and k not in TRAIN_ENTRY_CONFIG_META_KEYS)
    legacy_unknown = [k for k in unknown_keys if k in LEGACY_LOSS_TOPLEVEL_KEYS]
    if legacy_unknown:
        parser.error(_legacy_loss_keys_msg(legacy_unknown, context='config_json(top-level)'))
    removed_phase_reset = [k for k in unknown_keys if k in REMOVED_TRAINBASE_PHASE_RESET_KEYS]
    if removed_phase_reset:
        parser.error(_removed_trainbase_phase_reset_msg(removed_phase_reset, context='config_json(top-level)'))
    if unknown_keys:
        parser.error(f"[config_json] 存在未识别字段: {', '.join(unknown_keys)}")
    try:
        _assert_no_legacy_loss_keys_in_schedule(
            payload.get('freerun_stage_schedule'),
            context='config_json.freerun_stage_schedule',
        )
        _assert_no_removed_trainbase_stage_keys(
            payload.get('freerun_stage_schedule'),
            context='config_json.freerun_stage_schedule',
        )
    except ValueError as err:
        parser.error(str(err))
    print(f"[config_json] Loaded defaults from {cfg_path} ({len(payload)} keys)")
    return dict(payload)


def _apply_train_entry_config_overrides(
    namespace: argparse.Namespace,
    overrides: Optional[Sequence[str]],
    parser: argparse.ArgumentParser,
) -> None:
    if not overrides:
        return

    def _parse_literal(raw: str):
        txt = raw.strip()
        if not txt:
            return txt
        try:
            return ast.literal_eval(txt)
        except Exception:
            lowered = txt.lower()
            if lowered == 'none':
                return None
            return txt

    applied: Dict[str, Any] = {}
    for entry in overrides:
        if not entry:
            continue
        if '=' not in entry:
            parser.error(f"[config_override] 期望 KEY=VALUE，实际收到: {entry}")
        key, value_expr = entry.split('=', 1)
        key = key.strip()
        if not key:
            parser.error('[config_override] 键名不能为空')
        if key in REMOVED_TRAINBASE_PHASE_RESET_KEYS:
            parser.error(_removed_trainbase_phase_reset_msg([key], context='config_override'))
        if not hasattr(namespace, key):
            parser.error(f"[config_override] 未知键名: {key}")
        new_value = _parse_literal(value_expr)
        setattr(namespace, key, new_value)
        applied[key] = new_value
    if applied:
        formatted = ', '.join(f"{k}={applied[k]}" for k in sorted(applied))
        print(f"[config_override] Applied: {formatted}")


def _build_train_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        '--config_json',
        type=str,
        default=None,
        help='JSON 配置文件路径。键名需与 CLI 参数一致，并作为默认值参与解析。',
    )


    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[config_parser])
    p.add_argument('--val_mode', type=str, default='online', choices=['online','none'])
    p.add_argument(
        '--encoder_path',
        type=str,
        default='models/motion_encoder_equiv.pt',
        help='预训练 MotionEncoder bundle 路径（.pt，比如第二阶段导出的 motion_encoder_equiv.pt）',
    )
    p.add_argument('--norm_template', type=str, default='raw_data/processed_data/norm_template.json', help='数据归一化模板路径')
    p.add_argument('--pretrain_template', type=str, default='models/pretrain_template.json', help='预训练编码器模板（含角速度统计）')
    p.add_argument('--no_monitor', action='store_true', default=False)
    p.add_argument('--data', type=str, required=True, help='数据目录（含 *.npz）')
    p.add_argument('--out', type=str, default='./runs', help='输出目录根路径')
    p.add_argument('--run_name', type=str, default=None, help='子目录名；未给则用时间戳')
    p.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从 checkpoint(.pth) 初始化模型权重（仅加载 model state_dict；会自动跳过 shape 不匹配的权重）。',
    )
    p.add_argument('--config_override', action='append', default=None, metavar='KEY=VALUE',
                   help='在解析后覆写配置值，可重复，例如 --config_override lr=5e-5')
    p.add_argument('--train_files', type=str, default='', help='逗号分隔的路径/通配/或 @list.txt')
    p.add_argument('--diag_topk', type=int, default=8, help='free-run 评估时打印 X_norm 的 |z| Top-K')
    p.add_argument('--diag_thr', type=float, default=8.0, help='|z| 阈值，统计 X_norm 爆炸比例')
    p.add_argument("--bundle_json", type=str, default=None, help='UE 导出的运行时 bundle（可含 MuY/StdY、feature_layout、MuC_other/StdC_other 等）', required=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--min_lr_ratio', type=float, default=0.05)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--tf_mode', type=str, default='epoch_linear', choices=['global', 'epoch_linear'])
    p.add_argument('--tf_warmup_epochs', type=int, default=3)
    p.add_argument('--tf_start_epoch', type=int, default=0)
    p.add_argument('--tf_end_epoch', type=int, default=10)
    p.add_argument('--tf_max', type=float, default=1.0)
    p.add_argument('--tf_min', type=float, default=0.1)
    p.add_argument('--history_debug_steps', type=int, default=0,
                   help='>1 时，在训练批次中额外运行 train_free rollout 诊断历史漂移步数')
    p.add_argument('--history_adaptive_max_frames', type=int, default=None,
                   help='训练期允许的最大历史帧数（默认使用 norm_template 中的 pose_hist_len）')
    p.add_argument('--history_adaptive_hidden', type=int, default=256,
                   help='adaptive history 内部隐藏维度')
    p.add_argument('--history_adaptive_heads', type=int, default=2,
                   help='adaptive history 注意力头数')
    p.add_argument('--history_adaptive_train_variable', action='store_true',
                   help='训练时随机截断历史长度，提升部署鲁棒性')
    p.add_argument('--history_dropout_prob', type=float, default=0.10,
                   help='训练期以该概率完全屏蔽历史特征，迫使模型依赖未来条件信号进行纠错。')
    p.add_argument('--history_use_trend_features', action='store_true',
                   help='在 adaptive history 中显式注入历史 drift/趋势特征。')
    p.add_argument('--freerun_stage_schedule', type=str, default=None,
                   help='分阶段调度（TF/LR/history/direct-pose trainability 等）的 JSON/字符串配置。')
    p.add_argument('--ss_chunk_len', type=int, default=1,
                   help='scheduled sampling 的 chunk 长度（>1 启用 sticky 采样：每 chunk 采一次 use_gt）。')
    p.add_argument('--tf_warmup_steps', type=int, default=5000)
    p.add_argument('--tf_total_steps', type=int, default=200000)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--context_len', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--amp', action='store_true', help='启用自动混合精度 (torch.autocast)')
    p.add_argument('--w_rot_ortho', type=float, default=0.001)
    p.add_argument('--w_rot_local', type=float, default=0.0,
                   help='父子关节局部 geodesic 约束权重（0=关闭）。')
    p.add_argument('--w_root_vel', type=float, default=0.0,
                   help='根速度向量 MSE 损失权重（输出包含 RootVelocity 时生效）。')
    p.add_argument('--w_root_speed', type=float, default=0.0,
                   help='根速度模长 MAE 损失权重（输出包含 RootVelocity 时生效）。')
    # ---- Contact plan anchor (independent) ----
    p.add_argument('--contact_plan_enable', action='store_true', default=False,
                   help='启用 cond-only GRU contacts_plan（作为独立锚点）')
    p.add_argument(
        '--trainbase_contacts_source',
        type=str,
        default='auto',
        choices=('auto', 'whitebox', 'pretrain_contact'),
        help='Basetrain rollout contacts source：auto 优先接 frozen pretrain_contact，缺失时回退 whitebox。',
    )
    p.add_argument(
        '--trainbase_contacts_pretrain_clamp',
        type=float,
        default=1.0,
        help='当 basetrain rollout 使用 pretrain_contact 时，对 frozen encoder 输入做 [-k,+k] clamp。',
    )
    p.add_argument(
        '--trainbase_contacts_pretrain_affine_stats',
        type=str,
        default=None,
        help='可选的 pretrain_contact affine stats JSON / JSON-string；格式与 posttrain affine_stats.json 一致。',
    )
    p.add_argument('--contact_plan_hidden', type=int, default=64,
                   help='contacts_plan GRU hidden dim')
    p.add_argument('--contact_plan_dropout', type=float, default=0.0,
                   help='contacts_plan head dropout')
    p.add_argument('--w_contact_plan', type=float, default=0.0,
                   help='contacts_plan 监督权重（MSE vs GT soft_contacts）')
    p.add_argument('--contact_plan_inject', type=str, default='none', choices=['none', 'contacts', 'plan_z'],
                   help="Phase2: 将 contacts_plan / plan_z 前馈注入主干输入（none=关闭）")
    p.add_argument('--contact_plan_inject_detach', type=lambda x: str(x).lower() in ('1', 'true', 'yes'), default=True,
                   help="注入主干时对 plan 特征 stop-grad（保持 plan 语义为独立锚点；推荐开启）")
    p.add_argument('--contact_plan_time_pe_dim', type=int, default=0,
                   help='contacts_plan time positional encoding dim（0=关闭；推荐 8/16）')
    p.add_argument('--contact_plan_time_pe_base', type=float, default=10000.0,
                   help='contacts_plan time-PE 频率基数（默认 10000）')
    p.add_argument(
        '--contact_plan_init_mode',
        type=str,
        default='learnable',
        choices=['zeros', 'learnable', 'obs', 'learnable+obs'],
        help='contact plan 冷启动 init：learnable(默认)|obs|learnable+obs(推荐)|zeros',
    )
    p.add_argument('--contact_plan_init_hidden', type=int, default=128,
                   help='contact plan init MLP hidden dim（仅 init_mode=obs/learnable+obs 生效）')
    p.add_argument('--contact_plan_init_dropout', type=float, default=0.0,
                   help='contact plan init MLP dropout（仅 init_mode=obs/learnable+obs 生效）')
    # ---- Contact phase state (prev_phase_vec clock; step-stateful like plan_z) ----
    p.add_argument(
        '--contact_phase_state_enable',
        action='store_true',
        default=False,
        help='启用显式相位状态 prev_phase_vec（phase_z）并作为 contact_plan GRU 输入的一部分；见 docs/contact_phase_state_prevphase_tta.md',
    )
    p.add_argument(
        '--contact_phase_state_init_mode',
        type=str,
        default='obs',
        choices=['zeros', 'learnable', 'obs', 'learnable+obs'],
        help='phase_z 冷启动 init：obs(默认)|learnable+obs|learnable|zeros',
    )
    p.add_argument('--contact_phase_state_hidden', type=int, default=64,
                   help='phase Δφ head hidden dim')
    p.add_argument('--contact_phase_state_delta_max', type=float, default=0.5,
                   help='每步相位推进 Δφ 的最大幅度（rad/step，tanh 缩放）')
    p.add_argument('--contact_phase_state_delta_init', type=float, default=(6.283185307179586 / 80.0),
                   help='Δφ 初始 bias（rad/step；默认约等于 80 帧一周期）')
    # ---- Event-Clock v3 (contact_plan residual correction) ----
    p.add_argument('--use_event_clock', action='store_true', default=False,
                   help='启用 Event-Clock v3：在 contact_plan GRU loop 内做 gated residual correction')
    p.add_argument('--event_clock_max_delta', type=float, default=0.5,
                   help='Event-Clock Δz clip 幅度（0=不 clip）')
    p.add_argument('--event_clock_hidden_dim', type=int, default=64,
                   help='Event-Clock Δz head hidden dim')
    p.add_argument('--event_clock_gate_hidden_dim', type=int, default=32,
                   help='Event-Clock gate head hidden dim')
    p.add_argument('--event_clock_lambda_entropy_weight', type=float, default=0.01,
                   help='Event-Clock λ entropy 正则权重')
    p.add_argument('--event_clock_lambda_prior_weight', type=float, default=0.01,
                   help='Event-Clock λ dynamic prior 正则权重')
    p.add_argument('--event_clock_delta_z_l2_weight', type=float, default=0.001,
                   help='Event-Clock Δz L2 正则权重')
    # ---- Direct pose head (cond + contacts_plan -> absolute pose) ----
    p.add_argument('--direct_pose_enable', action='store_true', default=False,
                   help='启用 direct pose head（cond+contacts_plan -> out_direct，不走自回归）')
    p.add_argument('--direct_pose_hidden', type=int, default=256,
                   help='direct pose head hidden dim')
    p.add_argument('--direct_pose_dropout', type=float, default=0.0,
                   help='direct pose head dropout')
    p.add_argument('--direct_pose_detach_plan', type=lambda x: str(x).lower() in ('1', 'true', 'yes'), default=True,
                   help='direct head 输入 contacts_plan 时 stop-grad（推荐开启）')
    p.add_argument(
        '--direct_pose_meas_mode',
        type=str,
        default='concat',
        choices=['concat', 'mode_select'],
        help='Phase bridge: direct head 是否引入 contacts_meas（concat=D0; mode_select=D1）',
    )
    p.add_argument('--direct_pose_meas_drop_prob', type=float, default=0.0,
                   help='D2: 训练时对 direct 输入的 contacts_meas 执行整向量 drop(置0) 概率')
    p.add_argument('--direct_pose_meas_noise_std', type=float, default=0.0,
                   help='D2: 训练时对 direct 输入的 contacts_meas 加高斯噪声 std（随后 clamp 到[0,1]）')
    p.add_argument('--direct_pose_plan_drop_prob', type=float, default=0.0,
                   help='D2: 训练时对 direct 输入的 contacts_plan 执行整向量 drop(置0) 概率（防止 plan 成为 shortcut）')
    p.add_argument('--direct_pose_split_enable', action='store_true', default=False,
                   help='启用 direct_pose 输出分头（shared trunk + leg/non-leg split output heads）')
    p.add_argument('--direct_pose_arm_split_enable', action='store_true', default=False,
                   help='启用 direct_pose 非腿分支的 arm/else 再分头（3-way: leg/arm/else）')
    p.add_argument('--direct_pose_arm_bones', type=str, default=None,
                   help='arm 分支骨骼 CSV；未提供且启用 arm split 时默认使用 Stage6 3-way armchain 口径')
    p.add_argument('--direct_pose_nonleg_proj_dim', type=int, default=0,
                   help='non-leg/arm/else 分支投影维度；0=直接从 trunk readout')
    p.add_argument('--w_direct_pose', type=float, default=0.0,
                   help='direct pose 监督权重（geodesic vs GT pose；0=关闭）')
    p.add_argument('--direct_pose_loss_leg_split', action='store_true', default=False,
                   help='direct loss 按 leg/non-leg 拆分计算 base objective（Stage6 parity）')
    p.add_argument('--direct_pose_loss_arm_else_balance_enable', action='store_true', default=False,
                   help='启用 arm/else 组均衡 non-leg objective（按 group mean 重构 non-leg base）')
    p.add_argument('--direct_pose_loss_arm_weight', type=float, default=1.0,
                   help='arm/else rebalance 中 arm group 权重')
    p.add_argument('--direct_pose_loss_else_weight', type=float, default=1.0,
                   help='arm/else rebalance 中 else group 权重')
    p.add_argument('--direct_pose_grad_monitor_enable', action='store_true', default=False,
                   help='记录 direct trunk/leg/arm/else 输出头梯度范数与比值')
    p.add_argument('--direct_pose_grad_ratio_gate', type=float, default=0.35,
                   help='direct grad ratio 诊断阈值（nonleg/leg；仅日志告警）')
    p.add_argument('--direct_pose_loss_group_norm_enable', action='store_true', default=False,
                   help='启用 direct leg/non-leg group norm objective（Stage6 parity）')
    p.add_argument('--direct_pose_loss_group_norm_w_leg', type=float, default=1.0,
                   help='group norm objective 中 leg ratio 权重')
    p.add_argument('--direct_pose_loss_group_norm_w_nonleg', type=float, default=1.0,
                   help='group norm objective 中 non-leg ratio 权重')
    p.add_argument('--direct_pose_loss_group_norm_ema_beta', type=float, default=0.9,
                   help='group norm EMA beta')
    p.add_argument('--direct_pose_loss_group_norm_ratio_min', type=float, default=0.2,
                   help='group norm ratio clamp 最小值')
    p.add_argument('--direct_pose_loss_group_norm_ratio_max', type=float, default=5.0,
                   help='group norm ratio clamp 最大值')
    p.add_argument('--direct_pose_loss_group_norm_eps', type=float, default=1e-6,
                   help='group norm 数值稳定 epsilon')
    # ---- White-box contacts_meas knobs (P2 ground_z stability / ablations) ----
    # NOTE: when contact_plan_enable=True, training/rollout will resolve contacts_in_t from
    # trainbase_contacts_source=auto|whitebox|pretrain_contact; whitebox knobs below remain active
    # for fallback / ablation when the resolved source is whitebox.
    p.add_argument(
        '--contact_meas_gate_by_hit',
        type=str,
        default='auto',
        choices=('auto', 'true', 'false'),
        help="Override white-box gate_by_hit: auto uses bundle/meta; false disables discrete sweep hit gate (ablation).",
    )
    p.add_argument(
        '--contact_meas_ground_z_mode',
        type=str,
        default='window',
        choices=('ema', 'window', 'slew'),
        help="White-box ground_z update mode: ema | window(quantile) | slew(rate-limit).",
    )
    p.add_argument('--contact_meas_ground_z_beta', type=float, default=0.05,
                   help='EMA beta for contact_meas_ground_z_mode=ema.')
    p.add_argument('--contact_meas_ground_z_window', type=int, default=5,
                   help='Window length for contact_meas_ground_z_mode=window.')
    p.add_argument('--contact_meas_ground_z_quantile', type=float, default=0.2,
                   help='Low-quantile (0..1) for contact_meas_ground_z_mode=window.')
    p.add_argument('--contact_meas_ground_z_slew_up_cm', type=float, default=0.0,
                   help='Max upward change (cm/step) applied to ground_z after the chosen mode (0 disables).')
    p.add_argument('--contact_meas_ground_z_slew_down_cm', type=float, default=0.0,
                   help='Max downward change (cm/step) applied to ground_z after the chosen mode (0 disables).')
    # ---- (Reserved) post-train corrector knobs live in dedicated scripts ----
    p.add_argument('--adaptive_bone_weights', type=lambda x: str(x).lower() in ('1', 'true', 'yes'), default=True,
                   help='是否根据骨骼运动幅度自适应权重（默认开启）。')
    # unified bone weight (new)
    p.add_argument('--unified_downstream_power', type=float, default=0.6,
                   help='下游影响指数压缩 power (0.5~0.7 recommended)')
    p.add_argument('--unified_self_scale', type=float, default=1.5,
                   help='自身长度放大系数')
    p.add_argument('--unified_min_weight', type=float, default=0.05,
                   help='权重保底（相对均值）')
    p.add_argument('--rot_local_tail_weight', type=float, default=0.0,
                   help='rot_local 额外 tail loss 权重（CVaR/top-k，越大越压最差骨骼）。0=关闭。')
    p.add_argument('--rot_local_tail_k', type=int, default=0,
                   help='rot_local tail loss 的 top-k 骨骼数量（例如 13 骨骼取 3）。0=关闭。')
    p.add_argument('--rot_local_tail_scope', type=str, default='all', choices=['all', 'limbs', 'keybones'],
                   help="tail loss 选择范围：all=全骨骼；limbs=limb_monitor_names（若缺失则用skeleton leaves回退）；keybones=pelvis+limb_monitor_names（并用leaves补全）。")
    p.add_argument('--rot_local_tail_select', type=str, default='batch', choices=['batch', 'ema'],
                   help='tail loss top-k 选择打分：batch=当前batch均值；ema=跨batch EMA（更平滑、减少whack-a-mole）。')
    p.add_argument('--rot_local_tail_ema_beta', type=float, default=0.9,
                   help='rot_local_tail_select=ema 时的 EMA beta（越大越平滑）。')
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--yaw_aug_deg', type=float, default=0.0)
    p.add_argument('--normalize_c', action='store_true')
    p.add_argument('--aug_noise_std', type=float, default=0.0)
    p.add_argument('--aug_time_warp_prob', type=float, default=0.0)
    # TensorBoard 相关逻辑已移除，避免冗余参数
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--foot_contact_threshold', type=float, default=1.5, help='角速度阈值（rad/s），低于该值视为脚接触')
    p.add_argument('--monitor_batches', type=int, default=2, help='每个 epoch 在线指标采样的批次数')
    p.add_argument('--force_valfree_eval', action='store_true', default=False,
                   help='即使当前为纯 teacher 阶段，也强制执行一次 freerun 验证并写出 valfree 指标')
    p.add_argument('--teacher_eval_max_batches', type=int, default=None,
                   help='Teacher 评估最多跑多少个 batch；<=0 则跳过评估，用训练均值loss代填')
    p.add_argument('--eval_horizon', type=int, default=None,
                   help='在线 freerun 验证时的 horizon（帧数）；未指定则遍历整段序列')
    p.add_argument('--eval_warmup', type=int, default=0,
                   help='在线 freerun 验证前的 teacher forcing 帧数（warmup steps）')
    p.add_argument('--yaw_forward_axis', type=int, default=None, help='若提供，则覆盖数据推断的根骨前向轴(0/1/2)')
    p.add_argument('--yaw_forward_offset', type=float, default=None, help='额外指定 yaw 前向轴偏移（单位：度，优先于数据推断）')
    p.add_argument('--eval_angvel_dir_percentile', type=float, default=0.75, help='KeyBone 角速度方向指标仅统计大于该分位数的帧 (0~1)')
    p.add_argument('--diag_input_stats', action='store_true', help='启用输入特征统计（Teacher vs Free-run）')
    p.add_argument('--freerun_debug_path', type=str, default=None, help='若提供，则将首个 freerun batch 的诊断数据保存至该路径')
    p.add_argument('--no_grad_conn_test', action='store_true', help='跳过训练前的梯度连通性自检')

    return config_parser, p

def _parse_train_entry_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser, parser = _build_train_parser()
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    required_actions = []
    for action in parser._actions:
        if getattr(action, 'required', False):
            required_actions.append(action)
            action.required = False

    config_defaults = _load_train_entry_config_defaults(config_args.config_json, parser)
    legacy_cli_hits: list[str] = []
    for token in remaining_argv:
        for flag, key in LEGACY_LOSS_CLI_FLAGS.items():
            if token == flag or token.startswith(f'{flag}='):
                legacy_cli_hits.append(key)
    if legacy_cli_hits:
        parser.error(_legacy_loss_keys_msg(legacy_cli_hits, context='CLI args'))

    removed_phase_reset_cli_hits: list[str] = []
    for token in remaining_argv:
        for flag, key in REMOVED_TRAINBASE_PHASE_RESET_CLI_FLAGS.items():
            if token == flag or token.startswith(f'{flag}='):
                removed_phase_reset_cli_hits.append(key)
    if removed_phase_reset_cli_hits:
        parser.error(_removed_trainbase_phase_reset_msg(removed_phase_reset_cli_hits, context='CLI args'))

    namespace = argparse.Namespace(**config_defaults)
    namespace.config_json = config_args.config_json
    parsed_args = parser.parse_args(remaining_argv, namespace=namespace)
    set_global_args(parsed_args)
    _apply_train_entry_config_overrides(parsed_args, getattr(parsed_args, 'config_override', None), parser)
    parsed_args.config_override = None

    missing_required = [act for act in required_actions if getattr(parsed_args, act.dest, None) is None]
    if missing_required:
        missing_opts = []
        for act in missing_required:
            if act.option_strings:
                missing_opts.append(act.option_strings[-1])
            else:
                missing_opts.append(act.dest)
        parser.error(f"missing required arguments: {', '.join(missing_opts)}")
    return parsed_args


def _resolve_trainer_runtime_config(
    args: argparse.Namespace,
    trainer: Trainer,
    ds_train: Any,
    norm_template_path: Optional[Path],
    bundle_json_path: Optional[str],
    out_dir: Path,
    resolved_config: Dict[str, Any],
    run_name: str,
) -> TrainerRuntimeConfig:
    pose_norm = getattr(ds_train, 'pose_hist_norm', None)
    pose_hist_scales = None
    pose_hist_mu = None
    pose_hist_std = None
    if pose_norm is not None:
        pose_hist_scales = torch.as_tensor(pose_norm.scales, dtype=torch.float32)
        pose_hist_mu = (
            torch.as_tensor(pose_norm.mu, dtype=torch.float32)
            if getattr(pose_norm, 'mu', None) is not None
            else None
        )
        pose_hist_std = (
            torch.as_tensor(pose_norm.std, dtype=torch.float32)
            if getattr(pose_norm, 'std', None) is not None
            else None
        )

    try:
        up_cm = float(args.contact_meas_ground_z_slew_up_cm or 0.0)
    except Exception:
        up_cm = 0.0
    try:
        down_cm = float(args.contact_meas_ground_z_slew_down_cm or 0.0)
    except Exception:
        down_cm = 0.0

    forward_axis_override = args.yaw_forward_axis
    if forward_axis_override is not None:
        yaw_forward_axis = int(forward_axis_override)
    elif getattr(ds_train, 'forward_axis', None) is not None:
        yaw_forward_axis = int(ds_train.forward_axis)
    else:
        yaw_forward_axis = int(getattr(trainer, 'yaw_forward_axis', 2))

    offset_override = args.yaw_forward_offset
    if offset_override is not None:
        yaw_forward_axis_offset = float(_math.radians(float(offset_override)))
    else:
        yaw_forward_axis_offset = float(getattr(ds_train, 'forward_axis_offset', 0.0) or 0.0)

    monitor_batches = int(args.monitor_batches or 8)
    try:
        freerun_stage_schedule = _parse_stage_schedule(args.freerun_stage_schedule)
    except Exception as exc:
        freerun_stage_schedule = []
        print(
            f"[StageSchedule][WARN] failed to parse freerun_stage_schedule ({exc}); fallback to empty schedule."
        )
    _assert_no_legacy_loss_keys_in_schedule(
        freerun_stage_schedule,
        context='freerun_stage_schedule',
    )
    _assert_no_removed_trainbase_stage_keys(
        freerun_stage_schedule,
        context='freerun_stage_schedule',
    )

    gate_raw = getattr(args, 'contact_meas_gate_by_hit', 'auto')
    gate_value = str(gate_raw if gate_raw is not None else 'auto').strip().lower()
    if gate_value in ('true', '1', 'yes', 'y'):
        contact_meas_gate_by_hit_override: Optional[bool] = True
    elif gate_value in ('false', '0', 'no', 'n'):
        contact_meas_gate_by_hit_override = False
    else:
        contact_meas_gate_by_hit_override = None

    trainbase_contacts_source = str(
        getattr(args, '_trainbase_contacts_source_resolved', getattr(args, 'trainbase_contacts_source', 'auto')) or 'auto'
    ).strip().lower()
    if trainbase_contacts_source not in ('whitebox', 'pretrain_contact'):
        trainbase_contacts_source = 'whitebox'
    try:
        trainbase_contacts_pretrain_clamp = float(getattr(args, 'trainbase_contacts_pretrain_clamp', 1.0) or 0.0)
    except Exception:
        trainbase_contacts_pretrain_clamp = 1.0
    if (not _math.isfinite(float(trainbase_contacts_pretrain_clamp))) or float(trainbase_contacts_pretrain_clamp) < 0.0:
        trainbase_contacts_pretrain_clamp = 1.0
    trainbase_contacts_pretrain_affine_stats = getattr(args, 'trainbase_contacts_pretrain_affine_stats', None)
    trainbase_contacts_pretrain_affine = _parse_pretrain_contact_affine_spec(trainbase_contacts_pretrain_affine_stats)
    if trainbase_contacts_pretrain_affine_stats not in (None, '') and trainbase_contacts_pretrain_affine is None:
        print('[MPL][WARN] failed to parse --trainbase_contacts_pretrain_affine_stats; continuing without affine calibration.')

    return TrainerRuntimeConfig(
        norm_template_path=str(norm_template_path) if norm_template_path else None,
        bundle_json_path=bundle_json_path,
        out_dir=str(out_dir),
        direct_pose_grad_monitor_enable=bool(args.direct_pose_grad_monitor_enable),
        direct_pose_grad_ratio_gate=float(args.direct_pose_grad_ratio_gate or 0.35),
        full_config=resolved_config,
        pose_hist_len=int(getattr(ds_train, 'pose_hist_len', 0) or 0),
        pose_hist_dim=int(getattr(ds_train, 'pose_hist_dim', 0) or 0),
        pose_hist_scales=pose_hist_scales,
        pose_hist_mu=pose_hist_mu,
        pose_hist_std=pose_hist_std,
        foot_contact_threshold=float(args.foot_contact_threshold),
        bone_hz=float(getattr(ds_train, 'fps', 60.0) or 60.0),
        fps=float(getattr(ds_train, 'fps', 60.0) or 60.0),
        trainbase_contacts_source=trainbase_contacts_source,
        trainbase_contacts_pretrain_clamp=float(trainbase_contacts_pretrain_clamp),
        trainbase_contacts_pretrain_affine_stats=(
            str(trainbase_contacts_pretrain_affine_stats).strip()
            if trainbase_contacts_pretrain_affine_stats not in (None, '')
            else None
        ),
        trainbase_contacts_pretrain_affine=trainbase_contacts_pretrain_affine,
        contact_meas_gate_by_hit_override=contact_meas_gate_by_hit_override,
        contact_meas_ground_z_mode=str(getattr(args, 'contact_meas_ground_z_mode', 'window') or 'window').strip().lower(),
        contact_meas_ground_z_beta=float(getattr(args, 'contact_meas_ground_z_beta', 0.05) or 0.05),
        contact_meas_ground_z_window=int(getattr(args, 'contact_meas_ground_z_window', 5) or 5),
        contact_meas_ground_z_quantile=float(getattr(args, 'contact_meas_ground_z_quantile', 0.2) or 0.2),
        contact_meas_ground_z_max_up_m=max(0.0, up_cm) / 100.0,
        contact_meas_ground_z_max_down_m=max(0.0, down_cm) / 100.0,
        diag_topk=int(args.diag_topk or 8),
        diag_thr=float(args.diag_thr or 8.0),
        yaw_forward_axis=yaw_forward_axis,
        yaw_forward_axis_offset=yaw_forward_axis_offset,
        eval_angvel_dir_percentile=float(args.eval_angvel_dir_percentile),
        diag_input_stats=bool(args.diag_input_stats),
        val_mode=args.val_mode,
        no_monitor=bool(args.no_monitor),
        monitor_batches=monitor_batches,
        teacher_eval_max_batches=args.teacher_eval_max_batches,
        force_valfree_eval=bool(args.force_valfree_eval),
        eval_settings=FreeRunSettings(
            warmup_steps=int(args.eval_warmup or 0),
            horizon=args.eval_horizon,
            max_batches=monitor_batches,
        ),
        ss_chunk_len=int(getattr(args, 'ss_chunk_len', getattr(trainer, 'ss_chunk_len', 1)) or 1),
        tf_mode=args.tf_mode,
        tf_warmup_epochs=int(args.tf_warmup_epochs),
        tf_start_epoch=int(args.tf_start_epoch),
        tf_end_epoch=int(args.tf_end_epoch),
        tf_max=float(args.tf_max),
        tf_min=float(args.tf_min),
        history_debug_steps=int(args.history_debug_steps or 0),
        history_dropout_prob=float(args.history_dropout_prob or 0.0),
        history_dropout_prob_min=0.05,
        history_dropout_prob_max=0.30,
        freerun_stage_schedule=freerun_stage_schedule,
        adaptive_loss_module=None,
        hyperparam_scheduler=None,
        freerun_debug_path=args.freerun_debug_path,
        enable_grad_connection_test=not bool(args.no_grad_conn_test),
        current_run_name=run_name,
    )


def _apply_trainer_runtime_config(trainer: Trainer, runtime_cfg: TrainerRuntimeConfig) -> None:
    trainer._norm_template_path = runtime_cfg.norm_template_path
    trainer._bundle_json_path = runtime_cfg.bundle_json_path
    trainer.out_dir = runtime_cfg.out_dir
    trainer.direct_pose_grad_monitor_enable = runtime_cfg.direct_pose_grad_monitor_enable
    trainer.direct_pose_grad_ratio_gate = runtime_cfg.direct_pose_grad_ratio_gate
    trainer.full_config = runtime_cfg.full_config
    trainer.pose_hist_len = runtime_cfg.pose_hist_len
    trainer.pose_hist_dim = runtime_cfg.pose_hist_dim
    trainer.pose_hist_scales = runtime_cfg.pose_hist_scales
    trainer.pose_hist_mu = runtime_cfg.pose_hist_mu
    trainer.pose_hist_std = runtime_cfg.pose_hist_std
    trainer.foot_contact_threshold = runtime_cfg.foot_contact_threshold
    trainer.bone_hz = runtime_cfg.bone_hz
    trainer.fps = runtime_cfg.fps
    trainer.trainbase_contacts_source = runtime_cfg.trainbase_contacts_source
    trainer.trainbase_contacts_pretrain_clamp = runtime_cfg.trainbase_contacts_pretrain_clamp
    trainer.trainbase_contacts_pretrain_affine_stats_spec = runtime_cfg.trainbase_contacts_pretrain_affine_stats
    trainer.trainbase_contacts_pretrain_affine = runtime_cfg.trainbase_contacts_pretrain_affine
    trainer.contact_meas_gate_by_hit_override = runtime_cfg.contact_meas_gate_by_hit_override
    trainer.contact_meas_ground_z_mode = runtime_cfg.contact_meas_ground_z_mode
    trainer.contact_meas_ground_z_beta = runtime_cfg.contact_meas_ground_z_beta
    trainer.contact_meas_ground_z_window = runtime_cfg.contact_meas_ground_z_window
    trainer.contact_meas_ground_z_quantile = runtime_cfg.contact_meas_ground_z_quantile
    trainer.contact_meas_ground_z_max_up_m = runtime_cfg.contact_meas_ground_z_max_up_m
    trainer.contact_meas_ground_z_max_down_m = runtime_cfg.contact_meas_ground_z_max_down_m
    safe_set_slice(trainer, 'rootvel_x_slice', parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity'))
    safe_set_slice(trainer, 'angvel_x_slice', parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities'))
    trainer.diag_topk = runtime_cfg.diag_topk
    trainer.diag_thr = runtime_cfg.diag_thr
    trainer.yaw_forward_axis = runtime_cfg.yaw_forward_axis
    trainer.yaw_forward_axis_offset = runtime_cfg.yaw_forward_axis_offset
    trainer.eval_angvel_dir_percentile = runtime_cfg.eval_angvel_dir_percentile
    trainer.diag_input_stats = runtime_cfg.diag_input_stats
    trainer.val_mode = runtime_cfg.val_mode
    trainer.no_monitor = runtime_cfg.no_monitor
    trainer.monitor_batches = runtime_cfg.monitor_batches
    trainer.teacher_eval_max_batches = runtime_cfg.teacher_eval_max_batches
    trainer.force_valfree_eval = runtime_cfg.force_valfree_eval
    trainer.eval_settings = runtime_cfg.eval_settings
    trainer.ss_chunk_len = runtime_cfg.ss_chunk_len
    trainer.tf_mode = runtime_cfg.tf_mode
    trainer.tf_warmup_epochs = runtime_cfg.tf_warmup_epochs
    trainer.tf_start_epoch = runtime_cfg.tf_start_epoch
    trainer.tf_end_epoch = runtime_cfg.tf_end_epoch
    trainer.tf_max = runtime_cfg.tf_max
    trainer.tf_min = runtime_cfg.tf_min
    trainer.history_debug_steps = runtime_cfg.history_debug_steps
    trainer.history_dropout_prob = runtime_cfg.history_dropout_prob
    trainer.history_dropout_prob_min = runtime_cfg.history_dropout_prob_min
    trainer.history_dropout_prob_max = runtime_cfg.history_dropout_prob_max
    trainer.freerun_stage_schedule = runtime_cfg.freerun_stage_schedule
    trainer.adaptive_loss_module = runtime_cfg.adaptive_loss_module
    trainer.hyperparam_scheduler = runtime_cfg.hyperparam_scheduler
    trainer.freerun_debug_path = runtime_cfg.freerun_debug_path
    trainer.enable_grad_connection_test = runtime_cfg.enable_grad_connection_test
    trainer._current_run_name = runtime_cfg.current_run_name


def _build_train_components() -> Any:
    global GLOBAL_ARGS

    GLOBAL_ARGS = _parse_train_entry_args()
    args = GLOBAL_ARGS

    train_paths = expand_paths_from_specs(args.train_files)
    if not train_paths:
        if args.data and os.path.isdir(args.data):
            train_paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
        else:
            raise FileNotFoundError('No training files. Provide --train_files or --data with .npz')
    run_name = args.run_name or time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device('mps')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else torch.device('cuda')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    def _load_json_spec(path_str: Optional[str], label: str) -> Any:
        if not path_str:
            return None
        path = Path(path_str).expanduser()
        if not path.is_file():
            print(f"[Spec][WARN] {label} not found at {path}")
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[Spec] Loaded {label}: {path}")
            return data
        except Exception as err:
            print(f"[Spec][WARN] failed to read {label} {path}: {err}")
            return None

    norm_template_arg = args.norm_template
    norm_template_path = Path(norm_template_arg).expanduser() if norm_template_arg else None
    norm_spec = _load_json_spec(norm_template_arg, 'norm_template')
    if norm_spec is None:
        raise SystemExit(f"[FATAL] norm_template 缺失或无效，请确认路径：{norm_template_path}")
    require_standard_rotvec_spec(norm_spec, context=f"norm_template {norm_template_path}")
    pretrain_template_arg = args.pretrain_template
    pretrain_spec = _load_json_spec(pretrain_template_arg, 'pretrain_template')
    if pretrain_spec is not None:
        require_standard_rotvec_spec(
            pretrain_spec,
            context=f"pretrain_template {Path(pretrain_template_arg).expanduser() if pretrain_template_arg else 'None'}",
        )
        for key in (
            'MuAngVel',
            'StdAngVel',
            'tanh_scales_angvel',
            'pose_hist_len',
            'tanh_scales_pose_hist',
            'MuPoseHist',
            'StdPoseHist',
        ):
            if key in pretrain_spec and pretrain_spec[key] is not None:
                norm_spec[key] = pretrain_spec[key]
    pose_hist_len = 0
    if norm_spec is not None:
        try:
            pose_hist_len = int(norm_spec.get('pose_hist_len', 0) or 0)
        except Exception:
            pose_hist_len = 0

    return SimpleNamespace(
        args=args,
        train_paths=train_paths,
        run_name=run_name,
        out_dir=out_dir,
        device=device,
        norm_template_path=norm_template_path,
        norm_spec=norm_spec,
        pose_hist_len=pose_hist_len,
    )


def _build_train_loaders(train_ctx: Any) -> Any:
    from torch.utils.data import DataLoader

    args = train_ctx.args
    ds_train = MotionEventDataset(
        args.data,
        seq_len=args.seq_len,
        paths=train_ctx.train_paths,
        pose_hist_len=train_ctx.pose_hist_len,
        norm_spec=train_ctx.norm_spec,
    )
    ds_train = _maybe_optimize_dataset_index(ds_train, args)
    ds_train.is_train = True
    ds_train.yaw_aug_deg = float(args.yaw_aug_deg)
    ds_train.normalize_c = bool(args.normalize_c)
    if not hasattr(ds_train, 'state_layout'):
        ds_train.state_layout = getattr(ds_train, 'state_layout', {}) or {}
    pin = train_ctx.device.type == 'cuda'
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        **({'prefetch_factor': 2} if args.num_workers > 0 else {}),
    )
    loader_kwargs['collate_fn'] = make_fixedlen_collate(args.seq_len)
    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True, **loader_kwargs)
    dx, dy, dc = int(ds_train.Dx), int(ds_train.Dy), int(ds_train.Dc)
    print(f'[Export][Dims] Dx={dx}, Dy={dy}, Dc={dc} | L={int(args.depth)}, H={int(args.width)}, K={int(args.context_len)}')
    return SimpleNamespace(
        ds_train=ds_train,
        train_loader=train_loader,
        pin_memory=pin,
        dx=dx,
        dy=dy,
        dc=dc,
    )


def _build_train_model(
    train_ctx: Any,
    train_data: Any,
) -> Any:
    args = train_ctx.args
    ds_train = train_data.ds_train
    device = train_ctx.device

    pose_hist_dim_raw = int(getattr(ds_train, 'pose_hist_dim', 0) or 0)
    pose_hist_len_raw = int(getattr(ds_train, 'pose_hist_len', 0) or 0)
    history_export_frames = int(getattr(args, 'history_adaptive_export_frames', 0) or 0)
    history_frame_dim = (
        pose_hist_dim_raw // pose_hist_len_raw
        if pose_hist_len_raw > 0 and pose_hist_dim_raw % pose_hist_len_raw == 0
        else 0
    )
    pose_hist_dim_model = pose_hist_dim_raw
    if history_export_frames > 0 and history_frame_dim > 0:
        pose_hist_dim_model = history_export_frames * history_frame_dim

    direct_pose_split_enable = bool(args.direct_pose_split_enable)
    direct_pose_arm_split_enable = bool(args.direct_pose_arm_split_enable)
    arm_bones_raw = getattr(args, 'direct_pose_arm_bones', None)
    if not direct_pose_arm_split_enable:
        if arm_bones_raw is None:
            direct_pose_arm_bones_resolved = None
        else:
            arm_bones_txt = str(arm_bones_raw).strip()
            direct_pose_arm_bones_resolved = arm_bones_txt or None
    else:
        if arm_bones_raw is None:
            direct_pose_arm_bones_resolved = str(STAGE6_3WAY_ARMCHAIN_BONES_CSV)
        else:
            arm_bones_txt = str(arm_bones_raw).strip()
            direct_pose_arm_bones_resolved = arm_bones_txt or str(STAGE6_3WAY_ARMCHAIN_BONES_CSV)
    direct_pose_nonleg_proj_dim = int(args.direct_pose_nonleg_proj_dim or 0)

    model = EventMotionModel(
        in_state_dim=ds_train.Dx,
        out_motion_dim=ds_train.Dy,
        cond_dim=ds_train.Dc,
        period_dim=getattr(ds_train, 'period_dim', 0),
        hidden_dim=args.width,
        num_layers=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        context_len=args.context_len,
        contact_dim=getattr(ds_train, 'contact_dim', 0),
        angvel_dim=getattr(ds_train, 'angvel_dim', 0),
        pose_hist_dim=pose_hist_dim_model,
        state_layout=getattr(ds_train, 'state_layout', None),
        bone_names=getattr(ds_train, 'bone_names', None),
        output_layout=getattr(ds_train, 'output_layout', None),
        contact_plan_enable=bool(args.contact_plan_enable),
        contact_plan_hidden=int(args.contact_plan_hidden or 64),
        contact_plan_dropout=float(args.contact_plan_dropout or 0.0),
        contact_plan_inject=str(args.contact_plan_inject or 'none'),
        contact_plan_inject_detach=bool(args.contact_plan_inject_detach),
        contact_plan_time_pe_dim=int(args.contact_plan_time_pe_dim or 0),
        contact_plan_time_pe_base=float(args.contact_plan_time_pe_base or 10000.0),
        contact_plan_init_mode=str(getattr(args, 'contact_plan_init_mode', 'learnable') or 'learnable'),
        contact_plan_init_hidden=int(args.contact_plan_init_hidden or 128),
        contact_plan_init_dropout=float(args.contact_plan_init_dropout or 0.0),
        contact_phase_state_enable=bool(getattr(args, 'contact_phase_state_enable', False)),
        contact_phase_state_init_mode=str(getattr(args, 'contact_phase_state_init_mode', 'obs') or 'obs'),
        contact_phase_state_hidden=int(args.contact_phase_state_hidden or 64),
        contact_phase_state_delta_max=float(args.contact_phase_state_delta_max or 0.5),
        contact_phase_state_delta_init=float(args.contact_phase_state_delta_init or (6.283185307179586 / 80.0)),
        contact_phase_state_event_kind='none',
        contact_phase_state_event_thr=0.5,
        contact_phase_state_event_hyst=0.0,
        contact_phase_state_event_min_interval=0,
        phase_reset_source='none',
        use_event_clock=bool(args.use_event_clock),
        event_clock_max_delta=float(args.event_clock_max_delta or 0.5),
        event_clock_hidden_dim=int(args.event_clock_hidden_dim or 64),
        event_clock_gate_hidden_dim=int(args.event_clock_gate_hidden_dim or 32),
        direct_pose_enable=bool(args.direct_pose_enable),
        direct_pose_hidden=int(args.direct_pose_hidden or 256),
        direct_pose_dropout=float(args.direct_pose_dropout or 0.0),
        direct_pose_detach_plan=bool(args.direct_pose_detach_plan),
        direct_pose_meas_mode=str(getattr(args, 'direct_pose_meas_mode', 'concat') or 'concat'),
        direct_pose_meas_drop_prob=float(args.direct_pose_meas_drop_prob or 0.0),
        direct_pose_meas_noise_std=float(args.direct_pose_meas_noise_std or 0.0),
        direct_pose_plan_drop_prob=float(args.direct_pose_plan_drop_prob or 0.0),
        direct_pose_feat_source=str(getattr(args, 'direct_pose_feat_source', 'cond') or 'cond'),
        direct_pose_time_pe_dim=int(getattr(args, 'direct_pose_time_pe_dim', 0) or 0),
        direct_pose_time_pe_base=float(getattr(args, 'direct_pose_time_pe_base', 10000.0) or 10000.0),
        direct_pose_split_enable=bool(direct_pose_split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_nonleg_proj_dim),
        direct_pose_arm_split_enable=bool(direct_pose_arm_split_enable),
        direct_pose_arm_bones=direct_pose_arm_bones_resolved,
    ).to(device)
    return SimpleNamespace(
        model=model,
        pose_hist_dim_raw=pose_hist_dim_raw,
        pose_hist_len_raw=pose_hist_len_raw,
        history_export_frames=history_export_frames,
        history_frame_dim=history_frame_dim,
        direct_pose_split_enable=direct_pose_split_enable,
        direct_pose_arm_split_enable=direct_pose_arm_split_enable,
        direct_pose_arm_bones_resolved=direct_pose_arm_bones_resolved,
        direct_pose_nonleg_proj_dim=direct_pose_nonleg_proj_dim,
    )


def _prepare_train_model_runtime(
    train_ctx: Any,
    train_data: Any,
    model_artifacts: Any,
) -> None:
    args = train_ctx.args
    ds_train = train_data.ds_train
    device = train_ctx.device
    model = model_artifacts.model

    if model_artifacts.history_export_frames > 0:
        if model_artifacts.pose_hist_dim_raw <= 0 or model_artifacts.pose_hist_len_raw <= 0:
            print('[AdaptiveHistory][WARN] pose history not available; adaptive history disabled.')
        elif model_artifacts.pose_hist_dim_raw % model_artifacts.pose_hist_len_raw != 0:
            print('[AdaptiveHistory][WARN] pose history dim不整除帧数，跳过 adaptive history。')
        else:
            max_frames = args.history_adaptive_max_frames
            if max_frames is None:
                max_frames = model_artifacts.pose_hist_len_raw
            from .history import AdaptiveHistoryModule

            module_device = torch.device('cpu') if device.type == 'mps' else device
            history_module = AdaptiveHistoryModule(
                pose_dim=model_artifacts.history_frame_dim,
                hidden_dim=int(args.history_adaptive_hidden or int(args.width)),
                num_history_frames=model_artifacts.history_export_frames,
                max_history_frames=int(max_frames),
                cond_dim=0,
                num_heads=int(args.history_adaptive_heads or 2),
                train_variable_history=bool(args.history_adaptive_train_variable),
                history_dropout_prob=float(args.history_dropout_prob or 0.0),
                use_trend_features=bool(args.history_use_trend_features),
            ).to(module_device)
            model.enable_adaptive_history(history_module, pose_hist_len=model_artifacts.pose_hist_len_raw)

    validate_and_fix_model_(model, train_data.dx, train_data.dc)
    validate_and_fix_model_(model)

    encoder_path_cfg = getattr(args, 'encoder_path', '')
    resolved_bundle = None
    if encoder_path_cfg:
        base_candidate = Path(encoder_path_cfg).expanduser()
        search_roots = [
            base_candidate,
            Path(__file__).resolve().parent / base_candidate,
            Path(__file__).resolve().parent.parent / base_candidate,
        ]
        data_root = Path(args.data).expanduser() if getattr(args, 'data', None) else None
        if data_root is not None:
            search_roots.append(data_root.parent / base_candidate)
        for cand in search_roots:
            if cand is not None and cand.is_file():
                resolved_bundle = cand
                break
        if resolved_bundle is None:
            print(f'[MPL][WARN] MotionEncoder bundle not found (tried {encoder_path_cfg})')
        else:
            try:
                bundle = torch.load(str(resolved_bundle), map_location='cpu')
                model.attach_motion_encoder(bundle)
                print(f'[MPL] Attached MotionEncoder bundle: {resolved_bundle}')
                try:
                    ds_train.period_dim = getattr(model, 'period_dim', getattr(ds_train, 'period_dim', 0))
                except (AttributeError, TypeError, ValueError) as exc:
                    _phasec_warn_once(
                        "train_entry/attach_bundle_period_dim",
                        "failed to sync dataset period_dim from attached MotionEncoder bundle",
                        exc,
                    )
            except Exception as err:
                print(f'[MPL][WARN] failed to attach MotionEncoder bundle: {err}')

    requested_contacts_source = str(getattr(args, 'trainbase_contacts_source', 'auto') or 'auto').strip().lower()
    if requested_contacts_source not in ('auto', 'whitebox', 'pretrain_contact'):
        requested_contacts_source = 'auto'
    resolved_contacts_source = 'whitebox'
    if bool(getattr(model, 'contact_plan_enable', False)):
        has_pretrain_contact = (
            getattr(model, 'frozen_encoder', None) is not None
            and getattr(model, 'frozen_contact_head', None) is not None
        )
        if requested_contacts_source == 'pretrain_contact':
            if not has_pretrain_contact:
                raise SystemExit(
                    '[FATAL] trainbase_contacts_source=pretrain_contact requires --encoder_path to provide '
                    'a frozen encoder bundle with contact_head.'
                )
            resolved_contacts_source = 'pretrain_contact'
        elif requested_contacts_source == 'auto':
            resolved_contacts_source = 'pretrain_contact' if has_pretrain_contact else 'whitebox'
    setattr(args, '_trainbase_contacts_source_resolved', resolved_contacts_source)
    print(
        f'[MPL] trainbase_contacts_source: requested={requested_contacts_source}, '
        f'resolved={resolved_contacts_source}'
    )

    resume_path = getattr(args, 'resume', None)
    if resume_path:
        try:
            ckpt_path = Path(str(resume_path)).expanduser()
            if not ckpt_path.is_file():
                print(f'[Resume][WARN] checkpoint not found: {ckpt_path}')
            else:
                payload = torch.load(str(ckpt_path), map_location='cpu')
                state_dict = payload.get('model', payload) if isinstance(payload, dict) else payload
                if not isinstance(state_dict, dict):
                    print(f'[Resume][WARN] checkpoint has no state_dict: {ckpt_path}')
                else:
                    try:
                        model.adapt_legacy_state_dict_(state_dict)
                    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                        _phasec_warn_once(
                            "train_entry/resume_adapt_legacy_state",
                            "adapt_legacy_state_dict_ failed; loading matching keys directly",
                            exc,
                        )
                    cur = model.state_dict()
                    filtered = {}
                    skipped = []
                    for key, value in state_dict.items():
                        if key in cur and torch.is_tensor(value) and torch.is_tensor(cur[key]) and tuple(cur[key].shape) == tuple(value.shape):
                            filtered[key] = value
                        else:
                            skipped.append(key)
                    missing, unexpected = model.load_state_dict(filtered, strict=False)
                    print(
                        f'[Resume] loaded={len(filtered)}/{len(state_dict)} '
                        f'missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped)} ckpt={ckpt_path}'
                    )
        except Exception as err:
            print(f'[Resume][WARN] failed to load checkpoint: {err}')

    with torch.no_grad():
        first_linear = model.shared_encoder[0]
        if not torch.isfinite(first_linear.weight).all() or (
            first_linear.bias is not None and (not torch.isfinite(first_linear.bias).all())
        ):
            print('[Guard] first-linear became non-finite post-sanitize, reinitializing')
            torch.nn.init.kaiming_uniform_(first_linear.weight, a=_math.sqrt(5))
            if first_linear.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(first_linear.weight)
                bound = 1.0 / _math.sqrt(max(fan_in, 1))
                torch.nn.init.uniform_(first_linear.bias, -bound, bound)
            assert torch.isfinite(first_linear.weight).all() and (
                first_linear.bias is None or torch.isfinite(first_linear.bias).all()
            )
    with torch.no_grad():
        first_linear = model.shared_encoder[0]
        assert torch.isfinite(first_linear.weight).all() and (
            first_linear.bias is None or torch.isfinite(first_linear.bias).all()
        ), '[PostCheck] shared_encoder.0 still not finite'
    try:
        model._pasa_fps = float(getattr(ds_train, 'fps', 60.0))
    except (AttributeError, TypeError, ValueError) as exc:
        _phasec_warn_once(
            "train_entry/pasa_fps",
            "failed to set model._pasa_fps from dataset fps",
            exc,
        )


def _build_train_loss_and_trainer(
    train_ctx: Any,
    train_data: Any,
    model_artifacts: Any,
) -> Any:
    args = train_ctx.args
    ds_train = train_data.ds_train
    model = model_artifacts.model
    direct_pose_arm_split_enable = model_artifacts.direct_pose_arm_split_enable
    direct_pose_arm_bones_resolved = model_artifacts.direct_pose_arm_bones_resolved

    fps_data = float(getattr(ds_train, 'fps', 60.0) or 60.0)
    w_rot_local = float(args.w_rot_local or 0.0)
    w_root_vel = float(args.w_root_vel or 0.0)
    w_root_speed = float(args.w_root_speed or 0.0)
    adaptive_bone_weights = bool(args.adaptive_bone_weights)

    loss_fn = MotionJointLoss(
        output_layout=ds_train.output_layout,
        fps=fps_data,
        rot6d_spec=getattr(ds_train, 'rot6d_spec', {}),
        w_rot_ortho=args.w_rot_ortho,
        meta=getattr(ds_train, 'meta', None),
        w_rot_local=w_rot_local,
        w_root_vel=w_root_vel,
        w_root_speed=w_root_speed,
        w_contact_plan=float(args.w_contact_plan or 0.0),
        w_direct_pose=float(args.w_direct_pose or 0.0),
        direct_pose_loss_leg_split=bool(args.direct_pose_loss_leg_split),
        direct_pose_arm_split_enable=bool(direct_pose_arm_split_enable),
        direct_pose_arm_bones=direct_pose_arm_bones_resolved,
        direct_pose_loss_arm_else_balance_enable=bool(args.direct_pose_loss_arm_else_balance_enable),
        direct_pose_loss_arm_weight=float(args.direct_pose_loss_arm_weight or 1.0),
        direct_pose_loss_else_weight=float(args.direct_pose_loss_else_weight or 1.0),
        direct_pose_loss_group_norm_enable=bool(args.direct_pose_loss_group_norm_enable),
        direct_pose_loss_group_norm_w_leg=float(args.direct_pose_loss_group_norm_w_leg or 1.0),
        direct_pose_loss_group_norm_w_nonleg=float(args.direct_pose_loss_group_norm_w_nonleg or 1.0),
        direct_pose_loss_group_norm_ema_beta=float(args.direct_pose_loss_group_norm_ema_beta or 0.9),
        direct_pose_loss_group_norm_ratio_min=float(args.direct_pose_loss_group_norm_ratio_min or 0.2),
        direct_pose_loss_group_norm_ratio_max=float(args.direct_pose_loss_group_norm_ratio_max or 5.0),
        direct_pose_loss_group_norm_eps=float(args.direct_pose_loss_group_norm_eps or 1e-6),
        event_clock_lambda_entropy_weight=float(args.event_clock_lambda_entropy_weight or 0.01),
        event_clock_lambda_prior_weight=float(args.event_clock_lambda_prior_weight or 0.01),
        event_clock_delta_z_l2_weight=float(args.event_clock_delta_z_l2_weight or 0.001),
        adaptive_bone_weights=adaptive_bone_weights,
    )
    loss_fn.unified_downstream_power = float(args.unified_downstream_power or 0.6)
    loss_fn.unified_self_scale = float(args.unified_self_scale or 1.5)
    loss_fn.unified_min_weight = float(args.unified_min_weight or 0.05)
    loss_fn.rot_local_tail_weight = float(args.rot_local_tail_weight or 0.0)
    loss_fn.rot_local_tail_k = int(args.rot_local_tail_k or 0)
    loss_fn.rot_local_tail_scope = str(args.rot_local_tail_scope or 'all')
    loss_fn.rot_local_tail_select = str(args.rot_local_tail_select or 'batch')
    loss_fn.rot_local_tail_ema_beta = float(args.rot_local_tail_ema_beta or 0.9)
    loss_fn.unified_use_visual_importance = False
    if getattr(ds_train, 'bone_names', None):
        try:
            loss_fn.set_bone_names(ds_train.bone_names)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            _phasec_warn_once(
                "train_entry/loss_set_bone_names",
                "loss_fn.set_bone_names failed; continuing with default bone metadata",
                exc,
            )
    if getattr(ds_train, 'parents', None):
        try:
            loss_fn.set_skeleton(ds_train.parents, getattr(ds_train, 'bone_offsets', None))
        except Exception as exc:
            print(f'[Loss][WARN] set_skeleton failed: {exc}')
    bundle_json_arg = args.bundle_json
    bundle_json_path = str(Path(bundle_json_arg).expanduser()) if bundle_json_arg else None
    loss_fn.template_hint = str(train_ctx.norm_template_path) if train_ctx.norm_template_path else None
    loss_fn.bundle_hint = bundle_json_path

    print(
        f'[LossWeights] '
        f'w_rot_ortho={loss_fn.w_rot_ortho} '
        f'w_rot_local={loss_fn.w_rot_local} '
        f'rot_local_tail_weight={getattr(loss_fn, "rot_local_tail_weight", 0.0)} '
        f'rot_local_tail_k={getattr(loss_fn, "rot_local_tail_k", 0)} '
        f'rot_local_tail_scope={getattr(loss_fn, "rot_local_tail_scope", "all")} '
        f'rot_local_tail_select={getattr(loss_fn, "rot_local_tail_select", "batch")} '
        f'rot_local_tail_ema_beta={getattr(loss_fn, "rot_local_tail_ema_beta", 0.9)} '
        f'adaptive_bone_weights={loss_fn.use_adaptive_weights}'
    )

    loss_fn.dt_traj = 1.0 / max(1e-6, fps_data)
    loss_fn.dt_bone = 1.0 / max(1e-6, fps_data)
    print(f'[Dt] dt_traj={loss_fn.dt_traj:.6f}s | dt_bone={loss_fn.dt_bone:.6f}s (dataset fps={fps_data})')

    if hasattr(loss_fn, 'rot6d_eps'):
        loss_fn.rot6d_eps = 1e-6
    augmentor = MotionAugmentation(noise_std=args.aug_noise_std, time_warp_prob=args.aug_time_warp_prob)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        tf_warmup_steps=args.tf_warmup_steps,
        tf_total_steps=args.tf_total_steps,
        augmentor=augmentor,
        use_amp=args.amp,
        accum_steps=args.accum_steps,
        pin_memory=train_data.pin_memory,
        args=args,
    )
    if bool(args.direct_pose_loss_group_norm_enable) and (not bool(args.direct_pose_loss_leg_split)):
        print('[Loss][WARN] direct_pose_loss_group_norm_enable=true but direct_pose_loss_leg_split=false; group norm will have no effect.')
    try:
        resolved_config = dict(vars(args))
    except Exception:
        resolved_config = {}
    resolved_config['direct_pose_split_enable'] = bool(model_artifacts.direct_pose_split_enable)
    resolved_config['direct_pose_arm_split_enable'] = bool(direct_pose_arm_split_enable)
    resolved_config['direct_pose_arm_bones'] = direct_pose_arm_bones_resolved
    resolved_config['direct_pose_nonleg_proj_dim'] = int(model_artifacts.direct_pose_nonleg_proj_dim)
    try:
        cfg_out = train_ctx.out_dir / 'config_resolved.json'
        with open(cfg_out, 'w', encoding='utf-8') as f:
            json.dump(resolved_config, f, ensure_ascii=False, indent=2, default=str)
        print(f'[Config] saved resolved config to {cfg_out}')
    except Exception as exc:
        print(f'[Config][WARN] failed to save resolved config: {exc}')

    return SimpleNamespace(
        model=model,
        loss_fn=loss_fn,
        trainer=trainer,
        bundle_json_path=bundle_json_path,
        resolved_config=resolved_config,
    )


def _run_postfit_validation_and_export(
    train_ctx: Any,
    train_data: Any,
    build_artifacts: Any,
    best_ckpt: Optional[str],
) -> None:
    args = train_ctx.args
    trainer = build_artifacts.trainer
    model = build_artifacts.model

    try:
        vloader = train_data.train_loader
        monitor_batches = int(args.monitor_batches or 8)
        metrics = trainer.validate_autoreg_online(vloader, max_batches=monitor_batches)
        print(
            f"[ValFree@last] "
            f"GeoDeg={metrics.get('GeoDeg', float('nan')):.3f} "
            f"RootVelMAE={metrics.get('RootVelMAE', float('nan')):.5f} "
            f"AngVelMAE={metrics.get('AngVelMAE', float('nan')):.5f} rad/s"
        )
    except Exception as err:
        print(f'[ValFree] skipped due to error: {err}')
        import traceback

        traceback.print_exc()
        try:
            if best_ckpt:
                ckpt_path = Path(best_ckpt).expanduser()
                if ckpt_path.is_file():
                    ckpt = torch.load(str(ckpt_path), map_location=train_ctx.device)
                    missing, unexpected = model.load_state_dict(ckpt['model'], strict=True)
                    assert not missing and (not unexpected), f'state_dict mismatch: missing={missing}, unexpected={unexpected}'
                    print(f'[Load] best checkpoint loaded: {ckpt_path}')
                else:
                    print(f'[WARN] checkpoint not found: {ckpt_path}')
        except Exception as load_err:
            print(f'[Load][WARN] failed to load best ckpt: {load_err}')
    finally:
        _export_postfit_onnx(train_ctx, train_data, model)


def _export_postfit_onnx(
    train_ctx: Any,
    train_data: Any,
    model: torch.nn.Module,
) -> None:
    print('[Export][ENTER] preparing to export ONNX...')
    try:
        import traceback

        model_to_export = model.eval().cpu()
        onnx_path = os.path.join(str(train_ctx.out_dir), f'{train_ctx.run_name}_step_stateful_nophase.onnx')

        try:
            batch = next(iter(train_data.train_loader))
            dx = int(batch['motion'].shape[-1])
            dy = int(batch['gt_motion'].shape[-1])
            dc = int(batch['cond_in'].shape[-1]) if 'cond_in' in batch else 0
            print(f'[Export][ProbeDims] Dx={dx} Dy={dy} Dc={dc}')
            try:
                sanity_check_model_dims(model_to_export, dx, dy, dc)
                print('[Export][Sanity] input dims check OK')
            except Exception as sanity_err:
                print('[Export][Sanity][WARN]', sanity_err)
        except Exception as probe_err:
            print('[Export][ProbeDims][WARN] cannot read a batch for dim probe:', probe_err)

        os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
        export_onnx_step_stateful_nophase(
            model_to_export,
            train_data.train_loader,
            onnx_path,
            opset=18,
            dynamic_batch=False,
        )
    except Exception as export_err:
        print('[Export][ERROR]', export_err)
        traceback.print_exc()

def train_entry():
    train_ctx = _build_train_components()
    train_data = _build_train_loaders(train_ctx)
    model_artifacts = _build_train_model(train_ctx, train_data)
    _prepare_train_model_runtime(train_ctx, train_data, model_artifacts)
    build_artifacts = _build_train_loss_and_trainer(train_ctx, train_data, model_artifacts)
    args = train_ctx.args

    bundle_path = _arg('bundle_json', None)
    apply_layout_center(train_data.ds_train, build_artifacts.trainer, bundle_path)
    runtime_cfg = _resolve_trainer_runtime_config(
        args=args,
        trainer=build_artifacts.trainer,
        ds_train=train_data.ds_train,
        norm_template_path=train_ctx.norm_template_path,
        bundle_json_path=build_artifacts.bundle_json_path,
        out_dir=train_ctx.out_dir,
        resolved_config=build_artifacts.resolved_config,
        run_name=train_ctx.run_name,
    )
    _apply_trainer_runtime_config(build_artifacts.trainer, runtime_cfg)
    build_artifacts.loss_fn.mu_y = getattr(build_artifacts.trainer, 'mu_y', None)
    build_artifacts.loss_fn.std_y = getattr(build_artifacts.trainer, 'std_y', None)
    if getattr(build_artifacts.trainer, '_bundle_meta', None):
        try:
            build_artifacts.loss_fn.meta = dict(build_artifacts.trainer._bundle_meta)
        except (TypeError, ValueError, RuntimeError) as exc:
            _phasec_warn_once(
                "train_entry/loss_meta_from_bundle",
                "failed to copy bundle meta onto loss_fn.meta",
                exc,
            )

    _norm_debug_once(
        build_artifacts.trainer,
        train_data.train_loader,
        thr=float(args.diag_thr),
        topk=int(args.diag_topk),
        print_to_console=False,
    )

    best_ckpt, _history = build_artifacts.trainer.fit(
        train_data.train_loader,
        epochs=args.epochs,
        log_every=args.log_every,
        out_dir=str(train_ctx.out_dir),
        patience=args.patience,
        run_name=train_ctx.run_name,
    )
    _run_postfit_validation_and_export(train_ctx, train_data, build_artifacts, best_ckpt)
@torch.no_grad()
def export_onnx_step_stateful_nophase(model: torch.nn.Module, loader, onnx_path: str, opset: int = 18, dynamic_batch: bool = False):
    """
    单步（无隐式状态）ONNX 导出：
      输入:  state[B,Dx], cond[B,Dc], contacts[B,C], angvel[B,A], pose_hist[B,P],
            plan_z[B,Hp], phase_z[B,Hz]
      输出:  motion_pred[B,Dy], plan_z_next[B,Hp], phase_z_next[B,Hz]

    训练与推理均使用显式历史缓冲，对应 UE 中的 PoseHistoryBuffer。
    """
    import os, torch

    if loader is None:
        raise ValueError('loader is None；需要 DataLoader 来获取示例形状。')

    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        if batch and isinstance(batch[0], dict):
            batch = batch[0]
        else:
            tmp = {}
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                tmp['motion'] = batch[0]
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                tmp['gt_motion'] = batch[1]
            if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                tmp['cond_in'] = batch[2]
            batch = tmp
    if not isinstance(batch, dict):
        raise TypeError("DataLoader 必须返回 dict 才能导出 ONNX。")

    def _pick(*keys):
        for k in keys:
            v = batch.get(k)
            if v is not None:
                return v
        return None

    state_seq = _pick('motion', 'X', 'x_in_features')
    if state_seq is None:
        raise KeyError("Batch 缺少输入 X：需要 'motion' 或同义键。")

    try:
        shape_dbg = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
        print('[Export][BatchShapes]', shape_dbg)
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        _phasec_warn_once(
            "onnx_export/batch_shape_debug",
            "failed to print ONNX export batch shape debug info",
            exc,
        )

    cond_seq = _pick('cond_in', 'C', 'conditions')
    contacts_seq = _pick('contacts', 'soft_contact', 'contacts_in')
    angvel_seq = _pick('angvel', 'angular_velocity', 'angvel_in')
    pose_hist_seq = _pick('pose_hist', 'pose_history')

    state_seq = state_seq.to(torch.float32)
    if state_seq.dim() == 3:
        _, _, Dx = state_seq.shape
        state0 = state_seq[:1, 0, :].contiguous()
    elif state_seq.dim() == 2:
        Dx = state_seq.shape[-1]
        state0 = state_seq[:1, :].contiguous()
    else:
        raise ValueError(f'Unexpected X shape: {tuple(state_seq.shape)}')

    def _frame_or_zero(tensor, dim, dtype):
        if tensor is None:
            return torch.zeros((1, dim), dtype=dtype) if dim > 0 else torch.zeros((1, 0), dtype=dtype)
        tensor = tensor.to(dtype)
        if tensor.dim() == 3:
            return tensor[:1, 0, :dim].contiguous()
        if tensor.dim() == 2:
            return tensor[:1, :dim].contiguous()
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)[:, :dim].contiguous()
        raise ValueError(f'Unexpected tensor shape: {tuple(tensor.shape)}')

    cond_dim = int(getattr(model, 'cond_dim', cond_seq.shape[-1] if isinstance(cond_seq, torch.Tensor) else 0))
    contact_dim = int(getattr(model, 'contact_dim', contacts_seq.shape[-1] if isinstance(contacts_seq, torch.Tensor) else 0))
    angvel_dim = int(getattr(model, 'angvel_dim', angvel_seq.shape[-1] if isinstance(angvel_seq, torch.Tensor) else 0))
    pose_hist_dim = int(getattr(model, 'pose_hist_dim', pose_hist_seq.shape[-1] if isinstance(pose_hist_seq, torch.Tensor) else 0))
    plan_dim = int(getattr(model, 'contact_plan_hidden', 0) or 0) if bool(getattr(model, 'contact_plan_enable', False)) else 0
    phase_dim = int(getattr(model, '_contact_phase_state_dim', 0) or 0) if bool(getattr(model, 'contact_phase_state_enable', False)) else 0

    cond0 = _frame_or_zero(cond_seq, cond_dim, torch.float32)
    contacts0 = _frame_or_zero(contacts_seq, contact_dim, torch.float32)
    angvel0 = _frame_or_zero(angvel_seq, angvel_dim, torch.float32)
    pose_hist0 = _frame_or_zero(pose_hist_seq, pose_hist_dim, torch.float32)
    plan_z0 = torch.zeros((1, plan_dim), dtype=torch.float32) if plan_dim > 0 else None
    phase_z0 = torch.zeros((1, phase_dim), dtype=torch.float32) if phase_dim > 0 else None

    device = torch.device('cpu')
    model = model.to(device).eval()

    if plan_dim <= 0 or phase_dim <= 0 or plan_z0 is None or phase_z0 is None:
        raise RuntimeError(
            f"[Export][FATAL] ONNX export expects fixed mainchain contract plan_z+phase_z, "
            f"got plan_dim={plan_dim}, phase_dim={phase_dim}."
        )

    class _StatelessWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, state, cond, contacts, angvel, pose_hist, plan_z, phase_z):
            cond_in = cond if cond.shape[-1] > 0 else None
            contacts_in = contacts if contacts.shape[-1] > 0 else None
            angvel_in = angvel if angvel.shape[-1] > 0 else None
            pose_hist_in = pose_hist if pose_hist.shape[-1] > 0 else None
            plan_z_in = plan_z if plan_z.shape[-1] > 0 else None
            phase_z_in = phase_z if phase_z.shape[-1] > 0 else None
            out = self.core(
                state,
                cond_in,
                contacts=contacts_in,
                angvel=angvel_in,
                pose_history=pose_hist_in,
                plan_z=plan_z_in,
                phase_z=phase_z_in,
            )
            if isinstance(out, dict):
                pred = out.get('out')
                if pred is None:
                    raise RuntimeError("Model dict output missing 'out'.")
                z_next = out.get('plan_z_next')
                if z_next is None:
                    z_next = plan_z.new_zeros(plan_z.shape)
                p_next = out.get('phase_z_next')
                if p_next is None:
                    p_next = phase_z.new_zeros(phase_z.shape)
                return pred, z_next, p_next
            return out, plan_z, phase_z

    wrapper = _StatelessWrapper(model).cpu().eval()
    sample_out = wrapper(state0, cond0, contacts0, angvel0, pose_hist0, plan_z0, phase_z0)
    Dy = int(sample_out[0].shape[-1])

    inputs = (state0, cond0, contacts0, angvel0, pose_hist0, plan_z0, phase_z0)
    input_names = ['state', 'cond', 'contacts', 'angvel', 'pose_hist', 'plan_z', 'phase_z']
    output_names = ['motion_pred', 'plan_z_next', 'phase_z_next']
    dynamic_axes = {name: {0: 'B'} for name in input_names + output_names} if dynamic_batch else None

    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        f=onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(
        f'[Export][OK] saved: {onnx_path} | '
        f'Dx={Dx} Dy={Dy} Dc={cond_dim} C={contact_dim} A={angvel_dim} P={pose_hist_dim} '
        f'Hp={plan_dim} Hz={phase_dim}'
    )

def main():
    """
    包装器主函数：
    1. 首先执行预分析并生成 bundle.json。
    2. 然后调用真正的训练函数 train_entry()。
    """
    argv0 = sys.argv[:]
    is_export_only = '--arpg_export_only' in argv0
    if is_export_only:
        rest_argv = [arg for arg in argv0 if arg != '--arpg_export_only']
    else:
        rest_argv = argv0
    out_dir_arg = get_flag_value_from_argv(rest_argv, '--out') or get_flag_value_from_argv(rest_argv, '-o')
    run_name_arg = get_flag_value_from_argv(rest_argv, '--run_name')
    out_dir = out_dir_arg or './runs'
    run_name = run_name_arg or time.strftime('%Y%m%d-%H%M%S')
    train_files_flag = get_flag_values_from_argv(rest_argv, '--train_files')
    data_dir_flag = get_flag_value_from_argv(rest_argv, '--data')
    train_files = expand_paths_from_specs(train_files_flag)
    if not train_files and data_dir_flag and os.path.isdir(os.path.expanduser(data_dir_flag)):
        train_files = expand_paths_from_specs([data_dir_flag])


    from types import SimpleNamespace
    global GLOBAL_ARGS
    GLOBAL_ARGS = SimpleNamespace(out=out_dir, run_name=run_name, allow_val_on_train='--allow_val_on_train' in rest_argv, val_ratio=float(get_flag_value_from_argv(rest_argv, '--val_ratio') or 0))
    set_global_args(GLOBAL_ARGS)
    print(f'[ARPG-PATCH] 参数准备完毕，即将进入训练入口: train_entry()')
    sys.argv = rest_argv
    try:
        train_entry()
    finally:
        sys.argv = argv0
if __name__ == '__main__':
    main()
