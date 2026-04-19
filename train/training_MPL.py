from __future__ import annotations


# ===== Common Helpers (extracted) =====

# ========== [Unified Geometry Utilities] ==========
import argparse
import ast
import glob
import json
import math as _math
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Mapping, Sequence, List, Tuple

from .eval_utils import evaluate_teacher
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
    root_yaw_from_rot6d_torch,
    wrap_to_pi_np as _wrap_to_pi_np,
    wrap_to_pi_torch,
    gram_schmidt_renorm_np,
    _apply_so3_correction_to_delta_raw,
)
from .data.layout import (
    normalize_layout as _normalize_layout,
    layout_span as _layout_span,
)
from .configuration.norm_spec import (
    ContactPretrainRuntime,
    merge_norm_spec,
    parse_pretrain_contact_affine_spec,
    resolve_contact_pretrain_runtime,
)
from .data.dataset import (
    MotionEventDataset,
    ClipData,
    DatasetRuntimeArtifacts,
    make_fixedlen_collate,
    _infer_forward_axis_from_clip,
    build_and_attach_dataset_runtime,
    build_motion_dataloader,
    build_motion_dataset,
)
from .diagnostics import (
    _maybe_optimize_dataset_index,
    _norm_debug_once,
    _parse_stage_schedule,
    collect_direct_pose_grad_stats,
    dump_nan_grad_report,
    history_drift_debug,
    test_gradient_connection,
)
from .data.io import (
    load_soft_contacts_from_json as _load_soft_contacts_from_json,
    json_safe as _json_safe,
    write_json_payload as _write_json_payload_io,
    direction_yaw_from_array as _direction_yaw_from_array,
    velocity_yaw_from_array as _velocity_yaw_from_array,
    speed_from_X_layout as _speed_from_X_layout,
    npz_scalar_to_str as _npz_scalar_to_str,
)
from .utils import (
    _build_pretrain_contact_encoder_input,
    build_mlp,
    expand_paths_from_specs,
    _guard_first_linear_finite_,
    safe_int_scalar,
    validate_and_fix_model_,
    sanity_check_model_dims,
    set_global_args,
    get_global_arg,
    pick_first_present,
    warn_once,
)
from .history import attach_adaptive_history_runtime
from . import rollout_kernel as _rollout_kernel
from .rollout_kernel import (
    RolloutExecutionState,
    RolloutSequenceInputs,
    RolloutStepInputs,
)
from .data.normalizers import normalize_cond_tensor, prepare_runtime_stat_tensor
from .runtime_attach import (
    SharedTrainerRuntime,
    apply_contacts_pretrain_runtime,
    apply_loss_runtime_from_trainer,
    apply_shared_trainer_runtime,
    resolve_shared_trainer_runtime,
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
from .checkpoint.compat import resume_load_weights_compat as _resume_load_weights_compat
from .checkpoint.compat import attach_motion_encoder_bundle as _attach_motion_encoder_bundle


_arg = get_global_arg
_STATE_UPDATE_UNSET = object()


@dataclass(frozen=True)
class TrainEpochResult:
    avg_train: float
    train_metrics: Dict[str, Any]


@dataclass
class FitEpochValidationResult:
    teacher_metrics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ValidationRuntimeContext:
    epoch: int
    tf_ratio: float
    teacher_eval_max_batches: Optional[int]


@dataclass
class FitCheckpointState:
    last_payload: Optional[Dict[str, Any]] = None
    last_ckpt: Optional[str] = None
_PHASEC_WARN_ONCE_KEYS: set[str] = set()


def _phasec_warn_once(
    key: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> None:
    warn_once(
        _PHASEC_WARN_ONCE_KEYS,
        category="PhaseC",
        key=key,
        message=message,
        exc=exc,
    )


LEGACY_LOSS_KEYS: tuple[str, ...] = (
    "ignore_motion_groups",
    "bone_prior_stds",
    "use_hierarchy_weights",
    "hierarchy_mode",
    "hierarchy_alpha",
    "max_weight_ratio",
    "weight_gamma",
)

def _legacy_loss_keys_msg(keys: Sequence[str], *, context: str) -> str:
    keys_sorted = ", ".join(sorted({str(k) for k in keys}))
    return (
        f"[LegacyLossConfig] {context} contains removed keys: {keys_sorted}. "
        "Please remove them; MotionJointLoss now uses unified weights only "
        "(unified_* knobs)."
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
    def __init__(self, model, loss_fn, lr=0.0001, grad_clip=0.0, weight_decay=0.01, use_amp=None, accum_steps=1, *, pin_memory=False, args=None):
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
        self.contacts_pretrain_runtime_attached: bool = False
        self.nan_grad_reports: int = 0
        self.nan_grad_report_limit: int = 5
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
        self.hyperparam_scheduler: Optional[Any] = None
        self.teacher_forcing_ratio: float = 1.0
        # 自由运行时根部 yaw 的参考策略：
        #   - 'trajectory': 使用 cond_dir 定义世界/轨迹坐标系的 yaw（推荐）
        #   - 'skeleton' : 使用骨骼(pelvis)推断 yaw（旧行为，可能导致坐标系随误差漂移）
        self.freerun_yaw_strategy: str = str(
            getattr(args, 'freerun_yaw_strategy', _arg('freerun_yaw_strategy', 'trajectory')) or 'trajectory'
        )
        # Dedicated, unbounded summary state for durable basetrain panel artifacts.
        self._basetrain_summary_teacher_rows: Dict[int, Dict[str, Any]] = {}
        self._basetrain_summary_train_direct_rows: Dict[int, Dict[str, Any]] = {}

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

    def _pick_first(self, batch, keys):
        return pick_first_present(batch, keys)

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

    def _diag_norm_x(self, x_raw, mu_x=None, std_x=None):
        # 仅使用 DataNormalizer；缺失即视为致命错误
        self._require_normalizer("Trainer._diag_norm_x")
        try:
            return self.normalizer.norm(x_raw)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm 在 _diag_norm_x 中失败", exc)

    def _denorm(self, y):
        # 仅使用 DataNormalizer；缺失或异常直接终止
        self._require_normalizer("Trainer._denorm")
        try:
            return self.normalizer.denorm(y)
        except Exception as exc:
            self._raise_norm_error("normalizer.denorm 在 _denorm 中失败", exc)

    def _norm_y(self, y_raw):
        self._require_normalizer("Trainer._norm_y")
        try:
            return self.normalizer.norm_y(y_raw)
        except AttributeError as exc:
            self._raise_norm_error("DataNormalizer 缺少 norm_y 方法", exc)
        except Exception as exc:
            self._raise_norm_error("normalizer.norm_y 失败", exc)

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
        return normalize_cond_tensor(
            cond_raw,
            cond_mu,
            cond_std,
            cond_norm_clip=float(getattr(self, 'cond_norm_clip', 6.0) or 0.0),
        )

    def _pose_hist_params(self, ref: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns pose-history normalizer tensors aligned with the reference tensor's device/dtype.
        """
        if self.pose_hist_scales is None or self.pose_hist_dim <= 0:
            return None, None, None
        scales = prepare_runtime_stat_tensor(self.pose_hist_scales, ref_tensor=ref)
        mu = prepare_runtime_stat_tensor(self.pose_hist_mu, ref_tensor=ref)
        std = prepare_runtime_stat_tensor(self.pose_hist_std, ref_tensor=ref)
        return scales, mu, std

    def _infer_root_yaw_from_rot6d(self, y_denorm: "torch.Tensor"):
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
            rot6d = rot_flat.view(rot_flat.shape[0], J, 6)
        except Exception:
            return None
        root_idx = int(getattr(self, 'eval_root_idx', 0))
        up_axis = int(getattr(self, 'eval_up_axis', getattr(self, '_up_axis', 2)))
        forward_axis = int(getattr(self, 'yaw_forward_axis', 2))
        return root_yaw_from_rot6d_torch(
            rot6d,
            forward_axis=forward_axis,
            up_axis=up_axis,
            offset=float(getattr(self, 'yaw_forward_axis_offset', 0.0)),
            root_idx=root_idx,
            reproject=True,
        )

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
        cdim = int(getattr(model, 'contact_dim', 0) or 0)
        in_dim = int(getattr(model, 'encoder_input_dim', 0) or 0)
        if cdim <= 0 or in_dim <= 0:
            return None
        angvel_slice = getattr(self, 'angvel_x_slice', None)
        if not isinstance(angvel_slice, slice):
            angvel_slice = getattr(model, '_contact_meas_state_angvel_slice', None)
        contacts_pretrain_runtime_attached = bool(getattr(self, 'contacts_pretrain_runtime_attached', False))
        if contacts_pretrain_runtime_attached:
            missing = [
                field_name
                for field_name in (
                    'contacts_pretrain_clamp',
                    'contacts_pretrain_affine_stats_spec',
                    'contacts_pretrain_affine',
                )
                if not hasattr(self, field_name)
            ]
            if missing:
                raise RuntimeError(
                    f"contacts_pretrain runtime attached but missing neutral attrs: {', '.join(missing)}"
                )
        pre_clamp_raw = getattr(self, 'contacts_pretrain_clamp', 1.0)
        if contacts_pretrain_runtime_attached and pre_clamp_raw is None:
            raise RuntimeError("contacts_pretrain runtime attached but contacts_pretrain_clamp is None")
        try:
            pre_clamp = float(pre_clamp_raw or 0.0)
        except (TypeError, ValueError):
            pre_clamp = 1.0
        if not (_math.isfinite(float(pre_clamp)) and float(pre_clamp) > 0.0):
            pre_clamp = 0.0
        encoder_input = _build_pretrain_contact_encoder_input(
            motion_step_t,
            pose_hist_step_t,
            contact_dim=cdim,
            encoder_input_dim=in_dim,
            angvel_slice=angvel_slice,
            clamp_val=float(pre_clamp),
        )
        affine_spec = parse_pretrain_contact_affine_spec(getattr(self, 'contacts_pretrain_affine', None))
        try:
            with torch.no_grad():
                hidden = enc(encoder_input.unsqueeze(1), return_summary=False)
                logits = head(hidden)
        except Exception:
            return None
        if torch.is_tensor(logits) and logits.ndim == 3 and int(logits.size(1)) == 1:
            logits = logits[:, 0]
        if (not torch.is_tensor(logits)) or logits.ndim != 2:
            return None
        if affine_spec is not None and len(affine_spec['scale']) == int(logits.shape[-1]) == len(affine_spec['bias']):
            scale_t = logits.new_tensor(affine_spec['scale']).view(1, int(logits.shape[-1]))
            bias_t = logits.new_tensor(affine_spec['bias']).view(1, int(logits.shape[-1]))
            probs_affine = torch.sigmoid(logits).clamp(affine_spec['eps'], 1.0 - affine_spec['eps'])
            logits = bias_t + scale_t * (torch.log(probs_affine) - torch.log1p(-probs_affine))
        probs = torch.sigmoid(logits)
        if int(probs.shape[-1]) != cdim:
            if int(probs.shape[-1]) > cdim:
                probs = probs[..., :cdim]
            else:
                probs = F.pad(probs, (0, cdim - int(probs.shape[-1])))
        return probs

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
                std_t = prepare_runtime_stat_tensor(
                    std,
                    ref_tensor=delta_norm,
                    cache=self._norm_cache,
                    cache_key='std_y',
                )
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
        rot_slice = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
        if not isinstance(rot_slice, slice):
            rot_slice = slice(0, y_prev_raw.shape[-1])
        rot_len = int(rot_slice.stop - rot_slice.start)
        if rot_len % 6 != 0:
            self._raise_norm_error(f"compose_delta_to_raw: rot_slice 长度 {rot_len} 不是 6 的倍数。")

        if omega_hat is not None:
            gate_val = so3_gate
            if gate_val is None:
                logit = getattr(self.model, 'so3_corr_gate_logit', None)
                gate_val = float(torch.sigmoid(logit.detach()).item()) if torch.is_tensor(logit) else 0.0
            max_deg = so3_max_deg
            if max_deg is None:
                max_deg = float(getattr(self, 'so3_corr_max_deg', 20.0) or 20.0)
            try:
                delta_raw = _apply_so3_correction_to_delta_raw(
                    delta_raw,
                    rot_slice=rot_slice,
                    omega_hat=omega_hat,
                    gate_val=float(gate_val or 0.0),
                    max_deg=float(max_deg or 0.0),
                    columns=getattr(self.loss_fn, '_rot6d_columns', ("X", "Z")),
                    omega_detach=bool(omega_detach),
                )
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

    def _mul_lambda_reliability(
        self,
        lhs: Optional[torch.Tensor],
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        if lhs is None:
            return rhs
        if lhs.dim() == 1 and rhs.dim() == 2:
            lhs = lhs.unsqueeze(-1)
        elif lhs.dim() == 2 and rhs.dim() == 1:
            rhs = rhs.unsqueeze(-1)
        return lhs * rhs

    def _resolve_lambda_warmup_reliability(
        self,
        lam: torch.Tensor,
        *,
        step_idx: Optional[int],
    ) -> Optional[torch.Tensor]:
        warmup_steps = int(getattr(self, "lambda_reliability_warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return None
        batch_size = int(lam.shape[0])
        base_ratio = float(max(0, int(step_idx or 0))) / float(max(1, warmup_steps - 1))
        scalar_ratio = max(0.0, min(1.0, base_ratio))
        joint_scales = getattr(self, "lambda_reliability_warmup_joint_scales", None)
        joint_count = int(lam.shape[-1]) if lam.dim() >= 2 else 0
        if joint_scales is None or joint_count <= 0:
            return torch.full((batch_size,), scalar_ratio, device=lam.device, dtype=lam.dtype)
        try:
            if not torch.is_tensor(joint_scales):
                joint_scales = torch.as_tensor(joint_scales, device=lam.device, dtype=lam.dtype)
            joint_scales_t = joint_scales.to(device=lam.device, dtype=lam.dtype).reshape(-1)
            if int(joint_scales_t.numel()) != joint_count:
                return torch.full((batch_size,), scalar_ratio, device=lam.device, dtype=lam.dtype)
            base = torch.full((batch_size, 1), base_ratio, device=lam.device, dtype=lam.dtype)
            return (base * joint_scales_t.view(1, joint_count)).clamp(0.0, 1.0)
        except (RuntimeError, TypeError, ValueError) as exc:
            _phasec_warn_once(
                "lambda_reliability/warmup_joint_scales",
                "invalid warmup joint scales; fallback to scalar warmup reliability",
                exc,
            )
            return torch.full((batch_size,), scalar_ratio, device=lam.device, dtype=lam.dtype)

    def _resolve_lambda_contact_err_reliability(
        self,
        lam: torch.Tensor,
        *,
        ret: Optional[dict],
    ) -> Optional[torch.Tensor]:
        err = ret.get("contacts_err", None) if isinstance(ret, dict) else None
        if not torch.is_tensor(err):
            return None
        try:
            if err.dim() == 3:
                err = err[:, 0] if err.size(1) == 1 else err[:, -1]
            if err.dim() != 2 or err.shape[0] != int(lam.shape[0]):
                return None
            err_max = float(getattr(self, "lambda_reliability_contact_err_max", 1.0) or 1.0)
            if err_max <= 1e-8:
                return None
            err_abs_mean = err.detach().abs().mean(dim=-1)
            return (1.0 - err_abs_mean / err_max).clamp(0.0, 1.0).to(dtype=lam.dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            _phasec_warn_once(
                "lambda_reliability/contacts_err",
                "contacts_err reliability term failed; using warmup-only reliability path",
                exc,
            )
            return None

    def _broadcast_lambda_reliability(self, lam: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        batch_size = int(lam.shape[0])
        try:
            if reliability.dim() == 1:
                return reliability.view((batch_size,) + (1,) * max(0, int(lam.dim()) - 1))
            if reliability.dim() == 2 and lam.dim() >= 2:
                joint_count = int(reliability.shape[1])
                return reliability.view((batch_size,) + (1,) * max(0, int(lam.dim()) - 2) + (joint_count,))
        except (RuntimeError, TypeError, ValueError):
            pass
        return reliability

    def _lambda_fusion_apply_reliability(
        self,
        lambda_fusion,
        *,
        step_idx: Optional[int] = None,
        total_steps: Optional[int] = None,
        rollout_step=None,
        ret: Optional[dict] = None,
    ):
        _ = total_steps, rollout_step
        lam = lambda_fusion
        if lam is None or (not torch.is_tensor(lam)):
            return lambda_fusion, None
        mode = str(getattr(self, "lambda_reliability_mode", "none") or "none").strip().lower()
        if mode in ("", "none", "off", "false", "0", "disable", "disabled") or lam.dim() <= 0 or int(lam.shape[0]) <= 0:
            return lam, None
        tokens = [token.strip() for token in mode.replace(",", "+").split("+") if token.strip()]
        if not tokens:
            return lam, None

        reliability = None
        if "warmup" in tokens or "step_warmup" in tokens:
            warmup_rel = self._resolve_lambda_warmup_reliability(lam, step_idx=step_idx)
            if warmup_rel is not None:
                reliability = self._mul_lambda_reliability(reliability, warmup_rel)
        if "contacts_err" in tokens or "contact_err" in tokens:
            contact_rel = self._resolve_lambda_contact_err_reliability(lam, ret=ret)
            if contact_rel is not None:
                reliability = self._mul_lambda_reliability(reliability, contact_rel)
        if reliability is None:
            return lam, None

        reliability = reliability.clamp(0.0, 1.0)
        lam_eff = (lam * self._broadcast_lambda_reliability(lam, reliability)).clamp(0.0, 1.0)
        return lam_eff, reliability.detach()

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
        delta_yaw = wrap_to_pi_torch(delta_yaw)  # 归一化到 [-π, π]

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

    def _resolve_rollout_gt_yaw(
        self,
        rollout: RolloutExecutionState,
        rollout_inputs: RolloutSequenceInputs,
        *,
        step_idx: int,
    ) -> Optional[torch.Tensor]:
        if rollout_inputs.gt_seq is not None and rollout.has_time_dim.get('cond_raw'):
            gt_idx = min(rollout_inputs.gt_seq.shape[1] - 1, int(step_idx))
            return self._infer_root_yaw_from_rot6d(self._denorm(rollout_inputs.gt_seq[:, gt_idx]))
        if rollout_inputs.state_seq is not None:
            state_raw = self.normalizer.denorm_x(
                rollout_inputs.state_seq[:, int(step_idx)],
                prev_raw=rollout.motion_raw_local,
            )
            return self._infer_root_yaw_from_rot6d(state_raw)
        return None

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

    def _update_rollout_step_aux_state(
        self,
        rollout: RolloutExecutionState,
        step_inputs: RolloutStepInputs,
        ret: Mapping[str, Any],
    ) -> None:
        rollout.prev_foot_pos_meas = step_inputs.prev_foot_pos_meas
        if step_inputs.reprojection_applied:
            rollout.reprojection_applied_count += 1
        rollout.last_attn = ret.get('attn', rollout.last_attn)
        _rollout_kernel.update_rollout_plan_state(rollout, ret)
        meas_prob_step = ret.get('contacts_meas', None)
        if torch.is_tensor(meas_prob_step):
            rollout.meas_prev_prob = meas_prob_step.detach()

    def _should_apply_rollout_lambda_fusion(self, rollout: RolloutExecutionState) -> bool:
        if not bool(getattr(self, 'lambda_fusion_apply', False)):
            return False
        if rollout.mode not in ('free', 'train_free', 'mixed'):
            return False
        if rollout.mode != 'mixed':
            return True
        try:
            return float(rollout.tf_ratio) < 0.999
        except (TypeError, ValueError):
            return False

    def _maybe_apply_rollout_lambda_fusion(
        self,
        rollout: RolloutExecutionState,
        *,
        step_idx: int,
        step_inputs: RolloutStepInputs,
        ret: Mapping[str, Any],
        y_raw: torch.Tensor,
    ) -> torch.Tensor:
        if not self._should_apply_rollout_lambda_fusion(rollout):
            return y_raw
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
                except (RuntimeError, TypeError, ValueError, AttributeError, KeyError):
                    lam_eff = ret.get('lambda_fusion', None)
            return self._apply_lambda_fusion_to_raw(
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
            return y_raw

    def _compose_rollout_step_raw(
        self,
        rollout: RolloutExecutionState,
        *,
        step_idx: int,
        step_inputs: RolloutStepInputs,
        ret: Mapping[str, Any],
        delta_out: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        prev_raw_snapshot = rollout.y_raw_local.clone() if rollout.y_raw_local is not None else None
        y_raw = self._compose_delta_to_raw(rollout.y_raw_local, delta_out)
        if y_raw is None:
            self._raise_norm_error('compose_delta_to_raw 返回 None，缺少上一帧 RAW 数据。')
        y_raw = self._maybe_apply_rollout_lambda_fusion(
            rollout,
            step_idx=step_idx,
            step_inputs=step_inputs,
            ret=ret,
            y_raw=y_raw,
        )
        return prev_raw_snapshot, y_raw

    def _rollout_forward_step(
        self,
        rollout: RolloutExecutionState,
        rollout_inputs: RolloutSequenceInputs,
        *,
        step_idx: int,
    ) -> Dict[str, Optional[float]]:
        step_inputs = _rollout_kernel.resolve_rollout_step_inputs(
            self,
            rollout,
            rollout_inputs,
            step_idx=step_idx,
            yaw_gt_fn=lambda idx: self._resolve_rollout_gt_yaw(
                rollout,
                rollout_inputs,
                step_idx=int(idx),
            ),
            model=self.model,
        )
        with self._amp_context(rollout.amp_enabled):
            ret, delta_out, period_pred = _rollout_kernel.forward_rollout_model_step(
                self.model,
                motion=rollout.motion,
                cond_input=step_inputs.cond_input,
                contacts_in_t=step_inputs.contacts_in_t,
                angvel_t=step_inputs.angvel_t,
                pose_history_t=step_inputs.pose_history_t,
                plan_z=rollout.plan_z,
                meas_logits_prev=rollout.meas_prev_prob,
                time_index_t=step_inputs.time_index_t,
                rollout_step_t=step_inputs.rollout_step_t,
            )
        self._update_rollout_step_aux_state(rollout, step_inputs, ret)
        prev_raw_snapshot, y_raw = self._compose_rollout_step_raw(
            rollout,
            step_idx=step_idx,
            step_inputs=step_inputs,
            ret=ret,
            delta_out=delta_out,
        )
        rollout.y_raw_local = y_raw.clone() if rollout.allow_grad else y_raw.detach()
        y_norm = self._norm_y(y_raw)
        _rollout_kernel.record_rollout_step_outputs(
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

        _rollout_kernel.update_rollout_carry_state(
            self,
            rollout,
            rollout_inputs,
            step_idx=step_idx,
            warn_once_fn=_phasec_warn_once,
        )

    def _maybe_log_rollout_reprojection(self, rollout: RolloutExecutionState) -> None:
        if not (rollout.enable_reprojection and rollout.reprojection_applied_count > 0):
            return
        diag_limit = int(getattr(self, '_reprojection_diag_limit', 3))
        if not hasattr(self, '_reprojection_diag_count'):
            self._reprojection_diag_count = 0
        if self._reprojection_diag_count >= diag_limit:
            return
        epoch = getattr(self, 'cur_epoch', -1)
        print(
            f"[CondReprojection] Epoch {epoch}, Mode '{rollout.mode}': "
            f"Applied reprojection to {rollout.reprojection_applied_count}/{rollout.total_steps} steps"
        )
        self._reprojection_diag_count += 1

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
        rollout = _rollout_kernel.init_rollout_execution_state(
            self,
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
            preds = _rollout_kernel.finalize_rollout_outputs(rollout)
            self._maybe_log_rollout_reprojection(rollout)
            return preds, rollout.last_attn
        finally:
            self._commit_rollout_diag_update(mode=None, step=-1)

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
        tags = goal.get('tags') or ['teacher']
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
        if epoch == 1 and batch_idx == 1:
            if isinstance(stats, Mapping):
                cp_bce = stats.get('contact_plan_bce', None)
                if cp_bce is not None:
                    print(f'[Smoke] contact_plan_bce={cp_bce}')

        if getattr(self, 'history_debug_steps', 0) > 1 and batch_idx == 1:
            try:
                history_drift_debug(
                    self,
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

    def _backward_train_loss(
        self,
        loss: torch.Tensor,
        *,
        scaler: torch.amp.GradScaler,
        accum_steps: int,
    ) -> None:
        scaler.scale(loss / accum_steps).backward()

    def _prepare_train_optimizer_step(self, *, scaler: torch.amp.GradScaler) -> None:
        scaler.unscale_(self.optimizer)

    def _inject_direct_grad_monitor_stats(self, stats: Dict[str, Any]) -> None:
        if not (bool(getattr(self, 'direct_pose_grad_monitor_enable', False)) and isinstance(stats, dict)):
            return
        try:
            stats.update(collect_direct_pose_grad_stats(self))
        except Exception as exc:
            print(f'[DirectGrad][WARN] failed to collect grad stats: {exc}')

    def _handle_nonfinite_train_gradients(
        self,
        *,
        loss: torch.Tensor,
        stats: Dict[str, Any],
        preds_dict: Dict[str, Any],
        state_seq: torch.Tensor,
        gt_seq: torch.Tensor,
        batch: Any,
        epoch: int,
        batch_idx: int,
        log_every: int,
        scaler: torch.amp.GradScaler,
    ) -> bool:
        any_bad_grad = False
        bad_names: List[str] = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if torch.isfinite(param.grad).all():
                continue
            any_bad_grad = True
            if len(bad_names) < 3:
                bad_names.append(name)
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

        if not any_bad_grad:
            return False

        try:
            loss_val = float(loss.detach().cpu())
        except Exception:
            loss_val = float('nan')
        dump_nan_grad_report(self, epoch, batch_idx, batch, state_seq, gt_seq, preds_dict, loss_val, stats)
        if log_every:
            print(f"[Guard][Grad] non-finite grads on {', '.join(bad_names)} ... skip optimizer.step()")
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _apply_train_optimizer_step(
        self,
        loss: torch.Tensor,
        *,
        epoch: int,
        batch_idx: int,
        log_every: int,
        scaler: torch.amp.GradScaler,
        tf_ratio_local: float,
    ) -> float:
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
        return float(tf_ratio_local)

    def _post_train_optimizer_step(self, *, log_every: int) -> None:
        params_finite = True
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                if torch.isfinite(param).all():
                    continue
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
        if lr_scheduler is None:
            return
        try:
            lr_scheduler.step()
        except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
            _phasec_warn_once(
                "fit/lr_scheduler_step",
                "lr_scheduler.step() failed; keeping optimizer state unchanged",
                exc,
            )

    def _maybe_run_train_optimizer_step(
        self,
        *,
        batch: Any,
        loss: torch.Tensor,
        stats: Dict[str, Any],
        preds_dict: Dict[str, Any],
        state_seq: torch.Tensor,
        gt_seq: torch.Tensor,
        epoch: int,
        batch_idx: int,
        log_every: int,
        scaler: torch.amp.GradScaler,
        tf_ratio_local: float,
    ) -> Tuple[bool, float]:
        self._prepare_train_optimizer_step(scaler=scaler)
        self._inject_direct_grad_monitor_stats(stats)
        if self._handle_nonfinite_train_gradients(
            loss=loss,
            stats=stats,
            preds_dict=preds_dict,
            state_seq=state_seq,
            gt_seq=gt_seq,
            batch=batch,
            epoch=epoch,
            batch_idx=batch_idx,
            log_every=log_every,
            scaler=scaler,
        ):
            return True, float(tf_ratio_local)

        tf_ratio_local = self._apply_train_optimizer_step(
            loss,
            epoch=epoch,
            batch_idx=batch_idx,
            log_every=log_every,
            scaler=scaler,
            tf_ratio_local=tf_ratio_local,
        )
        self._post_train_optimizer_step(log_every=log_every)
        return False, float(tf_ratio_local)

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
            self._backward_train_loss(loss, scaler=scaler, accum_steps=accum_steps)

            if (batch_idx + 1) % accum_steps == 0:
                skip_batch, tf_ratio_local = self._maybe_run_train_optimizer_step(
                    batch=batch,
                    loss=loss,
                    stats=stats,
                    preds_dict=preds_dict,
                    state_seq=state_seq,
                    gt_seq=gt_seq,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    log_every=log_every,
                    scaler=scaler,
                    tf_ratio_local=tf_ratio_local,
                )
                if skip_batch:
                    continue

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

    @torch.no_grad()
    def eval_epoch(self, loader, mode='mixed', max_batches=None):
        self.model.eval()
        return evaluate_teacher(self, loader, mode='mixed', max_batches=max_batches)

    def _build_validation_runtime_context(self, *, epoch: int) -> ValidationRuntimeContext:
        tf_ratio = float(getattr(self, '_last_tf_ratio', 1.0))
        return ValidationRuntimeContext(
            epoch=epoch,
            tf_ratio=tf_ratio,
            teacher_eval_max_batches=getattr(self, 'teacher_eval_max_batches', None),
        )

    @staticmethod
    def _with_validation_phase(metrics: Mapping[str, Any], *, phase: str, tf_ratio: float) -> Dict[str, Any]:
        payload = dict(metrics)
        payload.setdefault('phase', phase)
        payload['tf_ratio'] = float(tf_ratio)
        return payload

    @staticmethod
    def _metric_first(metrics: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                return value
        return None

    def _run_epoch_validation(self, *, epoch: int) -> FitEpochValidationResult:
        context = self._build_validation_runtime_context(epoch=epoch)
        result = FitEpochValidationResult()
        try:
            max_teacher_batches = context.teacher_eval_max_batches
            if max_teacher_batches is not None and int(max_teacher_batches) <= 0:
                cached_batch = getattr(self, '_cached_train_batch', None)
                if cached_batch is not None:
                    teacher_metrics = self._with_validation_phase(
                        dict(self.eval_epoch([cached_batch], mode='teacher', max_batches=1) or {}),
                        phase='teacher',
                        tf_ratio=context.tf_ratio,
                    )
                    print(f'[ValTeacher@ep {context.epoch:03d}] cached-batch eval (no extra loader pass)')
                else:
                    teacher_metrics = None
                    print(f'[ValTeacher@ep {context.epoch:03d}] skipped: no cached batch available (teacher_eval_max_batches<=0)')
            else:
                teacher_metrics = self._with_validation_phase(
                    dict(self.eval_epoch(self.train_loader, mode='teacher', max_batches=max_teacher_batches) or {}),
                    phase='teacher',
                    tf_ratio=context.tf_ratio,
                )
            if isinstance(teacher_metrics, dict):
                teacher_metrics.setdefault('phase', 'teacher')
                try:
                    plateau_scheduler = getattr(self, 'lr_plateau_scheduler', None)
                    keybone_mean = None if plateau_scheduler is None else self._metric_first(teacher_metrics, 'KeyBone/GeoLocalDegMean', 'KeyBone/GeoDegMean', 'GeoLocalDeg', 'GeoDeg')
                    if keybone_mean is not None:
                        plateau_scheduler.step(float(keybone_mean))
                except Exception as plateau_exc:
                    print(f'[LR-Plateau][WARN] scheduler step failed: {plateau_exc}')
                self._save_epoch_diag_snapshot(epoch=context.epoch, phase='teacher', metrics=teacher_metrics)
                result.teacher_metrics = teacher_metrics
                print(
                    f'[ValTeacher@ep {context.epoch:03d}] '
                    f"loss={teacher_metrics.get('loss', float('nan')):.6f} | "
                    f"GeoDeg={teacher_metrics.get('GeoDeg', float('nan')):.3f}° | "
                    f"GeoLocalDeg={teacher_metrics.get('GeoLocalDeg', float('nan')):.3f}° | "
                    f"AngVelMAE={teacher_metrics.get('AngVelMAE', float('nan')):.5f} rad/s | "
                    f"AngMagRel={teacher_metrics.get('AngVelMagRel', float('nan')):.3f}"
                )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f'[ValTeacher@ep {context.epoch:03d}] skipped due to error: {exc}')

        return result

    def _persist_epoch_validation_outputs(self, *, epoch: int, validation_result: FitEpochValidationResult) -> None:
        if validation_result.teacher_metrics is not None:
            self._persist_epoch_metrics_artifacts(validation_result.teacher_metrics, tag='teacher', epoch=epoch)

        try:
            self._write_basetrain_keybone_group_summary()
        except Exception as exc:
            print(f'[MetricsWrite][WARN] failed to update basetrain_keybone_group_summary.json: {exc}')

    def _metrics_json_safe(self, value):
        return _json_safe(value)

    def _write_json_payload(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        warn_prefix: Optional[str] = None,
        warn_message: Optional[str] = None,
    ) -> Optional[Path]:
        return _write_json_payload_io(
            path,
            payload,
            warn_prefix=warn_prefix,
            warn_message=warn_message,
        )

    def _save_epoch_diag_snapshot(self, *, epoch: int, phase: str, metrics: Mapping[str, Any]) -> Optional[Path]:
        base_debug_path = getattr(self, 'freerun_debug_path', None)
        if not base_debug_path:
            return None
        ep_tag = f'ep{epoch:03d}'
        candidate = Path(base_debug_path)
        candidate = candidate / f'{phase}_diag_{ep_tag}.json' if candidate.is_dir() or str(base_debug_path).endswith('/') else candidate.with_name(candidate.stem + f'_{phase}_{ep_tag}.json')
        saved_path = self._write_json_payload(candidate, {'epoch': epoch, 'phase': phase, 'tf_ratio': float(getattr(self, '_last_tf_ratio', 1.0)), 'metrics': metrics}, warn_prefix=f'{phase.title()}Diag', warn_message='failed to save')
        if saved_path is not None:
            print(f'[{phase.title()}Diag] saved to {saved_path}')
        return saved_path

    def _persist_epoch_metrics_artifacts(
        self,
        metrics: Optional[Mapping[str, Any]],
        *,
        tag: str,
        epoch: int,
        finish_stage: bool = True,
    ) -> Optional[Path]:
        if metrics is None:
            return None
        metrics_payload = metrics if isinstance(metrics, dict) else dict(metrics)
        self._update_basetrain_summary_state(metrics_payload, tag=tag, epoch=epoch)
        json_path = self._dump_metrics_json(metrics_payload, tag=tag, epoch=epoch)
        if finish_stage:
            self._maybe_finish_stage(epoch, metrics_payload, tag=tag)
        return json_path

    def _dump_metrics_json(self, metrics: Dict[str, Any], *, tag: str, epoch: int) -> Optional[Path]:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return None
        json_path = Path(out_dir) / 'metrics' / f'{tag}_ep{int(epoch):03d}.json'
        return self._write_json_payload(
            json_path,
            {
                'epoch': int(epoch),
                'tag': str(tag),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                'metrics': self._metrics_json_safe(metrics),
            },
            warn_prefix='MetricsWrite',
            warn_message=f'failed to write {tag} metrics @ep{epoch}',
        )

    @staticmethod
    def _panel_metric_scalar(metrics: Mapping[str, Any], key: str, *, fallback: float = float('nan')) -> float:
        val = metrics.get(key, fallback)
        try:
            return float(val)
        except Exception:
            return float(fallback)

    def _build_basetrain_teacher_summary_row(self, *, epoch: int, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        keybone_summary = metrics.get('KeyBoneSummary', {}) if isinstance(metrics.get('KeyBoneSummary', {}), Mapping) else {}
        geo_local = self._panel_metric_scalar(metrics, 'GeoLocalDeg')
        key_geo_local = self._panel_metric_scalar(
            metrics,
            'KeyBone/GeoLocalDegMean',
            fallback=self._panel_metric_scalar(keybone_summary, 'GeoLocalDegMean'),
        )
        return {
            'epoch': int(epoch),
            'GeoLocalDeg': geo_local,
            'KeyBoneGeoLocalDegMean': key_geo_local,
            'group_mean': keybone_summary.get('group_mean', {}) if isinstance(keybone_summary.get('group_mean', {}), Mapping) else {},
        }

    def _build_basetrain_train_direct_summary_row(self, *, epoch: int, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            'epoch': int(epoch),
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

    def _update_basetrain_summary_state(self, metrics: Mapping[str, Any], *, tag: str, epoch: int) -> None:
        if not isinstance(metrics, Mapping):
            return
        metrics_safe = self._metrics_json_safe(metrics)
        if not isinstance(metrics_safe, Mapping):
            return
        epoch_int = int(epoch)
        if tag == 'teacher':
            self._basetrain_summary_teacher_rows[epoch_int] = self._build_basetrain_teacher_summary_row(
                epoch=epoch_int,
                metrics=metrics_safe,
            )
        elif tag == 'train':
            self._basetrain_summary_train_direct_rows[epoch_int] = self._build_basetrain_train_direct_summary_row(
                epoch=epoch_int,
                metrics=metrics_safe,
            )

    def _write_basetrain_keybone_group_summary(self) -> None:
        out_dir = getattr(self, 'out_dir', None)
        if not out_dir:
            return

        teacher_rows_by_epoch = getattr(self, '_basetrain_summary_teacher_rows', {})
        train_direct_rows_by_epoch = getattr(self, '_basetrain_summary_train_direct_rows', {})
        teacher_rows = [teacher_rows_by_epoch[ep] for ep in sorted(teacher_rows_by_epoch)]
        train_direct_rows = [train_direct_rows_by_epoch[ep] for ep in sorted(train_direct_rows_by_epoch)]

        payload: Dict[str, Any] = {
            'teacher': teacher_rows,
            'train_direct_group_norm': train_direct_rows,
        }

        summary_path = Path(out_dir) / 'basetrain_keybone_group_summary.json'
        self._write_json_payload(summary_path, payload)

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

    def _run_fit_epoch_cycle(
        self,
        train_loader,
        checkpoint_state: FitCheckpointState,
        *,
        epoch: int,
        total_epochs: int,
        log_every: int,
        scaler: torch.amp.GradScaler,
        accum_steps: int,
        tf_schedule: tuple[str, int, int, float, float],
    ) -> TrainEpochResult:
        tf_mode, tf_start, tf_end, tf_max_base, tf_min_base = tf_schedule
        tf_ratio = self._prepare_fit_epoch(epoch, total_epochs, tf_mode=tf_mode, tf_start=tf_start, tf_end=tf_end, tf_max_base=tf_max_base, tf_min_base=tf_min_base)
        train_epoch_result = self._run_one_train_epoch(train_loader, epoch=epoch, log_every=log_every, scaler=scaler, accum_steps=accum_steps, tf_ratio=tf_ratio)
        self._persist_epoch_metrics_artifacts(train_epoch_result.train_metrics, tag='train', epoch=epoch, finish_stage=False)
        validation_result = self._run_epoch_validation(epoch=epoch)
        self._persist_epoch_validation_outputs(epoch=epoch, validation_result=validation_result)
        return train_epoch_result

    def _finalize_fit_checkpoints(self, checkpoint_state: FitCheckpointState, *, out_dir: Optional[str], run_name: str) -> None:
        checkpoint_state.last_payload = self._fit_checkpoint_payload()
        if out_dir:
            for checkpoint_tag, ckpt_attr in (('last', 'last_ckpt'),):
                ckpt_path = self._save_fit_checkpoint_payload(
                    out_dir=out_dir,
                    run_name=run_name,
                    checkpoint_tag=checkpoint_tag,
                    payload=checkpoint_state.last_payload,
                )
                if ckpt_path is not None:
                    setattr(checkpoint_state, ckpt_attr, ckpt_path)

    def fit(self, train_loader, epochs=10, log_every=50, out_dir=None, run_name='run'):
        self.model.train()
        self.train_loader = train_loader
        device_type = getattr(self.device, 'type', 'cpu')
        scaler = torch.amp.GradScaler('cuda' if device_type=='cuda' else 'cpu', enabled=(getattr(self, 'use_amp', False) and device_type in ('cuda', 'mps')))
        accum_steps = int(getattr(self, 'accum_steps', 1) or 1)
        checkpoint_state = FitCheckpointState()
        history = {'train': []}
        total_epochs = int(epochs)
        tf_schedule = (
            getattr(self, 'tf_mode', 'epoch_linear'),
            int(getattr(self, 'tf_start_epoch', 0)),
            int(getattr(self, 'tf_end_epoch', 0)),
            float(getattr(self, 'tf_max', 1.0)),
            float(getattr(self, 'tf_min', 0.0)),
        )

        try:
            test_gradient_connection(self, train_loader)
        except Exception as _grad_exc:
            print(f"[GradConn] failed during warm-up: {_grad_exc}")
            raise

        for ep in range(1, total_epochs + 1):
            train_epoch_result = self._run_fit_epoch_cycle(
                train_loader,
                checkpoint_state,
                epoch=ep,
                total_epochs=total_epochs,
                log_every=log_every,
                scaler=scaler,
                accum_steps=accum_steps,
                tf_schedule=tf_schedule,
            )
            history['train'].append(train_epoch_result.avg_train)
        self._finalize_fit_checkpoints(checkpoint_state, out_dir=out_dir, run_name=run_name)
        return checkpoint_state.last_ckpt, history

TRAIN_ENTRY_CONFIG_META_KEYS = {'dataset_profile', 'strategy_meta'}


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    shared: SharedTrainerRuntime
    direct_pose_grad_monitor_enable: bool
    direct_pose_grad_ratio_gate: float
    contacts_pretrain: ContactPretrainRuntime
    diag_topk: int
    diag_thr: float
    teacher_eval_max_batches: Optional[int]
    ss_chunk_len: int
    tf_mode: str
    tf_start_epoch: int
    tf_end_epoch: int
    tf_max: float
    tf_min: float
    history_debug_steps: int
    history_dropout_prob: float
    history_dropout_prob_min: float
    history_dropout_prob_max: float
    freerun_stage_schedule: list[Any]
    hyperparam_scheduler: Any
    freerun_debug_path: Optional[str]
    enable_grad_connection_test: bool


def _resolve_freerun_stage_schedule(spec: Any) -> list[Any]:
    try:
        freerun_stage_schedule = _parse_stage_schedule(spec)
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
    return freerun_stage_schedule


def _resolve_trainer_runtime_config(
    args: argparse.Namespace,
    trainer: Trainer,
    dataset_artifacts: DatasetRuntimeArtifacts,
    norm_template_path: Optional[Path],
    bundle_json_path: Optional[str],
    out_dir: Path,
    resolved_config: Dict[str, Any],
    run_name: str,
) -> TrainerRuntimeConfig:
    shared_runtime = resolve_shared_trainer_runtime(
        dataset_artifacts=dataset_artifacts,
        trainer_default_yaw_forward_axis=int(getattr(trainer, 'yaw_forward_axis', 2)),
        yaw_forward_axis_override=args.yaw_forward_axis,
        yaw_forward_offset_deg_override=args.yaw_forward_offset,
        norm_template_path=norm_template_path,
        bundle_json_path=bundle_json_path,
        out_dir=out_dir,
        full_config=resolved_config,
        current_run_name=run_name,
    )
    contact_pretrain_runtime = resolve_contact_pretrain_runtime(
        clamp_raw=getattr(args, 'trainbase_contacts_pretrain_clamp', 1.0),
        affine_stats_raw=getattr(args, 'trainbase_contacts_pretrain_affine_stats', None),
        warn=True,
        warn_prefix='[MPL]',
    )
    eval_monitor = {
        'direct_pose_grad_monitor_enable': bool(args.direct_pose_grad_monitor_enable),
        'direct_pose_grad_ratio_gate': float(args.direct_pose_grad_ratio_gate or 0.35),
        'diag_topk': int(args.diag_topk or 8),
        'diag_thr': float(args.diag_thr or 8.0),
        'teacher_eval_max_batches': args.teacher_eval_max_batches,
    }
    history_schedule = {
        'ss_chunk_len': int(getattr(args, 'ss_chunk_len', getattr(trainer, 'ss_chunk_len', 1)) or 1),
        'tf_mode': args.tf_mode,
        'tf_start_epoch': int(args.tf_start_epoch),
        'tf_end_epoch': int(args.tf_end_epoch),
        'tf_max': float(args.tf_max),
        'tf_min': float(args.tf_min),
        'history_debug_steps': int(args.history_debug_steps or 0),
        'history_dropout_prob': float(args.history_dropout_prob or 0.0),
        'history_dropout_prob_min': 0.05,
        'history_dropout_prob_max': 0.30,
        'freerun_stage_schedule': _resolve_freerun_stage_schedule(args.freerun_stage_schedule),
        'hyperparam_scheduler': None,
        'freerun_debug_path': args.freerun_debug_path,
        'enable_grad_connection_test': not bool(args.no_grad_conn_test),
    }
    return TrainerRuntimeConfig(
        shared=shared_runtime,
        contacts_pretrain=contact_pretrain_runtime,
        **eval_monitor,
        **history_schedule,
    )


def _apply_trainer_runtime_config(trainer: Trainer, runtime_cfg: TrainerRuntimeConfig) -> None:
    apply_shared_trainer_runtime(trainer, runtime_cfg.shared)
    apply_contacts_pretrain_runtime(
        trainer,
        owner_prefix='trainbase',
        runtime=runtime_cfg.contacts_pretrain,
    )
    renamed_fields = {}
    direct_field_groups = (
        (
            'direct_pose_grad_monitor_enable', 'direct_pose_grad_ratio_gate', 'diag_topk', 'diag_thr',
            'teacher_eval_max_batches',
        ),
        (
            'ss_chunk_len', 'tf_mode', 'tf_start_epoch', 'tf_end_epoch', 'tf_max', 'tf_min',
            'history_debug_steps', 'history_dropout_prob', 'history_dropout_prob_min', 'history_dropout_prob_max',
            'freerun_stage_schedule', 'hyperparam_scheduler', 'freerun_debug_path', 'enable_grad_connection_test',
        ),
    )
    for trainer_attr, runtime_field in renamed_fields.items():
        setattr(trainer, trainer_attr, getattr(runtime_cfg, runtime_field))
    for direct_fields in direct_field_groups:
        for field_name in direct_fields:
            setattr(trainer, field_name, getattr(runtime_cfg, field_name))


# ===== Basetrain Entry Shell Band (Step 2) =====

@dataclass(frozen=True)
class TrainEntryContext:
    args: argparse.Namespace
    train_paths: list[str]
    run_name: str
    out_dir: Path
    device: torch.device
    norm_template_path: Optional[Path]
    norm_spec: Dict[str, Any]


@dataclass(frozen=True)
class TrainDataArtifacts:
    ds_train: MotionEventDataset
    train_loader: DataLoader
    pin_memory: bool
    dx: int
    dy: int
    dc: int


@dataclass(frozen=True)
class DirectPoseBuildOptions:
    split_enable: bool
    arm_split_enable: bool
    arm_bones_resolved: Optional[str]
    nonleg_proj_dim: int


@dataclass(frozen=True)
class TrainModelArtifacts:
    model: EventMotionModel
    pose_hist_dim_raw: int
    pose_hist_len_raw: int
    history_export_frames: int
    history_frame_dim: int
    direct_pose_options: DirectPoseBuildOptions


@dataclass(frozen=True)
class TrainBuildArtifacts:
    model: EventMotionModel
    loss_fn: MotionJointLoss
    trainer: Trainer
    bundle_json_path: Optional[str]
    resolved_config: Dict[str, Any]


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
    payload = dict(payload)
    valid_dests = {action.dest for action in parser._actions if action.dest and action.dest != 'help'}
    unknown_keys = sorted(str(k) for k in payload.keys() if str(k) not in valid_dests and str(k) not in TRAIN_ENTRY_CONFIG_META_KEYS)
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
        if not hasattr(namespace, key):
            parser.error(f"[config_override] 未知键名: {key}")
        new_value = _parse_literal(value_expr)
        setattr(namespace, key, new_value)
        applied[key] = new_value
    if applied:
        formatted = ', '.join(f"{k}={applied[k]}" for k in sorted(applied))
        print(f"[config_override] Applied: {formatted}")


def _parser_add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--encoder_path',
        type=str,
        default='models/motion_encoder_equiv.pt',
        help='预训练 MotionEncoder bundle 路径（.pt，比如第二阶段导出的 motion_encoder_equiv.pt）',
    )
    p.add_argument('--norm_template', type=str, default='raw_data/processed_data/norm_template.json', help='数据归一化模板路径')
    p.add_argument('--pretrain_template', type=str, default='models/pretrain_template.json', help='预训练编码器模板（含角速度统计）')
    p.add_argument('--data', type=str, required=True, help='数据目录（含 *.npz）')
    p.add_argument('--out', type=str, default='./runs', help='输出目录根路径')
    p.add_argument('--run_name', type=str, default=None, help='子目录名；未给则用时间戳')
    p.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从 checkpoint(.pth) 初始化模型权重（仅加载 model state_dict；会自动跳过 shape 不匹配的权重）。',
    )
    p.add_argument(
        '--config_override',
        action='append',
        default=None,
        metavar='KEY=VALUE',
        help='在解析后覆写配置值，可重复，例如 --config_override lr=5e-5',
    )
    p.add_argument('--train_files', type=str, default='', help='逗号分隔的路径/通配/或 @list.txt')
    p.add_argument('--diag_topk', type=int, default=8, help='free-run 评估时打印 X_norm 的 |z| Top-K')
    p.add_argument('--diag_thr', type=float, default=8.0, help='|z| 阈值，统计 X_norm 爆炸比例')
    p.add_argument("--bundle_json", type=str, default=None, help='UE 导出的运行时 bundle（可含 MuY/StdY、feature_layout、MuC_other/StdC_other 等）', required=True)


def _parser_add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--tf_mode', type=str, default='epoch_linear', choices=['global', 'epoch_linear'])
    p.add_argument('--tf_start_epoch', type=int, default=0)
    p.add_argument('--tf_end_epoch', type=int, default=10)
    p.add_argument('--tf_max', type=float, default=1.0)
    p.add_argument('--tf_min', type=float, default=0.1)
    p.add_argument(
        '--history_debug_steps',
        type=int,
        default=0,
        help='>1 时，在训练批次中额外运行 train_free rollout 诊断历史漂移步数',
    )
    p.add_argument(
        '--history_adaptive_max_frames',
        type=int,
        default=None,
        help='训练期允许的最大历史帧数（默认使用 norm_template 中的 pose_hist_len）',
    )
    p.add_argument('--history_adaptive_hidden', type=int, default=256, help='adaptive history 内部隐藏维度')
    p.add_argument('--history_adaptive_heads', type=int, default=2, help='adaptive history 注意力头数')
    p.add_argument('--history_adaptive_train_variable', action='store_true', help='训练时随机截断历史长度，提升部署鲁棒性')
    p.add_argument(
        '--history_dropout_prob',
        type=float,
        default=0.10,
        help='训练期以该概率完全屏蔽历史特征，迫使模型依赖未来条件信号进行纠错。',
    )
    p.add_argument('--history_use_trend_features', action='store_true', help='在 adaptive history 中显式注入历史 drift/趋势特征。')
    p.add_argument(
        '--freerun_stage_schedule',
        type=str,
        default=None,
        help='分阶段调度（TF/LR/history/direct-pose trainability 等）的 JSON/字符串配置。',
    )
    p.add_argument(
        '--ss_chunk_len',
        type=int,
        default=1,
        help='scheduled sampling 的 chunk 长度（>1 启用 sticky 采样：每 chunk 采一次 use_gt）。',
    )
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--context_len', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--amp', action='store_true', help='启用自动混合精度 (torch.autocast)')


def _parser_add_data_args(p: argparse.ArgumentParser) -> None:
    # ---- (Reserved) post-train corrector knobs live in dedicated scripts ----
    # unified bone weight (new)
    p.add_argument('--unified_downstream_power', type=float, default=0.6, help='下游影响指数压缩 power (0.5~0.7 recommended)')
    p.add_argument('--unified_self_scale', type=float, default=1.5, help='自身长度放大系数')
    p.add_argument('--unified_min_weight', type=float, default=0.05, help='权重保底（相对均值）')
    p.add_argument(
        '--rot_local_tail_weight',
        type=float,
        default=0.0,
        help='rot_local 额外 tail loss 权重（CVaR/top-k，越大越压最差骨骼）。0=关闭。',
    )
    p.add_argument(
        '--rot_local_tail_k',
        type=int,
        default=0,
        help='rot_local tail loss 的 top-k 骨骼数量（例如 13 骨骼取 3）。0=关闭。',
    )
    p.add_argument(
        '--rot_local_tail_scope',
        type=str,
        default='all',
        choices=['all', 'limbs', 'keybones'],
        help="tail loss 选择范围：all=全骨骼；limbs=limb_monitor_names（若缺失则用skeleton leaves回退）；keybones=pelvis+limb_monitor_names（并用leaves补全）。",
    )
    p.add_argument(
        '--rot_local_tail_select',
        type=str,
        default='batch',
        choices=['batch', 'ema'],
        help='tail loss top-k 选择打分：batch=当前batch均值；ema=跨batch EMA（更平滑、减少whack-a-mole）。',
    )
    p.add_argument('--rot_local_tail_ema_beta', type=float, default=0.9, help='rot_local_tail_select=ema 时的 EMA beta（越大越平滑）。')
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--yaw_aug_deg', type=float, default=0.0)
    p.add_argument('--normalize_c', action='store_true')


def _parser_add_loss_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--w_rot_ortho', type=float, default=0.001)
    p.add_argument('--w_rot_local', type=float, default=0.0, help='父子关节局部 geodesic 约束权重（0=关闭）。')
    p.add_argument('--w_root_vel', type=float, default=0.0, help='根速度向量 MSE 损失权重（输出包含 RootVelocity 时生效）。')
    p.add_argument('--w_root_speed', type=float, default=0.0, help='根速度模长 MAE 损失权重（输出包含 RootVelocity 时生效）。')


def _parser_add_model_args(p: argparse.ArgumentParser) -> None:
    # ---- Contact plan anchor (independent) ----
    p.add_argument('--contact_plan_enable', action='store_true', default=False, help='启用 cond-only GRU contacts_plan（作为独立锚点）')
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
    p.add_argument('--contact_plan_hidden', type=int, default=64, help='contacts_plan GRU hidden dim')
    p.add_argument('--contact_plan_dropout', type=float, default=0.0, help='contacts_plan head dropout')
    p.add_argument('--w_contact_plan', type=float, default=0.0, help='contacts_plan 监督权重（MSE vs GT soft_contacts）')
    p.add_argument(
        '--contact_plan_inject',
        type=str,
        default='none',
        choices=['none', 'contacts', 'plan_z'],
        help='Phase2: 将 contacts_plan / plan_z 前馈注入主干输入（none=关闭）',
    )
    p.add_argument(
        '--contact_plan_inject_detach',
        type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
        default=True,
        help='注入主干时对 plan 特征 stop-grad（保持 plan 语义为独立锚点；推荐开启）',
    )
    p.add_argument('--contact_plan_time_pe_dim', type=int, default=0, help='contacts_plan time positional encoding dim（0=关闭；推荐 8/16）')
    p.add_argument('--contact_plan_time_pe_base', type=float, default=10000.0, help='contacts_plan time-PE 频率基数（默认 10000）')
    p.add_argument(
        '--contact_plan_init_mode',
        type=str,
        default='learnable',
        choices=['zeros', 'learnable', 'obs', 'learnable+obs'],
        help='contact plan 冷启动 init：learnable(默认)|obs|learnable+obs(推荐)|zeros',
    )
    p.add_argument('--contact_plan_init_hidden', type=int, default=128, help='contact plan init MLP hidden dim（仅 init_mode=obs/learnable+obs 生效）')
    p.add_argument('--contact_plan_init_dropout', type=float, default=0.0, help='contact plan init MLP dropout（仅 init_mode=obs/learnable+obs 生效）')
    # ---- Event-Clock v3 (contact_plan residual correction) ----
    p.add_argument(
        '--use_event_clock',
        action='store_true',
        default=False,
        help='启用 Event-Clock v3：在 contact_plan GRU loop 内做 gated residual correction',
    )
    p.add_argument('--event_clock_max_delta', type=float, default=0.5, help='Event-Clock Δz clip 幅度（0=不 clip）')
    p.add_argument('--event_clock_hidden_dim', type=int, default=64, help='Event-Clock Δz head hidden dim')
    p.add_argument('--event_clock_gate_hidden_dim', type=int, default=32, help='Event-Clock gate head hidden dim')
    p.add_argument('--event_clock_lambda_entropy_weight', type=float, default=0.01, help='Event-Clock λ entropy 正则权重')
    p.add_argument('--event_clock_lambda_prior_weight', type=float, default=0.01, help='Event-Clock λ dynamic prior 正则权重')
    p.add_argument('--event_clock_delta_z_l2_weight', type=float, default=0.001, help='Event-Clock Δz L2 正则权重')
    # ---- Direct pose head (cond + contacts_plan -> absolute pose) ----
    p.add_argument('--direct_pose_enable', action='store_true', default=False, help='启用 direct pose head（cond+contacts_plan -> out_direct，不走自回归）')
    p.add_argument('--direct_pose_hidden', type=int, default=256, help='direct pose head hidden dim')
    p.add_argument('--direct_pose_dropout', type=float, default=0.0, help='direct pose head dropout')
    p.add_argument(
        '--direct_pose_detach_plan',
        type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
        default=True,
        help='direct head 输入 contacts_plan 时 stop-grad（推荐开启）',
    )
    p.add_argument(
        '--direct_pose_meas_mode',
        type=str,
        default='concat',
        choices=['concat', 'mode_select'],
        help='Phase bridge: direct head 是否引入 contacts_meas（concat=D0; mode_select=D1）',
    )
    p.add_argument('--direct_pose_meas_drop_prob', type=float, default=0.0, help='D2: 训练时对 direct 输入的 contacts_meas 执行整向量 drop(置0) 概率')
    p.add_argument('--direct_pose_meas_noise_std', type=float, default=0.0, help='D2: 训练时对 direct 输入的 contacts_meas 加高斯噪声 std（随后 clamp 到[0,1]）')
    p.add_argument(
        '--direct_pose_plan_drop_prob',
        type=float,
        default=0.0,
        help='D2: 训练时对 direct 输入的 contacts_plan 执行整向量 drop(置0) 概率（防止 plan 成为 shortcut）',
    )
    p.add_argument('--direct_pose_split_enable', action='store_true', default=False, help='启用 direct_pose 输出分头（shared trunk + leg/non-leg split output heads）')
    p.add_argument('--direct_pose_arm_split_enable', action='store_true', default=False, help='启用 direct_pose 非腿分支的 arm/else 再分头（3-way: leg/arm/else）')
    p.add_argument('--direct_pose_arm_bones', type=str, default=None, help='arm 分支骨骼 CSV；未提供且启用 arm split 时默认使用 Stage6 3-way armchain 口径')
    p.add_argument('--direct_pose_nonleg_proj_dim', type=int, default=0, help='non-leg/arm/else 分支投影维度；0=直接从 trunk readout')
    p.add_argument('--w_direct_pose', type=float, default=0.0, help='direct pose 监督权重（geodesic vs GT pose；0=关闭）')
    p.add_argument('--direct_pose_loss_leg_split', action='store_true', default=False, help='direct loss 按 leg/non-leg 拆分计算 base objective（Stage6 parity）')
    p.add_argument(
        '--direct_pose_loss_arm_else_balance_enable',
        action='store_true',
        default=False,
        help='启用 arm/else 组均衡 non-leg objective（按 group mean 重构 non-leg base）',
    )
    p.add_argument('--direct_pose_loss_arm_weight', type=float, default=1.0, help='arm/else rebalance 中 arm group 权重')
    p.add_argument('--direct_pose_loss_else_weight', type=float, default=1.0, help='arm/else rebalance 中 else group 权重')
    p.add_argument('--direct_pose_grad_monitor_enable', action='store_true', default=False, help='记录 direct trunk/leg/arm/else 输出头梯度范数与比值')
    p.add_argument('--direct_pose_grad_ratio_gate', type=float, default=0.35, help='direct grad ratio 诊断阈值（nonleg/leg；仅日志告警）')
    p.add_argument('--direct_pose_loss_group_norm_enable', action='store_true', default=False, help='启用 direct leg/non-leg group norm objective（Stage6 parity）')
    p.add_argument('--direct_pose_loss_group_norm_w_leg', type=float, default=1.0, help='group norm objective 中 leg ratio 权重')
    p.add_argument('--direct_pose_loss_group_norm_w_nonleg', type=float, default=1.0, help='group norm objective 中 non-leg ratio 权重')
    p.add_argument('--direct_pose_loss_group_norm_ema_beta', type=float, default=0.9, help='group norm EMA beta')
    p.add_argument('--direct_pose_loss_group_norm_ratio_min', type=float, default=0.2, help='group norm ratio clamp 最小值')
    p.add_argument('--direct_pose_loss_group_norm_ratio_max', type=float, default=5.0, help='group norm ratio clamp 最大值')
    p.add_argument('--direct_pose_loss_group_norm_eps', type=float, default=1e-6, help='group norm 数值稳定 epsilon')

def _parser_add_debug_export_args(p: argparse.ArgumentParser) -> None:
    # TensorBoard 相关逻辑已移除，避免冗余参数
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--teacher_eval_max_batches', type=int, default=None, help='Teacher 评估最多跑多少个 batch；<=0 则跳过评估，用训练均值loss代填')
    p.add_argument('--yaw_forward_axis', type=int, default=None, help='若提供，则覆盖数据推断的根骨前向轴(0/1/2)')
    p.add_argument('--yaw_forward_offset', type=float, default=None, help='额外指定 yaw 前向轴偏移（单位：度，优先于数据推断）')
    p.add_argument('--freerun_debug_path', type=str, default=None, help='若提供，则将首个 freerun batch 的诊断数据保存至该路径')
    p.add_argument('--no_grad_conn_test', action='store_true', help='跳过训练前的梯度连通性自检')


def _build_train_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        '--config_json',
        type=str,
        default=None,
        help='JSON 配置文件路径。键名需与 CLI 参数一致，并作为默认值参与解析。',
    )

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[config_parser])
    for parser_adder in (_parser_add_io_args, _parser_add_runtime_args, _parser_add_loss_args, _parser_add_model_args, _parser_add_data_args, _parser_add_debug_export_args):
        parser_adder(p)
    return config_parser, p

def _parse_train_entry_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser, parser = _build_train_parser()
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    required_actions = [action for action in parser._actions if getattr(action, 'required', False)]
    for action in required_actions:
        action.required = False

    config_defaults = _load_train_entry_config_defaults(config_args.config_json, parser)

    namespace = argparse.Namespace(**config_defaults)
    namespace.config_json = config_args.config_json
    parsed_args = parser.parse_args(remaining_argv, namespace=namespace)
    set_global_args(parsed_args)
    _apply_train_entry_config_overrides(parsed_args, getattr(parsed_args, 'config_override', None), parser)
    parsed_args.config_override = None

    missing_opts = [
        act.option_strings[-1] if act.option_strings else act.dest
        for act in required_actions
        if getattr(parsed_args, act.dest, None) is None
    ]
    if missing_opts:
        parser.error(f"missing required arguments: {', '.join(missing_opts)}")
    return parsed_args


def _build_direct_pose_options(args: argparse.Namespace) -> DirectPoseBuildOptions:
    arm_bones_raw = getattr(args, 'direct_pose_arm_bones', None)
    arm_split_enable = bool(args.direct_pose_arm_split_enable)
    arm_bones_txt = None if arm_bones_raw is None else str(arm_bones_raw).strip()
    arm_bones_default = None if not arm_split_enable else str(STAGE6_3WAY_ARMCHAIN_BONES_CSV)
    arm_bones_resolved = arm_bones_default if arm_bones_txt in (None, '') else arm_bones_txt
    return DirectPoseBuildOptions(
        split_enable=bool(args.direct_pose_split_enable),
        arm_split_enable=arm_split_enable,
        arm_bones_resolved=arm_bones_resolved,
        nonleg_proj_dim=int(args.direct_pose_nonleg_proj_dim or 0),
    )

def _build_train_entry_context(args: argparse.Namespace) -> TrainEntryContext:
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

    norm_template_arg = args.norm_template
    norm_template_path = Path(norm_template_arg).expanduser() if norm_template_arg else None
    pretrain_template_arg = args.pretrain_template
    pretrain_template_path = Path(pretrain_template_arg).expanduser() if pretrain_template_arg else None
    norm_spec = merge_norm_spec(
        norm_template_path,
        pretrain_template_path,
        strict=False,
        warn=True,
        warn_prefix='[Spec]',
    ) if norm_template_path is not None else None
    if norm_spec is None:
        raise SystemExit(f"[FATAL] norm_template 缺失或无效，请确认路径：{norm_template_path}")
    return TrainEntryContext(
        args=args,
        train_paths=train_paths,
        run_name=run_name,
        out_dir=out_dir,
        device=device,
        norm_template_path=norm_template_path,
        norm_spec=norm_spec,
    )


def _build_train_components(argv: Optional[Sequence[str]] = None) -> TrainEntryContext:
    args = _parse_train_entry_args(argv)
    return _build_train_entry_context(args)


def _build_train_loaders(train_ctx: TrainEntryContext) -> TrainDataArtifacts:
    args = train_ctx.args
    ds_train = build_motion_dataset(
        data_dir=args.data,
        seq_len=args.seq_len,
        paths=train_ctx.train_paths,
        norm_spec=train_ctx.norm_spec,
        optimize_index_fn=lambda dataset: _maybe_optimize_dataset_index(dataset, args),
        is_train=True,
        yaw_aug_deg=float(args.yaw_aug_deg),
        normalize_c=bool(args.normalize_c),
    )
    pin = train_ctx.device.type == 'cuda'
    train_loader = build_motion_dataloader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=make_fixedlen_collate(args.seq_len),
    )
    dx, dy, dc = int(ds_train.Dx), int(ds_train.Dy), int(ds_train.Dc)
    print(f'[Export][Dims] Dx={dx}, Dy={dy}, Dc={dc} | L={int(args.depth)}, H={int(args.width)}, K={int(args.context_len)}')
    return TrainDataArtifacts(
        ds_train=ds_train,
        train_loader=train_loader,
        pin_memory=pin,
        dx=dx,
        dy=dy,
        dc=dc,
    )


def _build_train_model(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
) -> TrainModelArtifacts:
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

    direct_pose_options = _build_direct_pose_options(args)

    model_kwargs = dict(
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
        direct_pose_split_enable=bool(direct_pose_options.split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_options.nonleg_proj_dim),
        direct_pose_arm_split_enable=bool(direct_pose_options.arm_split_enable),
        direct_pose_arm_bones=direct_pose_options.arm_bones_resolved,
    )
    model = EventMotionModel(**model_kwargs).to(device)
    return TrainModelArtifacts(
        model=model,
        pose_hist_dim_raw=pose_hist_dim_raw,
        pose_hist_len_raw=pose_hist_len_raw,
        history_export_frames=history_export_frames,
        history_frame_dim=history_frame_dim,
        direct_pose_options=direct_pose_options,
    )


def _sanitize_train_model_post_build(model: torch.nn.Module, train_data: TrainDataArtifacts) -> None:
    validate_and_fix_model_(model, train_data.dx, train_data.dc)
    validate_and_fix_model_(model)


def _prepare_motion_encoder_and_contacts_runtime(
    model: EventMotionModel,
    ds_train: MotionEventDataset,
    args: argparse.Namespace,
) -> None:
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
                _attach_motion_encoder_bundle(model, bundle)
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

    if bool(getattr(model, 'contact_plan_enable', False)):
        has_pretrain_contact = (
            getattr(model, 'frozen_encoder', None) is not None
            and getattr(model, 'frozen_contact_head', None) is not None
        )
        if not has_pretrain_contact:
            raise SystemExit(
                '[FATAL] contact_plan_enable now requires --encoder_path to provide '
                'a frozen encoder bundle with contact_head for rollout contact resolution.'
            )

def _prepare_train_model_runtime(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
    model_artifacts: TrainModelArtifacts,
) -> None:
    args = train_ctx.args
    ds_train = train_data.ds_train
    model = model_artifacts.model

    attach_adaptive_history_runtime(
        model,
        history_export_frames=model_artifacts.history_export_frames,
        pose_hist_dim_raw=model_artifacts.pose_hist_dim_raw,
        pose_hist_len_raw=model_artifacts.pose_hist_len_raw,
        history_frame_dim=model_artifacts.history_frame_dim,
        history_hidden_dim=int(args.history_adaptive_hidden or int(args.width)),
        max_history_frames=args.history_adaptive_max_frames,
        history_heads=int(args.history_adaptive_heads or 2),
        train_variable_history=bool(args.history_adaptive_train_variable),
        history_dropout_prob=float(args.history_dropout_prob or 0.0),
        use_trend_features=bool(args.history_use_trend_features),
        device=train_ctx.device,
    )
    _sanitize_train_model_post_build(model, train_data)
    _prepare_motion_encoder_and_contacts_runtime(model, ds_train, args)
    _resume_load_weights_compat(model, getattr(args, 'resume', None))
    _guard_first_linear_finite_(model)

    try:
        model._pasa_fps = float(getattr(ds_train, 'fps', 60.0))
    except (AttributeError, TypeError, ValueError) as exc:
        _phasec_warn_once(
            "train_entry/pasa_fps",
            "failed to set model._pasa_fps from dataset fps",
            exc,
        )


def _build_train_loss_and_trainer(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
    model_artifacts: TrainModelArtifacts,
) -> TrainBuildArtifacts:
    args = train_ctx.args
    ds_train = train_data.ds_train
    model = model_artifacts.model
    direct_pose_options = model_artifacts.direct_pose_options
    direct_pose_arm_split_enable = direct_pose_options.arm_split_enable
    direct_pose_arm_bones_resolved = direct_pose_options.arm_bones_resolved

    fps_data = float(getattr(ds_train, 'fps', 60.0) or 60.0)
    w_rot_local, w_root_vel, w_root_speed = (float(value or 0.0) for value in (args.w_rot_local, args.w_root_vel, args.w_root_speed))

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

    print(
        f'[LossWeights] '
        f'w_rot_ortho={loss_fn.w_rot_ortho} '
        f'w_rot_local={loss_fn.w_rot_local} '
        f'rot_local_tail_weight={getattr(loss_fn, "rot_local_tail_weight", 0.0)} '
        f'rot_local_tail_k={getattr(loss_fn, "rot_local_tail_k", 0)} '
        f'rot_local_tail_scope={getattr(loss_fn, "rot_local_tail_scope", "all")} '
        f'rot_local_tail_select={getattr(loss_fn, "rot_local_tail_select", "batch")} '
        f'rot_local_tail_ema_beta={getattr(loss_fn, "rot_local_tail_ema_beta", 0.9)} '
    )

    loss_fn.dt_traj = 1.0 / max(1e-6, fps_data)
    loss_fn.dt_bone = 1.0 / max(1e-6, fps_data)
    print(f'[Dt] dt_traj={loss_fn.dt_traj:.6f}s | dt_bone={loss_fn.dt_bone:.6f}s (dataset fps={fps_data})')

    if hasattr(loss_fn, 'rot6d_eps'):
        loss_fn.rot6d_eps = 1e-6
    trainer_kwargs = dict(
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        use_amp=args.amp,
        accum_steps=args.accum_steps,
        pin_memory=train_data.pin_memory,
        args=args,
    )
    trainer = Trainer(model=model, loss_fn=loss_fn, **trainer_kwargs)
    if bool(args.direct_pose_loss_group_norm_enable) and (not bool(args.direct_pose_loss_leg_split)):
        print('[Loss][WARN] direct_pose_loss_group_norm_enable=true but direct_pose_loss_leg_split=false; group norm will have no effect.')
    try:
        resolved_config = dict(vars(args))
    except Exception:
        resolved_config = {}
    resolved_config.update(
        direct_pose_split_enable=bool(direct_pose_options.split_enable),
        direct_pose_arm_split_enable=bool(direct_pose_arm_split_enable),
        direct_pose_arm_bones=direct_pose_arm_bones_resolved,
        direct_pose_nonleg_proj_dim=int(direct_pose_options.nonleg_proj_dim),
    )
    try:
        cfg_out = train_ctx.out_dir / 'config_resolved.json'
        with open(cfg_out, 'w', encoding='utf-8') as f:
            json.dump(resolved_config, f, ensure_ascii=False, indent=2, default=str)
        print(f'[Config] saved resolved config to {cfg_out}')
    except Exception as exc:
        print(f'[Config][WARN] failed to save resolved config: {exc}')

    return TrainBuildArtifacts(
        model=model,
        loss_fn=loss_fn,
        trainer=trainer,
        bundle_json_path=bundle_json_path,
        resolved_config=resolved_config,
    )


def _sync_train_entry_loss_runtime(build_artifacts: TrainBuildArtifacts) -> None:
    apply_loss_runtime_from_trainer(
        build_artifacts.loss_fn,
        build_artifacts.trainer,
        copy_bundle_meta=True,
        warn_once_fn=_phasec_warn_once,
        warn_key="train_entry/loss_meta_from_bundle",
        warn_message="failed to copy bundle meta onto loss_fn.meta",
    )


def _attach_train_entry_runtime(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
    build_artifacts: TrainBuildArtifacts,
) -> None:
    dataset_artifacts = build_and_attach_dataset_runtime(
        build_artifacts.trainer,
        train_data.ds_train,
        bundle_path=build_artifacts.bundle_json_path,
    )
    runtime_cfg = _resolve_trainer_runtime_config(
        args=train_ctx.args,
        trainer=build_artifacts.trainer,
        dataset_artifacts=dataset_artifacts,
        norm_template_path=train_ctx.norm_template_path,
        bundle_json_path=build_artifacts.bundle_json_path,
        out_dir=train_ctx.out_dir,
        resolved_config=build_artifacts.resolved_config,
        run_name=train_ctx.run_name,
    )
    _apply_trainer_runtime_config(build_artifacts.trainer, runtime_cfg)
    _sync_train_entry_loss_runtime(build_artifacts)


def _run_postfit_actions(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
    build_artifacts: TrainBuildArtifacts,
) -> None:
    model = build_artifacts.model
    _export_postfit_onnx(train_ctx, train_data, model)


def _export_postfit_onnx(
    train_ctx: TrainEntryContext,
    train_data: TrainDataArtifacts,
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
    _attach_train_entry_runtime(train_ctx, train_data, build_artifacts)

    _norm_debug_once(
        build_artifacts.trainer,
        train_data.train_loader,
        thr=float(train_ctx.args.diag_thr),
        topk=int(train_ctx.args.diag_topk),
        print_to_console=False,
    )

    build_artifacts.trainer.fit(
        train_data.train_loader,
        epochs=train_ctx.args.epochs,
        log_every=train_ctx.args.log_every,
        out_dir=str(train_ctx.out_dir),
        run_name=train_ctx.run_name,
    )
    _run_postfit_actions(train_ctx, train_data, build_artifacts)


@torch.no_grad()
def export_onnx_step_stateful_nophase(model: torch.nn.Module, loader, onnx_path: str, opset: int = 18, dynamic_batch: bool = False):
    """
    单步（无隐式状态）ONNX 导出：
      输入:  state[B,Dx], cond[B,Dc], contacts[B,C], angvel[B,A], pose_hist[B,P],
            plan_z[B,Hp]
      输出:  motion_pred[B,Dy], plan_z_next[B,Hp]

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

    cond0 = _frame_or_zero(cond_seq, cond_dim, torch.float32)
    contacts0 = _frame_or_zero(contacts_seq, contact_dim, torch.float32)
    angvel0 = _frame_or_zero(angvel_seq, angvel_dim, torch.float32)
    pose_hist0 = _frame_or_zero(pose_hist_seq, pose_hist_dim, torch.float32)
    plan_z0 = torch.zeros((1, plan_dim), dtype=torch.float32) if plan_dim > 0 else None

    device = torch.device('cpu')
    model = model.to(device).eval()

    if plan_dim <= 0 or plan_z0 is None:
        raise RuntimeError(
            f"[Export][FATAL] ONNX export expects contact_plan state, got plan_dim={plan_dim}."
        )

    class _StatelessWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, state, cond, contacts, angvel, pose_hist, plan_z):
            cond_in = cond if cond.shape[-1] > 0 else None
            contacts_in = contacts if contacts.shape[-1] > 0 else None
            angvel_in = angvel if angvel.shape[-1] > 0 else None
            pose_hist_in = pose_hist if pose_hist.shape[-1] > 0 else None
            plan_z_in = plan_z if plan_z.shape[-1] > 0 else None
            out = self.core(
                state,
                cond_in,
                contacts=contacts_in,
                angvel=angvel_in,
                pose_history=pose_hist_in,
                plan_z=plan_z_in,
            )
            if isinstance(out, dict):
                pred = out.get('out')
                if pred is None:
                    raise RuntimeError("Model dict output missing 'out'.")
                z_next = out.get('plan_z_next')
                if z_next is None:
                    z_next = plan_z.new_zeros(plan_z.shape)
                return pred, z_next
            return out, plan_z

    wrapper = _StatelessWrapper(model).cpu().eval()
    sample_out = wrapper(state0, cond0, contacts0, angvel0, pose_hist0, plan_z0)
    Dy = int(sample_out[0].shape[-1])

    inputs = (state0, cond0, contacts0, angvel0, pose_hist0, plan_z0)
    input_names = ['state', 'cond', 'contacts', 'angvel', 'pose_hist', 'plan_z']
    output_names = ['motion_pred', 'plan_z_next']
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
        f'Hp={plan_dim}'
    )

def main():
    train_entry()
if __name__ == '__main__':
    main()
