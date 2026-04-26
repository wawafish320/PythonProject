from __future__ import annotations

"""Loss seam home for motion training objectives."""

import math as _math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    rot6d_to_matrix,
    geodesic_R,
    angvel_vec_from_R_seq,
    reproject_rot6d,
    root_relative_matrices,
    parent_relative_matrices,
)
from .data.layout import infer_rot_joint_count, parse_layout_entry, resolve_rot6d_slice
from .utils import _resolve_joint_spec_indices

__all__ = [
    'MotionJointLoss',
    'DEFAULT_DIRECT_POSE_LEG_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES',
    'STAGE6_3WAY_ARMCHAIN_BONES_CSV',
]

DEFAULT_DIRECT_POSE_LEG_BONES: tuple[str, ...] = (
    'thigh_r', 'calf_r', 'foot_r', 'ball_r',
    'thigh_l', 'calf_l', 'foot_l', 'ball_l',
)

STAGE6_3WAY_ARMCHAIN_BONES: tuple[str, ...] = (
    'clavicle_l', 'upperarm_l', 'RUpArmTwist_l_01', 'RUpArmTwist_l_02',
    'lowerarm_l', 'L_ForeTwist_01', 'L_ForeTwist_02', 'hand_l',
    'index_01_l', 'middle_01_l', 'ring_01_l', 'pinky_01_l', 'thumb_01_l',
    'clavicle_r', 'upperarm_r', 'RUpArmTwist_r_01', 'RUpArmTwist_r_02',
    'lowerarm_r', 'R_ForeTwist_01', 'R_ForeTwist_02', 'hand_r',
    'index_01_r', 'middle_01_r', 'ring_01_r', 'pinky_01_r', 'thumb_01_r',
)

STAGE6_3WAY_ARMCHAIN_BONES_CSV = ','.join(STAGE6_3WAY_ARMCHAIN_BONES)

_DIRECT_POSE_DEFAULT_STATS_SPEC: tuple[tuple[str, str | None, float], ...] = (
    ('direct_pose_geo', None, 0.0),
    ('direct_pose_geo_deg', None, 0.0),
    ('direct_pose_objective', None, 0.0),
    ('direct_pose_weighted', None, 0.0),
    ('direct_pose_split_active', None, 0.0),
    ('direct_pose_arm_split_active', None, 0.0),
    ('dir_base', None, float('nan')),
    ('dir_leg_base', None, float('nan')),
    ('dir_nonleg_base', None, float('nan')),
    ('dir_nonleg_effective_base', None, float('nan')),
    ('dir_arm_base', None, float('nan')),
    ('dir_else_base', None, float('nan')),
    ('leg_over_nonleg', None, float('nan')),
    ('leg_over_nonleg_effective', None, float('nan')),
    ('arm_over_else', None, float('nan')),
    ('direct_pose_arm_else_balance_active', None, 0.0),
    ('direct_pose_loss_arm_weight', 'arm_weight', 0.0),
    ('direct_pose_loss_else_weight', 'else_weight', 0.0),
    ('dir_group_norm_used', None, 0.0),
    ('dir_group_norm_leg_raw', None, float('nan')),
    ('dir_group_norm_nonleg_raw', None, float('nan')),
    ('dir_group_norm_leg_clamped', None, float('nan')),
    ('dir_group_norm_nonleg_clamped', None, float('nan')),
    ('dir_group_norm_leg', None, float('nan')),
    ('dir_group_norm_nonleg', None, float('nan')),
    ('dir_group_norm_leg_ema', None, float('nan')),
    ('dir_group_norm_nonleg_ema', None, float('nan')),
    ('dir_group_norm_leg_hit_min', None, 0.0),
    ('dir_group_norm_leg_hit_max', None, 0.0),
    ('dir_group_norm_nonleg_hit_min', None, 0.0),
    ('dir_group_norm_nonleg_hit_max', None, 0.0),
    ('dir_group_norm_leg_hit_any', None, 0.0),
    ('dir_group_norm_nonleg_hit_any', None, 0.0),
)

_DIRECT_POSE_DEFAULT_STAT_KEYS: tuple[str, ...] = tuple(
    key for key, _, _ in _DIRECT_POSE_DEFAULT_STATS_SPEC
)


def _build_direct_pose_default_stats(feature_config: "_DirectPoseFeatureConfig") -> Dict[str, float]:
    return {
        key: float(getattr(feature_config, feature_attr)) if feature_attr is not None else float(default)
        for key, feature_attr, default in _DIRECT_POSE_DEFAULT_STATS_SPEC
    }

_DIRECT_POSE_COMPONENT_STAT_KEYS: tuple[str, ...] = (
    *_DIRECT_POSE_DEFAULT_STAT_KEYS,
    'dir_group_norm_w_leg',
    'dir_group_norm_w_nonleg',
)


def _masked_group_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return _masked_group_weighted_mean(values, mask, joint_weights=None)


def _masked_group_weighted_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    joint_weights: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if values is None or mask is None or values.numel() == 0:
        return None
    if mask.numel() != values.shape[-1]:
        return None
    if not bool(mask.any().detach().cpu().item()):
        return None
    group_values = values[..., mask]
    if joint_weights is None:
        return group_values.mean()
    if joint_weights.ndim != 1 or joint_weights.numel() != values.shape[-1]:
        return None
    group_weights = joint_weights.to(device=group_values.device, dtype=group_values.dtype)[mask]
    if group_weights.numel() != group_values.shape[-1]:
        return None
    group_weight_sum = group_weights.sum()
    if not bool((group_weight_sum > 0.0).detach().cpu().item()):
        return None
    weighted = (group_values * group_weights).sum(dim=-1) / group_weight_sum.clamp_min(1e-6)
    return weighted.mean()


def _stats_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _stats_float_or(value: Any, default: float = 0.0) -> float:
    try:
        return _stats_float(value)
    except (RuntimeError, TypeError, ValueError):
        return float(default)


def _ensure_temporal_axis(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    return tensor


def _setdefault_stats(stats: Dict[str, float], defaults: Dict[str, float]) -> None:
    for key, value in defaults.items():
        stats.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class _DirectPosePair:
    direct_seq: torch.Tensor
    gt_direct: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupBaseRequest:
    geo_theta: Optional[torch.Tensor] = None
    dir_base: Optional[torch.Tensor] = None
    dir_leg_base: Optional[torch.Tensor] = None
    dir_nonleg_base: Optional[torch.Tensor] = None
    dir_arm_base: Optional[torch.Tensor] = None
    dir_else_base: Optional[torch.Tensor] = None
    arm_split_enable: Optional[bool] = None
    arm_else_balance_enable: Optional[bool] = None
    arm_weight: Optional[float] = None
    else_weight: Optional[float] = None
    eps: Optional[float] = None


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupBaseTerms:
    dir_base: Optional[torch.Tensor] = None
    dir_leg_base: Optional[torch.Tensor] = None
    dir_nonleg_base: Optional[torch.Tensor] = None
    dir_arm_base: Optional[torch.Tensor] = None
    dir_else_base: Optional[torch.Tensor] = None

    def has_any_tensor(self) -> bool:
        return any(torch.is_tensor(value) for value in (
            self.dir_base,
            self.dir_leg_base,
            self.dir_nonleg_base,
            self.dir_arm_base,
            self.dir_else_base,
        ))


@dataclass(frozen=True, slots=True)
class _DirectPoseFeatureConfig:
    leg_split_enable: bool
    arm_split_enable: bool
    arm_else_balance_enable: bool
    arm_weight: float
    else_weight: float
    group_norm_enable: bool
    group_norm_w_leg: float
    group_norm_w_nonleg: float
    group_norm_beta: float
    group_norm_ratio_min: float
    group_norm_ratio_max: float
    group_norm_eps: float


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupBaseResult:
    dir_base: Optional[torch.Tensor]
    dir_leg_base: Optional[torch.Tensor]
    dir_nonleg_base: Optional[torch.Tensor]
    dir_nonleg_effective_base: Optional[torch.Tensor]
    dir_arm_base: Optional[torch.Tensor]
    dir_else_base: Optional[torch.Tensor]
    leg_over_nonleg: float
    leg_over_nonleg_effective: float
    arm_over_else: float
    direct_pose_arm_else_balance_active: float
    direct_pose_loss_arm_weight: float
    direct_pose_loss_else_weight: float

    def has_leg_split_tensors(self) -> bool:
        return (
            torch.is_tensor(self.dir_leg_base)
            and torch.is_tensor(self.dir_nonleg_base)
            and torch.is_tensor(self.dir_nonleg_effective_base)
        )

    def as_payload(self) -> Dict[str, Any]:
        return {
            'dir_base': self.dir_base if torch.is_tensor(self.dir_base) else float('nan'),
            'dir_leg_base': self.dir_leg_base if torch.is_tensor(self.dir_leg_base) else float('nan'),
            'dir_nonleg_base': self.dir_nonleg_base if torch.is_tensor(self.dir_nonleg_base) else float('nan'),
            'dir_nonleg_effective_base': (
                self.dir_nonleg_effective_base if torch.is_tensor(self.dir_nonleg_effective_base) else float('nan')
            ),
            'dir_arm_base': self.dir_arm_base if torch.is_tensor(self.dir_arm_base) else float('nan'),
            'dir_else_base': self.dir_else_base if torch.is_tensor(self.dir_else_base) else float('nan'),
            'leg_over_nonleg': float(self.leg_over_nonleg),
            'leg_over_nonleg_effective': float(self.leg_over_nonleg_effective),
            'arm_over_else': float(self.arm_over_else),
            'direct_pose_arm_else_balance_active': float(self.direct_pose_arm_else_balance_active),
            'direct_pose_loss_arm_weight': float(self.direct_pose_loss_arm_weight),
            'direct_pose_loss_else_weight': float(self.direct_pose_loss_else_weight),
        }


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormRequest:
    dir_leg_base: torch.Tensor
    dir_nonleg_base: torch.Tensor
    dir_nonleg_effective_base: torch.Tensor
    direct_group_w_leg: Optional[float] = None
    direct_group_w_nonleg: Optional[float] = None
    direct_group_beta: Optional[float] = None
    direct_group_ratio_min: Optional[float] = None
    direct_group_ratio_max: Optional[float] = None
    direct_group_eps: Optional[float] = None
    update_ema_state: bool = True


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormConfig:
    w_leg: float
    w_nonleg: float
    beta: float
    ratio_min: float
    ratio_max: float
    eps: float


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormMetrics:
    ema_leg_prev: torch.Tensor
    ema_non_prev: torch.Tensor
    leg_ratio_raw: torch.Tensor
    nonleg_ratio_raw: torch.Tensor
    leg_ratio: torch.Tensor
    nonleg_ratio: torch.Tensor
    leg_hit_min: torch.Tensor
    leg_hit_max: torch.Tensor
    nonleg_hit_min: torch.Tensor
    nonleg_hit_max: torch.Tensor

    def objective(self, config: _DirectPoseGroupNormConfig) -> torch.Tensor:
        return config.w_leg * self.leg_ratio + config.w_nonleg * self.nonleg_ratio

    def as_stats(self, config: _DirectPoseGroupNormConfig) -> Dict[str, float]:
        return {
            'dir_group_norm_used': 1.0,
            'dir_group_norm_leg_raw': _stats_float(self.leg_ratio_raw),
            'dir_group_norm_nonleg_raw': _stats_float(self.nonleg_ratio_raw),
            'dir_group_norm_leg_clamped': _stats_float(self.leg_ratio),
            'dir_group_norm_nonleg_clamped': _stats_float(self.nonleg_ratio),
            'dir_group_norm_leg': _stats_float(self.leg_ratio),
            'dir_group_norm_nonleg': _stats_float(self.nonleg_ratio),
            'dir_group_norm_leg_ema': _stats_float(self.ema_leg_prev),
            'dir_group_norm_nonleg_ema': _stats_float(self.ema_non_prev),
            'dir_group_norm_leg_hit_min': _stats_float(self.leg_hit_min),
            'dir_group_norm_leg_hit_max': _stats_float(self.leg_hit_max),
            'dir_group_norm_nonleg_hit_min': _stats_float(self.nonleg_hit_min),
            'dir_group_norm_nonleg_hit_max': _stats_float(self.nonleg_hit_max),
            'dir_group_norm_leg_hit_any': _stats_float(torch.maximum(self.leg_hit_min, self.leg_hit_max)),
            'dir_group_norm_nonleg_hit_any': _stats_float(torch.maximum(self.nonleg_hit_min, self.nonleg_hit_max)),
            'dir_group_norm_w_leg': float(config.w_leg),
            'dir_group_norm_w_nonleg': float(config.w_nonleg),
        }


@dataclass(frozen=True, slots=True)
class _DirectPoseGroupNormResult:
    objective: torch.Tensor
    stats: Dict[str, float]
    ema_update: Dict[str, Any]

    def as_tuple(self) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        return self.objective, self.stats, self.ema_update


@dataclass(frozen=True, slots=True)
class _DirectPosePayloadRequest:
    direct: torch.Tensor
    gt_motion: torch.Tensor
    deg_per_rad: float


@dataclass(frozen=True, slots=True)
class _DirectPosePayloadResult:
    objective: torch.Tensor
    extra: Dict[str, Any]

    def as_tuple(self) -> tuple[torch.Tensor, Dict[str, Any]]:
        return self.objective, self.extra


class MotionJointLoss(nn.Module):
    _masked_group_mean = staticmethod(_masked_group_mean)
    _masked_group_weighted_mean = staticmethod(_masked_group_weighted_mean)
    _stats_float = staticmethod(_stats_float)
    _stats_float_or = staticmethod(_stats_float_or)
    _ensure_temporal_axis = staticmethod(_ensure_temporal_axis)
    _setdefault_stats = staticmethod(_setdefault_stats)

    # === future: train/loss/init.py ===
    # Constructor / config bootstrap.
    def __init__(
        self,
        w_attn_reg: float = 0.01,
        output_layout: Dict[str, Any] = None,
        fps: float = 60.0,
        rot6d_spec: Dict[str, Any] = None,
        w_rot_ortho: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        w_rot_local: float = 0.0,
        w_root_vel: float = 0.0,
        w_root_speed: float = 0.0,
        # Contact plan (cond-only anchor) supervision
        w_contact_plan: float = 0.0,
        # Optional: supervise meas head (pose-derived contacts)
        w_contact_meas: float = 0.0,
        # Optional: supervise direct pose head (cond + contacts_plan -> absolute pose)
        w_direct_pose: float = 0.0,
        direct_pose_loss_leg_split: bool = False,
        direct_pose_leg_bones: Optional[Sequence[str] | str] = None,
        direct_pose_arm_split_enable: bool = False,
        direct_pose_arm_bones: Optional[Sequence[str] | str] = None,
        direct_pose_loss_arm_else_balance_enable: bool = False,
        direct_pose_loss_arm_weight: float = 1.0,
        direct_pose_loss_else_weight: float = 1.0,
        direct_pose_loss_group_norm_enable: bool = False,
        direct_pose_loss_group_norm_w_leg: float = 1.0,
        direct_pose_loss_group_norm_w_nonleg: float = 1.0,
        direct_pose_loss_group_norm_ema_beta: float = 0.9,
        direct_pose_loss_group_norm_ratio_min: float = 0.2,
        direct_pose_loss_group_norm_ratio_max: float = 5.0,
        direct_pose_loss_group_norm_eps: float = 1e-6,
        # Optional: regularize omega_hat magnitude (prevents aggressive corrections)
        w_omega_l2: float = 0.0,
        # Event-Clock v3 regularization (only active when model returns corresponding tensors)
        event_clock_lambda_entropy_weight: float = 0.0,
        event_clock_lambda_prior_weight: float = 0.0,
        event_clock_delta_z_l2_weight: float = 0.0,
        **legacy_kwargs: Any,
    ):
        super().__init__()

        # Fail-fast retired-key boundary.
        legacy_loss_keys = (
            "ignore_motion_groups",
            "bone_prior_stds",
            "use_hierarchy_weights",
            "hierarchy_mode",
            "hierarchy_alpha",
            "max_weight_ratio",
            "weight_gamma",
        )
        if legacy_kwargs:
            legacy_hits = sorted(k for k in legacy_kwargs.keys() if k in legacy_loss_keys)
            if legacy_hits:
                joined = ", ".join(legacy_hits)
                raise ValueError(
                    f"MotionJointLoss deprecated loss keys are no longer supported: {joined}. "
                    "Please remove them from config/CLI and use unified bone weighting knobs only."
                )
            unknown = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"MotionJointLoss got unexpected keyword arguments: {unknown}")

        # Local scalar normalization.
        self.meta = dict(meta) if isinstance(meta, dict) else {}

        def _resolve_positive_scalar(field_name: str, raw_value: Any, *, default_value: float) -> float:
            candidate = default_value if raw_value is None else raw_value
            try:
                resolved = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a finite scalar in range (0, inf); "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            if (not _math.isfinite(resolved)) or resolved <= 0.0:
                raise ValueError(
                    f"{field_name} must be a finite scalar in range (0, inf); "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        def _resolve_bounded_scalar(
            field_name: str,
            raw_value: Any,
            *,
            default_value: float,
            min_value: float,
            max_value: float,
            min_inclusive: bool,
            max_inclusive: bool,
            expected_range: str,
        ) -> float:
            candidate = default_value if raw_value is None else raw_value
            try:
                resolved = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a finite scalar in range {expected_range}; "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            lower_ok = resolved >= float(min_value) if min_inclusive else resolved > float(min_value)
            upper_ok = resolved <= float(max_value) if max_inclusive else resolved < float(max_value)
            if (not _math.isfinite(resolved)) or (not lower_ok) or (not upper_ok):
                raise ValueError(
                    f"{field_name} must be a finite scalar in range {expected_range}; "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        # Core loss weights / direct-pose config.
        self.w_attn_reg = float(w_attn_reg)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_local = float(w_rot_local)
        self.w_root_vel = float(w_root_vel)
        self.w_root_speed = float(w_root_speed)
        self.w_contact_plan = float(w_contact_plan)
        self.w_contact_meas = float(w_contact_meas)
        self.w_direct_pose = float(w_direct_pose)
        self.direct_pose_loss_leg_split = bool(direct_pose_loss_leg_split)
        self.direct_pose_leg_bones = direct_pose_leg_bones
        self.direct_pose_arm_split_enable = bool(direct_pose_arm_split_enable)
        self.direct_pose_arm_bones = direct_pose_arm_bones
        self.direct_pose_loss_arm_else_balance_enable = bool(direct_pose_loss_arm_else_balance_enable)
        self.direct_pose_loss_arm_weight = _resolve_positive_scalar(
            "direct_pose_loss_arm_weight",
            direct_pose_loss_arm_weight,
            default_value=1.0,
        )
        self.direct_pose_loss_else_weight = _resolve_positive_scalar(
            "direct_pose_loss_else_weight",
            direct_pose_loss_else_weight,
            default_value=1.0,
        )
        self.direct_pose_loss_group_norm_enable = bool(direct_pose_loss_group_norm_enable)
        self.direct_pose_loss_group_norm_w_leg = float(direct_pose_loss_group_norm_w_leg or 1.0)
        self.direct_pose_loss_group_norm_w_nonleg = float(direct_pose_loss_group_norm_w_nonleg or 1.0)
        self.direct_pose_loss_group_norm_ema_beta = _resolve_bounded_scalar(
            "direct_pose_loss_group_norm_ema_beta",
            direct_pose_loss_group_norm_ema_beta,
            default_value=0.9,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=False,
            max_inclusive=True,
            expected_range="(0.0, 0.9999]",
        )
        self.direct_pose_loss_group_norm_ratio_min = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_ratio_min",
            direct_pose_loss_group_norm_ratio_min,
            default_value=0.2,
        )
        self.direct_pose_loss_group_norm_ratio_max = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_ratio_max",
            direct_pose_loss_group_norm_ratio_max,
            default_value=5.0,
        )
        if self.direct_pose_loss_group_norm_ratio_min > self.direct_pose_loss_group_norm_ratio_max:
            raise ValueError(
                "direct_pose_loss_group_norm_ratio_min/direct_pose_loss_group_norm_ratio_max must satisfy "
                "ratio_min <= ratio_max with both finite scalars in range (0, inf); "
                f"got direct_pose_loss_group_norm_ratio_min={self.direct_pose_loss_group_norm_ratio_min!r}, "
                f"direct_pose_loss_group_norm_ratio_max={self.direct_pose_loss_group_norm_ratio_max!r}."
            )
        self.direct_pose_loss_group_norm_eps = _resolve_positive_scalar(
            "direct_pose_loss_group_norm_eps",
            direct_pose_loss_group_norm_eps,
            default_value=1e-6,
        )
        self._direct_pose_group_norm_ema: Dict[str, torch.Tensor] = {}
        self.w_omega_l2 = float(w_omega_l2)
        self.event_clock_lambda_entropy_weight = float(event_clock_lambda_entropy_weight or 0.0)
        self.event_clock_lambda_prior_weight = float(event_clock_lambda_prior_weight or 0.0)
        self.event_clock_delta_z_l2_weight = float(event_clock_delta_z_l2_weight or 0.0)
        # Layout / rot6d contract.
        self.fps = float(fps)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        self._rot6d_columns = self._resolve_rot6d_columns(self.rot6d_spec)
        layout = self.output_layout or {}
        slices_layout = layout.get('slices')
        inner = slices_layout if isinstance(slices_layout, dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.attn_lambda_local = getattr(self, 'attn_lambda_local', 0.02)
        self.attn_lambda_entropy = getattr(self, 'attn_lambda_entropy', 0.0)

        # Skeleton / cache bootstrap.
        self._warned_messages: set[str] = set()
        self._warned_bad_rot6d = False
        # Tail-risk regularization for per-bone rotation errors (CVaR / top-k style).
        # When enabled, adds an extra term on the worst bones (by mean GeoLocalDeg),
        # which reduces whack-a-mole without requiring explicit per-bone weight tables.
        self.rot_local_tail_weight = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        self.rot_local_tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        self.rot_local_tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all')
        self.rot_local_tail_select = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch')
        self.rot_local_tail_ema_beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        # 缓存几何骨骼权重（按 device/dtype）
        self._joint_weight_cache: dict[tuple, torch.Tensor] = {}
        # Tail-loss 辅助缓存（candidate pool 与选择打分）
        self._tail_candidate_cache: dict[tuple, torch.Tensor] = {}
        self._tail_score_cache: dict[tuple, torch.Tensor] = {}
        self.root_idx = 0
        self.bone_names: list[str] = []
        self._bone_name_to_idx: dict[str, int] = {}
        self.limb_monitor_names: list[str] = [
            'upperarm_l', 'lowerarm_l', 'hand_l',
            'upperarm_r', 'lowerarm_r', 'hand_r',
            'thigh_l', 'calf_l', 'foot_l',
            'thigh_r', 'calf_r', 'foot_r',
        ]
        self._limb_mask_cache: Optional[torch.Tensor] = None
        self._torso_mask_cache: Optional[torch.Tensor] = None
        self._limb_mask_joint_count: Optional[int] = None
        skeleton = self.meta.get('skeleton') if isinstance(self.meta, dict) else None
        self.parents: list[int] = []
        self._parents_tensor: Optional[torch.Tensor] = None
        self.bone_offsets: Optional[torch.Tensor] = None
        if isinstance(skeleton, dict):
            parents = skeleton.get('parents')
            if isinstance(parents, (list, tuple)):
                self.parents = [int(p) for p in parents]
                try:
                    self.root_idx = max(0, self.parents.index(-1))
                except ValueError:
                    self.root_idx = 0
            if 'ref_local_offsets_m' in skeleton:
                offsets = skeleton.get('ref_local_offsets_m')
                if offsets is not None:
                    try:
                        offsets_tensor = torch.as_tensor(offsets, dtype=torch.float32)
                    except (RuntimeError, TypeError, ValueError) as exc:
                        raise TypeError(
                            "skeleton.ref_local_offsets_m must be convertible to a float32 tensor with shape=(num_joints, 3); "
                            f"got value={offsets!r} (type={type(offsets).__name__})."
                        ) from exc
                    actual_shape = tuple(int(v) for v in offsets_tensor.shape)
                    if offsets_tensor.ndim != 2 or offsets_tensor.shape[-1] != 3:
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must have shape=(num_joints, 3); "
                            f"got actual_shape={actual_shape!r} (type={type(offsets).__name__})."
                        )
                    if self.parents and offsets_tensor.shape[0] != len(self.parents):
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must have shape=(num_joints, 3) with num_joints matching "
                            f"len(parents)={len(self.parents)}; got actual_shape={actual_shape!r}."
                        )
                    if not bool(torch.isfinite(offsets_tensor).all().detach().cpu().item()):
                        raise ValueError(
                            "skeleton.ref_local_offsets_m must contain only finite values; "
                            f"got actual_shape={actual_shape!r}."
                        )
                    self.bone_offsets = offsets_tensor

        # Orchestration tracker defaults.
        self._loss_group_totals: Dict[str, float] = {}
        self._loss_group_alias = {
            'attn': 'aux',
            'rot_ortho': 'core',
            'rot_local': 'core',
            'root_vel': 'core',
            'root_speed': 'core',
            'direct_pose': 'core',
        }

        # skeleton parents may be set later via set_skeleton; avoid early fallback here

    def _warn_once(
        self,
        key: str,
        message: str,
        *,
        category: type[Warning] = RuntimeWarning,
    ) -> None:
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        warnings.warn(message, category=category, stacklevel=2)

    @staticmethod
    def _resolve_rot6d_columns(spec: Optional[Dict[str, Any]]) -> tuple[str, str]:
        if isinstance(spec, dict):
            cols = spec.get('columns')
            if isinstance(cols, (list, tuple)) and len(cols) >= 2:
                a = str(cols[0]).strip().upper()
                b = str(cols[1]).strip().upper()
                if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                    return a, b
        return ("X", "Z")

    # === future: train/loss/skeleton_weights.py ===
    # Skeleton state / cache invalidation.
    def set_bone_names(self, names: Optional[Sequence[str]]) -> None:
        self.bone_names = [str(n) for n in (names or [])]
        self._bone_name_to_idx = {name: idx for idx, name in enumerate(self.bone_names)}
        self._limb_mask_cache = None
        self._torso_mask_cache = None
        self._limb_mask_joint_count = None
        self._tail_candidate_cache = {}
        self._tail_score_cache = {}
        # reset fk caches when bone count changes
        self._parents_tensor = None

    def set_skeleton(self, parents: Optional[Sequence[int]], offsets: Optional[Sequence[Sequence[float]]]) -> None:
        next_parents = self.parents
        if parents is not None:
            next_parents = [int(p) for p in parents]
        next_offsets = self.bone_offsets
        if offsets is not None:
            try:
                offsets_tensor = torch.as_tensor(offsets, dtype=torch.float32)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise TypeError(
                    "offsets must be convertible to a float32 tensor with shape=(num_joints, 3); "
                    f"got value={offsets!r} (type={type(offsets).__name__})."
                ) from exc
            actual_shape = tuple(int(v) for v in offsets_tensor.shape)
            if offsets_tensor.ndim != 2 or offsets_tensor.shape[-1] != 3:
                raise ValueError(
                    "offsets must have shape=(num_joints, 3); "
                    f"got actual_shape={actual_shape!r} (type={type(offsets).__name__})."
                )
            if next_parents and offsets_tensor.shape[0] != len(next_parents):
                raise ValueError(
                    "offsets must have shape=(num_joints, 3) with num_joints matching "
                    f"len(parents)={len(next_parents)}; got actual_shape={actual_shape!r}."
                )
            if not bool(torch.isfinite(offsets_tensor).all().detach().cpu().item()):
                raise ValueError(
                    "offsets must contain only finite values; "
                    f"got actual_shape={actual_shape!r}."
                )
            next_offsets = offsets_tensor
        if parents is not None:
            self.parents = next_parents
            self._parents_tensor = None
        if offsets is not None:
            self.bone_offsets = next_offsets
        self._tail_candidate_cache = {}

    def _invalidate_weight_cache(self) -> None:
        self._joint_weight_cache = {}

    # Mask resolution / skeleton-derived stats.
    def _resolve_named_joint_indices(
        self,
        *,
        joint_count: int,
        spec: Optional[Sequence[str] | str],
        default_items: Sequence[str],
    ) -> list[int]:
        if joint_count <= 0 or not self.bone_names or joint_count > len(self.bone_names):
            return []
        indices, _ = _resolve_joint_spec_indices(
            spec,
            default_items=default_items,
            bone_names=self.bone_names[:joint_count],
            joint_count=joint_count,
        )
        return indices

    def _resolve_limb_masks(self, joint_count: int, device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        import torch
        if joint_count <= 0:
            return None
        names = self.bone_names
        if not names or joint_count > len(names):
            return None
        monitor = getattr(self, 'limb_monitor_names', None) or []
        if not monitor:
            return None
        idx_map = {name: idx for idx, name in enumerate(names[:joint_count])}
        limb_indices = [idx_map[name] for name in monitor if name in idx_map]
        if not limb_indices:
            return None
        if self._limb_mask_joint_count != joint_count or self._limb_mask_cache is None:
            mask = torch.zeros(joint_count, dtype=torch.bool)
            mask[limb_indices] = True
            self._limb_mask_cache = mask
            self._torso_mask_cache = (~mask).clone()
            self._limb_mask_joint_count = joint_count
        limb_mask = self._limb_mask_cache.to(device=device)
        torso_mask = self._torso_mask_cache.to(device=device)
        if not torso_mask.any():
            torso_mask = (~limb_mask).clone()
            if torso_mask.numel() == 0 or not torso_mask.any():
                return None
        return limb_mask, torso_mask

    def _collect_limb_local_stats(self, geo_tensor: torch.Tensor) -> Dict[str, float]:
        import torch, math
        if geo_tensor is None or geo_tensor.numel() == 0:
            return {}
        J = geo_tensor.shape[-1]
        masks = self._resolve_limb_masks(J, geo_tensor.device)
        if not masks:
            return {}
        limb_mask, torso_mask = masks
        flat = geo_tensor.reshape(-1, J)
        joint_mean = torch.nanmean(flat, dim=0)
        stats: Dict[str, float] = {}
        deg = 180.0 / math.pi
        limb_val = torso_val = None
        if limb_mask.any():
            limb_val = joint_mean[limb_mask].mean()
            stats['rot_local_limb_deg'] = float((limb_val * deg).detach().cpu())
            stats['rot_local_limb_count'] = int(limb_mask.sum().item())
        if torso_mask.any():
            torso_val = joint_mean[torso_mask].mean()
            stats['rot_local_torso_deg'] = float((torso_val * deg).detach().cpu())
            stats['rot_local_torso_count'] = int(torso_mask.sum().item())
        if limb_val is not None and torso_val is not None:
            ratio = limb_val / torso_val.clamp_min(1e-6)
            stats['rot_local_limb_over_torso'] = float(ratio.detach().cpu())
        return stats

    # Weight computation.
    def _joint_weight_vector(self, device, dtype, joint_count: int) -> torch.Tensor:
        """
        influence = self_scale * |offset| + (sum lever_arm_to_descendants) ** power
        clamp to min_w, optional visual importance; normalize mean=1.
        """
        key = (str(device), str(dtype), int(joint_count))
        cache = getattr(self, '_joint_weight_cache', None)
        if cache is None:
            cache = {}
            self._joint_weight_cache = cache
        if key in cache:
            return cache[key]

        # 基于几何的统一权重
        weights_cpu = self._compute_unified_weights_cpu(joint_count)

        weights = weights_cpu.to(device=device, dtype=dtype)
        cache[key] = weights
        if not hasattr(self, '_weight_vector_logged'):
            self._weight_vector_logged = True
            self._warn_once(
                "unified_weight_vector",
                f"[Loss][UnifiedWeights] range=[{weights.min():.3f}, {weights.max():.3f}] "
                f"mean={weights.mean():.3f} std={weights.std():.3f} "
                f"power={getattr(self, 'unified_downstream_power', 0.6)} "
                f"self_scale={getattr(self, 'unified_self_scale', 1.5)} "
                f"min_w={getattr(self, 'unified_min_weight', 0.05)} "
                f"visual=False",
                category=UserWarning,
            )
        return weights

    def _compute_unified_weights_cpu(self, joint_count: int) -> torch.Tensor:
        power = float(getattr(self, 'unified_downstream_power', 0.6))
        self_scale = float(getattr(self, 'unified_self_scale', 1.5))
        min_w = float(getattr(self, 'unified_min_weight', 0.05))

        if not self.parents or self.bone_offsets is None or len(self.parents) < joint_count:
            return torch.ones(joint_count, dtype=torch.float32)

        J = joint_count
        parents = self.parents[:J]
        offsets = self.bone_offsets[:J].detach().cpu().float()

        global_pos = torch.zeros(J, 3, dtype=torch.float32)
        for j in range(J):
            p = parents[j]
            if p < 0:
                global_pos[j] = offsets[j]
            else:
                global_pos[j] = global_pos[p] + offsets[j]

        children = [[] for _ in range(J)]
        for j, p in enumerate(parents):
            if p >= 0 and p < J:
                children[p].append(j)

        def _descendants(idx: int):
            qs = [idx]
            seen = set()
            out = []
            while qs:
                cur = qs.pop(0)
                for c in children[cur]:
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
                        qs.append(c)
            return out

        weights = torch.zeros(J, dtype=torch.float32)
        for i in range(J):
            self_len = torch.norm(offsets[i])
            bone_i = global_pos[i]
            down = 0.0
            for j in _descendants(i):
                end = global_pos[j] + offsets[j]
                lever = torch.norm(end - bone_i)
                down += float(lever)
            down_scaled = (down ** power) if down > 0 else 0.0
            weights[i] = self_scale * self_len + down_scaled

        weights = weights / weights.mean().clamp_min(1e-6)
        weights = torch.clamp(weights, min=min_w)
        weights = weights / weights.mean().clamp_min(1e-6)

        return weights

    # Tail-risk selection helpers.
    def _rot_local_tail_scores(self, per_bone: torch.Tensor) -> torch.Tensor:
        """
        Selection scores for tail top-k:
        - batch: current batch mean (default)
        - ema:   exponential moving average across batches for smoother selection
        """
        mode = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch').lower()
        if mode != 'ema':
            return per_bone
        beta = float(getattr(self, 'rot_local_tail_ema_beta', 0.9) or 0.9)
        beta = max(0.0, min(0.999, beta))
        key = (str(per_bone.device), int(per_bone.numel()))
        cache = getattr(self, '_tail_score_cache', None)
        if cache is None:
            cache = {}
            self._tail_score_cache = cache
        prev = cache.get(key)
        if prev is None or (not torch.is_tensor(prev)) or prev.shape != per_bone.shape or prev.device != per_bone.device:
            score = per_bone.detach().clone()
        else:
            score = prev
            score.mul_(beta).add_(per_bone.detach(), alpha=(1.0 - beta))
        cache[key] = score
        return score

    def _rot_local_tail_candidates(self, scope: str, joint_count: int, device: torch.device, *, k: int) -> Optional[torch.Tensor]:
        """
        Candidate joint indices for tail selection. Uses:
        - explicit name list (limb_monitor_names / pelvis) when bone_names are known
        - skeleton leaves (end-effectors) as a dynamic fallback/augmentation
        """
        scope_norm = str(scope or 'all').lower()
        if scope_norm in ('all', '*'):
            return None
        if scope_norm not in ('limbs', 'limb', 'keybones', 'key_bones'):
            return None
        cache = getattr(self, '_tail_candidate_cache', None)
        if cache is None:
            cache = {}
            self._tail_candidate_cache = cache
        key = (scope_norm, int(joint_count), str(device))
        cached = cache.get(key)
        if cached is not None and torch.is_tensor(cached) and cached.device == device and cached.numel() > 0:
            return cached

        J = int(joint_count)
        idxs: list[int] = []

        # 1) Name-driven candidates (keeps old behavior when names match)
        name_to_idx = getattr(self, '_bone_name_to_idx', None)
        if isinstance(name_to_idx, dict) and name_to_idx:
            monitor = list(getattr(self, 'limb_monitor_names', None) or [])
            if scope_norm in ('keybones', 'key_bones'):
                monitor = ['pelvis'] + monitor
            for nm in monitor:
                idx = name_to_idx.get(str(nm))
                if isinstance(idx, int) and 0 <= idx < J:
                    idxs.append(int(idx))

        # 2) Skeleton-driven candidates: leaf joints (end-effectors) + root for keybones
        parents = getattr(self, 'parents', None)
        if isinstance(parents, list) and len(parents) >= J:
            child_counts = [0] * J
            for j, p in enumerate(parents[:J]):
                if isinstance(p, int) and 0 <= p < J:
                    child_counts[p] += 1
            leaves = [j for j in range(J) if child_counts[j] == 0]

            # Avoid selecting too many tiny leaves (e.g. fingers) as candidates.
            max_leaf = int(max(16, 4 * max(1, int(k))))
            if len(leaves) > max_leaf:
                w = self._compute_unified_weights_cpu(J)
                vals = w.index_select(0, torch.as_tensor(leaves, dtype=torch.long))
                keep = min(max_leaf, int(vals.numel()))
                if keep > 0:
                    _, sel = torch.topk(vals, k=keep, largest=True, sorted=False)
                    leaves = [leaves[int(i)] for i in sel.tolist()]
                else:
                    leaves = []

            if scope_norm in ('keybones', 'key_bones'):
                root_idx = int(getattr(self, 'root_idx', 0))
                if 0 <= root_idx < J:
                    idxs.append(root_idx)
            idxs.extend(leaves)

        # De-dup preserving order
        if not idxs:
            return None
        seen = set()
        idxs = [i for i in idxs if 0 <= i < J and not (i in seen or seen.add(i))]
        if not idxs:
            return None
        out = torch.as_tensor(idxs, device=device, dtype=torch.long)
        cache[key] = out
        return out

    # FK-relative rotation views.
    def _parent_relative_matrices(self, R: torch.Tensor) -> torch.Tensor:
        parents = getattr(self, 'parents', None)
        if not parents:
            return R
        J = int(R.shape[-3])
        if len(parents) < J or J <= 0:
            return R
        parents_tensor = getattr(self, '_parents_tensor', None)
        if parents_tensor is None or parents_tensor.device != R.device or parents_tensor.numel() < J:
            parents_tensor = torch.as_tensor(parents[:J], device=R.device, dtype=torch.long)
            self._parents_tensor = parents_tensor
        else:
            parents_tensor = parents_tensor[:J]
        return parent_relative_matrices(R, parents_tensor)

    def _root_relative(self, R: torch.Tensor) -> torch.Tensor:
        root_idx = int(getattr(self, 'root_idx', 0))
        return root_relative_matrices(R, root_idx)

    def _forward_base_inner(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor, attn_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            pred_motion: [B,T,D] or [T,D] or [B,D]
            gt_motion:   同形状
            attn_weights: None 或 [B,H,T,T]/[L,B,H,T,T] 或 list/tuple/dict 的任意嵌套
        返回:
            loss 标量, 分项 dict (float)
        """
        pm, gm = (pred_motion, gt_motion)
        assert pm.shape == gm.shape, f'pred/gt shape mismatch: {pm.shape} vs {gm.shape}'

        if attn_weights is not None:
            l_attn = self.compute_attention_regularization(attn_weights, geomask=None)
        else:
            l_attn = gm.new_zeros(())

        loss = self.w_attn_reg * l_attn
        self._accumulate_loss_contrib('attn', l_attn, self.w_attn_reg, group='aux')

        stats: Dict[str, float] = {
            'attn': float(l_attn.detach().cpu()),
            'rot_ortho': 0.0,
            'rot_ortho_raw': 0.0,
        }
        return loss, stats

    def _slice_if_exists(self, name: str, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        从 self.group_slices 中获取预先解析好的 slice，并应用于张量。
        """
        sl = self.group_slices.get(name)

        # 因为 self.group_slices 只包含 slice 对象，所以只需做一次类型检查即可。
        if isinstance(sl, slice):
            return X[..., sl]

        return None

    def compute_attention_regularization(self, attn_weights, geomask=None):
        """
        返回一个标量 loss：
        - 支持 Tensor: [B,H,T,T] 或 [L,B,H,T,T]
        - 支持 list/tuple/dict: 递归展开后逐个累加
        - geomask: None 或可广播到 [..., T, T] 的掩码（1=允许区域，0=不鼓励区域）
        """
        if attn_weights is None:
            if geomask is not None and torch.is_tensor(geomask):
                return torch.zeros((), device=geomask.device, dtype=geomask.dtype)
            return torch.tensor(0.0)

        def _flatten_items(x):
            if torch.is_tensor(x):
                return [x]
            if isinstance(x, (list, tuple)):
                items = []
                for y in x:
                    items.extend(_flatten_items(y))
                return items
            if isinstance(x, dict):
                items = []
                for y in x.values():
                    items.extend(_flatten_items(y))
                return items
            return []
        items = _flatten_items(attn_weights)
        if len(items) == 0:
            return torch.tensor(0.0, device=geomask.device if geomask is not None and torch.is_tensor(geomask) else None)
        loss_total = None
        count = 0
        for A in items:
            if A.dim() == 5:
                L, B, H, T, _ = A.shape
                A = A.reshape(L * B * H, T, T)
            elif A.dim() == 4:
                B, H, T, _ = A.shape
                A = A.reshape(B * H, T, T)
            elif A.dim() == 3:
                T = A.shape[-1]
            else:
                continue
            A = A.float()
            A = A / A.sum(-1, keepdim=True).clamp_min(1e-06)
            device = A.device
            if geomask is not None:
                if torch.is_tensor(geomask):
                    gm = geomask
                    gm_dim = int(gm.dim())
                    if gm_dim not in (2, 3, 4):
                        raise ValueError(
                            "geomask for attention regularization must be a tensor with rank 2, 3, or 4 "
                            "broadcastable to attention weights shaped (..., T, T); "
                            f"got actual_shape={tuple(int(v) for v in gm.shape)!r}, actual_ndim={gm_dim}, T={int(T)}."
                        )
                    try:
                        M = 1.0 - gm
                        loss_local = (A * M).mean()
                    except RuntimeError as direct_exc:
                        if gm_dim == 2:
                            try:
                                gm = gm.view(1, T, T)
                            except RuntimeError as reshape_exc:
                                raise RuntimeError(
                                    "geomask rank-2 fallback reshape failed in attention regularization: "
                                    "expected geomask shape (T, T) before view(1, T, T); "
                                    f"got geomask_shape={tuple(int(v) for v in geomask.shape)!r}, "
                                    f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                                ) from reshape_exc
                        elif gm_dim == 4:
                            gm = gm.mean(0)
                        else:
                            raise RuntimeError(
                                "geomask broadcast failed in attention regularization: "
                                "expected rank-3 geomask broadcastable to attention weights shaped (N, T, T); "
                                f"got geomask_shape={tuple(int(v) for v in gm.shape)!r}, "
                                f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                            ) from direct_exc
                        try:
                            M = 1.0 - gm
                            loss_local = (A * M).mean()
                        except RuntimeError as fallback_exc:
                            raise RuntimeError(
                                "geomask fallback broadcast failed in attention regularization: "
                                "expected rank-2 shape (T, T), rank-3 shape broadcastable to (N, T, T), "
                                "or rank-4 reducible via mean(0); "
                                f"got original_geomask_shape={tuple(int(v) for v in geomask.shape)!r}, "
                                f"fallback_geomask_shape={tuple(int(v) for v in gm.shape)!r}, "
                                f"attention_shape={tuple(int(v) for v in A.shape)!r}, T={int(T)}."
                            ) from fallback_exc
                else:
                    idx = torch.arange(T, device=device)
                    Dmat = (idx[None, :] - idx[:, None]).abs().float()
                    Dmat = Dmat / Dmat.max().clamp_min(1.0)
                    loss_local = (A * Dmat).mean()
            else:
                idx = torch.arange(T, device=device)
                Dmat = (idx[None, :] - idx[:, None]).abs().float()
                Dmat = Dmat / Dmat.max().clamp_min(1.0)
                loss_local = (A * Dmat).mean()
            Aeps = A.clamp_min(1e-06)
            entropy = -(Aeps * Aeps.log()).sum(-1).mean()
            loss_attn = self.attn_lambda_local * loss_local + self.attn_lambda_entropy * -entropy
            loss_total = loss_attn if loss_total is None else loss_total + loss_attn
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=items[0].device)
        return loss_total / count

    # === future: train/loss_rot6d.py ===
    # Rot6D slice / denorm / matrix helpers.
    def _maybe_get_rot6d(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        若存在 "BoneRotations6D" 切片，则返回该切片；否则 None。
        """
        rot = self._slice_if_exists('BoneRotations6D', X)
        return rot

    def _denorm_rot6d_flat(self, rot6d_flat: torch.Tensor) -> tuple[torch.Tensor, bool]:
        sl = self.group_slices.get('BoneRotations6D', None)
        if (
            not isinstance(sl, slice)
            or getattr(self, 'mu_y', None) is None
            or getattr(self, 'std_y', None) is None
        ):
            return rot6d_flat, False
        start = int(sl.start or 0)
        stop = int(sl.stop or start)
        width = max(0, stop - start)
        if width != int(rot6d_flat.shape[-1]):
            return rot6d_flat, False
        mu = torch.as_tensor(self.mu_y, device=rot6d_flat.device, dtype=rot6d_flat.dtype)[..., start:stop]
        std = torch.as_tensor(self.std_y, device=rot6d_flat.device, dtype=rot6d_flat.dtype)[..., start:stop]
        std = std.clamp(min=1e-6)
        while mu.dim() < rot6d_flat.dim():
            mu = mu.unsqueeze(0)
            std = std.unsqueeze(0)
        return rot6d_flat * std + mu, True

    def _extract_rot6d_flat(
        self,
        X: torch.Tensor,
        *,
        denorm: bool = True,
        reproject: bool = False,
        sanitize: bool = False,
    ) -> Optional[torch.Tensor]:
        rot6d = self._maybe_get_rot6d(X)
        if rot6d is None:
            return None
        if int(rot6d.shape[-1]) % 6 != 0:
            return None
        if denorm:
            rot6d, hit = self._denorm_rot6d_flat(rot6d)
            if hit and not hasattr(self, "_train_denorm_hit"):
                self._train_denorm_hit = True
                self._warn_once(
                    "train_denorm_rot6d_flat",
                    "[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.",
                    category=UserWarning,
                )
        if sanitize:
            rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=1.0, neginf=-1.0)
        if reproject:
            rot6d = reproject_rot6d(rot6d)
        return rot6d

    def _extract_rot6d_joint6(
        self,
        X: torch.Tensor,
        *,
        denorm: bool = True,
        reproject: bool = True,
        sanitize: bool = True,
    ) -> Optional[torch.Tensor]:
        rot6d = self._extract_rot6d_flat(
            X,
            denorm=denorm,
            reproject=reproject,
            sanitize=sanitize,
        )
        if rot6d is None:
            return None
        joint_count = int(rot6d.shape[-1]) // 6
        return rot6d.view(*rot6d.shape[:-1], joint_count, 6)

    def _extract_rot6d_mats(
        self,
        X: torch.Tensor,
        *,
        denorm: bool = True,
        reproject: bool = True,
        sanitize: bool = True,
    ) -> Optional[torch.Tensor]:
        rot6d = self._extract_rot6d_joint6(
            X,
            denorm=denorm,
            reproject=reproject,
            sanitize=sanitize,
        )
        if rot6d is None:
            return None
        return rot6d_to_matrix(rot6d)

    # Rot6D objective helpers.
    def compute_rot6d_ortho_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Ortho penalty on **raw 6D** (pre-GS):
        encourage unit-length columns and mutual orthogonality.
        This must NOT use rot6d_to_matrix (which orthonormalizes and would yield ~0 loss).
        """
        Z = lambda v: pred.new_tensor(float(v))
        pr_raw = self._maybe_get_rot6d(pred)  # (..., D) or None
        if pr_raw is None:
            return Z(0.0)
        D = pr_raw.shape[-1]
        if D % 6 != 0:
            if not getattr(self, '_warned_bad_rot6d_ortho', False):
                self._warned_bad_rot6d_ortho = True
                self._warn_once(
                    "bad_rot6d_ortho_dim",
                    f"[Loss][WARN] BoneRotations6D slice dim={D} not multiple of 6. Skip rot6d_ortho.",
                )
            return Z(0.0)
        a6 = self._extract_rot6d_joint6(pred, denorm=False, reproject=False, sanitize=False)
        if a6 is None:
            return Z(0.0)
        v1 = a6[..., 0:3]
        v2 = a6[..., 3:6]
        len_p = (v1.norm(dim=-1) - 1.0).pow(2) + (v2.norm(dim=-1) - 1.0).pow(2)
        ortho_p = (v1.mul(v2).sum(dim=-1)).pow(2)
        return (len_p + ortho_p).mean()

    def compute_rot6d_geo_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        return_per_joint: bool = False,
    ):
        Z = lambda v: gt.new_tensor(float(v))
        sl = self.group_slices.get('BoneRotations6D', None)

        pr_raw = self._maybe_get_rot6d(pred)
        gr_raw = self._maybe_get_rot6d(gt)
        if pr_raw is None or gr_raw is None:
            return Z(0.0)
        D = int(pr_raw.shape[-1])
        if D % 6 != 0:
            if not self._warned_bad_rot6d:
                self._warned_bad_rot6d = True
                self._warn_once(
                    "bad_rot6d_geo_dim",
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). "
                    f"slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.",
                )
            return Z(0.0)
        if int(gr_raw.shape[-1]) != D:
            return Z(0.0)
        pr = self._extract_rot6d_joint6(pred, denorm=True, reproject=True, sanitize=False)
        gr = self._extract_rot6d_joint6(gt, denorm=True, reproject=True, sanitize=False)
        if pr is None or gr is None:
            return Z(0.0)

        Rp = rot6d_to_matrix(pr)
        Rg = rot6d_to_matrix(gr)
        theta = geodesic_R(Rp, Rg)

        joint_count = int(pr.shape[-2])
        weights = self._joint_weight_vector(theta.device, theta.dtype, joint_count)
        view_shape = (1,) * (theta.dim() - 1) + (joint_count,)
        w = weights.view(*view_shape)
        theta_weighted = theta * w
        loss_val = theta_weighted.mean()
        if return_per_joint:
            return loss_val, theta, weights
        return loss_val

    def compute_rot6d_log_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: pred.new_tensor(float(v))
        Rp = self._extract_rot6d_mats(pred)
        Rg = self._extract_rot6d_mats(gt)
        if Rp is None or Rg is None:
            return Z(0.0)
        if Rp.dim() < 5:
            return Z(0.0)
        if int(Rp.shape[-4]) < 2:
            return Z(0.0)
        log_p = angvel_vec_from_R_seq(Rp, fps=1.0)
        log_g = angvel_vec_from_R_seq(Rg, fps=1.0)
        return torch.nn.functional.smooth_l1_loss(log_p, log_g)

    # === future: train/loss/direct_pose.py ===
    # Stats contract / pair normalization.
    def _direct_pose_default_stats(self) -> Dict[str, float]:
        feature_config = self._resolve_direct_pose_feature_config()
        return _build_direct_pose_default_stats(feature_config)

    def _resolve_direct_pose_feature_config(self) -> _DirectPoseFeatureConfig:
        return _DirectPoseFeatureConfig(
            leg_split_enable=bool(self.direct_pose_loss_leg_split),
            arm_split_enable=bool(self.direct_pose_arm_split_enable),
            arm_else_balance_enable=bool(self.direct_pose_loss_arm_else_balance_enable),
            arm_weight=float(self.direct_pose_loss_arm_weight),
            else_weight=float(self.direct_pose_loss_else_weight),
            group_norm_enable=bool(self.direct_pose_loss_group_norm_enable),
            group_norm_w_leg=float(self.direct_pose_loss_group_norm_w_leg),
            group_norm_w_nonleg=float(self.direct_pose_loss_group_norm_w_nonleg),
            group_norm_beta=float(self.direct_pose_loss_group_norm_ema_beta),
            group_norm_ratio_min=float(self.direct_pose_loss_group_norm_ratio_min),
            group_norm_ratio_max=float(self.direct_pose_loss_group_norm_ratio_max),
            group_norm_eps=float(self.direct_pose_loss_group_norm_eps),
        )

    def _resolve_direct_group_masks(self, joint_count: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        import torch

        if joint_count <= 0 or not self.bone_names or joint_count > len(self.bone_names):
            return None
        leg_idx = self._resolve_named_joint_indices(
            joint_count=joint_count,
            spec=self.direct_pose_leg_bones,
            default_items=DEFAULT_DIRECT_POSE_LEG_BONES,
        )
        arm_idx = self._resolve_named_joint_indices(
            joint_count=joint_count,
            spec=self.direct_pose_arm_bones,
            default_items=STAGE6_3WAY_ARMCHAIN_BONES,
        )
        leg_mask = torch.zeros(joint_count, dtype=torch.bool, device=device)
        arm_mask = torch.zeros(joint_count, dtype=torch.bool, device=device)
        if leg_idx:
            leg_mask[torch.as_tensor(leg_idx, dtype=torch.long, device=device)] = True
        if arm_idx:
            arm_mask[torch.as_tensor(arm_idx, dtype=torch.long, device=device)] = True
        nonleg_mask = ~leg_mask
        arm_mask = arm_mask & nonleg_mask
        else_mask = nonleg_mask & (~arm_mask)
        all_ex_root = torch.ones(joint_count, dtype=torch.bool, device=device)
        root_idx = int(getattr(self, 'root_idx', 0) or 0)
        if 0 <= root_idx < joint_count:
            all_ex_root[root_idx] = False
            leg_mask[root_idx] = False
            nonleg_mask[root_idx] = False
            arm_mask[root_idx] = False
            else_mask[root_idx] = False
        return {
            'all_ex_root': all_ex_root,
            'leg': leg_mask,
            'nonleg': nonleg_mask,
            'arm': arm_mask,
            'else': else_mask,
            'trunk': else_mask,
        }

    def _prepare_direct_pose_pair(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> Optional[_DirectPosePair]:
        if direct.dim() == 2 and gt_motion.dim() == 3:
            direct = direct.unsqueeze(1)
        gt_direct = gt_motion
        if gt_direct.dim() == 2 and direct.dim() == 3:
            gt_direct = gt_direct.unsqueeze(1)
        if direct.dim() != 3 or gt_direct.dim() != 3:
            return None
        steps = min(int(direct.shape[1]), int(gt_direct.shape[1]))
        if steps <= 0:
            return None
        return _DirectPosePair(direct_seq=direct[:, :steps], gt_direct=gt_direct[:, :steps])

    def _resolve_direct_pose_group_payload_weight(
        self,
        field_name: str,
        raw_value: Optional[float],
        default_value: float,
    ) -> float:
        candidate = default_value if raw_value is None else raw_value
        try:
            resolved = float(candidate)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{field_name} must be a finite scalar in range (0, inf) for direct-pose group payload; "
                f"got value={raw_value!r} (type={type(raw_value).__name__})."
            ) from exc
        if (not _math.isfinite(resolved)) or resolved <= 0.0:
            raise ValueError(
                f"{field_name} must be a finite scalar in range (0, inf) for direct-pose group payload; "
                f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
            )
        return float(resolved)

    def _resolve_direct_pose_group_base_terms(
        self,
        request: _DirectPoseGroupBaseRequest,
    ) -> Optional[_DirectPoseGroupBaseTerms]:
        terms = _DirectPoseGroupBaseTerms(
            dir_base=request.dir_base,
            dir_leg_base=request.dir_leg_base,
            dir_nonleg_base=request.dir_nonleg_base,
            dir_arm_base=request.dir_arm_base,
            dir_else_base=request.dir_else_base,
        )
        if not torch.is_tensor(request.geo_theta):
            return terms if terms.has_any_tensor() else None

        geo_theta = request.geo_theta
        split_masks = self._resolve_direct_group_masks(int(geo_theta.shape[-1]), geo_theta.device)
        if split_masks is None:
            return None
        terms = _DirectPoseGroupBaseTerms(
            dir_base=terms.dir_base if torch.is_tensor(terms.dir_base) else _masked_group_mean(geo_theta, split_masks.get('all_ex_root')),
            dir_leg_base=terms.dir_leg_base if torch.is_tensor(terms.dir_leg_base) else _masked_group_mean(geo_theta, split_masks.get('leg')),
            dir_nonleg_base=terms.dir_nonleg_base if torch.is_tensor(terms.dir_nonleg_base) else _masked_group_mean(geo_theta, split_masks.get('nonleg')),
            dir_arm_base=terms.dir_arm_base if torch.is_tensor(terms.dir_arm_base) else _masked_group_mean(geo_theta, split_masks.get('arm')),
            dir_else_base=terms.dir_else_base if torch.is_tensor(terms.dir_else_base) else _masked_group_mean(geo_theta, split_masks.get('else')),
        )
        return terms if terms.has_any_tensor() else None

    def _compute_direct_pose_group_base_from_request(
        self,
        request: _DirectPoseGroupBaseRequest,
        *,
        feature_config: _DirectPoseFeatureConfig,
    ) -> Optional[_DirectPoseGroupBaseResult]:
        def _ratio_stat(
            numerator: Optional[torch.Tensor],
            denominator: Optional[torch.Tensor],
        ) -> float:
            if not torch.is_tensor(numerator) or not torch.is_tensor(denominator):
                return float('nan')
            return _stats_float(numerator / denominator.clamp_min(eps_value))

        terms = self._resolve_direct_pose_group_base_terms(request)
        if terms is None:
            return None

        eps_value = float(feature_config.group_norm_eps if request.eps is None else request.eps)
        if (not _math.isfinite(eps_value)) or eps_value <= 0.0:
            eps_value = 1e-6
        arm_split_active = (
            bool(feature_config.arm_split_enable)
            if request.arm_split_enable is None
            else bool(request.arm_split_enable)
        )
        arm_else_balance_flag = (
            bool(feature_config.arm_else_balance_enable)
            if request.arm_else_balance_enable is None
            else bool(request.arm_else_balance_enable)
        )
        arm_w = self._resolve_direct_pose_group_payload_weight(
            "arm_weight",
            request.arm_weight,
            feature_config.arm_weight,
        )
        else_w = self._resolve_direct_pose_group_payload_weight(
            "else_weight",
            request.else_weight,
            feature_config.else_weight,
        )

        dir_nonleg_effective_base = terms.dir_nonleg_base
        arm_else_balance_active = 0.0
        if (
            arm_else_balance_flag
            and arm_split_active
            and torch.is_tensor(terms.dir_arm_base)
            and torch.is_tensor(terms.dir_else_base)
        ):
            denom = max(eps_value, arm_w + else_w)
            dir_nonleg_effective_base = (terms.dir_arm_base * arm_w + terms.dir_else_base * else_w) / denom
            arm_else_balance_active = 1.0

        return _DirectPoseGroupBaseResult(
            dir_base=terms.dir_base,
            dir_leg_base=terms.dir_leg_base,
            dir_nonleg_base=terms.dir_nonleg_base,
            dir_nonleg_effective_base=dir_nonleg_effective_base,
            dir_arm_base=terms.dir_arm_base,
            dir_else_base=terms.dir_else_base,
            leg_over_nonleg=_ratio_stat(terms.dir_leg_base, terms.dir_nonleg_base),
            leg_over_nonleg_effective=_ratio_stat(terms.dir_leg_base, dir_nonleg_effective_base),
            arm_over_else=_ratio_stat(terms.dir_arm_base, terms.dir_else_base),
            direct_pose_arm_else_balance_active=float(arm_else_balance_active),
            direct_pose_loss_arm_weight=float(arm_w),
            direct_pose_loss_else_weight=float(else_w),
        )

    # Group base payload.
    def _compute_direct_pose_group_base_payload(
        self,
        *,
        geo_theta: Optional[torch.Tensor] = None,
        dir_base: Optional[torch.Tensor] = None,
        dir_leg_base: Optional[torch.Tensor] = None,
        dir_nonleg_base: Optional[torch.Tensor] = None,
        dir_arm_base: Optional[torch.Tensor] = None,
        dir_else_base: Optional[torch.Tensor] = None,
        arm_split_enable: Optional[bool] = None,
        arm_else_balance_enable: Optional[bool] = None,
        arm_weight: Optional[float] = None,
        else_weight: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        feature_config = self._resolve_direct_pose_feature_config()
        result = self._compute_direct_pose_group_base_from_request(
            _DirectPoseGroupBaseRequest(
                geo_theta=geo_theta,
                dir_base=dir_base,
                dir_leg_base=dir_leg_base,
                dir_nonleg_base=dir_nonleg_base,
                dir_arm_base=dir_arm_base,
                dir_else_base=dir_else_base,
                arm_split_enable=arm_split_enable,
                arm_else_balance_enable=arm_else_balance_enable,
                arm_weight=arm_weight,
                else_weight=else_weight,
                eps=eps,
            ),
            feature_config=feature_config,
        )
        return result.as_payload() if result is not None else None

    # Group norm public wrapper / EMA helpers.
    def _direct_pose_group_norm_ema_value(
        self,
        ema_state: Dict[str, Any],
        key: str,
        default_value: torch.Tensor,
    ) -> torch.Tensor:
        candidate = ema_state.get(key, None)
        if torch.is_tensor(candidate):
            candidate = candidate.to(device=default_value.device, dtype=default_value.dtype)
            try:
                if bool(torch.isfinite(candidate).all().detach().cpu().item()):
                    return candidate
            except (RuntimeError, ValueError, TypeError):
                return default_value.detach()
        return default_value.detach()

    # Group norm typed implementation.
    def _compute_direct_pose_group_norm_shared(
        self,
        dir_leg_base: torch.Tensor,
        dir_nonleg_base: torch.Tensor,
        dir_nonleg_effective_base: torch.Tensor,
        *,
        direct_group_w_leg: Optional[float] = None,
        direct_group_w_nonleg: Optional[float] = None,
        direct_group_beta: Optional[float] = None,
        direct_group_ratio_min: Optional[float] = None,
        direct_group_ratio_max: Optional[float] = None,
        direct_group_eps: Optional[float] = None,
        update_ema_state: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        feature_config = self._resolve_direct_pose_feature_config()
        return self._compute_direct_pose_group_norm_result(
            _DirectPoseGroupNormRequest(
                dir_leg_base=dir_leg_base,
                dir_nonleg_base=dir_nonleg_base,
                dir_nonleg_effective_base=dir_nonleg_effective_base,
                direct_group_w_leg=direct_group_w_leg,
                direct_group_w_nonleg=direct_group_w_nonleg,
                direct_group_beta=direct_group_beta,
                direct_group_ratio_min=direct_group_ratio_min,
                direct_group_ratio_max=direct_group_ratio_max,
                direct_group_eps=direct_group_eps,
                update_ema_state=update_ema_state,
            ),
            feature_config=feature_config,
        ).as_tuple()

    def _compute_direct_pose_group_norm_result(
        self,
        request: _DirectPoseGroupNormRequest,
        *,
        feature_config: _DirectPoseFeatureConfig,
    ) -> _DirectPoseGroupNormResult:
        config = self._resolve_direct_pose_group_norm_config(request, feature_config=feature_config)
        ema_state_raw = getattr(self, '_direct_pose_group_norm_ema', None)
        ema_state = dict(ema_state_raw) if isinstance(ema_state_raw, dict) else {}
        metrics = self._compute_direct_pose_group_norm_metrics(
            request=request,
            config=config,
            ema_leg_prev=self._direct_pose_group_norm_ema_value(ema_state, 'leg', request.dir_leg_base),
            ema_non_prev=self._direct_pose_group_norm_ema_value(ema_state, 'nonleg', request.dir_nonleg_effective_base),
        )
        direct_objective = metrics.objective(config)
        payload = metrics.as_stats(config)
        ema_update_payload = dict(
            ema_state,
            leg=(config.beta * metrics.ema_leg_prev + (1.0 - config.beta) * request.dir_leg_base.detach()).detach(),
            nonleg=(
                config.beta * metrics.ema_non_prev + (1.0 - config.beta) * request.dir_nonleg_effective_base.detach()
            ).detach(),
        )
        if request.update_ema_state:
            self._direct_pose_group_norm_ema = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in ema_update_payload.items()
            }
        return _DirectPoseGroupNormResult(direct_objective, payload, ema_update_payload)

    def _resolve_direct_pose_group_norm_config(
        self,
        request: _DirectPoseGroupNormRequest,
        *,
        feature_config: _DirectPoseFeatureConfig,
    ) -> _DirectPoseGroupNormConfig:
        def _resolve_scalar(
            field_name: str,
            value: Optional[float],
            fallback: float,
            *,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            min_inclusive: bool = True,
            max_inclusive: bool = True,
            expected_range: Optional[str] = None,
        ) -> float:
            raw_value = fallback if value is None else value
            try:
                resolved = float(raw_value)
            except (TypeError, ValueError) as exc:
                range_text = f" in range {expected_range}" if expected_range is not None else ""
                raise TypeError(
                    f"{field_name} must be a finite scalar{range_text}; "
                    f"got value={raw_value!r} (type={type(raw_value).__name__})."
                ) from exc
            lower_ok = True
            upper_ok = True
            if min_value is not None:
                lower_ok = resolved >= float(min_value) if min_inclusive else resolved > float(min_value)
            if max_value is not None:
                upper_ok = resolved <= float(max_value) if max_inclusive else resolved < float(max_value)
            if (not _math.isfinite(resolved)) or (not lower_ok) or (not upper_ok):
                range_text = f" in range {expected_range}" if expected_range is not None else ""
                raise ValueError(
                    f"{field_name} must be a finite scalar{range_text}; "
                    f"got value={resolved!r} from raw={raw_value!r} (type={type(raw_value).__name__})."
                )
            return float(resolved)

        eps_value = _resolve_scalar(
            "direct_group_eps",
            request.direct_group_eps,
            feature_config.group_norm_eps,
            min_value=0.0,
            min_inclusive=False,
            expected_range="(0, inf)",
        )
        ratio_min = _resolve_scalar(
            "direct_group_ratio_min",
            request.direct_group_ratio_min,
            feature_config.group_norm_ratio_min,
            min_value=eps_value,
            expected_range=f"[direct_group_eps={eps_value}, inf)",
        )
        ratio_max = _resolve_scalar(
            "direct_group_ratio_max",
            request.direct_group_ratio_max,
            feature_config.group_norm_ratio_max,
            min_value=eps_value,
            expected_range=f"[direct_group_eps={eps_value}, inf)",
        )
        if ratio_min > ratio_max:
            raise ValueError(
                "direct_group_ratio_min/direct_group_ratio_max must satisfy direct_group_ratio_min <= direct_group_ratio_max "
                f"with both finite scalars >= direct_group_eps={eps_value}; got direct_group_ratio_min={ratio_min!r}, "
                f"direct_group_ratio_max={ratio_max!r}."
            )
        beta = _resolve_scalar(
            "direct_group_beta",
            request.direct_group_beta,
            feature_config.group_norm_beta,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=False,
            expected_range="(0.0, 0.9999]",
        )
        w_leg = _resolve_scalar(
            "direct_group_w_leg",
            request.direct_group_w_leg,
            feature_config.group_norm_w_leg,
        )
        w_nonleg = _resolve_scalar(
            "direct_group_w_nonleg",
            request.direct_group_w_nonleg,
            feature_config.group_norm_w_nonleg,
        )
        return _DirectPoseGroupNormConfig(
            w_leg=w_leg,
            w_nonleg=w_nonleg,
            beta=beta,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            eps=eps_value,
        )

    def _compute_direct_pose_group_norm_metrics(
        self,
        *,
        request: _DirectPoseGroupNormRequest,
        config: _DirectPoseGroupNormConfig,
        ema_leg_prev: torch.Tensor,
        ema_non_prev: torch.Tensor,
    ) -> _DirectPoseGroupNormMetrics:
        leg_ratio_raw = request.dir_leg_base / ema_leg_prev.clamp_min(config.eps)
        nonleg_ratio_raw = request.dir_nonleg_effective_base / ema_non_prev.clamp_min(config.eps)
        leg_ratio = leg_ratio_raw.clamp(config.ratio_min, config.ratio_max)
        nonleg_ratio = nonleg_ratio_raw.clamp(config.ratio_min, config.ratio_max)
        return _DirectPoseGroupNormMetrics(
            ema_leg_prev=ema_leg_prev,
            ema_non_prev=ema_non_prev,
            leg_ratio_raw=leg_ratio_raw,
            nonleg_ratio_raw=nonleg_ratio_raw,
            leg_ratio=leg_ratio,
            nonleg_ratio=nonleg_ratio,
            leg_hit_min=(leg_ratio_raw <= config.ratio_min).to(dtype=request.dir_leg_base.dtype),
            leg_hit_max=(leg_ratio_raw >= config.ratio_max).to(dtype=request.dir_leg_base.dtype),
            nonleg_hit_min=(nonleg_ratio_raw <= config.ratio_min).to(dtype=request.dir_nonleg_base.dtype),
            nonleg_hit_max=(nonleg_ratio_raw >= config.ratio_max).to(dtype=request.dir_nonleg_base.dtype),
        )

    # Direct-pose payload public wrapper / typed assembly.
    def _compute_direct_pose_payload(
        self,
        direct: torch.Tensor,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> Optional[tuple[torch.Tensor, Dict[str, Any]]]:
        result = self._compute_direct_pose_payload_from_request(_DirectPosePayloadRequest(
            direct=direct,
            gt_motion=gt_motion,
            deg_per_rad=float(deg_per_rad),
        ))
        return result.as_tuple() if result is not None else None

    def _compute_direct_pose_payload_from_request(
        self,
        request: _DirectPosePayloadRequest,
    ) -> Optional[_DirectPosePayloadResult]:
        pair = self._prepare_direct_pose_pair(request.direct, request.gt_motion)
        if pair is None:
            return None

        feature_config = self._resolve_direct_pose_feature_config()
        geo_payload = self.compute_rot6d_geo_loss(pair.direct_seq, pair.gt_direct, return_per_joint=True)
        if isinstance(geo_payload, tuple):
            geo_direct = geo_payload[0]
            geo_theta = geo_payload[1] if len(geo_payload) > 1 else None
        else:
            geo_direct = geo_payload
            geo_theta = None
        direct_objective = geo_direct
        extra: Dict[str, Any] = self._direct_pose_default_stats()
        extra.pop('direct_pose_objective', None)
        extra.pop('direct_pose_weighted', None)
        extra.update({
            'direct_pose_geo': geo_direct,
            'direct_pose_geo_deg': geo_direct * request.deg_per_rad,
            'direct_pose_split_active': 1.0 if bool(feature_config.leg_split_enable) else 0.0,
            'direct_pose_arm_split_active': 1.0 if bool(feature_config.arm_split_enable) else 0.0,
        })

        if torch.is_tensor(geo_theta) and geo_theta.ndim >= 3:
            base_result = self._compute_direct_pose_group_base_from_request(
                _DirectPoseGroupBaseRequest(geo_theta=geo_theta),
                feature_config=feature_config,
            )
            if base_result is not None:
                extra.update(base_result.as_payload())
                if bool(feature_config.leg_split_enable) and base_result.has_leg_split_tensors():
                    dir_leg_base = base_result.dir_leg_base
                    dir_nonleg_base = base_result.dir_nonleg_base
                    dir_nonleg_effective_base = base_result.dir_nonleg_effective_base
                    if (
                        torch.is_tensor(dir_leg_base)
                        and torch.is_tensor(dir_nonleg_base)
                        and torch.is_tensor(dir_nonleg_effective_base)
                    ):
                        direct_objective = dir_leg_base + dir_nonleg_effective_base
                        if bool(feature_config.group_norm_enable):
                            group_norm_request = _DirectPoseGroupNormRequest(
                                dir_leg_base=dir_leg_base,
                                dir_nonleg_base=dir_nonleg_base,
                                dir_nonleg_effective_base=dir_nonleg_effective_base,
                                update_ema_state=True,
                            )
                            group_norm_result = self._compute_direct_pose_group_norm_result(
                                group_norm_request,
                                feature_config=feature_config,
                            )
                            direct_objective = group_norm_result.objective
                            extra.update(group_norm_result.stats)

        return _DirectPosePayloadResult(direct_objective, extra)

    # === future: train/loss/components.py ===
    # Core motion applicators.
    def _apply_rot_ortho_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
    ) -> torch.Tensor:
        if self.w_rot_ortho > 0 and not delta_fallback:
            target_for_ortho = delta_motion if delta_motion is not None else pred_motion
            l_ortho = self.compute_rot6d_ortho_loss(target_for_ortho)
            return self._submit_component_loss(
                total_loss,
                stats=stats,
                name='rot_ortho',
                tensor=l_ortho,
                weight=self.w_rot_ortho,
                group='core',
                raw_key='rot_ortho',
                weighted_key='rot_ortho_weighted',
                extra={'rot_ortho_raw': l_ortho},
            )

        _setdefault_stats(stats, {
            'rot_ortho': 0.0,
            'rot_ortho_weighted': 0.0,
            'rot_ortho_raw': 0.0,
        })
        if delta_fallback and self.w_rot_ortho > 0 and delta_motion is not None:
            l_ortho = self.compute_rot6d_ortho_loss(delta_motion)
            stats['rot_ortho_fallback'] = _stats_float_or(l_ortho, default=float('nan'))
        return total_loss

    def _apply_rot_local_tail_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        geo_local: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        tail_w = float(getattr(self, 'rot_local_tail_weight', 0.0) or 0.0)
        tail_k = int(getattr(self, 'rot_local_tail_k', 0) or 0)
        tail_scope = str(getattr(self, 'rot_local_tail_scope', 'all') or 'all').lower()
        joint_count = int(geo_local.shape[-1])
        if tail_w <= 0.0 or tail_k <= 0 or joint_count <= 0:
            return total_loss

        k = min(max(1, tail_k), joint_count)
        cand_idx = self._rot_local_tail_candidates(tail_scope, joint_count, geo_local.device, k=k)
        per_bone = torch.nanmean(geo_local.detach(), dim=tuple(range(geo_local.dim() - 1)))
        scores = self._rot_local_tail_scores(per_bone)
        select_mode = str(getattr(self, 'rot_local_tail_select', 'batch') or 'batch').lower()
        score_count = int(scores.numel())
        if score_count <= 0:
            return total_loss
        if cand_idx is not None and cand_idx.numel() > 0:
            valid_idx = cand_idx[(cand_idx >= 0) & (cand_idx < score_count)]
            if valid_idx.numel() <= 0:
                return total_loss
            k_eff = min(k, int(valid_idx.numel()))
            vals = scores.index_select(0, valid_idx)
            _, sel = torch.topk(vals, k=k_eff, largest=True, sorted=False)
            idx = valid_idx.index_select(0, sel)
        else:
            k_eff = min(k, score_count)
            _, idx = torch.topk(scores, k=k_eff, largest=True, sorted=False)
        tail_loss = torch.nanmean(geo_local.index_select(-1, idx))
        total_loss = self._submit_component_loss(
            total_loss,
            stats=stats,
            name='rot_local_tail',
            tensor=tail_loss,
            weight=tail_w,
            group='core',
            extra={
                'rot_local_tail_deg': tail_loss * deg_per_rad,
                'rot_local_tail_k': float(k_eff),
                'rot_local_tail_scope': float({'all': 0.0, 'limbs': 1.0, 'keybones': 2.0}.get(tail_scope, 0.0)),
                'rot_local_tail_select': float({'batch': 0.0, 'ema': 1.0}.get(select_mode, 0.0)),
            },
        )
        return total_loss

    def _apply_rot_local_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        if self.w_rot_local <= 0.0:
            stats.setdefault('rot_local_deg', 0.0)
            return total_loss

        Rp_world = self._extract_rot6d_mats(pred_motion)
        Rg_world = self._extract_rot6d_mats(gt_motion)
        if Rp_world is None or Rg_world is None:
            return total_loss

        Rp_root = self._root_relative(Rp_world)
        Rg_root = self._root_relative(Rg_world)
        Rp_local = self._parent_relative_matrices(Rp_root)
        Rg_local = self._parent_relative_matrices(Rg_root)
        geo_local = geodesic_R(Rp_local, Rg_local)
        weights = self._joint_weight_vector(Rp_local.device, Rp_local.dtype, Rp_local.shape[-3])
        local_loss = (geo_local * weights.view(1, 1, -1)).mean()
        total_loss = self._submit_component_loss(
            total_loss,
            stats=stats,
            name='rot_local',
            tensor=local_loss,
            weight=self.w_rot_local,
            group='core',
            extra={'rot_local_deg': local_loss * deg_per_rad},
        )
        return self._apply_rot_local_tail_component(total_loss, stats, geo_local, deg_per_rad)

    def _apply_root_velocity_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> torch.Tensor:
        rv_slice = self.group_slices.get('RootVelocity')
        if rv_slice is None or (self.w_root_vel <= 0.0 and self.w_root_speed <= 0.0):
            _setdefault_stats(stats, {
                'root_vel_mse': 0.0,
                'root_speed_mae': 0.0,
            })
            return total_loss

        pred_vel = pred_motion[..., rv_slice]
        gt_vel = gt_motion[..., rv_slice]
        if self.w_root_vel > 0.0:
            vel_mse = F.mse_loss(pred_vel, gt_vel)
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='root_vel',
                tensor=vel_mse,
                weight=self.w_root_vel,
                group='core',
                raw_key='root_vel_mse',
            )
        else:
            stats.setdefault('root_vel_mse', 0.0)

        if self.w_root_speed > 0.0:
            pred_speed = torch.norm(pred_vel, dim=-1)
            gt_speed = torch.norm(gt_vel, dim=-1)
            speed_mae = F.l1_loss(pred_speed, gt_speed)
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='root_speed',
                tensor=speed_mae,
                weight=self.w_root_speed,
                group='core',
                raw_key='root_speed_mae',
            )
        else:
            stats.setdefault('root_speed_mae', 0.0)
        return total_loss

    # Motion component dispatch.
    def _apply_motion_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
        deg_per_rad: float,
    ) -> torch.Tensor:
        total_loss = self._apply_rot_ortho_component(
            total_loss,
            stats,
            pred_motion,
            delta_motion,
            delta_fallback,
        )
        total_loss = self._apply_rot_local_component(
            total_loss,
            stats,
            pred_motion,
            gt_motion,
            deg_per_rad,
        )
        return self._apply_root_velocity_components(total_loss, stats, pred_motion, gt_motion)

    # Direct-pose applicator.
    def _apply_direct_pose_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        gt_motion: torch.Tensor,
        deg_per_rad: float,
    ) -> torch.Tensor:
        if self.w_direct_pose <= 0.0 or not isinstance(pred_motion, dict):
            _setdefault_stats(stats, self._direct_pose_default_stats())
            return total_loss

        direct = pred_motion.get('out_direct', None)
        if torch.is_tensor(direct):
            payload = self._compute_direct_pose_payload(direct, gt_motion, deg_per_rad)
            if payload is not None:
                direct_objective, extra = payload
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='direct_pose',
                    tensor=direct_objective,
                    weight=self.w_direct_pose,
                    group='core',
                    raw_key='direct_pose_objective',
                    weighted_key='direct_pose_weighted',
                    extra=extra,
                )
        return total_loss

    # Auxiliary applicators.
    # Contact-plan applicator.
    def _contact_plan_extra_stats(
        self,
        plan: Any,
        logits: torch.Tensor,
        gt: torch.Tensor,
        steps: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        probs = plan if torch.is_tensor(plan) else torch.sigmoid(logits)
        probs = _ensure_temporal_axis(probs)
        if probs.dim() < 3 or probs.shape[-1] != gt.shape[-1] or int(probs.shape[1]) < steps:
            return None
        l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
        return {'contact_plan_mse': l_mse}

    def _apply_contact_plan_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if self.w_contact_plan <= 0.0 or not isinstance(pred_motion, dict) or not isinstance(batch, dict):
            return total_loss

        plan_logits = pred_motion.get('contacts_plan_logits', None)
        plan = pred_motion.get('contacts_plan', None)
        gt_contacts = batch.get('contacts', None)
        if torch.is_tensor(plan_logits) and torch.is_tensor(gt_contacts):
            logits, gt, steps = self._prepare_aux_supervision_pair(plan_logits, gt_contacts)
            if steps > 0 and logits.shape[-1] == gt.shape[-1]:
                gt_t = gt[:, :steps].clamp(0.0, 1.0)
                l_bce = F.binary_cross_entropy_with_logits(logits[:, :steps], gt_t)
                extra_stats = self._contact_plan_extra_stats(plan, logits, gt, steps)
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_plan',
                    tensor=l_bce,
                    weight=self.w_contact_plan,
                    group='aux',
                    raw_key='contact_plan_bce',
                    weighted_key='contact_plan_weighted',
                    extra=extra_stats,
                )
        elif torch.is_tensor(plan) and torch.is_tensor(gt_contacts):
            probs, gt, steps = self._prepare_aux_supervision_pair(plan, gt_contacts)
            if steps > 0 and probs.shape[-1] == gt.shape[-1]:
                l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_plan',
                    tensor=l_mse,
                    weight=self.w_contact_plan,
                    group='aux',
                    raw_key='contact_plan_mse',
                    weighted_key='contact_plan_weighted',
                )
        return total_loss

    # Event-clock applicators.
    def _apply_event_clock_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
    ) -> torch.Tensor:
        if not isinstance(pred_motion, dict):
            return total_loss
        if (
            self.event_clock_lambda_entropy_weight <= 0.0
            and self.event_clock_lambda_prior_weight <= 0.0
            and self.event_clock_delta_z_l2_weight <= 0.0
        ):
            return total_loss

        lam_logit = pred_motion.get('event_clock_lambda_logit', None)
        dyn_prior = pred_motion.get('event_clock_dynamic_prior', None)
        delta_z = pred_motion.get('event_clock_delta_z', None)
        if torch.is_tensor(lam_logit):
            logits = lam_logit.unsqueeze(1) if lam_logit.dim() == 2 else lam_logit
            p = torch.sigmoid(logits)
            eps = 1e-6
            ent = -p * torch.log(p + eps) - (1.0 - p) * torch.log(1.0 - p + eps)
            if self.event_clock_lambda_entropy_weight > 0.0:
                l_ent = -ent.mean()
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='event_clock_lambda_entropy',
                    tensor=l_ent,
                    weight=self.event_clock_lambda_entropy_weight,
                    group='aux',
                    raw_key='event_clock_lambda_entropy',
                    weighted_key='event_clock_lambda_entropy_weighted',
                )
            if self.event_clock_lambda_prior_weight > 0.0 and torch.is_tensor(dyn_prior):
                prior = dyn_prior.unsqueeze(1) if dyn_prior.dim() == 2 else dyn_prior
                if prior.shape == p.shape:
                    l_prior = (p - prior.detach()).pow(2).mean()
                    total_loss = self._submit_component_loss(
                        total_loss,
                        stats=stats,
                        name='event_clock_lambda_prior',
                        tensor=l_prior,
                        weight=self.event_clock_lambda_prior_weight,
                        group='aux',
                        raw_key='event_clock_lambda_prior',
                        weighted_key='event_clock_lambda_prior_weighted',
                    )
            stats['event_clock_lambda_mean'] = _stats_float_or(p.detach().mean(), default=float('nan'))

        if self.event_clock_delta_z_l2_weight > 0.0 and torch.is_tensor(delta_z) and delta_z.numel() > 0:
            dz = delta_z.unsqueeze(1) if delta_z.dim() == 2 else delta_z
            l2 = (dz * dz).mean()
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='event_clock_delta_z_l2',
                tensor=l2,
                weight=self.event_clock_delta_z_l2_weight,
                group='aux',
                raw_key='event_clock_delta_z_l2',
                weighted_key='event_clock_delta_z_l2_weighted',
            )
        return total_loss

    # Contact-measurement applicator.
    def _apply_contact_meas_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if self.w_contact_meas <= 0.0 or not isinstance(pred_motion, dict) or not isinstance(batch, dict):
            return total_loss

        meas = pred_motion.get('contacts_meas', None)
        gt_contacts = batch.get('contacts', None)
        if torch.is_tensor(meas) and torch.is_tensor(gt_contacts):
            probs, gt, steps = self._prepare_aux_supervision_pair(meas, gt_contacts)
            if steps > 0 and probs.shape[-1] == gt.shape[-1]:
                l_mse = F.mse_loss(probs[:, :steps], gt[:, :steps])
                total_loss = self._submit_component_loss(
                    total_loss,
                    stats=stats,
                    name='contact_meas',
                    tensor=l_mse,
                    weight=self.w_contact_meas,
                    group='aux',
                    raw_key='contact_meas_mse',
                    weighted_key='contact_meas_weighted',
                )
        return total_loss

    # Omega regularization applicator.
    def _apply_omega_l2_component(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
    ) -> torch.Tensor:
        if self.w_omega_l2 <= 0.0 or not isinstance(pred_motion, dict):
            return total_loss

        omega = pred_motion.get('omega_hat', None)
        if torch.is_tensor(omega) and omega.numel() > 0:
            if omega.dim() == 3:
                omega = omega.unsqueeze(1)
            l2 = (omega * omega).mean()
            total_loss = self._submit_component_loss(
                total_loss,
                stats=stats,
                name='omega_l2',
                tensor=l2,
                weight=self.w_omega_l2,
                group='aux',
                raw_key='omega_l2',
                weighted_key='omega_l2_weighted',
            )
        return total_loss

    # Auxiliary component dispatch.
    def _apply_aux_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        pred_motion: Any,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        total_loss = self._apply_contact_plan_component(total_loss, stats, pred_motion, batch)
        total_loss = self._apply_event_clock_components(total_loss, stats, pred_motion)
        total_loss = self._apply_contact_meas_component(total_loss, stats, pred_motion, batch)
        return self._apply_omega_l2_component(total_loss, stats, pred_motion)

    # === future: train/loss/orchestration.py ===
    # Forward input prep / base loss.
    def _prepare_forward_inputs(
        self,
        pred_motion: Any,
        gt_motion: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, Optional[torch.Tensor], bool]:
        delta_fallback = False
        if isinstance(pred_motion, dict):
            delta_fallback = bool(pred_motion.get('_delta_fallback', False))
            pred_core_motion = pred_motion.get('out')
            delta_motion = pred_motion.get('delta')
        else:
            pred_core_motion = pred_motion
            delta_motion = None
        return pred_core_motion, gt_motion, delta_motion, delta_fallback

    def _run_forward_base(
        self,
        pred_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        attn_weights=None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base_out = self._forward_base_inner(pred_motion, gt_motion, attn_weights=attn_weights)  # type: ignore[arg-type]
        if isinstance(base_out, tuple):
            loss, stats = base_out
        else:
            loss, stats = base_out, {}
        if isinstance(stats, dict):
            return loss, dict(stats)
        return loss, {}

    # Loss tracker / stats finalize.
    def _init_loss_group_tracker(self):
        self._loss_group_totals = {key: 0.0 for key in ('core', 'aux', 'long')}

    def _accumulate_loss_contrib(self, name: str, tensor: Optional[torch.Tensor], weight: float, group: Optional[str] = None):
        if tensor is None:
            return
        try:
            w = float(weight)
        except (RuntimeError, TypeError, ValueError):
            w = float(weight.item()) if hasattr(weight, 'item') else 0.0
        if not _math.isfinite(w) or abs(w) < 1e-9:
            return
        if group is None:
            group = self._loss_group_alias.get(name, 'core')
        if group not in self._loss_group_totals:
            self._loss_group_totals[group] = 0.0
        contrib = _stats_float_or(tensor.detach() * w, default=0.0)
        if _math.isfinite(contrib):
            self._loss_group_totals[group] += contrib

    def _loss_group_stats(self) -> Dict[str, float]:
        return {f'loss_group/{k}': float(v) for k, v in self._loss_group_totals.items()}

    def _prepare_aux_supervision_pair(
        self,
        pred_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        pred = _ensure_temporal_axis(pred_tensor)
        target = _ensure_temporal_axis(target_tensor.to(device=pred.device, dtype=pred.dtype))
        steps = min(int(pred.shape[1]), int(target.shape[1]))
        return pred, target, steps

    def _submit_component_loss(
        self,
        total_loss: torch.Tensor,
        *,
        stats: Dict[str, float],
        name: str,
        tensor: Optional[torch.Tensor],
        weight: float,
        group: Optional[str] = None,
        raw_key: Optional[str] = None,
        weighted_key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if tensor is None:
            return total_loss
        total_loss = total_loss + tensor * float(weight)
        self._accumulate_loss_contrib(name, tensor, weight, group=group)
        payload: Dict[str, float] = {}
        if raw_key is not None:
            payload[raw_key] = _stats_float(tensor)
        if weighted_key is not None:
            payload[weighted_key] = _stats_float(tensor * float(weight))
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload[key] = _stats_float(value)
        if payload:
            stats.update(payload)
        return total_loss

    # Applicator dispatch shell.
    def _dispatch_forward_components(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
        *,
        pred_motion: Any,
        pred_core_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        delta_motion: Optional[torch.Tensor],
        delta_fallback: bool,
        deg_per_rad: float,
        batch: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        total_loss = self._apply_motion_components(
            total_loss,
            stats,
            pred_core_motion,
            gt_motion,
            delta_motion,
            delta_fallback,
            deg_per_rad,
        )
        total_loss = self._apply_direct_pose_component(total_loss, stats, pred_motion, gt_motion, deg_per_rad)
        return self._apply_aux_components(total_loss, stats, pred_motion, batch)

    def _finalize_forward_outputs(
        self,
        total_loss: torch.Tensor,
        stats: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        stats.update(self._loss_group_stats())
        return total_loss, stats

    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        self._init_loss_group_tracker()
        pred_core_motion, gt_core_motion, delta_motion, delta_fallback = self._prepare_forward_inputs(pred_motion, gt_motion)
        loss, stats = self._run_forward_base(pred_core_motion, gt_core_motion, attn_weights=attn_weights)
        deg_per_rad = 180.0 / _math.pi
        loss = self._dispatch_forward_components(
            loss,
            stats,
            pred_motion=pred_motion,
            pred_core_motion=pred_core_motion,
            gt_motion=gt_core_motion,
            delta_motion=delta_motion,
            delta_fallback=delta_fallback,
            deg_per_rad=deg_per_rad,
            batch=batch,
        )
        return self._finalize_forward_outputs(loss, stats)
