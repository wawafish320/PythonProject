from __future__ import annotations

"""Single build/runtime contract for EventMotionModel construction."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from train.checkpoint.compat import (
    DirectPoseBuildOverrides,
    resolve_direct_pose_build_cfg as _resolve_direct_pose_build_cfg,
)
from train.checkpoint.contract import (
    normalize_contact_plan_init_mode,
    normalize_direct_pose_feat_source,
    normalize_direct_pose_leg_gate_mode,
    normalize_direct_pose_leg_gate_power,
    normalize_direct_pose_leg_mode,
    normalize_direct_pose_phase_z_mode,
    normalize_lambda_fusion_mode,
)
from train.configuration.norm_spec import ContactPretrainRuntime, resolve_contact_pretrain_runtime
from train.losses import MotionJointLoss, STAGE6_3WAY_ARMCHAIN_BONES_CSV


DEFAULT_TRAIN_TRAINER_LR = 1e-4
DEFAULT_TRAIN_TRAINER_GRAD_CLIP = 1.0
DEFAULT_TRAIN_TRAINER_WEIGHT_DECAY = 0.01
DEFAULT_TRAIN_TRAINER_USE_AMP = False
DEFAULT_TRAIN_TRAINER_ACCUM_STEPS = 1
DEFAULT_TRAIN_TRAINER_PIN_MEMORY = False

DEFAULT_POSTTRAIN_TRAINER_LR = 2e-4
DEFAULT_POSTTRAIN_TRAINER_GRAD_CLIP = 0.0
DEFAULT_POSTTRAIN_TRAINER_WEIGHT_DECAY = 0.0
DEFAULT_POSTTRAIN_TRAINER_USE_AMP = False
DEFAULT_POSTTRAIN_TRAINER_ACCUM_STEPS = 1
DEFAULT_POSTTRAIN_TRAINER_PIN_MEMORY = False

DEFAULT_TRAIN_HISTORY_ADAPTIVE_HIDDEN = 256
DEFAULT_TRAIN_HISTORY_ADAPTIVE_HEADS = 2
DEFAULT_TRAIN_HISTORY_ADAPTIVE_MAX_FRAMES = None
DEFAULT_TRAIN_HISTORY_ADAPTIVE_TRAIN_VARIABLE = False
DEFAULT_TRAIN_HISTORY_DROPOUT_PROB = 0.10
DEFAULT_POSTTRAIN_HISTORY_DROPOUT_PROB = 0.0
DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MIN = 0.05
DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MAX = 0.30
DEFAULT_TRAIN_HISTORY_USE_TREND_FEATURES = False
DEFAULT_TRAIN_DIAG_TOPK = 8
DEFAULT_TRAIN_DIAG_THR = 8.0
DEFAULT_DIRECT_POSE_GRAD_MONITOR_ENABLE = False
DEFAULT_DIRECT_POSE_GRAD_RATIO_GATE = 0.35
DEFAULT_TRAIN_TEACHER_EVAL_MAX_BATCHES = None
DEFAULT_TRAIN_SS_CHUNK_LEN = 1
DEFAULT_TRAIN_TF_MODE = "epoch_linear"
DEFAULT_TRAIN_TF_START_EPOCH = 0
DEFAULT_TRAIN_TF_END_EPOCH = 10
DEFAULT_TRAIN_TF_MAX = 1.0
DEFAULT_TRAIN_TF_MIN = 0.1
DEFAULT_TRAIN_HISTORY_DEBUG_STEPS = 0
DEFAULT_TRAIN_FREERUN_DEBUG_PATH = None
DEFAULT_TRAIN_ENABLE_GRAD_CONNECTION_TEST = True
DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_CLAMP = 1.0
DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_AFFINE_STATS = None
DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_CLAMP = 1.0
DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_AFFINE_STATS = None

DEFAULT_POSTTRAIN_CONTACT_MEAS_WEIGHT = 0.0
DEFAULT_POSTTRAIN_CONTACT_MEAS_GATE_BY_HIT = "auto"
DEFAULT_POSTTRAIN_CONTACT_MEAS_VXY_MODE = "abs"
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_MODE = "window"
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_BETA = 0.05
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_WINDOW = 5
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_QUANTILE = 0.2
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_SLEW_UP_CM = 0.0
DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_SLEW_DOWN_CM = 0.0

DEFAULT_POSTTRAIN_LAMBDA_FUSION_ENTROPY_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_FUSION_SMOOTH_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_STEPS = 0
DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_FUSION_MONOTONIC_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_PLAN_ENTROPY_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_PLAN_DYN_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MODE = "inv"
DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MAX = 2.0
DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_MODE = "none"
DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_WARMUP_STEPS = 0
DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_CONTACT_ERR_MAX = 1.0
DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_WARMUP_JOINT_SCALES = None
DEFAULT_POSTTRAIN_LAMBDA_L2SP_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_BOUNDARY_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_TAU_DEG = 2.5
DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_MARGIN_DEG = 1.0
DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_START_STEP = -1

DEFAULT_CONTACT_PLAN_HIDDEN = 64
DEFAULT_CONTACT_PLAN_DROPOUT = 0.0
DEFAULT_CONTACT_PLAN_INJECT = "none"
DEFAULT_CONTACT_PLAN_INJECT_DETACH = True
DEFAULT_CONTACT_PLAN_TIME_PE_DIM = 0
DEFAULT_CONTACT_PLAN_TIME_PE_BASE = 10000.0
DEFAULT_CONTACT_PLAN_INIT_MODE = "learnable"
DEFAULT_CONTACT_PLAN_INIT_HIDDEN = 128
DEFAULT_CONTACT_PLAN_INIT_DROPOUT = 0.0

DEFAULT_EVENT_CLOCK_MAX_DELTA = 0.5
DEFAULT_EVENT_CLOCK_HIDDEN_DIM = 64
DEFAULT_EVENT_CLOCK_GATE_HIDDEN_DIM = 32

DEFAULT_DIRECT_POSE_HIDDEN = 256
DEFAULT_DIRECT_POSE_DROPOUT = 0.0
DEFAULT_DIRECT_POSE_DETACH_PLAN = True
DEFAULT_DIRECT_POSE_MEAS_MODE = "concat"
DEFAULT_DIRECT_POSE_FEAT_SOURCE = "cond"
DEFAULT_DIRECT_POSE_TIME_PE_DIM = 0
DEFAULT_DIRECT_POSE_TIME_PE_BASE = 10000.0

DEFAULT_DIRECT_POSE_LEG_MODE = "rot6d_add"
DEFAULT_DIRECT_POSE_LEG_GATE_MODE = "none"
DEFAULT_DIRECT_POSE_LEG_GATE_POWER = 1.0
DEFAULT_DIRECT_POSE_LEG_SCALE_LOG_CLIP = 4.0
DEFAULT_DIRECT_POSE_LEG_SCALE_CLAMP_K = 0.0

DEFAULT_LAMBDA_FUSION_MODE = "per_joint"
DEFAULT_LAMBDA_FUSION_HIDDEN = 128
DEFAULT_LAMBDA_FUSION_DROPOUT = 0.0
DEFAULT_LAMBDA_FUSION_DETACH_ERR = True
DEFAULT_LAMBDA_FUSION_LOGIT_INIT = -2.0
DEFAULT_LAMBDA_FUSION_USE_ROLLOUT_STEP = False

DEFAULT_SO3_CORR_HIDDEN = 128
DEFAULT_SO3_CORR_DROPOUT = 0.0
DEFAULT_SO3_CORR_GATE_LOGIT_INIT = -5.0

DEFAULT_LOSS_W_ATTN_REG = 0.01
DEFAULT_TRAIN_LOSS_W_ROT_ORTHO = 0.001
DEFAULT_POSTTRAIN_LOSS_W_ROT_ORTHO = 0.0
DEFAULT_LOSS_W_ROT_LOCAL = 0.0
DEFAULT_LOSS_W_ROOT_VEL = 0.0
DEFAULT_LOSS_W_ROOT_SPEED = 0.0
DEFAULT_LOSS_W_CONTACT_PLAN = 0.0
DEFAULT_LOSS_W_CONTACT_MEAS = 0.0
DEFAULT_LOSS_W_DIRECT_POSE = 0.0
DEFAULT_LOSS_W_OMEGA_L2 = 0.0
DEFAULT_LOSS_DIRECT_POSE_LEG_SPLIT = False
DEFAULT_LOSS_DIRECT_POSE_ARM_ELSE_BALANCE_ENABLE = False
DEFAULT_LOSS_DIRECT_POSE_ARM_WEIGHT = 1.0
DEFAULT_LOSS_DIRECT_POSE_ELSE_WEIGHT = 1.0
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_ENABLE = False
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_W_LEG = 1.0
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_W_NONLEG = 1.0
DEFAULT_TRAIN_LOSS_DIRECT_POSE_GROUP_NORM_EMA_BETA = 0.9
DEFAULT_POSTTRAIN_LOSS_DIRECT_POSE_GROUP_NORM_EMA_BETA = 0.95
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_RATIO_MIN = 0.2
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_RATIO_MAX = 5.0
DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_EPS = 1e-6
DEFAULT_TRAIN_LOSS_EVENT_CLOCK_LAMBDA_ENTROPY_WEIGHT = 0.01
DEFAULT_TRAIN_LOSS_EVENT_CLOCK_LAMBDA_PRIOR_WEIGHT = 0.01
DEFAULT_TRAIN_LOSS_EVENT_CLOCK_DELTA_Z_L2_WEIGHT = 0.001
DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_LAMBDA_ENTROPY_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_LAMBDA_PRIOR_WEIGHT = 0.0
DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_DELTA_Z_L2_WEIGHT = 0.0
DEFAULT_LOSS_UNIFIED_DOWNSTREAM_POWER = 0.6
DEFAULT_LOSS_UNIFIED_SELF_SCALE = 1.5
DEFAULT_LOSS_UNIFIED_MIN_WEIGHT = 0.05
DEFAULT_LOSS_ROT_LOCAL_TAIL_WEIGHT = 0.0
DEFAULT_LOSS_ROT_LOCAL_TAIL_K = 0
DEFAULT_LOSS_ROT_LOCAL_TAIL_SCOPE = "all"
DEFAULT_LOSS_ROT_LOCAL_TAIL_SELECT = "batch"
DEFAULT_LOSS_ROT_LOCAL_TAIL_EMA_BETA = 0.9
DEFAULT_LOSS_UNIFIED_USE_VISUAL_IMPORTANCE = False
DEFAULT_LOSS_ROT6D_EPS = 1e-6


@dataclass(frozen=True)
class DatasetModelFacts:
    dx: int
    dy: int
    dc: int
    contact_dim: int
    angvel_dim: int
    pose_hist_dim: int
    pose_hist_len: int
    period_dim: int
    state_layout: Mapping[str, Any]
    output_layout: Mapping[str, Any]
    bone_names: Sequence[str]
    fps: float

    @classmethod
    def from_dataset(cls, dataset: Any, *, context: str = "dataset") -> "DatasetModelFacts":
        state_layout = _required_dataset_mapping(dataset, "state_layout", context=context)
        output_layout = _required_dataset_mapping(dataset, "output_layout", context=context)
        bone_names_raw = _required_dataset_field(dataset, "bone_names", context=context)
        if not isinstance(bone_names_raw, (list, tuple)):
            raise TypeError(
                f"{context}.bone_names must be a list/tuple for ModelBuildConfig; "
                f"got {type(bone_names_raw).__name__}."
            )
        return cls(
            dx=_required_dataset_int(dataset, "Dx", context=context),
            dy=_required_dataset_int(dataset, "Dy", context=context),
            dc=_required_dataset_int(dataset, "Dc", context=context),
            contact_dim=_required_dataset_int(dataset, "contact_dim", context=context, min_value=0),
            angvel_dim=_required_dataset_int(dataset, "angvel_dim", context=context, min_value=0),
            pose_hist_dim=_required_dataset_int(dataset, "pose_hist_dim", context=context, min_value=0),
            pose_hist_len=_required_dataset_int(dataset, "pose_hist_len", context=context, min_value=0),
            period_dim=_required_dataset_int(dataset, "period_dim", context=context, min_value=0),
            state_layout=state_layout,
            output_layout=output_layout,
            bone_names=tuple(str(name) for name in bone_names_raw),
            fps=_required_dataset_float(dataset, "fps", context=context, min_value=0.0),
        )


@dataclass(frozen=True)
class DatasetLossFacts:
    output_layout: Mapping[str, Any]
    fps: float
    rot6d_spec: Mapping[str, Any]
    meta: Mapping[str, Any]
    bone_names: Sequence[str]
    parents: Sequence[int]
    bone_offsets: Any

    @classmethod
    def from_dataset(cls, dataset: Any, *, context: str = "dataset") -> "DatasetLossFacts":
        meta = _required_dataset_loss_meta(dataset, context=context)
        output_layout = _required_dataset_mapping(dataset, "output_layout", context=context)
        rot6d_spec = _required_loss_mapping_value(
            _dataset_attr_or_meta_key(dataset, meta, "rot6d_spec", context=context),
            field=f"{context}.rot6d_spec",
            allow_empty=True,
        )
        bone_names_raw = _required_dataset_field(dataset, "bone_names", context=context)
        if not isinstance(bone_names_raw, (list, tuple)):
            raise TypeError(
                f"{context}.bone_names must be a list/tuple for DatasetLossFacts; "
                f"got {type(bone_names_raw).__name__}."
            )
        parents_raw = _required_dataset_field(dataset, "parents", context=context)
        if not isinstance(parents_raw, (list, tuple)):
            raise TypeError(
                f"{context}.parents must be a list/tuple for DatasetLossFacts; "
                f"got {type(parents_raw).__name__}."
            )
        bone_offsets = _required_dataset_field_allow_none(dataset, "bone_offsets", context=context)
        if bone_offsets is not None:
            offsets_tensor = torch.as_tensor(bone_offsets)
            if offsets_tensor.ndim != 2 or offsets_tensor.shape[-1] != 3:
                raise ValueError(
                    f"{context}.bone_offsets must be None or shape=(num_joints, 3) for DatasetLossFacts; "
                    f"got shape={tuple(int(v) for v in offsets_tensor.shape)!r}."
                )
        return cls(
            output_layout=output_layout,
            fps=_required_dataset_float(dataset, "fps", context=context, min_value=0.0),
            rot6d_spec=rot6d_spec,
            meta=dict(meta),
            bone_names=tuple(str(name) for name in bone_names_raw),
            parents=tuple(int(parent) for parent in parents_raw),
            bone_offsets=bone_offsets,
        )


@dataclass(frozen=True)
class ContactPlanConfig:
    enable: bool
    hidden: int
    dropout: float
    inject: str
    inject_detach: bool
    time_pe_dim: int
    time_pe_base: float
    init_mode: str
    init_hidden: int
    init_dropout: float


@dataclass(frozen=True)
class DirectPoseConfig:
    enable: bool
    hidden: int
    dropout: float
    detach_plan: bool
    meas_mode: str
    meas_drop_prob: float
    meas_noise_std: float
    plan_drop_prob: float
    feat_source: str
    time_pe_dim: int
    time_pe_base: float
    use_phase_z: bool
    phase_z_mode: str
    split_enable: bool
    nonleg_proj_dim: int
    arm_split_enable: bool
    arm_bones: Optional[str]
    drop_ckpt_weights: bool = False


@dataclass(frozen=True)
class DirectPoseLegConfig:
    enable: bool
    bones: Any
    mode: str
    stopgrad_main: bool
    detach_feat: bool
    max_deg: float
    gate_mode: str
    gate_power: float
    scale_log_clip: float
    scale_clamp_k: float
    side_routing: bool = False
    contact_order: str = "lr"
    side_embed_dim: int = 0
    side_plan_other: bool = False
    side_phase_other: bool = False
    side_phase_rel: bool = False
    side_cue: str = "none"
    side_cue_tau: float = 30.0
    side_sign_gate: bool = False
    side_rank1: bool = False


@dataclass(frozen=True)
class EventClockConfig:
    enable: bool
    max_delta: float
    hidden_dim: int
    gate_hidden_dim: int
    period_dim_init: int


@dataclass(frozen=True)
class LambdaFusionConfig:
    enable: bool
    mode: str
    hidden: int
    dropout: float
    detach_err: bool
    logit_init: float
    use_rollout_step: bool


@dataclass(frozen=True)
class ModelBuildConfig:
    facts: DatasetModelFacts
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    context_len: int
    pose_hist_dim_model: int
    pose_hist_dim_raw: int
    pose_hist_len_raw: int
    history_export_frames: int
    history_frame_dim: int
    contact_plan: ContactPlanConfig
    direct_pose: DirectPoseConfig
    direct_pose_leg: DirectPoseLegConfig
    event_clock: EventClockConfig
    lambda_fusion: LambdaFusionConfig
    so3_corr_hidden: int = DEFAULT_SO3_CORR_HIDDEN
    so3_corr_dropout: float = DEFAULT_SO3_CORR_DROPOUT
    so3_corr_gate_logit_init: float = DEFAULT_SO3_CORR_GATE_LOGIT_INIT

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "in_state_dim": int(self.facts.dx),
            "out_motion_dim": int(self.facts.dy),
            "cond_dim": int(self.facts.dc),
            "period_dim": int(self.event_clock.period_dim_init),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "dropout": float(self.dropout),
            "context_len": int(self.context_len),
            "contact_dim": int(self.facts.contact_dim),
            "angvel_dim": int(self.facts.angvel_dim),
            "pose_hist_dim": int(self.pose_hist_dim_model),
            "state_layout": dict(self.facts.state_layout),
            "bone_names": tuple(self.facts.bone_names),
            "output_layout": dict(self.facts.output_layout),
            "contact_plan_enable": bool(self.contact_plan.enable),
            "contact_plan_hidden": int(self.contact_plan.hidden),
            "contact_plan_dropout": float(self.contact_plan.dropout),
            "contact_plan_inject": str(self.contact_plan.inject),
            "contact_plan_inject_detach": bool(self.contact_plan.inject_detach),
            "contact_plan_time_pe_dim": int(self.contact_plan.time_pe_dim),
            "contact_plan_time_pe_base": float(self.contact_plan.time_pe_base),
            "contact_plan_init_mode": str(self.contact_plan.init_mode),
            "contact_plan_init_hidden": int(self.contact_plan.init_hidden),
            "contact_plan_init_dropout": float(self.contact_plan.init_dropout),
            "use_event_clock": bool(self.event_clock.enable),
            "event_clock_max_delta": float(self.event_clock.max_delta),
            "event_clock_hidden_dim": int(self.event_clock.hidden_dim),
            "event_clock_gate_hidden_dim": int(self.event_clock.gate_hidden_dim),
            "direct_pose_enable": bool(self.direct_pose.enable),
            "direct_pose_hidden": int(self.direct_pose.hidden),
            "direct_pose_dropout": float(self.direct_pose.dropout),
            "direct_pose_detach_plan": bool(self.direct_pose.detach_plan),
            "direct_pose_meas_mode": str(self.direct_pose.meas_mode),
            "direct_pose_meas_drop_prob": float(self.direct_pose.meas_drop_prob),
            "direct_pose_meas_noise_std": float(self.direct_pose.meas_noise_std),
            "direct_pose_plan_drop_prob": float(self.direct_pose.plan_drop_prob),
            "direct_pose_feat_source": str(self.direct_pose.feat_source),
            "direct_pose_time_pe_dim": int(self.direct_pose.time_pe_dim),
            "direct_pose_time_pe_base": float(self.direct_pose.time_pe_base),
            "direct_pose_use_phase_z": bool(self.direct_pose.use_phase_z),
            "direct_pose_phase_z_mode": str(self.direct_pose.phase_z_mode),
            "direct_pose_split_enable": bool(self.direct_pose.split_enable),
            "direct_pose_nonleg_proj_dim": int(self.direct_pose.nonleg_proj_dim),
            "direct_pose_arm_split_enable": bool(self.direct_pose.arm_split_enable),
            "direct_pose_arm_bones": self.direct_pose.arm_bones,
            "direct_pose_leg_enable": bool(self.direct_pose_leg.enable),
            "direct_pose_leg_bones": self.direct_pose_leg.bones,
            "direct_pose_leg_mode": str(self.direct_pose_leg.mode),
            "direct_pose_leg_stopgrad_main": bool(self.direct_pose_leg.stopgrad_main),
            "direct_pose_leg_detach_feat": bool(self.direct_pose_leg.detach_feat),
            "direct_pose_leg_max_deg": float(self.direct_pose_leg.max_deg),
            "direct_pose_leg_gate_mode": str(self.direct_pose_leg.gate_mode),
            "direct_pose_leg_gate_power": float(self.direct_pose_leg.gate_power),
            "direct_pose_leg_scale_log_clip": float(self.direct_pose_leg.scale_log_clip),
            "direct_pose_leg_scale_clamp_k": float(self.direct_pose_leg.scale_clamp_k),
            "direct_pose_leg_side_routing": bool(self.direct_pose_leg.side_routing),
            "direct_pose_leg_contact_order": str(self.direct_pose_leg.contact_order),
            "direct_pose_leg_side_embed_dim": int(self.direct_pose_leg.side_embed_dim),
            "direct_pose_leg_side_plan_other": bool(self.direct_pose_leg.side_plan_other),
            "direct_pose_leg_side_phase_other": bool(self.direct_pose_leg.side_phase_other),
            "direct_pose_leg_side_phase_rel": bool(self.direct_pose_leg.side_phase_rel),
            "direct_pose_leg_side_cue": str(self.direct_pose_leg.side_cue),
            "direct_pose_leg_side_cue_tau": float(self.direct_pose_leg.side_cue_tau),
            "direct_pose_leg_side_sign_gate": bool(self.direct_pose_leg.side_sign_gate),
            "direct_pose_leg_side_rank1": bool(self.direct_pose_leg.side_rank1),
            "lambda_fusion_enable": bool(self.lambda_fusion.enable),
            "lambda_fusion_mode": str(self.lambda_fusion.mode),
            "lambda_fusion_hidden": int(self.lambda_fusion.hidden),
            "lambda_fusion_dropout": float(self.lambda_fusion.dropout),
            "lambda_fusion_detach_err": bool(self.lambda_fusion.detach_err),
            "lambda_fusion_logit_init": float(self.lambda_fusion.logit_init),
            "lambda_fusion_use_rollout_step": bool(self.lambda_fusion.use_rollout_step),
            "so3_corr_hidden": int(self.so3_corr_hidden),
            "so3_corr_dropout": float(self.so3_corr_dropout),
            "so3_corr_gate_logit_init": float(self.so3_corr_gate_logit_init),
        }


@dataclass(frozen=True)
class LossBuildConfig:
    output_layout: Mapping[str, Any]
    fps: float
    rot6d_spec: Mapping[str, Any]
    meta: Mapping[str, Any]
    bone_names: Sequence[str]
    parents: Sequence[int]
    bone_offsets: Any
    w_attn_reg: float
    w_rot_ortho: float
    w_rot_local: float
    w_root_vel: float
    w_root_speed: float
    w_contact_plan: float
    w_contact_meas: float
    w_direct_pose: float
    w_omega_l2: float
    direct_pose_loss_leg_split: bool
    direct_pose_arm_split_enable: bool
    direct_pose_arm_bones: Optional[str]
    direct_pose_loss_arm_else_balance_enable: bool
    direct_pose_loss_arm_weight: float
    direct_pose_loss_else_weight: float
    direct_pose_loss_group_norm_enable: bool
    direct_pose_loss_group_norm_w_leg: float
    direct_pose_loss_group_norm_w_nonleg: float
    direct_pose_loss_group_norm_ema_beta: float
    direct_pose_loss_group_norm_ratio_min: float
    direct_pose_loss_group_norm_ratio_max: float
    direct_pose_loss_group_norm_eps: float
    event_clock_lambda_entropy_weight: float
    event_clock_lambda_prior_weight: float
    event_clock_delta_z_l2_weight: float
    unified_downstream_power: float
    unified_self_scale: float
    unified_min_weight: float
    rot_local_tail_weight: float
    rot_local_tail_k: int
    rot_local_tail_scope: str
    rot_local_tail_select: str
    rot_local_tail_ema_beta: float
    unified_use_visual_importance: bool
    rot6d_eps: float

    def to_loss_kwargs(self) -> dict[str, Any]:
        return {
            "w_attn_reg": float(self.w_attn_reg),
            "output_layout": dict(self.output_layout),
            "fps": float(self.fps),
            "rot6d_spec": dict(self.rot6d_spec),
            "w_rot_ortho": float(self.w_rot_ortho),
            "meta": dict(self.meta),
            "w_rot_local": float(self.w_rot_local),
            "w_root_vel": float(self.w_root_vel),
            "w_root_speed": float(self.w_root_speed),
            "w_contact_plan": float(self.w_contact_plan),
            "w_contact_meas": float(self.w_contact_meas),
            "w_direct_pose": float(self.w_direct_pose),
            "direct_pose_loss_leg_split": bool(self.direct_pose_loss_leg_split),
            "direct_pose_arm_split_enable": bool(self.direct_pose_arm_split_enable),
            "direct_pose_arm_bones": self.direct_pose_arm_bones,
            "direct_pose_loss_arm_else_balance_enable": bool(self.direct_pose_loss_arm_else_balance_enable),
            "direct_pose_loss_arm_weight": float(self.direct_pose_loss_arm_weight),
            "direct_pose_loss_else_weight": float(self.direct_pose_loss_else_weight),
            "direct_pose_loss_group_norm_enable": bool(self.direct_pose_loss_group_norm_enable),
            "direct_pose_loss_group_norm_w_leg": float(self.direct_pose_loss_group_norm_w_leg),
            "direct_pose_loss_group_norm_w_nonleg": float(self.direct_pose_loss_group_norm_w_nonleg),
            "direct_pose_loss_group_norm_ema_beta": float(self.direct_pose_loss_group_norm_ema_beta),
            "direct_pose_loss_group_norm_ratio_min": float(self.direct_pose_loss_group_norm_ratio_min),
            "direct_pose_loss_group_norm_ratio_max": float(self.direct_pose_loss_group_norm_ratio_max),
            "direct_pose_loss_group_norm_eps": float(self.direct_pose_loss_group_norm_eps),
            "w_omega_l2": float(self.w_omega_l2),
            "event_clock_lambda_entropy_weight": float(self.event_clock_lambda_entropy_weight),
            "event_clock_lambda_prior_weight": float(self.event_clock_lambda_prior_weight),
            "event_clock_delta_z_l2_weight": float(self.event_clock_delta_z_l2_weight),
        }

    def apply_post_init(self, loss_fn: MotionJointLoss) -> None:
        loss_fn.unified_downstream_power = float(self.unified_downstream_power)
        loss_fn.unified_self_scale = float(self.unified_self_scale)
        loss_fn.unified_min_weight = float(self.unified_min_weight)
        loss_fn.rot_local_tail_weight = float(self.rot_local_tail_weight)
        loss_fn.rot_local_tail_k = int(self.rot_local_tail_k)
        loss_fn.rot_local_tail_scope = str(self.rot_local_tail_scope)
        loss_fn.rot_local_tail_select = str(self.rot_local_tail_select)
        loss_fn.rot_local_tail_ema_beta = float(self.rot_local_tail_ema_beta)
        loss_fn.unified_use_visual_importance = bool(self.unified_use_visual_importance)
        loss_fn.rot6d_eps = float(self.rot6d_eps)

    def build_loss(self) -> MotionJointLoss:
        loss_fn = MotionJointLoss(**self.to_loss_kwargs())
        self.apply_post_init(loss_fn)
        return loss_fn


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    lr: float
    grad_clip: float
    weight_decay: float
    use_amp: bool
    accum_steps: int
    pin_memory: bool
    contacts_pretrain: ContactPretrainRuntime
    history_adaptive_hidden: int
    history_adaptive_heads: int
    history_adaptive_max_frames: Optional[int]
    history_adaptive_train_variable: bool
    history_dropout_prob: float
    history_dropout_prob_min: float
    history_dropout_prob_max: float
    history_use_trend_features: bool
    diag_topk: int
    diag_thr: float
    direct_pose_grad_monitor_enable: bool
    direct_pose_grad_ratio_gate: float
    teacher_eval_max_batches: Optional[int]
    ss_chunk_len: int
    tf_mode: str
    tf_start_epoch: int
    tf_end_epoch: int
    tf_max: float
    tf_min: float
    history_debug_steps: int
    freerun_stage_schedule_spec: Any
    hyperparam_scheduler: Any
    freerun_debug_path: Optional[str]
    enable_grad_connection_test: bool

    def to_trainer_kwargs(self, *, args: Any = None) -> dict[str, Any]:
        kwargs = {
            "lr": float(self.lr),
            "grad_clip": float(self.grad_clip),
            "weight_decay": float(self.weight_decay),
            "use_amp": bool(self.use_amp),
            "accum_steps": int(self.accum_steps),
            "pin_memory": bool(self.pin_memory),
        }
        if args is not None:
            kwargs["args"] = args
        return kwargs

    def to_adaptive_history_kwargs(self) -> dict[str, Any]:
        return {
            "history_hidden_dim": int(self.history_adaptive_hidden),
            "max_history_frames": self.history_adaptive_max_frames,
            "history_heads": int(self.history_adaptive_heads),
            "train_variable_history": bool(self.history_adaptive_train_variable),
            "history_dropout_prob": float(self.history_dropout_prob),
            "use_trend_features": bool(self.history_use_trend_features),
        }

    def to_train_owner_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "direct_pose_grad_monitor_enable": bool(self.direct_pose_grad_monitor_enable),
            "direct_pose_grad_ratio_gate": float(self.direct_pose_grad_ratio_gate),
            "diag_topk": int(self.diag_topk),
            "diag_thr": float(self.diag_thr),
            "teacher_eval_max_batches": self.teacher_eval_max_batches,
            "ss_chunk_len": int(self.ss_chunk_len),
            "tf_mode": str(self.tf_mode),
            "tf_start_epoch": int(self.tf_start_epoch),
            "tf_end_epoch": int(self.tf_end_epoch),
            "tf_max": float(self.tf_max),
            "tf_min": float(self.tf_min),
            "history_debug_steps": int(self.history_debug_steps),
            "history_dropout_prob": float(self.history_dropout_prob),
            "history_dropout_prob_min": float(self.history_dropout_prob_min),
            "history_dropout_prob_max": float(self.history_dropout_prob_max),
            "hyperparam_scheduler": self.hyperparam_scheduler,
            "freerun_debug_path": self.freerun_debug_path,
            "enable_grad_connection_test": bool(self.enable_grad_connection_test),
        }


@dataclass(frozen=True)
class PosttrainLocalRuntimeConfig:
    contact_meas_gate_by_hit: str
    contact_meas_gate_by_hit_override: Optional[bool]
    contact_meas_vxy_mode: str
    contact_meas_ground_z_mode: str
    contact_meas_ground_z_beta: float
    contact_meas_ground_z_window: int
    contact_meas_ground_z_quantile: float
    contact_meas_ground_z_slew_up_cm: float
    contact_meas_ground_z_slew_down_cm: float
    contact_meas_ground_z_max_up_m: float
    contact_meas_ground_z_max_down_m: float
    lambda_reliability_mode: str
    lambda_reliability_warmup_steps: int
    lambda_reliability_contact_err_max: float
    lambda_reliability_warmup_joint_scales: Optional[list[float]]

    def to_posttrain_config_dict(self) -> dict[str, Any]:
        return {
            "contact_meas_gate_by_hit": str(self.contact_meas_gate_by_hit),
            "contact_meas_vxy_mode": str(self.contact_meas_vxy_mode),
            "contact_meas_ground_z_mode": str(self.contact_meas_ground_z_mode),
            "contact_meas_ground_z_beta": float(self.contact_meas_ground_z_beta),
            "contact_meas_ground_z_window": int(self.contact_meas_ground_z_window),
            "contact_meas_ground_z_quantile": float(self.contact_meas_ground_z_quantile),
            "contact_meas_ground_z_slew_up_cm": float(self.contact_meas_ground_z_slew_up_cm),
            "contact_meas_ground_z_slew_down_cm": float(self.contact_meas_ground_z_slew_down_cm),
            "lambda_reliability_mode": str(self.lambda_reliability_mode),
            "lambda_reliability_warmup_steps": int(self.lambda_reliability_warmup_steps),
            "lambda_reliability_contact_err_max": float(self.lambda_reliability_contact_err_max),
            "lambda_reliability_warmup_joint_scales": (
                None
                if self.lambda_reliability_warmup_joint_scales is None
                else list(self.lambda_reliability_warmup_joint_scales)
            ),
        }

    def to_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "contact_meas_gate_by_hit_override": self.contact_meas_gate_by_hit_override,
            "contact_meas_vxy_mode": str(self.contact_meas_vxy_mode),
            "contact_meas_ground_z_mode": str(self.contact_meas_ground_z_mode),
            "contact_meas_ground_z_beta": float(self.contact_meas_ground_z_beta),
            "contact_meas_ground_z_window": int(self.contact_meas_ground_z_window),
            "contact_meas_ground_z_quantile": float(self.contact_meas_ground_z_quantile),
            "contact_meas_ground_z_max_up_m": float(self.contact_meas_ground_z_max_up_m),
            "contact_meas_ground_z_max_down_m": float(self.contact_meas_ground_z_max_down_m),
            "lambda_reliability_mode": str(self.lambda_reliability_mode),
            "lambda_reliability_warmup_steps": int(self.lambda_reliability_warmup_steps),
            "lambda_reliability_contact_err_max": float(self.lambda_reliability_contact_err_max),
            "lambda_reliability_warmup_joint_scales": (
                None
                if self.lambda_reliability_warmup_joint_scales is None
                else list(self.lambda_reliability_warmup_joint_scales)
            ),
        }


@dataclass(frozen=True)
class PosttrainLambdaObjectiveConfig:
    lambda_fusion_mode: str
    lambda_fusion_hidden: int
    lambda_fusion_dropout: float
    lambda_fusion_logit_init: float
    lambda_fusion_use_rollout_step: bool
    lambda_fusion_entropy_weight: float
    lambda_fusion_smooth_weight: float
    lambda_fusion_early_steps: int
    lambda_fusion_early_weight: float
    lambda_fusion_monotonic_weight: float
    lambda_plan_entropy_weight: float
    lambda_plan_dyn_weight: float
    lambda_time_weight_mode: str
    lambda_time_weight_max: float
    lambda_l2sp_weight: float
    lambda_boundary_weight: float
    lambda_gate_sup_weight: float
    lambda_gate_sup_tau_deg: float
    lambda_gate_sup_margin_deg: float
    lambda_gate_sup_start_step: int
    contact_meas_weight: float

    def to_posttrain_config_dict(self) -> dict[str, Any]:
        return {
            "lambda_fusion_mode": str(self.lambda_fusion_mode),
            "lambda_fusion_hidden": int(self.lambda_fusion_hidden),
            "lambda_fusion_dropout": float(self.lambda_fusion_dropout),
            "lambda_fusion_logit_init": float(self.lambda_fusion_logit_init),
            "lambda_fusion_use_rollout_step": bool(self.lambda_fusion_use_rollout_step),
            "lambda_fusion_entropy_weight": float(self.lambda_fusion_entropy_weight),
            "lambda_fusion_smooth_weight": float(self.lambda_fusion_smooth_weight),
            "lambda_fusion_early_steps": int(self.lambda_fusion_early_steps),
            "lambda_fusion_early_weight": float(self.lambda_fusion_early_weight),
            "lambda_fusion_monotonic_weight": float(self.lambda_fusion_monotonic_weight),
            "lambda_plan_entropy_weight": float(self.lambda_plan_entropy_weight),
            "lambda_plan_dyn_weight": float(self.lambda_plan_dyn_weight),
            "lambda_time_weight_mode": str(self.lambda_time_weight_mode),
            "lambda_time_weight_max": float(self.lambda_time_weight_max),
            "lambda_l2sp_weight": float(self.lambda_l2sp_weight),
            "lambda_boundary_weight": float(self.lambda_boundary_weight),
            "lambda_gate_sup_weight": float(self.lambda_gate_sup_weight),
            "lambda_gate_sup_tau_deg": float(self.lambda_gate_sup_tau_deg),
            "lambda_gate_sup_margin_deg": float(self.lambda_gate_sup_margin_deg),
            "lambda_gate_sup_start_step": int(self.lambda_gate_sup_start_step),
            "contact_meas_weight": float(self.contact_meas_weight),
        }

    def to_lambda_rollout_kwargs(self) -> dict[str, Any]:
        return {
            "lambda_entropy_weight": float(self.lambda_fusion_entropy_weight),
            "lambda_smooth_weight": float(self.lambda_fusion_smooth_weight),
            "lambda_early_steps": int(self.lambda_fusion_early_steps),
            "lambda_early_weight": float(self.lambda_fusion_early_weight),
            "lambda_monotonic_weight": float(self.lambda_fusion_monotonic_weight),
            "lambda_plan_entropy_weight": float(self.lambda_plan_entropy_weight),
            "lambda_plan_dyn_weight": float(self.lambda_plan_dyn_weight),
            "lambda_gate_sup_weight": float(self.lambda_gate_sup_weight),
            "lambda_gate_sup_tau_deg": float(self.lambda_gate_sup_tau_deg),
            "lambda_gate_sup_margin_deg": float(self.lambda_gate_sup_margin_deg),
            "lambda_gate_sup_start_step": int(self.lambda_gate_sup_start_step),
        }


def resolve_train_trainer_runtime_config(
    *,
    args: Any,
    dataset_facts: Optional[DatasetModelFacts] = None,
    loss_facts: Optional[DatasetLossFacts] = None,
    model_build_config: ModelBuildConfig,
    pin_memory: bool,
) -> TrainerRuntimeConfig:
    del dataset_facts, loss_facts
    history_hidden = _cfg_int(
        args,
        "history_adaptive_hidden",
        DEFAULT_TRAIN_HISTORY_ADAPTIVE_HIDDEN,
        min_value=0,
    )
    if history_hidden <= 0:
        history_hidden = int(model_build_config.hidden_dim)

    return TrainerRuntimeConfig(
        lr=_cfg_float(args, "lr", DEFAULT_TRAIN_TRAINER_LR, min_value=0.0),
        grad_clip=_cfg_float(args, "grad_clip", DEFAULT_TRAIN_TRAINER_GRAD_CLIP, min_value=0.0),
        weight_decay=_cfg_float(args, "weight_decay", DEFAULT_TRAIN_TRAINER_WEIGHT_DECAY, min_value=0.0),
        use_amp=_cfg_bool(args, "amp", DEFAULT_TRAIN_TRAINER_USE_AMP),
        accum_steps=_cfg_int(args, "accum_steps", DEFAULT_TRAIN_TRAINER_ACCUM_STEPS, min_value=1),
        pin_memory=bool(pin_memory),
        contacts_pretrain=resolve_contact_pretrain_runtime(
            clamp_raw=_cfg_float(
                args,
                "trainbase_contacts_pretrain_clamp",
                DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_CLAMP,
                min_value=0.0,
            ),
            affine_stats_raw=_cfg_value(
                args,
                "trainbase_contacts_pretrain_affine_stats",
                DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_AFFINE_STATS,
            ),
            warn=True,
            warn_prefix="[MPL]",
        ),
        history_adaptive_hidden=int(history_hidden),
        history_adaptive_heads=_cfg_int(
            args,
            "history_adaptive_heads",
            DEFAULT_TRAIN_HISTORY_ADAPTIVE_HEADS,
            min_value=1,
        ),
        history_adaptive_max_frames=_cfg_optional_int(args, "history_adaptive_max_frames", min_value=0),
        history_adaptive_train_variable=_cfg_bool(
            args,
            "history_adaptive_train_variable",
            DEFAULT_TRAIN_HISTORY_ADAPTIVE_TRAIN_VARIABLE,
        ),
        history_dropout_prob=_cfg_float(
            args,
            "history_dropout_prob",
            DEFAULT_TRAIN_HISTORY_DROPOUT_PROB,
            min_value=0.0,
        ),
        history_dropout_prob_min=DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MIN,
        history_dropout_prob_max=DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MAX,
        history_use_trend_features=_cfg_bool(
            args,
            "history_use_trend_features",
            DEFAULT_TRAIN_HISTORY_USE_TREND_FEATURES,
        ),
        diag_topk=_cfg_int(args, "diag_topk", DEFAULT_TRAIN_DIAG_TOPK, min_value=1),
        diag_thr=_cfg_float(args, "diag_thr", DEFAULT_TRAIN_DIAG_THR, min_value=0.0),
        direct_pose_grad_monitor_enable=_cfg_bool(
            args,
            "direct_pose_grad_monitor_enable",
            DEFAULT_DIRECT_POSE_GRAD_MONITOR_ENABLE,
        ),
        direct_pose_grad_ratio_gate=_cfg_float(
            args,
            "direct_pose_grad_ratio_gate",
            DEFAULT_DIRECT_POSE_GRAD_RATIO_GATE,
            min_value=0.0,
        ),
        teacher_eval_max_batches=_cfg_optional_int(args, "teacher_eval_max_batches"),
        ss_chunk_len=_cfg_int(args, "ss_chunk_len", DEFAULT_TRAIN_SS_CHUNK_LEN, min_value=1),
        tf_mode=_normalize_train_tf_mode(_cfg_value(args, "tf_mode", DEFAULT_TRAIN_TF_MODE)),
        tf_start_epoch=_cfg_int(args, "tf_start_epoch", DEFAULT_TRAIN_TF_START_EPOCH, min_value=0),
        tf_end_epoch=_cfg_int(args, "tf_end_epoch", DEFAULT_TRAIN_TF_END_EPOCH, min_value=0),
        tf_max=_cfg_float(args, "tf_max", DEFAULT_TRAIN_TF_MAX),
        tf_min=_cfg_float(args, "tf_min", DEFAULT_TRAIN_TF_MIN),
        history_debug_steps=_cfg_int(args, "history_debug_steps", DEFAULT_TRAIN_HISTORY_DEBUG_STEPS, min_value=0),
        freerun_stage_schedule_spec=_cfg_value(args, "freerun_stage_schedule", None),
        hyperparam_scheduler=None,
        freerun_debug_path=_cfg_optional_str(args, "freerun_debug_path", DEFAULT_TRAIN_FREERUN_DEBUG_PATH),
        enable_grad_connection_test=not _cfg_bool(args, "no_grad_conn_test", False),
    )


def resolve_posttrain_trainer_runtime_config(
    *,
    cfg: Any,
    dataset_facts: Optional[DatasetModelFacts] = None,
    loss_facts: Optional[DatasetLossFacts] = None,
    model_build_config: ModelBuildConfig,
) -> TrainerRuntimeConfig:
    del dataset_facts, loss_facts, model_build_config
    return TrainerRuntimeConfig(
        lr=_cfg_float(cfg, "lr", DEFAULT_POSTTRAIN_TRAINER_LR, min_value=0.0),
        grad_clip=DEFAULT_POSTTRAIN_TRAINER_GRAD_CLIP,
        weight_decay=_cfg_float(cfg, "weight_decay", DEFAULT_POSTTRAIN_TRAINER_WEIGHT_DECAY, min_value=0.0),
        use_amp=DEFAULT_POSTTRAIN_TRAINER_USE_AMP,
        accum_steps=DEFAULT_POSTTRAIN_TRAINER_ACCUM_STEPS,
        pin_memory=DEFAULT_POSTTRAIN_TRAINER_PIN_MEMORY,
        contacts_pretrain=resolve_contact_pretrain_runtime(
            clamp_raw=_cfg_float(
                cfg,
                "posttrain_contacts_pretrain_clamp",
                DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_CLAMP,
                min_value=0.0,
            ),
            affine_stats_raw=_cfg_value(
                cfg,
                "posttrain_contacts_pretrain_affine_stats",
                DEFAULT_POSTTRAIN_CONTACTS_PRETRAIN_AFFINE_STATS,
            ),
            warn=False,
        ),
        history_adaptive_hidden=DEFAULT_TRAIN_HISTORY_ADAPTIVE_HIDDEN,
        history_adaptive_heads=DEFAULT_TRAIN_HISTORY_ADAPTIVE_HEADS,
        history_adaptive_max_frames=DEFAULT_TRAIN_HISTORY_ADAPTIVE_MAX_FRAMES,
        history_adaptive_train_variable=DEFAULT_TRAIN_HISTORY_ADAPTIVE_TRAIN_VARIABLE,
        history_dropout_prob=DEFAULT_POSTTRAIN_HISTORY_DROPOUT_PROB,
        history_dropout_prob_min=DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MIN,
        history_dropout_prob_max=DEFAULT_TRAIN_HISTORY_DROPOUT_PROB_MAX,
        history_use_trend_features=DEFAULT_TRAIN_HISTORY_USE_TREND_FEATURES,
        diag_topk=DEFAULT_TRAIN_DIAG_TOPK,
        diag_thr=DEFAULT_TRAIN_DIAG_THR,
        direct_pose_grad_monitor_enable=_cfg_bool(
            cfg,
            "direct_pose_grad_monitor_enable",
            DEFAULT_DIRECT_POSE_GRAD_MONITOR_ENABLE,
        ),
        direct_pose_grad_ratio_gate=_cfg_float(
            cfg,
            "direct_pose_grad_ratio_gate",
            DEFAULT_DIRECT_POSE_GRAD_RATIO_GATE,
            min_value=0.0,
        ),
        teacher_eval_max_batches=DEFAULT_TRAIN_TEACHER_EVAL_MAX_BATCHES,
        ss_chunk_len=DEFAULT_TRAIN_SS_CHUNK_LEN,
        tf_mode=DEFAULT_TRAIN_TF_MODE,
        tf_start_epoch=DEFAULT_TRAIN_TF_START_EPOCH,
        tf_end_epoch=DEFAULT_TRAIN_TF_END_EPOCH,
        tf_max=DEFAULT_TRAIN_TF_MAX,
        tf_min=DEFAULT_TRAIN_TF_MIN,
        history_debug_steps=DEFAULT_TRAIN_HISTORY_DEBUG_STEPS,
        freerun_stage_schedule_spec=None,
        hyperparam_scheduler=None,
        freerun_debug_path=DEFAULT_TRAIN_FREERUN_DEBUG_PATH,
        enable_grad_connection_test=DEFAULT_TRAIN_ENABLE_GRAD_CONNECTION_TEST,
    )


def resolve_posttrain_local_runtime_config(cfg: Any) -> PosttrainLocalRuntimeConfig:
    gate_by_hit, gate_override = _normalize_posttrain_contact_meas_gate_by_hit(
        _cfg_value(cfg, "contact_meas_gate_by_hit", DEFAULT_POSTTRAIN_CONTACT_MEAS_GATE_BY_HIT)
    )
    slew_up_cm = max(
        0.0,
        _cfg_float(
            cfg,
            "contact_meas_ground_z_slew_up_cm",
            DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_SLEW_UP_CM,
        ),
    )
    slew_down_cm = max(
        0.0,
        _cfg_float(
            cfg,
            "contact_meas_ground_z_slew_down_cm",
            DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_SLEW_DOWN_CM,
        ),
    )
    return PosttrainLocalRuntimeConfig(
        contact_meas_gate_by_hit=str(gate_by_hit),
        contact_meas_gate_by_hit_override=gate_override,
        contact_meas_vxy_mode=_normalize_posttrain_contact_meas_vxy_mode(
            _cfg_value(cfg, "contact_meas_vxy_mode", DEFAULT_POSTTRAIN_CONTACT_MEAS_VXY_MODE)
        ),
        contact_meas_ground_z_mode=_normalize_posttrain_contact_meas_ground_z_mode(
            _cfg_value(cfg, "contact_meas_ground_z_mode", DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_MODE)
        ),
        contact_meas_ground_z_beta=_clamp_float_value(
            _cfg_float(
                cfg,
                "contact_meas_ground_z_beta",
                DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_BETA,
            ),
            min_value=0.0,
            max_value=1.0,
        ),
        contact_meas_ground_z_window=max(
            1,
            _cfg_int(
                cfg,
                "contact_meas_ground_z_window",
                DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_WINDOW,
            ),
        ),
        contact_meas_ground_z_quantile=_clamp_float_value(
            _cfg_float(
                cfg,
                "contact_meas_ground_z_quantile",
                DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_QUANTILE,
            ),
            min_value=0.0,
            max_value=1.0,
        ),
        contact_meas_ground_z_slew_up_cm=float(slew_up_cm),
        contact_meas_ground_z_slew_down_cm=float(slew_down_cm),
        contact_meas_ground_z_max_up_m=float(slew_up_cm) / 100.0,
        contact_meas_ground_z_max_down_m=float(slew_down_cm) / 100.0,
        lambda_reliability_mode=_normalize_posttrain_lambda_reliability_mode(
            _cfg_value(cfg, "lambda_reliability_mode", DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_MODE)
        ),
        lambda_reliability_warmup_steps=_cfg_int(
            cfg,
            "lambda_reliability_warmup_steps",
            DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_WARMUP_STEPS,
            min_value=0,
        ),
        lambda_reliability_contact_err_max=max(
            0.0,
            _cfg_float(
                cfg,
                "lambda_reliability_contact_err_max",
                DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_CONTACT_ERR_MAX,
            ),
        ),
        lambda_reliability_warmup_joint_scales=_cfg_float_list(
            _cfg_value(
                cfg,
                "lambda_reliability_warmup_joint_scales",
                DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_WARMUP_JOINT_SCALES,
            ),
            field="lambda_reliability_warmup_joint_scales",
        ),
    )


def resolve_posttrain_lambda_objective_config(cfg: Any) -> PosttrainLambdaObjectiveConfig:
    return PosttrainLambdaObjectiveConfig(
        lambda_fusion_mode=normalize_lambda_fusion_mode(
            _cfg_value(cfg, "lambda_fusion_mode", DEFAULT_LAMBDA_FUSION_MODE),
            default=DEFAULT_LAMBDA_FUSION_MODE,
            strict=True,
            context="lambda_fusion_mode",
        ),
        lambda_fusion_hidden=_cfg_int(cfg, "lambda_fusion_hidden", DEFAULT_LAMBDA_FUSION_HIDDEN, min_value=1),
        lambda_fusion_dropout=_cfg_float(cfg, "lambda_fusion_dropout", DEFAULT_LAMBDA_FUSION_DROPOUT, min_value=0.0),
        lambda_fusion_logit_init=_cfg_float(cfg, "lambda_fusion_logit_init", DEFAULT_LAMBDA_FUSION_LOGIT_INIT),
        lambda_fusion_use_rollout_step=_cfg_bool(
            cfg,
            "lambda_fusion_use_rollout_step",
            DEFAULT_LAMBDA_FUSION_USE_ROLLOUT_STEP,
        ),
        lambda_fusion_entropy_weight=_cfg_float(
            cfg,
            "lambda_fusion_entropy_weight",
            DEFAULT_POSTTRAIN_LAMBDA_FUSION_ENTROPY_WEIGHT,
        ),
        lambda_fusion_smooth_weight=_cfg_float(
            cfg,
            "lambda_fusion_smooth_weight",
            DEFAULT_POSTTRAIN_LAMBDA_FUSION_SMOOTH_WEIGHT,
        ),
        lambda_fusion_early_steps=_cfg_int(
            cfg,
            "lambda_fusion_early_steps",
            DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_STEPS,
            min_value=0,
        ),
        lambda_fusion_early_weight=_cfg_float(
            cfg,
            "lambda_fusion_early_weight",
            DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_WEIGHT,
        ),
        lambda_fusion_monotonic_weight=_cfg_float(
            cfg,
            "lambda_fusion_monotonic_weight",
            DEFAULT_POSTTRAIN_LAMBDA_FUSION_MONOTONIC_WEIGHT,
        ),
        lambda_plan_entropy_weight=_cfg_float(
            cfg,
            "lambda_plan_entropy_weight",
            DEFAULT_POSTTRAIN_LAMBDA_PLAN_ENTROPY_WEIGHT,
        ),
        lambda_plan_dyn_weight=_cfg_float(
            cfg,
            "lambda_plan_dyn_weight",
            DEFAULT_POSTTRAIN_LAMBDA_PLAN_DYN_WEIGHT,
        ),
        lambda_time_weight_mode=_normalize_posttrain_lambda_time_weight_mode(
            _cfg_value(cfg, "lambda_time_weight_mode", DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MODE)
        ),
        lambda_time_weight_max=max(
            1.0,
            _cfg_float(cfg, "lambda_time_weight_max", DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MAX),
        ),
        lambda_l2sp_weight=_cfg_float(cfg, "lambda_l2sp_weight", DEFAULT_POSTTRAIN_LAMBDA_L2SP_WEIGHT),
        lambda_boundary_weight=_cfg_float(
            cfg,
            "lambda_boundary_weight",
            DEFAULT_POSTTRAIN_LAMBDA_BOUNDARY_WEIGHT,
        ),
        lambda_gate_sup_weight=_cfg_float(
            cfg,
            "lambda_gate_sup_weight",
            DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_WEIGHT,
        ),
        lambda_gate_sup_tau_deg=_cfg_float(
            cfg,
            "lambda_gate_sup_tau_deg",
            DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_TAU_DEG,
        ),
        lambda_gate_sup_margin_deg=max(
            0.0,
            _cfg_float(
                cfg,
                "lambda_gate_sup_margin_deg",
                DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_MARGIN_DEG,
            ),
        ),
        lambda_gate_sup_start_step=_cfg_int(
            cfg,
            "lambda_gate_sup_start_step",
            DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_START_STEP,
        ),
        contact_meas_weight=_cfg_float(cfg, "contact_meas_weight", DEFAULT_POSTTRAIN_CONTACT_MEAS_WEIGHT),
    )


def resolve_train_model_build_config(*, args: Any, dataset_facts: DatasetModelFacts) -> ModelBuildConfig:
    pose_hist_dim_raw = int(dataset_facts.pose_hist_dim)
    pose_hist_len_raw = int(dataset_facts.pose_hist_len)
    history_export_frames = _cfg_int(args, "history_adaptive_export_frames", 0, min_value=0)
    history_frame_dim = (
        pose_hist_dim_raw // pose_hist_len_raw
        if pose_hist_len_raw > 0 and pose_hist_dim_raw % pose_hist_len_raw == 0
        else 0
    )


    pose_hist_dim_model = pose_hist_dim_raw
    if history_export_frames > 0 and history_frame_dim > 0:
        pose_hist_dim_model = int(history_export_frames * history_frame_dim)

    arm_split_enable = _cfg_bool(args, "direct_pose_arm_split_enable", False)
    arm_bones = _optional_csv(_cfg_value(args, "direct_pose_arm_bones", None))
    if arm_split_enable and arm_bones is None:
        arm_bones = str(STAGE6_3WAY_ARMCHAIN_BONES_CSV)

    contact_plan = ContactPlanConfig(
        enable=_cfg_bool(args, "contact_plan_enable", False),
        hidden=_cfg_int(args, "contact_plan_hidden", DEFAULT_CONTACT_PLAN_HIDDEN, min_value=1),
        dropout=_cfg_float(args, "contact_plan_dropout", DEFAULT_CONTACT_PLAN_DROPOUT, min_value=0.0),
        inject=_normalize_contact_plan_inject(_cfg_value(args, "contact_plan_inject", DEFAULT_CONTACT_PLAN_INJECT)),
        inject_detach=_cfg_bool(args, "contact_plan_inject_detach", DEFAULT_CONTACT_PLAN_INJECT_DETACH),
        time_pe_dim=_even_nonnegative_dim(
            _cfg_int(args, "contact_plan_time_pe_dim", DEFAULT_CONTACT_PLAN_TIME_PE_DIM, min_value=0),
            field="contact_plan_time_pe_dim",
        ),
        time_pe_base=_cfg_float(args, "contact_plan_time_pe_base", DEFAULT_CONTACT_PLAN_TIME_PE_BASE, min_value=0.0),
        init_mode=normalize_contact_plan_init_mode(
            _cfg_value(args, "contact_plan_init_mode", DEFAULT_CONTACT_PLAN_INIT_MODE),
            default=DEFAULT_CONTACT_PLAN_INIT_MODE,
            strict=True,
            context="contact_plan_init_mode",
        ),
        init_hidden=_cfg_int(args, "contact_plan_init_hidden", DEFAULT_CONTACT_PLAN_INIT_HIDDEN, min_value=1),
        init_dropout=_cfg_float(args, "contact_plan_init_dropout", DEFAULT_CONTACT_PLAN_INIT_DROPOUT, min_value=0.0),
    )
    direct_pose = DirectPoseConfig(
        enable=_cfg_bool(args, "direct_pose_enable", False),
        hidden=_cfg_int(args, "direct_pose_hidden", DEFAULT_DIRECT_POSE_HIDDEN, min_value=1),
        dropout=_cfg_float(args, "direct_pose_dropout", DEFAULT_DIRECT_POSE_DROPOUT, min_value=0.0),
        detach_plan=_cfg_bool(args, "direct_pose_detach_plan", DEFAULT_DIRECT_POSE_DETACH_PLAN),
        meas_mode=_normalize_direct_pose_meas_mode(
            _cfg_value(args, "direct_pose_meas_mode", DEFAULT_DIRECT_POSE_MEAS_MODE),
            field="direct_pose_meas_mode",
        ),
        meas_drop_prob=_cfg_float(args, "direct_pose_meas_drop_prob", 0.0, min_value=0.0),
        meas_noise_std=_cfg_float(args, "direct_pose_meas_noise_std", 0.0, min_value=0.0),
        plan_drop_prob=_cfg_float(args, "direct_pose_plan_drop_prob", 0.0, min_value=0.0),
        feat_source=normalize_direct_pose_feat_source(
            _cfg_value(args, "direct_pose_feat_source", DEFAULT_DIRECT_POSE_FEAT_SOURCE),
            default=DEFAULT_DIRECT_POSE_FEAT_SOURCE,
            strict=True,
            context="direct_pose_feat_source",
        ),
        time_pe_dim=_even_nonnegative_dim(
            _cfg_int(args, "direct_pose_time_pe_dim", DEFAULT_DIRECT_POSE_TIME_PE_DIM, min_value=0),
            field="direct_pose_time_pe_dim",
        ),
        time_pe_base=_cfg_float(args, "direct_pose_time_pe_base", DEFAULT_DIRECT_POSE_TIME_PE_BASE, min_value=0.0),
        use_phase_z=False,
        phase_z_mode="concat",
        split_enable=_cfg_bool(args, "direct_pose_split_enable", False),
        nonleg_proj_dim=_cfg_int(args, "direct_pose_nonleg_proj_dim", 0, min_value=0),
        arm_split_enable=arm_split_enable,
        arm_bones=arm_bones,
    )
    return ModelBuildConfig(
        facts=dataset_facts,
        hidden_dim=_cfg_int(args, "width", 512, min_value=1),
        num_layers=_cfg_int(args, "depth", 2, min_value=1),
        num_heads=_cfg_int(args, "num_heads", 4, min_value=1),
        dropout=_cfg_float(args, "dropout", 0.1, min_value=0.0),
        context_len=_cfg_int(args, "context_len", 16, min_value=1),
        pose_hist_dim_model=int(pose_hist_dim_model),
        pose_hist_dim_raw=int(pose_hist_dim_raw),
        pose_hist_len_raw=int(pose_hist_len_raw),
        history_export_frames=int(history_export_frames),
        history_frame_dim=int(history_frame_dim),
        contact_plan=contact_plan,
        direct_pose=direct_pose,
        direct_pose_leg=_default_direct_pose_leg_config(),
        event_clock=EventClockConfig(
            enable=_cfg_bool(args, "use_event_clock", False),
            max_delta=_cfg_float(args, "event_clock_max_delta", DEFAULT_EVENT_CLOCK_MAX_DELTA, min_value=0.0),
            hidden_dim=_cfg_int(args, "event_clock_hidden_dim", DEFAULT_EVENT_CLOCK_HIDDEN_DIM, min_value=1),
            gate_hidden_dim=_cfg_int(args, "event_clock_gate_hidden_dim", DEFAULT_EVENT_CLOCK_GATE_HIDDEN_DIM, min_value=1),
            period_dim_init=int(dataset_facts.period_dim),
        ),
        lambda_fusion=_default_lambda_fusion_config(),
    )


def resolve_train_loss_build_config(
    *,
    args: Any,
    loss_facts: DatasetLossFacts,
    model_build_config: ModelBuildConfig,
) -> LossBuildConfig:
    return _resolve_loss_build_config(
        source=args,
        loss_facts=loss_facts,
        model_build_config=model_build_config,
        direct_pose_group_norm_ema_beta_default=DEFAULT_TRAIN_LOSS_DIRECT_POSE_GROUP_NORM_EMA_BETA,
        event_clock_lambda_entropy_default=DEFAULT_TRAIN_LOSS_EVENT_CLOCK_LAMBDA_ENTROPY_WEIGHT,
        event_clock_lambda_prior_default=DEFAULT_TRAIN_LOSS_EVENT_CLOCK_LAMBDA_PRIOR_WEIGHT,
        event_clock_delta_z_l2_default=DEFAULT_TRAIN_LOSS_EVENT_CLOCK_DELTA_Z_L2_WEIGHT,
        w_rot_ortho_default=DEFAULT_TRAIN_LOSS_W_ROT_ORTHO,
    )


def resolve_posttrain_loss_build_config(
    *,
    cfg: Any,
    loss_facts: DatasetLossFacts,
    model_build_config: ModelBuildConfig,
) -> LossBuildConfig:
    return _resolve_loss_build_config(
        source=cfg,
        loss_facts=loss_facts,
        model_build_config=model_build_config,
        direct_pose_group_norm_ema_beta_default=DEFAULT_POSTTRAIN_LOSS_DIRECT_POSE_GROUP_NORM_EMA_BETA,
        event_clock_lambda_entropy_default=DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_LAMBDA_ENTROPY_WEIGHT,
        event_clock_lambda_prior_default=DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_LAMBDA_PRIOR_WEIGHT,
        event_clock_delta_z_l2_default=DEFAULT_POSTTRAIN_LOSS_EVENT_CLOCK_DELTA_Z_L2_WEIGHT,
        w_rot_ortho_default=DEFAULT_POSTTRAIN_LOSS_W_ROT_ORTHO,
    )


def resolve_posttrain_model_build_config(
    *,
    cfg: Any,
    dataset_facts: DatasetModelFacts,
    state_dict: Mapping[str, Any],
    ckpt_posttrain_cfg: Optional[Mapping[str, Any]],
    width: int,
    checkpoint_period_dim: int,
    has_encoder_bundle: bool,
) -> ModelBuildConfig:
    state = {str(key): value for key, value in state_dict.items()}
    _reject_direct_pose_reinit_without_train(cfg)

    direct_pose_cfg_raw = _resolve_direct_pose_build_cfg(
        out_motion_dim=int(dataset_facts.dy),
        state_dict=state,
        ckpt_posttrain_cfg=dict(ckpt_posttrain_cfg) if isinstance(ckpt_posttrain_cfg, Mapping) else None,
        contact_dim=int(dataset_facts.contact_dim),
        cond_dim=int(dataset_facts.dc),
        width=int(width),
        overrides=DirectPoseBuildOverrides(
            train_direct_pose=_cfg_bool(cfg, "train_direct_pose", False),
            direct_pose_reinit=_cfg_bool(cfg, "direct_pose_reinit", False),
            hidden_override=_cfg_optional_int(cfg, "direct_pose_hidden_override", min_value=1),
            meas_mode_override=_normalize_optional_direct_pose_meas_mode(
                _cfg_value(cfg, "direct_pose_meas_mode_override", None),
                field="direct_pose_meas_mode_override",
            ),
            feat_source=_posttrain_direct_pose_feat_source(cfg),
            time_pe_dim=_posttrain_direct_pose_time_pe_dim(cfg),
            time_pe_base=_cfg_float(cfg, "direct_pose_time_pe_base", DEFAULT_DIRECT_POSE_TIME_PE_BASE, min_value=0.0),
            use_phase_z=_cfg_bool(cfg, "direct_pose_use_phase_z", False),
            phase_z_mode=_posttrain_direct_pose_phase_z_mode(cfg),
            split_enable=_cfg_bool(cfg, "direct_pose_split_enable", False),
            arm_split_enable=_cfg_bool(cfg, "direct_pose_arm_split_enable", False),
            arm_bones=_optional_csv(_cfg_value(cfg, "direct_pose_arm_bones", None)),
            nonleg_proj_dim=_cfg_int(cfg, "direct_pose_nonleg_proj_dim", 0, min_value=0),
        ),
    )
    direct_pose = DirectPoseConfig(
        enable=bool(direct_pose_cfg_raw.enable),
        hidden=int(direct_pose_cfg_raw.hidden),
        dropout=0.0,
        detach_plan=True,
        meas_mode=str(direct_pose_cfg_raw.meas_mode),
        meas_drop_prob=0.0,
        meas_noise_std=0.0,
        plan_drop_prob=0.0,
        feat_source=str(direct_pose_cfg_raw.feat_source),
        time_pe_dim=int(direct_pose_cfg_raw.time_pe_dim),
        time_pe_base=float(direct_pose_cfg_raw.time_pe_base),
        use_phase_z=bool(direct_pose_cfg_raw.use_phase_z),
        phase_z_mode=str(direct_pose_cfg_raw.phase_z_mode),
        split_enable=bool(direct_pose_cfg_raw.split_enable),
        nonleg_proj_dim=int(direct_pose_cfg_raw.nonleg_proj_dim),
        arm_split_enable=bool(direct_pose_cfg_raw.arm_split_enable),
        arm_bones=_optional_csv(direct_pose_cfg_raw.arm_bones),
        drop_ckpt_weights=bool(direct_pose_cfg_raw.drop_ckpt_weights),
    )
    contact_plan = _resolve_posttrain_contact_plan_config(
        cfg=cfg,
        facts=dataset_facts,
        state_dict=state,
        direct_pose=direct_pose,
    )
    event_clock = _resolve_posttrain_event_clock_config(
        cfg=cfg,
        state_dict=state,
        contact_dim=int(dataset_facts.contact_dim),
        checkpoint_period_dim=int(checkpoint_period_dim),
        has_encoder_bundle=bool(has_encoder_bundle),
    )
    lambda_fusion = _resolve_posttrain_lambda_fusion_config(
        cfg=cfg,
        state_dict=state,
        width=int(width),
        contact_dim=int(dataset_facts.contact_dim),
        contact_plan_enable=bool(contact_plan.enable),
    )
    return ModelBuildConfig(
        facts=dataset_facts,
        hidden_dim=int(width),
        num_layers=_cfg_int(cfg, "depth", 2, min_value=1),
        num_heads=_cfg_int(cfg, "num_heads", 4, min_value=1),
        dropout=_cfg_float(cfg, "dropout", 0.0, min_value=0.0),
        context_len=_cfg_int(cfg, "context_len", 16, min_value=1),
        pose_hist_dim_model=int(dataset_facts.pose_hist_dim),
        pose_hist_dim_raw=int(dataset_facts.pose_hist_dim),
        pose_hist_len_raw=int(dataset_facts.pose_hist_len),
        history_export_frames=0,
        history_frame_dim=0,
        contact_plan=contact_plan,
        direct_pose=direct_pose,
        direct_pose_leg=_resolve_posttrain_direct_pose_leg_config(cfg),
        event_clock=event_clock,
        lambda_fusion=lambda_fusion,
    )


def _resolve_posttrain_contact_plan_config(
    *,
    cfg: Any,
    facts: DatasetModelFacts,
    state_dict: Mapping[str, Any],
    direct_pose: DirectPoseConfig,
) -> ContactPlanConfig:
    shared_encoder_weight = state_dict.get("shared_encoder.0.weight", None)
    if not (torch.is_tensor(shared_encoder_weight) and shared_encoder_weight.ndim == 2):
        raise KeyError("state_dict.shared_encoder.0.weight is required for ModelBuildConfig posttrain resolution.")
    shared_encoder_in_dim = int(shared_encoder_weight.shape[1])
    dataset_base_in_dim = int(facts.dx + facts.dc)
    extra_in_dim = max(0, int(shared_encoder_in_dim - dataset_base_in_dim))
    contact_plan_has_weights = any(str(key).startswith("contact_plan_cell.") for key in state_dict.keys())
    contact_plan_hidden: Optional[int] = None
    contact_plan_cell_weight = state_dict.get("contact_plan_cell.weight_ih", None)
    if torch.is_tensor(contact_plan_cell_weight) and contact_plan_cell_weight.ndim == 2:
        contact_plan_hidden = int(contact_plan_cell_weight.shape[0] // 3)
    if contact_plan_hidden is None:
        contact_plan_hidden = int(extra_in_dim) if extra_in_dim > 0 else DEFAULT_CONTACT_PLAN_HIDDEN

    if extra_in_dim == 0:
        contact_plan_inject = "none"
    elif int(facts.contact_dim) > 0 and extra_in_dim == int(facts.contact_dim):
        contact_plan_inject = "contacts"
    elif contact_plan_hidden > 0 and extra_in_dim == int(contact_plan_hidden):
        contact_plan_inject = "plan_z"
    elif extra_in_dim > 0 and contact_plan_has_weights:
        contact_plan_inject = "plan_z"
        contact_plan_hidden = int(extra_in_dim)
    else:
        contact_plan_inject = "none"

    contact_plan_time_pe_dim = 0
    contact_plan_time_head_weight = state_dict.get("contact_plan_time_head.weight", None)
    if torch.is_tensor(contact_plan_time_head_weight) and contact_plan_time_head_weight.ndim == 2:
        contact_plan_time_pe_dim = int(contact_plan_time_head_weight.shape[1])

    init_mode = normalize_contact_plan_init_mode(
        _cfg_value(cfg, "contact_plan_init_mode", DEFAULT_CONTACT_PLAN_INIT_MODE),
        default=DEFAULT_CONTACT_PLAN_INIT_MODE,
        strict=True,
        context="contact_plan_init_mode",
    )
    init_hidden = _cfg_int(cfg, "contact_plan_init_hidden", DEFAULT_CONTACT_PLAN_INIT_HIDDEN, min_value=1)
    init_dropout = _cfg_float(cfg, "contact_plan_init_dropout", DEFAULT_CONTACT_PLAN_INIT_DROPOUT, min_value=0.0)
    init_has_weights = any(str(key).startswith("contact_plan_init_head.") for key in state_dict.keys())
    if init_has_weights:
        if init_mode not in ("obs", "learnable+obs"):
            raise SystemExit(
                "[FATAL][Removed] field 'contact_plan_init_mode' no longer silently changes when "
                "ckpt contains contact_plan_init_head.* weights. This ModelBuildConfig unification removed "
                "the posttrain implicit override. Migration: set contact_plan_init_mode='learnable+obs' "
                "or use a checkpoint without contact_plan_init_head.* weights."
            )
        init_weight = state_dict.get("contact_plan_init_head.1.weight", None)
        if torch.is_tensor(init_weight) and init_weight.ndim == 2:
            init_hidden = int(init_weight.shape[0])
    return ContactPlanConfig(
        enable=bool(
            contact_plan_has_weights
            or direct_pose.enable
            or (extra_in_dim > 0 and facts.contact_dim > 0 and facts.dc > 0)
        ),
        hidden=int(contact_plan_hidden),
        dropout=0.0,
        inject=str(contact_plan_inject),
        inject_detach=True,
        time_pe_dim=int(contact_plan_time_pe_dim),
        time_pe_base=DEFAULT_CONTACT_PLAN_TIME_PE_BASE,
        init_mode=str(init_mode),
        init_hidden=int(init_hidden),
        init_dropout=float(init_dropout),
    )


def _resolve_posttrain_event_clock_config(
    *,
    cfg: Any,
    state_dict: Mapping[str, Any],
    contact_dim: int,
    checkpoint_period_dim: int,
    has_encoder_bundle: bool,
) -> EventClockConfig:
    event_clock_has_weights = any(
        str(key).startswith("event_clock_gate.") or str(key).startswith("event_clock_corrector.")
        for key in state_dict.keys()
    )
    mode = str(_cfg_value(cfg, "event_clock", "auto") or "auto").strip().lower()
    if mode not in ("auto", "on", "off"):
        raise SystemExit("[FATAL] unsupported field 'event_clock'; allowed values: auto | on | off.")
    enable = bool(event_clock_has_weights)
    if mode == "on":
        enable = True
    elif mode == "off":
        enable = False

    hidden_dim = DEFAULT_EVENT_CLOCK_HIDDEN_DIM
    corrector_weight = state_dict.get("event_clock_corrector.correction_head.0.weight", None)
    if torch.is_tensor(corrector_weight) and corrector_weight.ndim == 2:
        hidden_dim = int(corrector_weight.shape[0])
    gate_hidden_dim = DEFAULT_EVENT_CLOCK_GATE_HIDDEN_DIM
    gate_weight = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
    if torch.is_tensor(gate_weight) and gate_weight.ndim == 2:
        gate_hidden_dim = int(gate_weight.shape[0])
    override_hidden = _cfg_optional_int(cfg, "event_clock_hidden_dim", min_value=1)
    if override_hidden is not None:
        hidden_dim = int(override_hidden)
    override_gate = _cfg_optional_int(cfg, "event_clock_gate_hidden_dim", min_value=1)
    if override_gate is not None:
        gate_hidden_dim = int(override_gate)

    period_dim_init = int(checkpoint_period_dim)
    event_clock_period_feat_dim = None
    event_gate_weight = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
    if torch.is_tensor(event_gate_weight) and event_gate_weight.ndim == 2:
        base = int(contact_dim) * 2 + 1
        event_clock_period_feat_dim = max(0, int(event_gate_weight.shape[1]) - base)
    if event_clock_has_weights and event_clock_period_feat_dim is not None:
        period_dim_init = int(event_clock_period_feat_dim)
    if (
        event_clock_has_weights
        and int(period_dim_init) != int(checkpoint_period_dim)
        and not bool(has_encoder_bundle)
    ):
        raise SystemExit(
            "[FATAL][Removed] field 'encoder_bundle' is now required when ckpt Event-Clock period "
            f"features differ from period_encoder.weight (event_clock_period_dim={period_dim_init}, "
            f"checkpoint_period_dim={checkpoint_period_dim}). This ModelBuildConfig unification removed "
            "the implicit partial period reconstruction path. Migration: pass encoder_bundle or use a "
            "checkpoint with matching Event-Clock and period_encoder dimensions."
        )
    return EventClockConfig(
        enable=bool(enable),
        max_delta=_cfg_float(cfg, "event_clock_max_delta", DEFAULT_EVENT_CLOCK_MAX_DELTA, min_value=0.0),
        hidden_dim=int(hidden_dim),
        gate_hidden_dim=int(gate_hidden_dim),
        period_dim_init=int(period_dim_init),
    )


def _resolve_posttrain_lambda_fusion_config(
    *,
    cfg: Any,
    state_dict: Mapping[str, Any],
    width: int,
    contact_dim: int,
    contact_plan_enable: bool,
) -> LambdaFusionConfig:
    lambda_has_weights = any(str(key).startswith("lambda_fusion_head.") for key in state_dict.keys())
    mode = normalize_lambda_fusion_mode(
        _cfg_value(cfg, "lambda_fusion_mode", DEFAULT_LAMBDA_FUSION_MODE),
        default=DEFAULT_LAMBDA_FUSION_MODE,
        strict=True,
        context="lambda_fusion_mode",
    )
    hidden = _cfg_int(cfg, "lambda_fusion_hidden", DEFAULT_LAMBDA_FUSION_HIDDEN, min_value=1)
    use_rollout_step = _cfg_bool(cfg, "lambda_fusion_use_rollout_step", DEFAULT_LAMBDA_FUSION_USE_ROLLOUT_STEP)
    if lambda_has_weights:
        w_in = state_dict.get("lambda_fusion_head.1.weight", None)
        if torch.is_tensor(w_in) and w_in.ndim == 2:
            hidden = int(w_in.shape[0])
            base_in = int(width + (contact_dim if contact_plan_enable else 0))
            in_features = int(w_in.shape[1])
            inferred: Optional[bool] = None
            if in_features == base_in + 1:
                inferred = True
            elif in_features == base_in:
                inferred = False
            if inferred is not None and inferred != bool(use_rollout_step):
                raise SystemExit(
                    "[FATAL][Removed] field 'lambda_fusion_use_rollout_step' no longer silently changes "
                    "to match lambda_fusion_head.1.weight. This ModelBuildConfig unification removed "
                    f"the posttrain implicit override. Migration: set lambda_fusion_use_rollout_step={inferred} "
                    "to match the checkpoint, or train a new lambda_fusion_head."
                )
            if inferred is not None:
                use_rollout_step = bool(inferred)
        w_out = state_dict.get("lambda_fusion_head.4.weight", None)
        if torch.is_tensor(w_out) and w_out.ndim == 2:
            out_dim = int(w_out.shape[0])
            if out_dim == 1:
                mode = "global"
            elif out_dim > 1:
                mode = "per_joint"
    return LambdaFusionConfig(
        enable=bool(_cfg_bool(cfg, "train_lambda_head", False) or lambda_has_weights),
        mode=str(mode),
        hidden=int(hidden),
        dropout=_cfg_float(cfg, "lambda_fusion_dropout", DEFAULT_LAMBDA_FUSION_DROPOUT, min_value=0.0),
        detach_err=DEFAULT_LAMBDA_FUSION_DETACH_ERR,
        logit_init=_cfg_float(cfg, "lambda_fusion_logit_init", DEFAULT_LAMBDA_FUSION_LOGIT_INIT),
        use_rollout_step=bool(use_rollout_step),
    )


def _resolve_posttrain_direct_pose_leg_config(cfg: Any) -> DirectPoseLegConfig:
    gate_mode = normalize_direct_pose_leg_gate_mode(
        _cfg_value(cfg, "direct_pose_leg_gate_mode", DEFAULT_DIRECT_POSE_LEG_GATE_MODE),
        default=DEFAULT_DIRECT_POSE_LEG_GATE_MODE,
        strict=True,
        context="direct_pose_leg_gate_mode",
    )
    scale_clamp_k = _cfg_float(cfg, "direct_pose_leg_scale_clamp_k", DEFAULT_DIRECT_POSE_LEG_SCALE_CLAMP_K, min_value=0.0)
    if scale_clamp_k <= 1.0:
        scale_clamp_k = 0.0
    return DirectPoseLegConfig(
        enable=_cfg_bool(cfg, "direct_pose_leg_enable", False),
        bones=_cfg_value(cfg, "direct_pose_leg_bones", None),
        mode=normalize_direct_pose_leg_mode(
            _cfg_value(cfg, "direct_pose_leg_mode", DEFAULT_DIRECT_POSE_LEG_MODE),
            default=DEFAULT_DIRECT_POSE_LEG_MODE,
            strict=True,
            context="direct_pose_leg_mode",
        ),
        stopgrad_main=_cfg_bool(cfg, "direct_pose_leg_stopgrad_main", False),
        detach_feat=_cfg_bool(cfg, "direct_pose_leg_detach_feat", False),
        max_deg=_cfg_float(cfg, "direct_pose_leg_max_deg", 0.0, min_value=0.0),
        gate_mode=str(gate_mode),
        gate_power=normalize_direct_pose_leg_gate_power(
            _cfg_value(cfg, "direct_pose_leg_gate_power", DEFAULT_DIRECT_POSE_LEG_GATE_POWER),
            default=DEFAULT_DIRECT_POSE_LEG_GATE_POWER,
        ),
        scale_log_clip=_cfg_float(cfg, "direct_pose_leg_scale_log_clip", DEFAULT_DIRECT_POSE_LEG_SCALE_LOG_CLIP, min_value=1e-8),
        scale_clamp_k=float(scale_clamp_k),
    )


def _default_direct_pose_leg_config() -> DirectPoseLegConfig:
    return DirectPoseLegConfig(
        enable=False,
        bones=None,
        mode=DEFAULT_DIRECT_POSE_LEG_MODE,
        stopgrad_main=False,
        detach_feat=False,
        max_deg=0.0,
        gate_mode=DEFAULT_DIRECT_POSE_LEG_GATE_MODE,
        gate_power=DEFAULT_DIRECT_POSE_LEG_GATE_POWER,
        scale_log_clip=DEFAULT_DIRECT_POSE_LEG_SCALE_LOG_CLIP,
        scale_clamp_k=DEFAULT_DIRECT_POSE_LEG_SCALE_CLAMP_K,
    )


def _default_lambda_fusion_config() -> LambdaFusionConfig:
    return LambdaFusionConfig(
        enable=False,
        mode=DEFAULT_LAMBDA_FUSION_MODE,
        hidden=DEFAULT_LAMBDA_FUSION_HIDDEN,
        dropout=DEFAULT_LAMBDA_FUSION_DROPOUT,
        detach_err=DEFAULT_LAMBDA_FUSION_DETACH_ERR,
        logit_init=DEFAULT_LAMBDA_FUSION_LOGIT_INIT,
        use_rollout_step=DEFAULT_LAMBDA_FUSION_USE_ROLLOUT_STEP,
    )


def _reject_direct_pose_reinit_without_train(cfg: Any) -> None:
    if _cfg_bool(cfg, "direct_pose_reinit", False) and not _cfg_bool(cfg, "train_direct_pose", False):
        raise SystemExit(
            "[FATAL][Removed] field 'direct_pose_reinit' no longer silently no-ops when "
            "train_direct_pose=false. This ModelBuildConfig unification removed the posttrain implicit "
            "ignore path. Migration: set train_direct_pose=true with direct_pose_reinit=true, or remove "
            "direct_pose_reinit."
        )


def _resolve_loss_build_config(
    *,
    source: Any,
    loss_facts: DatasetLossFacts,
    model_build_config: ModelBuildConfig,
    direct_pose_group_norm_ema_beta_default: float,
    event_clock_lambda_entropy_default: float,
    event_clock_lambda_prior_default: float,
    event_clock_delta_z_l2_default: float,
    w_rot_ortho_default: float,
) -> LossBuildConfig:
    ratio_min = _cfg_positive_float(
        source,
        "direct_pose_loss_group_norm_ratio_min",
        DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_RATIO_MIN,
    )
    ratio_max = _cfg_positive_float(
        source,
        "direct_pose_loss_group_norm_ratio_max",
        DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_RATIO_MAX,
    )
    if ratio_min > ratio_max:
        raise ValueError(
            "direct_pose_loss_group_norm_ratio_min/direct_pose_loss_group_norm_ratio_max must satisfy "
            f"ratio_min <= ratio_max; got ratio_min={ratio_min!r}, ratio_max={ratio_max!r}."
        )
    return LossBuildConfig(
        output_layout=dict(loss_facts.output_layout),
        fps=float(loss_facts.fps),
        rot6d_spec=dict(loss_facts.rot6d_spec),
        meta=dict(loss_facts.meta),
        bone_names=tuple(loss_facts.bone_names),
        parents=tuple(int(parent) for parent in loss_facts.parents),
        bone_offsets=loss_facts.bone_offsets,
        w_attn_reg=DEFAULT_LOSS_W_ATTN_REG,
        w_rot_ortho=_cfg_float(source, "w_rot_ortho", w_rot_ortho_default, min_value=0.0),
        w_rot_local=_cfg_float(source, "w_rot_local", DEFAULT_LOSS_W_ROT_LOCAL, min_value=0.0),
        w_root_vel=_cfg_float(source, "w_root_vel", DEFAULT_LOSS_W_ROOT_VEL, min_value=0.0),
        w_root_speed=_cfg_float(source, "w_root_speed", DEFAULT_LOSS_W_ROOT_SPEED, min_value=0.0),
        w_contact_plan=_cfg_float(source, "w_contact_plan", DEFAULT_LOSS_W_CONTACT_PLAN, min_value=0.0),
        w_contact_meas=_cfg_float(source, "w_contact_meas", DEFAULT_LOSS_W_CONTACT_MEAS, min_value=0.0),
        w_direct_pose=_cfg_float(source, "w_direct_pose", DEFAULT_LOSS_W_DIRECT_POSE, min_value=0.0),
        w_omega_l2=_cfg_float(source, "w_omega_l2", DEFAULT_LOSS_W_OMEGA_L2, min_value=0.0),
        direct_pose_loss_leg_split=_cfg_bool(
            source,
            "direct_pose_loss_leg_split",
            DEFAULT_LOSS_DIRECT_POSE_LEG_SPLIT,
        ),
        direct_pose_arm_split_enable=bool(model_build_config.direct_pose.arm_split_enable),
        direct_pose_arm_bones=_optional_csv(model_build_config.direct_pose.arm_bones),
        direct_pose_loss_arm_else_balance_enable=_cfg_bool(
            source,
            "direct_pose_loss_arm_else_balance_enable",
            DEFAULT_LOSS_DIRECT_POSE_ARM_ELSE_BALANCE_ENABLE,
        ),
        direct_pose_loss_arm_weight=_cfg_positive_float(
            source,
            "direct_pose_loss_arm_weight",
            DEFAULT_LOSS_DIRECT_POSE_ARM_WEIGHT,
        ),
        direct_pose_loss_else_weight=_cfg_positive_float(
            source,
            "direct_pose_loss_else_weight",
            DEFAULT_LOSS_DIRECT_POSE_ELSE_WEIGHT,
        ),
        direct_pose_loss_group_norm_enable=_cfg_bool(
            source,
            "direct_pose_loss_group_norm_enable",
            DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_ENABLE,
        ),
        direct_pose_loss_group_norm_w_leg=_cfg_positive_float(
            source,
            "direct_pose_loss_group_norm_w_leg",
            DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_W_LEG,
        ),
        direct_pose_loss_group_norm_w_nonleg=_cfg_positive_float(
            source,
            "direct_pose_loss_group_norm_w_nonleg",
            DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_W_NONLEG,
        ),
        direct_pose_loss_group_norm_ema_beta=_cfg_float_range(
            source,
            "direct_pose_loss_group_norm_ema_beta",
            direct_pose_group_norm_ema_beta_default,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=False,
            max_inclusive=True,
        ),
        direct_pose_loss_group_norm_ratio_min=float(ratio_min),
        direct_pose_loss_group_norm_ratio_max=float(ratio_max),
        direct_pose_loss_group_norm_eps=_cfg_positive_float(
            source,
            "direct_pose_loss_group_norm_eps",
            DEFAULT_LOSS_DIRECT_POSE_GROUP_NORM_EPS,
        ),
        event_clock_lambda_entropy_weight=_cfg_float(
            source,
            "event_clock_lambda_entropy_weight",
            event_clock_lambda_entropy_default,
            min_value=0.0,
        ),
        event_clock_lambda_prior_weight=_cfg_float(
            source,
            "event_clock_lambda_prior_weight",
            event_clock_lambda_prior_default,
            min_value=0.0,
        ),
        event_clock_delta_z_l2_weight=_cfg_float(
            source,
            "event_clock_delta_z_l2_weight",
            event_clock_delta_z_l2_default,
            min_value=0.0,
        ),
        unified_downstream_power=_cfg_positive_float(
            source,
            "unified_downstream_power",
            DEFAULT_LOSS_UNIFIED_DOWNSTREAM_POWER,
        ),
        unified_self_scale=_cfg_positive_float(
            source,
            "unified_self_scale",
            DEFAULT_LOSS_UNIFIED_SELF_SCALE,
        ),
        unified_min_weight=_cfg_float(source, "unified_min_weight", DEFAULT_LOSS_UNIFIED_MIN_WEIGHT, min_value=0.0),
        rot_local_tail_weight=_cfg_float(
            source,
            "rot_local_tail_weight",
            DEFAULT_LOSS_ROT_LOCAL_TAIL_WEIGHT,
            min_value=0.0,
        ),
        rot_local_tail_k=_cfg_int(source, "rot_local_tail_k", DEFAULT_LOSS_ROT_LOCAL_TAIL_K, min_value=0),
        rot_local_tail_scope=_normalize_rot_local_tail_scope(
            _cfg_value(source, "rot_local_tail_scope", DEFAULT_LOSS_ROT_LOCAL_TAIL_SCOPE)
        ),
        rot_local_tail_select=_normalize_rot_local_tail_select(
            _cfg_value(source, "rot_local_tail_select", DEFAULT_LOSS_ROT_LOCAL_TAIL_SELECT)
        ),
        rot_local_tail_ema_beta=_cfg_float_range(
            source,
            "rot_local_tail_ema_beta",
            DEFAULT_LOSS_ROT_LOCAL_TAIL_EMA_BETA,
            min_value=0.0,
            max_value=0.9999,
            min_inclusive=True,
            max_inclusive=True,
        ),
        unified_use_visual_importance=DEFAULT_LOSS_UNIFIED_USE_VISUAL_IMPORTANCE,
        rot6d_eps=_cfg_positive_float(source, "rot6d_eps", DEFAULT_LOSS_ROT6D_EPS),
    )


def _required_dataset_loss_meta(dataset: Any, *, context: str) -> Mapping[str, Any]:
    if hasattr(dataset, "meta"):
        value = getattr(dataset, "meta")
        if isinstance(value, Mapping) and value:
            return dict(value)
    clips = getattr(dataset, "clips", None)
    if isinstance(clips, (list, tuple)) and len(clips) > 0:
        first = clips[0]
        value = getattr(first, "meta", None)
        if isinstance(value, Mapping) and value:
            return dict(value)
    raise AttributeError(f"{context}.meta or {context}.clips[0].meta is required for DatasetLossFacts.")


def _dataset_attr_or_meta_key(dataset: Any, meta: Mapping[str, Any], field: str, *, context: str) -> Any:
    if hasattr(dataset, field):
        value = getattr(dataset, field)
        if value is not None:
            return value
    if field in meta:
        value = meta[field]
        if value is not None:
            return value
    raise AttributeError(f"{context}.{field} is required for DatasetLossFacts.")


def _required_loss_mapping_value(value: Any, *, field: str, allow_empty: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping for DatasetLossFacts; got {type(value).__name__}.")
    if not allow_empty and not value:
        raise ValueError(f"{field} must be a non-empty mapping for DatasetLossFacts.")
    return dict(value)


def _required_dataset_field(dataset: Any, field: str, *, context: str) -> Any:
    if not hasattr(dataset, field):
        raise AttributeError(f"{context}.{field} is required for DatasetModelFacts.")
    value = getattr(dataset, field)
    if value is None:
        raise ValueError(f"{context}.{field} is required for DatasetModelFacts and cannot be None.")
    return value


def _required_dataset_field_allow_none(dataset: Any, field: str, *, context: str) -> Any:
    if not hasattr(dataset, field):
        raise AttributeError(f"{context}.{field} is required for DatasetLossFacts.")
    return getattr(dataset, field)


def _required_dataset_int(dataset: Any, field: str, *, context: str, min_value: Optional[int] = None) -> int:
    value = _required_dataset_field(dataset, field, context=context)
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context}.{field} must be an integer; got {value!r}.") from exc
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{context}.{field} must be >= {int(min_value)}; got {out}.")
    return int(out)


def _required_dataset_float(dataset: Any, field: str, *, context: str, min_value: Optional[float] = None) -> float:
    value = _required_dataset_field(dataset, field, context=context)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context}.{field} must be a finite float; got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{context}.{field} must be finite; got {out!r}.")
    if min_value is not None and out < float(min_value):
        raise ValueError(f"{context}.{field} must be >= {float(min_value)}; got {out}.")
    return float(out)


def _required_dataset_mapping(dataset: Any, field: str, *, context: str) -> Mapping[str, Any]:
    value = _required_dataset_field(dataset, field, context=context)
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{context}.{field} must be a non-empty mapping for ModelBuildConfig.")
    return dict(value)


def _normalize_train_tf_mode(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_TRAIN_TF_MODE).strip().lower()
    if text in ("global", "epoch_linear"):
        return text
    raise ValueError("tf_mode must be one of: global | epoch_linear.")


def _cfg_value(source: Any, field: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source[field] if field in source else default
    return getattr(source, field) if hasattr(source, field) else default


def _cfg_bool(source: Any, field: str, default: bool) -> bool:
    value = _cfg_value(source, field, default)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", "", "none", "null"):
        return False
    raise ValueError(f"{field} must be bool-like; got {value!r}.")


def _cfg_int(source: Any, field: str, default: int, *, min_value: Optional[int] = None) -> int:
    value = _cfg_value(source, field, default)
    if value is None:
        value = default
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer; got {value!r}.") from exc
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{field} must be >= {int(min_value)}; got {out}.")
    return int(out)


def _cfg_optional_int(source: Any, field: str, *, min_value: Optional[int] = None) -> Optional[int]:
    value = _cfg_value(source, field, None)
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer or None; got {value!r}.") from exc
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{field} must be >= {int(min_value)}; got {out}.")
    return int(out)


def _cfg_optional_str(source: Any, field: str, default: Optional[str]) -> Optional[str]:
    value = _cfg_value(source, field, default)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cfg_float(source: Any, field: str, default: float, *, min_value: Optional[float] = None) -> float:
    value = _cfg_value(source, field, default)
    if value is None:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite float; got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite; got {out!r}.")
    if min_value is not None and out < float(min_value):
        raise ValueError(f"{field} must be >= {float(min_value)}; got {out}.")
    return float(out)


def _cfg_positive_float(source: Any, field: str, default: float) -> float:
    out = _cfg_float(source, field, default)
    if out <= 0.0:
        raise ValueError(f"{field} must be > 0.0; got {out}.")
    return float(out)


def _cfg_float_range(
    source: Any,
    field: str,
    default: float,
    *,
    min_value: float,
    max_value: float,
    min_inclusive: bool,
    max_inclusive: bool,
) -> float:
    out = _cfg_float(source, field, default)
    lower_ok = out >= float(min_value) if min_inclusive else out > float(min_value)
    upper_ok = out <= float(max_value) if max_inclusive else out < float(max_value)
    if not lower_ok or not upper_ok:
        left = "[" if min_inclusive else "("
        right = "]" if max_inclusive else ")"
        raise ValueError(f"{field} must be in {left}{min_value}, {max_value}{right}; got {out}.")
    return float(out)


def _clamp_float_value(value: float, *, min_value: Optional[float], max_value: Optional[float]) -> float:
    out = float(value)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return float(out)


def _optional_csv(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null"):
        return None
    return text


def _normalize_posttrain_contact_meas_gate_by_hit(value: Any) -> tuple[str, Optional[bool]]:
    text = _normalize_text(value, default=DEFAULT_POSTTRAIN_CONTACT_MEAS_GATE_BY_HIT)
    if text in ("auto", ""):
        return "auto", None
    if text in ("true", "1", "yes", "y", "on"):
        return "true", True
    if text in ("false", "0", "no", "n", "off"):
        return "false", False
    raise ValueError("contact_meas_gate_by_hit must be one of: auto | true | false.")


def _normalize_posttrain_contact_meas_vxy_mode(value: Any) -> str:
    text = _normalize_text(value, default=DEFAULT_POSTTRAIN_CONTACT_MEAS_VXY_MODE)
    if text in ("abs", "absolute"):
        return "abs"
    if text in ("root", "root_rel", "root-relative", "rel", "relative"):
        return "root_rel"
    raise ValueError("contact_meas_vxy_mode must be one of: abs | root_rel.")


def _normalize_posttrain_contact_meas_ground_z_mode(value: Any) -> str:
    text = _normalize_text(value, default=DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_MODE)
    if text in ("ema", "window", "slew"):
        return text
    raise ValueError("contact_meas_ground_z_mode must be one of: ema | window | slew.")


def _normalize_posttrain_lambda_time_weight_mode(value: Any) -> str:
    text = _normalize_text(value, default=DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MODE)
    if text in ("inv", "inverse", "1/t", "one_over_t"):
        return "inv"
    if text in ("linear", "lin"):
        return "linear"
    if text in ("uniform", "flat", "ones", "one"):
        return "uniform"
    raise ValueError("lambda_time_weight_mode must be one of: inv | linear | uniform.")


def _normalize_posttrain_lambda_reliability_mode(value: Any) -> str:
    text = _normalize_text(value, default=DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_MODE)
    if text in ("", "none", "off", "false", "0", "disable", "disabled"):
        return "none"
    aliases = {
        "warmup": "warmup",
        "step_warmup": "warmup",
        "contacts_err": "contacts_err",
        "contact_err": "contacts_err",
    }
    tokens: list[str] = []
    for raw_token in text.replace(",", "+").split("+"):
        token = raw_token.strip()
        if not token:
            continue
        try:
            canonical = aliases[token]
        except KeyError as exc:
            raise ValueError(
                "lambda_reliability_mode must be one of: none | warmup | contacts_err | warmup+contacts_err."
            ) from exc
        if canonical not in tokens:
            tokens.append(canonical)
    return "+".join(tokens) if tokens else "none"


def _normalize_text(value: Any, *, default: str) -> str:
    if value is None:
        return str(default).strip().lower()
    text = str(value).strip().lower()
    return str(default).strip().lower() if text == "" else text


def _cfg_float_list(value: Any, *, field: str) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "none", "null", "off"):
            return None
        path = Path(text).expanduser()
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be a JSON list, JSON object with scales/values, or JSON file path.") from exc
    if isinstance(value, Mapping):
        if "scales" in value:
            value = value["scales"]
        elif "values" in value:
            value = value["values"]
        else:
            raise ValueError(f"{field} mapping must contain 'scales' or 'values'.")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of finite floats or None.")
    out: list[float] = []
    for item in value:
        try:
            item_float = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain only finite floats; got {item!r}.") from exc
        if not math.isfinite(item_float):
            raise ValueError(f"{field} must contain only finite floats; got {item_float!r}.")
        out.append(float(item_float))
    return out or None


def _normalize_contact_plan_inject(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_CONTACT_PLAN_INJECT).strip().lower()
    if text in ("none", "contacts", "plan_z"):
        return text
    raise ValueError("contact_plan_inject must be one of: none | contacts | plan_z.")


def _normalize_direct_pose_meas_mode(value: Any, *, field: str) -> str:
    text = str(value if value is not None else DEFAULT_DIRECT_POSE_MEAS_MODE).strip().lower()
    if text in ("concat", "mode_select"):
        return text
    raise ValueError(f"{field} must be one of: concat | mode_select.")


def _normalize_optional_direct_pose_meas_mode(value: Any, *, field: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null", "auto"):
        return None
    return _normalize_direct_pose_meas_mode(text, field=field)


def _even_nonnegative_dim(value: int, *, field: str) -> int:
    dim = int(value)
    if dim < 0:
        raise ValueError(f"{field} must be >= 0; got {dim}.")
    if dim % 2 != 0:
        raise ValueError(f"{field} must be even for sin/cos positional encoding; got {dim}.")
    return int(dim)


def _normalize_rot_local_tail_scope(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_LOSS_ROT_LOCAL_TAIL_SCOPE).strip().lower()
    if text in ("all", "limbs", "keybones"):
        return text
    raise ValueError("rot_local_tail_scope must be one of: all | limbs | keybones.")


def _normalize_rot_local_tail_select(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_LOSS_ROT_LOCAL_TAIL_SELECT).strip().lower()
    if text in ("batch", "ema"):
        return text
    raise ValueError("rot_local_tail_select must be one of: batch | ema.")


def _posttrain_direct_pose_time_pe_dim(cfg: Any) -> int:
    value = _cfg_int(cfg, "direct_pose_time_pe_dim", -1)
    if value >= 0:
        return _even_nonnegative_dim(value, field="direct_pose_time_pe_dim")
    return int(value)


def _posttrain_direct_pose_feat_source(cfg: Any) -> str:
    value = _cfg_value(cfg, "direct_pose_feat_source", "auto")
    text = str(value if value is not None else "auto").strip().lower()
    if text in ("", "auto"):
        return "auto"
    return normalize_direct_pose_feat_source(
        text,
        default=DEFAULT_DIRECT_POSE_FEAT_SOURCE,
        strict=True,
        context="direct_pose_feat_source",
    )


def _posttrain_direct_pose_phase_z_mode(cfg: Any) -> str:
    value = _cfg_value(cfg, "direct_pose_phase_z_mode", "auto")
    text = str(value if value is not None else "auto").strip().lower()
    if text in ("", "auto"):
        return "auto"
    return normalize_direct_pose_phase_z_mode(
        text,
        default="concat",
        strict=True,
        context="direct_pose_phase_z_mode",
    )
