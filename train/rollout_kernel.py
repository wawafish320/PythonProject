from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from .geometry import _rot6d_identity_like
from .history import (
    PoseHistState,
    advance_pose_hist_state,
    init_pose_hist_state,
    resolve_pose_hist_input,
)


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
class RolloutSequenceInputs:
    state_seq: torch.Tensor
    cond_seq: Optional[torch.Tensor] = None
    cond_raw_seq: Optional[torch.Tensor] = None
    contacts_seq: Optional[torch.Tensor] = None
    angvel_seq: Optional[torch.Tensor] = None
    pose_hist_seq: Optional[torch.Tensor] = None
    gt_seq: Optional[torch.Tensor] = None


def new_rollout_prediction_buffers() -> RolloutPredictionBuffers:
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
    meas_prev_prob: Optional[torch.Tensor] = None
    prev_foot_pos_meas: Optional[torch.Tensor] = None
    reprojection_applied_count: int = 0
    last_attn: Optional[torch.Tensor] = None
    latest_y_raw: Optional[torch.Tensor] = None
    latest_cond_raw_for_env: Optional[torch.Tensor] = None
    outs: List[torch.Tensor] = field(default_factory=list)
    delta_preds: List[torch.Tensor] = field(default_factory=list)
    buffers: RolloutPredictionBuffers = field(default_factory=new_rollout_prediction_buffers)


@dataclass(frozen=True)
class RolloutStepInputs:
    cond_input: Optional[torch.Tensor]
    contacts_in_t: Optional[torch.Tensor]
    angvel_t: Optional[torch.Tensor]
    pose_history_t: Optional[torch.Tensor]
    cond_raw_for_env: Any
    prev_foot_pos_meas: Optional[torch.Tensor]
    time_index_t: Optional[torch.Tensor]
    rollout_step_t: Optional[torch.Tensor]
    reprojection_applied: bool


def finalize_rollout_prediction_buffers(
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


def rollout_has_time_dim(value: Any) -> bool:
    dim_fn = getattr(value, "dim", None)
    return callable(dim_fn) and int(dim_fn()) == 3


def build_rollout_has_time_dim(rollout_inputs: RolloutSequenceInputs) -> Dict[str, bool]:
    return {
        "cond": rollout_has_time_dim(rollout_inputs.cond_seq),
        "cond_raw": rollout_has_time_dim(rollout_inputs.cond_raw_seq),
        "contacts": rollout_has_time_dim(rollout_inputs.contacts_seq),
        "angvel": rollout_has_time_dim(rollout_inputs.angvel_seq),
        "pose_hist": rollout_has_time_dim(rollout_inputs.pose_hist_seq),
    }


def normalize_pose_hist_state(value: Any) -> PoseHistState:
    if isinstance(value, PoseHistState):
        return value
    return PoseHistState(enabled=False, length=0, dim=0, stride=0)


def select_rollout_frame(
    value: Any,
    *,
    step_idx: int,
    has_time_dim: bool,
    offset: int = 0,
) -> Any:
    if has_time_dim and torch.is_tensor(value):
        idx = min(value.shape[1] - 1, max(0, int(step_idx) + int(offset)))
        return value[:, idx]
    return value


def resolve_rollout_pose_history(
    *,
    pose_hist_state: Any,
    pose_hist_seq: Optional[torch.Tensor],
    idx: int,
) -> tuple[PoseHistState, Optional[torch.Tensor]]:
    state = normalize_pose_hist_state(pose_hist_state)
    pose_history_t = resolve_pose_hist_input(
        state=state,
        pose_hist_seq=pose_hist_seq,
        idx=int(idx),
    )
    return state, pose_history_t


def prepare_rollout_pose_hist_state(
    trainer: Any,
    *,
    state_seq: torch.Tensor,
    pose_hist_seq: Optional[torch.Tensor],
    y_raw_local: Optional[torch.Tensor],
    rot6d_y_slice: slice,
    offset: int = 0,
    force_disable: Optional[bool] = None,
) -> PoseHistState:
    return init_pose_hist_state(
        ref_tensor=state_seq,
        pose_hist_seq=pose_hist_seq,
        y_prev_raw=y_raw_local,
        rot_slice=rot6d_y_slice,
        pose_hist_len=int(getattr(trainer, "pose_hist_len", 0) or 0),
        pose_hist_dim=int(getattr(trainer, "pose_hist_dim", 0) or 0),
        params_fn=trainer._pose_hist_params,
        offset=int(offset),
        force_disable=bool(
            getattr(trainer, "force_pose_hist_seq", False)
            if force_disable is None
            else force_disable
        ),
    )


def resolve_rollout_initial_raw_state(
    trainer: Any,
    state_seq: torch.Tensor,
    gt_seq: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], slice, slice]:
    motion = state_seq[:, 0]
    try:
        motion_raw_local = trainer.normalizer.denorm_x(motion)
    except Exception as exc:
        trainer._raise_norm_error("normalizer.denorm_x 在 roll-out 初始化时失败", exc)

    y_raw_local = None
    dy = 0
    if gt_seq is not None and gt_seq.dim() == 3:
        y0 = gt_seq[:, 0]
        dy = int(y0.shape[-1])
        y_raw_local = trainer._denorm(y0)
    if dy <= 0:
        dy = int(getattr(trainer, "Dy", 0) or (gt_seq.shape[-1] if gt_seq is not None else 0))
    rot6d_slice = getattr(getattr(trainer, "train_loader", None), "rot6d_x_slice", None)
    if rot6d_slice is None:
        rot6d_slice = getattr(trainer, "rot6d_x_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot6d_slice, slice):
        rot6d_slice = slice(0, motion.size(-1))
    rot6d_y_slice = getattr(trainer, "rot6d_y_slice", None) or rot6d_slice
    if y_raw_local is None and motion_raw_local is not None and dy:
        slice_len = rot6d_slice.stop - rot6d_slice.start
        if slice_len != dy:
            trainer._raise_norm_error("rollout 初始化 rot6d_x_slice 与 Dy 不匹配。")
        y_raw_local = motion_raw_local[..., rot6d_slice].clone()
    if y_raw_local is None and dy:
        joint_count = dy // 6
        if joint_count > 0:
            zeros = state_seq.new_zeros((state_seq.shape[0], joint_count, 6))
            y_raw_local = _rot6d_identity_like(zeros).view(state_seq.shape[0], dy)
    return motion, motion_raw_local, y_raw_local, rot6d_slice, rot6d_y_slice


def init_rollout_execution_state(
    trainer: Any,
    rollout_inputs: RolloutSequenceInputs,
    *,
    cond_norm_mu: Optional[torch.Tensor] = None,
    cond_norm_std: Optional[torch.Tensor] = None,
    mode: str = "mixed",
    tf_ratio: float = 1.0,
    time_base: Any = None,
) -> RolloutExecutionState:
    state_seq = rollout_inputs.state_seq
    gt_seq = rollout_inputs.gt_seq
    batch_size, total_steps, _ = state_seq.shape
    allow_grad = mode == "train_free"
    try:
        ss_chunk_len = int(getattr(trainer, "ss_chunk_len", 1) or 1)
    except (TypeError, ValueError):
        ss_chunk_len = 1
    ss_chunk_len = max(1, ss_chunk_len)
    motion, motion_raw_local, y_raw_local, rot6d_slice, rot6d_y_slice = resolve_rollout_initial_raw_state(
        trainer,
        state_seq,
        gt_seq,
    )
    pose_hist_state = prepare_rollout_pose_hist_state(
        trainer,
        state_seq=state_seq,
        pose_hist_seq=rollout_inputs.pose_hist_seq,
        y_raw_local=y_raw_local,
        rot6d_y_slice=rot6d_y_slice,
    )

    cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, state_seq) if cond_norm_mu is not None else None
    cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, state_seq) if cond_norm_std is not None else None

    enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))
    yaw_strategy = str(getattr(trainer, "freerun_yaw_strategy", "trajectory") or "trajectory")
    if yaw_strategy == "trajectory":
        enable_reprojection = False
    time_base_local = time_base
    if torch.is_tensor(time_base_local):
        try:
            time_base_local = time_base_local.to(device=state_seq.device)
        except RuntimeError:
            time_base_local = time_base

    return RolloutExecutionState(
        batch_size=batch_size,
        total_steps=total_steps,
        mode=mode,
        allow_grad=allow_grad,
        tf_ratio=float(tf_ratio),
        ss_chunk_len=ss_chunk_len,
        amp_enabled=bool(getattr(trainer, "use_amp", False)),
        rot6d_slice=rot6d_slice,
        rot6d_y_slice=rot6d_y_slice,
        has_time_dim=build_rollout_has_time_dim(rollout_inputs),
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
        enable_reprojection=enable_reprojection,
        plan_enable=bool(getattr(getattr(trainer, "model", None), "contact_plan_enable", False)),
        time_base_local=time_base_local,
        motion=motion,
        motion_raw_local=motion_raw_local,
        y_raw_local=y_raw_local,
        pose_hist_state=pose_hist_state,
    )


def prepare_rollout_cond(
    trainer: Any,
    *,
    cond_seq: Optional[torch.Tensor],
    cond_raw_seq: Optional[torch.Tensor],
    cond_norm_mu: Optional[torch.Tensor],
    cond_norm_std: Optional[torch.Tensor],
    step_idx: int,
    cond_idx: Optional[int] = None,
    cond_has_time_dim: bool = True,
    cond_raw_has_time_dim: bool = True,
    cond_raw_offset: int = 1,
    include_boundary: bool = False,
    cycle_len: int = 1,
    y_prev_raw: Optional[torch.Tensor],
    allow_reprojection: bool,
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
) -> tuple[Optional[torch.Tensor], Any, Any, bool]:
    idx = int(step_idx if cond_idx is None else cond_idx)
    cond_t = select_rollout_frame(
        cond_seq,
        step_idx=idx,
        has_time_dim=bool(cond_has_time_dim),
    )
    cond_raw_step = None
    if torch.is_tensor(cond_raw_seq):
        if bool(cond_raw_has_time_dim) and cond_raw_seq.dim() == 3:
            if include_boundary:
                idx_raw = int((idx + int(cond_raw_offset)) % max(1, int(cycle_len)))
            else:
                idx_raw = min(int(cond_raw_seq.shape[1]) - 1, idx + int(cond_raw_offset))
            cond_raw_step = cond_raw_seq[:, idx_raw]
        else:
            cond_raw_step = cond_raw_seq

    cond_raw_for_model = cond_raw_step
    reprojection_applied = False
    if bool(allow_reprojection) and torch.is_tensor(cond_raw_step):
        yaw_gt = None
        if callable(yaw_gt_fn):
            try:
                yaw_gt = yaw_gt_fn(int(idx))
            except Exception:
                yaw_gt = None
        yaw_pred = None
        if y_prev_raw is not None:
            try:
                yaw_pred = trainer._infer_root_yaw_from_rot6d(y_prev_raw)
            except Exception:
                yaw_pred = None
        if yaw_gt is not None and yaw_pred is not None:
            cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, yaw_gt, yaw_pred)
            if cond_proj is not None:
                cond_raw_for_model = cond_proj
                reprojection_applied = True

    if cond_raw_for_model is not None:
        try:
            cond_override = trainer._normalize_cond_from_raw(
                cond_raw_for_model,
                cond_norm_mu,
                cond_norm_std,
            )
        except Exception:
            cond_override = None
        if cond_override is not None:
            cond_t = cond_override
    return cond_t, cond_raw_step, cond_raw_for_model, reprojection_applied


def resolve_rollout_step_angvel(
    trainer: Any,
    *,
    motion: torch.Tensor,
    angvel_seq: Optional[torch.Tensor],
    step_idx: int,
    has_time_dim: bool,
) -> Optional[torch.Tensor]:
    angvel_slice = getattr(trainer, "angvel_x_slice", None)
    if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(angvel_slice, slice):
        return motion[..., angvel_slice].detach()
    return select_rollout_frame(
        angvel_seq,
        step_idx=step_idx,
        has_time_dim=bool(has_time_dim),
    )


def prepare_rollout_contacts_input(
    trainer: Any,
    model: Any,
    *,
    motion_t: torch.Tensor,
    pose_hist_t: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not bool(getattr(model, "contact_plan_enable", False)):
        return None
    contacts_in_t = None
    fn = getattr(trainer, "_predict_pretrain_contacts_from_frozen", None)
    if callable(fn):
        try:
            contacts_in_t = fn(motion_step_t=motion_t, pose_hist_step_t=pose_hist_t)
        except Exception:
            contacts_in_t = None
    if contacts_in_t is None:
        raise RuntimeError(
            "[FATAL] rollout contact resolution requires valid frozen encoder+contact_head "
            "and runtime-compatible encoder input dimensions."
        )
    return contacts_in_t


def build_rollout_step_time_inputs(
    ref_tensor: torch.Tensor,
    *,
    total_steps: int,
    step_idx: int,
    time_base: Any = None,
    time_index_seed: Optional[int] = None,
) -> tuple[Any, Optional[torch.Tensor]]:
    seed = time_index_seed
    time_index_t = seed
    if time_base is not None:
        base_offset = int(step_idx if seed is None else seed)
        try:
            time_index_t = time_base + base_offset
        except (RuntimeError, TypeError, ValueError):
            time_index_t = seed
    try:
        step_norm = (
            float(step_idx) / float(int(total_steps) - 1)
            if int(total_steps) > 1
            else 0.0
        )
        rollout_step_t = torch.full(
            (ref_tensor.shape[0], 1, 1),
            step_norm,
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
    except (RuntimeError, TypeError, ValueError):
        rollout_step_t = None
    return time_index_t, rollout_step_t


def resolve_rollout_time_controls(
    *,
    time_index_mode: str,
    time_base: Any,
    frame_idx: int,
    rollout_step_idx: int,
) -> tuple[Any, Optional[int]]:
    mode = str(time_index_mode)
    time_base_local = time_base
    if mode == "none":
        return None, None
    if mode == "cycle":
        return time_base_local, int(frame_idx)
    if time_base is not None:
        return time_base_local, int(frame_idx)
    return None, int(rollout_step_idx)


def ensure_rollout_time_axis(value: Any) -> Any:
    if torch.is_tensor(value) and value.dim() == 2:
        return value.unsqueeze(1)
    return value


def forward_rollout_model_step(
    model: Any,
    *,
    motion: torch.Tensor,
    cond_input: Optional[torch.Tensor],
    contacts_in_t: Optional[torch.Tensor],
    angvel_t: Optional[torch.Tensor],
    pose_history_t: Optional[torch.Tensor],
    plan_z: Any,
    meas_logits_prev: Any,
    time_index_t: Any,
    rollout_step_t: Optional[torch.Tensor],
) -> tuple[Mapping[str, Any], torch.Tensor, Optional[torch.Tensor]]:
    ret = model(
        motion,
        cond_input,
        contacts=contacts_in_t,
        angvel=angvel_t,
        pose_history=pose_history_t,
        plan_z=plan_z,
        meas_logits_prev=meas_logits_prev,
        time_index=time_index_t,
        rollout_step=rollout_step_t,
    )
    if not isinstance(ret, dict):
        raise RuntimeError("Model forward must return a dict with at least 'out'.")
    delta_out = ret.get("delta", ret.get("out", None))
    if delta_out is None:
        raise RuntimeError("Model forward must return 'delta' tensor.")
    return ret, delta_out, ret.get("period_pred", None)


def resolve_rollout_step_inputs(
    trainer: Any,
    rollout: RolloutExecutionState,
    rollout_inputs: RolloutSequenceInputs,
    *,
    step_idx: int,
    yaw_gt_fn: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    model: Any = None,
    time_index_seed: Optional[int] = None,
) -> RolloutStepInputs:
    _, pose_history_t = resolve_rollout_pose_history(
        pose_hist_state=rollout.pose_hist_state,
        pose_hist_seq=rollout_inputs.pose_hist_seq,
        idx=int(step_idx),
    )
    contacts_in_t = prepare_rollout_contacts_input(
        trainer,
        model if model is not None else getattr(trainer, "model", None),
        motion_t=rollout.motion,
        pose_hist_t=pose_history_t,
    )
    cond_input, cond_raw_for_env, _, reprojection_applied = prepare_rollout_cond(
        trainer,
        cond_seq=rollout_inputs.cond_seq,
        cond_raw_seq=rollout_inputs.cond_raw_seq,
        cond_norm_mu=rollout.cond_norm_mu,
        cond_norm_std=rollout.cond_norm_std,
        step_idx=int(step_idx),
        cond_has_time_dim=bool(rollout.has_time_dim["cond"]),
        cond_raw_has_time_dim=bool(rollout.has_time_dim["cond_raw"]),
        cond_raw_offset=1,
        include_boundary=False,
        cycle_len=int(rollout.total_steps),
        y_prev_raw=rollout.y_raw_local,
        allow_reprojection=bool(
            rollout.enable_reprojection
            and int(step_idx) > 0
            and rollout.mode in ("free", "train_free", "mixed")
        ),
        yaw_gt_fn=yaw_gt_fn,
    )
    time_index_t, rollout_step_t = build_rollout_step_time_inputs(
        rollout.motion,
        total_steps=rollout.total_steps,
        step_idx=int(step_idx),
        time_base=rollout.time_base_local,
        time_index_seed=time_index_seed,
    )
    return RolloutStepInputs(
        cond_input=cond_input,
        contacts_in_t=contacts_in_t,
        angvel_t=resolve_rollout_step_angvel(
            trainer,
            motion=rollout.motion,
            angvel_seq=rollout_inputs.angvel_seq,
            step_idx=int(step_idx),
            has_time_dim=bool(rollout.has_time_dim["angvel"]),
        ),
        pose_history_t=pose_history_t,
        cond_raw_for_env=cond_raw_for_env,
        prev_foot_pos_meas=rollout.prev_foot_pos_meas,
        time_index_t=time_index_t,
        rollout_step_t=rollout_step_t,
        reprojection_applied=reprojection_applied,
    )


def update_rollout_carry_state(
    trainer: Any,
    rollout: RolloutExecutionState,
    rollout_inputs: RolloutSequenceInputs,
    *,
    step_idx: int,
    warn_once_fn: Optional[Callable[[str, str, Optional[BaseException]], None]] = None,
) -> RolloutExecutionState:
    if rollout.motion_raw_local is None:
        trainer._raise_norm_error("rollout 更新需要 DataNormalizer 提供 RAW 状态写回。")
    free_raw = trainer._apply_free_carry(
        rollout.motion_raw_local,
        rollout.latest_y_raw,
        cond_next_raw=rollout.latest_cond_raw_for_env,
    )
    free_raw = free_raw.clone() if rollout.allow_grad else free_raw.detach()
    free_z = trainer._diag_norm_x(free_raw)
    gt_next = rollout_inputs.state_seq[:, step_idx + 1]
    ss_sel_hold = rollout.ss_sel_hold
    if ss_sel_hold is None or rollout.ss_chunk_len <= 1 or (step_idx % rollout.ss_chunk_len == 0):
        rand_device = getattr(trainer, "device", gt_next.device)
        ss_sel_hold = (
            torch.rand(rollout.batch_size, device=rand_device) < float(rollout.tf_ratio)
        ).float().unsqueeze(-1)
    sel = ss_sel_hold if ss_sel_hold.dtype == gt_next.dtype else ss_sel_hold.to(gt_next.dtype)
    motion = sel * gt_next + (1.0 - sel) * free_z

    try:
        gt_raw_next = trainer.normalizer.denorm_x(gt_next, prev_raw=rollout.motion_raw_local)
        motion_raw_local = sel * gt_raw_next + (1.0 - sel) * free_raw
    except Exception as exc:
        trainer._raise_norm_error("normalizer.denorm_x 在 rollout 更新时失败", exc)

    y_raw_local = rollout.y_raw_local
    if rollout_inputs.gt_seq is not None and rollout_inputs.gt_seq.dim() == 3:
        try:
            y_next = trainer._denorm(rollout_inputs.gt_seq[:, step_idx + 1]).detach()
            y_raw_local = sel * y_next + (1.0 - sel) * y_raw_local
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            if callable(warn_once_fn):
                warn_once_fn(
                    "rollout/carry_gt_denorm",
                    "failed to blend GT RAW carry in scheduled sampling; keeping model RAW carry",
                    exc,
                )

    pose_hist_state = advance_pose_hist_state(
        rollout.pose_hist_state,
        y_next_raw=y_raw_local,
        rot_slice=rollout.rot6d_y_slice,
    )
    rollout.motion = motion
    rollout.motion_raw_local = motion_raw_local
    rollout.y_raw_local = y_raw_local
    rollout.ss_sel_hold = ss_sel_hold
    rollout.pose_hist_state = pose_hist_state
    return rollout


def update_rollout_recurrent_state(
    model: Any,
    ret: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    if bool(getattr(model, "contact_plan_enable", False)):
        try:
            z_next = ret.get("plan_z_next", None)
            if torch.is_tensor(z_next):
                state["plan_z"] = z_next.detach()
        except Exception:
            pass
    try:
        mlog = ret.get("contacts_meas_logits", None)
        if torch.is_tensor(mlog):
            if mlog.dim() == 3:
                state["meas_logits_prev"] = mlog[:, -1].detach()
            elif mlog.dim() == 2:
                state["meas_logits_prev"] = mlog.detach()
    except Exception:
        pass


def apply_rollout_carry_state(
    trainer: Any,
    state: Dict[str, Any],
    *,
    y_next_raw: torch.Tensor,
    cond_raw_step: Optional[torch.Tensor],
) -> None:
    cond_env = cond_raw_step if torch.is_tensor(cond_raw_step) else None
    motion_raw = trainer._apply_free_carry(state["motion_raw"], y_next_raw, cond_next_raw=cond_env)
    motion_raw = torch.nan_to_num(motion_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    motion = trainer._diag_norm_x(motion_raw)
    state["motion_raw"] = motion_raw
    state["motion"] = motion

    pose_hist_state = normalize_pose_hist_state(state.get("pose_hist_state", None))
    state["pose_hist_state"] = advance_pose_hist_state(
        pose_hist_state,
        y_next_raw=y_next_raw,
        rot_slice=state.get("rot_slice", None),
    )

    state["y_prev_raw"] = y_next_raw


def get_rollout_step_tensor(
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


def update_rollout_plan_state(
    rollout: RolloutExecutionState,
    ret: Mapping[str, Any],
) -> None:
    if not rollout.plan_enable:
        return

    contacts_plan = get_rollout_step_tensor(ret, "contacts_plan")
    if contacts_plan is not None:
        rollout.buffers.contacts_plan.append(contacts_plan)

    contacts_plan_logits = get_rollout_step_tensor(ret, "contacts_plan_logits")
    if contacts_plan_logits is not None:
        rollout.buffers.contacts_plan_logits.append(contacts_plan_logits)

    direct_out = get_rollout_step_tensor(ret, "out_direct")
    if direct_out is not None:
        rollout.buffers.out_direct.append(direct_out)

    plan_z_next = ret.get("plan_z_next", None)
    if plan_z_next is not None:
        if torch.is_tensor(plan_z_next) and not rollout.allow_grad:
            plan_z_next = plan_z_next.detach()
        rollout.plan_z = plan_z_next


def record_rollout_step_outputs(
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

    hidden_step = ret.get("h_final", None)
    if torch.is_tensor(hidden_step):
        if hidden_step.dim() == 1:
            hidden_step = hidden_step.unsqueeze(0).unsqueeze(0)
        elif hidden_step.dim() == 2:
            hidden_step = hidden_step.unsqueeze(1)
        elif hidden_step.dim() >= 3 and hidden_step.size(1) != 1:
            hidden_step = hidden_step[:, -1:, ...]
        rollout.buffers.hidden_seq.append(hidden_step)

    for key, target in (
        ("contacts_meas", rollout.buffers.contacts_meas),
        ("contacts_err", rollout.buffers.contacts_err),
        ("event_clock_lambda_logit", rollout.buffers.event_clock_lambda_logit),
        ("event_clock_dynamic_prior", rollout.buffers.event_clock_dynamic_prior),
        ("event_clock_delta_z", rollout.buffers.event_clock_delta_z),
    ):
        value = get_rollout_step_tensor(ret, key)
        if value is not None:
            target.append(value)


def finalize_rollout_outputs(rollout: RolloutExecutionState) -> dict[str, torch.Tensor]:
    preds = {
        "out": torch.stack(rollout.outs, dim=1),
        "delta": torch.stack(rollout.delta_preds, dim=1),
    }
    finalize_rollout_prediction_buffers(preds, rollout.buffers)
    return preds
