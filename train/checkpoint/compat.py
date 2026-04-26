from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

if TYPE_CHECKING:
    from ..models import EventMotionModel

__all__ = [
    "ContactPlanBuildConfig",
    "ContactPlanBuildOverrides",
    "DirectPoseBuildConfig",
    "DirectPoseBuildOverrides",
    "DirectPoseCkptInference",
    "DirectPoseLegBuildConfig",
    "DirectPoseLoadCompatOptions",
    "EventMotionBuildState",
    "EventMotionCkptPayload",
    "EventMotionCoreDims",
    "EventClockBuildConfig",
    "EventClockBuildOverrides",
    "LambdaFusionBuildConfig",
    "LambdaFusionBuildOverrides",
    "ResumeLoadReport",
    "apply_direct_pose_ckpt_compat",
    "infer_event_motion_core_dims_from_ckpt",
    "infer_event_clock_build_cfg",
    "infer_lambda_fusion_build_cfg",
    "load_event_motion_ckpt_payload",
    "normalize_direct_pose_split_state_dict_schema",
    "prepare_event_motion_ckpt_state_for_load",
    "resume_load_weights_compat",
    "resolve_contact_plan_build_cfg",
    "resolve_direct_pose_build_cfg",
    "resolve_direct_pose_leg_build_cfg",
    "resolve_event_motion_build_state_from_ckpt",
]
_RETIRED_DIRECT_POSE_STEPC_LEG_PREFIXES: tuple[str, ...] = (
    "direct_pose_out_leg.",
)
_DIRECT_POSE_FEAT_SOURCE_CANONICAL: tuple[str, ...] = (
    "cond",
    "hidden",
    "hidden_pre",
    "cond+hidden",
    "cond+hidden_pre",
)
_DIRECT_POSE_FEAT_SOURCE_ALIAS_MAP: dict[str, str] = {
    "h": "hidden",
    "h_final": "hidden",
    "hidden_only": "hidden",
    "post": "hidden",
    "final": "hidden",
    "h_pre": "hidden_pre",
    "h_temporal": "hidden_pre",
    "pre": "hidden_pre",
    "temporal": "hidden_pre",
    "mid": "hidden_pre",
    "cond_hidden": "cond+hidden",
    "hidden_cond": "cond+hidden",
    "concat": "cond+hidden",
    "hidden+cond": "cond+hidden",
    "cond_hidden_pre": "cond+hidden_pre",
    "hidden_pre+cond": "cond+hidden_pre",
    "cond+pre": "cond+hidden_pre",
    "pre+cond": "cond+hidden_pre",
}
_DIRECT_POSE_LEG_GATE_ALIAS_MAP: dict[str, str] = {
    "": "none",
    "auto": "none",
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "no": "none",
    "n": "none",
    "disable": "none",
    "disabled": "none",
    "learned": "learned",
    "on": "learned",
    "true": "learned",
    "1": "learned",
    "yes": "learned",
    "y": "learned",
    "scale": "scale",
    "mag": "scale",
    "magnitude": "scale",
    "logmag": "scale",
    "log_mag": "scale",
    "exp": "scale",
    "alpha": "scale",
}


class RemovedCheckpointCompatError(RuntimeError):
    """Raised when a checkpoint enters a removed semantic compat path."""


class RetiredDirectPoseLayoutError(RemovedCheckpointCompatError):
    """Raised when a checkpoint still uses a retired direct-pose layout."""


_DIRECT_POSE_COMPAT_REMOVAL_CHANGE = "2026-04-25 semantic checkpoint compat removal"
_DIRECT_POSE_COMPAT_MIGRATION = (
    "retrain/resave with the current checkpoint schema; no in-loader replacement"
)
_DIRECT_POSE_WEIGHT_PREFIXES: tuple[str, ...] = (
    "direct_pose_head.",
    "direct_pose_leg_terminal.",
    "direct_pose_out_nonleg.",
    "direct_pose_out_arm.",
    "direct_pose_out_else.",
    "direct_pose_nonleg_proj.",
    "direct_pose_arm_proj.",
    "direct_pose_else_proj.",
    "direct_pose_leg_head.",
    "direct_pose_leg_gate_head.",
    "direct_pose_leg_head_shared.",
    "direct_pose_leg_gate_head_shared.",
    "direct_pose_leg_side_sign_gate_head.",
    "direct_pose_leg_side_embed.",
)
_DIRECT_POSE_WEIGHT_EXACT_KEYS: tuple[str, ...] = (
    "direct_pose_leg_joint_idx_tensor",
    "direct_pose_leg_side_pos_r_tensor",
    "direct_pose_leg_side_pos_l_tensor",
    "direct_pose_leg_out_idx",
    "direct_pose_nonleg_out_idx",
    "direct_pose_arm_out_idx",
    "direct_pose_else_out_idx",
)
_RETIRED_DIRECT_POSE_HIGHORDER_PREFIXES: tuple[str, ...] = (
    "direct_pose_leg_head_shared.",
    "direct_pose_leg_gate_head_shared.",
    "direct_pose_leg_side_sign_gate_head.",
    "direct_pose_leg_side_embed.",
)
_RETIRED_DIRECT_POSE_HIGHORDER_EXACT_KEYS: tuple[str, ...] = (
    "direct_pose_leg_side_pos_r_tensor",
    "direct_pose_leg_side_pos_l_tensor",
)
_DIRECT_POSE_PHASE_INPUT_KEYS: tuple[str, ...] = (
    "direct_pose_head.0.weight",
    "direct_pose_leg_head.0.weight",
    "direct_pose_leg_head_shared.0.weight",
    "direct_pose_leg_gate_head.0.weight",
    "direct_pose_leg_gate_head_shared.0.weight",
)
_DIRECT_POSE_LEG_PREFIXES: tuple[str, ...] = (
    "direct_pose_leg_head.",
    "direct_pose_leg_gate_head.",
    "direct_pose_leg_head_shared.",
    "direct_pose_leg_gate_head_shared.",
    "direct_pose_leg_side_sign_gate_head.",
    "direct_pose_leg_side_embed.",
)
_DIRECT_POSE_LEG_EXACT_KEYS: tuple[str, ...] = (
    "direct_pose_leg_joint_idx_tensor",
    "direct_pose_leg_side_pos_r_tensor",
    "direct_pose_leg_side_pos_l_tensor",
)
_DIRECT_POSE_SPLIT_INDEX_KEYS: tuple[str, ...] = (
    "direct_pose_leg_out_idx",
    "direct_pose_nonleg_out_idx",
    "direct_pose_arm_out_idx",
    "direct_pose_else_out_idx",
)
_INDEX_BUFFER_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _preview_items(items: Sequence[str], *, limit: int = 5) -> str:
    preview = ", ".join(items[:limit])
    if len(items) > limit:
        preview = f"{preview}, ..."
    return preview


def _find_state_keys(
    state_dict: Any,
    *,
    prefixes: tuple[str, ...] = (),
    exact: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if not isinstance(state_dict, dict):
        return ()
    exact_set = set(exact)
    return tuple(
        sorted(
            str(key)
            for key in state_dict.keys()
            if str(key) in exact_set or str(key).startswith(prefixes)
        )
    )


def _raise_removed_checkpoint_compat(
    *,
    context: str,
    key_pattern: str,
    hits: Sequence[str],
    reason: str,
    migration: str = _DIRECT_POSE_COMPAT_MIGRATION,
) -> None:
    hit_text = _preview_items(tuple(str(item) for item in hits))
    raise RemovedCheckpointCompatError(
        f"{context}: [Removed] ckpt key pattern `{key_pattern}` matched [{hit_text}]. "
        f"Removed by {_DIRECT_POSE_COMPAT_REMOVAL_CHANGE}: {reason}. "
        f"Migration: {migration}."
    )


def _raise_removed_checkpoint_field(
    *,
    context: str,
    field_name: str,
    reason: str,
    migration: str = _DIRECT_POSE_COMPAT_MIGRATION,
) -> None:
    raise RemovedCheckpointCompatError(
        f"{context}: [Removed] checkpoint/config field `{field_name}` entered a removed path. "
        f"Removed by {_DIRECT_POSE_COMPAT_REMOVAL_CHANGE}: {reason}. "
        f"Migration: {migration}."
    )


def _find_retired_direct_pose_stepc_leg_keys(state_dict: Any) -> tuple[str, ...]:
    if not isinstance(state_dict, dict):
        return ()
    return tuple(
        sorted(
            str(key)
            for key in state_dict.keys()
            if str(key).startswith(_RETIRED_DIRECT_POSE_STEPC_LEG_PREFIXES)
        )
    )


def _raise_if_retired_direct_pose_stepc_leg_layout(
    state_dict: Any,
    *,
    context: str,
) -> None:
    retired_keys = _find_retired_direct_pose_stepc_leg_keys(state_dict)
    if not retired_keys:
        return
    raise RetiredDirectPoseLayoutError(
        f"{context}: [Removed] ckpt key pattern `direct_pose_out_leg.*` matched "
        f"[{_preview_items(retired_keys)}]. Removed by {_DIRECT_POSE_COMPAT_REMOVAL_CHANGE}: "
        "retired stepc-leg direct-pose layout cannot be bit-exactly mapped to the current "
        "`direct_pose_leg_terminal.*` layout. Migration: "
        f"{_DIRECT_POSE_COMPAT_MIGRATION}."
    )


@dataclass(frozen=True)
class EventMotionCkptPayload:
    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    width: int
    period_dim: int
    stripped_frozen_key_count: int


@dataclass(frozen=True)
class ResumeLoadReport:
    ckpt_path: Optional[Path]
    loaded_count: int
    total_count: int
    missing_count: int
    unexpected_count: int
    skipped_shape_count: int
    warning: Optional[str] = None


@dataclass(frozen=True)
class EventMotionCoreDims:
    in_state_dim: int
    out_motion_dim: int
    cond_dim: int
    contact_dim: int
    angvel_dim: int
    pose_hist_dim: int
    period_dim: int


@dataclass(frozen=True)
class ContactPlanBuildConfig:
    enable: bool
    hidden: int
    inject: str
    time_pe_dim: int
    init_mode: str
    init_hidden: int
    init_dropout: float


@dataclass(frozen=True)
class ContactPlanBuildOverrides:
    init_mode: Optional[str] = None
    init_hidden: Optional[int] = None
    init_dropout: Optional[float] = None


@dataclass(frozen=True)
class DirectPoseBuildConfig:
    """Resolved direct-pose build decision for EventMotionModel."""

    enable: bool
    hidden: int
    meas_mode: str
    feat_source: str
    time_pe_dim: int
    time_pe_base: float
    use_phase_z: bool
    phase_z_mode: str
    split_enable: bool
    arm_split_enable: bool
    arm_bones: Any
    nonleg_proj_dim: int
    drop_ckpt_weights: bool


@dataclass(frozen=True)
class DirectPoseCkptInference:
    """Observed direct-pose layout facts inferred from checkpoint tensors."""

    has_weights: bool
    enable: bool
    hidden: int
    meas_mode: str
    feat_source: str
    time_pe_dim: int
    phase_z_mode: str
    split_enable: bool
    arm_split_enable: bool
    nonleg_proj_dim: int


@dataclass(frozen=True)
class DirectPoseBuildOverrides:
    train_direct_pose: bool
    direct_pose_reinit: bool
    hidden_override: Optional[int] = None
    meas_mode_override: Optional[str] = None
    feat_source: str = "auto"
    time_pe_dim: int = -1
    time_pe_base: float = 10000.0
    use_phase_z: Optional[bool] = None
    phase_z_mode: str = "auto"
    split_enable: Optional[bool] = None
    arm_split_enable: Optional[bool] = None
    arm_bones: Any = None
    nonleg_proj_dim: Optional[int] = None


@dataclass(frozen=True)
class DirectPoseLoadCompatOptions:
    train_direct_pose: bool
    leg_enable: bool
    leg_bones: Any = None


@dataclass(frozen=True)
class DirectPoseLegBuildConfig:
    enable: bool
    bones: Any
    mode: str
    stopgrad_main: bool
    detach_feat: bool
    max_deg: float
    side_routing: bool
    contact_order: str
    side_embed_dim: int
    side_plan_other: bool
    side_phase_other: bool
    side_phase_rel: bool
    side_cue: str
    side_cue_tau: float
    side_sign_gate: bool
    side_rank1: bool
    gate_mode: str
    gate_power: float
    scale_log_clip: float
    scale_clamp_k: float


@dataclass(frozen=True)
class EventClockBuildConfig:
    use_event_clock: bool
    hidden_dim: int
    gate_hidden_dim: int
    max_delta: float
    period_dim_init: int


@dataclass(frozen=True)
class EventClockBuildOverrides:
    mode: str = "auto"
    hidden_dim: Optional[int] = None
    gate_hidden_dim: Optional[int] = None
    max_delta: float = 0.5
    has_encoder_bundle: bool = False


@dataclass(frozen=True)
class LambdaFusionBuildConfig:
    enable: bool
    mode: str
    hidden: int
    dropout: float
    logit_init: float
    use_rollout_step: bool


@dataclass(frozen=True)
class LambdaFusionBuildOverrides:
    train_lambda_head: bool = False
    mode: str = "per_joint"
    hidden: int = 128
    dropout: float = 0.0
    logit_init: float = -2.0
    use_rollout_step: bool = False


@dataclass(frozen=True)
class EventMotionBuildState:
    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    width: int
    period_dim: int
    contact_plan_cfg: ContactPlanBuildConfig
    direct_pose_cfg: DirectPoseBuildConfig
    event_clock_cfg: EventClockBuildConfig
    lambda_fusion_cfg: LambdaFusionBuildConfig
    direct_pose_leg_cfg: DirectPoseLegBuildConfig


def _normalize_contact_plan_init_mode(value: Any) -> str:
    mode = str(value or "learnable").strip().lower()
    if mode in ("learnable_obs", "obs+learnable"):
        return "learnable+obs"
    return mode or "learnable"


def load_event_motion_ckpt_payload(
    ckpt_or_path: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> EventMotionCkptPayload:
    ckpt = (
        torch.load(ckpt_or_path, map_location=map_location)
        if isinstance(ckpt_or_path, (str, bytes, bytearray)) or hasattr(ckpt_or_path, "__fspath__")
        else ckpt_or_path
    )
    ckpt_posttrain_cfg = None
    if isinstance(ckpt, dict):
        cfg = ckpt.get("posttrain_cfg", None)
        if isinstance(cfg, dict):
            ckpt_posttrain_cfg = dict(cfg)
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_model_state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_model_state = ckpt["model_state_dict"]
    else:
        raw_model_state = ckpt
    if not isinstance(raw_model_state, dict):
        raise TypeError("Checkpoint payload does not contain a model state_dict.")
    state_dict = {}
    stripped = 0
    for key, value in raw_model_state.items():
        if (
            str(key).startswith("frozen_encoder.")
            or str(key).startswith("frozen_period_head.")
            or str(key).startswith("contact_plan_input_proj.")
        ):
            stripped += 1
            continue
        state_dict[str(key)] = value
    shared_encoder_weight = state_dict.get("shared_encoder.0.weight", None)
    if not (torch.is_tensor(shared_encoder_weight) and shared_encoder_weight.ndim == 2):
        raise KeyError("Checkpoint missing 'shared_encoder.0.weight' to infer hidden width.")
    width = int(shared_encoder_weight.shape[0])
    period_weight = state_dict.get("period_encoder.weight", None)
    period_dim = int(period_weight.shape[1]) if torch.is_tensor(period_weight) and period_weight.ndim == 2 else 0
    return EventMotionCkptPayload(
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        state_dict=state_dict,
        width=int(width),
        period_dim=int(period_dim),
        stripped_frozen_key_count=int(stripped),
    )


def attach_motion_encoder_bundle(
    model: "EventMotionModel",
    bundle: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    from ..models import MotionEncoder, PeriodHead
    from ..contracts.asset_semantics import require_standard_rotvec_bundle

    payload = (
        torch.load(bundle, map_location=map_location)
        if isinstance(bundle, (str, bytes, bytearray)) or hasattr(bundle, "__fspath__")
        else bundle
    )
    if not isinstance(payload, dict):
        raise TypeError("MotionEncoder bundle must be a dict or path to a dict.")
    require_standard_rotvec_bundle(payload, context="MotionEncoder bundle")

    encoder_state = payload.get("encoder")
    period_state = payload.get("period_head")
    contact_state = payload.get("contact_head")
    if encoder_state is None or period_state is None:
        raise KeyError("Bundle missing 'encoder' or 'period_head' state_dict.")

    meta = dict(payload.get("meta", {}))
    hint_mode = str(meta.get("period_hint_mode") or "").strip() or "contacts_tanh"
    if hint_mode != "contacts_tanh":
        raise ValueError(f"Unsupported MotionEncoder bundle period_hint_mode={hint_mode!r} (expected 'contacts_tanh').")
    weight0 = encoder_state.get("mlp.0.weight")
    if weight0 is None:
        weight0 = next(
            (value for key, value in encoder_state.items() if str(key).endswith("weight") and getattr(value, "ndim", None) == 2),
            None,
        )
    if weight0 is None:
        raise ValueError("Unable to infer MotionEncoder dimensions from state_dict.")

    input_dim = int(meta.get("input_dim", weight0.shape[1]))
    hidden_dim = int(meta.get("hidden_dim", weight0.shape[0]))
    encoder = MotionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=int(meta.get("z_dim", 0)),
        num_layers=int(meta.get("mlp_layers", 3)),
        dropout=float(meta.get("mlp_dropout", 0.0)),
    )
    encoder.load_state_dict(encoder_state)
    encoder.eval().requires_grad_(False)

    period_dim = int(period_state["fc.weight"].shape[0])
    period_head = PeriodHead(hidden_dim, period_dim)
    period_head.load_state_dict(period_state)
    period_head.eval().requires_grad_(False)

    frozen_contact_head = None
    if isinstance(contact_state, dict):
        try:
            weight = contact_state.get("fc.weight")
            bias = contact_state.get("fc.bias")
            if torch.is_tensor(weight):
                frozen_contact_head = torch.nn.Linear(hidden_dim, int(weight.shape[0]))
                state = {"weight": weight}
                if torch.is_tensor(bias):
                    state["bias"] = bias
                frozen_contact_head.load_state_dict(state, strict=False)
                frozen_contact_head.eval().requires_grad_(False)
        except Exception:
            frozen_contact_head = None

    if model.encoder_input_dim and model.encoder_input_dim != input_dim:
        raise ValueError(f"Encoder input dim mismatch: dataset={model.encoder_input_dim} vs bundle={input_dim}")
    model.encoder_input_dim = input_dim

    device = model._target_device()
    model.frozen_encoder = encoder.to(device)
    model.frozen_period_head = period_head.to(device)
    model.frozen_contact_head = frozen_contact_head.to(device) if frozen_contact_head is not None else None
    if model.period_dim != period_dim or model.period_encoder is None:
        model.period_dim = period_dim
        model.period_encoder = torch.nn.Linear(model.period_dim, model.hidden_dim).to(device)
    return meta


def resume_load_weights_compat(
    model: torch.nn.Module,
    resume_path: Optional[str],
) -> Optional[ResumeLoadReport]:
    if not resume_path:
        return None

    ckpt_path = Path(str(resume_path)).expanduser()
    if not ckpt_path.is_file():
        warning = f'checkpoint not found: {ckpt_path}'
        print(f'[Resume][WARN] {warning}')
        return ResumeLoadReport(
            ckpt_path=ckpt_path,
            loaded_count=0,
            total_count=0,
            missing_count=0,
            unexpected_count=0,
            skipped_shape_count=0,
            warning=warning,
        )

    try:
        payload = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = payload.get('model', payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            warning = f'checkpoint has no state_dict: {ckpt_path}'
            print(f'[Resume][WARN] {warning}')
            return ResumeLoadReport(
                ckpt_path=ckpt_path,
                loaded_count=0,
                total_count=0,
                missing_count=0,
                unexpected_count=0,
                skipped_shape_count=0,
                warning=warning,
            )

        _raise_if_retired_direct_pose_stepc_leg_layout(
            state_dict,
            context="resume_load_weights_compat",
        )
        normalize_direct_pose_split_state_dict_schema(model, state_dict)
        _raise_if_direct_pose_semantic_tensor_would_be_skipped(
            state_dict=state_dict,
            model=model,
            context="resume_load_weights_compat",
        )

        current_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key in current_state and torch.is_tensor(value) and torch.is_tensor(current_state[key]) and tuple(current_state[key].shape) == tuple(value.shape):
                filtered_state[key] = value
            else:
                skipped_keys.append(key)

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        report = ResumeLoadReport(
            ckpt_path=ckpt_path,
            loaded_count=len(filtered_state),
            total_count=len(state_dict),
            missing_count=len(missing),
            unexpected_count=len(unexpected),
            skipped_shape_count=len(skipped_keys),
        )
        print(
            f'[Resume] loaded={report.loaded_count}/{report.total_count} '
            f'missing={report.missing_count} unexpected={report.unexpected_count} '
            f'skipped_shape={report.skipped_shape_count} ckpt={ckpt_path}'
        )
        return report
    except Exception as err:
        if isinstance(err, RemovedCheckpointCompatError):
            raise
        warning = f'failed to load checkpoint: {err}'
        print(f'[Resume][WARN] {warning}')
        return ResumeLoadReport(
            ckpt_path=ckpt_path,
            loaded_count=0,
            total_count=0,
            missing_count=0,
            unexpected_count=0,
            skipped_shape_count=0,
            warning=warning,
        )


def _cfg_int(config: Optional[dict[str, Any]], *keys: str, default: Optional[int] = None) -> Optional[int]:
    if not isinstance(config, dict):
        return default
    for key in keys:
        if key not in config or config.get(key) is None:
            continue
        try:
            return int(config.get(key))
        except Exception:
            continue
    return default


def infer_event_motion_core_dims_from_ckpt(
    *,
    state_dict: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> EventMotionCoreDims:
    """Infer minimal EventMotionModel core dimensions from checkpoint tensors.

    This is intended for ckpt-backed tools that do not have a dataset object.
    It prefers an optional saved config, then falls back to tensor shapes.
    """
    shared_encoder_weight = state_dict.get("shared_encoder.0.weight", None)
    if not (torch.is_tensor(shared_encoder_weight) and shared_encoder_weight.ndim == 2):
        raise KeyError("Checkpoint missing 'shared_encoder.0.weight' to infer model input dimensions.")
    shared_in_dim = int(shared_encoder_weight.shape[1])

    cond_dim = _cfg_int(config, "cond_dim", "Dc")
    if cond_dim is None:
        weight_ih = state_dict.get("contact_plan_cell.weight_ih", None)
        if torch.is_tensor(weight_ih) and weight_ih.ndim == 2:
            cond_dim = int(weight_ih.shape[1])
    if cond_dim is None:
        raise KeyError("Unable to infer cond_dim from checkpoint config or contact_plan_cell.weight_ih.")

    out_motion_dim = _cfg_int(config, "out_motion_dim", "Dy")
    if out_motion_dim is None:
        motion_head_weights: list[tuple[int, torch.Tensor]] = []
        for key, value in state_dict.items():
            key_text = str(key)
            if not key_text.startswith("motion_head.") or not key_text.endswith(".weight"):
                continue
            if not (torch.is_tensor(value) and value.ndim == 2):
                continue
            try:
                index = int(key_text.split(".")[1])
            except Exception:
                continue
            motion_head_weights.append((index, value))
        if motion_head_weights:
            motion_head_weights.sort(key=lambda item: item[0])
            out_motion_dim = int(motion_head_weights[-1][1].shape[0])
        else:
            out_head_weight = state_dict.get("out_head.weight", None)
            if torch.is_tensor(out_head_weight) and out_head_weight.ndim == 2:
                out_motion_dim = int(out_head_weight.shape[0])
    if out_motion_dim is None:
        raise KeyError("Unable to infer out_motion_dim from checkpoint config or out_head.weight.")

    contact_dim = _cfg_int(config, "contact_dim")
    if contact_dim is None:
        for key in (
            "contact_plan_head.4.weight",
            "contact_plan_time_head.weight",
            "contact_meas_head.4.weight",
            "contact_meas_head.weight",
        ):
            weight = state_dict.get(key, None)
            if torch.is_tensor(weight) and weight.ndim == 2:
                contact_dim = int(weight.shape[0])
                break
    if contact_dim is None:
        contact_dim = 0

    period_dim = _cfg_int(config, "period_dim")
    if period_dim is None:
        period_weight = state_dict.get("period_encoder.weight", None)
        period_dim = int(period_weight.shape[1]) if torch.is_tensor(period_weight) and period_weight.ndim == 2 else 0

    angvel_dim = max(0, int(_cfg_int(config, "angvel_dim", default=0) or 0))
    pose_hist_dim = max(0, int(_cfg_int(config, "pose_hist_dim", default=0) or 0))
    init_head_weight = state_dict.get("contact_plan_init_head.1.weight", None)
    if torch.is_tensor(init_head_weight) and init_head_weight.ndim == 2:
        obs_dim = int(init_head_weight.shape[1])
        current_obs_dim = int(contact_dim + angvel_dim + pose_hist_dim)
        if obs_dim > 0 and current_obs_dim != obs_dim:
            # No-dataset fallback: preserve the total obs-conditioned init input dim.
            # The debug tools using this path do not rely on the angvel/pose split.
            angvel_dim = max(0, int(obs_dim - contact_dim))
            pose_hist_dim = 0

    in_state_dim = _cfg_int(config, "in_state_dim", "Dx")
    if in_state_dim is None:
        # Without a dataset there is no reliable way to distinguish trunk contact-plan
        # injection from state features. Use a shape-compatible fallback so checkpoint
        # shared_encoder weights can load; contact-plan-only diagnostics do not use it.
        in_state_dim = max(1, int(shared_in_dim - int(cond_dim)))

    return EventMotionCoreDims(
        in_state_dim=int(in_state_dim),
        out_motion_dim=int(out_motion_dim),
        cond_dim=int(cond_dim),
        contact_dim=int(contact_dim),
        angvel_dim=int(angvel_dim),
        pose_hist_dim=int(pose_hist_dim),
        period_dim=int(period_dim),
    )


def resolve_contact_plan_build_cfg(
    *,
    state_dict: dict[str, Any],
    in_state_dim: int,
    cond_dim: int,
    contact_dim: int,
    overrides: Optional[ContactPlanBuildOverrides] = None,
) -> ContactPlanBuildConfig:
    contact_plan_enable = any(
        str(key).startswith("contact_plan_cell.") or str(key).startswith("contact_plan_head.")
        for key in state_dict.keys()
    )
    contact_plan_hidden = 64
    try:
        w_hh = state_dict.get("contact_plan_cell.weight_hh", None)
        if torch.is_tensor(w_hh) and w_hh.ndim == 2:
            contact_plan_hidden = int(w_hh.shape[1])
    except Exception:
        pass
    contact_plan_time_pe_dim = 0
    try:
        w_time = state_dict.get("contact_plan_time_head.weight", None)
        if torch.is_tensor(w_time) and w_time.ndim == 2:
            contact_plan_time_pe_dim = int(w_time.shape[1])
    except Exception:
        contact_plan_time_pe_dim = 0

    contact_plan_init_mode = "learnable"
    contact_plan_init_hidden = 128
    try:
        init_has_weights = any(str(key).startswith("contact_plan_init_head.") for key in state_dict.keys())
        if init_has_weights:
            contact_plan_init_mode = "learnable+obs"
            w_init = state_dict.get("contact_plan_init_head.1.weight", None)
            if torch.is_tensor(w_init) and w_init.ndim == 2:
                contact_plan_init_hidden = int(w_init.shape[0])
    except Exception:
        contact_plan_init_mode = "learnable"
    contact_plan_init_dropout = 0.0

    if overrides is not None:
        if overrides.init_mode is not None:
            contact_plan_init_mode = _normalize_contact_plan_init_mode(overrides.init_mode)
        if overrides.init_hidden is not None:
            try:
                contact_plan_init_hidden = int(overrides.init_hidden)
            except Exception:
                pass
        if overrides.init_dropout is not None:
            try:
                contact_plan_init_dropout = float(overrides.init_dropout)
            except Exception:
                contact_plan_init_dropout = 0.0

    contact_plan_inject = "none"
    try:
        w0 = state_dict.get("shared_encoder.0.weight", None)
        if torch.is_tensor(w0) and w0.ndim == 2:
            nin = int(w0.shape[1])
            base_in = int(in_state_dim + cond_dim)
            extra = int(max(0, nin - base_in))
            if extra > 0:
                if int(contact_dim) > 0 and extra == int(contact_dim):
                    contact_plan_inject = "contacts"
                else:
                    contact_plan_inject = "plan_z"
                    if extra != int(contact_plan_hidden):
                        contact_plan_hidden = int(extra)
    except Exception:
        contact_plan_inject = "none"

    return ContactPlanBuildConfig(
        enable=bool(contact_plan_enable),
        hidden=int(contact_plan_hidden),
        inject=str(contact_plan_inject),
        time_pe_dim=int(contact_plan_time_pe_dim),
        init_mode=str(contact_plan_init_mode),
        init_hidden=int(contact_plan_init_hidden),
        init_dropout=float(contact_plan_init_dropout),
    )


def _normalize_direct_pose_feat_source(val: Any) -> Optional[str]:
    if val is None:
        return None
    text = str(val).strip().lower()
    if text in ("", "auto"):
        return None
    text = _DIRECT_POSE_FEAT_SOURCE_ALIAS_MAP.get(text, text)
    if text in _DIRECT_POSE_FEAT_SOURCE_CANONICAL:
        return text
    return None


def _normalize_direct_pose_phase_z_mode(val: Any) -> str:
    text = str(val or "").strip().lower()
    if text in ("", "auto"):
        return "concat"
    if text in ("replace", "replace_contacts", "phase", "phase_only"):
        return "replace_contacts"
    if text in ("concat", "append", "add", "plus", "contacts+phase"):
        return "concat"
    return text


def _normalize_direct_pose_time_pe_dim(val: Any) -> Optional[int]:
    try:
        dim = int(val)
    except (TypeError, ValueError):
        return None
    if dim < 0:
        return None
    if dim % 2 == 1:
        dim += 1
    return int(dim)


def _apply_direct_pose_concat_to_replace_contacts_phase_tail_compat(
    *,
    state_dict: dict[str, Any],
    model: EventMotionModel,
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    contact_dim: int,
    direct_pose_cfg: DirectPoseBuildConfig,
) -> tuple[str, ...]:
    # Narrow semantic compat for the direct-pose first-layer input projection only:
    # - source checkpoint was trained with `direct_pose_phase_z_mode='concat'`
    #   so its input layout is `[prefix, plan+meas, phase_tail]`
    # - target model runs with `direct_pose_phase_z_mode='replace_contacts'`
    #   so its input layout is `[prefix, phase_tail]`
    # - we only adapt the exact contraction where the removed middle block has the
    #   same width as `phase_dim = 2 * contact_dim`, i.e.
    #     src_in = prefix + phase_dim + phase_dim
    #     tgt_in = prefix + phase_dim
    # - mapping contract is:
    #     target[:, :prefix] <- source[:, :prefix]
    #     target[:, prefix:] <- source[:, src_in - phase_dim:src_in]
    # Unsupported layouts continue to fail via the strict mismatch guard below.
    if bool(direct_pose_cfg.drop_ckpt_weights) or (not bool(direct_pose_cfg.use_phase_z)):
        return ()
    if str(direct_pose_cfg.meas_mode or "concat").strip().lower() != "concat":
        return ()
    if _normalize_direct_pose_phase_z_mode(direct_pose_cfg.phase_z_mode) != "replace_contacts":
        return ()
    if not isinstance(ckpt_posttrain_cfg, dict):
        return ()
    if not bool(ckpt_posttrain_cfg.get("direct_pose_use_phase_z", False)):
        return ()
    if _normalize_direct_pose_phase_z_mode(ckpt_posttrain_cfg.get("direct_pose_phase_z_mode", None)) != "concat":
        return ()

    target_feat_source = _normalize_direct_pose_feat_source(direct_pose_cfg.feat_source)
    source_feat_source = _normalize_direct_pose_feat_source(ckpt_posttrain_cfg.get("direct_pose_feat_source", None))
    if target_feat_source is None or source_feat_source is None or source_feat_source != target_feat_source:
        return ()

    target_time_pe_dim = _normalize_direct_pose_time_pe_dim(direct_pose_cfg.time_pe_dim)
    source_time_pe_dim = _normalize_direct_pose_time_pe_dim(ckpt_posttrain_cfg.get("direct_pose_time_pe_dim", None))
    if target_time_pe_dim is None or source_time_pe_dim is None or source_time_pe_dim != target_time_pe_dim:
        return ()

    phase_dim = int(2 * int(contact_dim))
    if phase_dim <= 0:
        return ()

    model_sd = model.state_dict()
    adapted = []
    for key in _DIRECT_POSE_PHASE_INPUT_KEYS:
        value_ckpt = state_dict.get(key, None)
        value_model = model_sd.get(key, None)
        if not (
            torch.is_tensor(value_ckpt)
            and torch.is_tensor(value_model)
            and value_ckpt.ndim == 2
            and value_model.ndim == 2
        ):
            continue
        if int(value_ckpt.shape[0]) != int(value_model.shape[0]):
            continue
        src_in = int(value_ckpt.shape[1])
        tgt_in = int(value_model.shape[1])
        if src_in == tgt_in:
            continue
        prefix_len = int(tgt_in - phase_dim)
        removed_middle = int(src_in - tgt_in)
        if prefix_len < 0 or removed_middle != phase_dim:
            continue
        if src_in != int(prefix_len + removed_middle + phase_dim):
            continue
        patched = value_model.detach().clone()
        patched[:, :prefix_len] = value_ckpt[:, :prefix_len].to(dtype=patched.dtype, device=patched.device)
        patched[:, prefix_len:] = value_ckpt[:, src_in - phase_dim : src_in].to(
            dtype=patched.dtype,
            device=patched.device,
        )
        state_dict[key] = patched.to(dtype=value_ckpt.dtype, device=value_ckpt.device)
        adapted.append(
            f"{key} for phase replace: in_dim {src_in} -> {tgt_in} "
            f"(dropped plan+meas, kept phase tail dim={phase_dim})"
        )
    return tuple(adapted)


def _infer_direct_pose_head_shape(
    *,
    out_motion_dim: int,
    state_dict: dict[str, Any],
    contact_dim: int,
    cond_dim: int,
    width: int,
    direct_pose_use_phase_z_infer: bool,
    direct_pose_phase_z_mode_infer: str,
) -> Optional[tuple[int, str, str, int]]:
    w_in = state_dict.get("direct_pose_head.0.weight", None)
    if not (torch.is_tensor(w_in) and w_in.ndim == 2):
        return None

    in_dim = int(w_in.shape[1])
    hidden = int(w_in.shape[0])
    readout_specs = (
        (
            state_dict.get("direct_pose_leg_terminal.6.weight", None),
            state_dict.get("direct_pose_out_nonleg.weight", None),
        ),
        (
            state_dict.get("direct_pose_leg_terminal.6.weight", None),
            state_dict.get("direct_pose_out_arm.weight", None),
            state_dict.get("direct_pose_out_else.weight", None),
        ),
    )
    w_out = state_dict.get("direct_pose_head.6.weight", None)
    out_dim = int(w_out.shape[0]) if torch.is_tensor(w_out) and w_out.ndim == 2 else None
    for readouts in readout_specs if out_dim is None else ():
        if all(torch.is_tensor(weight) and weight.ndim == 2 for weight in readouts):
            out_dim = int(sum(int(weight.shape[0]) for weight in readouts))
            if int(readouts[0].shape[1]) > 0:
                hidden = int(readouts[0].shape[1])
            break
    if out_dim is None:
        raise SystemExit("[FATAL] direct_pose_head weights found but output readout weights are missing.")

    expected_out = int(out_motion_dim)
    expected_out_modes = int(out_motion_dim) * 2
    contact_channels = int(contact_dim)
    phase_mode_infer = str(direct_pose_phase_z_mode_infer or "concat").strip().lower()
    phase_dim = int(2 * contact_channels) if bool(direct_pose_use_phase_z_infer) else 0
    if out_dim == expected_out:
        meas_mode = "concat"
        contacts_in_dim = 0 if phase_mode_infer == "replace_contacts" else int(2 * contact_channels)
    elif out_dim == expected_out_modes:
        meas_mode = "mode_select"
        if phase_mode_infer == "replace_contacts":
            raise SystemExit(
                "[FATAL] direct_pose_phase_z_mode='replace_contacts' is not supported for direct_pose_meas_mode='mode_select'."
            )
        contacts_in_dim = int(contact_channels)
    else:
        raise SystemExit(
            f"[FATAL] Unrecognized direct_pose_head out_dim={out_dim} "
            f"(expected {expected_out} or {expected_out_modes})."
        )

    for base_dim, feat_source in (
        (int(cond_dim), "cond"),
        (int(width), "hidden"),
        (int(cond_dim + width), "cond+hidden"),
    ):
        time_pe_dim = int(in_dim - base_dim - contacts_in_dim - phase_dim)
        if time_pe_dim >= 0 and time_pe_dim % 2 == 0:
            return hidden, meas_mode, feat_source, int(time_pe_dim)
    raise SystemExit(
        f"[FATAL] Unrecognized direct_pose_head shape: in_dim={in_dim} out_dim={out_dim} "
        f"(cond_dim={cond_dim}, hidden_dim={width}, contact_dim={contact_dim})."
    )


def _infer_direct_pose_ckpt_layout(
    *,
    out_motion_dim: int,
    state_dict: dict[str, Any],
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    contact_dim: int,
    cond_dim: int,
    width: int,
    direct_has_weights: bool,
    direct_pose_reinit: bool,
) -> DirectPoseCkptInference:
    use_phase_z = False
    phase_z_mode = "concat"
    if isinstance(ckpt_posttrain_cfg, dict):
        use_phase_z = bool(ckpt_posttrain_cfg.get("direct_pose_use_phase_z", False))
        phase_mode_value = ckpt_posttrain_cfg.get("direct_pose_phase_z_mode", None)
        if phase_mode_value is not None:
            phase_z_mode = str(phase_mode_value).strip().lower() or "concat"

    enable, hidden, meas_mode, feat_source, time_pe_dim = False, 256, "concat", "cond", 0
    if direct_has_weights and contact_dim > 0 and (not direct_pose_reinit):
        try:
            head_shape = _infer_direct_pose_head_shape(
                out_motion_dim=out_motion_dim,
                state_dict=state_dict,
                contact_dim=contact_dim,
                cond_dim=cond_dim,
                width=width,
                direct_pose_use_phase_z_infer=use_phase_z,
                direct_pose_phase_z_mode_infer=phase_z_mode,
            )
            if head_shape is not None:
                hidden, meas_mode, feat_source, time_pe_dim = head_shape
                enable = True
        except Exception:
            enable = False

    split_enable, arm_split_enable = False, False
    try:
        has_leg_terminal = any(str(key).startswith("direct_pose_leg_terminal.") for key in state_dict.keys())
        has_nonleg_out = any(str(key).startswith("direct_pose_out_nonleg.") for key in state_dict.keys())
        has_arm_out = any(str(key).startswith("direct_pose_out_arm.") for key in state_dict.keys())
        has_else_out = any(str(key).startswith("direct_pose_out_else.") for key in state_dict.keys())
        split_enable = bool(has_leg_terminal and (has_nonleg_out or (has_arm_out and has_else_out)))
        arm_split_enable = bool(has_leg_terminal and has_arm_out and has_else_out)
    except Exception:
        split_enable, arm_split_enable = False, False
    try:
        if isinstance(ckpt_posttrain_cfg, dict):
            arm_split_enable = bool(
                ckpt_posttrain_cfg.get("direct_pose_arm_split_enable", arm_split_enable)
            )
            split_enable = bool(split_enable or arm_split_enable)
    except Exception:
        pass

    nonleg_proj_dim = 0
    try:
        w_non = state_dict.get("direct_pose_out_nonleg.weight", None)
        for proj_key in (
            "direct_pose_nonleg_proj.0.weight",
            "direct_pose_arm_proj.0.weight",
            "direct_pose_else_proj.0.weight",
        ):
            w_proj = state_dict.get(proj_key, None)
            if torch.is_tensor(w_proj) and w_proj.ndim == 2 and int(w_proj.shape[0]) > 0:
                nonleg_proj_dim = int(w_proj.shape[0])
                break
        if (
            torch.is_tensor(w_non)
            and w_non.ndim == 2
            and int(nonleg_proj_dim) <= 0
            and int(hidden) > 0
            and int(w_non.shape[1]) > 0
            and int(w_non.shape[1]) != int(hidden)
        ):
            nonleg_proj_dim = int(w_non.shape[1])
    except Exception:
        pass

    return DirectPoseCkptInference(
        has_weights=bool(direct_has_weights),
        enable=bool(enable),
        hidden=int(hidden),
        meas_mode=str(meas_mode),
        feat_source=str(feat_source),
        time_pe_dim=int(time_pe_dim),
        phase_z_mode=str(phase_z_mode),
        split_enable=bool(split_enable),
        arm_split_enable=bool(arm_split_enable),
        nonleg_proj_dim=int(nonleg_proj_dim),
    )


def _reject_removed_direct_pose_ckpt_semantic_policy(
    *,
    direct_pose_reinit: bool,
    ckpt_layout: DirectPoseCkptInference,
    direct_pose_hidden: int,
    direct_pose_meas_mode: str,
    direct_pose_feat_source: str,
    direct_pose_time_pe_dim: int,
    direct_pose_split_enable: bool,
    direct_pose_arm_split_enable: bool,
    direct_pose_nonleg_proj_dim: int,
) -> None:
    if bool(direct_pose_reinit) and bool(ckpt_layout.has_weights):
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_reinit",
            reason=(
                "`direct_pose_reinit` previously dropped existing `direct_pose_*` checkpoint tensors; "
                "that semantic tensor drop is no longer allowed"
            ),
        )
    shape_override = bool(
        ckpt_layout.has_weights
        and ckpt_layout.enable
        and (
            direct_pose_hidden != int(ckpt_layout.hidden)
            or direct_pose_meas_mode != str(ckpt_layout.meas_mode)
            or str(direct_pose_feat_source).replace("_pre", "")
            != str(ckpt_layout.feat_source).replace("_pre", "")
            or int(direct_pose_time_pe_dim) != int(ckpt_layout.time_pe_dim)
        )
    )
    nonleg_proj_mismatch = bool(int(direct_pose_nonleg_proj_dim) != int(ckpt_layout.nonleg_proj_dim))
    split_mismatch = bool(direct_pose_split_enable) != bool(ckpt_layout.split_enable)
    arm_split_mismatch = bool(direct_pose_arm_split_enable) != bool(ckpt_layout.arm_split_enable)
    if shape_override:
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_hidden/direct_pose_meas_mode/direct_pose_feat_source/direct_pose_time_pe_dim",
            reason=(
                "direct-pose shape overrides previously dropped or reinitialized old "
                "`direct_pose_head.*` tensors; non-bit-exact tensor replacement is removed"
            ),
        )
    if nonleg_proj_mismatch and bool(ckpt_layout.has_weights):
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_nonleg_proj_dim",
            reason=(
                "`direct_pose_nonleg_proj_dim` mismatch previously adapted/reinitialized non-leg "
                "readout tensors; non-bit-exact tensor adaptation is removed"
            ),
        )
    if split_mismatch and bool(ckpt_layout.has_weights):
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_split_enable",
            reason=(
                "old monolithic `direct_pose_head.6.*` to split-readout conversion/drop was removed; "
                "split mode must match the checkpoint schema exactly"
            ),
        )
    if arm_split_mismatch and bool(ckpt_layout.has_weights):
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_arm_split_enable",
            reason=(
                "direct-pose arm/else split restructuring previously remapped old readout tensors; "
                "non-bit-exact head restructuring is removed"
            ),
        )


def resolve_direct_pose_build_cfg(
    *,
    out_motion_dim: int,
    state_dict: dict[str, Any],
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    contact_dim: int,
    cond_dim: int,
    width: int,
    overrides: DirectPoseBuildOverrides,
) -> DirectPoseBuildConfig:
    _raise_if_retired_direct_pose_stepc_leg_layout(
        state_dict,
        context="resolve_direct_pose_build_cfg",
    )
    train_direct_pose = bool(overrides.train_direct_pose)
    direct_has_weights = any(
        key.startswith(
            (
                "direct_pose_head.",
                "direct_pose_leg_terminal.",
                "direct_pose_out_nonleg.",
                "direct_pose_out_arm.",
                "direct_pose_out_else.",
            )
        )
        for key in state_dict.keys()
    )
    direct_pose_reinit = bool(overrides.direct_pose_reinit)
    if direct_pose_reinit and (not train_direct_pose):
        _raise_removed_checkpoint_field(
            context="resolve_direct_pose_build_cfg",
            field_name="direct_pose_reinit",
            reason=(
                "`direct_pose_reinit=true` with `train_direct_pose=false` was formerly ignored; "
                "warning-and-continue behavior is removed"
            ),
            migration="set `direct_pose_reinit=false` or train a fresh current-schema direct-pose head; no replacement",
        )

    ckpt_layout = _infer_direct_pose_ckpt_layout(
        out_motion_dim=out_motion_dim,
        state_dict=state_dict,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        contact_dim=contact_dim,
        cond_dim=cond_dim,
        width=width,
        direct_has_weights=direct_has_weights,
        direct_pose_reinit=direct_pose_reinit,
    )

    direct_pose_enable = bool(
        ckpt_layout.enable or ckpt_layout.has_weights or train_direct_pose or direct_pose_reinit
    )
    direct_pose_hidden = int(overrides.hidden_override or ckpt_layout.hidden)
    direct_pose_meas_mode = str(overrides.meas_mode_override or ckpt_layout.meas_mode)
    direct_pose_feat_source = str(overrides.feat_source or "auto").lower().strip()
    direct_pose_time_pe_dim = int(overrides.time_pe_dim)
    direct_pose_time_pe_base = float(overrides.time_pe_base or 10000.0)
    if overrides.use_phase_z is None:
        direct_pose_use_phase_z = bool(
            isinstance(ckpt_posttrain_cfg, dict) and ckpt_posttrain_cfg.get("direct_pose_use_phase_z", False)
        )
    else:
        direct_pose_use_phase_z = bool(overrides.use_phase_z)
    direct_pose_phase_z_mode = str(overrides.phase_z_mode or "auto").strip().lower()
    direct_pose_split_enable = (
        bool(overrides.split_enable)
        if overrides.split_enable is not None
        else bool(ckpt_layout.split_enable)
    )
    direct_pose_arm_split_enable = (
        bool(overrides.arm_split_enable)
        if overrides.arm_split_enable is not None
        else bool(ckpt_layout.arm_split_enable)
    )
    direct_pose_split_enable = bool(
        direct_pose_split_enable
        or direct_pose_arm_split_enable
    )
    direct_pose_arm_bones = overrides.arm_bones
    direct_pose_nonleg_proj_dim = (
        max(0, int(overrides.nonleg_proj_dim))
        if overrides.nonleg_proj_dim is not None
        else int(ckpt_layout.nonleg_proj_dim)
    )
    if direct_pose_phase_z_mode in ("", "auto"):
        direct_pose_phase_z_mode = str(ckpt_layout.phase_z_mode or "concat").strip().lower() or "concat"
    if direct_pose_phase_z_mode in ("replace", "replace_contacts", "phase", "phase_only"):
        direct_pose_phase_z_mode = "replace_contacts"
    elif direct_pose_phase_z_mode in ("concat", "append", "add", "plus", "contacts+phase"):
        direct_pose_phase_z_mode = "concat"
    else:
        direct_pose_phase_z_mode = str(ckpt_layout.phase_z_mode or "concat").strip().lower() or "concat"

    if direct_pose_meas_mode not in ("concat", "mode_select"):
        direct_pose_meas_mode = ckpt_layout.meas_mode

    if direct_pose_feat_source == "auto":
        ckpt_feat_source_hint = None
        if isinstance(ckpt_posttrain_cfg, dict):
            ckpt_feat_source_hint = _normalize_direct_pose_feat_source(
                ckpt_posttrain_cfg.get("direct_pose_feat_source", None)
            )
        direct_pose_feat_source = ckpt_feat_source_hint or (
            ckpt_layout.feat_source if ckpt_layout.enable else "cond"
        )
    direct_pose_feat_source = _normalize_direct_pose_feat_source(direct_pose_feat_source) or "cond"
    if int(direct_pose_time_pe_dim) < 0:
        direct_pose_time_pe_dim = int(ckpt_layout.time_pe_dim)
    if int(direct_pose_time_pe_dim) % 2 == 1:
        print(
            f"[posttrain][WARN] direct_pose_time_pe_dim={direct_pose_time_pe_dim} is odd; rounding up to even."
        )
        direct_pose_time_pe_dim = int(direct_pose_time_pe_dim) + 1

    _reject_removed_direct_pose_ckpt_semantic_policy(
        direct_pose_reinit=direct_pose_reinit,
        ckpt_layout=ckpt_layout,
        direct_pose_hidden=direct_pose_hidden,
        direct_pose_meas_mode=direct_pose_meas_mode,
        direct_pose_feat_source=direct_pose_feat_source,
        direct_pose_time_pe_dim=direct_pose_time_pe_dim,
        direct_pose_split_enable=direct_pose_split_enable,
        direct_pose_arm_split_enable=direct_pose_arm_split_enable,
        direct_pose_nonleg_proj_dim=direct_pose_nonleg_proj_dim,
    )

    return DirectPoseBuildConfig(
        enable=bool(direct_pose_enable),
        hidden=int(direct_pose_hidden),
        meas_mode=str(direct_pose_meas_mode),
        feat_source=str(direct_pose_feat_source),
        time_pe_dim=int(direct_pose_time_pe_dim),
        time_pe_base=float(direct_pose_time_pe_base),
        use_phase_z=bool(direct_pose_use_phase_z),
        phase_z_mode=str(direct_pose_phase_z_mode),
        split_enable=bool(direct_pose_split_enable),
        arm_split_enable=bool(direct_pose_arm_split_enable),
        arm_bones=direct_pose_arm_bones,
        nonleg_proj_dim=int(direct_pose_nonleg_proj_dim),
        drop_ckpt_weights=False,
    )


def _raise_if_direct_pose_ckpt_weight_drop_requested(
    *,
    state_dict: dict[str, Any],
    context: str,
    drop_direct_pose_weights: bool,
) -> None:
    if not bool(drop_direct_pose_weights):
        return
    hits = _find_state_keys(
        state_dict,
        prefixes=_DIRECT_POSE_WEIGHT_PREFIXES,
        exact=_DIRECT_POSE_WEIGHT_EXACT_KEYS,
    )
    if hits:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_*",
            hits=hits,
            reason=(
                "direct-pose reinit/shape override previously dropped checkpoint tensors; "
                "semantic tensor drop is no longer allowed"
            ),
        )


def _raise_if_direct_pose_phase_z_input_mismatch(
    *,
    state_dict: dict[str, Any],
    model: EventMotionModel,
    contact_dim: int,
    direct_pose_use_phase_z: bool,
    direct_pose_phase_z_mode: str,
    drop_direct_pose_weights: bool,
    context: str,
) -> None:
    if bool(drop_direct_pose_weights) or not bool(direct_pose_use_phase_z):
        return
    if not any(str(key).startswith("direct_pose_") for key in state_dict.keys()):
        return
    model_sd = model.state_dict()
    phase_dim = int(2 * int(contact_dim))
    phase_mode = str(direct_pose_phase_z_mode or "concat").strip().lower()
    mismatches = []
    for key in _DIRECT_POSE_PHASE_INPUT_KEYS:
        value_ckpt = state_dict.get(key, None)
        value_model = model_sd.get(key, None)
        if not (
            torch.is_tensor(value_ckpt)
            and torch.is_tensor(value_model)
            and value_ckpt.ndim == 2
            and value_model.ndim == 2
        ):
            continue
        if tuple(value_ckpt.shape) != tuple(value_model.shape):
            mismatches.append(
                f"{key}: ckpt_shape={tuple(int(v) for v in value_ckpt.shape)} "
                f"model_shape={tuple(int(v) for v in value_model.shape)}"
            )
    if mismatches:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_head.0.weight/direct_pose_leg_head*.0.weight",
            hits=tuple(mismatches),
            reason=(
                "phase_z input-dimension padding/cropping was removed because zero-fill, "
                "contact-tail slicing, or head reinit is not bit-exact semantic equivalence; "
                f"direct_pose_phase_z_mode={phase_mode!r}, phase_dim={phase_dim}"
            ),
        )


def _raise_if_retired_direct_pose_highorder_ckpt_tensors(
    state_dict: dict[str, Any],
    *,
    context: str,
) -> None:
    retired_highorder = _find_state_keys(
        state_dict,
        prefixes=_RETIRED_DIRECT_POSE_HIGHORDER_PREFIXES,
        exact=_RETIRED_DIRECT_POSE_HIGHORDER_EXACT_KEYS,
    )
    if retired_highorder:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern=(
                "direct_pose_leg_head_shared.*|direct_pose_leg_gate_head_shared.*|"
                "direct_pose_leg_side_sign_gate_head.*|direct_pose_leg_side_embed.*"
            ),
            hits=retired_highorder,
            reason=(
                "retired high-order direct-pose tensors were previously dropped or used by a "
                "side-routing/sign-gate/rank1 shell; that semantic compat path is removed"
            ),
        )


def _norm_bones(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [item for item in items if item]


def _raise_if_incompatible_direct_pose_leg_ckpt_tensors(
    *,
    state_dict: dict[str, Any],
    model: EventMotionModel,
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    load_options: DirectPoseLoadCompatOptions,
    context: str,
) -> None:
    ckpt_leg_bones = []
    if isinstance(ckpt_posttrain_cfg, dict):
        ckpt_leg_bones = _norm_bones(ckpt_posttrain_cfg.get("direct_pose_leg_bones", None))
    tgt_leg_bones = _norm_bones(load_options.leg_bones)

    if bool(load_options.leg_enable) and ckpt_leg_bones and tgt_leg_bones and (tgt_leg_bones != ckpt_leg_bones):
        hits = _find_state_keys(
            state_dict,
            prefixes=_DIRECT_POSE_LEG_PREFIXES,
            exact=_DIRECT_POSE_LEG_EXACT_KEYS,
        )
        if hits:
            _raise_removed_checkpoint_compat(
                context=context,
                key_pattern="direct_pose_leg_bones + direct_pose_leg_*",
                hits=hits,
                reason=(
                    "`direct_pose_leg_bones` override previously dropped old leg-head/index tensors; "
                    f"ckpt_bones={ckpt_leg_bones!r}, requested_bones={tgt_leg_bones!r}"
                ),
            )

    model_sd = model.state_dict()
    shape_mismatches = []
    for key in state_dict.keys():
        key_text = str(key)
        if not any(key_text.startswith(prefix) for prefix in ("direct_pose_leg_head.", "direct_pose_leg_gate_head.")):
            continue
        value_ckpt = state_dict.get(key_text, None)
        value_model = model_sd.get(key_text, None)
        if (
            torch.is_tensor(value_ckpt)
            and torch.is_tensor(value_model)
            and tuple(value_ckpt.shape) != tuple(value_model.shape)
        ):
            shape_mismatches.append(
                f"{key_text}: ckpt_shape={tuple(int(v) for v in value_ckpt.shape)} "
                f"model_shape={tuple(int(v) for v in value_model.shape)}"
            )
    if shape_mismatches:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_leg_head.*|direct_pose_leg_gate_head.*",
            hits=tuple(shape_mismatches),
            reason=(
                "leg-head shape mismatch previously triggered pad/crop/drop/reinit behavior; "
                "non-bit-exact direct-pose leg tensor adaptation is removed"
            ),
        )


def apply_direct_pose_ckpt_compat(
    *,
    state_dict: dict[str, Any],
    model: EventMotionModel,
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    contact_dim: int,
    direct_pose_cfg: DirectPoseBuildConfig,
    load_options: DirectPoseLoadCompatOptions,
) -> None:
    _raise_if_retired_direct_pose_stepc_leg_layout(
        state_dict,
        context="apply_direct_pose_ckpt_compat",
    )
    normalize_direct_pose_split_state_dict_schema(model, state_dict)
    _raise_if_direct_pose_ckpt_weight_drop_requested(
        state_dict=state_dict,
        context="apply_direct_pose_ckpt_compat",
        drop_direct_pose_weights=direct_pose_cfg.drop_ckpt_weights,
    )
    for row in _apply_direct_pose_concat_to_replace_contacts_phase_tail_compat(
        state_dict=state_dict,
        model=model,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        contact_dim=contact_dim,
        direct_pose_cfg=direct_pose_cfg,
    ):
        print(f"[posttrain][INFO] adapted {row}")
    _raise_if_direct_pose_phase_z_input_mismatch(
        state_dict=state_dict,
        model=model,
        contact_dim=contact_dim,
        direct_pose_use_phase_z=direct_pose_cfg.use_phase_z,
        direct_pose_phase_z_mode=direct_pose_cfg.phase_z_mode,
        drop_direct_pose_weights=direct_pose_cfg.drop_ckpt_weights,
        context="apply_direct_pose_ckpt_compat",
    )
    _raise_if_retired_direct_pose_highorder_ckpt_tensors(
        state_dict,
        context="apply_direct_pose_ckpt_compat",
    )
    _raise_if_incompatible_direct_pose_leg_ckpt_tensors(
        state_dict=state_dict,
        model=model,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        load_options=load_options,
        context="apply_direct_pose_ckpt_compat",
    )
    _raise_if_direct_pose_semantic_tensor_would_be_skipped(
        state_dict=state_dict,
        model=model,
        context="apply_direct_pose_ckpt_compat",
    )


def _normalize_split_index_buffer_schema_only(state_dict: dict[str, Any], key: str, target_idx: torch.Tensor) -> bool:
    value = state_dict.get(key, None)
    if not torch.is_tensor(value):
        return False
    if tuple(value.shape) != tuple(target_idx.shape):
        _raise_removed_checkpoint_compat(
            context="normalize_direct_pose_split_state_dict_schema",
            key_pattern=key,
            hits=(f"{key}: ckpt_shape={tuple(int(v) for v in value.shape)} model_shape={tuple(int(v) for v in target_idx.shape)}",),
            reason=(
                "split index-buffer shape mismatch would change direct-pose output routing; "
                "shape drop/reconstruction is not schema-only"
            ),
        )
    if value.dtype == target_idx.dtype:
        return False
    if value.dtype not in _INDEX_BUFFER_DTYPES or target_idx.dtype not in _INDEX_BUFFER_DTYPES:
        _raise_removed_checkpoint_compat(
            context="normalize_direct_pose_split_state_dict_schema",
            key_pattern=key,
            hits=(f"{key}: ckpt_dtype={value.dtype} model_dtype={target_idx.dtype}",),
            reason=(
                "split index-buffer dtype normalization is only schema-only for integer index tensors; "
                "non-integer index adaptation has no replacement"
            ),
        )
    try:
        # Schema-only whitelist: split index buffers encode the same integer output
        # positions. Casting an existing same-shape integer buffer to the model's
        # integer dtype changes serialization/schema only, not routing semantics.
        state_dict[key] = value.to(dtype=target_idx.dtype)
    except Exception:
        _raise_removed_checkpoint_compat(
            context="normalize_direct_pose_split_state_dict_schema",
            key_pattern=key,
            hits=(f"{key}: ckpt_dtype={value.dtype} model_dtype={target_idx.dtype}",),
            reason=(
                "split index-buffer dtype normalization failed; silent index-buffer drop is removed"
            ),
        )
    return True


def _model_direct_pose_split_state(model: "EventMotionModel") -> Optional[dict[str, Any]]:
    try:
        split_state_fn = model._direct_pose_split_state
    except AttributeError:
        return None
    split_state = split_state_fn()
    return split_state if isinstance(split_state, dict) else None


def _raise_if_old_monolithic_direct_pose_readout_for_split_model(
    *,
    state_dict: dict[str, Any],
    model: "EventMotionModel",
    context: str,
    split_state: Optional[dict[str, Any]] = None,
) -> None:
    if split_state is None:
        split_state = _model_direct_pose_split_state(model)
    if split_state is None:
        return
    hits = tuple(
        key
        for key in ("direct_pose_head.6.weight", "direct_pose_head.6.bias")
        if key in state_dict
    )
    if hits:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_head.6.weight/direct_pose_head.6.bias",
            hits=hits,
            reason=(
                "old monolithic direct-pose readout to split readout (`direct_pose_leg_terminal.*`, "
                "`direct_pose_out_nonleg.*`, `direct_pose_out_arm.*`, `direct_pose_out_else.*`) "
                "was a semantic tensor re-index/restructure and has been removed"
            ),
        )


def _raise_if_direct_pose_semantic_tensor_would_be_skipped(
    state_dict: dict[str, Any],
    *,
    model: "EventMotionModel",
    context: str,
) -> None:
    model_sd = model.state_dict()
    missing = []
    mismatched = []
    for key, value_ckpt in state_dict.items():
        key_text = str(key)
        if key_text in _DIRECT_POSE_SPLIT_INDEX_KEYS:
            continue
        if key_text in _RETIRED_DIRECT_POSE_HIGHORDER_EXACT_KEYS:
            continue
        if not any(key_text.startswith(prefix) for prefix in _DIRECT_POSE_WEIGHT_PREFIXES):
            continue
        if not torch.is_tensor(value_ckpt):
            continue
        value_model = model_sd.get(key_text, None)
        if value_model is None:
            missing.append(key_text)
            continue
        if torch.is_tensor(value_model) and tuple(value_ckpt.shape) != tuple(value_model.shape):
            mismatched.append(
                f"{key_text}: ckpt_shape={tuple(int(v) for v in value_ckpt.shape)} "
                f"model_shape={tuple(int(v) for v in value_model.shape)}"
            )
    if missing:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_* unexpected/missing-in-model",
            hits=tuple(missing),
            reason=(
                "direct-pose tensor partial-load would silently drop checkpoint tensors that are not present "
                "in the current model; semantic direct-pose drops are removed"
            ),
        )
    if mismatched:
        _raise_removed_checkpoint_compat(
            context=context,
            key_pattern="direct_pose_* shape mismatch",
            hits=tuple(mismatched),
            reason=(
                "direct-pose tensor partial-load would skip shape-mismatched tensors; "
                "semantic shape adaptation/drop is removed"
            ),
        )


def normalize_direct_pose_split_state_dict_schema(model: "EventMotionModel", state_dict: dict[str, Any]) -> bool:
    _raise_if_retired_direct_pose_stepc_leg_layout(
        state_dict,
        context="normalize_direct_pose_split_state_dict_schema",
    )
    _raise_if_retired_direct_pose_highorder_ckpt_tensors(
        state_dict,
        context="normalize_direct_pose_split_state_dict_schema",
    )
    split_state = _model_direct_pose_split_state(model)
    if (not isinstance(state_dict, dict)) or split_state is None:
        return False
    _raise_if_old_monolithic_direct_pose_readout_for_split_model(
        state_dict=state_dict,
        model=model,
        context="normalize_direct_pose_split_state_dict_schema",
        split_state=split_state,
    )
    arm_split = bool(split_state["arm_split"])
    idx_leg = split_state["idx_leg"]
    idx_nonleg = split_state["idx_nonleg"]
    idx_arm = split_state["idx_arm"]
    idx_else = split_state["idx_else"]
    converted = False
    idx_pairs = [("direct_pose_leg_out_idx", idx_leg), ("direct_pose_nonleg_out_idx", idx_nonleg)]
    if arm_split:
        idx_pairs.append(("direct_pose_arm_out_idx", idx_arm))
        idx_pairs.append(("direct_pose_else_out_idx", idx_else))
    for key, idx_tgt in idx_pairs:
        converted = _normalize_split_index_buffer_schema_only(state_dict, key, idx_tgt) or converted
    return converted


def infer_event_clock_build_cfg(
    *,
    state_dict: dict[str, Any],
    contact_dim: int,
    period_dim: int,
    overrides: EventClockBuildOverrides,
) -> EventClockBuildConfig:
    event_clock_has_weights = any(
        str(key).startswith("event_clock_gate.") or str(key).startswith("event_clock_corrector.")
        for key in state_dict.keys()
    )
    event_clock_mode = str(overrides.mode or "auto").strip().lower()
    use_event_clock = bool(event_clock_has_weights)
    if event_clock_mode == "on":
        if not event_clock_has_weights:
            print(
                "[posttrain][WARN] --event_clock=on but ckpt has no event_clock_* weights; initializing Event-Clock randomly."
            )
        use_event_clock = True
    elif event_clock_mode == "off":
        if event_clock_has_weights:
            print(
                "[posttrain][WARN] --event_clock=off will drop event_clock_* weights when saving the posttrain checkpoint."
            )
        use_event_clock = False

    event_clock_hidden_dim = 64
    event_clock_gate_hidden_dim = 32
    try:
        w_ec = state_dict.get("event_clock_corrector.correction_head.0.weight", None)
        if torch.is_tensor(w_ec) and w_ec.ndim == 2:
            event_clock_hidden_dim = int(w_ec.shape[0])
    except Exception:
        pass
    try:
        w_gate = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
        if torch.is_tensor(w_gate) and w_gate.ndim == 2:
            event_clock_gate_hidden_dim = int(w_gate.shape[0])
    except Exception:
        pass
    if overrides.hidden_dim is not None:
        try:
            event_clock_hidden_dim = int(overrides.hidden_dim)
        except Exception:
            pass
    if overrides.gate_hidden_dim is not None:
        try:
            event_clock_gate_hidden_dim = int(overrides.gate_hidden_dim)
        except Exception:
            pass
    event_clock_max_delta = float(overrides.max_delta or 0.5)

    event_clock_period_feat_dim = None
    try:
        w0 = state_dict.get("event_clock_gate.confidence_head.0.weight", None)
        if torch.is_tensor(w0) and w0.ndim == 2:
            base = int(contact_dim) * 2 + 1
            event_clock_period_feat_dim = max(0, int(w0.shape[1]) - base)
    except Exception:
        event_clock_period_feat_dim = None
    period_dim_init = int(period_dim)
    try:
        if (
            bool(event_clock_has_weights)
            and event_clock_period_feat_dim is not None
            and int(event_clock_period_feat_dim) != int(period_dim)
        ):
            period_dim_init = int(event_clock_period_feat_dim)
    except Exception:
        period_dim_init = int(period_dim)
    if period_dim_init != int(period_dim) and bool(event_clock_has_weights):
        if not bool(overrides.has_encoder_bundle):
            print(
                f"[posttrain][WARN] ckpt period_dim={int(period_dim)} but Event-Clock was initialized with period_feat_dim={int(period_dim_init)}; "
                "no encoder_bundle provided so period_encoder weights may be dropped. "
                "Pass --encoder_bundle to fully reconstruct the model."
            )
        else:
            print(
                f"[posttrain][INFO] ckpt period_dim={int(period_dim)} but Event-Clock period_feat_dim={int(period_dim_init)}; "
                "initializing model with Event-Clock-compatible period_dim then attaching encoder bundle before loading weights."
            )
    return EventClockBuildConfig(
        use_event_clock=bool(use_event_clock),
        hidden_dim=int(event_clock_hidden_dim),
        gate_hidden_dim=int(event_clock_gate_hidden_dim),
        max_delta=float(event_clock_max_delta),
        period_dim_init=int(period_dim_init),
    )


def infer_lambda_fusion_build_cfg(
    *,
    state_dict: dict[str, Any],
    width: int,
    contact_dim: int,
    contact_plan_enable: bool,
    overrides: LambdaFusionBuildOverrides,
) -> LambdaFusionBuildConfig:
    lambda_has_weights = any(key.startswith("lambda_fusion_head.") for key in state_dict.keys())
    lambda_fusion_enable = bool(overrides.train_lambda_head or lambda_has_weights)
    lambda_fusion_mode = str(overrides.mode or "per_joint")
    lambda_fusion_hidden = int(overrides.hidden or 128)
    lambda_fusion_dropout = float(overrides.dropout or 0.0)
    lambda_fusion_logit_init = float(overrides.logit_init or -2.0)
    lambda_fusion_use_rollout_step_cfg = bool(overrides.use_rollout_step)
    lambda_fusion_use_rollout_step = bool(lambda_fusion_use_rollout_step_cfg)
    if lambda_has_weights:
        w_in = state_dict.get("lambda_fusion_head.1.weight", None)
        w_out = state_dict.get("lambda_fusion_head.4.weight", None)
        try:
            if torch.is_tensor(w_in) and w_in.ndim == 2:
                lambda_fusion_hidden = int(w_in.shape[0])
                base_in = int(width + (contact_dim if contact_plan_enable else 0))
                in_features = int(w_in.shape[1])
                inferred = None
                if in_features == base_in + 1:
                    inferred = True
                elif in_features == base_in:
                    inferred = False
                if inferred is not None and inferred != lambda_fusion_use_rollout_step_cfg:
                    print(
                        f"[posttrain][WARN] lambda_fusion_use_rollout_step={lambda_fusion_use_rollout_step_cfg} "
                        f"but ckpt expects {in_features} in_features (base={base_in}); overriding to {inferred}."
                    )
                if inferred is not None:
                    lambda_fusion_use_rollout_step = bool(inferred)
            if torch.is_tensor(w_out) and w_out.ndim == 2:
                out_dim = int(w_out.shape[0])
                lambda_fusion_mode = "global" if out_dim == 1 else "per_joint"
        except Exception:
            pass
    return LambdaFusionBuildConfig(
        enable=bool(lambda_fusion_enable),
        mode=str(lambda_fusion_mode),
        hidden=int(lambda_fusion_hidden),
        dropout=float(lambda_fusion_dropout),
        logit_init=float(lambda_fusion_logit_init),
        use_rollout_step=bool(lambda_fusion_use_rollout_step),
    )


def resolve_direct_pose_leg_build_cfg(
    *,
    state_dict: dict[str, Any],
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
) -> DirectPoseLegBuildConfig:
    direct_pose_leg_enable = False
    direct_pose_leg_bones = None
    direct_pose_leg_mode = "rot6d_add"
    direct_pose_leg_stopgrad_main = False
    direct_pose_leg_detach_feat = False
    direct_pose_leg_max_deg = 0.0
    direct_pose_leg_side_routing = False
    direct_pose_leg_contact_order = "lr"
    direct_pose_leg_side_embed_dim = 0
    direct_pose_leg_side_plan_other = False
    direct_pose_leg_side_phase_other = False
    direct_pose_leg_side_phase_rel = False
    direct_pose_leg_side_cue = "none"
    direct_pose_leg_side_cue_tau = 30.0
    direct_pose_leg_side_sign_gate = False
    direct_pose_leg_side_rank1 = False
    direct_pose_leg_gate_mode = "none"
    direct_pose_leg_gate_power = 1.0
    direct_pose_leg_scale_log_clip = 4.0
    direct_pose_leg_scale_clamp_k = 0.0
    try:
        if isinstance(ckpt_posttrain_cfg, dict):
            value = ckpt_posttrain_cfg.get("direct_pose_leg_enable", None)
            if value is not None:
                direct_pose_leg_enable = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_bones", None)
            if value is not None:
                direct_pose_leg_bones = str(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_mode", None)
            if value is not None:
                mode_raw = str(value).strip().lower()
                direct_pose_leg_mode = "so3" if mode_raw in ("so3", "omega", "compose", "so3_compose") else "rot6d_add"
            value = ckpt_posttrain_cfg.get("direct_pose_leg_stopgrad_main", None)
            if value is not None:
                direct_pose_leg_stopgrad_main = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_detach_feat", None)
            if value is not None:
                direct_pose_leg_detach_feat = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_max_deg", None)
            if value is not None:
                try:
                    direct_pose_leg_max_deg = float(value)
                except Exception:
                    direct_pose_leg_max_deg = 0.0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_gate_mode", None)
            if value is not None:
                gate_mode_raw = str(value).strip().lower()
                if gate_mode_raw in (
                    "signed_scale",
                    "signedscale",
                    "signed",
                    "signmag",
                    "sign_mag",
                    "signmagscale",
                    "signedmag",
                    "sscale",
                ):
                    raise SystemExit(
                        "[FATAL] ckpt posttrain_cfg uses direct_pose_leg_gate_mode='signed_scale', "
                        "which is removed in current train/eval main chain. "
                        "Migrate to direct_pose_leg_gate_mode='scale' (or 'learned')."
                    )
                direct_pose_leg_gate_mode = _DIRECT_POSE_LEG_GATE_ALIAS_MAP.get(gate_mode_raw, "none")
            value = ckpt_posttrain_cfg.get("direct_pose_leg_gate_power", None)
            if value is not None:
                try:
                    direct_pose_leg_gate_power = float(value)
                except Exception:
                    direct_pose_leg_gate_power = 1.0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_scale_log_clip", None)
            if value is not None:
                try:
                    direct_pose_leg_scale_log_clip = float(value)
                except Exception:
                    direct_pose_leg_scale_log_clip = 4.0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_scale_clamp_k", None)
            if value is not None:
                try:
                    direct_pose_leg_scale_clamp_k = float(value)
                except Exception:
                    direct_pose_leg_scale_clamp_k = 0.0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_routing", None)
            if value is not None:
                direct_pose_leg_side_routing = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_contact_order", None)
            if value is not None:
                direct_pose_leg_contact_order = str(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_embed_dim", None)
            if value is not None:
                try:
                    direct_pose_leg_side_embed_dim = int(value)
                except Exception:
                    direct_pose_leg_side_embed_dim = 0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_plan_other", None)
            if value is not None:
                direct_pose_leg_side_plan_other = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_phase_other", None)
            if value is not None:
                direct_pose_leg_side_phase_other = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_phase_rel", None)
            if value is not None:
                direct_pose_leg_side_phase_rel = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_cue", None)
            if value is not None:
                direct_pose_leg_side_cue = str(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_cue_tau", None)
            if value is not None:
                try:
                    direct_pose_leg_side_cue_tau = float(value)
                except Exception:
                    direct_pose_leg_side_cue_tau = 30.0
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_sign_gate", None)
            if value is not None:
                direct_pose_leg_side_sign_gate = bool(value)
            value = ckpt_posttrain_cfg.get("direct_pose_leg_side_rank1", None)
            if value is not None:
                direct_pose_leg_side_rank1 = bool(value)
    except Exception:
        pass
    try:
        if any(
            str(key).startswith("direct_pose_leg_head.")
            or str(key).startswith("direct_pose_leg_head_shared.")
            or str(key).startswith("direct_pose_leg_side_embed.")
            or str(key).startswith("direct_pose_leg_side_sign_gate_head.")
            for key in state_dict.keys()
        ):
            direct_pose_leg_enable = True
        if any(
            str(key).startswith("direct_pose_leg_gate_head.")
            or str(key).startswith("direct_pose_leg_gate_head_shared.")
            for key in state_dict.keys()
        ):
            direct_pose_leg_enable = True
            if str(direct_pose_leg_gate_mode).strip().lower() in ("", "none", "off", "false", "0"):
                direct_pose_leg_gate_mode = "learned"
        if any(str(key).startswith("direct_pose_leg_head_shared.") for key in state_dict.keys()):
            direct_pose_leg_side_routing = True
        if any(str(key).startswith("direct_pose_leg_gate_head_shared.") for key in state_dict.keys()):
            direct_pose_leg_side_routing = True
        if any(str(key).startswith("direct_pose_leg_side_sign_gate_head.") for key in state_dict.keys()):
            direct_pose_leg_side_routing = True
            direct_pose_leg_side_sign_gate = True
        if not bool(direct_pose_leg_side_rank1):
            weight = state_dict.get("direct_pose_leg_head_shared.6.weight", None)
            if torch.is_tensor(weight) and weight.ndim == 2 and int(weight.shape[0]) > 0 and (int(weight.shape[0]) % 3) != 0:
                direct_pose_leg_side_rank1 = True
    except Exception:
        pass
    return DirectPoseLegBuildConfig(
        enable=bool(direct_pose_leg_enable),
        bones=direct_pose_leg_bones,
        mode=str(direct_pose_leg_mode),
        stopgrad_main=bool(direct_pose_leg_stopgrad_main),
        detach_feat=bool(direct_pose_leg_detach_feat),
        max_deg=float(direct_pose_leg_max_deg),
        side_routing=bool(direct_pose_leg_side_routing),
        contact_order=str(direct_pose_leg_contact_order),
        side_embed_dim=int(direct_pose_leg_side_embed_dim),
        side_plan_other=bool(direct_pose_leg_side_plan_other),
        side_phase_other=bool(direct_pose_leg_side_phase_other),
        side_phase_rel=bool(direct_pose_leg_side_phase_rel),
        side_cue=str(direct_pose_leg_side_cue),
        side_cue_tau=float(direct_pose_leg_side_cue_tau),
        side_sign_gate=bool(direct_pose_leg_side_sign_gate),
        side_rank1=bool(direct_pose_leg_side_rank1),
        gate_mode=str(direct_pose_leg_gate_mode),
        gate_power=float(direct_pose_leg_gate_power),
        scale_log_clip=float(direct_pose_leg_scale_log_clip),
        scale_clamp_k=float(direct_pose_leg_scale_clamp_k),
    )


def resolve_event_motion_build_state_from_ckpt(
    *,
    ckpt_payload: EventMotionCkptPayload,
    in_state_dim: int,
    out_motion_dim: int,
    cond_dim: int,
    contact_dim: int,
    period_dim: int,
    contact_plan_overrides: Optional[ContactPlanBuildOverrides] = None,
    direct_pose_overrides: Optional[DirectPoseBuildOverrides] = None,
    event_clock_overrides: Optional[EventClockBuildOverrides] = None,
    lambda_fusion_overrides: Optional[LambdaFusionBuildOverrides] = None,
    default_direct_pose_arm_bones: Any = None,
    use_ckpt_direct_pose_posttrain_cfg: bool = True,
    include_direct_pose_leg_cfg: bool = False,
) -> EventMotionBuildState:
    state_dict = ckpt_payload.state_dict
    ckpt_posttrain_cfg = ckpt_payload.ckpt_posttrain_cfg
    contact_plan_cfg = resolve_contact_plan_build_cfg(
        state_dict=state_dict,
        in_state_dim=int(in_state_dim),
        cond_dim=int(cond_dim),
        contact_dim=int(contact_dim),
        overrides=contact_plan_overrides,
    )
    direct_pose_cfg = resolve_direct_pose_build_cfg(
        out_motion_dim=int(out_motion_dim),
        state_dict=state_dict,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg if use_ckpt_direct_pose_posttrain_cfg else None,
        contact_dim=int(contact_dim),
        cond_dim=int(cond_dim),
        width=int(ckpt_payload.width),
        overrides=direct_pose_overrides or DirectPoseBuildOverrides(
            train_direct_pose=False,
            direct_pose_reinit=False,
        ),
    )
    if direct_pose_cfg.arm_split_enable and direct_pose_cfg.arm_bones is None and default_direct_pose_arm_bones is not None:
        direct_pose_cfg = DirectPoseBuildConfig(
            enable=bool(direct_pose_cfg.enable),
            hidden=int(direct_pose_cfg.hidden),
            meas_mode=str(direct_pose_cfg.meas_mode),
            feat_source=str(direct_pose_cfg.feat_source),
            time_pe_dim=int(direct_pose_cfg.time_pe_dim),
            time_pe_base=float(direct_pose_cfg.time_pe_base),
            use_phase_z=bool(direct_pose_cfg.use_phase_z),
            phase_z_mode=str(direct_pose_cfg.phase_z_mode),
            split_enable=bool(direct_pose_cfg.split_enable),
            arm_split_enable=bool(direct_pose_cfg.arm_split_enable),
            arm_bones=default_direct_pose_arm_bones,
            nonleg_proj_dim=int(direct_pose_cfg.nonleg_proj_dim),
            drop_ckpt_weights=bool(direct_pose_cfg.drop_ckpt_weights),
        )
    contact_plan_model_enable = bool(
        contact_plan_cfg.enable
        or contact_plan_cfg.inject != "none"
        or direct_pose_cfg.enable
    )
    event_clock_cfg = infer_event_clock_build_cfg(
        state_dict=state_dict,
        contact_dim=int(contact_dim),
        period_dim=int(period_dim),
        overrides=event_clock_overrides or EventClockBuildOverrides(),
    )
    lambda_fusion_cfg = infer_lambda_fusion_build_cfg(
        state_dict=state_dict,
        width=int(ckpt_payload.width),
        contact_dim=int(contact_dim),
        contact_plan_enable=bool(contact_plan_model_enable),
        overrides=lambda_fusion_overrides or LambdaFusionBuildOverrides(),
    )
    direct_pose_leg_cfg = (
        resolve_direct_pose_leg_build_cfg(
            state_dict=state_dict,
            ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        )
        if include_direct_pose_leg_cfg
        else DirectPoseLegBuildConfig(
            enable=False,
            bones=None,
            mode="rot6d_add",
            stopgrad_main=False,
            detach_feat=False,
            max_deg=0.0,
            side_routing=False,
            contact_order="lr",
            side_embed_dim=0,
            side_plan_other=False,
            side_phase_other=False,
            side_phase_rel=False,
            side_cue="none",
            side_cue_tau=30.0,
            side_sign_gate=False,
            side_rank1=False,
            gate_mode="none",
            gate_power=1.0,
            scale_log_clip=4.0,
            scale_clamp_k=0.0,
        )
    )
    return EventMotionBuildState(
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        state_dict=state_dict,
        width=int(ckpt_payload.width),
        period_dim=int(ckpt_payload.period_dim),
        contact_plan_cfg=contact_plan_cfg,
        direct_pose_cfg=direct_pose_cfg,
        event_clock_cfg=event_clock_cfg,
        lambda_fusion_cfg=lambda_fusion_cfg,
        direct_pose_leg_cfg=direct_pose_leg_cfg,
    )


def prepare_event_motion_ckpt_state_for_load(
    *,
    state_dict: dict[str, Any],
    model: EventMotionModel,
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    contact_dim: int,
    direct_pose_cfg: DirectPoseBuildConfig,
    load_options: DirectPoseLoadCompatOptions,
    encoder_bundle: Any = None,
) -> dict[str, Any]:
    _raise_if_retired_direct_pose_stepc_leg_layout(
        state_dict,
        context="prepare_event_motion_ckpt_state_for_load",
    )
    load_state_dict = dict(state_dict)
    if encoder_bundle is not None:
        bundle_payload = (
            torch.load(str(encoder_bundle), map_location="cpu")
            if isinstance(encoder_bundle, (str, bytes, bytearray)) or hasattr(encoder_bundle, "__fspath__")
            else encoder_bundle
        )
        attach_motion_encoder_bundle(model, bundle_payload)
    apply_direct_pose_ckpt_compat(
        state_dict=load_state_dict,
        model=model,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        contact_dim=int(contact_dim),
        direct_pose_cfg=direct_pose_cfg,
        load_options=load_options,
    )
    return load_state_dict
