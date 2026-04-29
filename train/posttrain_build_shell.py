from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from train.checkpoint.fingerprint import (
    COMPONENT_SLOT_ORDER,
    FingerprintCompareSummary,
    build_event_motion_model_io_signature_manifest,
    build_event_motion_model_module_graph_manifest,
    build_posttrain_build_trace_manifest,
    compare_fingerprints,
    compute_build_order_hash,
    compute_io_signature_hash,
    compute_module_graph_hash,
    compute_train_policy_hash,
    compute_weights_hash,
    format_fingerprint_compare_summary,
)
from train.checkpoint.contract import enforce_strict_current_build_manifest_contract
from train.checkpoint.contract import attach_motion_encoder_bundle as _attach_motion_encoder_bundle
from train.configuration.model_build import (
    DatasetModelFacts,
    DirectPoseConfig,
    DirectPoseLegConfig,
    ModelBuildConfig,
    resolve_current_model_build_config_with_trace,
)
from train.models import EventMotionModel
from train.runtime.freeze import _freeze_all, _unfreeze_for_train_mode
from train.training_MPL import validate_and_fix_model_

if TYPE_CHECKING:
    from train.data.dataset import MotionEventDataset


_POSTTRAIN_COMPONENT_SLOT_PARAM_PREFIXES: dict[str, tuple[str, ...]] = {
    "shared_encoder": ("shared_encoder.",),
    "residual_proj": ("residual_proj.",),
    "pasa_attention_block": (
        "_pasa_q.",
        "_pasa_k.",
        "_pasa_v.",
        "_pasa_o.",
        "_pasa_lnq.",
        "_pasa_film.",
        "coupling_norm.",
    ),
    "motion_head": ("motion_head.",),
    "period_encoder": ("period_encoder.",),
    "bone_residual_adapter_bank": ("_bone_adapters.",),
    "contact_plan_cell": ("contact_plan_cell.",),
    "contact_plan_init_z": ("contact_plan_init_z",),
    "contact_plan_init_head": ("contact_plan_init_head.",),
    "contact_plan_head": ("contact_plan_head.",),
    "contact_plan_time_head": ("contact_plan_time_head.",),
    "event_clock_gate": ("event_clock_gate.",),
    "event_clock_corrector": ("event_clock_corrector.",),
    "direct_pose_head": ("direct_pose_head.",),
    "direct_pose_leg_terminal": ("direct_pose_leg_terminal.",),
    "direct_pose_out_nonleg": ("direct_pose_out_nonleg.",),
    "direct_pose_nonleg_proj": ("direct_pose_nonleg_proj.",),
    "direct_pose_out_arm": ("direct_pose_out_arm.",),
    "direct_pose_out_else": ("direct_pose_out_else.",),
    "direct_pose_arm_proj": ("direct_pose_arm_proj.",),
    "direct_pose_else_proj": ("direct_pose_else_proj.",),
    "direct_pose_leg_head": ("direct_pose_leg_head.",),
    "direct_pose_leg_gate_head": ("direct_pose_leg_gate_head.",),
    "direct_pose_leg_head_shared": ("direct_pose_leg_head_shared.",),
    "direct_pose_leg_gate_head_shared": ("direct_pose_leg_gate_head_shared.",),
    "direct_pose_leg_side_embed": ("direct_pose_leg_side_embed.",),
    "direct_pose_leg_side_sign_gate_head": ("direct_pose_leg_side_sign_gate_head.",),
    "lambda_fusion_head": ("lambda_fusion_head.",),
    "so3_delta_corrector": ("so3_delta_corrector.",),
    "so3_corr_gate_logit": ("so3_corr_gate_logit",),
    "adaptive_history_module": ("adaptive_history_module.",),
    "frozen_encoder": ("frozen_encoder.",),
    "frozen_period_head": ("frozen_period_head.",),
    "frozen_contact_head": ("frozen_contact_head.",),
}
_POSTTRAIN_FINGERPRINT_SHORT_HINTS: dict[str, str] = {
    "io_signature_hash": "forward IO signature changed",
    "module_graph_hash": "semantic module graph changed",
    "build_order_hash": "semantic build skeleton or runtime attach order changed",
    "weights_hash": "loaded model weights differ from checkpoint-declared weights",
    "train_policy_hash": "train policy changed; compare/report only",
}
_POSTTRAIN_ENFORCED_MISMATCH_SEGMENTS: tuple[str, ...] = (
    "io_signature_hash",
    "module_graph_hash",
    "build_order_hash",
)
_POSTTRAIN_LOAD_CONTEXT_CHOICES: tuple[str, ...] = ("resume", "chain_hop")
_POSTTRAIN_LOAD_CONTEXT_HINT = "caller must set load_context to one of: resume|chain_hop"
_STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE = "2026-04-28 strict direct-pose shape-inference unload"
_STRICT_DIRECT_POSE_PREFIXES: tuple[str, ...] = (
    "direct_pose_head.",
    "direct_pose_leg_terminal.",
    "direct_pose_out_nonleg.",
    "direct_pose_nonleg_proj.",
    "direct_pose_out_arm.",
    "direct_pose_out_else.",
    "direct_pose_arm_proj.",
    "direct_pose_else_proj.",
    "direct_pose_leg_head.",
    "direct_pose_leg_gate_head.",
    "direct_pose_leg_head_shared.",
    "direct_pose_leg_gate_head_shared.",
    "direct_pose_leg_side_embed.",
    "direct_pose_leg_side_sign_gate_head.",
)
_STRICT_DIRECT_POSE_EXACT_KEYS: tuple[str, ...] = (
    "direct_pose_leg_joint_idx_tensor",
    "direct_pose_leg_side_pos_r_tensor",
    "direct_pose_leg_side_pos_l_tensor",
    "direct_pose_leg_out_idx",
    "direct_pose_nonleg_out_idx",
    "direct_pose_arm_out_idx",
    "direct_pose_else_out_idx",
)


@dataclass
class PostTrainModelArtifacts:
    model: EventMotionModel
    build_state: "PostTrainModelBuildState"
    direct_pose_feat_source: str
    direct_pose_time_pe_dim: int
    direct_pose_time_pe_base: float
    direct_pose_use_phase_z: bool
    direct_pose_phase_z_mode: str
    direct_pose_split_enable: bool
    direct_pose_nonleg_proj_dim: int
    direct_pose_leg_gate_mode_model: str
    direct_pose_leg_gate_power_model: float
    fingerprint_compare_summary: Optional[FingerprintCompareSummary] = None
    current_fingerprints: Optional[dict[str, str]] = None
    resolved_build_manifest: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class PostTrainModelBuildState:
    """Resolved model-build inputs derived from checkpoint, dataset, and CLI policy."""

    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    model_build_config: ModelBuildConfig
    width: int

    contact_dim: int
    angvel_dim: int
    pose_hist_dim: int

    contact_plan_enable: bool
    contact_plan_hidden: int
    contact_plan_inject: str
    contact_plan_time_pe_dim: int
    contact_plan_init_mode: str
    contact_plan_init_hidden: int
    contact_plan_init_dropout: float

    use_event_clock: bool
    event_clock_hidden_dim: int
    event_clock_gate_hidden_dim: int
    event_clock_max_delta: float
    period_dim_init: int

    direct_pose_cfg: DirectPoseConfig
    direct_pose_leg_cfg: DirectPoseLegConfig

    lambda_fusion_enable: bool
    lambda_fusion_mode: str
    lambda_fusion_hidden: int
    lambda_fusion_dropout: float
    lambda_fusion_logit_init: float
    lambda_fusion_use_rollout_step: bool

    direct_pose_leg_gate_mode_model: str
    direct_pose_leg_gate_power_model: float
    fingerprint_schema_version: Optional[int] = None
    checkpoint_fingerprints: Optional[dict[str, str]] = None
    checkpoint_manifest_summary: Optional[dict[str, Any]] = None
    resolved_build_manifest: Optional[dict[str, Any]] = None


def _resolve_posttrain_train_mode(cfg: Any) -> str:
    train_direct_pose = bool(getattr(cfg, "train_direct_pose", False))
    train_lambda_head = bool(getattr(cfg, "train_lambda_head", False))
    selected = int(train_direct_pose) + int(train_lambda_head)
    if selected != 1:
        raise SystemExit("[FATAL] Choose exactly one: train_direct_pose | train_lambda_head.")
    return "direct" if train_direct_pose else "lambda"


def _validate_posttrain_load_context(load_context: Any) -> str:
    load_context_text = None if load_context is None else str(load_context).strip()
    if load_context_text in _POSTTRAIN_LOAD_CONTEXT_CHOICES:
        return str(load_context_text)
    if load_context is None or load_context_text == "":
        raise SystemExit(f"[FATAL] {_POSTTRAIN_LOAD_CONTEXT_HINT}")
    raise SystemExit(f"[FATAL] {_POSTTRAIN_LOAD_CONTEXT_HINT} (got {load_context!r})")


def _resolve_posttrain_load_context(cfg: Any) -> str:
    return _validate_posttrain_load_context(getattr(cfg, "load_context", None))


def _resolve_strict_current_width(cfg: Any) -> int:
    width = getattr(cfg, "width", None)
    if width is None:
        raise SystemExit(
            "[FATAL][strict_current_model_build] cfg.width/--width is required because strict current "
            "build cannot infer hidden width from checkpoint tensors."
        )
    try:
        width_int = int(width)
    except (TypeError, ValueError) as exc:
        raise SystemExit("[FATAL][strict_current_model_build] cfg.width/--width must be an integer.") from exc
    if width_int <= 0:
        raise SystemExit("[FATAL][strict_current_model_build] cfg.width/--width must be > 0.")
    return int(width_int)


def _filter_checkpoint_state_dict(
    raw_model_state: Any,
    *,
    ignored_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(raw_model_state, dict):
        raise TypeError(f"checkpoint model state must be a mapping; got {type(raw_model_state).__name__}.")
    return {
        str(key): value
        for key, value in raw_model_state.items()
        if not any(str(key).startswith(prefix) for prefix in ignored_prefixes)
    }


def _build_posttrain_train_policy_manifest(*, cfg: Any, train_mode: str) -> dict[str, Any]:
    return {
        "train_mode": str(train_mode),
        "direct_pose_leg_train_only": bool(getattr(cfg, "direct_pose_leg_train_only", False)),
        "direct_pose_leg_gate_train_only": bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
        "direct_pose_nonleg_train_only": bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
        "direct_pose_nonleg_trunk_mode": str(getattr(cfg, "direct_pose_nonleg_trunk_mode", "none") or "none"),
        "rollout_steps": int(getattr(cfg, "rollout_steps", 0) or 0),
        "rollout_cycles": int(getattr(cfg, "rollout_cycles", 1) or 1),
        "rollout_include_boundary": bool(getattr(cfg, "rollout_include_boundary", False)),
        "rollout_random_offset": bool(getattr(cfg, "rollout_random_offset", False)),
        "time_index_mode": str(getattr(cfg, "time_index_mode", "auto") or "auto"),
        "detach_rollout_state": bool(getattr(cfg, "detach_rollout_state", False)),
        "contact_meas_weight": float(getattr(cfg, "contact_meas_weight", 0.0) or 0.0),
        "lambda_reliability_mode": str(getattr(cfg, "lambda_reliability_mode", "none") or "none"),
    }


def _build_posttrain_structural_flags(*, build_state: PostTrainModelBuildState) -> dict[str, Any]:
    return {
        "direct_pose_enable": bool(build_state.direct_pose_cfg.enable),
        "lambda_fusion_enable": bool(build_state.lambda_fusion_enable),
        "contact_plan_enable": bool(build_state.contact_plan_enable),
        "use_event_clock": bool(build_state.use_event_clock),
        "direct_pose_leg_enable": bool(build_state.direct_pose_leg_cfg.enable),
        "direct_pose_leg_side_routing": bool(build_state.direct_pose_leg_cfg.side_routing),
        "direct_pose_arm_split_enable": bool(build_state.direct_pose_cfg.arm_split_enable),
        "direct_pose_leg_mode": str(build_state.direct_pose_leg_cfg.mode),
    }


def _build_component_slot_trainable_map(model: EventMotionModel) -> dict[str, bool]:
    requires_grad_names = {
        str(name)
        for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    trainable_slots: dict[str, bool] = {}
    for slot in COMPONENT_SLOT_ORDER:
        prefixes = _POSTTRAIN_COMPONENT_SLOT_PARAM_PREFIXES.get(slot, ())
        trainable_slots[slot] = any(
            any(name == prefix or name.startswith(prefix) for prefix in prefixes)
            for name in requires_grad_names
        )
    return trainable_slots


def _capture_parameter_requires_grad(model: EventMotionModel) -> dict[str, bool]:
    return {
        str(name): bool(getattr(parameter, "requires_grad", False))
        for name, parameter in model.named_parameters()
    }


def _restore_parameter_requires_grad(model: EventMotionModel, saved_flags: dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        if name in saved_flags:
            parameter.requires_grad_(bool(saved_flags[name]))


def _build_posttrain_current_fingerprints(
    *,
    cfg: Any,
    model: EventMotionModel,
    build_state: PostTrainModelBuildState,
    weights_state_dict: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    train_mode = _resolve_posttrain_train_mode(cfg)
    saved_requires_grad = _capture_parameter_requires_grad(model)
    try:
        _freeze_all(model)
        _unfreeze_for_train_mode(
            model,
            train_mode=train_mode,
            direct_pose_leg_train_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
            direct_pose_leg_gate_train_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
            direct_pose_nonleg_train_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
            direct_pose_nonleg_trunk_mode=str(getattr(cfg, "direct_pose_nonleg_trunk_mode", "none") or "none"),
        )
        trainable_slots = _build_component_slot_trainable_map(model)
        io_signature = build_event_motion_model_io_signature_manifest(model)
        module_graph = build_event_motion_model_module_graph_manifest(model)
        build_trace = build_posttrain_build_trace_manifest(
            structural_flags=_build_posttrain_structural_flags(build_state=build_state),
            train_mode=train_mode,
            trainable_slots=trainable_slots,
        )
        return {
            "io_signature_hash": compute_io_signature_hash(io_signature),
            "module_graph_hash": compute_module_graph_hash(module_graph),
            "build_order_hash": compute_build_order_hash(build_trace),
            "weights_hash": compute_weights_hash(model.state_dict() if weights_state_dict is None else weights_state_dict),
            "train_policy_hash": compute_train_policy_hash(
                _build_posttrain_train_policy_manifest(cfg=cfg, train_mode=train_mode)
            ),
        }
    finally:
        _restore_parameter_requires_grad(model, saved_requires_grad)


def _compare_posttrain_checkpoint_fingerprints(
    *,
    cfg: Any,
    model: EventMotionModel,
    build_state: PostTrainModelBuildState,
    weights_state_dict: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, str], FingerprintCompareSummary]:
    current_fingerprints = _build_posttrain_current_fingerprints(
        cfg=cfg,
        model=model,
        build_state=build_state,
        weights_state_dict=weights_state_dict,
    )
    summary = compare_fingerprints(
        build_state.checkpoint_fingerprints,
        current_fingerprints,
        short_diff_hints=_POSTTRAIN_FINGERPRINT_SHORT_HINTS,
    )
    return current_fingerprints, summary


def _emit_posttrain_checkpoint_fingerprint_report(
    *,
    summary: FingerprintCompareSummary,
    manifest_summary_present: bool,
    load_context: Optional[str] = None,
) -> None:
    manifest_state = "present" if manifest_summary_present else "missing"
    context_suffix = ""
    if load_context is not None:
        resolved_load_context = _validate_posttrain_load_context(load_context)
        context_suffix = f"; load_context={resolved_load_context}; policy=strict-current"
    print(
        f"[posttrain][fingerprint] compare summary follows "
        f"(manifest_summary={manifest_state}{context_suffix})."
    )
    for line in format_fingerprint_compare_summary(summary).splitlines():
        print(f"[posttrain][fingerprint] {line}")


def _posttrain_fingerprint_enforce_required(
    summary: FingerprintCompareSummary,
    *,
    load_context: str,
) -> bool:
    resolved_load_context = _validate_posttrain_load_context(load_context)
    required_segments = {str(segment) for segment in summary.required_segments}
    for result in summary.results:
        if result.status == "missing_required":
            return True
        if result.segment in required_segments and result.status == "mismatch":
            return True
        if (
            resolved_load_context == "resume"
            and result.segment in _POSTTRAIN_ENFORCED_MISMATCH_SEGMENTS
            and result.status == "mismatch"
        ):
            return True
    return False


def _enforce_posttrain_checkpoint_fingerprint(
    summary: FingerprintCompareSummary,
    *,
    load_context: str,
) -> None:
    if not _posttrain_fingerprint_enforce_required(
        summary,
        load_context=load_context,
    ):
        return
    raise SystemExit(format_fingerprint_compare_summary(summary))


def _resolve_posttrain_model_build_state(*, cfg: Any, ds: MotionEventDataset) -> PostTrainModelBuildState:
    """Resolve checkpoint-backed model build state for posttrain instantiation."""
    ckpt = torch.load(cfg.ckpt_in.expanduser(), map_location="cpu")
    fingerprint_schema_version_raw = ckpt.get("fingerprint_schema_version", None) if isinstance(ckpt, dict) else None
    try:
        fingerprint_schema_version = int(fingerprint_schema_version_raw)
    except (TypeError, ValueError):
        fingerprint_schema_version = None
    checkpoint_fingerprints_raw = ckpt.get("fingerprints", None) if isinstance(ckpt, dict) else None
    checkpoint_fingerprints: Optional[dict[str, str]]
    if isinstance(checkpoint_fingerprints_raw, dict):
        checkpoint_fingerprints = {
            str(key): str(value)
            for key, value in checkpoint_fingerprints_raw.items()
            if value is not None
        }
    else:
        checkpoint_fingerprints = None
    checkpoint_manifest_summary_raw = ckpt.get("manifest_summary", None) if isinstance(ckpt, dict) else None
    checkpoint_manifest_summary = (
        {str(key): value for key, value in checkpoint_manifest_summary_raw.items()}
        if isinstance(checkpoint_manifest_summary_raw, dict)
        else None
    )
    checkpoint_resolved_build_manifest = (
        ckpt.get("resolved_build_manifest", None) if isinstance(ckpt, dict) else None
    )
    checkpoint_contract_raw = ckpt.get("checkpoint_contract", None) if isinstance(ckpt, dict) else None
    checkpoint_resolved_build_manifest_hash = (
        ckpt.get("resolved_build_manifest_hash", None) if isinstance(ckpt, dict) else None
    )
    if checkpoint_resolved_build_manifest_hash is None and isinstance(checkpoint_contract_raw, dict):
        checkpoint_resolved_build_manifest_hash = checkpoint_contract_raw.get("resolved_build_manifest_hash", None)
    ckpt_posttrain_payload = ckpt.get("posttrain_cfg", None) if isinstance(ckpt, dict) else None
    ckpt_posttrain_cfg: Optional[dict[str, Any]] = ckpt_posttrain_payload if isinstance(ckpt_posttrain_payload, dict) else None
    raw_model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(checkpoint_resolved_build_manifest, dict):
        enforce_strict_current_build_manifest_contract(
            current_manifest={},
            checkpoint_manifest=checkpoint_resolved_build_manifest,
            checkpoint_manifest_hash=checkpoint_resolved_build_manifest_hash,
        )
    width = _resolve_strict_current_width(cfg)
    # strict/current intentionally strips no checkpoint prefixes; mismatches fail fast at shape/manifest validation.
    state_dict = _filter_checkpoint_state_dict(
        raw_model_state,
        ignored_prefixes=(),
    )
    period_dim = int(state_dict["period_encoder.weight"].shape[1]) if "period_encoder.weight" in state_dict else 0
    dataset_facts = DatasetModelFacts.from_dataset(ds, context="posttrain.dataset")
    encoder_bundle = getattr(cfg, "encoder_bundle", None)
    has_encoder_bundle = bool(encoder_bundle is not None and encoder_bundle.expanduser().is_file())
    resolved_model_build = resolve_current_model_build_config_with_trace(
        cfg=cfg,
        dataset_facts=dataset_facts,
        width=int(width),
    )
    model_build_config = resolved_model_build.config
    resolved_build_manifest = resolved_model_build.manifest()
    enforce_strict_current_build_manifest_contract(
        current_manifest=resolved_build_manifest,
        checkpoint_manifest=checkpoint_resolved_build_manifest,
        checkpoint_manifest_hash=checkpoint_resolved_build_manifest_hash,
    )
    contact_plan_cfg = model_build_config.contact_plan
    direct_pose_cfg = model_build_config.direct_pose
    direct_pose_leg_cfg = model_build_config.direct_pose_leg
    event_clock_cfg = model_build_config.event_clock
    lambda_fusion_cfg = model_build_config.lambda_fusion
    return PostTrainModelBuildState(
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        state_dict=state_dict,
        model_build_config=model_build_config,
        width=width,
        contact_dim=int(model_build_config.facts.contact_dim),
        angvel_dim=int(model_build_config.facts.angvel_dim),
        pose_hist_dim=int(model_build_config.facts.pose_hist_dim),
        contact_plan_enable=bool(contact_plan_cfg.enable),
        contact_plan_hidden=int(contact_plan_cfg.hidden),
        contact_plan_inject=str(contact_plan_cfg.inject),
        contact_plan_time_pe_dim=int(contact_plan_cfg.time_pe_dim),
        contact_plan_init_mode=str(contact_plan_cfg.init_mode),
        contact_plan_init_hidden=int(contact_plan_cfg.init_hidden),
        contact_plan_init_dropout=float(contact_plan_cfg.init_dropout),
        use_event_clock=bool(event_clock_cfg.enable),
        event_clock_hidden_dim=int(event_clock_cfg.hidden_dim),
        event_clock_gate_hidden_dim=int(event_clock_cfg.gate_hidden_dim),
        event_clock_max_delta=float(event_clock_cfg.max_delta),
        period_dim_init=int(event_clock_cfg.period_dim_init),
        direct_pose_cfg=direct_pose_cfg,
        direct_pose_leg_cfg=direct_pose_leg_cfg,
        lambda_fusion_enable=bool(lambda_fusion_cfg.enable),
        lambda_fusion_mode=str(lambda_fusion_cfg.mode),
        lambda_fusion_hidden=int(lambda_fusion_cfg.hidden),
        lambda_fusion_dropout=float(lambda_fusion_cfg.dropout),
        lambda_fusion_logit_init=float(lambda_fusion_cfg.logit_init),
        lambda_fusion_use_rollout_step=bool(lambda_fusion_cfg.use_rollout_step),
        direct_pose_leg_gate_mode_model=str(direct_pose_leg_cfg.gate_mode),
        direct_pose_leg_gate_power_model=float(direct_pose_leg_cfg.gate_power),
        fingerprint_schema_version=fingerprint_schema_version,
        checkpoint_fingerprints=checkpoint_fingerprints,
        checkpoint_manifest_summary=checkpoint_manifest_summary,
        resolved_build_manifest=resolved_build_manifest,
    )


def _instantiate_posttrain_model(
    *,
    cfg: Any,
    ds: MotionEventDataset,
    device: torch.device,
    build_state: PostTrainModelBuildState,
) -> EventMotionModel:
    model = EventMotionModel.from_config(build_state.model_build_config).to(device)
    validate_and_fix_model_(model, int(ds.Dx), int(ds.Dc))
    return model


def _shape_tuple(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _format_strict_shape_items(items: list[str], *, limit: int = 20) -> str:
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f"\n  ... ({len(items) - limit} more)"
    return "\n  " + "\n  ".join(shown) + suffix if shown else ""


def _is_strict_direct_pose_key(key: str) -> bool:
    return key in _STRICT_DIRECT_POSE_EXACT_KEYS or key.startswith(_STRICT_DIRECT_POSE_PREFIXES)


def _validate_strict_current_direct_pose_checkpoint_shapes(
    *,
    model: EventMotionModel,
    state_dict: dict[str, Any],
    stage: str,
) -> None:
    model_state = model.state_dict()
    missing: list[str] = []
    unexpected: list[str] = []
    mismatched: list[str] = []

    for key, model_value in model_state.items():
        if not _is_strict_direct_pose_key(str(key)) or not torch.is_tensor(model_value):
            continue
        if key not in state_dict:
            missing.append(str(key))
            continue
        ckpt_value = state_dict[key]
        if not torch.is_tensor(ckpt_value):
            mismatched.append(
                f"{key}: checkpoint value is {type(ckpt_value).__name__}, "
                f"model shape={_shape_tuple(model_value)}"
            )
            continue
        if _shape_tuple(ckpt_value) != _shape_tuple(model_value):
            mismatched.append(
                f"{key}: ckpt={_shape_tuple(ckpt_value)} model={_shape_tuple(model_value)}"
            )

    for key, ckpt_value in state_dict.items():
        if torch.is_tensor(ckpt_value) and _is_strict_direct_pose_key(str(key)) and key not in model_state:
            unexpected.append(str(key))

    if not (missing or unexpected or mismatched):
        return

    sections: list[str] = []
    if mismatched:
        sections.append("shape/type mismatches:" + _format_strict_shape_items(mismatched))
    if missing:
        sections.append("missing checkpoint tensors:" + _format_strict_shape_items(missing))
    if unexpected:
        sections.append("unexpected checkpoint tensors:" + _format_strict_shape_items(unexpected))
    raise SystemExit(
        f"[FATAL][strict_current_model_build][{stage}][Removed] direct-pose checkpoint keys do not match the "
        "resolved current build. Removed by "
        f"{_STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE}: strict/current load does not infer direct-pose "
        "semantics from checkpoint tensor shape or checkpoint posttrain_cfg. Migration: contractize/migrate "
        "to a shape-coherent strict checkpoint or update the resolved config; no load-time shape/posttrain_cfg "
        "replacement.\n"
        + "\n".join(sections)
    )


def _validate_strict_current_model_checkpoint_shapes(
    *,
    model: EventMotionModel,
    state_dict: dict[str, Any],
    stage: str,
) -> None:
    model_state = model.state_dict()
    missing: list[str] = []
    unexpected: list[str] = []
    mismatched: list[str] = []

    for key, model_value in model_state.items():
        if not torch.is_tensor(model_value):
            continue
        if key not in state_dict:
            missing.append(str(key))
            continue
        ckpt_value = state_dict[key]
        if not torch.is_tensor(ckpt_value):
            mismatched.append(
                f"{key}: checkpoint value is {type(ckpt_value).__name__}, "
                f"model shape={_shape_tuple(model_value)}"
            )
            continue
        if _shape_tuple(ckpt_value) != _shape_tuple(model_value):
            mismatched.append(
                f"{key}: ckpt={_shape_tuple(ckpt_value)} model={_shape_tuple(model_value)}"
            )

    for key, ckpt_value in state_dict.items():
        if torch.is_tensor(ckpt_value) and key not in model_state:
            unexpected.append(str(key))

    if not (missing or unexpected or mismatched):
        return

    sections: list[str] = []
    if mismatched:
        sections.append("shape/type mismatches:" + _format_strict_shape_items(mismatched))
    if missing:
        sections.append("missing checkpoint tensors:" + _format_strict_shape_items(missing))
    if unexpected:
        sections.append("unexpected checkpoint tensors:" + _format_strict_shape_items(unexpected))
    raise SystemExit(
        f"[FATAL][strict_current_model_build][{stage}] checkpoint does not match the current resolved model build. "
        "Migrate legacy checkpoints with tools/migrate_legacy_posttrain_ckpt.py and re-run with a "
        "strict/current contract checkpoint.\n"
        + "\n".join(sections)
    )


def _load_posttrain_checkpoint_into_model(
    *,
    cfg: Any,
    model: EventMotionModel,
    build_state: PostTrainModelBuildState,
) -> None:
    direct_pose_cfg = build_state.direct_pose_cfg
    state_dict = build_state.state_dict
    encoder_bundle_path = getattr(cfg, "encoder_bundle", None)
    has_encoder_bundle = bool(encoder_bundle_path is not None and encoder_bundle_path.expanduser().is_file())

    if has_encoder_bundle:
        _attach_motion_encoder_bundle(
            model,
            torch.load(str(encoder_bundle_path.expanduser()), map_location="cpu"),
        )
    _validate_strict_current_direct_pose_checkpoint_shapes(
        model=model,
        state_dict=state_dict,
        stage="pre-load",
    )
    _validate_strict_current_model_checkpoint_shapes(
        model=model,
        state_dict=state_dict,
        stage="pre-load",
    )
    model.load_state_dict(state_dict, strict=True)
    print("[posttrain][build] strict current model checkpoint shape validation passed.")

    if cfg.train_direct_pose:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] direct_pose_head is not instantiated; cannot train direct pose expert.")
        leg_only = bool(getattr(cfg, "direct_pose_leg_train_only", False))
        leg_gate_only = bool(getattr(cfg, "direct_pose_leg_gate_train_only", False))
        nonleg_only = bool(getattr(cfg, "direct_pose_nonleg_train_only", False))
        if nonleg_only and (leg_only or leg_gate_only):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true is incompatible with leg train_only modes. "
                "Pick exactly one train_only mode."
            )
        if (leg_only or leg_gate_only) and getattr(model, "direct_pose_leg_head", None) is None:
            raise SystemExit(
                "[FATAL] direct_pose_leg_*_train_only=true but no leg head is instantiated. "
                "Enable direct_pose_leg_enable and provide valid direct_pose_leg_bones."
            )
        has_nonleg_branch = (
            getattr(model, "direct_pose_out_nonleg", None) is not None
            or (
                getattr(model, "direct_pose_out_arm", None) is not None
                and getattr(model, "direct_pose_out_else", None) is not None
            )
        )
        if nonleg_only and (not has_nonleg_branch):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_train_only=true but no non-leg branch is instantiated. "
                "Enable direct_pose_split_enable (optionally with direct_pose_arm_split_enable)."
            )
        trunk_mode = str(getattr(cfg, "direct_pose_nonleg_trunk_mode", "none") or "none").strip().lower()
        if trunk_mode not in ("none", "last", "full"):
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_trunk_mode must be one of: none | last | full "
                f"(got {trunk_mode!r})."
            )
        if trunk_mode != "none" and not nonleg_only:
            raise SystemExit(
                "[FATAL] direct_pose_nonleg_trunk_mode requires direct_pose_nonleg_train_only=true."
            )
        if bool(leg_gate_only):
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_train_only=true but no leg gate/scale head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned'/'scale' and enable direct_pose_leg_enable with valid bones."
                )
        if float(getattr(cfg, "direct_pose_leg_gate_sup_weight", 0.0) or 0.0) > 0.0:
            leg_mode = str(getattr(model, "direct_pose_leg_mode", "rot6d_add") or "rot6d_add").strip().lower()
            if leg_mode != "so3":
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 requires direct_pose_leg_mode='so3' "
                    f"(got {leg_mode!r})."
                )
            has_leg_gate = (getattr(model, "direct_pose_leg_gate_head", None) is not None) or (
                getattr(model, "direct_pose_leg_gate_head_shared", None) is not None
            )
            if not has_leg_gate:
                raise SystemExit(
                    "[FATAL] direct_pose_leg_gate_sup_weight>0 but no learned leg gate head is instantiated. "
                    "Set direct_pose_leg_gate_mode='learned' and enable direct_pose_leg_enable with valid bones."
                )
    if cfg.train_lambda_head:
        if getattr(model, "direct_pose_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs direct_pose_head (out_direct), but checkpoint/model does not enable it.")
        if getattr(model, "lambda_fusion_head", None) is None:
            raise SystemExit("[FATAL] Stage2 needs lambda_fusion_head, but it is not instantiated.")

    if cfg.so3_corr_gate_logit_reset is not None:
        logit = getattr(model, "so3_corr_gate_logit", None)
        if torch.is_tensor(logit):
            with torch.no_grad():
                logit.fill_(float(cfg.so3_corr_gate_logit_reset))
            print(f"[posttrain] reset so3_corr_gate_logit={float(cfg.so3_corr_gate_logit_reset):.4f}")


def _build_posttrain_model_from_ckpt(
    *,
    cfg: Any,
    ds: MotionEventDataset,
    device: torch.device,
) -> PostTrainModelArtifacts:
    load_context = _resolve_posttrain_load_context(cfg)
    build_state = _resolve_posttrain_model_build_state(cfg=cfg, ds=ds)
    resolved_manifest = build_state.resolved_build_manifest
    if isinstance(resolved_manifest, dict):
        trace = resolved_manifest.get("trace", ())
        fields = len(trace) if isinstance(trace, (list, tuple)) else 0
        print(f"[posttrain][build] resolved model build manifest available: fields={fields}")
    model = _instantiate_posttrain_model(cfg=cfg, ds=ds, device=device, build_state=build_state)
    current_fingerprints, fingerprint_compare_summary = _compare_posttrain_checkpoint_fingerprints(
        cfg=cfg,
        model=model,
        build_state=build_state,
        weights_state_dict=getattr(build_state, "state_dict", None),
    )
    _emit_posttrain_checkpoint_fingerprint_report(
        summary=fingerprint_compare_summary,
        manifest_summary_present=build_state.checkpoint_manifest_summary is not None,
        load_context=load_context,
    )
    _enforce_posttrain_checkpoint_fingerprint(
        fingerprint_compare_summary,
        load_context=load_context,
    )
    _load_posttrain_checkpoint_into_model(cfg=cfg, model=model, build_state=build_state)
    direct_pose_cfg = build_state.direct_pose_cfg
    return PostTrainModelArtifacts(
        model=model,
        build_state=build_state,
        direct_pose_feat_source=str(direct_pose_cfg.feat_source),
        direct_pose_time_pe_dim=int(direct_pose_cfg.time_pe_dim),
        direct_pose_time_pe_base=float(direct_pose_cfg.time_pe_base),
        direct_pose_use_phase_z=bool(direct_pose_cfg.use_phase_z),
        direct_pose_phase_z_mode=str(direct_pose_cfg.phase_z_mode),
        direct_pose_split_enable=bool(direct_pose_cfg.split_enable),
        direct_pose_nonleg_proj_dim=int(direct_pose_cfg.nonleg_proj_dim),
        direct_pose_leg_gate_mode_model=str(build_state.direct_pose_leg_gate_mode_model),
        direct_pose_leg_gate_power_model=float(build_state.direct_pose_leg_gate_power_model),
        fingerprint_compare_summary=fingerprint_compare_summary,
        current_fingerprints=current_fingerprints,
        resolved_build_manifest=build_state.resolved_build_manifest,
    )
