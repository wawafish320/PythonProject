from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from .contract import (
    normalize_contact_plan_init_mode,
    normalize_direct_pose_feat_source,
    normalize_direct_pose_leg_gate_mode,
    normalize_direct_pose_leg_gate_power,
    normalize_direct_pose_leg_mode,
    normalize_direct_pose_phase_z_mode,
    normalize_lambda_fusion_mode,
)

FINGERPRINT_SCHEMA_VERSION = 1
FINGERPRINT_POLICY_INTRO_DATE = "2026-04-25"
REQUIRED_FINGERPRINT_SEGMENTS: tuple[str, ...] = (
    "io_signature_hash",
    "module_graph_hash",
    "build_order_hash",
    "weights_hash",
)
OPTIONAL_FINGERPRINT_SEGMENTS: tuple[str, ...] = ("train_policy_hash",)

COMPONENT_KIND_VOCAB: tuple[str, ...] = (
    "encoder_trunk",
    "residual_bypass_proj",
    "attention_coupler",
    "motion_readout",
    "phase_projection",
    "residual_adapter_bank",
    "recurrent_plan_core",
    "state_seed",
    "obs_seed_adapter",
    "contact_logit_head",
    "time_bias_head",
    "periodicity_gate",
    "plan_state_corrector",
    "pose_trunk",
    "pose_terminal",
    "pose_branch_readout",
    "feature_bottleneck",
    "leg_residual_head",
    "leg_gate_head",
    "side_routed_leg_head",
    "side_routed_leg_gate_head",
    "side_embedding",
    "side_sign_gate",
    "fusion_gate_head",
    "so3_delta_head",
    "scalar_gate_parameter",
    "adaptive_history_encoder",
    "external_frozen_encoder",
    "external_frozen_period_head",
    "external_frozen_contact_head",
)

COMPONENT_SLOT_ORDER: tuple[str, ...] = (
    "shared_encoder",
    "residual_proj",
    "pasa_attention_block",
    "motion_head",
    "period_encoder",
    "bone_residual_adapter_bank",
    "contact_plan_cell",
    "contact_plan_init_z",
    "contact_plan_init_head",
    "contact_plan_head",
    "contact_plan_time_head",
    "event_clock_gate",
    "event_clock_corrector",
    "direct_pose_head",
    "direct_pose_leg_terminal",
    "direct_pose_out_nonleg",
    "direct_pose_nonleg_proj",
    "direct_pose_out_arm",
    "direct_pose_out_else",
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_leg_head",
    "direct_pose_leg_gate_head",
    "direct_pose_leg_head_shared",
    "direct_pose_leg_gate_head_shared",
    "direct_pose_leg_side_embed",
    "direct_pose_leg_side_sign_gate_head",
    "lambda_fusion_head",
    "so3_delta_corrector",
    "so3_corr_gate_logit",
    "adaptive_history_module",
    "frozen_encoder",
    "frozen_period_head",
    "frozen_contact_head",
)

COMPONENT_SLOT_KIND_MAP: dict[str, str] = {
    "shared_encoder": "encoder_trunk",
    "residual_proj": "residual_bypass_proj",
    "pasa_attention_block": "attention_coupler",
    "motion_head": "motion_readout",
    "period_encoder": "phase_projection",
    "bone_residual_adapter_bank": "residual_adapter_bank",
    "contact_plan_cell": "recurrent_plan_core",
    "contact_plan_init_z": "state_seed",
    "contact_plan_init_head": "obs_seed_adapter",
    "contact_plan_head": "contact_logit_head",
    "contact_plan_time_head": "time_bias_head",
    "event_clock_gate": "periodicity_gate",
    "event_clock_corrector": "plan_state_corrector",
    "direct_pose_head": "pose_trunk",
    "direct_pose_leg_terminal": "pose_terminal",
    "direct_pose_out_nonleg": "pose_branch_readout",
    "direct_pose_nonleg_proj": "feature_bottleneck",
    "direct_pose_out_arm": "pose_branch_readout",
    "direct_pose_out_else": "pose_branch_readout",
    "direct_pose_arm_proj": "feature_bottleneck",
    "direct_pose_else_proj": "feature_bottleneck",
    "direct_pose_leg_head": "leg_residual_head",
    "direct_pose_leg_gate_head": "leg_gate_head",
    "direct_pose_leg_head_shared": "side_routed_leg_head",
    "direct_pose_leg_gate_head_shared": "side_routed_leg_gate_head",
    "direct_pose_leg_side_embed": "side_embedding",
    "direct_pose_leg_side_sign_gate_head": "side_sign_gate",
    "lambda_fusion_head": "fusion_gate_head",
    "so3_delta_corrector": "so3_delta_head",
    "so3_corr_gate_logit": "scalar_gate_parameter",
    "adaptive_history_module": "adaptive_history_encoder",
    "frozen_encoder": "external_frozen_encoder",
    "frozen_period_head": "external_frozen_period_head",
    "frozen_contact_head": "external_frozen_contact_head",
}

_COMPONENT_SLOT_INDEX = {slot: idx for idx, slot in enumerate(COMPONENT_SLOT_ORDER)}
_WEIGHTS_FINGERPRINT_ACTION = "weights changed; validate that the checkpoint replacement is intentional."
_REQUIRED_SEGMENT_ACTION = "regenerate checkpoint with current mainline or revert to the last supported semantic graph."
_MISSING_OPTIONAL_ACTION = "optional segment absent; no-check. Re-save on current mainline to materialize it."
_OPTIONAL_SEGMENT_ACTION = "optional segment drift observed; compare/report only."


@dataclass(frozen=True, slots=True)
class IOFieldManifest:
    key: str
    optional: bool
    shape_semantics: tuple[str, ...]
    dims: Mapping[str, int] = field(default_factory=dict)
    dtype: str = "floating_tensor"
    accepted_forms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class IOSignatureManifest:
    inputs: tuple[IOFieldManifest, ...] = field(default_factory=tuple)
    outputs: tuple[IOFieldManifest, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    component_slot: str
    component_kind: str
    enabled: bool
    consumes: tuple[str, ...] = field(default_factory=tuple)
    produces: tuple[str, ...] = field(default_factory=tuple)
    normalized_config: Mapping[str, Any] = field(default_factory=dict)
    children: tuple[str, ...] = field(default_factory=tuple)
    order_sensitive_consumes: bool = False
    order_sensitive_produces: bool = False


@dataclass(frozen=True, slots=True)
class ModuleGraphManifest:
    components: tuple[ComponentManifest, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BuildSubstepManifest:
    step_id: str
    consumes: tuple[str, ...] = field(default_factory=tuple)
    produces: tuple[str, ...] = field(default_factory=tuple)
    attached_attrs: tuple[str, ...] = field(default_factory=tuple)
    normalized_config: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BuildStepManifest:
    step_id: str
    step_order: int
    consumes: tuple[str, ...] = field(default_factory=tuple)
    produces: tuple[str, ...] = field(default_factory=tuple)
    attached_attrs: tuple[str, ...] = field(default_factory=tuple)
    normalized_config: Mapping[str, Any] = field(default_factory=dict)
    substeps: tuple[BuildSubstepManifest, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BuildTraceManifest:
    pipeline: str
    steps: tuple[BuildStepManifest, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CanonicalCheckpointManifest:
    fingerprint_schema_version: int = FINGERPRINT_SCHEMA_VERSION
    required_segments: tuple[str, ...] = REQUIRED_FINGERPRINT_SEGMENTS
    optional_segments: tuple[str, ...] = OPTIONAL_FINGERPRINT_SEGMENTS
    io_signature: IOSignatureManifest = field(default_factory=IOSignatureManifest)
    module_graph: ModuleGraphManifest = field(default_factory=ModuleGraphManifest)
    build_trace: Optional[BuildTraceManifest] = None
    train_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FingerprintSegmentComparison:
    segment: str
    status: str
    ckpt_hash: Optional[str]
    current_hash: Optional[str]
    short_diff_hint: str
    next_action: str


@dataclass(frozen=True, slots=True)
class FingerprintCompareSummary:
    required_segments: tuple[str, ...]
    optional_segments: tuple[str, ...]
    results: tuple[FingerprintSegmentComparison, ...]
    overall_status: str


def regularize_name_collection(
    values: Optional[Sequence[Any]],
    *,
    order_sensitive: bool = False,
) -> tuple[str, ...]:
    names: list[str] = []
    for value in values or ():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        names.append(text)
    if order_sensitive:
        return tuple(names)
    return tuple(sorted(set(names)))


def regularize_manifest_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_.name: regularize_manifest_value(getattr(value, field_.name))
            for field_ in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            text_key = str(key)
            if text_key in normalized:
                raise ValueError(f"regularized mapping key collision after string cast: {text_key!r}")
            normalized[text_key] = regularize_manifest_value(value[key])
        return normalized
    if isinstance(value, (set, frozenset)):
        items = [regularize_manifest_value(item) for item in value]
        return sorted(items, key=canonical_json_dumps)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Size):
        return [int(dim) for dim in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if torch.is_tensor(value):
        raise TypeError("live tensor values are not allowed in canonical fingerprint manifests")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not allowed in canonical fingerprint manifests: {value!r}")
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [regularize_manifest_value(item) for item in value]
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        regularize_manifest_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _require_component_slot(slot: str) -> None:
    if slot not in COMPONENT_SLOT_KIND_MAP:
        raise ValueError(f"unsupported component_slot={slot!r}; inventory is closed in schema v{FINGERPRINT_SCHEMA_VERSION}")


def _require_component_kind(kind: str) -> None:
    if kind not in COMPONENT_KIND_VOCAB:
        raise ValueError(f"unsupported component_kind={kind!r}; inventory is closed in schema v{FINGERPRINT_SCHEMA_VERSION}")


def _normalize_io_field(field_manifest: IOFieldManifest) -> dict[str, Any]:
    return {
        "key": str(field_manifest.key),
        "optional": bool(field_manifest.optional),
        "shape_semantics": list(regularize_name_collection(field_manifest.shape_semantics, order_sensitive=True)),
        "dims": regularize_manifest_value(field_manifest.dims),
        "dtype": str(field_manifest.dtype),
        "accepted_forms": list(regularize_name_collection(field_manifest.accepted_forms, order_sensitive=True)),
    }


def compute_io_signature_hash(io_signature: IOSignatureManifest | Mapping[str, Any]) -> str:
    if isinstance(io_signature, IOSignatureManifest):
        inputs = [_normalize_io_field(field_manifest) for field_manifest in io_signature.inputs]
        outputs = sorted(
            (_normalize_io_field(field_manifest) for field_manifest in io_signature.outputs),
            key=lambda item: str(item["key"]),
        )
        payload = {"inputs": inputs, "outputs": outputs}
    else:
        payload = regularize_manifest_value(io_signature)
    return _canonical_sha256(payload)


def _normalize_component_manifest(component: ComponentManifest) -> dict[str, Any]:
    _require_component_slot(component.component_slot)
    _require_component_kind(component.component_kind)
    return {
        "component_slot": str(component.component_slot),
        "component_kind": str(component.component_kind),
        "enabled": bool(component.enabled),
        "consumes": list(
            regularize_name_collection(
                component.consumes,
                order_sensitive=bool(component.order_sensitive_consumes),
            )
        ),
        "produces": list(
            regularize_name_collection(
                component.produces,
                order_sensitive=bool(component.order_sensitive_produces),
            )
        ),
        "normalized_config": regularize_manifest_value(component.normalized_config),
        "children": list(regularize_name_collection(component.children, order_sensitive=True)),
    }


def _empty_component(slot: str) -> ComponentManifest:
    return ComponentManifest(
        component_slot=slot,
        component_kind=COMPONENT_SLOT_KIND_MAP[slot],
        enabled=False,
    )


def compute_module_graph_hash(module_graph: ModuleGraphManifest | Mapping[str, Any]) -> str:
    if isinstance(module_graph, ModuleGraphManifest):
        by_slot = {component.component_slot: component for component in module_graph.components}
        extra_slots = sorted(set(by_slot) - set(COMPONENT_SLOT_ORDER))
        if extra_slots:
            raise ValueError(f"unsupported component_slot(s) in module_graph manifest: {extra_slots!r}")
        payload = {
            "components": [
                _normalize_component_manifest(by_slot.get(slot, _empty_component(slot)))
                for slot in COMPONENT_SLOT_ORDER
            ]
        }
    else:
        payload = regularize_manifest_value(module_graph)
    return _canonical_sha256(payload)


def _normalize_substep_manifest(substep: BuildSubstepManifest) -> dict[str, Any]:
    return {
        "step_id": str(substep.step_id),
        "consumes": list(regularize_name_collection(substep.consumes)),
        "produces": list(regularize_name_collection(substep.produces)),
        "attached_attrs": list(regularize_name_collection(substep.attached_attrs)),
        "normalized_config": regularize_manifest_value(substep.normalized_config),
    }


def _normalize_build_step_manifest(step: BuildStepManifest) -> dict[str, Any]:
    ordered_substeps = [_normalize_substep_manifest(substep) for substep in step.substeps]
    return {
        "step_id": str(step.step_id),
        "step_order": int(step.step_order),
        "consumes": list(regularize_name_collection(step.consumes)),
        "produces": list(regularize_name_collection(step.produces)),
        "attached_attrs": list(regularize_name_collection(step.attached_attrs)),
        "normalized_config": regularize_manifest_value(step.normalized_config),
        "substeps_folded": ordered_substeps,
    }


def compute_build_order_hash(build_trace: BuildTraceManifest | Mapping[str, Any]) -> str:
    if isinstance(build_trace, BuildTraceManifest):
        ordered_steps = sorted(build_trace.steps, key=lambda step: int(step.step_order))
        payload = {
            "pipeline": str(build_trace.pipeline),
            "steps": [_normalize_build_step_manifest(step) for step in ordered_steps],
        }
    else:
        payload = regularize_manifest_value(build_trace)
    return _canonical_sha256(payload)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def compute_weights_hash(state_dict: Mapping[str, Any]) -> str:
    aggregate = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(f"weights fingerprint expects tensor state_dict values; key={key!r} type={type(value).__name__}")
        meta = {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        aggregate.update(str(key).encode("utf-8"))
        aggregate.update(canonical_json_dumps(meta).encode("utf-8"))
    return aggregate.hexdigest()


def compute_train_policy_hash(train_policy: Mapping[str, Any]) -> str:
    return _canonical_sha256(regularize_manifest_value(dict(train_policy)))


def _module_direct_parameter_signature(module: nn.Module) -> list[dict[str, Any]]:
    signatures = []
    for parameter in module.parameters(recurse=False):
        signatures.append(
            {
                "shape": [int(dim) for dim in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    return sorted(signatures, key=canonical_json_dumps)


def _module_direct_buffer_signature(module: nn.Module) -> list[dict[str, Any]]:
    signatures = []
    for buffer in module.buffers(recurse=False):
        signatures.append(
            {
                "shape": [int(dim) for dim in buffer.shape],
                "dtype": str(buffer.dtype),
            }
        )
    return sorted(signatures, key=canonical_json_dumps)


def _leaf_module_signature(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, nn.Linear):
        return {
            "kind": "linear",
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": bool(module.bias is not None),
        }
    if isinstance(module, nn.LayerNorm):
        return {
            "kind": "layer_norm",
            "normalized_shape": [int(dim) for dim in module.normalized_shape],
            "elementwise_affine": bool(module.elementwise_affine),
        }
    if isinstance(module, nn.Dropout):
        return {"kind": "dropout", "p": float(module.p)}
    if isinstance(module, nn.ReLU):
        return {"kind": "relu", "inplace": bool(module.inplace)}
    if isinstance(module, nn.GELU):
        return {"kind": "gelu", "approximate": str(module.approximate)}
    if isinstance(module, nn.Identity):
        return {"kind": "identity"}
    if isinstance(module, nn.Embedding):
        return {
            "kind": "embedding",
            "num_embeddings": int(module.num_embeddings),
            "embedding_dim": int(module.embedding_dim),
        }
    if isinstance(module, nn.GRUCell):
        return {
            "kind": "gru_cell",
            "input_size": int(module.input_size),
            "hidden_size": int(module.hidden_size),
            "bias": bool(module.bias),
        }
    return {
        "kind": "leaf_module",
        "parameters": _module_direct_parameter_signature(module),
        "buffers": _module_direct_buffer_signature(module),
    }


def _module_structure_signature(module: nn.Module) -> dict[str, Any]:
    children = [_module_structure_signature(child) for child in module.children()]
    if not children:
        return _leaf_module_signature(module)
    if isinstance(module, nn.Sequential):
        kind = "sequential"
        ordered_children = children
    elif isinstance(module, nn.ModuleList):
        kind = "module_list"
        ordered_children = children
    else:
        kind = "composite_module"
        ordered_children = sorted(children, key=canonical_json_dumps)
    return {
        "kind": kind,
        "parameters": _module_direct_parameter_signature(module),
        "buffers": _module_direct_buffer_signature(module),
        "children": ordered_children,
    }


def _parameter_signature(parameter: Optional[torch.Tensor]) -> Mapping[str, Any]:
    if parameter is None:
        return {}
    return {
        "shape": [int(dim) for dim in parameter.shape],
        "dtype": str(parameter.dtype),
        "requires_grad": bool(getattr(parameter, "requires_grad", False)),
    }


def _first_linear_features(module: Optional[nn.Module]) -> Mapping[str, int]:
    if module is None:
        return {}
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            return {
                "first_linear_in_features": int(submodule.in_features),
                "first_linear_out_features": int(submodule.out_features),
            }
    return {}


def _count_tensor_index(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel())
    if value is None:
        return 0
    try:
        return int(len(value))
    except (TypeError, AttributeError):
        return 0


def _slice_signature(slice_value: slice) -> tuple[int, int, int]:
    return (
        int(0 if slice_value.start is None else slice_value.start),
        int(0 if slice_value.stop is None else slice_value.stop),
        int(1 if slice_value.step is None else slice_value.step),
    )


def _append_component(
    components: list[ComponentManifest],
    *,
    slot: str,
    enabled: bool,
    consumes: Sequence[Any] = (),
    produces: Sequence[Any] = (),
    normalized_config: Optional[Mapping[str, Any]] = None,
    children: Sequence[Any] = (),
    order_sensitive_consumes: bool = False,
    order_sensitive_produces: bool = False,
) -> None:
    components.append(
        ComponentManifest(
            component_slot=slot,
            component_kind=COMPONENT_SLOT_KIND_MAP[slot],
            enabled=bool(enabled),
            consumes=regularize_name_collection(consumes, order_sensitive=order_sensitive_consumes),
            produces=regularize_name_collection(produces, order_sensitive=order_sensitive_produces),
            normalized_config=regularize_manifest_value(normalized_config or {}),
            children=regularize_name_collection(children, order_sensitive=True),
            order_sensitive_consumes=bool(order_sensitive_consumes),
            order_sensitive_produces=bool(order_sensitive_produces),
        )
    )


def build_event_motion_model_io_signature_manifest(model: Any) -> IOSignatureManifest:
    contact_dim = int(getattr(model, "contact_dim", 0) or 0)
    hidden_dim = int(getattr(model, "hidden_dim", 0) or 0)
    period_dim = int(getattr(model, "period_dim", 0) or 0)
    leg_joint_count = int(len(getattr(model, "direct_pose_leg_joint_idx", []) or []))
    lambda_joint_count = int(getattr(model, "lambda_fusion_joint_count", 0) or 0)
    so3_joint_count = int(getattr(model, "so3_corr_joint_count", 0) or 0)
    phase_dim = int(getattr(model, "_direct_pose_phase_dim", 0) or 0)
    direct_side_dim = int(getattr(model, "direct_pose_side_channel_dim", 0) or 0)
    direct_pose_enabled = getattr(model, "direct_pose_head", None) is not None
    leg_mode = normalize_direct_pose_leg_mode(getattr(model, "direct_pose_leg_mode", "rot6d_add"))
    gate_mode = normalize_direct_pose_leg_gate_mode(getattr(model, "direct_pose_leg_gate_mode", "none"))
    outputs: list[IOFieldManifest] = [
        IOFieldManifest(
            key="out",
            optional=False,
            shape_semantics=("batch", "query_steps", "out_motion_dim"),
            dims={"out_motion_dim": int(getattr(model, "out_motion_dim", 0) or 0)},
        ),
        IOFieldManifest(
            key="delta",
            optional=False,
            shape_semantics=("batch", "query_steps", "out_motion_dim"),
            dims={"out_motion_dim": int(getattr(model, "out_motion_dim", 0) or 0)},
        ),
        IOFieldManifest(
            key="attn",
            optional=False,
            shape_semantics=("batch", "query_steps", "query_steps"),
        ),
        IOFieldManifest(
            key="h_final",
            optional=False,
            shape_semantics=("batch", "query_steps", "hidden_dim"),
            dims={"hidden_dim": hidden_dim},
        ),
    ]
    if getattr(model, "contact_plan_enable", False) and getattr(model, "contact_plan_head", None) is not None:
        outputs.extend(
            [
                IOFieldManifest(
                    key="contacts_meas",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_dim"),
                    dims={"contact_dim": contact_dim},
                ),
                IOFieldManifest(
                    key="contacts_plan",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_dim"),
                    dims={"contact_dim": contact_dim},
                ),
                IOFieldManifest(
                    key="contacts_plan_logits",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_dim"),
                    dims={"contact_dim": contact_dim},
                ),
                IOFieldManifest(
                    key="plan_z_next",
                    optional=False,
                    shape_semantics=("batch", "contact_plan_hidden"),
                    dims={"contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0)},
                ),
                IOFieldManifest(
                    key="contacts_err",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_dim"),
                    dims={"contact_dim": contact_dim},
                ),
            ]
        )
    if getattr(model, "use_event_clock", False) and getattr(model, "event_clock_gate", None) is not None:
        outputs.extend(
            [
                IOFieldManifest(
                    key="event_clock_lambda_corr",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "scalar_dim"),
                    dims={"scalar_dim": 1},
                ),
                IOFieldManifest(
                    key="event_clock_lambda_logit",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "scalar_dim"),
                    dims={"scalar_dim": 1},
                ),
                IOFieldManifest(
                    key="event_clock_dynamic_prior",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "scalar_dim"),
                    dims={"scalar_dim": 1},
                ),
                IOFieldManifest(
                    key="event_clock_delta_z",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_plan_hidden"),
                    dims={"contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0)},
                ),
                IOFieldManifest(
                    key="event_clock_delta_meas",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "contact_dim"),
                    dims={"contact_dim": contact_dim},
                ),
                IOFieldManifest(
                    key="event_clock_lr_diff",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "scalar_dim"),
                    dims={"scalar_dim": 1},
                ),
            ]
        )
    if direct_pose_enabled:
        outputs.append(
            IOFieldManifest(
                key="out_direct",
                optional=False,
                shape_semantics=("batch", "query_steps", "out_motion_dim"),
                dims={"out_motion_dim": int(getattr(model, "out_motion_dim", 0) or 0)},
            )
        )
    if direct_pose_enabled and leg_mode == "so3" and leg_joint_count > 0:
        outputs.extend(
            [
                IOFieldManifest(
                    key="direct_leg_omega",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "leg_joint_count", "axis_dim"),
                    dims={"leg_joint_count": leg_joint_count, "axis_dim": 3},
                ),
                IOFieldManifest(
                    key="direct_leg_omega_raw",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "leg_joint_count", "axis_dim"),
                    dims={"leg_joint_count": leg_joint_count, "axis_dim": 3},
                ),
            ]
        )
        if gate_mode == "learned":
            outputs.extend(
                [
                    IOFieldManifest(
                        key="direct_leg_gate",
                        optional=False,
                        shape_semantics=("batch", "query_steps", "leg_joint_count"),
                        dims={"leg_joint_count": leg_joint_count},
                    ),
                    IOFieldManifest(
                        key="direct_leg_gate_logits",
                        optional=False,
                        shape_semantics=("batch", "query_steps", "leg_joint_count"),
                        dims={"leg_joint_count": leg_joint_count},
                    ),
                ]
            )
        if gate_mode == "scale":
            outputs.extend(
                [
                    IOFieldManifest(
                        key="direct_leg_scale",
                        optional=False,
                        shape_semantics=("batch", "query_steps", "leg_joint_count"),
                        dims={"leg_joint_count": leg_joint_count},
                    ),
                    IOFieldManifest(
                        key="direct_leg_scale_log",
                        optional=False,
                        shape_semantics=("batch", "query_steps", "leg_joint_count"),
                        dims={"leg_joint_count": leg_joint_count},
                    ),
                    IOFieldManifest(
                        key="direct_leg_scale_log_raw",
                        optional=False,
                        shape_semantics=("batch", "query_steps", "leg_joint_count"),
                        dims={"leg_joint_count": leg_joint_count},
                    ),
                ]
            )
        if bool(getattr(model, "direct_pose_leg_side_sign_gate", False)):
            outputs.append(
                IOFieldManifest(
                    key="direct_leg_side_sign_gate",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "side_count"),
                    dims={"side_count": 2},
                )
            )
    if getattr(model, "lambda_fusion_head", None) is not None:
        logit_dim = 1 if normalize_lambda_fusion_mode(getattr(model, "lambda_fusion_mode", "per_joint")) == "global" else lambda_joint_count
        outputs.extend(
            [
                IOFieldManifest(
                    key="lambda_fusion_logits",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "lambda_logit_dim"),
                    dims={"lambda_logit_dim": int(logit_dim)},
                ),
                IOFieldManifest(
                    key="lambda_fusion",
                    optional=False,
                    shape_semantics=("batch", "query_steps", "lambda_joint_count"),
                    dims={"lambda_joint_count": int(lambda_joint_count)},
                ),
            ]
        )
    if getattr(model, "so3_delta_corrector", None) is not None and so3_joint_count > 0:
        outputs.append(
            IOFieldManifest(
                key="omega_hat",
                optional=False,
                shape_semantics=("batch", "query_steps", "so3_joint_count", "axis_dim"),
                dims={"so3_joint_count": int(so3_joint_count), "axis_dim": 3},
            )
        )
    if getattr(model, "frozen_period_head", None) is not None:
        outputs.append(
            IOFieldManifest(
                key="period_pred",
                optional=False,
                shape_semantics=("batch", "query_steps", "period_dim"),
                dims={"period_dim": period_dim},
            )
        )
    inputs_list = [
        IOFieldManifest(
            key="state",
            optional=False,
            shape_semantics=("batch", "query_steps", "in_state_dim"),
            dims={"in_state_dim": int(getattr(model, "in_state_dim", 0) or 0)},
            accepted_forms=("state_seq",),
        ),
        IOFieldManifest(
            key="cond",
            optional=True,
            shape_semantics=("batch", "query_steps", "cond_dim"),
            dims={"cond_dim": int(getattr(model, "cond_dim", 0) or 0)},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="contacts",
            optional=True,
            shape_semantics=("batch", "query_steps", "contact_dim"),
            dims={"contact_dim": contact_dim},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="angvel",
            optional=True,
            shape_semantics=("batch", "query_steps", "angvel_dim"),
            dims={"angvel_dim": int(getattr(model, "angvel_dim", 0) or 0)},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="pose_history",
            optional=True,
            shape_semantics=("batch", "query_steps", "pose_hist_dim"),
            dims={"pose_hist_dim": int(getattr(model, "pose_hist_dim", 0) or 0)},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="plan_z",
            optional=True,
            shape_semantics=("batch", "contact_plan_hidden"),
            dims={"contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0)},
            accepted_forms=("batch_state", "batch_time_state"),
        ),
        IOFieldManifest(
            key="phase_z",
            optional=True,
            shape_semantics=("batch", "query_steps", "phase_dim"),
            dims={"phase_dim": int(phase_dim)},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="phase_event_age",
            optional=True,
            shape_semantics=("batch", "query_steps", "contact_dim"),
            dims={"contact_dim": contact_dim},
            accepted_forms=("broadcastable_seq",),
        ),
        IOFieldManifest(
            key="meas_logits_prev",
            optional=True,
            shape_semantics=("batch", "contact_dim"),
            dims={"contact_dim": contact_dim},
            accepted_forms=("contact_dim", "batch_contact_dim", "batch_single_step_contact_dim"),
        ),
        IOFieldManifest(
            key="time_index",
            optional=True,
            shape_semantics=("batch", "query_steps", "scalar_dim"),
            dims={"scalar_dim": 1},
            accepted_forms=("scalar", "batch_scalar", "batch_time_scalar"),
        ),
        IOFieldManifest(
            key="rollout_step",
            optional=True,
            shape_semantics=("batch", "query_steps", "scalar_dim"),
            dims={"scalar_dim": 1},
            accepted_forms=("scalar", "batch_scalar", "batch_time_scalar", "batch_time_feature"),
        ),
    ]
    if direct_side_dim > 0:
        inputs_list.append(
            IOFieldManifest(
                key="direct_pose_side_channel",
                optional=True,
                shape_semantics=("batch", "query_steps", "direct_pose_side_channel_dim"),
                dims={"direct_pose_side_channel_dim": int(direct_side_dim)},
                accepted_forms=("broadcastable_seq",),
            )
        )
    return IOSignatureManifest(inputs=tuple(inputs_list), outputs=tuple(outputs))


def build_event_motion_model_module_graph_manifest(model: Any) -> ModuleGraphManifest:
    components: list[ComponentManifest] = []
    contact_plan_inject = str(getattr(model, "contact_plan_inject", "none") or "none").strip().lower()
    shared_encoder_consumes = ["state", "cond"]
    if contact_plan_inject == "contacts":
        shared_encoder_consumes.append("contacts_plan")
    elif contact_plan_inject == "plan_z":
        shared_encoder_consumes.append("plan_z")
    shared_encoder = getattr(model, "shared_encoder", None)
    _append_component(
        components,
        slot="shared_encoder",
        enabled=shared_encoder is not None,
        consumes=shared_encoder_consumes,
        produces=("h",),
        normalized_config={
            "hidden_dim": int(getattr(model, "hidden_dim", 0) or 0),
            "encoder_residual": bool(getattr(model, "_encoder_residual", False)),
            "contact_plan_inject": contact_plan_inject,
            "structure": _module_structure_signature(shared_encoder) if shared_encoder is not None else {},
            **_first_linear_features(shared_encoder),
        },
    )
    residual_proj = getattr(model, "residual_proj", None)
    residual_proj_config: dict[str, Any] = {
        "identity": bool(isinstance(residual_proj, nn.Identity)),
        "structure": _module_structure_signature(residual_proj) if isinstance(residual_proj, nn.Module) else {},
    }
    if isinstance(residual_proj, nn.Linear):
        residual_proj_config["in_features"] = int(residual_proj.in_features)
        residual_proj_config["out_features"] = int(residual_proj.out_features)
    _append_component(
        components,
        slot="residual_proj",
        enabled=residual_proj is not None,
        consumes=shared_encoder_consumes,
        produces=("h_temporal_residual",),
        normalized_config=residual_proj_config,
    )
    pasa_children = {
        "query_proj": getattr(model, "_pasa_q", None),
        "key_proj": getattr(model, "_pasa_k", None),
        "value_proj": getattr(model, "_pasa_v", None),
        "out_proj": getattr(model, "_pasa_o", None),
        "query_norm": getattr(model, "_pasa_lnq", None),
        "cond_film": getattr(model, "_pasa_film", None),
        "coupling_norm": getattr(model, "coupling_norm", None),
    }
    _append_component(
        components,
        slot="pasa_attention_block",
        enabled=all(isinstance(module, nn.Module) for module in pasa_children.values()),
        consumes=("h_temporal", "cond"),
        produces=("attn", "h_final"),
        normalized_config={
            "pasa_heads": int(getattr(model, "_pasa_heads", 0) or 0),
            "pasa_dhead": int(getattr(model, "_pasa_dhead", 0) or 0),
            "children": {
                name: _module_structure_signature(module) if isinstance(module, nn.Module) else {}
                for name, module in pasa_children.items()
            },
        },
        children=tuple(pasa_children.keys()),
    )
    motion_head = getattr(model, "motion_head", None)
    _append_component(
        components,
        slot="motion_head",
        enabled=motion_head is not None,
        consumes=("h_final",),
        produces=("out", "delta"),
        normalized_config={
            "out_motion_dim": int(getattr(model, "out_motion_dim", 0) or 0),
            "structure": _module_structure_signature(motion_head) if motion_head is not None else {},
        },
    )
    period_encoder = getattr(model, "period_encoder", None)
    _append_component(
        components,
        slot="period_encoder",
        enabled=period_encoder is not None and int(getattr(model, "period_dim", 0) or 0) > 0,
        consumes=("soft_period",),
        produces=("period_emb",),
        normalized_config={
            "period_dim": int(getattr(model, "period_dim", 0) or 0),
            "hidden_dim": int(getattr(model, "hidden_dim", 0) or 0),
            "structure": _module_structure_signature(period_encoder) if period_encoder is not None else {},
        },
    )
    bone_adapters = list(getattr(model, "_bone_adapters", []) or [])
    bone_adapter_slices = list(getattr(model, "_bone_adapter_slices", []) or [])
    bone_adapter_names = [str(name) for name in (getattr(model, "_bone_adapter_names", []) or [])]
    _append_component(
        components,
        slot="bone_residual_adapter_bank",
        enabled=bool(bone_adapters) and bool(bone_adapter_slices) and bool(bone_adapter_names),
        consumes=("h_final", "selected_output_slices"),
        produces=("bone_residuals",),
        normalized_config={
            "adapter_count": int(len(bone_adapters)),
            "adapter_names": bone_adapter_names,
            "adapter_slices": [_slice_signature(slice_value) for slice_value in bone_adapter_slices],
            "adapter_structures": [_module_structure_signature(adapter) for adapter in bone_adapters],
        },
    )
    contact_plan_cell = getattr(model, "contact_plan_cell", None)
    _append_component(
        components,
        slot="contact_plan_cell",
        enabled=bool(getattr(model, "contact_plan_enable", False)) and contact_plan_cell is not None,
        consumes=("cond", "plan_z_prev"),
        produces=("plan_z_raw",),
        normalized_config={
            "contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0),
            "inject_mode": contact_plan_inject,
            "structure": _module_structure_signature(contact_plan_cell) if contact_plan_cell is not None else {},
        },
    )
    contact_plan_init_z = getattr(model, "contact_plan_init_z", None)
    _append_component(
        components,
        slot="contact_plan_init_z",
        enabled=bool(getattr(model, "contact_plan_enable", False)) and torch.is_tensor(contact_plan_init_z),
        produces=("plan_z_init",),
        normalized_config=_parameter_signature(contact_plan_init_z),
    )
    init_mode = normalize_contact_plan_init_mode(getattr(model, "contact_plan_init_mode", "learnable"))
    init_head = getattr(model, "contact_plan_init_head", None)
    init_head_consumes: list[str] = []
    if int(getattr(model, "contact_dim", 0) or 0) > 0:
        init_head_consumes.append("contacts_meas")
    if int(getattr(model, "angvel_dim", 0) or 0) > 0:
        init_head_consumes.append("angvel")
    if int(getattr(model, "pose_hist_dim", 0) or 0) > 0:
        init_head_consumes.append("pose_history")
    _append_component(
        components,
        slot="contact_plan_init_head",
        enabled=bool(getattr(model, "contact_plan_enable", False))
        and init_mode in ("obs", "learnable+obs")
        and int(getattr(model, "_contact_plan_init_obs_dim", 0) or 0) > 0
        and init_head is not None,
        consumes=init_head_consumes,
        produces=("plan_z_init_delta",),
        normalized_config={
            "init_mode": init_mode,
            "obs_dim": int(getattr(model, "_contact_plan_init_obs_dim", 0) or 0),
            "hidden_dim": int(getattr(model, "contact_plan_init_hidden", 0) or 0),
            "dropout": float(getattr(model, "_contact_plan_init_dropout", 0.0) or 0.0),
            "structure": _module_structure_signature(init_head) if init_head is not None else {},
            **_first_linear_features(init_head),
        },
    )
    contact_plan_head = getattr(model, "contact_plan_head", None)
    _append_component(
        components,
        slot="contact_plan_head",
        enabled=bool(getattr(model, "contact_plan_enable", False)) and contact_plan_head is not None,
        consumes=("plan_z",),
        produces=("contacts_plan_logits", "contacts_plan"),
        normalized_config={
            "contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0),
            "contact_dim": int(getattr(model, "contact_dim", 0) or 0),
            "structure": _module_structure_signature(contact_plan_head) if contact_plan_head is not None else {},
        },
    )
    time_head = getattr(model, "contact_plan_time_head", None)
    _append_component(
        components,
        slot="contact_plan_time_head",
        enabled=time_head is not None and int(getattr(model, "contact_plan_time_pe_dim", 0) or 0) > 0,
        consumes=("time_pe",),
        produces=("contact_plan_time_bias",),
        normalized_config={
            "time_pe_dim": int(getattr(model, "contact_plan_time_pe_dim", 0) or 0),
            "contact_dim": int(getattr(model, "contact_dim", 0) or 0),
            "structure": _module_structure_signature(time_head) if time_head is not None else {},
        },
    )
    event_clock_gate = getattr(model, "event_clock_gate", None)
    _append_component(
        components,
        slot="event_clock_gate",
        enabled=bool(getattr(model, "use_event_clock", False)) and event_clock_gate is not None,
        consumes=("contacts_err", "event_clock_delta_meas", "event_clock_lr_diff", "period_feat"),
        produces=("lambda_corr", "lambda_logit", "dynamic_prior"),
        normalized_config={
            "contact_dim": int(getattr(model, "contact_dim", 0) or 0),
            "period_feat_dim": int(getattr(model, "period_dim", 0) or 0),
            "hidden_dim": int(getattr(model, "event_clock_gate_hidden_dim", 0) or 0),
            "structure": _module_structure_signature(event_clock_gate) if event_clock_gate is not None else {},
        },
    )
    event_clock_corrector = getattr(model, "event_clock_corrector", None)
    _append_component(
        components,
        slot="event_clock_corrector",
        enabled=bool(getattr(model, "use_event_clock", False)) and event_clock_corrector is not None,
        consumes=("plan_z_raw", "contacts_meas", "event_clock_delta_meas", "contacts_err", "lambda_corr", "period_feat"),
        produces=("plan_z", "delta_z"),
        normalized_config={
            "plan_z_dim": int(getattr(model, "contact_plan_hidden", 0) or 0),
            "contact_dim": int(getattr(model, "contact_dim", 0) or 0),
            "period_feat_dim": int(getattr(model, "period_dim", 0) or 0),
            "hidden_dim": int(getattr(model, "event_clock_hidden_dim", 0) or 0),
            "max_delta": float(getattr(model, "event_clock_max_delta", 0.0) or 0.0),
            "structure": _module_structure_signature(event_clock_corrector) if event_clock_corrector is not None else {},
        },
    )
    direct_pose_head = getattr(model, "direct_pose_head", None)
    direct_feat_source = normalize_direct_pose_feat_source(getattr(model, "direct_pose_feat_source", "cond"))
    direct_phase_mode = normalize_direct_pose_phase_z_mode(getattr(model, "direct_pose_phase_z_mode", "concat"))
    direct_meas_mode = str(getattr(model, "direct_pose_meas_mode", "concat") or "concat").strip().lower()
    direct_use_phase = bool(getattr(model, "direct_pose_use_phase_z", False)) and int(getattr(model, "_direct_pose_phase_dim", 0) or 0) > 0
    direct_side_dim = int(getattr(model, "direct_pose_side_channel_dim", 0) or 0)
    direct_consumes: list[str] = []
    if direct_feat_source == "cond":
        direct_consumes.append("cond")
    elif direct_feat_source == "hidden":
        direct_consumes.append("h_final")
    elif direct_feat_source == "hidden_pre":
        direct_consumes.append("h_temporal")
    elif direct_feat_source == "cond+hidden":
        direct_consumes.extend(("cond", "h_final"))
    elif direct_feat_source == "cond+hidden_pre":
        direct_consumes.extend(("cond", "h_temporal"))
    if int(getattr(model, "direct_pose_time_pe_dim", 0) or 0) > 0:
        direct_consumes.append("time_pe")
    if direct_side_dim > 0:
        direct_consumes.append("direct_pose_side_channel")
    if direct_phase_mode == "replace_contacts" and direct_use_phase:
        direct_consumes.append("phase_z")
    else:
        direct_consumes.append("contacts_plan")
        if direct_meas_mode in ("concat", "mode_select"):
            direct_consumes.append("contacts_meas")
        if direct_use_phase:
            direct_consumes.append("phase_z")
    if bool(getattr(model, "direct_pose_split_enable", False)):
        direct_produces = ("direct_pose_trunk_feature",)
    elif direct_meas_mode == "mode_select":
        direct_produces = ("mode_select_logits",)
    else:
        direct_produces = ("out_direct",)
    direct_pose_normalized_config = {
        "hidden_dim": int(getattr(model, "direct_pose_hidden", 0) or 0),
        "feat_source": direct_feat_source,
        "meas_mode": direct_meas_mode,
        "time_pe_dim": int(getattr(model, "direct_pose_time_pe_dim", 0) or 0),
        "use_phase_z": direct_use_phase,
        "phase_z_mode": direct_phase_mode,
        "split_enable": bool(getattr(model, "direct_pose_split_enable", False)),
        "arm_split_enable": bool(getattr(model, "direct_pose_arm_split_enable", False)),
        "nonleg_proj_dim": int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0),
        "leg_side_routing": bool(getattr(model, "direct_pose_leg_side_routing", False)),
        "leg_mode": normalize_direct_pose_leg_mode(getattr(model, "direct_pose_leg_mode", "rot6d_add")),
        "structure": _module_structure_signature(direct_pose_head) if direct_pose_head is not None else {},
        **_first_linear_features(direct_pose_head),
    }
    if direct_side_dim > 0:
        direct_pose_normalized_config["side_channel_dim"] = int(direct_side_dim)
    _append_component(
        components,
        slot="direct_pose_head",
        enabled=bool(getattr(model, "contact_plan_enable", False)) and bool(getattr(model, "direct_pose_enable", False)) and direct_pose_head is not None,
        consumes=direct_consumes,
        produces=direct_produces,
        normalized_config=direct_pose_normalized_config,
        order_sensitive_consumes=True,
    )
    direct_pose_leg_terminal = getattr(model, "direct_pose_leg_terminal", None)
    _append_component(
        components,
        slot="direct_pose_leg_terminal",
        enabled=bool(getattr(model, "direct_pose_split_enable", False)) and direct_pose_leg_terminal is not None,
        consumes=("direct_pose_trunk_feature",),
        produces=("leg_output_slice",),
        normalized_config={
            "output_index_count": _count_tensor_index(getattr(model, "direct_pose_leg_out_idx", None)),
            "structure": _module_structure_signature(direct_pose_leg_terminal) if direct_pose_leg_terminal is not None else {},
        },
    )
    direct_pose_out_nonleg = getattr(model, "direct_pose_out_nonleg", None)
    _append_component(
        components,
        slot="direct_pose_out_nonleg",
        enabled=bool(getattr(model, "direct_pose_split_enable", False))
        and (not bool(getattr(model, "direct_pose_arm_split_enable", False)))
        and direct_pose_out_nonleg is not None,
        consumes=("projected_nonleg_feature",)
        if getattr(model, "direct_pose_nonleg_proj", None) is not None
        else ("direct_pose_trunk_feature",),
        produces=("nonleg_output_slice",),
        normalized_config={
            "output_index_count": _count_tensor_index(getattr(model, "direct_pose_nonleg_out_idx", None)),
            "structure": _module_structure_signature(direct_pose_out_nonleg) if direct_pose_out_nonleg is not None else {},
        },
    )
    direct_pose_nonleg_proj = getattr(model, "direct_pose_nonleg_proj", None)
    _append_component(
        components,
        slot="direct_pose_nonleg_proj",
        enabled=bool(getattr(model, "direct_pose_split_enable", False))
        and (not bool(getattr(model, "direct_pose_arm_split_enable", False)))
        and int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0) > 0
        and direct_pose_nonleg_proj is not None,
        consumes=("direct_pose_trunk_feature",),
        produces=("projected_nonleg_feature",),
        normalized_config={
            "proj_dim": int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0),
            "structure": _module_structure_signature(direct_pose_nonleg_proj) if direct_pose_nonleg_proj is not None else {},
        },
    )
    direct_pose_out_arm = getattr(model, "direct_pose_out_arm", None)
    _append_component(
        components,
        slot="direct_pose_out_arm",
        enabled=bool(getattr(model, "direct_pose_arm_split_enable", False)) and direct_pose_out_arm is not None,
        consumes=("projected_arm_feature",) if getattr(model, "direct_pose_arm_proj", None) is not None else ("direct_pose_trunk_feature",),
        produces=("arm_output_slice",),
        normalized_config={
            "output_index_count": _count_tensor_index(getattr(model, "direct_pose_arm_out_idx", None)),
            "structure": _module_structure_signature(direct_pose_out_arm) if direct_pose_out_arm is not None else {},
        },
    )
    direct_pose_out_else = getattr(model, "direct_pose_out_else", None)
    _append_component(
        components,
        slot="direct_pose_out_else",
        enabled=bool(getattr(model, "direct_pose_arm_split_enable", False)) and direct_pose_out_else is not None,
        consumes=("projected_else_feature",)
        if getattr(model, "direct_pose_else_proj", None) is not None
        else ("direct_pose_trunk_feature",),
        produces=("else_output_slice",),
        normalized_config={
            "output_index_count": _count_tensor_index(getattr(model, "direct_pose_else_out_idx", None)),
            "structure": _module_structure_signature(direct_pose_out_else) if direct_pose_out_else is not None else {},
        },
    )
    direct_pose_arm_proj = getattr(model, "direct_pose_arm_proj", None)
    _append_component(
        components,
        slot="direct_pose_arm_proj",
        enabled=bool(getattr(model, "direct_pose_arm_split_enable", False))
        and int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0) > 0
        and direct_pose_arm_proj is not None,
        consumes=("direct_pose_trunk_feature",),
        produces=("projected_arm_feature",),
        normalized_config={
            "proj_dim": int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0),
            "structure": _module_structure_signature(direct_pose_arm_proj) if direct_pose_arm_proj is not None else {},
        },
    )
    direct_pose_else_proj = getattr(model, "direct_pose_else_proj", None)
    _append_component(
        components,
        slot="direct_pose_else_proj",
        enabled=bool(getattr(model, "direct_pose_arm_split_enable", False))
        and int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0) > 0
        and direct_pose_else_proj is not None,
        consumes=("direct_pose_trunk_feature",),
        produces=("projected_else_feature",),
        normalized_config={
            "proj_dim": int(getattr(model, "direct_pose_nonleg_proj_dim", 0) or 0),
            "structure": _module_structure_signature(direct_pose_else_proj) if direct_pose_else_proj is not None else {},
        },
    )
    leg_joint_count = int(len(getattr(model, "direct_pose_leg_joint_idx", []) or []))
    leg_mode = normalize_direct_pose_leg_mode(getattr(model, "direct_pose_leg_mode", "rot6d_add"))
    direct_pose_leg_head = getattr(model, "direct_pose_leg_head", None)
    side_routing = bool(getattr(model, "direct_pose_leg_side_routing", False))
    _append_component(
        components,
        slot="direct_pose_leg_head",
        enabled=bool(getattr(model, "direct_pose_leg_enable", False)) and leg_joint_count > 0 and direct_pose_leg_head is not None,
        consumes=("direct_feature_stream",),
        produces=("leg_residual",),
        normalized_config={
            "leg_joint_count": leg_joint_count,
            "leg_mode": leg_mode,
            "side_routing_enabled": side_routing,
            "stopgrad_main": bool(getattr(model, "direct_pose_leg_stopgrad_main", False)),
            "detach_feat": bool(getattr(model, "direct_pose_leg_detach_feat", False)),
            "max_deg": float(getattr(model, "direct_pose_leg_max_deg", 0.0) or 0.0),
            "structure": _module_structure_signature(direct_pose_leg_head) if direct_pose_leg_head is not None else {},
        },
    )
    gate_mode = normalize_direct_pose_leg_gate_mode(getattr(model, "direct_pose_leg_gate_mode", "none"))
    gate_power = normalize_direct_pose_leg_gate_power(getattr(model, "direct_pose_leg_gate_power", 1.0))
    direct_pose_leg_gate_head = getattr(model, "direct_pose_leg_gate_head", None)
    _append_component(
        components,
        slot="direct_pose_leg_gate_head",
        enabled=bool(getattr(model, "direct_pose_leg_enable", False))
        and gate_mode in {"learned", "scale"}
        and direct_pose_leg_gate_head is not None,
        consumes=("direct_feature_stream",),
        produces=("leg_gate",),
        normalized_config={
            "gate_mode": gate_mode,
            "gate_power": gate_power,
            "scale_log_clip": float(getattr(model, "direct_pose_leg_scale_log_clip", 0.0) or 0.0),
            "scale_clamp_k": float(getattr(model, "direct_pose_leg_scale_clamp_k", 0.0) or 0.0),
            "leg_joint_count": leg_joint_count,
            "structure": _module_structure_signature(direct_pose_leg_gate_head) if direct_pose_leg_gate_head is not None else {},
        },
    )
    side_k = int(getattr(model, "direct_pose_leg_side_k", 0) or 0)
    direct_pose_leg_head_shared = getattr(model, "direct_pose_leg_head_shared", None)
    _append_component(
        components,
        slot="direct_pose_leg_head_shared",
        enabled=side_routing and side_k > 0 and direct_pose_leg_head_shared is not None,
        consumes=("side_routed_leg_feature_stream",),
        produces=("side_shared_leg_residual",),
        normalized_config={
            "side_k": side_k,
            "side_embed_dim": int(getattr(model, "direct_pose_leg_side_embed_dim", 0) or 0),
            "plan_other": bool(getattr(model, "direct_pose_leg_side_plan_other", False)),
            "phase_other": bool(getattr(model, "direct_pose_leg_side_phase_other", False)),
            "phase_rel": bool(getattr(model, "direct_pose_leg_side_phase_rel", False)),
            "side_cue": str(getattr(model, "direct_pose_leg_side_cue", "none") or "none"),
            "side_cue_tau": float(getattr(model, "direct_pose_leg_side_cue_tau", 0.0) or 0.0),
            "rank1": bool(getattr(model, "direct_pose_leg_side_rank1", False)),
            "structure": _module_structure_signature(direct_pose_leg_head_shared) if direct_pose_leg_head_shared is not None else {},
        },
    )
    direct_pose_leg_gate_head_shared = getattr(model, "direct_pose_leg_gate_head_shared", None)
    _append_component(
        components,
        slot="direct_pose_leg_gate_head_shared",
        enabled=side_routing and gate_mode in {"learned", "scale"} and direct_pose_leg_gate_head_shared is not None,
        consumes=("side_routed_leg_feature_stream",),
        produces=("side_shared_leg_gate",),
        normalized_config={
            "gate_mode": gate_mode,
            "gate_power": gate_power,
            "side_k": side_k,
            "structure": _module_structure_signature(direct_pose_leg_gate_head_shared)
            if direct_pose_leg_gate_head_shared is not None
            else {},
        },
    )
    direct_pose_leg_side_embed = getattr(model, "direct_pose_leg_side_embed", None)
    _append_component(
        components,
        slot="direct_pose_leg_side_embed",
        enabled=side_routing and int(getattr(model, "direct_pose_leg_side_embed_dim", 0) or 0) > 0 and direct_pose_leg_side_embed is not None,
        consumes=("side_id",),
        produces=("side_embedding",),
        normalized_config={
            "embedding_dim": int(getattr(model, "direct_pose_leg_side_embed_dim", 0) or 0),
            "structure": _module_structure_signature(direct_pose_leg_side_embed) if direct_pose_leg_side_embed is not None else {},
        },
    )
    direct_pose_leg_side_sign_gate_head = getattr(model, "direct_pose_leg_side_sign_gate_head", None)
    _append_component(
        components,
        slot="direct_pose_leg_side_sign_gate_head",
        enabled=side_routing and bool(getattr(model, "direct_pose_leg_side_sign_gate", False)) and direct_pose_leg_side_sign_gate_head is not None,
        consumes=("side_routed_leg_feature_stream",),
        produces=("side_shared_sign_gate",),
        normalized_config={
            "side_k": side_k,
            "structure": _module_structure_signature(direct_pose_leg_side_sign_gate_head)
            if direct_pose_leg_side_sign_gate_head is not None
            else {},
        },
    )
    lambda_fusion_head = getattr(model, "lambda_fusion_head", None)
    _append_component(
        components,
        slot="lambda_fusion_head",
        enabled=bool(getattr(model, "lambda_fusion_enable", False)) and lambda_fusion_head is not None,
        consumes=("h_final", "contacts_plan", "rollout_step"),
        produces=("lambda_fusion_logits", "lambda_fusion"),
        normalized_config={
            "mode": normalize_lambda_fusion_mode(getattr(model, "lambda_fusion_mode", "per_joint")),
            "joint_count": int(getattr(model, "lambda_fusion_joint_count", 0) or 0),
            "hidden_dim": int(getattr(model, "lambda_fusion_hidden", 0) or 0),
            "detach_err": bool(getattr(model, "lambda_fusion_detach_err", False)),
            "use_rollout_step": bool(getattr(model, "lambda_fusion_use_rollout_step", False)),
            "structure": _module_structure_signature(lambda_fusion_head) if lambda_fusion_head is not None else {},
        },
    )
    so3_delta_corrector = getattr(model, "so3_delta_corrector", None)
    _append_component(
        components,
        slot="so3_delta_corrector",
        enabled=int(getattr(model, "so3_corr_joint_count", 0) or 0) > 0 and so3_delta_corrector is not None,
        consumes=("h_final", "contacts_plan"),
        produces=("omega_hat",),
        normalized_config={
            "joint_count": int(getattr(model, "so3_corr_joint_count", 0) or 0),
            "structure": _module_structure_signature(so3_delta_corrector) if so3_delta_corrector is not None else {},
        },
    )
    so3_corr_gate_logit = getattr(model, "so3_corr_gate_logit", None)
    _append_component(
        components,
        slot="so3_corr_gate_logit",
        enabled=int(getattr(model, "so3_corr_joint_count", 0) or 0) > 0 and torch.is_tensor(so3_corr_gate_logit),
        produces=("so3_corr_gate",),
        normalized_config=_parameter_signature(so3_corr_gate_logit),
    )
    adaptive_history_module = getattr(model, "adaptive_history_module", None)
    _append_component(
        components,
        slot="adaptive_history_module",
        enabled=adaptive_history_module is not None,
        consumes=("pose_history",),
        produces=("pose_history_feature",),
        normalized_config={
            "pose_hist_len": int(getattr(model, "pose_hist_len", 0) or 0),
            "structure": _module_structure_signature(adaptive_history_module)
            if adaptive_history_module is not None
            else {},
        },
    )
    frozen_encoder = getattr(model, "frozen_encoder", None)
    _append_component(
        components,
        slot="frozen_encoder",
        enabled=frozen_encoder is not None,
        consumes=("encoder_input",),
        produces=("frozen_hidden_summary",),
        normalized_config={
            "structure": _module_structure_signature(frozen_encoder) if frozen_encoder is not None else {},
        },
    )
    frozen_period_head = getattr(model, "frozen_period_head", None)
    _append_component(
        components,
        slot="frozen_period_head",
        enabled=frozen_period_head is not None,
        consumes=("frozen_hidden_summary",),
        produces=("soft_period",),
        normalized_config={
            "structure": _module_structure_signature(frozen_period_head) if frozen_period_head is not None else {},
        },
    )
    frozen_contact_head = getattr(model, "frozen_contact_head", None)
    _append_component(
        components,
        slot="frozen_contact_head",
        enabled=frozen_contact_head is not None,
        consumes=("frozen_hidden_summary",),
        produces=("external_contact_hint",),
        normalized_config={
            "structure": _module_structure_signature(frozen_contact_head) if frozen_contact_head is not None else {},
        },
    )
    components.sort(key=lambda component: _COMPONENT_SLOT_INDEX[component.component_slot])
    return ModuleGraphManifest(components=tuple(components))


def build_basetrain_build_trace_manifest() -> BuildTraceManifest:
    steps = (
        BuildStepManifest(
            step_id="basetrain.parse_context",
            step_order=1,
            consumes=("argv", "config_defaults", "cli_overrides"),
            produces=("TrainEntryContext",),
            attached_attrs=("args", "train_paths", "device", "norm_template_path"),
            notes=("volatile_excluded: out_dir, run_name",),
        ),
        BuildStepManifest(
            step_id="basetrain.build_dataset_loader",
            step_order=2,
            consumes=("TrainEntryContext",),
            produces=("TrainDataArtifacts",),
            attached_attrs=("ds_train", "train_loader", "dx", "dy", "dc"),
            notes=("volatile_excluded: dataloader worker runtime noise, loader object id",),
        ),
        BuildStepManifest(
            step_id="basetrain.instantiate_model",
            step_order=3,
            consumes=("TrainEntryContext", "TrainDataArtifacts"),
            produces=("TrainModelArtifacts",),
            attached_attrs=("model", "direct_pose_options", "history_export_dim"),
            notes=("volatile_excluded: device placement detail, raw out_dir paths",),
        ),
        BuildStepManifest(
            step_id="basetrain.prepare_model_runtime",
            step_order=4,
            consumes=("TrainEntryContext", "TrainDataArtifacts", "TrainModelArtifacts"),
            produces=("TrainModelArtifacts",),
            attached_attrs=("adaptive_history_runtime", "external_motion_bundle", "resume_weights", "_pasa_fps"),
            notes=("volatile_excluded: bundle path, checkpoint path, RNG advancement side effects",),
        ),
        BuildStepManifest(
            step_id="basetrain.build_loss_and_trainer",
            step_order=5,
            consumes=("TrainEntryContext", "TrainDataArtifacts", "TrainModelArtifacts"),
            produces=("TrainBuildArtifacts",),
            attached_attrs=("loss_fn", "trainer", "resolved_config_snapshot"),
            notes=("volatile_excluded: resolved_config dump path",),
        ),
        BuildStepManifest(
            step_id="basetrain.attach_entry_runtime",
            step_order=6,
            consumes=("TrainEntryContext", "TrainDataArtifacts", "TrainBuildArtifacts"),
            produces=("Trainer",),
            attached_attrs=(
                "dataset_runtime",
                "shared_trainer_runtime",
                "trainbase_contacts_pretrain_runtime",
                "history_schedule_attrs",
                "loss_runtime_mirror",
            ),
            substeps=(
                BuildSubstepManifest(
                    step_id="basetrain.attach_dataset_runtime",
                    consumes=("trainer", "ds_train", "bundle_json_path"),
                    produces=("DatasetRuntimeArtifacts",),
                    attached_attrs=("dataset_runtime", "yaw_bundle_metadata"),
                ),
                BuildSubstepManifest(
                    step_id="basetrain.resolve_runtime_cfg",
                    consumes=("args", "trainer", "DatasetRuntimeArtifacts", "path_meta_inputs"),
                    produces=("TrainerRuntimeConfig",),
                    attached_attrs=("shared_runtime_cfg", "contacts_pretrain_runtime", "history_schedule_cfg"),
                ),
                BuildSubstepManifest(
                    step_id="basetrain.apply_runtime_cfg",
                    consumes=("trainer", "TrainerRuntimeConfig"),
                    produces=("Trainer",),
                    attached_attrs=(
                        "pose_hist_runtime_attrs",
                        "yaw_runtime_attrs",
                        "trainbase_contacts_pretrain_attrs",
                        "tf_runtime_attrs",
                        "freerun_stage_schedule",
                        "diagnostics_attrs",
                    ),
                ),
                BuildSubstepManifest(
                    step_id="basetrain.sync_loss_runtime",
                    consumes=("loss_fn", "trainer"),
                    produces=("loss_fn",),
                    attached_attrs=("mu_y", "std_y", "bundle_meta"),
                ),
            ),
            notes=("volatile_excluded: out_dir, run_name, bundle_json_path, full config blob, trainer object id",),
        ),
    )
    return BuildTraceManifest(pipeline="basetrain", steps=steps)


def build_posttrain_build_trace_manifest(
    *,
    structural_flags: Optional[Mapping[str, Any]] = None,
    train_mode: Optional[str] = None,
    trainable_slots: Optional[Mapping[str, bool]] = None,
) -> BuildTraceManifest:
    structural_flags = regularize_manifest_value(structural_flags or {})
    trainable_slot_map = {
        str(key): bool(value)
        for key, value in sorted((trainable_slots or {}).items(), key=lambda item: str(item[0]))
    }
    steps = (
        BuildStepManifest(
            step_id="posttrain.parse_cfg_and_seed",
            step_order=1,
            consumes=("config_json_payload", "cli_overrides"),
            produces=("PostTrainConfig", "train_mode"),
            attached_attrs=("typed_cfg", "seed_setup", "device_intent"),
            notes=("volatile_excluded: output dir, run directory, RNG state values",),
        ),
        BuildStepManifest(
            step_id="posttrain.build_dataset_loader",
            step_order=2,
            consumes=("PostTrainConfig",),
            produces=("norm_spec", "MotionEventDataset", "batch_iter"),
            attached_attrs=("dataset_runtime_primitives",),
            notes=("volatile_excluded: iterator object id, batch iterator state",),
        ),
        BuildStepManifest(
            step_id="posttrain.resolve_ckpt_build_state",
            step_order=3,
            consumes=("cfg", "ds", "checkpoint_payload"),
            produces=("PostTrainModelBuildState",),
            attached_attrs=("structural_enables", "head_branch_dims"),
            normalized_config=structural_flags,
            notes=("volatile_excluded: checkpoint path, raw dict insertion order",),
        ),
        BuildStepManifest(
            step_id="posttrain.instantiate_model",
            step_order=4,
            consumes=("cfg", "ds", "device", "PostTrainModelBuildState"),
            produces=("EventMotionModel",),
            attached_attrs=("model",),
            notes=("volatile_excluded: device id, storage ptr",),
        ),
        BuildStepManifest(
            step_id="posttrain.load_ckpt_into_model",
            step_order=5,
            consumes=("cfg", "model", "PostTrainModelBuildState"),
            produces=("EventMotionModel",),
            attached_attrs=("encoder_bundle_attach", "compat_transform", "state_load"),
            notes=("volatile_excluded: bundle path, checkpoint path",),
        ),
        BuildStepManifest(
            step_id="posttrain.verify_rollout_contracts",
            step_order=6,
            consumes=("cfg", "model"),
            produces=("verified_rollout_contracts",),
            attached_attrs=("rollout_contract_gate",),
            notes=("volatile_excluded: raw file path spelling",),
        ),
        BuildStepManifest(
            step_id="posttrain.build_loss_and_trainer",
            step_order=7,
            consumes=("cfg", "ds", "model"),
            produces=("Trainer",),
            attached_attrs=("loss_fn", "trainer", "bone_name_sync"),
        ),
        BuildStepManifest(
            step_id="posttrain.attach_trainer_runtime",
            step_order=8,
            consumes=("cfg", "ds", "trainer", "norm_spec"),
            produces=("Trainer",),
            attached_attrs=(
                "dataset_runtime",
                "shared_trainer_runtime",
                "loss_runtime_mirror",
                "posttrain_local_overlay",
            ),
            substeps=(
                BuildSubstepManifest(
                    step_id="posttrain.attach_dataset_runtime",
                    consumes=("trainer", "ds", "bundle_json", "norm_spec"),
                    produces=("DatasetRuntimeArtifacts",),
                    attached_attrs=("dataset_runtime",),
                ),
                BuildSubstepManifest(
                    step_id="posttrain.apply_shared_runtime",
                    consumes=("DatasetRuntimeArtifacts", "cfg_meta"),
                    produces=("Trainer",),
                    attached_attrs=("pose_hist_runtime_attrs", "yaw_runtime_attrs", "output_meta_attrs"),
                ),
                BuildSubstepManifest(
                    step_id="posttrain.sync_loss_runtime",
                    consumes=("loss_fn", "trainer"),
                    produces=("loss_fn",),
                    attached_attrs=("mu_y", "std_y"),
                ),
                BuildSubstepManifest(
                    step_id="posttrain.apply_local_overlay",
                    consumes=("cfg", "trainer"),
                    produces=("Trainer",),
                    attached_attrs=(
                        "posttrain_contacts_pretrain_attrs",
                        "contact_meas_policy_attrs",
                        "lambda_reliability_attrs",
                        "rollout_local_runtime_attrs",
                    ),
                ),
            ),
            notes=("volatile_excluded: bundle_json_path, out_dir, full config blob, run name",),
        ),
        BuildStepManifest(
            step_id="posttrain.configure_trainable_slots",
            step_order=9,
            consumes=("cfg", "train_mode", "model"),
            produces=("trainable_parameter_mask",),
            attached_attrs=("freeze_all", "unfreeze_selected_slots"),
            normalized_config={
                "train_mode": None if train_mode is None else str(train_mode),
                "trainable_slots": trainable_slot_map,
            },
            notes=("volatile_excluded: parameter object ids",),
        ),
    )
    return BuildTraceManifest(pipeline="posttrain", steps=steps)


def build_canonical_checkpoint_manifest(
    *,
    io_signature: IOSignatureManifest,
    module_graph: ModuleGraphManifest,
    build_trace: Optional[BuildTraceManifest] = None,
    train_policy: Optional[Mapping[str, Any]] = None,
) -> CanonicalCheckpointManifest:
    return CanonicalCheckpointManifest(
        io_signature=io_signature,
        module_graph=module_graph,
        build_trace=build_trace,
        train_policy=regularize_manifest_value(train_policy or {}),
    )


def build_checkpoint_fingerprint_metadata(
    *,
    io_signature: IOSignatureManifest,
    module_graph: ModuleGraphManifest,
    build_trace: BuildTraceManifest,
    state_dict: Mapping[str, Any],
    train_policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    manifest = build_canonical_checkpoint_manifest(
        io_signature=io_signature,
        module_graph=module_graph,
        build_trace=build_trace,
        train_policy=train_policy,
    )
    fingerprints = {
        "io_signature_hash": compute_io_signature_hash(manifest.io_signature),
        "module_graph_hash": compute_module_graph_hash(manifest.module_graph),
        "build_order_hash": compute_build_order_hash(build_trace),
        "weights_hash": compute_weights_hash(state_dict),
        "train_policy_hash": compute_train_policy_hash(manifest.train_policy),
    }
    return {
        "fingerprint_schema_version": int(manifest.fingerprint_schema_version),
        "fingerprints": fingerprints,
        "manifest_summary": regularize_manifest_value(manifest),
    }


def compare_fingerprints(
    checkpoint_fingerprints: Optional[Mapping[str, str]],
    current_fingerprints: Mapping[str, str],
    *,
    required_segments: Sequence[str] = REQUIRED_FINGERPRINT_SEGMENTS,
    optional_segments: Sequence[str] = OPTIONAL_FINGERPRINT_SEGMENTS,
    short_diff_hints: Optional[Mapping[str, str]] = None,
) -> FingerprintCompareSummary:
    required = tuple(str(segment) for segment in required_segments)
    optional = tuple(str(segment) for segment in optional_segments)
    hints = {str(key): str(value) for key, value in (short_diff_hints or {}).items()}
    missing_current = sorted(set(required + optional) - set(current_fingerprints))
    if missing_current:
        raise ValueError(f"current_fingerprints is missing declared segment(s): {missing_current!r}")
    if checkpoint_fingerprints is None:
        result = FingerprintSegmentComparison(
            segment="fingerprint_block",
            status="missing_required",
            ckpt_hash=None,
            current_hash=None,
            short_diff_hint=f"checkpoint predates fingerprint policy introduced on {FINGERPRINT_POLICY_INTRO_DATE}",
            next_action="regenerate this checkpoint with current mainline; no legacy lane is provided.",
        )
        return FingerprintCompareSummary(
            required_segments=required,
            optional_segments=optional,
            results=(result,),
            overall_status="fail",
        )
    results: list[FingerprintSegmentComparison] = []
    overall_status = "pass"
    for segment in required + optional:
        ckpt_hash = checkpoint_fingerprints.get(segment)
        current_hash = current_fingerprints[segment]
        if ckpt_hash is None:
            status = "missing_required" if segment in required else "missing_optional"
            hint = hints.get(
                segment,
                "segment absent in checkpoint metadata"
                if status == "missing_required"
                else "segment absent in checkpoint metadata; optional no-check",
            )
            action = _REQUIRED_SEGMENT_ACTION if status == "missing_required" else _MISSING_OPTIONAL_ACTION
        elif str(ckpt_hash) == str(current_hash):
            status = "match"
            hint = hints.get(segment, "hash match")
            action = "none"
        else:
            status = "mismatch"
            hint = hints.get(segment, "checkpoint hash differs from current semantic fingerprint")
            if segment == "weights_hash":
                action = _WEIGHTS_FINGERPRINT_ACTION
            elif segment in optional:
                action = _OPTIONAL_SEGMENT_ACTION
            else:
                action = _REQUIRED_SEGMENT_ACTION
        if status == "missing_required":
            overall_status = "fail"
        elif status == "mismatch" and segment in required and segment != "weights_hash":
            overall_status = "fail"
        elif status == "mismatch" and overall_status != "fail":
            overall_status = "warn"
        results.append(
            FingerprintSegmentComparison(
                segment=segment,
                status=status,
                ckpt_hash=None if ckpt_hash is None else str(ckpt_hash),
                current_hash=None if current_hash is None else str(current_hash),
                short_diff_hint=hint,
                next_action=action,
            )
        )
    return FingerprintCompareSummary(
        required_segments=required,
        optional_segments=optional,
        results=tuple(results),
        overall_status=overall_status,
    )


def format_fingerprint_compare_summary(summary: FingerprintCompareSummary) -> str:
    if len(summary.results) == 1 and summary.results[0].segment == "fingerprint_block":
        result = summary.results[0]
        lines = [
            "[FATAL] checkpoint missing required fingerprint metadata.",
            f"- segment: {result.segment}",
            f"- status: {result.status}",
            f"- hint: {result.short_diff_hint}",
            f"- action: {result.next_action}",
        ]
        return "\n".join(lines)
    status_prefix = {"pass": "[OK]", "warn": "[WARN]", "fail": "[FATAL]"}[summary.overall_status]
    lines = [f"{status_prefix} checkpoint fingerprint comparison summary."]
    for result in summary.results:
        lines.extend(
            [
                f"- segment: {result.segment}",
                f"  status: {result.status}",
                f"  ckpt_hash: {result.ckpt_hash}",
                f"  current_hash: {result.current_hash}",
                f"  hint: {result.short_diff_hint}",
                f"  action: {result.next_action}",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "BuildStepManifest",
    "BuildSubstepManifest",
    "BuildTraceManifest",
    "CanonicalCheckpointManifest",
    "COMPONENT_KIND_VOCAB",
    "COMPONENT_SLOT_KIND_MAP",
    "COMPONENT_SLOT_ORDER",
    "FINGERPRINT_POLICY_INTRO_DATE",
    "FINGERPRINT_SCHEMA_VERSION",
    "FingerprintCompareSummary",
    "FingerprintSegmentComparison",
    "IOFieldManifest",
    "IOSignatureManifest",
    "ModuleGraphManifest",
    "OPTIONAL_FINGERPRINT_SEGMENTS",
    "REQUIRED_FINGERPRINT_SEGMENTS",
    "ComponentManifest",
    "build_basetrain_build_trace_manifest",
    "build_canonical_checkpoint_manifest",
    "build_checkpoint_fingerprint_metadata",
    "build_event_motion_model_io_signature_manifest",
    "build_event_motion_model_module_graph_manifest",
    "build_posttrain_build_trace_manifest",
    "canonical_json_dumps",
    "compare_fingerprints",
    "compute_build_order_hash",
    "compute_io_signature_hash",
    "compute_module_graph_hash",
    "compute_train_policy_hash",
    "compute_weights_hash",
    "format_fingerprint_compare_summary",
    "regularize_manifest_value",
    "regularize_name_collection",
]
