from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

if TYPE_CHECKING:
    from ..models import EventMotionModel

__all__ = [
    "ContactPlanBuildConfig",
    "DirectPoseBuildConfig",
    "DirectPoseLegBuildConfig",
    "EventClockBuildConfig",
    "LambdaFusionBuildConfig",
    "POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY",
    "POSTTRAIN_CHECKPOINT_CONTRACT_NAME",
    "POSTTRAIN_CHECKPOINT_CONTRACT_VERSION",
    "PosttrainContractBuildState",
    "PosttrainContractCkptPayload",
    "attach_motion_encoder_bundle",
    "compute_resolved_build_manifest_hash",
    "diff_resolved_build_manifests",
    "dump_posttrain_build_cfg",
    "enforce_strict_current_build_manifest_contract",
    "flatten_resolved_build_manifest",
    "load_posttrain_effective_cfg",
    "load_posttrain_contract_ckpt_payload",
    "normalize_contact_plan_init_mode",
    "normalize_direct_pose_feat_source",
    "normalize_direct_pose_leg_gate_mode",
    "normalize_direct_pose_leg_mode",
    "normalize_direct_pose_leg_gate_power",
    "normalize_direct_pose_phase_z_mode",
    "normalize_lambda_fusion_mode",
    "resolve_posttrain_build_state_from_contract",
]

POSTTRAIN_CHECKPOINT_CONTRACT_NAME = "posttrain_newflow"
POSTTRAIN_CHECKPOINT_CONTRACT_VERSION = 2
POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY = "train.posttrain"
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
    "mlp": "learned",
    "net": "learned",
    "nn": "learned",
    "learn": "learned",
    "learned": "learned",
    "gate": "learned",
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
_DIRECT_POSE_LEG_GATE_REMOVED_ALIASES: tuple[str, ...] = (
    "signed_scale",
    "signedscale",
    "signed",
    "signmag",
    "sign_mag",
    "signmagscale",
    "signedmag",
    "sscale",
)
_DIRECT_POSE_LEG_MODE_SO3_ALIASES: tuple[str, ...] = (
    "so3",
    "omega",
    "so3_compose",
    "compose",
    "exp",
    "expmap",
    "log",
    "axisangle",
    "axis_angle",
)
_CONTACT_PLAN_INIT_MODE_CANONICAL: tuple[str, ...] = ("zeros", "learnable", "obs", "learnable+obs")
_LAMBDA_FUSION_MODE_CANONICAL: tuple[str, ...] = ("global", "per_joint")
_MISSING_MANIFEST_VALUE = object()
_STRICT_CURRENT_CONTRACT_HINT = (
    "If loading a legacy checkpoint, migrate it with tools/migrate_legacy_posttrain_ckpt.py. "
    "If this checkpoint is expected to be strict/current, re-save it with the current build contract."
)


def _json_friendly_manifest_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_friendly_manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_friendly_manifest_value(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() <= 16:
            return _json_friendly_manifest_value(value.detach().cpu().tolist())
        return {
            "type": "tensor",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    return str(value)


def _canonical_resolved_build_manifest_json(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise TypeError(f"resolved build manifest must be a mapping; got {type(manifest).__name__}.")
    hash_payload: Any = {"config": manifest.get("config", None)} if "config" in manifest else manifest
    return json.dumps(
        _json_friendly_manifest_value(hash_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_resolved_build_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return sha256 hex for the canonical JSON hard-contract subset of a build manifest."""
    payload = _canonical_resolved_build_manifest_json(manifest).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_resolved_build_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten manifest['config'] into dotted field paths for contract diffs."""
    if not isinstance(manifest, Mapping):
        return {"<manifest>": _json_friendly_manifest_value(manifest)}
    config = manifest.get("config", None)
    if not isinstance(config, Mapping):
        return {"config": _json_friendly_manifest_value(config)}

    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if not value:
                flattened[prefix] = {}
            for key in sorted(value.keys(), key=lambda item: str(item)):
                child_path = str(key) if prefix == "" else f"{prefix}.{key}"
                visit(child_path, value[key])
            return
        flattened[prefix] = _json_friendly_manifest_value(value)

    for key in sorted(config.keys(), key=lambda item: str(item)):
        visit(str(key), config[key])
    return flattened


def _format_manifest_diff_value(value: Any) -> str:
    if value is _MISSING_MANIFEST_VALUE:
        return "<missing>"
    return repr(value)


def diff_resolved_build_manifests(
    current: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> list[str]:
    """Return field-level config diffs as checkpoint/current strings."""
    current_flat = flatten_resolved_build_manifest(current)
    checkpoint_flat = flatten_resolved_build_manifest(checkpoint)
    diffs: list[str] = []
    for key in sorted(set(current_flat.keys()) | set(checkpoint_flat.keys())):
        current_value = current_flat.get(key, _MISSING_MANIFEST_VALUE)
        checkpoint_value = checkpoint_flat.get(key, _MISSING_MANIFEST_VALUE)
        if checkpoint_value != current_value:
            diffs.append(
                f"{key}: checkpoint={_format_manifest_diff_value(checkpoint_value)} "
                f"current={_format_manifest_diff_value(current_value)}"
            )
    return diffs


def _format_manifest_diff_lines(diffs: list[str], *, limit: int = 20) -> str:
    shown = diffs[:limit]
    suffix = "" if len(diffs) <= limit else f"\n  ... ({len(diffs) - limit} more)"
    return "\n  " + "\n  ".join(shown) + suffix if shown else "\n  <no config field diffs>"


def enforce_strict_current_build_manifest_contract(
    *,
    current_manifest: Mapping[str, Any],
    checkpoint_manifest: Any,
    checkpoint_manifest_hash: Any,
) -> None:
    """Fail fast when a strict-current checkpoint build contract is absent or stale."""
    if not isinstance(checkpoint_manifest, Mapping):
        raise SystemExit(
            "[FATAL][strict_current_model_build] checkpoint missing resolved_build_manifest; "
            "this is not a strict-current contract checkpoint. "
            + _STRICT_CURRENT_CONTRACT_HINT
        )

    computed_checkpoint_hash = compute_resolved_build_manifest_hash(checkpoint_manifest)
    stored_checkpoint_hash = None if checkpoint_manifest_hash is None else str(checkpoint_manifest_hash).strip()
    if stored_checkpoint_hash != computed_checkpoint_hash:
        raise SystemExit(
            "[FATAL][strict_current_model_build] checkpoint resolved_build_manifest_hash mismatch; "
            f"stored={stored_checkpoint_hash or '<missing>'} computed={computed_checkpoint_hash}. "
            "Checkpoint contract is corrupted or stale. "
            + _STRICT_CURRENT_CONTRACT_HINT
        )

    current_hash = compute_resolved_build_manifest_hash(current_manifest)
    if current_hash == computed_checkpoint_hash:
        return

    diffs = diff_resolved_build_manifests(current_manifest, checkpoint_manifest)
    raise SystemExit(
        "[FATAL][strict_current_model_build] resolved build manifest mismatch; "
        f"checkpoint_hash={computed_checkpoint_hash} current_hash={current_hash}. "
        + _STRICT_CURRENT_CONTRACT_HINT
        + "\nField diffs (first 20):"
        + _format_manifest_diff_lines(diffs, limit=20)
    )


def normalize_direct_pose_leg_gate_mode(
    value: Any,
    *,
    default: str = "none",
    strict: bool = False,
    context: str = "direct_pose_leg_gate_mode",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in _DIRECT_POSE_LEG_GATE_REMOVED_ALIASES:
        raise SystemExit(
            f"[FATAL] {context}='signed_scale' is removed in current train/eval main chain. "
            "Migrate to 'scale' (or 'learned')."
        )
    canonical = _DIRECT_POSE_LEG_GATE_ALIAS_MAP.get(raw, None)
    if canonical is not None:
        return str(canonical)
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: none | learned | scale."
        )
    return str(default)


def normalize_contact_plan_init_mode(
    value: Any,
    *,
    default: str = "learnable",
    strict: bool = False,
    context: str = "contact_plan_init_mode",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in ("learnable_obs", "obs+learnable", "learnable+obs"):
        return "learnable+obs"
    if raw in _CONTACT_PLAN_INIT_MODE_CANONICAL:
        return str(raw)
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: zeros | learnable | obs | learnable+obs."
        )
    return str(default)


def normalize_direct_pose_phase_z_mode(
    value: Any,
    *,
    default: str = "concat",
    strict: bool = False,
    context: str = "direct_pose_phase_z_mode",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in ("replace", "replace_contacts", "replace_contact", "phase", "phase_only", "phase_only_hint"):
        return "replace_contacts"
    if raw in ("concat", "append", "add", "plus", "contacts+phase"):
        return "concat"
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: concat | replace_contacts."
        )
    return str(default)


def normalize_direct_pose_feat_source(
    value: Any,
    *,
    default: str = "cond",
    strict: bool = False,
    context: str = "direct_pose_feat_source",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in ("", "auto"):
        if strict:
            raise SystemExit(
                f"[FATAL] unsupported {context}={raw!r}; allowed values: "
                "cond | hidden | hidden_pre | cond+hidden | cond+hidden_pre."
            )
        return str(default)
    canonical = _DIRECT_POSE_FEAT_SOURCE_ALIAS_MAP.get(raw, raw)
    if canonical in _DIRECT_POSE_FEAT_SOURCE_CANONICAL:
        return str(canonical)
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: "
            "cond | hidden | hidden_pre | cond+hidden | cond+hidden_pre."
        )
    return str(default)


def normalize_direct_pose_leg_mode(
    value: Any,
    *,
    default: str = "rot6d_add",
    strict: bool = False,
    context: str = "direct_pose_leg_mode",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in _DIRECT_POSE_LEG_MODE_SO3_ALIASES:
        return "so3"
    if raw == "rot6d_add":
        return "rot6d_add"
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: rot6d_add | so3."
        )
    return str(default)


def normalize_lambda_fusion_mode(
    value: Any,
    *,
    default: str = "per_joint",
    strict: bool = False,
    context: str = "lambda_fusion_mode",
) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw in _LAMBDA_FUSION_MODE_CANONICAL:
        return str(raw)
    if strict:
        raise SystemExit(
            f"[FATAL] unsupported {context}={raw!r}; allowed values: global | per_joint."
        )
    return str(default)


def normalize_direct_pose_leg_gate_power(
    value: Any,
    *,
    default: float = 1.0,
) -> float:
    try:
        power = float(default if value is None else value)
    except Exception:
        power = float(default)
    if (not math.isfinite(power)) or power <= 0.0:
        return float(default)
    return float(power)


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
class DirectPoseLegBuildConfig:
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


@dataclass(frozen=True)
class EventClockBuildConfig:
    use_event_clock: bool
    hidden_dim: int
    gate_hidden_dim: int
    max_delta: float
    period_dim_init: int


@dataclass(frozen=True)
class LambdaFusionBuildConfig:
    enable: bool
    mode: str
    hidden: int
    dropout: float
    logit_init: float
    use_rollout_step: bool


@dataclass(frozen=True)
class PosttrainContractCkptPayload:
    checkpoint_contract: dict[str, Any]
    build_cfg: dict[str, Any]
    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    stripped_frozen_key_count: int


@dataclass(frozen=True)
class PosttrainContractBuildState:
    ckpt_posttrain_cfg: Optional[dict[str, Any]]
    state_dict: dict[str, Any]
    in_state_dim: int
    out_motion_dim: int
    cond_dim: int
    hidden_dim: int
    depth: int
    num_heads: int
    dropout: float
    context_len: int
    contact_dim: int
    angvel_dim: int
    pose_hist_dim: int
    period_dim: int
    contact_plan_cfg: ContactPlanBuildConfig
    direct_pose_cfg: DirectPoseBuildConfig
    direct_pose_leg_cfg: DirectPoseLegBuildConfig
    event_clock_cfg: EventClockBuildConfig
    lambda_fusion_cfg: LambdaFusionBuildConfig


def _normalize_contact_plan_init_mode(value: Any) -> str:
    return normalize_contact_plan_init_mode(value, default="learnable", strict=False)


def _load_ckpt_payload(
    ckpt_or_path: Any,
    *,
    map_location: str | torch.device,
) -> Any:
    return (
        torch.load(ckpt_or_path, map_location=map_location)
        if isinstance(ckpt_or_path, (str, bytes, bytearray)) or hasattr(ckpt_or_path, "__fspath__")
        else ckpt_or_path
    )


def _extract_ckpt_posttrain_cfg(ckpt: Any) -> Optional[dict[str, Any]]:
    if isinstance(ckpt, dict):
        cfg = ckpt.get("posttrain_cfg", None)
        if isinstance(cfg, dict):
            return dict(cfg)
    return None


def _extract_event_motion_state_dict_from_ckpt(ckpt: Any) -> tuple[dict[str, Any], int]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_model_state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_model_state = ckpt["model_state_dict"]
    else:
        raw_model_state = ckpt
    if not isinstance(raw_model_state, dict):
        raise TypeError("Checkpoint payload does not contain a model state_dict.")
    state_dict: dict[str, Any] = {}
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
    return state_dict, int(stripped)


def _require_contract_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"[FATAL] checkpoint missing {context}; regenerate with current train.posttrain.")
    return dict(value)


def _require_contract_key(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise SystemExit(f"[FATAL] checkpoint missing {context}.{key}; regenerate with current train.posttrain.")
    return mapping[key]


def _require_contract_int(mapping: dict[str, Any], key: str, *, context: str) -> int:
    value = _require_contract_key(mapping, key, context=context)
    if isinstance(value, bool):
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected int.")
    try:
        return int(value)
    except Exception as exc:
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected int.") from exc


def _require_contract_float(mapping: dict[str, Any], key: str, *, context: str) -> float:
    value = _require_contract_key(mapping, key, context=context)
    if isinstance(value, bool):
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected float.")
    try:
        result = float(value)
    except Exception as exc:
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected float.") from exc
    if not math.isfinite(result):
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected finite float.")
    return float(result)


def _require_contract_bool(mapping: dict[str, Any], key: str, *, context: str) -> bool:
    value = _require_contract_key(mapping, key, context=context)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
    raise SystemExit(f"[FATAL] invalid {context}.{key}; expected bool.")


def _require_contract_str(mapping: dict[str, Any], key: str, *, context: str) -> str:
    value = _require_contract_key(mapping, key, context=context)
    text = str(value).strip()
    if text == "":
        raise SystemExit(f"[FATAL] invalid {context}.{key}; expected non-empty string.")
    return text


def _normalize_direct_pose_phase_z_mode(value: Any) -> str:
    return normalize_direct_pose_phase_z_mode(
        value,
        default="concat",
        strict=True,
        context="build_cfg.direct_pose.phase_z_mode",
    )


def dump_posttrain_build_cfg(
    *,
    model: "EventMotionModel",
    build_state: PosttrainContractBuildState,
) -> dict[str, Any]:
    contact_plan_cfg = build_state.contact_plan_cfg
    direct_pose_cfg = build_state.direct_pose_cfg
    direct_pose_leg_cfg = build_state.direct_pose_leg_cfg
    event_clock_cfg = build_state.event_clock_cfg
    lambda_fusion_cfg = build_state.lambda_fusion_cfg

    return {
        "in_state_dim": int(build_state.in_state_dim),
        "out_motion_dim": int(build_state.out_motion_dim),
        "cond_dim": int(build_state.cond_dim),
        "hidden_dim": int(getattr(model, "hidden_dim", build_state.hidden_dim)),
        "depth": int(getattr(model, "num_layers", build_state.depth)),
        "num_heads": int(build_state.num_heads),
        "dropout": float(build_state.dropout),
        "context_len": int(getattr(model, "context_len", build_state.context_len)),
        "contact_dim": int(getattr(model, "contact_dim", build_state.contact_dim)),
        "angvel_dim": int(getattr(model, "angvel_dim", build_state.angvel_dim)),
        "pose_hist_dim": int(getattr(model, "pose_hist_dim", build_state.pose_hist_dim)),
        "period_dim": int(getattr(model, "period_dim", build_state.period_dim)),
        "contact_plan": {
            "enable": bool(getattr(model, "contact_plan_enable", contact_plan_cfg.enable)),
            "hidden": int(getattr(model, "contact_plan_hidden", contact_plan_cfg.hidden)),
            "inject": str(getattr(model, "contact_plan_inject", contact_plan_cfg.inject) or contact_plan_cfg.inject),
            "time_pe_dim": int(getattr(model, "contact_plan_time_pe_dim", contact_plan_cfg.time_pe_dim)),
            "init_mode": str(getattr(model, "contact_plan_init_mode", contact_plan_cfg.init_mode) or contact_plan_cfg.init_mode),
            "init_hidden": int(getattr(model, "contact_plan_init_hidden", contact_plan_cfg.init_hidden)),
            "init_dropout": float(getattr(model, "_contact_plan_init_dropout", contact_plan_cfg.init_dropout) or contact_plan_cfg.init_dropout),
        },
        "direct_pose": {
            "enable": bool(getattr(model, "direct_pose_enable", direct_pose_cfg.enable)),
            "hidden": int(getattr(model, "direct_pose_hidden", direct_pose_cfg.hidden)),
            "meas_mode": str(getattr(model, "direct_pose_meas_mode", direct_pose_cfg.meas_mode) or direct_pose_cfg.meas_mode),
            "feat_source": str(getattr(model, "direct_pose_feat_source", direct_pose_cfg.feat_source) or direct_pose_cfg.feat_source),
            "time_pe_dim": int(getattr(model, "direct_pose_time_pe_dim", direct_pose_cfg.time_pe_dim)),
            "time_pe_base": float(getattr(model, "_direct_pose_time_pe_base", direct_pose_cfg.time_pe_base) or direct_pose_cfg.time_pe_base),
            "use_phase_z": bool(getattr(model, "direct_pose_use_phase_z", direct_pose_cfg.use_phase_z)),
            "phase_z_mode": str(getattr(model, "direct_pose_phase_z_mode", direct_pose_cfg.phase_z_mode) or direct_pose_cfg.phase_z_mode),
            "split_enable": bool(getattr(model, "direct_pose_split_enable", direct_pose_cfg.split_enable)),
            "arm_split_enable": bool(getattr(model, "direct_pose_arm_split_enable", direct_pose_cfg.arm_split_enable)),
            "arm_bones": getattr(model, "direct_pose_arm_bones", direct_pose_cfg.arm_bones),
            "nonleg_proj_dim": int(getattr(model, "direct_pose_nonleg_proj_dim", direct_pose_cfg.nonleg_proj_dim)),
        },
        "direct_pose_leg": {
            "enable": bool(getattr(model, "direct_pose_leg_enable", direct_pose_leg_cfg.enable)),
            "bones": getattr(model, "direct_pose_leg_bones", direct_pose_leg_cfg.bones),
            "mode": str(getattr(model, "direct_pose_leg_mode", direct_pose_leg_cfg.mode) or direct_pose_leg_cfg.mode),
            "stopgrad_main": bool(getattr(model, "direct_pose_leg_stopgrad_main", direct_pose_leg_cfg.stopgrad_main)),
            "detach_feat": bool(getattr(model, "direct_pose_leg_detach_feat", direct_pose_leg_cfg.detach_feat)),
            "max_deg": float(getattr(model, "direct_pose_leg_max_deg", direct_pose_leg_cfg.max_deg) or direct_pose_leg_cfg.max_deg),
            "gate_mode": normalize_direct_pose_leg_gate_mode(
                getattr(model, "direct_pose_leg_gate_mode", direct_pose_leg_cfg.gate_mode),
                default=str(direct_pose_leg_cfg.gate_mode or "none"),
                strict=False,
                context="direct_pose_leg_gate_mode",
            ),
            "gate_power": normalize_direct_pose_leg_gate_power(
                getattr(model, "direct_pose_leg_gate_power", direct_pose_leg_cfg.gate_power),
                default=float(direct_pose_leg_cfg.gate_power),
            ),
            "scale_log_clip": float(
                getattr(model, "direct_pose_leg_scale_log_clip", direct_pose_leg_cfg.scale_log_clip)
                or direct_pose_leg_cfg.scale_log_clip
            ),
            "scale_clamp_k": float(
                getattr(model, "direct_pose_leg_scale_clamp_k", direct_pose_leg_cfg.scale_clamp_k)
                or direct_pose_leg_cfg.scale_clamp_k
            ),
        },
        "event_clock": {
            "enable": bool(getattr(model, "use_event_clock", event_clock_cfg.use_event_clock)),
            "hidden_dim": int(getattr(model, "event_clock_hidden_dim", event_clock_cfg.hidden_dim)),
            "gate_hidden_dim": int(getattr(model, "event_clock_gate_hidden_dim", event_clock_cfg.gate_hidden_dim)),
            "max_delta": float(getattr(model, "event_clock_max_delta", event_clock_cfg.max_delta) or event_clock_cfg.max_delta),
        },
        "lambda_fusion": {
            "enable": bool(getattr(model, "lambda_fusion_enable", lambda_fusion_cfg.enable)),
            "mode": str(getattr(model, "lambda_fusion_mode", lambda_fusion_cfg.mode) or lambda_fusion_cfg.mode),
            "hidden": int(getattr(model, "lambda_fusion_hidden", lambda_fusion_cfg.hidden)),
            "dropout": float(getattr(model, "_lambda_fusion_dropout", lambda_fusion_cfg.dropout) or lambda_fusion_cfg.dropout),
            "logit_init": float(
                getattr(model, "_lambda_fusion_logit_init", lambda_fusion_cfg.logit_init) or lambda_fusion_cfg.logit_init
            ),
            "use_rollout_step": bool(
                getattr(model, "lambda_fusion_use_rollout_step", lambda_fusion_cfg.use_rollout_step)
            ),
        },
    }


def load_posttrain_contract_ckpt_payload(
    ckpt_or_path: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> PosttrainContractCkptPayload:
    ckpt = _load_ckpt_payload(ckpt_or_path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint payload must be a dict for contract-based posttrain loading.")

    contract = ckpt.get("checkpoint_contract", None)
    if not isinstance(contract, dict) or str(contract.get("name", "")).strip() != POSTTRAIN_CHECKPOINT_CONTRACT_NAME:
        raise SystemExit(
            "[FATAL] unsupported checkpoint contract; regenerate with current train.posttrain."
        )
    try:
        version = int(contract.get("version", -1))
    except Exception as exc:
        raise SystemExit(
            f"[FATAL] invalid posttrain checkpoint contract version={contract.get('version', None)!r}; "
            f"current mainline expects v{int(POSTTRAIN_CHECKPOINT_CONTRACT_VERSION)}."
        ) from exc
    if version == 1:
        raise SystemExit(
            "[FATAL] posttrain checkpoint contract v1 is retired in current mainline after "
            "side-routing/shared-leg-head removal. No v1 compatibility adapter remains. "
            "Regenerate this checkpoint with current train.posttrain to produce contract v2, "
            "or use archived historical repro lanes for v1 artifacts."
        )
    if version != int(POSTTRAIN_CHECKPOINT_CONTRACT_VERSION):
        raise SystemExit(
            f"[FATAL] unsupported posttrain checkpoint contract version={version}; "
            f"current mainline expects v{int(POSTTRAIN_CHECKPOINT_CONTRACT_VERSION)}."
        )

    build_cfg = _require_contract_mapping(ckpt.get("build_cfg", None), context="build_cfg")
    ckpt_posttrain_cfg = _extract_ckpt_posttrain_cfg(ckpt)
    state_dict, stripped = _extract_event_motion_state_dict_from_ckpt(ckpt)
    return PosttrainContractCkptPayload(
        checkpoint_contract={
            "name": str(contract.get("name", POSTTRAIN_CHECKPOINT_CONTRACT_NAME)),
            "version": int(version),
            "created_by": str(contract.get("created_by", "")),
        },
        build_cfg=build_cfg,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        state_dict=state_dict,
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


def _normalize_direct_pose_feat_source(val: Any) -> Optional[str]:
    if val is None:
        return None
    text = str(val).strip()
    if text.lower() in ("", "auto"):
        return None
    try:
        return normalize_direct_pose_feat_source(
            text,
            default="cond",
            strict=True,
            context="build_cfg.direct_pose.feat_source",
        )
    except SystemExit:
        return None


def resolve_posttrain_build_state_from_contract(
    *,
    ckpt_payload: PosttrainContractCkptPayload,
) -> PosttrainContractBuildState:
    build_cfg = _require_contract_mapping(ckpt_payload.build_cfg, context="build_cfg")
    contact_plan_raw = _require_contract_mapping(
        build_cfg.get("contact_plan", None),
        context="build_cfg.contact_plan",
    )
    direct_pose_raw = _require_contract_mapping(
        build_cfg.get("direct_pose", None),
        context="build_cfg.direct_pose",
    )
    direct_pose_leg_raw = _require_contract_mapping(
        build_cfg.get("direct_pose_leg", None),
        context="build_cfg.direct_pose_leg",
    )
    event_clock_raw = _require_contract_mapping(
        build_cfg.get("event_clock", None),
        context="build_cfg.event_clock",
    )
    lambda_fusion_raw = _require_contract_mapping(
        build_cfg.get("lambda_fusion", None),
        context="build_cfg.lambda_fusion",
    )

    direct_pose_feat_source = _normalize_direct_pose_feat_source(
        _require_contract_str(direct_pose_raw, "feat_source", context="build_cfg.direct_pose")
    )
    if direct_pose_feat_source is None:
        raise SystemExit(
            "[FATAL] unsupported build_cfg.direct_pose.feat_source; "
            "allowed values: cond | hidden | hidden_pre | cond+hidden | cond+hidden_pre."
        )
    direct_pose_meas_mode = _require_contract_str(
        direct_pose_raw,
        "meas_mode",
        context="build_cfg.direct_pose",
    ).strip().lower()
    if direct_pose_meas_mode not in ("concat", "mode_select"):
        raise SystemExit(
            f"[FATAL] unsupported build_cfg.direct_pose.meas_mode={direct_pose_meas_mode!r}; "
            "allowed values: concat | mode_select."
        )
    lambda_fusion_mode = normalize_lambda_fusion_mode(
        _require_contract_str(lambda_fusion_raw, "mode", context="build_cfg.lambda_fusion"),
        default="per_joint",
        strict=True,
        context="build_cfg.lambda_fusion.mode",
    )
    contact_plan_init_mode = normalize_contact_plan_init_mode(
        _require_contract_str(contact_plan_raw, "init_mode", context="build_cfg.contact_plan"),
        default="learnable",
        strict=True,
        context="build_cfg.contact_plan.init_mode",
    )
    direct_pose_phase_z_mode = normalize_direct_pose_phase_z_mode(
        _require_contract_str(direct_pose_raw, "phase_z_mode", context="build_cfg.direct_pose"),
        default="concat",
        strict=True,
        context="build_cfg.direct_pose.phase_z_mode",
    )
    direct_pose_leg_gate_mode = normalize_direct_pose_leg_gate_mode(
        _require_contract_str(direct_pose_leg_raw, "gate_mode", context="build_cfg.direct_pose_leg"),
        default="none",
        strict=True,
        context="build_cfg.direct_pose_leg.gate_mode",
    )
    direct_pose_leg_mode = normalize_direct_pose_leg_mode(
        _require_contract_str(direct_pose_leg_raw, "mode", context="build_cfg.direct_pose_leg"),
        default="rot6d_add",
        strict=True,
        context="build_cfg.direct_pose_leg.mode",
    )
    direct_pose_leg_gate_power = normalize_direct_pose_leg_gate_power(
        _require_contract_float(direct_pose_leg_raw, "gate_power", context="build_cfg.direct_pose_leg"),
        default=1.0,
    )

    period_dim = _require_contract_int(build_cfg, "period_dim", context="build_cfg")

    return PosttrainContractBuildState(
        ckpt_posttrain_cfg=ckpt_payload.ckpt_posttrain_cfg,
        state_dict=ckpt_payload.state_dict,
        in_state_dim=_require_contract_int(build_cfg, "in_state_dim", context="build_cfg"),
        out_motion_dim=_require_contract_int(build_cfg, "out_motion_dim", context="build_cfg"),
        cond_dim=_require_contract_int(build_cfg, "cond_dim", context="build_cfg"),
        hidden_dim=_require_contract_int(build_cfg, "hidden_dim", context="build_cfg"),
        depth=_require_contract_int(build_cfg, "depth", context="build_cfg"),
        num_heads=_require_contract_int(build_cfg, "num_heads", context="build_cfg"),
        dropout=_require_contract_float(build_cfg, "dropout", context="build_cfg"),
        context_len=_require_contract_int(build_cfg, "context_len", context="build_cfg"),
        contact_dim=_require_contract_int(build_cfg, "contact_dim", context="build_cfg"),
        angvel_dim=_require_contract_int(build_cfg, "angvel_dim", context="build_cfg"),
        pose_hist_dim=_require_contract_int(build_cfg, "pose_hist_dim", context="build_cfg"),
        period_dim=int(period_dim),
        contact_plan_cfg=ContactPlanBuildConfig(
            enable=_require_contract_bool(contact_plan_raw, "enable", context="build_cfg.contact_plan"),
            hidden=_require_contract_int(contact_plan_raw, "hidden", context="build_cfg.contact_plan"),
            inject=_require_contract_str(contact_plan_raw, "inject", context="build_cfg.contact_plan"),
            time_pe_dim=_require_contract_int(contact_plan_raw, "time_pe_dim", context="build_cfg.contact_plan"),
            init_mode=str(contact_plan_init_mode),
            init_hidden=_require_contract_int(contact_plan_raw, "init_hidden", context="build_cfg.contact_plan"),
            init_dropout=_require_contract_float(contact_plan_raw, "init_dropout", context="build_cfg.contact_plan"),
        ),
        direct_pose_cfg=DirectPoseBuildConfig(
            enable=_require_contract_bool(direct_pose_raw, "enable", context="build_cfg.direct_pose"),
            hidden=_require_contract_int(direct_pose_raw, "hidden", context="build_cfg.direct_pose"),
            meas_mode=str(direct_pose_meas_mode),
            feat_source=str(direct_pose_feat_source),
            time_pe_dim=_require_contract_int(direct_pose_raw, "time_pe_dim", context="build_cfg.direct_pose"),
            time_pe_base=_require_contract_float(direct_pose_raw, "time_pe_base", context="build_cfg.direct_pose"),
            use_phase_z=_require_contract_bool(direct_pose_raw, "use_phase_z", context="build_cfg.direct_pose"),
            phase_z_mode=str(direct_pose_phase_z_mode),
            split_enable=_require_contract_bool(direct_pose_raw, "split_enable", context="build_cfg.direct_pose"),
            arm_split_enable=_require_contract_bool(
                direct_pose_raw,
                "arm_split_enable",
                context="build_cfg.direct_pose",
            ),
            arm_bones=direct_pose_raw.get("arm_bones", None),
            nonleg_proj_dim=_require_contract_int(direct_pose_raw, "nonleg_proj_dim", context="build_cfg.direct_pose"),
            drop_ckpt_weights=False,
        ),
        direct_pose_leg_cfg=DirectPoseLegBuildConfig(
            enable=_require_contract_bool(direct_pose_leg_raw, "enable", context="build_cfg.direct_pose_leg"),
            bones=direct_pose_leg_raw.get("bones", None),
            mode=str(direct_pose_leg_mode),
            stopgrad_main=_require_contract_bool(
                direct_pose_leg_raw,
                "stopgrad_main",
                context="build_cfg.direct_pose_leg",
            ),
            detach_feat=_require_contract_bool(
                direct_pose_leg_raw,
                "detach_feat",
                context="build_cfg.direct_pose_leg",
            ),
            max_deg=_require_contract_float(direct_pose_leg_raw, "max_deg", context="build_cfg.direct_pose_leg"),
            gate_mode=str(direct_pose_leg_gate_mode),
            gate_power=float(direct_pose_leg_gate_power),
            scale_log_clip=_require_contract_float(
                direct_pose_leg_raw,
                "scale_log_clip",
                context="build_cfg.direct_pose_leg",
            ),
            scale_clamp_k=_require_contract_float(
                direct_pose_leg_raw,
                "scale_clamp_k",
                context="build_cfg.direct_pose_leg",
            ),
        ),
        event_clock_cfg=EventClockBuildConfig(
            use_event_clock=_require_contract_bool(event_clock_raw, "enable", context="build_cfg.event_clock"),
            hidden_dim=_require_contract_int(event_clock_raw, "hidden_dim", context="build_cfg.event_clock"),
            gate_hidden_dim=_require_contract_int(
                event_clock_raw,
                "gate_hidden_dim",
                context="build_cfg.event_clock",
            ),
            max_delta=_require_contract_float(event_clock_raw, "max_delta", context="build_cfg.event_clock"),
            period_dim_init=int(period_dim),
        ),
        lambda_fusion_cfg=LambdaFusionBuildConfig(
            enable=_require_contract_bool(lambda_fusion_raw, "enable", context="build_cfg.lambda_fusion"),
            mode=str(lambda_fusion_mode),
            hidden=_require_contract_int(lambda_fusion_raw, "hidden", context="build_cfg.lambda_fusion"),
            dropout=_require_contract_float(lambda_fusion_raw, "dropout", context="build_cfg.lambda_fusion"),
            logit_init=_require_contract_float(lambda_fusion_raw, "logit_init", context="build_cfg.lambda_fusion"),
            use_rollout_step=_require_contract_bool(
                lambda_fusion_raw,
                "use_rollout_step",
                context="build_cfg.lambda_fusion",
            ),
        ),
    )


def load_posttrain_effective_cfg(
    ckpt_or_path: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = _load_ckpt_payload(ckpt_or_path, map_location=map_location)
    cfg = _extract_ckpt_posttrain_cfg(ckpt) or {}
    effective_cfg = dict(cfg)
    try:
        ckpt_payload = load_posttrain_contract_ckpt_payload(ckpt, map_location=map_location)
        build_state = resolve_posttrain_build_state_from_contract(ckpt_payload=ckpt_payload)
    except (SystemExit, TypeError, ValueError, KeyError):
        return effective_cfg

    contact_plan_cfg = build_state.contact_plan_cfg
    direct_pose_cfg = build_state.direct_pose_cfg
    direct_pose_leg_cfg = build_state.direct_pose_leg_cfg

    effective_cfg.update(
        {
            "depth": int(build_state.depth),
            "num_heads": int(build_state.num_heads),
            "dropout": float(build_state.dropout),
            "context_len": int(build_state.context_len),
            "contact_plan_init_mode": str(contact_plan_cfg.init_mode),
            "contact_plan_init_hidden": int(contact_plan_cfg.init_hidden),
            "contact_plan_init_dropout": float(contact_plan_cfg.init_dropout),
            "direct_pose_feat_source": str(direct_pose_cfg.feat_source),
            "direct_pose_time_pe_dim": int(direct_pose_cfg.time_pe_dim),
            "direct_pose_time_pe_base": float(direct_pose_cfg.time_pe_base),
            "direct_pose_use_phase_z": bool(direct_pose_cfg.use_phase_z),
            "direct_pose_phase_z_mode": str(direct_pose_cfg.phase_z_mode),
            "direct_pose_split_enable": bool(direct_pose_cfg.split_enable),
            "direct_pose_arm_split_enable": bool(direct_pose_cfg.arm_split_enable),
            "direct_pose_arm_bones": direct_pose_cfg.arm_bones,
            "direct_pose_nonleg_proj_dim": int(direct_pose_cfg.nonleg_proj_dim),
            "direct_pose_leg_gate_mode": str(direct_pose_leg_cfg.gate_mode),
            "direct_pose_leg_gate_power": float(direct_pose_leg_cfg.gate_power),
        }
    )
    return effective_cfg
