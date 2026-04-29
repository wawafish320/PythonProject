#!/usr/bin/env python3
from __future__ import annotations

"""Migrate a legacy posttrain checkpoint into a strict-current contract checkpoint."""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain as _posttrain
from train import posttrain_build_shell as _build_shell
from train.checkpoint.load_schema import (
    DirectPoseLoadCompatOptions,
    normalize_and_validate_direct_pose_ckpt_for_load,
)
from train.checkpoint.contract import (
    POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY,
    POSTTRAIN_CHECKPOINT_CONTRACT_NAME,
    POSTTRAIN_CHECKPOINT_CONTRACT_VERSION,
    compute_resolved_build_manifest_hash,
    diff_resolved_build_manifests,
    normalize_contact_plan_init_mode,
)
from train.checkpoint.fingerprint import (
    build_checkpoint_fingerprint_metadata,
    build_event_motion_model_io_signature_manifest,
    build_event_motion_model_module_graph_manifest,
    build_posttrain_build_trace_manifest,
    compute_weights_hash,
)
from train.configuration import model_build as _model_build_cfg
from train.configuration.io import load_json
from train.configuration.model_build import (
    DatasetModelFacts,
    resolve_current_model_build_config_with_trace,
)
from train.configuration.norm_spec import merge_norm_spec
from train.data.dataset import build_motion_dataset
from train.data.io import config_to_jsonable as _cfg_to_jsonable
from train.models import EventMotionModel
from train.runtime.freeze import _freeze_all, _unfreeze_for_train_mode

_MIGRATION_LEGACY_STRIPPED_CHECKPOINT_PREFIXES: tuple[str, ...] = (
    "frozen_encoder.",
    "frozen_period_head.",
    "contact_plan_input_proj.",
)
_STRICT_LOAD_IGNORED_CHECKPOINT_PREFIXES: tuple[str, ...] = ()


def _parse_set_values(items: Optional[Sequence[str]]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items or ():
        if "=" not in str(item):
            raise SystemExit(f"[FATAL] --set expects KEY=VALUE, got {item!r}.")
        key, value = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"[FATAL] --set expects non-empty KEY, got {item!r}.")
        overrides[key] = value
    return overrides


def _parse_unknown_overrides(items: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    index = 0
    while index < len(items):
        token = str(items[index])
        if not token.startswith("--"):
            raise SystemExit(f"[FATAL] unexpected positional argument {token!r}.")
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            index += 1
        else:
            key = key_value
            if index + 1 >= len(items) or str(items[index + 1]).startswith("--"):
                value = "true"
                index += 1
            else:
                value = str(items[index + 1])
                index += 2
        key = key.strip().replace("-", "_")
        if not key:
            raise SystemExit(f"[FATAL] malformed override flag {token!r}.")
        overrides[key] = value
    return overrides


def _load_payload(args: argparse.Namespace, unknown: Sequence[str]) -> dict[str, Any]:
    payload = load_json(args.config.expanduser()) if args.config is not None else {}
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    payload["ckpt_in"] = str(args.ckpt_in.expanduser())
    payload.setdefault("out_dir", str(args.out.expanduser().parent if args.out is not None else Path("./models/posttrain")))
    payload.setdefault("run_name", args.run_name or "migrated_strict")

    explicit: dict[str, Any] = {}
    for key in (
        "data",
        "bundle_json",
        "pretrain_template",
        "seq_len",
        "width",
        "event_clock",
        "load_context",
    ):
        value = getattr(args, key, None)
        if value is not None:
            explicit[key] = str(value) if isinstance(value, Path) else value
    if args.paths:
        explicit["paths"] = [str(path) for path in args.paths]
    if args.train_mode == "direct":
        explicit["train_direct_pose"] = "true"
        explicit["train_lambda_head"] = "false"
    elif args.train_mode == "lambda":
        explicit["train_direct_pose"] = "false"
        explicit["train_lambda_head"] = "true"

    payload.update(explicit)
    payload.update(_parse_set_values(args.set_values))
    payload.update(_parse_unknown_overrides(unknown))
    return payload


def _load_checkpoint(path: Path) -> Any:
    return torch.load(path.expanduser(), map_location="cpu")


def _raise_if_already_strict_contract_checkpoint(ckpt: Any, *, ckpt_path: Path) -> None:
    if not isinstance(ckpt, Mapping):
        return
    if not bool(ckpt.get("strict_current_model_build", False)):
        return
    if not isinstance(ckpt.get("resolved_build_manifest"), Mapping):
        return
    manifest_hash = ckpt.get("resolved_build_manifest_hash", None)
    raise SystemExit(
        "[FATAL][AlreadyStrict] checkpoint already carries a strict/current build contract and must not be "
        f"double-migrated: ckpt={ckpt_path.expanduser()} resolved_build_manifest_hash={manifest_hash!r}. "
        "Migration: use this checkpoint directly on the strict/current path; do not run "
        "tools/migrate_legacy_posttrain_ckpt.py on an already strict contract checkpoint."
    )


def _extract_ckpt_posttrain_cfg(ckpt: Any) -> Optional[dict[str, Any]]:
    cfg = ckpt.get("posttrain_cfg", None) if isinstance(ckpt, dict) else None
    return dict(cfg) if isinstance(cfg, Mapping) else None


def _extract_raw_model_state(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
    else:
        raw = ckpt
    if not isinstance(raw, dict):
        raise SystemExit(f"[FATAL] checkpoint model state must be a mapping; got {type(raw).__name__}.")
    return {str(key): value for key, value in raw.items()}


def _infer_width(state_dict: Mapping[str, Any], width_override: Any) -> int:
    if width_override is not None:
        try:
            width = int(width_override)
        except (TypeError, ValueError) as exc:
            raise SystemExit("[FATAL] --width must be an integer.") from exc
        if width <= 0:
            raise SystemExit("[FATAL] --width must be > 0.")
        return int(width)
    shared = state_dict.get("shared_encoder.0.weight", None)
    if not torch.is_tensor(shared) or shared.ndim < 1:
        raise SystemExit("[FATAL] unable to infer width; pass --width or --set width=...")
    return int(shared.shape[0])


def _strip_safe_to_exit_migration_tensors(
    *,
    state_dict: dict[str, Any],
    cfg: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strip only audited safe-to-exit tensors before legacy build resolution."""
    filtered = dict(state_dict)
    strips: list[dict[str, Any]] = []

    contact_plan_init_mode = normalize_contact_plan_init_mode(
        getattr(cfg, "contact_plan_init_mode", "learnable"),
        default="learnable",
        strict=True,
        context="migration contact_plan_init_mode",
    )
    if contact_plan_init_mode == "learnable":
        removed = sorted(key for key in filtered if key.startswith("contact_plan_init_head."))
        for key in removed:
            filtered.pop(key, None)
        if removed:
            strips.append(
                {
                    "prefix": "contact_plan_init_head.",
                    "reason": "contact_plan_init_mode=learnable",
                    "removed_count": len(removed),
                    "removed_keys": removed,
                }
            )

    if (not bool(getattr(cfg, "train_lambda_head", False))) and (not bool(getattr(cfg, "lambda_fusion_enable", False))):
        removed = sorted(key for key in filtered if key.startswith("lambda_fusion_head."))
        for key in removed:
            filtered.pop(key, None)
        if removed:
            strips.append(
                {
                    "prefix": "lambda_fusion_head.",
                    "reason": "non-lambda target with lambda_fusion_enable=false",
                    "removed_count": len(removed),
                    "removed_keys": removed,
                }
            )

    return filtered, strips


def _dataset_facts_from_json(path: Path) -> DatasetModelFacts:
    data = load_json(path.expanduser())
    required = (
        "dx",
        "dy",
        "dc",
        "contact_dim",
        "angvel_dim",
        "pose_hist_dim",
        "pose_hist_len",
        "period_dim",
        "state_layout",
        "output_layout",
        "bone_names",
        "fps",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"[FATAL] dataset facts JSON missing keys: {', '.join(missing)}")
    return DatasetModelFacts(
        dx=int(data["dx"]),
        dy=int(data["dy"]),
        dc=int(data["dc"]),
        contact_dim=int(data["contact_dim"]),
        angvel_dim=int(data["angvel_dim"]),
        pose_hist_dim=int(data["pose_hist_dim"]),
        pose_hist_len=int(data["pose_hist_len"]),
        period_dim=int(data["period_dim"]),
        state_layout=dict(data["state_layout"]),
        output_layout=dict(data["output_layout"]),
        bone_names=tuple(str(name) for name in data["bone_names"]),
        fps=float(data["fps"]),
    )


def _resolve_dataset_facts(cfg: Any, facts_json: Optional[Path]) -> DatasetModelFacts:
    if facts_json is not None:
        return _dataset_facts_from_json(facts_json)
    norm_spec = merge_norm_spec(
        cfg.bundle_json.expanduser().resolve(),
        cfg.pretrain_template,
        pretrain_keys=None,
        strict=True,
    )
    dataset = build_motion_dataset(
        data_dir=str(cfg.data.expanduser().resolve()),
        seq_len=max(2, int(cfg.seq_len)),
        paths=[str(path.expanduser().resolve()) for path in cfg.paths] if cfg.paths else None,
        norm_spec=norm_spec,
        index_mode=str(getattr(cfg, "dataset_index_mode", "sliding") or "sliding"),
        is_train=True,
    )
    return DatasetModelFacts.from_dataset(dataset, context="migration.dataset")


def _cfg_with_width(cfg: Any, width: int, *, strict_current: bool) -> Any:
    values = dict(vars(cfg))
    values["width"] = int(width)
    values["strict_current_model_build"] = bool(strict_current)
    return SimpleNamespace(**values)


def _resolve_legacy_model_build_config_with_trace(
    *,
    cfg: Any,
    dataset_facts: DatasetModelFacts,
    state_dict: Mapping[str, Any],
    ckpt_posttrain_cfg: Optional[Mapping[str, Any]],
    width: int,
    checkpoint_period_dim: int,
    has_encoder_bundle: bool,
) -> _model_build_cfg.ResolvedModelBuildConfig:
    state = {str(key): value for key, value in state_dict.items()}
    _model_build_cfg._reject_direct_pose_reinit_without_train(cfg)

    direct_pose_cfg_raw = _model_build_cfg._resolve_direct_pose_build_cfg(
        out_motion_dim=int(dataset_facts.dy),
        state_dict=state,
        ckpt_posttrain_cfg=dict(ckpt_posttrain_cfg) if isinstance(ckpt_posttrain_cfg, Mapping) else None,
        contact_dim=int(dataset_facts.contact_dim),
        cond_dim=int(dataset_facts.dc),
        width=int(width),
        overrides=_model_build_cfg.DirectPoseBuildOverrides(
            train_direct_pose=_model_build_cfg._cfg_bool(cfg, "train_direct_pose", False),
            direct_pose_reinit=_model_build_cfg._cfg_bool(cfg, "direct_pose_reinit", False),
            hidden_override=_model_build_cfg._cfg_optional_int(cfg, "direct_pose_hidden_override", min_value=1),
            meas_mode_override=_model_build_cfg._normalize_optional_direct_pose_meas_mode(
                _model_build_cfg._cfg_value(cfg, "direct_pose_meas_mode_override", None),
                field="direct_pose_meas_mode_override",
            ),
            feat_source=_model_build_cfg._posttrain_direct_pose_feat_source(cfg),
            time_pe_dim=_model_build_cfg._posttrain_direct_pose_time_pe_dim(cfg),
            time_pe_base=_model_build_cfg._cfg_float(
                cfg,
                "direct_pose_time_pe_base",
                _model_build_cfg.DEFAULT_DIRECT_POSE_TIME_PE_BASE,
                min_value=0.0,
            ),
            use_phase_z=_model_build_cfg._cfg_bool(cfg, "direct_pose_use_phase_z", False),
            phase_z_mode=_model_build_cfg._posttrain_direct_pose_phase_z_mode(cfg),
            split_enable=_model_build_cfg._cfg_bool(cfg, "direct_pose_split_enable", False),
            arm_split_enable=_model_build_cfg._cfg_bool(cfg, "direct_pose_arm_split_enable", False),
            arm_bones=_model_build_cfg._optional_csv(_model_build_cfg._cfg_value(cfg, "direct_pose_arm_bones", None)),
            nonleg_proj_dim=_model_build_cfg._cfg_int(cfg, "direct_pose_nonleg_proj_dim", 0, min_value=0),
        ),
    )
    direct_pose = _model_build_cfg.DirectPoseConfig(
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
        arm_bones=_model_build_cfg._optional_csv(direct_pose_cfg_raw.arm_bones),
        drop_ckpt_weights=bool(direct_pose_cfg_raw.drop_ckpt_weights),
    )
    contact_plan = _model_build_cfg._resolve_posttrain_contact_plan_config(
        cfg=cfg,
        facts=dataset_facts,
        state_dict=state,
        direct_pose=direct_pose,
    )
    event_clock = _model_build_cfg._resolve_posttrain_event_clock_config(
        cfg=cfg,
        state_dict=state,
        contact_dim=int(dataset_facts.contact_dim),
        checkpoint_period_dim=int(checkpoint_period_dim),
        has_encoder_bundle=bool(has_encoder_bundle),
    )
    lambda_fusion = _model_build_cfg._resolve_posttrain_lambda_fusion_config(
        cfg=cfg,
        state_dict=state,
        width=int(width),
        contact_dim=int(dataset_facts.contact_dim),
        contact_plan_enable=bool(contact_plan.enable),
    )
    config = _model_build_cfg.ModelBuildConfig(
        facts=dataset_facts,
        hidden_dim=int(width),
        num_layers=_model_build_cfg._cfg_int(cfg, "depth", 2, min_value=1),
        num_heads=_model_build_cfg._cfg_int(cfg, "num_heads", 4, min_value=1),
        dropout=_model_build_cfg._cfg_float(cfg, "dropout", 0.0, min_value=0.0),
        context_len=_model_build_cfg._cfg_int(cfg, "context_len", 16, min_value=1),
        pose_hist_dim_model=int(dataset_facts.pose_hist_dim),
        pose_hist_dim_raw=int(dataset_facts.pose_hist_dim),
        pose_hist_len_raw=int(dataset_facts.pose_hist_len),
        history_export_frames=0,
        history_frame_dim=0,
        contact_plan=contact_plan,
        direct_pose=direct_pose,
        direct_pose_leg=_model_build_cfg._resolve_posttrain_direct_pose_leg_config(cfg),
        event_clock=event_clock,
        lambda_fusion=lambda_fusion,
    )
    trace = _model_build_cfg._build_resolved_model_build_trace(
        cfg=cfg,
        config=config,
        dataset_facts=dataset_facts,
    )
    return _model_build_cfg.ResolvedModelBuildConfig(config=config, trace=tuple(trace))


def _resolve_legacy_and_strict_manifests(
    *,
    cfg: Any,
    facts: DatasetModelFacts,
    state_dict: dict[str, Any],
    ckpt_posttrain_cfg: Optional[Mapping[str, Any]],
    width: int,
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    period_dim = int(state_dict["period_encoder.weight"].shape[1]) if "period_encoder.weight" in state_dict else 0
    encoder_bundle = getattr(cfg, "encoder_bundle", None)
    has_encoder_bundle = bool(encoder_bundle is not None and encoder_bundle.expanduser().is_file())
    legacy = _resolve_legacy_model_build_config_with_trace(
        cfg=cfg,
        dataset_facts=facts,
        state_dict=state_dict,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        width=int(width),
        checkpoint_period_dim=int(period_dim),
        has_encoder_bundle=has_encoder_bundle,
    )
    strict_cfg = _cfg_with_width(cfg, width, strict_current=True)
    strict = resolve_current_model_build_config_with_trace(
        cfg=strict_cfg,
        dataset_facts=facts,
        width=int(width),
    )
    return legacy, legacy.manifest(), strict, strict.manifest()


def _format_diffs(diffs: Sequence[str], *, limit: int = 50) -> str:
    if not diffs:
        return "  <none>"
    shown = list(diffs[:limit])
    suffix = "" if len(diffs) <= limit else f"\n  ... ({len(diffs) - limit} more)"
    return "\n  " + "\n  ".join(shown) + suffix


def _load_legacy_weights_into_model(
    *,
    cfg: Any,
    model_config: Any,
    state_dict: dict[str, Any],
    ckpt_posttrain_cfg: Optional[dict[str, Any]],
    facts: DatasetModelFacts,
) -> EventMotionModel:
    model = EventMotionModel.from_config(model_config)
    normalize_and_validate_direct_pose_ckpt_for_load(
        state_dict=state_dict,
        model=model,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        contact_dim=int(facts.contact_dim),
        direct_pose_cfg=model_config.direct_pose,
        load_options=DirectPoseLoadCompatOptions(
            train_direct_pose=bool(getattr(cfg, "train_direct_pose", False)),
            leg_enable=bool(model_config.direct_pose_leg.enable),
            leg_bones=model_config.direct_pose_leg.bones,
        ),
        context="apply_direct_pose_ckpt_compat",
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_result, "missing_keys", ()) or ())
    unexpected = list(getattr(load_result, "unexpected_keys", ()) or ())
    if missing:
        print(f"[migrate][WARN] legacy load missing keys after compat: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[migrate][WARN] legacy load unexpected keys after compat: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    return model


def _clone_model_state_dict_to_cpu(model: EventMotionModel) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        output[str(key)] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    return output


def _build_trainable_slots(cfg: Any, model: EventMotionModel) -> tuple[str, dict[str, bool]]:
    train_mode = _build_shell._resolve_posttrain_train_mode(cfg)
    _freeze_all(model)
    _unfreeze_for_train_mode(
        model,
        train_mode=train_mode,
        direct_pose_leg_train_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
        direct_pose_leg_gate_train_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
        direct_pose_nonleg_train_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
        direct_pose_nonleg_trunk_mode=str(getattr(cfg, "direct_pose_nonleg_trunk_mode", "none") or "none"),
    )
    return train_mode, _build_shell._build_component_slot_trainable_map(model)


def _build_structural_flags(model_config: Any) -> dict[str, Any]:
    return {
        "direct_pose_enable": bool(model_config.direct_pose.enable),
        "lambda_fusion_enable": bool(model_config.lambda_fusion.enable),
        "contact_plan_enable": bool(model_config.contact_plan.enable),
        "use_event_clock": bool(model_config.event_clock.enable),
        "direct_pose_leg_enable": bool(model_config.direct_pose_leg.enable),
        "direct_pose_leg_side_routing": bool(model_config.direct_pose_leg.side_routing),
        "direct_pose_arm_split_enable": bool(model_config.direct_pose.arm_split_enable),
        "direct_pose_leg_mode": str(model_config.direct_pose_leg.mode),
    }


def _attach_runtime_assets_for_contract(cfg: Any, model: EventMotionModel) -> dict[str, Any]:
    """Attach runtime assets that strict posttrain attaches before fingerprint enforcement."""
    encoder_bundle = getattr(cfg, "encoder_bundle", None)
    if encoder_bundle is None or not encoder_bundle.expanduser().is_file():
        return {"encoder_bundle_attached": False}
    meta = _build_shell._attach_motion_encoder_bundle(
        model,
        torch.load(str(encoder_bundle.expanduser()), map_location="cpu"),
    )
    return {
        "encoder_bundle_attached": True,
        "encoder_bundle": str(encoder_bundle.expanduser()),
        "encoder_bundle_meta": dict(meta) if isinstance(meta, Mapping) else {},
    }


def _validate_strict_load_ready_final_state(
    *,
    cfg: Any,
    strict_model_config: Any,
    model_state: dict[str, Any],
    expected_weights_hash: str,
) -> dict[str, Any]:
    strict_model = EventMotionModel.from_config(strict_model_config)
    attach_meta = _attach_runtime_assets_for_contract(cfg, strict_model)
    load_ready_state = _build_shell._filter_checkpoint_state_dict(
        model_state,
        ignored_prefixes=_STRICT_LOAD_IGNORED_CHECKPOINT_PREFIXES,
    )
    strict_model.load_state_dict(load_ready_state, strict=True)
    recomputed_hash = compute_weights_hash(_clone_model_state_dict_to_cpu(strict_model))
    if str(recomputed_hash) != str(expected_weights_hash):
        raise SystemExit(
            "[FATAL] migrated strict checkpoint is not load-ready: "
            f"expected_weights_hash={expected_weights_hash} recomputed_after_strict_load={recomputed_hash}"
        )
    return {
        "strict_load_ready_validated": True,
        "strict_load_ready_weights_hash": recomputed_hash,
        "strict_load_ready_loaded_key_count": len(load_ready_state),
        **attach_meta,
    }


def _build_migrated_payload(
    *,
    ckpt: Any,
    cfg: Any,
    model: EventMotionModel,
    strict_manifest: dict[str, Any],
    strict_model_config: Any,
    migration_time_strips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attach_meta = _attach_runtime_assets_for_contract(cfg, model)
    model_state = _clone_model_state_dict_to_cpu(model)
    train_mode, trainable_slots = _build_trainable_slots(cfg, model)
    payload = dict(ckpt) if isinstance(ckpt, dict) else {}
    payload["model"] = model_state
    payload.pop("model_state_dict", None)

    posttrain_cfg = payload.get("posttrain_cfg", None)
    posttrain_cfg = dict(posttrain_cfg) if isinstance(posttrain_cfg, Mapping) else _cfg_to_jsonable(cfg)
    posttrain_cfg["strict_current_model_build"] = True
    posttrain_cfg["width"] = int(strict_model_config.hidden_dim)
    payload["posttrain_cfg"] = posttrain_cfg
    payload["checkpoint_contract"] = {
        "name": POSTTRAIN_CHECKPOINT_CONTRACT_NAME,
        "version": int(POSTTRAIN_CHECKPOINT_CONTRACT_VERSION),
        "created_by": POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY,
    }
    payload["resolved_build_manifest"] = strict_manifest
    payload["resolved_build_manifest_hash"] = compute_resolved_build_manifest_hash(strict_manifest)
    payload["strict_current_model_build"] = True
    fingerprint_metadata = build_checkpoint_fingerprint_metadata(
        io_signature=build_event_motion_model_io_signature_manifest(model),
        module_graph=build_event_motion_model_module_graph_manifest(model),
        build_trace=build_posttrain_build_trace_manifest(
            structural_flags=_build_structural_flags(strict_model_config),
            train_mode=str(train_mode),
            trainable_slots=trainable_slots,
        ),
        state_dict=model_state,
        train_policy=_build_shell._build_posttrain_train_policy_manifest(
            cfg=cfg,
            train_mode=train_mode,
        ),
    )
    payload.update(fingerprint_metadata)
    strict_load_ready_meta = _validate_strict_load_ready_final_state(
        cfg=cfg,
        strict_model_config=strict_model_config,
        model_state=model_state,
        expected_weights_hash=str(fingerprint_metadata["fingerprints"]["weights_hash"]),
    )
    migration_meta = dict(payload.get("migration_meta", {})) if isinstance(payload.get("migration_meta"), Mapping) else {}
    migration_meta["legacy_to_strict_contract"] = {
        "tool": "tools/migrate_legacy_posttrain_ckpt.py",
        "resolved_build_manifest_hash": payload["resolved_build_manifest_hash"],
        "fingerprint_basis": "strict_load_ready_final_model_state",
        "final_model_state_key_count": len(model_state),
        "migration_time_strips": [dict(item) for item in migration_time_strips],
        **attach_meta,
        **strict_load_ready_meta,
    }
    payload["migration_meta"] = migration_meta
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a legacy posttrain checkpoint into a strict-current contract checkpoint. "
            "Unknown --key value flags are forwarded into the posttrain config payload."
        )
    )
    parser.add_argument("--ckpt_in", type=Path, required=True, help="Legacy checkpoint path.")
    parser.add_argument("--out", type=Path, help="Output strict-compatible checkpoint path.")
    parser.add_argument("--config", type=Path, help="Posttrain config JSON used to resolve build flags.")
    parser.add_argument("--dataset_facts_json", type=Path, help="JSON DatasetModelFacts source; skips dataset loading.")
    parser.add_argument("--dry_run", action="store_true", help="Print legacy-vs-strict build diff and do not write.")
    parser.add_argument("--run_name", type=str, help="Config run_name override for migrated metadata.")
    parser.add_argument("--data", type=Path, help="Dataset root override.")
    parser.add_argument("--paths", type=Path, nargs="*", help="Optional dataset clip path overrides.")
    parser.add_argument("--bundle_json", type=Path, help="Norm bundle JSON override.")
    parser.add_argument("--pretrain_template", type=Path, help="Optional pretrain template override.")
    parser.add_argument("--seq_len", type=int, help="Dataset seq_len override.")
    parser.add_argument("--width", type=int, help="Strict/current model width; defaults to legacy checkpoint inference.")
    parser.add_argument("--event_clock", type=str, choices=("on", "off"), help="Required by strict resolver if config is auto.")
    parser.add_argument("--load_context", type=str, choices=("resume", "chain_hop"), help="Metadata/config override.")
    parser.add_argument("--train_mode", choices=("direct", "lambda"), help="Set exactly one posttrain target mode.")
    parser.add_argument("--set", dest="set_values", action="append", help="Config override KEY=VALUE; may repeat.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args, unknown = parser.parse_known_args()
    if not args.dry_run and args.out is None:
        raise SystemExit("[FATAL] --out is required unless --dry_run is set.")

    payload = _load_payload(args, unknown)
    cfg = _posttrain._cfg_from_payload(payload)
    ckpt = _load_checkpoint(args.ckpt_in)
    _raise_if_already_strict_contract_checkpoint(ckpt, ckpt_path=args.ckpt_in)
    raw_state = _extract_raw_model_state(ckpt)
    state_dict = _build_shell._filter_checkpoint_state_dict(
        raw_state,
        ignored_prefixes=_MIGRATION_LEGACY_STRIPPED_CHECKPOINT_PREFIXES,
    )
    ckpt_posttrain_cfg = _extract_ckpt_posttrain_cfg(ckpt)
    width = _infer_width(state_dict, getattr(cfg, "width", None))
    cfg = _cfg_with_width(cfg, width, strict_current=False)
    state_dict, migration_time_strips = _strip_safe_to_exit_migration_tensors(
        state_dict=state_dict,
        cfg=cfg,
    )
    if migration_time_strips:
        print(f"[migrate] migration-time strips: {migration_time_strips}")
    facts = _resolve_dataset_facts(cfg, args.dataset_facts_json)

    legacy, legacy_manifest, strict, strict_manifest = _resolve_legacy_and_strict_manifests(
        cfg=cfg,
        facts=facts,
        state_dict=copy.copy(state_dict),
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        width=int(width),
    )
    diffs = diff_resolved_build_manifests(strict_manifest, legacy_manifest)
    legacy_hash = compute_resolved_build_manifest_hash(legacy_manifest)
    strict_hash = compute_resolved_build_manifest_hash(strict_manifest)
    print(f"[migrate] legacy_hash={legacy_hash}")
    print(f"[migrate] strict_hash={strict_hash}")
    print(f"[migrate] config field diffs legacy(checkpoint) vs requested strict(current):\n{_format_diffs(diffs)}")

    if args.dry_run:
        print("[migrate] dry-run: no checkpoint written.")
        return
    if diffs:
        raise SystemExit(
            "[FATAL] requested strict build does not match legacy-inferred build; "
            "fix config/CLI flags, then rerun. Use --dry_run to inspect diffs without writing."
        )

    model = _load_legacy_weights_into_model(
        cfg=cfg,
        model_config=legacy.config,
        state_dict=state_dict,
        ckpt_posttrain_cfg=ckpt_posttrain_cfg,
        facts=facts,
    )
    payload_out = _build_migrated_payload(
        ckpt=ckpt,
        cfg=_cfg_with_width(cfg, width, strict_current=True),
        model=model,
        strict_manifest=strict_manifest,
        strict_model_config=strict.config,
        migration_time_strips=migration_time_strips,
    )
    out_path = args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload_out, out_path)
    print(f"[migrate] wrote strict-compatible checkpoint: {out_path}")


if __name__ == "__main__":
    main()
