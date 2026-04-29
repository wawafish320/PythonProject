#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain as _posttrain
from train import posttrain_build_shell as shell
from train.checkpoint.contract import (
    POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY,
    POSTTRAIN_CHECKPOINT_CONTRACT_NAME,
    POSTTRAIN_CHECKPOINT_CONTRACT_VERSION,
    compute_resolved_build_manifest_hash,
    normalize_contact_plan_init_mode,
)
from train.checkpoint.fingerprint import (
    build_checkpoint_fingerprint_metadata,
    build_event_motion_model_io_signature_manifest,
    build_event_motion_model_module_graph_manifest,
    build_posttrain_build_trace_manifest,
    compute_weights_hash,
)
from train.configuration.model_build import DatasetModelFacts, resolve_current_model_build_config_with_trace
from train.configuration.norm_spec import merge_norm_spec
from train.data.dataset import build_motion_dataset
from train.data.io import config_to_jsonable as _cfg_to_jsonable
from train.models import EventMotionModel
from tools import migrate_legacy_posttrain_ckpt as mig


_DIRECT_POSE_PHASE_INPUT_KEYS = (
    "direct_pose_head.0.weight",
    "direct_pose_leg_head.0.weight",
)


def _load_dataset_facts(cfg: Any) -> DatasetModelFacts:
    norm = merge_norm_spec(
        cfg.bundle_json.expanduser().resolve(),
        cfg.pretrain_template,
        pretrain_keys=None,
        strict=True,
    )
    ds = build_motion_dataset(
        data_dir=str(cfg.data.expanduser().resolve()),
        seq_len=max(2, int(cfg.seq_len)),
        paths=[str(path.expanduser().resolve()) for path in cfg.paths] if cfg.paths else None,
        norm_spec=norm,
        index_mode=str(getattr(cfg, "dataset_index_mode", "sliding") or "sliding"),
        is_train=True,
    )
    return DatasetModelFacts.from_dataset(ds, context="strict_handoff_contractize.dataset")


def _clone_state(model: EventMotionModel) -> dict[str, Any]:
    return {
        str(key): (value.detach().cpu().clone() if torch.is_tensor(value) else value)
        for key, value in model.state_dict().items()
    }


def _attach_runtime_assets(cfg: Any, model: EventMotionModel) -> None:
    encoder_bundle = getattr(cfg, "encoder_bundle", None)
    if encoder_bundle is not None and encoder_bundle.expanduser().is_file():
        shell._attach_motion_encoder_bundle(
            model,
            torch.load(str(encoder_bundle.expanduser()), map_location="cpu"),
        )


def _source_state(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        return {str(key): value for key, value in ckpt["model"].items()}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
        return {str(key): value for key, value in ckpt["model_state_dict"].items()}
    if isinstance(ckpt, dict):
        return {str(key): value for key, value in ckpt.items()}
    raise SystemExit(f"[FATAL] source checkpoint state is not a dict: {type(ckpt).__name__}")


def _source_posttrain_cfg(ckpt: Any) -> dict[str, Any] | None:
    if not isinstance(ckpt, dict):
        return None
    cfg = ckpt.get("posttrain_cfg")
    if not isinstance(cfg, dict):
        return None
    return dict(cfg)


def _strip_safe_to_exit_source_tensors(
    *,
    source_state: dict[str, Any],
    target_cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only audited migration-time strips before strict handoff copying."""
    strip_specs: list[tuple[str, str]] = []
    contact_plan_init_mode = normalize_contact_plan_init_mode(
        getattr(target_cfg, "contact_plan_init_mode", "learnable"),
        default="learnable",
        strict=True,
        context="handoff target contact_plan_init_mode",
    )
    if contact_plan_init_mode == "learnable":
        strip_specs.append(("contact_plan_init_head.", "contact_plan_init_mode=learnable"))

    is_lambda_stage = bool(getattr(target_cfg, "train_lambda_head", False))
    lambda_fusion_enable = bool(getattr(target_cfg, "lambda_fusion_enable", False))
    if (not is_lambda_stage) and (not lambda_fusion_enable):
        strip_specs.append(("lambda_fusion_head.", "non-lambda target with lambda_fusion_enable=false"))

    stripped: list[dict[str, Any]] = []
    filtered = dict(source_state)
    for prefix, reason in strip_specs:
        removed = sorted(key for key in filtered if str(key).startswith(prefix))
        for key in removed:
            filtered.pop(key, None)
        if removed:
            stripped.append(
                {
                    "prefix": prefix,
                    "reason": reason,
                    "removed_count": len(removed),
                    "removed_keys": removed,
                }
            )
    return filtered, {"migration_time_strips": stripped}


def _should_zero_init_phase_expansion(
    *,
    key: str,
    source_ckpt_cfg: dict[str, Any] | None,
    target_cfg: Any,
    source_value: torch.Tensor,
    target_value: torch.Tensor,
    phase_dim: int,
) -> bool:
    if key not in _DIRECT_POSE_PHASE_INPUT_KEYS:
        return False
    if phase_dim <= 0:
        return False
    if int(target_value.shape[1]) <= int(source_value.shape[1]):
        return False
    if int(target_value.shape[1] - source_value.shape[1]) != int(phase_dim):
        return False
    source_use_phase_z = bool(isinstance(source_ckpt_cfg, dict) and source_ckpt_cfg.get("direct_pose_use_phase_z", False))
    target_use_phase_z = bool(getattr(target_cfg, "direct_pose_use_phase_z", False))
    if source_use_phase_z or (not target_use_phase_z):
        return False
    return True


def _copy_handoff_state(
    *,
    source_state: dict[str, Any],
    source_ckpt_cfg: dict[str, Any] | None,
    target_cfg: Any,
    target_state: dict[str, Any],
    phase_dim: int,
    allow_missing_prefix: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_state = dict(target_state)
    copied: list[str] = []
    expanded: list[dict[str, Any]] = []
    truncated: list[dict[str, Any]] = []
    missing: list[str] = []
    initialized_missing: list[dict[str, Any]] = []
    zero_initialized_expanded: list[dict[str, Any]] = []
    unexpected = sorted(str(key) for key in source_state.keys() if key not in target_state)

    for key, target_value in list(target_state.items()):
        source_value = source_state.get(key)
        if source_value is None:
            if any(str(key).startswith(prefix) for prefix in allow_missing_prefix):
                initialized_missing.append(
                    {
                        "key": str(key),
                        "target_shape": [int(dim) for dim in target_value.shape]
                        if torch.is_tensor(target_value)
                        else None,
                    }
                )
            else:
                missing.append(str(key))
            continue
        if not (torch.is_tensor(target_value) and torch.is_tensor(source_value)):
            out_state[key] = source_value
            copied.append(str(key))
            continue
        if tuple(target_value.shape) == tuple(source_value.shape):
            out_state[key] = source_value.detach().clone()
            copied.append(str(key))
            continue
        if target_value.ndim == 2 and source_value.ndim == 2 and target_value.shape[0] == source_value.shape[0]:
            is_phase_zero_fill = _should_zero_init_phase_expansion(
                key=str(key),
                source_ckpt_cfg=source_ckpt_cfg,
                target_cfg=target_cfg,
                source_value=source_value,
                target_value=target_value,
                phase_dim=int(phase_dim),
            )
            if not is_phase_zero_fill:
                raise SystemExit(
                    f"[FATAL][Removed] uncontracted 2D tensor shape adaptation for {key}: "
                    f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}. "
                    "Removed by 2026-04-28 strict branch unload cleanup: strict handoff only permits "
                    "source-no-phase-z to target-phase-z zero-fill for direct_pose_head.0.weight and "
                    "direct_pose_leg_head.0.weight. Migration: add an explicit migration rule or use a "
                    "shape-coherent checkpoint; no generic pad/crop/truncate handoff."
                )
            new_value = target_value.detach().clone()
            cols = min(int(target_value.shape[1]), int(source_value.shape[1]))
            new_value[:, :cols] = source_value.detach()[:, :cols]
            new_value[:, cols:] = 0.0
            out_state[key] = new_value
            entry = {
                "key": str(key),
                "source_shape": [int(dim) for dim in source_value.shape],
                "target_shape": [int(dim) for dim in target_value.shape],
                "copied_columns": cols,
            }
            if target_value.shape[1] > source_value.shape[1]:
                entry["initialized_new_columns"] = int(target_value.shape[1] - source_value.shape[1])
                expanded.append(entry)
                zero_initialized_expanded.append(
                    {
                        **entry,
                        "policy": "source_no_phase_z_to_target_phase_z_zero_fill",
                    }
                )
            else:
                raise SystemExit(
                    f"[FATAL][Removed] phase-z handoff only permits target expansion for {key}; "
                    f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}. "
                    "Migration: use a source without phase-z and a replace target with explicit phase-z, "
                    "or provide a coherent checkpoint."
                )
            continue
        raise SystemExit(
            f"[FATAL] unhandled tensor mismatch for {key}: "
            f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
        )

    if missing or unexpected:
        raise SystemExit(f"[FATAL] unhandled missing/unexpected tensors: missing={missing[:20]} unexpected={unexpected[:20]}")

    report = {
        "copied_tensor_count": len(copied),
        "expanded_tensors": expanded,
        "initialized_missing_tensors": initialized_missing,
        "zero_initialized_expanded_tensors": zero_initialized_expanded,
        "truncated_tensors": truncated,
    }
    return out_state, report


def _apply_tensor_transplant(
    *,
    target_state: dict[str, Any],
    donor_state: dict[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if not prefixes:
        return {"transplanted_tensor_count": 0, "transplanted_tensors": []}
    if any(prefix.startswith("direct_pose") for prefix in prefixes) and prefixes != ("direct_pose_",):
        raise SystemExit(
            "[FATAL] direct_pose transplant must use exactly --transplant-prefix direct_pose_ "
            "so the donor is a coherent full direct_pose_* bundle."
        )
    if prefixes == ("direct_pose_",):
        target_direct = sorted(key for key in target_state if key.startswith("direct_pose_"))
        donor_direct = sorted(key for key in donor_state if key.startswith("direct_pose_"))
        missing_direct = [key for key in target_direct if key not in donor_state]
        if missing_direct:
            raise SystemExit(
                "[FATAL] direct_pose donor is not a full coherent bundle; missing target tensors: "
                f"{missing_direct[:20]}"
            )
        if len(donor_direct) < len(target_direct):
            raise SystemExit(
                "[FATAL] direct_pose donor has fewer direct_pose_* tensors than target; "
                f"donor={len(donor_direct)} target={len(target_direct)}"
            )
    transplanted: list[dict[str, Any]] = []
    for key in sorted(str(k) for k in target_state.keys()):
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if key not in donor_state:
            raise SystemExit(f"[FATAL] transplant donor missing tensor: {key}")
        target_value = target_state[key]
        donor_value = donor_state[key]
        if not (torch.is_tensor(target_value) and torch.is_tensor(donor_value)):
            raise SystemExit(f"[FATAL] transplant expects tensors for {key}")
        if tuple(target_value.shape) != tuple(donor_value.shape):
            raise SystemExit(
                f"[FATAL] transplant shape mismatch for {key}: "
                f"target={tuple(target_value.shape)} donor={tuple(donor_value.shape)}"
            )
        target_state[key] = donor_value.detach().clone()
        transplanted.append(
            {
                "key": key,
                "shape": [int(dim) for dim in target_value.shape],
            }
        )
    return {
        "transplanted_tensor_count": len(transplanted),
        "transplanted_tensors": transplanted,
        "transplant_prefixes": list(prefixes),
        "transplant_policy": "coherent_full_direct_pose_bundle" if prefixes == ("direct_pose_",) else "explicit_prefix_bundle",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Contractize a strict posttrain handoff for a target config.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--allow-missing-prefix", action="append", default=[])
    parser.add_argument("--tensor-donor", type=Path, help="Optional checkpoint donor for post-handoff tensor transplant.")
    parser.add_argument("--transplant-prefix", action="append", default=[], help="Tensor prefix to copy from --tensor-donor.")
    args = parser.parse_args()

    cfg_payload = json.loads(args.target_config.expanduser().read_text(encoding="utf-8"))
    cfg = _posttrain._cfg_from_payload(cfg_payload)
    if not bool(getattr(cfg, "strict_current_model_build", False)):
        raise SystemExit("[FATAL] target config must set strict_current_model_build=true")
    torch.manual_seed(int(getattr(cfg, "seed", 0) or 0))

    facts = _load_dataset_facts(cfg)
    resolved = resolve_current_model_build_config_with_trace(
        cfg=cfg,
        dataset_facts=facts,
        width=int(cfg.width),
    )
    model = EventMotionModel.from_config(resolved.config)
    _attach_runtime_assets(cfg, model)

    source_ckpt = torch.load(args.source.expanduser(), map_location="cpu")
    source_state, strip_report = _strip_safe_to_exit_source_tensors(
        source_state=_source_state(source_ckpt),
        target_cfg=cfg,
    )
    target_state, copy_report = _copy_handoff_state(
        source_state=source_state,
        source_ckpt_cfg=_source_posttrain_cfg(source_ckpt),
        target_cfg=cfg,
        target_state=model.state_dict(),
        phase_dim=int(2 * int(facts.contact_dim)),
        allow_missing_prefix=tuple(str(prefix) for prefix in args.allow_missing_prefix),
    )
    transplant_report: dict[str, Any] = {"transplanted_tensor_count": 0, "transplanted_tensors": []}
    if args.tensor_donor is not None:
        donor_ckpt = torch.load(args.tensor_donor.expanduser(), map_location="cpu")
        transplant_report = _apply_tensor_transplant(
            target_state=target_state,
            donor_state=_source_state(donor_ckpt),
            prefixes=tuple(str(prefix) for prefix in args.transplant_prefix),
        )
        transplant_report["tensor_donor"] = str(args.tensor_donor.expanduser())
    model.load_state_dict(target_state, strict=True)
    model_state = _clone_state(model)
    train_mode, trainable_slots = mig._build_trainable_slots(cfg, model)
    manifest = resolved.manifest()
    metadata = build_checkpoint_fingerprint_metadata(
        io_signature=build_event_motion_model_io_signature_manifest(model),
        module_graph=build_event_motion_model_module_graph_manifest(model),
        build_trace=build_posttrain_build_trace_manifest(
            structural_flags=mig._build_structural_flags(resolved.config),
            train_mode=str(train_mode),
            trainable_slots=trainable_slots,
        ),
        state_dict=model_state,
        train_policy=shell._build_posttrain_train_policy_manifest(cfg=cfg, train_mode=train_mode),
    )
    payload: dict[str, Any] = {
        "model": model_state,
        "posttrain_cfg": _cfg_to_jsonable(cfg),
        "checkpoint_contract": {
            "name": POSTTRAIN_CHECKPOINT_CONTRACT_NAME,
            "version": int(POSTTRAIN_CHECKPOINT_CONTRACT_VERSION),
            "created_by": POSTTRAIN_CHECKPOINT_CONTRACT_CREATED_BY,
        },
        "resolved_build_manifest": manifest,
        "resolved_build_manifest_hash": compute_resolved_build_manifest_hash(manifest),
        "strict_current_model_build": True,
        "migration_meta": {
            "strict_posttrain_handoff_contractize": {
                "label": str(args.label),
                "source": str(args.source.expanduser()),
                "target_config": str(args.target_config.expanduser()),
                "policy": "migration-time tensor handoff; no load-time compat",
                "fingerprint_basis": "target strict load-ready final model state",
                "source_manifest_hash": source_ckpt.get("resolved_build_manifest_hash")
                if isinstance(source_ckpt, dict)
                else None,
                "target_manifest_hash": compute_resolved_build_manifest_hash(manifest),
                **strip_report,
                **copy_report,
                **transplant_report,
            }
        },
    }
    payload.update(metadata)

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out.expanduser())

    verify = EventMotionModel.from_config(resolved.config)
    _attach_runtime_assets(cfg, verify)
    verify.load_state_dict(payload["model"], strict=True)
    verify_hash = compute_weights_hash(_clone_state(verify))
    if verify_hash != payload["fingerprints"]["weights_hash"]:
        raise SystemExit(f"[FATAL] verify hash mismatch: {verify_hash} != {payload['fingerprints']['weights_hash']}")

    report = {
        "status": "pass",
        "label": str(args.label),
        "source": str(args.source.expanduser()),
        "output": str(args.out.expanduser()),
        "target_config": str(args.target_config.expanduser()),
        "source_manifest_hash": payload["migration_meta"]["strict_posttrain_handoff_contractize"]["source_manifest_hash"],
        "target_manifest_hash": payload["resolved_build_manifest_hash"],
        "weights_hash": payload["fingerprints"]["weights_hash"],
        "verify_weights_hash": verify_hash,
        "model_key_count": len(model_state),
        **strip_report,
        **copy_report,
        **transplant_report,
    }
    args.report.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
