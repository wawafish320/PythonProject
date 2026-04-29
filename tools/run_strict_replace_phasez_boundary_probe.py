#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain as _posttrain
from train import posttrain_build_shell as shell
from train.checkpoint.contract import compute_resolved_build_manifest_hash
from train.checkpoint.fingerprint import (
    build_checkpoint_fingerprint_metadata,
    build_event_motion_model_io_signature_manifest,
    build_event_motion_model_module_graph_manifest,
    build_posttrain_build_trace_manifest,
    compute_weights_hash,
)
from train.configuration.model_build import resolve_current_model_build_config_with_trace
from train.data.io import config_to_jsonable as _cfg_to_jsonable
from train.models import EventMotionModel
from tools import migrate_legacy_posttrain_ckpt as mig
from tools.contractize_strict_posttrain_handoff import (
    _attach_runtime_assets,
    _clone_state,
    _load_dataset_facts,
)

CPU_WRAPPER = ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
SUMMARY_TOOL = ROOT / "tools/phasea_group_summary.py"
TEACHER_JSON = ROOT / "validate/teacher_batches/Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models/motion_encoder_equiv.pt.best.pt"

CURRENT_RUN_ROOT = ROOT / "debug_output/_tmp_strict_stageB_finalstate_20260427_080658"
DEFAULT_BASE_CONFIG = CURRENT_RUN_ROOT / "stageB_strict/configs/replace.json"
DEFAULT_SOURCE_70A_CKPT = CURRENT_RUN_ROOT / "stageB_strict/70a/checkpoints/ckpt_last_70a_strictB_20260427_080803.pth"
DEFAULT_DONOR_REPLACE_CKPT = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/models/replace_clean"
    / "ckpt_last_WalkF_stage7_replace_from_stage6step360_70a_20260426_173158.pth"
)

GROUP_KEYS = ("all_ex_root", "leg", "nonleg", "arm", "else")
PATCH_KEYS = ("direct_pose_head.0.weight", "direct_pose_leg_head.0.weight")
DEFAULT_VARIANTS = ("baseline", "zero-new-cols")
_PHASE_Z_CARRIER_POLICY_BY_VARIANT = {
    "baseline": "preserve-phase-z-carrier",
    "zero-new-cols": "zero-phase-z-carrier",
    "donor-new-cols": "donor-phase-z-carrier",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _env(run_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["RUNTIME_SEMANTICS_TRACE_EVENTS"] = str(run_root / "runtime_semantics_events.jsonl")
    return env


def _cmd_text(cmd: Sequence[object]) -> str:
    return shlex.join([str(part) for part in cmd])


def _python_cmd() -> list[object]:
    if CPU_WRAPPER.is_file():
        return [CPU_WRAPPER]
    return [sys.executable]


def _run(cmd: Sequence[object], *, log_path: Path, run_root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = [str(part) for part in cmd]
    print(f"[RUN] {_cmd_text(cmd_list)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {_cmd_text(cmd_list)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(ROOT),
            env=_env(run_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = int(proc.wait())
        log.write(f"\n[exit_code] {rc}\n")
    if rc != 0:
        raise RuntimeError(f"command failed with exit={rc}: {_cmd_text(cmd_list)}")


def _canonicalize_direct_pose_config_fields(payload: dict[str, Any]) -> None:
    hidden_override = payload.pop("direct_pose_hidden_override", None)
    if hidden_override is not None and "direct_pose_hidden" not in payload:
        payload["direct_pose_hidden"] = int(hidden_override)

    meas_mode_override = payload.pop("direct_pose_meas_mode_override", None)
    if meas_mode_override is not None and str(meas_mode_override).strip():
        payload["direct_pose_meas_mode"] = str(meas_mode_override).strip()
    elif "direct_pose_meas_mode" not in payload:
        payload["direct_pose_meas_mode"] = "concat"


def _make_config(
    *,
    base_config: Path,
    variant_root: Path,
    handoff_ckpt: Path,
    run_name: str,
    steps_per_epoch: int,
) -> Path:
    payload = dict(_load_json(base_config))
    _canonicalize_direct_pose_config_fields(payload)
    for key in ("contact_plan_init_mode", "contact_plan_init_hidden", "contact_plan_init_dropout"):
        payload.pop(key, None)
    for key in tuple(payload.keys()):
        if str(key).startswith("lambda_fusion_"):
            payload.pop(str(key), None)
    payload.update(
        {
            "ckpt_in": str(handoff_ckpt),
            "out_dir": str(variant_root / "checkpoints"),
            "run_name": run_name,
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "epochs": 1,
            "steps_per_epoch": int(steps_per_epoch),
            "save_step_ckpts": f"0,1,20,{int(steps_per_epoch)}",
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
            "train_lambda_head": False,
        }
    )
    path = variant_root / "configs/replace_probe.json"
    _dump_json(path, payload)
    return path


def _contractize(
    *,
    source_ckpt: Path,
    target_config: Path,
    handoff_ckpt: Path,
    report: Path,
    variant_root: Path,
) -> None:
    cmd: list[object] = [
        sys.executable,
        ROOT / "tools/contractize_strict_posttrain_handoff.py",
        "--source",
        source_ckpt,
        "--target-config",
        target_config,
        "--out",
        handoff_ckpt,
        "--report",
        report,
        "--label",
        "70a_to_replace_phasez_probe",
    ]
    log_path = variant_root / "logs/contractize_70a_to_replace.log"
    _run(
        cmd,
        log_path=log_path,
        run_root=variant_root,
    )
    _rewrite_contractize_phase_z_log(log_path)


def _rewrite_contractize_phase_z_report(report_path: Path) -> None:
    payload = _load_json(report_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"contractize report must be a JSON object: {report_path}")

    def _rewrite_entries(entries: Any) -> list[dict[str, Any]]:
        if not isinstance(entries, list):
            return []
        rewritten_entries: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            rewritten = {str(key): value for key, value in entry.items()}
            initialized = rewritten.pop("initialized_new_columns", None)
            if initialized is not None:
                rewritten["initialized_phase_z_carrier_columns"] = int(initialized)
            rewritten["phase_z_carrier_semantics"] = (
                "trailing direct-pose phase-z carrier slice; not a generic layout expansion"
            )
            rewritten_entries.append(rewritten)
        return rewritten_entries

    phase_z_carrier_tensors = _rewrite_entries(payload.pop("expanded_tensors", []))
    zero_initialized_phase_z_carrier_tensors = _rewrite_entries(
        payload.pop("zero_initialized_expanded_tensors", [])
    )
    if phase_z_carrier_tensors or zero_initialized_phase_z_carrier_tensors:
        payload["phase_z_carrier_note"] = (
            "replace_contacts keeps direct-pose semantics fixed and only rewrites the trailing phase-z carrier "
            "slice; this probe is not auditing a generic width/layout expansion."
        )
        payload["phase_z_carrier_tensors"] = phase_z_carrier_tensors
        payload["zero_initialized_phase_z_carrier_tensors"] = zero_initialized_phase_z_carrier_tensors
    _dump_json(report_path, payload)


def _rewrite_contractize_phase_z_log(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rewritten = (
        text.replace("expanded_tensors", "phase_z_carrier_tensors")
        .replace("zero_initialized_expanded_tensors", "zero_initialized_phase_z_carrier_tensors")
        .replace("initialized_new_columns", "initialized_phase_z_carrier_columns")
    )
    if rewritten != text:
        log_path.write_text(rewritten, encoding="utf-8")


def _extract_model_state(ckpt: Mapping[str, Any]) -> dict[str, Any]:
    model_state = ckpt.get("model")
    if not isinstance(model_state, dict):
        raise RuntimeError("checkpoint missing model state dict")
    return {str(key): value for key, value in model_state.items()}


def _build_metadata_for_state(*, cfg_payload: Mapping[str, Any], model_state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    cfg = _posttrain._cfg_from_payload(dict(cfg_payload))
    if not bool(getattr(cfg, "strict_current_model_build", False)):
        raise RuntimeError("target config must set strict_current_model_build=true")
    torch.manual_seed(int(getattr(cfg, "seed", 0) or 0))

    facts = _load_dataset_facts(cfg)
    resolved = resolve_current_model_build_config_with_trace(
        cfg=cfg,
        dataset_facts=facts,
        width=int(cfg.width),
    )
    model = EventMotionModel.from_config(resolved.config)
    _attach_runtime_assets(cfg, model)
    model.load_state_dict(dict(model_state), strict=True)
    final_state = _clone_state(model)
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
        state_dict=final_state,
        train_policy=shell._build_posttrain_train_policy_manifest(cfg=cfg, train_mode=train_mode),
    )
    payload_updates = {
        "model": final_state,
        "posttrain_cfg": _cfg_to_jsonable(cfg),
        "resolved_build_manifest": manifest,
        "resolved_build_manifest_hash": compute_resolved_build_manifest_hash(manifest),
        "strict_current_model_build": True,
    }
    payload_updates.update(metadata)
    verify_hash = compute_weights_hash(final_state)
    if verify_hash != payload_updates["fingerprints"]["weights_hash"]:
        raise RuntimeError(f"weights hash mismatch after refingerprint: {verify_hash}")
    return payload_updates, verify_hash


def _phase_z_carrier_column_span(
    *,
    in_features: int,
    contact_dim: int,
    use_phase_z: bool,
    phase_z_mode: str,
) -> tuple[int, int]:
    if not bool(use_phase_z):
        raise RuntimeError("phase-z carrier patch requires direct_pose_use_phase_z=true")
    phase_dim = int(2 * int(contact_dim))
    if phase_dim <= 0:
        raise RuntimeError(f"phase-z carrier patch requires positive phase_dim, got contact_dim={int(contact_dim)}")
    width = int(in_features)
    if width < phase_dim:
        raise RuntimeError(
            f"tensor width cannot contain trailing phase-z carrier cols: width={width}, phase_dim={phase_dim}, "
            f"phase_z_mode={phase_z_mode!r}"
        )
    return int(width - phase_dim), int(width)


def _resolve_phase_z_carrier_column_span(*, target_config: Path, in_features: int) -> tuple[int, int]:
    cfg = _posttrain._cfg_from_payload(_load_json(target_config))
    facts = _load_dataset_facts(cfg)
    phase_mode = str(getattr(cfg, "direct_pose_phase_z_mode", "concat") or "concat").strip().lower()
    return _phase_z_carrier_column_span(
        in_features=int(in_features),
        contact_dim=int(facts.contact_dim),
        use_phase_z=bool(getattr(cfg, "direct_pose_use_phase_z", False)),
        phase_z_mode=phase_mode,
    )


def _rewrite_handoff_phase_z_carrier_variant(
    *,
    ckpt_path: Path,
    target_config: Path,
    variant: str,
    donor_ckpt: Path | None,
    report_path: Path,
) -> dict[str, Any]:
    phase_z_carrier_policy = _PHASE_Z_CARRIER_POLICY_BY_VARIANT.get(variant, variant)
    if variant == "baseline":
        report = {
            "variant": variant,
            "phase_z_carrier_policy": phase_z_carrier_policy,
            "patched_phase_z_carrier_tensors": [],
            "donor_ckpt": None,
            "weights_hash_after": None,
        }
        _dump_json(report_path, report)
        return report

    payload = torch.load(ckpt_path, map_location="cpu")
    model_state = _extract_model_state(payload)
    donor_state: dict[str, Any] | None = None
    if variant == "donor-new-cols":
        if donor_ckpt is None:
            raise RuntimeError("donor-new-cols requires --donor-replace-ckpt")
        donor_payload = torch.load(donor_ckpt, map_location="cpu")
        donor_state = _extract_model_state(donor_payload)

    changed: list[dict[str, Any]] = []
    for key in PATCH_KEYS:
        value = model_state.get(key)
        if not torch.is_tensor(value) or value.ndim != 2:
            raise RuntimeError(f"expected 2D tensor for {key}")
        carrier_start, carrier_end = _resolve_phase_z_carrier_column_span(
            target_config=target_config,
            in_features=int(value.shape[1]),
        )
        new_value = value.detach().clone()
        before = new_value[:, carrier_start:carrier_end].detach().clone()
        if variant == "zero-new-cols":
            new_value[:, carrier_start:carrier_end] = 0.0
        elif variant == "donor-new-cols":
            assert donor_state is not None
            donor_value = donor_state.get(key)
            if not torch.is_tensor(donor_value) or donor_value.ndim != 2:
                raise RuntimeError(f"donor missing 2D tensor for {key}")
            if int(donor_value.shape[0]) != int(new_value.shape[0]):
                raise RuntimeError(
                    f"donor tensor shape mismatch for {key}: donor={tuple(donor_value.shape)} target={tuple(new_value.shape)}"
                )
            donor_start, donor_end = _resolve_phase_z_carrier_column_span(
                target_config=target_config,
                in_features=int(donor_value.shape[1]),
            )
            new_value[:, carrier_start:carrier_end] = donor_value[:, donor_start:donor_end]
        else:
            raise RuntimeError(f"unsupported variant: {variant}")
        model_state[key] = new_value
        changed.append(
            {
                "tensor_key": key,
                "phase_z_carrier_col_start": int(carrier_start),
                "phase_z_carrier_col_end": int(carrier_end),
                "phase_z_carrier_col_count": int(carrier_end - carrier_start),
                "before_phase_z_carrier_abs_mean": float(before.abs().mean()),
                "after_phase_z_carrier_abs_mean": float(new_value[:, carrier_start:carrier_end].abs().mean()),
                "before_phase_z_carrier_abs_sum": float(before.abs().sum()),
                "after_phase_z_carrier_abs_sum": float(new_value[:, carrier_start:carrier_end].abs().sum()),
            }
        )

    payload_updates, verify_hash = _build_metadata_for_state(
        cfg_payload=_load_json(target_config),
        model_state=model_state,
    )
    payload.update(payload_updates)
    migration_meta = dict(payload.get("migration_meta") or {})
    migration_meta["strict_replace_phasez_boundary_probe"] = {
        "variant": variant,
        "phase_z_carrier_policy": phase_z_carrier_policy,
        "target_config": str(target_config.expanduser()),
        "patched_phase_z_carrier_tensors": changed,
        "donor_ckpt": str(donor_ckpt.expanduser()) if donor_ckpt is not None else None,
        "policy": "post-contract trailing phase-z carrier rewrite; re-fingerprinted strict load-ready final model state",
    }
    payload["migration_meta"] = migration_meta
    torch.save(payload, ckpt_path)

    report = {
        "variant": variant,
        "phase_z_carrier_policy": phase_z_carrier_policy,
        "patched_phase_z_carrier_tensors": changed,
        "donor_ckpt": str(donor_ckpt.expanduser()) if donor_ckpt is not None else None,
        "weights_hash_after": payload["fingerprints"]["weights_hash"],
        "verify_weights_hash": verify_hash,
    }
    _dump_json(report_path, report)
    return report


def _train(config: Path, variant_root: Path) -> None:
    _run(
        [*_python_cmd(), "-m", "train.posttrain", "--config", config],
        log_path=variant_root / "logs/replace_train.log",
        run_root=variant_root,
    )


def _eval_ckpt(*, ckpt: Path, label: str, variant_root: Path) -> Path:
    out_dir = variant_root / "evals" / label / "eval_model_source"
    group_json = out_dir.parent / "group_summary.json"
    _run(
        [
            *_python_cmd(),
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            TEACHER_JSON,
            "--model",
            ckpt,
            "--encoder-bundle",
            ENCODER_BUNDLE,
            "--out",
            out_dir,
            "--device",
            "cpu",
            "--contacts_meas_source",
            "model",
            "--log_contacts",
            "--export_direct_arm_probe",
            "--export_joint_direct_geolocal_series",
            "--rounds",
            "5",
            "--depth",
            "3",
            "--time-index-mode",
            "cycle",
            "--event_clock",
            "on",
            "--contact_plan_init_mode",
            "learnable",
            "--phase_reset_source",
            "none",
            "--force",
        ],
        log_path=out_dir.parent / "eval.log",
        run_root=variant_root,
    )
    _run(
        [
            sys.executable,
            SUMMARY_TOOL,
            out_dir / "Walk_F_freerun_cycles.json",
            "--cycle_gte",
            "1",
            "--drop_wrap",
            "--out",
            group_json,
        ],
        log_path=out_dir.parent / "group_summary.log",
        run_root=variant_root,
    )
    return group_json


def _group_means(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    groups = payload.get("groups", {})
    return {key: float(groups.get(key, {}).get("mean", float("nan"))) for key in GROUP_KEYS}


def _delta(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {key: float(b.get(key, float("nan")) - a.get(key, float("nan"))) for key in GROUP_KEYS}


def _ckpt_model_state(path: Path) -> Mapping[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), Mapping):
        return ckpt["model"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), Mapping):
        return ckpt["model_state_dict"]
    if isinstance(ckpt, Mapping):
        return ckpt
    raise RuntimeError(f"checkpoint state is not a mapping: {path}")


def _checkpoint_tensor_diffs(a_path: Path, b_path: Path) -> list[str]:
    a_state = _ckpt_model_state(a_path)
    b_state = _ckpt_model_state(b_path)
    changed: list[str] = []
    for key in sorted(set(str(k) for k in a_state.keys()) & set(str(k) for k in b_state.keys())):
        a_value = a_state[key]
        b_value = b_state[key]
        if not (torch.is_tensor(a_value) and torch.is_tensor(b_value)):
            continue
        if tuple(a_value.shape) != tuple(b_value.shape):
            changed.append(key)
            continue
        if not torch.equal(a_value, b_value):
            changed.append(key)
    return changed


def _train_log_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    json_path = None
    match = re.search(r"\[posttrain\]\[OK\] saved: (.+)", text)
    if match:
        ckpt_last = Path(match.group(1).strip())
        suffix = ckpt_last.name[len("ckpt_last_") : -len(".pth")]
        json_path = ckpt_last.parent / f"posttrain_log_{suffix}.json"
    trainable_match = re.search(r"\[posttrain\] trainable=(\d+) params: ([^\n]+)", text)
    return {
        "log": str(path),
        "posttrain_json": str(json_path) if json_path is not None else None,
        "contains_chain_hop_report_only_policy": "chain-hop-report-only" in text,
        "contains_policy_strict_current": "policy=strict-current" in text,
        "contains_legacy_checkpoint_compat_true": "legacy_checkpoint_compat=true" in text,
        "contains_strict_shape_validation": "strict current model checkpoint shape validation passed" in text,
        "trainable_count": int(trainable_match.group(1)) if trainable_match else None,
        "trainable_preview": trainable_match.group(2).strip() if trainable_match else None,
    }


def _run_variant(
    *,
    variant: str,
    base_config: Path,
    source_70a_ckpt: Path,
    donor_replace_ckpt: Path | None,
    run_root: Path,
    steps_per_epoch: int,
) -> dict[str, Any]:
    variant_root = run_root / variant
    variant_root.mkdir(parents=True, exist_ok=False)
    run_name = f"replace_phasez_{variant}_{run_root.name}"
    handoff_ckpt = variant_root / "handoffs/70a_to_replace_strict_contract.pth"
    contract_report = variant_root / "handoffs/contractize_70a_to_replace.json"
    config = _make_config(
        base_config=base_config,
        variant_root=variant_root,
        handoff_ckpt=handoff_ckpt,
        run_name=run_name,
        steps_per_epoch=steps_per_epoch,
    )
    _contractize(
        source_ckpt=source_70a_ckpt,
        target_config=config,
        handoff_ckpt=handoff_ckpt,
        report=contract_report,
        variant_root=variant_root,
    )
    _rewrite_contractize_phase_z_report(contract_report)
    rewrite_report = _rewrite_handoff_phase_z_carrier_variant(
        ckpt_path=handoff_ckpt,
        target_config=config,
        variant=variant,
        donor_ckpt=donor_replace_ckpt,
        report_path=variant_root / "handoffs/phase_z_carrier_patch_report.json",
    )
    _train(config, variant_root)

    out_dir = variant_root / "checkpoints"
    step0 = out_dir / f"ckpt_step_000000_{run_name}.pth"
    step20 = out_dir / f"ckpt_step_000020_{run_name}.pth"
    last = out_dir / f"ckpt_last_{run_name}.pth"
    for path in (step0, step20, last):
        if not path.is_file():
            raise RuntimeError(f"missing expected checkpoint: {path}")

    group0 = _eval_ckpt(ckpt=step0, label="step_000000", variant_root=variant_root)
    group20 = _eval_ckpt(ckpt=step20, label="step_000020", variant_root=variant_root)
    group_last = _eval_ckpt(ckpt=last, label="last", variant_root=variant_root)
    g0 = _group_means(group0)
    g20 = _group_means(group20)
    glast = _group_means(group_last)

    train_summary = _train_log_summary(variant_root / "logs/replace_train.log")
    changed_0_20 = _checkpoint_tensor_diffs(step0, step20)
    changed_20_last = _checkpoint_tensor_diffs(step20, last)
    summary = {
        "variant": variant,
        "run_root": str(variant_root),
        "config": str(config),
        "contractized_handoff": str(handoff_ckpt),
        "contractize_report": str(contract_report),
        "phase_z_carrier_patch_report": str(variant_root / "handoffs/phase_z_carrier_patch_report.json"),
        "source_70a_ckpt": str(source_70a_ckpt.expanduser()),
        "donor_replace_ckpt": str(donor_replace_ckpt.expanduser()) if donor_replace_ckpt is not None else None,
        "strict_policy": {
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "contains_chain_hop_report_only_policy": bool(train_summary["contains_chain_hop_report_only_policy"]),
            "contains_policy_strict_current": bool(train_summary["contains_policy_strict_current"]),
            "contains_strict_shape_validation": bool(train_summary["contains_strict_shape_validation"]),
        },
        "train": train_summary,
        "handoff_phase_z_carrier_patch": rewrite_report,
        "evals": {
            "step0_group_summary": str(group0),
            "step20_group_summary": str(group20),
            "last_group_summary": str(group_last),
            "step0": g0,
            "step20": g20,
            "last": glast,
            "delta_step20_minus_step0": _delta(g0, g20),
            "delta_last_minus_step0": _delta(g0, glast),
            "delta_last_minus_step20": _delta(g20, glast),
        },
        "checkpoint_tensor_diffs": {
            "step0_to_step20": changed_0_20,
            "step20_to_last": changed_20_last,
        },
    }
    _dump_json(variant_root / "probe_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict replace boundary probes for phase-z carrier handling."
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--source-70a-ckpt", type=Path, default=DEFAULT_SOURCE_70A_CKPT)
    parser.add_argument("--donor-replace-ckpt", type=Path, default=DEFAULT_DONOR_REPLACE_CKPT)
    parser.add_argument("--variant", action="append", choices=("baseline", "zero-new-cols", "donor-new-cols"))
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()

    variants = tuple(args.variant) if args.variant else DEFAULT_VARIANTS
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or (ROOT / "debug_output" / f"_tmp_strict_replace_phasez_boundary_probe_{stamp}")).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=False)

    run_summary = {
        "run_root": str(run_root),
        "base_config": str(args.base_config.expanduser()),
        "source_70a_ckpt": str(args.source_70a_ckpt.expanduser()),
        "donor_replace_ckpt": str(args.donor_replace_ckpt.expanduser()) if args.donor_replace_ckpt is not None else None,
        "variants": {},
    }
    for variant in variants:
        donor = args.donor_replace_ckpt.expanduser() if variant == "donor-new-cols" and args.donor_replace_ckpt else None
        run_summary["variants"][variant] = _run_variant(
            variant=variant,
            base_config=args.base_config.expanduser(),
            source_70a_ckpt=args.source_70a_ckpt.expanduser(),
            donor_replace_ckpt=donor,
            run_root=run_root,
            steps_per_epoch=int(args.steps_per_epoch),
        )

    _dump_json(run_root / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
