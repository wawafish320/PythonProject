#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain as _posttrain
from train.checkpoint.contract import compute_resolved_build_manifest_hash
from train.configuration.model_build import DatasetModelFacts, resolve_current_model_build_config_with_trace
from train.configuration.norm_spec import merge_norm_spec
from train.data.dataset import build_motion_dataset
from tools import run_strict_70r_trunkfull_probe as trunk_probe

CPU_WRAPPER = ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
SUMMARY_TOOL = ROOT / "tools/phasea_group_summary.py"
TEACHER_JSON = ROOT / "validate/teacher_batches/Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models/motion_encoder_equiv.pt.best.pt"

TEMPLATE_RUN_ROOT = ROOT / "debug_output/_tmp_strict_stageB_finalstate_20260427_080658"
TEMPLATE_CONFIG_DIR = TEMPLATE_RUN_ROOT / "stageB_strict/configs"
STRICT_DONOR_CKPT = ROOT / "debug_output/_tmp_strict_fingerprint_contract_fix_20260427_075813/ckpts/basetrain_donor_strict_contract_finalstate.pth"
RAW_BASETRAIN_SOURCE = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/models/basetrain"
    / "fresh_tail_top7_basetrain_20260426_173158/ckpt_last_fresh_tail_top7_basetrain_20260426_173158.pth"
)
DIRECT_POSE_DONOR_STEP0 = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/70R_lr_probe/lr1e4_step20/checkpoints"
    / "ckpt_step_000000_WalkF_stage7_70R_lr1e4_step20_20260426_173158.pth"
)
PREVIOUS_DIRECT_70R_STEP0_GROUP = TEMPLATE_RUN_ROOT / "stageB_strict/evals/70R/step_000000/group_summary.json"
PREVIOUS_DIRECT_70R_STEP20_GROUP = TEMPLATE_RUN_ROOT / "stageB_strict/evals/70R/step_000020/group_summary.json"
CONTACT_PLAN_PREFLIGHT_AUDIT = (
    ROOT
    / "debug_output/_tmp_strict_resolved_config_migration_preflight_20260426_235509/preflight/checkpoint_contract_audit.json"
)

GROUP_KEYS = ("all_ex_root", "leg", "nonleg", "arm", "else")
METRICS = ("mean", "p50", "p90", "p95")

_RESOLVED_CACHE: dict[Path, dict[str, Any]] = {}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def env(run_root: Path) -> dict[str, str]:
    out = dict(os.environ)
    out["PYTHONPATH"] = str(ROOT)
    out["PYTHONUNBUFFERED"] = "1"
    out["MPLBACKEND"] = "Agg"
    out["RUNTIME_SEMANTICS_TRACE_EVENTS"] = str(run_root / "runtime_semantics_events.jsonl")
    return out


def python_cmd() -> list[object]:
    if CPU_WRAPPER.is_file():
        return [CPU_WRAPPER]
    return [sys.executable]


def command_text(cmd: Sequence[object]) -> str:
    return shlex.join([str(part) for part in cmd])


def run_logged(*, label: str, cmd: Sequence[object], log_path: Path, run_root: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = [str(part) for part in cmd]
    started = time.time()
    print(f"\n[RUN:{label}] {command_text(cmd_list)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(cmd_list)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(ROOT),
            env=env(run_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = int(proc.wait())
        elapsed = time.time() - started
        log.write(f"\n[exit_code] {code}\n[elapsed_sec] {elapsed:.3f}\n")
    return code


def require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing required file: {path}")


def git_status_lines() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        env=env(ROOT / "debug_output"),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def load_template_config(name: str) -> dict[str, Any]:
    return load_json(TEMPLATE_CONFIG_DIR / f"{name}.json")


def canonicalize_direct_pose_config_fields(payload: dict[str, Any]) -> None:
    hidden_override = payload.pop("direct_pose_hidden_override", None)
    if hidden_override is not None and "direct_pose_hidden" not in payload:
        payload["direct_pose_hidden"] = int(hidden_override)

    meas_mode_override = payload.pop("direct_pose_meas_mode_override", None)
    if meas_mode_override is not None and str(meas_mode_override).strip():
        payload["direct_pose_meas_mode"] = str(meas_mode_override).strip()
    elif "direct_pose_meas_mode" not in payload:
        payload["direct_pose_meas_mode"] = "concat"


def build_stage_config(
    *,
    name: str,
    run_root: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = load_template_config(name)
    canonicalize_direct_pose_config_fields(payload)
    for key in ("contact_plan_init_mode", "contact_plan_init_hidden", "contact_plan_init_dropout"):
        payload.pop(key, None)
    if name != "replace":
        for key in ("direct_pose_use_phase_z", "direct_pose_phase_z_mode"):
            payload.pop(key, None)
    if name != "lambda":
        for key in tuple(payload.keys()):
            if str(key).startswith("lambda_fusion_"):
                payload.pop(str(key), None)
        payload["train_lambda_head"] = False
        payload.pop("lambda_fusion_enable", None)
    else:
        payload["train_lambda_head"] = True
        payload["lambda_fusion_enable"] = True
    payload.update(
        {
            "ckpt_in": str(ckpt_in),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "event_clock": "on",
            "width": 512,
        }
    )
    if extra:
        payload.update(extra)
    config_path = run_root / "configs" / f"{name}.json"
    dump_json(config_path, payload)
    return config_path


def expected_last(out_dir: Path, run_name: str) -> Path:
    return out_dir / f"ckpt_last_{run_name}.pth"


def expected_step(out_dir: Path, run_name: str, step: int) -> Path:
    return out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"


def checkpoint_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return {str(key): value for key, value in payload["model"].items()}
    if isinstance(payload, Mapping) and isinstance(payload.get("model_state_dict"), Mapping):
        return {str(key): value for key, value in payload["model_state_dict"].items()}
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    raise RuntimeError(f"checkpoint payload is not mapping: {type(payload).__name__}")


def ckpt_posttrain_cfg(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    raw = ckpt.get("posttrain_cfg", {}) if isinstance(ckpt, Mapping) else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    ckpt = torch.load(path, map_location="cpu")
    state = checkpoint_state(ckpt)
    contract = ckpt.get("checkpoint_contract", {}) if isinstance(ckpt, Mapping) else {}
    return {
        "path": str(path),
        "exists": True,
        "strict_current_model_build": ckpt.get("strict_current_model_build") if isinstance(ckpt, Mapping) else None,
        "has_legacy_checkpoint_compat_field": bool(isinstance(ckpt, Mapping) and "legacy_checkpoint_compat" in ckpt),
        "resolved_build_manifest_hash": ckpt.get("resolved_build_manifest_hash") if isinstance(ckpt, Mapping) else None,
        "checkpoint_contract_name": contract.get("name") if isinstance(contract, Mapping) else None,
        "checkpoint_contract_version": contract.get("version") if isinstance(contract, Mapping) else None,
        "model_key_count": len(state),
        "contact_plan_init_head_keys": sorted(key for key in state if key.startswith("contact_plan_init_head.")),
        "lambda_fusion_head_keys": sorted(key for key in state if key.startswith("lambda_fusion_head.")),
    }


def scan_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        "log": str(path),
        "contains_chain_hop_waiver": "chain_hop-waiver" in text,
        "contains_fingerprint_waiver": "fingerprint waiver" in text.lower() or "skip fingerprint" in text.lower(),
        "contains_legacy_checkpoint_compat_true": "legacy_checkpoint_compat=true" in text,
        "contains_direct_pose_temp_compat": "direct_pose temp compat" in text.lower() or "direct pose input dim" in text.lower(),
        "contains_shape_inference_semantics": "shape/posttrain_cfg inference" in text.lower() or "shape inference" in text.lower(),
        "contains_lambda_rollout_override": "lambda_fusion_use_rollout_step=" in text and "overriding to true" in text.lower(),
        "contains_policy_strict_current": "policy=strict-current" in text,
        "contains_strict_shape_validation": "strict current model checkpoint shape validation passed" in text,
        "contains_event_clock_auto": "event_clock=auto" in text or "event_clock auto" in text.lower(),
    }


def assert_no_forbidden_log_tokens(scan: Mapping[str, Any], *, label: str) -> None:
    forbidden = (
        "contains_chain_hop_waiver",
        "contains_fingerprint_waiver",
        "contains_legacy_checkpoint_compat_true",
        "contains_direct_pose_temp_compat",
        "contains_shape_inference_semantics",
        "contains_lambda_rollout_override",
        "contains_event_clock_auto",
    )
    bad = {key: bool(scan.get(key)) for key in forbidden if bool(scan.get(key))}
    if bad:
        raise RuntimeError(f"{label} used forbidden compat path: {bad}")


def load_dataset_facts(cfg: Any) -> DatasetModelFacts:
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
    return DatasetModelFacts.from_dataset(ds, context="strict_stageb_resolvedcfg_rerun.dataset")


def resolve_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    cached = _RESOLVED_CACHE.get(config_path)
    if cached is not None:
        return cached
    payload = load_json(config_path)
    cfg = _posttrain._cfg_from_payload(payload)
    facts = load_dataset_facts(cfg)
    resolved = resolve_current_model_build_config_with_trace(
        cfg=cfg,
        dataset_facts=facts,
        width=int(cfg.width),
    )
    manifest = resolved.manifest()
    item = {
        "config_path": str(config_path),
        "payload": payload,
        "cfg": cfg,
        "manifest": manifest,
        "manifest_hash": compute_resolved_build_manifest_hash(manifest),
    }
    _RESOLVED_CACHE[config_path] = item
    return item


def runtime_effective_fields(*, config_path: Path, ckpt_path: Path) -> dict[str, Any]:
    resolved = resolve_config(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    manifest_cfg = ((ckpt.get("resolved_build_manifest") or {}).get("config") or {}) if isinstance(ckpt, Mapping) else {}
    post_cfg = ckpt.get("posttrain_cfg", {}) if isinstance(ckpt, Mapping) else {}
    if not isinstance(post_cfg, Mapping):
        post_cfg = {}
    payload_cfg = resolved["payload"]
    contract = ckpt.get("checkpoint_contract", {}) if isinstance(ckpt, Mapping) else {}
    state = checkpoint_state(ckpt)
    return {
        "contact_plan_init_mode": manifest_cfg.get("contact_plan_init_mode"),
        "event_clock_enabled": str(payload_cfg.get("event_clock", "")).strip().lower() == "on",
        "event_clock_effective": bool(manifest_cfg.get("use_event_clock")),
        "direct_pose_enabled": bool(payload_cfg.get("direct_pose_enable")),
        "direct_pose_effective": bool(manifest_cfg.get("direct_pose_enable")),
        "contact_plan_enabled": bool(payload_cfg.get("contact_plan_enable")),
        "contact_plan_effective": bool(manifest_cfg.get("contact_plan_enable")),
        "lambda_fusion_enabled": bool(payload_cfg.get("lambda_fusion_enable")),
        "lambda_fusion_effective": bool(manifest_cfg.get("lambda_fusion_enable")),
        "width": int(manifest_cfg.get("hidden_dim", payload_cfg.get("width", 0))),
        "load_context": post_cfg.get("load_context"),
        "strict_current_model_build": bool(ckpt.get("strict_current_model_build", post_cfg.get("strict_current_model_build", False))),
        "has_legacy_checkpoint_compat_field": bool(
            ("legacy_checkpoint_compat" in ckpt if isinstance(ckpt, Mapping) else False)
            or ("legacy_checkpoint_compat" in post_cfg if isinstance(post_cfg, Mapping) else False)
        ),
        "checkpoint_contract_version": contract.get("version") if isinstance(contract, Mapping) else None,
        "resolved_build_manifest_hash": ckpt.get("resolved_build_manifest_hash"),
        "resolved_build_manifest_matches_target": ckpt.get("resolved_build_manifest_hash") == resolved["manifest_hash"],
        "resolved_build_manifest_target_hash": resolved["manifest_hash"],
        "contact_plan_init_head_key_count": sum(1 for key in state if key.startswith("contact_plan_init_head.")),
        "lambda_fusion_head_key_count": sum(1 for key in state if key.startswith("lambda_fusion_head.")),
    }


def write_resolved_artifact(
    *,
    out_path: Path,
    label: str,
    kind: str,
    config_path: Path,
    ckpt_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "label": label,
        "kind": kind,
        "config_path": str(config_path),
        "ckpt_path": str(ckpt_path),
        "runtime_effective": runtime_effective_fields(config_path=config_path, ckpt_path=ckpt_path),
        "checkpoint_summary": inspect_checkpoint(ckpt_path),
    }
    if extra:
        artifact["extra"] = dict(extra)
    dump_json(out_path, artifact)
    return artifact


def run_posttrain(*, stage: str, config_path: Path, log_path: Path, run_root: Path) -> int:
    return run_logged(
        label=stage,
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_path],
        log_path=log_path,
        run_root=run_root,
    )


def run_contractize(
    *,
    label: str,
    source_ckpt: Path,
    target_config: Path,
    out_ckpt: Path,
    report_path: Path,
    run_root: Path,
    tensor_donor: Path | None = None,
    transplant_prefixes: Sequence[str] = (),
    allow_missing_prefixes: Sequence[str] = (),
) -> None:
    cmd: list[object] = [
        sys.executable,
        ROOT / "tools/contractize_strict_posttrain_handoff.py",
        "--source",
        source_ckpt,
        "--target-config",
        target_config,
        "--out",
        out_ckpt,
        "--report",
        report_path,
        "--label",
        label,
    ]
    for prefix in allow_missing_prefixes:
        cmd.extend(["--allow-missing-prefix", str(prefix)])
    if tensor_donor is not None:
        cmd.extend(["--tensor-donor", tensor_donor])
        for prefix in transplant_prefixes:
            cmd.extend(["--transplant-prefix", str(prefix)])
    rc = run_logged(
        label=f"contractize:{label}",
        cmd=cmd,
        log_path=run_root / "logs" / f"contractize_{label}.log",
        run_root=run_root,
    )
    if rc != 0:
        raise RuntimeError(f"contractize failed for {label} with exit={rc}")
    require_file(out_ckpt, label=f"contractize:{label}")
    require_file(report_path, label=f"contractize:{label}:report")


def build_direct_handoff_report(
    *,
    label: str,
    source_ckpt: Path,
    target_config: Path,
    out_path: Path,
) -> dict[str, Any]:
    resolved = resolve_config(target_config)
    summary = inspect_checkpoint(source_ckpt)
    report = {
        "label": label,
        "kind": "direct_strict_load",
        "source_ckpt": str(source_ckpt),
        "target_config": str(target_config),
        "source_manifest_hash": summary.get("resolved_build_manifest_hash"),
        "target_manifest_hash": resolved["manifest_hash"],
        "source_strict_current_model_build": summary.get("strict_current_model_build"),
        "source_has_legacy_checkpoint_compat_field": summary.get("has_legacy_checkpoint_compat_field"),
        "status": "pass",
        "runtime_effective_target": {
            "contact_plan_init_mode": resolved["manifest"]["config"].get("contact_plan_init_mode"),
            "event_clock_effective": bool(resolved["manifest"]["config"].get("use_event_clock")),
            "direct_pose_effective": bool(resolved["manifest"]["config"].get("direct_pose_enable")),
            "contact_plan_effective": bool(resolved["manifest"]["config"].get("contact_plan_enable")),
            "lambda_fusion_effective": bool(resolved["manifest"]["config"].get("lambda_fusion_enable")),
            "width": int(resolved["manifest"]["config"].get("hidden_dim", 0)),
            "load_context": resolved["payload"].get("load_context"),
            "strict_current_model_build": bool(resolved["payload"].get("strict_current_model_build")),
            "checkpoint_contract_version": summary.get("checkpoint_contract_version"),
            "resolved_build_manifest_hash": resolved["manifest_hash"],
        },
    }
    if summary.get("resolved_build_manifest_hash") != resolved["manifest_hash"]:
        report["status"] = "fail"
        report["reason"] = "resolved_build_manifest_hash mismatch"
    if summary.get("strict_current_model_build") is not True:
        report["status"] = "fail"
        report["reason"] = "source checkpoint is not strict_current_model_build=true"
    if summary.get("has_legacy_checkpoint_compat_field"):
        report["status"] = "fail"
        report["reason"] = "source checkpoint still carries retired legacy_checkpoint_compat metadata"
    dump_json(out_path, report)
    if report["status"] != "pass":
        raise RuntimeError(f"{label} failed: {report}")
    return report


def group_means(path: Path) -> dict[str, float]:
    payload = load_json(path)
    groups = payload.get("groups", {})
    return {key: float(groups.get(key, {}).get("mean", float("nan"))) for key in GROUP_KEYS}


def delta_groups(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> dict[str, float]:
    return {key: float(rhs.get(key, float("nan")) - lhs.get(key, float("nan"))) for key in GROUP_KEYS}


def compare_group_summary(current_path: Path, baseline_path: Path, *, baseline_label: str, current_label: str) -> dict[str, Any]:
    current = load_json(current_path).get("groups", {})
    baseline = load_json(baseline_path).get("groups", {})
    out: dict[str, Any] = {
        "baseline_label": baseline_label,
        "current_label": current_label,
        "baseline": str(baseline_path),
        "current": str(current_path),
        "groups": {},
    }
    for group in GROUP_KEYS:
        out["groups"][group] = {}
        for metric in METRICS:
            b = float(baseline.get(group, {}).get(metric, float("nan")))
            c = float(current.get(group, {}).get(metric, float("nan")))
            out["groups"][group][metric] = {"baseline": b, "current": c, "delta": c - b}
    return out


def eval_checkpoint(*, label: str, ckpt: Path, out_dir: Path, run_root: Path) -> Path:
    log_path = out_dir.parent / "eval.log"
    cmd: list[object] = [
        *python_cmd(),
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
    ]
    rc = run_logged(label=f"eval:{label}", cmd=cmd, log_path=log_path, run_root=run_root)
    if rc != 0:
        raise RuntimeError(f"eval failed for {label} with exit={rc}")
    scan = scan_log(log_path)
    assert_no_forbidden_log_tokens(scan, label=f"eval:{label}")
    eval_json = out_dir / "Walk_F_freerun_cycles.json"
    require_file(eval_json, label=f"eval:{label}")
    group_json = out_dir.parent / "group_summary.json"
    rc = run_logged(
        label=f"group:{label}",
        cmd=[sys.executable, SUMMARY_TOOL, eval_json, "--cycle_gte", "1", "--drop_wrap", "--out", group_json],
        log_path=out_dir.parent / "group_summary.log",
        run_root=run_root,
    )
    if rc != 0:
        raise RuntimeError(f"group summary failed for {label} with exit={rc}")
    require_file(group_json, label=f"group:{label}")
    return group_json


def strip_checkpoint_prefixes(*, source_ckpt: Path, output_ckpt: Path, prefixes: Sequence[str]) -> dict[str, Any]:
    payload = torch.load(source_ckpt, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"strip source is not a mapping: {source_ckpt}")
    payload = dict(payload)
    state = checkpoint_state(payload)
    removed = sorted(key for key in state if any(key.startswith(prefix) for prefix in prefixes))
    filtered = {key: value for key, value in state.items() if key not in removed}
    if "model" in payload and isinstance(payload["model"], Mapping):
        payload["model"] = filtered
    elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
        payload["model_state_dict"] = filtered
    else:
        payload = filtered
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_ckpt)
    return {
        "source_ckpt": str(source_ckpt),
        "output_ckpt": str(output_ckpt),
        "strip_prefixes": [str(prefix) for prefix in prefixes],
        "removed_keys": removed,
        "removed_count": len(removed),
    }


def run_strip_audits(
    *,
    run_root: Path,
    stage6_config: Path,
    config_72: Path,
    current_lambda_ckpt: Path,
    current_strict_donor: Path,
) -> dict[str, Any]:
    audit_root = run_root / "strip_audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    preflight = load_json(CONTACT_PLAN_PREFLIGHT_AUDIT)
    basetrain_audit = preflight["migrated_checkpoints"]["basetrain"]
    donor_summary = inspect_checkpoint(current_strict_donor)
    contact_plan_strip = {
        "rule": "contact_plan_init_mode=learnable => strip contact_plan_init_head.*",
        "target_config": str(stage6_config),
        "target_contact_plan_init_mode": resolve_config(stage6_config)["manifest"]["config"].get("contact_plan_init_mode"),
        "source_raw_checkpoint": basetrain_audit.get("source"),
        "migration_output_checkpoint": basetrain_audit.get("output"),
        "removed_keys": basetrain_audit.get("removed_keys", []),
        "strict_contract_ready": bool(basetrain_audit.get("strict_contract_ready")),
        "current_chain_input_checkpoint": str(current_strict_donor),
        "current_chain_input_contact_plan_init_head_keys": donor_summary.get("contact_plan_init_head_keys", []),
        "status": "pass",
        "verdict": "safe_to_strip",
        "reason": "resolved config fixes contact_plan_init_mode=learnable and the strict donor used by this rerun no longer carries contact_plan_init_head.* weights",
    }

    lambda_stripped_source = audit_root / "lambda_step200_strip_lambda_fusion_head_source.pth"
    strip_meta = strip_checkpoint_prefixes(
        source_ckpt=current_lambda_ckpt,
        output_ckpt=lambda_stripped_source,
        prefixes=("lambda_fusion_head.",),
    )
    lambda_to_72_handoff = audit_root / "lambda_to_72_after_strip_strict_contract.pth"
    lambda_to_72_report = audit_root / "lambda_to_72_after_strip_contractize.json"
    run_contractize(
        label="lambda_to_72_after_strip",
        source_ckpt=lambda_stripped_source,
        target_config=config_72,
        out_ckpt=lambda_to_72_handoff,
        report_path=lambda_to_72_report,
        run_root=run_root,
    )
    lambda_strip_report = load_json(lambda_to_72_report)
    lambda_strip = {
        "rule": "non-lambda stage with lambda_fusion_enable=false => lambda_fusion_head.* strip/ignore",
        "target_config": str(config_72),
        "target_lambda_fusion_enable": bool(resolve_config(config_72)["payload"].get("lambda_fusion_enable")),
        "strip_meta": strip_meta,
        "contractize_report": lambda_strip_report,
        "status": "pass" if lambda_strip_report.get("status") == "pass" else "fail",
        "verdict": "safe_to_strip" if lambda_strip_report.get("status") == "pass" else "keep_runtime_responsibility",
    }

    report = {
        "run_root": str(run_root),
        "contact_plan_init_head": contact_plan_strip,
        "lambda_fusion_head_nonlambda": lambda_strip,
    }
    dump_json(audit_root / "strip_ignore_audit_report.json", report)
    return report


def finalize(
    *,
    run_root: Path,
    stage_ckpts: Mapping[str, Path],
    stage_records: Mapping[str, Any],
    handoff_reports: Mapping[str, Any],
    resolved_artifacts: Mapping[str, Any],
    eval_records: Mapping[str, Any],
    strip_audit: Mapping[str, Any] | None,
    status: str,
    failed_stage: str | None,
    error: str | None,
) -> None:
    all_stage = {
        "status": status,
        "failed_stage": failed_stage,
        "stages": {key: str(path) for key, path in stage_ckpts.items()},
    }
    dump_json(run_root / "all_stage_ckpts.json", all_stage)
    trace = {
        "run_root": str(run_root),
        "status": status,
        "failed_stage": failed_stage,
        "error": error,
        "stage_records": stage_records,
        "handoff_reports": handoff_reports,
        "resolved_config_artifacts": resolved_artifacts,
        "eval_records": eval_records,
        "strip_ignore_audit_report": str(run_root / "strip_audit/strip_ignore_audit_report.json") if strip_audit else None,
    }
    dump_json(run_root / "strict_contract_enforcement_trace.json", trace)
    (run_root / "git_status_after_run.txt").write_text("\n".join(git_status_lines()) + "\n", encoding="utf-8")


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT / "debug_output" / f"_tmp_strict_stageB_resolvedcfg_rerun_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    for subdir in ("configs", "logs", "handoffs", "resolved_config_artifacts", "evals", "strip_audit"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    stage_ckpts: dict[str, Path] = {}
    stage_records: dict[str, Any] = {}
    handoff_reports: dict[str, Any] = {}
    resolved_artifacts: dict[str, Any] = {}
    eval_records: dict[str, Any] = {}
    strip_audit: dict[str, Any] | None = None
    failed_stage: str | None = None

    try:
        stage6_out = run_root / "stage6/checkpoints"
        stage6_name = f"stage6_resolvedcfg_{stamp}"
        config_stage6 = build_stage_config(
            name="stage6",
            run_root=run_root,
            ckpt_in=STRICT_DONOR_CKPT,
            out_dir=stage6_out,
            run_name=stage6_name,
        )
        ckpt_stage6_step360 = expected_step(stage6_out, stage6_name, 360)
        ckpt_stage6_last = expected_last(stage6_out, stage6_name)
        failed_stage = "stage6"
        rc = run_posttrain(stage="stage6", config_path=config_stage6, log_path=run_root / "logs/stage6.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"stage6 failed with exit={rc}")
        require_file(ckpt_stage6_step360, label="stage6_step360")
        stage_ckpts["stage6_step360"] = ckpt_stage6_step360
        stage_ckpts["stage6_last"] = ckpt_stage6_last
        scan = scan_log(run_root / "logs/stage6.log")
        assert_no_forbidden_log_tokens(scan, label="stage6")
        stage_records["stage6"] = {
            "config": str(config_stage6),
            "log": str(run_root / "logs/stage6.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_stage6_step360),
        }
        resolved_artifacts["stage6"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/stage6.json",
            label="stage6",
            kind="stage",
            config_path=config_stage6,
            ckpt_path=ckpt_stage6_step360,
        )

        config_70a = build_stage_config(
            name="70a",
            run_root=run_root,
            ckpt_in=ckpt_stage6_step360,
            out_dir=run_root / "70a/checkpoints",
            run_name=f"70a_resolvedcfg_{stamp}",
        )
        failed_stage = "stage6_to_70a"
        handoff_reports["stage6_to_70a"] = build_direct_handoff_report(
            label="stage6_to_70a",
            source_ckpt=ckpt_stage6_step360,
            target_config=config_70a,
            out_path=run_root / "handoffs/stage6_to_70a_direct_report.json",
        )

        ckpt_70a_last = expected_last(run_root / "70a/checkpoints", f"70a_resolvedcfg_{stamp}")
        failed_stage = "70a"
        rc = run_posttrain(stage="70a", config_path=config_70a, log_path=run_root / "logs/70a.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"70a failed with exit={rc}")
        require_file(ckpt_70a_last, label="70a_last")
        stage_ckpts["70a_last"] = ckpt_70a_last
        scan = scan_log(run_root / "logs/70a.log")
        assert_no_forbidden_log_tokens(scan, label="70a")
        stage_records["70a"] = {
            "config": str(config_70a),
            "log": str(run_root / "logs/70a.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_70a_last),
        }
        resolved_artifacts["70a"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/70a.json",
            label="70a",
            kind="stage",
            config_path=config_70a,
            ckpt_path=ckpt_70a_last,
        )

        config_replace = build_stage_config(
            name="replace",
            run_root=run_root,
            ckpt_in=run_root / "handoffs/70a_to_replace_strict_contract.pth",
            out_dir=run_root / "replace/checkpoints",
            run_name=f"replace_resolvedcfg_{stamp}",
        )
        failed_stage = "70a_to_replace"
        run_contractize(
            label="70a_to_replace",
            source_ckpt=ckpt_70a_last,
            target_config=config_replace,
            out_ckpt=run_root / "handoffs/70a_to_replace_strict_contract.pth",
            report_path=run_root / "handoffs/70a_to_replace_contractize.json",
            run_root=run_root,
        )
        handoff_reports["70a_to_replace"] = load_json(run_root / "handoffs/70a_to_replace_contractize.json")
        resolved_artifacts["70a_to_replace"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/70a_to_replace.json",
            label="70a_to_replace",
            kind="handoff",
            config_path=config_replace,
            ckpt_path=run_root / "handoffs/70a_to_replace_strict_contract.pth",
        )

        ckpt_replace_last = expected_last(run_root / "replace/checkpoints", f"replace_resolvedcfg_{stamp}")
        failed_stage = "replace"
        rc = run_posttrain(stage="replace", config_path=config_replace, log_path=run_root / "logs/replace.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"replace failed with exit={rc}")
        require_file(ckpt_replace_last, label="replace_last")
        stage_ckpts["replace_last"] = ckpt_replace_last
        scan = scan_log(run_root / "logs/replace.log")
        assert_no_forbidden_log_tokens(scan, label="replace")
        stage_records["replace"] = {
            "config": str(config_replace),
            "log": str(run_root / "logs/replace.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_replace_last),
        }
        resolved_artifacts["replace"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/replace.json",
            label="replace",
            kind="stage",
            config_path=config_replace,
            ckpt_path=ckpt_replace_last,
        )

        config_70r = build_stage_config(
            name="70R",
            run_root=run_root,
            ckpt_in=run_root / "handoffs/replace_to_70R_warmstart_bridge_strict_contract.pth",
            out_dir=run_root / "70R/checkpoints",
            run_name=f"70R_resolvedcfg_{stamp}",
            extra={
                "direct_pose_nonleg_train_only": True,
                "direct_pose_nonleg_trunk_mode": "full",
                "save_step_ckpts": "0,1,20",
                "epochs": 1,
                "steps_per_epoch": 60,
            },
        )
        failed_stage = "replace_to_70R"
        run_contractize(
            label="replace_to_70R_warmstart_bridge",
            source_ckpt=ckpt_replace_last,
            target_config=config_70r,
            out_ckpt=run_root / "handoffs/replace_to_70R_warmstart_bridge_strict_contract.pth",
            report_path=run_root / "handoffs/replace_to_70R_warmstart_bridge_contractize.json",
            run_root=run_root,
            tensor_donor=DIRECT_POSE_DONOR_STEP0,
            transplant_prefixes=("direct_pose_",),
        )
        handoff_reports["replace_to_70R"] = load_json(run_root / "handoffs/replace_to_70R_warmstart_bridge_contractize.json")
        resolved_artifacts["replace_to_70R"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/replace_to_70R.json",
            label="replace_to_70R",
            kind="handoff",
            config_path=config_70r,
            ckpt_path=run_root / "handoffs/replace_to_70R_warmstart_bridge_strict_contract.pth",
            extra={
                "bridge_kind": "migration_time_70R_warmstart_bridge",
                "tensor_donor": str(DIRECT_POSE_DONOR_STEP0),
                "transplant_prefixes": ["direct_pose_"],
            },
        )

        ckpt_70r_step0 = expected_step(run_root / "70R/checkpoints", f"70R_resolvedcfg_{stamp}", 0)
        ckpt_70r_step20 = expected_step(run_root / "70R/checkpoints", f"70R_resolvedcfg_{stamp}", 20)
        failed_stage = "70R"
        rc = run_posttrain(stage="70R", config_path=config_70r, log_path=run_root / "logs/70R.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"70R failed with exit={rc}")
        require_file(ckpt_70r_step0, label="70R_step0")
        require_file(ckpt_70r_step20, label="70R_step20")
        stage_ckpts["70R_step0"] = ckpt_70r_step0
        stage_ckpts["70R_step20"] = ckpt_70r_step20
        scan = scan_log(run_root / "logs/70R.log")
        assert_no_forbidden_log_tokens(scan, label="70R")
        train_summary = trunk_probe._train_log_summary(run_root / "logs/70R.log")
        if int(train_summary.get("direct_grad_norm_trunk_finite_count", 0) or 0) <= 0:
            raise RuntimeError("70R trunk grad norms never became finite")
        stage_records["70R"] = {
            "config": str(config_70r),
            "log": str(run_root / "logs/70R.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_70r_step20),
            "train_summary": train_summary,
        }
        resolved_artifacts["70R_step0"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/70R_step0.json",
            label="70R_step0",
            kind="stage_step",
            config_path=config_70r,
            ckpt_path=ckpt_70r_step0,
            extra={"step": 0},
        )
        resolved_artifacts["70R_step20"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/70R_step20.json",
            label="70R_step20",
            kind="stage_step",
            config_path=config_70r,
            ckpt_path=ckpt_70r_step20,
            extra={"step": 20},
        )

        failed_stage = "eval_70R_step0"
        group_70r_step0 = eval_checkpoint(
            label="70R_step0",
            ckpt=ckpt_70r_step0,
            out_dir=run_root / "evals/70R/step_000000/eval_model_source",
            run_root=run_root,
        )
        failed_stage = "eval_70R_step20"
        group_70r_step20 = eval_checkpoint(
            label="70R_step20",
            ckpt=ckpt_70r_step20,
            out_dir=run_root / "evals/70R/step_000020/eval_model_source",
            run_root=run_root,
        )
        eval_records["70R"] = {
            "step0_group_summary": str(group_70r_step0),
            "step20_group_summary": str(group_70r_step20),
            "step0": group_means(group_70r_step0),
            "step20": group_means(group_70r_step20),
            "delta_step20_minus_step0": delta_groups(group_means(group_70r_step0), group_means(group_70r_step20)),
        }
        if PREVIOUS_DIRECT_70R_STEP0_GROUP.is_file() and PREVIOUS_DIRECT_70R_STEP20_GROUP.is_file():
            eval_records["70R"]["old_direct_handoff_reference"] = {
                "step0_group_summary": str(PREVIOUS_DIRECT_70R_STEP0_GROUP),
                "step20_group_summary": str(PREVIOUS_DIRECT_70R_STEP20_GROUP),
                "compare_step0": compare_group_summary(
                    group_70r_step0,
                    PREVIOUS_DIRECT_70R_STEP0_GROUP,
                    baseline_label="old_strictB_direct_handoff_step0",
                    current_label="resolvedcfg_warmstart_bridge_step0",
                ),
                "compare_step20": compare_group_summary(
                    group_70r_step20,
                    PREVIOUS_DIRECT_70R_STEP20_GROUP,
                    baseline_label="old_strictB_direct_handoff_step20",
                    current_label="resolvedcfg_warmstart_bridge_step20",
                ),
            }

        config_71 = build_stage_config(
            name="71",
            run_root=run_root,
            ckpt_in=run_root / "handoffs/70R_to_71_strict_contract.pth",
            out_dir=run_root / "71/checkpoints",
            run_name=f"71_resolvedcfg_{stamp}",
        )
        failed_stage = "70R_to_71"
        run_contractize(
            label="70R_to_71",
            source_ckpt=ckpt_70r_step20,
            target_config=config_71,
            out_ckpt=run_root / "handoffs/70R_to_71_strict_contract.pth",
            report_path=run_root / "handoffs/70R_to_71_contractize.json",
            run_root=run_root,
        )
        handoff_reports["70R_to_71"] = load_json(run_root / "handoffs/70R_to_71_contractize.json")
        resolved_artifacts["70R_to_71"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/70R_to_71.json",
            label="70R_to_71",
            kind="handoff",
            config_path=config_71,
            ckpt_path=run_root / "handoffs/70R_to_71_strict_contract.pth",
        )

        ckpt_71_step120 = expected_step(run_root / "71/checkpoints", f"71_resolvedcfg_{stamp}", 120)
        failed_stage = "71"
        rc = run_posttrain(stage="71", config_path=config_71, log_path=run_root / "logs/71.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"71 failed with exit={rc}")
        require_file(ckpt_71_step120, label="71_step120")
        stage_ckpts["71_step120"] = ckpt_71_step120
        scan = scan_log(run_root / "logs/71.log")
        assert_no_forbidden_log_tokens(scan, label="71")
        stage_records["71"] = {
            "config": str(config_71),
            "log": str(run_root / "logs/71.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_71_step120),
        }
        resolved_artifacts["71"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/71.json",
            label="71",
            kind="stage",
            config_path=config_71,
            ckpt_path=ckpt_71_step120,
        )

        config_72 = build_stage_config(
            name="72",
            run_root=run_root,
            ckpt_in=ckpt_71_step120,
            out_dir=run_root / "72/checkpoints",
            run_name=f"72_resolvedcfg_{stamp}",
        )
        failed_stage = "71_to_72"
        handoff_reports["71_to_72"] = build_direct_handoff_report(
            label="71_to_72",
            source_ckpt=ckpt_71_step120,
            target_config=config_72,
            out_path=run_root / "handoffs/71_to_72_direct_report.json",
        )

        ckpt_72_step150 = expected_step(run_root / "72/checkpoints", f"72_resolvedcfg_{stamp}", 150)
        failed_stage = "72"
        rc = run_posttrain(stage="72", config_path=config_72, log_path=run_root / "logs/72.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"72 failed with exit={rc}")
        require_file(ckpt_72_step150, label="72_step150")
        stage_ckpts["72_step150"] = ckpt_72_step150
        scan = scan_log(run_root / "logs/72.log")
        assert_no_forbidden_log_tokens(scan, label="72")
        stage_records["72"] = {
            "config": str(config_72),
            "log": str(run_root / "logs/72.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_72_step150),
        }
        resolved_artifacts["72"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/72.json",
            label="72",
            kind="stage",
            config_path=config_72,
            ckpt_path=ckpt_72_step150,
        )

        config_lambda = build_stage_config(
            name="lambda",
            run_root=run_root,
            ckpt_in=run_root / "handoffs/72_to_lambda_strict_contract.pth",
            out_dir=run_root / "lambda/checkpoints",
            run_name=f"lambda_resolvedcfg_{stamp}",
        )
        failed_stage = "72_to_lambda"
        run_contractize(
            label="72_to_lambda",
            source_ckpt=ckpt_72_step150,
            target_config=config_lambda,
            out_ckpt=run_root / "handoffs/72_to_lambda_strict_contract.pth",
            report_path=run_root / "handoffs/72_to_lambda_contractize.json",
            run_root=run_root,
            allow_missing_prefixes=("lambda_fusion_head.",),
        )
        handoff_reports["72_to_lambda"] = load_json(run_root / "handoffs/72_to_lambda_contractize.json")
        resolved_artifacts["72_to_lambda"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/72_to_lambda.json",
            label="72_to_lambda",
            kind="handoff",
            config_path=config_lambda,
            ckpt_path=run_root / "handoffs/72_to_lambda_strict_contract.pth",
        )

        ckpt_lambda_step200 = expected_step(run_root / "lambda/checkpoints", f"lambda_resolvedcfg_{stamp}", 200)
        failed_stage = "lambda"
        rc = run_posttrain(stage="lambda", config_path=config_lambda, log_path=run_root / "logs/lambda.log", run_root=run_root)
        if rc != 0:
            raise RuntimeError(f"lambda failed with exit={rc}")
        require_file(ckpt_lambda_step200, label="lambda_step200")
        stage_ckpts["lambda_step200"] = ckpt_lambda_step200
        scan = scan_log(run_root / "logs/lambda.log")
        assert_no_forbidden_log_tokens(scan, label="lambda")
        stage_records["lambda"] = {
            "config": str(config_lambda),
            "log": str(run_root / "logs/lambda.log"),
            "log_scan": scan,
            "checkpoint": inspect_checkpoint(ckpt_lambda_step200),
        }
        resolved_artifacts["lambda"] = write_resolved_artifact(
            out_path=run_root / "resolved_config_artifacts/lambda.json",
            label="lambda",
            kind="stage",
            config_path=config_lambda,
            ckpt_path=ckpt_lambda_step200,
        )

        failed_stage = "eval_lambda_final"
        group_lambda_final = eval_checkpoint(
            label="lambda_final",
            ckpt=ckpt_lambda_step200,
            out_dir=run_root / "evals/lambda/step_000200/eval_model_source",
            run_root=run_root,
        )
        eval_records["lambda"] = {
            "final_group_summary": str(group_lambda_final),
            "final": group_means(group_lambda_final),
        }

        failed_stage = "strip_audit"
        strip_audit = run_strip_audits(
            run_root=run_root,
            stage6_config=config_stage6,
            config_72=config_72,
            current_lambda_ckpt=ckpt_lambda_step200,
            current_strict_donor=STRICT_DONOR_CKPT,
        )

        finalize(
            run_root=run_root,
            stage_ckpts=stage_ckpts,
            stage_records=stage_records,
            handoff_reports=handoff_reports,
            resolved_artifacts=resolved_artifacts,
            eval_records=eval_records,
            strip_audit=strip_audit,
            status="completed",
            failed_stage=None,
            error=None,
        )
        print(json.dumps({"run_root": str(run_root), "status": "completed"}, indent=2))
        return 0
    except Exception as exc:
        finalize(
            run_root=run_root,
            stage_ckpts=stage_ckpts,
            stage_records=stage_records,
            handoff_reports=handoff_reports,
            resolved_artifacts=resolved_artifacts,
            eval_records=eval_records,
            strip_audit=strip_audit,
            status="failed",
            failed_stage=failed_stage,
            error=repr(exc),
        )
        print(json.dumps({"run_root": str(run_root), "status": "failed", "failed_stage": failed_stage, "error": repr(exc)}, indent=2))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
