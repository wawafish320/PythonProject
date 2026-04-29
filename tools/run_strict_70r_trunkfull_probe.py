#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
CPU_WRAPPER = ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
SUMMARY_TOOL = ROOT / "tools/phasea_group_summary.py"
TEACHER_JSON = ROOT / "validate/teacher_batches/Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models/motion_encoder_equiv.pt.best.pt"

CURRENT_RUN_ROOT = ROOT / "debug_output/_tmp_strict_stageB_finalstate_20260427_080658"
DEFAULT_BASE_CONFIG = CURRENT_RUN_ROOT / "stageB_strict/configs/70R.json"
DEFAULT_REPLACE_CKPT = CURRENT_RUN_ROOT / "stageB_strict/replace/checkpoints/ckpt_last_replace_strictB_20260427_080803.pth"
DEFAULT_NO_TRUNK_STEP0 = CURRENT_RUN_ROOT / "stageB_strict/evals/70R/step_000000/group_summary.json"
DEFAULT_NO_TRUNK_STEP20 = CURRENT_RUN_ROOT / "stageB_strict/evals/70R/step_000020/group_summary.json"
DEFAULT_0426_STEP0_CKPT = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158"
    / "70R_lr_probe/lr1e4_step20/checkpoints"
    / "ckpt_step_000000_WalkF_stage7_70R_lr1e4_step20_20260426_173158.pth"
)

GROUP_KEYS = ("all_ex_root", "leg", "nonleg", "arm", "else")


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


def _make_config(*, base_config: Path, run_root: Path, handoff_ckpt: Path, run_name: str) -> Path:
    payload = dict(_load_json(base_config))
    _canonicalize_direct_pose_config_fields(payload)
    for key in ("contact_plan_init_mode", "contact_plan_init_hidden", "contact_plan_init_dropout"):
        payload.pop(key, None)
    for key in ("direct_pose_use_phase_z", "direct_pose_phase_z_mode"):
        payload.pop(key, None)
    for key in tuple(payload.keys()):
        if str(key).startswith("lambda_fusion_"):
            payload.pop(str(key), None)
    payload.update(
        {
            "ckpt_in": str(handoff_ckpt),
            "out_dir": str(run_root / "checkpoints"),
            "run_name": run_name,
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "lr": 1e-4,
            "epochs": 1,
            "steps_per_epoch": 60,
            "save_step_ckpts": "0,1,20",
            "train_direct_pose": True,
            "train_lambda_head": False,
            "direct_pose_nonleg_train_only": True,
            "direct_pose_nonleg_trunk_mode": "full",
        }
    )
    payload.pop("posttrain_contacts_source", None)
    path = run_root / "configs/70R_trunkfull.json"
    _dump_json(path, payload)
    return path


def _contractize(
    *,
    source_ckpt: Path,
    target_config: Path,
    handoff_ckpt: Path,
    report: Path,
    run_root: Path,
    tensor_donor_ckpt: Path | None = None,
    transplant_prefixes: Sequence[str] = (),
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
            "replace_to_70R_trunkfull",
    ]
    if tensor_donor_ckpt is not None:
        cmd.extend(["--tensor-donor", tensor_donor_ckpt])
        for prefix in transplant_prefixes:
            cmd.extend(["--transplant-prefix", str(prefix)])
    _run(
        cmd,
        log_path=run_root / "logs/contractize_replace_to_70R.log",
        run_root=run_root,
    )


def _train(config: Path, run_root: Path) -> None:
    _run(
        [*_python_cmd(), "-m", "train.posttrain", "--config", config],
        log_path=run_root / "logs/70R_train.log",
        run_root=run_root,
    )


def _eval_ckpt(*, ckpt: Path, step: int, run_root: Path) -> Path:
    out_dir = run_root / "evals" / f"step_{step:06d}" / "eval_model_source"
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
        run_root=run_root,
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
        run_root=run_root,
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


def _changed_tensors(step0: Path, step20: Path) -> list[str]:
    s0 = _ckpt_model_state(step0)
    s20 = _ckpt_model_state(step20)
    changed: list[str] = []
    for key in sorted(set(str(k) for k in s0.keys()) & set(str(k) for k in s20.keys())):
        v0 = s0[key]
        v20 = s20[key]
        if not (torch.is_tensor(v0) and torch.is_tensor(v20)):
            continue
        if tuple(v0.shape) != tuple(v20.shape):
            changed.append(key)
            continue
        if not torch.equal(v0, v20):
            changed.append(key)
    return changed


def _train_log_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    json_path = None
    match = re.search(r"\[posttrain\]\[OK\] saved: (.+)", text)
    if match:
        ckpt_last = Path(match.group(1).strip())
        json_path = ckpt_last.parent / f"posttrain_log_{ckpt_last.name[len('ckpt_last_'):-len('.pth')]}.json"
    trunk_vals: list[float] = []
    if json_path is not None and json_path.is_file():
        rows = _load_json(json_path).get("log", [])
        for row in rows:
            try:
                value = float(row.get("direct_grad_norm_trunk", float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                trunk_vals.append(value)
    trainable_match = re.search(r"\[posttrain\] trainable=(\d+) params: ([^\n]+)", text)
    return {
        "log": str(path),
        "posttrain_json": str(json_path) if json_path is not None else None,
        "contains_chain_hop_waiver": "chain_hop-waiver" in text,
        "contains_policy_strict_current": "policy=strict-current" in text,
        "contains_legacy_checkpoint_compat_true": "legacy_checkpoint_compat=true" in text,
        "contains_strict_shape_validation": "strict current model checkpoint shape validation passed" in text,
        "trainable_count": int(trainable_match.group(1)) if trainable_match else None,
        "trainable_preview": trainable_match.group(2).strip() if trainable_match else None,
        "direct_grad_norm_trunk_finite_count": len(trunk_vals),
        "direct_grad_norm_trunk_first": trunk_vals[0] if trunk_vals else float("nan"),
        "direct_grad_norm_trunk_last": trunk_vals[-1] if trunk_vals else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict 70R-only trunk-full regression probe.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--source-replace-ckpt", type=Path, default=DEFAULT_REPLACE_CKPT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--no-trunk-step0-group", type=Path, default=DEFAULT_NO_TRUNK_STEP0)
    parser.add_argument("--no-trunk-step20-group", type=Path, default=DEFAULT_NO_TRUNK_STEP20)
    parser.add_argument("--direct-pose-donor-ckpt", type=Path, help="Deprecated alias for --tensor-donor-ckpt with default prefix `direct_pose_`.")
    parser.add_argument("--tensor-donor-ckpt", type=Path, help="Optional donor checkpoint for targeted start-state transplant.")
    parser.add_argument("--transplant-prefix", action="append", default=[], help="Tensor prefix to copy from --tensor-donor-ckpt.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or (ROOT / "debug_output" / f"_tmp_strict_70R_trunkfull_contract_probe_{stamp}")).expanduser().resolve()
    run_name = args.run_name or f"70R_strict_trunkfull_probe_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    tensor_donor_ckpt = args.tensor_donor_ckpt.expanduser() if args.tensor_donor_ckpt else None
    transplant_prefixes = tuple(str(prefix) for prefix in args.transplant_prefix)
    if args.direct_pose_donor_ckpt is not None:
        if tensor_donor_ckpt is not None:
            raise RuntimeError("use either --direct-pose-donor-ckpt or --tensor-donor-ckpt, not both")
        tensor_donor_ckpt = args.direct_pose_donor_ckpt.expanduser()
        transplant_prefixes = ("direct_pose_",)
    if tensor_donor_ckpt is not None and not transplant_prefixes:
        raise RuntimeError("--tensor-donor-ckpt requires at least one --transplant-prefix")

    handoff_ckpt = run_root / "handoffs/replace_to_70R_trunkfull_strict_contract.pth"
    contract_report = run_root / "handoffs/replace_to_70R_trunkfull_contractize.json"
    config = _make_config(base_config=args.base_config.expanduser(), run_root=run_root, handoff_ckpt=handoff_ckpt, run_name=run_name)
    _contractize(
        source_ckpt=args.source_replace_ckpt.expanduser(),
        target_config=config,
        handoff_ckpt=handoff_ckpt,
        report=contract_report,
        run_root=run_root,
        tensor_donor_ckpt=tensor_donor_ckpt,
        transplant_prefixes=transplant_prefixes,
    )
    _train(config, run_root)

    out_dir = run_root / "checkpoints"
    step0 = out_dir / f"ckpt_step_000000_{run_name}.pth"
    step20 = out_dir / f"ckpt_step_000020_{run_name}.pth"
    if not step0.is_file() or not step20.is_file():
        raise RuntimeError(f"missing required step ckpts: step0={step0.is_file()} step20={step20.is_file()}")

    group0 = _eval_ckpt(ckpt=step0, step=0, run_root=run_root)
    group20 = _eval_ckpt(ckpt=step20, step=20, run_root=run_root)

    g0 = _group_means(group0)
    g20 = _group_means(group20)
    probe_delta = _delta(g0, g20)
    no_trunk = None
    if args.no_trunk_step0_group.is_file() and args.no_trunk_step20_group.is_file():
        nt0 = _group_means(args.no_trunk_step0_group)
        nt20 = _group_means(args.no_trunk_step20_group)
        no_trunk = {"step0": nt0, "step20": nt20, "delta": _delta(nt0, nt20)}

    train_summary = _train_log_summary(run_root / "logs/70R_train.log")
    changed = _changed_tensors(step0, step20)
    required_changed = [
        "direct_pose_head.0.weight",
        "direct_pose_head.0.bias",
        "direct_pose_head.3.weight",
        "direct_pose_head.3.bias",
    ]
    summary = {
        "run_root": str(run_root),
        "config": str(config),
        "contractized_handoff": str(handoff_ckpt),
        "contractize_report": str(contract_report),
        "direct_pose_donor_ckpt": str(args.direct_pose_donor_ckpt.expanduser()) if args.direct_pose_donor_ckpt else None,
        "tensor_donor_ckpt": str(tensor_donor_ckpt) if tensor_donor_ckpt is not None else None,
        "transplant_prefixes": list(transplant_prefixes),
        "source_replace_ckpt": str(args.source_replace_ckpt.expanduser()),
        "strict_policy": {
            "strict_current_model_build": True,
            "load_context": "chain_hop",
            "contains_chain_hop_waiver": bool(train_summary["contains_chain_hop_waiver"]),
            "contains_policy_strict_current": bool(train_summary["contains_policy_strict_current"]),
            "contains_strict_shape_validation": bool(train_summary["contains_strict_shape_validation"]),
        },
        "train": train_summary,
        "evals": {
            "step0_group_summary": str(group0),
            "step20_group_summary": str(group20),
            "step0": g0,
            "step20": g20,
            "delta_step20_minus_step0": probe_delta,
            "current_no_trunk_reference": no_trunk,
        },
        "changed_tensors": changed,
        "required_trunk_tensors_changed": {key: key in changed for key in required_changed},
    }
    _dump_json(run_root / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
