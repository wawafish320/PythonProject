#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        make_generated_config,
        run_cmd,
        run_eval,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        make_generated_config,
        run_cmd,
        run_eval,
        write_json,
    )


RUN_DATE = "20260407"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_plan_drop_competition_probe_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_plan_drop_competition_probe_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
COADAPT_240_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "coadapt_allrot_interface_bestlr_longer_4x"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth"
)
COADAPT_240_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "configs"
    / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json"
)

CASE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "coadapt_plan_drop_0p3",
        "plan_drop_prob": 0.3,
        "steps_per_epoch": 240,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_plan_drop_0p5",
        "plan_drop_prob": 0.5,
        "steps_per_epoch": 240,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_plan_drop_sched_1p0_to_0p3_240",
        "plan_drop_schedule": (
            {"start_step": 0, "end_step": 80, "prob": 1.0},
            {"start_step": 80, "end_step": 160, "prob": 0.7},
            {"start_step": 160, "end_step": 240, "prob": 0.3},
        ),
        "steps_per_epoch": 240,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_plan_drop_sched_1p0_to_0p0_240",
        "plan_drop_schedule": (
            {"start_step": 0, "end_step": 80, "prob": 1.0},
            {"start_step": 80, "end_step": 160, "prob": 0.5},
            {"start_step": 160, "end_step": 240, "prob": 0.0},
        ),
        "steps_per_epoch": 240,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))


def _run_name(case_name: str) -> str:
    return f"WalkF_stage7_70b_replace_lowdrift_{case_name}_{RUN_DATE}"


def _schedule_payload(spec: Mapping[str, Any]) -> Optional[Tuple[Dict[str, Any], ...]]:
    raw = spec.get("plan_drop_schedule")
    if not raw:
        return None
    out: List[Dict[str, Any]] = []
    for item in raw:
        out.append(
            {
                "start_step": int(item["start_step"]),
                "end_step": None if item.get("end_step") is None else int(item["end_step"]),
                "prob": float(item["prob"]),
            }
        )
    return tuple(out)


def _schedule_label(schedule: Optional[Tuple[Dict[str, Any], ...]]) -> str:
    if not schedule:
        return "fixed"
    parts: List[str] = []
    for item in schedule:
        start_step = int(item["start_step"])
        end_step = item.get("end_step", None)
        prob = float(item["prob"])
        if end_step is None:
            parts.append(f"[{start_step},∞)->{prob:.3f}")
        else:
            parts.append(f"[{start_step},{int(end_step)})->{prob:.3f}")
    return "; ".join(parts)


def _parse_trainable_log(case_name: str) -> Dict[str, Any]:
    if not LOG_FILE.is_file():
        return {"found": False, "case_name": str(case_name), "log_file": str(LOG_FILE)}
    run_name = _run_name(case_name)
    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_block = False
    train_mode = None
    trainable_count = None
    sample_names: List[str] = []
    for line in lines:
        if line.startswith("$ "):
            in_block = run_name in line and "train.posttrain" in line
            continue
        if not in_block:
            continue
        if "[posttrain] mode=" in line and train_mode is None:
            train_mode = line.split("=", 1)[-1].strip()
            continue
        if "[posttrain] trainable=" not in line:
            continue
        match = re.search(r"trainable=(\d+)\s+params:\s*(.+)$", line)
        if match is not None:
            trainable_count = int(match.group(1))
            sample_names = [part.strip() for part in match.group(2).split(",") if part.strip()]
            break
    return {
        "found": trainable_count is not None,
        "case_name": str(case_name),
        "run_name": run_name,
        "log_file": str(LOG_FILE),
        "train_mode": train_mode,
        "trainable_param_count": trainable_count,
        "sample_names": sample_names,
        "all_sample_names_are_direct_pose": bool(sample_names) and all(
            name.startswith("direct_pose_") for name in sample_names
        ),
    }


def _make_case_config(spec: Mapping[str, Any]) -> Tuple[Path, Path, str]:
    case_name = str(spec["name"])
    out_dir = MODEL_ROOT / case_name
    run_name = _run_name(case_name)
    cfg_json = CONFIG_ROOT / f"{case_name}_{RUN_DATE}.json"
    overrides: Dict[str, Any] = {
        "ckpt_in": str(COADAPT_240_CKPT),
        "out_dir": str(out_dir),
        "run_name": run_name,
        "device": "cpu",
        "epochs": 1,
        "steps_per_epoch": int(spec["steps_per_epoch"]),
        "lr": float(spec["lr"]),
        "weight_decay": float(spec.get("weight_decay", 0.0)),
        "encoder_bundle": str(ENCODER_BUNDLE),
        "posttrain_contacts_source": "pretrain_contact",
        "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        "optimizer_param_group_overrides": None,
        "train_direct_pose": True,
        "train_incremental_replace": False,
        "train_lambda_head": False,
        "train_arm_residual": False,
        "train_arm_leg_residual": False,
        "incremental_motion_head_row_ranges": None,
        "incremental_interface_mode": "off",
        "incremental_interface_lr_scale": 0.0,
        "direct_pose_leg_train_only": False,
        "direct_pose_leg_gate_train_only": False,
        "direct_pose_nonleg_train_only": False,
    }
    if "plan_drop_prob" in spec:
        overrides["direct_pose_plan_drop_prob"] = float(spec["plan_drop_prob"])
    plan_drop_schedule = _schedule_payload(spec)
    if plan_drop_schedule is not None:
        overrides["direct_pose_plan_drop_schedule"] = plan_drop_schedule
    make_generated_config(
        COADAPT_240_CONFIG,
        cfg_json,
        overrides,
    )
    return cfg_json, out_dir, run_name


def _run_case(spec: Mapping[str, Any]) -> Dict[str, Any]:
    case_name = str(spec["name"])
    cfg_json, out_dir, run_name = _make_case_config(spec)
    ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    if not ckpt.is_file():
        run_cmd(
            [
                sys.executable,
                str(CPU_EXEC),
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_json),
                "--ckpt_in",
                str(COADAPT_240_CKPT),
                "--out_dir",
                str(out_dir),
                "--run_name",
                run_name,
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                str(PRETRAIN_CLAMP),
                "--encoder_bundle",
                str(ENCODER_BUNDLE),
                "--posttrain_contacts_pretrain_affine_stats",
                str(AFFINE_STATS),
            ],
            log_file=LOG_FILE,
        )

    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
    if not eval_json.is_file():
        run_eval(
            model_ckpt=ckpt,
            out_dir=eval_dir,
            contacts_source="model",
            log_file=LOG_FILE,
        )

    plan_drop_schedule = _schedule_payload(spec)
    changed_keys = ["direct_pose_plan_drop_schedule"] if plan_drop_schedule is not None else ["direct_pose_plan_drop_prob"]
    payload = {
        "name": case_name,
        "warmstart": "coadapt_allrot_interface_bestlr_longer_4x",
        "warmstart_ckpt": str(COADAPT_240_CKPT),
        "trainable_scope": "train_direct_pose only",
        "trainable_scope_detail": {
            "train_direct_pose": True,
            "train_incremental_replace": False,
            "train_lambda_head": False,
            "train_arm_residual": False,
            "train_arm_leg_residual": False,
            "incremental_interface_mode": "off",
            "incremental_interface_lr_scale": 0.0,
        },
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "eval": str(eval_json),
        "steps_per_epoch": int(spec["steps_per_epoch"]),
        "lr": float(spec["lr"]),
        "weight_decay": float(spec.get("weight_decay", 0.0)),
        "direct_pose_plan_drop_prob": float(spec.get("plan_drop_prob", 0.0)),
        "direct_pose_plan_drop_schedule": plan_drop_schedule,
        "plan_drop_schedule_label": _schedule_label(plan_drop_schedule),
        "changed_config_keys": changed_keys,
        "self_contained": True,
        "event_clock_enabled": True,
        "analysis_artifact_root": str(OUT_ROOT),
        "trainable_log_report": _parse_trainable_log(case_name),
    }
    write_json(OUT_ROOT / "candidates" / f"{case_name}.json", payload)
    return payload


def _build_summary_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# cp015 tailk7 minimal plan-drop competition probe",
        "",
        "## Training design",
        "",
        f"- warmstart: `{summary['training_design']['warmstart']}`",
        f"- trainable scope: `{summary['training_design']['trainable_scope']}`",
        f"- old recipe defaults changed: `{str(summary['training_design']['old_recipe_defaults_changed']).lower()}`",
        "",
        "| candidate | plan_drop_prob | plan_drop_schedule | steps | lr | wd | ckpt | eval |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['name']}` | `{case['direct_pose_plan_drop_prob']}` | "
            f"`{case['plan_drop_schedule_label']}` | `{case['steps_per_epoch']}` | "
            f"`{case['lr']}` | `{case['weight_decay']}` | `{case['ckpt']}` | `{case['eval']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_exists(
        [
            CPU_EXEC,
            COADAPT_240_CKPT,
            COADAPT_240_CONFIG,
            ENCODER_BUNDLE,
            AFFINE_STATS,
        ]
    )
    cases: List[Dict[str, Any]] = []
    total = len(CASE_SPECS)
    for idx, spec in enumerate(CASE_SPECS, start=1):
        log(f"running {spec['name']} ({idx}/{total})")
        cases.append(_run_case(spec))
        write_json(
            STATUS_JSON,
            {
                "completed": False,
                "done_cases": [case["name"] for case in cases],
                "total_cases": total,
            },
        )

    summary = {
        "run_date": RUN_DATE,
        "artifacts": {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "log_file": str(LOG_FILE),
        },
        "training_design": {
            "warmstart": "coadapt_allrot_interface_bestlr_longer_4x",
            "warmstart_ckpt": str(COADAPT_240_CKPT),
            "trainable_scope": "train_direct_pose only",
            "trainable_scope_detail": {
                "train_direct_pose": True,
                "train_incremental_replace": False,
                "train_lambda_head": False,
                "train_arm_residual": False,
                "train_arm_leg_residual": False,
                "incremental_interface_mode": "off",
                "incremental_interface_lr_scale": 0.0,
            },
            "changed_config_keys": ["direct_pose_plan_drop_prob", "direct_pose_plan_drop_schedule"],
            "old_recipe_defaults_changed": False,
            "base_config": str(COADAPT_240_CONFIG),
        },
        "cases": cases,
    }
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(_build_summary_md(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "completed": True,
            "done_cases": [case["name"] for case in cases],
            "total_cases": total,
        },
    )
    log(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
