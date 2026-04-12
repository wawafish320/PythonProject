#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        load_json,
        make_generated_config,
        run_cmd,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        load_json,
        make_generated_config,
        run_cmd,
        write_json,
    )

from tools.analyze_cp015_tailk7_rot_row_group_pose_swaps import (  # noqa: E402
    PRIMARY_METRICS,
    _case_summary,
    _relative_improvement,
    _safe_float,
)


RUN_DATE = "20260406"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
ROOT_NAME = "pelvis"
ROW_ALL_ROT = ((0, 276),)
REQUIRED_BUCKETS: Tuple[str, ...] = ("d0_9", "d10_20", "d21_43", "sic0_10", "sic11_21", "sic22_43")
FOCUS_BUCKETS: Tuple[str, ...] = ("d10_20", "d21_43", "sic11_21", "sic22_43")
SECONDARY_METRICS: Tuple[str, ...] = ("GeoLocalDeg",)
BASE_EPOCHS = 1
BASE_STEPS_PER_EPOCH = 60
READOUT_LR = 5e-5

CURRENT_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
CURRENT_CONTROL_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "e3x60_adapter_factorized"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth"
)
CURRENT_CONTROL_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "eval_model_source"
    / "e3x60_adapter_factorized"
    / "Walk_F_freerun_cycles.json"
)
BASELINE_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
BASELINE_REPLACE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)
LOWLR_WINNER_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_replace_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
)
PREV_0P02_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_ablation_20260405"
    / "configs"
    / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_20260405.json"
)
PREV_0P02_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_ablation_20260405"
    / "coadapt_allrot_interface"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_20260405.pth"
)
PREV_0P02_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_ablation_20260405"
    / "eval_model_source"
    / "coadapt_allrot_interface"
    / "Walk_F_freerun_cycles.json"
)

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_{RUN_DATE}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"

LR_CASE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "coadapt_allrot_interface_lrscale_0p01",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "tail",
        "interface_lr_scale": 0.01,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": BASE_STEPS_PER_EPOCH,
    },
    {
        "name": "coadapt_allrot_interface_lrscale_0p02",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "tail",
        "interface_lr_scale": 0.02,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": BASE_STEPS_PER_EPOCH,
        "reuse_artifacts": {
            "config": PREV_0P02_CONFIG,
            "ckpt": PREV_0P02_CKPT,
            "eval": PREV_0P02_EVAL,
            "note": "reuse prior 2026-04-05 best-anchor artifacts",
        },
    },
    {
        "name": "coadapt_allrot_interface_lrscale_0p04",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "tail",
        "interface_lr_scale": 0.04,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": BASE_STEPS_PER_EPOCH,
    },
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))


def _fmt(x: Any) -> str:
    val = _safe_float(x)
    return "nan" if not math.isfinite(val) else f"{val:.6f}"


def _fmt_pct(x: Any) -> str:
    val = _safe_float(x)
    return "nan" if not math.isfinite(val) else f"{100.0 * val:+.2f}%"


def _load_eval_summary(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    return _case_summary(
        {
            "metrics_per_round": list(payload.get("metrics_per_round", []) or []),
            "metrics_per_step": list(payload.get("metrics_per_step", []) or []),
        },
        root_name=ROOT_NAME,
    )


def _primary_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        var = (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _primary_bucket_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any], bucket: str) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (
            (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        var = (
            (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _overall_primary_row(label: str, case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": label}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        row[metric] = (
            (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        )
    row["primary_relative_improvement_vs_current"] = _primary_relative_mean(case_summary, ref_summary)
    return row


def _bucket_row(label: str, case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": label}
    for bucket in REQUIRED_BUCKETS:
        row[bucket] = _primary_bucket_relative_mean(case_summary, ref_summary, bucket)
    return row


def _primary_metric_mean(case_summary: Mapping[str, Any], metric: str) -> float:
    return _safe_float((((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean"))


def _primary_metric_win_count(case_summary: Mapping[str, Any], other_summary: Mapping[str, Any]) -> int:
    wins = 0
    for metric in PRIMARY_METRICS:
        cur = _primary_metric_mean(case_summary, metric)
        oth = _primary_metric_mean(other_summary, metric)
        if math.isfinite(cur) and math.isfinite(oth) and cur < oth:
            wins += 1
    return int(wins)


def _focus_bucket_score(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> float:
    return float(sum(_primary_bucket_relative_mean(case_summary, ref_summary, bucket) for bucket in FOCUS_BUCKETS))


def _best_case_name(case_names: Sequence[str], case_payloads: Mapping[str, Mapping[str, Any]], current_summary: Mapping[str, Any]) -> str:
    ranked = sorted(
        [str(name) for name in case_names],
        key=lambda name: (
            _safe_float(_primary_relative_mean(case_payloads[name]["summary"], current_summary)),
            _safe_float(_focus_bucket_score(case_payloads[name]["summary"], current_summary)),
        ),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("no candidate cases to rank")
    return str(ranked[0])


def _is_clear_best(
    best_name: str,
    case_names: Sequence[str],
    case_payloads: Mapping[str, Mapping[str, Any]],
    current_summary: Mapping[str, Any],
) -> bool:
    best_summary = case_payloads[best_name]["summary"]
    if len(case_names) <= 1:
        return True
    for other_name in case_names:
        if str(other_name) == str(best_name):
            continue
        other_summary = case_payloads[str(other_name)]["summary"]
        if _primary_metric_win_count(best_summary, other_summary) < 4:
            return False
        if _primary_relative_mean(best_summary, current_summary) <= _primary_relative_mean(other_summary, current_summary):
            return False
    return True


def _longer_still_improving(
    longer_summary: Mapping[str, Any],
    base_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
) -> bool:
    if _primary_relative_mean(longer_summary, current_summary) <= _primary_relative_mean(base_summary, current_summary):
        return False
    return _primary_metric_win_count(longer_summary, base_summary) >= 3


def _delta_vs_anchor_row(
    label: str,
    case_summary: Mapping[str, Any],
    anchor_summary: Mapping[str, Any],
    ref_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": label,
        "overall_primary_rel_delta_vs_anchor": (
            _primary_relative_mean(case_summary, ref_summary) - _primary_relative_mean(anchor_summary, ref_summary)
        ),
    }
    for bucket in REQUIRED_BUCKETS:
        row[bucket] = _primary_bucket_relative_mean(case_summary, ref_summary, bucket) - _primary_bucket_relative_mean(
            anchor_summary,
            ref_summary,
            bucket,
        )
    return row


def _case_total_steps(case: Mapping[str, Any]) -> int:
    return int(case["epochs"]) * int(case["steps_per_epoch"])


def _make_case_config(case: Mapping[str, Any]) -> Tuple[Path, Path, str]:
    case_name = str(case["name"])
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_{RUN_DATE}.json"
    make_generated_config(
        LOWLR_WINNER_CONFIG,
        cfg_json,
        {
            "ckpt_in": str(WARMSTART_CKPT),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "device": "cpu",
            "epochs": int(case["epochs"]),
            "steps_per_epoch": int(case["steps_per_epoch"]),
            "lr": READOUT_LR,
            "weight_decay": 0.0,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "optimizer_param_group_overrides": None,
            "train_direct_pose": False,
            "train_incremental_replace": True,
            "train_lambda_head": False,
            "train_arm_residual": False,
            "train_arm_leg_residual": False,
            "incremental_motion_head_row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
            "incremental_interface_mode": str(case["interface_mode"]),
            "incremental_interface_lr_scale": float(case["interface_lr_scale"]),
        },
    )
    return cfg_json, out_dir, run_name


def _run_posttrain_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    case_name = str(case["name"])
    reuse_artifacts = case.get("reuse_artifacts")
    if isinstance(reuse_artifacts, Mapping):
        config_path = Path(str(reuse_artifacts["config"]))
        ckpt = Path(str(reuse_artifacts["ckpt"]))
        eval_json = Path(str(reuse_artifacts["eval"]))
        if config_path.is_file() and ckpt.is_file() and eval_json.is_file():
            case_summary = _load_eval_summary(eval_json)
            return {
                "name": case_name,
                "config": str(config_path),
                "ckpt": str(ckpt),
                "eval": str(eval_json),
                "summary": case_summary,
                "row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
                "interface_mode": str(case["interface_mode"]),
                "interface_lr_scale": float(case["interface_lr_scale"]),
                "epochs": int(case["epochs"]),
                "steps_per_epoch": int(case["steps_per_epoch"]),
                "total_steps": int(_case_total_steps(case)),
                "reuse_existing_artifacts": True,
                "reuse_note": str(reuse_artifacts.get("note", "")),
            }

    cfg_json, out_dir, run_name = _make_case_config(case)
    ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
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
                str(WARMSTART_CKPT),
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
    if not eval_json.is_file():
        run_cmd(
            [
                sys.executable,
                str(CPU_EXEC),
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"),
                "--model",
                str(ckpt),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--event_clock",
                "auto",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "model",
                "--lambda_fusion_apply",
                "--log_contacts",
                "--export_direct_arm_probe",
                "--export_joint_direct_geolocal_series",
                "--out",
                str(eval_dir),
                "--force",
            ],
            log_file=LOG_FILE,
        )
    case_summary = _load_eval_summary(eval_json)
    return {
        "name": case_name,
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "eval": str(eval_json),
        "summary": case_summary,
        "row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
        "interface_mode": str(case["interface_mode"]),
        "interface_lr_scale": float(case["interface_lr_scale"]),
        "epochs": int(case["epochs"]),
        "steps_per_epoch": int(case["steps_per_epoch"]),
        "total_steps": int(_case_total_steps(case)),
        "reuse_existing_artifacts": False,
    }


def _write_status(case_payloads: Mapping[str, Any], executed_order: Sequence[str], planned_total: int, notes: Mapping[str, Any]) -> None:
    write_json(
        STATUS_JSON,
        {
            "done_cases": list(case_payloads.keys()),
            "executed_order": list(executed_order),
            "total_cases_planned_so_far": int(planned_total),
            "notes": dict(notes),
        },
    )


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# cp015 tailk7 replace interface co-adaptation saturation sweep",
        "",
        "## Code Facts",
        "",
    ]
    for fact in summary["code_facts"]["current_train_incremental_replace"]:
        lines.append(f"- {fact}")
    lines.append("")
    for fact in summary["code_facts"]["new_runner_and_config_plumbing"]:
        lines.append(f"- {fact}")
    lines.extend(
        [
            "",
            "## Experiment Matrix",
            "",
            "| variant | row_ranges | interface_mode | interface_lr_scale | epochs | steps_per_epoch | total_steps | reuse |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["experiment_matrix"]:
        lines.append(
            f"| {row['name']} | {row['row_ranges']} | {row['interface_mode']} | {float(row['interface_lr_scale']):.4f} | "
            f"{int(row['epochs'])} | {int(row['steps_per_epoch'])} | {int(row['total_steps'])} | {row['reuse_existing_artifacts']} |"
        )
    lines.extend(
        [
            "",
            "## Primary Pose Table",
            "",
            "| variant | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs current |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["primary_pose"]:
        lines.append(
            f"| {row['variant']} | {_fmt(row['Rot6dLocalL2'])} | {_fmt(row['Rot6dLocalL2Weighted'])} | "
            f"{_fmt(row['GeoDeg'])} | {_fmt(row['KeyBoneGeoDegMean'])} | {_fmt(row['KeyBoneGeoLocalDegMean'])} | "
            f"{_fmt(row['GeoLocalDeg'])} | {_fmt_pct(row['primary_relative_improvement_vs_current'])} |"
        )
    lines.extend(
        [
            "",
            "## Bucket-Wise Table",
            "",
            "| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["bucket_primary_relative_vs_current"]:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row['d0_9'])} | {_fmt_pct(row['d10_20'])} | {_fmt_pct(row['d21_43'])} | "
            f"{_fmt_pct(row['sic0_10'])} | {_fmt_pct(row['sic11_21'])} | {_fmt_pct(row['sic22_43'])} |"
        )
    lines.extend(
        [
            "",
            "## LR Delta vs 0p02 Anchor",
            "",
            "| variant | overall primary delta vs 0p02 | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["delta_vs_0p02_anchor"]:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row['overall_primary_rel_delta_vs_anchor'])} | {_fmt_pct(row['d0_9'])} | "
            f"{_fmt_pct(row['d10_20'])} | {_fmt_pct(row['d21_43'])} | {_fmt_pct(row['sic0_10'])} | "
            f"{_fmt_pct(row['sic11_21'])} | {_fmt_pct(row['sic22_43'])} |"
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            f"- best_lr_case: {summary['judgements']['best_lr_case']}",
            f"- best_lr_scale: {summary['judgements']['best_lr_scale']}",
            f"- clear_best_lr: {summary['judgements']['clear_best_lr']}",
            f"- longer_1p5x_run: {summary['judgements']['longer_1p5x_run']}",
            f"- longer_1p5x_improved: {summary['judgements']['longer_1p5x_improved']}",
            f"- longer_2x_run: {summary['judgements']['longer_2x_run']}",
            f"- longer_2x_improved: {summary['judgements']['longer_2x_improved']}",
            f"- plateau_after_lr_epochs: {summary['judgements']['plateau_after_lr_epochs']}",
            f"- subset_probe_run: {summary['judgements']['subset_probe_run']}",
            f"- next_priority: {summary['judgements']['next_priority']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_exists(
        [
            CURRENT_70A_CKPT,
            CURRENT_CONTROL_CKPT,
            CURRENT_CONTROL_EVAL,
            BASELINE_REPLACE_CKPT,
            BASELINE_REPLACE_EVAL,
            LOWLR_WINNER_CONFIG,
            CPU_EXEC,
            ENCODER_BUNDLE,
            AFFINE_STATS,
            PREV_0P02_CONFIG,
            PREV_0P02_CKPT,
            PREV_0P02_EVAL,
        ]
    )
    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    current_summary = _load_eval_summary(CURRENT_CONTROL_EVAL)
    baseline_summary = _load_eval_summary(BASELINE_REPLACE_EVAL)

    case_payloads: Dict[str, Any] = {}
    executed_order: List[str] = []
    decision_notes: Dict[str, Any] = {}

    for case in LR_CASE_SPECS:
        log(f"running {case['name']}")
        case_name = str(case["name"])
        case_payloads[case_name] = _run_posttrain_case(case)
        executed_order.append(case_name)
        _write_status(case_payloads, executed_order, len(LR_CASE_SPECS), decision_notes)

    lr_case_names = [str(case["name"]) for case in LR_CASE_SPECS]
    best_lr_name = _best_case_name(lr_case_names, case_payloads, current_summary)
    clear_best_lr = _is_clear_best(best_lr_name, lr_case_names, case_payloads, current_summary)
    decision_notes["best_lr_name"] = best_lr_name
    decision_notes["clear_best_lr"] = bool(clear_best_lr)

    longer_1p5x_name: Optional[str] = None
    longer_2x_name: Optional[str] = None
    longer_1p5x_improved = False
    longer_2x_improved = False

    if clear_best_lr:
        best_lr_scale = float(case_payloads[best_lr_name]["interface_lr_scale"])
        longer_1p5x_case = {
            "name": "coadapt_allrot_interface_bestlr_longer_1p5x",
            "row_ranges": ROW_ALL_ROT,
            "interface_mode": "tail",
            "interface_lr_scale": best_lr_scale,
            "epochs": BASE_EPOCHS,
            "steps_per_epoch": int(round(BASE_STEPS_PER_EPOCH * 1.5)),
        }
        log(f"running {longer_1p5x_case['name']} with lr_scale={best_lr_scale:.4f}")
        longer_1p5x_name = str(longer_1p5x_case["name"])
        case_payloads[longer_1p5x_name] = _run_posttrain_case(longer_1p5x_case)
        executed_order.append(longer_1p5x_name)
        longer_1p5x_improved = _longer_still_improving(
            case_payloads[longer_1p5x_name]["summary"],
            case_payloads[best_lr_name]["summary"],
            current_summary,
        )
        decision_notes["longer_1p5x_improved"] = bool(longer_1p5x_improved)
        _write_status(case_payloads, executed_order, len(LR_CASE_SPECS) + 1, decision_notes)

        if longer_1p5x_improved:
            longer_2x_case = {
                "name": "coadapt_allrot_interface_bestlr_longer_2x",
                "row_ranges": ROW_ALL_ROT,
                "interface_mode": "tail",
                "interface_lr_scale": best_lr_scale,
                "epochs": BASE_EPOCHS,
                "steps_per_epoch": int(BASE_STEPS_PER_EPOCH * 2),
            }
            log(f"running {longer_2x_case['name']} with lr_scale={best_lr_scale:.4f}")
            longer_2x_name = str(longer_2x_case["name"])
            case_payloads[longer_2x_name] = _run_posttrain_case(longer_2x_case)
            executed_order.append(longer_2x_name)
            longer_2x_improved = _longer_still_improving(
                case_payloads[longer_2x_name]["summary"],
                case_payloads[longer_1p5x_name]["summary"],
                current_summary,
            )
            decision_notes["longer_2x_improved"] = bool(longer_2x_improved)
            _write_status(case_payloads, executed_order, len(LR_CASE_SPECS) + 2, decision_notes)

    epoch_anchor_candidates = list(lr_case_names)
    if longer_1p5x_name is not None:
        epoch_anchor_candidates.append(longer_1p5x_name)
    if longer_2x_name is not None:
        epoch_anchor_candidates.append(longer_2x_name)
    best_epoch_case_name = _best_case_name(epoch_anchor_candidates, case_payloads, current_summary)

    plateau_after_lr_epochs = False
    if not clear_best_lr:
        plateau_after_lr_epochs = True
    elif longer_1p5x_name is None:
        plateau_after_lr_epochs = True
    elif not longer_1p5x_improved:
        plateau_after_lr_epochs = True
    elif longer_2x_name is not None and not longer_2x_improved:
        plateau_after_lr_epochs = True

    subset_case_names: List[str] = []
    if plateau_after_lr_epochs:
        subset_anchor = case_payloads[best_epoch_case_name]
        subset_specs = (
            {
                "name": "coadapt_allrot_interface_no_sharedenc_lastblock",
                "row_ranges": ROW_ALL_ROT,
                "interface_mode": "tail_no_sharedenc_lastblock",
                "interface_lr_scale": float(subset_anchor["interface_lr_scale"]),
                "epochs": int(subset_anchor["epochs"]),
                "steps_per_epoch": int(subset_anchor["steps_per_epoch"]),
            },
            {
                "name": "coadapt_allrot_interface_no_pasa_stack",
                "row_ranges": ROW_ALL_ROT,
                "interface_mode": "tail_no_pasa_stack",
                "interface_lr_scale": float(subset_anchor["interface_lr_scale"]),
                "epochs": int(subset_anchor["epochs"]),
                "steps_per_epoch": int(subset_anchor["steps_per_epoch"]),
            },
        )
        for case in subset_specs:
            log(f"running {case['name']} with lr_scale={float(case['interface_lr_scale']):.4f}")
            case_name = str(case["name"])
            case_payloads[case_name] = _run_posttrain_case(case)
            executed_order.append(case_name)
            subset_case_names.append(case_name)
            _write_status(case_payloads, executed_order, len(executed_order), decision_notes)

    ordered_case_names: List[str] = list(lr_case_names)
    if longer_1p5x_name is not None:
        ordered_case_names.append(longer_1p5x_name)
    if longer_2x_name is not None:
        ordered_case_names.append(longer_2x_name)
    ordered_case_names.extend(subset_case_names)

    primary_pose_rows: List[Dict[str, Any]] = [
        _overall_primary_row("current_frozen_trunk_replace_control", current_summary, current_summary)
    ]
    primary_pose_rows.extend(
        _overall_primary_row(name, case_payloads[name]["summary"], current_summary) for name in ordered_case_names
    )
    primary_pose_rows.append(_overall_primary_row("baseline_replace", baseline_summary, current_summary))

    bucket_rows: List[Dict[str, Any]] = [
        _bucket_row("current_frozen_trunk_replace_control", current_summary, current_summary)
    ]
    bucket_rows.extend(_bucket_row(name, case_payloads[name]["summary"], current_summary) for name in ordered_case_names)
    bucket_rows.append(_bucket_row("baseline_replace", baseline_summary, current_summary))

    delta_vs_0p02_rows = [
        _delta_vs_anchor_row(name, case_payloads[name]["summary"], case_payloads["coadapt_allrot_interface_lrscale_0p02"]["summary"], current_summary)
        for name in ordered_case_names
    ]

    experiment_matrix = [
        {
            "name": name,
            "row_ranges": case_payloads[name]["row_ranges"],
            "interface_mode": case_payloads[name]["interface_mode"],
            "interface_lr_scale": float(case_payloads[name]["interface_lr_scale"]),
            "epochs": int(case_payloads[name]["epochs"]),
            "steps_per_epoch": int(case_payloads[name]["steps_per_epoch"]),
            "total_steps": int(case_payloads[name]["total_steps"]),
            "reuse_existing_artifacts": bool(case_payloads[name].get("reuse_existing_artifacts", False)),
        }
        for name in ordered_case_names
    ]

    best_lr_scale = float(case_payloads[best_lr_name]["interface_lr_scale"])
    judgements = {
        "main_question": "replace-time co-adapt is not being re-proven; the only question is whether the current replace-stage broad-tail co-adapt is under-tuned or already saturated.",
        "primary_metric_family": "pose-side metrics only: Rot6dLocalL2 / Rot6dLocalL2Weighted / GeoDeg / KeyBoneGeoDegMean / KeyBoneGeoLocalDegMean; GeoLocalDeg is secondary only.",
        "best_lr_case": best_lr_name,
        "best_lr_scale": best_lr_scale,
        "clear_best_lr": bool(clear_best_lr),
        "longer_1p5x_run": longer_1p5x_name is not None,
        "longer_1p5x_improved": bool(longer_1p5x_improved),
        "longer_2x_run": longer_2x_name is not None,
        "longer_2x_improved": bool(longer_2x_improved),
        "best_epoch_case": best_epoch_case_name,
        "plateau_after_lr_epochs": bool(plateau_after_lr_epochs),
        "subset_probe_run": bool(subset_case_names),
        "subset_anchor_case": best_epoch_case_name if subset_case_names else None,
        "next_priority": (
            "continue replace-stage full[0:276]+broad-tail co-adaptation tuning"
            if not plateau_after_lr_epochs or not subset_case_names
            else "subset probe finished; only consider upstream robustness if replace-stage broad-tail co-adapt stays flat"
        ),
        "adapter_not_first_priority": True,
    }

    summary = {
        "run_date": RUN_DATE,
        "artifacts": {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "warmstart_ckpt": str(WARMSTART_CKPT),
            "warmstart_report": str(WARMSTART_REPORT),
            "log_file": str(LOG_FILE),
        },
        "references": {
            "current_70a_ckpt": str(CURRENT_70A_CKPT),
            "current_frozen_trunk_replace_control_ckpt": str(CURRENT_CONTROL_CKPT),
            "current_frozen_trunk_replace_control_eval": str(CURRENT_CONTROL_EVAL),
            "baseline_replace_ckpt": str(BASELINE_REPLACE_CKPT),
            "baseline_replace_eval": str(BASELINE_REPLACE_EVAL),
            "reused_0p02_config": str(PREV_0P02_CONFIG),
            "reused_0p02_ckpt": str(PREV_0P02_CKPT),
            "reused_0p02_eval": str(PREV_0P02_EVAL),
            "lowlr_winner_config": str(LOWLR_WINNER_CONFIG),
        },
        "code_facts": {
            "current_train_incremental_replace": [
                "train_incremental_replace still trains the deployed incremental path with objective='inc'; this round does not change runtime contract or loss family.",
                "incremental_motion_head_row_ranges still masks grads only on the final motion_head Linear weight/bias rows; unselected rows are zeroed by hooks.",
                "readout parameters stay on the default AdamW group at cfg.lr, while interface params get an auto-created incremental_interface group at cfg.lr * incremental_interface_lr_scale.",
                "broad tail mode still resolves to shared_encoder last residual block + residual_proj + _pasa_lnq/_q/_k/_v/_o/_film + coupling_norm.",
            ],
            "new_runner_and_config_plumbing": [
                "this round adds tools/run_cp015_tailk7_replace_interface_coadapt_saturation_sweep.py to enforce the decision order LR scale -> longer training -> interface subset.",
                "the runner keeps final rot rows fixed to full [0:276] for the mandatory sweep and reuses the 2026-04-05 0p02 broad-tail anchor unless those artifacts are missing.",
                "train/posttrain.py now accepts incremental_interface_mode=tail_no_sharedenc_lastblock and tail_no_pasa_stack so subset probes can isolate shared_encoder last block vs PASA stack without changing runtime semantics.",
                "longer training is implemented as total-step scaling on the same replace-stage config: 60 -> 90 -> 120 steps at the chosen best LR.",
            ],
        },
        "experiment_matrix": experiment_matrix,
        "cases": case_payloads,
        "tables": {
            "primary_pose": primary_pose_rows,
            "bucket_primary_relative_vs_current": bucket_rows,
            "delta_vs_0p02_anchor": delta_vs_0p02_rows,
        },
        "judgements": judgements,
    }
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(_build_markdown(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "done_cases": list(case_payloads.keys()),
            "executed_order": executed_order,
            "completed": True,
            "judgements": judgements,
        },
    )
    log(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
