#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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


RUN_DATE = "20260405"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_interface_coadapt_ablation_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_interface_coadapt_ablation_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
ROOT_NAME = "pelvis"
ROW_ALL_ROT = ((0, 276),)
ROW_ALL_LEG = ((192, 276),)
REQUIRED_BUCKETS: Tuple[str, ...] = ("d0_9", "d10_20", "d21_43", "sic0_10", "sic11_21", "sic22_43")
SECONDARY_METRICS: Tuple[str, ...] = ("GeoLocalDeg",)

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

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_{RUN_DATE}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"

CASE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "readout_only_allrot",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "off",
        "interface_lr_scale": 0.0,
    },
    {
        "name": "readout_only_allleg",
        "row_ranges": ROW_ALL_LEG,
        "interface_mode": "off",
        "interface_lr_scale": 0.0,
    },
    {
        "name": "coadapt_allrot_interface",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "tail",
        "interface_lr_scale": 0.02,
    },
    {
        "name": "coadapt_allleg_interface",
        "row_ranges": ROW_ALL_LEG,
        "interface_mode": "tail",
        "interface_lr_scale": 0.02,
    },
    {
        "name": "coadapt_allrot_couplingnorm_only",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": "coupling_norm_only",
        "interface_lr_scale": 0.02,
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


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# cp015 tailk7 replace interface co-adaptation ablation",
        "",
        "## Experiment Matrix",
        "",
        "| variant | row_ranges | interface_mode | interface_lr_scale |",
        "|---|---|---|---:|",
    ]
    for row in summary["experiment_matrix"]:
        lines.append(
            f"| {row['name']} | {row['row_ranges']} | {row['interface_mode']} | {float(row['interface_lr_scale']):.4f} |"
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
            "## Bucket Relative Improvement vs Current",
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
            "## Judgement",
            "",
            f"- tiny-LR interface co-adaptation beats readout-only: {summary['judgements']['tiny_lr_interface_beats_readout_only']}",
            f"- all_leg_rows is the better first co-adapt target: {summary['judgements']['all_leg_better_than_all_rot_for_first_target']}",
            f"- root-cause side to prioritize: {summary['judgements']['root_cause_priority']}",
            f"- adapter remains lower priority: {summary['judgements']['adapter_not_first_priority']}",
        ]
    )
    return "\n".join(lines) + "\n"


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
            "epochs": 1,
            "steps_per_epoch": 60,
            "lr": 5e-5,
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
    }


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
        ]
    )
    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    current_summary = _load_eval_summary(CURRENT_CONTROL_EVAL)
    baseline_summary = _load_eval_summary(BASELINE_REPLACE_EVAL)

    case_payloads: Dict[str, Any] = {}
    for case in CASE_SPECS:
        log(f"running {case['name']}")
        case_payloads[str(case["name"])] = _run_posttrain_case(case)
        write_json(
            STATUS_JSON,
            {
                "done_cases": list(case_payloads.keys()),
                "total_cases": len(CASE_SPECS),
            },
        )

    summaries: Dict[str, Any] = {
        "tail_current_control": current_summary,
        "baseline_replace": baseline_summary,
    }
    for name, payload in case_payloads.items():
        summaries[name] = payload["summary"]

    primary_pose_rows: List[Dict[str, Any]] = [
        _overall_primary_row("current_frozen_trunk_replace_control", current_summary, current_summary)
    ]
    primary_pose_rows.extend(
        _overall_primary_row(name, payload["summary"], current_summary) for name, payload in case_payloads.items()
    )
    primary_pose_rows.append(_overall_primary_row("baseline_replace", baseline_summary, current_summary))

    bucket_rows: List[Dict[str, Any]] = [
        _bucket_row("current_frozen_trunk_replace_control", current_summary, current_summary)
    ]
    bucket_rows.extend(_bucket_row(name, payload["summary"], current_summary) for name, payload in case_payloads.items())
    bucket_rows.append(_bucket_row("baseline_replace", baseline_summary, current_summary))

    ranked_rows: List[Dict[str, Any]] = []
    for name, payload in case_payloads.items():
        row = {
            "variant": name,
            "primary_relative_improvement_vs_current": _primary_relative_mean(payload["summary"], current_summary),
            "primary_relative_improvement_vs_baseline_gap_anchor": _primary_relative_mean(payload["summary"], baseline_summary),
        }
        for bucket in REQUIRED_BUCKETS:
            row[bucket] = _primary_bucket_relative_mean(payload["summary"], current_summary, bucket)
        ranked_rows.append(row)
    ranked_rows.sort(
        key=lambda row: (
            _safe_float(row["primary_relative_improvement_vs_current"]),
            _safe_float(row["d10_20"]) + _safe_float(row["d21_43"]) + _safe_float(row["sic11_21"]) + _safe_float(row["sic22_43"]),
        ),
        reverse=True,
    )

    judgements = {
        "tiny_lr_interface_beats_readout_only": (
            _safe_float(_primary_relative_mean(summaries["coadapt_allrot_interface"], current_summary))
            > _safe_float(_primary_relative_mean(summaries["readout_only_allrot"], current_summary))
            and _safe_float(_primary_relative_mean(summaries["coadapt_allleg_interface"], current_summary))
            > _safe_float(_primary_relative_mean(summaries["readout_only_allleg"], current_summary))
        ),
        "all_leg_better_than_all_rot_for_first_target": (
            _safe_float(_primary_relative_mean(summaries["coadapt_allleg_interface"], current_summary))
            > _safe_float(_primary_relative_mean(summaries["coadapt_allrot_interface"], current_summary))
        ),
        "root_cause_priority": (
            "allow interface bidirectional negotiation"
            if max(
                _safe_float(_primary_relative_mean(summaries["coadapt_allrot_interface"], current_summary)),
                _safe_float(_primary_relative_mean(summaries["coadapt_allleg_interface"], current_summary)),
            )
            > max(
                _safe_float(_primary_relative_mean(summaries["readout_only_allrot"], current_summary)),
                _safe_float(_primary_relative_mean(summaries["readout_only_allleg"], current_summary)),
            )
            else "continue readout-only unilateral adaptation"
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
        "code_facts": {
            "current_posttrain_train_modes_before_this_followup": [
                "train_direct_pose -> direct_pose_* only",
                "train_lambda_head -> lambda_fusion_head only",
                "train_arm_residual -> arm_residual_corrector only",
                "train_arm_leg_residual -> arm_residual_corrector only",
            ],
            "new_train_mode": "train_incremental_replace",
            "new_param_group_plumbing": {
                "readout_group": "default AdamW group at cfg.lr for the final motion_head Linear",
                "interface_group": "auto-created incremental_interface group at cfg.lr * incremental_interface_lr_scale",
                "interface_modes": ["off", "tail", "coupling_norm_only"],
            },
            "new_row_mask_plumbing": {
                "target_module": "motion_head last Linear only",
                "config_key": "incremental_motion_head_row_ranges",
                "mask_behavior": "zero grad for unselected output rows on weight and bias",
            },
            "tail_interface_modules": [
                "shared_encoder last block",
                "residual_proj",
                "_pasa_lnq",
                "_pasa_q",
                "_pasa_k",
                "_pasa_v",
                "_pasa_o",
                "_pasa_film",
                "coupling_norm",
            ],
        },
        "references": {
            "current_70a_ckpt": str(CURRENT_70A_CKPT),
            "current_frozen_trunk_replace_control_ckpt": str(CURRENT_CONTROL_CKPT),
            "current_frozen_trunk_replace_control_eval": str(CURRENT_CONTROL_EVAL),
            "baseline_replace_ckpt": str(BASELINE_REPLACE_CKPT),
            "baseline_replace_eval": str(BASELINE_REPLACE_EVAL),
            "lowlr_winner_config": str(LOWLR_WINNER_CONFIG),
        },
        "experiment_matrix": [
            {
                "name": str(case["name"]),
                "row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
                "interface_mode": str(case["interface_mode"]),
                "interface_lr_scale": float(case["interface_lr_scale"]),
            }
            for case in CASE_SPECS
        ],
        "cases": case_payloads,
        "tables": {
            "primary_pose": primary_pose_rows,
            "bucket_primary_relative_vs_current": bucket_rows,
            "ranking": ranked_rows,
        },
        "judgements": judgements,
    }
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(_build_markdown(summary), encoding="utf-8")
    write_json(STATUS_JSON, {"done_cases": list(case_payloads.keys()), "total_cases": len(CASE_SPECS), "completed": True})
    log(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
