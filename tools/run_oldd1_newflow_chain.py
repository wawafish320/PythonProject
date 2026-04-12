#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        parse_reference_payload,
        run_70r_promote,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        parse_reference_payload,
        run_70r_promote,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_oldd1_newflow_chain_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_oldd1_newflow_chain_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
STAGE_MODEL_EVAL_ROOT = OUT_ROOT / "stage_eval_model"

STAGE6_COMPARE_JSON = ROOT / "debug_output" / "_tmp_stage6_basetrain_compare_20260313" / "compare_summary.json"
FULL_OLDPLAN_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_oldplan_downstream_chain_20260314" / "summary.json"
ROLLBACK_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_oldplan_component_ablation_20260314" / "summary.json"
BASE_CKPT = ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth"
STAGE6_CASE_NAME = "old_bestfree"

TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ("stage6", "70a"),
    ("70a", "70b"),
    ("70b", "new70b_replace"),
    ("new70b_replace", "70R"),
    ("70R", "71"),
    ("71", "72"),
    ("72", "lambda"),
)

DIRECT_GROUP_KEYS = ("all_ex_root", "leg", "nonleg", "arm", "else")
MASKED_KEYS = (
    "DirectGeoLocalDeg",
    "DirectGeoLocalDegWeighted",
    "BlendGeoLocalDeg",
    "BlendGeoLocalDegWeighted",
    "GeoLocalDeg",
    "GeoLocalDegWeighted",
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_stage6_case(name: str) -> Dict[str, Any]:
    payload = load_json(STAGE6_COMPARE_JSON)
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        raise RuntimeError(f"invalid lanes in {STAGE6_COMPARE_JSON}")
    for lane in lanes:
        if isinstance(lane, dict) and str(lane.get("name")) == name:
            return lane
    raise RuntimeError(f"missing lane {name} in {STAGE6_COMPARE_JSON}")


def load_full_oldplan_reference() -> Dict[str, Any]:
    return load_json(FULL_OLDPLAN_SUMMARY_JSON)


def load_rollback_reference() -> Dict[str, Any]:
    payload = load_json(ROLLBACK_SUMMARY_JSON)
    promoted = payload.get("downstream", {}).get("promoted_lanes", [])
    if not isinstance(promoted, list):
        raise RuntimeError(f"invalid promoted_lanes in {ROLLBACK_SUMMARY_JSON}")
    for lane in promoted:
        if isinstance(lane, dict) and str(lane.get("name")) == "rollback_planner_core":
            return lane
    raise RuntimeError(f"missing rollback_planner_core in {ROLLBACK_SUMMARY_JSON}")


def build_paths() -> Dict[str, Any]:
    return {
        "lane_log": OUT_ROOT / "lane.log",
        "status_json": OUT_ROOT / "status.json",
        "summary_json": OUT_ROOT / "summary.json",
        "summary_md": OUT_ROOT / "summary.md",
        "cfg_70b_replace": CONFIG_ROOT / f"posttrain_70b_replacecontacts_from_oldd1_{RUN_DATE}.json",
        "cfg_70r": CONFIG_ROOT / f"posttrain_70R_from_oldd1_replace_lr3e4_e1_s60_{RUN_DATE}.json",
        "warmstart_ckpt": MODEL_ROOT / "warmstart" / f"ckpt_last_oldd1_70a_replacecontacts_zerophase_{RUN_DATE}.pth",
        "warmstart_report": OUT_ROOT / "warmstart" / "replace_zerophase_report.json",
        "ckpt_70a": MODEL_ROOT / "70a" / f"ckpt_last_WalkF_stage7_70a_from_oldd1_newflow_{RUN_DATE}.pth",
        "ckpt_70b": MODEL_ROOT / "70b" / f"ckpt_last_WalkF_stage7_70b_from_oldd1_newflow_{RUN_DATE}.pth",
        "ckpt_70b_replace": MODEL_ROOT / "70b_replace" / f"ckpt_last_WalkF_stage7_70b_replace_from_oldd1_newflow_{RUN_DATE}.pth",
        "ckpt_70r": MODEL_ROOT / "70R" / f"ckpt_last_WalkF_stage7_70R_from_oldd1_newflow_s180_{RUN_DATE}.pth",
        "ckpt_71": MODEL_ROOT / "71" / f"ckpt_last_WalkF_stage7_71_from_oldd1_newflow_{RUN_DATE}.pth",
        "ckpt_72": MODEL_ROOT / "72" / f"ckpt_last_WalkF_stage7_72_from_oldd1_newflow_{RUN_DATE}.pth",
        "ckpt_lambda": MODEL_ROOT / "lambda" / f"ckpt_last_WalkF_stage7_lambda_from_oldd1_newflow_{RUN_DATE}.pth",
        "stage_eval_model": {
            "stage6": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "stage6",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "stage6" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "stage6_group_summary.json",
            },
            "70a": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "70a",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "70a" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "70a_group_summary.json",
            },
            "70b": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "70b",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "70b" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "70b_group_summary.json",
            },
            "new70b_replace": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "new70b_replace",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "new70b_replace" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "new70b_replace_group_summary.json",
            },
            "70R": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "70R",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "70R" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "70R_group_summary.json",
            },
            "71": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "71",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "71" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "71_group_summary.json",
            },
            "72": {
                "eval_dir": STAGE_MODEL_EVAL_ROOT / "72",
                "eval_json": STAGE_MODEL_EVAL_ROOT / "72" / "Walk_F_freerun_cycles.json",
                "group_json": STAGE_MODEL_EVAL_ROOT / "72_group_summary.json",
            },
            "lambda": {
                "eval_dir": OUT_ROOT / "eval_model_source",
                "eval_json": OUT_ROOT / "eval_model_source" / "Walk_F_freerun_cycles.json",
                "group_json": OUT_ROOT / "eval_model_source_group_summary.json",
            },
        },
        "eval_strict_dir": OUT_ROOT / "eval_pretrain_contact",
        "eval_strict_json": OUT_ROOT / "eval_pretrain_contact" / "Walk_F_freerun_cycles.json",
        "eval_strict_group": OUT_ROOT / "eval_pretrain_contact_group_summary.json",
    }


def collect_eval(eval_json: Path, group_json: Path) -> Dict[str, Any]:
    return {
        "masked_means": masked_metric_means(eval_json),
        "direct_group_summary": group_metrics(group_json),
        "window_summary": window_group_stats(eval_json),
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def run_stage_model_eval(
    *,
    stage_name: str,
    model_ckpt: Path,
    eval_dir: Path,
    group_json: Path,
    log_file: Path,
) -> Dict[str, Any]:
    log(f"=== model-source eval: {stage_name} ===")
    eval_json = run_eval(
        model_ckpt=model_ckpt,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=log_file,
    )
    ensure_group_summary(eval_json, group_json, log_file=log_file)
    return collect_eval(eval_json, group_json)


def delta_block(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def transition_deltas(stage_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for prev_key, cur_key in TRANSITIONS:
        prev = stage_metrics[prev_key]
        cur = stage_metrics[cur_key]
        out[f"{prev_key}_to_{cur_key}"] = {
            "masked_means_delta": delta_block(cur["masked_means"], prev["masked_means"], MASKED_KEYS),
            "direct_group_delta": delta_block(cur["direct_group_summary"], prev["direct_group_summary"], DIRECT_GROUP_KEYS),
        }
    return out


def classify_direct_group(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    deltas = [safe_float(diff(cur.get(key), ref.get(key))) for key in ("all_ex_root", "leg", "nonleg")]
    neg = sum(1 for v in deltas if math.isfinite(v) and v < 0.0)
    pos = sum(1 for v in deltas if math.isfinite(v) and v > 0.0)
    if neg > 0 and pos == 0:
        return "better"
    if pos > 0 and neg == 0:
        return "worse"
    if neg == 0 and pos == 0:
        return "tie"
    return "mixed"


def best_and_worst_transition(
    transition_payload: Mapping[str, Any],
    metric_family: str,
    metric_key: str,
) -> Dict[str, Any]:
    rows: List[Tuple[str, float]] = []
    for name, payload in transition_payload.items():
        family = payload.get(metric_family, {})
        value = safe_float(family.get(metric_key))
        if math.isfinite(value):
            rows.append((name, value))
    if not rows:
        return {
            "best_improvement": {"transition": None, "delta": float("nan")},
            "worst_regression": {"transition": None, "delta": float("nan")},
        }
    best_name, best_val = min(rows, key=lambda item: item[1])
    worst_name, worst_val = max(rows, key=lambda item: item[1])
    return {
        "best_improvement": {"transition": best_name, "delta": best_val},
        "worst_regression": {"transition": worst_name, "delta": worst_val},
    }


def final_compare_block(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": classify_direct_group(cur["direct_group_summary"], ref["direct_group_summary"]),
        "masked_means_delta": delta_block(cur["masked_means"], ref["masked_means"], MASKED_KEYS),
        "direct_group_delta": delta_block(cur["direct_group_summary"], ref["direct_group_summary"], DIRECT_GROUP_KEYS),
    }


def build_summary(
    *,
    stage6_case: Mapping[str, Any],
    paths: Mapping[str, Any],
    stage_model_metrics: Mapping[str, Any],
    final_strict_eval: Mapping[str, Any],
) -> Dict[str, Any]:
    accepted_refs = parse_reference_payload()
    accepted_anchor = accepted_refs["accepted_final_model_source"]
    full_oldplan = load_full_oldplan_reference()
    rollback_lane = load_rollback_reference()

    full_oldplan_model = full_oldplan["final_evals"]["model_source"]
    full_oldplan_strict = full_oldplan["final_evals"]["strict_pretrain_contact"]
    rollback_model = rollback_lane["final_evals"]["model_source"]
    rollback_strict = rollback_lane["final_evals"]["strict_pretrain_contact"]

    stage6_strict = {
        "direct_group_summary": {
            "all_ex_root": safe_float(stage6_case["stage6_exit"]["all_ex_root_mean"]),
            "leg": safe_float(stage6_case["stage6_exit"]["leg_mean"]),
            "nonleg": safe_float(stage6_case["stage6_exit"]["nonleg_mean"]),
            "arm": safe_float(stage6_case["stage6_exit"]["arm_mean"]),
            "else": safe_float(stage6_case["stage6_exit"]["else_mean"]),
        },
        "paths": {
            "group_summary": str(stage6_case["paths"]["stage6_group_summary"]),
            "stage6_ckpt": str(stage6_case["paths"]["stage6_ckpt"]),
        },
    }

    transitions = transition_deltas(stage_model_metrics)
    all_root_progress = best_and_worst_transition(transitions, "direct_group_delta", "all_ex_root")
    leg_progress = best_and_worst_transition(transitions, "direct_group_delta", "leg")

    final_model = stage_model_metrics["lambda"]
    final_model_vs_anchor = final_compare_block(final_model, accepted_anchor)
    final_model_vs_full_oldplan = final_compare_block(final_model, full_oldplan_model)
    final_model_vs_rollback = final_compare_block(final_model, rollback_model)
    final_strict_vs_full_oldplan = final_compare_block(final_strict_eval, full_oldplan_strict)
    final_strict_vs_rollback = final_compare_block(final_strict_eval, rollback_strict)

    stage6_vs_full_oldplan = delta_block(
        stage6_strict["direct_group_summary"],
        full_oldplan["stage6_case"]["stage6_exit"],
        DIRECT_GROUP_KEYS,
    )
    stage6_vs_rollback = delta_block(
        stage6_strict["direct_group_summary"],
        rollback_lane["stage6_exit"],
        DIRECT_GROUP_KEYS,
    )

    strict_vs_model_consistency = {
        "vs_full_oldplan_chain": final_strict_vs_full_oldplan["status"] == final_model_vs_full_oldplan["status"],
        "vs_rollback_planner_core": final_strict_vs_rollback["status"] == final_model_vs_rollback["status"],
    }

    worth_challenger_lane = final_model_vs_full_oldplan["status"] in ("better", "mixed") and final_model_vs_rollback["status"] in ("better", "mixed")
    eligible_for_promote = final_model_vs_anchor["status"] == "better"

    return {
        "run_date": RUN_DATE,
        "policy": {
            "base_ckpt": str(BASE_CKPT),
            "stage6_source_summary": str(STAGE6_COMPARE_JSON),
            "full_oldplan_summary": str(FULL_OLDPLAN_SUMMARY_JSON),
            "rollback_component_summary": str(ROLLBACK_SUMMARY_JSON),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "final_eval_strict": {
                "contacts_meas_source": "pretrain_contact",
                "contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
                "contacts_meas_pretrain_affine_stats": str(AFFINE_STATS),
                "encoder_bundle": str(ENCODER_BUNDLE),
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "lambda_fusion_apply": True,
                "log_contacts": True,
                "export_direct_arm_probe": True,
                "export_joint_direct_geolocal_series": True,
            },
            "stage_progress_eval_model_source": {
                "contacts_meas_source": "model",
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "lambda_fusion_apply": True,
                "log_contacts": True,
                "export_direct_arm_probe": True,
                "export_joint_direct_geolocal_series": True,
            },
        },
        "stage6": {
            "name": stage6_case["name"],
            "strict_pretrain_contact": stage6_strict,
            "model_source": stage_model_metrics["stage6"],
        },
        "checkpoints": {
            "stage6": str(stage6_case["paths"]["stage6_ckpt"]),
            "70a": str(paths["ckpt_70a"]),
            "70b": str(paths["ckpt_70b"]),
            "70a_replace_warmstart": str(paths["warmstart_ckpt"]),
            "new70b_replace": str(paths["ckpt_70b_replace"]),
            "70R": str(paths["ckpt_70r"]),
            "71": str(paths["ckpt_71"]),
            "72": str(paths["ckpt_72"]),
            "lambda": str(paths["ckpt_lambda"]),
        },
        "configs": {
            "stage6": str(ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"),
            "70a": str(CONFIG_70A),
            "70b": str(CONFIG_70B),
            "new70b_replace": str(paths["cfg_70b_replace"]),
            "70R": str(paths["cfg_70r"]),
            "71": str(CONFIG_71),
            "72": str(CONFIG_72),
            "lambda": str(CONFIG_LAMBDA),
        },
        "references": {
            "accepted_final_anchor_model_source": accepted_anchor,
            "full_oldplan_chain": {
                "stage6_strict": full_oldplan["stage6_case"]["stage6_exit"],
                "final_model_source": full_oldplan_model,
                "final_strict": full_oldplan_strict,
            },
            "rollback_planner_core_challenger": {
                "stage6_strict": rollback_lane["stage6_exit"],
                "final_model_source": rollback_model,
                "final_strict": rollback_strict,
            },
        },
        "stage_progress_model_source": stage_model_metrics,
        "stage_progress_model_source_deltas": transitions,
        "final_evals": {
            "model_source": final_model,
            "strict_pretrain_contact": final_strict_eval,
        },
        "comparisons": {
            "stage6_strict_vs_full_oldplan_control": stage6_vs_full_oldplan,
            "stage6_strict_vs_rollback_planner_core": stage6_vs_rollback,
            "final_model_source_vs_accepted_anchor": final_model_vs_anchor,
            "final_model_source_vs_full_oldplan_chain": final_model_vs_full_oldplan,
            "final_model_source_vs_rollback_planner_core": final_model_vs_rollback,
            "final_strict_vs_full_oldplan_chain": final_strict_vs_full_oldplan,
            "final_strict_vs_rollback_planner_core": final_strict_vs_rollback,
            "strict_vs_model_consistency": strict_vs_model_consistency,
        },
        "answers": {
            "q1_stage6_vs_controls": {
                "overall": "worse" if stage6_vs_full_oldplan["all_ex_root"] > 0.0 or stage6_vs_full_oldplan["leg"] > 0.0 else "better_or_tie",
                "note": (
                    "Stage6 is judged mainly by all_ex_root and leg; nonleg is included as a secondary check."
                ),
            },
            "q2_main_progress_step_all_ex_root": all_root_progress,
            "q2_main_progress_step_leg": leg_progress,
            "q3_final_model_source_vs_full_oldplan_chain": final_model_vs_full_oldplan["status"],
            "q3_final_model_source_vs_rollback_planner_core": final_model_vs_rollback["status"],
            "q3_final_model_source_vs_current_accepted_anchor": final_model_vs_anchor["status"],
            "q3_final_strict_vs_full_oldplan_chain": final_strict_vs_full_oldplan["status"],
            "q3_final_strict_vs_rollback_planner_core": final_strict_vs_rollback["status"],
            "q3_final_strict_vs_current_accepted_anchor": "cross_contract_only",
            "q4_strict_vs_model_consistency": strict_vs_model_consistency,
            "q5_worth_new_challenger_lane": bool(worth_challenger_lane),
            "q6_eligible_for_baseline_or_promote_discussion": bool(eligible_for_promote),
            "q6_note": (
                "Accepted anchor is archived only as model-source in repo; strict-vs-accepted remains cross-contract."
            ),
        },
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    stage6 = summary["stage6"]
    refs = summary["references"]
    comps = summary["comparisons"]
    answers = summary["answers"]
    final_model = summary["final_evals"]["model_source"]
    final_strict = summary["final_evals"]["strict_pretrain_contact"]

    lines: List[str] = []
    lines.append("# old d1 basetrain + new posttrain flow")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- base_ckpt: `{summary['policy']['base_ckpt']}`")
    lines.append(f"- stage6_ckpt: `{summary['checkpoints']['stage6']}`")
    lines.append(
        f"- stage6 strict: all_ex_root={fmt(stage6['strict_pretrain_contact']['direct_group_summary']['all_ex_root'])}, "
        f"leg={fmt(stage6['strict_pretrain_contact']['direct_group_summary']['leg'])}, "
        f"nonleg={fmt(stage6['strict_pretrain_contact']['direct_group_summary']['nonleg'])}, "
        f"arm={fmt(stage6['strict_pretrain_contact']['direct_group_summary']['arm'])}, "
        f"else={fmt(stage6['strict_pretrain_contact']['direct_group_summary']['else'])}"
    )
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append("| stage | ckpt |")
    lines.append("|---|---|")
    for key in ("stage6", "70a", "70b", "70a_replace_warmstart", "new70b_replace", "70R", "71", "72", "lambda"):
        lines.append(f"| {key} | `{summary['checkpoints'][key]}` |")
    lines.append("")
    lines.append("## Stage progress (model-source)")
    lines.append("")
    lines.append("| stage | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg | arm | else |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("stage6", "70a", "70b", "new70b_replace", "70R", "71", "72", "lambda"):
        row = summary["stage_progress_model_source"][key]
        mm = row["masked_means"]
        gg = row["direct_group_summary"]
        lines.append(
            f"| {key} | {fmt(mm.get('DirectGeoLocalDeg'))} | {fmt(mm.get('BlendGeoLocalDeg'))} | {fmt(mm.get('GeoLocalDeg'))} | "
            f"{fmt(gg.get('all_ex_root'))} | {fmt(gg.get('leg'))} | {fmt(gg.get('nonleg'))} | {fmt(gg.get('arm'))} | {fmt(gg.get('else'))} |"
        )
    lines.append("")
    lines.append("| transition | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ("stage6_to_70a", "70a_to_70b", "70b_to_new70b_replace", "new70b_replace_to_70R", "70R_to_71", "71_to_72", "72_to_lambda"):
        row = summary["stage_progress_model_source_deltas"][key]["direct_group_delta"]
        lines.append(
            f"| {key} | {fmt(row.get('all_ex_root'))} | {fmt(row.get('leg'))} | {fmt(row.get('nonleg'))} | {fmt(row.get('arm'))} | {fmt(row.get('else'))} |"
        )
    lines.append("")
    lines.append("## Final evals")
    lines.append("")
    lines.append("| lane | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg | arm | else |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lane_name, lane in (("model_source", final_model), ("strict_pretrain_contact", final_strict)):
        mm = lane["masked_means"]
        gg = lane["direct_group_summary"]
        lines.append(
            f"| {lane_name} | {fmt(mm.get('DirectGeoLocalDeg'))} | {fmt(mm.get('BlendGeoLocalDeg'))} | {fmt(mm.get('GeoLocalDeg'))} | "
            f"{fmt(gg.get('all_ex_root'))} | {fmt(gg.get('leg'))} | {fmt(gg.get('nonleg'))} | {fmt(gg.get('arm'))} | {fmt(gg.get('else'))} |"
        )
    lines.append("")
    lines.append("## Final direct-path windows")
    lines.append("")
    lines.append("| lane | section | legs_main | arms_main | left_arm_main | right_arm_main |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for lane_name, lane in (("model_source", final_model), ("strict_pretrain_contact", final_strict)):
        for sec in ("overall", "A_52_59", "B_76_80"):
            row = lane["window_summary"][sec]
            lines.append(
                f"| {lane_name} | {sec} | {fmt(row.get('legs_main'))} | {fmt(row.get('arms_main'))} | "
                f"{fmt(row.get('left_arm_main'))} | {fmt(row.get('right_arm_main'))} |"
            )
    lines.append("")
    lines.append("| lane | foot_l_ball_l_SIC12_15 | calf_r_SIC2_4 |")
    lines.append("|---|---:|---:|")
    for lane_name, lane in (("model_source", final_model), ("strict_pretrain_contact", final_strict)):
        hot = lane["window_summary"]["hotspots"]
        lines.append(f"| {lane_name} | {fmt(hot.get('foot_l_ball_l_SIC12_15'))} | {fmt(hot.get('calf_r_SIC2_4'))} |")
    lines.append("")
    lines.append("## Reference controls")
    lines.append("")
    lines.append("| ref | contract | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    anchor = refs["accepted_final_anchor_model_source"]
    lines.append(
        f"| accepted_final_anchor | model | {fmt(anchor['masked_means'].get('DirectGeoLocalDeg'))} | "
        f"{fmt(anchor['masked_means'].get('BlendGeoLocalDeg'))} | {fmt(anchor['masked_means'].get('GeoLocalDeg'))} | "
        f"{fmt(anchor['direct_group_summary'].get('all_ex_root'))} | {fmt(anchor['direct_group_summary'].get('leg'))} | "
        f"{fmt(anchor['direct_group_summary'].get('nonleg'))} |"
    )
    full_oldplan = refs["full_oldplan_chain"]
    lines.append(
        f"| full_oldplan_chain | model | {fmt(full_oldplan['final_model_source']['masked_means'].get('DirectGeoLocalDeg'))} | "
        f"{fmt(full_oldplan['final_model_source']['masked_means'].get('BlendGeoLocalDeg'))} | {fmt(full_oldplan['final_model_source']['masked_means'].get('GeoLocalDeg'))} | "
        f"{fmt(full_oldplan['final_model_source']['direct_group_summary'].get('all_ex_root'))} | {fmt(full_oldplan['final_model_source']['direct_group_summary'].get('leg'))} | "
        f"{fmt(full_oldplan['final_model_source']['direct_group_summary'].get('nonleg'))} |"
    )
    lines.append(
        f"| full_oldplan_chain | strict | {fmt(full_oldplan['final_strict']['masked_means'].get('DirectGeoLocalDeg'))} | "
        f"{fmt(full_oldplan['final_strict']['masked_means'].get('BlendGeoLocalDeg'))} | {fmt(full_oldplan['final_strict']['masked_means'].get('GeoLocalDeg'))} | "
        f"{fmt(full_oldplan['final_strict']['direct_group_summary'].get('all_ex_root'))} | {fmt(full_oldplan['final_strict']['direct_group_summary'].get('leg'))} | "
        f"{fmt(full_oldplan['final_strict']['direct_group_summary'].get('nonleg'))} |"
    )
    rollback = refs["rollback_planner_core_challenger"]
    lines.append(
        f"| rollback_planner_core | model | {fmt(rollback['final_model_source']['masked_means'].get('DirectGeoLocalDeg'))} | "
        f"{fmt(rollback['final_model_source']['masked_means'].get('BlendGeoLocalDeg'))} | {fmt(rollback['final_model_source']['masked_means'].get('GeoLocalDeg'))} | "
        f"{fmt(rollback['final_model_source']['direct_group_summary'].get('all_ex_root'))} | {fmt(rollback['final_model_source']['direct_group_summary'].get('leg'))} | "
        f"{fmt(rollback['final_model_source']['direct_group_summary'].get('nonleg'))} |"
    )
    lines.append(
        f"| rollback_planner_core | strict | {fmt(rollback['final_strict']['masked_means'].get('DirectGeoLocalDeg'))} | "
        f"{fmt(rollback['final_strict']['masked_means'].get('BlendGeoLocalDeg'))} | {fmt(rollback['final_strict']['masked_means'].get('GeoLocalDeg'))} | "
        f"{fmt(rollback['final_strict']['direct_group_summary'].get('all_ex_root'))} | {fmt(rollback['final_strict']['direct_group_summary'].get('leg'))} | "
        f"{fmt(rollback['final_strict']['direct_group_summary'].get('nonleg'))} |"
    )
    lines.append("")
    lines.append("## Final deltas")
    lines.append("")
    lines.append("| compare | contract | d_DirectGeoLocalDeg | d_BlendGeoLocalDeg | d_GeoLocalDeg | d_all_ex_root | d_leg | d_nonleg |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, contract, payload in (
        ("accepted_anchor", "model", comps["final_model_source_vs_accepted_anchor"]),
        ("full_oldplan_chain", "model", comps["final_model_source_vs_full_oldplan_chain"]),
        ("rollback_planner_core", "model", comps["final_model_source_vs_rollback_planner_core"]),
        ("full_oldplan_chain", "strict", comps["final_strict_vs_full_oldplan_chain"]),
        ("rollback_planner_core", "strict", comps["final_strict_vs_rollback_planner_core"]),
    ):
        lines.append(
            f"| {name} | {contract} | {fmt(payload['masked_means_delta'].get('DirectGeoLocalDeg'))} | "
            f"{fmt(payload['masked_means_delta'].get('BlendGeoLocalDeg'))} | {fmt(payload['masked_means_delta'].get('GeoLocalDeg'))} | "
            f"{fmt(payload['direct_group_delta'].get('all_ex_root'))} | {fmt(payload['direct_group_delta'].get('leg'))} | "
            f"{fmt(payload['direct_group_delta'].get('nonleg'))} |"
        )
    lines.append("")
    lines.append("## Requested answers")
    lines.append("")
    lines.append(
        f"1. Stage6 vs controls: old d1 Stage6 is `worse overall` than both full oldplan control and rollback_planner_core on strict Stage6 "
        f"(vs full oldplan d_all_ex_root={fmt(comps['stage6_strict_vs_full_oldplan_control']['all_ex_root'])}, "
        f"d_leg={fmt(comps['stage6_strict_vs_full_oldplan_control']['leg'])}; "
        f"vs rollback d_all_ex_root={fmt(comps['stage6_strict_vs_rollback_planner_core']['all_ex_root'])}, "
        f"d_leg={fmt(comps['stage6_strict_vs_rollback_planner_core']['leg'])}). "
        f"Nonleg is slightly better in both comparisons."
    )
    root_best = answers["q2_main_progress_step_all_ex_root"]["best_improvement"]
    root_worst = answers["q2_main_progress_step_all_ex_root"]["worst_regression"]
    leg_best = answers["q2_main_progress_step_leg"]["best_improvement"]
    leg_worst = answers["q2_main_progress_step_leg"]["worst_regression"]
    lines.append(
        f"2. Main improvement step: all_ex_root `{root_best['transition']}` ({fmt(root_best['delta'])}); "
        f"main regression step: `{root_worst['transition']}` ({fmt(root_worst['delta'])}). "
        f"For leg, best=`{leg_best['transition']}` ({fmt(leg_best['delta'])}), worst=`{leg_worst['transition']}` ({fmt(leg_worst['delta'])})."
    )
    lines.append(
        f"3. Final model-source status: vs full oldplan `{answers['q3_final_model_source_vs_full_oldplan_chain']}`, "
        f"vs rollback_planner_core `{answers['q3_final_model_source_vs_rollback_planner_core']}`, "
        f"vs accepted anchor `{answers['q3_final_model_source_vs_current_accepted_anchor']}`."
    )
    lines.append(
        f"4. Final strict status: vs full oldplan `{answers['q3_final_strict_vs_full_oldplan_chain']}`, "
        f"vs rollback_planner_core `{answers['q3_final_strict_vs_rollback_planner_core']}`; "
        f"vs accepted anchor is `cross_contract_only` because the archived accepted anchor is model-source only."
    )
    lines.append(
        f"5. Strict/model consistency: vs full oldplan `{str(bool(comps['strict_vs_model_consistency']['vs_full_oldplan_chain'])).lower()}`, "
        f"vs rollback `{str(bool(comps['strict_vs_model_consistency']['vs_rollback_planner_core'])).lower()}`."
    )
    lines.append(
        f"6. Worth a challenger lane: `{str(bool(answers['q5_worth_new_challenger_lane'])).lower()}`. "
        f"Eligible for baseline/promote discussion now: `{str(bool(answers['q6_eligible_for_baseline_or_promote_discussion'])).lower()}` "
        f"({answers['q6_note']})."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        STAGE6_COMPARE_JSON,
        FULL_OLDPLAN_SUMMARY_JSON,
        ROLLBACK_SUMMARY_JSON,
        BASE_CKPT,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE_MODEL_EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    stage6_case = load_stage6_case(STAGE6_CASE_NAME)
    stage6_ckpt = Path(str(stage6_case["paths"]["stage6_ckpt"]))
    if not stage6_ckpt.is_file():
        raise SystemExit(f"missing stage6 ckpt: {stage6_ckpt}")

    paths = build_paths()

    run_name_70a = f"WalkF_stage7_70a_from_oldd1_newflow_{RUN_DATE}"
    run_name_70b = f"WalkF_stage7_70b_from_oldd1_newflow_{RUN_DATE}"
    run_name_70b_replace = f"WalkF_stage7_70b_replace_from_oldd1_newflow_{RUN_DATE}"
    run_name_70r = f"WalkF_stage7_70R_from_oldd1_newflow_s180_{RUN_DATE}"
    run_name_71 = f"WalkF_stage7_71_from_oldd1_newflow_{RUN_DATE}"
    run_name_72 = f"WalkF_stage7_72_from_oldd1_newflow_{RUN_DATE}"
    run_name_lambda = f"WalkF_stage7_lambda_from_oldd1_newflow_{RUN_DATE}"

    log("=== stage 70a ===")
    ckpt_70a = run_posttrain_stage(
        config=CONFIG_70A,
        ckpt_in=stage6_ckpt,
        out_dir=MODEL_ROOT / "70a",
        run_name=run_name_70a,
        log_file=paths["lane_log"],
    )

    log("=== stage 70b ===")
    ckpt_70b = run_posttrain_stage(
        config=CONFIG_70B,
        ckpt_in=ckpt_70a,
        out_dir=MODEL_ROOT / "70b",
        run_name=run_name_70b,
        log_file=paths["lane_log"],
    )

    log("=== 70a replace zerophase warmstart ===")
    create_replace_zerophase_warmstart(
        src_ckpt=ckpt_70a,
        dst_ckpt=paths["warmstart_ckpt"],
        report_json=paths["warmstart_report"],
    )

    log("=== new70b replace ===")
    cfg_70b_replace = make_generated_config(
        CONFIG_70B,
        paths["cfg_70b_replace"],
        {
            "ckpt_in": str(paths["warmstart_ckpt"]),
            "out_dir": str(MODEL_ROOT / "70b_replace"),
            "run_name": run_name_70b_replace,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70b_replace = run_posttrain_stage(
        config=cfg_70b_replace,
        ckpt_in=paths["warmstart_ckpt"],
        out_dir=MODEL_ROOT / "70b_replace",
        run_name=run_name_70b_replace,
        log_file=paths["lane_log"],
    )

    log("=== promoted 70R s180 ===")
    cfg_70r = make_generated_config(
        CONFIG_70R,
        paths["cfg_70r"],
        {
            "ckpt_in": str(ckpt_70b_replace),
            "out_dir": str(MODEL_ROOT / "70R"),
            "run_name": run_name_70r,
            "lr": 3e-4,
            "epochs": 1,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70r = run_70r_promote(
        config_json=cfg_70r,
        out_dir=MODEL_ROOT / "70R",
        run_name=run_name_70r,
        log_file=paths["lane_log"],
    )

    log("=== stage 71 ===")
    ckpt_71 = run_posttrain_stage(
        config=CONFIG_71,
        ckpt_in=ckpt_70r,
        out_dir=MODEL_ROOT / "71",
        run_name=run_name_71,
        log_file=paths["lane_log"],
    )

    log("=== stage 72 ===")
    ckpt_72 = run_posttrain_stage(
        config=CONFIG_72,
        ckpt_in=ckpt_71,
        out_dir=MODEL_ROOT / "72",
        run_name=run_name_72,
        log_file=paths["lane_log"],
    )

    log("=== lambda final ===")
    ckpt_lambda = run_posttrain_stage(
        config=CONFIG_LAMBDA,
        ckpt_in=ckpt_72,
        out_dir=MODEL_ROOT / "lambda",
        run_name=run_name_lambda,
        log_file=paths["lane_log"],
    )

    stage_ckpts = {
        "stage6": stage6_ckpt,
        "70a": ckpt_70a,
        "70b": ckpt_70b,
        "new70b_replace": ckpt_70b_replace,
        "70R": ckpt_70r,
        "71": ckpt_71,
        "72": ckpt_72,
        "lambda": ckpt_lambda,
    }
    stage_model_metrics: Dict[str, Any] = {}
    for stage_name in ("stage6", "70a", "70b", "new70b_replace", "70R", "71", "72", "lambda"):
        spec = paths["stage_eval_model"][stage_name]
        stage_model_metrics[stage_name] = run_stage_model_eval(
            stage_name=stage_name,
            model_ckpt=stage_ckpts[stage_name],
            eval_dir=spec["eval_dir"],
            group_json=spec["group_json"],
            log_file=paths["lane_log"],
        )

    log("=== final strict eval ===")
    eval_strict_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_strict_dir"],
        contacts_source="pretrain_contact",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_strict_json, paths["eval_strict_group"], log_file=paths["lane_log"])
    final_strict_eval = collect_eval(eval_strict_json, paths["eval_strict_group"])

    status_payload = {
        "base_ckpt": str(BASE_CKPT),
        "stage6_case": STAGE6_CASE_NAME,
        "stage6_ckpt": str(stage6_ckpt),
        "stage_ckpts": {key: str(value) for key, value in stage_ckpts.items()},
        "warmstart_ckpt": str(paths["warmstart_ckpt"]),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(paths["status_json"], status_payload)

    summary = build_summary(
        stage6_case=stage6_case,
        paths=paths,
        stage_model_metrics=stage_model_metrics,
        final_strict_eval=final_strict_eval,
    )
    write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
