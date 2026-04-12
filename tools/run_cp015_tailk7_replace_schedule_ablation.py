#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        ENCODER_BUNDLE,
        ROOT,
        create_replace_zerophase_warmstart,
        ensure_group_summary,
        fmt,
        load_json,
        make_generated_config,
        run_eval,
        run_posttrain_stage,
        safe_float,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        ENCODER_BUNDLE,
        ROOT,
        create_replace_zerophase_warmstart,
        ensure_group_summary,
        fmt,
        load_json,
        make_generated_config,
        run_eval,
        run_posttrain_stage,
        safe_float,
        write_json,
    )


RUN_DATE = "20260402"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_schedule_ablation_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_schedule_ablation_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CURRENT_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
CURRENT_70A_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source_group_summary.json"
)
CURRENT_REPLACE_SUMMARY = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_from_70a_20260402"
    / "summary.json"
)
BASELINE_REPLACE_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift_group_summary.json"
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

GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "arm", "else")
PERCENTILES: Tuple[str, ...] = ("mean", "p50", "p90", "p95")
PRIMARY_CASES: Tuple[Tuple[str, int, int], ...] = (
    ("e2x60", 2, 60),
    ("e1x120", 1, 120),
)
CONDITIONAL_CASE: Tuple[str, int, int] = ("e3x60", 3, 60)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))


def group_percentiles(path: Path) -> Dict[str, Dict[str, float]]:
    obj = load_json(path)
    groups = obj.get("groups", {})
    out: Dict[str, Dict[str, float]] = {}
    for group_name in GROUPS:
        group_payload = groups.get(group_name, {}) if isinstance(groups, dict) else {}
        out[group_name] = {pct: safe_float(group_payload.get(pct)) for pct in PERCENTILES}
    return out


def aggregate_16(metrics: Mapping[str, Mapping[str, Any]]) -> float:
    total = 0.0
    count = 0
    for group_name in GROUPS:
        for pct in PERCENTILES:
            v = safe_float(metrics.get(group_name, {}).get(pct))
            if math.isfinite(v):
                total += v
                count += 1
    return float(total if count else float("nan"))


def compare_slots(cur: Mapping[str, Mapping[str, Any]], ref: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    delta: Dict[str, Dict[str, float]] = {}
    wins = 0
    ties = 0
    losses = 0
    worst_slot = ""
    worst_delta = float("-inf")
    all_leq = True
    any_lt = False

    for group_name in GROUPS:
        delta[group_name] = {}
        for pct in PERCENTILES:
            cur_v = safe_float(cur.get(group_name, {}).get(pct))
            ref_v = safe_float(ref.get(group_name, {}).get(pct))
            d = float(cur_v - ref_v) if math.isfinite(cur_v) and math.isfinite(ref_v) else float("nan")
            delta[group_name][pct] = d
            if not math.isfinite(d):
                all_leq = False
                continue
            if d < -1e-9:
                wins += 1
                any_lt = True
            elif d > 1e-9:
                losses += 1
                all_leq = False
                if d > worst_delta:
                    worst_delta = d
                    worst_slot = f"{group_name}.{pct}"
            else:
                ties += 1
            if d > worst_delta:
                worst_delta = d
                worst_slot = f"{group_name}.{pct}"

    return {
        "delta": delta,
        "slot_wins": wins,
        "slot_ties": ties,
        "slot_losses": losses,
        "all_leq": bool(all_leq and any_lt),
        "worst_positive_delta_slot": worst_slot if worst_delta > 0 else "",
        "worst_positive_delta": float(worst_delta if worst_delta > 0 else 0.0),
    }


def format_quad(metrics: Mapping[str, Any]) -> str:
    return " / ".join(fmt(metrics[pct]) for pct in PERCENTILES)


def build_command_list(case_name: str, cfg_json: Path, out_dir: Path, run_name: str, ckpt: Path, eval_dir: Path, group_json: Path) -> Dict[str, str]:
    train_cmd = (
        "PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain "
        f"--config {cfg_json} "
        f"--ckpt_in {WARMSTART_CKPT} "
        f"--out_dir {out_dir} "
        f"--run_name {run_name} "
        "--posttrain_contacts_source pretrain_contact "
        "--posttrain_contacts_pretrain_clamp 1.0 "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {ROOT / 'debug_output' / '_tmp_phaseb_affine_20260304' / 'affine_fit_mix08' / 'affine_stats.json'}"
    )
    eval_cmd = (
        "PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles "
        "--teacher validate/teacher_batches/Walk_F_teacher.json "
        f"--model {ckpt} "
        "--rounds 5 "
        "--depth 3 "
        "--time-index-mode cycle "
        "--event_clock auto "
        "--phase_reset_source none "
        "--contacts_meas_source model "
        "--lambda_fusion_apply "
        "--log_contacts "
        "--export_direct_arm_probe "
        "--export_joint_direct_geolocal_series "
        f"--out {eval_dir} "
        "--force"
    )
    summary_cmd = (
        "PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py "
        f"tools/phasea_group_summary.py {eval_dir / 'Walk_F_freerun_cycles.json'} "
        "--cycle_gte 1 "
        "--drop_wrap "
        f"--out {group_json}"
    )
    return {
        "case_name": case_name,
        "train": train_cmd,
        "eval": eval_cmd,
        "group_summary": summary_cmd,
    }


def run_case(case_name: str, epochs: int, steps_per_epoch: int) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_lr5e5_from_cp015_tailk7_70a_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_lr5e5_from_cp015_tailk7_70a_{RUN_DATE}.json"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"

    make_generated_config(
        LOWLR_WINNER_CONFIG,
        cfg_json,
        {
            "ckpt_in": str(WARMSTART_CKPT),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": 5e-5,
            "epochs": int(epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "weight_decay": 0.0,
            "encoder_bundle": str(ENCODER_BUNDLE),
        },
    )

    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=WARMSTART_CKPT,
        out_dir=out_dir,
        run_name=run_name,
        log_file=LOG_FILE,
    )
    eval_json = run_eval(model_ckpt=ckpt, out_dir=eval_dir, contacts_source="model", log_file=LOG_FILE)
    ensure_group_summary(eval_json, group_json, log_file=LOG_FILE)
    metrics = group_percentiles(group_json)
    return {
        "case_name": case_name,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "log": str(out_dir / f"posttrain_log_{run_name}.json"),
        "eval": str(eval_json),
        "group_summary": str(group_json),
        "metrics": metrics,
        "aggregate_16": aggregate_16(metrics),
        "commands": build_command_list(case_name, cfg_json, out_dir, run_name, ckpt, eval_dir, group_json),
    }


def row(label: str, metrics: Mapping[str, Mapping[str, Any]]) -> str:
    return (
        f"| {label} | {format_quad(metrics['all_ex_root'])} | {format_quad(metrics['leg'])} | "
        f"{format_quad(metrics['arm'])} | {format_quad(metrics['else'])} |"
    )


def delta_lines(label: str, delta_payload: Mapping[str, Any]) -> Iterable[str]:
    yield f"### {label}"
    yield ""
    yield "| group | d_mean | d_p50 | d_p90 | d_p95 |"
    yield "|---|---:|---:|---:|---:|"
    for group_name in GROUPS:
        yield (
            f"| {group_name} | {fmt(delta_payload['delta'][group_name]['mean'])} | "
            f"{fmt(delta_payload['delta'][group_name]['p50'])} | "
            f"{fmt(delta_payload['delta'][group_name]['p90'])} | "
            f"{fmt(delta_payload['delta'][group_name]['p95'])} |"
        )
    yield ""
    yield (
        f"- slot wins/ties/losses: {delta_payload['slot_wins']} / "
        f"{delta_payload['slot_ties']} / {delta_payload['slot_losses']}"
    )
    if delta_payload["worst_positive_delta_slot"]:
        yield (
            f"- worst positive delta: {delta_payload['worst_positive_delta_slot']} = "
            f"{fmt(delta_payload['worst_positive_delta'])}"
        )
    else:
        yield "- worst positive delta: none"
    yield ""


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]
    lines = [
        "# cp015 tailk7 replace schedule ablation",
        "",
        "## Policy",
        "",
        "- fixed donor: current 70a",
        "- fixed warmstart: copy-only replace_zerophase",
        "- fixed lr: 5e-5",
        f"- conditional third case executed: `{summary['policy']['conditional_case_executed']}`",
        "",
        "## Stage replace exit table",
        "",
        "| lane | all_ex_root mean/p50/p90/p95 | leg mean/p50/p90/p95 | arm mean/p50/p90/p95 | else mean/p50/p90/p95 |",
        "|---|---|---|---|---|",
        row("current_replace_1x60", refs["current_replace_1x60"]["metrics"]),
    ]
    for case_name in summary["cases_in_order"]:
        lines.append(row(case_name, summary["cases"][case_name]["metrics"]))
    lines.append(row("baseline_replace", refs["baseline_replace"]["metrics"]))
    lines.extend(["", "## Delta vs current replace 1x60", ""])
    for case_name in summary["cases_in_order"]:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_current_replace"]))
    lines.extend(["## Delta vs baseline replace", ""])
    for case_name in summary["cases_in_order"]:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_baseline_replace"]))
    lines.extend(
        [
            "## Verdict",
            "",
            f"- best schedule case by 16-slot aggregate: `{summary['answers']['best_case_by_aggregate_16']}`",
            f"- best case clean beat current 1x60: `{summary['answers']['best_case_clean_beat_current_1x60']}`",
            f"- best case clean beat baseline replace: `{summary['answers']['best_case_clean_beat_baseline_replace']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    assert_exists(
        [
            CURRENT_70A_CKPT,
            CURRENT_70A_GROUP,
            CURRENT_REPLACE_SUMMARY,
            BASELINE_REPLACE_GROUP,
            LOWLR_WINNER_CONFIG,
            ENCODER_BUNDLE,
        ]
    )

    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    warmstart_report = load_json(WARMSTART_REPORT)

    current_replace_summary = load_json(CURRENT_REPLACE_SUMMARY)
    current_replace_ref = current_replace_summary["cases"]["lr5e5"]
    current_replace_metrics = current_replace_ref["metrics"]
    current_replace_aggregate = safe_float(current_replace_ref["aggregate_16"])
    baseline_replace_metrics = group_percentiles(BASELINE_REPLACE_GROUP)

    cases: Dict[str, Any] = {}
    cases_in_order: List[str] = []
    command_list: List[Dict[str, str]] = []

    for case_name, epochs, steps_per_epoch in PRIMARY_CASES:
        log(f"=== running schedule case {case_name} ({epochs}x{steps_per_epoch}) ===")
        payload = run_case(case_name, epochs, steps_per_epoch)
        payload["delta_vs_current_replace"] = compare_slots(payload["metrics"], current_replace_metrics)
        payload["delta_vs_baseline_replace"] = compare_slots(payload["metrics"], baseline_replace_metrics)
        cases[case_name] = payload
        cases_in_order.append(case_name)
        command_list.append(payload["commands"])

    should_run_conditional = any(
        safe_float(cases[name]["aggregate_16"]) < current_replace_aggregate - 1e-9
        for name in cases_in_order
    )

    if should_run_conditional:
        case_name, epochs, steps_per_epoch = CONDITIONAL_CASE
        log(f"=== running conditional schedule case {case_name} ({epochs}x{steps_per_epoch}) ===")
        payload = run_case(case_name, epochs, steps_per_epoch)
        payload["delta_vs_current_replace"] = compare_slots(payload["metrics"], current_replace_metrics)
        payload["delta_vs_baseline_replace"] = compare_slots(payload["metrics"], baseline_replace_metrics)
        cases[case_name] = payload
        cases_in_order.append(case_name)
        command_list.append(payload["commands"])

    best_case_name = min(cases_in_order, key=lambda name: safe_float(cases[name]["aggregate_16"]))
    best_case = cases[best_case_name]

    summary = {
        "run_date": RUN_DATE,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "lane_log": str(LOG_FILE),
        "policy": {
            "fixed_lr": 5e-5,
            "primary_cases": [{"name": name, "epochs": epochs, "steps_per_epoch": spe} for name, epochs, spe in PRIMARY_CASES],
            "conditional_case": {"name": CONDITIONAL_CASE[0], "epochs": CONDITIONAL_CASE[1], "steps_per_epoch": CONDITIONAL_CASE[2]},
            "conditional_case_executed": bool(should_run_conditional),
            "conditional_trigger": "any primary case aggregate_16 < current replace 1x60 aggregate_16",
        },
        "references": {
            "warmstart": warmstart_report,
            "current_70a": {
                "ckpt": str(CURRENT_70A_CKPT),
                "group_summary": str(CURRENT_70A_GROUP),
                "metrics": group_percentiles(CURRENT_70A_GROUP),
            },
            "current_replace_1x60": {
                "ckpt": str(current_replace_ref["ckpt"]),
                "log": str(current_replace_ref["log"]),
                "group_summary": str(current_replace_ref["group_summary"]),
                "metrics": current_replace_metrics,
                "aggregate_16": current_replace_aggregate,
            },
            "baseline_replace": {
                "group_summary": str(BASELINE_REPLACE_GROUP),
                "metrics": baseline_replace_metrics,
            },
        },
        "cases": cases,
        "cases_in_order": cases_in_order,
        "commands": command_list,
        "answers": {
            "best_case_by_aggregate_16": best_case_name,
            "best_case_clean_beat_current_1x60": bool(best_case["delta_vs_current_replace"]["all_leq"]),
            "best_case_clean_beat_baseline_replace": bool(best_case["delta_vs_baseline_replace"]["all_leq"]),
            "best_case_worst_gap_vs_baseline_slot": best_case["delta_vs_baseline_replace"]["worst_positive_delta_slot"],
            "best_case_worst_gap_vs_baseline_value": best_case["delta_vs_baseline_replace"]["worst_positive_delta"],
        },
    }

    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "lane_log": str(LOG_FILE),
            "best_case": best_case_name,
            "conditional_case_executed": bool(should_run_conditional),
            "warmstart_ckpt": str(WARMSTART_CKPT),
        },
    )
    log(f"summary: {SUMMARY_JSON}")
    log(f"markdown: {SUMMARY_MD}")
    log(f"status: {STATUS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
