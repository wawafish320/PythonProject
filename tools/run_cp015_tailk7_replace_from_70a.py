#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
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
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
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

try:
    from train import posttrain
except Exception:
    posttrain = None


RUN_DATE = "20260402"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_from_70a_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_from_70a_{RUN_DATE}"
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
CURRENT_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.json"
)
CURRENT_70A_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source"
    / "Walk_F_freerun_cycles.json"
)
CURRENT_70A_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source_group_summary.json"
)

BASELINE_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
BASELINE_REPLACE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "configs"
    / "posttrain_70b_replace_lowdrift_fromfresh_20260317.json"
)
BASELINE_REPLACE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)
BASELINE_REPLACE_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift_group_summary.json"
)

REPLACE_SEMANTICS_BASE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_70alr3e4_lowdrift_20260328"
    / "configs"
    / "posttrain_70b_replace_lowdrift_from_ep014center_70alr3e4_20260328.json"
)
LOWLR_WINNER_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_replace_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
)
LOWLR_WINNER_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_replace_lowlr_sweep_20260328"
    / "eval_model_source"
    / "lr5e5_group_summary.json"
)

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_{RUN_DATE}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"

GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "arm", "else")
PERCENTILES: Tuple[str, ...] = ("mean", "p50", "p90", "p95")
CASES: Tuple[Tuple[str, float], ...] = (
    ("lr5e5", 5e-5),
    ("lr1e4", 1e-4),
    ("lr2e4", 2e-4),
)


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


def flatten_slots(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for group_name in GROUPS:
        for pct in PERCENTILES:
            flat[f"{group_name}.{pct}"] = safe_float(metrics.get(group_name, {}).get(pct))
    return flat


def sum_slots(metrics: Mapping[str, Mapping[str, Any]]) -> float:
    total = 0.0
    count = 0
    for value in flatten_slots(metrics).values():
        if math.isfinite(value):
            total += value
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
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
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


def legacy_phase_flag_report(config_json: Path) -> Dict[str, Any]:
    payload = load_json(config_json)
    report = {
        "config_json": str(config_json),
        "raw_has_direct_pose_use_phase_z": bool("direct_pose_use_phase_z" in payload),
        "raw_has_direct_pose_phase_z_mode": bool("direct_pose_phase_z_mode" in payload),
    }
    if posttrain is None:
        report["parsed_check_available"] = False
        return report
    cfg = posttrain._cfg_from_payload(payload)
    report.update(
        {
            "parsed_check_available": True,
            "parsed_has_direct_pose_use_phase_z": bool(hasattr(cfg, "direct_pose_use_phase_z")),
            "parsed_has_direct_pose_phase_z_mode": bool(hasattr(cfg, "direct_pose_phase_z_mode")),
        }
    )
    return report


def replay_case(case_name: str, lr: float) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_from_cp015_tailk7_70a_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_from_cp015_tailk7_70a_{RUN_DATE}.json"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"

    make_generated_config(
        LOWLR_WINNER_CONFIG,
        cfg_json,
        {
            "ckpt_in": str(WARMSTART_CKPT),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "epochs": 1,
            "steps_per_epoch": 60,
            "lr": float(lr),
            "weight_decay": 0.0,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            # These keys are legacy and ignored by the current parser, but keeping
            # them aligned with the replace path makes the generated config easier
            # to inspect against the historical docs/pipeline naming.
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
        },
    )

    commands = build_command_list(case_name, cfg_json, out_dir, run_name, out_dir / f"ckpt_last_{run_name}.pth", eval_dir, group_json)

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
        "lr": float(lr),
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "log": str(out_dir / f"posttrain_log_{run_name}.json"),
        "eval": str(eval_json),
        "group_summary": str(group_json),
        "metrics": metrics,
        "aggregate_16": sum_slots(metrics),
        "commands": commands,
        "legacy_phase_flag_report": legacy_phase_flag_report(cfg_json),
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
    delta = delta_payload["delta"]
    for group_name in GROUPS:
        yield (
            f"| {group_name} | {fmt(delta[group_name]['mean'])} | {fmt(delta[group_name]['p50'])} | "
            f"{fmt(delta[group_name]['p90'])} | {fmt(delta[group_name]['p95'])} |"
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
        "# cp015 tailk7 replace from current 70a",
        "",
        "## Semantics facts",
        "",
        f"- warmstart source: `{refs['warmstart']['source_ckpt']}`",
        f"- warmstart output: `{refs['warmstart']['output_ckpt']}`",
        f"- warmstart is copy-only: `{refs['warmstart']['copied_without_phase_z_direct_adaptation']}`",
        f"- semantic reference config: `{summary['semantics']['replace_semantics_base_config']}`",
        f"- low-LR winner reference config: `{summary['semantics']['lowlr_winner_config']}`",
        "",
        "## Stage replace exit table",
        "",
        "| lane | all_ex_root mean/p50/p90/p95 | leg mean/p50/p90/p95 | arm mean/p50/p90/p95 | else mean/p50/p90/p95 |",
        "|---|---|---|---|---|",
        row("current_70a", refs["current_70a"]["metrics"]),
    ]
    for case_name in CASES:
        case_payload = summary["cases"][case_name[0]]
        lines.append(row(case_name[0], case_payload["metrics"]))
    lines.append(row("baseline_replace", refs["baseline_replace"]["metrics"]))
    lines.extend(
        [
            "",
            "## Case deltas vs current 70a",
            "",
        ]
    )
    for case_name, _ in CASES:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_current_70a"]))
    lines.extend(
        [
            "## Case deltas vs baseline replace",
            "",
        ]
    )
    for case_name, _ in CASES:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_baseline_replace"]))
    lines.extend(
        [
            "## Verdict",
            "",
            f"- best case by 16-slot aggregate: `{summary['answers']['best_case_by_aggregate_16']}`",
            f"- best case clean improve vs current 70a: `{summary['answers']['best_case_clean_improve_vs_current_70a']}`",
            f"- best case clean beat vs baseline replace: `{summary['answers']['best_case_clean_beat_vs_baseline_replace']}`",
            f"- promote verdict: `{summary['answers']['promote_verdict']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    assert_exists(
        [
            CURRENT_70A_CKPT,
            CURRENT_70A_LOG,
            CURRENT_70A_EVAL,
            CURRENT_70A_GROUP,
            BASELINE_REPLACE_CKPT,
            BASELINE_REPLACE_CONFIG,
            BASELINE_REPLACE_EVAL,
            BASELINE_REPLACE_GROUP,
            REPLACE_SEMANTICS_BASE_CONFIG,
            LOWLR_WINNER_CONFIG,
            LOWLR_WINNER_GROUP,
        ]
    )

    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    warmstart_report = load_json(WARMSTART_REPORT)

    current_70a_metrics = group_percentiles(CURRENT_70A_GROUP)
    baseline_replace_metrics = group_percentiles(BASELINE_REPLACE_GROUP)
    historical_lowlr_metrics = group_percentiles(LOWLR_WINNER_GROUP)

    cases: Dict[str, Any] = {}
    command_list = []
    for case_name, lr in CASES:
        log(f"=== running replace case {case_name} (lr={lr}) ===")
        case_payload = replay_case(case_name, lr)
        case_payload["delta_vs_current_70a"] = compare_slots(case_payload["metrics"], current_70a_metrics)
        case_payload["delta_vs_baseline_replace"] = compare_slots(case_payload["metrics"], baseline_replace_metrics)
        cases[case_name] = case_payload
        command_list.append(case_payload["commands"])

    best_case_name = min(cases.items(), key=lambda kv: safe_float(kv[1]["aggregate_16"]))[0]
    best_case = cases[best_case_name]
    best_vs_current = best_case["delta_vs_current_70a"]
    best_vs_baseline = best_case["delta_vs_baseline_replace"]

    answers = {
        "best_case_by_aggregate_16": best_case_name,
        "best_case_clean_improve_vs_current_70a": bool(best_vs_current["all_leq"]),
        "best_case_clean_beat_vs_baseline_replace": bool(best_vs_baseline["all_leq"]),
        "promote_verdict": "promote" if bool(best_vs_baseline["all_leq"]) else "no promote",
        "replacement_message": (
            "replace promoted over incumbent"
            if bool(best_vs_baseline["all_leq"])
            else "replace did not replace incumbent"
        ),
        "best_case_worst_gap_vs_baseline_slot": best_vs_baseline["worst_positive_delta_slot"],
        "best_case_worst_gap_vs_baseline_value": best_vs_baseline["worst_positive_delta"],
    }

    summary = {
        "run_date": RUN_DATE,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "lane_log": str(LOG_FILE),
        "semantics": {
            "replace_semantics_base_config": str(REPLACE_SEMANTICS_BASE_CONFIG),
            "lowlr_winner_config": str(LOWLR_WINNER_CONFIG),
            "baseline_replace_config": str(BASELINE_REPLACE_CONFIG),
            "semantic_reference_phase_flags": {
                "replace_semantics_base_config": legacy_phase_flag_report(REPLACE_SEMANTICS_BASE_CONFIG),
                "lowlr_winner_config": legacy_phase_flag_report(LOWLR_WINNER_CONFIG),
                "baseline_replace_config": legacy_phase_flag_report(BASELINE_REPLACE_CONFIG),
            },
        },
        "references": {
            "warmstart": warmstart_report,
            "current_70a": {
                "ckpt": str(CURRENT_70A_CKPT),
                "log": str(CURRENT_70A_LOG),
                "eval": str(CURRENT_70A_EVAL),
                "group_summary": str(CURRENT_70A_GROUP),
                "metrics": current_70a_metrics,
            },
            "baseline_replace": {
                "ckpt": str(BASELINE_REPLACE_CKPT),
                "config": str(BASELINE_REPLACE_CONFIG),
                "eval": str(BASELINE_REPLACE_EVAL),
                "group_summary": str(BASELINE_REPLACE_GROUP),
                "metrics": baseline_replace_metrics,
            },
            "historical_lowlr_winner": {
                "config": str(LOWLR_WINNER_CONFIG),
                "group_summary": str(LOWLR_WINNER_GROUP),
                "metrics": historical_lowlr_metrics,
            },
        },
        "cases": cases,
        "commands": command_list,
        "answers": answers,
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
            "promote_verdict": answers["promote_verdict"],
            "replacement_message": answers["replacement_message"],
            "warmstart_ckpt": str(WARMSTART_CKPT),
        },
    )

    log(f"summary: {SUMMARY_JSON}")
    log(f"markdown: {SUMMARY_MD}")
    log(f"status: {STATUS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
