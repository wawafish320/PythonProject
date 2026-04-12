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
        run_cmd,
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
        run_cmd,
        safe_float,
        write_json,
    )


RUN_TAG = "20260402_armelse_fixgroupmask"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_loss_weight_ablation_{RUN_TAG}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_loss_weight_ablation_{RUN_TAG}"
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
CURRENT_SCHEDULE_SUMMARY = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_schedule_ablation_20260402"
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

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_{RUN_TAG}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"
CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"

GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "arm", "else")
PERCENTILES: Tuple[str, ...] = ("mean", "p50", "p90", "p95")
PRIMARY_CASES: Tuple[Tuple[str, float, float], ...] = (
    ("e2x60_armelse_on_eq", 1.0, 1.0),
    ("e2x60_armelse_else125", 1.0, 1.25),
)
CONDITIONAL_CASE: Tuple[str, float, float] = ("e2x60_armelse_else150", 1.0, 1.5)


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


def tail_stats(log_path: Path) -> Dict[str, Any]:
    payload = load_json(log_path)
    rows = payload.get("log", []) if isinstance(payload, dict) else []
    last = rows[-1] if isinstance(rows, list) and rows else {}
    keys = (
        "dir_geo",
        "dir_leg_base",
        "dir_nonleg_base",
        "dir_nonleg_effective",
        "dir_nonleg_plain",
        "dir_arm_base",
        "dir_else_base",
        "direct_pose_arm_else_balance_active",
        "direct_pose_loss_arm_weight",
        "direct_pose_loss_else_weight",
        "dir_group_norm_used",
        "dir_group_norm_leg",
        "dir_group_norm_nonleg",
        "leg_over_nonleg",
        "leg_over_nonleg_effective",
        "arm_over_else",
    )
    return {key: last.get(key) for key in keys}


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


def run_case(case_name: str, arm_weight: float, else_weight: float) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_lr5e5_from_cp015_tailk7_70a_{RUN_TAG}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_lr5e5_from_cp015_tailk7_70a_{RUN_TAG}.json"
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
            "epochs": 2,
            "steps_per_epoch": 60,
            "weight_decay": 0.0,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "direct_pose_loss_arm_else_balance_enable": True,
            "direct_pose_loss_arm_weight": float(arm_weight),
            "direct_pose_loss_else_weight": float(else_weight),
        },
    )

    ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    if not ckpt.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
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
                "1.0",
                "--encoder_bundle",
                str(ENCODER_BUNDLE),
                "--posttrain_contacts_pretrain_affine_stats",
                str(ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"),
            ],
            log_file=LOG_FILE,
        )
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
    if not eval_json.is_file():
        run_cmd(
            [
                str(CPU_EXEC),
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                "validate/teacher_batches/Walk_F_teacher.json",
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
    if not group_json.is_file():
        ensure_group_summary(eval_json, group_json, log_file=LOG_FILE)
    metrics = group_percentiles(group_json)
    log_json = out_dir / f"posttrain_log_{run_name}.json"
    return {
        "case_name": case_name,
        "arm_weight": float(arm_weight),
        "else_weight": float(else_weight),
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "log": str(log_json),
        "eval": str(eval_json),
        "group_summary": str(group_json),
        "metrics": metrics,
        "aggregate_16": aggregate_16(metrics),
        "log_tail": tail_stats(log_json),
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
    audit = summary["code_audit"]
    lines = [
        "# cp015 tailk7 replace loss-weight ablation",
        "",
        "## Code Audit",
        "",
        f"- pre-patch config parser read arm/else balance fields: `{audit['pre_patch']['config_parser_reads_arm_else_balance']}`",
        f"- pre-patch direct rollout objective implemented arm/else balance: `{audit['pre_patch']['direct_rollout_implements_arm_else_balance']}`",
        f"- current e2x60 config arm/else keys: `enable={audit['pre_patch']['current_e2x60_config'].get('direct_pose_loss_arm_else_balance_enable')}` / "
        f"`arm_w={audit['pre_patch']['current_e2x60_config'].get('direct_pose_loss_arm_weight')}` / "
        f"`else_w={audit['pre_patch']['current_e2x60_config'].get('direct_pose_loss_else_weight')}`",
        "",
        "## Policy",
        "",
        "- fixed donor: current 70a",
        "- fixed warmstart: copy-only replace_zerophase",
        "- fixed replace budget: e2x60",
        "- fixed lr/wd: 5e-5 / 0.0",
        "- fixed encoder bundle: current bundle",
        f"- conditional else150 executed: `{summary['policy']['conditional_case_executed']}`",
        "",
        "## Stage replace exit table",
        "",
        "| lane | all_ex_root mean/p50/p90/p95 | leg mean/p50/p90/p95 | arm mean/p50/p90/p95 | else mean/p50/p90/p95 |",
        "|---|---|---|---|---|",
        row("current_70a", refs["current_70a"]["metrics"]),
        row("current_replace_1x60", refs["current_replace_1x60"]["metrics"]),
        row("current_replace_e2x60", refs["current_replace_e2x60"]["metrics"]),
    ]
    for case_name in summary["cases_in_order"]:
        lines.append(row(case_name, summary["cases"][case_name]["metrics"]))
    lines.append(row("baseline_replace", refs["baseline_replace"]["metrics"]))
    lines.extend(["", "## Delta vs current 70a", ""])
    for case_name in summary["cases_in_order"]:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_current_70a"]))
    lines.extend(["## Delta vs current replace e2x60", ""])
    for case_name in summary["cases_in_order"]:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_current_replace_e2x60"]))
    lines.extend(["## Delta vs baseline replace", ""])
    for case_name in summary["cases_in_order"]:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_baseline_replace"]))
    lines.extend(["## Tail Stats", ""])
    for case_name in summary["cases_in_order"]:
        tail = summary["cases"][case_name]["log_tail"]
        lines.extend(
            [
                f"### {case_name}",
                "",
                f"- arm/else balance active: `{tail.get('direct_pose_arm_else_balance_active')}`",
                f"- dir_leg / dir_nonleg_plain / dir_nonleg_effective: `{tail.get('dir_leg_base')}` / `{tail.get('dir_nonleg_plain')}` / `{tail.get('dir_nonleg_effective')}`",
                f"- dir_arm / dir_else / arm_over_else: `{tail.get('dir_arm_base')}` / `{tail.get('dir_else_base')}` / `{tail.get('arm_over_else')}`",
                f"- group norm used leg/nonleg: `{tail.get('dir_group_norm_used')}` / `{tail.get('dir_group_norm_leg')}` / `{tail.get('dir_group_norm_nonleg')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Verdict",
            "",
            f"- best case by 16-slot aggregate: `{summary['answers']['best_case_by_aggregate_16']}`",
            f"- best case clean improve vs current 70a: `{summary['answers']['best_case_clean_beat_current_70a']}`",
            f"- best case clean improve vs current replace e2x60: `{summary['answers']['best_case_clean_beat_current_replace_e2x60']}`",
            f"- best case clean beat baseline replace: `{summary['answers']['best_case_clean_beat_baseline_replace']}`",
            f"- promote verdict: `{summary['answers']['promote_verdict']}`",
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
            CURRENT_SCHEDULE_SUMMARY,
            BASELINE_REPLACE_GROUP,
            LOWLR_WINNER_CONFIG,
            ENCODER_BUNDLE,
        ]
    )

    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    warmstart_report = load_json(WARMSTART_REPORT)

    replace_summary = load_json(CURRENT_REPLACE_SUMMARY)
    schedule_summary = load_json(CURRENT_SCHEDULE_SUMMARY)
    current_replace_1x60 = replace_summary["cases"]["lr5e5"]
    current_replace_e2x60 = schedule_summary["cases"]["e2x60"]
    current_replace_e2x60_metrics = current_replace_e2x60["metrics"]
    current_replace_e2x60_aggregate = safe_float(current_replace_e2x60["aggregate_16"])
    baseline_replace_metrics = group_percentiles(BASELINE_REPLACE_GROUP)

    current_e2x60_config = load_json(Path(current_replace_e2x60["config"]))
    current_e2x60_log_tail = tail_stats(Path(current_replace_e2x60["log"]))

    cases: Dict[str, Any] = {}
    cases_in_order: List[str] = []
    command_list: List[Dict[str, str]] = []

    for case_name, arm_weight, else_weight in PRIMARY_CASES:
        log(f"=== running loss-weight case {case_name} (arm={arm_weight}, else={else_weight}) ===")
        payload = run_case(case_name, arm_weight, else_weight)
        payload["delta_vs_current_70a"] = compare_slots(payload["metrics"], group_percentiles(CURRENT_70A_GROUP))
        payload["delta_vs_current_replace_e2x60"] = compare_slots(payload["metrics"], current_replace_e2x60_metrics)
        payload["delta_vs_baseline_replace"] = compare_slots(payload["metrics"], baseline_replace_metrics)
        cases[case_name] = payload
        cases_in_order.append(case_name)
        command_list.append(payload["commands"])

    should_run_conditional = any(
        safe_float(cases[name]["aggregate_16"]) < current_replace_e2x60_aggregate - 1e-9
        for name in cases_in_order
    )

    if should_run_conditional:
        case_name, arm_weight, else_weight = CONDITIONAL_CASE
        log(f"=== running conditional loss-weight case {case_name} (arm={arm_weight}, else={else_weight}) ===")
        payload = run_case(case_name, arm_weight, else_weight)
        payload["delta_vs_current_70a"] = compare_slots(payload["metrics"], group_percentiles(CURRENT_70A_GROUP))
        payload["delta_vs_current_replace_e2x60"] = compare_slots(payload["metrics"], current_replace_e2x60_metrics)
        payload["delta_vs_baseline_replace"] = compare_slots(payload["metrics"], baseline_replace_metrics)
        cases[case_name] = payload
        cases_in_order.append(case_name)
        command_list.append(payload["commands"])

    best_case_name = min(cases_in_order, key=lambda name: safe_float(cases[name]["aggregate_16"]))
    best_case = cases[best_case_name]

    summary = {
        "run_tag": RUN_TAG,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "lane_log": str(LOG_FILE),
        "policy": {
            "fixed_schedule": {"epochs": 2, "steps_per_epoch": 60},
            "fixed_lr": 5e-5,
            "fixed_weight_decay": 0.0,
            "primary_cases": [{"name": name, "arm_weight": arm_w, "else_weight": else_w} for name, arm_w, else_w in PRIMARY_CASES],
            "conditional_case": {"name": CONDITIONAL_CASE[0], "arm_weight": CONDITIONAL_CASE[1], "else_weight": CONDITIONAL_CASE[2]},
            "conditional_case_executed": bool(should_run_conditional),
            "conditional_trigger": "any primary case aggregate_16 < current replace e2x60 aggregate_16",
        },
        "code_audit": {
            "pre_patch": {
                "config_parser_reads_arm_else_balance": False,
                "direct_rollout_implements_arm_else_balance": False,
                "current_e2x60_config": {
                    "direct_pose_loss_leg_split": current_e2x60_config.get("direct_pose_loss_leg_split"),
                    "direct_pose_arm_split_enable": current_e2x60_config.get("direct_pose_arm_split_enable"),
                    "direct_pose_loss_group_norm_enable": current_e2x60_config.get("direct_pose_loss_group_norm_enable"),
                    "direct_pose_loss_arm_else_balance_enable": current_e2x60_config.get("direct_pose_loss_arm_else_balance_enable"),
                    "direct_pose_loss_arm_weight": current_e2x60_config.get("direct_pose_loss_arm_weight"),
                    "direct_pose_loss_else_weight": current_e2x60_config.get("direct_pose_loss_else_weight"),
                },
                "current_e2x60_log_tail": current_e2x60_log_tail,
            },
            "post_patch": {
                "config_parser_reads_arm_else_balance": True,
                "direct_rollout_implements_arm_else_balance": True,
            },
        },
        "references": {
            "warmstart": warmstart_report,
            "current_70a": {
                "ckpt": str(CURRENT_70A_CKPT),
                "group_summary": str(CURRENT_70A_GROUP),
                "metrics": group_percentiles(CURRENT_70A_GROUP),
            },
            "current_replace_1x60": {
                "ckpt": str(current_replace_1x60["ckpt"]),
                "log": str(current_replace_1x60["log"]),
                "group_summary": str(current_replace_1x60["group_summary"]),
                "metrics": current_replace_1x60["metrics"],
                "aggregate_16": safe_float(current_replace_1x60["aggregate_16"]),
            },
            "current_replace_e2x60": {
                "ckpt": str(current_replace_e2x60["ckpt"]),
                "log": str(current_replace_e2x60["log"]),
                "group_summary": str(current_replace_e2x60["group_summary"]),
                "metrics": current_replace_e2x60_metrics,
                "aggregate_16": current_replace_e2x60_aggregate,
                "log_tail": current_e2x60_log_tail,
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
            "best_case_clean_beat_current_70a": bool(best_case["delta_vs_current_70a"]["all_leq"]),
            "best_case_clean_beat_current_replace_e2x60": bool(best_case["delta_vs_current_replace_e2x60"]["all_leq"]),
            "best_case_clean_beat_baseline_replace": bool(best_case["delta_vs_baseline_replace"]["all_leq"]),
            "best_case_worst_gap_vs_baseline_slot": best_case["delta_vs_baseline_replace"]["worst_positive_delta_slot"],
            "best_case_worst_gap_vs_baseline_value": best_case["delta_vs_baseline_replace"]["worst_positive_delta"],
            "promote_verdict": "promote" if bool(best_case["delta_vs_baseline_replace"]["all_leq"]) else "no promote",
            "replace_replaces_incumbent": bool(best_case["delta_vs_baseline_replace"]["all_leq"]),
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
