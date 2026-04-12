#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        ensure_group_summary,
        fmt,
        load_json,
        make_generated_config,
        run_cmd,
        run_eval,
        safe_float,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        ensure_group_summary,
        fmt,
        load_json,
        make_generated_config,
        run_cmd,
        run_eval,
        safe_float,
        write_json,
    )


RUN_DATE = "20260402"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_encoder_ablation_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_encoder_ablation_{RUN_DATE}"
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
CURRENT_70A_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source_group_summary.json"
)

CURRENT_REPLACE_SWEEP_SUMMARY = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_from_70a_20260402"
    / "summary.json"
)
BASELINE_REPLACE_LOG = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "posttrain_log_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.json"
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

CURRENT_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
BUNDLE_20260317 = ROOT / "models" / "motion_encoder_equiv_20260317.pt.best.pt"
BUNDLE_STAGEA = ROOT / "models" / "motion_encoder_equiv_stageA.pt"

CASES: Tuple[Tuple[str, Path], ...] = (
    ("enc20260317", BUNDLE_20260317),
    ("encStageA", BUNDLE_STAGEA),
)
GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "arm", "else")
PERCENTILES: Tuple[str, ...] = ("mean", "p50", "p90", "p95")


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


def bundle_meta(path: Path) -> Dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": hasher.hexdigest(),
    }


def run_posttrain_stage_with_encoder(
    *,
    config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    encoder_bundle: Path,
    log_file: Path,
) -> Path:
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_out.is_file():
        return ckpt_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-m",
            "train.posttrain",
            "--config",
            str(config),
            "--ckpt_in",
            str(ckpt_in),
            "--out_dir",
            str(out_dir),
            "--run_name",
            run_name,
            "--posttrain_contacts_source",
            "pretrain_contact",
            "--posttrain_contacts_pretrain_clamp",
            PRETRAIN_CLAMP,
            "--encoder_bundle",
            str(encoder_bundle),
            "--posttrain_contacts_pretrain_affine_stats",
            str(AFFINE_STATS),
        ],
        log_file=log_file,
    )
    return ckpt_out


def build_command_list(case_name: str, cfg_json: Path, out_dir: Path, run_name: str, encoder_bundle: Path, ckpt: Path, eval_dir: Path, group_json: Path) -> Dict[str, str]:
    train_cmd = (
        "PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain "
        f"--config {cfg_json} "
        f"--ckpt_in {WARMSTART_CKPT} "
        f"--out_dir {out_dir} "
        f"--run_name {run_name} "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {encoder_bundle} "
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


def replay_case(case_name: str, encoder_bundle: Path) -> Dict[str, Any]:
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
            "epochs": 1,
            "steps_per_epoch": 60,
            "lr": 5e-5,
            "weight_decay": 0.0,
            "encoder_bundle": str(encoder_bundle),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )

    ckpt = run_posttrain_stage_with_encoder(
        config=cfg_json,
        ckpt_in=WARMSTART_CKPT,
        out_dir=out_dir,
        run_name=run_name,
        encoder_bundle=encoder_bundle,
        log_file=LOG_FILE,
    )
    eval_json = run_eval(model_ckpt=ckpt, out_dir=eval_dir, contacts_source="model", log_file=LOG_FILE)
    ensure_group_summary(eval_json, group_json, log_file=LOG_FILE)
    metrics = group_percentiles(group_json)
    return {
        "case_name": case_name,
        "encoder_bundle": bundle_meta(encoder_bundle),
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "log": str(out_dir / f"posttrain_log_{run_name}.json"),
        "eval": str(eval_json),
        "group_summary": str(group_json),
        "metrics": metrics,
        "commands": build_command_list(case_name, cfg_json, out_dir, run_name, encoder_bundle, ckpt, eval_dir, group_json),
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
        "# cp015 tailk7 replace encoder ablation",
        "",
        "## Bundle facts",
        "",
        f"- warmstart is copy-only: `{refs['warmstart']['copied_without_phase_z_direct_adaptation']}`",
        f"- donor 70a encoder_bundle: `{refs['current_70a']['encoder_bundle']}`",
        f"- current replace reference encoder_bundle: `{refs['current_replace_current_bundle']['encoder_bundle']}`",
        f"- baseline replace encoder_bundle: `{refs['baseline_replace']['encoder_bundle']}`",
        "",
        "## Stage replace exit table",
        "",
        "| lane | all_ex_root mean/p50/p90/p95 | leg mean/p50/p90/p95 | arm mean/p50/p90/p95 | else mean/p50/p90/p95 |",
        "|---|---|---|---|---|",
        row("current_replace_current_bundle", refs["current_replace_current_bundle"]["metrics"]),
    ]
    for case_name, _ in CASES:
        lines.append(row(case_name, summary["cases"][case_name]["metrics"]))
    lines.append(row("baseline_replace", refs["baseline_replace"]["metrics"]))
    lines.extend(
        [
            "",
            "## Delta vs current replace current bundle",
            "",
        ]
    )
    for case_name, _ in CASES:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_current_replace"]))
    lines.extend(
        [
            "## Delta vs baseline replace",
            "",
        ]
    )
    for case_name, _ in CASES:
        lines.extend(delta_lines(case_name, summary["cases"][case_name]["delta_vs_baseline_replace"]))
    lines.extend(
        [
            "## Verdict",
            "",
            f"- best encoder case by 16-slot aggregate: `{summary['answers']['best_case_by_aggregate_16']}`",
            f"- beats current replace current bundle cleanly: `{summary['answers']['best_case_clean_beat_current_replace']}`",
            f"- beats baseline replace cleanly: `{summary['answers']['best_case_clean_beat_baseline_replace']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    assert_exists(
        [
            CURRENT_70A_CKPT,
            CURRENT_70A_LOG,
            CURRENT_70A_GROUP,
            CURRENT_REPLACE_SWEEP_SUMMARY,
            BASELINE_REPLACE_LOG,
            BASELINE_REPLACE_GROUP,
            LOWLR_WINNER_CONFIG,
            CURRENT_BUNDLE,
            BUNDLE_20260317,
            BUNDLE_STAGEA,
            AFFINE_STATS,
        ]
    )

    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)
    warmstart_report = load_json(WARMSTART_REPORT)

    current_70a_log = load_json(CURRENT_70A_LOG)
    current_replace_summary = load_json(CURRENT_REPLACE_SWEEP_SUMMARY)
    current_replace_ref = current_replace_summary["cases"]["lr5e5"]
    baseline_replace_log = load_json(BASELINE_REPLACE_LOG)

    current_replace_metrics = group_percentiles(Path(str(current_replace_ref["group_summary"])))
    baseline_replace_metrics = group_percentiles(BASELINE_REPLACE_GROUP)

    cases: Dict[str, Any] = {}
    command_list = []
    for case_name, encoder_bundle in CASES:
        log(f"=== running encoder ablation {case_name} ===")
        payload = replay_case(case_name, encoder_bundle)
        payload["delta_vs_current_replace"] = compare_slots(payload["metrics"], current_replace_metrics)
        payload["delta_vs_baseline_replace"] = compare_slots(payload["metrics"], baseline_replace_metrics)
        cases[case_name] = payload
        command_list.append(payload["commands"])

    best_case_name = min(
        cases,
        key=lambda name: sum(
            safe_float(cases[name]["metrics"][group_name][pct])
            for group_name in GROUPS
            for pct in PERCENTILES
        ),
    )
    best_case = cases[best_case_name]

    answers = {
        "best_case_by_aggregate_16": best_case_name,
        "best_case_clean_beat_current_replace": bool(best_case["delta_vs_current_replace"]["all_leq"]),
        "best_case_clean_beat_baseline_replace": bool(best_case["delta_vs_baseline_replace"]["all_leq"]),
        "best_case_worst_gap_vs_baseline_slot": best_case["delta_vs_baseline_replace"]["worst_positive_delta_slot"],
        "best_case_worst_gap_vs_baseline_value": best_case["delta_vs_baseline_replace"]["worst_positive_delta"],
    }

    summary = {
        "run_date": RUN_DATE,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "lane_log": str(LOG_FILE),
        "references": {
            "warmstart": warmstart_report,
            "current_70a": {
                "ckpt": str(CURRENT_70A_CKPT),
                "group_summary": str(CURRENT_70A_GROUP),
                "metrics": group_percentiles(CURRENT_70A_GROUP),
                "encoder_bundle": str(current_70a_log.get("config", {}).get("encoder_bundle")),
            },
            "current_replace_current_bundle": {
                "ckpt": str(current_replace_ref["ckpt"]),
                "log": str(current_replace_ref["log"]),
                "group_summary": str(current_replace_ref["group_summary"]),
                "metrics": current_replace_metrics,
                "encoder_bundle": str(load_json(Path(str(current_replace_ref["log"]))).get("config", {}).get("encoder_bundle")),
            },
            "baseline_replace": {
                "log": str(BASELINE_REPLACE_LOG),
                "group_summary": str(BASELINE_REPLACE_GROUP),
                "metrics": baseline_replace_metrics,
                "encoder_bundle": str(baseline_replace_log.get("config", {}).get("encoder_bundle")),
            },
            "bundle_meta": {
                "current_bundle": bundle_meta(CURRENT_BUNDLE),
                "bundle_20260317": bundle_meta(BUNDLE_20260317),
                "bundle_stageA": bundle_meta(BUNDLE_STAGEA),
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
            "warmstart_ckpt": str(WARMSTART_CKPT),
        },
    )
    log(f"summary: {SUMMARY_JSON}")
    log(f"markdown: {SUMMARY_MD}")
    log(f"status: {STATUS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
