#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70R,
        CONFIG_71,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
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
        CONFIG_70R,
        CONFIG_71,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        run_70r_promote,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260314"
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_newflow_chain_20260314" / "summary.json"
LOWDRIFT_REPLACE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_skip70b_lowdrift_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_oldd1_skip70b_lowdrift_to71_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_oldd1_skip70b_lowdrift_to71_{RUN_DATE}"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def delta_block(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def stage_snapshot(summary: Mapping[str, Any], stage_name: str) -> Dict[str, Any]:
    stages = summary.get("stage_progress_model_source", {})
    if not isinstance(stages, Mapping) or stage_name not in stages:
        raise RuntimeError(f"missing stage {stage_name} in {SOURCE_SUMMARY_JSON}")
    stage = stages[stage_name]
    if not isinstance(stage, dict):
        raise RuntimeError(f"invalid stage payload for {stage_name}")
    return dict(stage)


def build_summary(
    *,
    source_summary: Mapping[str, Any],
    replace_summary: Mapping[str, Any],
    candidate_70r: Mapping[str, Any],
    candidate_71: Mapping[str, Any],
    ckpt_70r: Path,
    ckpt_71: Path,
    cfg_70r: Path,
) -> Dict[str, Any]:
    keys_direct = ("all_ex_root", "leg", "nonleg", "arm", "else")
    keys_masked = ("DirectGeoLocalDeg", "BlendGeoLocalDeg", "GeoLocalDeg")
    hotspot_keys = ("foot_l_ball_l_SIC12_15", "calf_r_SIC2_4")

    current_70r = stage_snapshot(source_summary, "70R")
    current_71 = stage_snapshot(source_summary, "71")
    candidate_replace = replace_summary["candidate"]["eval"]

    def compare(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "masked_means_delta": delta_block(cur["masked_means"], ref["masked_means"], keys_masked),
            "direct_group_delta": delta_block(cur["direct_group_summary"], ref["direct_group_summary"], keys_direct),
            "overall_window_delta": delta_block(cur["window_summary"]["overall"], ref["window_summary"]["overall"], ("legs_main", "arms_main", "left_arm_main", "right_arm_main")),
            "hotspot_delta": delta_block(cur["window_summary"]["hotspots"], ref["window_summary"]["hotspots"], hotspot_keys),
        }

    return {
        "run_date": RUN_DATE,
        "source_summary": str(SOURCE_SUMMARY_JSON),
        "replace_summary": str(LOWDRIFT_REPLACE_SUMMARY_JSON),
        "policy": {
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "compare_contract": "model_source",
        },
        "control": {
            "current_70R": current_70r,
            "current_71": current_71,
            "candidate_replace": candidate_replace,
        },
        "candidate": {
            "ckpt_70R": str(ckpt_70r),
            "ckpt_71": str(ckpt_71),
            "config_70R": str(cfg_70r),
            "eval_70R": candidate_70r,
            "eval_71": candidate_71,
        },
        "comparisons": {
            "candidate_70R_vs_current_70R": compare(candidate_70r, current_70r),
            "candidate_71_vs_current_71": compare(candidate_71, current_71),
            "candidate_70R_vs_candidate_replace": compare(candidate_70r, candidate_replace),
            "candidate_71_vs_candidate_70R": compare(candidate_71, candidate_70r),
        },
        "answers": {
            "calf_recovers_at_70R_vs_replace": safe_float(diff(candidate_70r["window_summary"]["hotspots"]["calf_r_SIC2_4"], candidate_replace["window_summary"]["hotspots"]["calf_r_SIC2_4"])) < 0.0,
            "calf_recovers_at_71_vs_70R": safe_float(diff(candidate_71["window_summary"]["hotspots"]["calf_r_SIC2_4"], candidate_70r["window_summary"]["hotspots"]["calf_r_SIC2_4"])) < 0.0,
            "candidate_71_beats_current_71_leg": safe_float(diff(candidate_71["direct_group_summary"]["leg"], current_71["direct_group_summary"]["leg"])) < 0.0,
            "candidate_71_beats_current_71_calf_hotspot": safe_float(diff(candidate_71["window_summary"]["hotspots"]["calf_r_SIC2_4"], current_71["window_summary"]["hotspots"]["calf_r_SIC2_4"])) < 0.0,
        },
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    cur70r = summary["control"]["current_70R"]
    cur71 = summary["control"]["current_71"]
    repl = summary["control"]["candidate_replace"]
    cand70r = summary["candidate"]["eval_70R"]
    cand71 = summary["candidate"]["eval_71"]
    c70r_cur = summary["comparisons"]["candidate_70R_vs_current_70R"]
    c71_cur = summary["comparisons"]["candidate_71_vs_current_71"]
    c70r_rep = summary["comparisons"]["candidate_70R_vs_candidate_replace"]
    c71_70r = summary["comparisons"]["candidate_71_vs_candidate_70R"]
    ans = summary["answers"]

    lines = [
        "# old d1 lowdrift replace -> 70R -> 71",
        "",
        f"- source_summary: `{summary['source_summary']}`",
        f"- replace_summary: `{summary['replace_summary']}`",
        f"- candidate_70R_ckpt: `{summary['candidate']['ckpt_70R']}`",
        f"- candidate_71_ckpt: `{summary['candidate']['ckpt_71']}`",
        "",
        "## Direct-path metrics (model-source)",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else | calf_r@SIC2-4 | foot_l/ball_l@SIC12-15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| candidate_replace | {fmt(repl['masked_means']['DirectGeoLocalDeg'])} | {fmt(repl['direct_group_summary']['all_ex_root'])} | {fmt(repl['direct_group_summary']['leg'])} | {fmt(repl['direct_group_summary']['nonleg'])} | {fmt(repl['direct_group_summary']['arm'])} | {fmt(repl['direct_group_summary']['else'])} | {fmt(repl['window_summary']['hotspots']['calf_r_SIC2_4'])} | {fmt(repl['window_summary']['hotspots']['foot_l_ball_l_SIC12_15'])} |",
        f"| current_70R | {fmt(cur70r['masked_means']['DirectGeoLocalDeg'])} | {fmt(cur70r['direct_group_summary']['all_ex_root'])} | {fmt(cur70r['direct_group_summary']['leg'])} | {fmt(cur70r['direct_group_summary']['nonleg'])} | {fmt(cur70r['direct_group_summary']['arm'])} | {fmt(cur70r['direct_group_summary']['else'])} | {fmt(cur70r['window_summary']['hotspots']['calf_r_SIC2_4'])} | {fmt(cur70r['window_summary']['hotspots']['foot_l_ball_l_SIC12_15'])} |",
        f"| candidate_70R | {fmt(cand70r['masked_means']['DirectGeoLocalDeg'])} | {fmt(cand70r['direct_group_summary']['all_ex_root'])} | {fmt(cand70r['direct_group_summary']['leg'])} | {fmt(cand70r['direct_group_summary']['nonleg'])} | {fmt(cand70r['direct_group_summary']['arm'])} | {fmt(cand70r['direct_group_summary']['else'])} | {fmt(cand70r['window_summary']['hotspots']['calf_r_SIC2_4'])} | {fmt(cand70r['window_summary']['hotspots']['foot_l_ball_l_SIC12_15'])} |",
        f"| current_71 | {fmt(cur71['masked_means']['DirectGeoLocalDeg'])} | {fmt(cur71['direct_group_summary']['all_ex_root'])} | {fmt(cur71['direct_group_summary']['leg'])} | {fmt(cur71['direct_group_summary']['nonleg'])} | {fmt(cur71['direct_group_summary']['arm'])} | {fmt(cur71['direct_group_summary']['else'])} | {fmt(cur71['window_summary']['hotspots']['calf_r_SIC2_4'])} | {fmt(cur71['window_summary']['hotspots']['foot_l_ball_l_SIC12_15'])} |",
        f"| candidate_71 | {fmt(cand71['masked_means']['DirectGeoLocalDeg'])} | {fmt(cand71['direct_group_summary']['all_ex_root'])} | {fmt(cand71['direct_group_summary']['leg'])} | {fmt(cand71['direct_group_summary']['nonleg'])} | {fmt(cand71['direct_group_summary']['arm'])} | {fmt(cand71['direct_group_summary']['else'])} | {fmt(cand71['window_summary']['hotspots']['calf_r_SIC2_4'])} | {fmt(cand71['window_summary']['hotspots']['foot_l_ball_l_SIC12_15'])} |",
        "",
        "## Deltas",
        "",
        "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_calf_r@SIC2-4 | d_foot_l/ball_l@SIC12-15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| candidate_70R - current_70R | {fmt(c70r_cur['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(c70r_cur['direct_group_delta']['all_ex_root'])} | {fmt(c70r_cur['direct_group_delta']['leg'])} | {fmt(c70r_cur['direct_group_delta']['nonleg'])} | {fmt(c70r_cur['direct_group_delta']['arm'])} | {fmt(c70r_cur['hotspot_delta']['calf_r_SIC2_4'])} | {fmt(c70r_cur['hotspot_delta']['foot_l_ball_l_SIC12_15'])} |",
        f"| candidate_71 - current_71 | {fmt(c71_cur['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(c71_cur['direct_group_delta']['all_ex_root'])} | {fmt(c71_cur['direct_group_delta']['leg'])} | {fmt(c71_cur['direct_group_delta']['nonleg'])} | {fmt(c71_cur['direct_group_delta']['arm'])} | {fmt(c71_cur['hotspot_delta']['calf_r_SIC2_4'])} | {fmt(c71_cur['hotspot_delta']['foot_l_ball_l_SIC12_15'])} |",
        f"| candidate_70R - candidate_replace | {fmt(c70r_rep['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(c70r_rep['direct_group_delta']['all_ex_root'])} | {fmt(c70r_rep['direct_group_delta']['leg'])} | {fmt(c70r_rep['direct_group_delta']['nonleg'])} | {fmt(c70r_rep['direct_group_delta']['arm'])} | {fmt(c70r_rep['hotspot_delta']['calf_r_SIC2_4'])} | {fmt(c70r_rep['hotspot_delta']['foot_l_ball_l_SIC12_15'])} |",
        f"| candidate_71 - candidate_70R | {fmt(c71_70r['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(c71_70r['direct_group_delta']['all_ex_root'])} | {fmt(c71_70r['direct_group_delta']['leg'])} | {fmt(c71_70r['direct_group_delta']['nonleg'])} | {fmt(c71_70r['direct_group_delta']['arm'])} | {fmt(c71_70r['hotspot_delta']['calf_r_SIC2_4'])} | {fmt(c71_70r['hotspot_delta']['foot_l_ball_l_SIC12_15'])} |",
        "",
        "## Answers",
        "",
        f"- calf recovers at 70R vs replace: `{str(bool(ans['calf_recovers_at_70R_vs_replace'])).lower()}`",
        f"- calf recovers at 71 vs 70R: `{str(bool(ans['calf_recovers_at_71_vs_70R'])).lower()}`",
        f"- candidate 71 beats current 71 on leg: `{str(bool(ans['candidate_71_beats_current_71_leg'])).lower()}`",
        f"- candidate 71 beats current 71 on calf hotspot: `{str(bool(ans['candidate_71_beats_current_71_calf_hotspot'])).lower()}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [SOURCE_SUMMARY_JSON, LOWDRIFT_REPLACE_SUMMARY_JSON, CONFIG_70R, CONFIG_71, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    source_summary = load_json(SOURCE_SUMMARY_JSON)
    replace_summary = load_json(LOWDRIFT_REPLACE_SUMMARY_JSON)
    replace_ckpt = Path(str(replace_summary["candidate"]["ckpt"]))
    if not replace_ckpt.is_file():
        raise SystemExit(f"missing replace ckpt: {replace_ckpt}")

    lane_log = OUT_ROOT / "lane.log"
    cfg_70r = OUT_ROOT / "configs" / f"posttrain_70R_from_oldd1_lowdrift_replace_{RUN_DATE}.json"
    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    status_json = OUT_ROOT / "status.json"

    run_name_70r = f"WalkF_stage7_70R_from_oldd1_lowdrift_replace_{RUN_DATE}"
    run_name_71 = f"WalkF_stage7_71_from_oldd1_lowdrift_replace_{RUN_DATE}"

    log("=== candidate 70R ===")
    cfg_70r = make_generated_config(
        CONFIG_70R,
        cfg_70r,
        {
            "ckpt_in": str(replace_ckpt),
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
        log_file=lane_log,
    )

    log("=== candidate 71 ===")
    ckpt_71 = run_posttrain_stage(
        config=CONFIG_71,
        ckpt_in=ckpt_70r,
        out_dir=MODEL_ROOT / "71",
        run_name=run_name_71,
        log_file=lane_log,
    )

    log("=== eval candidate 70R ===")
    eval_70r_dir = OUT_ROOT / "eval_70R_model"
    eval_70r_group = OUT_ROOT / "eval_70R_model_group_summary.json"
    eval_70r_json = run_eval(
        model_ckpt=ckpt_70r,
        out_dir=eval_70r_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_70r_json, eval_70r_group, log_file=lane_log)
    cand70r = collect_eval(eval_70r_json, eval_70r_group)

    log("=== eval candidate 71 ===")
    eval_71_dir = OUT_ROOT / "eval_71_model"
    eval_71_group = OUT_ROOT / "eval_71_model_group_summary.json"
    eval_71_json = run_eval(
        model_ckpt=ckpt_71,
        out_dir=eval_71_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_71_json, eval_71_group, log_file=lane_log)
    cand71 = collect_eval(eval_71_json, eval_71_group)

    write_json(
        status_json,
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "replace_ckpt": str(replace_ckpt),
            "candidate_70R_ckpt": str(ckpt_70r),
            "candidate_71_ckpt": str(ckpt_71),
            "candidate_70R_eval": str(eval_70r_json),
            "candidate_71_eval": str(eval_71_json),
        },
    )

    summary = build_summary(
        source_summary=source_summary,
        replace_summary=replace_summary,
        candidate_70r=cand70r,
        candidate_71=cand71,
        ckpt_70r=ckpt_70r,
        ckpt_71=ckpt_71,
        cfg_70r=cfg_70r,
    )
    write_json(summary_json, summary)
    summary_md.write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
