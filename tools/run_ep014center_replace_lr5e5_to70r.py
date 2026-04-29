#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70R,
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
        safe_float,
        window_group_stats,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70R,
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
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260328"
REPLACE_SWEEP_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_replace_lr5e5_to70r_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_replace_lr5e5_to70r_{RUN_DATE}"
PROMOTED_REPLACE_CASE = "lr5e5"
SELECTED_METRICS = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def collect_eval(eval_json: Path, group_json: Path) -> Dict[str, Any]:
    return {
        "masked_means": masked_metric_means(eval_json),
        "direct_group_summary": group_metrics(group_json),
        "window_summary": window_group_stats(eval_json),
        "selected_metrics": {
            "DirectGeoLocalDeg": safe_float(masked_metric_means(eval_json).get("DirectGeoLocalDeg")),
            "all_ex_root": safe_float(group_metrics(group_json).get("all_ex_root")),
            "leg": safe_float(group_metrics(group_json).get("leg")),
            "nonleg": safe_float(group_metrics(group_json).get("nonleg")),
            "arm": safe_float(group_metrics(group_json).get("arm")),
            "foot_l_ball_l_SIC12_15": safe_float(window_group_stats(eval_json).get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
            "calf_r_SIC2_4": safe_float(window_group_stats(eval_json).get("hotspots", {}).get("calf_r_SIC2_4")),
        },
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_METRICS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def stage_row_to_eval(stage_row: Mapping[str, Any]) -> Dict[str, Any]:
    eval_json = Path(str(stage_row["eval_json"]))
    group_json = Path(str(stage_row["group_json"]))
    return collect_eval(eval_json, group_json)


def build_markdown(summary: Mapping[str, Any]) -> str:
    cand_replace = summary["references"]["candidate_replace"]["eval"]["selected_metrics"]
    docs_70r = summary["references"]["docs_baseline_70R"]["eval"]["selected_metrics"]
    cand_70r = summary["candidate_70R"]["eval"]["selected_metrics"]
    d_rep = summary["candidate_70R"]["delta_vs_candidate_replace"]
    d_docs = summary["candidate_70R"]["delta_vs_docs_baseline_70R"]
    lines = [
        "# ep014center replace(lr5e-5) -> 70R",
        "",
        f"- promoted replace case: `{PROMOTED_REPLACE_CASE}`",
        f"- replace ckpt: `{summary['references']['candidate_replace']['ckpt']}`",
        f"- 70R config: `{summary['candidate_70R']['config']}`",
        f"- 70R ckpt: `{summary['candidate_70R']['ckpt']}`",
        "",
        "## Metrics",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| candidate_replace_lr5e5 | {fmt(cand_replace['DirectGeoLocalDeg'])} | {fmt(cand_replace['all_ex_root'])} | {fmt(cand_replace['leg'])} | {fmt(cand_replace['nonleg'])} | {fmt(cand_replace['arm'])} | {fmt(cand_replace['foot_l_ball_l_SIC12_15'])} | {fmt(cand_replace['calf_r_SIC2_4'])} |",
        f"| docs_baseline_70R | {fmt(docs_70r['DirectGeoLocalDeg'])} | {fmt(docs_70r['all_ex_root'])} | {fmt(docs_70r['leg'])} | {fmt(docs_70r['nonleg'])} | {fmt(docs_70r['arm'])} | {fmt(docs_70r['foot_l_ball_l_SIC12_15'])} | {fmt(docs_70r['calf_r_SIC2_4'])} |",
        f"| candidate_70R | {fmt(cand_70r['DirectGeoLocalDeg'])} | {fmt(cand_70r['all_ex_root'])} | {fmt(cand_70r['leg'])} | {fmt(cand_70r['nonleg'])} | {fmt(cand_70r['arm'])} | {fmt(cand_70r['foot_l_ball_l_SIC12_15'])} | {fmt(cand_70r['calf_r_SIC2_4'])} |",
        "",
        "## Deltas",
        "",
        "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| candidate_70R - candidate_replace_lr5e5 | {fmt(d_rep['DirectGeoLocalDeg'])} | {fmt(d_rep['all_ex_root'])} | {fmt(d_rep['leg'])} | {fmt(d_rep['nonleg'])} | {fmt(d_rep['arm'])} | {fmt(d_rep['foot_l_ball_l_SIC12_15'])} | {fmt(d_rep['calf_r_SIC2_4'])} |",
        f"| candidate_70R - docs_baseline_70R | {fmt(d_docs['DirectGeoLocalDeg'])} | {fmt(d_docs['all_ex_root'])} | {fmt(d_docs['leg'])} | {fmt(d_docs['nonleg'])} | {fmt(d_docs['arm'])} | {fmt(d_docs['foot_l_ball_l_SIC12_15'])} | {fmt(d_docs['calf_r_SIC2_4'])} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    raise SystemExit(
        "[FATAL][Removed] replace->70R direct handoff script entry was removed by "
        "2026-04-28 strict branch unload cleanup. Migration: use "
        "`tools/run_strict_70r_warmstart_bridge_probe.py` or "
        "`tools/contractize_strict_posttrain_handoff.py --tensor-donor ... "
        "--transplant-prefix direct_pose_` to build an explicit warmstart bridge; "
        "no direct replace ckpt_in handoff replacement."
    )
    required = [REPLACE_SWEEP_SUMMARY_JSON, DOCS_BASELINE_SUMMARY_JSON, CONFIG_70R, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    replace_sweep = load_json(REPLACE_SWEEP_SUMMARY_JSON)
    replace_case = replace_sweep["cases"][PROMOTED_REPLACE_CASE]
    replace_ckpt = Path(str(replace_case["last_ckpt"]))
    if not replace_ckpt.is_file():
        raise SystemExit(f"missing replace ckpt: {replace_ckpt}")

    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)
    docs_70r = stage_row_to_eval(docs_summary["stage_progress_model_source"]["70R"])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"
    cfg_70r = OUT_ROOT / "configs" / f"posttrain_70R_from_ep014center_replace_lr5e5_{RUN_DATE}.json"
    run_name_70r = f"WalkF_stage7_70R_from_ep014center_replace_lr5e5_s180_{RUN_DATE}"

    log("=== candidate 70R from replace lr5e5 ===")
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

    log("=== eval candidate 70R ===")
    eval_dir = OUT_ROOT / "eval_model_source" / "70R"
    group_json = OUT_ROOT / "eval_model_source" / "70R_group_summary.json"
    eval_json = run_eval(
        model_ckpt=ckpt_70r,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_json, group_json, log_file=lane_log)
    candidate_70r_eval = collect_eval(eval_json, group_json)

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "promoted_replace_case": PROMOTED_REPLACE_CASE,
            "source_summary": str(REPLACE_SWEEP_SUMMARY_JSON),
            "docs_baseline_summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "compare_contract": "model_source",
            "70R_helper": "run_70r_promote(full trunk, s180 promote path)",
        },
        "references": {
            "candidate_replace": {
                "ckpt": str(replace_ckpt),
                "eval": replace_case["eval"],
            },
            "docs_baseline_70R": {
                "eval": docs_70r,
            },
        },
        "candidate_70R": {
            "config": str(cfg_70r),
            "ckpt": str(ckpt_70r),
            "eval": candidate_70r_eval,
            "delta_vs_candidate_replace": metric_delta(
                candidate_70r_eval["selected_metrics"],
                replace_case["eval"]["selected_metrics"],
            ),
            "delta_vs_docs_baseline_70R": metric_delta(
                candidate_70r_eval["selected_metrics"],
                docs_70r["selected_metrics"],
            ),
        },
    }

    write_json(
        OUT_ROOT / "status.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "replace_ckpt": str(replace_ckpt),
            "candidate_70R_ckpt": str(ckpt_70r),
            "candidate_70R_eval": str(eval_json),
        },
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
