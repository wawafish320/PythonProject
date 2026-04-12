#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
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
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
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
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260328"
REPLACE_SWEEP_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_replace_lr5e5_to71_lr3e4_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_replace_lr5e5_to71_lr3e4_{RUN_DATE}"
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
    masked = masked_metric_means(eval_json)
    groups = group_metrics(group_json)
    window = window_group_stats(eval_json)
    return {
        "masked_means": masked,
        "direct_group_summary": groups,
        "window_summary": window,
        "selected_metrics": {
            "DirectGeoLocalDeg": safe_float(masked.get("DirectGeoLocalDeg")),
            "all_ex_root": safe_float(groups.get("all_ex_root")),
            "leg": safe_float(groups.get("leg")),
            "nonleg": safe_float(groups.get("nonleg")),
            "arm": safe_float(groups.get("arm")),
            "foot_l_ball_l_SIC12_15": safe_float(window.get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
            "calf_r_SIC2_4": safe_float(window.get("hotspots", {}).get("calf_r_SIC2_4")),
        },
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_METRICS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def stage_row_to_eval(stage_row: Mapping[str, Any]) -> Dict[str, Any]:
    return collect_eval(Path(str(stage_row["eval_json"])), Path(str(stage_row["group_json"])))


def build_markdown(summary: Mapping[str, Any]) -> str:
    replace_metrics = summary["references"]["replace_input"]["selected_metrics"]
    docs_71_metrics = summary["references"]["docs_baseline_71_lr3e4"]["selected_metrics"]
    cand_metrics = summary["candidate_71"]["eval"]["selected_metrics"]
    d_replace = summary["candidate_71"]["delta_vs_replace_input"]
    d_docs = summary["candidate_71"]["delta_vs_docs_baseline_71_lr3e4"]
    lines = [
        "# ep014center replace(lr5e-5) -> 71(lr=3e-4)",
        "",
        f"- replace ckpt: `{summary['references']['replace_ckpt']}`",
        f"- 71 config: `{summary['candidate_71']['config']}`",
        f"- 71 ckpt: `{summary['candidate_71']['ckpt']}`",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| replace_input_lr5e5 | {fmt(replace_metrics['DirectGeoLocalDeg'])} | {fmt(replace_metrics['all_ex_root'])} | {fmt(replace_metrics['leg'])} | {fmt(replace_metrics['nonleg'])} | {fmt(replace_metrics['arm'])} | {fmt(replace_metrics['foot_l_ball_l_SIC12_15'])} | {fmt(replace_metrics['calf_r_SIC2_4'])} |",
        f"| docs_baseline_71_lr3e4 | {fmt(docs_71_metrics['DirectGeoLocalDeg'])} | {fmt(docs_71_metrics['all_ex_root'])} | {fmt(docs_71_metrics['leg'])} | {fmt(docs_71_metrics['nonleg'])} | {fmt(docs_71_metrics['arm'])} | {fmt(docs_71_metrics['foot_l_ball_l_SIC12_15'])} | {fmt(docs_71_metrics['calf_r_SIC2_4'])} |",
        f"| candidate_71 | {fmt(cand_metrics['DirectGeoLocalDeg'])} | {fmt(cand_metrics['all_ex_root'])} | {fmt(cand_metrics['leg'])} | {fmt(cand_metrics['nonleg'])} | {fmt(cand_metrics['arm'])} | {fmt(cand_metrics['foot_l_ball_l_SIC12_15'])} | {fmt(cand_metrics['calf_r_SIC2_4'])} |",
        "",
        "## Deltas",
        "",
        "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| candidate_71 - replace_input | {fmt(d_replace['DirectGeoLocalDeg'])} | {fmt(d_replace['all_ex_root'])} | {fmt(d_replace['leg'])} | {fmt(d_replace['nonleg'])} | {fmt(d_replace['arm'])} | {fmt(d_replace['foot_l_ball_l_SIC12_15'])} | {fmt(d_replace['calf_r_SIC2_4'])} |",
        f"| candidate_71 - docs_baseline_71_lr3e4 | {fmt(d_docs['DirectGeoLocalDeg'])} | {fmt(d_docs['all_ex_root'])} | {fmt(d_docs['leg'])} | {fmt(d_docs['nonleg'])} | {fmt(d_docs['arm'])} | {fmt(d_docs['foot_l_ball_l_SIC12_15'])} | {fmt(d_docs['calf_r_SIC2_4'])} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [REPLACE_SWEEP_SUMMARY_JSON, DOCS_BASELINE_SUMMARY_JSON, CONFIG_71, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    replace_sweep = load_json(REPLACE_SWEEP_SUMMARY_JSON)
    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)
    replace_case = replace_sweep["cases"]["lr5e5"]
    replace_ckpt = Path(str(replace_case["last_ckpt"]))
    if not replace_ckpt.is_file():
        raise SystemExit(f"missing replace ckpt: {replace_ckpt}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"
    cfg_71 = OUT_ROOT / "configs" / f"posttrain_71_lr3e4_from_ep014center_replace_lr5e5_{RUN_DATE}.json"
    run_name_71 = f"WalkF_stage7_71_lr3e4_from_ep014center_replace_lr5e5_{RUN_DATE}"

    log("=== candidate 71 directly from replace lr5e5 ===")
    cfg_71 = make_generated_config(
        CONFIG_71,
        cfg_71,
        {
            "ckpt_in": str(replace_ckpt),
            "out_dir": str(MODEL_ROOT / "71"),
            "run_name": run_name_71,
            "lr": 3e-4,
            "epochs": 3,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_71 = run_posttrain_stage(
        config=cfg_71,
        ckpt_in=replace_ckpt,
        out_dir=MODEL_ROOT / "71",
        run_name=run_name_71,
        log_file=lane_log,
    )

    log("=== eval candidate 71 ===")
    eval_dir = OUT_ROOT / "eval_model_source" / "71"
    group_json = OUT_ROOT / "eval_model_source" / "71_group_summary.json"
    eval_json = run_eval(
        model_ckpt=ckpt_71,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_json, group_json, log_file=lane_log)
    candidate_eval = collect_eval(eval_json, group_json)
    docs_71 = stage_row_to_eval(docs_summary["stage_progress_model_source"]["71_lr3e4"])

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "source_replace_case": "lr5e5",
            "71_lr": 3e-4,
            "71_base_config": str(CONFIG_71),
            "compare_contract": "model_source",
        },
        "references": {
            "replace_ckpt": str(replace_ckpt),
            "replace_input": replace_case["eval"],
            "docs_baseline_71_lr3e4": docs_71,
        },
        "candidate_71": {
            "config": str(cfg_71),
            "ckpt": str(ckpt_71),
            "eval": candidate_eval,
            "delta_vs_replace_input": metric_delta(candidate_eval["selected_metrics"], replace_case["eval"]["selected_metrics"]),
            "delta_vs_docs_baseline_71_lr3e4": metric_delta(candidate_eval["selected_metrics"], docs_71["selected_metrics"]),
        },
    }
    write_json(
        OUT_ROOT / "status.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "replace_ckpt": str(replace_ckpt),
            "candidate_71_ckpt": str(ckpt_71),
            "candidate_71_eval": str(eval_json),
        },
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
