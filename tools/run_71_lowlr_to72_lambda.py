#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
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
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        masked_metric_means,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260314"
LOWLR_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "summary.json"
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_newflow_chain_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_71_lowlr_to72lambda_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_71_lowlr_to72lambda_{RUN_DATE}"

SELECTED_KEYS = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "legs_main",
    "arms_main",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def selected_metrics(stage_payload: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "DirectGeoLocalDeg": safe_float(stage_payload["masked_means"]["DirectGeoLocalDeg"]),
        "all_ex_root": safe_float(stage_payload["direct_group_summary"]["all_ex_root"]),
        "leg": safe_float(stage_payload["direct_group_summary"]["leg"]),
        "nonleg": safe_float(stage_payload["direct_group_summary"]["nonleg"]),
        "arm": safe_float(stage_payload["direct_group_summary"]["arm"]),
        "legs_main": safe_float(stage_payload["window_summary"]["overall"]["legs_main"]),
        "arms_main": safe_float(stage_payload["window_summary"]["overall"]["arms_main"]),
        "foot_l_ball_l_SIC12_15": safe_float(stage_payload["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"]),
        "calf_r_SIC2_4": safe_float(stage_payload["window_summary"]["hotspots"]["calf_r_SIC2_4"]),
    }


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
            "legs_main": safe_float(window_group_stats(eval_json).get("overall", {}).get("legs_main")),
            "arms_main": safe_float(window_group_stats(eval_json).get("overall", {}).get("arms_main")),
            "foot_l_ball_l_SIC12_15": safe_float(window_group_stats(eval_json).get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
            "calf_r_SIC2_4": safe_float(window_group_stats(eval_json).get("hotspots", {}).get("calf_r_SIC2_4")),
        },
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_KEYS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def eval_model(model_ckpt: Path, out_dir: Path, group_json: Path, log_file: Path) -> Dict[str, Any]:
    eval_json = run_eval(
        model_ckpt=model_ckpt,
        out_dir=out_dir,
        contacts_source="model",
        log_file=log_file,
    )
    ensure_group_summary(eval_json, group_json, log_file=log_file)
    return collect_eval(eval_json, group_json)


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]
    cand = summary["candidate"]
    lines = [
        "# low-LR 71 -> 72 -> lambda",
        "",
        f"- 71 source ckpt: `{refs['candidate_71_ckpt']}`",
        "- start lane: candidate 70R -> 71(lr=3e-4)",
        "- eval contract: model-source only",
        "",
        "## Metrics",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = [
        ("candidate_71_lowlr", cand["eval_71"]["selected_metrics"]),
        ("current_72", refs["current_72"]),
        ("candidate_72", cand["eval_72"]["selected_metrics"]),
        ("current_lambda", refs["current_lambda"]),
        ("candidate_lambda", cand["eval_lambda"]["selected_metrics"]),
    ]
    for label, row in table_rows:
        lines.append(
            f"| {label} | {fmt(row['DirectGeoLocalDeg'])} | {fmt(row['all_ex_root'])} | {fmt(row['leg'])} | {fmt(row['nonleg'])} | {fmt(row['arm'])} | {fmt(row['legs_main'])} | {fmt(row['arms_main'])} | {fmt(row['foot_l_ball_l_SIC12_15'])} | {fmt(row['calf_r_SIC2_4'])} |"
        )
    lines.extend(
        [
            "",
            "## Key deltas",
            "",
            f"- candidate 72 vs current 72: all_ex_root={fmt(summary['comparisons']['candidate_72_vs_current_72']['all_ex_root'])}, leg={fmt(summary['comparisons']['candidate_72_vs_current_72']['leg'])}, foot_l/ball_l@SIC12-15={fmt(summary['comparisons']['candidate_72_vs_current_72']['foot_l_ball_l_SIC12_15'])}, calf_r@SIC2-4={fmt(summary['comparisons']['candidate_72_vs_current_72']['calf_r_SIC2_4'])}",
            f"- candidate lambda vs current lambda: all_ex_root={fmt(summary['comparisons']['candidate_lambda_vs_current_lambda']['all_ex_root'])}, leg={fmt(summary['comparisons']['candidate_lambda_vs_current_lambda']['leg'])}, foot_l/ball_l@SIC12-15={fmt(summary['comparisons']['candidate_lambda_vs_current_lambda']['foot_l_ball_l_SIC12_15'])}, calf_r@SIC2-4={fmt(summary['comparisons']['candidate_lambda_vs_current_lambda']['calf_r_SIC2_4'])}",
            f"- candidate 72 vs candidate 71: all_ex_root={fmt(summary['comparisons']['candidate_72_vs_candidate_71']['all_ex_root'])}, leg={fmt(summary['comparisons']['candidate_72_vs_candidate_71']['leg'])}, foot_l/ball_l@SIC12-15={fmt(summary['comparisons']['candidate_72_vs_candidate_71']['foot_l_ball_l_SIC12_15'])}, calf_r@SIC2-4={fmt(summary['comparisons']['candidate_72_vs_candidate_71']['calf_r_SIC2_4'])}",
            f"- candidate lambda vs candidate 72: all_ex_root={fmt(summary['comparisons']['candidate_lambda_vs_candidate_72']['all_ex_root'])}, leg={fmt(summary['comparisons']['candidate_lambda_vs_candidate_72']['leg'])}, foot_l/ball_l@SIC12-15={fmt(summary['comparisons']['candidate_lambda_vs_candidate_72']['foot_l_ball_l_SIC12_15'])}, calf_r@SIC2-4={fmt(summary['comparisons']['candidate_lambda_vs_candidate_72']['calf_r_SIC2_4'])}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [LOWLR_SUMMARY_JSON, SOURCE_SUMMARY_JSON, CONFIG_72, CONFIG_LAMBDA, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    lowlr_summary = load_json(LOWLR_SUMMARY_JSON)
    source_summary = load_json(SOURCE_SUMMARY_JSON)
    candidate_71_ckpt = Path(str(lowlr_summary["cases"]["lr3e4"]["last_ckpt"]))
    if not candidate_71_ckpt.is_file():
        raise RuntimeError(f"missing candidate lowlr 71 ckpt: {candidate_71_ckpt}")

    log("=== candidate 72 from lowlr 71 ===")
    ckpt_72 = run_posttrain_stage(
        config=CONFIG_72,
        ckpt_in=candidate_71_ckpt,
        out_dir=MODEL_ROOT / "72",
        run_name=f"WalkF_stage7_72_from_lowlr71_{RUN_DATE}",
        log_file=lane_log,
    )

    log("=== candidate lambda from lowlr 72 ===")
    ckpt_lambda = run_posttrain_stage(
        config=CONFIG_LAMBDA,
        ckpt_in=ckpt_72,
        out_dir=MODEL_ROOT / "lambda",
        run_name=f"WalkF_stage7_lambda_from_lowlr72_{RUN_DATE}",
        log_file=lane_log,
    )

    log("=== eval candidate 72 ===")
    eval_72 = eval_model(
        model_ckpt=ckpt_72,
        out_dir=OUT_ROOT / "eval_72_model",
        group_json=OUT_ROOT / "eval_72_model_group_summary.json",
        log_file=lane_log,
    )

    log("=== eval candidate lambda ===")
    eval_lambda = eval_model(
        model_ckpt=ckpt_lambda,
        out_dir=OUT_ROOT / "eval_lambda_model",
        group_json=OUT_ROOT / "eval_lambda_model_group_summary.json",
        log_file=lane_log,
    )

    refs = {
        "current_71": selected_metrics(source_summary["stage_progress_model_source"]["71"]),
        "current_72": selected_metrics(source_summary["stage_progress_model_source"]["72"]),
        "current_lambda": selected_metrics(source_summary["stage_progress_model_source"]["lambda"]),
        "candidate_71_ckpt": str(candidate_71_ckpt),
        "candidate_71_metrics": lowlr_summary["cases"]["lr3e4"]["snapshots"]["s180"]["eval"]["selected_metrics"],
        "source_summary": str(SOURCE_SUMMARY_JSON),
        "lowlr_summary": str(LOWLR_SUMMARY_JSON),
    }

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "compare_contract": "model_source",
            "note": "downstream continuation from candidate 71(lr=3e-4) only",
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        },
        "references": refs,
        "candidate": {
            "ckpt_72": str(ckpt_72),
            "ckpt_lambda": str(ckpt_lambda),
            "eval_71": {"selected_metrics": refs["candidate_71_metrics"]},
            "eval_72": eval_72,
            "eval_lambda": eval_lambda,
        },
        "comparisons": {
            "candidate_72_vs_current_72": metric_delta(eval_72["selected_metrics"], refs["current_72"]),
            "candidate_lambda_vs_current_lambda": metric_delta(eval_lambda["selected_metrics"], refs["current_lambda"]),
            "candidate_72_vs_candidate_71": metric_delta(eval_72["selected_metrics"], refs["candidate_71_metrics"]),
            "candidate_lambda_vs_candidate_72": metric_delta(eval_lambda["selected_metrics"], eval_72["selected_metrics"]),
        },
    }

    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    write_json(summary_json, summary)
    summary_md.write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
