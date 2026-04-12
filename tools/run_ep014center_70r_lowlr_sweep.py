#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

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
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_70r_lowlr_sweep_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_70r_lowlr_sweep_{RUN_DATE}"
REPLACE_SWEEP_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
CURRENT_70R_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to70r_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"

CASES: Tuple[Tuple[str, float], ...] = (
    ("lr2e4", 2e-4),
    ("lr1e4", 1e-4),
    ("lr5e5", 5e-5),
)
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


def replay_case(*, case_name: str, lr: float, replace_ckpt: Path, lane_log: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    cfg_json = OUT_ROOT / "configs" / f"posttrain_70R_{case_name}_from_ep014center_replace_{RUN_DATE}.json"
    run_name = f"WalkF_stage7_70R_{case_name}_from_ep014center_replace_lr5e5_s180_{RUN_DATE}"
    make_generated_config(
        CONFIG_70R,
        cfg_json,
        {
            "ckpt_in": str(replace_ckpt),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": float(lr),
            "epochs": 1,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt = run_70r_promote(
        config_json=cfg_json,
        out_dir=out_dir,
        run_name=run_name,
        log_file=lane_log,
    )
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"
    eval_json = run_eval(
        model_ckpt=ckpt,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_json, group_json, log_file=lane_log)
    return {
        "lr": float(lr),
        "config": str(cfg_json),
        "run_name": run_name,
        "last_ckpt": str(ckpt),
        "eval": collect_eval(eval_json, group_json),
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center 70R low-LR sweep",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("replace_lr5e5_input", refs["replace_input"]["selected_metrics"]),
        row("current_70R_lr3e4", refs["current_70R_lr3e4"]["selected_metrics"]),
        row("docs_baseline_70R", refs["docs_baseline_70R"]["selected_metrics"]),
    ]
    for case_name, payload in summary["cases"].items():
        lines.append(row(case_name, payload["eval"]["selected_metrics"]))

    lines.extend(
        [
            "",
            "## Delta vs replace input",
            "",
            "| lane | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, payload in summary["cases"].items():
        lines.append(row(case_name, payload["delta_vs_replace_input"]))

    lines.extend(
        [
            "",
            "## Delta vs docs baseline 70R",
            "",
            "| lane | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, payload in summary["cases"].items():
        lines.append(row(case_name, payload["delta_vs_docs_baseline_70R"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [REPLACE_SWEEP_SUMMARY_JSON, CURRENT_70R_SUMMARY_JSON, DOCS_BASELINE_SUMMARY_JSON, CONFIG_70R, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    replace_sweep = load_json(REPLACE_SWEEP_SUMMARY_JSON)
    current_70r = load_json(CURRENT_70R_SUMMARY_JSON)
    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)

    replace_case = replace_sweep["cases"]["lr5e5"]
    replace_ckpt = Path(str(replace_case["last_ckpt"]))
    if not replace_ckpt.is_file():
        raise SystemExit(f"missing replace ckpt: {replace_ckpt}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    refs = {
        "replace_input": replace_case["eval"],
        "current_70R_lr3e4": current_70r["candidate_70R"]["eval"],
        "docs_baseline_70R": stage_row_to_eval(docs_summary["stage_progress_model_source"]["70R"]),
    }

    cases: Dict[str, Any] = {}
    for case_name, lr in CASES:
        log(f"=== replay {case_name} lr={lr:.6g} ===")
        payload = replay_case(case_name=case_name, lr=lr, replace_ckpt=replace_ckpt, lane_log=lane_log)
        payload["delta_vs_replace_input"] = metric_delta(payload["eval"]["selected_metrics"], refs["replace_input"]["selected_metrics"])
        payload["delta_vs_docs_baseline_70R"] = metric_delta(payload["eval"]["selected_metrics"], refs["docs_baseline_70R"]["selected_metrics"])
        cases[case_name] = payload

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "input_replace_case": "lr5e5",
            "sweep": [{"name": name, "lr": lr} for name, lr in CASES],
            "compare_contract": "model_source",
            "70R_base_config": str(CONFIG_70R),
        },
        "references": refs,
        "cases": cases,
        "answers": {
            "best_case_by_all_ex_root": min(cases, key=lambda name: safe_float(cases[name]["eval"]["selected_metrics"]["all_ex_root"])),
            "best_case_by_nonleg": min(cases, key=lambda name: safe_float(cases[name]["eval"]["selected_metrics"]["nonleg"])),
        },
    }
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
