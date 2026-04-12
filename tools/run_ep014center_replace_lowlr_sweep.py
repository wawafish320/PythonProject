#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
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
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_replace_lowlr_sweep_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_replace_lowlr_sweep_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"

SOURCE_70A_ARTIFACTS_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "artifacts.json"
CURRENT_REPLACE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_70alr3e4_lowdrift_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"

SELECTED_METRICS: Tuple[str, ...] = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)

CASES: Tuple[Tuple[str, float], ...] = (
    ("lr2e4", 2e-4),
    ("lr1e4", 1e-4),
    ("lr5e5", 5e-5),
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def selected_metrics_from_eval(eval_json: Path, group_json: Path) -> Dict[str, float]:
    masked = masked_metric_means(eval_json)
    groups = group_metrics(group_json)
    window = window_group_stats(eval_json)
    return {
        "DirectGeoLocalDeg": safe_float(masked.get("DirectGeoLocalDeg")),
        "all_ex_root": safe_float(groups.get("all_ex_root")),
        "leg": safe_float(groups.get("leg")),
        "nonleg": safe_float(groups.get("nonleg")),
        "arm": safe_float(groups.get("arm")),
        "foot_l_ball_l_SIC12_15": safe_float(window.get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
        "calf_r_SIC2_4": safe_float(window.get("hotspots", {}).get("calf_r_SIC2_4")),
    }


def collect_eval(eval_json: Path, group_json: Path) -> Dict[str, Any]:
    return {
        "masked_means": masked_metric_means(eval_json),
        "direct_group_summary": group_metrics(group_json),
        "window_summary": window_group_stats(eval_json),
        "selected_metrics": selected_metrics_from_eval(eval_json, group_json),
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_METRICS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def _stage_row_to_eval(stage_row: Mapping[str, Any]) -> Dict[str, Any]:
    eval_json = Path(str(stage_row["eval_json"]))
    group_json = Path(str(stage_row["group_json"]))
    if eval_json.is_file() and group_json.is_file():
        return collect_eval(eval_json, group_json)
    selected = {
        "DirectGeoLocalDeg": safe_float(stage_row.get("DirectGeoLocalDeg")),
        "all_ex_root": safe_float(stage_row.get("all_ex_root")),
        "leg": safe_float(stage_row.get("leg")),
        "nonleg": safe_float(stage_row.get("nonleg")),
        "arm": safe_float(stage_row.get("arm")),
        "foot_l_ball_l_SIC12_15": safe_float(stage_row.get("foot_l_ball_l_SIC12_15")),
        "calf_r_SIC2_4": safe_float(stage_row.get("calf_r_SIC2_4")),
    }
    return {
        "masked_means": {"DirectGeoLocalDeg": safe_float(stage_row.get("DirectGeoLocalDeg"))},
        "direct_group_summary": {
            "all_ex_root": safe_float(stage_row.get("all_ex_root")),
            "leg": safe_float(stage_row.get("leg")),
            "nonleg": safe_float(stage_row.get("nonleg")),
            "arm": safe_float(stage_row.get("arm")),
        },
        "window_summary": {
            "hotspots": {
                "foot_l_ball_l_SIC12_15": safe_float(stage_row.get("foot_l_ball_l_SIC12_15")),
                "calf_r_SIC2_4": safe_float(stage_row.get("calf_r_SIC2_4")),
            }
        },
        "selected_metrics": selected,
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def load_references() -> Dict[str, Any]:
    source_artifacts = load_json(SOURCE_70A_ARTIFACTS_JSON)
    source_70a = source_artifacts["lr3e4"]
    source_70a_eval = collect_eval(Path(str(source_70a["eval_json"])), Path(str(source_70a["group_json"])))

    current_replace_summary = load_json(CURRENT_REPLACE_SUMMARY_JSON)
    current_replace_eval = collect_eval(
        Path(str(current_replace_summary["eval_json"])),
        Path(str(current_replace_summary["group_json"])),
    )

    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)
    docs_stage = docs_summary["stage_progress_model_source"]["new70b_replace_lowdrift"]
    docs_replace_eval = _stage_row_to_eval(docs_stage)

    return {
        "source_70a": {
            "ckpt": str(source_70a["ckpt"]),
            "config": str(source_70a["config_json"]),
            "eval": source_70a_eval,
        },
        "current_replace_lr3e4": {
            "ckpt": str(current_replace_summary["candidate_ckpt"]),
            "config": str(current_replace_summary["config_json"]),
            "warmstart_ckpt": str(current_replace_summary["warmstart_ckpt"]),
            "eval": current_replace_eval,
        },
        "docs_baseline_replace": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": docs_replace_eval,
        },
    }


def replay_case(
    *,
    case_name: str,
    lr: float,
    base_config: Path,
    warmstart_ckpt: Path,
    log_file: Path,
) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_from_ep014center_70alr3e4_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_from_ep014center_{RUN_DATE}.json"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"

    make_generated_config(
        base_config,
        cfg_json,
        {
            "ckpt_in": str(warmstart_ckpt),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": float(lr),
        },
    )

    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=warmstart_ckpt,
        out_dir=out_dir,
        run_name=run_name,
        log_file=log_file,
    )
    eval_json = run_eval(
        model_ckpt=ckpt,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=log_file,
    )
    ensure_group_summary(eval_json, group_json, log_file=log_file)
    eval_payload = collect_eval(eval_json, group_json)
    return {
        "lr": float(lr),
        "config": str(cfg_json),
        "run_name": run_name,
        "last_ckpt": str(ckpt),
        "eval": eval_payload,
    }


def _leq(a: Any, b: Any, eps: float = 1e-9) -> bool:
    va = safe_float(a)
    vb = safe_float(b)
    if not math.isfinite(va) or not math.isfinite(vb):
        return False
    return va <= vb + eps


def build_markdown(summary: Mapping[str, Any]) -> str:
    source = summary["references"]["source_70a"]["eval"]["selected_metrics"]
    current = summary["references"]["current_replace_lr3e4"]["eval"]["selected_metrics"]
    docs = summary["references"]["docs_baseline_replace"]["eval"]["selected_metrics"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    def delta_row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center replace low-LR sweep",
        "",
        "- fixed source is `70a(lr=3e-4)` from the exact `ep014center` Stage6 winner",
        "- replace semantics are cloned from the current `new70b_replace_lowdrift` generated config",
        "- only `lr` changes across candidates",
        "",
        "## Reference rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("source_70a_lr3e4", source),
        row("current_replace_lr3e4", current),
        row("docs_baseline_replace", docs),
        "",
        "## Candidate rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, payload in summary["cases"].items():
        lines.append(row(case_name, payload["eval"]["selected_metrics"]))

    lines.extend(
        [
            "",
            "## Deltas vs source 70a(lr=3e-4)",
            "",
            "| lane | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, payload in summary["cases"].items():
        lines.append(delta_row(case_name, payload["delta_vs_source_70a"]))

    lines.extend(
        [
            "",
            "## Deltas vs docs baseline replace",
            "",
            "| lane | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, payload in summary["cases"].items():
        lines.append(delta_row(case_name, payload["delta_vs_docs_baseline"]))

    lines.extend(
        [
            "",
            "## Hard gates",
            "",
            "| lane | calf_delta_vs_source<=0.40 | final_calf<=docs | broad<=docs(all4) | foot<=docs |",
            "|---|---|---|---|---|",
        ]
    )
    for case_name, payload in summary["cases"].items():
        judge = payload["judgement"]
        lines.append(
            f"| {case_name} | {str(bool(judge['calf_delta_vs_source_leq_0p40'])).lower()} | "
            f"{str(bool(judge['final_calf_leq_docs_baseline'])).lower()} | "
            f"{str(bool(judge['broad_all4_leq_docs_baseline'])).lower()} | "
            f"{str(bool(judge['foot_leq_docs_baseline'])).lower()} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [SOURCE_70A_ARTIFACTS_JSON, CURRENT_REPLACE_SUMMARY_JSON, DOCS_BASELINE_SUMMARY_JSON]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    references = load_references()
    current_replace = load_json(CURRENT_REPLACE_SUMMARY_JSON)
    base_config = Path(str(current_replace["config_json"]))
    warmstart_ckpt = Path(str(current_replace["warmstart_ckpt"]))
    source_70a_ckpt = Path(str(references["source_70a"]["ckpt"]))
    if not base_config.is_file():
        raise SystemExit(f"missing base generated config: {base_config}")
    if not warmstart_ckpt.is_file():
        raise SystemExit(f"missing warmstart ckpt: {warmstart_ckpt}")
    if not source_70a_ckpt.is_file():
        raise SystemExit(f"missing source 70a ckpt: {source_70a_ckpt}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    source_metrics = references["source_70a"]["eval"]["selected_metrics"]
    docs_metrics = references["docs_baseline_replace"]["eval"]["selected_metrics"]

    cases: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    for case_name, lr in CASES:
        log(f"=== replay {case_name} lr={lr:.6g} ===")
        case_payload = replay_case(
            case_name=case_name,
            lr=lr,
            base_config=base_config,
            warmstart_ckpt=warmstart_ckpt,
            log_file=lane_log,
        )
        selected = case_payload["eval"]["selected_metrics"]
        case_payload["delta_vs_source_70a"] = metric_delta(selected, source_metrics)
        case_payload["delta_vs_docs_baseline"] = metric_delta(selected, docs_metrics)
        case_payload["judgement"] = {
            "calf_delta_vs_source_leq_0p40": _leq(case_payload["delta_vs_source_70a"]["calf_r_SIC2_4"], 0.40),
            "final_calf_leq_docs_baseline": _leq(selected["calf_r_SIC2_4"], docs_metrics["calf_r_SIC2_4"]),
            "broad_all4_leq_docs_baseline": all(
                _leq(selected[key], docs_metrics[key]) for key in ("all_ex_root", "leg", "nonleg", "arm")
            ),
            "foot_leq_docs_baseline": _leq(selected["foot_l_ball_l_SIC12_15"], docs_metrics["foot_l_ball_l_SIC12_15"]),
        }
        cases[case_name] = case_payload
        artifacts[case_name] = {
            "config_json": case_payload["config"],
            "ckpt": case_payload["last_ckpt"],
            "eval_json": case_payload["eval"]["paths"]["eval_json"],
            "group_json": case_payload["eval"]["paths"]["group_summary"],
        }

    feasible_cases = [
        case_name
        for case_name, payload in cases.items()
        if bool(payload["judgement"]["calf_delta_vs_source_leq_0p40"])
        and bool(payload["judgement"]["final_calf_leq_docs_baseline"])
    ]

    best_case_by_all_ex_root = min(
        cases.items(),
        key=lambda item: safe_float(item[1]["eval"]["selected_metrics"]["all_ex_root"]),
    )[0]
    best_feasible_case_by_all_ex_root: Optional[str] = None
    if feasible_cases:
        best_feasible_case_by_all_ex_root = min(
            feasible_cases,
            key=lambda name: safe_float(cases[name]["eval"]["selected_metrics"]["all_ex_root"]),
        )

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "source_stage": "70a(lr=3e-4) from exact ep014center Stage6 winner",
            "replace_semantics_base_config": str(base_config),
            "shared_warmstart_ckpt": str(warmstart_ckpt),
            "compare_contract": "model_source",
            "sweep": [{"name": name, "lr": lr} for name, lr in CASES],
            "note": "only replace LR changes; warmstart, contract, epochs=1, steps_per_epoch=60 stay fixed",
        },
        "references": references,
        "cases": cases,
        "answers": {
            "feasible_cases_by_calf_rules": feasible_cases,
            "best_case_by_all_ex_root": best_case_by_all_ex_root,
            "best_feasible_case_by_all_ex_root": best_feasible_case_by_all_ex_root,
            "current_replace_lr3e4_fails_calf_delta_vs_source": not _leq(
                references["current_replace_lr3e4"]["eval"]["selected_metrics"]["calf_r_SIC2_4"]
                - references["source_70a"]["eval"]["selected_metrics"]["calf_r_SIC2_4"],
                0.40,
            ),
            "current_replace_lr3e4_fails_final_calf_vs_docs": not _leq(
                references["current_replace_lr3e4"]["eval"]["selected_metrics"]["calf_r_SIC2_4"],
                references["docs_baseline_replace"]["eval"]["selected_metrics"]["calf_r_SIC2_4"],
            ),
        },
    }

    write_json(OUT_ROOT / "artifacts.json", artifacts)
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
