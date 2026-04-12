#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        ROOT,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        masked_metric_means,
        run_eval,
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
        masked_metric_means,
        run_eval,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260328"
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 1, 5, 20, 60, 180)
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to70r_20260328" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_70r_snapshot_sweep_{RUN_DATE}"
STEP_TAGS = {step: f"s{step:03d}" for step in SNAPSHOT_STEPS}
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


def build_markdown(summary: Mapping[str, Any]) -> str:
    replace_metrics = summary["references"]["candidate_replace"]["selected_metrics"]
    baseline_70r_metrics = summary["references"]["docs_baseline_70R"]["selected_metrics"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center 70R snapshot sweep",
        "",
        f"- source summary: `{SOURCE_SUMMARY_JSON}`",
        "",
        "## References",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("candidate_replace_lr5e5", replace_metrics),
        row("docs_baseline_70R", baseline_70r_metrics),
        "",
        "## Snapshots",
        "",
        "| snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS]:
        lines.append(row(tag, summary["snapshots"][tag]["eval"]["selected_metrics"]))

    lines.extend(
        [
            "",
            "## Delta vs replace input",
            "",
            "| snapshot | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS]:
        lines.append(row(tag, summary["snapshots"][tag]["delta_vs_replace"]))

    lines.extend(
        [
            "",
            "## Delta vs docs baseline 70R",
            "",
            "| snapshot | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS]:
        lines.append(row(tag, summary["snapshots"][tag]["delta_vs_docs_baseline_70R"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    if not SOURCE_SUMMARY_JSON.is_file():
        raise SystemExit(f"missing source summary: {SOURCE_SUMMARY_JSON}")
    source_summary = load_json(SOURCE_SUMMARY_JSON)
    candidate_70r = source_summary["candidate_70R"]
    ckpt_last = Path(str(candidate_70r["ckpt"]))
    model_dir = ckpt_last.parent
    run_name = ckpt_last.name[len("ckpt_last_") : -len(".pth")]
    lane_log = OUT_ROOT / "lane.log"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    replace_metrics = source_summary["references"]["candidate_replace"]["eval"]["selected_metrics"]
    docs_70r_metrics = source_summary["references"]["docs_baseline_70R"]["eval"]["selected_metrics"]

    snapshots: Dict[str, Any] = {}
    for step in SNAPSHOT_STEPS:
        tag = STEP_TAGS[step]
        ckpt = model_dir / f"ckpt_step_{step:06d}_{run_name}.pth"
        if not ckpt.is_file():
            raise SystemExit(f"missing snapshot ckpt: {ckpt}")
        log(f"=== eval {tag} ===")
        eval_dir = OUT_ROOT / "eval_model" / tag
        group_json = OUT_ROOT / "eval_model" / f"{tag}_group_summary.json"
        eval_json = run_eval(
            model_ckpt=ckpt,
            out_dir=eval_dir,
            contacts_source="model",
            log_file=lane_log,
        )
        ensure_group_summary(eval_json, group_json, log_file=lane_log)
        eval_payload = collect_eval(eval_json, group_json)
        snapshots[tag] = {
            "step": int(step),
            "ckpt": str(ckpt),
            "eval": eval_payload,
            "delta_vs_replace": metric_delta(eval_payload["selected_metrics"], replace_metrics),
            "delta_vs_docs_baseline_70R": metric_delta(eval_payload["selected_metrics"], docs_70r_metrics),
        }

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "compare_contract": "model_source",
            "question": "Does candidate 70R drift immediately or only after later steps?",
        },
        "references": {
            "candidate_replace": source_summary["references"]["candidate_replace"]["eval"],
            "docs_baseline_70R": source_summary["references"]["docs_baseline_70R"]["eval"],
            "candidate_70R_final": source_summary["candidate_70R"]["eval"],
        },
        "snapshots": snapshots,
        "answers": {
            "earliest_step_nonleg_worse_than_replace": next(
                (tag for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS] if safe_float(snapshots[tag]["delta_vs_replace"]["nonleg"]) > 0.0),
                None,
            ),
            "earliest_step_arm_worse_than_replace": next(
                (tag for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS] if safe_float(snapshots[tag]["delta_vs_replace"]["arm"]) > 0.0),
                None,
            ),
            "earliest_step_all_ex_root_worse_than_replace": next(
                (tag for tag in [STEP_TAGS[step] for step in SNAPSHOT_STEPS] if safe_float(snapshots[tag]["delta_vs_replace"]["all_ex_root"]) > 0.0),
                None,
            ),
            "best_snapshot_by_all_ex_root": min(
                [STEP_TAGS[step] for step in SNAPSHOT_STEPS],
                key=lambda tag: safe_float(snapshots[tag]["eval"]["selected_metrics"]["all_ex_root"]),
            ),
            "best_snapshot_by_nonleg": min(
                [STEP_TAGS[step] for step in SNAPSHOT_STEPS],
                key=lambda tag: safe_float(snapshots[tag]["eval"]["selected_metrics"]["nonleg"]),
            ),
        },
    }

    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
