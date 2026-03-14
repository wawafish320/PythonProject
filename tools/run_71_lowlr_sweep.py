#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        run_cmd,
        run_eval,
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
        run_cmd,
        run_eval,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260314"
ATTRIBUTION_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_71_regression_attribution_20260314" / "summary.json"
CANDIDATE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_skip70b_lowdrift_to71_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_71_lowlr_sweep_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_71_lowlr_sweep_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 5, 10, 20, 40, 60, 120, 180)
CASES: Tuple[Tuple[str, float], ...] = (
    ("lr5e4", 5e-4),
    ("lr3e4", 3e-4),
)
SELECTED_METRICS: Tuple[str, ...] = (
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
        "legs_main": safe_float(window.get("overall", {}).get("legs_main")),
        "arms_main": safe_float(window.get("overall", {}).get("arms_main")),
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


def replay_case(*, case_name: str, lr: float, ckpt_in: Path, log_file: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_71_{case_name}_from_candidate70R_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_71_{case_name}_{RUN_DATE}.json"
    make_generated_config(
        CONFIG_71,
        cfg_json,
        {
            "ckpt_in": str(ckpt_in),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": float(lr),
            "save_step_ckpts": ",".join(str(x) for x in SNAPSHOT_STEPS),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    required_ckpts = [out_dir / f"ckpt_step_{step:06d}_{run_name}.pth" for step in SNAPSHOT_STEPS]
    last_ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    if not (last_ckpt.is_file() and all(path.is_file() for path in required_ckpts)):
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_json),
            ],
            log_file=log_file,
        )
    snapshots: Dict[str, Any] = {}
    eval_root = OUT_ROOT / "eval_model" / case_name
    for step in SNAPSHOT_STEPS:
        tag = f"s{step:03d}"
        ckpt = out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"
        eval_dir = eval_root / tag
        group_json = eval_root / f"{tag}_group_summary.json"
        eval_json = run_eval(
            model_ckpt=ckpt,
            out_dir=eval_dir,
            contacts_source="model",
            log_file=log_file,
        )
        ensure_group_summary(eval_json, group_json, log_file=log_file)
        snapshots[tag] = {
            "step": int(step),
            "ckpt": str(ckpt),
            "eval": collect_eval(eval_json, group_json),
        }
    return {
        "lr": float(lr),
        "config": str(cfg_json),
        "run_name": run_name,
        "last_ckpt": str(last_ckpt),
        "snapshots": snapshots,
    }


def extract_reference_snapshot_metrics(summary: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tag, payload in summary["replay"]["candidate"]["snapshots"].items():
        if not isinstance(payload, Mapping):
            continue
        out[str(tag)] = dict(payload["eval"]["selected_metrics"])
    return out


def extract_current_snapshot_metrics(summary: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tag, payload in summary["replay"]["current"]["snapshots"].items():
        if not isinstance(payload, Mapping):
            continue
        out[str(tag)] = dict(payload["eval"]["selected_metrics"])
    return out


def shared_snapshot_deltas(case_snapshots: Mapping[str, Any], ref_snapshots: Mapping[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tag, ref in ref_snapshots.items():
        if tag not in case_snapshots:
            continue
        cur = case_snapshots[tag]["eval"]["selected_metrics"]
        out[str(tag)] = metric_delta(cur, ref)
    return out


def choose_best_snapshot(case_snapshots: Mapping[str, Any], metric_key: str) -> Tuple[str, Dict[str, float]]:
    rows = [(tag, payload["eval"]["selected_metrics"]) for tag, payload in case_snapshots.items()]
    return min(rows, key=lambda item: safe_float(item[1][metric_key]))


def build_markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["references"]["candidate_baseline_snapshots"]
    current = summary["references"]["current_snapshots"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | {fmt(metrics['nonleg'])} | "
            f"{fmt(metrics['arm'])} | {fmt(metrics['legs_main'])} | {fmt(metrics['arms_main'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# 71 lower-LR sweep",
        "",
        "- start point is candidate `70R` from the lowdrift replace lane",
        "- `71` semantics are unchanged; only `lr` is changed",
        "- eval contract is model-source only",
        "",
        "## Reference rows",
        "",
        "| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("current_s180", current["s180"]),
        row("baseline_candidate_s000", baseline["s000"]),
        row("baseline_candidate_s020", baseline["s020"]),
        row("baseline_candidate_s060", baseline["s060"]),
        row("baseline_candidate_s120", baseline["s120"]),
        row("baseline_candidate_s180", baseline["s180"]),
        "",
        "## Sweep snapshots",
        "",
        "| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, case_payload in summary["cases"].items():
        for tag in sorted(case_payload["snapshots"], key=lambda x: int(x[1:])):
            metrics = case_payload["snapshots"][tag]["eval"]["selected_metrics"]
            lines.append(row(f"{case_name}_{tag}", metrics))
        lines.append("")

    lines.extend(
        [
            "## Best snapshots",
            "",
            "| case | best_all_ex_root_snapshot | value | best_leg_snapshot | value |",
            "|---|---|---:|---|---:|",
        ]
    )
    for case_name, case_payload in summary["cases"].items():
        best_all = case_payload["best_snapshot"]["all_ex_root"]
        best_leg = case_payload["best_snapshot"]["leg"]
        lines.append(
            f"| {case_name} | {best_all['tag']} | {fmt(best_all['value'])} | {best_leg['tag']} | {fmt(best_leg['value'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [ATTRIBUTION_SUMMARY_JSON, CANDIDATE_SUMMARY_JSON, CONFIG_71, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    attribution_summary = load_json(ATTRIBUTION_SUMMARY_JSON)
    candidate_summary = load_json(CANDIDATE_SUMMARY_JSON)
    candidate_70r_ckpt = Path(str(candidate_summary["candidate"]["ckpt_70R"]))
    if not candidate_70r_ckpt.is_file():
        raise RuntimeError(f"missing candidate 70R ckpt: {candidate_70r_ckpt}")

    baseline_snapshots = extract_reference_snapshot_metrics(attribution_summary)
    current_snapshots = extract_current_snapshot_metrics(attribution_summary)
    current_final = current_snapshots["s180"]

    cases: Dict[str, Any] = {}
    for case_name, lr in CASES:
        log(f"=== replay {case_name} lr={lr:.6g} ===")
        case_payload = replay_case(case_name=case_name, lr=lr, ckpt_in=candidate_70r_ckpt, log_file=lane_log)
        case_snapshots = case_payload["snapshots"]
        case_payload["delta_vs_current_final"] = {
            tag: metric_delta(payload["eval"]["selected_metrics"], current_final)
            for tag, payload in case_snapshots.items()
        }
        case_payload["delta_vs_baseline_candidate_shared"] = shared_snapshot_deltas(case_snapshots, baseline_snapshots)
        best_all_tag, best_all_metrics = choose_best_snapshot(case_snapshots, "all_ex_root")
        best_leg_tag, best_leg_metrics = choose_best_snapshot(case_snapshots, "leg")
        case_payload["best_snapshot"] = {
            "all_ex_root": {"tag": best_all_tag, "value": safe_float(best_all_metrics["all_ex_root"])},
            "leg": {"tag": best_leg_tag, "value": safe_float(best_leg_metrics["leg"])},
        }
        cases[case_name] = case_payload

    recommendation_rows: List[Tuple[float, str]] = []
    for case_name, case_payload in cases.items():
        best_all = case_payload["best_snapshot"]["all_ex_root"]["value"]
        recommendation_rows.append((safe_float(best_all), case_name))
    recommendation_rows.sort()
    best_case = recommendation_rows[0][1]

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "compare_contract": "model_source",
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "sweep": [{"name": name, "lr": lr} for name, lr in CASES],
            "note": "candidate 70R handoff only; unchanged 71 semantics, lower-LR sweep",
        },
        "references": {
            "attribution_summary": str(ATTRIBUTION_SUMMARY_JSON),
            "candidate_summary": str(CANDIDATE_SUMMARY_JSON),
            "candidate_70R_ckpt": str(candidate_70r_ckpt),
            "current_snapshots": current_snapshots,
            "candidate_baseline_snapshots": baseline_snapshots,
        },
        "cases": cases,
        "answers": {
            "best_case_by_all_ex_root": best_case,
            "best_case_beats_current_final": safe_float(cases[best_case]["best_snapshot"]["all_ex_root"]["value"]) < safe_float(current_final["all_ex_root"]),
            "best_case_beats_baseline_candidate_final": safe_float(cases[best_case]["best_snapshot"]["all_ex_root"]["value"]) < safe_float(baseline_snapshots["s180"]["all_ex_root"]),
            "best_case_earliest_shared_improvement_vs_baseline": next(
                (
                    tag
                    for tag, delta_row in sorted(cases[best_case]["delta_vs_baseline_candidate_shared"].items(), key=lambda item: int(item[0][1:]))
                    if safe_float(delta_row.get("all_ex_root")) < 0.0 and safe_float(delta_row.get("leg")) < 0.0
                ),
                None,
            ),
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
