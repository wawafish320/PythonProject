#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_72,
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
        CONFIG_72,
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
ATTRIBUTION_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_72_loss_curve_attribution_20260314" / "summary.json"
LOWLR71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_72_lowlr_sweep_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_72_lowlr_sweep_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 5, 10, 20, 40, 60, 120, 180)
CASES: Tuple[Tuple[str, float], ...] = (
    ("lr5e4", 5e-4),
    ("lr3e4", 3e-4),
    ("lr1e4", 1e-4),
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
LOSS_KEYS: Tuple[str, ...] = (
    "total",
    "dir_geo",
    "leg_align_weighted",
    "dir_group_norm_leg",
    "dir_leg_base",
    "dir_nonleg_base",
    "boundary_dir_geo",
)
WINDOWS: Dict[str, Tuple[int, int]] = {
    "start20": (0, 20),
    "mid20": (80, 100),
    "late20": (160, 180),
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


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


def summarize_log(log_json: Path) -> Dict[str, Any]:
    obj = load_json(log_json)
    rows = obj["log"]
    out: Dict[str, Any] = {}
    for key in LOSS_KEYS:
        vals = [safe_float(row.get(key)) for row in rows]
        if not any(math.isfinite(v) for v in vals):
            continue
        out[key] = {
            "windows": {name: mean(vals[start:end]) for name, (start, end) in WINDOWS.items()},
            "epoch_means": {
                f"epoch{epoch}": mean(
                    safe_float(row.get(key))
                    for row in rows
                    if int(safe_float(row.get("epoch", 0)) or 0) == epoch
                )
                for epoch in (1, 2, 3)
            },
            "peak_first20": max((v for v in vals[:20] if math.isfinite(v)), default=float("nan")),
        }
    return out


def replay_case(*, case_name: str, lr: float, ckpt_in: Path, log_file: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_72_{case_name}_from_lowlr71_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_72_{case_name}_{RUN_DATE}.json"
    make_generated_config(
        CONFIG_72,
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
    log_json = out_dir / f"posttrain_log_{run_name}.json"
    return {
        "lr": float(lr),
        "config": str(cfg_json),
        "run_name": run_name,
        "last_ckpt": str(last_ckpt),
        "log_json": str(log_json),
        "log_summary": summarize_log(log_json),
        "snapshots": snapshots,
    }


def choose_best_snapshot(case_snapshots: Mapping[str, Any], metric_key: str) -> Tuple[str, Dict[str, float]]:
    rows = [(tag, payload["eval"]["selected_metrics"]) for tag, payload in case_snapshots.items()]
    return min(rows, key=lambda item: safe_float(item[1][metric_key]))


def choose_best_post_start_snapshot(case_snapshots: Mapping[str, Any], metric_key: str) -> Tuple[str, Dict[str, float]]:
    rows = [(tag, payload["eval"]["selected_metrics"]) for tag, payload in case_snapshots.items() if tag != "s000"]
    return min(rows, key=lambda item: safe_float(item[1][metric_key]))


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]
    cases = summary["cases"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | {fmt(metrics['nonleg'])} | "
            f"{fmt(metrics['arm'])} | {fmt(metrics['legs_main'])} | {fmt(metrics['arms_main'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# 72 lower-LR sweep",
        "",
        "- start point is candidate `71 (lr=3e-4)`",
        "- `72` semantics are unchanged; only `lr` is changed",
        "- eval contract is model-source only",
        "",
        "## Reference rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("current_72", refs["current_72"]),
        row("baseline_candidate_72", refs["baseline_candidate_72"]),
    ]
    for case_name, payload in cases.items():
        lines.append(row(case_name, payload["snapshots"]["s180"]["eval"]["selected_metrics"]))
    lines.extend(
        [
            "",
            "## Early snapshots",
            "",
            "| lane_snapshot | all_ex_root | leg | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|",
            f"| baseline_candidate_s000 | {fmt(refs['candidate_baseline_snapshots']['s000']['all_ex_root'])} | {fmt(refs['candidate_baseline_snapshots']['s000']['leg'])} | {fmt(refs['candidate_baseline_snapshots']['s000']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_baseline_snapshots']['s000']['calf_r_SIC2_4'])} |",
            f"| baseline_candidate_s005 | {fmt(refs['candidate_baseline_snapshots']['s005']['all_ex_root'])} | {fmt(refs['candidate_baseline_snapshots']['s005']['leg'])} | {fmt(refs['candidate_baseline_snapshots']['s005']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_baseline_snapshots']['s005']['calf_r_SIC2_4'])} |",
            f"| baseline_candidate_s010 | {fmt(refs['candidate_baseline_snapshots']['s010']['all_ex_root'])} | {fmt(refs['candidate_baseline_snapshots']['s010']['leg'])} | {fmt(refs['candidate_baseline_snapshots']['s010']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_baseline_snapshots']['s010']['calf_r_SIC2_4'])} |",
            f"| baseline_candidate_s020 | {fmt(refs['candidate_baseline_snapshots']['s020']['all_ex_root'])} | {fmt(refs['candidate_baseline_snapshots']['s020']['leg'])} | {fmt(refs['candidate_baseline_snapshots']['s020']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_baseline_snapshots']['s020']['calf_r_SIC2_4'])} |",
        ]
    )
    for case_name, payload in cases.items():
        for tag in ("s005", "s010", "s020", "s120", "s180"):
            metrics = payload["snapshots"][tag]["eval"]["selected_metrics"]
            lines.append(
                f"| {case_name}_{tag} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
            )
    lines.extend(
        [
            "",
            "## Start20 loss summary",
            "",
            "| case | total | dir_geo | leg_align_weighted | dir_group_norm_leg | dir_leg_base | dir_nonleg_base | boundary_dir_geo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| baseline_candidate_72 (lr=1e-3) | {fmt(refs['baseline_candidate_72_loss']['total']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['dir_geo']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['leg_align_weighted']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['dir_group_norm_leg']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['dir_leg_base']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['dir_nonleg_base']['windows']['start20'])} | {fmt(refs['baseline_candidate_72_loss']['boundary_dir_geo']['windows']['start20'])} |",
        ]
    )
    for case_name, payload in cases.items():
        ls = payload["log_summary"]
        lines.append(
            f"| {case_name} | {fmt(ls['total']['windows']['start20'])} | {fmt(ls['dir_geo']['windows']['start20'])} | {fmt(ls['leg_align_weighted']['windows']['start20'])} | {fmt(ls['dir_group_norm_leg']['windows']['start20'])} | {fmt(ls['dir_leg_base']['windows']['start20'])} | {fmt(ls['dir_nonleg_base']['windows']['start20'])} | {fmt(ls['boundary_dir_geo']['windows']['start20'])} |"
        )
    lines.extend(
        [
            "",
            "## Best snapshots",
            "",
            "| case | best_all_snapshot | all_ex_root | best_leg_snapshot | leg | best_post_start_all | all_ex_root | best_post_start_leg | leg |",
            "|---|---|---:|---|---:|---|---:|---|---:|",
        ]
    )
    for case_name, payload in cases.items():
        best_all = payload["best_snapshot"]["all_ex_root"]
        best_leg = payload["best_snapshot"]["leg"]
        best_all_ps = payload["best_post_start_snapshot"]["all_ex_root"]
        best_leg_ps = payload["best_post_start_snapshot"]["leg"]
        lines.append(
            f"| {case_name} | {best_all['tag']} | {fmt(best_all['value'])} | {best_leg['tag']} | {fmt(best_leg['value'])} | {best_all_ps['tag']} | {fmt(best_all_ps['value'])} | {best_leg_ps['tag']} | {fmt(best_leg_ps['value'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [ATTRIBUTION_SUMMARY_JSON, LOWLR71_SUMMARY_JSON, CONFIG_72, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    attribution_summary = load_json(ATTRIBUTION_SUMMARY_JSON)
    lowlr71_summary = load_json(LOWLR71_SUMMARY_JSON)
    candidate_71_ckpt = Path(str(lowlr71_summary["cases"]["lr3e4"]["last_ckpt"]))
    if not candidate_71_ckpt.is_file():
        raise RuntimeError(f"missing candidate 71 ckpt: {candidate_71_ckpt}")

    current_72 = dict(attribution_summary["reference_metrics"]["current_72"])
    baseline_candidate_72 = dict(attribution_summary["reference_metrics"]["candidate_72"])
    candidate_baseline_snapshots = {
        tag: dict(payload["eval"]["selected_metrics"])
        for tag, payload in attribution_summary["replay"]["candidate"]["snapshots"].items()
    }
    current_snapshots = {
        tag: dict(payload["eval"]["selected_metrics"])
        for tag, payload in attribution_summary["replay"]["current"]["snapshots"].items()
    }
    baseline_candidate_72_loss = attribution_summary["loss_curves"]["candidate"]["series"]

    cases: Dict[str, Any] = {}
    for case_name, lr in CASES:
        log(f"=== replay {case_name} lr={lr:.6g} ===")
        case_payload = replay_case(case_name=case_name, lr=lr, ckpt_in=candidate_71_ckpt, log_file=lane_log)
        case_snapshots = case_payload["snapshots"]
        case_payload["delta_vs_current_final"] = {
            tag: metric_delta(payload["eval"]["selected_metrics"], current_72)
            for tag, payload in case_snapshots.items()
        }
        case_payload["delta_vs_baseline_candidate_shared"] = {
            tag: metric_delta(payload["eval"]["selected_metrics"], candidate_baseline_snapshots[tag])
            for tag, payload in case_snapshots.items()
        }
        best_all_tag, best_all_metrics = choose_best_snapshot(case_snapshots, "all_ex_root")
        best_leg_tag, best_leg_metrics = choose_best_snapshot(case_snapshots, "leg")
        best_all_ps_tag, best_all_ps_metrics = choose_best_post_start_snapshot(case_snapshots, "all_ex_root")
        best_leg_ps_tag, best_leg_ps_metrics = choose_best_post_start_snapshot(case_snapshots, "leg")
        case_payload["best_snapshot"] = {
            "all_ex_root": {"tag": best_all_tag, "value": safe_float(best_all_metrics["all_ex_root"])},
            "leg": {"tag": best_leg_tag, "value": safe_float(best_leg_metrics["leg"])},
        }
        case_payload["best_post_start_snapshot"] = {
            "all_ex_root": {"tag": best_all_ps_tag, "value": safe_float(best_all_ps_metrics["all_ex_root"])},
            "leg": {"tag": best_leg_ps_tag, "value": safe_float(best_leg_ps_metrics["leg"])},
        }
        cases[case_name] = case_payload

    best_final_all_case = min(cases.items(), key=lambda item: safe_float(item[1]["snapshots"]["s180"]["eval"]["selected_metrics"]["all_ex_root"]))[0]
    best_final_leg_case = min(cases.items(), key=lambda item: safe_float(item[1]["snapshots"]["s180"]["eval"]["selected_metrics"]["leg"]))[0]
    best_post_start_all_case = min(cases.items(), key=lambda item: safe_float(item[1]["best_post_start_snapshot"]["all_ex_root"]["value"]))[0]
    best_post_start_leg_case = min(cases.items(), key=lambda item: safe_float(item[1]["best_post_start_snapshot"]["leg"]["value"]))[0]

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "compare_contract": "model_source",
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "sweep": [{"name": name, "lr": lr} for name, lr in CASES],
            "note": "candidate 71(lr=3e-4) handoff only; unchanged 72 semantics, lower-LR sweep",
        },
        "references": {
            "attribution_summary": str(ATTRIBUTION_SUMMARY_JSON),
            "lowlr71_summary": str(LOWLR71_SUMMARY_JSON),
            "candidate_71_ckpt": str(candidate_71_ckpt),
            "current_72": current_72,
            "baseline_candidate_72": baseline_candidate_72,
            "current_snapshots": current_snapshots,
            "candidate_baseline_snapshots": candidate_baseline_snapshots,
            "baseline_candidate_72_loss": baseline_candidate_72_loss,
        },
        "cases": cases,
        "answers": {
            "best_final_all_case": best_final_all_case,
            "best_final_leg_case": best_final_leg_case,
            "best_post_start_all_case": best_post_start_all_case,
            "best_post_start_leg_case": best_post_start_leg_case,
            "best_final_all_beats_current72": safe_float(cases[best_final_all_case]["snapshots"]["s180"]["eval"]["selected_metrics"]["all_ex_root"]) < safe_float(current_72["all_ex_root"]),
            "best_final_leg_beats_current72": safe_float(cases[best_final_leg_case]["snapshots"]["s180"]["eval"]["selected_metrics"]["leg"]) < safe_float(current_72["leg"]),
            "best_post_start_all_beats_current72": safe_float(cases[best_post_start_all_case]["best_post_start_snapshot"]["all_ex_root"]["value"]) < safe_float(current_72["all_ex_root"]),
            "best_post_start_leg_beats_current72": safe_float(cases[best_post_start_leg_case]["best_post_start_snapshot"]["leg"]["value"]) < safe_float(current_72["leg"]),
            "earliest_joint_improvement_vs_baseline_by_case": {
                case_name: next(
                    (
                        tag
                        for tag, delta_row in sorted(case_payload["delta_vs_baseline_candidate_shared"].items(), key=lambda item: int(item[0][1:]))
                        if safe_float(delta_row.get("all_ex_root")) < 0.0 and safe_float(delta_row.get("leg")) < 0.0
                    ),
                    None,
                )
                for case_name, case_payload in cases.items()
            },
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
