#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70B,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
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
        CONFIG_70B,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
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


RUN_DATE = "20260314"
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_newflow_chain_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_oldd1_skip70b_lowdrift_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_oldd1_skip70b_lowdrift_{RUN_DATE}"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_source_summary() -> Dict[str, Any]:
    return load_json(SOURCE_SUMMARY_JSON)


def extract_stage(summary: Mapping[str, Any], stage_name: str) -> Dict[str, Any]:
    stages = summary.get("stage_progress_model_source", {})
    if not isinstance(stages, Mapping) or stage_name not in stages:
        raise RuntimeError(f"missing stage_progress_model_source[{stage_name}] in {SOURCE_SUMMARY_JSON}")
    stage = stages[stage_name]
    if not isinstance(stage, dict):
        raise RuntimeError(f"invalid stage payload for {stage_name}")
    return dict(stage)


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


def build_summary(
    *,
    source_summary: Mapping[str, Any],
    candidate_eval: Mapping[str, Any],
    candidate_ckpt: Path,
    candidate_cfg: Path,
    warmstart_ckpt: Path,
    warmstart_report: Path,
) -> Dict[str, Any]:
    keys_direct = ("all_ex_root", "leg", "nonleg", "arm", "else")
    keys_masked = ("DirectGeoLocalDeg", "BlendGeoLocalDeg", "GeoLocalDeg")

    stage70a = extract_stage(source_summary, "70a")
    current_replace = extract_stage(source_summary, "new70b_replace")

    delta_70a_to_current = {
        "masked_means_delta": delta_block(current_replace["masked_means"], stage70a["masked_means"], keys_masked),
        "direct_group_delta": delta_block(current_replace["direct_group_summary"], stage70a["direct_group_summary"], keys_direct),
    }
    delta_70a_to_candidate = {
        "masked_means_delta": delta_block(candidate_eval["masked_means"], stage70a["masked_means"], keys_masked),
        "direct_group_delta": delta_block(candidate_eval["direct_group_summary"], stage70a["direct_group_summary"], keys_direct),
    }
    delta_candidate_to_current = {
        "masked_means_delta": delta_block(candidate_eval["masked_means"], current_replace["masked_means"], keys_masked),
        "direct_group_delta": delta_block(candidate_eval["direct_group_summary"], current_replace["direct_group_summary"], keys_direct),
    }

    current_leg_gain = safe_float(stage70a["direct_group_summary"]["leg"]) - safe_float(current_replace["direct_group_summary"]["leg"])
    candidate_leg_gain = safe_float(stage70a["direct_group_summary"]["leg"]) - safe_float(candidate_eval["direct_group_summary"]["leg"])

    return {
        "run_date": RUN_DATE,
        "source_summary": str(SOURCE_SUMMARY_JSON),
        "policy": {
            "base_config": str(CONFIG_70B),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "compare_contract": "model_source",
        },
        "control": {
            "stage70a": stage70a,
            "current_new70b_replace": current_replace,
        },
        "candidate": {
            "name": "new70b_replace_lowdrift",
            "config": str(candidate_cfg),
            "warmstart_ckpt": str(warmstart_ckpt),
            "warmstart_report": str(warmstart_report),
            "ckpt": str(candidate_ckpt),
            "eval": candidate_eval,
        },
        "comparisons": {
            "stage70a_to_current_new70b_replace": delta_70a_to_current,
            "stage70a_to_candidate_new70b_replace_lowdrift": delta_70a_to_candidate,
            "candidate_new70b_replace_lowdrift_to_current_new70b_replace": delta_candidate_to_current,
        },
        "answers": {
            "candidate_reduces_drift_vs_current": {
                "all_ex_root": safe_float(delta_candidate_to_current["direct_group_delta"]["all_ex_root"]) < 0.0,
                "nonleg": safe_float(delta_candidate_to_current["direct_group_delta"]["nonleg"]) < 0.0,
                "arm": safe_float(delta_candidate_to_current["direct_group_delta"]["arm"]) < 0.0,
            },
            "candidate_keeps_leg_gain_vs_70a": {
                "current_leg_gain": current_leg_gain,
                "candidate_leg_gain": candidate_leg_gain,
                "retains_at_least_half": candidate_leg_gain >= 0.5 * current_leg_gain,
            },
        },
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    stage70a = summary["control"]["stage70a"]
    current_replace = summary["control"]["current_new70b_replace"]
    candidate = summary["candidate"]["eval"]
    cmp_70a_current = summary["comparisons"]["stage70a_to_current_new70b_replace"]
    cmp_70a_candidate = summary["comparisons"]["stage70a_to_candidate_new70b_replace_lowdrift"]
    cmp_candidate_current = summary["comparisons"]["candidate_new70b_replace_lowdrift_to_current_new70b_replace"]
    ans = summary["answers"]

    lines = [
        "# old d1 skip-raw70b replace compare",
        "",
        f"- source_summary: `{summary['source_summary']}`",
        f"- candidate_config: `{summary['candidate']['config']}`",
        f"- candidate_ckpt: `{summary['candidate']['ckpt']}`",
        "",
        "## Direct-path metrics (model-source)",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 70a | {fmt(stage70a['masked_means']['DirectGeoLocalDeg'])} | {fmt(stage70a['direct_group_summary']['all_ex_root'])} | {fmt(stage70a['direct_group_summary']['leg'])} | {fmt(stage70a['direct_group_summary']['nonleg'])} | {fmt(stage70a['direct_group_summary']['arm'])} | {fmt(stage70a['direct_group_summary']['else'])} |",
        f"| current_new70b_replace | {fmt(current_replace['masked_means']['DirectGeoLocalDeg'])} | {fmt(current_replace['direct_group_summary']['all_ex_root'])} | {fmt(current_replace['direct_group_summary']['leg'])} | {fmt(current_replace['direct_group_summary']['nonleg'])} | {fmt(current_replace['direct_group_summary']['arm'])} | {fmt(current_replace['direct_group_summary']['else'])} |",
        f"| candidate_new70b_replace_lowdrift | {fmt(candidate['masked_means']['DirectGeoLocalDeg'])} | {fmt(candidate['direct_group_summary']['all_ex_root'])} | {fmt(candidate['direct_group_summary']['leg'])} | {fmt(candidate['direct_group_summary']['nonleg'])} | {fmt(candidate['direct_group_summary']['arm'])} | {fmt(candidate['direct_group_summary']['else'])} |",
        "",
        "## Deltas",
        "",
        "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 70a -> current_new70b_replace | {fmt(cmp_70a_current['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(cmp_70a_current['direct_group_delta']['all_ex_root'])} | {fmt(cmp_70a_current['direct_group_delta']['leg'])} | {fmt(cmp_70a_current['direct_group_delta']['nonleg'])} | {fmt(cmp_70a_current['direct_group_delta']['arm'])} | {fmt(cmp_70a_current['direct_group_delta']['else'])} |",
        f"| 70a -> candidate_new70b_replace_lowdrift | {fmt(cmp_70a_candidate['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(cmp_70a_candidate['direct_group_delta']['all_ex_root'])} | {fmt(cmp_70a_candidate['direct_group_delta']['leg'])} | {fmt(cmp_70a_candidate['direct_group_delta']['nonleg'])} | {fmt(cmp_70a_candidate['direct_group_delta']['arm'])} | {fmt(cmp_70a_candidate['direct_group_delta']['else'])} |",
        f"| candidate - current_new70b_replace | {fmt(cmp_candidate_current['masked_means_delta']['DirectGeoLocalDeg'])} | {fmt(cmp_candidate_current['direct_group_delta']['all_ex_root'])} | {fmt(cmp_candidate_current['direct_group_delta']['leg'])} | {fmt(cmp_candidate_current['direct_group_delta']['nonleg'])} | {fmt(cmp_candidate_current['direct_group_delta']['arm'])} | {fmt(cmp_candidate_current['direct_group_delta']['else'])} |",
        "",
        "## Decision hooks",
        "",
        f"- candidate reduces drift vs current: all_ex_root=`{str(bool(ans['candidate_reduces_drift_vs_current']['all_ex_root'])).lower()}`, nonleg=`{str(bool(ans['candidate_reduces_drift_vs_current']['nonleg'])).lower()}`, arm=`{str(bool(ans['candidate_reduces_drift_vs_current']['arm'])).lower()}`",
        f"- current leg gain vs 70a: `{fmt(ans['candidate_keeps_leg_gain_vs_70a']['current_leg_gain'])}`",
        f"- candidate leg gain vs 70a: `{fmt(ans['candidate_keeps_leg_gain_vs_70a']['candidate_leg_gain'])}`",
        f"- candidate keeps at least half the current leg gain: `{str(bool(ans['candidate_keeps_leg_gain_vs_70a']['retains_at_least_half'])).lower()}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [SOURCE_SUMMARY_JSON, CONFIG_70B, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    source_summary = load_source_summary()
    stage70a_ckpt = Path(str(source_summary["checkpoints"]["70a"]))
    if not stage70a_ckpt.is_file():
        raise SystemExit(f"missing stage70a ckpt: {stage70a_ckpt}")

    warmstart_ckpt = MODEL_ROOT / "warmstart" / f"ckpt_last_oldd1_skip70b_zerophase_{RUN_DATE}.pth"
    warmstart_report = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"
    cfg_json = OUT_ROOT / "configs" / f"posttrain_70b_replace_lowdrift_from_oldd1_{RUN_DATE}.json"
    lane_log = OUT_ROOT / "lane.log"
    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    status_json = OUT_ROOT / "status.json"

    run_name = f"WalkF_stage7_70b_replace_lowdrift_from_oldd1_{RUN_DATE}"
    model_dir = MODEL_ROOT / "70b_replace_lowdrift"
    candidate_ckpt = model_dir / f"ckpt_last_{run_name}.pth"
    eval_dir = OUT_ROOT / "eval_model_source"
    eval_group = OUT_ROOT / "eval_model_source_group_summary.json"

    log("=== build warmstart from 70a ===")
    create_replace_zerophase_warmstart(
        src_ckpt=stage70a_ckpt,
        dst_ckpt=warmstart_ckpt,
        report_json=warmstart_report,
    )

    log("=== candidate new70b_replace_lowdrift ===")
    cfg_json.parent.mkdir(parents=True, exist_ok=True)
    cfg_json = make_generated_config(
        CONFIG_70B,
        cfg_json,
        {
            "ckpt_in": str(warmstart_ckpt),
            "out_dir": str(model_dir),
            "run_name": run_name,
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
            "lr": 3e-4,
            "epochs": 1,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    candidate_ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=warmstart_ckpt,
        out_dir=model_dir,
        run_name=run_name,
        log_file=lane_log,
    )

    log("=== model-source eval ===")
    eval_json = run_eval(
        model_ckpt=candidate_ckpt,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_json, eval_group, log_file=lane_log)
    candidate_eval = collect_eval(eval_json, eval_group)

    status_payload = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage70a_ckpt": str(stage70a_ckpt),
        "warmstart_ckpt": str(warmstart_ckpt),
        "candidate_config": str(cfg_json),
        "candidate_ckpt": str(candidate_ckpt),
        "candidate_eval_json": str(eval_json),
    }
    write_json(status_json, status_payload)

    summary = build_summary(
        source_summary=source_summary,
        candidate_eval=candidate_eval,
        candidate_ckpt=candidate_ckpt,
        candidate_cfg=cfg_json,
        warmstart_ckpt=warmstart_ckpt,
        warmstart_report=warmstart_report,
    )
    write_json(summary_json, summary)
    summary_md.write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
