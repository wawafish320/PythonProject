#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain
from train.posttrain_build_shell import _build_posttrain_model_from_ckpt
from train.runtime.freeze import _freeze_all, _select_trainable_params, _unfreeze_for_train_mode


RUN_DATE = "20260328"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_per_head_merged_stage_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_per_head_merged_stage_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"

FIXED_BASETRAIN_CONFIG = ROOT / "config" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324.json"
FIXED_STAGE6_WINNER = ROOT / "models" / "__tmp_ep014center_main_anchor_check" / "tmp_ep014center_main_selector_check__last" / "ckpt_last_tmp_ep014center_main_selector_check__last_stage6_anchor_ep014center_main_check.pth"
FIXED_70A_WINNER = ROOT / "models" / "__tmp_ep014center_70a_lowlr_sweep_20260328" / "lr3e4" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth"
CURRENT_REPLACE_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lowlr_sweep_20260328" / "lr5e5" / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_ep014center_70alr3e4_20260328.pth"
CURRENT_SKIP71_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "71" / "ckpt_last_WalkF_stage7_71_lr3e4_from_ep014center_replace_lr5e5_20260328.pth"

SOURCE_70A_ARTIFACTS_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "artifacts.json"
CURRENT_REPLACE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
CURRENT_SKIP71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "summary.json"
CURRENT_LOWDRIFT_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_70alr3e4_lowdrift_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"

CURRENT_REPLACE_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "configs" / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
CURRENT_SKIP71_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "configs" / "posttrain_71_lr3e4_from_ep014center_replace_lr5e5_20260328.json"
SOURCE_70A_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "configs" / "posttrain_70a_lr3e4_from_ep014center_20260328.json"

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_ep014center_per_head_replace_zerophase_{RUN_DATE}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"

SELECTED_METRICS: Tuple[str, ...] = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)

MAIN_CASES: Tuple[Tuple[str, float], ...] = (
    ("nonleg1e4", 1e-4),
    ("nonleg2e4", 2e-4),
    ("nonleg3e4", 3e-4),
)

INSPECT_MODULE_NAMES: Tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_out_leg",
    "direct_pose_out_nonleg",
    "direct_pose_out_arm",
    "direct_pose_out_else",
    "direct_pose_nonleg_proj",
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_leg_head",
    "direct_pose_leg_gate_head",
    "direct_pose_leg_head_shared",
    "direct_pose_leg_gate_head_shared",
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


def stage_row_to_eval(stage_row: Mapping[str, Any]) -> Dict[str, Any]:
    eval_json = Path(str(stage_row.get("eval_json", "")))
    group_json = Path(str(stage_row.get("group_json", "")))
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


def build_optimizer_group_overrides(nonleg_lr: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "conservative_shared_and_proj",
            "lr": 5e-5,
            "module_prefixes": [
                "direct_pose_head",
                "direct_pose_nonleg_proj",
                "direct_pose_arm_proj",
                "direct_pose_else_proj",
            ],
        },
        {
            "name": "leg_bucket",
            "lr": 5e-5,
            "module_prefixes": [
                "direct_pose_out_leg",
                "direct_pose_leg_head",
                "direct_pose_leg_gate_head",
                "direct_pose_leg_head_shared",
                "direct_pose_leg_gate_head_shared",
            ],
        },
        {
            "name": "nonleg_heads",
            "lr": float(nonleg_lr),
            "module_prefixes": [
                "direct_pose_out_nonleg",
                "direct_pose_out_arm",
                "direct_pose_out_else",
            ],
        },
    ]


def build_dataset_for_inspection() -> tuple[dict[str, Any], Any]:
    payload = load_json(CURRENT_REPLACE_CONFIG_JSON)
    payload["ckpt_in"] = str(FIXED_70A_WINNER)
    cfg = posttrain._cfg_from_payload(payload)
    norm_spec, ds, _ = posttrain._build_dataset_and_loader(cfg)
    return norm_spec, ds


def inspect_model_scope(
    *,
    config_json: Path,
    ckpt_in: Path,
    ds: Any,
) -> Dict[str, Any]:
    payload = load_json(config_json)
    payload["ckpt_in"] = str(ckpt_in)
    cfg = posttrain._cfg_from_payload(payload)
    artifacts = _build_posttrain_model_from_ckpt(
        cfg=cfg,
        ds=ds,
        device=torch.device("cpu"),
    )
    model = artifacts.model
    _freeze_all(model)
    train_mode = posttrain._resolve_train_mode(cfg)
    _unfreeze_for_train_mode(
        model,
        train_mode=train_mode,
        direct_pose_leg_train_only=bool(getattr(cfg, "direct_pose_leg_train_only", False)),
        direct_pose_leg_gate_train_only=bool(getattr(cfg, "direct_pose_leg_gate_train_only", False)),
        direct_pose_nonleg_train_only=bool(getattr(cfg, "direct_pose_nonleg_train_only", False)),
    )
    params, names = _select_trainable_params(model)
    _, group_summaries = posttrain._resolve_optimizer_param_groups(cfg=cfg, params=params, names=names)

    instantiated: dict[str, Any] = {}
    for module_name in INSPECT_MODULE_NAMES:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        instantiated[module_name] = {
            "type": module.__class__.__name__,
            "param_count": int(sum(int(p.numel()) for p in module.parameters())),
        }

    return {
        "config_json": str(config_json),
        "ckpt_in": str(ckpt_in),
        "train_mode": train_mode,
        "resolved": {
            "direct_pose_feat_source": str(artifacts.direct_pose_feat_source),
            "direct_pose_time_pe_dim": int(artifacts.direct_pose_time_pe_dim),
            "direct_pose_time_pe_base": float(artifacts.direct_pose_time_pe_base),
            "direct_pose_split_enable": bool(artifacts.direct_pose_split_enable),
            "direct_pose_nonleg_proj_dim": int(artifacts.direct_pose_nonleg_proj_dim),
            "direct_pose_leg_gate_mode_model": str(artifacts.direct_pose_leg_gate_mode_model),
            "direct_pose_leg_gate_power_model": float(artifacts.direct_pose_leg_gate_power_model),
        },
        "instantiated_modules": instantiated,
        "trainable_param_names": list(names),
        "optimizer_param_groups": group_summaries,
    }


def write_scope_summary(path: Path, payload: Mapping[str, Any]) -> Path:
    write_json(path, payload)
    return path


def load_references() -> Dict[str, Any]:
    source_70a = load_json(SOURCE_70A_ARTIFACTS_JSON)["lr3e4"]
    current_replace = load_json(CURRENT_REPLACE_SUMMARY_JSON)["cases"]["lr5e5"]
    current_skip71 = load_json(CURRENT_SKIP71_SUMMARY_JSON)["candidate_71"]
    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)["stage_progress_model_source"]
    return {
        "source_70a": {
            "config": str(source_70a["config_json"]),
            "ckpt": str(source_70a["ckpt"]),
            "eval": collect_eval(Path(str(source_70a["eval_json"])), Path(str(source_70a["group_json"]))),
        },
        "current_replace_lr5e5": {
            "config": str(current_replace["config"]),
            "ckpt": str(current_replace["last_ckpt"]),
            "eval": current_replace["eval"],
        },
        "current_skip70R_to71_lr3e4": {
            "config": str(current_skip71["config"]),
            "ckpt": str(current_skip71["ckpt"]),
            "eval": current_skip71["eval"],
        },
        "docs_baseline_new70b_replace_lowdrift": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["new70b_replace_lowdrift"]),
        },
        "docs_baseline_70R": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["70R"]),
        },
        "docs_baseline_71_lr3e4": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["71_lr3e4"]),
        },
        "current_lowdrift_replace_direct": {
            "summary": str(CURRENT_LOWDRIFT_SUMMARY_JSON),
            "eval": {
                "selected_metrics": {
                    "DirectGeoLocalDeg": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["masked_means"]["DirectGeoLocalDeg"]),
                    "all_ex_root": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["direct_group_summary"]["all_ex_root"]),
                    "leg": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["direct_group_summary"]["leg"]),
                    "nonleg": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["direct_group_summary"]["nonleg"]),
                    "arm": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["direct_group_summary"]["arm"]),
                    "foot_l_ball_l_SIC12_15": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"]),
                    "calf_r_SIC2_4": safe_float(load_json(CURRENT_LOWDRIFT_SUMMARY_JSON)["window_summary"]["hotspots"]["calf_r_SIC2_4"]),
                }
            },
        },
    }


def run_candidate(
    *,
    case_name: str,
    nonleg_lr: float,
    ds: Any,
    lane_log: Path,
) -> Dict[str, Any]:
    run_name = f"WalkF_stage7_merged_perhead_{case_name}_from_ep014center_70alr3e4_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_merged_perhead_{case_name}_from_ep014center_70alr3e4_{RUN_DATE}.json"
    out_dir = MODEL_ROOT / case_name
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"
    scope_json = OUT_ROOT / "scope" / f"{case_name}_scope_summary.json"

    cfg_json = make_generated_config(
        CURRENT_REPLACE_CONFIG_JSON,
        cfg_json,
        {
            "ckpt_in": str(WARMSTART_CKPT),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": 5e-5,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "optimizer_param_group_overrides": build_optimizer_group_overrides(nonleg_lr),
        },
    )
    scope_summary = inspect_model_scope(config_json=cfg_json, ckpt_in=WARMSTART_CKPT, ds=ds)
    write_scope_summary(scope_json, scope_summary)

    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=WARMSTART_CKPT,
        out_dir=out_dir,
        run_name=run_name,
        log_file=lane_log,
    )
    eval_json = run_eval(
        model_ckpt=ckpt,
        out_dir=eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(eval_json, group_json, log_file=lane_log)
    eval_payload = collect_eval(eval_json, group_json)
    return {
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "scope_summary": str(scope_json),
        "nonleg_lr": float(nonleg_lr),
        "eval": eval_payload,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center per-head LR merged-stage",
        "",
        f"- fixed basetrain config: `{FIXED_BASETRAIN_CONFIG}`",
        f"- fixed stage6 winner: `{FIXED_STAGE6_WINNER}`",
        f"- fixed 70a winner: `{FIXED_70A_WINNER}`",
        f"- warmstart ckpt: `{WARMSTART_CKPT}`",
        "",
        "## Reference Rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("source_70a_lr3e4", refs["source_70a"]["eval"]["selected_metrics"]),
        row("current_replace_lr5e5", refs["current_replace_lr5e5"]["eval"]["selected_metrics"]),
        row("current_skip70R_to71_lr3e4", refs["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]),
        row("docs_new70b_replace_lowdrift", refs["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"]),
        row("docs_70R", refs["docs_baseline_70R"]["eval"]["selected_metrics"]),
        row("docs_71_lr3e4", refs["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        "",
        "## Candidate Rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, case_payload in summary["cases"].items():
        lines.append(row(case_name, case_payload["eval"]["selected_metrics"]))
    lines.extend(
        [
            "",
            "## Candidate Deltas",
            "",
            "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, case_payload in summary["cases"].items():
        deltas = case_payload["deltas"]
        for delta_name in (
            "vs_source_70a",
            "vs_current_replace_lr5e5",
            "vs_current_skip70R_to71_lr3e4",
            "vs_docs_new70b_replace_lowdrift",
            "vs_docs_70R",
            "vs_docs_71_lr3e4",
        ):
            metric_block = deltas[delta_name]
            lines.append(
                f"| {case_name} {delta_name} | {fmt(metric_block['DirectGeoLocalDeg'])} | {fmt(metric_block['all_ex_root'])} | "
                f"{fmt(metric_block['leg'])} | {fmt(metric_block['nonleg'])} | {fmt(metric_block['arm'])} | "
                f"{fmt(metric_block['foot_l_ball_l_SIC12_15'])} | {fmt(metric_block['calf_r_SIC2_4'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_case_filter(raw: Optional[str]) -> Sequence[Tuple[str, float]]:
    if raw is None or not str(raw).strip():
        return MAIN_CASES
    wanted = {tok.strip() for tok in str(raw).replace(";", ",").split(",") if tok.strip()}
    return tuple((name, lr) for name, lr in MAIN_CASES if name in wanted)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ep014center merged-stage per-head LR experiments.")
    ap.add_argument("--cases", type=str, default=None, help="Comma-separated subset of case names to run.")
    args = ap.parse_args()

    required = [
        FIXED_BASETRAIN_CONFIG,
        FIXED_STAGE6_WINNER,
        FIXED_70A_WINNER,
        CURRENT_REPLACE_CKPT,
        CURRENT_SKIP71_CKPT,
        SOURCE_70A_ARTIFACTS_JSON,
        CURRENT_REPLACE_SUMMARY_JSON,
        CURRENT_SKIP71_SUMMARY_JSON,
        CURRENT_LOWDRIFT_SUMMARY_JSON,
        DOCS_BASELINE_SUMMARY_JSON,
        CURRENT_REPLACE_CONFIG_JSON,
        CURRENT_SKIP71_CONFIG_JSON,
        SOURCE_70A_CONFIG_JSON,
        ENCODER_BUNDLE,
        AFFINE_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    cases = parse_case_filter(args.cases)
    if not cases:
        raise SystemExit("no cases selected")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    log("=== build replace zerophase warmstart from fixed 70a ===")
    create_replace_zerophase_warmstart(
        src_ckpt=FIXED_70A_WINNER,
        dst_ckpt=WARMSTART_CKPT,
        report_json=WARMSTART_REPORT,
    )

    log("=== inspect reference scopes ===")
    _, ds = build_dataset_for_inspection()
    reference_scope = {
        "source_70a": inspect_model_scope(config_json=SOURCE_70A_CONFIG_JSON, ckpt_in=FIXED_STAGE6_WINNER, ds=ds),
        "current_replace_lr5e5": inspect_model_scope(config_json=CURRENT_REPLACE_CONFIG_JSON, ckpt_in=FIXED_70A_WINNER, ds=ds),
        "current_skip70R_to71_lr3e4": inspect_model_scope(config_json=CURRENT_SKIP71_CONFIG_JSON, ckpt_in=CURRENT_REPLACE_CKPT, ds=ds),
    }
    write_scope_summary(OUT_ROOT / "reference_scope_summary.json", reference_scope)

    references = load_references()
    summary: Dict[str, Any] = {
        "run_date": RUN_DATE,
        "policy": {
            "main_route": "70a winner -> replace semantics merged stage with optimizer param-group LR",
            "base_config": str(CURRENT_REPLACE_CONFIG_JSON),
            "fixed_basetrain_config": str(FIXED_BASETRAIN_CONFIG),
            "fixed_stage6_winner": str(FIXED_STAGE6_WINNER),
            "fixed_70a_winner": str(FIXED_70A_WINNER),
            "warmstart_ckpt": str(WARMSTART_CKPT),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        },
        "references": references,
        "reference_scope_summary": str(OUT_ROOT / "reference_scope_summary.json"),
        "cases": {},
    }

    for case_name, nonleg_lr in cases:
        log(f"=== run case {case_name} nonleg_lr={nonleg_lr:.2e} ===")
        case_payload = run_candidate(case_name=case_name, nonleg_lr=nonleg_lr, ds=ds, lane_log=lane_log)
        selected = case_payload["eval"]["selected_metrics"]
        case_payload["deltas"] = {
            "vs_source_70a": metric_delta(selected, references["source_70a"]["eval"]["selected_metrics"]),
            "vs_current_replace_lr5e5": metric_delta(selected, references["current_replace_lr5e5"]["eval"]["selected_metrics"]),
            "vs_current_skip70R_to71_lr3e4": metric_delta(selected, references["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]),
            "vs_docs_new70b_replace_lowdrift": metric_delta(selected, references["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"]),
            "vs_docs_70R": metric_delta(selected, references["docs_baseline_70R"]["eval"]["selected_metrics"]),
            "vs_docs_71_lr3e4": metric_delta(selected, references["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        }
        summary["cases"][case_name] = case_payload

    write_json(
        OUT_ROOT / "status.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warmstart_ckpt": str(WARMSTART_CKPT),
            "cases": list(summary["cases"].keys()),
        },
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
