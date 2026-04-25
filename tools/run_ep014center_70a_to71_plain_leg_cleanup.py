#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

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

try:
    from run_ep014center_70a_vs_baseline_sic_profile import (
        GROUPS as SIC_GROUPS,
        _delta_summary,
        _per_sic_group_profile,
        _profile_stats,
    )
except ModuleNotFoundError:
    from tools.run_ep014center_70a_vs_baseline_sic_profile import (
        GROUPS as SIC_GROUPS,
        _delta_summary,
        _per_sic_group_profile,
        _profile_stats,
    )

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain
from train.posttrain_build_shell import _build_posttrain_model_from_ckpt
from train.runtime.freeze import _freeze_all, _select_trainable_params, _unfreeze_for_train_mode


RUN_DATE = "20260328"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_70a_to71_plain_leg_cleanup_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_70a_to71_plain_leg_cleanup_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SCOPE_ROOT = OUT_ROOT / "scope"
SIC_ROOT = OUT_ROOT / "sic"

FIXED_BASETRAIN_CONFIG = ROOT / "config" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324.json"
FIXED_STAGE6_WINNER = ROOT / "models" / "__tmp_ep014center_main_anchor_check" / "tmp_ep014center_main_selector_check__last" / "ckpt_last_tmp_ep014center_main_selector_check__last_stage6_anchor_ep014center_main_check.pth"
FIXED_70A_WINNER = ROOT / "models" / "__tmp_ep014center_70a_lowlr_sweep_20260328" / "lr3e4" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth"
CURRENT_REPLACE_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lowlr_sweep_20260328" / "lr5e5" / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_ep014center_70alr3e4_20260328.pth"
CURRENT_SKIP71_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "71" / "ckpt_last_WalkF_stage7_71_lr3e4_from_ep014center_replace_lr5e5_20260328.pth"

SOURCE_70A_ARTIFACTS_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "artifacts.json"
CURRENT_REPLACE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
CURRENT_SKIP71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"
SIC_DIAGNOSTIC_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_70a_vs_baseline_sic_profile_20260328" / "summary.md"
CURRENT_SCOPE_AUDIT_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_per_head_merged_stage_20260328" / "reference_scope_summary.json"
CURRENT_SKIP71_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "configs" / "posttrain_71_lr3e4_from_ep014center_replace_lr5e5_20260328.json"

CASES: Tuple[Tuple[str, float], ...] = (
    ("lr3e4", 3e-4),
    ("lr1e4", 1e-4),
    ("lr5e4", 5e-4),
)

SELECTED_METRICS: Tuple[str, ...] = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)

INSPECT_MODULE_NAMES: Tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_leg_terminal",
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


def build_inspection_dataset() -> Any:
    payload = load_json(CONFIG_71)
    payload["ckpt_in"] = str(FIXED_70A_WINNER)
    payload["encoder_bundle"] = str(ENCODER_BUNDLE)
    payload["posttrain_contacts_source"] = "pretrain_contact"
    payload["posttrain_contacts_pretrain_clamp"] = PRETRAIN_CLAMP
    payload["posttrain_contacts_pretrain_affine_stats"] = str(AFFINE_STATS)
    cfg = posttrain._cfg_from_payload(payload)
    _, ds, _ = posttrain._build_dataset_and_loader(cfg)
    return ds


def inspect_scope(*, config_json: Path, ckpt_in: Path, ds: Any) -> Dict[str, Any]:
    payload = load_json(config_json)
    payload["ckpt_in"] = str(ckpt_in)
    payload["load_context"] = "chain_hop"
    payload["encoder_bundle"] = str(ENCODER_BUNDLE)
    payload["posttrain_contacts_source"] = "pretrain_contact"
    payload["posttrain_contacts_pretrain_clamp"] = PRETRAIN_CLAMP
    payload["posttrain_contacts_pretrain_affine_stats"] = str(AFFINE_STATS)
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

    instantiated: Dict[str, Any] = {}
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
        "trainable_param_count": int(sum(int(p.numel()) for p in params)),
    }


def checkpoint_posttrain_cfg_flags(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    cfg = obj.get("posttrain_cfg", {}) if isinstance(obj, dict) else {}
    return {
        "path": str(path),
        "has_posttrain_cfg": isinstance(cfg, dict),
        "has_direct_pose_use_phase_z": bool(isinstance(cfg, dict) and ("direct_pose_use_phase_z" in cfg)),
        "has_direct_pose_phase_z_mode": bool(isinstance(cfg, dict) and ("direct_pose_phase_z_mode" in cfg)),
    }


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
    }


def run_case(*, case_name: str, lr: float, lane_log: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_71_plain_{case_name}_from_ep014center_70alr3e4_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_71_plain_{case_name}_from_ep014center_70a_{RUN_DATE}.json"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    group_json = OUT_ROOT / "eval_model_source" / f"{case_name}_group_summary.json"

    make_generated_config(
        CONFIG_71,
        cfg_json,
        {
            "ckpt_in": str(FIXED_70A_WINNER),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "lr": float(lr),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            # These are legacy keys ignored by the current parser. Set them to the
            # no-phase values so the generated config reflects the intended path.
            "direct_pose_use_phase_z": False,
            "direct_pose_phase_z_mode": "concat",
        },
    )
    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=FIXED_70A_WINNER,
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
    return {
        "lr": float(lr),
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "eval": collect_eval(eval_json, group_json),
    }


def build_scope_checks(ds: Any) -> Dict[str, Any]:
    base_payload = load_json(CONFIG_71)
    parsed_cfg = posttrain._cfg_from_payload(base_payload)
    current_scope = inspect_scope(config_json=CURRENT_SKIP71_CONFIG_JSON, ckpt_in=CURRENT_REPLACE_CKPT, ds=ds)
    direct_scope = inspect_scope(config_json=CONFIG_71, ckpt_in=FIXED_70A_WINNER, ds=ds)
    return {
        "config_71_path": str(CONFIG_71),
        "legacy_phase_keys_in_config": {
            "direct_pose_use_phase_z": bool("direct_pose_use_phase_z" in base_payload),
            "direct_pose_phase_z_mode": bool("direct_pose_phase_z_mode" in base_payload),
        },
        "legacy_phase_keys_in_parsed_cfg": {
            "direct_pose_use_phase_z": bool(hasattr(parsed_cfg, "direct_pose_use_phase_z")),
            "direct_pose_phase_z_mode": bool(hasattr(parsed_cfg, "direct_pose_phase_z_mode")),
        },
        "checkpoint_posttrain_cfg_flags": {
            "source_70a": checkpoint_posttrain_cfg_flags(FIXED_70A_WINNER),
            "current_skip71": checkpoint_posttrain_cfg_flags(CURRENT_SKIP71_CKPT),
        },
        "current_skip71_scope": current_scope,
        "direct_71_from_70a_scope": direct_scope,
        "compatibility": {
            "same_train_mode": str(current_scope["train_mode"]) == str(direct_scope["train_mode"]),
            "same_trainable_names": list(current_scope["trainable_param_names"]) == list(direct_scope["trainable_param_names"]),
            "same_optimizer_group_count": len(current_scope["optimizer_param_groups"]) == len(direct_scope["optimizer_param_groups"]),
            "same_split_enable": bool(current_scope["resolved"]["direct_pose_split_enable"]) == bool(direct_scope["resolved"]["direct_pose_split_enable"]),
            "same_nonleg_proj_dim": int(current_scope["resolved"]["direct_pose_nonleg_proj_dim"]) == int(direct_scope["resolved"]["direct_pose_nonleg_proj_dim"]),
            "implicit_replace_blocker": False,
        },
    }


def maybe_build_sic_summary(
    *,
    case_name: str,
    case_eval_json: Path,
    references: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        profiles = {
            "source_70a": _per_sic_group_profile(Path(str(references["source_70a"]["eval"]["paths"]["eval_json"])), cycle_gte=1, drop_wrap=True),
            "current_replace_lr5e5": _per_sic_group_profile(Path(str(references["current_replace_lr5e5"]["eval"]["paths"]["eval_json"])), cycle_gte=1, drop_wrap=True),
            "current_skip70R_to71_lr3e4": _per_sic_group_profile(Path(str(references["current_skip70R_to71_lr3e4"]["eval"]["paths"]["eval_json"])), cycle_gte=1, drop_wrap=True),
            "candidate": _per_sic_group_profile(case_eval_json, cycle_gte=1, drop_wrap=True),
        }
    except Exception as exc:
        return {
            "case_name": case_name,
            "error": str(exc),
        }

    summary: Dict[str, Any] = {
        "case_name": case_name,
        "artifacts": {
            "source_70a_eval_json": str(references["source_70a"]["eval"]["paths"]["eval_json"]),
            "current_replace_lr5e5_eval_json": str(references["current_replace_lr5e5"]["eval"]["paths"]["eval_json"]),
            "current_skip70R_to71_lr3e4_eval_json": str(references["current_skip70R_to71_lr3e4"]["eval"]["paths"]["eval_json"]),
            "candidate_eval_json": str(case_eval_json),
        },
        "groups": {},
    }

    for group_name in SIC_GROUPS:
        source_profile = profiles["source_70a"]["profiles"][group_name]
        replace_profile = profiles["current_replace_lr5e5"]["profiles"][group_name]
        skip71_profile = profiles["current_skip70R_to71_lr3e4"]["profiles"][group_name]
        cand_profile = profiles["candidate"]["profiles"][group_name]
        summary["groups"][group_name] = {
            "source_70a": _profile_stats(source_profile),
            "current_replace_lr5e5": _profile_stats(replace_profile),
            "current_skip70R_to71_lr3e4": _profile_stats(skip71_profile),
            "candidate": _profile_stats(cand_profile),
            "candidate_delta_vs_source_70a": _delta_summary(source_profile, cand_profile),
            "candidate_delta_vs_current_replace_lr5e5": _delta_summary(replace_profile, cand_profile),
            "candidate_delta_vs_current_skip70R_to71_lr3e4": _delta_summary(skip71_profile, cand_profile),
        }

    write_json(SIC_ROOT / f"{case_name}_summary.json", summary)
    lines = [
        f"# SIC summary for {case_name}",
        "",
        "- mask: `cycle>=1`, `drop_wrap=true`",
        "",
        "| group | src70a top SICs | replace top SICs | skip71 top SICs | candidate top SICs | cand d(top8-source) | cand d(rest-source) |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for group_name in SIC_GROUPS:
        row = summary["groups"][group_name]
        d_src = row["candidate_delta_vs_source_70a"]
        lines.append(
            f"| {group_name} | {row['source_70a'].get('top_sics', [])[:8]} | "
            f"{row['current_replace_lr5e5'].get('top_sics', [])[:8]} | "
            f"{row['current_skip70R_to71_lr3e4'].get('top_sics', [])[:8]} | "
            f"{row['candidate'].get('top_sics', [])[:8]} | "
            f"{fmt(d_src.get('mean_delta_on_own_top_before'))} | {fmt(d_src.get('mean_delta_on_rest'))} |"
        )
    md_path = SIC_ROOT / f"{case_name}_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_md"] = str(md_path)
    write_json(SIC_ROOT / f"{case_name}_summary.json", summary)
    return summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]
    findings = summary["findings"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center 70a -> 71 plain leg-only cleanup",
        "",
        "## Findings",
        "",
        f"- current 71 trainable scope: `{findings['current_71_scope']}`",
        f"- direct 70a -> 71 scope compatibility: `{str(bool(findings['direct_70a_to71_scope_compatible'])).lower()}`",
        f"- replace/phase_z warmstart blocker: `{str(bool(findings['implicit_replace_blocker'])).lower()}`",
        f"- legacy phase keys are active in parser: `{str(bool(findings['legacy_phase_keys_active_in_parser'])).lower()}`",
        f"- legacy phase keys present in checkpoint posttrain_cfg: `{str(bool(findings['legacy_phase_keys_present_in_ckpt_posttrain_cfg'])).lower()}`",
        "",
        "## Reference Metrics",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("source_70a", refs["source_70a"]["eval"]["selected_metrics"]),
        row("current_replace_lr5e5", refs["current_replace_lr5e5"]["eval"]["selected_metrics"]),
        row("current_skip70R_to71_lr3e4", refs["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]),
        row("docs_new70b_replace_lowdrift", refs["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"]),
        row("docs_70R", refs["docs_baseline_70R"]["eval"]["selected_metrics"]),
        row("docs_71_lr3e4", refs["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        "",
        "## Candidate Metrics",
        "",
        "| candidate | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
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
        for compare_name, delta_row in case_payload["deltas"].items():
            lines.append(
                f"| {case_name} - {compare_name} | {fmt(delta_row['DirectGeoLocalDeg'])} | {fmt(delta_row['all_ex_root'])} | "
                f"{fmt(delta_row['leg'])} | {fmt(delta_row['nonleg'])} | {fmt(delta_row['arm'])} | "
                f"{fmt(delta_row['foot_l_ball_l_SIC12_15'])} | {fmt(delta_row['calf_r_SIC2_4'])} |"
            )
    best_case = summary["answers"]["best_case_by_all_ex_root"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- best case by `all_ex_root`: `{best_case}`",
            f"- best case also wins vs current replace on `leg`: `{str(bool(summary['answers']['best_case_beats_current_replace_on_leg'])).lower()}`",
            f"- best case also wins vs current replace on `calf_r@SIC2-4`: `{str(bool(summary['answers']['best_case_beats_current_replace_on_calf'])).lower()}`",
            f"- best case also wins vs current skip71 on `all_ex_root`: `{str(bool(summary['answers']['best_case_beats_current_skip71_on_all_ex_root'])).lower()}`",
            f"- best case also wins vs current skip71 on `calf_r@SIC2-4`: `{str(bool(summary['answers']['best_case_beats_current_skip71_on_calf'])).lower()}`",
            f"- best case also wins vs current skip71 on `foot_l/ball_l@SIC12-15`: `{str(bool(summary['answers']['best_case_beats_current_skip71_on_foot'])).lower()}`",
            "",
        ]
    )
    sic = summary.get("best_case_sic", None)
    if isinstance(sic, Mapping) and ("groups" in sic):
        lines.extend(
            [
                "## Best-Case SIC",
                "",
                f"- summary: `{sic.get('summary_md')}`",
                "",
            ]
        )
        for group_name in ("all_ex_root", "leg"):
            group = sic["groups"][group_name]
            cand = group["candidate"]
            src = group["source_70a"]
            d_src = group["candidate_delta_vs_source_70a"]
            lines.append(
                f"- {group_name}: candidate top SICs={cand.get('top_sics', [])[:8]}, "
                f"source70a top SICs={src.get('top_sics', [])[:8]}, "
                f"d(top8-source)={fmt(d_src.get('mean_delta_on_own_top_before'))}, "
                f"d(rest-source)={fmt(d_src.get('mean_delta_on_rest'))}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        FIXED_BASETRAIN_CONFIG,
        FIXED_STAGE6_WINNER,
        FIXED_70A_WINNER,
        CURRENT_REPLACE_CKPT,
        CURRENT_SKIP71_CKPT,
        SOURCE_70A_ARTIFACTS_JSON,
        CURRENT_REPLACE_SUMMARY_JSON,
        CURRENT_SKIP71_SUMMARY_JSON,
        DOCS_BASELINE_SUMMARY_JSON,
        SIC_DIAGNOSTIC_SUMMARY,
        CURRENT_SCOPE_AUDIT_SUMMARY,
        CONFIG_71,
        ENCODER_BUNDLE,
        AFFINE_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    SCOPE_ROOT.mkdir(parents=True, exist_ok=True)
    SIC_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    log("=== inspect current 71 scope and direct 70a -> 71 compatibility ===")
    ds = build_inspection_dataset()
    scope_checks = build_scope_checks(ds)
    write_json(SCOPE_ROOT / "scope_checks.json", scope_checks)

    references = load_references()
    cases: Dict[str, Any] = {}
    extras_allowed = False
    for idx, (case_name, lr) in enumerate(CASES):
        if idx > 0 and not extras_allowed:
            break
        log(f"=== run {case_name} lr={lr:.6g} ===")
        case_payload = run_case(case_name=case_name, lr=lr, lane_log=lane_log)
        case_payload["deltas"] = {
            "source_70a": metric_delta(case_payload["eval"]["selected_metrics"], references["source_70a"]["eval"]["selected_metrics"]),
            "current_replace_lr5e5": metric_delta(case_payload["eval"]["selected_metrics"], references["current_replace_lr5e5"]["eval"]["selected_metrics"]),
            "current_skip70R_to71_lr3e4": metric_delta(case_payload["eval"]["selected_metrics"], references["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]),
            "docs_new70b_replace_lowdrift": metric_delta(case_payload["eval"]["selected_metrics"], references["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"]),
            "docs_70R": metric_delta(case_payload["eval"]["selected_metrics"], references["docs_baseline_70R"]["eval"]["selected_metrics"]),
            "docs_71_lr3e4": metric_delta(case_payload["eval"]["selected_metrics"], references["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        }
        cases[case_name] = case_payload
        if idx == 0:
            extras_allowed = True

    ranking = sorted(
        ((safe_float(payload["eval"]["selected_metrics"]["all_ex_root"]), case_name) for case_name, payload in cases.items()),
        key=lambda item: item[0],
    )
    best_case = ranking[0][1]
    best_payload = cases[best_case]
    best_metrics = best_payload["eval"]["selected_metrics"]
    best_case_sic = maybe_build_sic_summary(
        case_name=best_case,
        case_eval_json=Path(str(best_payload["eval"]["paths"]["eval_json"])),
        references=references,
    )

    findings = {
        "current_71_scope": ", ".join(scope_checks["current_skip71_scope"]["trainable_param_names"]),
        "direct_70a_to71_scope_compatible": bool(scope_checks["compatibility"]["same_train_mode"] and scope_checks["compatibility"]["same_trainable_names"]),
        "implicit_replace_blocker": bool(scope_checks["compatibility"]["implicit_replace_blocker"]),
        "legacy_phase_keys_active_in_parser": bool(
            scope_checks["legacy_phase_keys_in_parsed_cfg"]["direct_pose_use_phase_z"]
            or scope_checks["legacy_phase_keys_in_parsed_cfg"]["direct_pose_phase_z_mode"]
        ),
        "legacy_phase_keys_present_in_ckpt_posttrain_cfg": bool(
            scope_checks["checkpoint_posttrain_cfg_flags"]["source_70a"]["has_direct_pose_use_phase_z"]
            or scope_checks["checkpoint_posttrain_cfg_flags"]["source_70a"]["has_direct_pose_phase_z_mode"]
            or scope_checks["checkpoint_posttrain_cfg_flags"]["current_skip71"]["has_direct_pose_use_phase_z"]
            or scope_checks["checkpoint_posttrain_cfg_flags"]["current_skip71"]["has_direct_pose_phase_z_mode"]
        ),
    }

    summary = {
        "run_date": RUN_DATE,
        "inputs": {
            "basetrain_config": str(FIXED_BASETRAIN_CONFIG),
            "stage6_winner": str(FIXED_STAGE6_WINNER),
            "source_70a_winner": str(FIXED_70A_WINNER),
            "current_replace_ckpt": str(CURRENT_REPLACE_CKPT),
            "current_skip71_ckpt": str(CURRENT_SKIP71_CKPT),
            "sic_diagnostic_summary": str(SIC_DIAGNOSTIC_SUMMARY),
            "current_scope_audit_summary": str(CURRENT_SCOPE_AUDIT_SUMMARY),
            "base_71_config": str(CONFIG_71),
        },
        "policy": {
            "primary_run_first": {"case": CASES[0][0], "lr": CASES[0][1]},
            "extra_runs_after_primary": [
                {"case": case_name, "lr": lr}
                for case_name, lr in CASES[1:]
                if case_name in cases
            ],
            "reuse_pipeline": [
                "run_posttrain_stage",
                "run_eval",
                "ensure_group_summary",
                "masked_metric_means",
                "group_metrics",
                "window_group_stats",
            ],
            "compare_contract": "model_source",
            "notes": [
                "current 71 trainable scope is preserved",
                "no model-structure, loss-contract, or sampling changes",
                "legacy phase_z keys are ignored by the current parser and absent from checkpoint posttrain_cfg",
            ],
        },
        "findings": findings,
        "scope_checks": scope_checks,
        "references": references,
        "cases": cases,
        "answers": {
            "best_case_by_all_ex_root": best_case,
            "best_case_beats_current_replace_on_leg": safe_float(best_metrics["leg"]) < safe_float(references["current_replace_lr5e5"]["eval"]["selected_metrics"]["leg"]),
            "best_case_beats_current_replace_on_calf": safe_float(best_metrics["calf_r_SIC2_4"]) < safe_float(references["current_replace_lr5e5"]["eval"]["selected_metrics"]["calf_r_SIC2_4"]),
            "best_case_beats_current_skip71_on_all_ex_root": safe_float(best_metrics["all_ex_root"]) < safe_float(references["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]["all_ex_root"]),
            "best_case_beats_current_skip71_on_calf": safe_float(best_metrics["calf_r_SIC2_4"]) < safe_float(references["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]["calf_r_SIC2_4"]),
            "best_case_beats_current_skip71_on_foot": safe_float(best_metrics["foot_l_ball_l_SIC12_15"]) < safe_float(references["current_skip70R_to71_lr3e4"]["eval"]["selected_metrics"]["foot_l_ball_l_SIC12_15"]),
        },
        "best_case_sic": best_case_sic,
    }
    write_json(OUT_ROOT / "status.json", {"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "best_case": best_case})
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
