#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
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
    from run_ep014center_70a_vs_baseline_sic_profile import _profile_stats
except ModuleNotFoundError:
    from tools.run_ep014center_70a_vs_baseline_sic_profile import _profile_stats

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import posttrain
from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
from train.posttrain_build_shell import _build_posttrain_model_from_ckpt
from train.runtime.freeze import _freeze_all, _select_trainable_params, _unfreeze_for_train_mode


RUN_DATE = "20260328"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_stage6_early_live_cleanup_bias_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_stage6_early_live_cleanup_bias_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SCOPE_ROOT = OUT_ROOT / "scope"
SIC_ROOT = OUT_ROOT / "sic"

FIXED_BASETRAIN_CONFIG = ROOT / "config" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324.json"
FIXED_STAGE6_WINNER = ROOT / "models" / "__tmp_ep014center_main_anchor_check" / "tmp_ep014center_main_selector_check__last" / "ckpt_last_tmp_ep014center_main_selector_check__last_stage6_anchor_ep014center_main_check.pth"
FIXED_70A_WINNER = ROOT / "models" / "__tmp_ep014center_70a_lowlr_sweep_20260328" / "lr3e4" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth"
CURRENT_REPLACE_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lowlr_sweep_20260328" / "lr5e5" / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_ep014center_70alr3e4_20260328.pth"
CURRENT_SKIP71_CKPT = ROOT / "models" / "__tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "71" / "ckpt_last_WalkF_stage7_71_lr3e4_from_ep014center_replace_lr5e5_20260328.pth"

SOURCE_70A_ARTIFACTS_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "artifacts.json"
SOURCE_70A_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "configs" / "posttrain_70a_lr3e4_from_ep014center_20260328.json"
CURRENT_REPLACE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
CURRENT_REPLACE_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "configs" / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
CURRENT_SKIP71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "summary.json"
CURRENT_SKIP71_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "configs" / "posttrain_71_lr3e4_from_ep014center_replace_lr5e5_20260328.json"
PLAIN71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_to71_plain_leg_cleanup_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"
SIC_DIAGNOSTIC_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_70a_vs_baseline_sic_profile_20260328" / "summary.md"
CURRENT_SCOPE_AUDIT_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_per_head_merged_stage_20260328" / "reference_scope_summary.json"
CURRENT_REPLACE_REDESIGN_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_replace_redesign_20260328" / "summary.json"

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

CANDIDATES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "main_align3_lr1e4",
        "kind": "early_align",
        "role": "main",
        "hypothesis": (
            "Keep the exact 70a coupled cleanup base and full 20-tensor scope, but inject a mild live "
            "`direct_pose_leg_align_*` oracle prior from step 0 so broad cleanup bias participates while the "
            "Stage6(last) concentrated spike is still present."
        ),
        "cleanup_overrides": {
            "lr": 1e-4,
            "epochs": 5,
            "steps_per_epoch": 60,
            "direct_pose_leg_stopgrad_main": False,
            "direct_pose_leg_detach_feat": False,
            "direct_pose_leg_gate_mode": "none",
            "direct_pose_leg_align_weight": 3.0,
            "direct_pose_leg_align_oracle_min_deg": 0.5,
            "direct_pose_leg_align_mode": "proj",
            "direct_pose_leg_align_schedule": "linear",
            "direct_pose_leg_align_start_weight": 1.0,
            "direct_pose_leg_align_warmup_steps": 0,
            "direct_pose_leg_align_ramp_steps": 20,
            "save_step_ckpts": "0,1,5,20,60,120,180,240,300",
        },
    },
    {
        "name": "backup_scale_lr1e4",
        "kind": "mild_scale",
        "role": "fallback",
        "hypothesis": (
            "Keep the 70a coupled cleanup base but replace the extra dense oracle prior with a live `scale` gate "
            "head so leg omega gets a mild magnitude calibration path from step 0 without full trunk decoupling."
        ),
        "cleanup_overrides": {
            "lr": 1e-4,
            "epochs": 5,
            "steps_per_epoch": 60,
            "direct_pose_leg_stopgrad_main": False,
            "direct_pose_leg_detach_feat": False,
            "direct_pose_leg_gate_mode": "scale",
            "direct_pose_leg_scale_log_clip": 1.5,
            "direct_pose_leg_scale_clamp_k": 2.0,
            "direct_pose_leg_align_weight": 0.0,
            "save_step_ckpts": "0,1,5,20,60,120,180,240,300",
        },
    },
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
    payload = load_json(SOURCE_70A_CONFIG_JSON)
    payload["ckpt_in"] = str(FIXED_STAGE6_WINNER)
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
    artifacts = _build_posttrain_model_from_ckpt(cfg=cfg, ds=ds, device=torch.device("cpu"))
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
            "direct_pose_meas_mode_model": str(getattr(model, "direct_pose_meas_mode", "unknown")),
            "direct_pose_leg_mode_model": str(getattr(model, "direct_pose_leg_mode", "unknown")),
            "direct_pose_leg_stopgrad_main_model": bool(getattr(model, "direct_pose_leg_stopgrad_main", False)),
            "direct_pose_leg_detach_feat_model": bool(getattr(model, "direct_pose_leg_detach_feat", False)),
            "direct_pose_leg_gate_mode_model": str(artifacts.direct_pose_leg_gate_mode_model),
            "direct_pose_leg_gate_power_model": float(artifacts.direct_pose_leg_gate_power_model),
            "direct_pose_loss_leg_split_cfg": bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
            "direct_pose_loss_group_norm_enable_cfg": bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
            "direct_pose_leg_align_weight_cfg": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0)),
            "direct_pose_leg_align_mode_cfg": str(getattr(cfg, "direct_pose_leg_align_mode", "cos")),
        },
        "instantiated_modules": instantiated,
        "trainable_param_names": list(names),
        "optimizer_param_groups": group_summaries,
        "trainable_param_count": int(sum(int(p.numel()) for p in params)),
    }


def raw_and_parsed_phase_flags(config_json: Path) -> Dict[str, Any]:
    payload = load_json(config_json)
    cfg = posttrain._cfg_from_payload(payload)
    return {
        "config_json": str(config_json),
        "raw_has_direct_pose_use_phase_z": "direct_pose_use_phase_z" in payload,
        "raw_has_direct_pose_phase_z_mode": "direct_pose_phase_z_mode" in payload,
        "parsed_has_direct_pose_use_phase_z": hasattr(cfg, "direct_pose_use_phase_z"),
        "parsed_has_direct_pose_phase_z_mode": hasattr(cfg, "direct_pose_phase_z_mode"),
    }


def checkpoint_phase_flags(path: Path) -> Dict[str, Any]:
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
    plain71 = load_json(PLAIN71_SUMMARY_JSON)
    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)["stage_progress_model_source"]

    best_plain71_name = str(plain71.get("answers", {}).get("best_case_by_all_ex_root") or "lr1e4")
    best_plain71 = plain71["cases"][best_plain71_name]

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
        "current_best_downstream_71_lr3e4": {
            "config": str(current_skip71["config"]),
            "ckpt": str(current_skip71["ckpt"]),
            "eval": current_skip71["eval"],
        },
        "docs_baseline_new70b_replace_lowdrift": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["new70b_replace_lowdrift"]),
        },
        "docs_baseline_71_lr3e4": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["71_lr3e4"]),
        },
        "plain71_best_case": {
            "case_name": best_plain71_name,
            "config": str(best_plain71["config"]),
            "ckpt": str(best_plain71["ckpt"]),
            "eval": best_plain71["eval"],
        },
    }


def candidate_by_name(name: str) -> Dict[str, Any]:
    for candidate in CANDIDATES:
        if str(candidate["name"]) == str(name):
            return dict(candidate)
    raise KeyError(name)


def parse_cases(raw: Optional[str]) -> Tuple[Dict[str, Any], ...]:
    if raw is None or not str(raw).strip():
        return (candidate_by_name("main_align3_lr1e4"),)
    wanted = {tok.strip() for tok in str(raw).replace(";", ",").split(",") if tok.strip()}
    out = []
    for candidate in CANDIDATES:
        if str(candidate["name"]) in wanted:
            out.append(dict(candidate))
    if not out:
        raise SystemExit(f"no candidates selected from {sorted(wanted)}")
    return tuple(out)


def run_cleanup_candidate(*, candidate: Mapping[str, Any], ds: Any, lane_log: Path) -> Dict[str, Any]:
    case_name = str(candidate["name"])
    run_name = f"WalkF_stage7_70a_earlylive_{case_name}_from_ep014center_stage6last_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70a_earlylive_{case_name}_{RUN_DATE}.json"
    out_dir = MODEL_ROOT / case_name / "cleanup"
    eval_dir = OUT_ROOT / "cleanup_eval" / case_name
    group_json = OUT_ROOT / "cleanup_eval" / f"{case_name}_group_summary.json"
    scope_json = SCOPE_ROOT / f"{case_name}_cleanup_scope.json"

    overrides = dict(candidate["cleanup_overrides"])
    overrides.update(
        ckpt_in=str(FIXED_STAGE6_WINNER),
        out_dir=str(out_dir),
        run_name=run_name,
        encoder_bundle=str(ENCODER_BUNDLE),
        posttrain_contacts_source="pretrain_contact",
        posttrain_contacts_pretrain_clamp=PRETRAIN_CLAMP,
        posttrain_contacts_pretrain_affine_stats=str(AFFINE_STATS),
    )
    make_generated_config(SOURCE_70A_CONFIG_JSON, cfg_json, overrides)

    scope_summary = inspect_scope(config_json=cfg_json, ckpt_in=FIXED_STAGE6_WINNER, ds=ds)
    write_json(scope_json, scope_summary)

    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=FIXED_STAGE6_WINNER,
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
        "eval": eval_payload,
    }


def run_downstream_candidate(*, candidate: Mapping[str, Any], cleanup_ckpt: Path, ds: Any, lane_log: Path) -> Dict[str, Any]:
    case_name = str(candidate["name"])
    run_name = f"WalkF_stage7_71_from_earlylive_{case_name}_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_71_from_earlylive_{case_name}_{RUN_DATE}.json"
    out_dir = MODEL_ROOT / case_name / "71"
    eval_dir = OUT_ROOT / "downstream_eval" / case_name
    group_json = OUT_ROOT / "downstream_eval" / f"{case_name}_group_summary.json"
    scope_json = SCOPE_ROOT / f"{case_name}_71_scope.json"

    overrides = {
        "ckpt_in": str(cleanup_ckpt),
        "out_dir": str(out_dir),
        "run_name": run_name,
        "lr": 3e-4,
        "epochs": 3,
        "steps_per_epoch": 60,
        "encoder_bundle": str(ENCODER_BUNDLE),
        "posttrain_contacts_source": "pretrain_contact",
        "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
    }
    make_generated_config(CURRENT_SKIP71_CONFIG_JSON, cfg_json, overrides)

    scope_summary = inspect_scope(config_json=cfg_json, ckpt_in=cleanup_ckpt, ds=ds)
    write_json(scope_json, scope_summary)

    ckpt = run_posttrain_stage(
        config=cfg_json,
        ckpt_in=cleanup_ckpt,
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
        "eval": eval_payload,
    }


def _group_indices(names: Sequence[str], root_idx: int) -> Dict[str, list[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    name_to_idx = {str(name): int(i) for i, name in enumerate(names)}
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    idx_nonleg = [i for i in idx_all if i not in set(idx_leg)]
    idx_foot_ball_left = [name_to_idx[name] for name in ("foot_l", "ball_l") if name in name_to_idx]
    idx_calf_r = [name_to_idx["calf_r"]] if "calf_r" in name_to_idx else []
    return {
        "all_ex_root": idx_all,
        "leg": idx_leg,
        "nonleg": idx_nonleg,
        "arm": idx_arm,
        "foot_l_ball_l": idx_foot_ball_left,
        "calf_r": idx_calf_r,
    }


def per_sic_profiles(eval_json: Path, *, cycle_gte: int = 1, drop_wrap: bool = True) -> Dict[str, Any]:
    obj = load_json(eval_json)
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    names = [str(x) for x in per.get("bone_names", [])]
    mat = per.get("DirectGeoLocalDeg", [])
    root_idx = int(per.get("root_idx", 0) or 0)
    cycle_len = int(obj.get("cycle_len", 0) or 0)
    if not isinstance(steps, list) or not isinstance(mat, list) or not names or cycle_len <= 0:
        raise RuntimeError(f"invalid freerun payload: {eval_json}")

    groups = _group_indices(names, root_idx)
    accum: Dict[str, Dict[int, list[float]]] = {
        key: {sic: [] for sic in range(cycle_len)}
        for key in groups
    }
    for step_i, step in enumerate(steps):
        if step_i >= len(mat):
            break
        if int(step.get("cycle", 0) or 0) < int(cycle_gte):
            continue
        if bool(drop_wrap) and bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", 0) or 0)
        row = mat[step_i]
        if not isinstance(row, list):
            continue
        for group_name, joint_indices in groups.items():
            vals = []
            for joint_i in joint_indices:
                if int(joint_i) >= len(row):
                    continue
                value = safe_float(row[joint_i])
                if math.isfinite(value):
                    vals.append(value)
            if vals:
                accum[group_name][sic].append(float(np.mean(np.asarray(vals, dtype=np.float64))))

    profiles: Dict[str, list[float]] = {}
    for group_name, sic_map in accum.items():
        profiles[group_name] = [
            float(np.mean(np.asarray(sic_map[sic], dtype=np.float64))) if sic_map[sic] else float("nan")
            for sic in range(cycle_len)
        ]
    return {
        "eval_json": str(eval_json),
        "cycle_len": int(cycle_len),
        "profiles": profiles,
    }


def window_mean(arr: Sequence[float], start: int, end: int) -> float:
    vals = [safe_float(arr[sic]) for sic in range(max(0, int(start)), min(len(arr), int(end) + 1))]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def rest_mean_excluding(arr: Sequence[float], windows: Sequence[Tuple[int, int]]) -> float:
    keep = []
    for sic, raw in enumerate(arr):
        value = safe_float(raw)
        if not math.isfinite(value):
            continue
        inside = False
        for lo, hi in windows:
            if int(lo) <= sic <= int(hi):
                inside = True
                break
        if not inside:
            keep.append(value)
    if not keep:
        return float("nan")
    return float(sum(keep) / len(keep))


def build_sic_summary(
    *,
    candidate_name: str,
    references: Mapping[str, Any],
    cleanup_eval_json: Path,
    downstream_eval_json: Path,
) -> Dict[str, Any]:
    lane_eval_jsons = {
        "source_70a": Path(str(references["source_70a"]["eval"]["paths"]["eval_json"])),
        "current_replace": Path(str(references["current_replace_lr5e5"]["eval"]["paths"]["eval_json"])),
        "current_best_downstream_71": Path(str(references["current_best_downstream_71_lr3e4"]["eval"]["paths"]["eval_json"])),
        "candidate_cleanup": cleanup_eval_json,
        "candidate_downstream_71": downstream_eval_json,
    }

    lanes: Dict[str, Any] = {}
    for lane_name, eval_json in lane_eval_jsons.items():
        sic = per_sic_profiles(eval_json)
        lane_payload: Dict[str, Any] = {
            "eval_json": str(eval_json),
            "groups": {},
        }
        for group_name, arr in sic["profiles"].items():
            lane_payload["groups"][group_name] = {
                "stats": _profile_stats(arr),
                "window_66_77": window_mean(arr, 66, 77),
                "window_57_64": window_mean(arr, 57, 64),
                "window_12_15": window_mean(arr, 12, 15),
                "window_2_4": window_mean(arr, 2, 4),
                "rest_ex_57_64": rest_mean_excluding(arr, [(57, 64)]),
                "rest_ex_66_77": rest_mean_excluding(arr, [(66, 77)]),
            }
        lanes[lane_name] = lane_payload

    comparisons = {
        "cleanup_vs_source_70a": {},
        "cleanup_vs_current_replace": {},
        "downstream_vs_current_best_downstream_71": {},
        "downstream_vs_source_70a": {},
    }
    for group_name in ("all_ex_root", "leg", "nonleg", "arm", "foot_l_ball_l", "calf_r"):
        cleanup = lanes["candidate_cleanup"]["groups"][group_name]
        source = lanes["source_70a"]["groups"][group_name]
        replace = lanes["current_replace"]["groups"][group_name]
        down = lanes["candidate_downstream_71"]["groups"][group_name]
        best71 = lanes["current_best_downstream_71"]["groups"][group_name]

        comparisons["cleanup_vs_source_70a"][group_name] = {
            "window_66_77": diff(cleanup["window_66_77"], source["window_66_77"]),
            "window_57_64": diff(cleanup["window_57_64"], source["window_57_64"]),
            "window_12_15": diff(cleanup["window_12_15"], source["window_12_15"]),
            "window_2_4": diff(cleanup["window_2_4"], source["window_2_4"]),
        }
        comparisons["cleanup_vs_current_replace"][group_name] = {
            "window_66_77": diff(cleanup["window_66_77"], replace["window_66_77"]),
            "window_57_64": diff(cleanup["window_57_64"], replace["window_57_64"]),
            "window_12_15": diff(cleanup["window_12_15"], replace["window_12_15"]),
            "window_2_4": diff(cleanup["window_2_4"], replace["window_2_4"]),
        }
        comparisons["downstream_vs_current_best_downstream_71"][group_name] = {
            "window_57_64": diff(down["window_57_64"], best71["window_57_64"]),
            "window_12_15": diff(down["window_12_15"], best71["window_12_15"]),
            "window_2_4": diff(down["window_2_4"], best71["window_2_4"]),
        }
        comparisons["downstream_vs_source_70a"][group_name] = {
            "window_57_64": diff(down["window_57_64"], source["window_57_64"]),
            "rest_ex_57_64": diff(down["rest_ex_57_64"], source["rest_ex_57_64"]),
        }

    return {
        "candidate_name": candidate_name,
        "lanes": lanes,
        "comparisons": comparisons,
    }


def build_sic_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# SIC summary for {payload['candidate_name']}",
        "",
        "| lane | all_ex_root max@sic | all_ex_root 66-77 | all_ex_root 57-64 | leg 66-77 | leg 57-64 | foot 12-15 | calf 2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane_name in (
        "source_70a",
        "current_replace",
        "current_best_downstream_71",
        "candidate_cleanup",
        "candidate_downstream_71",
    ):
        lane = payload["lanes"][lane_name]["groups"]
        max_sic = lane["all_ex_root"]["stats"].get("max_sic", float("nan"))
        max_val = lane["all_ex_root"]["stats"].get("max", float("nan"))
        lines.append(
            f"| {lane_name} | {fmt(max_val)}@{int(max_sic) if math.isfinite(safe_float(max_sic)) else 'nan'} | "
            f"{fmt(lane['all_ex_root']['window_66_77'])} | {fmt(lane['all_ex_root']['window_57_64'])} | "
            f"{fmt(lane['leg']['window_66_77'])} | {fmt(lane['leg']['window_57_64'])} | "
            f"{fmt(lane['foot_l_ball_l']['window_12_15'])} | {fmt(lane['calf_r']['window_2_4'])} |"
        )
    lines.extend(
        [
            "",
            "## Key deltas",
            "",
            "| compare | group | d_66_77 | d_57_64 | d_12_15 | d_2_4 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for compare_name in (
        "cleanup_vs_source_70a",
        "cleanup_vs_current_replace",
        "downstream_vs_current_best_downstream_71",
    ):
        compare = payload["comparisons"][compare_name]
        for group_name in ("all_ex_root", "leg", "nonleg", "arm", "foot_l_ball_l", "calf_r"):
            row = compare[group_name]
            lines.append(
                f"| {compare_name} | {group_name} | {fmt(row.get('window_66_77'))} | {fmt(row.get('window_57_64'))} | "
                f"{fmt(row.get('window_12_15'))} | {fmt(row.get('window_2_4'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["references"]

    def row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | {fmt(metrics['leg'])} | "
            f"{fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | {fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# Stage6-start early-injected live cleanup bias",
        "",
        f"- basetrain config: `{summary['inputs']['basetrain_config']}`",
        f"- fixed Stage6 winner: `{summary['inputs']['stage6_winner']}`",
        f"- fixed source 70a winner: `{summary['inputs']['source_70a_winner']}`",
        f"- current replace ckpt: `{summary['inputs']['current_replace_ckpt']}`",
        f"- current best downstream ckpt: `{summary['inputs']['current_best_downstream_ckpt']}`",
        "",
        "## Reference rows",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("source_70a_lr3e4", refs["source_70a"]["eval"]["selected_metrics"]),
        row("current_replace_lr5e5", refs["current_replace_lr5e5"]["eval"]["selected_metrics"]),
        row("current_best_downstream_71_lr3e4", refs["current_best_downstream_71_lr3e4"]["eval"]["selected_metrics"]),
        row("docs_new70b_replace_lowdrift", refs["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"]),
        row("docs_71_lr3e4", refs["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        row(
            f"plain71_best_{refs['plain71_best_case']['case_name']}",
            refs["plain71_best_case"]["eval"]["selected_metrics"],
        ),
        "",
    ]

    for candidate_name, payload in summary["candidates"].items():
        lines.extend(
            [
                f"## Candidate `{candidate_name}`",
                "",
                f"- role: `{payload['role']}`",
                f"- cleanup config: `{payload['cleanup_stage']['config']}`",
                f"- cleanup ckpt: `{payload['cleanup_stage']['ckpt']}`",
                f"- downstream config: `{payload['downstream_71']['config']}`",
                f"- downstream ckpt: `{payload['downstream_71']['ckpt']}`",
                "",
                "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                row(f"{candidate_name}_cleanup", payload["cleanup_stage"]["eval"]["selected_metrics"]),
                row(f"{candidate_name}_to71", payload["downstream_71"]["eval"]["selected_metrics"]),
                "",
                "### Cleanup deltas",
                "",
                "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                row("vs_source_70a", payload["cleanup_stage"]["delta_vs_source_70a"]),
                row("vs_current_replace", payload["cleanup_stage"]["delta_vs_current_replace_lr5e5"]),
                row("vs_docs_new70b_replace", payload["cleanup_stage"]["delta_vs_docs_new70b_replace_lowdrift"]),
                "",
                "### Downstream deltas",
                "",
                "| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                row("vs_current_best_downstream", payload["downstream_71"]["delta_vs_current_best_downstream"]),
                row("vs_docs_71_lr3e4", payload["downstream_71"]["delta_vs_docs_71_lr3e4"]),
                row("vs_plain71_best", payload["downstream_71"]["delta_vs_plain71_best"]),
                row("vs_source_70a", payload["downstream_71"]["delta_vs_source_70a"]),
                "",
                f"- SIC summary: `{payload['sic']['summary_md']}`",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Stage6-start early injected live cleanup bias experiments.")
    ap.add_argument("--cases", type=str, default=None, help="Comma-separated candidate names.")
    args = ap.parse_args()

    required = [
        FIXED_BASETRAIN_CONFIG,
        FIXED_STAGE6_WINNER,
        FIXED_70A_WINNER,
        CURRENT_REPLACE_CKPT,
        CURRENT_SKIP71_CKPT,
        SOURCE_70A_ARTIFACTS_JSON,
        SOURCE_70A_CONFIG_JSON,
        CURRENT_REPLACE_SUMMARY_JSON,
        CURRENT_REPLACE_CONFIG_JSON,
        CURRENT_SKIP71_SUMMARY_JSON,
        CURRENT_SKIP71_CONFIG_JSON,
        PLAIN71_SUMMARY_JSON,
        DOCS_BASELINE_SUMMARY_JSON,
        SIC_DIAGNOSTIC_SUMMARY,
        CURRENT_SCOPE_AUDIT_SUMMARY,
        CURRENT_REPLACE_REDESIGN_SUMMARY,
        ENCODER_BUNDLE,
        AFFINE_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    candidates = parse_cases(args.cases)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    SCOPE_ROOT.mkdir(parents=True, exist_ok=True)
    SIC_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    log("=== build inspection dataset ===")
    ds = build_inspection_dataset()

    log("=== inspect reference scopes ===")
    reference_scope = {
        "source_70a": inspect_scope(config_json=SOURCE_70A_CONFIG_JSON, ckpt_in=FIXED_STAGE6_WINNER, ds=ds),
        "current_replace_lr5e5": inspect_scope(config_json=CURRENT_REPLACE_CONFIG_JSON, ckpt_in=FIXED_70A_WINNER, ds=ds),
        "current_best_downstream_71_lr3e4": inspect_scope(config_json=CURRENT_SKIP71_CONFIG_JSON, ckpt_in=CURRENT_REPLACE_CKPT, ds=ds),
    }
    write_json(OUT_ROOT / "reference_scope_summary.json", reference_scope)

    references = load_references()
    summary: Dict[str, Any] = {
        "run_date": RUN_DATE,
        "inputs": {
            "basetrain_config": str(FIXED_BASETRAIN_CONFIG),
            "stage6_winner": str(FIXED_STAGE6_WINNER),
            "source_70a_winner": str(FIXED_70A_WINNER),
            "current_replace_ckpt": str(CURRENT_REPLACE_CKPT),
            "current_best_downstream_ckpt": str(CURRENT_SKIP71_CKPT),
            "sic_diagnostic_summary": str(SIC_DIAGNOSTIC_SUMMARY),
            "current_scope_audit": str(CURRENT_SCOPE_AUDIT_SUMMARY),
            "current_replace_redesign_summary": str(CURRENT_REPLACE_REDESIGN_SUMMARY),
        },
        "policy": {
            "experiment": "Stage6(last) -> cleanup-with-early-live-bias(5x60) -> skip-70R -> 71(lr=3e-4)",
            "no_path_selection": True,
            "main_lr_policy": {"cleanup_main_lr": 1e-4, "cleanup_backup_lr": 2e-4},
            "source_cleanup_base": "exact generated source 70a config from fixed ep014center Stage6 winner",
            "source_downstream_base": "exact generated current best downstream 71 config",
            "selected_early_bias_layer": {
                "semantic_layer": "direct rollout loss contract on live leg omega residual",
                "keys": [
                    "direct_pose_leg_align_weight",
                    "direct_pose_leg_align_mode",
                    "direct_pose_leg_align_oracle_min_deg",
                    "direct_pose_leg_align_schedule",
                    "direct_pose_leg_align_start_weight",
                    "direct_pose_leg_align_warmup_steps",
                    "direct_pose_leg_align_ramp_steps",
                ],
                "reason": (
                    "These keys are actively parsed in train/posttrain.py and only affect training loss, "
                    "so they can inject broad cleanup bias from step 0 without reviving retired phase_z plumbing."
                ),
            },
            "active_key_audit": {
                "legacy_dead": ["direct_pose_use_phase_z", "direct_pose_phase_z_mode"],
                "model_construction_or_grad_path": [
                    "direct_pose_split_enable",
                    "direct_pose_arm_split_enable",
                    "direct_pose_leg_enable",
                    "direct_pose_leg_mode",
                    "direct_pose_leg_stopgrad_main",
                    "direct_pose_leg_detach_feat",
                    "direct_pose_leg_gate_mode",
                    "direct_pose_leg_scale_log_clip",
                    "direct_pose_leg_scale_clamp_k",
                ],
                "optimizer_scope": [
                    "train_direct_pose",
                    "direct_pose_leg_train_only",
                    "direct_pose_leg_gate_train_only",
                    "direct_pose_nonleg_train_only",
                    "optimizer_param_group_overrides",
                ],
                "loss_contract": [
                    "direct_pose_leg_align_*",
                    "direct_pose_leg_gate_sup_weight",
                    "direct_pose_loss_leg_split",
                    "direct_pose_loss_group_norm_*",
                ],
            },
        },
        "code_review": {
            "phase_flags_replace_config": raw_and_parsed_phase_flags(CURRENT_REPLACE_CONFIG_JSON),
            "phase_flags_downstream_config": raw_and_parsed_phase_flags(CURRENT_SKIP71_CONFIG_JSON),
            "checkpoint_phase_flags": {
                "source_70a": checkpoint_phase_flags(FIXED_70A_WINNER),
                "current_replace": checkpoint_phase_flags(CURRENT_REPLACE_CKPT),
                "current_best_downstream": checkpoint_phase_flags(CURRENT_SKIP71_CKPT),
            },
            "reference_scope_summary": str(OUT_ROOT / "reference_scope_summary.json"),
            "coupled_cleanup_base": reference_scope["source_70a"],
        },
        "references": references,
        "candidates": {},
    }

    for candidate in candidates:
        case_name = str(candidate["name"])
        log(f"=== run cleanup candidate {case_name} ===")
        cleanup_payload = run_cleanup_candidate(candidate=candidate, ds=ds, lane_log=lane_log)
        cleanup_selected = cleanup_payload["eval"]["selected_metrics"]
        cleanup_payload["delta_vs_source_70a"] = metric_delta(
            cleanup_selected,
            references["source_70a"]["eval"]["selected_metrics"],
        )
        cleanup_payload["delta_vs_current_replace_lr5e5"] = metric_delta(
            cleanup_selected,
            references["current_replace_lr5e5"]["eval"]["selected_metrics"],
        )
        cleanup_payload["delta_vs_docs_new70b_replace_lowdrift"] = metric_delta(
            cleanup_selected,
            references["docs_baseline_new70b_replace_lowdrift"]["eval"]["selected_metrics"],
        )

        log(f"=== run downstream 71 for {case_name} ===")
        downstream_payload = run_downstream_candidate(
            candidate=candidate,
            cleanup_ckpt=Path(str(cleanup_payload["ckpt"])),
            ds=ds,
            lane_log=lane_log,
        )
        downstream_selected = downstream_payload["eval"]["selected_metrics"]
        downstream_payload["delta_vs_current_best_downstream"] = metric_delta(
            downstream_selected,
            references["current_best_downstream_71_lr3e4"]["eval"]["selected_metrics"],
        )
        downstream_payload["delta_vs_docs_71_lr3e4"] = metric_delta(
            downstream_selected,
            references["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"],
        )
        downstream_payload["delta_vs_plain71_best"] = metric_delta(
            downstream_selected,
            references["plain71_best_case"]["eval"]["selected_metrics"],
        )
        downstream_payload["delta_vs_source_70a"] = metric_delta(
            downstream_selected,
            references["source_70a"]["eval"]["selected_metrics"],
        )

        log(f"=== build SIC summary for {case_name} ===")
        sic_payload = build_sic_summary(
            candidate_name=case_name,
            references=references,
            cleanup_eval_json=Path(str(cleanup_payload["eval"]["paths"]["eval_json"])),
            downstream_eval_json=Path(str(downstream_payload["eval"]["paths"]["eval_json"])),
        )
        sic_json = SIC_ROOT / f"{case_name}_summary.json"
        sic_md = SIC_ROOT / f"{case_name}_summary.md"
        write_json(sic_json, sic_payload)
        sic_md.write_text(build_sic_markdown(sic_payload) + "\n", encoding="utf-8")

        summary["candidates"][case_name] = {
            "role": str(candidate["role"]),
            "kind": str(candidate["kind"]),
            "hypothesis": str(candidate["hypothesis"]),
            "cleanup_stage": cleanup_payload,
            "downstream_71": downstream_payload,
            "sic": {
                "summary_json": str(sic_json),
                "summary_md": str(sic_md),
            },
        }

    write_json(
        OUT_ROOT / "status.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candidates": list(summary["candidates"].keys()),
        },
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
