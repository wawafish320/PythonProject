#!/usr/bin/env python3
from __future__ import annotations

import json
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


RUN_DATE = "20260328"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_replace_redesign_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_ep014center_replace_redesign_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SCOPE_ROOT = OUT_ROOT / "scope"
SIC_ROOT = OUT_ROOT / "sic"

FIXED_BASETRAIN_CONFIG = ROOT / "config" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_seed2024_20260324.json"
FIXED_STAGE6_WINNER = ROOT / "models" / "__tmp_ep014center_main_anchor_check" / "tmp_ep014center_main_selector_check__last" / "ckpt_last_tmp_ep014center_main_selector_check__last_stage6_anchor_ep014center_main_check.pth"
FIXED_70A_WINNER = ROOT / "models" / "__tmp_ep014center_70a_lowlr_sweep_20260328" / "lr3e4" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth"

SOURCE_70A_ARTIFACTS_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "artifacts.json"
CURRENT_REPLACE_BASE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_70alr3e4_lowdrift_20260328" / "summary.json"
CURRENT_REPLACE_WARMSTART_REPORT_JSON = ROOT / "debug_output" / "_tmp_ep014center_70alr3e4_lowdrift_20260328" / "warmstart" / "replace_zerophase_report.json"
CURRENT_REPLACE_SWEEP_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "summary.json"
CURRENT_DOWNSTREAM_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "summary.json"
PLAIN71_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_ep014center_70a_to71_plain_leg_cleanup_20260328" / "summary.json"
DOCS_BASELINE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "manual_summary.json"
SIC_DIAGNOSTIC_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_70a_vs_baseline_sic_profile_20260328" / "summary.md"
CURRENT_SCOPE_AUDIT_SUMMARY = ROOT / "debug_output" / "_tmp_ep014center_per_head_merged_stage_20260328" / "reference_scope_summary.json"

CURRENT_REPLACE_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "configs" / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
CURRENT_DOWNSTREAM_CONFIG_JSON = ROOT / "debug_output" / "_tmp_ep014center_replace_lr5e5_to71_lr3e4_20260328" / "configs" / "posttrain_71_lr3e4_from_ep014center_replace_lr5e5_20260328.json"
BASE_71_CONFIG_JSON = ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json"

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

DEFAULT_REPLACE_LR = 5e-5
DEFAULT_REPLACE_EPOCHS = 1
DEFAULT_REPLACE_STEPS = 60
DEFAULT_DOWNSTREAM_LR = 3e-4
DEFAULT_DOWNSTREAM_EPOCHS = 3
DEFAULT_DOWNSTREAM_STEPS = 60

CANDIDATES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "decouple",
        "hypothesis": (
            "Keep current replace trainable scope/budget, but set "
            "`direct_pose_leg_stopgrad_main=true` and `direct_pose_leg_detach_feat=true` "
            "so leg cleanup becomes a true residual calibrator on top of fixed 70a main-leg output, "
            "instead of letting diffuse leg error back-drive the shared direct trunk."
        ),
        "replace_overrides": {
            "lr": DEFAULT_REPLACE_LR,
            "epochs": DEFAULT_REPLACE_EPOCHS,
            "steps_per_epoch": DEFAULT_REPLACE_STEPS,
            "direct_pose_leg_stopgrad_main": True,
            "direct_pose_leg_detach_feat": True,
        },
    },
    {
        "name": "decouple_align3",
        "hypothesis": (
            "Start from the decoupled residual calibrator and add a mild `proj` oracle-align prior "
            "(`direct_pose_leg_align_weight=3`) with a short linear ramp, so the leg omega head gets "
            "dense direction/magnitude supervision across many bins instead of only reacting to sparse geodesic spikes."
        ),
        "replace_overrides": {
            "lr": DEFAULT_REPLACE_LR,
            "epochs": DEFAULT_REPLACE_EPOCHS,
            "steps_per_epoch": DEFAULT_REPLACE_STEPS,
            "direct_pose_leg_stopgrad_main": True,
            "direct_pose_leg_detach_feat": True,
            "direct_pose_leg_align_weight": 3.0,
            "direct_pose_leg_align_oracle_min_deg": 0.5,
            "direct_pose_leg_align_mode": "proj",
            "direct_pose_leg_align_schedule": "linear",
            "direct_pose_leg_align_start_weight": 0.0,
            "direct_pose_leg_align_warmup_steps": 10,
            "direct_pose_leg_align_ramp_steps": 20,
        },
    },
)

UNRUN_HYPOTHESES: Tuple[str, ...] = (
    "If both decoupled variants fail, the next low-cost fallback is a decoupled replace with `direct_pose_leg_gate_mode=\"scale\"` to add per-joint magnitude calibration without reopening retired SIC-focus branches.",
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
    payload = load_json(BASE_71_CONFIG_JSON)
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
    payload["encoder_bundle"] = str(ENCODER_BUNDLE)
    payload["posttrain_contacts_source"] = "pretrain_contact"
    payload["posttrain_contacts_pretrain_clamp"] = PRETRAIN_CLAMP
    payload["posttrain_contacts_pretrain_affine_stats"] = str(AFFINE_STATS)
    cfg = posttrain._cfg_from_payload(payload)
    (
        model,
        direct_pose_feat_source,
        direct_pose_time_pe_dim,
        direct_pose_time_pe_base,
        direct_pose_split_enable,
        direct_pose_nonleg_proj_dim,
        direct_pose_leg_gate_mode_model,
        direct_pose_leg_gate_power_model,
    ) = posttrain._build_posttrain_model_from_ckpt(cfg=cfg, ds=ds, device=torch.device("cpu"))
    posttrain._freeze_all(model)
    train_mode = posttrain._resolve_train_mode(cfg)
    posttrain._unfreeze_for_train_mode(model, cfg, train_mode)
    params, names = posttrain._select_trainable_params(model)
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
            "direct_pose_feat_source": str(direct_pose_feat_source),
            "direct_pose_time_pe_dim": int(direct_pose_time_pe_dim),
            "direct_pose_time_pe_base": float(direct_pose_time_pe_base),
            "direct_pose_split_enable": bool(direct_pose_split_enable),
            "direct_pose_nonleg_proj_dim": int(direct_pose_nonleg_proj_dim),
            "direct_pose_meas_mode_model": str(getattr(model, "direct_pose_meas_mode", "unknown")),
            "direct_pose_leg_mode_model": str(getattr(model, "direct_pose_leg_mode", "unknown")),
            "direct_pose_leg_stopgrad_main_model": bool(getattr(model, "direct_pose_leg_stopgrad_main", False)),
            "direct_pose_leg_detach_feat_model": bool(getattr(model, "direct_pose_leg_detach_feat", False)),
            "direct_pose_leg_gate_mode_model": str(direct_pose_leg_gate_mode_model),
            "direct_pose_leg_gate_power_model": float(direct_pose_leg_gate_power_model),
            "direct_pose_loss_leg_split_cfg": bool(getattr(cfg, "direct_pose_loss_leg_split", False)),
            "direct_pose_loss_group_norm_enable_cfg": bool(getattr(cfg, "direct_pose_loss_group_norm_enable", False)),
            "direct_pose_loss_group_norm_w_leg_cfg": float(getattr(cfg, "direct_pose_loss_group_norm_w_leg", 1.0)),
            "direct_pose_loss_group_norm_w_nonleg_cfg": float(getattr(cfg, "direct_pose_loss_group_norm_w_nonleg", 1.0)),
            "direct_pose_leg_align_weight_cfg": float(getattr(cfg, "direct_pose_leg_align_weight", 0.0)),
            "direct_pose_leg_align_mode_cfg": str(getattr(cfg, "direct_pose_leg_align_mode", "cos")),
            "direct_pose_nonleg_focus_bones_cfg": str(getattr(cfg, "direct_pose_nonleg_focus_bones", "") or ""),
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
    current_replace_base = load_json(CURRENT_REPLACE_BASE_SUMMARY_JSON)
    current_replace = load_json(CURRENT_REPLACE_SWEEP_SUMMARY_JSON)["cases"]["lr5e5"]
    current_downstream = load_json(CURRENT_DOWNSTREAM_SUMMARY_JSON)["candidate_71"]
    plain71_summary = load_json(PLAIN71_SUMMARY_JSON)
    plain71_best_case = str(plain71_summary["answers"]["best_case_by_all_ex_root"])
    plain71_best = plain71_summary["cases"][plain71_best_case]
    docs_summary = load_json(DOCS_BASELINE_SUMMARY_JSON)["stage_progress_model_source"]
    return {
        "source_70a": {
            "config": str(source_70a["config_json"]),
            "ckpt": str(source_70a["ckpt"]),
            "eval": collect_eval(Path(str(source_70a["eval_json"])), Path(str(source_70a["group_json"]))),
        },
        "current_replace": {
            "config": str(current_replace["config"]),
            "ckpt": str(current_replace["last_ckpt"]),
            "warmstart_ckpt": str(current_replace_base["warmstart_ckpt"]),
            "warmstart_report": str(CURRENT_REPLACE_WARMSTART_REPORT_JSON),
            "eval": current_replace["eval"],
        },
        "current_best_downstream": {
            "config": str(current_downstream["config"]),
            "ckpt": str(current_downstream["ckpt"]),
            "eval": current_downstream["eval"],
        },
        "plain71_best": {
            "case": plain71_best_case,
            "config": str(plain71_best["config"]),
            "ckpt": str(plain71_best["ckpt"]),
            "eval": plain71_best["eval"],
        },
        "docs_baseline_replace": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["new70b_replace_lowdrift"]),
        },
        "docs_baseline_71_lr3e4": {
            "summary": str(DOCS_BASELINE_SUMMARY_JSON),
            "eval": stage_row_to_eval(docs_summary["71_lr3e4"]),
        },
    }


def build_code_review(*, ds: Any, references: Mapping[str, Any]) -> Dict[str, Any]:
    current_replace_payload = load_json(CURRENT_REPLACE_CONFIG_JSON)
    current_downstream_payload = load_json(CURRENT_DOWNSTREAM_CONFIG_JSON)
    parsed_replace_cfg = posttrain._cfg_from_payload(current_replace_payload)
    parsed_downstream_cfg = posttrain._cfg_from_payload(current_downstream_payload)
    replace_scope = inspect_scope(
        config_json=CURRENT_REPLACE_CONFIG_JSON,
        ckpt_in=Path(str(references["current_replace"]["warmstart_ckpt"])),
        ds=ds,
    )
    downstream_scope = inspect_scope(
        config_json=CURRENT_DOWNSTREAM_CONFIG_JSON,
        ckpt_in=Path(str(references["current_replace"]["ckpt"])),
        ds=ds,
    )
    warmstart_report_path = Path(str(references["current_replace"]["warmstart_report"]))
    warmstart_report = load_json(warmstart_report_path) if warmstart_report_path.is_file() else {}

    live_replace_keys = {
        "train_mode": str(replace_scope["train_mode"]),
        "train_direct_pose": bool(getattr(parsed_replace_cfg, "train_direct_pose", False)),
        "direct_pose_feat_source": str(replace_scope["resolved"]["direct_pose_feat_source"]),
        "direct_pose_meas_mode_model": str(replace_scope["resolved"]["direct_pose_meas_mode_model"]),
        "direct_pose_time_pe_dim": int(replace_scope["resolved"]["direct_pose_time_pe_dim"]),
        "direct_pose_split_enable": bool(replace_scope["resolved"]["direct_pose_split_enable"]),
        "direct_pose_nonleg_proj_dim": int(replace_scope["resolved"]["direct_pose_nonleg_proj_dim"]),
        "direct_pose_arm_split_enable": bool(getattr(parsed_replace_cfg, "direct_pose_arm_split_enable", False)),
        "direct_pose_leg_enable": bool(getattr(parsed_replace_cfg, "direct_pose_leg_enable", False)),
        "direct_pose_leg_mode": str(replace_scope["resolved"]["direct_pose_leg_mode_model"]),
        "direct_pose_leg_stopgrad_main": bool(replace_scope["resolved"]["direct_pose_leg_stopgrad_main_model"]),
        "direct_pose_leg_detach_feat": bool(replace_scope["resolved"]["direct_pose_leg_detach_feat_model"]),
        "direct_pose_leg_gate_mode": str(replace_scope["resolved"]["direct_pose_leg_gate_mode_model"]),
        "direct_pose_leg_align_weight": float(replace_scope["resolved"]["direct_pose_leg_align_weight_cfg"]),
        "direct_pose_loss_leg_split": bool(replace_scope["resolved"]["direct_pose_loss_leg_split_cfg"]),
        "direct_pose_loss_group_norm_enable": bool(replace_scope["resolved"]["direct_pose_loss_group_norm_enable_cfg"]),
        "direct_pose_loss_group_norm_w_leg": float(replace_scope["resolved"]["direct_pose_loss_group_norm_w_leg_cfg"]),
        "direct_pose_loss_group_norm_w_nonleg": float(replace_scope["resolved"]["direct_pose_loss_group_norm_w_nonleg_cfg"]),
        "direct_pose_nonleg_focus_bones": str(replace_scope["resolved"]["direct_pose_nonleg_focus_bones_cfg"]),
        "epochs": int(getattr(parsed_replace_cfg, "epochs", 0)),
        "steps_per_epoch": int(getattr(parsed_replace_cfg, "steps_per_epoch", 0)),
        "lr": float(getattr(parsed_replace_cfg, "lr", float("nan"))),
    }

    live_mechanisms = [
        "Current replace is `train_mode=direct`; no lambda/contact shell is trainable.",
        "The live direct path uses `direct_pose_feat_source=cond` plus 32-d time PE; the model-side `direct_pose_meas_mode` stays `concat`.",
        "Replace trains the split direct trunk and readouts together: `direct_pose_head`, `direct_pose_out_leg`, `direct_pose_out_arm`, `direct_pose_out_else`, `direct_pose_arm_proj`, `direct_pose_else_proj`, and `direct_pose_leg_head`.",
        "Leg cleanup is an on-manifold SO(3) omega head (`direct_pose_leg_mode=so3`) composed inside posttrain/eval; current replace does not stop-grad the main leg base and does not detach leg-head features.",
        "The direct objective is live in leg/non-leg split form (`direct_pose_loss_leg_split=true`) with EMA group-norm enabled and symmetric weights (`w_leg=1`, `w_nonleg=1`).",
        "Arm/else split is active inside the non-leg branch (`direct_pose_arm_split_enable=true`), but there is no extra non-leg focus mask in current replace.",
        "The shared `replace_zerophase` warmstart is not a semantic transform: the current helper just copies the 70a checkpoint unchanged before running replace.",
        "Current downstream 71 is a different live mechanism: only the 6 tensors in `direct_pose_leg_head` are trainable, with `stopgrad_main=true` and `detach_feat=true`.",
    ]

    active_inductive_biases = [
        "Because the replace loss is only the end geodesic objective plus symmetric leg/non-leg group norm, the optimizer is driven by whichever bins produce the largest immediate loss reductions.",
        "With leg loss still coupled into the main direct path, leg corrections can back-drive the shared trunk/readouts instead of staying in the residual omega head.",
        "No dense oracle prior is active on leg omega (`direct_pose_leg_align_weight=0`, `gate_mode=none`), so the residual head has no explicit bias toward broad low-amplitude correction.",
        "Replace still updates non-leg branches in the same stage, so sparse high-gain bins can win the optimization budget over smaller diffuse leg cleanup.",
    ]

    baseline_concentrated_spike_biases = [
        "Coupled trunk + leg-head training is naturally good at chasing a few high-amplitude hotspots, because shared updates can move many outputs at once.",
        "Symmetric leg/non-leg group norm equalizes aggregate group magnitudes, not SIC spread, so it does not reward covering a wider 57-64 band over winning a smaller set of taller spikes.",
        "No stop-grad/detach means the system can rewrite the main leg base rather than adding a local corrective omega, which matches concentrated cleanup better than diffuse residual polish.",
    ]

    diffuse_mismatch_points = [
        "For the new-chain 57-64 diffuse hotspot, current replace has no mechanism that says 'keep 70a main output fixed and only apply small residuals where needed'.",
        "Because SIC-focus is retired in mainline, the only live way to better fit diffuse bins is to change residual coupling/objective semantics, not to re-enable old phase-bin branches.",
        "The current replace config still carries `direct_pose_use_phase_z` / `direct_pose_phase_z_mode`, but parser/model ignore them, so they cannot be the reason current replace works or fails.",
    ]

    return {
        "legacy_phase_keys_in_replace_config": {
            "direct_pose_use_phase_z": current_replace_payload.get("direct_pose_use_phase_z", None),
            "direct_pose_phase_z_mode": current_replace_payload.get("direct_pose_phase_z_mode", None),
        },
        "legacy_phase_keys_in_downstream_config": {
            "direct_pose_use_phase_z": current_downstream_payload.get("direct_pose_use_phase_z", None),
            "direct_pose_phase_z_mode": current_downstream_payload.get("direct_pose_phase_z_mode", None),
        },
        "legacy_phase_keys_active_in_parsed_cfg": {
            "replace_has_direct_pose_use_phase_z": hasattr(parsed_replace_cfg, "direct_pose_use_phase_z"),
            "replace_has_direct_pose_phase_z_mode": hasattr(parsed_replace_cfg, "direct_pose_phase_z_mode"),
            "downstream_has_direct_pose_use_phase_z": hasattr(parsed_downstream_cfg, "direct_pose_use_phase_z"),
            "downstream_has_direct_pose_phase_z_mode": hasattr(parsed_downstream_cfg, "direct_pose_phase_z_mode"),
        },
        "checkpoint_posttrain_cfg_flags": {
            "source_70a": checkpoint_posttrain_cfg_flags(FIXED_70A_WINNER),
            "current_replace": checkpoint_posttrain_cfg_flags(Path(str(references["current_replace"]["ckpt"]))),
            "current_best_downstream": checkpoint_posttrain_cfg_flags(Path(str(references["current_best_downstream"]["ckpt"]))),
        },
        "replace_scope": replace_scope,
        "downstream_scope": downstream_scope,
        "warmstart_report": warmstart_report,
        "live_replace_keys": live_replace_keys,
        "live_mechanisms": live_mechanisms,
        "active_inductive_biases": active_inductive_biases,
        "baseline_concentrated_spike_biases": baseline_concentrated_spike_biases,
        "diffuse_mismatch_points": diffuse_mismatch_points,
    }


def run_replace_candidate(
    *,
    candidate: Mapping[str, Any],
    warmstart_ckpt: Path,
    lane_log: Path,
    ds: Any,
) -> Dict[str, Any]:
    case_name = str(candidate["name"])
    replace_dir = MODEL_ROOT / case_name / "replace"
    replace_run_name = f"WalkF_stage7_70b_replace_redesign_{case_name}_from_ep014center_70alr3e4_{RUN_DATE}"
    replace_cfg = CONFIG_ROOT / f"posttrain_70b_replace_redesign_{case_name}_{RUN_DATE}.json"
    replace_eval_dir = OUT_ROOT / "replace_eval" / case_name
    replace_group_json = OUT_ROOT / "replace_eval" / f"{case_name}_group_summary.json"

    replace_overrides = {
        "ckpt_in": str(warmstart_ckpt),
        "out_dir": str(replace_dir),
        "run_name": replace_run_name,
        "lr": DEFAULT_REPLACE_LR,
        "epochs": DEFAULT_REPLACE_EPOCHS,
        "steps_per_epoch": DEFAULT_REPLACE_STEPS,
    }
    replace_overrides.update(dict(candidate["replace_overrides"]))
    make_generated_config(CURRENT_REPLACE_CONFIG_JSON, replace_cfg, replace_overrides)

    replace_scope = inspect_scope(config_json=replace_cfg, ckpt_in=warmstart_ckpt, ds=ds)
    replace_ckpt = run_posttrain_stage(
        config=replace_cfg,
        ckpt_in=warmstart_ckpt,
        out_dir=replace_dir,
        run_name=replace_run_name,
        log_file=lane_log,
    )
    replace_eval_json = run_eval(
        model_ckpt=replace_ckpt,
        out_dir=replace_eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(replace_eval_json, replace_group_json, log_file=lane_log)
    replace_eval = collect_eval(replace_eval_json, replace_group_json)

    downstream_dir = MODEL_ROOT / case_name / "71"
    downstream_run_name = f"WalkF_stage7_71_from_replace_redesign_{case_name}_{RUN_DATE}"
    downstream_cfg = CONFIG_ROOT / f"posttrain_71_from_replace_redesign_{case_name}_{RUN_DATE}.json"
    downstream_eval_dir = OUT_ROOT / "downstream_eval" / case_name
    downstream_group_json = OUT_ROOT / "downstream_eval" / f"{case_name}_group_summary.json"

    make_generated_config(
        CURRENT_DOWNSTREAM_CONFIG_JSON,
        downstream_cfg,
        {
            "ckpt_in": str(replace_ckpt),
            "out_dir": str(downstream_dir),
            "run_name": downstream_run_name,
            "lr": DEFAULT_DOWNSTREAM_LR,
            "epochs": DEFAULT_DOWNSTREAM_EPOCHS,
            "steps_per_epoch": DEFAULT_DOWNSTREAM_STEPS,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    downstream_scope = inspect_scope(config_json=downstream_cfg, ckpt_in=replace_ckpt, ds=ds)
    downstream_ckpt = run_posttrain_stage(
        config=downstream_cfg,
        ckpt_in=replace_ckpt,
        out_dir=downstream_dir,
        run_name=downstream_run_name,
        log_file=lane_log,
    )
    downstream_eval_json = run_eval(
        model_ckpt=downstream_ckpt,
        out_dir=downstream_eval_dir,
        contacts_source="model",
        log_file=lane_log,
    )
    ensure_group_summary(downstream_eval_json, downstream_group_json, log_file=lane_log)
    downstream_eval = collect_eval(downstream_eval_json, downstream_group_json)

    return {
        "name": case_name,
        "hypothesis": str(candidate["hypothesis"]),
        "replace": {
            "config": str(replace_cfg),
            "ckpt": str(replace_ckpt),
            "eval": replace_eval,
            "scope": replace_scope,
        },
        "downstream": {
            "config": str(downstream_cfg),
            "ckpt": str(downstream_ckpt),
            "eval": downstream_eval,
            "scope": downstream_scope,
        },
    }


def _per_sic_subset_profile(
    path: Path,
    *,
    bones: Sequence[str],
    cycle_gte: int,
    drop_wrap: bool,
) -> Dict[str, Any]:
    obj = load_json(path)
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise RuntimeError(f"invalid freerun json: {path}")
    names = [str(x) for x in per.get("bone_names", [])]
    mat = per.get("DirectGeoLocalDeg", [])
    if not names or not isinstance(mat, list):
        raise RuntimeError(f"missing direct geolocal matrix in {path}")
    cycle_len = int(obj.get("cycle_len", 0) or 0)
    if cycle_len <= 0:
        raise RuntimeError(f"invalid cycle_len in {path}")
    want = {str(name) for name in bones}
    indices = [i for i, name in enumerate(names) if name in want]
    if not indices:
        raise RuntimeError(f"missing requested bones={list(bones)} in {path}")
    grouped = {sic: [] for sic in range(cycle_len)}
    for step_i, step in enumerate(steps):
        cycle = int(step.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(drop_wrap) and bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", 0) or 0)
        if step_i >= len(mat):
            continue
        row = mat[step_i]
        if not isinstance(row, list):
            continue
        values = []
        for joint_i in indices:
            if joint_i >= len(row):
                continue
            value = safe_float(row[joint_i])
            if math.isfinite(value):
                values.append(value)
        if values:
            grouped[sic].append(float(np.mean(np.asarray(values, dtype=np.float64))))
    arr = [
        float(np.mean(np.asarray(grouped[sic], dtype=np.float64))) if grouped[sic] else float("nan")
        for sic in range(cycle_len)
    ]
    return {
        "source": str(path),
        "bones": list(bones),
        "indices": indices,
        "cycle_len": cycle_len,
        "mask": {"cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
        "profile": arr,
    }


def _window_mean(arr: Sequence[float], start: int, end: int) -> float:
    vals = [safe_float(arr[idx]) for idx in range(int(start), int(end) + 1)]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _window_delta(before: Sequence[float], after: Sequence[float], start: int, end: int) -> float:
    a = _window_mean(before, start, end)
    b = _window_mean(after, start, end)
    return diff(b, a)


def _outside_window_delta(before: Sequence[float], after: Sequence[float], start: int, end: int) -> float:
    pairs = [
        (safe_float(before[idx]), safe_float(after[idx]))
        for idx in range(min(len(before), len(after)))
        if idx < int(start) or idx > int(end)
    ]
    valid = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if not valid:
        return float("nan")
    before_mean = float(np.mean(np.asarray([a for a, _ in valid], dtype=np.float64)))
    after_mean = float(np.mean(np.asarray([b for _, b in valid], dtype=np.float64)))
    return diff(after_mean, before_mean)


def build_best_replace_sic_summary(
    *,
    best_candidate_name: str,
    best_replace_eval_json: Path,
    references: Mapping[str, Any],
) -> Dict[str, Any]:
    source_eval_json = Path(str(references["source_70a"]["eval"]["paths"]["eval_json"]))
    current_replace_eval_json = Path(str(references["current_replace"]["eval"]["paths"]["eval_json"]))
    current_best_downstream_eval_json = Path(str(references["current_best_downstream"]["eval"]["paths"]["eval_json"]))

    profiles = {
        "source_70a": _per_sic_group_profile(source_eval_json, cycle_gte=1, drop_wrap=True),
        "current_replace": _per_sic_group_profile(current_replace_eval_json, cycle_gte=1, drop_wrap=True),
        "current_best_downstream": _per_sic_group_profile(current_best_downstream_eval_json, cycle_gte=1, drop_wrap=True),
        "candidate_replace": _per_sic_group_profile(best_replace_eval_json, cycle_gte=1, drop_wrap=True),
    }
    subset_profiles = {
        "foot_l_ball_l": {
            "source_70a": _per_sic_subset_profile(source_eval_json, bones=("foot_l", "ball_l"), cycle_gte=1, drop_wrap=True),
            "current_replace": _per_sic_subset_profile(current_replace_eval_json, bones=("foot_l", "ball_l"), cycle_gte=1, drop_wrap=True),
            "current_best_downstream": _per_sic_subset_profile(current_best_downstream_eval_json, bones=("foot_l", "ball_l"), cycle_gte=1, drop_wrap=True),
            "candidate_replace": _per_sic_subset_profile(best_replace_eval_json, bones=("foot_l", "ball_l"), cycle_gte=1, drop_wrap=True),
        },
        "calf_r": {
            "source_70a": _per_sic_subset_profile(source_eval_json, bones=("calf_r",), cycle_gte=1, drop_wrap=True),
            "current_replace": _per_sic_subset_profile(current_replace_eval_json, bones=("calf_r",), cycle_gte=1, drop_wrap=True),
            "current_best_downstream": _per_sic_subset_profile(current_best_downstream_eval_json, bones=("calf_r",), cycle_gte=1, drop_wrap=True),
            "candidate_replace": _per_sic_subset_profile(best_replace_eval_json, bones=("calf_r",), cycle_gte=1, drop_wrap=True),
        },
    }

    summary: Dict[str, Any] = {
        "best_candidate_name": best_candidate_name,
        "artifacts": {
            "source_70a_eval_json": str(source_eval_json),
            "current_replace_eval_json": str(current_replace_eval_json),
            "current_best_downstream_eval_json": str(current_best_downstream_eval_json),
            "candidate_replace_eval_json": str(best_replace_eval_json),
        },
        "groups": {},
        "focus_profiles": {},
        "answers": {},
    }

    for group_name in SIC_GROUPS:
        source_profile = profiles["source_70a"]["profiles"][group_name]
        current_replace_profile = profiles["current_replace"]["profiles"][group_name]
        current_best_downstream_profile = profiles["current_best_downstream"]["profiles"][group_name]
        candidate_profile = profiles["candidate_replace"]["profiles"][group_name]
        summary["groups"][group_name] = {
            "source_70a": _profile_stats(source_profile),
            "current_replace": _profile_stats(current_replace_profile),
            "current_best_downstream": _profile_stats(current_best_downstream_profile),
            "candidate_replace": _profile_stats(candidate_profile),
            "candidate_delta_vs_source_70a": _delta_summary(source_profile, candidate_profile),
            "candidate_delta_vs_current_replace": _delta_summary(current_replace_profile, candidate_profile),
            "candidate_delta_vs_current_best_downstream": _delta_summary(current_best_downstream_profile, candidate_profile),
            "candidate_window_delta_vs_source_57_64": _window_delta(source_profile, candidate_profile, 57, 64),
            "candidate_window_delta_vs_source_rest_ex57_64": _outside_window_delta(
                source_profile, candidate_profile, 57, 64
            ),
        }

    for focus_name, focus_map in subset_profiles.items():
        src = focus_map["source_70a"]["profile"]
        cur = focus_map["current_replace"]["profile"]
        skip71 = focus_map["current_best_downstream"]["profile"]
        cand = focus_map["candidate_replace"]["profile"]
        summary["focus_profiles"][focus_name] = {
            "source_70a": _profile_stats(src),
            "current_replace": _profile_stats(cur),
            "current_best_downstream": _profile_stats(skip71),
            "candidate_replace": _profile_stats(cand),
            "candidate_delta_vs_source_70a": _delta_summary(src, cand),
            "candidate_delta_vs_current_replace": _delta_summary(cur, cand),
            "candidate_delta_vs_current_best_downstream": _delta_summary(skip71, cand),
            "candidate_window_12_15_delta_vs_source_70a": _window_delta(src, cand, 12, 15),
            "candidate_window_12_15_delta_vs_current_replace": _window_delta(cur, cand, 12, 15),
            "candidate_window_57_64_delta_vs_source_70a": _window_delta(src, cand, 57, 64),
            "candidate_window_57_64_delta_vs_current_replace": _window_delta(cur, cand, 57, 64),
        }

    all_ex_root = summary["groups"]["all_ex_root"]
    leg = summary["groups"]["leg"]
    foot = summary["focus_profiles"]["foot_l_ball_l"]
    summary["answers"] = {
        "candidate_cleans_57_64_more_than_rest_all_ex_root": safe_float(
            all_ex_root["candidate_window_delta_vs_source_57_64"]
        ) < safe_float(all_ex_root["candidate_window_delta_vs_source_rest_ex57_64"]),
        "candidate_window_delta_vs_source_57_64_all_ex_root": safe_float(all_ex_root["candidate_window_delta_vs_source_57_64"]),
        "candidate_window_delta_vs_source_rest_ex57_64_all_ex_root": safe_float(
            all_ex_root["candidate_window_delta_vs_source_rest_ex57_64"]
        ),
        "candidate_window_delta_vs_source_57_64_leg": safe_float(leg["candidate_window_delta_vs_source_57_64"]),
        "candidate_window_delta_vs_source_57_64_nonleg": safe_float(summary["groups"]["nonleg"]["candidate_window_delta_vs_source_57_64"]),
        "candidate_window_delta_vs_source_57_64_arm": safe_float(summary["groups"]["arm"]["candidate_window_delta_vs_source_57_64"]),
        "foot_cost_top_regress_sics_vs_current_replace": foot["candidate_delta_vs_current_replace"]["worst_regress_sics"][:8],
        "foot_cost_top_regress_sics_vs_source_70a": foot["candidate_delta_vs_source_70a"]["worst_regress_sics"][:8],
    }

    lines = [
        f"# SIC compare for best redesigned replace: {best_candidate_name}",
        "",
        "- mask: `cycle>=1`, `drop_wrap=true`",
        "",
        "## Group SIC Table",
        "",
        "| group | src70a top SICs | current replace top SICs | current best downstream71 top SICs | candidate replace top SICs | cand d57-64 vs src | cand d(top8-src) | cand d(rest!=57-64 vs src) |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for group_name in ("all_ex_root", "leg", "nonleg", "arm"):
        row = summary["groups"][group_name]
        lines.append(
            f"| {group_name} | {row['source_70a'].get('top_sics', [])[:8]} | "
            f"{row['current_replace'].get('top_sics', [])[:8]} | "
            f"{row['current_best_downstream'].get('top_sics', [])[:8]} | "
            f"{row['candidate_replace'].get('top_sics', [])[:8]} | "
            f"{fmt(row['candidate_window_delta_vs_source_57_64'])} | "
            f"{fmt(row['candidate_delta_vs_source_70a'].get('mean_delta_on_own_top_before'))} | "
            f"{fmt(row['candidate_window_delta_vs_source_rest_ex57_64'])} |"
        )
    lines.extend(
        [
            "",
            "## Foot/Calf Focus",
            "",
            "| focus | candidate d12-15 vs src | candidate d12-15 vs current replace | candidate d57-64 vs src | candidate worst regress SICs vs current replace |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for focus_name in ("foot_l_ball_l", "calf_r"):
        row = summary["focus_profiles"][focus_name]
        lines.append(
            f"| {focus_name} | {fmt(row['candidate_window_12_15_delta_vs_source_70a'])} | "
            f"{fmt(row['candidate_window_12_15_delta_vs_current_replace'])} | "
            f"{fmt(row['candidate_window_57_64_delta_vs_source_70a'])} | "
            f"{row['candidate_delta_vs_current_replace'].get('worst_regress_sics', [])[:6]} |"
        )
    md_path = SIC_ROOT / "best_replace_sic_compare.md"
    json_path = SIC_ROOT / "best_replace_sic_compare.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_md"] = str(md_path)
    summary["summary_json"] = str(json_path)
    write_json(json_path, summary)
    return summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    code_review = summary["code_review"]
    refs = summary["references"]

    def lane_row(label: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | "
            f"{fmt(metrics['leg'])} | {fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | "
            f"{fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    def candidate_row(candidate: str, stage: str, metrics: Mapping[str, Any]) -> str:
        return (
            f"| {candidate} | {stage} | {fmt(metrics['DirectGeoLocalDeg'])} | {fmt(metrics['all_ex_root'])} | "
            f"{fmt(metrics['leg'])} | {fmt(metrics['nonleg'])} | {fmt(metrics['arm'])} | "
            f"{fmt(metrics['foot_l_ball_l_SIC12_15'])} | {fmt(metrics['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# ep014center replace redesign",
        "",
        "## Live Review",
        "",
        "- current replace trainable params:",
        f"  `{', '.join(code_review['replace_scope']['trainable_param_names'])}`",
        "- current downstream71 trainable params:",
        f"  `{', '.join(code_review['downstream_scope']['trainable_param_names'])}`",
        f"- warmstart is copy-only: `{str(bool(code_review['warmstart_report'].get('copied_without_phase_z_direct_adaptation', False))).lower()}`",
        f"- parser sees legacy phase_z keys as live: `{str(bool(any(code_review['legacy_phase_keys_active_in_parsed_cfg'].values()))).lower()}`",
        "",
        "## Replace References",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        lane_row("source_70a", refs["source_70a"]["eval"]["selected_metrics"]),
        lane_row("current_replace", refs["current_replace"]["eval"]["selected_metrics"]),
        lane_row("docs_baseline_replace", refs["docs_baseline_replace"]["eval"]["selected_metrics"]),
        lane_row("current_best_downstream71", refs["current_best_downstream"]["eval"]["selected_metrics"]),
        lane_row("docs_baseline_71_lr3e4", refs["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"]),
        lane_row("plain71_best", refs["plain71_best"]["eval"]["selected_metrics"]),
        "",
        "## Candidates",
        "",
        "| candidate | stage | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand in summary["candidates"]:
        lines.append(candidate_row(str(cand["name"]), "replace", cand["replace"]["eval"]["selected_metrics"]))
        lines.append(candidate_row(str(cand["name"]), "downstream71", cand["downstream"]["eval"]["selected_metrics"]))
    lines.extend(
        [
            "",
            "## Best Verdict",
            "",
            f"- best candidate by downstream `all_ex_root`: `{summary['answers']['best_candidate_name']}`",
            f"- replace redesign successful vs current best downstream: `{str(bool(summary['answers']['redesign_success_beats_current_best_downstream'])).lower()}`",
            f"- switch mainline away from current best path: `{str(bool(summary['answers']['switch_mainline'])).lower()}`",
            "",
        ]
    )
    if isinstance(summary.get("best_replace_sic"), Mapping):
        lines.extend(
            [
                "## SIC",
                "",
                f"- best replace SIC summary: `{summary['best_replace_sic'].get('summary_md')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        FIXED_BASETRAIN_CONFIG,
        FIXED_STAGE6_WINNER,
        FIXED_70A_WINNER,
        SOURCE_70A_ARTIFACTS_JSON,
        CURRENT_REPLACE_BASE_SUMMARY_JSON,
        CURRENT_REPLACE_SWEEP_SUMMARY_JSON,
        CURRENT_DOWNSTREAM_SUMMARY_JSON,
        PLAIN71_SUMMARY_JSON,
        DOCS_BASELINE_SUMMARY_JSON,
        SIC_DIAGNOSTIC_SUMMARY,
        CURRENT_SCOPE_AUDIT_SUMMARY,
        CURRENT_REPLACE_CONFIG_JSON,
        CURRENT_DOWNSTREAM_CONFIG_JSON,
        BASE_71_CONFIG_JSON,
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

    log("=== build inspection context ===")
    ds = build_inspection_dataset()
    references = load_references()

    log("=== code review: current replace live semantics ===")
    code_review = build_code_review(ds=ds, references=references)
    write_json(SCOPE_ROOT / "code_review.json", code_review)

    warmstart_ckpt = Path(str(references["current_replace"]["warmstart_ckpt"]))
    if not warmstart_ckpt.is_file():
        raise SystemExit(f"missing warmstart ckpt: {warmstart_ckpt}")

    candidates_out: list[Dict[str, Any]] = []
    for candidate in CANDIDATES:
        log(f"=== candidate {candidate['name']} ===")
        payload = run_replace_candidate(candidate=candidate, warmstart_ckpt=warmstart_ckpt, lane_log=lane_log, ds=ds)
        payload["replace"]["delta_vs_source_70a"] = metric_delta(
            payload["replace"]["eval"]["selected_metrics"],
            references["source_70a"]["eval"]["selected_metrics"],
        )
        payload["replace"]["delta_vs_current_replace"] = metric_delta(
            payload["replace"]["eval"]["selected_metrics"],
            references["current_replace"]["eval"]["selected_metrics"],
        )
        payload["replace"]["delta_vs_docs_baseline_replace"] = metric_delta(
            payload["replace"]["eval"]["selected_metrics"],
            references["docs_baseline_replace"]["eval"]["selected_metrics"],
        )
        payload["downstream"]["delta_vs_current_best_downstream"] = metric_delta(
            payload["downstream"]["eval"]["selected_metrics"],
            references["current_best_downstream"]["eval"]["selected_metrics"],
        )
        payload["downstream"]["delta_vs_docs_baseline_71_lr3e4"] = metric_delta(
            payload["downstream"]["eval"]["selected_metrics"],
            references["docs_baseline_71_lr3e4"]["eval"]["selected_metrics"],
        )
        payload["downstream"]["delta_vs_plain71_best"] = metric_delta(
            payload["downstream"]["eval"]["selected_metrics"],
            references["plain71_best"]["eval"]["selected_metrics"],
        )
        payload["downstream"]["delta_vs_source_70a"] = metric_delta(
            payload["downstream"]["eval"]["selected_metrics"],
            references["source_70a"]["eval"]["selected_metrics"],
        )
        candidates_out.append(payload)

    best_candidate = min(
        candidates_out,
        key=lambda item: safe_float(item["downstream"]["eval"]["selected_metrics"]["all_ex_root"]),
    )
    best_candidate_name = str(best_candidate["name"])
    best_replace_sic = build_best_replace_sic_summary(
        best_candidate_name=best_candidate_name,
        best_replace_eval_json=Path(str(best_candidate["replace"]["eval"]["paths"]["eval_json"])),
        references=references,
    )

    best_downstream_metrics = best_candidate["downstream"]["eval"]["selected_metrics"]
    current_best_downstream_metrics = references["current_best_downstream"]["eval"]["selected_metrics"]
    current_replace_metrics = references["current_replace"]["eval"]["selected_metrics"]

    redesign_success = safe_float(best_downstream_metrics["all_ex_root"]) < safe_float(current_best_downstream_metrics["all_ex_root"])
    pareto_candidate = (
        safe_float(best_downstream_metrics["leg"]) <= safe_float(current_best_downstream_metrics["leg"])
        and safe_float(best_downstream_metrics["calf_r_SIC2_4"]) <= safe_float(current_best_downstream_metrics["calf_r_SIC2_4"])
        and safe_float(best_downstream_metrics["foot_l_ball_l_SIC12_15"]) <= safe_float(current_best_downstream_metrics["foot_l_ball_l_SIC12_15"])
    )

    findings: list[Dict[str, Any]] = []
    if not redesign_success:
        findings.append(
            {
                "severity": "high",
                "title": "No redesigned replace beats the current best downstream path on all_ex_root",
                "detail": (
                    f"Best redesigned candidate is `{best_candidate_name}`, but its downstream "
                    f"`all_ex_root={fmt(best_downstream_metrics['all_ex_root'])}` does not beat the locked current best "
                    f"`{fmt(current_best_downstream_metrics['all_ex_root'])}`."
                ),
            }
        )
    if safe_float(best_candidate["replace"]["eval"]["selected_metrics"]["calf_r_SIC2_4"]) > safe_float(current_replace_metrics["calf_r_SIC2_4"]):
        findings.append(
            {
                "severity": "medium",
                "title": "Best redesigned replace still does not fully solve calf_r exposure at replace stage",
                "detail": (
                    f"Best replace candidate `{best_candidate_name}` has replace-stage calf_r@SIC2-4="
                    f"{fmt(best_candidate['replace']['eval']['selected_metrics']['calf_r_SIC2_4'])} "
                    f"vs current replace {fmt(current_replace_metrics['calf_r_SIC2_4'])}."
                ),
            }
        )
    if safe_float(best_downstream_metrics["foot_l_ball_l_SIC12_15"]) > safe_float(current_best_downstream_metrics["foot_l_ball_l_SIC12_15"]):
        findings.append(
            {
                "severity": "medium",
                "title": "Best redesigned downstream candidate pays extra foot_l/ball_l cost",
                "detail": (
                    f"Best candidate `{best_candidate_name}` downstream foot_l/ball_l@SIC12-15="
                    f"{fmt(best_downstream_metrics['foot_l_ball_l_SIC12_15'])} "
                    f"vs current best downstream {fmt(current_best_downstream_metrics['foot_l_ball_l_SIC12_15'])}."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "severity": "info",
                "title": "At least one redesigned replace candidate beats the current best downstream path",
                "detail": f"`{best_candidate_name}` is a clear downstream winner on the primary metric.",
            }
        )

    summary = {
        "run_date": RUN_DATE,
        "inputs": {
            "basetrain_config": str(FIXED_BASETRAIN_CONFIG),
            "stage6_winner": str(FIXED_STAGE6_WINNER),
            "source_70a_winner": str(FIXED_70A_WINNER),
            "current_replace_config": str(CURRENT_REPLACE_CONFIG_JSON),
            "current_replace_ckpt": str(references["current_replace"]["ckpt"]),
            "current_best_downstream_config": str(CURRENT_DOWNSTREAM_CONFIG_JSON),
            "current_best_downstream_ckpt": str(references["current_best_downstream"]["ckpt"]),
            "sic_diagnostic_summary": str(SIC_DIAGNOSTIC_SUMMARY),
            "current_scope_audit_summary": str(CURRENT_SCOPE_AUDIT_SUMMARY),
            "plain71_summary": str(PLAIN71_SUMMARY_JSON),
        },
        "policy": {
            "fixed_upstream": "Stage6(last) -> 70a(lr=3e-4)",
            "fixed_replace_base": "current replace(lr=5e-5) config + shared copy-only warmstart from 70a",
            "fixed_downstream": "best redesigned replace -> skip-70R -> 71(lr=3e-4)",
            "replace_lr_policy": {
                "main_lr": DEFAULT_REPLACE_LR,
                "backup_lr": None,
                "reason": "keep LR fixed at the current best replace LR so redesign quality reflects semantic changes, not new LR search",
            },
            "reuse_pipeline": [
                "run_posttrain_stage",
                "run_eval",
                "ensure_group_summary",
                "masked_metric_means",
                "group_metrics",
                "window_group_stats",
                "_per_sic_group_profile",
            ],
            "executed_candidates": [cand["name"] for cand in CANDIDATES],
            "unrun_hypotheses": list(UNRUN_HYPOTHESES),
        },
        "code_review": code_review,
        "hypotheses": [
            {"name": cand["name"], "hypothesis": cand["hypothesis"], "executed": True}
            for cand in CANDIDATES
        ] + [{"name": f"fallback_{idx+1}", "hypothesis": text, "executed": False} for idx, text in enumerate(UNRUN_HYPOTHESES)],
        "references": references,
        "candidates": candidates_out,
        "best_replace_sic": best_replace_sic,
        "findings": findings,
        "answers": {
            "best_candidate_name": best_candidate_name,
            "redesign_success_beats_current_best_downstream": redesign_success,
            "best_candidate_is_pareto_tradeoff": bool((not redesign_success) and pareto_candidate),
            "switch_mainline": redesign_success,
            "best_candidate_replace_beats_current_replace_all_ex_root": safe_float(best_candidate["replace"]["eval"]["selected_metrics"]["all_ex_root"]) < safe_float(current_replace_metrics["all_ex_root"]),
            "best_candidate_replace_beats_current_replace_leg": safe_float(best_candidate["replace"]["eval"]["selected_metrics"]["leg"]) < safe_float(current_replace_metrics["leg"]),
            "best_candidate_replace_beats_current_replace_calf": safe_float(best_candidate["replace"]["eval"]["selected_metrics"]["calf_r_SIC2_4"]) < safe_float(current_replace_metrics["calf_r_SIC2_4"]),
            "best_candidate_downstream_beats_current_best_all_ex_root": redesign_success,
            "best_candidate_downstream_beats_current_best_leg": safe_float(best_downstream_metrics["leg"]) < safe_float(current_best_downstream_metrics["leg"]),
            "best_candidate_downstream_beats_current_best_calf": safe_float(best_downstream_metrics["calf_r_SIC2_4"]) < safe_float(current_best_downstream_metrics["calf_r_SIC2_4"]),
            "best_candidate_downstream_beats_current_best_foot": safe_float(best_downstream_metrics["foot_l_ball_l_SIC12_15"]) < safe_float(current_best_downstream_metrics["foot_l_ball_l_SIC12_15"]),
        },
    }

    write_json(
        OUT_ROOT / "status.json",
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_candidate": best_candidate_name,
            "redesign_success_beats_current_best_downstream": redesign_success,
        },
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
