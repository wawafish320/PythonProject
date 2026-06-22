#!/usr/bin/env python3
"""GT-only lifted contract exactness repair audit.

Debug-only/read-only probe for support_anchor_keep_inter_anchor exactness. It
does not train a model, does not forward production Trainer/runtime/gate, does
not mutate checkpoints, and does not attach a production path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    FPS,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_command_demotion_replay import (  # noqa: E402
    _attach_demoted_acceptance,
    _calibrate_command_bands,
    _command_compatibility,
    _support_side_core,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _fmt,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    DecoderItem,
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _evaluate_seq_common,
    _reconstructed_gt_seq,
)
from tools.run_action_handoff_signal_representation_audit import (  # noqa: E402
    _anchor_root_path,
    _ego_from_world_vel,
    _rng_for,
    _state_and_seq_from_state,
    _support_foot_world_displacement,
)
from tools.run_action_handoff_support_contract_tightening_probe import _support_contract  # noqa: E402
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_lifted_contract_exactness_repair_20260603")
EPS = 1e-8
FLOAT32_FOOT_SLIP_ABS_EPS_MPS = 1e-6
FLOAT32_FOOT_SLIP_REL_EPS = 2e-6

RATE_SUBITEMS = (
    ("angvel_step_rms_p95", "angvel_step_rms"),
    ("angvel_component_p95_p95", "angvel_step_component_p95"),
    ("rootvel_step_l2_p95", "rootvel_step_l2"),
    ("yaw_rate_step_abs_p95", "yaw_rate_step_abs"),
)
SUPPORT_SUBITEMS = (
    ("contact_step_l2_p95", "contact_step_l2"),
    ("foot_slip_p95_mps", "foot_slip_contacted_speed_mps"),
)
CORE_ACCEPTANCE_KEYS = (
    "regime_reached",
    "rate_budget",
    "support_honesty",
    "support_side_core",
    "command_compatibility",
    "pose_continuity",
    "endpoint_bridgeability",
)
SUPPORT_SIDE_CORE_DIAGNOSTIC_KEYS = (
    "regime_reached",
    "support_side_core",
    "command_compatibility",
    "pose_continuity",
    "endpoint_bridgeability",
)


def _jsonify(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonify(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _finite_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _finite_seq(seq: Mapping[str, np.ndarray]) -> bool:
    for key in ("rot6d", "root_pos", "root_vel", "bone_angvel", "cond_dir", "contact", "yaw_rate"):
        arr = np.asarray(seq[key])
        if not np.all(np.isfinite(arr)):
            return False
    return True


def _world_fd_forward(root_pos: np.ndarray) -> np.ndarray:
    root = np.asarray(root_pos, dtype=np.float32).reshape(-1, 3)
    h = int(root.shape[0])
    vel = np.zeros((h, 2), dtype=np.float32)
    if h > 1:
        vel[:-1] = (root[1:, :2] - root[:-1, :2]) * float(FPS)
        vel[-1] = vel[-2]
    return vel


def _world_fd_central_endpoint(root_pos: np.ndarray) -> np.ndarray:
    root = np.asarray(root_pos, dtype=np.float32).reshape(-1, 3)
    h = int(root.shape[0])
    vel = np.zeros((h, 2), dtype=np.float32)
    if h == 1:
        return vel
    step_vel = (root[1:, :2] - root[:-1, :2]) * float(FPS)
    vel[0] = step_vel[0]
    vel[-1] = step_vel[-1]
    if h > 2:
        vel[1:-1] = 0.5 * (step_vel[:-1] + step_vel[1:])
    return vel.astype(np.float32, copy=False)


def _state_from_world_root_vel(base_state: np.ndarray, world_root_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    state = np.asarray(base_state, dtype=np.float32).reshape(-1, STATE_DIM).copy()
    state[:, EGO_VEL_SLICE] = _ego_from_world_vel(world_root_vel, cond_dir)
    return state.astype(np.float32, copy=False)


def _seq_from_components(
    *,
    base_seq: Mapping[str, np.ndarray],
    root_pos: np.ndarray,
    root_vel: np.ndarray,
) -> Dict[str, np.ndarray]:
    h = int(np.asarray(base_seq["rot6d"]).shape[0])
    return {
        "rot6d": np.asarray(base_seq["rot6d"], dtype=np.float32).reshape(h, POSE_SLICE.stop - POSE_SLICE.start),
        "root_pos": np.asarray(root_pos, dtype=np.float32).reshape(h, 3),
        "root_vel": np.asarray(root_vel, dtype=np.float32).reshape(h, 2),
        "bone_angvel": np.asarray(base_seq["bone_angvel"], dtype=np.float32).reshape(h, ANGVEL_DIM),
        "cond_dir": np.asarray(base_seq["cond_dir"], dtype=np.float32).reshape(h, 2),
        "contact": np.asarray(base_seq["contact"], dtype=np.float32).reshape(h, 2),
        "yaw_rate": np.asarray(base_seq["yaw_rate"], dtype=np.float32).reshape(h),
    }


def _variant_sequences(
    *,
    item_i: int,
    item: DecoderItem,
    skeleton: Any,
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    aux: np.ndarray,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray], str]]:
    anchored_root = _anchor_root_path(
        item,
        skeleton,
        baseline_seq,
        keep_inter_anchor=True,
        noise_scale=0.0,
        rng=_rng_for(seed, item_i, "anchored_root_exact"),
    )

    baseline_state = _state_from_world_root_vel(
        true_state,
        _world_fd_forward(anchored_root),
        np.asarray(baseline_seq["cond_dir"], dtype=np.float32),
    )
    baseline_state, baseline_seq_current = _state_and_seq_from_state(item, baseline_state, aux)

    copied_state = np.asarray(true_state, dtype=np.float32).reshape(-1, STATE_DIM).copy()
    copied_seq = _seq_from_components(
        base_seq=baseline_seq,
        root_pos=anchored_root,
        root_vel=np.asarray(baseline_seq["root_vel"], dtype=np.float32),
    )

    central_root_vel = _world_fd_central_endpoint(anchored_root)
    central_state = _state_from_world_root_vel(
        true_state,
        central_root_vel,
        np.asarray(baseline_seq["cond_dir"], dtype=np.float32),
    )
    central_seq = _seq_from_components(
        base_seq=baseline_seq,
        root_pos=anchored_root,
        root_vel=central_root_vel,
    )

    contact_state = baseline_state.copy()
    contact_state[:, CONTACT_SLICE] = np.asarray(true_state[:, CONTACT_SLICE], dtype=np.float32)
    contact_seq = dict(baseline_seq_current)
    contact_seq["contact"] = np.asarray(baseline_seq["contact"], dtype=np.float32).copy()

    support_core_state = baseline_state.copy()
    support_core_seq = dict(baseline_seq_current)

    return {
        "baseline_current": (
            baseline_state,
            baseline_seq_current,
            "current root_path-to-root_vel forward finite-diff plus reconstructed-domain integration",
        ),
        "copied_gt_root_vel": (
            copied_state,
            copied_seq,
            "anchored keep root_path with reconstructed GT root_vel copied into state281 velocity channels",
        ),
        "endpoint_consistent_fd": (
            central_state,
            central_seq,
            "anchored keep root_path with central finite-diff root_vel and endpoint copied one-sided steps",
        ),
        "contact_passthrough_check": (
            contact_state,
            contact_seq,
            "baseline_current with oracle contact explicitly passed through",
        ),
        "support_side_core_only": (
            support_core_state,
            support_core_seq,
            "diagnostic only: command-ish support-side keys excluded and rate/support_honesty not allowed to license decoder",
        ),
    }


def _metric_with_band(row: Mapping[str, Any], metric_key: str, band_key: str) -> Tuple[float, float, float, bool]:
    metrics = row.get("metrics", {}) or {}
    thresholds = row.get("thresholds", {}) or {}
    value = _finite_float(metrics.get(metric_key))
    band = _finite_float(thresholds.get(band_key))
    margin = value - band
    return value, band, margin, bool(value > band + EPS)


def _failed_subitems(row: Mapping[str, Any], specs: Sequence[Tuple[str, str]]) -> List[str]:
    out: List[str] = []
    for metric_key, band_key in specs:
        _value, _band, _margin, failed = _metric_with_band(row, metric_key, band_key)
        if failed:
            out.append(metric_key)
    return out


def _attach_exactness_fields(
    row: Dict[str, Any],
    *,
    variant: str,
    item: DecoderItem,
    seq: Mapping[str, np.ndarray],
    baseline_seq: Mapping[str, np.ndarray],
    state: np.ndarray,
    true_state: np.ndarray,
    skeleton: Any,
) -> Dict[str, Any]:
    labels = [str(x) for x in _support_contract(seq["contact"], min_run_frames=2)["normalized_label_sequence"]]
    disp = _support_foot_world_displacement(seq, baseline_seq, labels, skeleton)
    root_err = np.linalg.norm(
        np.asarray(seq["root_pos"], dtype=np.float64) - np.asarray(baseline_seq["root_pos"], dtype=np.float64),
        axis=1,
    )
    state_delta = np.asarray(state, dtype=np.float64) - np.asarray(true_state, dtype=np.float64)
    contact_delta = state_delta[:, CONTACT_SLICE]

    row["variant"] = variant
    row["clip"] = item.clip
    row["start"] = int(item.start)
    row["end"] = int(item.end)
    row["root_path_error_p95_m"] = _safe_percentile(root_err, 95.0)
    row["root_path_error_max_m"] = float(np.max(root_err)) if root_err.size else 0.0
    row["support_foot_world_displacement_p95_m"] = float(disp["p95_m"])
    row["support_foot_world_displacement_max_m"] = float(disp["max_m"])
    row["support_foot_world_displacement_count"] = int(disp["count"])
    row["max_abs_state_delta"] = float(np.max(np.abs(state_delta))) if state_delta.size else 0.0
    row["max_abs_contact_delta"] = float(np.max(np.abs(contact_delta))) if contact_delta.size else 0.0
    row["finite_ok"] = bool(np.all(np.isfinite(state)) and _finite_seq(seq))
    row["heading_error_p95_rad"] = _finite_float((row.get("metrics") or {}).get("heading_error_p95_rad"))
    row["foot_slip_p95_to_band_ratio"] = _finite_float(row.get("foot_slip_p95_to_band_ratio"))

    exceeded: List[str] = []
    for metric_key, band_key in RATE_SUBITEMS + SUPPORT_SUBITEMS:
        value, band, margin, failed = _metric_with_band(row, metric_key, band_key)
        row[metric_key] = value
        row[f"{metric_key}_band"] = band
        row[f"{metric_key}_over_band"] = max(0.0, margin)
        row[f"{metric_key}_fails_band"] = bool(failed)
        if failed:
            exceeded.append(metric_key)
    row["rate_budget_failed_subitems"] = ",".join(_failed_subitems(row, RATE_SUBITEMS))
    row["support_honesty_failed_subitems"] = ",".join(_failed_subitems(row, SUPPORT_SUBITEMS))
    row["exceeded_subitems"] = ",".join(exceeded)
    return row


def _foot_slip_float32_tolerance(row: Mapping[str, Any]) -> float:
    band = abs(_finite_float(row.get("foot_slip_p95_mps_band"), 0.0))
    return max(FLOAT32_FOOT_SLIP_ABS_EPS_MPS, FLOAT32_FOOT_SLIP_REL_EPS * max(1.0, band))


def _is_float32_precision_only_support_failure(row: Mapping[str, Any]) -> bool:
    failed = [x for x in str(row.get("demoted_failed_family") or "").split(",") if x]
    if failed != ["support_honesty"]:
        return False
    if str(row.get("support_honesty_failed_subitems") or "") != "foot_slip_p95_mps":
        return False
    if bool(row.get("contact_step_l2_p95_fails_band", False)):
        return False
    over = _finite_float(row.get("foot_slip_p95_mps_over_band"), 0.0)
    return bool(0.0 < over <= _foot_slip_float32_tolerance(row))


def _attach_float32_precision_tolerant_acceptance(
    row: Dict[str, Any],
    *,
    acceptance_keys: Sequence[str],
) -> Dict[str, Any]:
    precision_only = _is_float32_precision_only_support_failure(row)
    row["float32_foot_slip_abs_eps_mps"] = FLOAT32_FOOT_SLIP_ABS_EPS_MPS
    row["float32_foot_slip_rel_eps"] = FLOAT32_FOOT_SLIP_REL_EPS
    row["float32_foot_slip_tolerance_mps"] = _foot_slip_float32_tolerance(row)
    row["float32_precision_only_support_failure"] = bool(precision_only)
    row["float32_precision_tolerant_support_honesty"] = bool(row.get("support_honesty", False) or precision_only)

    failed: List[str] = []
    for key in acceptance_keys:
        ok = bool(row.get(key, False))
        if key == "support_honesty":
            ok = bool(row["float32_precision_tolerant_support_honesty"])
        if not ok:
            failed.append(str(key))
    row["float32_precision_tolerant_demoted_acceptance_pass"] = bool(not failed)
    row["float32_precision_tolerant_failed_family"] = ",".join(failed)
    return row


def _evaluate_variant_seq(
    *,
    variant: str,
    acceptance_keys: Sequence[str],
    item: DecoderItem,
    state: np.ndarray,
    seq: Mapping[str, np.ndarray],
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    command_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    note: str,
) -> Dict[str, Any]:
    row = _evaluate_seq_common(
        variant=variant,
        split="lifted_contract_exactness_repair",
        split_kind="gt_read_only",
        partition="support_anchor_keep_inter_anchor",
        item=item,
        seq=seq,
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=min_run_frames,
        endpoint_note=note,
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
        calibration_domain="reconstructed_state281",
    )
    contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
    labels = [str(x) for x in contract["normalized_label_sequence"]]
    row.update(_support_side_core(seq, labels, skeleton, support_bands[item.clip]["feature_bands"]))
    row.update(_command_compatibility(seq, command_bands.get(item.clip)))
    row["legacy_acceptance_pass"] = bool(row.get("acceptance_proxy_pass", False))
    row["legacy_failed_family"] = str(row.get("failed_family") or "")
    row["acceptance_definition"] = ",".join(acceptance_keys)

    if tuple(acceptance_keys) == CORE_ACCEPTANCE_KEYS:
        _attach_demoted_acceptance(row, include_support_side_core=True)
    else:
        failed = [key for key in acceptance_keys if not bool(row.get(key, False))]
        row["demoted_acceptance_pass"] = bool(not failed)
        row["demoted_failed_family"] = ",".join(failed)
        row["legacy_command_response_diagnostic"] = bool(row.get("command_response", False))
    row = _attach_exactness_fields(
        row,
        variant=variant,
        item=item,
        seq=seq,
        baseline_seq=baseline_seq,
        state=state,
        true_state=true_state,
        skeleton=skeleton,
    )
    return _attach_float32_precision_tolerant_acceptance(row, acceptance_keys=acceptance_keys)


def _pass_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([bool(r.get(key, False)) for r in rows])) if rows else 0.0


def _numeric_stats(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    vals = np.asarray([_finite_float(r.get(key)) for r in rows], dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "p95": _safe_percentile(vals, 95.0),
        "max": float(np.max(vals)),
    }


def _summarize_variant(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failed = Counter()
    for row in rows:
        for family in str(row.get("demoted_failed_family") or "").split(","):
            if family:
                failed[family] += 1
    root = _numeric_stats(rows, "root_path_error_p95_m")
    disp = _numeric_stats(rows, "support_foot_world_displacement_p95_m")
    foot = _numeric_stats(rows, "foot_slip_p95_to_band_ratio")
    heading = _numeric_stats(rows, "heading_error_p95_rad")
    precision_failed = Counter()
    for row in rows:
        for family in str(row.get("float32_precision_tolerant_failed_family") or "").split(","):
            if family:
                precision_failed[family] += 1
    return {
        "n": int(len(rows)),
        "demoted_acceptance_pass_rate": _pass_rate(rows, "demoted_acceptance_pass"),
        "float32_precision_tolerant_demoted_pass_rate": _pass_rate(
            rows,
            "float32_precision_tolerant_demoted_acceptance_pass",
        ),
        "rate_budget_pass_rate": _pass_rate(rows, "rate_budget"),
        "support_honesty_pass_rate": _pass_rate(rows, "support_honesty"),
        "float32_precision_tolerant_support_honesty_pass_rate": _pass_rate(
            rows,
            "float32_precision_tolerant_support_honesty",
        ),
        "support_side_core_pass_rate": _pass_rate(rows, "support_side_core"),
        "command_compatibility_pass_rate": _pass_rate(rows, "command_compatibility"),
        "pose_continuity_pass_rate": _pass_rate(rows, "pose_continuity"),
        "endpoint_bridgeability_pass_rate": _pass_rate(rows, "endpoint_bridgeability"),
        "failed_family_counts": dict(failed),
        "float32_precision_tolerant_failed_family_counts": dict(precision_failed),
        "float32_precision_only_support_failure_count": int(
            sum(bool(r.get("float32_precision_only_support_failure", False)) for r in rows)
        ),
        "root_path_error_p95_m_mean": root["mean"],
        "root_path_error_p95_m_p95": root["p95"],
        "root_path_error_p95_m_max": root["max"],
        "support_foot_world_displacement_p95_m_mean": disp["mean"],
        "support_foot_world_displacement_p95_m_p95": disp["p95"],
        "support_foot_world_displacement_p95_m_max": disp["max"],
        "foot_slip_p95_to_band_ratio_mean": foot["mean"],
        "foot_slip_p95_to_band_ratio_p95": foot["p95"],
        "foot_slip_p95_to_band_ratio_max": foot["max"],
        "heading_error_p95_rad_mean": heading["mean"],
        "heading_error_p95_rad_p95": heading["p95"],
        "heading_error_p95_rad_max": heading["max"],
    }


def _decision(variant_summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    baseline = variant_summary.get("baseline_current", {})
    copied = variant_summary.get("copied_gt_root_vel", {})
    endpoint_fd = variant_summary.get("endpoint_consistent_fd", {})
    support_core = variant_summary.get("support_side_core_only", {})
    copied_pass = _finite_float(copied.get("demoted_acceptance_pass_rate"))
    endpoint_pass = _finite_float(endpoint_fd.get("demoted_acceptance_pass_rate"))
    baseline_pass = _finite_float(baseline.get("demoted_acceptance_pass_rate"))
    endpoint_precision_pass = _finite_float(endpoint_fd.get("float32_precision_tolerant_demoted_pass_rate"))
    copied_precision_pass = _finite_float(copied.get("float32_precision_tolerant_demoted_pass_rate"))
    support_core_pass = _finite_float(support_core.get("demoted_acceptance_pass_rate"))
    acceptance_grade = bool(copied_pass >= 0.95 or endpoint_pass >= 0.95 or baseline_pass >= 0.95)
    lossless_under_precision = bool(endpoint_precision_pass >= 0.999)
    if lossless_under_precision:
        blocker = "none_float32_band_edge_precision_noise_only"
    elif copied_pass >= 0.95 and baseline_pass < 0.95:
        blocker = "root_pos_to_root_vel_finite_diff_exactness_calibration"
    elif copied_pass < 0.95:
        blocker = "support_fk_contact_exactness_or_acceptance_band"
    else:
        blocker = "none_for_gt_only_reconstructability"
    return {
        "acceptance_threshold": 0.95,
        "baseline_current_acceptance_grade": bool(baseline_pass >= 0.95),
        "copied_gt_root_vel_acceptance_grade": bool(copied_pass >= 0.95),
        "endpoint_consistent_fd_acceptance_grade": bool(endpoint_pass >= 0.95),
        "copied_gt_root_vel_float32_precision_tolerant_pass_rate": copied_precision_pass,
        "endpoint_consistent_fd_float32_precision_tolerant_pass_rate": endpoint_precision_pass,
        "anchored_keep_gt_lossless_under_float32_roundtrip_precision": bool(lossless_under_precision),
        "committed_reconstruction_contract": "endpoint_consistent_fd",
        "copied_gt_root_vel_role": "oracle upper bound only; not a deployable reconstruction contract",
        "support_side_core_only_acceptance_grade": bool(support_core_pass >= 0.95),
        "anchored_keep_reaches_acceptance_grade": bool(acceptance_grade),
        "primary_blocker_class": blocker,
        "allow_fair_perturbation_next": bool(acceptance_grade),
        "allow_decoder_toy_smoke": False,
        "decoder_toy_smoke_reason": "GT-only lifted reconstructability can license fair perturbation next, not decoder toy smoke",
    }


def _failure_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if bool(row.get("demoted_acceptance_pass", False)):
            continue
        out.append(
            {
                "clip": row.get("clip"),
                "start": int(row.get("start", 0)),
                "end": int(row.get("end", 0)),
                "failed_family": row.get("demoted_failed_family", ""),
                "exceeded_subitems": row.get("exceeded_subitems", ""),
                "rate_budget_failed_subitems": row.get("rate_budget_failed_subitems", ""),
                "support_honesty_failed_subitems": row.get("support_honesty_failed_subitems", ""),
                "angvel_step_rms_p95": row.get("angvel_step_rms_p95"),
                "angvel_step_rms_p95_band": row.get("angvel_step_rms_p95_band"),
                "angvel_component_p95_p95": row.get("angvel_component_p95_p95"),
                "angvel_component_p95_p95_band": row.get("angvel_component_p95_p95_band"),
                "rootvel_step_l2_p95": row.get("rootvel_step_l2_p95"),
                "rootvel_step_l2_p95_band": row.get("rootvel_step_l2_p95_band"),
                "yaw_rate_step_abs_p95": row.get("yaw_rate_step_abs_p95"),
                "yaw_rate_step_abs_p95_band": row.get("yaw_rate_step_abs_p95_band"),
                "contact_step_l2_p95": row.get("contact_step_l2_p95"),
                "contact_step_l2_p95_band": row.get("contact_step_l2_p95_band"),
                "foot_slip_p95_mps": row.get("foot_slip_p95_mps"),
                "foot_slip_band": row.get("foot_slip_p95_mps_band"),
                "root_path_error_p95_m": row.get("root_path_error_p95_m"),
                "support_foot_world_displacement_p95_m": row.get("support_foot_world_displacement_p95_m"),
                "heading_error_p95_rad": row.get("heading_error_p95_rad"),
                "max_abs_state_delta": row.get("max_abs_state_delta"),
            }
        )
    return sorted(out, key=lambda r: (str(r["clip"]), int(r["start"])))


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "variant",
        "clip",
        "start",
        "end",
        "demoted_acceptance_pass",
        "demoted_failed_family",
        "float32_precision_tolerant_demoted_acceptance_pass",
        "float32_precision_tolerant_failed_family",
        "float32_precision_only_support_failure",
        "exceeded_subitems",
        "rate_budget_failed_subitems",
        "support_honesty_failed_subitems",
        "rate_budget",
        "support_honesty",
        "support_side_core",
        "command_compatibility",
        "pose_continuity",
        "endpoint_bridgeability",
        "angvel_step_rms_p95",
        "angvel_step_rms_p95_band",
        "angvel_step_rms_p95_over_band",
        "angvel_component_p95_p95",
        "angvel_component_p95_p95_band",
        "angvel_component_p95_p95_over_band",
        "rootvel_step_l2_p95",
        "rootvel_step_l2_p95_band",
        "rootvel_step_l2_p95_over_band",
        "yaw_rate_step_abs_p95",
        "yaw_rate_step_abs_p95_band",
        "yaw_rate_step_abs_p95_over_band",
        "contact_step_l2_p95",
        "contact_step_l2_p95_band",
        "contact_step_l2_p95_over_band",
        "foot_slip_p95_mps",
        "foot_slip_p95_mps_band",
        "foot_slip_p95_mps_over_band",
        "float32_foot_slip_tolerance_mps",
        "foot_slip_p95_to_band_ratio",
        "root_path_error_p95_m",
        "root_path_error_max_m",
        "support_foot_world_displacement_p95_m",
        "support_foot_world_displacement_max_m",
        "heading_error_p95_rad",
        "max_abs_state_delta",
        "max_abs_contact_delta",
        "finite_ok",
        "acceptance_definition",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Lifted Contract Exactness Repair")
    lines.append("")
    lines.append("Date: 2026-06-03")
    lines.append("")
    lines.append("Debug-only GT/read-only audit. No model training, no production Trainer/runtime/gate forward or edit, no checkpoint mutation, no residual head, and no decoder toy smoke.")
    lines.append("")
    lines.append("## Guard Verdict")
    dec = payload["decision"]
    lines.append(f"- anchored keep acceptance-grade: `{str(dec['anchored_keep_reaches_acceptance_grade']).lower()}`")
    lines.append(
        "- anchored keep GT lossless under float32 round-trip precision: "
        f"`{str(dec['anchored_keep_gt_lossless_under_float32_roundtrip_precision']).lower()}`"
    )
    lines.append(f"- committed reconstruction contract: `{dec['committed_reconstruction_contract']}`")
    lines.append(f"- copied GT root_vel role: `{dec['copied_gt_root_vel_role']}`")
    lines.append(f"- allow fair perturbation next: `{str(dec['allow_fair_perturbation_next']).lower()}`")
    lines.append(f"- allow decoder toy smoke: `{str(dec['allow_decoder_toy_smoke']).lower()}`")
    lines.append(f"- primary blocker class: `{dec['primary_blocker_class']}`")
    lines.append("")
    lines.append("## Preflight Caveat Consistency")
    lines.append("")
    pre = payload["preflight_caveat_consistency"]
    lines.append(f"- signal artifact high-frequency caveat present: `{str(pre['signal_artifact_high_frequency_caveat_present']).lower()}`")
    lines.append(f"- stale conditioning verdict patched before this run: `{str(pre['stale_conditioning_verdict_patched']).lower()}`")
    lines.append("- Current `1e-3` perturbation rows are per-frame independent Gaussian / high-frequency diagnostics only; flat integration low-passes and lifted finite-diff high-passes that noise.")
    lines.append("- Fair perturbation still requires native-space correlated/bias noise, equal reconstructed-state281 MSE, and position/velocity double-sided reporting.")
    lines.append(
        f"- Float32 support-slip band-edge tolerance for this debug audit: abs "
        f"`{FLOAT32_FOOT_SLIP_ABS_EPS_MPS:g} m/s`, rel `{FLOAT32_FOOT_SLIP_REL_EPS:g}`."
    )
    lines.append("")
    lines.append("## Failure Row Decomposition")
    lines.append("")
    lines.append("| clip | start | end | failed family | exceeded subitems | rootvel p95/band | foot slip p95/band | root path p95 m | support foot disp p95 m | heading p95 rad | max abs state delta |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["failure_row_decomposition"]:
        lines.append(
            f"| {row['clip']} | {row['start']} | {row['end']} | {row['failed_family']} | "
            f"{row['exceeded_subitems']} | "
            f"{_fmt(row['rootvel_step_l2_p95'], 9)}/{_fmt(row['rootvel_step_l2_p95_band'], 9)} | "
            f"{_fmt(row['foot_slip_p95_mps'], 9)}/{_fmt(row['foot_slip_band'], 9)} | "
            f"{_fmt(row['root_path_error_p95_m'], 12)} | "
            f"{_fmt(row['support_foot_world_displacement_p95_m'], 12)} | "
            f"{_fmt(row['heading_error_p95_rad'], 9)} | "
            f"{_fmt(row['max_abs_state_delta'], 9)} |"
        )
    lines.append("")
    lines.append("## Exactness Variants")
    lines.append("")
    lines.append("| variant | n | demoted pass | float32 pass | rate | support honest / float32 | support core | command compat | pose | endpoint | failed families | float32 failed families | root p95 mean/p95/max m | support foot disp mean/p95/max m | foot ratio mean/p95/max | heading mean/p95/max rad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for name, rec in payload["variant_summary"].items():
        lines.append(
            f"| {name} | {int(rec.get('n', 0))} | {_fmt(rec.get('demoted_acceptance_pass_rate'))} | "
            f"{_fmt(rec.get('float32_precision_tolerant_demoted_pass_rate'))} | "
            f"{_fmt(rec.get('rate_budget_pass_rate'))} | "
            f"{_fmt(rec.get('support_honesty_pass_rate'))}/{_fmt(rec.get('float32_precision_tolerant_support_honesty_pass_rate'))} | "
            f"{_fmt(rec.get('support_side_core_pass_rate'))} | {_fmt(rec.get('command_compatibility_pass_rate'))} | "
            f"{_fmt(rec.get('pose_continuity_pass_rate'))} | {_fmt(rec.get('endpoint_bridgeability_pass_rate'))} | "
            f"{rec.get('failed_family_counts', {})} | "
            f"{rec.get('float32_precision_tolerant_failed_family_counts', {})} | "
            f"{_fmt(rec.get('root_path_error_p95_m_mean'), 9)}/{_fmt(rec.get('root_path_error_p95_m_p95'), 9)}/{_fmt(rec.get('root_path_error_p95_m_max'), 9)} | "
            f"{_fmt(rec.get('support_foot_world_displacement_p95_m_mean'), 9)}/{_fmt(rec.get('support_foot_world_displacement_p95_m_p95'), 9)}/{_fmt(rec.get('support_foot_world_displacement_p95_m_max'), 9)} | "
            f"{_fmt(rec.get('foot_slip_p95_to_band_ratio_mean'), 6)}/{_fmt(rec.get('foot_slip_p95_to_band_ratio_p95'), 6)}/{_fmt(rec.get('foot_slip_p95_to_band_ratio_max'), 6)} | "
            f"{_fmt(rec.get('heading_error_p95_rad_mean'), 9)}/{_fmt(rec.get('heading_error_p95_rad_p95'), 9)}/{_fmt(rec.get('heading_error_p95_rad_max'), 9)} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- summary json: `{payload['artifacts']['summary_json']}`")
    lines.append(f"- rows csv: `{payload['artifacts']['rows_csv']}`")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root))
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    baseline_bands = _calibrate_reconstructed_baseline_bands(
        all_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    support_bands = _calibrate_reconstructed_support_side_bands(
        all_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    command_bands = _calibrate_command_bands(
        clips,
        horizon=int(args.horizon),
        quantile=float(args.command_quantile),
    )

    rows: List[Dict[str, Any]] = []
    for item_i, item in enumerate(main_items):
        baseline_seq = _reconstructed_gt_seq(
            item,
            oracle_contact_passthrough=True,
            command_align_root_vel=False,
        )
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(int(args.horizon), STATE_DIM)
        aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(int(args.horizon), ANGVEL_DIM)
        variants = _variant_sequences(
            item_i=item_i,
            item=item,
            skeleton=skeleton,
            baseline_seq=baseline_seq,
            true_state=true_state,
            aux=aux,
            seed=int(args.seed),
        )
        for variant, (state, seq, note) in variants.items():
            keys = SUPPORT_SIDE_CORE_DIAGNOSTIC_KEYS if variant == "support_side_core_only" else CORE_ACCEPTANCE_KEYS
            rows.append(
                _evaluate_variant_seq(
                    variant=variant,
                    acceptance_keys=keys,
                    item=item,
                    state=state,
                    seq=seq,
                    baseline_seq=baseline_seq,
                    true_state=true_state,
                    baseline_bands=baseline_bands,
                    support_bands=support_bands,
                    command_bands=command_bands,
                    skeleton=skeleton,
                    min_run_frames=int(args.min_run_frames),
                    note=note,
                )
            )

    rows_by_variant: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summary = {name: _summarize_variant(vrows) for name, vrows in rows_by_variant.items()}
    failure_decomp = _failure_rows(rows_by_variant.get("baseline_current", []))
    decision = _decision(variant_summary)
    payload = {
        "task": "GT-only lifted contract repair audit",
        "scope": (
            "debug-only GT/read-only exactness variants for support_anchor_keep_inter_anchor; "
            "no training, no production Trainer/runtime/gate forward or edit, no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "out_dir": str(args.out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "command_quantile": float(args.command_quantile),
            "seed": int(args.seed),
            "dtype": "float32",
            "device": "cpu",
            "float32_foot_slip_abs_eps_mps": FLOAT32_FOOT_SLIP_ABS_EPS_MPS,
            "float32_foot_slip_rel_eps": FLOAT32_FOOT_SLIP_REL_EPS,
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "matched_window_count": int(len(main_items)),
            "per_clip_windows": dict(Counter(it.clip for it in main_items)),
            "horizon": int(args.horizon),
        },
        "input_output_contract": {
            "state281": {"shape": [int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "root_pos": {"shape": [int(args.horizon), 3], "dtype": "float32", "device": "cpu"},
            "root_vel": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "contact": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "bone_angvel": {"shape": [int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
        },
        "preflight_caveat_consistency": {
            "signal_artifact_high_frequency_caveat_present": True,
            "stale_conditioning_verdict_patched": True,
            "current_noise_model": "per-frame independent Gaussian / high-frequency",
            "flat_path_effect": "velocity integration low-pass",
            "lifted_path_effect": "position finite-diff high-pass",
            "not_conditioning_verdict": True,
            "fair_gate_required": (
                "native-space correlated/bias noise + equal reconstructed-state281 MSE + "
                "position/velocity double-sided metrics"
            ),
        },
        "variant_summary": variant_summary,
        "failure_row_decomposition": failure_decomp,
        "decision": decision,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_new_model": False,
            "forwarded_production_runtime_or_trainer": False,
            "edited_production_runtime_trainer_gate": False,
            "mutated_checkpoint": False,
            "residual_head": False,
            "decoder_toy_smoke": False,
        },
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "lifted_contract_exactness_repair_summary.json"
    rows_csv = args.out_dir / "lifted_contract_exactness_repair_rows.csv"
    summary_md = args.out_dir / "lifted_contract_exactness_repair_summary.md"
    payload["artifacts"] = {
        "summary_json": str(summary_json),
        "rows_csv": str(rows_csv),
        "summary_md": str(summary_md),
    }
    _dump_json(summary_json, payload)
    _write_rows_csv(rows_csv, rows)
    _write_summary_md(summary_md, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--command-quantile", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=20260603)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"wrote {payload['artifacts']['summary_md']}")
    print(f"wrote {payload['artifacts']['summary_json']}")
    print(f"wrote {payload['artifacts']['rows_csv']}")
    print(json.dumps(_jsonify(payload["decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
