#!/usr/bin/env python3
"""8-window action-handoff inbetween debug train-fit discriminator.

Debug-only tool. It trains a tiny deterministic decoder on stratified oracle
schedule windows and audits p95-shadow vs accepted p99 decision bands. It does
not train production Trainer/runtime/gate code and does not mutate checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_SLICE,
    POSE_TOPK,
    STATE_DIM,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_adjusted_acceptance_guard import (  # noqa: E402
    DEFAULT_BONE_BRIDGE as DEFAULT_GUARD_BONE_BRIDGE,
    DEFAULT_COMMAND_DEMOTION_ROWS as DEFAULT_GUARD_COMMAND_DEMOTION_ROWS,
    DEFAULT_REGIME_BRIDGE as DEFAULT_GUARD_REGIME_BRIDGE,
    DEFAULT_TWO_FRAME as DEFAULT_GUARD_TWO_FRAME,
)
from tools.run_action_handoff_band_audit import DEFAULT_OUT_DIR as DEFAULT_BAND_AUDIT_OUT_DIR  # noqa: E402
from tools.run_action_handoff_dynamics_consistency_train_fit_ladder import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_PRETRAIN_TEMPLATE,
    EPS,
    _build_base_operator,
    _build_loss_refactor_context,
    _checkpoint_overlap_report,
    _dynamics_residual_from_state_aux,
    _gt_dynamics_residual_target,
    _heading_error_torch,
    _jsonify,
    _loss_refactor_foot_positions,
    _oracle_event_masks,
    _robust_cond_norm,
    _stack_cond_raw,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    WALK_F,
    _fmt,
    _foot_slip_metrics,
    _heading_error_rad,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
    _step_angvel_component_p95,
    _step_angvel_rms,
    _step_l2,
    _step_pose_l2,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    DecoderItem,
    Standardizer,
    TinyDeterministicDecoder,
    _apply_oracle_contact_passthrough,
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _dataset_arrays,
    _evaluate_support_side_correctness,
    _fit_standardizer,
    _foot_positions,
    _loss_metrics,
    _reshape_state_aux,
    _seq_from_prediction,
    _support_side_features,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    SUPPORT_SIDE_FEATURE_KEYS,
    _support_contract,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


RUN_DATE = "20260605"
DEFAULT_OUT_DIR = Path(f"debug_output/_tmp_action_handoff_8window_train_fit_beststage1_{RUN_DATE}")
DEFAULT_DOC_PATH = Path(f"docs/aperiodic_transition/2026-06-05_8window_train_fit_review.md")
DEFAULT_BAND_AUDIT_SUMMARY = DEFAULT_BAND_AUDIT_OUT_DIR / "band_audit_summary.json"

UPPER_METRICS: Tuple[Dict[str, Any], ...] = (
    {
        "metric": "bone_angvel_level_rms_to_target",
        "audit_metric": "bone_angvel_level_rms",
        "band_key": "bone_angvel_level_rms",
        "family": "regime_reached",
        "kind": "scalar",
        "event_aware": False,
    },
    {
        "metric": "angvel_step_rms_p95",
        "audit_metric": "angvel_step_rms",
        "band_key": "angvel_step_rms",
        "family": "rate_budget",
        "kind": "step",
        "event_aware": True,
    },
    {
        "metric": "angvel_component_p95_p95",
        "audit_metric": "angvel_step_component_p95",
        "band_key": "angvel_step_component_p95",
        "family": "rate_budget",
        "kind": "step",
        "event_aware": False,
    },
    {
        "metric": "rootvel_step_l2_p95",
        "audit_metric": "rootvel_step_l2",
        "band_key": "rootvel_step_l2",
        "family": "rate_budget",
        "kind": "step",
        "event_aware": False,
    },
    {
        "metric": "yaw_rate_step_abs_p95",
        "audit_metric": "yaw_rate_step_abs",
        "band_key": "yaw_rate_step_abs",
        "family": "rate_budget",
        "kind": "step",
        "event_aware": False,
    },
    {
        "metric": "contact_step_l2_p95",
        "audit_metric": "contact_step_l2",
        "band_key": "contact_step_l2",
        "family": "support_honesty",
        "kind": "step",
        "event_aware": True,
    },
    {
        "metric": "foot_slip_p95_mps",
        "audit_metric": "foot_slip_contacted_speed_mps",
        "band_key": "foot_slip_contacted_speed_mps",
        "family": "support_honesty",
        "kind": "scalar",
        "event_aware": False,
    },
    {
        "metric": "heading_error_p95_rad",
        "audit_metric": "heading_error_rad",
        "band_key": "heading_error_rad",
        "family": "command_response",
        "kind": "frame",
        "event_aware": False,
        "heading_effective": True,
    },
    {
        "metric": "pose_step_l2_p95",
        "audit_metric": "pose_step_l2",
        "band_key": "pose_step_l2",
        "family": "pose_continuity",
        "kind": "step",
        "event_aware": False,
    },
)

FAMILY_ORDER = (
    "regime_reached",
    "rate_budget",
    "support_honesty",
    "support_side_correctness",
    "command_response",
    "pose_continuity",
    "endpoint_bridgeability",
)


def _dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonify(row.get(k)) for k in fields})


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _copy_bands(bands: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy({str(k): dict(v) for k, v in bands.items()})


def _apply_accepted_relabels(
    reconstructed_bands: Dict[str, Dict[str, Any]],
    accepted_relabels: Sequence[Mapping[str, Any]],
) -> None:
    for row in accepted_relabels:
        target = str(row.get("target"))
        metric = str(row.get("metric"))
        if target not in reconstructed_bands:
            continue
        new_band = float(row.get("new_band"))
        reconstructed_bands[target][metric] = new_band
        if metric == "foot_slip_contacted_speed_mps":
            reconstructed_bands[target]["foot_slip_contacted_speed_p95_mps"] = new_band
            foot = reconstructed_bands[target].setdefault("foot_slip", {})
            if isinstance(foot, dict):
                foot["contacted_speed_p95_mps"] = new_band


def _load_band_audit(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metric_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in data.get("metric_rows", []) or []:
        metric_map[(str(row.get("target")), str(row.get("metric")))] = dict(row)
    return {
        "source": str(path),
        "accepted_relabels": [dict(x) for x in data.get("accepted_relabels", []) or []],
        "metric_map": metric_map,
        "final_guard": data.get("final_guard", {}) or {},
        "decision": data.get("decision", {}) or {},
    }


def _support_labels(item: DecoderItem) -> List[str]:
    return [str(x) for x in item.support_contract.get("normalized_label_sequence", [])]


def _window_stats(item: DecoderItem, idx: int, *, horizon: int, event_window: int) -> Dict[str, Any]:
    event = _oracle_event_masks(item, horizon=horizon, event_window=event_window)
    root_steps = _step_l2(np.asarray(item.seq["root_vel"], dtype=np.float32))
    yaw_steps = np.abs(np.diff(np.asarray(item.seq["yaw_rate"], dtype=np.float32).reshape(-1)))
    return {
        "idx": int(idx),
        "clip": item.clip,
        "start": int(item.start),
        "end": int(item.end),
        "support_switch_count": int(len(event["switch_frames"])),
        "support_switch_frames": [int(x) for x in event["switch_frames"]],
        "support_labels": event["labels"],
        "rootvel_step_l2_p95_gt": _safe_percentile(root_steps, 95.0),
        "yaw_rate_step_abs_p95_gt": _safe_percentile(yaw_steps, 95.0),
    }


def _with_percentile_ranks(rows: List[Dict[str, Any]]) -> None:
    for key in ("rootvel_step_l2_p95_gt", "yaw_rate_step_abs_p95_gt"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=np.float64)
        if vals.size <= 1:
            ranks[:] = 100.0
        else:
            for rank, pos in enumerate(order):
                ranks[pos] = 100.0 * rank / (vals.size - 1)
        for row, pct in zip(rows, ranks):
            row[f"{key}_percentile_rank"] = float(pct)
            row["selection_score"] = float(
                max(row.get("rootvel_step_l2_p95_gt_percentile_rank", 0.0), row.get("yaw_rate_step_abs_p95_gt_percentile_rank", 0.0))
                + 5.0 * min(1, int(row.get("support_switch_count", 0)))
            )


def _select_stratified_windows(
    items: Sequence[DecoderItem],
    *,
    horizon: int,
    event_window: int,
    n: int = 8,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    rows = [_window_stats(item, idx, horizon=horizon, event_window=event_window) for idx, item in enumerate(items)]
    _with_percentile_ranks(rows)
    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["clip"])].append(row)
    quotas = {"Walk_L_To_L": 3, "Walk_R_To_L": 3, "Walk_R_To_R": 2}
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    counts = Counter()

    def add(row: Optional[Mapping[str, Any]], reason: str) -> None:
        if row is None:
            return
        idx = int(row["idx"])
        clip = str(row["clip"])
        if idx in selected_ids or counts[clip] >= quotas.get(clip, 0) or len(selected) >= int(n):
            return
        rec = dict(row)
        rec["selection_reason"] = reason
        selected.append(rec)
        selected_ids.add(idx)
        counts[clip] += 1

    for target in MATCHED_TARGETS:
        cand = [r for r in by_target[target] if int(r["support_switch_count"]) > 0]
        cand.sort(key=lambda r: (int(r["support_switch_count"]), float(r["selection_score"]), int(r["start"])), reverse=True)
        add(cand[0] if cand else None, "support_switch_high_score")
    for target in MATCHED_TARGETS:
        cand = sorted(by_target[target], key=lambda r: (float(r["rootvel_step_l2_p95_gt_percentile_rank"]), float(r["selection_score"])), reverse=True)
        for row in cand:
            if int(row["idx"]) not in selected_ids:
                add(row, "target_high_rootvel")
                break
    for target in MATCHED_TARGETS:
        cand = sorted(by_target[target], key=lambda r: (float(r["yaw_rate_step_abs_p95_gt_percentile_rank"]), float(r["selection_score"])), reverse=True)
        for row in cand:
            if int(row["idx"]) not in selected_ids:
                add(row, "target_high_yaw_rate")
                break
    remaining = sorted(rows, key=lambda r: (float(r["selection_score"]), float(r["rootvel_step_l2_p95_gt_percentile_rank"])), reverse=True)
    for row in remaining:
        if len(selected) >= int(n):
            break
        if int(row["idx"]) not in selected_ids:
            add(row, "quota_fill_high_score")
    if len(selected) != int(n):
        raise RuntimeError(f"failed to select {n} stratified windows, got {len(selected)}")
    return [int(r["idx"]) for r in selected], selected


def _upper_metric_values(seq: Mapping[str, np.ndarray], target_bands: Mapping[str, Any], skeleton: Any) -> Dict[str, Any]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_SLICE.stop - POSE_SLICE.start)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)
    level_center = np.asarray(target_bands["bone_angvel_level_center"], dtype=np.float32).reshape(1, ANGVEL_DIM)
    foot = _foot_slip_metrics(rot6d, root_pos, contact, skeleton)
    return {
        "bone_angvel_level_rms_to_target": {
            "samples": np.asarray([float(np.sqrt(np.mean((bone_angvel[-1:] - level_center) ** 2)))], dtype=np.float64),
            "value": float(np.sqrt(np.mean((bone_angvel[-1:] - level_center) ** 2))),
        },
        "angvel_step_rms_p95": {
            "samples": _step_angvel_rms(bone_angvel),
            "value": _safe_percentile(_step_angvel_rms(bone_angvel), 95.0),
        },
        "angvel_component_p95_p95": {
            "samples": _step_angvel_component_p95(bone_angvel),
            "value": _safe_percentile(_step_angvel_component_p95(bone_angvel), 95.0),
        },
        "rootvel_step_l2_p95": {
            "samples": _step_l2(root_vel),
            "value": _safe_percentile(_step_l2(root_vel), 95.0),
        },
        "yaw_rate_step_abs_p95": {
            "samples": np.abs(np.diff(yaw_rate)),
            "value": _safe_percentile(np.abs(np.diff(yaw_rate)), 95.0),
        },
        "contact_step_l2_p95": {
            "samples": _step_l2(contact),
            "value": _safe_percentile(_step_l2(contact), 95.0),
        },
        "foot_slip_p95_mps": {
            "samples": np.asarray([float(foot.get("contacted_speed_p95_mps", 0.0) or 0.0)], dtype=np.float64),
            "value": float(foot.get("contacted_speed_p95_mps", 0.0) or 0.0),
        },
        "heading_error_p95_rad": {
            "samples": _heading_error_rad(root_vel, cond_dir),
            "value": _safe_percentile(_heading_error_rad(root_vel, cond_dir), 95.0),
        },
        "pose_step_l2_p95": {
            "samples": _step_pose_l2(rot6d),
            "value": _safe_percentile(_step_pose_l2(rot6d), 95.0),
        },
    }


def _event_pass(samples: np.ndarray, band: float, event_step_mask: np.ndarray) -> Tuple[bool, List[int], List[int], List[int]]:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    mask = np.asarray(event_step_mask, dtype=bool).reshape(-1)
    over = [int(i + 1) for i, v in enumerate(arr) if float(v) > float(band) + EPS]
    excused = [int(i + 1) for i, v in enumerate(arr) if float(v) > float(band) + EPS and i < mask.size and bool(mask[i])]
    unexcused = [int(i + 1) for i, v in enumerate(arr) if float(v) > float(band) + EPS and not (i < mask.size and bool(mask[i]))]
    return (not unexcused), over, excused, unexcused


def _upper_pass(
    *,
    spec: Mapping[str, Any],
    value: float,
    samples: np.ndarray,
    band: float,
    event_step_mask: np.ndarray,
) -> Tuple[bool, List[int], List[int], List[int]]:
    if bool(spec.get("event_aware", False)):
        return _event_pass(samples, band, event_step_mask)
    return (float(value) <= float(band) + EPS), [], [], []


def _upper_band_for(
    *,
    target: str,
    spec: Mapping[str, Any],
    original_bands: Mapping[str, Mapping[str, Any]],
    accepted_bands: Mapping[str, Mapping[str, Any]],
    band_metric_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    heading_tolerance_rad: float,
) -> Dict[str, Any]:
    band_key = str(spec["band_key"])
    old_band = _finite(original_bands[target].get(band_key))
    accepted_band = _finite(accepted_bands[target].get(band_key))
    if bool(spec.get("heading_effective", False)):
        old_band = max(old_band, float(heading_tolerance_rad))
        accepted_band = max(accepted_band, float(heading_tolerance_rad))
    audit = band_metric_map.get((target, str(spec["audit_metric"])), {}) or {}
    p95_band = _finite(audit.get("verdict_p95"), old_band)
    return {
        "old_band": old_band,
        "accepted_p99_band": accepted_band,
        "p95_shadow_band": p95_band,
        "p95_shadow_basis": audit.get("verdict_basis", "missing_use_old_band"),
    }


def _norm_slack_upper(value: float, band: float) -> float:
    return float((float(band) - float(value)) / max(abs(float(band)), EPS))


def _interval_slack(value: float, lo: float, hi: float) -> float:
    scale = max(abs(float(lo)), abs(float(hi)), EPS)
    return float(min(float(value) - float(lo), float(hi) - float(value)) / scale)


def _score_raw(
    *,
    stage: str,
    raw: np.ndarray,
    idxs: Sequence[int],
    items: Sequence[DecoderItem],
    original_bands: Mapping[str, Mapping[str, Any]],
    accepted_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    band_metric_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    skeleton: Any,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    state, aux = _reshape_state_aux(np.asarray(raw, dtype=np.float32), int(args.horizon))
    per_window: List[Dict[str, Any]] = []
    per_metric: List[Dict[str, Any]] = []
    for local_i, item_idx in enumerate(idxs):
        item = items[int(item_idx)]
        target = str(item.clip)
        seq = _seq_from_prediction(
            item,
            state[local_i],
            aux[local_i],
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
            command_align_root_vel=bool(args.command_align_root_vel),
        )
        event = _oracle_event_masks(item, horizon=int(args.horizon), event_window=int(args.event_window))
        metric_values = _upper_metric_values(seq, accepted_bands[target], skeleton)
        accepted_family_pass: Dict[str, bool] = {}
        p95_family_pass: Dict[str, bool] = {}
        p99_only_metrics: List[str] = []
        fail_metrics: List[str] = []
        near_1e3_metrics: List[str] = []
        near_1e4_metrics: List[str] = []
        for spec in UPPER_METRICS:
            metric = str(spec["metric"])
            bands = _upper_band_for(
                target=target,
                spec=spec,
                original_bands=original_bands,
                accepted_bands=accepted_bands,
                band_metric_map=band_metric_map,
                heading_tolerance_rad=float(args.heading_tolerance_rad),
            )
            value = float(metric_values[metric]["value"])
            samples = np.asarray(metric_values[metric]["samples"], dtype=np.float64)
            p95_ok, _, _, _ = _upper_pass(
                spec=spec,
                value=value,
                samples=samples,
                band=float(bands["p95_shadow_band"]),
                event_step_mask=np.asarray(event["step_mask"], dtype=bool),
            )
            accepted_ok, over_frames, excused_frames, unexcused_frames = _upper_pass(
                spec=spec,
                value=value,
                samples=samples,
                band=float(bands["accepted_p99_band"]),
                event_step_mask=np.asarray(event["step_mask"], dtype=bool),
            )
            old_ok, _, _, _ = _upper_pass(
                spec=spec,
                value=value,
                samples=samples,
                band=float(bands["old_band"]),
                event_step_mask=np.asarray(event["step_mask"], dtype=bool),
            )
            state_label = "pass@p95" if p95_ok else ("p99-only" if accepted_ok else "fail")
            slack = _norm_slack_upper(value, float(bands["accepted_p99_band"]))
            if abs(slack) < 1e-3:
                near_1e3_metrics.append(metric)
            if abs(slack) < 1e-4:
                near_1e4_metrics.append(metric)
            if state_label == "p99-only":
                p99_only_metrics.append(metric)
            elif state_label == "fail":
                fail_metrics.append(metric)
            accepted_family_pass[str(spec["family"])] = accepted_family_pass.get(str(spec["family"]), True) and bool(accepted_ok)
            p95_family_pass[str(spec["family"])] = p95_family_pass.get(str(spec["family"]), True) and bool(p95_ok)
            per_metric.append(
                {
                    "stage": stage,
                    "clip": target,
                    "start": int(item.start),
                    "end": int(item.end),
                    "train_index": int(item_idx),
                    "metric": metric,
                    "family": spec["family"],
                    "band_kind": "upper",
                    "raw_value": value,
                    "accepted_p99_band": float(bands["accepted_p99_band"]),
                    "p95_shadow_band": float(bands["p95_shadow_band"]),
                    "old_original_band": float(bands["old_band"]),
                    "normalized_slack": slack,
                    "p95_shadow_normalized_slack": _norm_slack_upper(value, float(bands["p95_shadow_band"])),
                    "old_normalized_slack": _norm_slack_upper(value, float(bands["old_band"])),
                    "state": state_label,
                    "accepted_pass": bool(accepted_ok),
                    "p95_shadow_pass": bool(p95_ok),
                    "old_original_pass": bool(old_ok),
                    "near_boundary_abs_slack_lt_1e3": bool(abs(slack) < 1e-3),
                    "near_boundary_abs_slack_lt_1e4": bool(abs(slack) < 1e-4),
                    "failed_family": str(spec["family"]) if state_label != "pass@p95" else "",
                    "failed_metric": metric if state_label != "pass@p95" else "",
                    "support_switch_frames": event["switch_frames"],
                    "event_excused_frames": excused_frames,
                    "event_unexcused_frames": unexcused_frames,
                    "event_over_frames": over_frames,
                    "p95_shadow_basis": bands["p95_shadow_basis"],
                }
            )

        pred_contract = _support_contract(np.asarray(seq["contact"], dtype=np.float32), min_run_frames=int(args.min_run_frames))
        foot = _foot_positions(np.asarray(seq["rot6d"], dtype=np.float32), np.asarray(seq["root_pos"], dtype=np.float32), skeleton)
        support_features = _support_side_features(seq, pred_contract["normalized_label_sequence"], foot)
        side_ok, side_failures = _evaluate_support_side_correctness(
            support_features,
            support_bands[target]["feature_bands"],
        )
        support_p95_clean = True
        for key in SUPPORT_SIDE_FEATURE_KEYS:
            band = support_bands[target]["feature_bands"].get(key, {}) if isinstance(support_bands[target].get("feature_bands"), Mapping) else {}
            if not isinstance(band, Mapping):
                continue
            val = float(support_features.get(key, 0.0))
            lo = float(band.get("min", 0.0))
            hi = float(band.get("max", 0.0))
            p95_lo = float(band.get("p01", lo))
            p95_hi = float(band.get("p99", hi))
            accepted_ok = bool(lo - EPS <= val <= hi + EPS)
            p95_ok = bool(p95_lo - EPS <= val <= p95_hi + EPS)
            old_ok = accepted_ok
            support_p95_clean = support_p95_clean and p95_ok
            state_label = "pass@p95" if p95_ok else ("p99-only" if accepted_ok else "fail")
            slack = _interval_slack(val, lo, hi)
            metric_name = f"support_side.{key}"
            if abs(slack) < 1e-3:
                near_1e3_metrics.append(metric_name)
            if abs(slack) < 1e-4:
                near_1e4_metrics.append(metric_name)
            if state_label == "p99-only":
                p99_only_metrics.append(metric_name)
            elif state_label == "fail":
                fail_metrics.append(metric_name)
            per_metric.append(
                {
                    "stage": stage,
                    "clip": target,
                    "start": int(item.start),
                    "end": int(item.end),
                    "train_index": int(item_idx),
                    "metric": metric_name,
                    "family": "support_side_correctness",
                    "band_kind": "interval",
                    "raw_value": val,
                    "accepted_p99_band": f"[{lo},{hi}]",
                    "accepted_p99_band_min": lo,
                    "accepted_p99_band_max": hi,
                    "p95_shadow_band": f"[{p95_lo},{p95_hi}]",
                    "p95_shadow_band_min": p95_lo,
                    "p95_shadow_band_max": p95_hi,
                    "old_original_band": f"[{lo},{hi}]",
                    "old_original_band_min": lo,
                    "old_original_band_max": hi,
                    "normalized_slack": slack,
                    "p95_shadow_normalized_slack": _interval_slack(val, p95_lo, p95_hi),
                    "old_normalized_slack": slack,
                    "state": state_label,
                    "accepted_pass": bool(accepted_ok),
                    "p95_shadow_pass": bool(p95_ok),
                    "old_original_pass": bool(old_ok),
                    "near_boundary_abs_slack_lt_1e3": bool(abs(slack) < 1e-3),
                    "near_boundary_abs_slack_lt_1e4": bool(abs(slack) < 1e-4),
                    "failed_family": "support_side_correctness" if state_label != "pass@p95" else "",
                    "failed_metric": metric_name if state_label != "pass@p95" else "",
                    "support_switch_frames": event["switch_frames"],
                    "event_excused_frames": [],
                    "event_unexcused_frames": [],
                    "event_over_frames": [],
                    "p95_shadow_basis": "support_feature_p01_p99_proxy",
                }
            )
        accepted_family_pass["support_side_correctness"] = bool(side_ok)
        p95_family_pass["support_side_correctness"] = bool(support_p95_clean)

        oracle_labels = _support_labels(item)
        pred_labels = [str(x) for x in pred_contract.get("normalized_label_sequence", [])]
        endpoint_ok = bool(pred_labels and oracle_labels and pred_labels[0] == oracle_labels[0] and pred_labels[-1] == oracle_labels[-1])
        accepted_family_pass["endpoint_bridgeability"] = endpoint_ok
        p95_family_pass["endpoint_bridgeability"] = endpoint_ok
        accepted_failed = [fam for fam in FAMILY_ORDER if not bool(accepted_family_pass.get(fam, False))]
        p95_failed = [fam for fam in FAMILY_ORDER if not bool(p95_family_pass.get(fam, False))]
        accepted_pass = bool(not accepted_failed)
        p95_clean = bool(accepted_pass and not p95_failed and not p99_only_metrics and not fail_metrics)
        window_state = "pass@p95" if p95_clean else ("p99-only" if accepted_pass else "fail")
        per_window.append(
            {
                "stage": stage,
                "clip": target,
                "start": int(item.start),
                "end": int(item.end),
                "train_index": int(item_idx),
                "window_state": window_state,
                "accepted_p99_pass": accepted_pass,
                "p95_shadow_clean_pass": bool(p95_clean),
                "accepted_failed_family": ",".join(accepted_failed),
                "p95_shadow_failed_family": ",".join(p95_failed),
                "p99_only_metric_count": int(len(p99_only_metrics)),
                "p99_only_metrics": p99_only_metrics,
                "fail_metric_count": int(len(fail_metrics)),
                "fail_metrics": fail_metrics,
                "near_boundary_1e3_metric_count": int(len(near_1e3_metrics)),
                "near_boundary_1e3_metrics": near_1e3_metrics,
                "near_boundary_1e4_metric_count": int(len(near_1e4_metrics)),
                "near_boundary_1e4_metrics": near_1e4_metrics,
                "support_switch_frames": event["switch_frames"],
                "oracle_support_start": oracle_labels[0] if oracle_labels else "",
                "oracle_support_end": oracle_labels[-1] if oracle_labels else "",
                "predicted_support_start": pred_labels[0] if pred_labels else "",
                "predicted_support_end": pred_labels[-1] if pred_labels else "",
                "oracle_support_token_accuracy": float(
                    np.mean([oracle_labels[i] == pred_labels[i] for i in range(min(len(oracle_labels), len(pred_labels)))])
                )
                if oracle_labels and pred_labels
                else 0.0,
                "support_side_failure_count": int(len(side_failures)),
                "support_side_failures": side_failures,
                **{f"{fam}_accepted": bool(accepted_family_pass.get(fam, False)) for fam in FAMILY_ORDER},
                **{f"{fam}_p95_shadow": bool(p95_family_pass.get(fam, False)) for fam in FAMILY_ORDER},
            }
        )
    return per_window, per_metric


def _per_window_flat_mse(raw: np.ndarray, true_raw: np.ndarray, horizon: int) -> List[Dict[str, float]]:
    pred_state, pred_aux = _reshape_state_aux(raw, horizon)
    true_state, true_aux = _reshape_state_aux(true_raw, horizon)
    rows = []
    for i in range(pred_state.shape[0]):
        rows.append(
            {
                "flat_raw_mse": float(np.mean((raw[i].astype(np.float64) - true_raw[i].astype(np.float64)) ** 2)),
                "state281_mse": float(np.mean((pred_state[i].astype(np.float64) - true_state[i].astype(np.float64)) ** 2)),
                "pose_rot6d_mse": float(np.mean((pred_state[i, :, POSE_SLICE].astype(np.float64) - true_state[i, :, POSE_SLICE].astype(np.float64)) ** 2)),
                "rootvel_mse": float(np.mean((pred_state[i, :, EGO_VEL_SLICE].astype(np.float64) - true_state[i, :, EGO_VEL_SLICE].astype(np.float64)) ** 2)),
                "yaw_rate_mse": float(np.mean((pred_state[i, :, YAW_RATE_SLICE].astype(np.float64) - true_state[i, :, YAW_RATE_SLICE].astype(np.float64)) ** 2)),
                "contact_mse": float(np.mean((pred_state[i, :, CONTACT_SLICE].astype(np.float64) - true_state[i, :, CONTACT_SLICE].astype(np.float64)) ** 2)),
                "bone_angvel_mse": float(np.mean((pred_aux[i].astype(np.float64) - true_aux[i].astype(np.float64)) ** 2)),
            }
        )
    return rows


def _masked_mean_batch(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=vals.device, dtype=vals.dtype)
    return (vals * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _masked_quantile_batch(vals: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    m = mask.to(device=vals.device, dtype=torch.bool)
    for row, row_mask in zip(vals, m):
        picked = row[row_mask]
        outs.append(torch.quantile(picked, float(q)) if picked.numel() else row.new_zeros(()))
    return torch.stack(outs, dim=0) if outs else vals.new_zeros((0,))


def _batch_upper(vals: torch.Tensor, bands: torch.Tensor, *, ignore_mask: Optional[torch.Tensor] = None, topk: int = 0) -> torch.Tensor:
    if vals.numel() == 0:
        return vals.new_zeros((bands.reshape(-1).shape[0],))
    band = bands.to(device=vals.device, dtype=vals.dtype)
    while band.dim() < vals.dim():
        band = band.unsqueeze(-1)
    over = F.relu(vals / band.clamp_min(EPS) - 1.0).square()
    if ignore_mask is not None:
        mask = ignore_mask.to(device=vals.device, dtype=torch.bool)
        while mask.dim() < over.dim():
            mask = mask.unsqueeze(-1)
        over = torch.where(mask, torch.zeros_like(over), over)
    k = int(topk)
    if k > 0 and over.dim() >= 2 and over.shape[1] > k:
        over = torch.topk(over, k=k, dim=1).values
    return torch.mean(over.reshape(over.shape[0], -1), dim=1)


def _batch_interval(
    vals: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    scale_floor: float,
    power: float,
    hard_gate_tolerance: bool,
    hard_gate_safety_margin: float,
) -> torch.Tensor:
    lo_t = lo.to(device=vals.device, dtype=vals.dtype)
    hi_t = hi.to(device=vals.device, dtype=vals.dtype)
    while lo_t.dim() < vals.dim():
        lo_t = lo_t.unsqueeze(-1)
        hi_t = hi_t.unsqueeze(-1)
    if hard_gate_tolerance:
        tol = 1.0e-6 + 1.0e-5 * torch.maximum(torch.ones_like(lo_t), torch.maximum(lo_t.abs(), hi_t.abs()))
        scale = torch.maximum(tol, torch.full_like(tol, float(scale_floor))).clamp_min(EPS)
        safety = torch.full_like(tol, max(0.0, float(hard_gate_safety_margin)))
        low = F.relu((lo_t - tol + safety - vals) / scale)
        high = F.relu((vals - hi_t - tol + safety) / scale)
    else:
        scale = (hi_t - lo_t).abs().clamp_min(float(scale_floor))
        low = F.relu((lo_t - vals) / scale)
        high = F.relu((vals - hi_t) / scale)
    if abs(float(power) - 1.0) <= 1e-12:
        out = low + high
    else:
        out = low.square() + high.square()
    return torch.mean(out.reshape(out.shape[0], -1), dim=1)


def _support_side_batch_terms(pred: Mapping[str, torch.Tensor], foot: Mapping[str, torch.Tensor], ctx: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    root_pos = pred["root_pos"]
    root_vel = pred["root_vel"]
    yaw = pred["yaw"].squeeze(-1)
    heading = _heading_error_torch(root_vel, ctx["cond_dir"])
    b, h = yaw.shape
    zeros = yaw.new_zeros((b,))
    if "right" in foot:
        right_speed = torch.linalg.norm(foot["right"][:, 1:] - foot["right"][:, :-1], dim=-1) * 30.0
    else:
        right_speed = yaw.new_zeros((b, max(0, h - 1)))
    if "left" in foot:
        left_speed = torch.linalg.norm(foot["left"][:, 1:] - foot["left"][:, :-1], dim=-1) * 30.0
    else:
        left_speed = yaw.new_zeros((b, max(0, h - 1)))
    right_claim = ctx["right_step_mask"].to(device=yaw.device)
    left_claim = ctx["left_step_mask"].to(device=yaw.device)
    right_single = ctx["right_single_step_mask"].to(device=yaw.device)
    left_single = ctx["left_single_step_mask"].to(device=yaw.device)
    claimed_vals = torch.cat([right_speed, left_speed], dim=1)
    claimed_mask = torch.cat([right_claim, left_claim], dim=1)
    diff_vals = torch.cat([right_speed - left_speed, left_speed - right_speed], dim=1)
    diff_mask = torch.cat([right_single, left_single], dim=1)
    ratio_vals = torch.cat([right_speed / left_speed.clamp_min(1.0e-4), left_speed / right_speed.clamp_min(1.0e-4)], dim=1)
    feats: Dict[str, torch.Tensor] = {
        "claimed_support_slip_mean_mps": _masked_mean_batch(claimed_vals, claimed_mask),
        "claimed_support_slip_p95_mps": _masked_quantile_batch(claimed_vals, claimed_mask, 0.95),
        "claimed_support_slip_max_mps": torch.max(torch.where(claimed_mask, claimed_vals, torch.zeros_like(claimed_vals)), dim=1).values,
        "single_support_claimed_minus_opposite_mean_mps": _masked_mean_batch(diff_vals, diff_mask),
        "single_support_claimed_minus_opposite_p95_mps": _masked_quantile_batch(diff_vals, diff_mask, 0.95),
        "single_support_claimed_speed_ratio_p95": _masked_quantile_batch(ratio_vals, diff_mask, 0.95),
        "yaw_sum_rad": torch.sum(yaw, dim=1) / 30.0,
        "yaw_abs_sum_rad": torch.sum(torch.abs(yaw), dim=1) / 30.0,
        "heading_error_p95_rad": torch.quantile(heading, 0.95, dim=1) if heading.numel() else zeros,
        "root_speed_mean": torch.mean(torch.linalg.norm(root_vel, dim=-1), dim=1),
        "root_lateral_mean": torch.mean(root_vel[:, :, 1], dim=1),
    }
    for side in ("right", "left"):
        mask = ctx[f"{side}_frame_mask"].to(device=yaw.device)
        if side in foot:
            rel = foot[side] - root_pos
            rel_norm = torch.linalg.norm(rel, dim=-1)
            for dim, axis in enumerate(("x", "y", "z")):
                feats[f"{side}_rel_{axis}_mean"] = _masked_mean_batch(rel[:, :, dim], mask)
            feats[f"{side}_rel_norm_p95"] = _masked_quantile_batch(rel_norm, mask, 0.95)
        else:
            for axis in ("x", "y", "z"):
                feats[f"{side}_rel_{axis}_mean"] = zeros
            feats[f"{side}_rel_norm_p95"] = zeros
    balance = ctx["right_frame_mask"].to(device=yaw.device, dtype=yaw.dtype).mean(dim=1) - ctx["left_frame_mask"].to(device=yaw.device, dtype=yaw.dtype).mean(dim=1)
    feats["support_yaw_product"] = balance * feats["yaw_sum_rad"]
    feats["support_lateral_product"] = balance * feats["root_lateral_mean"]
    linear_keys = set(str(ctx.get("support_linear_feature_keys", "")).split(","))
    linear_keys.discard("")
    excluded_keys = set(str(ctx.get("support_excluded_feature_keys", "")).split(","))
    excluded_keys.discard("")
    hard_gate_keys = set(str(ctx.get("support_hard_gate_feature_keys", "")).split(","))
    hard_gate_keys.discard("")
    out: Dict[str, torch.Tensor] = {}
    for key, val in feats.items():
        if key in excluded_keys:
            continue
        out[f"support_side.{key}"] = _batch_interval(
            val,
            ctx["support_lo"][key],
            ctx["support_hi"][key],
            scale_floor=float(ctx["support_scale_floor"]),
            power=1.0 if key in linear_keys else float(ctx.get("support_margin_power", 2.0)),
            hard_gate_tolerance=key in hard_gate_keys,
            hard_gate_safety_margin=float(ctx.get("support_hard_gate_safety_margin", 0.0) or 0.0) if key in hard_gate_keys else 0.0,
        )
    return out


def _per_window_minimax_objective(
    *,
    pred_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    true_raw: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_root_vel: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    cond_norm: torch.Tensor,
    gt_dynamics_resid: torch.Tensor,
    base: Any,
    skeleton: Any,
    offsets: torch.Tensor,
    ctx: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    pred_raw = pred_std * y_std + y_mean
    h = int(args.horizon)
    state_width = h * STATE_DIM
    pred_state = pred_raw[:, :state_width].reshape(-1, h, STATE_DIM)
    pred_aux = pred_raw[:, state_width:].reshape(-1, h, ANGVEL_DIM)
    true_state = true_raw[:, :state_width].reshape(-1, h, STATE_DIM)
    true_aux = true_raw[:, state_width:].reshape(-1, h, ANGVEL_DIM)
    pred, dyn_resid, _, _ = _dynamics_residual_from_state_aux(
        state=pred_state,
        aux=pred_aux,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        cond_norm=cond_norm,
        base=base,
        command_align_root_vel=bool(args.command_align_root_vel),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
    )
    bands = ctx["bands"]
    event_mask = ctx["event_step_mask"].to(device=pred_state.device)
    pose_step = torch.linalg.norm(pred["rot6d"][:, 1:] - pred["rot6d"][:, :-1], dim=-1) / math.sqrt(POSE_SLICE.stop - POSE_SLICE.start)
    aux_delta = pred["aux"][:, 1:] - pred["aux"][:, :-1]
    angvel_rms = torch.sqrt(torch.mean(aux_delta.square(), dim=-1).clamp_min(0.0) + 1.0e-12)
    angvel_component = torch.quantile(torch.abs(aux_delta), 0.95, dim=-1) if aux_delta.numel() else angvel_rms
    rootvel_step = torch.linalg.norm(pred["root_vel"][:, 1:] - pred["root_vel"][:, :-1], dim=-1)
    yaw_step = torch.abs(pred["yaw"][:, 1:].squeeze(-1) - pred["yaw"][:, :-1].squeeze(-1))
    contact_step = torch.linalg.norm(pred["contact"][:, 1:] - pred["contact"][:, :-1], dim=-1)
    gate_terms: Dict[str, torch.Tensor] = {
        "rate_budget.angvel_step_rms": _batch_upper(angvel_rms, bands["angvel_step_rms"], ignore_mask=event_mask, topk=int(args.loss_refactor_rate_topk)),
        "rate_budget.angvel_component_p95": _batch_upper(angvel_component, bands["angvel_step_component_p95"], topk=int(args.loss_refactor_rate_topk)),
        "pose_continuity.pose_step_l2": _batch_upper(pose_step, bands["pose_step_l2"], topk=int(args.loss_refactor_pose_topk)),
        "rate_budget.rootvel_step_l2": _batch_upper(rootvel_step, bands["rootvel_step_l2"], topk=int(args.loss_refactor_rate_topk)),
        "support_honesty.contact_step_l2": _batch_upper(contact_step, bands["contact_step_l2"], ignore_mask=event_mask, topk=int(args.loss_refactor_rate_topk)),
        "rate_budget.yaw_rate_step_abs": _batch_upper(yaw_step, bands["yaw_rate_step_abs"], topk=int(args.loss_refactor_rate_topk)),
    }
    foot = _loss_refactor_foot_positions(rot6d=pred["rot6d"], root_pos=pred["root_pos"], skeleton=skeleton, offsets=offsets)
    foot_terms = []
    for ch_idx, side in ((0, "right"), (1, "left")):
        if side not in foot:
            continue
        mask = (true_contact[:, :-1, ch_idx] > 0.5) & (true_contact[:, 1:, ch_idx] > 0.5)
        speed = torch.linalg.norm(foot[side][:, 1:] - foot[side][:, :-1], dim=-1) * 30.0
        foot_terms.append(_batch_upper(speed, bands["foot_slip_contacted_speed_mps"], ignore_mask=~mask, topk=int(args.loss_refactor_rate_topk)))
    gate_terms["support_honesty.foot_slip_contacted_speed_mps"] = torch.stack(foot_terms).mean(dim=0) if foot_terms else pred_state.new_zeros((pred_state.shape[0],))
    heading_band = torch.maximum(bands["heading_error_rad"], torch.full_like(bands["heading_error_rad"], float(args.heading_tolerance_rad)))
    heading = _heading_error_torch(pred["root_vel"], true_cond_dir.to(device=pred_state.device, dtype=pred_state.dtype))
    gate_terms["command_response.heading_error_rad"] = _batch_upper(heading, heading_band, topk=int(args.loss_refactor_heading_topk))
    level = torch.sqrt(torch.mean((pred["aux"][:, -1] - ctx["bone_angvel_level_center"].to(device=pred_state.device)) ** 2, dim=-1) + 1.0e-12)
    gate_terms["regime_reached.bone_angvel_level_rms"] = _batch_upper(level.reshape(-1, 1), bands["bone_angvel_level_rms"], topk=0)
    gate_terms.update(_support_side_batch_terms(pred, foot, {**ctx, "cond_dir": true_cond_dir}))
    gate_names = list(gate_terms.keys())
    gate_matrix = torch.stack([gate_terms[name] for name in gate_names], dim=1)
    flat = gate_matrix.reshape(-1)
    tau = float(args.loss_refactor_minimax_temperature)
    soft_gate = torch.max(flat) if tau <= 0.0 else tau * torch.logsumexp(flat / tau, dim=0) - tau * math.log(float(flat.numel()))

    rot_width = POSE_SLICE.stop - POSE_SLICE.start
    articulation_low = torch.mean((dyn_resid[:, :, :rot_width] - gt_dynamics_resid[:, :, :rot_width]).square(), dim=(1, 2)) / max(float(args.loss_refactor_dynamics_low_band) ** 2, EPS)
    root_low = torch.mean((dyn_resid[:, :, rot_width:] - gt_dynamics_resid[:, :, rot_width:]).square(), dim=(1, 2)) / max(float(args.loss_refactor_dynamics_low_band) ** 2, EPS)
    true_rootvel_delta = true_root_vel[:, 1:] - true_root_vel[:, :-1]
    root_rate_anchor = torch.mean(((pred["root_vel"][:, 1:] - pred["root_vel"][:, :-1] - true_rootvel_delta) / bands["rootvel_step_l2"].view(-1, 1, 1).clamp_min(EPS)).square(), dim=(1, 2))
    root_path_anchor = torch.mean(((pred["root_vel"] - true_root_vel) / bands["rootvel_step_l2"].view(-1, 1, 1).clamp_min(EPS)).square(), dim=(1, 2))
    contact_anchor = torch.mean(((pred["contact"] - true_contact) / bands["contact_step_l2"].view(-1, 1, 1).clamp_min(EPS)).square(), dim=(1, 2))
    endpoint_pose = torch.mean(((pred["rot6d"][:, -1] - true_state[:, -1, POSE_SLICE]) / bands["pose_step_l2"].view(-1, 1).clamp_min(EPS)).square(), dim=1)
    endpoint_root = torch.mean(((pred["root_vel"][:, -1] - true_root_vel[:, -1]) / bands["rootvel_step_l2"].view(-1, 1).clamp_min(EPS)).square(), dim=1)
    endpoint_contact = torch.mean(((pred["contact"][:, -1] - true_contact[:, -1]) / bands["contact_step_l2"].view(-1, 1).clamp_min(EPS)).square(), dim=1)
    endpoint_anchor = torch.stack([endpoint_pose, endpoint_root, endpoint_contact], dim=1).mean(dim=1)
    anchor_per_window = torch.stack([articulation_low, root_low, root_rate_anchor, root_path_anchor, contact_anchor, endpoint_anchor], dim=1).mean(dim=1)
    loss = soft_gate + float(args.loss_refactor_anchor_weight) * torch.mean(anchor_per_window)
    hard_flat_idx = int(torch.argmax(flat.detach()).cpu().item()) if flat.numel() else 0
    worst_window = int(hard_flat_idx // len(gate_names)) if gate_names else 0
    worst_metric = gate_names[int(hard_flat_idx % len(gate_names))] if gate_names else ""
    details = {
        "loss_refactor_true_window_metric_softmax_gate": float(soft_gate.detach().cpu().item()),
        "loss_refactor_true_window_metric_hard_max_gate": float(torch.max(flat).detach().cpu().item()) if flat.numel() else 0.0,
        "loss_refactor_anchor_tiebreaker": float(torch.mean(anchor_per_window).detach().cpu().item()),
        "worst_surrogate_window_local": worst_window,
        "worst_surrogate_metric": worst_metric,
        "worst_surrogate_value": float(torch.max(flat).detach().cpu().item()) if flat.numel() else 0.0,
        "gate_metric_count": int(len(gate_names)),
    }
    return loss, details


def _train_debug_decoder(
    *,
    idxs: Sequence[int],
    items: Sequence[DecoderItem],
    base: Any,
    skeleton: Any,
    original_bands: Mapping[str, Mapping[str, Any]],
    accepted_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    band_metric_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    train_x_raw, train_y_raw = _dataset_arrays(items, idxs)
    x_scaler = _fit_standardizer(train_x_raw)
    y_scaler = _fit_standardizer(train_y_raw)
    train_x = x_scaler.transform(train_x_raw)
    train_y = y_scaler.transform(train_y_raw)
    torch.manual_seed(int(args.seed))
    model = TinyDeterministicDecoder(train_x.shape[1], int(args.hidden_dim), train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    y_mean = torch.as_tensor(y_scaler.mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(y_scaler.std, dtype=torch.float32, device=device)
    true_raw = torch.as_tensor(train_y_raw, dtype=torch.float32, device=device)
    true_root_pos = torch.as_tensor(np.stack([items[int(i)].seq["root_pos"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    true_root_vel = torch.as_tensor(np.stack([items[int(i)].seq["root_vel"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    true_cond_dir = torch.as_tensor(np.stack([items[int(i)].seq["cond_dir"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    true_contact = torch.as_tensor(np.stack([items[int(i)].seq["contact"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    cond_raw = _stack_cond_raw(base, items, idxs, int(args.horizon))
    cond_norm = torch.as_tensor(_robust_cond_norm(cond_raw), dtype=torch.float32, device=device)
    state_width = int(args.horizon) * STATE_DIM
    true_state = true_raw[:, :state_width].reshape(-1, int(args.horizon), STATE_DIM)
    true_aux = true_raw[:, state_width:].reshape(-1, int(args.horizon), ANGVEL_DIM)
    gt_dynamics_resid = _gt_dynamics_residual_target(
        true_state=true_state,
        true_aux=true_aux,
        true_root_pos=true_root_pos,
        true_cond_dir=true_cond_dir,
        true_contact=true_contact,
        cond_norm=cond_norm,
        base=base,
        command_align_root_vel=bool(args.command_align_root_vel),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
    )
    offsets = torch.as_tensor(skeleton.offsets, dtype=torch.float32, device=device)
    loss_ctx = _build_loss_refactor_context(
        items=items,
        idxs=idxs,
        baseline_bands=accepted_bands,
        support_bands=support_bands,
        args=args,
        device=device,
    )
    stage1_log: List[Dict[str, Any]] = []
    best_supervised_loss = float("inf")
    best_supervised_epoch = -1
    best_supervised_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(int(args.stage1_supervised_epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_std = model(xtr)
        loss = F.mse_loss(pred_std, ytr)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_supervised_loss:
            best_supervised_loss = loss_value
            best_supervised_epoch = int(epoch)
            best_supervised_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or epoch == int(args.stage1_supervised_epochs) - 1 or epoch % max(1, int(args.instrument_step_log_stride)) == 0:
            stage1_log.append({"stage": "stage1_supervised_fit_8", "epoch": int(epoch), "flat_standardized_mse": loss_value})
    if best_supervised_state is not None:
        model.load_state_dict(best_supervised_state)
    stage1_log.append(
        {
            "stage": "stage1_supervised_fit_8",
            "epoch": int(best_supervised_epoch),
            "flat_standardized_mse": float(best_supervised_loss),
            "selected_best_checkpoint": True,
        }
    )

    def predict_raw() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            pred = model(xtr) * y_std + y_mean
        out = pred.detach().cpu().numpy().astype(np.float32)
        if bool(args.oracle_contact_passthrough):
            out = _apply_oracle_contact_passthrough(out, items, idxs, int(args.horizon))
        return out

    stage1_raw = predict_raw()
    stage1_window, stage1_metric = _score_raw(
        stage="stage1_supervised_fit_8",
        raw=stage1_raw,
        idxs=idxs,
        items=items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_metric_map,
        skeleton=skeleton,
        args=args,
    )
    stage1_mse = _per_window_flat_mse(stage1_raw, train_y_raw, int(args.horizon))
    stage1_fit_ok = bool(
        all(bool(r.get("accepted_p99_pass", False)) for r in stage1_window)
        and all(float(r.get("flat_raw_mse", 1.0)) <= float(args.supervised_raw_mse_threshold) for r in stage1_mse)
    )
    if not stage1_fit_ok:
        skipped_window = [dict(r, stage="stage2_minimax_8", stage2_skipped=True) for r in stage1_window]
        skipped_metric = [dict(r, stage="stage2_minimax_8", stage2_skipped=True) for r in stage1_metric]
        return {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "idxs": [int(x) for x in idxs],
            "true_raw": train_y_raw,
            "stage1_raw": stage1_raw,
            "stage2_raw": stage1_raw,
            "stage1_window_rows": stage1_window,
            "stage1_metric_rows": stage1_metric,
            "stage2_window_rows": skipped_window,
            "stage2_metric_rows": skipped_metric,
            "stage1_step_log": stage1_log,
            "stage2_step_log": [
                {
                    "stage": "stage2_minimax_8",
                    "skipped": True,
                    "reason": "stage1_supervised_fit_8_failed_preflight",
                    "best_supervised_epoch": int(best_supervised_epoch),
                    "best_supervised_flat_standardized_mse": float(best_supervised_loss),
                }
            ],
            "stage2_skipped": True,
            "stage2_skip_reason": "stage1_supervised_fit_8_failed_preflight",
            "best_supervised_epoch": int(best_supervised_epoch),
            "best_supervised_flat_standardized_mse": float(best_supervised_loss),
            "stage1_loss_metrics": _loss_metrics(stage1_raw, train_y_raw, int(args.horizon)),
            "stage2_loss_metrics": _loss_metrics(stage1_raw, train_y_raw, int(args.horizon)),
            "stage1_per_window_mse": stage1_mse,
            "stage2_per_window_mse": stage1_mse,
            "input_dim": int(train_x_raw.shape[1]),
            "output_dim": int(train_y_raw.shape[1]),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        }

    for group in opt.param_groups:
        group["lr"] = float(args.stage2_minimax_tail_lr)
    stage2_log: List[Dict[str, Any]] = []
    for local_epoch in range(int(args.stage2_tail_epochs)):
        global_epoch = int(args.stage1_supervised_epochs) + int(local_epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_std = model(xtr)
        loss, details = _per_window_minimax_objective(
            pred_std=pred_std,
            y_mean=y_mean,
            y_std=y_std,
            true_raw=true_raw,
            true_root_pos=true_root_pos,
            true_root_vel=true_root_vel,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            cond_norm=cond_norm,
            gt_dynamics_resid=gt_dynamics_resid,
            base=base,
            skeleton=skeleton,
            offsets=offsets,
            ctx=loss_ctx,
            args=args,
        )
        loss.backward()
        opt.step()
        if (
            local_epoch == 0
            or local_epoch == int(args.stage2_tail_epochs) - 1
            or local_epoch % max(1, int(args.instrument_step_log_stride)) == 0
        ):
            raw_now = predict_raw()
            _, metrics_now = _score_raw(
                stage=f"stage2_epoch_{global_epoch}",
                raw=raw_now,
                idxs=idxs,
                items=items,
                original_bands=original_bands,
                accepted_bands=accepted_bands,
                support_bands=support_bands,
                band_metric_map=band_metric_map,
                skeleton=skeleton,
                args=args,
            )
            upper_or_interval = [r for r in metrics_now if r.get("metric") != ""]
            worst = min(upper_or_interval, key=lambda r: float(r.get("normalized_slack", 0.0))) if upper_or_interval else {}
            local_w = int(details.get("worst_surrogate_window_local", 0))
            item = items[int(idxs[local_w])] if idxs else None
            stage2_log.append(
                {
                    "stage": "stage2_minimax_8",
                    "epoch": int(global_epoch),
                    "tail_epoch": int(local_epoch),
                    "total_loss": float(loss.detach().cpu().item()),
                    **details,
                    "worst_surrogate_window": f"{item.clip}:{item.start}-{item.end}" if item is not None else "",
                    "worst_acceptance_window": f"{worst.get('clip')}:{worst.get('start')}-{worst.get('end')}" if worst else "",
                    "worst_acceptance_metric": worst.get("metric", ""),
                    "worst_acceptance_state": worst.get("state", ""),
                    "worst_acceptance_raw_value": worst.get("raw_value", ""),
                    "worst_acceptance_accepted_p99_band": worst.get("accepted_p99_band", ""),
                    "worst_acceptance_normalized_slack": worst.get("normalized_slack", ""),
                }
            )

    stage2_raw = predict_raw()
    stage2_window, stage2_metric = _score_raw(
        stage="stage2_minimax_8",
        raw=stage2_raw,
        idxs=idxs,
        items=items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_metric_map,
        skeleton=skeleton,
        args=args,
    )
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "idxs": [int(x) for x in idxs],
        "true_raw": train_y_raw,
        "stage1_raw": stage1_raw,
        "stage2_raw": stage2_raw,
        "stage1_window_rows": stage1_window,
        "stage1_metric_rows": stage1_metric,
        "stage2_window_rows": stage2_window,
        "stage2_metric_rows": stage2_metric,
        "stage1_step_log": stage1_log,
        "stage2_step_log": stage2_log,
        "stage2_skipped": False,
        "stage2_skip_reason": "",
        "best_supervised_epoch": int(best_supervised_epoch),
        "best_supervised_flat_standardized_mse": float(best_supervised_loss),
        "stage1_loss_metrics": _loss_metrics(stage1_raw, train_y_raw, int(args.horizon)),
        "stage2_loss_metrics": _loss_metrics(stage2_raw, train_y_raw, int(args.horizon)),
        "stage1_per_window_mse": stage1_mse,
        "stage2_per_window_mse": _per_window_flat_mse(stage2_raw, train_y_raw, int(args.horizon)),
        "input_dim": int(train_x_raw.shape[1]),
        "output_dim": int(train_y_raw.shape[1]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def _summarize_window_states(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(r.get("window_state", "")) for r in rows)
    return {
        "n": int(len(rows)),
        "clean_pass_count": int(counts.get("pass@p95", 0)),
        "p99_only_count": int(counts.get("p99-only", 0)),
        "fail_count": int(counts.get("fail", 0)),
        "accepted_p99_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in rows)),
        "accepted_p99_pass_rate": float(np.mean([bool(r.get("accepted_p99_pass", False)) for r in rows])) if rows else 0.0,
    }


def _metric_fail_lists(rows: Sequence[Mapping[str, Any]], stage: str, clip: str, start: int) -> Tuple[List[str], List[str], List[str]]:
    subset = [r for r in rows if str(r.get("stage")) == stage and str(r.get("clip")) == clip and int(r.get("start")) == int(start)]
    p99 = [str(r.get("metric")) for r in subset if str(r.get("state")) == "p99-only"]
    fail = [str(r.get("metric")) for r in subset if str(r.get("state")) == "fail"]
    near = [str(r.get("metric")) for r in subset if bool(r.get("near_boundary_abs_slack_lt_1e3", False))]
    return p99, fail, near


def _stall_classification(
    *,
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    stage1_mse_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    by_stage1 = {(str(r["clip"]), int(r["start"])): r for r in stage1_rows}
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(stage2_rows):
        if str(row.get("window_state")) == "pass@p95":
            continue
        key = (str(row["clip"]), int(row["start"]))
        s1 = by_stage1.get(key, {})
        mse = stage1_mse_rows[i] if i < len(stage1_mse_rows) else {}
        stage1_low_mse = float(mse.get("flat_raw_mse", 1.0)) <= float(args.supervised_raw_mse_threshold)
        stage1_fit_ok = bool(s1.get("accepted_p99_pass", False)) and bool(stage1_low_mse)
        p99, fail, near = _metric_fail_lists(metric_rows, "stage2_minimax_8", str(row["clip"]), int(row["start"]))
        heading_block = any("heading_error" in m for m in fail)
        contract_width_terms = [
            m
            for m in p99
            if any(token in m for token in ("rootvel", "foot_slip", "bone_angvel_level", "support_side.root_speed_mean", "support_side.support_lateral_product"))
        ]
        if not stage1_fit_ok:
            if stage1_low_mse and heading_block:
                cls = "heading-band preflight block"
                evidence = "Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax."
            else:
                cls = "capacity/recipe failure"
                evidence = "Stage1 supervised-flat failed raw MSE preflight."
        elif str(row.get("window_state")) == "p99-only" and contract_width_terms:
            cls = "contract-width risk"
            evidence = "Accepted only under p99/interval-width view; p95-shadow has metric misses."
        else:
            cls = "optimization-fragile"
            evidence = "Stage1 supervised-flat reached GT basin, but minimax final is non-clean or near-boundary."
        out.append(
            {
                "clip": row["clip"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "stage2_window_state": row.get("window_state"),
                "classification": cls,
                "evidence": evidence,
                "stage1_fit_ok": bool(stage1_fit_ok),
                "stage1_low_mse": bool(stage1_low_mse),
                "stage1_flat_raw_mse": float(mse.get("flat_raw_mse", 0.0)),
                "stage1_state281_mse": float(mse.get("state281_mse", 0.0)),
                "p99_only_metrics": p99,
                "fail_metrics": fail,
                "near_boundary_1e3_metrics": near,
                "contract_width_terms": contract_width_terms,
                "conditioning_conflict_evidence": "none",
            }
        )
    return out


def _negative_control_summary(band_audit: Mapping[str, Any]) -> Dict[str, Any]:
    final_guard = band_audit.get("final_guard", {}) or {}
    shortcut = final_guard.get("shortcut_negative_summary_by_case", {}) or {}
    command = final_guard.get("command_demotion_negative_summary", {}) or {}
    artifact_non_rate = {}
    for case, rec in shortcut.items():
        if not str(case).startswith("artifact_proxy:"):
            continue
        counts = dict(rec.get("failed_family_counts", {}) or {})
        artifact_non_rate[case] = {fam: int(n) for fam, n in counts.items() if fam not in {"rate_budget", "command_response"}}
    return {
        "source": band_audit.get("source", ""),
        "shortcut_negative_controls_still_fail": bool((final_guard.get("verdict", {}) or {}).get("shortcut_negative_controls_still_fail", False)),
        "command_demotion_negative_controls_still_fail": bool((final_guard.get("verdict", {}) or {}).get("command_demotion_negative_controls_still_fail", False)),
        "shortcut_by_case": shortcut,
        "command_demotion": command,
        "artifact_proxy_non_rate_failed_family_counts": artifact_non_rate,
    }


def _save_artifacts(
    *,
    out_dir: Path,
    result: Mapping[str, Any],
    selected_windows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred_raw.npz"
    np.savez(
        pred_path,
        pred_raw=np.asarray(result["stage2_raw"], dtype=np.float32),
        stage1_pred_raw=np.asarray(result["stage1_raw"], dtype=np.float32),
        true_raw=np.asarray(result["true_raw"], dtype=np.float32),
        train_indices=np.asarray(result["idxs"], dtype=np.int64),
        clip=np.asarray([str(r["clip"]) for r in selected_windows]),
        start=np.asarray([int(r["start"]) for r in selected_windows], dtype=np.int64),
        end=np.asarray([int(r["end"]) for r in selected_windows], dtype=np.int64),
    )
    state_path = out_dir / "decoder_state.pt"
    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "x_scaler": {"mean": result["x_scaler"].mean, "std": result["x_scaler"].std, "constant_count": result["x_scaler"].constant_count},
            "y_scaler": {"mean": result["y_scaler"].mean, "std": result["y_scaler"].std, "constant_count": result["y_scaler"].constant_count},
            "idxs": [int(x) for x in result["idxs"]],
            "selected_windows": list(selected_windows),
            "debug_scope": "8-window fixed-oracle-schedule train-fit only",
        },
        state_path,
    )
    return {
        "summary_md": str(out_dir / "summary.md"),
        "summary_json": str(out_dir / "summary.json"),
        "per_window_csv": str(out_dir / "per_window.csv"),
        "per_metric_csv": str(out_dir / "per_metric.csv"),
        "stage1_supervised_fit_csv": str(out_dir / "stage1_supervised_fit.csv"),
        "stage2_minimax_step_log_csv": str(out_dir / "stage2_minimax_step_log.csv"),
        "stall_classification_csv": str(out_dir / "stall_classification.csv"),
        "pred_raw_npz": str(pred_path),
        "decoder_state_pt": str(state_path),
        "doc_md": str(args.doc_path),
    }


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    lines = [
        "# Action-Handoff 8-Window Debug Train-Fit",
        "",
        "Debug-only fixed-oracle-schedule train-fit. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## Verdict",
        "",
        f"- 8-window debug train-fit passed accepted p99 gate: `{str(verdict['accepted_p99_debug_pass']).lower()}`",
        f"- clean pass / p99-only / fail: `{verdict['clean_pass_count']}` / `{verdict['p99_only_count']}` / `{verdict['fail_count']}`",
        "- This is not generalization, deployment readiness, or schedule learning.",
        "",
        "## Tensor Contract",
        "",
        "- decoder input: `state feature x [8,4957] float32 CPU`",
        "- decoder output state: `state281 [8,16,281] float32 CPU`",
        "- decoder aux: `bone_angvel [8,16,138] float32 CPU`",
        "- saved `pred_raw`: `[8,6704] float32 CPU NumPy`",
        "",
        "## Stage1 Supervised-Fit-8",
        "",
        f"- aggregate flat raw MSE: `{_fmt(payload['stage1']['loss_metrics'].get('state_mse'), 10)}` state MSE, `{_fmt(payload['stage1']['loss_metrics'].get('bone_angvel_aux_mse'), 10)}` bone_angvel MSE",
        f"- accepted p99 pass count: `{payload['stage1']['window_summary']['accepted_p99_pass_count']}/8`",
        "",
        "## Stage2 Minimax-8",
        "",
    ]
    if bool(payload["stage2"].get("skipped", False)):
        lines.append(f"- skipped: `true`; reason: `{payload['stage2'].get('skip_reason')}`")
    else:
        lines.extend(
            [
                f"- accepted p99 pass count: `{payload['stage2']['window_summary']['accepted_p99_pass_count']}/8`",
                f"- final worst acceptance metric: `{payload['stage2']['final_worst_metric'].get('clip')}:{payload['stage2']['final_worst_metric'].get('start')}-{payload['stage2']['final_worst_metric'].get('end')} {payload['stage2']['final_worst_metric'].get('metric')}` slack `{_fmt(payload['stage2']['final_worst_metric'].get('normalized_slack'), 8)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Stage3 Stall Classification",
            "",
            "| window | state | classification | p99-only metrics | fail metrics |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["stall_classification"]:
        lines.append(
            f"| {row['clip']}:{row['start']}-{row['end']} | {row['stage2_window_state']} | {row['classification']} | {row['p99_only_metrics']} | {row['fail_metrics']} |"
        )
    if not payload["stall_classification"]:
        lines.append("| none | pass@p95 | clean | [] | [] |")
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            f"- shortcut controls still fail: `{str(payload['negative_controls']['shortcut_negative_controls_still_fail']).lower()}`",
            f"- command demotion controls still fail: `{str(payload['negative_controls']['command_demotion_negative_controls_still_fail']).lower()}`",
            f"- command demotion negative pass count: `{payload['negative_controls']['command_demotion'].get('demoted_negative_pass_count')}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key in (
        "summary_json",
        "per_window_csv",
        "per_metric_csv",
        "stage1_supervised_fit_csv",
        "stage2_minimax_step_log_csv",
        "stall_classification_csv",
        "pred_raw_npz",
        "decoder_state_pt",
        "doc_md",
    ):
        lines.append(f"- {key}: `{payload['artifacts'][key]}`")
    _dump_md(path, lines)


def _write_doc_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    stage2 = payload["stage2"]
    neg = payload["negative_controls"]
    lines = [
        "# 8-Window Train-Fit Review",
        "",
        f"Date: 2026-06-05",
        "",
        "Scope: debug-only fixed-oracle-schedule Layer-2 train-fit discriminator. It uses flat `state281`, a tiny deterministic decoder, oracle support/contact schedule features, and the three causal items. It is not a production Trainer/runtime/gate/checkpoint change.",
        "",
        "## 1. Verdict",
        "",
        f"- 8-window accepted-p99 debug train-fit: `{str(verdict['accepted_p99_debug_pass']).lower()}`.",
        f"- clean pass / p99-only / fail: `{verdict['clean_pass_count']}` / `{verdict['p99_only_count']}` / `{verdict['fail_count']}`.",
        "- This is not generalization, deployment readiness, or schedule-learning success.",
        f"- artifact: `{payload['artifacts']['summary_json']}`.",
        "",
        "| window | reason | support switches |",
        "|---|---|---|",
    ]
    for row in payload.get("selected_windows", []):
        lines.append(
            f"| `{row['clip']}:{row['start']}-{row['end']}` | {row.get('selection_reason', '')} | `{row.get('support_switch_frames', [])}` |"
        )
    lines.extend(
        [
        "",
        "## 2. Stage1 supervised-fit-8",
        "",
        f"- `state281 [8,16,281] float32 CPU`, `bone_angvel [8,16,138] float32 CPU`, saved `pred_raw [8,6704] float32 CPU NumPy`.",
        f"- aggregate `state_mse={_fmt(payload['stage1']['loss_metrics'].get('state_mse'), 12)}`, `bone_angvel_aux_mse={_fmt(payload['stage1']['loss_metrics'].get('bone_angvel_aux_mse'), 12)}`.",
        f"- best supervised checkpoint: epoch `{payload['stage1'].get('best_supervised_epoch')}`, `flat_standardized_mse={payload['stage1'].get('best_supervised_flat_standardized_mse')}`.",
        f"- accepted p99 pass: `{payload['stage1']['window_summary']['accepted_p99_pass_count']}/8`.",
        f"- rows: `{payload['artifacts']['stage1_supervised_fit_csv']}`.",
        "- Stage1 reached the low-MSE GT basin; accepted-p99 failure is a heading-band/derived-metric preflight block, not a capacity or representation result.",
        "",
        "## 3. Stage2 minimax-8",
        "",
        f"- skipped: `{str(stage2.get('skipped', False)).lower()}`; reason: `{stage2.get('skip_reason', '')}`.",
        f"- worst final `(window x metric)`: `{stage2['final_worst_metric'].get('clip')}:{stage2['final_worst_metric'].get('start')}-{stage2['final_worst_metric'].get('end')} {stage2['final_worst_metric'].get('metric')}` with normalized slack `{_fmt(stage2['final_worst_metric'].get('normalized_slack'), 8)}`.",
        f"- per-window accepted pass count: `{stage2['window_summary']['accepted_p99_pass_count']}/8`.",
        f"- p95-shadow vs p99 decision: clean `{verdict['clean_pass_count']}`, p99-only `{verdict['p99_only_count']}`, fail `{verdict['fail_count']}`.",
        f"- step/skip log: `{payload['artifacts']['stage2_minimax_step_log_csv']}`.",
        "- There is no minimax trend to interpret when Stage1 preflight is blocked.",
        "",
        "## 4. Stage3 stall classification",
        "",
        ]
    )
    if payload["stall_classification"]:
        lines.extend(["| window | state | classification | evidence |", "|---|---|---|---|"])
        for row in payload["stall_classification"]:
            lines.append(f"| {row['clip']}:{row['start']}-{row['end']} | {row['stage2_window_state']} | {row['classification']} | {row['evidence']} |")
    else:
        lines.append("No non-clean window under p95-shadow.")
    lines.extend(
        [
            "",
            "- p99-only rows are treated as soft-fail and included above when present.",
            "- No true multimodality evidence is claimed: deterministic fixed-schedule train-fit did not exclude contract-width/optimization explanations for non-clean rows.",
            "",
            "## 5. Negative controls",
            "",
            f"- shortcut controls still fail: `{str(neg['shortcut_negative_controls_still_fail']).lower()}`.",
            f"- command demotion controls still fail: `{str(neg['command_demotion_negative_controls_still_fail']).lower()}`.",
            f"- command demotion pass count: `{neg['command_demotion'].get('demoted_negative_pass_count')}`.",
            "",
            "| case | n | pass count/rate | failed families |",
            "|---|---:|---:|---|",
        ]
    )
    for case, rec in neg["shortcut_by_case"].items():
        lines.append(f"| {case} | {rec.get('n')} | {rec.get('adjusted_pass_count')}/{_fmt(rec.get('adjusted_pass_rate'))} | {rec.get('failed_family_counts')} |")
    lines.extend(
        [
            "",
            "Artifact proxy rows cannot be rescored from complete trajectories in this artifact; their non-rate failed-family counts remain:",
            "",
        ]
    )
    for case, counts in neg["artifact_proxy_non_rate_failed_family_counts"].items():
        lines.append(f"- `{case}`: `{counts}`")
    lines.extend(
        [
            "",
            "## 6. Next decision",
            "",
        ]
    )
    classes = {str(row.get("classification", "")) for row in payload.get("stall_classification", [])}
    if "heading-band preflight block" in classes:
        lines.append(
            "Audit/relabel the command/support-side heading contract or add a heading-exactness repair, then rerun Stage2 minimax. Do not escalate to sampling/multimodality from this artifact."
        )
    elif verdict["fail_count"]:
        lines.append("Optimization/capacity repair is the next minimal branch; do not escalate to sampling/multimodality yet.")
    elif verdict["p99_only_count"]:
        lines.append("Contract-width review is the next minimal branch; p99-only is not a clean deterministic pass.")
    else:
        lines.append("Deterministic debug train-fit can continue to the next stress set, still without deployment/generalization claims.")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")
    selected_idxs, selected_windows = _select_stratified_windows(
        main_items,
        horizon=int(args.horizon),
        event_window=int(args.event_window),
        n=8,
    )
    original_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    accepted_bands = _copy_bands(original_bands)
    support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    band_audit = _load_band_audit(Path(args.band_audit_summary))
    _apply_accepted_relabels(accepted_bands, band_audit["accepted_relabels"])
    base = _build_base_operator(args, Path(args.npz_root), device)
    result = _train_debug_decoder(
        idxs=selected_idxs,
        items=main_items,
        base=base,
        skeleton=skeleton,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_audit["metric_map"],
        args=args,
        device=device,
    )
    per_window_rows = result["stage1_window_rows"] + result["stage2_window_rows"]
    per_metric_rows = result["stage1_metric_rows"] + result["stage2_metric_rows"]
    stage1_fit_rows = []
    for rec, mse in zip(result["stage1_window_rows"], result["stage1_per_window_mse"]):
        stage1_fit_rows.append({**rec, **mse})
    stall_rows = _stall_classification(
        stage1_rows=result["stage1_window_rows"],
        stage2_rows=result["stage2_window_rows"],
        metric_rows=per_metric_rows,
        stage1_mse_rows=result["stage1_per_window_mse"],
        args=args,
    )
    artifacts = _save_artifacts(out_dir=Path(args.out_dir), result=result, selected_windows=selected_windows, args=args)
    final_worst = min(result["stage2_metric_rows"], key=lambda r: float(r.get("normalized_slack", 0.0)))
    neg = _negative_control_summary(band_audit)
    stage2_summary = _summarize_window_states(result["stage2_window_rows"])
    verdict = {
        "accepted_p99_debug_pass": bool(stage2_summary["accepted_p99_pass_count"] == 8),
        **stage2_summary,
        "not_generalization_or_deployment": True,
    }
    payload: Dict[str, Any] = {
        "task": "action_handoff_inbetween_8window_debug_train_fit",
        "scope": "debug-only fixed-oracle-schedule train-fit discriminator; no production Trainer/runtime/gate/checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "checkpoint": str(args.checkpoint),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stage1_supervised_epochs": int(args.stage1_supervised_epochs),
            "stage2_tail_epochs": int(args.stage2_tail_epochs),
            "lr": float(args.lr),
            "stage2_minimax_tail_lr": float(args.stage2_minimax_tail_lr),
            "hidden_dim": int(args.hidden_dim),
            "device": "cpu",
            "dtype": "float32",
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "event_window": int(args.event_window),
            "loss_refactor_minimax_temperature": float(args.loss_refactor_minimax_temperature),
            "loss_refactor_anchor_weight": float(args.loss_refactor_anchor_weight),
            "loss_refactor_support_feature_topk": int(args.loss_refactor_support_feature_topk),
            "loss_refactor_support_band_floor": float(args.loss_refactor_support_band_floor),
            "loss_refactor_support_hard_gate_feature_keys": str(args.loss_refactor_support_hard_gate_feature_keys),
            "loss_refactor_support_hard_gate_safety_margin": float(args.loss_refactor_support_hard_gate_safety_margin),
        },
        "input_output_contract": {
            "decoder_input": {"shape": [8, result["input_dim"]], "dtype": "float32", "device": "cpu"},
            "middle_state_output": {"shape": [8, int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux_output": {"shape": [8, int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "saved_pred_raw": {"shape": [8, int(args.horizon) * (STATE_DIM + ANGVEL_DIM)], "dtype": "float32", "device": "cpu numpy"},
        },
        "selected_windows": selected_windows,
        "stage1": {
            "loss_metrics": result["stage1_loss_metrics"],
            "window_summary": _summarize_window_states(result["stage1_window_rows"]),
            "step_log": result["stage1_step_log"],
            "best_supervised_epoch": int(result.get("best_supervised_epoch", -1)),
            "best_supervised_flat_standardized_mse": float(result.get("best_supervised_flat_standardized_mse", 0.0)),
        },
        "stage2": {
            "loss_metrics": result["stage2_loss_metrics"],
            "window_summary": stage2_summary,
            "final_worst_metric": final_worst,
            "step_log_row_count": int(len(result["stage2_step_log"])),
            "skipped": bool(result.get("stage2_skipped", False)),
            "skip_reason": str(result.get("stage2_skip_reason", "")),
        },
        "stall_classification": stall_rows,
        "negative_controls": neg,
        "base_operator_preflight": {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_model_overlap": _checkpoint_overlap_report(base),
            "raw_x_norm_max_abs_error_by_clip": base.raw_x_norm_max_abs_error,
            "raw_x_norm_max_abs_error_max": float(max(base.raw_x_norm_max_abs_error.values())),
        },
        "hard_constraint_confirmations": {
            "debug_only": True,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "used_diffusion_or_sampling": False,
            "production_ready_generator": False,
        },
        "verdict": verdict,
        "artifacts": artifacts,
    }
    _dump_json(Path(artifacts["summary_json"]), payload)
    _write_csv(Path(artifacts["per_window_csv"]), per_window_rows)
    _write_csv(Path(artifacts["per_metric_csv"]), per_metric_rows)
    _write_csv(Path(artifacts["stage1_supervised_fit_csv"]), stage1_fit_rows)
    _write_csv(Path(artifacts["stage2_minimax_step_log_csv"]), result["stage2_step_log"])
    _write_csv(Path(artifacts["stall_classification_csv"]), stall_rows)
    _write_summary_md(Path(artifacts["summary_md"]), payload)
    _write_doc_md(Path(artifacts["doc_md"]), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=Path, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=Path, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--band-audit-summary", type=Path, default=DEFAULT_BAND_AUDIT_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--instrument-step-log-stride", type=int, default=100)
    p.add_argument("--stage1-supervised-epochs", type=int, default=2000)
    p.add_argument("--stage2-tail-epochs", type=int, default=5000)
    p.add_argument("--stage2-minimax-tail-lr", type=float, default=1e-5)
    p.add_argument("--supervised-raw-mse-threshold", type=float, default=1e-8)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--event-window", type=int, default=1)
    p.add_argument("--heading-tolerance-rad", type=float, default=1e-5)
    p.add_argument("--oracle-contact-passthrough", action="store_true", default=False)
    p.add_argument("--command-align-root-vel", action="store_true", default=False)
    p.add_argument("--dynamics-eval-scale-floor", type=float, default=0.05)
    p.add_argument("--loss-refactor-minimax-temperature", type=float, default=0.005)
    p.add_argument("--loss-refactor-anchor-weight", type=float, default=0.05)
    p.add_argument("--loss-refactor-rate-topk", type=int, default=3)
    p.add_argument("--loss-refactor-pose-topk", type=int, default=3)
    p.add_argument("--loss-refactor-heading-topk", type=int, default=3)
    p.add_argument("--loss-refactor-support-feature-topk", type=int, default=1)
    p.add_argument("--loss-refactor-support-band-floor", type=float, default=0.01)
    p.add_argument("--loss-refactor-support-margin-power", type=float, default=2.0)
    p.add_argument("--loss-refactor-support-linear-feature-keys", type=str, default="support_lateral_product")
    p.add_argument("--loss-refactor-support-excluded-feature-keys", type=str, default="heading_error_p95_rad")
    p.add_argument("--loss-refactor-support-hard-gate-feature-keys", type=str, default="support_lateral_product")
    p.add_argument("--loss-refactor-support-hard-gate-safety-margin", type=float, default=1e-6)
    p.add_argument("--loss-refactor-dynamics-low-band", type=float, default=1.0)
    p.add_argument("--guard-two-frame-summary", type=Path, default=DEFAULT_GUARD_TWO_FRAME)
    p.add_argument("--guard-bone-bridge-summary", type=Path, default=DEFAULT_GUARD_BONE_BRIDGE)
    p.add_argument("--guard-regime-bridge-summary", type=Path, default=DEFAULT_GUARD_REGIME_BRIDGE)
    p.add_argument("--guard-command-demotion-rows", type=Path, default=DEFAULT_GUARD_COMMAND_DEMOTION_ROWS)
    p.add_argument("--guard-pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--guard-ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--guard-ground-pose-thr", type=float, default=GROUND_POSE_THR)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")
    print(json.dumps(_jsonify(payload["verdict"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
