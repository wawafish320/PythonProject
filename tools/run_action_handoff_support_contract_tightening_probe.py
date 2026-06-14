#!/usr/bin/env python3
"""Tighten support contract diagnostics for middle-state feasibility.

Read-only probe. No training, no model forward, no checkpoint mutation, no
production trainer/runtime/gate change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_TOPK,
    TURN_CLIPS,
    WALK_F,
    full_state_align,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    POSE_DIM,
    ClipData,
    SkeletonMeta,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _evaluate_sequence,
    _fmt,
    _foot_positions,
    _heading_error_rad,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
    _support_label,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_BASELINE_QUANTILE,
    DEFAULT_BRIDGE_BUDGET_QUANTILE,
    DEFAULT_HORIZON,
    DEFAULT_POSE_BUCKET_RADIUS,
    _bridge_budgets,
    _endpoint_bridge,
    _replace_seq,
    _root_shifted_target_sequence,
    _shuffled_contact,
    _support_entropy,
    _support_transition_count,
    _walk_cue_sequence,
    _wrong_side_contact,
)


FPS = 60.0
EPS = 1e-8
LABEL_ORDER = ("right", "left", "dual", "flight_or_unknown")
SIDE_ORDER = ("right", "left")

DEFAULT_AVAILABLE_BUCKET_RADIUS = 0.035
DEFAULT_VELOCITY_BUCKET_RADIUS = 0.045

SUPPORT_SIDE_FEATURE_KEYS = (
    "right_frame_fraction",
    "left_frame_fraction",
    "dual_frame_fraction",
    "flight_frame_fraction",
    "transition_count",
    "support_entropy_bits",
    "right_run_max",
    "left_run_max",
    "single_support_run_min",
    "single_support_run_max",
    "claimed_support_slip_mean_mps",
    "claimed_support_slip_p95_mps",
    "claimed_support_slip_max_mps",
    "single_support_claimed_minus_opposite_mean_mps",
    "single_support_claimed_minus_opposite_p95_mps",
    "single_support_claimed_speed_ratio_p95",
    "right_rel_x_mean",
    "right_rel_y_mean",
    "right_rel_z_mean",
    "right_rel_norm_p95",
    "left_rel_x_mean",
    "left_rel_y_mean",
    "left_rel_z_mean",
    "left_rel_norm_p95",
    "yaw_sum_rad",
    "yaw_abs_sum_rad",
    "heading_error_p95_rad",
    "root_speed_mean",
    "root_lateral_mean",
    "support_side_balance",
    "support_yaw_product",
    "support_lateral_product",
)


@dataclass(frozen=True)
class Run:
    label: str
    start: int
    length: int


@dataclass
class WindowItem:
    clip: str
    start: int
    end: int
    seq: Dict[str, np.ndarray]
    ctx: np.ndarray
    support_contract: Dict[str, Any]
    support_side_correctness: bool
    support_side_failures: List[Dict[str, Any]]
    feature_by_tier: Dict[str, np.ndarray]


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _support_labels(contact: np.ndarray) -> List[str]:
    return [_support_label(c) for c in np.asarray(contact, dtype=np.float32).reshape(-1, 2)]


def _rle(labels: Sequence[str]) -> List[Run]:
    if not labels:
        return []
    out: List[Run] = []
    cur = str(labels[0])
    start = 0
    for idx, label in enumerate(labels[1:], start=1):
        label = str(label)
        if label == cur:
            continue
        out.append(Run(cur, start, idx - start))
        cur = label
        start = idx
    out.append(Run(cur, start, len(labels) - start))
    return out


def _dominant_label(labels: Sequence[str]) -> str:
    right = sum(1 for x in labels if _label_has_side(str(x), "right"))
    left = sum(1 for x in labels if _label_has_side(str(x), "left"))
    if right > left + 1:
        return "right"
    if left > right + 1:
        return "left"
    return "mixed"


def _signature_from_labels(labels: Sequence[str], ambiguous_runs: Sequence[Mapping[str, Any]]) -> str:
    runs = _rle(labels)
    if not runs:
        return "empty"
    rle_text = ">".join(f"{r.label}:{r.length}" for r in runs)
    return (
        f"{labels[0]}->{labels[-1]}|runs={rle_text}|dom={_dominant_label(labels)}|"
        f"trans={_support_transition_count(labels)}|amb={len(ambiguous_runs)}"
    )


def _normalize_support_labels(labels: Sequence[str], min_run_frames: int) -> Dict[str, Any]:
    min_run = max(1, int(min_run_frames))
    work = [str(x) for x in labels]
    merge_events: List[Dict[str, Any]] = []
    while True:
        runs = _rle(work)
        changed = False
        for idx, run in enumerate(runs):
            if run.length >= min_run or idx == 0 or idx == len(runs) - 1:
                continue
            prev_run = runs[idx - 1]
            next_run = runs[idx + 1]
            if prev_run.label != next_run.label:
                continue
            for j in range(run.start, run.start + run.length):
                work[j] = prev_run.label
            merge_events.append(
                {
                    "start": int(run.start),
                    "length": int(run.length),
                    "from": run.label,
                    "to": prev_run.label,
                }
            )
            changed = True
            break
        if not changed:
            break

    ambiguous: List[Dict[str, Any]] = []
    runs = _rle(work)
    for idx, run in enumerate(runs):
        if run.length >= min_run:
            continue
        prev_label = runs[idx - 1].label if idx > 0 else None
        next_label = runs[idx + 1].label if idx + 1 < len(runs) else None
        ambiguous.append(
            {
                "start": int(run.start),
                "length": int(run.length),
                "label": run.label,
                "prev": prev_label,
                "next": next_label,
                "reason": "short_run_not_mergeable",
            }
        )
    return {
        "labels": work,
        "runs": [{"label": r.label, "start": int(r.start), "length": int(r.length)} for r in runs],
        "merge_events": merge_events,
        "ambiguous_runs": ambiguous,
        "ambiguous": bool(ambiguous),
    }


def _support_contract(contact: np.ndarray, min_run_frames: int) -> Dict[str, Any]:
    raw = _support_labels(contact)
    norm = _normalize_support_labels(raw, min_run_frames=min_run_frames)
    norm_labels = list(norm["labels"])
    raw_ambiguous: List[Dict[str, Any]] = []
    return {
        "schedule_label_sequence": raw,
        "normalized_label_sequence": norm_labels,
        "raw_signature": _signature_from_labels(raw, raw_ambiguous),
        "normalized_signature": _signature_from_labels(norm_labels, norm["ambiguous_runs"]),
        "raw_transition_count": int(_support_transition_count(raw)),
        "normalized_transition_count": int(_support_transition_count(norm_labels)),
        "raw_entropy_bits": float(_support_entropy(raw)),
        "normalized_entropy_bits": float(_support_entropy(norm_labels)),
        "normalization": norm,
    }


def _label_has_side(label: str, side: str) -> bool:
    if label == "dual":
        return True
    return label == side


def _mask_run_lengths(mask: np.ndarray) -> List[int]:
    vals = [bool(x) for x in mask.reshape(-1)]
    lengths: List[int] = []
    cur = 0
    for v in vals:
        if v:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def _p95(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 95))


def _mean(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _support_side_features(
    seq: Mapping[str, np.ndarray],
    labels: Sequence[str],
    foot: Optional[Mapping[str, np.ndarray]],
) -> Dict[str, float]:
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)
    labels = [str(x) for x in labels]
    h = len(labels)

    right_mask = np.asarray([_label_has_side(x, "right") for x in labels], dtype=bool)
    left_mask = np.asarray([_label_has_side(x, "left") for x in labels], dtype=bool)
    single_mask = np.asarray([x in {"right", "left"} for x in labels], dtype=bool)
    right_runs = _mask_run_lengths(right_mask)
    left_runs = _mask_run_lengths(left_mask)
    single_runs = _mask_run_lengths(single_mask)

    speeds: Dict[str, np.ndarray] = {}
    if foot is not None:
        for side in SIDE_ORDER:
            if side in foot and np.asarray(foot[side]).shape[0] >= h:
                pos = np.asarray(foot[side], dtype=np.float32).reshape(-1, 3)[:h]
                speeds[side] = np.linalg.norm(pos[1:] - pos[:-1], axis=1) * FPS

    claimed_speeds: List[float] = []
    claimed_minus_opp: List[float] = []
    claimed_ratio: List[float] = []
    for i in range(max(0, h - 1)):
        for side, opp in (("right", "left"), ("left", "right")):
            side_claimed = _label_has_side(labels[i], side) and _label_has_side(labels[i + 1], side)
            opp_claimed = _label_has_side(labels[i], opp) and _label_has_side(labels[i + 1], opp)
            if not side_claimed or side not in speeds:
                continue
            sp = float(speeds[side][i])
            claimed_speeds.append(sp)
            if (not opp_claimed) and opp in speeds:
                opp_sp = float(speeds[opp][i])
                claimed_minus_opp.append(sp - opp_sp)
                claimed_ratio.append(sp / max(opp_sp, 1e-4))

    feats: Dict[str, float] = {
        "right_frame_fraction": float(np.mean(right_mask)) if h else 0.0,
        "left_frame_fraction": float(np.mean(left_mask)) if h else 0.0,
        "dual_frame_fraction": float(sum(1 for x in labels if x == "dual") / max(1, h)),
        "flight_frame_fraction": float(sum(1 for x in labels if x == "flight_or_unknown") / max(1, h)),
        "transition_count": float(_support_transition_count(labels)),
        "support_entropy_bits": float(_support_entropy(labels)),
        "right_run_max": float(max(right_runs) if right_runs else 0),
        "left_run_max": float(max(left_runs) if left_runs else 0),
        "single_support_run_min": float(min(single_runs) if single_runs else 0),
        "single_support_run_max": float(max(single_runs) if single_runs else 0),
        "claimed_support_slip_mean_mps": _mean(claimed_speeds),
        "claimed_support_slip_p95_mps": _p95(claimed_speeds),
        "claimed_support_slip_max_mps": float(max(claimed_speeds) if claimed_speeds else 0.0),
        "single_support_claimed_minus_opposite_mean_mps": _mean(claimed_minus_opp),
        "single_support_claimed_minus_opposite_p95_mps": _p95(claimed_minus_opp),
        "single_support_claimed_speed_ratio_p95": _p95(claimed_ratio),
        "yaw_sum_rad": float(np.sum(yaw_rate) / FPS),
        "yaw_abs_sum_rad": float(np.sum(np.abs(yaw_rate)) / FPS),
        "heading_error_p95_rad": _safe_percentile(_heading_error_rad(root_vel, cond_dir), 95),
        "root_speed_mean": float(np.mean(np.linalg.norm(root_vel, axis=1))) if root_vel.size else 0.0,
        "root_lateral_mean": float(np.mean(root_vel[:, 1])) if root_vel.size else 0.0,
    }

    for side, mask in (("right", right_mask), ("left", left_mask)):
        if foot is None or side not in foot or not np.any(mask):
            rel = np.zeros((0, 3), dtype=np.float32)
        else:
            side_pos = np.asarray(foot[side], dtype=np.float32).reshape(-1, 3)[:h]
            rel = side_pos[mask] - root_pos[mask]
        if rel.size == 0:
            feats[f"{side}_rel_x_mean"] = 0.0
            feats[f"{side}_rel_y_mean"] = 0.0
            feats[f"{side}_rel_z_mean"] = 0.0
            feats[f"{side}_rel_norm_p95"] = 0.0
        else:
            feats[f"{side}_rel_x_mean"] = float(np.mean(rel[:, 0]))
            feats[f"{side}_rel_y_mean"] = float(np.mean(rel[:, 1]))
            feats[f"{side}_rel_z_mean"] = float(np.mean(rel[:, 2]))
            feats[f"{side}_rel_norm_p95"] = _p95(np.linalg.norm(rel, axis=1))

    balance = feats["right_frame_fraction"] - feats["left_frame_fraction"]
    feats["support_side_balance"] = float(balance)
    feats["support_yaw_product"] = float(balance * feats["yaw_sum_rad"])
    feats["support_lateral_product"] = float(balance * feats["root_lateral_mean"])
    return {k: float(feats.get(k, 0.0)) for k in SUPPORT_SIDE_FEATURE_KEYS}


def _feature_bands(feature_rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    bands: Dict[str, Any] = {}
    for key in SUPPORT_SIDE_FEATURE_KEYS:
        vals = np.asarray([float(r.get(key, 0.0)) for r in feature_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            bands[key] = {"n": 0, "min": 0.0, "max": 0.0, "p01": 0.0, "p99": 0.0}
            continue
        bands[key] = {
            "n": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p01": float(np.percentile(vals, 1)),
            "p99": float(np.percentile(vals, 99)),
        }
    return bands


def _evaluate_support_side_correctness(
    features: Mapping[str, float],
    bands: Mapping[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    for key in SUPPORT_SIDE_FEATURE_KEYS:
        band = bands.get(key)
        if not isinstance(band, Mapping):
            continue
        val = float(features.get(key, 0.0))
        lo = float(band.get("min", 0.0))
        hi = float(band.get("max", 0.0))
        tol = 1e-6 + 1e-5 * max(1.0, abs(lo), abs(hi))
        if val < lo - tol or val > hi + tol:
            failures.append(
                {
                    "feature": key,
                    "value": val,
                    "band_min": lo,
                    "band_max": hi,
                    "band_p01": float(band.get("p01", lo)),
                    "band_p99": float(band.get("p99", hi)),
                }
            )
    return (not failures), failures


def _sequence_foot_positions(
    seq: Mapping[str, np.ndarray],
    skeleton: SkeletonMeta,
) -> Optional[Dict[str, np.ndarray]]:
    return _foot_positions(
        np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_DIM),
        np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3),
        skeleton,
    )


def _make_sequence(clip: ClipData, start: int, horizon: int) -> Dict[str, np.ndarray]:
    s = int(start)
    e = s + int(horizon)
    return {
        "state281": clip.state281[s:e].astype(np.float32, copy=False),
        "rot6d": clip.rot6d[s:e].astype(np.float32, copy=False),
        "root_pos": clip.root_pos[s:e].astype(np.float32, copy=False),
        "root_vel": clip.root_vel[s:e].astype(np.float32, copy=False),
        "bone_angvel": clip.bone_angvel[s:e].astype(np.float32, copy=False),
        "cond_dir": clip.cond_dir[s:e].astype(np.float32, copy=False),
        "contact": clip.contact[s:e].astype(np.float32, copy=False),
        "yaw_rate": clip.yaw_rate[s:e].astype(np.float32, copy=False),
    }


def _context_window(clip: ClipData, start: int, context_len: int, *, wrap: bool) -> np.ndarray:
    c = int(context_len)
    s = int(start)
    state = np.asarray(clip.state281, dtype=np.float32)
    if wrap:
        idx = (np.arange(s - c, s, dtype=np.int64) % state.shape[0]).astype(np.int64)
        return state[idx].copy()
    lo = max(0, s - c)
    ctx = state[lo:s].copy()
    if ctx.shape[0] >= c:
        return ctx[-c:].copy()
    pad_src = state[max(0, min(s, state.shape[0] - 1))].reshape(1, -1)
    pad = np.repeat(pad_src, c - ctx.shape[0], axis=0)
    return np.concatenate([pad, ctx], axis=0).astype(np.float32, copy=False)


def _label_one_hot(label: str) -> np.ndarray:
    return np.asarray([1.0 if label == x else 0.0 for x in LABEL_ORDER], dtype=np.float32)


def _available_context_feature(item: WindowItem) -> np.ndarray:
    seq = item.seq
    ctx = np.asarray(item.ctx, dtype=np.float32).reshape(-1)
    command = np.concatenate(
        [
            np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1),
            np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )
    # Soft endpoint cue excludes future endpoint contact channels.
    endpoint_state = np.asarray(seq["state281"][-1, : CONTACT_SLICE.start], dtype=np.float32).reshape(-1)
    start_label = item.support_contract["schedule_label_sequence"][0]
    seam_label = item.support_contract["normalized_label_sequence"][0]
    return np.concatenate(
        [ctx, command, endpoint_state, _label_one_hot(start_label), _label_one_hot(seam_label)],
        axis=0,
    ).astype(np.float32)


def _pose_only_feature(item: WindowItem) -> np.ndarray:
    seq = item.seq
    return np.concatenate([seq["rot6d"][0], seq["rot6d"][-1]], axis=0).astype(np.float32)


def _velocity_phase_feature(item: WindowItem, include_bone_angvel: bool) -> np.ndarray:
    seq = item.seq
    labels = item.support_contract["normalized_label_sequence"]
    support_stats = np.asarray(
        [
            sum(1 for x in labels if _label_has_side(x, "right")) / max(1, len(labels)),
            sum(1 for x in labels if _label_has_side(x, "left")) / max(1, len(labels)),
            sum(1 for x in labels if x == "dual") / max(1, len(labels)),
            sum(1 for x in labels if x == "flight_or_unknown") / max(1, len(labels)),
            _support_transition_count(labels),
        ],
        dtype=np.float32,
    )
    parts = [
        _available_context_feature(item),
        np.asarray(seq["root_vel"][0], dtype=np.float32).reshape(-1),
        np.asarray(seq["root_vel"][-1], dtype=np.float32).reshape(-1),
        np.asarray([seq["yaw_rate"][0], seq["yaw_rate"][-1]], dtype=np.float32),
        _label_one_hot(labels[0]),
        _label_one_hot(labels[-1]),
        support_stats,
    ]
    if include_bone_angvel:
        ang = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
        parts.extend([ang[0], ang[-1], np.mean(ang, axis=0).astype(np.float32)])
    return np.concatenate(parts, axis=0).astype(np.float32)


def _calibrate_support_side_bands(
    clips: Mapping[str, ClipData],
    skeleton: SkeletonMeta,
    *,
    horizon: int,
    min_run_frames: int,
    only_clips: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    foot_cache: Dict[str, Dict[str, np.ndarray]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for name in only_clips:
        clip = clips[name]
        foot = _foot_positions(clip.rot6d, clip.root_pos, skeleton) or {}
        foot_cache[name] = foot
        rows: List[Dict[str, float]] = []
        max_start = int(clip.rot6d.shape[0]) - int(horizon)
        for start in range(max_start + 1):
            seq = _make_sequence(clip, start, horizon)
            contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
            foot_win = {side: arr[start : start + horizon] for side, arr in foot.items()}
            rows.append(_support_side_features(seq, contract["normalized_label_sequence"], foot_win))
        out[name] = {
            "horizon": int(horizon),
            "n_windows": int(len(rows)),
            "feature_bands": _feature_bands(rows),
            "band_rule": "inclusive min/max over real continuous target windows; no oracle schedule equality",
        }
    return out, foot_cache


def _feature_vector_for_tier(
    item: WindowItem,
    tier: str,
    *,
    include_bone_angvel_witness: bool,
) -> np.ndarray:
    if tier == "pose_only":
        return _pose_only_feature(item)
    if tier == "available_context":
        return _available_context_feature(item)
    if tier == "velocity_phase_enriched":
        return _velocity_phase_feature(item, include_bone_angvel=include_bone_angvel_witness)
    raise KeyError(tier)


def _build_window_items(
    clips: Mapping[str, ClipData],
    skeleton: SkeletonMeta,
    support_bands: Mapping[str, Mapping[str, Any]],
    foot_cache: Mapping[str, Mapping[str, np.ndarray]],
    *,
    horizon: int,
    context_len: int,
    min_run_frames: int,
    only_clips: Sequence[str],
    stride: int,
    include_bone_angvel_witness: bool,
) -> List[WindowItem]:
    items: List[WindowItem] = []
    for name in only_clips:
        clip = clips[name]
        max_start = int(clip.rot6d.shape[0]) - int(horizon)
        if max_start < 0:
            continue
        foot = foot_cache.get(name, {})
        for start in range(0, max_start + 1, max(1, int(stride))):
            seq = _make_sequence(clip, start, horizon)
            ctx = _context_window(clip, start, context_len, wrap=(name == WALK_F))
            contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
            foot_win = {side: arr[start : start + horizon] for side, arr in foot.items()}
            features = _support_side_features(seq, contract["normalized_label_sequence"], foot_win)
            band = support_bands[name]["feature_bands"]
            ok, failures = _evaluate_support_side_correctness(features, band)
            item = WindowItem(
                clip=name,
                start=int(start),
                end=int(start + horizon - 1),
                seq=seq,
                ctx=ctx,
                support_contract=contract,
                support_side_correctness=bool(ok),
                support_side_failures=failures,
                feature_by_tier={},
            )
            for tier in ("pose_only", "available_context", "velocity_phase_enriched"):
                item.feature_by_tier[tier] = _feature_vector_for_tier(
                    item,
                    tier,
                    include_bone_angvel_witness=include_bone_angvel_witness,
                )
            items.append(item)
    return items


def _cluster_items(
    items: Sequence[WindowItem],
    *,
    tier: str,
    radius: float,
) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        feat = np.asarray(item.feature_by_tier[tier], dtype=np.float64).reshape(-1)
        feat = np.where(np.isfinite(feat), feat, 0.0)
        best_j: Optional[int] = None
        best_d = float("inf")
        for j, bucket in enumerate(buckets):
            center = np.asarray(bucket["center"], dtype=np.float64)
            d = float(np.linalg.norm(feat - center) / math.sqrt(max(1, feat.shape[0])))
            if d <= float(radius) and d < best_d:
                best_j = j
                best_d = d
        if best_j is None:
            buckets.append({"center": feat.copy(), "items": [idx], "distances": [0.0]})
        else:
            buckets[best_j]["items"].append(idx)
            buckets[best_j]["distances"].append(best_d)
    return buckets


def _bucket_size_distribution(sizes: Sequence[int]) -> Dict[str, Any]:
    arr = np.asarray(list(sizes), dtype=np.float64)
    if arr.size == 0:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0.0, "singleton": 0, "ge2": 0, "ge3": 0}
    return {
        "min": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "singleton": int(np.sum(arr == 1)),
        "ge2": int(np.sum(arr >= 2)),
        "ge3": int(np.sum(arr >= 3)),
    }


def _item_example(item: WindowItem, distance: float) -> Dict[str, Any]:
    c = item.support_contract
    return {
        "clip": item.clip,
        "start": int(item.start),
        "end": int(item.end),
        "distance_to_center": float(distance),
        "raw_signature": c["raw_signature"],
        "normalized_signature": c["normalized_signature"],
        "ambiguous": bool(c["normalization"]["ambiguous"]),
        "support_side_correctness": bool(item.support_side_correctness),
    }


def _summarize_multimodality_tier(
    items: Sequence[WindowItem],
    *,
    tier: str,
    radius: float,
) -> Dict[str, Any]:
    buckets = _cluster_items(items, tier=tier, radius=radius)
    norm_multi: List[Dict[str, Any]] = []
    raw_only: List[Dict[str, Any]] = []
    raw_multi_count = 0
    max_raw = 0
    max_norm = 0
    changed_items = 0
    ambiguous_item_count = 0
    for bidx, bucket in enumerate(buckets):
        idxs = list(bucket["items"])
        distances = list(bucket["distances"])
        raw_sigs = {items[i].support_contract["raw_signature"] for i in idxs}
        norm_sigs = {items[i].support_contract["normalized_signature"] for i in idxs}
        max_raw = max(max_raw, len(raw_sigs))
        max_norm = max(max_norm, len(norm_sigs))
        if len(raw_sigs) > 1:
            raw_multi_count += 1
        for i in idxs:
            c = items[i].support_contract
            if c["raw_signature"] != c["normalized_signature"]:
                changed_items += 1
            if c["normalization"]["ambiguous"]:
                ambiguous_item_count += 1
        examples = [_item_example(items[i], distances[j]) for j, i in enumerate(idxs[:8])]
        rec = {
            "bucket": int(bidx),
            "n": int(len(idxs)),
            "raw_signature_count": int(len(raw_sigs)),
            "normalized_signature_count": int(len(norm_sigs)),
            "raw_signatures": sorted(raw_sigs),
            "normalized_signatures": sorted(norm_sigs),
            "examples": examples,
        }
        if len(norm_sigs) > 1:
            norm_multi.append(rec)
        if len(raw_sigs) > 1 and len(norm_sigs) == 1:
            raw_only.append(rec)

    sizes = [len(b["items"]) for b in buckets]
    norm_multi_count = len(norm_multi)
    return {
        "tier": tier,
        "radius": float(radius),
        "bucket_count": int(len(buckets)),
        "bucket_size_distribution": _bucket_size_distribution(sizes),
        "raw_multi_signature_bucket_count": int(raw_multi_count),
        "normalized_multi_signature_bucket_count": int(norm_multi_count),
        "normalized_multi_signature_bucket_fraction": _rate(norm_multi_count, len(buckets)),
        "max_raw_signatures_per_bucket": int(max_raw),
        "max_signatures_per_bucket": int(max_norm),
        "raw_vs_normalized_signature_deltas": {
            "items_with_changed_signature": int(changed_items),
            "ambiguous_item_count": int(ambiguous_item_count),
            "raw_only_multi_bucket_count": int(len(raw_only)),
            "normalized_multi_minus_raw_multi": int(norm_multi_count - raw_multi_count),
            "raw_only_multi_examples": raw_only[:8],
        },
        "top_ambiguous_buckets": sorted(
            norm_multi,
            key=lambda r: (-int(r["normalized_signature_count"]), -int(r["n"]), int(r["bucket"])),
        )[:10],
    }


def _architecture_decision(available: Mapping[str, Any]) -> Dict[str, Any]:
    frac = float(available.get("normalized_multi_signature_bucket_fraction", 0.0))
    max_sig = int(available.get("max_signatures_per_bucket", 0))
    if frac <= 0.05 and max_sig <= 2:
        decision = "deterministic_masked_transformer"
        reason = "available_context normalized multi-signature fraction <= 0.05 and max signatures <= 2"
    elif frac >= 0.15 or max_sig >= 3:
        decision = "retain_sampling_or_diffusion_branch"
        reason = "available_context normalized multi-signature fraction >= 0.15 or max signatures >= 3"
    else:
        decision = "inconclusive"
        reason = "available_context falls between deterministic and sampling thresholds"
    return {
        "decision_tier": "available_context",
        "decision": decision,
        "reason": reason,
        "observed_available_context_fraction": frac,
        "observed_available_context_max_signatures": max_sig,
    }


def _load_base_state_contract(npz_root: Path, clips: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for clip in clips:
        path = Path(npz_root) / f"{clip}.npz"
        if not path.is_file():
            out[clip] = {"loaded": False, "reason": f"missing {path}"}
            continue
        with np.load(path, allow_pickle=True) as z:
            key = "X_flat" if "X_flat" in z.files else ("x_in_features" if "x_in_features" in z.files else None)
            if key is None:
                out[clip] = {"loaded": False, "reason": "no X_flat or x_in_features key"}
                continue
            arr = np.asarray(z[key], dtype=np.float32)
        out[clip] = {
            "loaded": True,
            "source_key": key,
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "device": "cpu",
            "finite": bool(np.isfinite(arr).all()),
        }
    return out


def _sequence_contract_shapes(
    *,
    ctx: np.ndarray,
    middle: np.ndarray,
    contact: np.ndarray,
    bone_angvel: np.ndarray,
) -> Dict[str, Any]:
    return {
        "ctx": {
            "shape_contract": "[C,281]",
            "actual_shape": [int(x) for x in ctx.shape],
            "dtype": str(np.asarray(ctx).dtype),
            "device": "cpu",
            "role": "available-context feature only; not fed to any model in this probe",
        },
        "candidate_oracle_middle": {
            "shape_contract": "[H,281]",
            "actual_shape": [int(x) for x in middle.shape],
            "dtype": str(np.asarray(middle).dtype),
            "device": "cpu",
        },
        "support_schedule_contact": {
            "shape_contract": "[H,2]",
            "actual_shape": [int(x) for x in contact.shape],
            "dtype": str(np.asarray(contact).dtype),
            "device": "cpu",
        },
        "bone_angvel_witness": {
            "shape_contract": "[H,138]",
            "actual_shape": [int(x) for x in bone_angvel.shape],
            "dtype": str(np.asarray(bone_angvel).dtype),
            "device": "cpu",
        },
    }


def _augment_acceptance_row_with_support_side(
    row: Dict[str, Any],
    *,
    support_contract: Mapping[str, Any],
    support_side_correctness: bool,
    support_side_failures: Sequence[Mapping[str, Any]],
    seam_support_start: str,
    seam_support_end: str,
    horizon_support_start: str,
    horizon_support_end: str,
) -> Dict[str, Any]:
    families = [
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
    ]
    row["support_side_correctness"] = bool(support_side_correctness)
    failed = [k for k in families if not bool(row.get(k, False))]
    row["failed_family"] = ",".join(failed)
    row["pass"] = bool(not failed)
    row["support_contract"] = {
        "seam_support_start": seam_support_start,
        "seam_support_end": seam_support_end,
        "horizon_support_start": horizon_support_start,
        "horizon_support_end": horizon_support_end,
        "schedule_label_sequence": list(support_contract["schedule_label_sequence"]),
        "normalized_label_sequence": list(support_contract["normalized_label_sequence"]),
        "raw_signature": support_contract["raw_signature"],
        "normalized_signature": support_contract["normalized_signature"],
        "normalization": support_contract["normalization"],
    }
    row["support_side_failures"] = [dict(x) for x in support_side_failures[:12]]
    return row


def _endpoint_schedule_ok(
    support_contract: Mapping[str, Any],
    *,
    seam_support_start: str,
    target_horizon_support_end: str,
    horizon_bridgeable: bool,
) -> bool:
    labels = list(support_contract["normalized_label_sequence"])
    if not labels:
        return False
    return bool(horizon_bridgeable and labels[0] == seam_support_start and labels[-1] == target_horizon_support_end)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "target",
        "start_phase",
        "expected_label",
        "pass",
        "failed_family",
        "support_honesty",
        "support_side_correctness",
        "endpoint_bridgeability",
        "seam_support_start",
        "seam_support_end",
        "horizon_support_start",
        "horizon_support_end",
        "raw_signature",
        "normalized_signature",
        "ambiguous",
        "support_side_failure_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            c = row.get("support_contract", {})
            norm = c.get("normalization", {}) if isinstance(c, Mapping) else {}
            writer.writerow(
                {
                    "case": row.get("case"),
                    "target": row.get("target"),
                    "start_phase": row.get("start_phase"),
                    "expected_label": row.get("expected_label"),
                    "pass": row.get("pass"),
                    "failed_family": row.get("failed_family"),
                    "support_honesty": row.get("support_honesty"),
                    "support_side_correctness": row.get("support_side_correctness"),
                    "endpoint_bridgeability": row.get("endpoint_bridgeability"),
                    "seam_support_start": c.get("seam_support_start") if isinstance(c, Mapping) else None,
                    "seam_support_end": c.get("seam_support_end") if isinstance(c, Mapping) else None,
                    "horizon_support_start": c.get("horizon_support_start") if isinstance(c, Mapping) else None,
                    "horizon_support_end": c.get("horizon_support_end") if isinstance(c, Mapping) else None,
                    "raw_signature": c.get("raw_signature") if isinstance(c, Mapping) else None,
                    "normalized_signature": c.get("normalized_signature") if isinstance(c, Mapping) else None,
                    "ambiguous": norm.get("ambiguous") if isinstance(norm, Mapping) else None,
                    "support_side_failure_count": len(row.get("support_side_failures", []) or []),
                }
            )


def _summarize_family_attribution(
    rows: Sequence[Mapping[str, Any]],
    real_items: Sequence[WindowItem],
    alt_items: Sequence[WindowItem],
) -> Dict[str, Any]:
    wrong = [r for r in rows if str(r.get("case", "")).endswith("wrong_side_support")]
    shuffled = [r for r in rows if str(r.get("case", "")).endswith("shuffled_support")]
    real_fp = [it for it in real_items if not it.support_side_correctness]
    alt_fp = [it for it in alt_items if not it.support_side_correctness]
    return {
        "wrong_side_n": int(len(wrong)),
        "wrong_side_pass_rate": _rate(sum(1 for r in wrong if bool(r.get("pass"))), len(wrong)),
        "wrong_side_rejected_by_support_side_correctness_rate": _rate(
            sum(1 for r in wrong if not bool(r.get("support_side_correctness"))),
            len(wrong),
        ),
        "wrong_side_rejected_by_endpoint_bridgeability_rate": _rate(
            sum(1 for r in wrong if not bool(r.get("endpoint_bridgeability"))),
            len(wrong),
        ),
        "wrong_side_support_side_independent_reject_rate": _rate(
            sum(1 for r in wrong if not bool(r.get("support_side_correctness"))),
            len(wrong),
        ),
        "shuffled_n": int(len(shuffled)),
        "shuffled_rejected_by_support_side_correctness_rate": _rate(
            sum(1 for r in shuffled if not bool(r.get("support_side_correctness"))),
            len(shuffled),
        ),
        "shuffled_rejected_by_support_honesty_rate": _rate(
            sum(1 for r in shuffled if not bool(r.get("support_honesty"))),
            len(shuffled),
        ),
        "real_continuous_support_side_false_positive_rate": _rate(len(real_fp), len(real_items)),
        "real_continuous_support_side_false_positive_cases": [
            {
                "clip": it.clip,
                "start": int(it.start),
                "end": int(it.end),
                "normalized_signature": it.support_contract["normalized_signature"],
                "failures": it.support_side_failures[:8],
            }
            for it in real_fp[:20]
        ],
        "alternative_real_mode_n": int(len(alt_items)),
        "alternative_real_mode_false_positive_rate": _rate(len(alt_fp), len(alt_items)),
        "alternative_real_mode_false_positive_cases": [
            {
                "clip": it.clip,
                "start": int(it.start),
                "end": int(it.end),
                "normalized_signature": it.support_contract["normalized_signature"],
                "failures": it.support_side_failures[:8],
            }
            for it in alt_fp[:20]
        ],
    }


def _alternative_items_from_multisignature_buckets(
    items: Sequence[WindowItem],
    *,
    tier: str,
    radius: float,
) -> List[WindowItem]:
    out: Dict[Tuple[str, int], WindowItem] = {}
    for bucket in _cluster_items(items, tier=tier, radius=radius):
        idxs = list(bucket["items"])
        norm_sigs = {items[i].support_contract["normalized_signature"] for i in idxs}
        if len(norm_sigs) <= 1:
            continue
        for i in idxs:
            item = items[i]
            out[(item.clip, item.start)] = item
    return list(out.values())


def _walk_l_to_r_report(
    *,
    clips: Mapping[str, ClipData],
    matched_pairs: Mapping[str, Any],
    horizon: int,
    pose_topk: int,
    ground_contact_thr: float,
    ground_pose_thr: float,
    min_run_frames: int,
) -> Dict[str, Any]:
    target_name = "Walk_L_To_R"
    walk = clips[WALK_F]
    target = clips[target_name]
    align = full_state_align(
        walk.state281,
        target.state281[0],
        topk=int(pose_topk),
        contact_thr=float(ground_contact_thr),
        pose_thr=float(ground_pose_thr),
    )
    pair = matched_pairs.get(target_name)
    phi = int(pair["phi"]) if pair else int(align.full_state_phi)
    onset = int(pair["onset"]) if pair else 0
    h_ok = onset + int(horizon) <= target.contact.shape[0]
    seam_start = _support_label(walk.contact[phi])
    seam_end = _support_label(target.contact[onset])
    horizon: Optional[Dict[str, Any]]
    normalized_signature: Optional[str]
    raw_signature: Optional[str]
    if h_ok:
        contract = _support_contract(target.contact[onset : onset + int(horizon)], min_run_frames=min_run_frames)
        horizon = {
            "start": contract["schedule_label_sequence"][0],
            "end": contract["schedule_label_sequence"][-1],
            "normalized_start": contract["normalized_label_sequence"][0],
            "normalized_end": contract["normalized_label_sequence"][-1],
        }
        raw_signature = contract["raw_signature"]
        normalized_signature = contract["normalized_signature"]
        v1_out = bool(
            "flight_or_unknown" in {
                seam_start,
                seam_end,
                horizon["start"],
                horizon["end"],
                horizon["normalized_start"],
                horizon["normalized_end"],
            }
            or contract["normalization"]["ambiguous"]
        )
    else:
        horizon = None
        raw_signature = None
        normalized_signature = None
        v1_out = bool("flight_or_unknown" in {seam_start, seam_end})
    reasons = []
    if align.full_state_contact_d > float(ground_contact_thr):
        reasons.append(f"contact_d {align.full_state_contact_d:.6f} > ground_contact_thr {float(ground_contact_thr):.6f}")
    if align.full_state_pose_d > float(ground_pose_thr):
        reasons.append(f"pose_d {align.full_state_pose_d:.6f} > ground_pose_thr {float(ground_pose_thr):.6f}")
    if pair is None:
        reasons.append("no matched_pair in two-frame artifact")
    return {
        "target": target_name,
        "matched_pair_available": bool(pair is not None),
        "pose_d": float(align.full_state_pose_d),
        "contact_d": float(align.full_state_contact_d),
        "phi": int(phi),
        "onset": int(onset),
        "seam_support": {"walk_phi": seam_start, "target_onset": seam_end},
        "horizon_support": horizon,
        "raw_signature": raw_signature,
        "normalized_signature": normalized_signature,
        "ungroundable_reason": "; ".join(reasons) if reasons else "",
        "v1_out_of_scope_flight_or_unknown_phase": bool(v1_out),
        "averaged_into_matched_target_rates": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only support contract tightening and multimodality audit.")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--baseline-quantile", type=float, default=DEFAULT_BASELINE_QUANTILE)
    p.add_argument("--bridge-budget-quantile", type=float, default=DEFAULT_BRIDGE_BUDGET_QUANTILE)
    p.add_argument("--pose-bucket-radius", type=float, default=DEFAULT_POSE_BUCKET_RADIUS)
    p.add_argument("--available-bucket-radius", type=float, default=DEFAULT_AVAILABLE_BUCKET_RADIUS)
    p.add_argument("--velocity-bucket-radius", type=float, default=DEFAULT_VELOCITY_BUCKET_RADIUS)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--no-bone-angvel-witness", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_support_contract_tightening_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = int(args.horizon)
    context_len = int(args.context_len)
    min_run = int(args.min_run_frames)
    stride = max(1, int(args.stride))
    include_bone = not bool(args.no_bone_angvel_witness)

    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    baseline_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    bridge_budgets = _bridge_budgets(clips, quantile=float(args.bridge_budget_quantile))
    support_bands, foot_cache = _calibrate_support_side_bands(
        clips,
        skeleton,
        horizon=h,
        min_run_frames=min_run,
        only_clips=TURN_CLIPS,
    )
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}

    walk = clips[WALK_F]
    rows: List[Dict[str, Any]] = []
    endpoint_contract_rows: List[Dict[str, Any]] = []
    matched_targets: List[str] = []

    for target_name in TURN_CLIPS:
        if target_name == "Walk_L_To_R":
            continue
        target = clips[target_name]
        align = full_state_align(
            walk.state281,
            target.state281[0],
            topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        pair = matched_pairs.get(target_name)
        if pair is None:
            endpoint_contract_rows.append(
                {
                    "target": target_name,
                    "matched_pair_available": False,
                    "pose_d": float(align.full_state_pose_d),
                    "contact_d": float(align.full_state_contact_d),
                    "decision": "separate_unmatched",
                }
            )
            continue
        phi = int(pair["phi"])
        onset = int(pair["onset"])
        if onset + h > target.rot6d.shape[0]:
            endpoint_contract_rows.append(
                {
                    "target": target_name,
                    "matched_pair_available": True,
                    "phi": phi,
                    "onset": onset,
                    "decision": "skip_too_short_for_horizon",
                }
            )
            continue
        matched_targets.append(target_name)

        bridge = _endpoint_bridge(
            walk,
            target,
            phi,
            onset,
            bridge_budgets[target_name],
            horizon=h,
            groundable=bool(align.groundable),
        )
        oracle_seq = _root_shifted_target_sequence(walk, target, phi, onset, h)
        oracle_seq["state281"] = target.state281[onset : onset + h].astype(np.float32, copy=False)
        walk_cue = _walk_cue_sequence(walk, phi, h)
        seam_start = _support_label(walk.contact[phi])
        seam_end = _support_label(target.contact[onset])
        oracle_contract = _support_contract(oracle_seq["contact"], min_run_frames=min_run)
        target_horizon_end = oracle_contract["normalized_label_sequence"][-1]
        endpoint_row = {
            "target": target_name,
            "matched_pair_available": True,
            "phi": phi,
            "onset": onset,
            "pose_d": float(pair.get("current_pose_l2", align.full_state_pose_d)),
            "contact_d": float(align.full_state_contact_d),
            "seam_support_start": seam_start,
            "seam_support_end": seam_end,
            "horizon_support_start": oracle_contract["schedule_label_sequence"][0],
            "horizon_support_end": oracle_contract["schedule_label_sequence"][-1],
            "horizon_support_start_normalized": oracle_contract["normalized_label_sequence"][0],
            "horizon_support_end_normalized": oracle_contract["normalized_label_sequence"][-1],
            "raw_signature": oracle_contract["raw_signature"],
            "normalized_signature": oracle_contract["normalized_signature"],
            **bridge,
        }
        endpoint_contract_rows.append(endpoint_row)

        condition_specs = [
            (
                "endpoint_command_oracle_support",
                oracle_seq,
                "pass",
            ),
            (
                "endpoint_command_shuffled_support",
                _replace_seq(oracle_seq, contact=_shuffled_contact(oracle_seq["contact"])),
                "fail",
            ),
            (
                "endpoint_command_wrong_side_support",
                _replace_seq(oracle_seq, contact=_wrong_side_contact(oracle_seq["contact"])),
                "fail",
            ),
            (
                "endpoint_command_walk_support",
                _replace_seq(oracle_seq, contact=walk_cue["contact"]),
                "diagnostic_fail_or_incomplete",
            ),
        ]
        for condition, seq, expected in condition_specs:
            if "state281" not in seq:
                seq["state281"] = oracle_seq["state281"]
            contract = _support_contract(seq["contact"], min_run_frames=min_run)
            schedule_ok = _endpoint_schedule_ok(
                contract,
                seam_support_start=seam_start,
                target_horizon_support_end=target_horizon_end,
                horizon_bridgeable=bool(bridge.get("horizon_bridgeable")),
            )
            foot = _sequence_foot_positions(seq, skeleton)
            side_features = _support_side_features(seq, contract["normalized_label_sequence"], foot)
            side_ok, side_failures = _evaluate_support_side_correctness(
                side_features,
                support_bands[target_name]["feature_bands"],
            )
            row = _evaluate_sequence(
                seq,
                target=target_name,
                target_bands=baseline_bands[target_name],
                skeleton=skeleton,
                case=f"support_contract:{condition}",
                expected_label=expected,
                start_phase=f"phi={phi};onset={onset};H={h}",
                endpoint_bridgeability=bool(schedule_ok),
                endpoint_details={
                    "phi": phi,
                    "onset": onset,
                    "horizon_bridgeable": bool(bridge.get("horizon_bridgeable")),
                    "endpoint_bridgeability_definition": "normalized schedule first/last support + horizon bridgeable only",
                    "support_side_correctness_definition": "candidate schedule + realized trajectory within real continuous feature bands; no oracle schedule equality",
                },
            )
            rows.append(
                _augment_acceptance_row_with_support_side(
                    row,
                    support_contract=contract,
                    support_side_correctness=side_ok,
                    support_side_failures=side_failures,
                    seam_support_start=seam_start,
                    seam_support_end=seam_end,
                    horizon_support_start=contract["normalized_label_sequence"][0],
                    horizon_support_end=contract["normalized_label_sequence"][-1],
                )
            )

    all_items = _build_window_items(
        clips,
        skeleton,
        support_bands,
        foot_cache,
        horizon=h,
        context_len=context_len,
        min_run_frames=min_run,
        only_clips=TURN_CLIPS,
        stride=stride,
        include_bone_angvel_witness=include_bone,
    )
    matched_real_items = [it for it in all_items if it.clip in matched_targets]
    multimodality = {
        "pose_only": _summarize_multimodality_tier(
            all_items,
            tier="pose_only",
            radius=float(args.pose_bucket_radius),
        ),
        "available_context": _summarize_multimodality_tier(
            all_items,
            tier="available_context",
            radius=float(args.available_bucket_radius),
        ),
        "velocity_phase_enriched": _summarize_multimodality_tier(
            all_items,
            tier="velocity_phase_enriched",
            radius=float(args.velocity_bucket_radius),
        ),
    }
    alternative_items = _alternative_items_from_multisignature_buckets(
        all_items,
        tier="available_context",
        radius=float(args.available_bucket_radius),
    )
    family_attribution = _summarize_family_attribution(rows, matched_real_items, alternative_items)
    decision = _architecture_decision(multimodality["available_context"])
    walk_l_to_r = _walk_l_to_r_report(
        clips=clips,
        matched_pairs=matched_pairs,
        horizon=h,
        pose_topk=int(args.pose_topk),
        ground_contact_thr=float(args.ground_contact_thr),
        ground_pose_thr=float(args.ground_pose_thr),
        min_run_frames=min_run,
    )
    base_state_contract = _load_base_state_contract(Path(args.npz_root), [WALK_F, *TURN_CLIPS])

    example_ctx = _context_window(walk, int(endpoint_contract_rows[0].get("phi", 0)) if endpoint_contract_rows else 0, context_len, wrap=True)
    example_middle = next(iter(clips.values())).state281[:h].astype(np.float32, copy=False)
    example_contact = next(iter(clips.values())).contact[:h].astype(np.float32, copy=False)
    example_angvel = next(iter(clips.values())).bone_angvel[:h].astype(np.float32, copy=False)
    io_contract = _sequence_contract_shapes(
        ctx=example_ctx,
        middle=example_middle,
        contact=example_contact,
        bone_angvel=example_angvel,
    )
    io_contract["base_state_if_loaded"] = {
        "shape_contract": "[T,419]",
        "per_clip": base_state_contract,
    }

    payload = {
        "task": "support_contract_tightening_probe",
        "scope": "read-only diagnostics; no training; no model forward; no production trainer/runtime/gate edit",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "two_frame_summary": str(args.two_frame_summary),
            "horizon": h,
            "context_len": context_len,
            "stride": stride,
            "min_run_frames": min_run,
            "baseline_quantile": float(args.baseline_quantile),
            "bridge_budget_quantile": float(args.bridge_budget_quantile),
            "pose_bucket_radius": float(args.pose_bucket_radius),
            "available_bucket_radius": float(args.available_bucket_radius),
            "velocity_bucket_radius": float(args.velocity_bucket_radius),
            "include_bone_angvel_witness": bool(include_bone),
            "device_policy": "cpu only",
        },
        "input_output_contract": io_contract,
        "metric_boundary_confirmations": {
            "support_honesty_current_scope": "contact step + FK foot slip; no left/right semantic side check",
            "endpoint_bridgeability_current_scope": "first/last support bridge + horizon bridgeable; no middle schedule semantics",
            "previous_support_summary_support_endpoint": "horizon-end support, not seam support",
            "support_metric_leak_false_interpretation": "`support_metric_leak=False` does not prove side metric works",
        },
        "architecture_decision_rules": {
            "decision_tier": "available_context",
            "deterministic_masked_transformer": {
                "normalized_multi_signature_bucket_fraction_lte": 0.05,
                "max_signatures_lte": 2,
            },
            "retain_sampling_or_diffusion_branch": {
                "normalized_multi_signature_bucket_fraction_gte": 0.15,
                "max_signatures_gte": 3,
            },
            "middle": "inconclusive; do not force an architecture",
            "velocity_phase_enriched_role": "explanatory upper-bound only, not inference condition",
        },
        "support_side_correctness_contract": {
            "family": "support_side_correctness",
            "definition": (
                "feasibility-based: candidate support schedule plus realized trajectory must fall within "
                "real continuous target feature bands; does not compare candidate schedule to oracle schedule"
            ),
            "features": list(SUPPORT_SIDE_FEATURE_KEYS),
            "band_rule": "inclusive min/max over real continuous target windows",
        },
        "support_contract_three_split": endpoint_contract_rows,
        "condition_rows": rows,
        "support_side_bands": support_bands,
        "family_attribution": family_attribution,
        "multimodality": multimodality,
        "architecture_decision": decision,
        "walk_l_to_r": walk_l_to_r,
        "window_counts": {
            "all_multimodality_windows": int(len(all_items)),
            "matched_real_continuous_windows": int(len(matched_real_items)),
            "matched_targets": list(matched_targets),
            "per_clip": dict(Counter(it.clip for it in all_items)),
        },
        "artifacts": {
            "summary_json": str(out_dir / "support_contract_tightening_summary.json"),
            "summary_md": str(out_dir / "support_contract_tightening_summary.md"),
            "rows_csv": str(out_dir / "support_contract_tightening_rows.csv"),
        },
    }
    _dump_json(out_dir / "support_contract_tightening_summary.json", payload)
    _write_rows_csv(out_dir / "support_contract_tightening_rows.csv", rows)

    lines: List[str] = []
    lines.append("# Support Contract Tightening Probe")
    lines.append("")
    lines.append("Read-only diagnostics. No training, no model forward, no checkpoint mutation, no production trainer/runtime/gate edit.")
    lines.append("")
    lines.append("## Support Contract Three Split")
    lines.append("")
    lines.append("| target | seam start | seam end | horizon start | horizon end | raw signature | normalized signature |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in endpoint_contract_rows:
        lines.append(
            f"| {row.get('target')} | {row.get('seam_support_start')} | {row.get('seam_support_end')} | "
            f"{row.get('horizon_support_start_normalized', row.get('horizon_support_start'))} | "
            f"{row.get('horizon_support_end_normalized', row.get('horizon_support_end'))} | "
            f"{row.get('raw_signature', '-')} | {row.get('normalized_signature', '-')} |"
        )
    lines.append("")
    lines.append("## Family Attribution")
    lines.append("")
    lines.append(f"- wrong_side pass rate: `{_fmt(family_attribution['wrong_side_pass_rate'])}`")
    lines.append(
        "- wrong_side rejected by support_side_correctness: "
        f"`{_fmt(family_attribution['wrong_side_rejected_by_support_side_correctness_rate'])}`"
    )
    lines.append(
        "- wrong_side rejected by endpoint_bridgeability: "
        f"`{_fmt(family_attribution['wrong_side_rejected_by_endpoint_bridgeability_rate'])}`"
    )
    lines.append(
        "- shuffled rejected by support_side_correctness: "
        f"`{_fmt(family_attribution['shuffled_rejected_by_support_side_correctness_rate'])}`"
    )
    lines.append(
        "- shuffled rejected by support_honesty: "
        f"`{_fmt(family_attribution['shuffled_rejected_by_support_honesty_rate'])}`"
    )
    lines.append(
        "- real continuous support_side false positive rate: "
        f"`{_fmt(family_attribution['real_continuous_support_side_false_positive_rate'])}`"
    )
    lines.append(
        "- alternative real mode false positive rate: "
        f"`{_fmt(family_attribution['alternative_real_mode_false_positive_rate'])}`"
    )
    lines.append("")
    lines.append("## Multimodality")
    lines.append("")
    lines.append("| tier | buckets | size max | normalized multi buckets | fraction | max signatures | raw-only multi buckets |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tier in ("pose_only", "available_context", "velocity_phase_enriched"):
        rec = multimodality[tier]
        dist = rec["bucket_size_distribution"]
        delta = rec["raw_vs_normalized_signature_deltas"]
        lines.append(
            f"| {tier} | {rec['bucket_count']} | {dist['max']} | "
            f"{rec['normalized_multi_signature_bucket_count']} | "
            f"{_fmt(rec['normalized_multi_signature_bucket_fraction'])} | "
            f"{rec['max_signatures_per_bucket']} | {delta['raw_only_multi_bucket_count']} |"
        )
    lines.append("")
    lines.append("## Architecture Decision")
    lines.append("")
    lines.append(f"- decision tier: `{decision['decision_tier']}`")
    lines.append(f"- decision: `{decision['decision']}`")
    lines.append(f"- reason: {decision['reason']}")
    lines.append("")
    lines.append("## Walk_L_To_R")
    lines.append("")
    lines.append(f"- matched_pair_available: `{bool(walk_l_to_r['matched_pair_available'])}`")
    lines.append(f"- pose_d: `{_fmt(walk_l_to_r['pose_d'])}`")
    lines.append(f"- contact_d: `{_fmt(walk_l_to_r['contact_d'])}`")
    lines.append(f"- seam_support: `{walk_l_to_r['seam_support']}`")
    lines.append(f"- horizon_support: `{walk_l_to_r['horizon_support']}`")
    lines.append(f"- ungroundable_reason: `{walk_l_to_r['ungroundable_reason']}`")
    lines.append(
        "- v1_out_of_scope_flight_or_unknown_phase: "
        f"`{bool(walk_l_to_r['v1_out_of_scope_flight_or_unknown_phase'])}`"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{out_dir / 'support_contract_tightening_summary.json'}`")
    lines.append(f"- `{out_dir / 'support_contract_tightening_rows.csv'}`")
    _dump_md(out_dir / "support_contract_tightening_summary.md", lines)

    print(f"wrote {out_dir / 'support_contract_tightening_summary.md'}")
    print(f"wrote {out_dir / 'support_contract_tightening_summary.json'}")


if __name__ == "__main__":
    main()
