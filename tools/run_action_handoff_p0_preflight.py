#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.checkpoint.load_schema import load_event_motion_ckpt_payload
from train.data.io import load_soft_contacts_from_json
from train.geometry import fk_positions_from_rot6d


LOCKED_CLIPS = [
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
]

DEFAULT_CKPT = (
    "debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/"
    "ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth"
)
DEFAULT_SUBSTRATE = "debug_output/_tmp_turn_a_to_b_entry_probe_20260515"
DEFAULT_TEACHER_ROOT = "validate/teacher_batches"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
ENERGY_MAD_EPS = 1e-6
OVERLAP_COST_EPS = 1e-8
OVERLAP_COST_WEIGHTS = {
    "contact_l2": 0.4,
    "foot_pos_l2": 0.3,
    "root_l2": 0.2,
    "pose_l2": 0.1,
}
OVERLAP_AGGREGATE_Q = 0.25
OVERLAP_RUNTIME_TOP_K = 5


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _npz_scalar_to_text(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def _find_foot_joint_indices(bone_names: list[str]) -> list[int]:
    idx: list[int] = []
    for i, name in enumerate(bone_names):
        n = str(name).strip().lower()
        if n in ("foot_l", "ball_l", "foot_r", "ball_r"):
            idx.append(int(i))
    return idx


def _l2_vecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64), dtype=np.float64)


def _metric_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "worst": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "worst": float(np.max(arr)),
    }


def _pooled_p50_scale(values: list[np.ndarray], *, eps: float) -> dict[str, Any]:
    if not values:
        raise RuntimeError("overlap-cost scale requires non-empty value list")
    pooled = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values], axis=0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size <= 0:
        raise RuntimeError("overlap-cost scale got no finite values")
    p50_raw = float(np.percentile(pooled, 50.0))
    p50_used = float(max(p50_raw, float(eps)))
    return {
        "p50_raw": p50_raw,
        "p50_used": p50_used,
        "eps": float(eps),
        "pooled_count": int(pooled.size),
    }


def _overlap_cost_from_metrics(
    *,
    contact_l2: np.ndarray,
    foot_pos_l2: np.ndarray,
    root_l2: np.ndarray,
    pose_l2: np.ndarray,
    scales: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> np.ndarray:
    cost = (
        float(weights["contact_l2"])
        * (np.asarray(contact_l2, dtype=np.float64) / float(scales["contact_l2"]["p50_used"]))
        + float(weights["foot_pos_l2"])
        * (np.asarray(foot_pos_l2, dtype=np.float64) / float(scales["foot_pos_l2"]["p50_used"]))
        + float(weights["root_l2"])
        * (np.asarray(root_l2, dtype=np.float64) / float(scales["root_l2"]["p50_used"]))
        + float(weights["pose_l2"])
        * (np.asarray(pose_l2, dtype=np.float64) / float(scales["pose_l2"]["p50_used"]))
    )
    if not np.all(np.isfinite(cost)):
        raise RuntimeError("overlap cost produced non-finite values")
    return cost.astype(np.float64, copy=False)


def _empty_overlap_bucket() -> dict[str, list[float]]:
    return {"top1": [], "candidate_count": []}


def _summ_overlap_bucket(bucket: dict[str, list[float]], *, dropped_queries: int = 0) -> dict[str, Any]:
    top1 = np.asarray(bucket.get("top1", []), dtype=np.float64)
    candidate_count = np.asarray(bucket.get("candidate_count", []), dtype=np.float64)
    return {
        "num_queries": int(top1.size),
        "dropped_queries_no_overlap_candidates": int(dropped_queries),
        "top1_oracle_agreement": float(np.mean(top1)) if top1.size else None,
        "candidate_count": _metric_stats(candidate_count.tolist()),
    }


def _energy_top1_hit_within_candidates(
    *,
    src_energy: np.ndarray,
    tgt_energy: np.ndarray,
    src_frame: int,
    candidate_frames: np.ndarray,
    oracle_j: int,
) -> bool:
    frames = np.asarray(candidate_frames, dtype=np.int64).reshape(-1)
    if frames.size <= 0:
        raise RuntimeError("energy candidate frame list must be non-empty")
    if int(oracle_j) not in {int(x) for x in frames.tolist()}:
        raise RuntimeError(f"oracle target frame {oracle_j} missing from candidate frames")
    if np.any(frames < 0) or np.any(frames >= int(tgt_energy.shape[0])):
        raise RuntimeError("candidate frame out of range for target energy")
    q = float(src_energy[int(src_frame), 0])
    candidate_energy = np.asarray([float(tgt_energy[int(j), 0]) for j in frames.tolist()], dtype=np.float32)
    order = np.argsort(np.abs(candidate_energy - np.float32(q)), kind="stable")
    ranked_frames = [int(frames[int(idx)]) for idx in order.tolist()]
    return bool(ranked_frames and ranked_frames[0] == int(oracle_j))


def _extract_mean_angvel2(
    *,
    clip: str,
    npz_data: Any,
    state_layout_json: dict[str, Any],
) -> tuple[np.ndarray, str]:
    # Prefer raw angular velocity tensor from npz; fail-fast if unavailable.
    raw_key_candidates = (
        "bone_ang_vel",
        "bone_angular_vel",
        "bone_angvel",
        "bone_omega",
    )
    for key in raw_key_candidates:
        if key not in npz_data.files:
            continue
        ang = np.asarray(npz_data[key], dtype=np.float32)
        if ang.ndim != 3 or ang.shape[2] != 3:
            raise RuntimeError(f"{clip}: invalid raw angvel shape for key={key}, got {tuple(ang.shape)}")
        ang2 = np.sum(ang * ang, axis=2, dtype=np.float32)
        mean_angvel2 = np.mean(ang2, axis=1, dtype=np.float32, keepdims=True)
        return mean_angvel2.astype(np.float32, copy=False), f"npz:{key}"

    if "X_flat" in npz_data.files and isinstance(state_layout_json, dict):
        sec = state_layout_json.get("BoneAngularVelocities")
        if isinstance(sec, dict) and "start" in sec and "size" in sec:
            x_flat = np.asarray(npz_data["X_flat"], dtype=np.float32)
            if x_flat.ndim != 2:
                raise RuntimeError(f"{clip}: X_flat must be rank-2, got {tuple(x_flat.shape)}")
            start = int(sec["start"])
            size = int(sec["size"])
            if size <= 0 or (start + size) > int(x_flat.shape[1]):
                raise RuntimeError(
                    f"{clip}: invalid BoneAngularVelocities slice start={start} size={size} "
                    f"for X_flat shape={tuple(x_flat.shape)}"
                )
            if size % 3 != 0:
                raise RuntimeError(f"{clip}: BoneAngularVelocities size must be multiple of 3, got {size}")
            ang_flat = x_flat[:, start : start + size]
            ang = ang_flat.reshape(int(ang_flat.shape[0]), size // 3, 3).astype(np.float32, copy=False)
            ang2 = np.sum(ang * ang, axis=2, dtype=np.float32)
            mean_angvel2 = np.mean(ang2, axis=1, dtype=np.float32, keepdims=True)
            return mean_angvel2.astype(np.float32, copy=False), "npz:X_flat[BoneAngularVelocities]"

    raise RuntimeError(
        f"{clip}: raw angular velocity field not found in npz; "
        "refusing to use normalized-only state fallback."
    )


def _robust_pool_stats(values: list[np.ndarray], *, eps: float) -> dict[str, float]:
    if not values:
        raise RuntimeError("robust pooling requires non-empty value list")
    pooled = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
    if pooled.size <= 0:
        raise RuntimeError("robust pooling got empty pooled array")
    if not np.all(np.isfinite(pooled)):
        raise RuntimeError("robust pooling got non-finite values")
    median = float(np.median(pooled))
    mad_raw = float(np.median(np.abs(pooled - median)))
    mad_used = float(max(mad_raw, float(eps)))
    return {
        "median": median,
        "mad_raw": mad_raw,
        "mad_used": mad_used,
        "mad_eps": float(eps),
        "pooled_count": int(pooled.size),
    }


def _apply_robust_z(x: np.ndarray, *, median: float, mad_used: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    out = (x_arr - np.float32(median)) / np.float32(mad_used)
    return out.astype(np.float32, copy=False)


def _chance_topk_without_replacement(*, n_pos: int, n_total: int, k: int) -> float:
    if n_total <= 0 or n_pos <= 0:
        return 0.0
    kk = min(int(k), int(n_total))
    if kk <= 0:
        return 0.0
    n_neg = int(n_total) - int(n_pos)
    if kk > n_neg:
        return 1.0
    return float(1.0 - (math.comb(n_neg, kk) / math.comb(n_total, kk)))


def _build_energy_retrieval_separability(
    energy_by_clip: dict[str, np.ndarray],
) -> dict[str, Any]:
    all_vals: list[float] = []
    all_clips: list[str] = []
    all_frames: list[int] = []
    for clip in LOCKED_CLIPS:
        arr = np.asarray(energy_by_clip[clip], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise RuntimeError(f"{clip}: energy tensor must be [T,1], got {tuple(arr.shape)}")
        for i in range(int(arr.shape[0])):
            all_vals.append(float(arr[i, 0]))
            all_clips.append(str(clip))
            all_frames.append(int(i))

    vals = np.asarray(all_vals, dtype=np.float32)
    clips = np.asarray(all_clips, dtype=object)
    frames = np.asarray(all_frames, dtype=np.int64)

    if int(vals.shape[0]) <= 1:
        raise RuntimeError("energy retrieval requires at least 2 frames in pooled set")

    global_top1_hits: list[float] = []
    global_top3_hits: list[float] = []
    global_chance_top1: list[float] = []
    global_chance_top3: list[float] = []
    per_clip_bucket: dict[str, dict[str, list[float]]] = {
        clip: {"top1": [], "top3": [], "chance_top1": [], "chance_top3": []} for clip in LOCKED_CLIPS
    }

    n_total_frames = int(vals.shape[0])
    for q_idx in range(n_total_frames):
        q_clip = str(clips[q_idx])
        q_frame = int(frames[q_idx])
        q_val = float(vals[q_idx])

        mask_not_self = np.ones(n_total_frames, dtype=bool)
        mask_not_self[q_idx] = False
        pool_vals = vals[mask_not_self]
        pool_clips = clips[mask_not_self]
        pool_frames = frames[mask_not_self]
        pool_pos = (pool_clips == q_clip) & (pool_frames != q_frame)
        n_pos = int(np.sum(pool_pos))
        n_pool = int(pool_vals.shape[0])
        if n_pos <= 0:
            raise RuntimeError(f"{q_clip}: no positive candidates for query frame={q_frame}")

        d = np.abs(pool_vals - np.float32(q_val))
        order = np.argsort(d, kind="stable")
        top1_hit = bool(pool_pos[int(order[0])])
        top3_idx = order[: min(3, n_pool)]
        top3_hit = bool(np.any(pool_pos[top3_idx]))
        chance1 = float(n_pos / n_pool)
        chance3 = _chance_topk_without_replacement(n_pos=n_pos, n_total=n_pool, k=3)

        global_top1_hits.append(1.0 if top1_hit else 0.0)
        global_top3_hits.append(1.0 if top3_hit else 0.0)
        global_chance_top1.append(chance1)
        global_chance_top3.append(chance3)

        b = per_clip_bucket[q_clip]
        b["top1"].append(1.0 if top1_hit else 0.0)
        b["top3"].append(1.0 if top3_hit else 0.0)
        b["chance_top1"].append(chance1)
        b["chance_top3"].append(chance3)

    per_clip: dict[str, Any] = {}
    for clip in LOCKED_CLIPS:
        b = per_clip_bucket[clip]
        per_clip[clip] = {
            "num_queries": int(len(b["top1"])),
            "chance_top1": float(np.mean(np.asarray(b["chance_top1"], dtype=np.float64))),
            "chance_top3": float(np.mean(np.asarray(b["chance_top3"], dtype=np.float64))),
            "top1_accuracy": float(np.mean(np.asarray(b["top1"], dtype=np.float64))),
            "top3_accuracy": float(np.mean(np.asarray(b["top3"], dtype=np.float64))),
        }

    return {
        "task": "energy scalar same-clip retrieval over pooled 5 clips",
        "distance": "absolute energy difference |e_i - e_j|",
        "positive_definition": "same clip id as query, excluding the identical frame index",
        "negative_definition": "different clip id from query",
        "candidate_pool": "all pooled frames from 5 clips except the query frame itself",
        "metrics": {
            "num_queries": int(len(global_top1_hits)),
            "chance_top1": float(np.mean(np.asarray(global_chance_top1, dtype=np.float64))),
            "chance_top3": float(np.mean(np.asarray(global_chance_top3, dtype=np.float64))),
            "top1_accuracy": float(np.mean(np.asarray(global_top1_hits, dtype=np.float64))),
            "top3_accuracy": float(np.mean(np.asarray(global_top3_hits, dtype=np.float64))),
        },
        "per_clip": per_clip,
    }


def _build_mm_oracle_table(
    *,
    feature_bank: dict[str, dict[str, np.ndarray]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_stats: dict[str, Any] = {}
    priority = ["contact_l2", "foot_pos_l2", "root_l2", "pose_l2"]

    for src_clip, src_payload in feature_bank.items():
        src_pose = src_payload["pose"]
        src_root = src_payload["root"]
        src_contact = src_payload["contact"]
        src_foot_rot6d = src_payload["foot_rot6d"]
        src_foot_pos = src_payload["foot_pos"]
        src_t = int(src_pose.shape[0])

        for tgt_clip, tgt_payload in feature_bank.items():
            if tgt_clip == src_clip:
                continue

            tgt_pose = tgt_payload["pose"]
            tgt_root = tgt_payload["root"]
            tgt_contact = tgt_payload["contact"]
            tgt_foot_rot6d = tgt_payload["foot_rot6d"]
            tgt_foot_pos = tgt_payload["foot_pos"]
            tgt_t = int(tgt_pose.shape[0])

            pair_key = f"{src_clip}->{tgt_clip}"
            pair_best_contact: list[float] = []
            pair_best_foot_rot6d: list[float] = []
            pair_best_foot_pos: list[float] = []
            pair_best_pose: list[float] = []
            pair_best_root: list[float] = []

            for i in range(src_t):
                c_i = src_contact[i : i + 1]
                r_i = src_root[i : i + 1]
                p_i = src_pose[i : i + 1]
                f6_i = src_foot_rot6d[i : i + 1]
                fp_i = src_foot_pos[i : i + 1]

                contact_l2 = _l2_vecs(tgt_contact, c_i)
                root_l2 = _l2_vecs(tgt_root, r_i)
                pose_l2 = _l2_vecs(tgt_pose, p_i) / math.sqrt(float(tgt_pose.shape[1]))
                foot_l2 = _l2_vecs(tgt_foot_rot6d, f6_i) / math.sqrt(float(tgt_foot_rot6d.shape[1]))
                foot_pos_l2 = _l2_vecs(tgt_foot_pos, fp_i) / math.sqrt(float(tgt_foot_pos.shape[1]))

                order = np.lexsort((pose_l2, root_l2, foot_pos_l2, contact_l2))
                best_j = int(order[0])
                k = max(1, min(int(top_k), int(tgt_t)))
                shortlist_idx = order[:k].tolist()

                pair_best_contact.append(float(contact_l2[best_j]))
                pair_best_foot_rot6d.append(float(foot_l2[best_j]))
                pair_best_foot_pos.append(float(foot_pos_l2[best_j]))
                pair_best_pose.append(float(pose_l2[best_j]))
                pair_best_root.append(float(root_l2[best_j]))

                shortlist = []
                for rank, j in enumerate(shortlist_idx, start=1):
                    jj = int(j)
                    shortlist.append(
                        {
                            "rank": int(rank),
                            "target_frame": int(jj),
                            "contact_l2": float(contact_l2[jj]),
                            "foot_l2": float(foot_pos_l2[jj]),
                            "foot_l2_rot6d": float(foot_l2[jj]),
                            "foot_pos_l2": float(foot_pos_l2[jj]),
                            "root_l2": float(root_l2[jj]),
                            "pose_l2": float(pose_l2[jj]),
                        }
                    )

                rows.append(
                    {
                        "source_clip": src_clip,
                        "source_frame": int(i),
                        "target_clip": tgt_clip,
                        "target_frame": int(best_j),
                        "contact_l2": float(contact_l2[best_j]),
                        "foot_l2": float(foot_pos_l2[best_j]),
                        "foot_l2_rot6d": float(foot_l2[best_j]),
                        "foot_pos_l2": float(foot_pos_l2[best_j]),
                        "root_l2": float(root_l2[best_j]),
                        "pose_l2": float(pose_l2[best_j]),
                        "metric_priority": list(priority),
                        "retrieval_topk": shortlist,
                    }
                )

            pair_stats[pair_key] = {
                "rows": int(src_t),
                "target_frames": int(tgt_t),
                "contact_l2": _metric_stats(pair_best_contact),
                "foot_l2_rot6d": _metric_stats(pair_best_foot_rot6d),
                "foot_pos_l2": _metric_stats(pair_best_foot_pos),
                "root_l2": _metric_stats(pair_best_root),
                "pose_l2": _metric_stats(pair_best_pose),
            }

    return rows, pair_stats


def _build_pair_metric_matrices(
    *,
    src_payload: dict[str, np.ndarray],
    tgt_payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    src_pose = src_payload["pose"]
    src_root = src_payload["root"]
    src_contact = src_payload["contact"]
    src_foot_pos = src_payload["foot_pos"]
    tgt_pose = tgt_payload["pose"]
    tgt_root = tgt_payload["root"]
    tgt_contact = tgt_payload["contact"]
    tgt_foot_pos = tgt_payload["foot_pos"]

    src_t = int(src_pose.shape[0])
    tgt_t = int(tgt_pose.shape[0])
    contact_l2 = np.empty((src_t, tgt_t), dtype=np.float64)
    foot_pos_l2 = np.empty((src_t, tgt_t), dtype=np.float64)
    root_l2 = np.empty((src_t, tgt_t), dtype=np.float64)
    pose_l2 = np.empty((src_t, tgt_t), dtype=np.float64)

    pose_scale = math.sqrt(float(tgt_pose.shape[1]))
    foot_pos_scale = math.sqrt(float(tgt_foot_pos.shape[1]))
    for i in range(src_t):
        contact_l2[i] = _l2_vecs(tgt_contact, src_contact[i : i + 1])
        foot_pos_l2[i] = _l2_vecs(tgt_foot_pos, src_foot_pos[i : i + 1]) / foot_pos_scale
        root_l2[i] = _l2_vecs(tgt_root, src_root[i : i + 1])
        pose_l2[i] = _l2_vecs(tgt_pose, src_pose[i : i + 1]) / pose_scale

    return {
        "contact_l2": contact_l2,
        "foot_pos_l2": foot_pos_l2,
        "root_l2": root_l2,
        "pose_l2": pose_l2,
    }


def _add_overlap_result(
    *,
    bucket: dict[str, list[float]],
    per_source: dict[str, dict[str, list[float]]],
    per_pair: dict[str, dict[str, list[float]]],
    src_clip: str,
    pair_key: str,
    hit: bool,
    candidate_count: int,
) -> None:
    val = 1.0 if hit else 0.0
    bucket["top1"].append(val)
    bucket["candidate_count"].append(float(candidate_count))
    per_source.setdefault(src_clip, _empty_overlap_bucket())
    per_source[src_clip]["top1"].append(val)
    per_source[src_clip]["candidate_count"].append(float(candidate_count))
    per_pair.setdefault(pair_key, _empty_overlap_bucket())
    per_pair[pair_key]["top1"].append(val)
    per_pair[pair_key]["candidate_count"].append(float(candidate_count))


def _build_energy_overlap_restricted_entry_proxy(
    *,
    energy_by_clip: dict[str, np.ndarray],
    feature_bank: dict[str, dict[str, np.ndarray]],
    aggregate_q: float,
    runtime_top_k: int,
) -> dict[str, Any]:
    """Diagnostic P4 proxy inside priority-aligned natural-overlap regions."""
    metric_names = ["contact_l2", "foot_pos_l2", "root_l2", "pose_l2"]
    pair_metrics: dict[str, dict[str, np.ndarray]] = {}
    scale_values: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}

    if not (0.0 < float(aggregate_q) <= 1.0):
        raise RuntimeError(f"aggregate_q must be in (0,1], got {aggregate_q}")
    if int(runtime_top_k) <= 0:
        raise RuntimeError(f"runtime_top_k must be positive, got {runtime_top_k}")

    for src_clip in LOCKED_CLIPS:
        for tgt_clip in LOCKED_CLIPS:
            if tgt_clip == src_clip:
                continue
            pair_key = f"{src_clip}->{tgt_clip}"
            metrics = _build_pair_metric_matrices(
                src_payload=feature_bank[src_clip],
                tgt_payload=feature_bank[tgt_clip],
            )
            pair_metrics[pair_key] = metrics
            for name in metric_names:
                scale_values[name].append(metrics[name])

    scales = {name: _pooled_p50_scale(vals, eps=OVERLAP_COST_EPS) for name, vals in scale_values.items()}

    aggregate_bucket = _empty_overlap_bucket()
    aggregate_per_source: dict[str, dict[str, list[float]]] = {}
    aggregate_per_pair: dict[str, dict[str, list[float]]] = {}
    aggregate_dropped = 0
    aggregate_dropped_source = {clip: 0 for clip in LOCKED_CLIPS}
    aggregate_dropped_pair: dict[str, int] = {}

    runtime_bucket = _empty_overlap_bucket()
    runtime_per_source: dict[str, dict[str, list[float]]] = {}
    runtime_per_pair: dict[str, dict[str, list[float]]] = {}

    aggregate_pair_cost_stats: dict[str, Any] = {}
    runtime_pair_cost_stats: dict[str, Any] = {}

    for src_clip in LOCKED_CLIPS:
        src_energy = np.asarray(energy_by_clip[src_clip], dtype=np.float32)
        for tgt_clip in LOCKED_CLIPS:
            if tgt_clip == src_clip:
                continue
            tgt_energy = np.asarray(energy_by_clip[tgt_clip], dtype=np.float32)
            pair_key = f"{src_clip}->{tgt_clip}"
            metrics = pair_metrics[pair_key]
            cost = _overlap_cost_from_metrics(
                contact_l2=metrics["contact_l2"],
                foot_pos_l2=metrics["foot_pos_l2"],
                root_l2=metrics["root_l2"],
                pose_l2=metrics["pose_l2"],
                scales=scales,
                weights=OVERLAP_COST_WEIGHTS,
            )
            src_t = int(cost.shape[0])
            tgt_t = int(cost.shape[1])

            aggregate_threshold = float(np.percentile(cost.reshape(-1), 100.0 * float(aggregate_q)))
            aggregate_pair_cost_stats[pair_key] = {
                "all_pair_cost": _metric_stats(cost.reshape(-1).tolist()),
                "threshold_bottom_q": aggregate_threshold,
                "selected_candidate_pairs": int(np.sum(cost <= aggregate_threshold)),
                "total_candidate_pairs": int(cost.size),
            }
            runtime_top = max(1, min(int(runtime_top_k), int(tgt_t)))
            runtime_selected_costs: list[float] = []

            for i in range(src_t):
                aggregate_candidates = np.flatnonzero(cost[i] <= aggregate_threshold).astype(np.int64, copy=False)
                if aggregate_candidates.size <= 0:
                    aggregate_dropped += 1
                    aggregate_dropped_source[src_clip] += 1
                    aggregate_dropped_pair[pair_key] = int(aggregate_dropped_pair.get(pair_key, 0) + 1)
                else:
                    local_order = np.argsort(cost[i, aggregate_candidates], kind="stable")
                    aggregate_oracle_j = int(aggregate_candidates[int(local_order[0])])
                    aggregate_hit = _energy_top1_hit_within_candidates(
                        src_energy=src_energy,
                        tgt_energy=tgt_energy,
                        src_frame=i,
                        candidate_frames=aggregate_candidates,
                        oracle_j=aggregate_oracle_j,
                    )
                    _add_overlap_result(
                        bucket=aggregate_bucket,
                        per_source=aggregate_per_source,
                        per_pair=aggregate_per_pair,
                        src_clip=src_clip,
                        pair_key=pair_key,
                        hit=aggregate_hit,
                        candidate_count=int(aggregate_candidates.size),
                    )

                runtime_candidates = np.argsort(cost[i], kind="stable")[:runtime_top].astype(np.int64, copy=False)
                runtime_selected_costs.extend([float(cost[i, int(j)]) for j in runtime_candidates.tolist()])
                runtime_oracle_j = int(runtime_candidates[0])
                runtime_hit = _energy_top1_hit_within_candidates(
                    src_energy=src_energy,
                    tgt_energy=tgt_energy,
                    src_frame=i,
                    candidate_frames=runtime_candidates,
                    oracle_j=runtime_oracle_j,
                )
                _add_overlap_result(
                    bucket=runtime_bucket,
                    per_source=runtime_per_source,
                    per_pair=runtime_per_pair,
                    src_clip=src_clip,
                    pair_key=pair_key,
                    hit=runtime_hit,
                    candidate_count=int(runtime_candidates.size),
                )

            runtime_pair_cost_stats[pair_key] = {
                "selected_candidate_cost": _metric_stats(runtime_selected_costs),
                "selected_candidates_per_query": int(runtime_top),
            }

    return {
        "task": "energy scalar overlap-restricted cross-clip entry proxy against combined-cost P0/MM oracle",
        "status": "diagnostic_only_not_p6_pass_fail",
        "caveat": (
            "P4 overlap-restricted agreement measures consistency with, or stable deviation from, "
            "the P0/MM priority-defined ranking inside natural-overlap regions. It does not prove "
            "absolute transition quality; only P6 boundary stress can do that."
        ),
        "overlap_cost": {
            "formula": (
                "0.4*contact_l2/contact_pooled_p50 + 0.3*foot_pos_l2/foot_pos_pooled_p50 + "
                "0.2*root_l2/root_pooled_p50 + 0.1*pose_l2/pose_pooled_p50"
            ),
            "weights": dict(OVERLAP_COST_WEIGHTS),
            "normalization": "pooled p50 over all ordered cross-clip source-target frame pairs from the locked 5 clips",
            "eps": float(OVERLAP_COST_EPS),
            "scales": scales,
            "metrics": metric_names,
        },
        "aggregate_bottom_q": {
            "selection": "bottom Q fraction by overlap_cost per ordered source_clip->target_clip pair; source-frame queries with no selected target candidate are dropped",
            "q": float(aggregate_q),
            "metrics": _summ_overlap_bucket(aggregate_bucket, dropped_queries=aggregate_dropped),
            "per_source_clip": {
                clip: _summ_overlap_bucket(
                    aggregate_per_source.get(clip, _empty_overlap_bucket()),
                    dropped_queries=aggregate_dropped_source.get(clip, 0),
                )
                for clip in LOCKED_CLIPS
            },
            "per_pair": {
                key: _summ_overlap_bucket(bucket, dropped_queries=aggregate_dropped_pair.get(key, 0))
                for key, bucket in sorted(aggregate_per_pair.items())
            },
            "pair_cost_stats": aggregate_pair_cost_stats,
        },
        "runtime_topk": {
            "selection": "top K target frames by overlap_cost per source frame and target clip",
            "k": int(runtime_top_k),
            "metrics": _summ_overlap_bucket(runtime_bucket, dropped_queries=0),
            "per_source_clip": {
                clip: _summ_overlap_bucket(runtime_per_source.get(clip, _empty_overlap_bucket()), dropped_queries=0)
                for clip in LOCKED_CLIPS
            },
            "per_pair": {
                key: _summ_overlap_bucket(bucket, dropped_queries=0)
                for key, bucket in sorted(runtime_per_pair.items())
            },
            "pair_cost_stats": runtime_pair_cost_stats,
        },
    }


def _build_energy_cross_clip_entry_proxy(
    *,
    energy_by_clip: dict[str, np.ndarray],
    mm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Diagnostic-only P4 proxy: can energy match the P0 oracle top-1 within its top-k set."""
    global_top1: list[float] = []
    per_pair: dict[str, dict[str, list[float]]] = {}
    per_source: dict[str, dict[str, list[float]]] = {}

    for row in mm_rows:
        src_clip = str(row["source_clip"])
        tgt_clip = str(row["target_clip"])
        src_frame = int(row["source_frame"])
        oracle_j = int(row["target_frame"])
        pair_key = f"{src_clip}->{tgt_clip}"

        src_energy = np.asarray(energy_by_clip[src_clip], dtype=np.float32)
        tgt_energy = np.asarray(energy_by_clip[tgt_clip], dtype=np.float32)
        if src_energy.ndim != 2 or src_energy.shape[1] != 1:
            raise RuntimeError(f"{src_clip}: energy tensor must be [T,1], got {tuple(src_energy.shape)}")
        if tgt_energy.ndim != 2 or tgt_energy.shape[1] != 1:
            raise RuntimeError(f"{tgt_clip}: energy tensor must be [T,1], got {tuple(tgt_energy.shape)}")
        if not (0 <= src_frame < int(src_energy.shape[0])):
            raise RuntimeError(f"{pair_key}: source frame out of range: {src_frame}")
        if not (0 <= oracle_j < int(tgt_energy.shape[0])):
            raise RuntimeError(f"{pair_key}: oracle target frame out of range: {oracle_j}")

        q = float(src_energy[src_frame, 0])
        shortlist = row.get("retrieval_topk") or []
        candidate_frames = [int(x["target_frame"]) for x in shortlist if isinstance(x, dict) and "target_frame" in x]
        if oracle_j not in candidate_frames:
            candidate_frames.insert(0, oracle_j)
        if not candidate_frames:
            raise RuntimeError(f"{pair_key}: empty P0 candidate frame list for source frame={src_frame}")

        candidate_energy = np.asarray([float(tgt_energy[j, 0]) for j in candidate_frames], dtype=np.float32)
        order = np.argsort(np.abs(candidate_energy - np.float32(q)), kind="stable")
        ranked_frames = [candidate_frames[int(idx)] for idx in order.tolist()]
        top1_hit = bool(ranked_frames and ranked_frames[0] == oracle_j)

        top1_val = 1.0 if top1_hit else 0.0
        global_top1.append(top1_val)

        per_pair.setdefault(pair_key, {"top1": []})
        per_pair[pair_key]["top1"].append(top1_val)
        per_source.setdefault(src_clip, {"top1": []})
        per_source[src_clip]["top1"].append(top1_val)

    def _summ(bucket: dict[str, list[float]]) -> dict[str, Any]:
        top1 = np.asarray(bucket.get("top1", []), dtype=np.float64)
        return {
            "num_queries": int(top1.size),
            "top1_oracle_agreement": float(np.mean(top1)) if top1.size else None,
        }

    return {
        "task": "energy scalar cross-clip entry retrieval proxy against P0 oracle top-k",
        "status": "diagnostic_only_not_p6_pass_fail",
        "source": "P0 Motion Matching oracle table; correct target is P0 top-1 within each row's retrieval_topk candidate set",
        "distance": "absolute energy difference |e(source_i) - e(target_j)|",
        "topk_note": "Only top1 agreement is reported: oracle top1 is part of the candidate set, so top-k containment would be tautological.",
        "metrics": {
            "num_queries": int(len(global_top1)),
            "top1_oracle_agreement": float(np.mean(np.asarray(global_top1, dtype=np.float64))) if global_top1 else None,
        },
        "per_source_clip": {clip: _summ(per_source.get(clip, {"top1": []})) for clip in LOCKED_CLIPS},
        "per_pair": {key: _summ(bucket) for key, bucket in sorted(per_pair.items())},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="P0 preflight for action-handoff predictive-contrastive z probe (no z training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--teacher-root", type=str, default=DEFAULT_TEACHER_ROOT)
    ap.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--substrate-dir", type=str, default=DEFAULT_SUBSTRATE)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    today = datetime.now().strftime("%Y%m%d")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (repo_root / "debug_output" / f"_tmp_action_handoff_p0_preflight_{today}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    teacher_root = Path(args.teacher_root).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    substrate_dir = Path(args.substrate_dir).expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    ckpt_payload = load_event_motion_ckpt_payload(ckpt_path, map_location="cpu")
    ckpt_summary = {
        "path": str(ckpt_path),
        "width": int(ckpt_payload.width),
        "period_dim": int(ckpt_payload.period_dim),
        "state_dict_keys": int(len(ckpt_payload.state_dict)),
        "stripped_frozen_key_count": int(ckpt_payload.stripped_frozen_key_count),
        "sample_state_keys": sorted(list(ckpt_payload.state_dict.keys()))[:12],
    }

    feature_bank: dict[str, dict[str, np.ndarray]] = {}
    clip_reports: list[dict[str, Any]] = []
    energy_inputs: dict[str, dict[str, np.ndarray]] = {}

    for clip in LOCKED_CLIPS:
        teacher_path = teacher_root / f"{clip}_teacher.json"
        npz_path = npz_root / f"{clip}.npz"
        if not teacher_path.is_file():
            raise FileNotFoundError(f"teacher batch missing: {teacher_path}")
        if not npz_path.is_file():
            raise FileNotFoundError(f"npz missing: {npz_path}")

        teacher = _load_json(teacher_path)
        teacher_block = teacher.get("teacher", {})
        teacher_state = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
        teacher_cond = np.asarray(teacher_block.get("cond"), dtype=np.float32)
        teacher_target = np.asarray(teacher_block.get("target_norm"), dtype=np.float32)
        teacher_layout = teacher.get("layouts", {})

        with np.load(npz_path, allow_pickle=True) as d:
            bone_rot6d = np.asarray(d["bone_rot6d"], dtype=np.float32)
            root_vel = np.asarray(d["root_vel"], dtype=np.float32)
            root_pos = np.asarray(d["root_pos"], dtype=np.float32)
            parents = np.asarray(d["parents"], dtype=np.int64)
            bone_names = [str(x) for x in d["bone_names"].tolist()]
            source_json = Path(_npz_scalar_to_text(d["source_json"])).expanduser().resolve()
            state_layout_json = json.loads(_npz_scalar_to_text(d["state_layout_json"]))
            output_layout_json = json.loads(_npz_scalar_to_text(d["output_layout_json"]))
            meta_json = json.loads(_npz_scalar_to_text(d["meta_json"]))
            mean_angvel2_full, mean_angvel2_source = _extract_mean_angvel2(
                clip=clip,
                npz_data=d,
                state_layout_json=state_layout_json,
            )

        contacts_full = load_soft_contacts_from_json(str(source_json)).astype(np.float32, copy=False)
        foot_idx = _find_foot_joint_indices(bone_names)
        if not foot_idx:
            raise RuntimeError(f"{clip}: foot joint indices unresolved from bone_names")

        t_feat = min(
            int(teacher_state.shape[0]),
            int(teacher_cond.shape[0]),
            int(teacher_target.shape[0]),
            int(bone_rot6d.shape[0]),
            int(root_vel.shape[0]),
            int(root_pos.shape[0]),
            int(contacts_full.shape[0]),
        )
        if t_feat <= 0:
            raise RuntimeError(f"{clip}: no valid aligned frame count")

        pose = bone_rot6d[:t_feat].reshape(t_feat, -1).astype(np.float32, copy=False)
        foot_pose_rot6d = bone_rot6d[:t_feat, foot_idx, :].reshape(t_feat, -1).astype(np.float32, copy=False)
        root = root_vel[:t_feat].astype(np.float32, copy=False)
        contact = contacts_full[:t_feat].astype(np.float32, copy=False)
        root_speed2 = np.sum(root * root, axis=1, dtype=np.float32, keepdims=True).astype(np.float32, copy=False)
        mean_angvel2 = mean_angvel2_full[:t_feat].astype(np.float32, copy=False)

        skeleton = meta_json.get("skeleton", {}) if isinstance(meta_json, dict) else {}
        offsets = np.asarray(skeleton.get("ref_local_offsets_m", []), dtype=np.float32)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise RuntimeError(f"{clip}: invalid skeleton.ref_local_offsets_m shape={tuple(offsets.shape)}")
        if int(offsets.shape[0]) < int(bone_rot6d.shape[1]):
            raise RuntimeError(
                f"{clip}: offsets joints={int(offsets.shape[0])} < rot joints={int(bone_rot6d.shape[1])}"
            )

        rot_t = torch.from_numpy(bone_rot6d[:t_feat]).to(dtype=torch.float32, device="cpu")
        root_pos_t = torch.from_numpy(root_pos[:t_feat]).to(dtype=torch.float32, device="cpu")
        offsets_t = torch.from_numpy(offsets[: rot_t.shape[1]]).to(dtype=torch.float32, device="cpu")
        parents_t = torch.from_numpy(parents[: rot_t.shape[1]]).to(dtype=torch.long, device="cpu")
        foot_pos = (
            fk_positions_from_rot6d(
                rot6d=rot_t,
                parents=parents_t,
                offsets=offsets_t,
                root_pos=root_pos_t,
                columns=("X", "Z"),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        foot_pos_feat = foot_pos[:, foot_idx, :].reshape(t_feat, -1).astype(np.float32, copy=False)

        feature_bank[clip] = {
            "pose": pose,
            "root": root,
            "contact": contact,
            "foot_rot6d": foot_pose_rot6d,
            "foot_pos": foot_pos_feat,
        }
        energy_inputs[clip] = {
            "root_speed2": root_speed2,
            "mean_angvel2": mean_angvel2,
        }

        clip_reports.append(
            {
                "clip": clip,
                "teacher_path": str(teacher_path),
                "npz_path": str(npz_path),
                "source_json": str(source_json),
                "teacher": {
                    "state_norm": {
                        "shape": list(teacher_state.shape),
                        "dtype": str(teacher_state.dtype),
                        "device": "cpu",
                    },
                    "cond": {
                        "shape": list(teacher_cond.shape),
                        "dtype": str(teacher_cond.dtype),
                        "device": "cpu",
                    },
                    "target_norm": {
                        "shape": list(teacher_target.shape),
                        "dtype": str(teacher_target.dtype),
                        "device": "cpu",
                    },
                    "layouts": teacher_layout,
                },
                "npz_raw": {
                    "bone_rot6d": {
                        "shape": list(bone_rot6d.shape),
                        "dtype": str(bone_rot6d.dtype),
                        "device": "cpu",
                    },
                    "root_vel": {
                        "shape": list(root_vel.shape),
                        "dtype": str(root_vel.dtype),
                        "device": "cpu",
                    },
                    "root_pos": {
                        "shape": list(root_pos.shape),
                        "dtype": str(root_pos.dtype),
                        "device": "cpu",
                    },
                    "parents": {
                        "shape": list(parents.shape),
                        "dtype": str(parents.dtype),
                        "device": "cpu",
                    },
                    "contacts_soft": {
                        "shape": list(contacts_full.shape),
                        "dtype": str(contacts_full.dtype),
                        "device": "cpu",
                    },
                    "fk_offsets_local_m": {
                        "shape": list(offsets.shape),
                        "dtype": str(offsets.dtype),
                        "device": "cpu",
                        "source": "meta_json.skeleton.ref_local_offsets_m",
                    },
                    "state_layout_json": state_layout_json,
                    "output_layout_json": output_layout_json,
                },
                "aligned_export": {
                    "frames_used": int(t_feat),
                    "pose_feature": {
                        "shape": list(pose.shape),
                        "dtype": str(pose.dtype),
                        "device": "cpu",
                        "layout": "BoneRotations6D flattened from [T,J,6] to [T, J*6]",
                    },
                    "root_feature": {
                        "shape": list(root.shape),
                        "dtype": str(root.dtype),
                        "device": "cpu",
                        "layout": "RootVelocity planar [vx, vy]",
                    },
                    "root_speed2_feature": {
                        "shape": list(root_speed2.shape),
                        "dtype": str(root_speed2.dtype),
                        "device": "cpu",
                        "layout": "||root_vel||^2 per frame",
                    },
                    "mean_angvel2_feature": {
                        "shape": list(mean_angvel2.shape),
                        "dtype": str(mean_angvel2.dtype),
                        "device": "cpu",
                        "layout": "mean_j ||bone_ang_vel_j||^2 per frame",
                        "source": mean_angvel2_source,
                    },
                    "contact_feature": {
                        "shape": list(contact.shape),
                        "dtype": str(contact.dtype),
                        "device": "cpu",
                        "layout": "[soft_contact_left, soft_contact_right]",
                    },
                    "foot_pose_rot6d_feature": {
                        "shape": list(foot_pose_rot6d.shape),
                        "dtype": str(foot_pose_rot6d.dtype),
                        "device": "cpu",
                        "layout": "Foot subset rot6d flattened from [T,4,6] to [T,24]",
                        "foot_joint_indices": [int(x) for x in foot_idx],
                        "foot_joint_names": [bone_names[i] for i in foot_idx],
                    },
                    "foot_fk_position_feature": {
                        "shape": list(foot_pos_feat.shape),
                        "dtype": str(foot_pos_feat.dtype),
                        "device": "cpu",
                        "layout": "Foot subset FK world positions flattened from [T,4,3] to [T,12]",
                        "fk_columns_convention": ["X", "Z"],
                        "foot_joint_indices": [int(x) for x in foot_idx],
                        "foot_joint_names": [bone_names[i] for i in foot_idx],
                    },
                },
            }
        )

    root_speed2_stats = _robust_pool_stats(
        [energy_inputs[c]["root_speed2"] for c in LOCKED_CLIPS],
        eps=ENERGY_MAD_EPS,
    )
    mean_angvel2_stats = _robust_pool_stats(
        [energy_inputs[c]["mean_angvel2"] for c in LOCKED_CLIPS],
        eps=ENERGY_MAD_EPS,
    )

    energy_by_clip: dict[str, np.ndarray] = {}
    energy_clip_stats: dict[str, Any] = {}
    for clip in LOCKED_CLIPS:
        rs2 = energy_inputs[clip]["root_speed2"]
        av2 = energy_inputs[clip]["mean_angvel2"]
        energy = _apply_robust_z(
            rs2,
            median=root_speed2_stats["median"],
            mad_used=root_speed2_stats["mad_used"],
        ) + _apply_robust_z(
            av2,
            median=mean_angvel2_stats["median"],
            mad_used=mean_angvel2_stats["mad_used"],
        )
        energy = energy.astype(np.float32, copy=False)
        energy_by_clip[clip] = energy
        energy_clip_stats[clip] = _metric_stats(energy.reshape(-1).tolist())
        feature_bank[clip]["energy"] = energy

    for rep in clip_reports:
        clip = rep["clip"]
        ae = rep["aligned_export"]
        ae["energy_feature"] = {
            "shape": list(energy_by_clip[clip].shape),
            "dtype": str(energy_by_clip[clip].dtype),
            "device": "cpu",
            "layout": "robust_z(root_speed2) + robust_z(mean_angvel2), pooled across 5 clips",
        }
        rep["energy_distribution"] = energy_clip_stats[clip]

    energy_retrieval = _build_energy_retrieval_separability(energy_by_clip)

    mm_rows, mm_pair_stats = _build_mm_oracle_table(feature_bank=feature_bank, top_k=int(args.top_k))
    energy_cross_clip_entry = _build_energy_cross_clip_entry_proxy(
        energy_by_clip=energy_by_clip,
        mm_rows=mm_rows,
    )
    energy_cross_clip_overlap = _build_energy_overlap_restricted_entry_proxy(
        energy_by_clip=energy_by_clip,
        feature_bank=feature_bank,
        aggregate_q=OVERLAP_AGGREGATE_Q,
        runtime_top_k=OVERLAP_RUNTIME_TOP_K,
    )

    substrate = {
        "substrate_dir": str(substrate_dir),
        "exists": bool(substrate_dir.is_dir()),
    }
    if substrate_dir.is_dir():
        sweep_path = substrate_dir / "sweep_config.json"
        contract_path = substrate_dir / "contract_check_report.json"
        p2_path = substrate_dir / "p2_entry_probe_check_report.json"
        trial_path = substrate_dir / "trials" / "trial_001_Walk_L_To_R_M0_N40" / "Walk_F_freerun_cycles.json"
        substrate["files"] = {
            "sweep_config": str(sweep_path) if sweep_path.is_file() else None,
            "contract_check_report": str(contract_path) if contract_path.is_file() else None,
            "p2_entry_probe_check_report": str(p2_path) if p2_path.is_file() else None,
            "trial_example": str(trial_path) if trial_path.is_file() else None,
        }

        if sweep_path.is_file():
            sweep = _load_json(sweep_path)
            substrate["sweep_summary"] = {
                "model": sweep.get("model"),
                "teacher": sweep.get("teacher"),
                "turn_npz_list": sweep.get("turn_npz_list"),
                "inject_at_steps": sweep.get("inject_at_steps"),
                "inject_fields": sweep.get("inject_fields"),
                "entry_window_pre_k": sweep.get("entry_window_pre_k"),
                "entry_window_post_k": sweep.get("entry_window_post_k"),
                "recovery_window_k": sweep.get("recovery_window_k"),
            }
        if contract_path.is_file():
            contract = _load_json(contract_path)
            substrate["contract_summary"] = {
                "trial_count": contract.get("trial_count"),
                "all_ok": contract.get("all_ok"),
                "entry_window": contract.get("entry_window"),
                "post_inject_recovery_window": contract.get("post_inject_recovery_window"),
                "metric_classification": contract.get("metric_classification"),
            }
        if p2_path.is_file():
            p2 = _load_json(p2_path)
            substrate["p2_summary"] = {
                "trial_count": p2.get("trial_count"),
                "ok_count": p2.get("ok_count"),
                "all_subprocess_ok": p2.get("all_subprocess_ok"),
                "all_metadata_ok": p2.get("all_metadata_ok"),
            }
        if trial_path.is_file():
            trial = _load_json(trial_path)
            step0 = (trial.get("metrics_per_step") or [{}])[0]
            substrate["trial_example_summary"] = {
                "metrics_per_step_len": len(trial.get("metrics_per_step") or []),
                "top_level_keys_sample": sorted(list(trial.keys()))[:24],
                "metrics_step0_keys_sample": sorted(list(step0.keys()))[:40],
            }

    p6_minimal_change_points = [
        {
            "file": "train/validate/run_freerun_cycles.py",
            "line_hint": "1826",
            "change": "run_clip: 保持 _run_freerun_cycles() 返回 extra -> payload.update(extra) 机制，新增 z 检索决策元数据写入 extra。",
        },
        {
            "file": "train/validate/run_freerun_cycles.py",
            "line_hint": "9313",
            "change": "per_step entry 构建处追加 z_distance / retrieval_margin / entry_retrieval_decision 字段。",
        },
        {
            "file": "train/validate/run_freerun_cycles.py",
            "line_hint": "2144",
            "change": "payload.update(extra) 继续承载 trial-level entry retrieval summary，不改训练入口。",
        },
    ]

    preflight_payload = {
        "date": today,
        "scope": "P0 + probe data extraction preflight only; no z training",
        "checkpoint": ckpt_summary,
        "clips": clip_reports,
        "energy_baseline": {
            "definition": "energy_t = robust_z(root_speed2_t) + robust_z(mean_angvel2_t)",
            "root_speed2": {
                "shape": [None, 1],
                "dtype": "float32",
                "device": "cpu",
                "formula": "||root_vel_t||^2",
                "robust_pool_stats": root_speed2_stats,
            },
            "mean_angvel2": {
                "shape": [None, 1],
                "dtype": "float32",
                "device": "cpu",
                "formula": "mean_j ||bone_ang_vel_{t,j}||^2",
                "robust_pool_stats": mean_angvel2_stats,
                "source_policy": "prefer raw npz angular velocity field; fail-fast if unavailable",
            },
            "normalization_policy": {
                "pooling_scope": "union of all frames from locked 5 clips",
                "method": "median/MAD robust z-score",
                "mad_eps": float(ENERGY_MAD_EPS),
                "per_clip_normalization": False,
            },
            "clip_distribution": energy_clip_stats,
            "retrieval_separability": energy_retrieval,
            "cross_clip_entry_proxy": energy_cross_clip_entry,
            "cross_clip_overlap_restricted_proxy": energy_cross_clip_overlap,
        },
        "mm_oracle_retrieval_table": {
            "row_count": int(len(mm_rows)),
            "priority": ["contact_l2", "foot_pos_l2", "root_l2", "pose_l2"],
            "foot_metric_note": "foot_l2 is an alias for FK foot position distance; foot_l2_rot6d is retained as secondary proxy stat.",
            "table_path": str(out_dir / "mm_oracle_frame_retrieval_table.json"),
            "pair_stats": mm_pair_stats,
        },
        "substrate_reuse": substrate,
        "p6_minimal_change_points": p6_minimal_change_points,
    }

    mm_table_payload = {
        "description": "Motion Matching oracle frame retrieval table draft (cross-clip only).",
        "metric_priority": ["contact_l2", "foot_pos_l2", "root_l2", "pose_l2"],
        "foot_metric_note": "foot_l2 is an alias for FK foot position distance; foot_l2_rot6d is retained as secondary proxy stat.",
        "pair_stats": mm_pair_stats,
        "rows": mm_rows,
    }

    _dump_json(out_dir / "preflight_summary.json", preflight_payload)
    _dump_json(out_dir / "mm_oracle_frame_retrieval_table.json", mm_table_payload)

    md_lines = [
        f"# Action Handoff P0 Preflight ({today})",
        "",
        "## Scope",
        "- P0 + probe data extraction preflight only",
        "- No model posttrain, no z training, no contract write",
        "",
        "## Checkpoint",
        f"- path: `{ckpt_summary['path']}`",
        f"- width: {ckpt_summary['width']}, period_dim: {ckpt_summary['period_dim']}",
        f"- state_dict_keys: {ckpt_summary['state_dict_keys']}, stripped_frozen_key_count: {ckpt_summary['stripped_frozen_key_count']}",
        "",
        "## Clip Export (pose/root/contact)",
    ]
    for rep in clip_reports:
        ae = rep["aligned_export"]
        md_lines.extend(
            [
                f"- {rep['clip']}: frames_used={ae['frames_used']}",
                f"  - pose: shape={ae['pose_feature']['shape']} dtype={ae['pose_feature']['dtype']} device={ae['pose_feature']['device']}",
                f"  - root: shape={ae['root_feature']['shape']} dtype={ae['root_feature']['dtype']} device={ae['root_feature']['device']}",
                f"  - root_speed2: shape={ae['root_speed2_feature']['shape']} dtype={ae['root_speed2_feature']['dtype']} device={ae['root_speed2_feature']['device']}",
                f"  - mean_angvel2: shape={ae['mean_angvel2_feature']['shape']} dtype={ae['mean_angvel2_feature']['dtype']} device={ae['mean_angvel2_feature']['device']} source={ae['mean_angvel2_feature']['source']}",
                f"  - contact: shape={ae['contact_feature']['shape']} dtype={ae['contact_feature']['dtype']} device={ae['contact_feature']['device']}",
                f"  - foot_fk_pos: shape={ae['foot_fk_position_feature']['shape']} dtype={ae['foot_fk_position_feature']['dtype']} device={ae['foot_fk_position_feature']['device']}",
                f"  - energy: shape={ae['energy_feature']['shape']} dtype={ae['energy_feature']['dtype']} device={ae['energy_feature']['device']}",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Energy Baseline (P1)",
            "- definition: `energy_t = robust_z(root_speed2_t) + robust_z(mean_angvel2_t)`",
            f"- robust_z MAD eps: {ENERGY_MAD_EPS}",
            "- pooled normalization scope: union of all frames from locked 5 clips (no per-clip normalization)",
            (
                "- pooled root_speed2 stats: "
                f"median={root_speed2_stats['median']:.6f}, mad_raw={root_speed2_stats['mad_raw']:.6f}, mad_used={root_speed2_stats['mad_used']:.6f}, "
                f"count={root_speed2_stats['pooled_count']}"
            ),
            (
                "- pooled mean_angvel2 stats: "
                f"median={mean_angvel2_stats['median']:.6f}, mad_raw={mean_angvel2_stats['mad_raw']:.6f}, mad_used={mean_angvel2_stats['mad_used']:.6f}, "
                f"count={mean_angvel2_stats['pooled_count']}"
            ),
            "- per-clip energy distribution:",
        ]
    )
    for clip in LOCKED_CLIPS:
        st = energy_clip_stats[clip]
        md_lines.append(
            (
                f"  - {clip}: mean={st['mean']:.6f}, p50={st['p50']:.6f}, "
                f"p90={st['p90']:.6f}, p95={st['p95']:.6f}, worst={st['worst']:.6f}, n={st['count']}"
            )
        )
    er_m = energy_retrieval["metrics"]
    md_lines.extend(
        [
            "- retrieval/separability:",
            f"  - positive: {energy_retrieval['positive_definition']}",
            f"  - negative: {energy_retrieval['negative_definition']}",
            (
                "  - global metrics: "
                f"chance_top1={er_m['chance_top1']:.6f}, chance_top3={er_m['chance_top3']:.6f}, "
                f"top1={er_m['top1_accuracy']:.6f}, top3={er_m['top3_accuracy']:.6f}, n={er_m['num_queries']}"
            ),
        ]
    )
    for clip in LOCKED_CLIPS:
        cs = energy_retrieval["per_clip"][clip]
        md_lines.append(
            (
                f"  - {clip}: chance_top1={cs['chance_top1']:.6f}, chance_top3={cs['chance_top3']:.6f}, "
                f"top1={cs['top1_accuracy']:.6f}, top3={cs['top3_accuracy']:.6f}, n={cs['num_queries']}"
            )
        )
    er_x = energy_cross_clip_entry["metrics"]
    md_lines.extend(
        [
            "- cross-clip entry proxy (diagnostic only, not P6 pass/fail):",
            f"  - task: {energy_cross_clip_entry['task']}",
            f"  - note: {energy_cross_clip_entry['topk_note']}",
            (
                "  - global oracle agreement: "
                f"top1={er_x['top1_oracle_agreement']:.6f}, n={er_x['num_queries']}"
            ),
        ]
    )
    for clip in LOCKED_CLIPS:
        cs = energy_cross_clip_entry["per_source_clip"][clip]
        md_lines.append(
            (
                f"  - source {clip}: top1={cs['top1_oracle_agreement']:.6f}, n={cs['num_queries']}"
            )
        )
    er_ov = energy_cross_clip_overlap
    ov_agg = er_ov["aggregate_bottom_q"]["metrics"]
    ov_rt = er_ov["runtime_topk"]["metrics"]
    md_lines.extend(
        [
            "- overlap-restricted cross-clip entry proxy (diagnostic only, not P6 pass/fail):",
            f"  - caveat: {er_ov['caveat']}",
            f"  - overlap cost: {er_ov['overlap_cost']['formula']}",
            (
                "  - aggregate bottom-q: "
                f"q={er_ov['aggregate_bottom_q']['q']:.2f}, "
                f"top1={ov_agg['top1_oracle_agreement']:.6f}, n={ov_agg['num_queries']}, "
                f"dropped={ov_agg['dropped_queries_no_overlap_candidates']}"
            ),
            (
                "  - runtime top-k: "
                f"k={er_ov['runtime_topk']['k']}, "
                f"top1={ov_rt['top1_oracle_agreement']:.6f}, n={ov_rt['num_queries']}"
            ),
            "  - aggregate per-source:",
        ]
    )
    for clip in LOCKED_CLIPS:
        cs = er_ov["aggregate_bottom_q"]["per_source_clip"][clip]
        md_lines.append(
            (
                f"    - {clip}: top1={cs['top1_oracle_agreement']:.6f}, "
                f"n={cs['num_queries']}, dropped={cs['dropped_queries_no_overlap_candidates']}"
            )
        )
    md_lines.append("  - runtime per-source:")
    for clip in LOCKED_CLIPS:
        cs = er_ov["runtime_topk"]["per_source_clip"][clip]
        md_lines.append(
            (
                f"    - {clip}: top1={cs['top1_oracle_agreement']:.6f}, n={cs['num_queries']}"
            )
        )
    md_lines.extend(
        [
            "",
            "## Motion Matching Oracle Table",
            f"- rows: {len(mm_rows)}",
            "- metric priority: contact -> foot -> root -> pose",
            f"- table json: `{(out_dir / 'mm_oracle_frame_retrieval_table.json').resolve()}`",
            "",
            "## P6 Substrate Reuse",
            f"- substrate: `{substrate_dir}` (exists={substrate.get('exists')})",
            "- minimal P6 append points:",
            "  - train/validate/run_freerun_cycles.py:1826",
            "  - train/validate/run_freerun_cycles.py:9313",
            "  - train/validate/run_freerun_cycles.py:2144",
            "",
            "## Outputs",
            f"- `{(out_dir / 'preflight_summary.json').resolve()}`",
            f"- `{(out_dir / 'mm_oracle_frame_retrieval_table.json').resolve()}`",
        ]
    )
    (out_dir / "preflight_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote preflight outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
