#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.checkpoint.load_schema import load_event_motion_ckpt_payload
from train.data.io import load_soft_contacts_from_json
from train.geometry import fk_positions_from_rot6d
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _build_full_cycle_sample


LOCKED_CLIPS = (
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)

DEFAULT_CKPT = (
    "debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/"
    "ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth"
)
DEFAULT_TEACHER_ROOT = "validate/teacher_batches"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"

DEFAULT_PRETRAIN_TEMPLATE = "models/pretrain_template.json"
DEFAULT_BUNDLE = "raw_data/processed_data/norm_template.json"
DEFAULT_ENCODER_BUNDLE = "models/motion_encoder_equiv_stageA.pt"

ENERGY_MAD_EPS = 1e-6
OVERLAP_COST_EPS = 1e-8
OVERLAP_COST_WEIGHTS = {
    "contact_l2": 0.4,
    "foot_pos_l2": 0.3,
    "root_l2": 0.2,
    "pose_l2": 0.1,
}

P4_GATE_BASELINE = {
    "global": {"value": 0.343243, "queries": 1480},
    "runtime_topk": {"value": 0.237838, "queries": 1480},
    "aggregate_bottom_q": {"value": 0.126446, "queries": 1210, "dropped_queries": 270},
}

P4_FIXED_GATES = {
    "global_min": 0.343243,
    "runtime_min": 0.400000,
    "aggregate_min": 0.300000,
}

DEFAULT_ABLATION_BETA_SWEEP = (0.0, 0.03, 0.1, 0.25, 1.0, 4.0)
DEFAULT_ABLATION_ZDIM_SWEEP = (32, 64, 128, 256)
DEFAULT_ABLATION_TOP_SEEDS = (0, 1, 2)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _npz_scalar_to_text(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


def _find_foot_joint_indices(bone_names: Sequence[str]) -> list[int]:
    idx: list[int] = []
    for i, name in enumerate(bone_names):
        n = str(name).strip().lower()
        if n in ("foot_l", "ball_l", "foot_r", "ball_r"):
            idx.append(int(i))
    return idx


def _metric_stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
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


def _l2_vecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64), dtype=np.float64)


def _extract_mean_angvel2(
    *,
    clip: str,
    npz_data: Any,
    state_layout_json: dict[str, Any],
) -> tuple[np.ndarray, str]:
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
        f"{clip}: raw angular velocity field not found in npz; refusing normalized-only fallback."
    )


def _robust_pool_stats(values: list[np.ndarray], *, eps: float) -> dict[str, float]:
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


def _build_energy_retrieval_separability(energy_by_clip: dict[str, np.ndarray]) -> dict[str, Any]:
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


def _pooled_p50_scale(values: list[np.ndarray], *, eps: float) -> dict[str, Any]:
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


@dataclass(frozen=True)
class FrameRef:
    clip: str
    frame: int


@dataclass
class ClipData:
    clip: str
    teacher_path: Path
    npz_path: Path
    source_json: Path
    hidden_pre: np.ndarray
    pose: np.ndarray
    root: np.ndarray
    contact: np.ndarray
    foot_rot6d: np.ndarray
    foot_pos: np.ndarray
    energy: np.ndarray
    future_desc: np.ndarray
    teacher_state_shape: list[int]
    teacher_cond_shape: list[int]
    teacher_target_shape: list[int]
    state_layout_json: dict[str, Any]
    output_layout_json: dict[str, Any]
    mean_angvel2_source: str


def _future_desc_group_norm_and_balance(
    pose_by_clip: dict[str, np.ndarray],
    root_by_clip: dict[str, np.ndarray],
    contact_by_clip: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pose_cat = np.concatenate([pose_by_clip[c] for c in LOCKED_CLIPS], axis=0).astype(np.float32, copy=False)
    root_cat = np.concatenate([root_by_clip[c] for c in LOCKED_CLIPS], axis=0).astype(np.float32, copy=False)
    contact_cat = np.concatenate([contact_by_clip[c] for c in LOCKED_CLIPS], axis=0).astype(np.float32, copy=False)

    def _mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.mean(x, axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.std(x, axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.where(np.isfinite(std), std, np.float32(1.0)).astype(np.float32, copy=False)
        std = np.maximum(std, np.float32(1e-6)).astype(np.float32, copy=False)
        mu = np.where(np.isfinite(mu), mu, np.float32(0.0)).astype(np.float32, copy=False)
        return mu, std

    pose_mu, pose_std = _mean_std(pose_cat)
    root_mu, root_std = _mean_std(root_cat)
    contact_mu, contact_std = _mean_std(contact_cat)

    d_pose = int(pose_cat.shape[1])
    d_root = int(root_cat.shape[1])
    d_contact = int(contact_cat.shape[1])
    bal_pose = float(1.0 / math.sqrt(max(1, d_pose)))
    bal_root = float(1.0 / math.sqrt(max(1, d_root)))
    bal_contact = float(1.0 / math.sqrt(max(1, d_contact)))

    out: dict[str, np.ndarray] = {}
    for clip in LOCKED_CLIPS:
        pose_n = ((pose_by_clip[clip] - pose_mu) / pose_std).astype(np.float32, copy=False) * np.float32(bal_pose)
        root_n = ((root_by_clip[clip] - root_mu) / root_std).astype(np.float32, copy=False) * np.float32(bal_root)
        contact_n = ((contact_by_clip[clip] - contact_mu) / contact_std).astype(np.float32, copy=False) * np.float32(
            bal_contact
        )
        desc = np.concatenate([pose_n, root_n, contact_n], axis=1).astype(np.float32, copy=False)
        out[clip] = desc

    meta = {
        "groups": {
            "pose": {
                "dim": d_pose,
                "balance_scale": bal_pose,
                "mean_shape": list(pose_mu.shape),
                "std_shape": list(pose_std.shape),
            },
            "root_vel": {
                "dim": d_root,
                "balance_scale": bal_root,
                "mean_shape": list(root_mu.shape),
                "std_shape": list(root_std.shape),
            },
            "contact": {
                "dim": d_contact,
                "balance_scale": bal_contact,
                "mean_shape": list(contact_mu.shape),
                "std_shape": list(contact_std.shape),
            },
        },
        "formula": "concat(zscore(group) * (1/sqrt(dim_group)))",
    }
    return out, meta


class ZEncoder(nn.Module):
    def __init__(self, in_dim: int, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, int(z_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HorizonPredictHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HorizonQueryHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatchedReadout(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, horizons: Sequence[int]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(in_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({str(int(k)): nn.Linear(128, int(out_dim)) for k in horizons})

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        h = self.trunk(x)
        return self.heads[str(int(k))](h)


def _trainable_param_count(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters() if p.requires_grad))


def _make_freerun_runner_args(checkpoint: Path, *, device: str, bundle: str, pretrain_template: str, encoder_bundle: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(checkpoint),
        bundle=str(bundle),
        pretrain_template=str(pretrain_template),
        encoder_bundle=str(encoder_bundle),
        device=str(device),
        num_heads=4,
        dropout=0.1,
        context_len=16,
        depth=2,
        lambda_fusion_apply=True,
        allow_lambda_apply_off_ablation=False,
    )


def _extract_hidden_pre_from_sample(
    *,
    model: nn.Module,
    sample: dict[str, torch.Tensor],
) -> torch.Tensor:
    capture: dict[str, torch.Tensor] = {}

    def _pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not inputs:
            raise RuntimeError("failed to capture hidden_pre: _pasa_lnq received empty inputs")
        x = inputs[0]
        if not torch.is_tensor(x):
            raise RuntimeError("failed to capture hidden_pre: _pasa_lnq input is not tensor")
        capture["hidden_pre"] = x.detach().cpu()

    hook = model._pasa_lnq.register_forward_pre_hook(_pre_hook)  # type: ignore[attr-defined]
    try:
        with torch.no_grad():
            _ = model(
                sample["motion"].unsqueeze(0),
                sample["cond_in"].unsqueeze(0),
                contacts=sample["contacts"].unsqueeze(0),
                angvel=sample["angvel"].unsqueeze(0),
                pose_history=sample["pose_hist"].unsqueeze(0),
            )
    finally:
        hook.remove()
    hidden_pre = capture.get("hidden_pre")
    if hidden_pre is None:
        raise RuntimeError("hidden_pre capture failed: no hook payload from model._pasa_lnq")
    if hidden_pre.ndim != 3:
        raise RuntimeError(f"hidden_pre capture expects [B,T,D], got {tuple(hidden_pre.shape)}")
    return hidden_pre


def _build_clip_data(
    *,
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[dict[str, ClipData], dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    teacher_root = Path(args.teacher_root).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")

    ckpt_payload = load_event_motion_ckpt_payload(checkpoint, map_location="cpu")
    ckpt_summary = {
        "path": str(checkpoint),
        "width": int(ckpt_payload.width),
        "period_dim": int(ckpt_payload.period_dim),
        "state_dict_keys": int(len(ckpt_payload.state_dict)),
        "stripped_frozen_key_count": int(ckpt_payload.stripped_frozen_key_count),
        "sample_state_keys": sorted(list(ckpt_payload.state_dict.keys()))[:12],
    }

    runner_args = _make_freerun_runner_args(
        checkpoint,
        device=str(args.device),
        bundle=str(Path(args.bundle).expanduser()),
        pretrain_template=str(Path(args.pretrain_template).expanduser()),
        encoder_bundle=str(Path(args.encoder_bundle).expanduser()),
    )
    runner = FreeRunCycleRunner(runner_args)

    pose_by_clip: dict[str, np.ndarray] = {}
    root_by_clip: dict[str, np.ndarray] = {}
    contact_by_clip: dict[str, np.ndarray] = {}
    foot_rot6d_by_clip: dict[str, np.ndarray] = {}
    foot_pos_by_clip: dict[str, np.ndarray] = {}
    hidden_pre_by_clip: dict[str, np.ndarray] = {}
    clip_meta: dict[str, dict[str, Any]] = {}
    energy_inputs: dict[str, dict[str, np.ndarray]] = {}
    mean_angvel2_source_by_clip: dict[str, str] = {}

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
        if teacher_state.ndim != 2 or teacher_cond.ndim != 2 or teacher_target.ndim != 2:
            raise RuntimeError(f"{clip}: invalid teacher payload shapes")
        teacher_t = int(teacher_state.shape[0])

        ds = runner._build_dataset(npz_path, seq_len=teacher_t)
        runner._ensure_model_ready(ds)
        clip_obj = ds.clips[0]
        sample = _build_full_cycle_sample(ds, clip_obj, seq_len=teacher_t)
        hidden_pre_t = _extract_hidden_pre_from_sample(model=runner.model, sample=sample)  # type: ignore[arg-type]
        hidden_pre_np = hidden_pre_t.squeeze(0).contiguous().float().cpu().numpy().astype(np.float32, copy=False)

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
            int(hidden_pre_np.shape[0]),
            int(teacher_state.shape[0]),
            int(teacher_cond.shape[0]),
            int(teacher_target.shape[0]),
            int(bone_rot6d.shape[0]),
            int(root_vel.shape[0]),
            int(root_pos.shape[0]),
            int(contacts_full.shape[0]),
        )
        if args.max_frames_per_clip is not None:
            t_feat = min(t_feat, int(args.max_frames_per_clip))
        if t_feat <= 32:
            raise RuntimeError(f"{clip}: aligned frame count too small for horizons, got {t_feat}")

        pose = bone_rot6d[:t_feat].reshape(t_feat, -1).astype(np.float32, copy=False)
        foot_pose_rot6d = bone_rot6d[:t_feat, foot_idx, :].reshape(t_feat, -1).astype(np.float32, copy=False)
        root = root_vel[:t_feat].astype(np.float32, copy=False)
        contact = contacts_full[:t_feat].astype(np.float32, copy=False)
        hidden_pre = hidden_pre_np[:t_feat].astype(np.float32, copy=False)
        root_speed2 = np.sum(root * root, axis=1, dtype=np.float32, keepdims=True).astype(np.float32, copy=False)
        mean_angvel2 = mean_angvel2_full[:t_feat].astype(np.float32, copy=False)

        skeleton = meta_json.get("skeleton", {}) if isinstance(meta_json, dict) else {}
        offsets = np.asarray(skeleton.get("ref_local_offsets_m", []), dtype=np.float32)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise RuntimeError(f"{clip}: invalid skeleton.ref_local_offsets_m shape={tuple(offsets.shape)}")
        if int(offsets.shape[0]) < int(bone_rot6d.shape[1]):
            raise RuntimeError(f"{clip}: offsets joints={int(offsets.shape[0])} < rot joints={int(bone_rot6d.shape[1])}")

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

        pose_by_clip[clip] = pose
        root_by_clip[clip] = root
        contact_by_clip[clip] = contact
        foot_rot6d_by_clip[clip] = foot_pose_rot6d
        foot_pos_by_clip[clip] = foot_pos_feat
        hidden_pre_by_clip[clip] = hidden_pre
        energy_inputs[clip] = {"root_speed2": root_speed2, "mean_angvel2": mean_angvel2}
        mean_angvel2_source_by_clip[clip] = mean_angvel2_source
        clip_meta[clip] = {
            "teacher_path": str(teacher_path),
            "npz_path": str(npz_path),
            "source_json": str(source_json),
            "teacher_state_shape": list(teacher_state.shape),
            "teacher_cond_shape": list(teacher_cond.shape),
            "teacher_target_shape": list(teacher_target.shape),
            "state_layout_json": state_layout_json,
            "output_layout_json": output_layout_json,
        }

    root_speed2_stats = _robust_pool_stats([energy_inputs[c]["root_speed2"] for c in LOCKED_CLIPS], eps=ENERGY_MAD_EPS)
    mean_angvel2_stats = _robust_pool_stats([energy_inputs[c]["mean_angvel2"] for c in LOCKED_CLIPS], eps=ENERGY_MAD_EPS)

    energy_by_clip: dict[str, np.ndarray] = {}
    for clip in LOCKED_CLIPS:
        rs2 = energy_inputs[clip]["root_speed2"]
        av2 = energy_inputs[clip]["mean_angvel2"]
        energy = _apply_robust_z(rs2, median=root_speed2_stats["median"], mad_used=root_speed2_stats["mad_used"]) + _apply_robust_z(
            av2,
            median=mean_angvel2_stats["median"],
            mad_used=mean_angvel2_stats["mad_used"],
        )
        energy_by_clip[clip] = energy.astype(np.float32, copy=False)

    future_desc_by_clip, future_desc_meta = _future_desc_group_norm_and_balance(
        pose_by_clip=pose_by_clip,
        root_by_clip=root_by_clip,
        contact_by_clip=contact_by_clip,
    )

    clip_data: dict[str, ClipData] = {}
    for clip in LOCKED_CLIPS:
        meta = clip_meta[clip]
        clip_data[clip] = ClipData(
            clip=clip,
            teacher_path=Path(meta["teacher_path"]),
            npz_path=Path(meta["npz_path"]),
            source_json=Path(meta["source_json"]),
            hidden_pre=hidden_pre_by_clip[clip],
            pose=pose_by_clip[clip],
            root=root_by_clip[clip],
            contact=contact_by_clip[clip],
            foot_rot6d=foot_rot6d_by_clip[clip],
            foot_pos=foot_pos_by_clip[clip],
            energy=energy_by_clip[clip],
            future_desc=future_desc_by_clip[clip],
            teacher_state_shape=meta["teacher_state_shape"],
            teacher_cond_shape=meta["teacher_cond_shape"],
            teacher_target_shape=meta["teacher_target_shape"],
            state_layout_json=meta["state_layout_json"],
            output_layout_json=meta["output_layout_json"],
            mean_angvel2_source=mean_angvel2_source_by_clip[clip],
        )

    energy_summary = {
        "definition": "energy_t = robust_z(root_speed2_t) + robust_z(mean_angvel2_t)",
        "root_speed2_pool": root_speed2_stats,
        "mean_angvel2_pool": mean_angvel2_stats,
        "clip_distribution": {clip: _metric_stats(clip_data[clip].energy.reshape(-1).tolist()) for clip in LOCKED_CLIPS},
        "retrieval_separability": _build_energy_retrieval_separability(energy_by_clip),
    }

    shape_summary: dict[str, Any] = {}
    for clip in LOCKED_CLIPS:
        cd = clip_data[clip]
        shape_summary[clip] = {
            "hidden_pre": {"shape": list(cd.hidden_pre.shape), "dtype": str(cd.hidden_pre.dtype), "device": "cpu"},
            "pose": {"shape": list(cd.pose.shape), "dtype": str(cd.pose.dtype), "device": "cpu"},
            "root": {"shape": list(cd.root.shape), "dtype": str(cd.root.dtype), "device": "cpu"},
            "contact": {"shape": list(cd.contact.shape), "dtype": str(cd.contact.dtype), "device": "cpu"},
            "future_desc": {"shape": list(cd.future_desc.shape), "dtype": str(cd.future_desc.dtype), "device": "cpu"},
            "energy": {"shape": list(cd.energy.shape), "dtype": str(cd.energy.dtype), "device": "cpu"},
            "teacher_state_shape": list(cd.teacher_state_shape),
            "teacher_cond_shape": list(cd.teacher_cond_shape),
            "teacher_target_shape": list(cd.teacher_target_shape),
            "teacher_path": str(cd.teacher_path),
            "npz_path": str(cd.npz_path),
            "source_json": str(cd.source_json),
        }

    return clip_data, ckpt_summary, shape_summary, {"energy": energy_summary, "future_desc": future_desc_meta}


def _build_global_frame_index(clip_data: dict[str, ClipData]) -> tuple[list[FrameRef], dict[str, int], np.ndarray]:
    refs: list[FrameRef] = []
    offset: dict[str, int] = {}
    for clip in LOCKED_CLIPS:
        offset[clip] = len(refs)
        t = int(clip_data[clip].pose.shape[0])
        for i in range(t):
            refs.append(FrameRef(clip=clip, frame=i))
    clip_ids = np.asarray([LOCKED_CLIPS.index(r.clip) for r in refs], dtype=np.int64)
    return refs, offset, clip_ids


def _frame_to_global(offset: dict[str, int], clip: str, frame: int) -> int:
    return int(offset[clip] + int(frame))


def _build_pose_hard_neighbor_order(clip_data: dict[str, ClipData], refs: list[FrameRef]) -> np.ndarray:
    pose_all = np.concatenate([clip_data[c].pose for c in LOCKED_CLIPS], axis=0).astype(np.float32, copy=False)
    x2 = np.sum(pose_all * pose_all, axis=1, dtype=np.float64, keepdims=True)
    dist2 = np.maximum(x2 + x2.T - 2.0 * (pose_all.astype(np.float64) @ pose_all.astype(np.float64).T), 0.0)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1, kind="stable")
    if order.shape[0] != len(refs):
        raise RuntimeError("hard neighbor order shape mismatch")
    return order.astype(np.int64, copy=False)


def _select_negatives_for_target(
    *,
    target_global_idx: int,
    refs: list[FrameRef],
    hard_order: np.ndarray,
    pos_set: set[int],
    same_window_radius: int,
    n_hard: int,
    n_easy: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target_ref = refs[int(target_global_idx)]
    all_idx = np.arange(len(refs), dtype=np.int64)

    def _eligible(idx: int) -> bool:
        if idx in pos_set:
            return False
        r = refs[int(idx)]
        if r.clip == target_ref.clip and abs(int(r.frame) - int(target_ref.frame)) <= int(same_window_radius):
            return False
        return True

    hard_list: list[int] = []
    for idx in hard_order[int(target_global_idx)].tolist():
        ii = int(idx)
        if not _eligible(ii):
            continue
        hard_list.append(ii)
        if len(hard_list) >= int(n_hard):
            break

    easy_pool = [int(x) for x in all_idx.tolist() if _eligible(int(x))]
    if len(easy_pool) <= 0:
        raise RuntimeError(f"no eligible negatives for target_global_idx={target_global_idx}")
    if len(easy_pool) <= int(n_easy):
        easy_list = easy_pool
    else:
        easy_list = rng.choice(np.asarray(easy_pool, dtype=np.int64), size=int(n_easy), replace=False).astype(np.int64).tolist()

    neg = sorted(set(hard_list + easy_list))
    if not neg:
        raise RuntimeError(f"negative sampling produced empty set for target_global_idx={target_global_idx}")
    return np.asarray(neg, dtype=np.int64)


def _cosine_top1_hit(
    src_repr: np.ndarray,
    tgt_repr: np.ndarray,
    src_frame: int,
    candidate_frames: np.ndarray,
    oracle_j: int,
    *,
    scalar_abs_mode: bool,
) -> bool:
    frames = np.asarray(candidate_frames, dtype=np.int64).reshape(-1)
    if frames.size <= 0:
        raise RuntimeError("candidate frame list must be non-empty")
    if int(oracle_j) not in {int(x) for x in frames.tolist()}:
        raise RuntimeError(f"oracle target frame {oracle_j} missing from candidate frames")
    if np.any(frames < 0) or np.any(frames >= int(tgt_repr.shape[0])):
        raise RuntimeError("candidate frame out of range")

    q = src_repr[int(src_frame)]
    cand = tgt_repr[frames]
    if scalar_abs_mode:
        qv = float(q.reshape(-1)[0])
        score = np.abs(cand.reshape(cand.shape[0], -1)[:, 0] - np.float32(qv)).astype(np.float64, copy=False)
        order = np.argsort(score, kind="stable")
    else:
        qn = q / max(float(np.linalg.norm(q)), 1e-8)
        cn = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-8)
        score = (cn @ qn).astype(np.float64, copy=False)
        order = np.argsort(-score, kind="stable")
    ranked_frames = [int(frames[int(idx)]) for idx in order.tolist()]
    return bool(ranked_frames and ranked_frames[0] == int(oracle_j))


def _build_cross_clip_global_agreement(
    *,
    repr_by_clip: dict[str, np.ndarray],
    mm_rows: list[dict[str, Any]],
    scalar_abs_mode: bool,
) -> dict[str, Any]:
    global_top1: list[float] = []
    per_pair: dict[str, dict[str, list[float]]] = {}
    per_source: dict[str, dict[str, list[float]]] = {}

    for row in mm_rows:
        src_clip = str(row["source_clip"])
        tgt_clip = str(row["target_clip"])
        src_frame = int(row["source_frame"])
        oracle_j = int(row["target_frame"])
        pair_key = f"{src_clip}->{tgt_clip}"

        src_repr = np.asarray(repr_by_clip[src_clip], dtype=np.float32)
        tgt_repr = np.asarray(repr_by_clip[tgt_clip], dtype=np.float32)
        shortlist = row.get("retrieval_topk") or []
        candidate_frames = [int(x["target_frame"]) for x in shortlist if isinstance(x, dict) and "target_frame" in x]
        if oracle_j not in candidate_frames:
            candidate_frames.insert(0, oracle_j)
        if not candidate_frames:
            raise RuntimeError(f"{pair_key}: empty P0 candidate frame list for source frame={src_frame}")
        top1_hit = _cosine_top1_hit(
            src_repr=src_repr,
            tgt_repr=tgt_repr,
            src_frame=src_frame,
            candidate_frames=np.asarray(candidate_frames, dtype=np.int64),
            oracle_j=oracle_j,
            scalar_abs_mode=bool(scalar_abs_mode),
        )
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
        "task": "cross-clip global agreement against P0 oracle shortlist",
        "distance": "absolute diff on scalar arm; cosine similarity on vector arms",
        "metrics": {
            "num_queries": int(len(global_top1)),
            "top1_oracle_agreement": float(np.mean(np.asarray(global_top1, dtype=np.float64))) if global_top1 else None,
        },
        "per_source_clip": {clip: _summ(per_source.get(clip, {"top1": []})) for clip in LOCKED_CLIPS},
        "per_pair": {key: _summ(bucket) for key, bucket in sorted(per_pair.items())},
    }


def _build_cross_clip_overlap_restricted_agreement(
    *,
    repr_by_clip: dict[str, np.ndarray],
    feature_bank: dict[str, dict[str, np.ndarray]],
    aggregate_q: float,
    runtime_top_k: int,
    scalar_abs_mode: bool,
) -> dict[str, Any]:
    metric_names = ["contact_l2", "foot_pos_l2", "root_l2", "pose_l2"]
    pair_metrics: dict[str, dict[str, np.ndarray]] = {}
    scale_values: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
    for src_clip in LOCKED_CLIPS:
        for tgt_clip in LOCKED_CLIPS:
            if tgt_clip == src_clip:
                continue
            pair_key = f"{src_clip}->{tgt_clip}"
            metrics = _build_pair_metric_matrices(src_payload=feature_bank[src_clip], tgt_payload=feature_bank[tgt_clip])
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
        src_repr = np.asarray(repr_by_clip[src_clip], dtype=np.float32)
        for tgt_clip in LOCKED_CLIPS:
            if tgt_clip == src_clip:
                continue
            tgt_repr = np.asarray(repr_by_clip[tgt_clip], dtype=np.float32)
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
                    aggregate_hit = _cosine_top1_hit(
                        src_repr=src_repr,
                        tgt_repr=tgt_repr,
                        src_frame=i,
                        candidate_frames=aggregate_candidates,
                        oracle_j=aggregate_oracle_j,
                        scalar_abs_mode=bool(scalar_abs_mode),
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
                runtime_hit = _cosine_top1_hit(
                    src_repr=src_repr,
                    tgt_repr=tgt_repr,
                    src_frame=i,
                    candidate_frames=runtime_candidates,
                    oracle_j=runtime_oracle_j,
                    scalar_abs_mode=bool(scalar_abs_mode),
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
        "task": "overlap-restricted cross-clip agreement against P0/MM oracle",
        "overlap_cost": {
            "formula": (
                "0.4*contact_l2/contact_pooled_p50 + 0.3*foot_pos_l2/foot_pos_pooled_p50 + "
                "0.2*root_l2/root_pooled_p50 + 0.1*pose_l2/pose_pooled_p50"
            ),
            "weights": dict(OVERLAP_COST_WEIGHTS),
            "normalization": "pooled p50 over all ordered cross-clip source-target frame pairs from locked 5 clips",
            "eps": float(OVERLAP_COST_EPS),
            "scales": scales,
            "metrics": metric_names,
        },
        "aggregate_bottom_q": {
            "selection": "bottom Q fraction by overlap_cost per ordered source_clip->target_clip pair; no-candidate queries are dropped",
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
                key: _summ_overlap_bucket(bucket, dropped_queries=0) for key, bucket in sorted(runtime_per_pair.items())
            },
            "pair_cost_stats": runtime_pair_cost_stats,
        },
    }


def _flatten_repr(repr_by_clip: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    out: dict[str, np.ndarray] = {}
    dim: int | None = None
    for clip in LOCKED_CLIPS:
        arr = np.asarray(repr_by_clip[clip], dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"{clip}: repr must be [T,D], got {tuple(arr.shape)}")
        if dim is None:
            dim = int(arr.shape[1])
        elif int(arr.shape[1]) != int(dim):
            raise RuntimeError("representation dim mismatch across clips")
        out[clip] = arr
    if dim is None:
        raise RuntimeError("empty repr")
    return out, int(dim)


def _split_indices_by_clip(clip_data: dict[str, ClipData], *, train_ratio: float) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for clip in LOCKED_CLIPS:
        t = int(clip_data[clip].future_desc.shape[0])
        if t < 8:
            raise RuntimeError(f"{clip}: too few frames for split: {t}")
        n_train = int(max(1, min(t - 1, math.floor(t * float(train_ratio)))))
        train_idx = np.arange(0, n_train, dtype=np.int64)
        test_idx = np.arange(n_train, t, dtype=np.int64)
        out[clip] = {"train": train_idx, "test": test_idx}
    return out


def _compute_weighted_predictive_loss(
    *,
    features_by_clip_t: dict[str, torch.Tensor],
    future_desc_by_clip_t: dict[str, torch.Tensor],
    horizons: Sequence[int],
    weights_by_h: dict[int, float],
    split: dict[str, dict[str, np.ndarray]],
    model: MatchedReadout,
    subset: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], int]:
    total = torch.zeros((), device=device, dtype=torch.float32)
    total_weight = torch.zeros((), device=device, dtype=torch.float32)
    per_clip_vals: dict[str, list[float]] = {clip: [] for clip in LOCKED_CLIPS}
    per_clip_w: dict[str, list[float]] = {clip: [] for clip in LOCKED_CLIPS}
    query_count = 0

    for clip in LOCKED_CLIPS:
        x = features_by_clip_t[clip]
        y = future_desc_by_clip_t[clip]
        idx_base = split[clip][subset]
        for k in horizons:
            valid = idx_base[idx_base + int(k) < int(y.shape[0])]
            if valid.size <= 0:
                continue
            idx_t = torch.as_tensor(valid, dtype=torch.long, device=device)
            pred = model(x.index_select(0, idx_t), int(k))
            tgt = y.index_select(0, idx_t + int(k))
            l = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)
            wk = float(weights_by_h[int(k)])
            total = total + torch.mean(l) * wk
            total_weight = total_weight + torch.tensor(wk, device=device, dtype=torch.float32)
            per_clip_vals[clip].append(float(torch.mean(l).detach().cpu()))
            per_clip_w[clip].append(float(wk))
            query_count += int(valid.size)
    if float(total_weight.detach().cpu()) <= 0.0:
        raise RuntimeError(f"no valid predictive terms for subset={subset}")

    per_clip_metric: dict[str, float] = {}
    for clip in LOCKED_CLIPS:
        if not per_clip_vals[clip]:
            per_clip_metric[clip] = float("nan")
            continue
        arr = np.asarray(per_clip_vals[clip], dtype=np.float64)
        ww = np.asarray(per_clip_w[clip], dtype=np.float64)
        per_clip_metric[clip] = float(np.sum(arr * ww) / np.sum(ww))
    return (total / total_weight), per_clip_metric, query_count


def _train_matched_readout(
    *,
    repr_by_clip: dict[str, np.ndarray],
    clip_data: dict[str, ClipData],
    horizons: Sequence[int],
    weights_by_h: dict[int, float],
    split: dict[str, dict[str, np.ndarray]],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict[str, Any]:
    repr_by_clip_np, in_dim = _flatten_repr(repr_by_clip)
    out_dim = int(next(iter(clip_data.values())).future_desc.shape[1])
    readout = MatchedReadout(in_dim=in_dim, out_dim=out_dim, horizons=horizons).to(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    feat_t = {c: torch.from_numpy(repr_by_clip_np[c]).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}
    desc_t = {c: torch.from_numpy(clip_data[c].future_desc).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}

    train_curve: list[float] = []
    for _ep in range(int(epochs)):
        readout.train()
        opt.zero_grad(set_to_none=True)
        loss, _, _ = _compute_weighted_predictive_loss(
            features_by_clip_t=feat_t,
            future_desc_by_clip_t=desc_t,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            model=readout,
            subset="train",
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), max_norm=5.0)
        opt.step()
        train_curve.append(float(loss.detach().cpu()))

    readout.eval()
    with torch.no_grad():
        train_loss, train_per_clip, train_q = _compute_weighted_predictive_loss(
            features_by_clip_t=feat_t,
            future_desc_by_clip_t=desc_t,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            model=readout,
            subset="train",
            device=device,
        )
        test_loss, test_per_clip, test_q = _compute_weighted_predictive_loss(
            features_by_clip_t=feat_t,
            future_desc_by_clip_t=desc_t,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            model=readout,
            subset="test",
            device=device,
        )

    return {
        "repr_dim": int(in_dim),
        "readout_trainable_params": _trainable_param_count(readout),
        "loss": {
            "train_weighted_huber": float(train_loss.detach().cpu()),
            "test_weighted_huber": float(test_loss.detach().cpu()),
        },
        "queries": {
            "train_terms": int(train_q),
            "test_terms": int(test_q),
        },
        "per_clip": {
            "train_weighted_huber": {k: float(v) for k, v in train_per_clip.items()},
            "test_weighted_huber": {k: float(v) for k, v in test_per_clip.items()},
        },
        "train_curve": {
            "first": float(train_curve[0]) if train_curve else None,
            "last": float(train_curve[-1]) if train_curve else None,
            "best": float(min(train_curve)) if train_curve else None,
            "epochs": int(epochs),
        },
    }


def _train_z_probe(
    *,
    clip_data: dict[str, ClipData],
    horizons: Sequence[int],
    beta: float,
    tau: float,
    z_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_easy_neg: int,
    n_hard_neg: int,
    same_window_radius: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], nn.Module, nn.ModuleDict]:
    if tau <= 0.0:
        raise RuntimeError(f"tau must be positive, got {tau}")
    rng = np.random.default_rng(int(seed))
    refs, offset, _clip_ids = _build_global_frame_index(clip_data)
    hard_order = _build_pose_hard_neighbor_order(clip_data, refs)

    hidden_in_dim = int(next(iter(clip_data.values())).hidden_pre.shape[1])
    desc_dim = int(next(iter(clip_data.values())).future_desc.shape[1])
    encoder = ZEncoder(in_dim=hidden_in_dim, z_dim=int(z_dim)).to(device)
    pred_heads = nn.ModuleDict({str(int(k)): HorizonPredictHead(z_dim=int(z_dim), out_dim=desc_dim).to(device) for k in horizons})
    q_heads = nn.ModuleDict({str(int(k)): HorizonQueryHead(z_dim=int(z_dim), out_dim=desc_dim).to(device) for k in horizons})

    params = list(encoder.parameters()) + list(pred_heads.parameters()) + list(q_heads.parameters())
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    hidden_t = {c: torch.from_numpy(clip_data[c].hidden_pre).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}
    desc_t = {c: torch.from_numpy(clip_data[c].future_desc).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}
    global_desc_t = torch.from_numpy(np.concatenate([clip_data[c].future_desc for c in LOCKED_CLIPS], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    weights_by_h = {int(k): float(1.0 / math.sqrt(float(k))) for k in horizons}
    info_pos_offsets = (-2, -1, 0, 1, 2)

    curve_total: list[float] = []
    curve_predict: list[float] = []
    curve_nce: list[float] = []
    curve_terms: list[int] = []

    for _ep in range(int(epochs)):
        encoder.train()
        pred_heads.train()
        q_heads.train()
        opt.zero_grad(set_to_none=True)
        z_by_clip = {c: encoder(hidden_t[c]) for c in LOCKED_CLIPS}
        total_predict = torch.zeros((), device=device, dtype=torch.float32)
        total_predict_w = torch.zeros((), device=device, dtype=torch.float32)
        total_nce = torch.zeros((), device=device, dtype=torch.float32)
        nce_terms = 0

        for clip in LOCKED_CLIPS:
            zc = z_by_clip[clip]
            yc = desc_t[clip]
            t_clip = int(zc.shape[0])
            for k in horizons:
                k_int = int(k)
                n = t_clip - k_int
                if n <= 0:
                    continue
                wk = float(weights_by_h[k_int])
                z_anchor = zc[:n]
                pred = pred_heads[str(k_int)](z_anchor)
                tgt = yc[k_int : k_int + n]
                pred_loss = F.smooth_l1_loss(pred, tgt, reduction="mean")
                total_predict = total_predict + pred_loss * wk
                total_predict_w = total_predict_w + torch.tensor(wk, device=device, dtype=torch.float32)

                q_vec = F.normalize(q_heads[str(k_int)](z_anchor), dim=1)
                for i in range(n):
                    target_frame = i + k_int
                    pos_ids: list[int] = []
                    for d in info_pos_offsets:
                        j = int(target_frame + d)
                        if 0 <= j < t_clip:
                            pos_ids.append(_frame_to_global(offset, clip, j))
                    pos_ids = sorted(set(pos_ids))
                    if not pos_ids:
                        continue
                    pos_set = set(pos_ids)
                    target_global = _frame_to_global(offset, clip, target_frame)
                    neg_ids = _select_negatives_for_target(
                        target_global_idx=target_global,
                        refs=refs,
                        hard_order=hard_order,
                        pos_set=pos_set,
                        same_window_radius=int(same_window_radius),
                        n_hard=int(n_hard_neg),
                        n_easy=int(n_easy_neg),
                        rng=rng,
                    )

                    candidate_ids = np.asarray(pos_ids + neg_ids.tolist(), dtype=np.int64)
                    labels = np.asarray([1] * len(pos_ids) + [0] * len(neg_ids), dtype=np.int64)
                    desc_cand = global_desc_t.index_select(
                        0,
                        torch.from_numpy(candidate_ids).to(device=device, dtype=torch.long),
                    )
                    desc_cand = F.normalize(desc_cand, dim=1)
                    logits = torch.matmul(desc_cand, q_vec[i].unsqueeze(-1)).squeeze(-1) / float(tau)
                    pos_mask = torch.from_numpy(labels).to(device=device, dtype=torch.bool)
                    log_denom = torch.logsumexp(logits, dim=0)
                    log_pos = torch.logsumexp(logits[pos_mask], dim=0)
                    nce = -(log_pos - log_denom)
                    total_nce = total_nce + nce
                    nce_terms += 1

        if float(total_predict_w.detach().cpu()) <= 0.0:
            raise RuntimeError("z training produced no predictive terms")
        lp = total_predict / total_predict_w
        lnce = total_nce / max(1, int(nce_terms))
        loss = lp + float(beta) * lnce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        curve_total.append(float(loss.detach().cpu()))
        curve_predict.append(float(lp.detach().cpu()))
        curve_nce.append(float(lnce.detach().cpu()))
        curve_terms.append(int(nce_terms))

    encoder.eval()
    with torch.no_grad():
        z_by_clip_np = {
            clip: encoder(hidden_t[clip]).detach().cpu().numpy().astype(np.float32, copy=False) for clip in LOCKED_CLIPS
        }

    summary = {
        "encoder_arch": f"MLP({hidden_in_dim}->256->128->{int(z_dim)}) + LayerNorm + GELU",
        "loss": {
            "formula": "L_predict + 0.25 * L_InfoNCE",
            "beta": float(beta),
            "tau": float(tau),
            "horizons": [int(k) for k in horizons],
            "weights_w_k": {str(int(k)): float(weights_by_h[int(k)]) for k in horizons},
            "multi_positive_offsets": list(info_pos_offsets),
            "hard_negatives": {
                "method": "pose L2 nearest over pooled locked clips",
                "filter": "exclude same-clip same-window around target frame",
                "same_window_radius": int(same_window_radius),
                "n_hard": int(n_hard_neg),
                "n_easy": int(n_easy_neg),
            },
        },
        "trainable_params": {
            "encoder": _trainable_param_count(encoder),
            "predict_heads": _trainable_param_count(pred_heads),
            "query_heads": _trainable_param_count(q_heads),
            "total": _trainable_param_count(encoder) + _trainable_param_count(pred_heads) + _trainable_param_count(q_heads),
        },
        "curve": {
            "epochs": int(epochs),
            "loss_total_first": float(curve_total[0]) if curve_total else None,
            "loss_total_last": float(curve_total[-1]) if curve_total else None,
            "loss_total_best": float(min(curve_total)) if curve_total else None,
            "loss_predict_first": float(curve_predict[0]) if curve_predict else None,
            "loss_predict_last": float(curve_predict[-1]) if curve_predict else None,
            "loss_nce_first": float(curve_nce[0]) if curve_nce else None,
            "loss_nce_last": float(curve_nce[-1]) if curve_nce else None,
            "nce_terms_last_epoch": int(curve_terms[-1]) if curve_terms else 0,
        },
    }
    return z_by_clip_np, summary, encoder, pred_heads


def _parse_csv_ints(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in str(spec).split(",") if str(x).strip()]
    if sorted(set(vals)) != vals:
        raise RuntimeError(f"values must be strict ascending unique ints, got {vals}")
    return vals


def _safe_tag(v: Any) -> str:
    s = str(v)
    s = s.replace("-", "m").replace(".", "p")
    out = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)
    return out or "na"


def _build_config_id(*, beta: float, z_dim: int, seed: int, epochs: int, readout_epochs: int) -> str:
    return (
        f"beta_{_safe_tag(f'{float(beta):.6g}')}"
        f"__dz_{int(z_dim)}__seed_{int(seed)}"
        f"__ep_{int(epochs)}__rep_{int(readout_epochs)}"
    )


def _weighted_huber_for_pred_heads(
    *,
    z_by_clip_t: dict[str, torch.Tensor],
    future_desc_by_clip_t: dict[str, torch.Tensor],
    pred_heads: nn.ModuleDict,
    horizons: Sequence[int],
    weights_by_h: dict[int, float],
    split: dict[str, dict[str, np.ndarray]],
    subset: str,
    device: torch.device,
) -> tuple[float, int]:
    total = torch.zeros((), device=device, dtype=torch.float32)
    total_w = torch.zeros((), device=device, dtype=torch.float32)
    q_terms = 0
    for clip in LOCKED_CLIPS:
        zc = z_by_clip_t[clip]
        yc = future_desc_by_clip_t[clip]
        base = split[clip][subset]
        for k in horizons:
            valid = base[base + int(k) < int(yc.shape[0])]
            if valid.size <= 0:
                continue
            idx_t = torch.as_tensor(valid, dtype=torch.long, device=device)
            pred = pred_heads[str(int(k))](zc.index_select(0, idx_t))
            tgt = yc.index_select(0, idx_t + int(k))
            l = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)
            wk = float(weights_by_h[int(k)])
            total = total + torch.mean(l) * wk
            total_w = total_w + torch.tensor(wk, device=device, dtype=torch.float32)
            q_terms += int(valid.size)
    if float(total_w.detach().cpu()) <= 0.0:
        raise RuntimeError(f"pred-head audit has no valid terms for subset={subset}")
    return float((total / total_w).detach().cpu()), int(q_terms)


def _build_run_artifacts(
    *,
    clip_data: dict[str, ClipData],
    ckpt_summary: dict[str, Any],
    shape_summary: dict[str, Any],
    feature_meta: dict[str, Any],
    horizons: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    beta: float,
    z_dim: int,
    seed: int,
    epochs: int,
    readout_epochs: int,
    today: str,
) -> dict[str, Any]:
    _set_seed(int(seed))
    z_by_clip, z_train_summary, encoder, pred_heads = _train_z_probe(
        clip_data=clip_data,
        horizons=horizons,
        beta=float(beta),
        tau=float(args.tau),
        z_dim=int(z_dim),
        epochs=int(epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        n_easy_neg=int(args.easy_negs),
        n_hard_neg=int(args.hard_negs),
        same_window_radius=int(args.neg_same_window_radius),
        seed=int(seed),
        device=device,
    )

    split = _split_indices_by_clip(clip_data, train_ratio=float(args.train_ratio))
    weights_by_h = {int(k): float(1.0 / math.sqrt(float(k))) for k in horizons}
    arms: dict[str, dict[str, Any]] = {}
    arm_repr: dict[str, dict[str, np.ndarray]] = {
        "energy_scalar": {clip: clip_data[clip].energy.astype(np.float32, copy=False) for clip in LOCKED_CLIPS},
        "raw_hidden_pre": {clip: clip_data[clip].hidden_pre.astype(np.float32, copy=False) for clip in LOCKED_CLIPS},
        "z_bottleneck": {clip: z_by_clip[clip].astype(np.float32, copy=False) for clip in LOCKED_CLIPS},
    }

    for arm_name in ("energy_scalar", "raw_hidden_pre", "z_bottleneck"):
        readout_summary = _train_matched_readout(
            repr_by_clip=arm_repr[arm_name],
            clip_data=clip_data,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            epochs=int(readout_epochs),
            lr=float(args.readout_lr),
            weight_decay=float(args.readout_weight_decay),
            device=device,
        )
        per_clip_shape = {
            clip: {
                "shape": list(arm_repr[arm_name][clip].shape),
                "dtype": str(arm_repr[arm_name][clip].dtype),
                "device": "cpu",
            }
            for clip in LOCKED_CLIPS
        }
        arms[arm_name] = {
            "feature_shape_dtype_device_per_clip": per_clip_shape,
            "matched_readout_arch": "Linear(in,128)+LayerNorm+GELU + per-horizon Linear(128,D_desc)",
            **readout_summary,
        }

    p1 = {
        "task": "P1 predictive compare (matched readout)",
        "split_policy": {
            "train_ratio": float(args.train_ratio),
            "policy": "per-clip temporal split (first ratio for train, tail for test)",
        },
        "horizons": [int(k) for k in horizons],
        "weights_w_k": {str(int(k)): float(weights_by_h[int(k)]) for k in horizons},
        "arms": arms,
    }

    feature_bank = {
        clip: {
            "pose": clip_data[clip].pose,
            "root": clip_data[clip].root,
            "contact": clip_data[clip].contact,
            "foot_rot6d": clip_data[clip].foot_rot6d,
            "foot_pos": clip_data[clip].foot_pos,
        }
        for clip in LOCKED_CLIPS
    }
    mm_rows, mm_pair_stats = _build_mm_oracle_table(feature_bank=feature_bank, top_k=int(args.mm_oracle_top_k))

    p4_arms: dict[str, Any] = {}
    for arm_name in ("energy_scalar", "raw_hidden_pre", "z_bottleneck"):
        scalar_abs_mode = arm_name == "energy_scalar"
        global_view = _build_cross_clip_global_agreement(
            repr_by_clip=arm_repr[arm_name],
            mm_rows=mm_rows,
            scalar_abs_mode=scalar_abs_mode,
        )
        overlap_view = _build_cross_clip_overlap_restricted_agreement(
            repr_by_clip=arm_repr[arm_name],
            feature_bank=feature_bank,
            aggregate_q=float(args.overlap_aggregate_q),
            runtime_top_k=int(args.overlap_runtime_top_k),
            scalar_abs_mode=scalar_abs_mode,
        )
        p4_arms[arm_name] = {
            "global": global_view,
            "overlap_restricted_runtime": overlap_view["runtime_topk"],
            "overlap_restricted_aggregate": overlap_view["aggregate_bottom_q"],
            "overlap_cost": overlap_view["overlap_cost"],
        }

    z_global = float(p4_arms["z_bottleneck"]["global"]["metrics"]["top1_oracle_agreement"])
    z_runtime = float(p4_arms["z_bottleneck"]["overlap_restricted_runtime"]["metrics"]["top1_oracle_agreement"])
    z_aggregate = float(p4_arms["z_bottleneck"]["overlap_restricted_aggregate"]["metrics"]["top1_oracle_agreement"])
    p4_gate = {
        "global_ge_0p343243": bool(z_global >= float(P4_FIXED_GATES["global_min"])),
        "runtime_ge_0p400000": bool(z_runtime >= float(P4_FIXED_GATES["runtime_min"])),
        "aggregate_ge_0p300000": bool(z_aggregate >= float(P4_FIXED_GATES["aggregate_min"])),
    }
    p4_gate["pass_all"] = bool(all(p4_gate.values()))

    p4 = {
        "task": "P4 cross-clip entry retrieval agreement",
        "views": {
            "global": "agreement against P0 oracle shortlist (top-k rows from MM lexicographic ranking)",
            "overlap_restricted_runtime": "per (source_clip, source_frame, target_clip), top K target frames by overlap_cost",
            "overlap_restricted_aggregate": "bottom Q overlap region per ordered clip pair; dropped when query has no target candidate",
        },
        "mm_oracle_top_k_for_global": int(args.mm_oracle_top_k),
        "runtime_top_k": int(args.overlap_runtime_top_k),
        "aggregate_bottom_q": float(args.overlap_aggregate_q),
        "arms": p4_arms,
        "gate": {
            "z_thresholds": {
                "global_min": float(P4_FIXED_GATES["global_min"]),
                "runtime_topk_min": float(P4_FIXED_GATES["runtime_min"]),
                "aggregate_bottom_q_min": float(P4_FIXED_GATES["aggregate_min"]),
            },
            "z_metrics": {
                "global": z_global,
                "runtime_topk": z_runtime,
                "aggregate_bottom_q": z_aggregate,
            },
            "z_pass": p4_gate,
            "energy_baseline_reference": P4_GATE_BASELINE,
        },
        "aggregate_dropped_semantics": {
            "definition": "dropped is query-level (source_clip, source_frame, target_clip) with no target inside bottom-Q overlap region",
            "not_equivalent_to": "source frame having no candidate for all target clips",
            "runtime_behavior": "no_good_candidate_for_target / defer / extend / fallback, not force-pick j",
        },
        "mm_pair_stats": mm_pair_stats,
    }

    hidden_t = {c: torch.from_numpy(clip_data[c].hidden_pre).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}
    desc_t = {c: torch.from_numpy(clip_data[c].future_desc).to(device=device, dtype=torch.float32) for c in LOCKED_CLIPS}
    encoder.eval()
    pred_heads.eval()
    with torch.no_grad():
        z_by_clip_t = {c: encoder(hidden_t[c]) for c in LOCKED_CLIPS}
        internal_train, internal_train_terms = _weighted_huber_for_pred_heads(
            z_by_clip_t=z_by_clip_t,
            future_desc_by_clip_t=desc_t,
            pred_heads=pred_heads,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            subset="train",
            device=device,
        )
        internal_test, internal_test_terms = _weighted_huber_for_pred_heads(
            z_by_clip_t=z_by_clip_t,
            future_desc_by_clip_t=desc_t,
            pred_heads=pred_heads,
            horizons=horizons,
            weights_by_h=weights_by_h,
            split=split,
            subset="test",
            device=device,
        )

    internal_audit = {
        "internal_pred_head_train_weighted_huber": float(internal_train),
        "internal_pred_head_test_weighted_huber": float(internal_test),
        "matched_readout_train_weighted_huber": float(p1["arms"]["z_bottleneck"]["loss"]["train_weighted_huber"]),
        "matched_readout_test_weighted_huber": float(p1["arms"]["z_bottleneck"]["loss"]["test_weighted_huber"]),
        "queries": {
            "internal_train_terms": int(internal_train_terms),
            "internal_test_terms": int(internal_test_terms),
            "matched_readout_train_terms": int(p1["arms"]["z_bottleneck"]["queries"]["train_terms"]),
            "matched_readout_test_terms": int(p1["arms"]["z_bottleneck"]["queries"]["test_terms"]),
        },
    }

    config = {
        "beta": float(beta),
        "z_dim": int(z_dim),
        "seed": int(seed),
        "epochs": int(epochs),
        "readout_epochs": int(readout_epochs),
        "tau": float(args.tau),
        "horizons": [int(k) for k in horizons],
        "hidden_pre_feature_source": "hidden_pre captured from model._pasa_lnq input",
    }

    z_train_summary_payload = {
        "date": today,
        "checkpoint": ckpt_summary,
        "locked_clips": list(LOCKED_CLIPS),
        "feature_schema": shape_summary,
        "future_desc_meta": feature_meta["future_desc"],
        "energy_meta": feature_meta["energy"],
        "config": config,
        "z_training": z_train_summary,
        "internal_prediction_head_audit": internal_audit,
    }

    return {
        "config": config,
        "z_by_clip": z_by_clip,
        "z_train_summary_payload": z_train_summary_payload,
        "p1": p1,
        "p4": p4,
        "internal_audit": internal_audit,
        "z_losses": {
            "loss_total": float(z_train_summary["curve"]["loss_total_last"]),
            "loss_predict": float(z_train_summary["curve"]["loss_predict_last"]),
            "loss_nce": float(z_train_summary["curve"]["loss_nce_last"]),
        },
    }


def _write_run_outputs(
    *,
    out_dir: Path,
    run: dict[str, Any],
    clip_data: dict[str, ClipData],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = run["p1"]
    p4 = run["p4"]
    z_train_summary_payload = run["z_train_summary_payload"]
    z_by_clip = run["z_by_clip"]
    config = run["config"]
    internal = run["internal_audit"]

    _dump_json(out_dir / "z_train_summary.json", z_train_summary_payload)
    _dump_json(out_dir / "p1_predictive_compare.json", p1)
    _dump_json(out_dir / "p4_cross_clip_entry.json", p4)

    npz_payload: dict[str, Any] = {"clip_order": np.asarray(list(LOCKED_CLIPS), dtype=object)}
    for clip in LOCKED_CLIPS:
        npz_payload[f"{clip}__hidden_pre"] = clip_data[clip].hidden_pre.astype(np.float32, copy=False)
        npz_payload[f"{clip}__z"] = z_by_clip[clip].astype(np.float32, copy=False)
        npz_payload[f"{clip}__energy"] = clip_data[clip].energy.astype(np.float32, copy=False)
        npz_payload[f"{clip}__future_desc"] = clip_data[clip].future_desc.astype(np.float32, copy=False)
    np.savez_compressed(out_dir / "z_features_per_clip.npz", **npz_payload)

    z_metrics = p4["gate"]["z_metrics"]
    z_pass = p4["gate"]["z_pass"]
    md: list[str] = []
    md.append(f"# Action Handoff z Probe v1 ({z_train_summary_payload['date']})")
    md.append("")
    md.append("## Config")
    md.append(
        f"- beta={config['beta']}, z_dim={config['z_dim']}, seed={config['seed']}, "
        f"epochs={config['epochs']}, readout_epochs={config['readout_epochs']}, tau={config['tau']}"
    )
    md.append(f"- horizons={config['horizons']}")
    md.append(f"- hidden_pre feature source: {config['hidden_pre_feature_source']}")
    md.append("")
    md.append("## P1 Predictive Compare")
    for arm_name in ("energy_scalar", "raw_hidden_pre", "z_bottleneck"):
        arm = p1["arms"][arm_name]
        md.append(
            f"- {arm_name}: test_weighted_huber={arm['loss']['test_weighted_huber']:.6f}, "
            f"train_weighted_huber={arm['loss']['train_weighted_huber']:.6f}, "
            f"params={arm['readout_trainable_params']}, repr_dim={arm['repr_dim']}"
        )
        for clip in LOCKED_CLIPS:
            v = arm["per_clip"]["test_weighted_huber"][clip]
            md.append(f"  - {clip}: test_weighted_huber={float(v):.6f}")
    md.append("")
    md.append("## Internal Prediction-Head Audit")
    md.append(f"- internal_pred_head_train_weighted_huber={internal['internal_pred_head_train_weighted_huber']:.6f}")
    md.append(f"- internal_pred_head_test_weighted_huber={internal['internal_pred_head_test_weighted_huber']:.6f}")
    md.append(f"- matched_readout_train_weighted_huber={internal['matched_readout_train_weighted_huber']:.6f}")
    md.append(f"- matched_readout_test_weighted_huber={internal['matched_readout_test_weighted_huber']:.6f}")
    md.append("")
    md.append("## P4 Gates (z)")
    md.append(
        f"- global: z={z_metrics['global']:.6f} vs gate>={P4_FIXED_GATES['global_min']:.6f} -> {'PASS' if z_pass['global_ge_0p343243'] else 'FAIL'}"
    )
    md.append(
        f"- overlap_restricted_runtime(topK=5): z={z_metrics['runtime_topk']:.6f} vs gate>={P4_FIXED_GATES['runtime_min']:.6f} -> {'PASS' if z_pass['runtime_ge_0p400000'] else 'FAIL'}"
    )
    md.append(
        f"- overlap_restricted_aggregate(bottomQ=25%): z={z_metrics['aggregate_bottom_q']:.6f} vs gate>={P4_FIXED_GATES['aggregate_min']:.6f} -> {'PASS' if z_pass['aggregate_ge_0p300000'] else 'FAIL'}"
    )
    md.append(f"- overall: {'PASS' if z_pass['pass_all'] else 'FAIL'}")
    md.append("")
    md.append("## z Training Final Losses")
    md.append(f"- loss_total={run['z_losses']['loss_total']:.6f}")
    md.append(f"- loss_predict={run['z_losses']['loss_predict']:.6f}")
    md.append(f"- loss_nce={run['z_losses']['loss_nce']:.6f}")
    md.append("")
    md.append("## Artifacts")
    md.append(f"- `{(out_dir / 'z_train_summary.json').resolve()}`")
    md.append(f"- `{(out_dir / 'p1_predictive_compare.json').resolve()}`")
    md.append(f"- `{(out_dir / 'p4_cross_clip_entry.json').resolve()}`")
    _dump_md(out_dir / "summary.md", md)


def _summarize_config_for_ablation(*, config_id: str, out_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    p1 = run["p1"]
    p4 = run["p4"]
    z_metrics = p4["gate"]["z_metrics"]
    z_pass = p4["gate"]["z_pass"]
    return {
        "config_id": str(config_id),
        "config": dict(run["config"]),
        "paths": {
            "dir": str(out_dir.resolve()),
            "z_train_summary": str((out_dir / "z_train_summary.json").resolve()),
            "p1_predictive_compare": str((out_dir / "p1_predictive_compare.json").resolve()),
            "p4_cross_clip_entry": str((out_dir / "p4_cross_clip_entry.json").resolve()),
            "summary_md": str((out_dir / "summary.md").resolve()),
        },
        "p1": {
            "energy": {
                "test_weighted_huber": float(p1["arms"]["energy_scalar"]["loss"]["test_weighted_huber"]),
                "per_clip_test_weighted_huber": dict(p1["arms"]["energy_scalar"]["per_clip"]["test_weighted_huber"]),
            },
            "raw_hidden_pre": {
                "test_weighted_huber": float(p1["arms"]["raw_hidden_pre"]["loss"]["test_weighted_huber"]),
                "per_clip_test_weighted_huber": dict(p1["arms"]["raw_hidden_pre"]["per_clip"]["test_weighted_huber"]),
            },
            "z": {
                "test_weighted_huber": float(p1["arms"]["z_bottleneck"]["loss"]["test_weighted_huber"]),
                "per_clip_test_weighted_huber": dict(p1["arms"]["z_bottleneck"]["per_clip"]["test_weighted_huber"]),
            },
        },
        "p4_z": {
            "global": float(z_metrics["global"]),
            "overlap_restricted_runtime": float(z_metrics["runtime_topk"]),
            "overlap_restricted_aggregate": float(z_metrics["aggregate_bottom_q"]),
            "gate": {
                "global_ge_0p343243": bool(z_pass["global_ge_0p343243"]),
                "runtime_ge_0p400000": bool(z_pass["runtime_ge_0p400000"]),
                "aggregate_ge_0p300000": bool(z_pass["aggregate_ge_0p300000"]),
                "pass_all": bool(z_pass["pass_all"]),
            },
        },
        "internal_prediction_head_audit": dict(run["internal_audit"]),
        "z_training_final_losses": dict(run["z_losses"]),
    }


def _beta_seed_records(records: list[dict[str, Any]], *, seed: int, z_dim: int) -> list[dict[str, Any]]:
    out = [
        r
        for r in records
        if int(r["config"]["seed"]) == int(seed) and int(r["config"]["z_dim"]) == int(z_dim)
    ]
    return sorted(out, key=lambda x: float(x["config"]["beta"]))


def _dz_seed_records(records: list[dict[str, Any]], *, seed: int, beta: float) -> list[dict[str, Any]]:
    out = [
        r
        for r in records
        if int(r["config"]["seed"]) == int(seed) and abs(float(r["config"]["beta"]) - float(beta)) <= 1e-12
    ]
    return sorted(out, key=lambda x: int(x["config"]["z_dim"]))


def _build_ablation_summary_md(*, out_dir: Path, records: list[dict[str, Any]], focused_meta: dict[str, Any]) -> None:
    md: list[str] = []
    md.append("# Action Handoff z Probe v1 Focused Ablation Summary")
    md.append("")
    md.append(f"- artifact root: `{out_dir.resolve()}`")
    md.append(f"- total configs: {len(records)}")
    md.append("")

    beta_rows = _beta_seed_records(
        records,
        seed=int(focused_meta["primary_seed"]),
        z_dim=int(focused_meta["beta_sweep_z_dim"]),
    )
    dz_rows = _dz_seed_records(
        records,
        seed=int(focused_meta["primary_seed"]),
        beta=float(focused_meta["dz_sweep_beta"]),
    )

    md.append("## Beta Sweep (single seed)")
    md.append("| beta | z_dim | seed | P1 energy | P1 raw | P1 z | P4 global | P4 runtime | P4 aggregate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in beta_rows:
        md.append(
            "| "
            f"{r['config']['beta']:.6g} | {r['config']['z_dim']} | {r['config']['seed']} | "
            f"{r['p1']['energy']['test_weighted_huber']:.6f} | "
            f"{r['p1']['raw_hidden_pre']['test_weighted_huber']:.6f} | "
            f"{r['p1']['z']['test_weighted_huber']:.6f} | "
            f"{r['p4_z']['global']:.6f} | {r['p4_z']['overlap_restricted_runtime']:.6f} | "
            f"{r['p4_z']['overlap_restricted_aggregate']:.6f} |"
        )
    md.append("")

    md.append("## Dz Sweep (single seed)")
    md.append("| beta | z_dim | seed | P1 energy | P1 raw | P1 z | P4 global | P4 runtime | P4 aggregate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in dz_rows:
        md.append(
            "| "
            f"{r['config']['beta']:.6g} | {r['config']['z_dim']} | {r['config']['seed']} | "
            f"{r['p1']['energy']['test_weighted_huber']:.6f} | "
            f"{r['p1']['raw_hidden_pre']['test_weighted_huber']:.6f} | "
            f"{r['p1']['z']['test_weighted_huber']:.6f} | "
            f"{r['p4_z']['global']:.6f} | {r['p4_z']['overlap_restricted_runtime']:.6f} | "
            f"{r['p4_z']['overlap_restricted_aggregate']:.6f} |"
        )
    md.append("")

    all_seed0 = [r for r in records if int(r["config"]["seed"]) == int(focused_meta["primary_seed"])]
    if not all_seed0:
        raise RuntimeError("ablation summary needs at least one primary-seed record")
    best = min(all_seed0, key=lambda x: float(x["p1"]["z"]["test_weighted_huber"]))
    best_z = float(best["p1"]["z"]["test_weighted_huber"])
    best_energy = float(best["p1"]["energy"]["test_weighted_huber"])
    energy_ref = float(P4_FIXED_GATES["global_min"])
    _ = energy_ref

    base_seed0 = None
    for r in all_seed0:
        if abs(float(r["config"]["beta"]) - 0.25) <= 1e-12 and int(r["config"]["z_dim"]) == 32:
            base_seed0 = r
            break
    if base_seed0 is None:
        base_seed0 = min(all_seed0, key=lambda x: float(x["p1"]["z"]["test_weighted_huber"]))

    beta_min = min(beta_rows, key=lambda x: float(x["p1"]["z"]["test_weighted_huber"])) if beta_rows else None
    dz_min = min(dz_rows, key=lambda x: float(x["p1"]["z"]["test_weighted_huber"])) if dz_rows else None
    baseline_energy = float(base_seed0["p1"]["energy"]["test_weighted_huber"])
    baseline_raw = float(base_seed0["p1"]["raw_hidden_pre"]["test_weighted_huber"])
    baseline_z = float(base_seed0["p1"]["z"]["test_weighted_huber"])

    md.append("## Decision Logic")
    if beta_min and float(beta_min["p1"]["z"]["test_weighted_huber"]) < baseline_z:
        md.append(
            f"- beta↓ improves P1 from {baseline_z:.6f} to {float(beta_min['p1']['z']['test_weighted_huber']):.6f}; likely root cause: InfoNCE domination."
        )
    else:
        md.append("- beta sweep did not improve P1 over baseline; no evidence for pure InfoNCE domination.")

    if dz_min and int(dz_min["config"]["z_dim"]) > 32 and float(dz_min["p1"]["z"]["test_weighted_huber"]) < baseline_z:
        md.append(
            f"- Dz↑ improves P1 from {baseline_z:.6f} to {float(dz_min['p1']['z']['test_weighted_huber']):.6f}; likely root cause: bottleneck too tight."
        )
    else:
        md.append("- Dz sweep did not show clear gain vs baseline; weak evidence for bottleneck-only issue.")

    internal_best = best["internal_prediction_head_audit"]
    ipk_test = float(internal_best["internal_pred_head_test_weighted_huber"])
    mr_test = float(internal_best["matched_readout_test_weighted_huber"])
    if ipk_test <= baseline_energy * 1.2 and mr_test >= ipk_test * 1.5:
        md.append(
            "- internal P_k test loss is low but matched readout test loss is high; likely issue: matched readout instability / evaluation noise."
        )
    elif ipk_test >= baseline_energy * 1.5 and mr_test >= baseline_energy * 1.5:
        md.append(
            "- internal P_k and matched readout are both high; likely issue: z encoder/objective discards predictive information."
        )
    else:
        md.append(
            "- internal P_k vs matched readout gap is mixed; no single dominant failure mode from this heuristic alone."
        )

    near_energy = best_z <= (baseline_energy + 5e-4)
    if not near_energy:
        md.append(
            "- beta + Dz cannot bring z P1 near energy baseline; do not proceed to P4/P5/P6, recommend fallback discussion for magnitude-preserving scoring."
        )
    else:
        md.append("- beta + Dz brings z P1 near/below energy baseline; P1 recovery is plausible for next-stage gating.")

    md.append("")
    md.append("## Best Primary-Seed Config")
    md.append(
        f"- config_id={best['config_id']}, beta={best['config']['beta']}, z_dim={best['config']['z_dim']}, seed={best['config']['seed']}"
    )
    md.append(
        f"- P1 z={best_z:.6f}, energy={best_energy:.6f}, raw={float(best['p1']['raw_hidden_pre']['test_weighted_huber']):.6f}"
    )
    md.append(
        f"- P4 z: global={best['p4_z']['global']:.6f}, runtime={best['p4_z']['overlap_restricted_runtime']:.6f}, aggregate={best['p4_z']['overlap_restricted_aggregate']:.6f}"
    )

    _dump_md(out_dir / "ablation_summary.md", md)


def _load_ablation_configs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        cfgs = payload.get("configs")
        if not isinstance(cfgs, list):
            raise RuntimeError("ablation json dict must contain list field `configs`")
    elif isinstance(payload, list):
        cfgs = payload
    else:
        raise RuntimeError("ablation json must be list or dict with `configs`")
    out: list[dict[str, Any]] = []
    for row in cfgs:
        if not isinstance(row, dict):
            raise RuntimeError("each ablation config must be an object")
        out.append(dict(row))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="v1 predictive-contrastive z probe for action handoff (probe-only, no training/posttrain entry changes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--teacher-root", type=str, default=DEFAULT_TEACHER_ROOT)
    ap.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    ap.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    ap.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    ap.add_argument("--seed", type=int, default=20260524)

    ap.add_argument("--horizons", type=str, default="1,3,6,12,24")
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--easy-negs", type=int, default=32)
    ap.add_argument("--hard-negs", type=int, default=16)
    ap.add_argument("--neg-same-window-radius", type=int, default=24)

    ap.add_argument("--readout-epochs", type=int, default=80)
    ap.add_argument("--readout-lr", type=float, default=3e-4)
    ap.add_argument("--readout-weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-ratio", type=float, default=0.7)

    ap.add_argument("--mm-oracle-top-k", type=int, default=3)
    ap.add_argument("--overlap-runtime-top-k", type=int, default=5)
    ap.add_argument("--overlap-aggregate-q", type=float, default=0.25)

    ap.add_argument("--max-frames-per-clip", type=int, default=None)

    ap.add_argument("--ablation-json", type=str, default=None)
    ap.add_argument("--sweep-mode", type=str, default="none", choices=("none", "focused"))
    ap.add_argument("--ablation-top-k", type=int, default=3)
    ap.add_argument("--ablation-top-seeds", type=str, default="0,1,2")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    today = datetime.now().strftime("%Y%m%d")

    horizons = _parse_csv_ints(str(args.horizons))
    if any(int(k) <= 0 for k in horizons):
        raise RuntimeError(f"horizons must be positive, got {horizons}")

    is_ablation = bool(args.ablation_json) or str(args.sweep_mode) != "none"
    default_out = (
        f"_tmp_action_handoff_z_probe_v1_ablation_{today}" if is_ablation else f"_tmp_action_handoff_z_probe_v1_{today}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo_root / "debug_output" / default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_data, ckpt_summary, shape_summary, feature_meta = _build_clip_data(args=args, repo_root=repo_root)
    device = torch.device("cpu" if str(args.device) == "auto" else str(args.device))

    if not is_ablation:
        run = _build_run_artifacts(
            clip_data=clip_data,
            ckpt_summary=ckpt_summary,
            shape_summary=shape_summary,
            feature_meta=feature_meta,
            horizons=horizons,
            args=args,
            device=device,
            beta=float(args.beta),
            z_dim=int(args.z_dim),
            seed=int(args.seed),
            epochs=int(args.epochs),
            readout_epochs=int(args.readout_epochs),
            today=today,
        )
        _write_run_outputs(out_dir=out_dir, run=run, clip_data=clip_data)
        print(f"[OK] wrote artifacts under {out_dir}")
        return 0

    per_cfg_root = out_dir / "per_config"
    per_cfg_root.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[float, int, int, int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    def _run_and_record(beta: float, z_dim: int, seed: int, epochs: int, readout_epochs: int) -> dict[str, Any]:
        key = (float(beta), int(z_dim), int(seed), int(epochs), int(readout_epochs))
        cfg_id = _build_config_id(
            beta=float(beta),
            z_dim=int(z_dim),
            seed=int(seed),
            epochs=int(epochs),
            readout_epochs=int(readout_epochs),
        )
        cfg_dir = per_cfg_root / cfg_id
        if key in cache:
            return cache[key]
        run = _build_run_artifacts(
            clip_data=clip_data,
            ckpt_summary=ckpt_summary,
            shape_summary=shape_summary,
            feature_meta=feature_meta,
            horizons=horizons,
            args=args,
            device=device,
            beta=float(beta),
            z_dim=int(z_dim),
            seed=int(seed),
            epochs=int(epochs),
            readout_epochs=int(readout_epochs),
            today=today,
        )
        _write_run_outputs(out_dir=cfg_dir, run=run, clip_data=clip_data)
        rec = _summarize_config_for_ablation(config_id=cfg_id, out_dir=cfg_dir, run=run)
        records.append(rec)
        cache[key] = rec
        return rec

    primary_seed = int(args.seed)
    focused_meta = {
        "plan": None,
        "primary_seed": int(primary_seed),
        "beta_sweep_z_dim": 32,
        "dz_sweep_beta": 0.25,
        "top_k": int(max(2, min(3, int(args.ablation_top_k)))),
        "top_seed_list": _parse_csv_ints(str(args.ablation_top_seeds)),
    }

    if args.ablation_json:
        cfgs = _load_ablation_configs(Path(args.ablation_json).expanduser().resolve())
        focused_meta["plan"] = "ablation_json"
        for row in cfgs:
            _run_and_record(
                beta=float(row.get("beta", args.beta)),
                z_dim=int(row.get("z_dim", args.z_dim)),
                seed=int(row.get("seed", args.seed)),
                epochs=int(row.get("epochs", args.epochs)),
                readout_epochs=int(row.get("readout_epochs", args.readout_epochs)),
            )
    else:
        focused_meta["plan"] = "focused"
        for b in DEFAULT_ABLATION_BETA_SWEEP:
            _run_and_record(
                beta=float(b),
                z_dim=int(focused_meta["beta_sweep_z_dim"]),
                seed=int(primary_seed),
                epochs=int(args.epochs),
                readout_epochs=int(args.readout_epochs),
            )
        for dz in DEFAULT_ABLATION_ZDIM_SWEEP:
            _run_and_record(
                beta=float(focused_meta["dz_sweep_beta"]),
                z_dim=int(dz),
                seed=int(primary_seed),
                epochs=int(args.epochs),
                readout_epochs=int(args.readout_epochs),
            )

        seed0_records = [r for r in records if int(r["config"]["seed"]) == int(primary_seed)]
        top_cfg = sorted(seed0_records, key=lambda r: float(r["p1"]["z"]["test_weighted_huber"]))[: int(focused_meta["top_k"])]
        focused_meta["top_config_ids"] = [str(r["config_id"]) for r in top_cfg]
        for r in top_cfg:
            beta_v = float(r["config"]["beta"])
            dz_v = int(r["config"]["z_dim"])
            for s in focused_meta["top_seed_list"]:
                _run_and_record(
                    beta=beta_v,
                    z_dim=dz_v,
                    seed=int(s),
                    epochs=int(args.epochs),
                    readout_epochs=int(args.readout_epochs),
                )

    records_sorted = sorted(
        records,
        key=lambda r: (
            float(r["config"]["beta"]),
            int(r["config"]["z_dim"]),
            int(r["config"]["seed"]),
            int(r["config"]["epochs"]),
            int(r["config"]["readout_epochs"]),
        ),
    )
    ablation_summary = {
        "date": today,
        "artifact_root": str(out_dir.resolve()),
        "fixed_p4_gates": {
            "global_min": float(P4_FIXED_GATES["global_min"]),
            "runtime_min": float(P4_FIXED_GATES["runtime_min"]),
            "aggregate_min": float(P4_FIXED_GATES["aggregate_min"]),
        },
        "focused_meta": focused_meta,
        "records": records_sorted,
    }
    _dump_json(out_dir / "ablation_summary.json", ablation_summary)
    _build_ablation_summary_md(out_dir=out_dir, records=records_sorted, focused_meta=focused_meta)
    print(f"[OK] wrote ablation artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
