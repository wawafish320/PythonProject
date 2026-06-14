#!/usr/bin/env python3
"""Oracle-schedule-conditioned layer-2 trajectory decoder smoke.

Temporary debug tool only. This trains small deterministic CPU MLP decoders over
existing continuous target windows. It does not use production Trainer, does not
edit or forward production runtime/gate, does not mutate checkpoints, and does
not package a production generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    FPS,
    POSE_SLICE,
    STATE_DIM,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    CONTACT_THRESHOLD,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _evaluate_sequence,
    _fmt,
    _foot_positions,
    _heading_error_rad,
    _load_clips,
    _load_skeleton_meta,
    _rms_rows,
    _safe_percentile,
    _step_angvel_component_p95,
    _step_angvel_rms,
    _step_l2,
    _step_pose_l2,
)
from train.geometry import fk_positions_from_rot6d, rot6d_to_matrix  # noqa: E402
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    LABEL_ORDER,
    _calibrate_support_side_bands,
    _context_window,
    _evaluate_support_side_correctness,
    _feature_bands,
    _make_sequence,
    _support_contract,
    _support_side_features,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
    SplitDef,
    UNMATCHED_TARGET,
    _build_splits,
)


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_oracle_schedule_trajectory_decoder_smoke_20260602")
EPS = 1e-8
INSTRUMENTED_TERM_KEYS = (
    ("state_mse", "state_raw_mse"),
    ("flat_state", "flat_standardized"),
    ("foot_vel", "contacted_foot_velocity"),
    ("root_pos", "root_pos"),
    ("command", "command_response"),
    ("pose_step", "pose_step"),
    ("aux", "aux_bone_angvel"),
)


@dataclass
class DecoderItem:
    clip: str
    start: int
    end: int
    seq: Dict[str, np.ndarray]
    ctx: np.ndarray
    support_contract: Dict[str, Any]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    constant_count: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return (arr * self.std + self.mean).astype(np.float32)


class TinyDeterministicDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _finite_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)


def _fit_standardizer(x: np.ndarray) -> Standardizer:
    arr = np.asarray(x, dtype=np.float32)
    mean = np.mean(arr, axis=0, keepdims=True).astype(np.float32)
    std = np.std(arr, axis=0, keepdims=True).astype(np.float32)
    keep = std > 1e-6
    std = np.where(keep, std, 1.0).astype(np.float32)
    return Standardizer(mean=mean, std=std, constant_count=int(np.sum(~keep)))


def _support_labels(contact: np.ndarray) -> List[str]:
    labels = _support_contract(np.asarray(contact, dtype=np.float32), min_run_frames=2)["normalized_label_sequence"]
    return [str(x) for x in labels]


def _one_hot_labels(labels: Sequence[str]) -> np.ndarray:
    rows = []
    for label in labels:
        rows.append([1.0 if str(label) == ref else 0.0 for ref in LABEL_ORDER])
    return np.asarray(rows, dtype=np.float32)


def _run_phase_features(labels: Sequence[str]) -> np.ndarray:
    labels = [str(x) for x in labels]
    n = len(labels)
    out = np.zeros((n, 2), dtype=np.float32)
    start = 0
    while start < n:
        end = start + 1
        while end < n and labels[end] == labels[start]:
            end += 1
        length = max(1, end - start)
        for j in range(start, end):
            phase = (j - start) / max(1, length - 1)
            out[j, 0] = math.sin(2.0 * math.pi * phase)
            out[j, 1] = math.cos(2.0 * math.pi * phase)
        start = end
    return out


def _support_stats(labels: Sequence[str]) -> np.ndarray:
    labels = [str(x) for x in labels]
    n = max(1, len(labels))
    transitions = sum(a != b for a, b in zip(labels[:-1], labels[1:]))
    counts = Counter(labels)
    probs = np.asarray([c / n for c in counts.values()], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log2(np.maximum(probs, EPS)))) if probs.size else 0.0
    return np.asarray(
        [
            sum(x in {"right", "dual"} for x in labels) / n,
            sum(x in {"left", "dual"} for x in labels) / n,
            counts.get("dual", 0) / n,
            counts.get("flight_or_unknown", 0) / n,
            transitions / max(1, len(labels) - 1),
            entropy,
        ],
        dtype=np.float32,
    )


def _build_items(
    clips: Mapping[str, Any],
    *,
    horizon: int,
    context_len: int,
    min_run_frames: int,
    stride: int,
) -> List[DecoderItem]:
    items: List[DecoderItem] = []
    for name in TURN_CLIPS:
        clip = clips[name]
        max_start = int(clip.state281.shape[0]) - int(horizon)
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, int(stride))):
            seq = _make_sequence(clip, start, horizon)
            ctx = _context_window(clip, start, context_len, wrap=(name == WALK_F))
            contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
            items.append(
                DecoderItem(
                    clip=name,
                    start=int(start),
                    end=int(start + horizon - 1),
                    seq=seq,
                    ctx=ctx,
                    support_contract=contract,
                )
            )
    return items


def _feature_from_item(item: DecoderItem) -> np.ndarray:
    seq = item.seq
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    endpoint_state_no_contact = np.asarray(seq["state281"][-1, : CONTACT_SLICE.start], dtype=np.float32)
    parts = [
        np.asarray(item.ctx, dtype=np.float32).reshape(-1),
        np.asarray(seq["contact"], dtype=np.float32).reshape(-1),
        np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1),
        np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1),
        endpoint_state_no_contact.reshape(-1),
        _one_hot_labels(labels).reshape(-1),
        _run_phase_features(labels).reshape(-1),
        _support_stats(labels).reshape(-1),
    ]
    return _finite_float32(np.concatenate(parts, axis=0))


def _target_from_item(item: DecoderItem) -> np.ndarray:
    state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(-1)
    aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(-1)
    return _finite_float32(np.concatenate([state, aux], axis=0))


def _dataset_arrays(items: Sequence[DecoderItem], idxs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([_feature_from_item(items[int(i)]) for i in idxs], axis=0).astype(np.float32, copy=False)
    y = np.stack([_target_from_item(items[int(i)]) for i in idxs], axis=0).astype(np.float32, copy=False)
    return x, y


def _reshape_state_aux(y: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y, dtype=np.float32)
    state_w = int(horizon) * STATE_DIM
    state = arr[:, :state_w].reshape(arr.shape[0], int(horizon), STATE_DIM)
    aux = arr[:, state_w:].reshape(arr.shape[0], int(horizon), ANGVEL_DIM)
    return state, aux


def _loss_metrics(y_pred: np.ndarray, y_true: np.ndarray, horizon: int) -> Dict[str, float]:
    pred_state, pred_aux = _reshape_state_aux(y_pred, horizon)
    true_state, true_aux = _reshape_state_aux(y_true, horizon)

    def mse(a: np.ndarray, b: np.ndarray) -> float:
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.mean(d * d)) if d.size else 0.0

    token_acc = []
    for p, t in zip(pred_state, true_state):
        pred_labels = _support_labels(p[:, CONTACT_SLICE])
        true_labels = _support_labels(t[:, CONTACT_SLICE])
        n = min(len(pred_labels), len(true_labels))
        token_acc.append(float(np.mean([pred_labels[i] == true_labels[i] for i in range(n)])) if n else 0.0)

    return {
        "state_mse": mse(pred_state, true_state),
        "pose_rot6d_mse": mse(pred_state[:, :, POSE_SLICE], true_state[:, :, POSE_SLICE]),
        "ego_vel_mse": mse(pred_state[:, :, EGO_VEL_SLICE], true_state[:, :, EGO_VEL_SLICE]),
        "yaw_rate_mse": mse(pred_state[:, :, YAW_RATE_SLICE], true_state[:, :, YAW_RATE_SLICE]),
        "contact_mse": mse(pred_state[:, :, CONTACT_SLICE], true_state[:, :, CONTACT_SLICE]),
        "bone_angvel_aux_mse": mse(pred_aux, true_aux),
        "support_token_accuracy": float(np.mean(token_acc)) if token_acc else 0.0,
        "contact_range_violation_fraction": float(
            np.mean((pred_state[:, :, CONTACT_SLICE] < -1e-4) | (pred_state[:, :, CONTACT_SLICE] > 1.0 + 1e-4))
        ),
        "finite": float(np.mean(np.isfinite(y_pred))),
    }


def _world_root_vel_from_ego(
    ego_vel: np.ndarray,
    cond_dir: np.ndarray,
    *,
    command_align_root_vel: bool,
) -> np.ndarray:
    ego = np.asarray(ego_vel, dtype=np.float32).reshape(-1, 2)
    cmd = np.asarray(cond_dir, dtype=np.float32).reshape(-1, 2)
    n = min(ego.shape[0], cmd.shape[0])
    ego = ego[:n]
    cmd = cmd[:n]
    norm = np.maximum(np.linalg.norm(cmd, axis=1, keepdims=True), EPS)
    fwd = cmd / norm
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    lateral = 0.0 if command_align_root_vel else ego[:, 1:2]
    return (ego[:, 0:1] * fwd + lateral * lat).astype(np.float32)


def _world_root_vel_from_ego_torch(
    ego_vel: torch.Tensor,
    cond_dir: torch.Tensor,
    *,
    command_align_root_vel: bool,
) -> torch.Tensor:
    norm = torch.linalg.norm(cond_dir, dim=-1, keepdim=True).clamp_min(EPS)
    fwd = cond_dir / norm
    lat = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
    lateral = torch.zeros_like(ego_vel[..., 1:2]) if command_align_root_vel else ego_vel[..., 1:2]
    return ego_vel[..., 0:1] * fwd + lateral * lat


def _integrate_root_pos(root_vel: np.ndarray, start_root: np.ndarray) -> np.ndarray:
    vel = np.asarray(root_vel, dtype=np.float32).reshape(-1, 2)
    root0 = np.asarray(start_root, dtype=np.float32).reshape(3)
    out = np.zeros((vel.shape[0], 3), dtype=np.float32)
    out[0] = root0
    for i in range(1, vel.shape[0]):
        out[i, :2] = out[i - 1, :2] + vel[i - 1] / float(FPS)
        out[i, 2] = root0[2]
    return out


def _integrate_root_pos_torch(root_vel: torch.Tensor, start_root: torch.Tensor) -> torch.Tensor:
    b, h, _ = root_vel.shape
    out = root_vel.new_zeros((b, h, 3))
    out[:, 0, :] = start_root
    if h > 1:
        steps = root_vel[:, :-1, :] / float(FPS)
        xy = torch.cumsum(steps, dim=1)
        out[:, 1:, :2] = start_root[:, None, :2] + xy
        out[:, 1:, 2] = start_root[:, None, 2]
    return out


def _state_with_oracle_contact(item: DecoderItem, state: np.ndarray) -> np.ndarray:
    pred_state = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM).copy()
    pred_state[:, CONTACT_SLICE] = np.asarray(item.seq["contact"], dtype=np.float32).reshape(
        pred_state.shape[0], 2
    )
    return pred_state


def _apply_oracle_contact_passthrough(
    y_pred: np.ndarray,
    items: Sequence[DecoderItem],
    idxs: Sequence[int],
    horizon: int,
) -> np.ndarray:
    out = np.asarray(y_pred, dtype=np.float32).copy()
    state, _ = _reshape_state_aux(out, horizon)
    for local_i, item_idx in enumerate(idxs):
        state[local_i, :, CONTACT_SLICE] = np.asarray(items[int(item_idx)].seq["contact"], dtype=np.float32).reshape(
            int(horizon), 2
        )
    return out


def _seq_from_prediction(
    item: DecoderItem,
    state: np.ndarray,
    aux: np.ndarray,
    *,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, np.ndarray]:
    if oracle_contact_passthrough:
        pred_state = _state_with_oracle_contact(item, state)
    else:
        pred_state = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    cond_dir = np.asarray(item.seq["cond_dir"], dtype=np.float32).reshape(pred_state.shape[0], 2)
    root_vel = _world_root_vel_from_ego(
        pred_state[:, EGO_VEL_SLICE],
        cond_dir,
        command_align_root_vel=command_align_root_vel,
    )
    return {
        "rot6d": pred_state[:, POSE_SLICE].astype(np.float32, copy=False),
        "root_pos": _integrate_root_pos(root_vel, np.asarray(item.seq["root_pos"], dtype=np.float32)[0]),
        "root_vel": root_vel,
        "bone_angvel": np.asarray(aux, dtype=np.float32).reshape(pred_state.shape[0], ANGVEL_DIM),
        "cond_dir": cond_dir,
        "contact": pred_state[:, CONTACT_SLICE].astype(np.float32, copy=False),
        "yaw_rate": pred_state[:, YAW_RATE_SLICE].reshape(-1).astype(np.float32, copy=False),
    }


def _endpoint_bridgeability_proxy(item: DecoderItem, pred_contract: Mapping[str, Any]) -> bool:
    pred = [str(x) for x in pred_contract.get("normalized_label_sequence", [])]
    oracle = [str(x) for x in item.support_contract.get("normalized_label_sequence", [])]
    if not pred or not oracle:
        return False
    return bool(pred[0] == oracle[0] and pred[-1] == oracle[-1])


def _evaluate_seq_common(
    *,
    variant: str,
    split: str,
    split_kind: str,
    partition: str,
    item: DecoderItem,
    seq: Mapping[str, np.ndarray],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    endpoint_note: str,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
    calibration_domain: str,
) -> Dict[str, Any]:
    pred_contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
    endpoint_ok = _endpoint_bridgeability_proxy(item, pred_contract)
    row = _evaluate_sequence(
        seq,
        target=item.clip,
        target_bands=baseline_bands[item.clip],
        skeleton=skeleton,
        case="oracle_schedule_decoder_smoke:predicted_middle",
        expected_label="diagnostic",
        start_phase=f"{item.clip}:{item.start}-{item.end}",
        endpoint_bridgeability=endpoint_ok,
        endpoint_details={
            "endpoint_bridgeability_proxy": endpoint_note,
            "oracle_contact_passthrough": bool(oracle_contact_passthrough),
            "command_align_root_vel": bool(command_align_root_vel),
            "calibration_domain": str(calibration_domain),
        },
    )
    foot = _foot_positions(seq["rot6d"], seq["root_pos"], skeleton)
    side_features = _support_side_features(seq, pred_contract["normalized_label_sequence"], foot)
    side_ok, side_failures = _evaluate_support_side_correctness(
        side_features,
        support_bands[item.clip]["feature_bands"],
    )
    families = [
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
    ]
    row["support_side_correctness"] = bool(side_ok)
    failed = [name for name in families if not bool(row.get(name, False))]
    row["acceptance_proxy_pass"] = bool(not failed)
    row["pass"] = bool(not failed)
    row["failed_family"] = ",".join(failed)
    row["split"] = split
    row["split_kind"] = split_kind
    row["partition"] = partition
    row["variant"] = variant
    row["calibration_domain"] = str(calibration_domain)
    row["clip"] = item.clip
    row["start"] = int(item.start)
    row["end"] = int(item.end)
    row["support_side_failure_count"] = int(len(side_failures))
    row["support_side_failures"] = [dict(x) for x in side_failures[:8]]
    oracle_labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    pred_labels = [str(x) for x in pred_contract["normalized_label_sequence"]]
    n = min(len(oracle_labels), len(pred_labels))
    row["oracle_support_token_accuracy"] = (
        float(np.mean([oracle_labels[i] == pred_labels[i] for i in range(n)])) if n else 0.0
    )
    row["oracle_contact_mse"] = float(
        np.mean((np.asarray(seq["contact"], dtype=np.float64) - np.asarray(item.seq["contact"], dtype=np.float64)) ** 2)
    )
    foot_thr = float(baseline_bands[item.clip].get("foot_slip_contacted_speed_mps", 0.0) or 0.0)
    foot_p95 = float(row.get("metrics", {}).get("foot_slip_p95_mps", 0.0) or 0.0)
    row["foot_slip_p95_to_band_ratio"] = float(foot_p95 / max(foot_thr, EPS))
    row["old_pop_safe_diagnostic"] = bool(row.get("pose_continuity", False) and row.get("rate_budget", False))
    row["predicted_support_start"] = pred_labels[0] if pred_labels else None
    row["predicted_support_end"] = pred_labels[-1] if pred_labels else None
    row["oracle_support_start"] = oracle_labels[0] if oracle_labels else None
    row["oracle_support_end"] = oracle_labels[-1] if oracle_labels else None
    return row


def _evaluate_prediction(
    *,
    variant: str,
    split: str,
    split_kind: str,
    partition: str,
    item: DecoderItem,
    state: np.ndarray,
    aux: np.ndarray,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
    calibration_domain: str,
) -> Dict[str, Any]:
    seq = _seq_from_prediction(
        item,
        state,
        aux,
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
    )
    return _evaluate_seq_common(
        variant=variant,
        split=split,
        split_kind=split_kind,
        partition=partition,
        item=item,
        seq=seq,
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=min_run_frames,
        endpoint_note="predicted normalized support first/last matches oracle schedule first/last",
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
        calibration_domain=calibration_domain,
    )


def _reconstructed_gt_seq(
    item: DecoderItem,
    *,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, np.ndarray]:
    raw = _target_from_item(item).reshape(1, -1)
    state, aux = _reshape_state_aux(raw, int(item.seq["state281"].shape[0]))
    return _seq_from_prediction(
        item,
        state[0],
        aux[0],
        oracle_contact_passthrough=oracle_contact_passthrough,
        command_align_root_vel=command_align_root_vel,
    )


def _contacted_speed_pool_from_seq(seq: Mapping[str, np.ndarray], skeleton: Any) -> np.ndarray:
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    foot = _foot_positions(
        np.asarray(seq["rot6d"], dtype=np.float32),
        np.asarray(seq["root_pos"], dtype=np.float32),
        skeleton,
    )
    if foot is None:
        return np.zeros((0,), dtype=np.float64)
    vals: List[float] = []
    for ch_idx, side in ((0, "right"), (1, "left")):
        if side not in foot or contact.shape[1] <= ch_idx:
            continue
        pos = np.asarray(foot[side], dtype=np.float32).reshape(-1, 3)
        mask = (contact[:-1, ch_idx] > CONTACT_THRESHOLD) & (contact[1:, ch_idx] > CONTACT_THRESHOLD)
        speed = np.linalg.norm(pos[1:] - pos[:-1], axis=1) * float(FPS)
        vals.extend(float(x) for x in speed[mask].tolist())
    return np.asarray(vals, dtype=np.float64)


def _seq_baseline_metrics(seq: Mapping[str, np.ndarray], skeleton: Any) -> Dict[str, float]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_SLICE.stop - POSE_SLICE.start)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)
    speeds = _contacted_speed_pool_from_seq(seq, skeleton)
    foot_p95 = _safe_percentile(speeds, 95.0)
    return {
        "pose_step_l2": _safe_percentile(_step_pose_l2(rot6d), 95.0),
        "angvel_step_rms": _safe_percentile(_step_angvel_rms(bone_angvel), 95.0),
        "angvel_step_component_p95": _safe_percentile(_step_angvel_component_p95(bone_angvel), 95.0),
        "rootvel_step_l2": _safe_percentile(_step_l2(root_vel), 95.0),
        "yaw_rate_step_abs": _safe_percentile(np.abs(np.diff(yaw_rate)), 95.0),
        "contact_step_l2": _safe_percentile(_step_l2(contact), 95.0),
        "heading_error_rad": _safe_percentile(_heading_error_rad(root_vel, cond_dir), 95.0),
        "foot_slip_contacted_speed_p95_mps": foot_p95,
        "foot_slip_contacted_speed_mps": foot_p95,
        "foot_speed_sample_count": float(speeds.size),
    }


def _calibrate_reconstructed_baseline_bands(
    items: Sequence[DecoderItem],
    skeleton: Any,
    *,
    quantile: float,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, Dict[str, Any]]:
    by_clip: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
    for item in items:
        by_clip[item.clip].append(
            _reconstructed_gt_seq(
                item,
                oracle_contact_passthrough=oracle_contact_passthrough,
                command_align_root_vel=command_align_root_vel,
            )
        )

    out: Dict[str, Dict[str, Any]] = {}
    metric_keys = (
        "pose_step_l2",
        "angvel_step_rms",
        "angvel_step_component_p95",
        "rootvel_step_l2",
        "yaw_rate_step_abs",
        "contact_step_l2",
        "heading_error_rad",
        "foot_slip_contacted_speed_p95_mps",
        "foot_slip_contacted_speed_mps",
    )
    for clip, seqs in by_clip.items():
        metrics = [_seq_baseline_metrics(seq, skeleton) for seq in seqs]
        if seqs:
            bone_frames = np.concatenate(
                [np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM) for seq in seqs],
                axis=0,
            )
            level_center = np.mean(bone_frames, axis=0).astype(np.float32)
            endpoint_bone = np.stack(
                [np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)[-1] for seq in seqs],
                axis=0,
            )
            level_dist = _rms_rows(endpoint_bone - level_center.reshape(1, -1))
        else:
            level_center = np.zeros((ANGVEL_DIM,), dtype=np.float32)
            level_dist = np.zeros((0,), dtype=np.float64)
        rec: Dict[str, Any] = {
            "quantile": float(quantile),
            "calibration_domain": "reconstructed_state281",
            "n_windows": int(len(seqs)),
            "bone_angvel_level_rms": _safe_percentile(level_dist, quantile),
            "bone_angvel_level_center": level_center,
            "foot_slip": {
                "contacted_speed_p95_mps": _safe_percentile(
                    np.asarray([m["foot_slip_contacted_speed_p95_mps"] for m in metrics], dtype=np.float64),
                    quantile,
                ),
            },
        }
        for key in metric_keys:
            rec[key] = _safe_percentile(
                np.asarray([float(m.get(key, 0.0)) for m in metrics], dtype=np.float64),
                quantile,
            )
        out[clip] = rec
    return out


def _calibrate_reconstructed_support_side_bands(
    items: Sequence[DecoderItem],
    skeleton: Any,
    *,
    horizon: int,
    min_run_frames: int,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> Dict[str, Dict[str, Any]]:
    rows_by_clip: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for item in items:
        seq = _reconstructed_gt_seq(
            item,
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
        )
        contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
        foot = _foot_positions(seq["rot6d"], seq["root_pos"], skeleton)
        rows_by_clip[item.clip].append(_support_side_features(seq, contract["normalized_label_sequence"], foot))
    return {
        clip: {
            "horizon": int(horizon),
            "n_windows": int(len(rows)),
            "feature_bands": _feature_bands(rows),
            "calibration_domain": "reconstructed_state281",
            "band_rule": "inclusive min/max over GT state281 reconstructed through smoke path",
        }
        for clip, rows in rows_by_clip.items()
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    def rate(key: str) -> float:
        return float(np.mean([bool(r.get(key, False)) for r in rows]))

    def mean_metric(key: str) -> float:
        vals = []
        for r in rows:
            metrics = r.get("metrics", {}) or {}
            vals.append(float(r.get(key, metrics.get(key, 0.0)) or 0.0))
        return float(np.mean(vals)) if vals else 0.0

    failed = Counter()
    for row in rows:
        for fam in str(row.get("failed_family") or "").split(","):
            if fam:
                failed[fam] += 1
    return {
        "n": int(len(rows)),
        "acceptance_proxy_pass_rate": rate("acceptance_proxy_pass"),
        "regime_reached_pass_rate": rate("regime_reached"),
        "rate_budget_pass_rate": rate("rate_budget"),
        "support_honesty_pass_rate": rate("support_honesty"),
        "support_side_correctness_pass_rate": rate("support_side_correctness"),
        "command_response_pass_rate": rate("command_response"),
        "pose_continuity_pass_rate": rate("pose_continuity"),
        "endpoint_bridgeability_pass_rate": rate("endpoint_bridgeability"),
        "old_pop_safe_diagnostic_pass_rate": rate("old_pop_safe_diagnostic"),
        "oracle_support_token_accuracy_mean": mean_metric("oracle_support_token_accuracy"),
        "oracle_contact_mse_mean": mean_metric("oracle_contact_mse"),
        "foot_slip_p95_mps_mean": mean_metric("foot_slip_p95_mps"),
        "foot_slip_p95_to_band_ratio_mean": mean_metric("foot_slip_p95_to_band_ratio"),
        "heading_error_p95_rad_mean": mean_metric("heading_error_p95_rad"),
        "failed_family_counts": dict(failed),
    }


def _predict_raw(
    model: nn.Module,
    x_raw: np.ndarray,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(x_scaler.transform(x_raw), dtype=torch.float32, device=device)
        pred_std = model(x).detach().cpu().numpy().astype(np.float32)
    return y_scaler.inverse(pred_std)


def _stack_seq(items: Sequence[DecoderItem], idxs: Sequence[int], key: str) -> np.ndarray:
    return np.stack([np.asarray(items[int(i)].seq[key], dtype=np.float32) for i in idxs], axis=0).astype(
        np.float32,
        copy=False,
    )


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    weight = mask.to(dtype=x.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (x * weight).sum() / denom


def _fk_positions_from_rot6d_no_inplace(
    rot6d: torch.Tensor,
    parents: Sequence[int],
    offsets: torch.Tensor,
    root_pos: torch.Tensor,
) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must be [...,J,6], got shape={tuple(rot6d.shape)}")
    j_count = int(rot6d.shape[-2])
    lead_shape = rot6d.shape[:-2]
    flat = rot6d.reshape(-1, j_count, 6)
    root_flat = root_pos.reshape(-1, 3).to(device=rot6d.device, dtype=rot6d.dtype)
    offsets_t = offsets[:j_count].to(device=rot6d.device, dtype=rot6d.dtype)
    r_local = rot6d_to_matrix(flat)
    global_r: List[torch.Tensor] = []
    global_p: List[torch.Tensor] = []
    parents_list = [int(x) for x in list(parents)[:j_count]]
    for j in range(j_count):
        parent = parents_list[j]
        if parent < 0 or parent >= j_count:
            rj = r_local[:, j]
            pj = root_flat + offsets_t[j].view(1, 3)
        else:
            rj = torch.matmul(global_r[parent], r_local[:, j])
            off = offsets_t[j].view(1, 3, 1)
            pj = global_p[parent] + torch.matmul(global_r[parent], off).squeeze(-1)
        global_r.append(rj)
        global_p.append(pj)
    return torch.stack(global_p, dim=1).reshape(*lead_shape, j_count, 3)


def _foot_velocity_loss(
    *,
    pred_rot6d: torch.Tensor,
    pred_root_pos: torch.Tensor,
    true_rot6d: torch.Tensor,
    true_root_pos: torch.Tensor,
    contact: torch.Tensor,
    skeleton: Any,
    offsets: torch.Tensor,
    speed_penalty_weight: float,
) -> torch.Tensor:
    b, h, pose_dim = pred_rot6d.shape
    joints = pose_dim // 6
    pred_pos = _fk_positions_from_rot6d_no_inplace(
        pred_rot6d.reshape(b, h, joints, 6),
        skeleton.parents,
        offsets,
        root_pos=pred_root_pos,
    )
    with torch.no_grad():
        true_pos = _fk_positions_from_rot6d_no_inplace(
            true_rot6d.reshape(b, h, joints, 6),
            skeleton.parents,
            offsets,
            root_pos=true_root_pos,
        )
    losses = []
    for ch_idx, joint_idx in ((0, skeleton.right_foot_idx), (1, skeleton.left_foot_idx)):
        if joint_idx is None:
            continue
        mask = (contact[:, :-1, ch_idx] > CONTACT_THRESHOLD) & (contact[:, 1:, ch_idx] > CONTACT_THRESHOLD)
        pred_speed = torch.linalg.norm(pred_pos[:, 1:, joint_idx] - pred_pos[:, :-1, joint_idx], dim=-1) * float(FPS)
        true_speed = torch.linalg.norm(true_pos[:, 1:, joint_idx] - true_pos[:, :-1, joint_idx], dim=-1) * float(FPS)
        losses.append(_masked_mean((pred_speed - true_speed).square(), mask))
        penalty_weight = float(speed_penalty_weight)
        if penalty_weight:
            losses.append(penalty_weight * _masked_mean(pred_speed.square(), mask))
    if not losses:
        return pred_rot6d.new_zeros(())
    return torch.stack(losses).sum()


def _fk_support_objective(
    *,
    pred_std: torch.Tensor,
    ytr_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    true_raw: torch.Tensor,
    true_root_pos: torch.Tensor,
    true_root_vel: torch.Tensor,
    true_cond_dir: torch.Tensor,
    true_contact: torch.Tensor,
    skeleton: Any,
    offsets: torch.Tensor,
    horizon: int,
    loss_weights: Mapping[str, float],
    command_align_root_vel: bool,
    oracle_contact_passthrough: bool,
    foot_speed_penalty_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    pred_raw = pred_std * y_std + y_mean
    state_width = int(horizon) * STATE_DIM
    pred_state = pred_raw[:, :state_width].reshape(-1, int(horizon), STATE_DIM)
    pred_aux = pred_raw[:, state_width:].reshape(-1, int(horizon), ANGVEL_DIM)
    true_state = true_raw[:, :state_width].reshape(-1, int(horizon), STATE_DIM)
    true_aux = true_raw[:, state_width:].reshape(-1, int(horizon), ANGVEL_DIM)
    if oracle_contact_passthrough:
        pred_state = pred_state.clone()
        pred_state[:, :, CONTACT_SLICE] = true_contact

    pred_ego = pred_state[:, :, EGO_VEL_SLICE]
    pred_root_vel = _world_root_vel_from_ego_torch(
        pred_ego,
        true_cond_dir,
        command_align_root_vel=command_align_root_vel,
    )
    pred_root_pos = _integrate_root_pos_torch(pred_root_vel, true_root_pos[:, 0])

    pose = F.mse_loss(pred_state[:, :, POSE_SLICE], true_state[:, :, POSE_SLICE])
    ego = F.mse_loss(pred_state[:, :, EGO_VEL_SLICE], true_state[:, :, EGO_VEL_SLICE])
    yaw = F.mse_loss(pred_state[:, :, YAW_RATE_SLICE], true_state[:, :, YAW_RATE_SLICE])
    aux = F.mse_loss(pred_aux, true_aux)
    root_vel = F.mse_loss(pred_root_vel, true_root_vel)
    root_pos = F.mse_loss(pred_root_pos, true_root_pos)
    pose_step = F.mse_loss(
        pred_state[:, 1:, POSE_SLICE] - pred_state[:, :-1, POSE_SLICE],
        true_state[:, 1:, POSE_SLICE] - true_state[:, :-1, POSE_SLICE],
    )
    ego_step = F.mse_loss(
        pred_state[:, 1:, EGO_VEL_SLICE] - pred_state[:, :-1, EGO_VEL_SLICE],
        true_state[:, 1:, EGO_VEL_SLICE] - true_state[:, :-1, EGO_VEL_SLICE],
    )
    yaw_step = F.mse_loss(
        pred_state[:, 1:, YAW_RATE_SLICE] - pred_state[:, :-1, YAW_RATE_SLICE],
        true_state[:, 1:, YAW_RATE_SLICE] - true_state[:, :-1, YAW_RATE_SLICE],
    )
    aux_rate = F.mse_loss(pred_aux[:, 1:] - pred_aux[:, :-1], true_aux[:, 1:] - true_aux[:, :-1])
    state_raw_mse = F.mse_loss(pred_state, true_state)
    lateral = pred_ego[:, :, 1]
    speed = torch.linalg.norm(pred_ego, dim=-1).clamp_min(EPS)
    command = torch.mean((lateral / speed).square()) + torch.mean(lateral.square())
    foot_vel = _foot_velocity_loss(
        pred_rot6d=pred_state[:, :, POSE_SLICE],
        pred_root_pos=pred_root_pos,
        true_rot6d=true_state[:, :, POSE_SLICE],
        true_root_pos=true_root_pos,
        contact=true_contact,
        skeleton=skeleton,
        offsets=offsets,
        speed_penalty_weight=float(foot_speed_penalty_weight),
    )
    flat = F.mse_loss(pred_std, ytr_std)

    terms = {
        "state_raw_mse": state_raw_mse,
        "flat_standardized": flat,
        "pose": pose,
        "ego": ego,
        "yaw": yaw,
        "aux_bone_angvel": aux,
        "root_vel": root_vel,
        "root_pos": root_pos,
        "pose_step": pose_step,
        "ego_step": ego_step,
        "yaw_step": yaw_step,
        "aux_rate": aux_rate,
        "command_response": command,
        "contacted_foot_velocity": foot_vel,
    }
    loss = pred_std.new_zeros(())
    details: Dict[str, float] = {}
    for key, val in terms.items():
        w = float(loss_weights.get(key, 0.0))
        if w:
            loss = loss + w * val
        details[key] = float(val.detach().cpu().item())
    return loss, details, terms


def _grad_l2_norm(term: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> float:
    if not bool(getattr(term, "requires_grad", False)):
        return 0.0
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float(torch.sum(grad.detach().double().square()).cpu().item())
    return float(math.sqrt(max(total, 0.0)))


def _instrument_step_record(
    *,
    stage: str,
    arm: str,
    epoch: int,
    total_loss: torch.Tensor,
    details: Mapping[str, float],
    terms: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
    params: Sequence[torch.nn.Parameter],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage": stage,
        "arm": arm,
        "epoch": int(epoch),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    for label, key in INSTRUMENTED_TERM_KEYS:
        val = float(details.get(key, 0.0))
        weight = float(loss_weights.get(key, 0.0))
        term = terms.get(key)
        row[f"{label}_loss"] = val
        row[f"{label}_weight"] = weight
        row[f"{label}_grad_norm"] = _grad_l2_norm(term, params) if term is not None else 0.0
        row[f"{label}_weighted_grad_norm"] = (
            _grad_l2_norm(weight * term, params) if term is not None and weight != 0.0 else 0.0
        )
    return row


def _train_one_split(
    *,
    split_name: str,
    split_kind: str,
    train_items: Sequence[DecoderItem],
    test_items: Sequence[DecoderItem],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    all_items: Sequence[DecoderItem],
    horizon: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    loss_weights: Mapping[str, float],
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
    skeleton: Any,
    instrumentation_stage: Optional[str] = None,
    instrumentation_arm: Optional[str] = None,
    foot_speed_penalty_weight: float = 0.1,
) -> Dict[str, Any]:
    del train_items, test_items
    train_x_raw, train_y_raw = _dataset_arrays(all_items, train_idx)
    test_x_raw, test_y_raw = _dataset_arrays(all_items, test_idx)
    x_scaler = _fit_standardizer(train_x_raw)
    y_scaler = _fit_standardizer(train_y_raw)
    train_x = x_scaler.transform(train_x_raw)
    train_y = y_scaler.transform(train_y_raw)

    torch.manual_seed(int(seed))
    model = TinyDeterministicDecoder(train_x.shape[1], int(hidden_dim), train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    y_mean = torch.as_tensor(y_scaler.mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(y_scaler.std, dtype=torch.float32, device=device)
    true_raw = torch.as_tensor(train_y_raw, dtype=torch.float32, device=device)
    true_root_pos = torch.as_tensor(_stack_seq(all_items, train_idx, "root_pos"), dtype=torch.float32, device=device)
    true_root_vel = torch.as_tensor(_stack_seq(all_items, train_idx, "root_vel"), dtype=torch.float32, device=device)
    true_cond_dir = torch.as_tensor(_stack_seq(all_items, train_idx, "cond_dir"), dtype=torch.float32, device=device)
    true_contact = torch.as_tensor(_stack_seq(all_items, train_idx, "contact"), dtype=torch.float32, device=device)
    offsets = torch.as_tensor(skeleton.offsets, dtype=torch.float32, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    step_log: List[Dict[str, Any]] = []

    final_loss = 0.0
    final_terms: Dict[str, float] = {}
    for epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_std = model(xtr)
        loss, final_terms, term_tensors = _fk_support_objective(
            pred_std=pred_std,
            ytr_std=ytr,
            y_mean=y_mean,
            y_std=y_std,
            true_raw=true_raw,
            true_root_pos=true_root_pos,
            true_root_vel=true_root_vel,
            true_cond_dir=true_cond_dir,
            true_contact=true_contact,
            skeleton=skeleton,
            offsets=offsets,
            horizon=int(horizon),
            loss_weights=loss_weights,
            command_align_root_vel=command_align_root_vel,
            oracle_contact_passthrough=oracle_contact_passthrough,
            foot_speed_penalty_weight=float(foot_speed_penalty_weight),
        )
        if instrumentation_stage is not None and instrumentation_arm is not None:
            step_log.append(
                _instrument_step_record(
                    stage=str(instrumentation_stage),
                    arm=str(instrumentation_arm),
                    epoch=int(epoch),
                    total_loss=loss,
                    details=final_terms,
                    terms=term_tensors,
                    loss_weights=loss_weights,
                    params=params,
                )
            )
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())

    train_pred_raw = _predict_raw(model, train_x_raw, x_scaler, y_scaler, device)
    test_pred_raw = _predict_raw(model, test_x_raw, x_scaler, y_scaler, device)
    if oracle_contact_passthrough:
        train_pred_raw = _apply_oracle_contact_passthrough(train_pred_raw, all_items, train_idx, int(horizon))
        test_pred_raw = _apply_oracle_contact_passthrough(test_pred_raw, all_items, test_idx, int(horizon))
    params = int(sum(p.numel() for p in model.parameters()))
    return {
        "split": split_name,
        "split_kind": split_kind,
        "train_idx": [int(x) for x in train_idx],
        "test_idx": [int(x) for x in test_idx],
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_pred_raw": train_pred_raw,
        "test_pred_raw": test_pred_raw,
        "train_y_raw": train_y_raw,
        "test_y_raw": test_y_raw,
        "train_loss_metrics": _loss_metrics(train_pred_raw, train_y_raw, horizon),
        "test_loss_metrics": _loss_metrics(test_pred_raw, test_y_raw, horizon),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "input_dim": int(train_x_raw.shape[1]),
        "output_dim": int(train_y_raw.shape[1]),
        "parameter_count": params,
        "final_train_objective": final_loss,
        "x_constant_features_train": int(x_scaler.constant_count),
        "y_constant_outputs_train": int(y_scaler.constant_count),
        "device": str(device),
        "dtype": "float32",
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "oracle_contact_passthrough": bool(oracle_contact_passthrough),
        "command_align_root_vel": bool(command_align_root_vel),
        "final_train_objective_terms": final_terms,
        "instrumentation_stage": instrumentation_stage,
        "instrumentation_arm": instrumentation_arm,
        "instrumentation_step_log": step_log,
        "foot_speed_penalty_weight": float(foot_speed_penalty_weight),
    }


def _evaluate_split_predictions(
    *,
    split_result: Mapping[str, Any],
    all_items: Sequence[DecoderItem],
    horizon: int,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    variant: str,
    calibration_domain: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for partition, pred_key, idx_key in (
        ("train", "train_pred_raw", "train_idx"),
        ("test", "test_pred_raw", "test_idx"),
    ):
        pred_raw = np.asarray(split_result[pred_key], dtype=np.float32)
        pred_state, pred_aux = _reshape_state_aux(pred_raw, horizon)
        for local_i, item_idx in enumerate(split_result[idx_key]):
            item = all_items[int(item_idx)]
            rows.append(
                _evaluate_prediction(
                    variant=variant,
                    split=str(split_result["split"]),
                    split_kind=str(split_result["split_kind"]),
                    partition=partition,
                    item=item,
                    state=pred_state[local_i],
                    aux=pred_aux[local_i],
                    baseline_bands=baseline_bands,
                    support_bands=support_bands,
                    skeleton=skeleton,
                    min_run_frames=min_run_frames,
                    oracle_contact_passthrough=bool(split_result.get("oracle_contact_passthrough", False)),
                    command_align_root_vel=bool(split_result.get("command_align_root_vel", False)),
                    calibration_domain=calibration_domain,
                )
            )
    return rows


def _evaluate_direct_items(
    *,
    variant: str,
    split: str,
    split_kind: str,
    partition: str,
    items: Sequence[DecoderItem],
    idxs: Sequence[int],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    calibration_domain: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item_idx in idxs:
        item = items[int(item_idx)]
        rows.append(
            _evaluate_seq_common(
                variant=variant,
                split=split,
                split_kind=split_kind,
                partition=partition,
                item=item,
                seq=item.seq,
                baseline_bands=baseline_bands,
                support_bands=support_bands,
                skeleton=skeleton,
                min_run_frames=min_run_frames,
                endpoint_note="direct continuous oracle trajectory copy",
                oracle_contact_passthrough=False,
                command_align_root_vel=False,
                calibration_domain=calibration_domain,
            )
        )
    return rows


def _evaluate_raw_items(
    *,
    variant: str,
    split: str,
    split_kind: str,
    partition: str,
    items: Sequence[DecoderItem],
    idxs: Sequence[int],
    raw: np.ndarray,
    horizon: int,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
    calibration_domain: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    state, aux = _reshape_state_aux(raw, int(horizon))
    for local_i, item_idx in enumerate(idxs):
        rows.append(
            _evaluate_prediction(
                variant=variant,
                split=split,
                split_kind=split_kind,
                partition=partition,
                item=items[int(item_idx)],
                state=state[local_i],
                aux=aux[local_i],
                baseline_bands=baseline_bands,
                support_bands=support_bands,
                skeleton=skeleton,
                min_run_frames=min_run_frames,
                oracle_contact_passthrough=oracle_contact_passthrough,
                command_align_root_vel=command_align_root_vel,
                calibration_domain=calibration_domain,
            )
        )
    return rows


def _nearest_neighbor_copy_raw(
    *,
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    query_x_raw: np.ndarray,
    exclude_self: bool,
) -> np.ndarray:
    x_scaler = _fit_standardizer(train_x_raw)
    xtr = x_scaler.transform(train_x_raw).astype(np.float64)
    xq = x_scaler.transform(query_x_raw).astype(np.float64)
    out = np.zeros((xq.shape[0], train_y_raw.shape[1]), dtype=np.float32)
    for i, x in enumerate(xq):
        d = np.linalg.norm(xtr - x.reshape(1, -1), axis=1) / math.sqrt(max(1, xtr.shape[1]))
        if exclude_self and xq.shape[0] == xtr.shape[0] and i < d.shape[0]:
            d[i] = np.inf
        best = int(np.argmin(d))
        out[i] = train_y_raw[best]
    return out


def _baseline_rows_for_split(
    *,
    split: Any,
    items: Sequence[DecoderItem],
    horizon: int,
    raw_baseline_bands: Mapping[str, Mapping[str, Any]],
    raw_support_bands: Mapping[str, Mapping[str, Any]],
    reconstructed_baseline_bands: Mapping[str, Mapping[str, Any]],
    reconstructed_support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for partition, idxs in (("train", split.train_idx), ("test", split.test_idx)):
        x_raw, y_raw = _dataset_arrays(items, idxs)
        rows.extend(
            _evaluate_direct_items(
                variant="oracle_copy_direct",
                split=split.name,
                split_kind=split.kind,
                partition=partition,
                items=items,
                idxs=idxs,
                baseline_bands=raw_baseline_bands,
                support_bands=raw_support_bands,
                skeleton=skeleton,
                min_run_frames=min_run_frames,
                calibration_domain="raw_continuous",
            )
        )
        rows.extend(
            _evaluate_raw_items(
                variant="gt_through_smoke_guard",
                split=split.name,
                split_kind=split.kind,
                partition=partition,
                items=items,
                idxs=idxs,
                raw=y_raw,
                horizon=horizon,
                baseline_bands=reconstructed_baseline_bands,
                support_bands=reconstructed_support_bands,
                skeleton=skeleton,
                min_run_frames=min_run_frames,
                oracle_contact_passthrough=oracle_contact_passthrough,
                command_align_root_vel=command_align_root_vel,
                calibration_domain="reconstructed_state281",
            )
        )
    train_x, train_y = _dataset_arrays(items, split.train_idx)
    test_x, _ = _dataset_arrays(items, split.test_idx)
    nn_train = _nearest_neighbor_copy_raw(
        train_x_raw=train_x,
        train_y_raw=train_y,
        query_x_raw=train_x,
        exclude_self=True,
    )
    nn_test = _nearest_neighbor_copy_raw(
        train_x_raw=train_x,
        train_y_raw=train_y,
        query_x_raw=test_x,
        exclude_self=False,
    )
    rows.extend(
        _evaluate_raw_items(
            variant="nearest_neighbor_copy",
            split=split.name,
            split_kind=split.kind,
            partition="train",
            items=items,
            idxs=split.train_idx,
            raw=nn_train,
            horizon=horizon,
            baseline_bands=reconstructed_baseline_bands,
            support_bands=reconstructed_support_bands,
            skeleton=skeleton,
            min_run_frames=min_run_frames,
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
            calibration_domain="reconstructed_state281",
        )
    )
    rows.extend(
        _evaluate_raw_items(
            variant="nearest_neighbor_copy",
            split=split.name,
            split_kind=split.kind,
            partition="test",
            items=items,
            idxs=split.test_idx,
            raw=nn_test,
            horizon=horizon,
            baseline_bands=reconstructed_baseline_bands,
            support_bands=reconstructed_support_bands,
            skeleton=skeleton,
            min_run_frames=min_run_frames,
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
            calibration_domain="reconstructed_state281",
        )
    )
    return rows


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "calibration_domain",
        "split",
        "split_kind",
        "partition",
        "clip",
        "start",
        "end",
        "acceptance_proxy_pass",
        "failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
        "oracle_support_token_accuracy",
        "oracle_contact_mse",
        "foot_slip_p95_to_band_ratio",
        "old_pop_safe_diagnostic",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _summarize_split(
    split_result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    split = str(split_result["split"])
    variant = "decoder_fk_support_objective"
    train_rows = [
        r for r in rows if r.get("split") == split and r.get("partition") == "train" and r.get("variant") == variant
    ]
    test_rows = [
        r for r in rows if r.get("split") == split and r.get("partition") == "test" and r.get("variant") == variant
    ]
    return {
        "split": split,
        "split_kind": split_result["split_kind"],
        "train_n": split_result["train_n"],
        "test_n": split_result["test_n"],
        "model": {
            "type": "TinyDeterministicDecoder(flattened MLP)",
            "input_dim": split_result["input_dim"],
            "hidden_dim": split_result["hidden_dim"],
            "output_dim": split_result["output_dim"],
            "parameter_count": split_result["parameter_count"],
            "dtype": split_result["dtype"],
            "device": split_result["device"],
            "epochs": split_result["epochs"],
            "final_train_objective": split_result["final_train_objective"],
            "objective": "FK/support-aware raw objective; flat standardized MSE is metric/optional anchor only",
            "final_train_objective_terms": split_result.get("final_train_objective_terms", {}),
        },
        "scaling": {
            "fit": "train split only",
            "x_constant_features_train": split_result["x_constant_features_train"],
            "y_constant_outputs_train": split_result["y_constant_outputs_train"],
        },
        "loss": {
            "train": split_result["train_loss_metrics"],
            "test": split_result["test_loss_metrics"],
        },
        "acceptance_realized_motion": {
            "train": _summarize_rows(train_rows),
            "test": _summarize_rows(test_rows),
        },
    }


def _summarize_baselines(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for variant in sorted(set(str(r.get("variant")) for r in rows if r.get("variant"))):
        if variant == "decoder_fk_support_objective":
            continue
        vrows = [r for r in rows if r.get("variant") == variant]
        by_split: Dict[str, Any] = {}
        for split in sorted(set(str(r.get("split")) for r in vrows)):
            srows = [r for r in vrows if str(r.get("split")) == split]
            by_split[split] = {
                "train": _summarize_rows([r for r in srows if r.get("partition") == "train"]),
                "test": _summarize_rows([r for r in srows if r.get("partition") == "test"]),
            }
        out[variant] = by_split
    return out


def _domain_guard_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    specs = [
        ("raw_gt_direct", "oracle_copy_direct"),
        ("reconstructed_gt_from_state281", "gt_through_smoke_guard"),
    ]
    out: List[Dict[str, Any]] = []
    for label, variant in specs:
        vrows = [r for r in rows if r.get("variant") == variant]
        train_summaries = []
        test_summaries = []
        for split in sorted(set(str(r.get("split")) for r in vrows)):
            train_summaries.append(
                _summarize_rows(
                    [
                        r
                        for r in vrows
                        if str(r.get("split")) == split and str(r.get("partition")) == "train"
                    ]
                )
            )
            test_summaries.append(
                _summarize_rows(
                    [
                        r
                        for r in vrows
                        if str(r.get("split")) == split and str(r.get("partition")) == "test"
                    ]
                )
            )

        def min_key(recs: Sequence[Mapping[str, Any]], key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in recs if int(r.get("n", 0) or 0) > 0]
            return float(min(vals)) if vals else 0.0

        def max_key(recs: Sequence[Mapping[str, Any]], key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in recs if int(r.get("n", 0) or 0) > 0]
            return float(max(vals)) if vals else 0.0

        domains = sorted(set(str(r.get("calibration_domain")) for r in vrows if r.get("calibration_domain")))
        out.append(
            {
                "label": label,
                "source_variant": variant,
                "calibration_domains": domains,
                "min_train_acceptance_proxy_pass_rate": min_key(train_summaries, "acceptance_proxy_pass_rate"),
                "min_test_acceptance_proxy_pass_rate": min_key(test_summaries, "acceptance_proxy_pass_rate"),
                "min_test_support_honesty_pass_rate": min_key(test_summaries, "support_honesty_pass_rate"),
                "min_test_support_side_correctness_pass_rate": min_key(
                    test_summaries,
                    "support_side_correctness_pass_rate",
                ),
                "min_test_command_response_pass_rate": min_key(test_summaries, "command_response_pass_rate"),
                "max_test_foot_slip_p95_to_band_ratio_mean": max_key(
                    test_summaries,
                    "foot_slip_p95_to_band_ratio_mean",
                ),
            }
        )
    return out


def _min_rate_for_variant(
    baseline_summaries: Mapping[str, Any],
    variant: str,
    partition: str,
    key: str,
) -> float:
    rec = baseline_summaries.get(variant, {}) if isinstance(baseline_summaries, Mapping) else {}
    vals = []
    for split_rec in rec.values():
        if isinstance(split_rec, Mapping):
            vals.append(float((split_rec.get(partition, {}) or {}).get(key, 0.0)))
    return float(min(vals)) if vals else 0.0


def _decision(
    split_summaries: Sequence[Mapping[str, Any]],
    baseline_summaries: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    gt_guard_min = _min_rate_for_variant(
        baseline_summaries,
        "gt_through_smoke_guard",
        "test",
        "acceptance_proxy_pass_rate",
    )
    if gt_guard_min < threshold:
        return {
            "deterministic_decoder_feasible": False,
            "threshold": float(threshold),
            "train_acceptance_gate_pass": False,
            "gt_through_smoke_guard_pass": False,
            "gt_through_smoke_guard_domain": "reconstructed_state281",
            "reason": "reconstructed-domain GT-through-smoke harness guard failed; repair representation/eval harness before model claims",
            "layer2_diffusion_sampling_evidence": "none; harness guard failure is not diffusion evidence",
            "production_ready_generator": False,
        }

    train_checks = [
        s["acceptance_realized_motion"]["train"]
        for s in split_summaries
        if str(s.get("split_kind")) != "random_optimistic_diagnostic"
    ]
    min_train = min(float(t.get("acceptance_proxy_pass_rate", 0.0)) for t in train_checks) if train_checks else 0.0
    if min_train < threshold:
        return {
            "deterministic_decoder_feasible": False,
            "threshold": float(threshold),
            "train_acceptance_gate_pass": False,
            "gt_through_smoke_guard_pass": True,
            "gt_through_smoke_guard_domain": "reconstructed_state281",
            "min_train_acceptance_proxy_pass_rate": min_train,
            "reason": "decoder train acceptance failed after reconstructed-domain GT guard passed; blocked/leave-clip-out is not yet decision-eligible",
            "layer2_diffusion_sampling_evidence": "none; train failure under repaired objective is not diffusion evidence",
            "production_ready_generator": False,
        }

    tests = [
        s["acceptance_realized_motion"]["test"]
        for s in split_summaries
        if str(s.get("split_kind")) != "random_optimistic_diagnostic"
    ]
    if not tests:
        return {"deterministic_decoder_feasible": False, "reason": "no non-random test splits"}
    acc_min = min(float(t.get("acceptance_proxy_pass_rate", 0.0)) for t in tests)
    honesty_min = min(float(t.get("support_honesty_pass_rate", 0.0)) for t in tests)
    side_min = min(float(t.get("support_side_correctness_pass_rate", 0.0)) for t in tests)
    token_min = min(float(t.get("oracle_support_token_accuracy_mean", 0.0)) for t in tests)
    feasible = bool(acc_min >= threshold and honesty_min >= threshold and side_min >= threshold and token_min >= threshold)
    if feasible:
        evidence = "fixed oracle schedule deterministic decoder clears acceptance/support thresholds in this smoke"
        diffusion = "no layer-2 diffusion/sampling evidence from this smoke"
    else:
        evidence = "one or more blocked/leave-clip-out acceptance/support thresholds failed"
        diffusion = (
            "retain sampling/diffusion only as a layer-2 branch if failures persist under fixed oracle schedule; "
            "this smoke alone is not production evidence"
        )
    return {
        "deterministic_decoder_feasible": feasible,
        "threshold": float(threshold),
        "train_acceptance_gate_pass": True,
        "gt_through_smoke_guard_pass": True,
        "gt_through_smoke_guard_domain": "reconstructed_state281",
        "min_test_acceptance_proxy_pass_rate": acc_min,
        "min_test_support_honesty_pass_rate": honesty_min,
        "min_test_support_side_correctness_pass_rate": side_min,
        "min_test_oracle_support_token_accuracy": token_min,
        "reason": evidence,
        "layer2_diffusion_sampling_evidence": diffusion,
        "production_ready_generator": False,
    }


def _guard_path_identity(
    *,
    items: Sequence[DecoderItem],
    idxs: Sequence[int],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
    horizon: int,
    oracle_contact_passthrough: bool,
    command_align_root_vel: bool,
    pass_rate_threshold: float,
) -> Dict[str, Any]:
    del horizon
    seq_keys = ("rot6d", "root_pos", "root_vel", "bone_angvel", "cond_dir", "contact", "yaw_rate")
    max_abs_delta = 0.0
    rows_equal = True
    reconstructed_rows: List[Dict[str, Any]] = []
    decoder_path_rows: List[Dict[str, Any]] = []
    for item_idx in idxs:
        item = items[int(item_idx)]
        raw = _target_from_item(item).reshape(1, -1)
        state, aux = _reshape_state_aux(raw, int(item.seq["state281"].shape[0]))
        reconstructed_seq = _reconstructed_gt_seq(
            item,
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
        )
        decoder_path_seq = _seq_from_prediction(
            item,
            state[0],
            aux[0],
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
        )
        for key in seq_keys:
            a = np.asarray(reconstructed_seq[key], dtype=np.float64)
            b = np.asarray(decoder_path_seq[key], dtype=np.float64)
            max_abs_delta = max(max_abs_delta, float(np.max(np.abs(a - b))) if a.size else 0.0)
        rec_row = _evaluate_seq_common(
            variant="guard_reconstructed_gt_seq",
            split="guard_path_identity",
            split_kind="preflight",
            partition="guard",
            item=item,
            seq=reconstructed_seq,
            baseline_bands=baseline_bands,
            support_bands=support_bands,
            skeleton=skeleton,
            min_run_frames=min_run_frames,
            endpoint_note="reconstructed GT seq path",
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
            calibration_domain="reconstructed_state281",
        )
        dec_row = _evaluate_seq_common(
            variant="guard_decoder_output_path_from_gt_raw",
            split="guard_path_identity",
            split_kind="preflight",
            partition="guard",
            item=item,
            seq=decoder_path_seq,
            baseline_bands=baseline_bands,
            support_bands=support_bands,
            skeleton=skeleton,
            min_run_frames=min_run_frames,
            endpoint_note="decoder eval path fed exact GT raw vector",
            oracle_contact_passthrough=oracle_contact_passthrough,
            command_align_root_vel=command_align_root_vel,
            calibration_domain="reconstructed_state281",
        )
        compare_keys = (
            "acceptance_proxy_pass",
            "failed_family",
            "regime_reached",
            "rate_budget",
            "support_honesty",
            "support_side_correctness",
            "command_response",
            "pose_continuity",
            "endpoint_bridgeability",
        )
        rows_equal = rows_equal and all(rec_row.get(k) == dec_row.get(k) for k in compare_keys)
        reconstructed_rows.append(rec_row)
        decoder_path_rows.append(dec_row)
    rec_summary = _summarize_rows(reconstructed_rows)
    dec_summary = _summarize_rows(decoder_path_rows)
    rec_pass_rate = float(rec_summary.get("acceptance_proxy_pass_rate", 0.0))
    dec_pass_rate = float(dec_summary.get("acceptance_proxy_pass_rate", 0.0))
    passed = bool(
        max_abs_delta <= 1e-8
        and rows_equal
        and rec_pass_rate >= float(pass_rate_threshold)
        and dec_pass_rate >= float(pass_rate_threshold)
    )
    return {
        "passed": passed,
        "n": int(len(idxs)),
        "max_abs_seq_delta": max_abs_delta,
        "acceptance_rows_equal": bool(rows_equal),
        "reconstructed_gt_acceptance_rate": rec_pass_rate,
        "decoder_path_from_gt_raw_acceptance_rate": dec_pass_rate,
        "reconstructed_summary": rec_summary,
        "decoder_path_summary": dec_summary,
        "reason": (
            "same reconstruction + acceptance path; exact GT raw vector is decision-eligible"
            if passed
            else "guard path identity failed; repair reconstruction/eval path before overfit"
        ),
    }


def _make_overfit_split(stage: str, idxs: Sequence[int]) -> SplitDef:
    clean = tuple(int(x) for x in idxs)
    return SplitDef(
        name=str(stage),
        kind="instrumented_overfit",
        train_idx=clean,
        test_idx=clean,
        low_n_diagnostic=True,
        note="debug-only train=test overfit ladder; no generalization claim",
    )


def _loss_weights_for_arm(base: Mapping[str, float], arm: str) -> Dict[str, float]:
    out = {str(k): float(v) for k, v in base.items()}
    single_term_alias = {
        "pose": "pose",
        "ego": "ego",
        "yaw": "yaw",
        "aux": "aux_bone_angvel",
        "aux_rate": "aux_rate",
        "root_vel": "root_vel",
        "root_pos": "root_pos",
        "pose_step": "pose_step",
        "ego_step": "ego_step",
        "yaw_step": "yaw_step",
        "command": "command_response",
        "foot_vel": "contacted_foot_velocity",
    }
    if arm == "aux_off":
        out["aux_bone_angvel"] = 0.0
        out["aux_rate"] = 0.0
    elif arm == "state_anchor_1x":
        out["flat_standardized"] = 1.0
    elif arm == "state_anchor_10x":
        out["flat_standardized"] = 10.0
    elif arm == "flat_only_10x":
        for key in list(out):
            out[key] = 0.0
        out["flat_standardized"] = 10.0
    elif arm == "state_anchor_10x_no_command":
        out["flat_standardized"] = 10.0
        out["command_response"] = 0.0
    elif arm.startswith("flat_plus_"):
        term = arm[len("flat_plus_") :]
        if term not in single_term_alias:
            raise ValueError(f"unknown flat_plus term: {term}")
        active_key = single_term_alias[term]
        for key in list(out):
            out[key] = 0.0
        out["flat_standardized"] = 10.0
        out[active_key] = float(base.get(active_key, 0.0))
    elif arm == "state_anchor_pose_root_only":
        out["flat_standardized"] = 0.0
        out["aux_bone_angvel"] = 0.0
        out["aux_rate"] = 0.0
        out["pose"] = max(float(out.get("pose", 0.0)), 10.0)
        out["ego"] = max(float(out.get("ego", 0.0)), 10.0)
        out["yaw"] = max(float(out.get("yaw", 0.0)), 10.0)
        out["root_vel"] = max(float(out.get("root_vel", 0.0)), 10.0)
        out["root_pos"] = max(float(out.get("root_pos", 0.0)), 10.0)
    return out


def _step_log_summary(rows: Sequence[Mapping[str, Any]], arm: str, stage: str) -> Dict[str, Any]:
    selected = [r for r in rows if str(r.get("arm")) == arm and str(r.get("stage")) == stage]
    if not selected:
        return {"n": 0}
    first = selected[0]
    last = selected[-1]
    out: Dict[str, Any] = {
        "n": int(len(selected)),
        "initial_total_loss": float(first.get("total_loss", 0.0)),
        "final_total_loss": float(last.get("total_loss", 0.0)),
    }
    for label, _ in INSTRUMENTED_TERM_KEYS:
        initial = float(first.get(f"{label}_loss", 0.0))
        final = float(last.get(f"{label}_loss", 0.0))
        grad_vals = np.asarray([float(r.get(f"{label}_weighted_grad_norm", 0.0)) for r in selected], dtype=np.float64)
        out[f"{label}_initial_loss"] = initial
        out[f"{label}_final_loss"] = final
        out[f"{label}_loss_ratio"] = float(final / max(initial, EPS))
        out[f"{label}_weighted_grad_norm_initial"] = float(first.get(f"{label}_weighted_grad_norm", 0.0))
        out[f"{label}_weighted_grad_norm_final"] = float(last.get(f"{label}_weighted_grad_norm", 0.0))
        out[f"{label}_weighted_grad_norm_max"] = float(np.max(grad_vals)) if grad_vals.size else 0.0
    return out


def _classify_overfit_result(
    *,
    guard: Mapping[str, Any],
    train_acceptance: Mapping[str, Any],
    train_loss: Mapping[str, float],
    step_summary: Mapping[str, Any],
    pass_rate_threshold: float,
    near_zero_state_mse: float,
    foot_vel_no_drop_ratio: float,
) -> str:
    if not bool(guard.get("passed", False)):
        return "guard_path_identity_failed"
    train_acc = float(train_acceptance.get("acceptance_proxy_pass_rate", 0.0))
    if train_acc >= float(pass_rate_threshold):
        return "train_fit_acceptance_pass"
    state_mse = float(train_loss.get("state_mse", float("inf")))
    if state_mse <= float(near_zero_state_mse):
        return "operator_mismatch_state_mse_near_zero_accept_fail"
    foot_ratio = float(step_summary.get("foot_vel_loss_ratio", 0.0))
    if foot_ratio >= float(foot_vel_no_drop_ratio):
        return "gradient_swamping_or_loss_balance_foot_vel_not_decreasing"
    return "train_fit_failure_unclassified_from_current_instrumentation"


def _write_instrumentation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", "arm", "epoch", "total_loss"]
    for label, _ in INSTRUMENTED_TERM_KEYS:
        fields.extend(
            [
                f"{label}_loss",
                f"{label}_weight",
                f"{label}_grad_norm",
                f"{label}_weighted_grad_norm",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_ladder_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Instrumented Oracle-Schedule Overfit Ladder",
        "",
        "Debug-only layer-2 overfit ladder. No production Trainer/runtime/gate attachment and no checkpoint mutation.",
        "",
        "## Guard",
        "",
    ]
    guard = payload.get("guard_path_identity", {})
    lines.extend(
        [
            f"- passed: `{guard.get('passed')}`",
            f"- n: `{guard.get('n')}`",
            f"- max_abs_seq_delta: `{guard.get('max_abs_seq_delta')}`",
            f"- reconstructed_gt_acceptance_rate: `{_fmt(guard.get('reconstructed_gt_acceptance_rate', 0.0))}`",
            f"- decoder_path_from_gt_raw_acceptance_rate: `{_fmt(guard.get('decoder_path_from_gt_raw_acceptance_rate', 0.0))}`",
            f"- reason: {guard.get('reason')}",
            "",
            "## Stage Results",
            "",
            "| stage | arm | windows | state mse | aux mse | train accept | support honest | side correct | foot ratio | diagnosis |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rec in payload.get("stage_results", []):
        loss = rec.get("train_loss", {}) or {}
        acc = rec.get("train_acceptance", {}) or {}
        lines.append(
            f"| {rec.get('stage')} | {rec.get('arm')} | {rec.get('train_n')} | "
            f"{_fmt(loss.get('state_mse', 0.0), 8)} | "
            f"{_fmt(loss.get('bone_angvel_aux_mse', 0.0), 8)} | "
            f"{_fmt(acc.get('acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_honesty_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_side_correctness_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('foot_slip_p95_to_band_ratio_mean', 0.0))} | "
            f"{rec.get('diagnosis')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- one-window baseline passed: `{payload.get('one_window_baseline_passed')}`",
            f"- ran 8-window: `{payload.get('ran_8window')}`",
            f"- next step: {payload.get('next_step')}",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
            f"- step log csv: `{payload['artifacts']['step_log_csv']}`",
        ]
    )
    _dump_md(path, lines)


def run_instrumented_overfit_ladder(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if not main_items:
        raise RuntimeError("no matched items available for instrumented overfit ladder")
    start = int(args.overfit_start_index)
    if start < 0 or start >= len(main_items):
        raise ValueError(f"--overfit-start-index out of range: {start}, n={len(main_items)}")

    raw_baseline_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    del raw_baseline_bands
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    base_loss_weights = {
        "flat_standardized": float(args.flat_standardized_loss_weight),
        "pose": float(args.pose_loss_weight),
        "ego": float(args.ego_loss_weight),
        "yaw": float(args.yaw_loss_weight),
        "aux_bone_angvel": float(args.aux_bone_angvel_loss_weight),
        "root_vel": float(args.root_vel_loss_weight),
        "root_pos": float(args.root_pos_loss_weight),
        "pose_step": float(args.pose_step_loss_weight),
        "ego_step": float(args.ego_step_loss_weight),
        "yaw_step": float(args.yaw_step_loss_weight),
        "aux_rate": float(args.aux_rate_loss_weight),
        "command_response": float(args.command_response_loss_weight),
        "contacted_foot_velocity": float(args.contacted_foot_velocity_loss_weight),
    }

    one_idxs = tuple(range(start, start + 1))
    eight_count = min(int(args.overfit_eight_window_count), len(main_items) - start)
    eight_idxs = tuple(range(start, start + eight_count))
    guard = _guard_path_identity(
        items=main_items,
        idxs=eight_idxs if eight_idxs else one_idxs,
        baseline_bands=reconstructed_baseline_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=float(args.pass_rate_threshold),
    )

    rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    stage_results: List[Dict[str, Any]] = []

    if args.overfit_arms:
        arms = tuple(a.strip() for a in str(args.overfit_arms).split(",") if a.strip())
    elif bool(args.state_anchor_rerun):
        arms = ("state_anchor_1x", "state_anchor_10x", "state_anchor_pose_root_only")
    else:
        arms = ("baseline", "aux_off")

    def run_stage(stage: str, idxs: Sequence[int]) -> None:
        split = _make_overfit_split(stage, idxs)
        for arm in arms:
            arm_weights = _loss_weights_for_arm(base_loss_weights, arm)
            result = _train_one_split(
                split_name=split.name,
                split_kind=split.kind,
                train_items=[main_items[int(i)] for i in split.train_idx],
                test_items=[main_items[int(i)] for i in split.test_idx],
                train_idx=split.train_idx,
                test_idx=split.test_idx,
                all_items=main_items,
                horizon=int(args.horizon),
                hidden_dim=int(args.hidden_dim),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed) + (0 if arm == "baseline" else 100_003) + len(stage_results) * 1009,
                device=device,
                loss_weights=arm_weights,
                oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
                command_align_root_vel=bool(args.command_align_root_vel),
                skeleton=skeleton,
                instrumentation_stage=stage,
                instrumentation_arm=arm,
                foot_speed_penalty_weight=float(args.foot_speed_penalty_weight),
            )
            step_rows.extend(result.get("instrumentation_step_log", []))
            variant = f"decoder_instrumented_{arm}"
            pred_rows = _evaluate_split_predictions(
                split_result=result,
                all_items=main_items,
                horizon=int(args.horizon),
                baseline_bands=reconstructed_baseline_bands,
                support_bands=reconstructed_support_bands,
                skeleton=skeleton,
                min_run_frames=int(args.min_run_frames),
                variant=variant,
                calibration_domain="reconstructed_state281",
            )
            rows.extend(pred_rows)
            train_rows = [r for r in pred_rows if r.get("partition") == "train"]
            train_acceptance = _summarize_rows(train_rows)
            step_summary = _step_log_summary(step_rows, arm=arm, stage=stage)
            diagnosis = _classify_overfit_result(
                guard=guard,
                train_acceptance=train_acceptance,
                train_loss=result["train_loss_metrics"],
                step_summary=step_summary,
                pass_rate_threshold=float(args.pass_rate_threshold),
                near_zero_state_mse=float(args.near_zero_state_mse),
                foot_vel_no_drop_ratio=float(args.foot_vel_no_drop_ratio),
            )
            stage_results.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "train_n": int(len(split.train_idx)),
                    "train_indices": [int(x) for x in split.train_idx],
                    "train_windows": [
                        {
                            "clip": main_items[int(i)].clip,
                            "start": int(main_items[int(i)].start),
                            "end": int(main_items[int(i)].end),
                        }
                        for i in split.train_idx
                    ],
                    "loss_weights": arm_weights,
                    "train_loss": result["train_loss_metrics"],
                    "train_acceptance": train_acceptance,
                    "step_log_summary": step_summary,
                    "diagnosis": diagnosis,
                    "final_train_objective": result["final_train_objective"],
                    "final_train_objective_terms": result["final_train_objective_terms"],
                    "parameter_count": result["parameter_count"],
                    "input_dim": result["input_dim"],
                    "output_dim": result["output_dim"],
                    "dtype": result["dtype"],
                    "device": result["device"],
                }
            )

    if bool(guard.get("passed", False)):
        run_stage("one_window", one_idxs)
    gate_arm = arms[0] if args.overfit_arms else ("state_anchor_10x" if bool(args.state_anchor_rerun) else "baseline")
    one_base = next((r for r in stage_results if r["stage"] == "one_window" and r["arm"] == gate_arm), None)
    one_passed = bool(
        one_base
        and one_base["diagnosis"] == "train_fit_acceptance_pass"
        and float(one_base["train_acceptance"].get("acceptance_proxy_pass_rate", 0.0))
        >= float(args.pass_rate_threshold)
    )
    ran_8window = False
    if one_passed and bool(args.run_8window_after_pass):
        run_stage("eight_window", eight_idxs)
        ran_8window = True

    artifacts = {
        "summary_json": str(args.out_dir / "instrumented_overfit_ladder_summary.json"),
        "summary_md": str(args.out_dir / "instrumented_overfit_ladder_summary.md"),
        "rows_csv": str(args.out_dir / "instrumented_overfit_ladder_rows.csv"),
        "step_log_csv": str(args.out_dir / "instrumented_overfit_step_log.csv"),
    }
    return {
        "task": "instrumented_oracle_schedule_overfit_ladder",
        "scope": (
            "debug-only layer-2 one-window/8-window overfit ladder; no production Trainer/runtime/gate; "
            "no checkpoint mutation; not a production generator"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "out_dir": str(args.out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": "cpu",
            "dtype": "float32",
            "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
            "command_align_root_vel": bool(args.command_align_root_vel),
            "pass_rate_threshold": float(args.pass_rate_threshold),
            "near_zero_state_mse": float(args.near_zero_state_mse),
            "foot_vel_no_drop_ratio": float(args.foot_vel_no_drop_ratio),
            "foot_speed_penalty_weight": float(args.foot_speed_penalty_weight),
            "overfit_arms": list(arms),
            "gate_arm": gate_arm,
        },
        "input_output_contract": {
            "train_input": {"shape": "[B,input_dim]", "dtype": "float32", "device": "cpu"},
            "state_output": {"shape": [None, int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "aux_bone_angvel_output": {
                "shape": [None, int(args.horizon), ANGVEL_DIM],
                "dtype": "float32",
                "device": "cpu",
            },
        },
        "guard_path_identity": guard,
        "stage_results": stage_results,
        "one_window_baseline_passed": one_passed,
        "ran_8window": ran_8window,
        "next_step": (
            "inspect 8-window diagnostics"
            if ran_8window
            else (
                "repair guard path before overfit"
                if not bool(guard.get("passed", False))
                else "inspect one-window loss/gradient diagnosis before 8-window"
            )
        ),
        "rows": rows,
        "step_log_row_count": int(len(step_rows)),
        "artifacts": artifacts,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "used_production_trainer": False,
            "forwarded_production_runtime_or_trainer": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "continued_endpoint_yaw_discriminator_instrumentation": False,
            "used_yaw_or_cond_dir_as_prediction_target": False,
            "attached_to_runtime": False,
            "production_ready_generator": False,
        },
        "_step_rows_for_csv": step_rows,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    clips = _load_clips(args.npz_root, args.z_features)
    skeleton = _load_skeleton_meta(args.npz_root)
    raw_baseline_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    raw_support_bands, _ = _calibrate_support_side_bands(
        clips,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        only_clips=tuple(MATCHED_TARGETS) + (UNMATCHED_TARGET,),
    )
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        all_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        all_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    unmatched_items = [it for it in all_items if it.clip == UNMATCHED_TARGET]
    splits = _build_splits(
        main_items,
        train_fraction=float(args.train_fraction),
        block_gap=int(args.block_gap),
        seed=int(args.seed),
        low_n_threshold=int(args.split_low_n_threshold),
        include_random=bool(args.include_random_diagnostic),
    )
    loss_weights = {
        "flat_standardized": float(args.flat_standardized_loss_weight),
        "pose": float(args.pose_loss_weight),
        "ego": float(args.ego_loss_weight),
        "yaw": float(args.yaw_loss_weight),
        "aux_bone_angvel": float(args.aux_bone_angvel_loss_weight),
        "root_vel": float(args.root_vel_loss_weight),
        "root_pos": float(args.root_pos_loss_weight),
        "pose_step": float(args.pose_step_loss_weight),
        "ego_step": float(args.ego_step_loss_weight),
        "yaw_step": float(args.yaw_step_loss_weight),
        "aux_rate": float(args.aux_rate_loss_weight),
        "command_response": float(args.command_response_loss_weight),
        "contacted_foot_velocity": float(args.contacted_foot_velocity_loss_weight),
    }

    split_results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for split_i, split in enumerate(splits):
        if not split.train_idx or not split.test_idx:
            continue
        rows.extend(
            _baseline_rows_for_split(
                split=split,
                items=main_items,
                horizon=int(args.horizon),
                raw_baseline_bands=raw_baseline_bands,
                raw_support_bands=raw_support_bands,
                reconstructed_baseline_bands=reconstructed_baseline_bands,
                reconstructed_support_bands=reconstructed_support_bands,
                skeleton=skeleton,
                min_run_frames=int(args.min_run_frames),
                oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
                command_align_root_vel=bool(args.command_align_root_vel),
            )
        )
        result = _train_one_split(
            split_name=split.name,
            split_kind=split.kind,
            train_items=[main_items[int(i)] for i in split.train_idx],
            test_items=[main_items[int(i)] for i in split.test_idx],
            train_idx=split.train_idx,
            test_idx=split.test_idx,
            all_items=main_items,
            horizon=int(args.horizon),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed) + split_i * 1009,
            device=device,
            loss_weights=loss_weights,
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
            command_align_root_vel=bool(args.command_align_root_vel),
            skeleton=skeleton,
            foot_speed_penalty_weight=float(args.foot_speed_penalty_weight),
        )
        rows.extend(
            _evaluate_split_predictions(
                split_result=result,
                all_items=main_items,
                horizon=int(args.horizon),
                baseline_bands=reconstructed_baseline_bands,
                support_bands=reconstructed_support_bands,
                skeleton=skeleton,
                min_run_frames=int(args.min_run_frames),
                variant="decoder_fk_support_objective",
                calibration_domain="reconstructed_state281",
            )
        )
        split_results.append(result)

    split_summaries = [_summarize_split(result, rows) for result in split_results]
    baseline_summaries = _summarize_baselines(rows)
    gt_domain_comparison = _domain_guard_summary(rows)

    walk_l_to_r_summary: Dict[str, Any] = {"n": 0, "evaluated_with_split": None}
    if split_results and unmatched_items:
        primary = split_results[0]
        x_raw = np.stack([_feature_from_item(it) for it in unmatched_items], axis=0).astype(np.float32)
        y_raw = np.stack([_target_from_item(it) for it in unmatched_items], axis=0).astype(np.float32)
        pred_raw = _predict_raw(primary["model"], x_raw, primary["x_scaler"], primary["y_scaler"], device)
        if args.oracle_contact_passthrough:
            pred_raw = _apply_oracle_contact_passthrough(
                pred_raw,
                unmatched_items,
                tuple(range(len(unmatched_items))),
                int(args.horizon),
            )
        pred_state, pred_aux = _reshape_state_aux(pred_raw, int(args.horizon))
        diag_rows = []
        for i, item in enumerate(unmatched_items):
            diag_rows.append(
                _evaluate_prediction(
                    variant="decoder_fk_support_objective",
                    split="walk_l_to_r_diagnostic",
                    split_kind="unmatched_diagnostic",
                    partition="diagnostic",
                    item=item,
                    state=pred_state[i],
                    aux=pred_aux[i],
                    baseline_bands=reconstructed_baseline_bands,
                    support_bands=reconstructed_support_bands,
                    skeleton=skeleton,
                    min_run_frames=int(args.min_run_frames),
                    oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
                    command_align_root_vel=bool(args.command_align_root_vel),
                    calibration_domain="reconstructed_state281",
                )
            )
        rows.extend(diag_rows)
        walk_l_to_r_summary = {
            "n": int(len(unmatched_items)),
            "evaluated_with_split": primary["split"],
            "not_mixed_into_training": True,
            "loss": _loss_metrics(pred_raw, y_raw, int(args.horizon)),
            "acceptance_realized_motion": _summarize_rows(diag_rows),
        }

    payload = {
        "task": "oracle_schedule_trajectory_decoder_smoke",
        "scope": (
            "temporary layer-2 oracle-schedule deterministic decoder smoke; no production Trainer/runtime/gate; "
            "no checkpoint mutation; not a production generator"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "out_dir": str(args.out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "train_fraction": float(args.train_fraction),
            "block_gap": int(args.block_gap),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "baseline_quantile": float(args.baseline_quantile),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "decoder_evaluation_calibration_domain": "reconstructed_state281",
            "device": "cpu",
            "dtype": "float32",
            "loss_weights": loss_weights,
            "oracle_contact_passthrough": bool(args.oracle_contact_passthrough),
            "command_align_root_vel": bool(args.command_align_root_vel),
            "foot_speed_penalty_weight": float(args.foot_speed_penalty_weight),
        },
        "input_output_contract": {
            "ctx": {
                "shape_contract": "[C,281]",
                "actual_shape": [int(args.context_len), STATE_DIM],
                "dtype": "float32",
                "device": "cpu",
            },
            "oracle_support_schedule_contact": {
                "shape_contract": "[H,2]",
                "actual_shape": [int(args.horizon), 2],
                "dtype": "float32",
                "device": "cpu",
                "role": "oracle condition for smoke only; not runtime",
            },
            "commanded_cond_dir_yaw_cue": {
                "shape_contract": "[H,3]",
                "actual_shape": [int(args.horizon), 3],
                "dtype": "float32",
                "device": "cpu",
                "role": "commanded cue only; not yaw prediction target",
            },
            "soft_endpoint_cue": {
                "shape_contract": "[279]",
                "actual_shape": [CONTACT_SLICE.start],
                "dtype": "float32",
                "device": "cpu",
                "allowed_fields": "endpoint state prefix state281[-1,:279], endpoint contact excluded",
            },
            "oracle_topology_timing_tokens": {
                "shape_contract": "[H,4] one-hot + [H,2] run phase + [6] stats",
                "dtype": "float32",
                "device": "cpu",
                "role": "oracle-schedule smoke condition only",
            },
            "middle_state_output": {
                "shape_contract": "[H,281]",
                "actual_shape": [int(args.horizon), STATE_DIM],
                "dtype": "float32",
                "device": "cpu",
                "contact_channels": (
                    "oracle deterministic pass-through from fixed schedule condition"
                    if args.oracle_contact_passthrough
                    else "learned MLP output"
                ),
            },
            "bone_angvel_aux_output": {
                "shape_contract": "[H,138]",
                "actual_shape": [int(args.horizon), ANGVEL_DIM],
                "dtype": "float32",
                "device": "cpu",
                "role": "aux/witness loss and acceptance rate witness; not part of handoff 281 schema",
            },
        },
        "calibration_contract": {
            "raw_gt_direct_domain": {
                "calibration_domain": "raw_continuous",
                "baseline_quantile": float(args.baseline_quantile),
                "support_side_rule": "existing raw continuous inclusive min/max support-side bands",
                "used_for_variants": ["oracle_copy_direct"],
            },
            "reconstructed_state281_domain": {
                "calibration_domain": "reconstructed_state281",
                "baseline_quantile": float(args.reconstructed_baseline_quantile),
                "support_side_rule": "inclusive min/max over GT state281 reconstructed through smoke path",
                "used_for_variants": [
                    "gt_through_smoke_guard",
                    "nearest_neighbor_copy",
                    "decoder_fk_support_objective",
                    "walk_l_to_r_diagnostic",
                ],
            },
            "decision_gate": "requires gt_through_smoke_guard in reconstructed_state281 domain before decoder train/test claims",
        },
        "dataset": {
            "matched_train_eval_clips": list(MATCHED_TARGETS),
            "walk_l_to_r": "diagnostic only; never mixed into training",
            "matched_window_count": int(len(main_items)),
            "walk_l_to_r_window_count": int(len(unmatched_items)),
            "per_clip_windows": dict(Counter(it.clip for it in all_items)),
        },
        "split_summaries": split_summaries,
        "baseline_summaries": baseline_summaries,
        "gt_domain_comparison": gt_domain_comparison,
        "walk_l_to_r_diagnostic": walk_l_to_r_summary,
        "decision": _decision(
            split_summaries,
            baseline_summaries,
            threshold=float(args.pass_rate_threshold),
        ),
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "used_production_trainer": False,
            "forwarded_production_runtime_or_trainer": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "continued_endpoint_yaw_discriminator_instrumentation": False,
            "used_yaw_or_cond_dir_as_prediction_target": False,
            "oracle_contact_passthrough_used": bool(args.oracle_contact_passthrough),
            "command_aligned_root_velocity_used": bool(args.command_align_root_vel),
            "attached_to_runtime": False,
            "production_ready_generator": False,
        },
        "artifacts": {
            "summary_json": str(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_summary.json"),
            "summary_md": str(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_summary.md"),
            "rows_csv": str(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_rows.csv"),
        },
    }

    for result in split_results:
        result.pop("model", None)
        result.pop("x_scaler", None)
        result.pop("y_scaler", None)
        result.pop("train_pred_raw", None)
        result.pop("test_pred_raw", None)
        result.pop("train_y_raw", None)
        result.pop("test_y_raw", None)
    payload["debug_train_results"] = split_results
    payload["rows"] = rows
    return payload


def _write_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Oracle-Schedule Trajectory Decoder Smoke",
        "",
        "Temporary debug tool only. Not a production generator; no runtime/trainer/gate attachment.",
        "",
        "## Dataset / Contract",
        "",
        f"- matched windows: `{payload['dataset']['matched_window_count']}` from `{payload['dataset']['matched_train_eval_clips']}`",
        f"- Walk_L_To_R diagnostic windows: `{payload['dataset']['walk_l_to_r_window_count']}`; not mixed into training",
        "- input: `ctx [C,281]`, oracle contact `[H,2]`, commanded cond_dir/yaw `[H,3]`, endpoint non-contact `[279]`, oracle support tokens",
        "- output: middle `state281 [H,281]` plus aux `bone_angvel [H,138]` witness",
        f"- contact channels: `{payload['input_output_contract']['middle_state_output']['contact_channels']}`",
        "",
        "## Harness / Baselines",
        "",
        "| variant | min train accept | min test accept | min test support honest | min test side correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, by_split in payload.get("baseline_summaries", {}).items():
        train_acc = []
        test_acc = []
        test_honest = []
        test_side = []
        for split_rec in by_split.values():
            train = split_rec.get("train", {}) if isinstance(split_rec, Mapping) else {}
            test = split_rec.get("test", {}) if isinstance(split_rec, Mapping) else {}
            train_acc.append(float(train.get("acceptance_proxy_pass_rate", 0.0)))
            test_acc.append(float(test.get("acceptance_proxy_pass_rate", 0.0)))
            test_honest.append(float(test.get("support_honesty_pass_rate", 0.0)))
            test_side.append(float(test.get("support_side_correctness_pass_rate", 0.0)))
        lines.append(
            f"| {variant} | {_fmt(min(train_acc) if train_acc else 0.0)} | "
            f"{_fmt(min(test_acc) if test_acc else 0.0)} | "
            f"{_fmt(min(test_honest) if test_honest else 0.0)} | "
            f"{_fmt(min(test_side) if test_side else 0.0)} |"
        )
    lines.extend(
        [
            "",
            "## GT Domain Calibration Guard",
            "",
            "| comparison | calibration domain | min train accept | min test accept | support honest | side correct | command | max foot ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rec in payload.get("gt_domain_comparison", []):
        domains = ",".join(str(x) for x in rec.get("calibration_domains", []))
        lines.append(
            f"| {rec.get('label')} | {domains} | "
            f"{_fmt(rec.get('min_train_acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(rec.get('min_test_acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(rec.get('min_test_support_honesty_pass_rate', 0.0))} | "
            f"{_fmt(rec.get('min_test_support_side_correctness_pass_rate', 0.0))} | "
            f"{_fmt(rec.get('min_test_command_response_pass_rate', 0.0))} | "
            f"{_fmt(rec.get('max_test_foot_slip_p95_to_band_ratio_mean', 0.0))} |"
        )
    lines.extend(
        [
            "",
            (
                "- decision gate: reconstructed GT must pass reconstructed-domain bands before decoder "
                "train/test claims are decision-eligible"
            ),
            "",
        "## Split Results",
        "",
            "| split | train/test | state mse | aux mse | train accept | test accept | support honest | side correct | foot ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for s in payload["split_summaries"]:
        loss = s["loss"]["test"]
        train_acc = s["acceptance_realized_motion"]["train"]
        acc = s["acceptance_realized_motion"]["test"]
        lines.append(
            f"| {s['split']} | {s['train_n']}/{s['test_n']} | {_fmt(loss['state_mse'], 6)} | "
            f"{_fmt(loss['bone_angvel_aux_mse'], 6)} | "
            f"{_fmt(train_acc.get('acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('acceptance_proxy_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_honesty_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('support_side_correctness_pass_rate', 0.0))} | "
            f"{_fmt(acc.get('foot_slip_p95_to_band_ratio_mean', 0.0))} |"
        )
    decision = payload["decision"]
    w = payload["walk_l_to_r_diagnostic"]
    w_acc = w.get("acceptance_realized_motion", {})
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- deterministic oracle-schedule decoder feasible: `{decision['deterministic_decoder_feasible']}`",
            f"- reason: {decision['reason']}",
            f"- layer-2 diffusion/sampling evidence: {decision['layer2_diffusion_sampling_evidence']}",
            f"- production-ready generator: `{decision['production_ready_generator']}`",
            "",
            "## Walk_L_To_R",
            "",
            f"- diagnostic n: `{w.get('n')}`; evaluated with split: `{w.get('evaluated_with_split')}`",
            f"- acceptance proxy pass rate: `{_fmt(w_acc.get('acceptance_proxy_pass_rate', 0.0))}`",
            f"- support token accuracy: `{_fmt(w_acc.get('oracle_support_token_accuracy_mean', 0.0))}`",
        ]
    )
    _dump_md(path, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--block-gap", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--split-low-n-threshold", type=int, default=20)
    p.add_argument("--include-random-diagnostic", action="store_true")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--torch-num-threads", type=int, default=8)
    p.add_argument(
        "--instrumented-overfit-ladder",
        action="store_true",
        help="debug-only one-window then optional 8-window train-fit ladder with per-step loss/grad logs",
    )
    p.add_argument(
        "--state-anchor-rerun",
        action="store_true",
        help="with --instrumented-overfit-ladder, run state_anchor_1x/state_anchor_10x/state_anchor_pose_root_only arms only",
    )
    p.add_argument(
        "--overfit-arms",
        type=str,
        default="",
        help="comma-separated instrumented overfit arms to run; overrides default arm set",
    )
    p.add_argument("--overfit-start-index", type=int, default=0)
    p.add_argument("--overfit-eight-window-count", type=int, default=8)
    p.add_argument(
        "--skip-8window-after-pass",
        dest="run_8window_after_pass",
        action="store_false",
        help="diagnostic escape hatch; default ladder runs 8-window after baseline one-window passes",
    )
    p.set_defaults(run_8window_after_pass=True)
    p.add_argument("--near-zero-state-mse", type=float, default=1e-6)
    p.add_argument("--foot-vel-no-drop-ratio", type=float, default=0.9)
    p.add_argument(
        "--foot-speed-penalty-weight",
        type=float,
        default=0.1,
        help="debug foot objective penalty on pred support speed; set 0 to keep only (pred_speed-true_speed)^2",
    )
    p.add_argument("--baseline-quantile", type=float, default=99.5)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--pass-rate-threshold", type=float, default=0.75)
    p.add_argument("--flat-standardized-loss-weight", type=float, default=0.0)
    p.add_argument("--pose-loss-weight", type=float, default=1.0)
    p.add_argument("--ego-loss-weight", type=float, default=4.0)
    p.add_argument("--yaw-loss-weight", type=float, default=4.0)
    p.add_argument("--contact-loss-weight", type=float, default=0.0)
    p.add_argument("--aux-bone-angvel-loss-weight", type=float, default=0.5)
    p.add_argument("--root-vel-loss-weight", type=float, default=4.0)
    p.add_argument("--root-pos-loss-weight", type=float, default=8.0)
    p.add_argument("--pose-step-loss-weight", type=float, default=16.0)
    p.add_argument("--ego-step-loss-weight", type=float, default=4.0)
    p.add_argument("--yaw-step-loss-weight", type=float, default=4.0)
    p.add_argument("--aux-rate-loss-weight", type=float, default=1.0)
    p.add_argument("--command-response-loss-weight", type=float, default=24.0)
    p.add_argument("--contacted-foot-velocity-loss-weight", type=float, default=0.35)
    p.add_argument(
        "--learn-contact-output",
        dest="oracle_contact_passthrough",
        action="store_false",
        help="diagnostic ablation: evaluate learned contact output instead of fixed oracle contact pass-through",
    )
    p.set_defaults(oracle_contact_passthrough=True)
    p.add_argument(
        "--command-align-root-vel",
        dest="command_align_root_vel",
        action="store_true",
        help="diagnostic ablation: ignore ego lateral velocity and reconstruct root velocity along commanded cond_dir",
    )
    p.add_argument(
        "--use-ego-lateral-root-vel",
        dest="command_align_root_vel",
        action="store_false",
        help="default: reconstruct root velocity from both ego forward and lateral channels",
    )
    p.set_defaults(command_align_root_vel=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.instrumented_overfit_ladder):
        payload = run_instrumented_overfit_ladder(args)
        step_rows = payload.pop("_step_rows_for_csv", [])
        _dump_json(args.out_dir / "instrumented_overfit_ladder_summary.json", payload)
        _write_ladder_md(args.out_dir / "instrumented_overfit_ladder_summary.md", payload)
        _write_rows_csv(args.out_dir / "instrumented_overfit_ladder_rows.csv", payload["rows"])
        _write_instrumentation_csv(args.out_dir / "instrumented_overfit_step_log.csv", step_rows)
        print(f"wrote {args.out_dir / 'instrumented_overfit_ladder_summary.json'}")
        print(f"wrote {args.out_dir / 'instrumented_overfit_ladder_summary.md'}")
        print(f"wrote {args.out_dir / 'instrumented_overfit_ladder_rows.csv'}")
        print(f"wrote {args.out_dir / 'instrumented_overfit_step_log.csv'}")
        return
    payload = run(args)
    _dump_json(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_summary.json", payload)
    _write_md(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_summary.md", payload)
    _write_rows_csv(args.out_dir / "oracle_schedule_trajectory_decoder_smoke_rows.csv", payload["rows"])
    print(f"wrote {args.out_dir / 'oracle_schedule_trajectory_decoder_smoke_summary.json'}")
    print(f"wrote {args.out_dir / 'oracle_schedule_trajectory_decoder_smoke_summary.md'}")
    print(f"wrote {args.out_dir / 'oracle_schedule_trajectory_decoder_smoke_rows.csv'}")


if __name__ == "__main__":
    main()
