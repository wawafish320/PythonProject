#!/usr/bin/env python3
"""Learner-vs-condition ablation for support topology feasibility.

Read-only lightweight probe. This script trains only small CPU classifiers over
existing support-topology labels. It does not train a full trajectory generator,
forward production runtime/trainer code, mutate checkpoints, or edit any
production gate.
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
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _fmt,
    _load_clips,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    LABEL_ORDER,
    WindowItem,
    _available_context_feature,
    _context_window,
    _label_has_side,
    _make_sequence,
    _support_contract,
    _walk_l_to_r_report,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
    UNMATCHED_TARGET,
    ScheduleTarget,
    SplitDef,
    _build_splits,
    _class_id_maps,
    _conditional_entropy,
    _entropy_from_values,
    _target_from_item,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)


DEFAULT_OUT_DIR = Path(
    "debug_output/_tmp_action_handoff_support_topology_learner_condition_ablation_20260602"
)
DEFAULT_PREVIOUS_BASELINE = Path(
    "debug_output/_tmp_action_handoff_support_schedule_predictive_baseline_20260602/"
    "support_schedule_predictive_baseline_summary.json"
)

EPS = 1e-8
POSE_PROXY_DIM = 48

FEATURE_TIERS = (
    "base_available",
    "non_leaky_ctx_history",
    "non_leaky_ctx_history_plus_endpoint",
    "non_leaky_ctx_history_plus_command",
    "leaky_upper_bound_oracle_phase_support",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    role: str
    decision_eligible: bool
    leaky_upper_bound: bool
    description: str


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    role: str
    decision_eligible: bool
    diagnostic_only: bool


@dataclass(frozen=True)
class PredResult:
    probs_train: np.ndarray
    probs_test: np.ndarray
    pred_train: np.ndarray
    pred_test: np.ndarray
    details: Dict[str, Any]


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "base_available": FeatureSpec(
        name="base_available",
        role="baseline",
        decision_eligible=True,
        leaky_upper_bound=False,
        description="previous available_context feature from the schedule baseline",
    ),
    "non_leaky_ctx_history": FeatureSpec(
        name="non_leaky_ctx_history",
        role="runtime_available",
        decision_eligible=True,
        leaky_upper_bound=False,
        description="ctx-only support/contact, ego/yaw, and pose-phase proxy",
    ),
    "non_leaky_ctx_history_plus_endpoint": FeatureSpec(
        name="non_leaky_ctx_history_plus_endpoint",
        role="runtime_available",
        decision_eligible=True,
        leaky_upper_bound=False,
        description="ctx history plus soft endpoint pose/root/yaw cue; endpoint contact channels excluded",
    ),
    "non_leaky_ctx_history_plus_command": FeatureSpec(
        name="non_leaky_ctx_history_plus_command",
        role="runtime_available",
        decision_eligible=True,
        leaky_upper_bound=False,
        description="ctx history plus soft endpoint cue plus commanded cond_dir/yaw cue",
    ),
    "leaky_upper_bound_oracle_phase_support": FeatureSpec(
        name="leaky_upper_bound_oracle_phase_support",
        role="leaky_upper_bound",
        decision_eligible=False,
        leaky_upper_bound=True,
        description="oracle future support/phase/bone_angvel witness; diagnostic upper-bound only",
    ),
}

LEARNER_SPECS: Dict[str, LearnerSpec] = {
    "torch_logistic": LearnerSpec(
        name="torch_logistic",
        role="decision_learner",
        decision_eligible=True,
        diagnostic_only=False,
    ),
    "torch_small_mlp": LearnerSpec(
        name="torch_small_mlp",
        role="decision_learner",
        decision_eligible=True,
        diagnostic_only=False,
    ),
    "diagnostic_knn": LearnerSpec(
        name="diagnostic_knn",
        role="diagnostic_only",
        decision_eligible=False,
        diagnostic_only=True,
    ),
    "diagnostic_nearest_centroid": LearnerSpec(
        name="diagnostic_nearest_centroid",
        role="diagnostic_only",
        decision_eligible=False,
        diagnostic_only=True,
    ),
}


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _as_float32_finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)


def _support_labels_from_contact(contact: np.ndarray) -> List[str]:
    arr = np.asarray(contact, dtype=np.float32).reshape(-1, 2)
    out: List[str] = []
    for row in arr:
        right = bool(row[0] > 0.5)
        left = bool(row[1] > 0.5)
        if right and left:
            out.append("dual")
        elif right:
            out.append("right")
        elif left:
            out.append("left")
        else:
            out.append("flight_or_unknown")
    return out


def _one_hot_label(label: str) -> np.ndarray:
    return np.asarray([1.0 if str(label) == x else 0.0 for x in LABEL_ORDER], dtype=np.float32)


def _one_hot_labels(labels: Sequence[str]) -> np.ndarray:
    if not labels:
        return np.zeros((0, len(LABEL_ORDER)), dtype=np.float32)
    return np.stack([_one_hot_label(str(x)) for x in labels], axis=0).astype(np.float32, copy=False)


def _support_transition_count(labels: Sequence[str]) -> int:
    return int(sum(str(a) != str(b) for a, b in zip(labels[:-1], labels[1:])))


def _support_entropy_bits(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = np.asarray(list(Counter(str(x) for x in labels).values()), dtype=np.float64)
    p = counts / max(float(np.sum(counts)), EPS)
    return float(-np.sum(p * np.log2(np.maximum(p, EPS))))


def _recent_run_length(labels: Sequence[str]) -> int:
    if not labels:
        return 0
    cur = str(labels[-1])
    n = 0
    for label in reversed(labels):
        if str(label) != cur:
            break
        n += 1
    return int(n)


def _run_phase_features(labels: Sequence[str]) -> np.ndarray:
    n = len(labels)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.zeros((n, 2), dtype=np.float32)
    start = 0
    while start < n:
        end = start + 1
        while end < n and str(labels[end]) == str(labels[start]):
            end += 1
        length = max(1, end - start)
        for j in range(start, end):
            phase = (j - start) / max(1, length - 1)
            out[j, 0] = math.sin(2.0 * math.pi * phase)
            out[j, 1] = math.cos(2.0 * math.pi * phase)
        start = end
    return out


def _ctx_history_feature(item: WindowItem) -> np.ndarray:
    ctx = np.asarray(item.ctx, dtype=np.float32)
    if ctx.ndim != 2 or ctx.shape[1] < CONTACT_SLICE.stop:
        raise ValueError(f"ctx must be [C,281]-compatible, got shape={tuple(ctx.shape)}")

    contact = ctx[:, CONTACT_SLICE].astype(np.float32, copy=False)
    ego_vel = ctx[:, EGO_VEL_SLICE].astype(np.float32, copy=False)
    yaw_rate = ctx[:, YAW_RATE_SLICE].reshape(ctx.shape[0], -1).astype(np.float32, copy=False)
    pose = ctx[:, POSE_SLICE].astype(np.float32, copy=False)
    pose_proxy_width = min(POSE_PROXY_DIM, pose.shape[1])
    pose_proxy = pose[:, :pose_proxy_width]
    if pose.shape[0] >= 2:
        pose_step = np.linalg.norm(np.diff(pose, axis=0), axis=1) / math.sqrt(max(1, pose.shape[1]))
    else:
        pose_step = np.zeros((0,), dtype=np.float32)
    pose_stats = np.asarray(
        [
            float(np.mean(pose_step)) if pose_step.size else 0.0,
            float(np.std(pose_step)) if pose_step.size else 0.0,
            float(np.max(pose_step)) if pose_step.size else 0.0,
            float(np.linalg.norm(pose[-1] - pose[0]) / math.sqrt(max(1, pose.shape[1]))),
            float(np.linalg.norm(ego_vel[-1])),
            float(np.mean(np.abs(yaw_rate))),
        ],
        dtype=np.float32,
    )

    labels = _support_labels_from_contact(contact)
    recent_len = _recent_run_length(labels)
    transition_count = _support_transition_count(labels)
    support_stats = np.asarray(
        [
            recent_len / max(1, len(labels)),
            transition_count / max(1, len(labels) - 1),
            _support_entropy_bits(labels),
            sum(1 for x in labels if _label_has_side(x, "right")) / max(1, len(labels)),
            sum(1 for x in labels if _label_has_side(x, "left")) / max(1, len(labels)),
            sum(1 for x in labels if x == "flight_or_unknown") / max(1, len(labels)),
        ],
        dtype=np.float32,
    )
    run_phase = _run_phase_features(labels).reshape(-1)
    cyclic_recent = np.asarray(
        [
            math.sin(2.0 * math.pi * recent_len / max(1, len(labels))),
            math.cos(2.0 * math.pi * recent_len / max(1, len(labels))),
            math.sin(2.0 * math.pi * transition_count / max(1, len(labels))),
            math.cos(2.0 * math.pi * transition_count / max(1, len(labels))),
        ],
        dtype=np.float32,
    )

    parts = [
        contact.reshape(-1),
        ego_vel.reshape(-1),
        yaw_rate.reshape(-1),
        pose_proxy[-1],
        np.mean(pose_proxy, axis=0),
        np.std(pose_proxy, axis=0),
        pose_proxy[-1] - pose_proxy[0],
        pose_step.astype(np.float32, copy=False).reshape(-1),
        pose_stats,
        _one_hot_label(labels[-1] if labels else "flight_or_unknown"),
        support_stats,
        run_phase,
        cyclic_recent,
    ]
    return _as_float32_finite(np.concatenate(parts, axis=0))


def _endpoint_feature(item: WindowItem) -> np.ndarray:
    seq = item.seq
    endpoint_state = np.asarray(seq["state281"][-1, : CONTACT_SLICE.start], dtype=np.float32).reshape(-1)
    selected_seam = str(item.support_contract["normalized_label_sequence"][0])
    raw_start = str(item.support_contract["schedule_label_sequence"][0])
    return _as_float32_finite(
        np.concatenate([endpoint_state, _one_hot_label(selected_seam), _one_hot_label(raw_start)], axis=0)
    )


def _command_feature(item: WindowItem) -> np.ndarray:
    seq = item.seq
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)
    return _as_float32_finite(np.concatenate([cond_dir, yaw_rate], axis=0))


def _leaky_oracle_feature(item: WindowItem) -> np.ndarray:
    seq = item.seq
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    label_oh = _one_hot_labels(labels)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    phase = _run_phase_features(labels)
    ang = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    if ang.shape[0] >= 2:
        ang_step = np.sqrt(np.mean(np.diff(ang, axis=0).astype(np.float64) ** 2, axis=1)).astype(np.float32)
    else:
        ang_step = np.zeros((0,), dtype=np.float32)
    support_stats = np.asarray(
        [
            sum(1 for x in labels if _label_has_side(x, "right")) / max(1, len(labels)),
            sum(1 for x in labels if _label_has_side(x, "left")) / max(1, len(labels)),
            sum(1 for x in labels if x == "dual") / max(1, len(labels)),
            sum(1 for x in labels if x == "flight_or_unknown") / max(1, len(labels)),
            _support_transition_count(labels),
            _support_entropy_bits(labels),
            len(Counter(labels)),
        ],
        dtype=np.float32,
    )
    return _as_float32_finite(
        np.concatenate(
            [
                label_oh.reshape(-1),
                contact.reshape(-1),
                phase.reshape(-1),
                _one_hot_label(labels[0] if labels else "flight_or_unknown"),
                _one_hot_label(labels[-1] if labels else "flight_or_unknown"),
                support_stats,
                ang.reshape(-1),
                ang[0],
                ang[-1],
                np.mean(ang, axis=0).astype(np.float32),
                np.std(ang, axis=0).astype(np.float32),
                ang_step.reshape(-1),
            ],
            axis=0,
        )
    )


def _feature_vector(item: WindowItem, tier: str) -> np.ndarray:
    if tier == "base_available":
        return _as_float32_finite(_available_context_feature(item))
    history = _ctx_history_feature(item)
    if tier == "non_leaky_ctx_history":
        return history
    if tier == "non_leaky_ctx_history_plus_endpoint":
        return _as_float32_finite(np.concatenate([history, _endpoint_feature(item)], axis=0))
    if tier == "non_leaky_ctx_history_plus_command":
        return _as_float32_finite(
            np.concatenate([history, _endpoint_feature(item), _command_feature(item)], axis=0)
        )
    if tier == "leaky_upper_bound_oracle_phase_support":
        return _leaky_oracle_feature(item)
    raise KeyError(tier)


def _build_items(
    clips: Mapping[str, Any],
    *,
    horizon: int,
    context_len: int,
    min_run_frames: int,
    stride: int,
) -> List[WindowItem]:
    items: List[WindowItem] = []
    for name in TURN_CLIPS:
        clip = clips[name]
        max_start = int(clip.state281.shape[0]) - int(horizon)
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, int(stride))):
            seq = _make_sequence(clip, start, horizon)
            seq["state281"] = clip.state281[start : start + int(horizon)].astype(np.float32, copy=False)
            ctx = _context_window(clip, start, context_len, wrap=(name == WALK_F))
            contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
            item = WindowItem(
                clip=name,
                start=int(start),
                end=int(start + horizon - 1),
                seq=seq,
                ctx=ctx,
                support_contract=contract,
                support_side_correctness=True,
                support_side_failures=[],
                feature_by_tier={},
            )
            for tier in FEATURE_TIERS:
                item.feature_by_tier[tier] = _feature_vector(item, tier)
            items.append(item)
    return items


def _targets_to_ids(
    targets: Sequence[ScheduleTarget],
    idxs: Sequence[int],
    id_map: Mapping[str, int],
) -> np.ndarray:
    return np.asarray([int(id_map[targets[int(i)].topology_key]) for i in idxs], dtype=np.int64)


def _features(items: Sequence[WindowItem], idxs: Sequence[int], tier: str) -> np.ndarray:
    return np.stack([items[int(i)].feature_by_tier[tier] for i in idxs], axis=0).astype(np.float32, copy=False)


def _standardize_train_only(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    train = np.where(np.isfinite(train), train, 0.0)
    test = np.where(np.isfinite(test), test, 0.0)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("features must be rank-2")
    mean = np.mean(train, axis=0, keepdims=True) if train.shape[0] else np.zeros((1, train.shape[1]))
    std = np.std(train, axis=0, keepdims=True) if train.shape[0] else np.ones((1, train.shape[1]))
    keep = std > 1e-6
    std = np.where(keep, std, 1.0)
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        {
            "feature_dim": int(train.shape[1]),
            "constant_features_train": int(np.sum(~keep)),
            "normalization": "mean/std fit on train split only, then applied to train/test",
        },
    )


def _topk_order(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    return np.argsort(-arr, axis=1).astype(np.int64)


def _coverage_at_k(order: np.ndarray, y: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        mask = np.ones((y.shape[0],), dtype=bool)
    idxs = np.where(mask)[0]
    if idxs.size == 0 or order.size == 0:
        return 0.0
    kk = min(int(k), order.shape[1])
    return float(np.mean([int(y[i]) in set(int(x) for x in order[i, :kk]) for i in idxs]))


def _prediction_entropy(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return -np.sum(arr * np.log2(np.maximum(arr, EPS)), axis=1)


def _macro_top1(order: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0 or order.size == 0:
        return 0.0
    pred = order[:, 0]
    vals = []
    for cls in sorted(set(int(x) for x in y.tolist())):
        mask = y == cls
        vals.append(float(np.mean(pred[mask] == cls)) if np.any(mask) else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _eval_probs(
    *,
    probs: np.ndarray,
    y_true: np.ndarray,
    y_train: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    train_classes = set(int(x) for x in np.asarray(y_train, dtype=np.int64).tolist())
    test_classes = set(int(x) for x in y.tolist())
    seen_mask = np.asarray([int(x) in train_classes for x in y.tolist()], dtype=bool)
    order = _topk_order(probs)
    pred = order[:, 0] if order.size else np.zeros((0,), dtype=np.int64)
    entropy = _prediction_entropy(probs)

    per_class = []
    for cls in sorted(test_classes):
        mask = y == cls
        per_class.append(
            {
                "class_id": int(cls),
                "class": class_names[int(cls)],
                "train_support": int(np.sum(np.asarray(y_train, dtype=np.int64) == cls)),
                "test_support": int(np.sum(mask)),
                "seen_in_train": bool(cls in train_classes),
                "top1": float(np.mean(pred[mask] == cls)) if np.any(mask) else 0.0,
                "top2": _coverage_at_k(order, y, 2, mask=mask),
                "top3": _coverage_at_k(order, y, 3, mask=mask),
            }
        )

    confusion_counter = Counter((int(t), int(p)) for t, p in zip(y.tolist(), pred.tolist()))
    confusion = [
        {
            "true_id": int(t),
            "true": class_names[int(t)],
            "pred_id": int(p),
            "pred": class_names[int(p)],
            "count": int(c),
        }
        for (t, p), c in sorted(confusion_counter.items())
    ]
    unseen_sample_count = int(np.sum(~seen_mask))
    unseen_classes = sorted(test_classes - train_classes)
    return {
        "n": int(y.size),
        "top1": _coverage_at_k(order, y, 1),
        "top2": _coverage_at_k(order, y, 2),
        "top3": _coverage_at_k(order, y, 3),
        "seen_top1": _coverage_at_k(order, y, 1, mask=seen_mask),
        "seen_top2": _coverage_at_k(order, y, 2, mask=seen_mask),
        "seen_top3": _coverage_at_k(order, y, 3, mask=seen_mask),
        "seen_sample_count": int(np.sum(seen_mask)),
        "seen_sample_fraction": _rate(int(np.sum(seen_mask)), int(y.size)),
        "unseen_sample_count": unseen_sample_count,
        "unseen_sample_fraction": _rate(unseen_sample_count, int(y.size)),
        "unseen_class_count": int(len(unseen_classes)),
        "unseen_classes": [class_names[int(x)] for x in unseen_classes],
        "macro_accuracy": _macro_top1(order, y),
        "prediction_entropy_bits_mean": float(np.mean(entropy)) if entropy.size else 0.0,
        "prediction_entropy_bits_p95": float(np.percentile(entropy, 95)) if entropy.size else 0.0,
        "empirical_entropy_bits": _entropy_from_values(y.tolist()),
        "per_class_support": per_class,
        "confusion": confusion,
    }


def _class_weight(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=int(num_classes)).astype(np.float64)
    weights = np.zeros((int(num_classes),), dtype=np.float32)
    present = counts > 0
    if np.any(present):
        weights[present] = float(np.sum(counts[present])) / (
            float(np.sum(present)) * np.maximum(counts[present], 1.0)
        )
    return torch.as_tensor(weights, dtype=torch.float32)


def _torch_predict(
    *,
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    learner: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
) -> PredResult:
    torch.manual_seed(int(seed))
    xtr = torch.as_tensor(train_x, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.int64)
    xte = torch.as_tensor(test_x, dtype=torch.float32)
    d = int(train_x.shape[1])
    if learner == "torch_logistic":
        model: nn.Module = nn.Linear(d, int(num_classes))
    elif learner == "torch_small_mlp":
        model = nn.Sequential(
            nn.Linear(d, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )
    else:
        raise KeyError(learner)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss(weight=_class_weight(y_train, int(num_classes)))
    last_loss = 0.0
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        logits_train = model(xtr)
        logits_test = model(xte)
        probs_train = torch.softmax(logits_train, dim=1).cpu().numpy().astype(np.float64)
        probs_test = torch.softmax(logits_test, dim=1).cpu().numpy().astype(np.float64)
    pred_train = np.argmax(probs_train, axis=1).astype(np.int64)
    pred_test = np.argmax(probs_test, axis=1).astype(np.int64)
    params = int(sum(p.numel() for p in model.parameters()))
    return PredResult(
        probs_train=probs_train,
        probs_test=probs_test,
        pred_train=pred_train,
        pred_test=pred_test,
        details={
            "backend": "torch",
            "device": "cpu",
            "dtype": "float32",
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "hidden_dim": int(hidden_dim) if learner == "torch_small_mlp" else 0,
            "dropout": float(dropout) if learner == "torch_small_mlp" else 0.0,
            "parameter_count": params,
            "final_train_loss": last_loss,
            "loss_class_weighting": "inverse train class frequency; train split only",
        },
    )


def _knn_predict(
    *,
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    k: int,
) -> PredResult:
    kk = min(max(1, int(k)), int(train_x.shape[0]))

    def _predict(query_x: np.ndarray, allow_self: bool) -> np.ndarray:
        probs = np.zeros((query_x.shape[0], int(num_classes)), dtype=np.float64)
        for i, x in enumerate(query_x):
            d = np.linalg.norm(train_x - x.reshape(1, -1), axis=1) / math.sqrt(max(1, train_x.shape[1]))
            if allow_self and query_x.shape[0] == train_x.shape[0]:
                d[i] = np.inf
            nn = np.argsort(d)[:kk]
            weights = 1.0 / np.maximum(d[nn], 1e-6)
            finite = np.isfinite(weights)
            if not np.any(finite):
                weights = np.ones_like(weights)
            for cls, w in zip(y_train[nn].astype(np.int64), weights):
                probs[i, int(cls)] += float(w)
            denom = float(np.sum(probs[i]))
            probs[i] = probs[i] / denom if denom > 0.0 else (1.0 / max(1, int(num_classes)))
        return probs

    probs_train = _predict(train_x, allow_self=train_x.shape[0] > 1)
    probs_test = _predict(test_x, allow_self=False)
    return PredResult(
        probs_train=probs_train,
        probs_test=probs_test,
        pred_train=np.argmax(probs_train, axis=1).astype(np.int64),
        pred_test=np.argmax(probs_test, axis=1).astype(np.int64),
        details={"k": int(kk), "diagnostic_only": True},
    )


def _centroid_predict(
    *,
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
) -> PredResult:
    present = sorted(set(int(x) for x in np.asarray(y_train, dtype=np.int64).tolist()))
    centroids = {cls: np.mean(train_x[y_train == cls], axis=0) for cls in present}

    def _predict(query_x: np.ndarray) -> np.ndarray:
        probs = np.zeros((query_x.shape[0], int(num_classes)), dtype=np.float64)
        for i, x in enumerate(query_x):
            logits = np.full((int(num_classes),), -1e9, dtype=np.float64)
            for cls, center in centroids.items():
                logits[cls] = -float(np.linalg.norm(x - center) / math.sqrt(max(1, x.shape[0])))
            logits -= np.max(logits)
            p = np.exp(logits)
            denom = float(np.sum(p))
            probs[i] = p / denom if denom > 0.0 else (1.0 / max(1, int(num_classes)))
        return probs

    probs_train = _predict(train_x)
    probs_test = _predict(test_x)
    return PredResult(
        probs_train=probs_train,
        probs_test=probs_test,
        pred_train=np.argmax(probs_train, axis=1).astype(np.int64),
        pred_test=np.argmax(probs_test, axis=1).astype(np.int64),
        details={"diagnostic_only": True},
    )


def _split_overlap(
    *,
    split: SplitDef,
    targets: Sequence[ScheduleTarget],
) -> Dict[str, Any]:
    train_counts = Counter(targets[int(i)].topology_key for i in split.train_idx)
    test_counts = Counter(targets[int(i)].topology_key for i in split.test_idx)
    train_classes = set(train_counts)
    test_classes = set(test_counts)
    overlap = sorted(train_classes & test_classes)
    unseen = sorted(test_classes - train_classes)
    return {
        "name": split.name,
        "kind": split.kind,
        "diagnostic_only": bool(split.kind == "random_optimistic_diagnostic"),
        "train_indices_contract": {
            "shape_contract": "[N_train]",
            "actual_shape": [int(len(split.train_idx))],
            "dtype": "int64",
            "device": "cpu",
        },
        "test_indices_contract": {
            "shape_contract": "[N_test]",
            "actual_shape": [int(len(split.test_idx))],
            "dtype": "int64",
            "device": "cpu",
        },
        "train_n": int(len(split.train_idx)),
        "test_n": int(len(split.test_idx)),
        "low_n_diagnostic": bool(split.low_n_diagnostic),
        "note": split.note,
        "train_per_clip": dict(Counter()),
        "train_topology_class_count": int(len(train_classes)),
        "test_topology_class_count": int(len(test_classes)),
        "overlap_topology_class_count": int(len(overlap)),
        "unseen_topology_class_count": int(len(unseen)),
        "unseen_topologies": unseen,
        "train_topology_support": dict(train_counts),
        "test_topology_support": dict(test_counts),
    }


def _feature_contract(items: Sequence[WindowItem], targets: Sequence[ScheduleTarget]) -> Dict[str, Any]:
    ex = items[0]
    target = targets[0]
    out = {}
    for tier in FEATURE_TIERS:
        feat = np.asarray(ex.feature_by_tier[tier], dtype=np.float32)
        out[tier] = {
            "shape_contract": "[D]",
            "actual_shape": [int(feat.shape[0])],
            "dtype": str(feat.dtype),
            "device": "cpu",
            "finite": bool(np.isfinite(feat).all()),
            "role": FEATURE_SPECS[tier].role,
            "decision_eligible": bool(FEATURE_SPECS[tier].decision_eligible),
            "leaky_upper_bound": bool(FEATURE_SPECS[tier].leaky_upper_bound),
            "description": FEATURE_SPECS[tier].description,
        }
    return {
        "base_available_feature": out["base_available"],
        "non_leaky_phase_support_feature": out["non_leaky_ctx_history_plus_command"],
        "leaky_upper_bound_feature": out["leaky_upper_bound_oracle_phase_support"],
        "all_feature_tiers": out,
        "topology_target": {
            "shape_contract": "scalar",
            "actual_shape": [],
            "dtype": "int64",
            "device": "cpu",
            "example_class_id": 0,
            "meaning": "class id for debounced support topology with durations removed",
        },
        "topology_tokens": {
            "shape_contract": "variable-length labels",
            "actual_shape": [int(len(target.topology_tokens))],
            "dtype": "object/string",
            "device": "cpu",
            "example": list(target.topology_tokens),
        },
    }


def _leak_audit() -> Dict[str, Any]:
    return {
        "global_forbidden_for_decision_features": [
            "oracle future support/contact schedule beyond selected seam support",
            "future endpoint contact channels",
            "future phase labels or future support run labels",
            "future bone_angvel/root dynamics witness",
            "topology target id/string or normalized topology tokens as inputs",
            "test split statistics in scaling/normalization",
            "random split as architecture evidence",
        ],
        "tiers": {
            "base_available": {
                "allowed_sources": [
                    "previous available_context baseline",
                    "start ctx [C,281] float32/cpu",
                    "commanded cond_dir [H,2] and yaw cue [H,1]",
                    "soft endpoint state prefix excluding contact channels",
                    "selected start/seam support one-hot cue",
                ],
                "forbidden_sources": [
                    "future endpoint contact channels",
                    "oracle future support schedule after seam",
                    "future bone_angvel witness",
                    "topology target",
                ],
                "decision_role": "baseline only; runtime-available under previous probe contract",
            },
            "non_leaky_ctx_history": {
                "allowed_sources": [
                    "start ctx [C,281] only",
                    "past contact ctx[:,279:281] [C,2]",
                    "past ego_vel ctx[:,276:278] [C,2]",
                    "past yaw_rate ctx[:,278:279] [C,1]",
                    "past pose phase proxy from ctx[:,0:276] low-dim summaries",
                    "ctx last support label, recent support run length, recent support transition count",
                ],
                "forbidden_sources": [
                    "horizon support/contact labels",
                    "soft endpoint cue",
                    "commanded future cond_dir/yaw cue",
                    "future bone_angvel",
                    "topology target",
                ],
                "decision_role": "non-leaky runtime history feature",
            },
            "non_leaky_ctx_history_plus_endpoint": {
                "allowed_sources": [
                    "all non_leaky_ctx_history sources",
                    "soft endpoint cue seq[-1,:279] float32/cpu",
                    "selected seam support one-hot cue",
                ],
                "forbidden_sources": [
                    "endpoint contact channels seq[-1,279:281]",
                    "future support/contact schedule after seam",
                    "future phase labels",
                    "future bone_angvel",
                    "topology target",
                ],
                "endpoint_contact_channels_included": False,
                "decision_role": "non-leaky endpoint-conditioned feature",
            },
            "non_leaky_ctx_history_plus_command": {
                "allowed_sources": [
                    "all non_leaky_ctx_history_plus_endpoint sources",
                    "commanded cond_dir [H,2] cue",
                    "commanded yaw/yaw_rate [H,1] cue",
                ],
                "forbidden_sources": [
                    "oracle future support/contact schedule after seam",
                    "future endpoint contact channels",
                    "future bone_angvel",
                    "topology target",
                    "using yaw/cond_dir as prediction target",
                ],
                "yaw_cond_dir_role": "commanded cue only, never prediction target",
                "decision_role": "primary non-leaky decision tier",
            },
            "leaky_upper_bound_oracle_phase_support": {
                "allowed_sources": [
                    "oracle middle normalized support labels [H]",
                    "future support/contact schedule [H,2]",
                    "future support run phase sin/cos",
                    "future support stats and first/last labels",
                    "future bone_angvel witness [H,138]",
                ],
                "forbidden_sources": [
                    "architecture decision based on this tier",
                    "runtime condition claim",
                    "topology granularity reduction based on accuracy",
                ],
                "decision_role": "leaky_upper_bound only; answers oracle separability",
            },
            "random_optimistic_diagnostic": {
                "allowed_sources": [
                    "same feature tensors as the selected tier",
                    "random train/test split for leakage/overlap diagnostic only",
                ],
                "forbidden_sources": [
                    "architecture decision",
                    "condition-missing conclusion",
                ],
                "decision_role": "diagnostic only",
            },
        },
    }


def _load_previous_calibration(path: Path) -> Dict[str, Any]:
    expected = {
        "expected_previous_available_context_multi_fraction": 0.2917,
        "expected_previous_topology_bucket_entropy_bits_weighted": 0.1357,
    }
    if not path.is_file():
        return {"loaded": False, "path": str(path), **expected}
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("empirical_entropy_calibration", {}) or {}
    return {
        "loaded": True,
        "path": str(path),
        "previous_available_context_multi_fraction": float(
            cal.get("empirical_available_context_multi_fraction", expected["expected_previous_available_context_multi_fraction"])
        ),
        "previous_topology_bucket_entropy_bits_weighted": float(
            cal.get(
                "current_topology_bucket_entropy_bits_weighted",
                expected["expected_previous_topology_bucket_entropy_bits_weighted"],
            )
        ),
        "previous_flat_signature_bucket_entropy_bits_weighted": float(
            (cal.get("current_flat_signature_bucket_entropy", {}) or {}).get("entropy_bits_mean_weighted", 0.0)
        ),
        **expected,
    }


def _row_from_result(
    *,
    split: SplitDef,
    tier: str,
    learner: str,
    train_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    details: Mapping[str, Any],
    prep: Mapping[str, Any],
) -> Dict[str, Any]:
    feature_spec = FEATURE_SPECS[tier]
    learner_spec = LEARNER_SPECS[learner]
    return {
        "split": split.name,
        "split_kind": split.kind,
        "split_diagnostic_only": bool(split.kind == "random_optimistic_diagnostic"),
        "feature_tier": tier,
        "feature_role": feature_spec.role,
        "feature_leaky_upper_bound": bool(feature_spec.leaky_upper_bound),
        "feature_decision_eligible": bool(feature_spec.decision_eligible),
        "learner": learner,
        "learner_role": learner_spec.role,
        "learner_diagnostic_only": bool(learner_spec.diagnostic_only),
        "decision_eligible": bool(
            feature_spec.decision_eligible
            and learner_spec.decision_eligible
            and split.kind != "random_optimistic_diagnostic"
        ),
        "train_n": int(train_metrics["n"]),
        "test_n": int(test_metrics["n"]),
        "feature_dim": int(prep.get("feature_dim", 0)),
        "constant_features_train": int(prep.get("constant_features_train", 0)),
        "train_top1": float(train_metrics["top1"]),
        "train_top2": float(train_metrics["top2"]),
        "train_top3": float(train_metrics["top3"]),
        "train_macro_accuracy": float(train_metrics["macro_accuracy"]),
        "test_top1": float(test_metrics["top1"]),
        "test_top2": float(test_metrics["top2"]),
        "test_top3": float(test_metrics["top3"]),
        "seen_top1": float(test_metrics["seen_top1"]),
        "seen_top2": float(test_metrics["seen_top2"]),
        "seen_top3": float(test_metrics["seen_top3"]),
        "macro_accuracy": float(test_metrics["macro_accuracy"]),
        "prediction_entropy_bits_mean": float(test_metrics["prediction_entropy_bits_mean"]),
        "prediction_entropy_bits_p95": float(test_metrics["prediction_entropy_bits_p95"]),
        "empirical_test_entropy_bits": float(test_metrics["empirical_entropy_bits"]),
        "seen_sample_count": int(test_metrics["seen_sample_count"]),
        "seen_sample_fraction": float(test_metrics["seen_sample_fraction"]),
        "unseen_sample_count": int(test_metrics["unseen_sample_count"]),
        "unseen_sample_fraction": float(test_metrics["unseen_sample_fraction"]),
        "unseen_class_count": int(test_metrics["unseen_class_count"]),
        "parameter_count": int(details.get("parameter_count", 0)),
        "epochs": int(details.get("epochs", 0)),
        "final_train_loss": float(details.get("final_train_loss", 0.0)),
        "details": dict(details),
        "per_class_support": list(test_metrics["per_class_support"]),
        "confusion": list(test_metrics["confusion"]),
    }


def _evaluate_all(
    *,
    items: Sequence[WindowItem],
    targets: Sequence[ScheduleTarget],
    topology_id: Mapping[str, int],
    topology_names: Sequence[str],
    splits: Sequence[SplitDef],
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
    knn_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_i, split in enumerate(splits):
        if not split.train_idx or not split.test_idx:
            continue
        y_train = _targets_to_ids(targets, split.train_idx, topology_id)
        y_test = _targets_to_ids(targets, split.test_idx, topology_id)
        for tier_i, tier in enumerate(FEATURE_TIERS):
            raw_train_x = _features(items, split.train_idx, tier)
            raw_test_x = _features(items, split.test_idx, tier)
            train_x, test_x, prep = _standardize_train_only(raw_train_x, raw_test_x)

            for learner_i, learner in enumerate(
                ("torch_logistic", "torch_small_mlp", "diagnostic_knn", "diagnostic_nearest_centroid")
            ):
                run_seed = int(seed) + split_i * 1000 + tier_i * 100 + learner_i
                if learner in {"torch_logistic", "torch_small_mlp"}:
                    pred = _torch_predict(
                        train_x=train_x,
                        y_train=y_train,
                        test_x=test_x,
                        num_classes=len(topology_names),
                        learner=learner,
                        epochs=int(epochs),
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        hidden_dim=int(hidden_dim),
                        dropout=float(dropout),
                        seed=run_seed,
                    )
                elif learner == "diagnostic_knn":
                    pred = _knn_predict(
                        train_x=train_x,
                        y_train=y_train,
                        test_x=test_x,
                        num_classes=len(topology_names),
                        k=int(knn_k),
                    )
                else:
                    pred = _centroid_predict(
                        train_x=train_x,
                        y_train=y_train,
                        test_x=test_x,
                        num_classes=len(topology_names),
                    )
                train_metrics = _eval_probs(
                    probs=pred.probs_train,
                    y_true=y_train,
                    y_train=y_train,
                    class_names=topology_names,
                )
                test_metrics = _eval_probs(
                    probs=pred.probs_test,
                    y_true=y_test,
                    y_train=y_train,
                    class_names=topology_names,
                )
                rows.append(
                    _row_from_result(
                        split=split,
                        tier=tier,
                        learner=learner,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        details=pred.details,
                        prep=prep,
                    )
                )
    return rows


def _random_gap_diagnostic(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("learner") not in {"torch_logistic", "torch_small_mlp"}:
            continue
        key = (str(row.get("feature_tier")), str(row.get("learner")))
        by_key[key][str(row.get("split"))] = row
    out: List[Dict[str, Any]] = []
    for (tier, learner), recs in sorted(by_key.items()):
        block = recs.get("contiguous_block")
        rand = recs.get("random_optimistic_diagnostic")
        if not block or not rand:
            continue
        out.append(
            {
                "feature_tier": tier,
                "learner": learner,
                "diagnostic_only": True,
                "random_minus_contiguous_top1": float(rand.get("test_top1", 0.0)) - float(block.get("test_top1", 0.0)),
                "random_minus_contiguous_top3": float(rand.get("test_top3", 0.0)) - float(block.get("test_top3", 0.0)),
                "random_minus_contiguous_seen_top3": float(rand.get("seen_top3", 0.0))
                - float(block.get("seen_top3", 0.0)),
                "interpretation": "overlap/leakage diagnostic only; not architecture evidence",
            }
        )
    return out


def _mean_row_value(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def _decision(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    thresholds = {
        "condition_top3_all_main_splits_gte": 0.80,
        "condition_seen_top3_all_main_splits_gte": 0.90,
        "train_high_top1_gte": 0.85,
        "train_low_top1_lt": 0.70,
        "test_low_top3_lt": 0.80,
        "unseen_high_fraction_gte": 0.20,
        "unseen_low_fraction_lte": 0.05,
        "leaky_seen_top3_high_gte": 0.90,
        "leaky_minus_nonleaky_seen_top3_gap_gte": 0.20,
    }
    main_rows = [
        r
        for r in rows
        if r.get("split_kind") != "random_optimistic_diagnostic"
        and r.get("learner") in {"torch_logistic", "torch_small_mlp"}
    ]
    primary_tier = "non_leaky_ctx_history_plus_command"
    eligible = [r for r in main_rows if r.get("feature_tier") == primary_tier]
    by_learner: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_learner[str(row.get("learner"))].append(row)

    condition_sufficient_by_learner: Dict[str, bool] = {}
    for learner, learner_rows in by_learner.items():
        if not learner_rows:
            condition_sufficient_by_learner[learner] = False
            continue
        condition_sufficient_by_learner[learner] = bool(
            all(float(r.get("test_top3", 0.0)) >= thresholds["condition_top3_all_main_splits_gte"] for r in learner_rows)
            and all(
                float(r.get("seen_top3", 0.0)) >= thresholds["condition_seen_top3_all_main_splits_gte"]
                for r in learner_rows
            )
        )

    mlp_rows = by_learner.get("torch_small_mlp", [])
    if not mlp_rows:
        mlp_rows = [r for r in eligible if r.get("learner") == "torch_logistic"]
    train_high = bool(_mean_row_value(mlp_rows, "train_top1") >= thresholds["train_high_top1_gte"])
    train_low = bool(_mean_row_value(mlp_rows, "train_top1") < thresholds["train_low_top1_lt"])
    test_low = bool(any(float(r.get("test_top3", 0.0)) < thresholds["test_low_top3_lt"] for r in mlp_rows))
    unseen_high = bool(
        any(float(r.get("unseen_sample_fraction", 0.0)) >= thresholds["unseen_high_fraction_gte"] for r in mlp_rows)
    )
    unseen_low = bool(
        all(float(r.get("unseen_sample_fraction", 0.0)) <= thresholds["unseen_low_fraction_lte"] for r in mlp_rows)
        if mlp_rows
        else False
    )

    leaky_rows = [
        r
        for r in main_rows
        if r.get("feature_tier") == "leaky_upper_bound_oracle_phase_support"
        and r.get("learner") == "torch_small_mlp"
    ]
    nonleaky_rows = [r for r in mlp_rows if r.get("feature_tier") == primary_tier]
    leaky_seen_top3 = _mean_row_value(leaky_rows, "seen_top3")
    nonleaky_seen_top3 = _mean_row_value(nonleaky_rows, "seen_top3")
    leaky_high_nonleaky_low = bool(
        leaky_seen_top3 >= thresholds["leaky_seen_top3_high_gte"]
        and (leaky_seen_top3 - nonleaky_seen_top3)
        >= thresholds["leaky_minus_nonleaky_seen_top3_gap_gte"]
    )
    leaky_all_top3 = _mean_row_value(leaky_rows, "test_top3")
    nonleaky_all_top3 = _mean_row_value(nonleaky_rows, "test_top3")

    if any(condition_sufficient_by_learner.values()):
        primary = "condition_sufficient_previous_knn_or_centroid_learner_weak"
        next_step = "enter_layer1_topology_head_deterministic_vs_small_categorical_decision"
        reason = "non-leaky plus-command true learner clears top3/seen-top3 thresholds on all main splits"
    elif train_high and test_low and unseen_high:
        primary = "data_coverage_insufficient_expand_clips_no_generator"
        next_step = "expand clips or coverage before any generator"
        reason = "true learner fits train but blocked/leave-clip test is low with high unseen topology fraction"
    elif leaky_high_nonleaky_low:
        primary = "runtime_available_condition_missing_phase_support_cue"
        next_step = "refine non-leaky runtime phase/support history cue; do not train diffusion"
        reason = "oracle future phase/support is separable but non-leaky condition is not"
    elif train_high and test_low and unseen_low:
        primary = "generalization_or_feature_construction_problem"
        next_step = "refine non-leaky feature construction"
        reason = "true learner fits train, test remains low, and unseen topology is low"
    elif train_low:
        primary = "feature_insufficient_or_topology_target_too_fine"
        next_step = "inspect decoder need for topology distinctions before any granularity change"
        reason = "MLP train accuracy is low; do not reduce topology granularity for accuracy alone"
    else:
        primary = "inconclusive_hold_layer1"
        next_step = "inspect per-class support and feature construction"
        reason = "pre-registered thresholds did not isolate a single failure family"

    return {
        "decision_rules_preregistered": {
            "rule_1_condition_enough": (
                "If a true learner on non_leaky_ctx_history_plus_command has top3 >= 0.80 and "
                "seen-class top3 >= 0.90 on contiguous-block and leave-clip-out, condition is enough; "
                "previous kNN/centroid was too weak; move only to deterministic vs small categorical topology head."
            ),
            "rule_2_data_coverage": (
                "If true learner train acc is high, blocked/leave-clip-out test is low, and unseen topology is high, "
                "data coverage is insufficient; expand clips, do not train generator."
            ),
            "rule_3_generalization_feature": (
                "If true learner train acc is high and test is low but unseen topology is low, refine non-leaky features."
            ),
            "rule_4_train_low": (
                "If true learner train acc is low, feature is insufficient or topology target may be too fine; "
                "do not coarsen topology unless decoder does not need the distinction."
            ),
            "rule_5_leaky_high_nonleaky_low": (
                "If leaky upper-bound is high and non-leaky is low, oracle phase/support is separable but runtime "
                "condition misses phase/support cue; do not jump to diffusion."
            ),
            "rule_6_nonleaky_high": "If non-leaky is high, enter layer-1 topology head modeling only.",
            "rule_7_no_diffusion_required": "No layer-1 result may be written as diffusion required.",
            "numeric_thresholds": thresholds,
        },
        "primary_decision": primary,
        "next_step": next_step,
        "reason": reason,
        "condition_sufficient_by_learner": condition_sufficient_by_learner,
        "primary_nonleaky_mlp_main_split_mean": {
            "train_top1": _mean_row_value(mlp_rows, "train_top1"),
            "test_top1": _mean_row_value(mlp_rows, "test_top1"),
            "test_top3": _mean_row_value(mlp_rows, "test_top3"),
            "seen_top3": _mean_row_value(mlp_rows, "seen_top3"),
            "unseen_sample_fraction_max": float(
                max([float(r.get("unseen_sample_fraction", 0.0)) for r in mlp_rows], default=0.0)
            ),
        },
        "leaky_vs_nonleaky": {
            "leaky_mlp_seen_top3_mean": leaky_seen_top3,
            "nonleaky_plus_command_mlp_seen_top3_mean": nonleaky_seen_top3,
            "seen_top3_gap": leaky_seen_top3 - nonleaky_seen_top3,
            "leaky_mlp_all_top3_mean": leaky_all_top3,
            "nonleaky_plus_command_mlp_all_top3_mean": nonleaky_all_top3,
            "all_top3_gap": leaky_all_top3 - nonleaky_all_top3,
            "leaky_high_nonleaky_low": leaky_high_nonleaky_low,
            "architecture_decision_allowed_from_leaky": False,
        },
        "diffusion_statement": "Layer-1 conclusion is never diffusion required in this probe.",
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "split_kind",
        "split_diagnostic_only",
        "feature_tier",
        "feature_role",
        "feature_leaky_upper_bound",
        "learner",
        "learner_role",
        "learner_diagnostic_only",
        "decision_eligible",
        "train_n",
        "test_n",
        "feature_dim",
        "parameter_count",
        "epochs",
        "train_top1",
        "train_top2",
        "train_top3",
        "train_macro_accuracy",
        "test_top1",
        "test_top2",
        "test_top3",
        "seen_top1",
        "seen_top2",
        "seen_top3",
        "macro_accuracy",
        "prediction_entropy_bits_mean",
        "prediction_entropy_bits_p95",
        "empirical_test_entropy_bits",
        "seen_sample_count",
        "seen_sample_fraction",
        "unseen_sample_count",
        "unseen_sample_fraction",
        "unseen_class_count",
        "final_train_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_confusion_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "split_kind", "feature_tier", "learner", "true", "pred", "count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for c in row.get("confusion", []) or []:
                writer.writerow(
                    {
                        "split": row.get("split"),
                        "split_kind": row.get("split_kind"),
                        "feature_tier": row.get("feature_tier"),
                        "learner": row.get("learner"),
                        "true": c.get("true"),
                        "pred": c.get("pred"),
                        "count": c.get("count"),
                    }
                )


def _topology_table_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        r
        for r in rows
        if r.get("learner") in {"torch_logistic", "torch_small_mlp"}
        and r.get("feature_tier")
        in {
            "base_available",
            "non_leaky_ctx_history",
            "non_leaky_ctx_history_plus_endpoint",
            "non_leaky_ctx_history_plus_command",
            "leaky_upper_bound_oracle_phase_support",
        }
    ]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only support topology learner-vs-condition ablation.")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--previous-baseline-summary", type=Path, default=DEFAULT_PREVIOUS_BASELINE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--block-gap", type=int, default=None)
    p.add_argument("--low-n-threshold", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--no-random-diagnostic", action="store_true")
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = int(args.horizon)
    block_gap = int(args.block_gap) if args.block_gap is not None else max(0, h // 2)

    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    all_items = _build_items(
        clips,
        horizon=h,
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=max(1, int(args.stride)),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    unmatched_items = [it for it in all_items if it.clip == UNMATCHED_TARGET]
    if not main_items:
        raise RuntimeError("no matched target windows found")

    targets = [_target_from_item(it) for it in main_items]
    topology_id, topology_names = _class_id_maps([t.topology_key for t in targets])
    topology_values = [t.topology_key for t in targets]
    timing_values = [t.timing_key for t in targets]

    splits = _build_splits(
        main_items,
        train_fraction=float(args.train_fraction),
        block_gap=block_gap,
        seed=int(args.seed),
        low_n_threshold=int(args.low_n_threshold),
        include_random=not bool(args.no_random_diagnostic),
    )
    split_summaries = []
    for split in splits:
        rec = _split_overlap(split=split, targets=targets)
        rec["train_per_clip"] = dict(Counter(main_items[int(i)].clip for i in split.train_idx))
        rec["test_per_clip"] = dict(Counter(main_items[int(i)].clip for i in split.test_idx))
        split_summaries.append(rec)

    rows = _evaluate_all(
        items=main_items,
        targets=targets,
        topology_id=topology_id,
        topology_names=topology_names,
        splits=splits,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        seed=int(args.seed),
        knn_k=int(args.knn_k),
    )

    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    walk_l_to_r = _walk_l_to_r_report(
        clips=clips,
        matched_pairs=two_frame.get("matched_pairs", {}) or {},
        horizon=h,
        pose_topk=int(args.pose_topk),
        ground_contact_thr=float(args.ground_contact_thr),
        ground_pose_thr=float(args.ground_pose_thr),
        min_run_frames=int(args.min_run_frames),
    )
    previous = _load_previous_calibration(Path(args.previous_baseline_summary))
    random_gap = _random_gap_diagnostic(rows)
    decision = _decision(rows)
    target_counts = Counter(topology_values)

    payload = {
        "task": "support_topology_learner_condition_ablation",
        "scope": (
            "read-only lightweight CPU topology classifiers; no full trajectory generator training; "
            "no production trainer/runtime/gate forward or edit; no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "two_frame_summary": str(args.two_frame_summary),
            "previous_baseline_summary": str(args.previous_baseline_summary),
            "horizon": h,
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "train_fraction": float(args.train_fraction),
            "block_gap": int(block_gap),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "knn_k": int(args.knn_k),
            "device_policy": "cpu only",
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "input_output_contract": _feature_contract(main_items, targets),
        "leak_audit": _leak_audit(),
        "target_contract": {
            "topology_definition": "debounced support event order with durations removed",
            "target_dtype": "int64",
            "target_device": "cpu",
            "topology_class_count": int(len(topology_names)),
            "topology_class_names": list(topology_names),
            "topology_counts": dict(target_counts),
            "topology_tokens_role": "variable-length cpu labels; target metadata only",
            "yaw_cond_dir_role": "commanded cue only, not prediction target",
            "topology_granularity_policy": (
                "not changed for accuracy; can only change if trajectory decoder does not need the distinction"
            ),
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "unmatched_out_of_scope": UNMATCHED_TARGET,
            "main_window_count": int(len(main_items)),
            "unmatched_window_count": int(len(unmatched_items)),
            "per_clip_windows_all": dict(Counter(it.clip for it in all_items)),
            "topology_entropy_bits": _entropy_from_values(topology_values),
            "timing_entropy_bits_conditional_on_topology": _conditional_entropy(timing_values, topology_values),
        },
        "splits": split_summaries,
        "entropy_calibration": {
            **previous,
            "current_prediction_entropy_is_reported_per_row": True,
            "alignment_statement": (
                "prediction entropy is interpreted against previous empirical topology bucket entropy 0.1357 "
                "and available-context multi fraction 0.2917; high random-vs-blocked gap is leakage/overlap only"
            ),
        },
        "learner_results": rows,
        "random_vs_blocked_gap_diagnostic": random_gap,
        "decision": decision,
        "walk_l_to_r": walk_l_to_r,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_full_middle_trajectory_generator": False,
            "forwarded_production_runtime_or_trainer": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "continued_endpoint_yaw_discriminator_instrumentation": False,
            "used_yaw_or_cond_dir_as_prediction_target": False,
            "used_random_split_for_architecture_decision": False,
            "used_knn_or_nearest_centroid_as_architecture_decision_learner": False,
            "used_leaky_upper_bound_as_runtime_condition": False,
            "reduced_topology_granularity_for_accuracy": False,
            "wrote_diffusion_required_conclusion": False,
        },
        "artifacts": {
            "summary_json": str(out_dir / "support_topology_learner_condition_ablation_summary.json"),
            "summary_md": str(out_dir / "support_topology_learner_condition_ablation_summary.md"),
            "rows_csv": str(out_dir / "support_topology_learner_condition_ablation_rows.csv"),
            "confusion_csv": str(out_dir / "support_topology_learner_condition_ablation_confusion.csv"),
        },
    }

    _dump_json(out_dir / "support_topology_learner_condition_ablation_summary.json", payload)
    _write_rows_csv(out_dir / "support_topology_learner_condition_ablation_rows.csv", rows)
    _write_confusion_csv(out_dir / "support_topology_learner_condition_ablation_confusion.csv", rows)

    lines: List[str] = []
    lines.append("# Support Topology Learner-Condition Ablation")
    lines.append("")
    lines.append(
        "Read-only lightweight CPU classifier ablation. No full trajectory generator training, "
        "no production runtime/trainer/gate forward or edit, no checkpoint mutation."
    )
    lines.append("")
    lines.append("## Leak Audit")
    lines.append("")
    lines.append("- non-leaky decision tiers: `base_available`, `non_leaky_ctx_history`, `non_leaky_ctx_history_plus_endpoint`, `non_leaky_ctx_history_plus_command`.")
    lines.append("- `non_leaky_ctx_history` uses only start ctx `[C,281]`: past contact `[C,2]`, ego_vel `[C,2]`, yaw_rate `[C,1]`, pose proxy, and recent support history.")
    lines.append("- endpoint tier adds soft endpoint `seq[-1,:279]` and selected seam support; endpoint contact channels `seq[-1,279:281]` are excluded.")
    lines.append("- command tier adds commanded `cond_dir/yaw` cue only; yaw/cond_dir are never targets.")
    lines.append("- `leaky_upper_bound_oracle_phase_support` uses future support labels/contact phase/bone_angvel and is diagnostic upper-bound only.")
    lines.append("")
    lines.append("## Dataset / Splits")
    lines.append("")
    lines.append(f"- main matched windows: `{len(main_items)}` from `{MATCHED_TARGETS}`")
    lines.append(f"- unmatched Walk_L_To_R windows: `{len(unmatched_items)}` diagnostic/out-of-scope only")
    for s in split_summaries:
        lines.append(
            f"- {s['name']}: train `{s['train_n']}`, test `{s['test_n']}`, "
            f"unseen test samples `{sum(s['test_topology_support'].get(x, 0) for x in s['unseen_topologies'])}`, "
            f"unseen topologies `{s['unseen_topology_class_count']}`; {s['note']}"
        )
    lines.append("")
    lines.append("## True Learners")
    lines.append("")
    lines.append("| split | tier | learner | train top1 | test top1 | test top3 | seen top3 | unseen n | entropy |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in _topology_table_rows(rows):
        if r.get("learner") not in {"torch_logistic", "torch_small_mlp"}:
            continue
        suffix = " (random diag)" if r.get("split_kind") == "random_optimistic_diagnostic" else ""
        lines.append(
            f"| {r['split']}{suffix} | {r['feature_tier']} | {r['learner']} | "
            f"{_fmt(r['train_top1'])} | {_fmt(r['test_top1'])} | {_fmt(r['test_top3'])} | "
            f"{_fmt(r['seen_top3'])} | {r['unseen_sample_count']} | "
            f"{_fmt(r['prediction_entropy_bits_mean'])} |"
        )
    lines.append("")
    lines.append("## Random Gap Diagnostic")
    lines.append("")
    lines.append("| tier | learner | random-contig top3 | random-contig seen top3 |")
    lines.append("|---|---|---:|---:|")
    for r in random_gap:
        lines.append(
            f"| {r['feature_tier']} | {r['learner']} | "
            f"{_fmt(r['random_minus_contiguous_top3'])} | "
            f"{_fmt(r['random_minus_contiguous_seen_top3'])} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- primary decision: `{decision['primary_decision']}`")
    lines.append(f"- reason: {decision['reason']}")
    lines.append(f"- next step: `{decision['next_step']}`")
    lines.append(f"- leaky vs non-leaky seen top3 gap: `{_fmt(decision['leaky_vs_nonleaky']['seen_top3_gap'])}`")
    lines.append("- layer-1 conclusion is not `diffusion required`.")
    lines.append("")
    lines.append("## Walk_L_To_R")
    lines.append("")
    lines.append(f"- matched_pair_available: `{bool(walk_l_to_r['matched_pair_available'])}`")
    lines.append(f"- pose_d: `{_fmt(walk_l_to_r['pose_d'])}`")
    lines.append(f"- contact_d: `{_fmt(walk_l_to_r['contact_d'])}`")
    lines.append(f"- seam_support: `{walk_l_to_r['seam_support']}`")
    lines.append(f"- horizon_support: `{walk_l_to_r['horizon_support']}`")
    lines.append(f"- unmatched reason: `{walk_l_to_r['ungroundable_reason']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for path in payload["artifacts"].values():
        lines.append(f"- `{path}`")
    _dump_md(out_dir / "support_topology_learner_condition_ablation_summary.md", lines)

    print(f"wrote {out_dir / 'support_topology_learner_condition_ablation_summary.md'}")
    print(f"wrote {out_dir / 'support_topology_learner_condition_ablation_summary.json'}")
    print(f"wrote {out_dir / 'support_topology_learner_condition_ablation_rows.csv'}")
    print(f"wrote {out_dir / 'support_topology_learner_condition_ablation_confusion.csv'}")


if __name__ == "__main__":
    main()
