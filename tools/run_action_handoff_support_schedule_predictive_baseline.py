#!/usr/bin/env python3
"""Predictive baselines for support/event schedules.

Read-only feasibility probe. This script fits lightweight numpy/sklearn
baselines over existing continuous target windows. It does not train a full
trajectory generator, forward production runtime/trainer, mutate checkpoints,
or edit any production gate.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTEXT_LEN_C,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_TOPK,
    TURN_CLIPS,
    WALK_F,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _evaluate_sequence,
    _fmt,
    _load_clips,
    _load_skeleton_meta,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    LABEL_ORDER,
    WindowItem,
    _build_window_items,
    _calibrate_support_side_bands,
    _cluster_items,
    _evaluate_support_side_correctness,
    _support_contract,
    _support_side_features,
    _walk_l_to_r_report,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
    _replace_seq,
)


MATCHED_TARGETS = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")
UNMATCHED_TARGET = "Walk_L_To_R"
DEFAULT_PREVIOUS_SUMMARY = Path(
    "debug_output/_tmp_action_handoff_support_contract_tightening_20260602/"
    "support_contract_tightening_summary.json"
)
DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_support_schedule_predictive_baseline_20260602")
CONTACT_BY_LABEL = {
    "right": np.asarray([1.0, 0.0], dtype=np.float32),
    "left": np.asarray([0.0, 1.0], dtype=np.float32),
    "dual": np.asarray([1.0, 1.0], dtype=np.float32),
    "flight_or_unknown": np.asarray([0.0, 0.0], dtype=np.float32),
}
EPS = 1e-8


@dataclass(frozen=True)
class ScheduleTarget:
    raw_labels: Tuple[str, ...]
    normalized_labels: Tuple[str, ...]
    topology_tokens: Tuple[str, ...]
    durations: Tuple[int, ...]
    event_tokens: Tuple[int, ...]
    topology_key: str
    timing_key: str
    raw_signature: str
    normalized_signature: str
    ambiguous_count: int
    merge_count: int


@dataclass(frozen=True)
class SplitDef:
    name: str
    kind: str
    train_idx: Tuple[int, ...]
    test_idx: Tuple[int, ...]
    low_n_diagnostic: bool
    note: str


@dataclass(frozen=True)
class ClassifierPred:
    model: str
    probs: np.ndarray
    pred: np.ndarray
    details: Mapping[str, Any]


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _entropy_from_counts(counts: Iterable[int]) -> float:
    arr = np.asarray([int(x) for x in counts if int(x) > 0], dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log2(np.maximum(p, EPS))))


def _entropy_from_values(values: Sequence[Any]) -> float:
    return _entropy_from_counts(Counter(values).values())


def _conditional_entropy(child: Sequence[Any], parent: Sequence[Any]) -> float:
    if len(child) != len(parent) or not child:
        return 0.0
    total = len(child)
    by_parent: Dict[Any, List[Any]] = defaultdict(list)
    for c, p in zip(child, parent):
        by_parent[p].append(c)
    out = 0.0
    for vals in by_parent.values():
        out += (len(vals) / total) * _entropy_from_values(vals)
    return float(out)


def _rle(labels: Sequence[str]) -> List[Tuple[str, int]]:
    if not labels:
        return []
    out: List[Tuple[str, int]] = []
    cur = str(labels[0])
    length = 1
    for label in labels[1:]:
        label = str(label)
        if label == cur:
            length += 1
            continue
        out.append((cur, length))
        cur = label
        length = 1
    out.append((cur, length))
    return out


def _target_from_item(item: WindowItem) -> ScheduleTarget:
    c = item.support_contract
    raw = tuple(str(x) for x in c["schedule_label_sequence"])
    norm = tuple(str(x) for x in c["normalized_label_sequence"])
    runs = _rle(norm)
    topology = tuple(label for label, _ in runs)
    durations = tuple(int(length) for _, length in runs)
    topology_key = ">".join(topology) if topology else "empty"
    timing_key = ",".join(str(x) for x in durations) if durations else "empty"
    token_map = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    norm_block = c.get("normalization", {}) or {}
    return ScheduleTarget(
        raw_labels=raw,
        normalized_labels=norm,
        topology_tokens=topology,
        durations=durations,
        event_tokens=tuple(int(token_map.get(x, -1)) for x in norm),
        topology_key=topology_key,
        timing_key=timing_key,
        raw_signature=str(c["raw_signature"]),
        normalized_signature=str(c["normalized_signature"]),
        ambiguous_count=len(norm_block.get("ambiguous_runs", []) or []),
        merge_count=len(norm_block.get("merge_events", []) or []),
    )


def _labels_to_contact(labels: Sequence[str], horizon: int) -> np.ndarray:
    rows = [CONTACT_BY_LABEL.get(str(label), CONTACT_BY_LABEL["flight_or_unknown"]) for label in labels]
    if len(rows) < int(horizon):
        rows.extend([rows[-1] if rows else CONTACT_BY_LABEL["flight_or_unknown"]] * (int(horizon) - len(rows)))
    if len(rows) > int(horizon):
        rows = rows[: int(horizon)]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _fix_durations(durations: Sequence[int], horizon: int) -> Tuple[int, ...]:
    arr = np.asarray([max(1, int(x)) for x in durations], dtype=np.int64)
    if arr.size == 0:
        return (int(horizon),)
    diff = int(horizon) - int(np.sum(arr))
    while diff != 0:
        if diff > 0:
            idx = int(np.argmax(arr))
            arr[idx] += 1
            diff -= 1
        else:
            eligible = np.where(arr > 1)[0]
            if eligible.size == 0:
                break
            idx = int(eligible[np.argmax(arr[eligible])])
            arr[idx] -= 1
            diff += 1
    return tuple(int(x) for x in arr.tolist())


def _expand_topology(topology_tokens: Sequence[str], durations: Sequence[int], horizon: int) -> Tuple[str, ...]:
    tokens = [str(x) for x in topology_tokens]
    if not tokens:
        tokens = ["flight_or_unknown"]
    durations = _fix_durations(durations, int(horizon))
    if len(durations) != len(tokens):
        if len(durations) < len(tokens):
            durations = _fix_durations(list(durations) + [1] * (len(tokens) - len(durations)), int(horizon))
        else:
            durations = _fix_durations(list(durations[: len(tokens)]), int(horizon))
    labels: List[str] = []
    for label, length in zip(tokens, durations):
        labels.extend([label] * int(length))
    if len(labels) < int(horizon):
        labels.extend([labels[-1] if labels else "flight_or_unknown"] * (int(horizon) - len(labels)))
    return tuple(labels[: int(horizon)])


def _boundaries_from_durations(durations: Sequence[int]) -> Tuple[int, ...]:
    if len(durations) <= 1:
        return ()
    arr = np.cumsum(np.asarray(durations, dtype=np.int64))
    return tuple(int(x) for x in arr[:-1].tolist())


def _boundary_mae(true_dur: Sequence[int], pred_dur: Sequence[int], horizon: int) -> float:
    true_b = list(_boundaries_from_durations(true_dur))
    pred_b = list(_boundaries_from_durations(pred_dur))
    n = max(len(true_b), len(pred_b))
    if n == 0:
        return 0.0
    true_b.extend([int(horizon)] * (n - len(true_b)))
    pred_b.extend([int(horizon)] * (n - len(pred_b)))
    return float(np.mean(np.abs(np.asarray(true_b, dtype=np.float64) - np.asarray(pred_b, dtype=np.float64))))


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    train = np.where(np.isfinite(train), train, 0.0)
    test = np.where(np.isfinite(test), test, 0.0)
    mean = np.mean(train, axis=0, keepdims=True) if train.size else np.zeros((1, test.shape[1]), dtype=np.float64)
    std = np.std(train, axis=0, keepdims=True) if train.size else np.ones((1, test.shape[1]), dtype=np.float64)
    keep = std > 1e-6
    std = np.where(keep, std, 1.0)
    return (train - mean) / std, (test - mean) / std, {"d": int(train.shape[1]), "constant_features": int(np.sum(~keep))}


def _majority_classifier(y_train: np.ndarray, n_test: int, num_classes: int) -> ClassifierPred:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    if float(np.sum(counts)) <= 0.0:
        probs = np.full((n_test, num_classes), 1.0 / max(1, num_classes), dtype=np.float64)
    else:
        probs = np.repeat((counts / np.sum(counts)).reshape(1, -1), int(n_test), axis=0)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    return ClassifierPred("majority", probs, pred, {"train_class_count": int(np.sum(counts > 0))})


def _knn_classifier(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    *,
    k: int,
) -> ClassifierPred:
    xtr, xte, prep = _standardize(train_x, test_x)
    kk = min(max(1, int(k)), int(xtr.shape[0]))
    probs = np.zeros((xte.shape[0], num_classes), dtype=np.float64)
    for i, x in enumerate(xte):
        d = np.linalg.norm(xtr - x.reshape(1, -1), axis=1) / math.sqrt(max(1, xtr.shape[1]))
        nn = np.argsort(d)[:kk]
        weights = 1.0 / np.maximum(d[nn], 1e-6)
        for cls, w in zip(y_train[nn].astype(np.int64), weights):
            probs[i, int(cls)] += float(w)
        denom = float(np.sum(probs[i]))
        if denom <= 0.0:
            probs[i] = 1.0 / max(1, num_classes)
        else:
            probs[i] /= denom
    pred = np.argmax(probs, axis=1).astype(np.int64)
    return ClassifierPred("knn_available_context", probs, pred, {"k": int(kk), **prep})


def _nearest_centroid_classifier(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    *,
    reason: str,
) -> ClassifierPred:
    xtr, xte, prep = _standardize(train_x, test_x)
    present = sorted(int(x) for x in set(y_train.astype(np.int64).tolist()))
    centroids: Dict[int, np.ndarray] = {}
    for cls in present:
        mask = y_train == cls
        centroids[cls] = np.mean(xtr[mask], axis=0)
    probs = np.zeros((xte.shape[0], num_classes), dtype=np.float64)
    for i, x in enumerate(xte):
        logits = np.full((num_classes,), -1e9, dtype=np.float64)
        for cls, cen in centroids.items():
            logits[cls] = -float(np.linalg.norm(x - cen) / math.sqrt(max(1, x.shape[0])))
        logits -= np.max(logits)
        p = np.exp(logits)
        denom = float(np.sum(p))
        probs[i] = p / denom if denom > 0.0 else (1.0 / max(1, num_classes))
    pred = np.argmax(probs, axis=1).astype(np.int64)
    return ClassifierPred(
        "linear_or_nearest_centroid",
        probs,
        pred,
        {"fallback": "nearest_centroid", "reason": reason, **prep},
    )


def _linear_classifier(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
) -> ClassifierPred:
    if len(set(y_train.astype(np.int64).tolist())) <= 1:
        return _nearest_centroid_classifier(train_x, y_train, test_x, num_classes, reason="single_train_class")
    try:
        from sklearn.exceptions import ConvergenceWarning  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        import warnings

        xtr, xte, prep = _standardize(train_x, test_x)
        clf = LogisticRegression(max_iter=1000, solver="liblinear", multi_class="ovr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(xtr, y_train.astype(np.int64))
        local = clf.predict_proba(xte)
        probs = np.zeros((xte.shape[0], num_classes), dtype=np.float64)
        for col, cls in enumerate(clf.classes_.astype(np.int64)):
            probs[:, int(cls)] = local[:, col]
        row_sum = np.sum(probs, axis=1, keepdims=True)
        probs = np.divide(probs, np.maximum(row_sum, EPS))
        pred = np.argmax(probs, axis=1).astype(np.int64)
        return ClassifierPred("linear_logistic", probs, pred, {"backend": "sklearn_logistic", **prep})
    except Exception as exc:  # pragma: no cover - depends on local sklearn install.
        return _nearest_centroid_classifier(
            train_x,
            y_train,
            test_x,
            num_classes,
            reason=f"{type(exc).__name__}: {exc}",
        )


def _classifier_suite(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    *,
    num_classes: int,
    k: int,
) -> List[ClassifierPred]:
    return [
        _majority_classifier(y_train, int(test_x.shape[0]), int(num_classes)),
        _knn_classifier(train_x, y_train, test_x, int(num_classes), k=k),
        _linear_classifier(train_x, y_train, test_x, int(num_classes)),
    ]


def _class_metrics(
    pred: ClassifierPred,
    y_true: np.ndarray,
    y_train: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    probs = np.asarray(pred.probs, dtype=np.float64)
    y = y_true.astype(np.int64)
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0] if order.size else np.zeros((0,), dtype=np.int64)

    def _topk(k: int) -> float:
        if y.size == 0:
            return 0.0
        kk = min(k, order.shape[1])
        hit = [int(y[i]) in set(int(x) for x in order[i, :kk]) for i in range(y.size)]
        return float(np.mean(hit))

    present_test = sorted(set(int(x) for x in y.tolist()))
    macro_vals = []
    per_class = []
    for cls in present_test:
        mask = y == cls
        acc = float(np.mean(top1[mask] == cls)) if np.any(mask) else 0.0
        macro_vals.append(acc)
        per_class.append(
            {
                "class_id": int(cls),
                "class": class_names[cls],
                "support": int(np.sum(mask)),
                "top1_acc": acc,
                "top2_coverage": _rate(
                    sum(cls in set(int(x) for x in order[i, : min(2, order.shape[1])]) for i in np.where(mask)[0]),
                    int(np.sum(mask)),
                ),
                "top3_coverage": _rate(
                    sum(cls in set(int(x) for x in order[i, : min(3, order.shape[1])]) for i in np.where(mask)[0]),
                    int(np.sum(mask)),
                ),
            }
        )
    confusion_counter = Counter((int(t), int(p)) for t, p in zip(y.tolist(), top1.tolist()))
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
    train_classes = set(int(x) for x in y_train.astype(np.int64).tolist())
    entropy = -np.sum(probs * np.log2(np.maximum(probs, EPS)), axis=1) if probs.size else np.zeros((0,))
    return {
        "model": pred.model,
        "n_test": int(y.size),
        "top1_coverage": _topk(1),
        "top2_coverage": _topk(2),
        "top3_coverage": _topk(3),
        "macro_accuracy": float(np.mean(macro_vals)) if macro_vals else 0.0,
        "predicted_entropy_bits_mean": float(np.mean(entropy)) if entropy.size else 0.0,
        "predicted_entropy_bits_p95": float(np.percentile(entropy, 95)) if entropy.size else 0.0,
        "empirical_test_entropy_bits": _entropy_from_values(y.tolist()),
        "unseen_true_class_count": int(sum(int(x) not in train_classes for x in y.tolist())),
        "per_class_support": per_class,
        "confusion": confusion,
        "details": dict(pred.details),
    }


def _class_id_maps(values: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    names = sorted(set(str(x) for x in values))
    return {name: idx for idx, name in enumerate(names)}, names


def _build_splits(
    items: Sequence[WindowItem],
    *,
    train_fraction: float,
    block_gap: int,
    seed: int,
    low_n_threshold: int,
    include_random: bool,
) -> List[SplitDef]:
    by_clip: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_clip[item.clip].append(idx)
    splits: List[SplitDef] = []

    train_block: List[int] = []
    test_block: List[int] = []
    notes = []
    for clip in MATCHED_TARGETS:
        idxs = sorted(by_clip.get(clip, []), key=lambda i: items[i].start)
        n = len(idxs)
        if n < 2:
            notes.append(f"{clip}: n={n} skipped")
            continue
        train_n = max(1, min(n - 1, int(math.floor(float(train_fraction) * n))))
        test_start = train_n + max(0, int(block_gap))
        if test_start >= n:
            test_start = train_n
        train_block.extend(idxs[:train_n])
        test_block.extend(idxs[test_start:])
        notes.append(f"{clip}: train={train_n} gap={max(0, test_start - train_n)} test={n - test_start}")
    splits.append(
        SplitDef(
            name="contiguous_block",
            kind="contiguous_block",
            train_idx=tuple(train_block),
            test_idx=tuple(test_block),
            low_n_diagnostic=len(test_block) < int(low_n_threshold),
            note="; ".join(notes),
        )
    )

    for clip in MATCHED_TARGETS:
        test = sorted(by_clip.get(clip, []), key=lambda i: items[i].start)
        train: List[int] = []
        for other in MATCHED_TARGETS:
            if other != clip:
                train.extend(by_clip.get(other, []))
        splits.append(
            SplitDef(
                name=f"leave_clip_out:{clip}",
                kind="leave_clip_out",
                train_idx=tuple(sorted(train)),
                test_idx=tuple(test),
                low_n_diagnostic=len(test) < int(low_n_threshold),
                note=f"held_out={clip}; low_n_threshold={int(low_n_threshold)}",
            )
        )

    if include_random:
        rng = np.random.default_rng(int(seed))
        idxs = np.arange(len(items), dtype=np.int64)
        rng.shuffle(idxs)
        train_n = max(1, min(len(idxs) - 1, int(math.floor(float(train_fraction) * len(idxs)))))
        splits.append(
            SplitDef(
                name="random_optimistic_diagnostic",
                kind="random_optimistic_diagnostic",
                train_idx=tuple(int(x) for x in idxs[:train_n].tolist()),
                test_idx=tuple(int(x) for x in idxs[train_n:].tolist()),
                low_n_diagnostic=False,
                note="optimistic diagnostic only; adjacent overlapping windows can leak phase",
            )
        )
    return splits


def _features(items: Sequence[WindowItem], idxs: Sequence[int]) -> np.ndarray:
    return np.stack([items[i].feature_by_tier["available_context"] for i in idxs], axis=0).astype(np.float32)


def _ids(targets: Sequence[ScheduleTarget], idxs: Sequence[int], id_map: Mapping[str, int], attr: str) -> np.ndarray:
    vals = [getattr(targets[i], attr) for i in idxs]
    return np.asarray([int(id_map[str(v)]) for v in vals], dtype=np.int64)


def _empirical_bucket_entropy(
    items: Sequence[WindowItem],
    values: Sequence[str],
    *,
    radius: float,
) -> Dict[str, Any]:
    buckets = _cluster_items(items, tier="available_context", radius=float(radius))
    entropies = []
    weighted = []
    multi = 0
    max_labels = 0
    for bucket in buckets:
        idxs = [int(i) for i in bucket["items"]]
        vals = [values[i] for i in idxs]
        counts = Counter(vals)
        ent = _entropy_from_counts(counts.values())
        entropies.append(ent)
        weighted.append(ent * len(idxs))
        max_labels = max(max_labels, len(counts))
        if len(counts) > 1:
            multi += 1
    total_items = sum(len(b["items"]) for b in buckets)
    return {
        "radius": float(radius),
        "bucket_count": int(len(buckets)),
        "multi_bucket_count": int(multi),
        "multi_bucket_fraction": _rate(multi, len(buckets)),
        "max_labels_per_bucket": int(max_labels),
        "entropy_bits_mean_unweighted": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_bits_mean_weighted": float(np.sum(weighted) / max(1, total_items)),
    }


def _prototype_duration(
    train_targets: Sequence[ScheduleTarget],
    pred_topology: str,
    *,
    horizon: int,
) -> Tuple[int, ...]:
    rows = [t.durations for t in train_targets if t.topology_key == pred_topology]
    tokens = tuple(pred_topology.split(">")) if pred_topology and pred_topology != "empty" else ("flight_or_unknown",)
    if not rows:
        return _fix_durations([int(horizon)], int(horizon))
    width = len(tokens)
    compatible = [r for r in rows if len(r) == width]
    if not compatible:
        compatible = rows
    padded = []
    for r in compatible:
        if len(r) < width:
            padded.append(list(r) + [1] * (width - len(r)))
        else:
            padded.append(list(r[:width]))
    med = np.rint(np.median(np.asarray(padded, dtype=np.float64), axis=0)).astype(np.int64)
    return _fix_durations(med.tolist(), int(horizon))


def _nearest_token_sequence(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_targets: Sequence[ScheduleTarget],
    pred_topology: str,
    test_row: int,
) -> Tuple[str, ...]:
    xtr, xte, _ = _standardize(train_x, test_x)
    candidates = [i for i, t in enumerate(train_targets) if t.topology_key == pred_topology]
    if not candidates:
        candidates = list(range(len(train_targets)))
    if not candidates:
        return ("flight_or_unknown",)
    x = xte[int(test_row)]
    cand = np.asarray(candidates, dtype=np.int64)
    d = np.linalg.norm(xtr[cand] - x.reshape(1, -1), axis=1) / math.sqrt(max(1, xtr.shape[1]))
    best = int(cand[int(np.argmin(d))])
    return tuple(train_targets[best].normalized_labels)


def _foot_window(item: WindowItem, foot_cache: Mapping[str, Mapping[str, np.ndarray]], horizon: int) -> Dict[str, np.ndarray]:
    foot = foot_cache.get(item.clip, {})
    return {side: arr[item.start : item.start + int(horizon)] for side, arr in foot.items()}


def _timing_eval_one(
    *,
    item: WindowItem,
    true_target: ScheduleTarget,
    pred_topology: str,
    pred_labels: Sequence[str],
    pred_durations: Sequence[int],
    support_bands: Mapping[str, Mapping[str, Any]],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    foot_cache: Mapping[str, Mapping[str, np.ndarray]],
    horizon: int,
) -> Dict[str, Any]:
    pred_labels = tuple(str(x) for x in pred_labels)
    pred_contact = _labels_to_contact(pred_labels, int(horizon))
    side_features = _support_side_features(item.seq, pred_labels, _foot_window(item, foot_cache, int(horizon)))
    side_ok, side_failures = _evaluate_support_side_correctness(
        side_features,
        support_bands[item.clip]["feature_bands"],
    )
    seq = _replace_seq(item.seq, contact=pred_contact)
    support_row = _evaluate_sequence(
        seq,
        target=item.clip,
        target_bands=baseline_bands[item.clip],
        skeleton=skeleton,
        case="schedule_predictive_baseline:reconstructed_schedule",
        expected_label="diagnostic",
        start_phase=f"{item.clip}:{item.start}-{item.end}",
        endpoint_bridgeability=True,
        endpoint_details={"acceptance_proxy_scope": "support_honesty plus support_side_correctness only"},
    )
    token_acc = float(np.mean(np.asarray(pred_labels, dtype=object) == np.asarray(true_target.normalized_labels, dtype=object)))
    duration_exact = bool(
        pred_topology == true_target.topology_key
        and tuple(int(x) for x in pred_durations) == tuple(int(x) for x in true_target.durations)
    )
    b_mae = _boundary_mae(true_target.durations, pred_durations, int(horizon))
    return {
        "duration_exact": duration_exact,
        "boundary_mae_frames": b_mae,
        "token_accuracy": token_acc,
        "support_side_correctness": bool(side_ok),
        "support_honesty": bool(support_row.get("support_honesty", False)),
        "acceptance_proxy_pass": bool(side_ok and bool(support_row.get("support_honesty", False))),
        "support_side_failure_count": int(len(side_failures)),
        "contact_step_l2_p95": float(support_row.get("metrics", {}).get("contact_step_l2_p95", 0.0)),
        "foot_slip_p95_mps": float(support_row.get("metrics", {}).get("foot_slip_p95_mps", 0.0)),
    }


def _summarize_timing_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "exact_duration_match_rate": 0.0,
            "boundary_mae_frames_mean": 0.0,
            "boundary_mae_frames_median": 0.0,
            "reconstructed_schedule_token_accuracy_mean": 0.0,
            "support_side_correctness_pass_rate": 0.0,
            "support_honesty_pass_rate": 0.0,
            "acceptance_proxy_pass_rate": 0.0,
        }
    vals = lambda key: np.asarray([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "exact_duration_match_rate": float(np.mean([bool(r.get("duration_exact")) for r in rows])),
        "boundary_mae_frames_mean": float(np.mean(vals("boundary_mae_frames"))),
        "boundary_mae_frames_median": float(np.median(vals("boundary_mae_frames"))),
        "reconstructed_schedule_token_accuracy_mean": float(np.mean(vals("token_accuracy"))),
        "support_side_correctness_pass_rate": float(np.mean([bool(r.get("support_side_correctness")) for r in rows])),
        "support_honesty_pass_rate": float(np.mean([bool(r.get("support_honesty")) for r in rows])),
        "acceptance_proxy_pass_rate": float(np.mean([bool(r.get("acceptance_proxy_pass")) for r in rows])),
    }


def _evaluate_timing_models(
    *,
    split: SplitDef,
    items: Sequence[WindowItem],
    targets: Sequence[ScheduleTarget],
    topology_pred: ClassifierPred,
    topology_class_names: Sequence[str],
    support_bands: Mapping[str, Mapping[str, Any]],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    foot_cache: Mapping[str, Mapping[str, np.ndarray]],
    horizon: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_targets = [targets[i] for i in split.train_idx]
    train_x = _features(items, split.train_idx)
    test_x = _features(items, split.test_idx)
    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    by_model_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for local_j, item_idx in enumerate(split.test_idx):
        item = items[item_idx]
        true_target = targets[item_idx]
        pred_topology = topology_class_names[int(topology_pred.pred[local_j])]
        pred_tokens = tuple(pred_topology.split(">")) if pred_topology and pred_topology != "empty" else ("flight_or_unknown",)

        proto_dur = _prototype_duration(train_targets, pred_topology, horizon=int(horizon))
        proto_labels = _expand_topology(pred_tokens, proto_dur, int(horizon))
        proto_eval = _timing_eval_one(
            item=item,
            true_target=true_target,
            pred_topology=pred_topology,
            pred_labels=proto_labels,
            pred_durations=proto_dur,
            support_bands=support_bands,
            baseline_bands=baseline_bands,
            skeleton=skeleton,
            foot_cache=foot_cache,
            horizon=int(horizon),
        )
        row = {
            "split": split.name,
            "split_kind": split.kind,
            "topology_model": topology_pred.model,
            "timing_model": "duration_median_prototype",
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "true_topology": true_target.topology_key,
            "pred_topology": pred_topology,
            "true_durations": ";".join(str(x) for x in true_target.durations),
            "pred_durations": ";".join(str(x) for x in proto_dur),
            "true_normalized_signature": true_target.normalized_signature,
            "pred_labels": ">".join(proto_labels),
            **proto_eval,
        }
        rows.append(row)
        by_model_rows["duration_median_prototype"].append(row)

        nn_labels = _nearest_token_sequence(train_x, test_x, train_targets, pred_topology, local_j)
        nn_runs = _rle(nn_labels)
        nn_dur = tuple(int(length) for _, length in nn_runs)
        nn_eval = _timing_eval_one(
            item=item,
            true_target=true_target,
            pred_topology=pred_topology,
            pred_labels=nn_labels,
            pred_durations=nn_dur,
            support_bands=support_bands,
            baseline_bands=baseline_bands,
            skeleton=skeleton,
            foot_cache=foot_cache,
            horizon=int(horizon),
        )
        row = {
            "split": split.name,
            "split_kind": split.kind,
            "topology_model": topology_pred.model,
            "timing_model": "event_token_nn_continuation",
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "true_topology": true_target.topology_key,
            "pred_topology": pred_topology,
            "true_durations": ";".join(str(x) for x in true_target.durations),
            "pred_durations": ";".join(str(x) for x in nn_dur),
            "true_normalized_signature": true_target.normalized_signature,
            "pred_labels": ">".join(nn_labels),
            **nn_eval,
        }
        rows.append(row)
        by_model_rows["event_token_nn_continuation"].append(row)

    for timing_model, model_rows in by_model_rows.items():
        summary_rows.append(
            {
                "split": split.name,
                "split_kind": split.kind,
                "topology_model": topology_pred.model,
                "timing_model": timing_model,
                **_summarize_timing_rows(model_rows),
            }
        )
    return rows, summary_rows


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "split_kind",
        "topology_model",
        "timing_model",
        "clip",
        "start",
        "end",
        "true_topology",
        "pred_topology",
        "true_durations",
        "pred_durations",
        "duration_exact",
        "boundary_mae_frames",
        "token_accuracy",
        "support_side_correctness",
        "support_honesty",
        "acceptance_proxy_pass",
        "support_side_failure_count",
        "contact_step_l2_p95",
        "foot_slip_p95_mps",
        "true_normalized_signature",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_confusion_csv(path: Path, topology_results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "split_kind", "model", "true", "pred", "count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in topology_results:
            for row in rec.get("confusion", []) or []:
                writer.writerow(
                    {
                        "split": rec.get("split"),
                        "split_kind": rec.get("split_kind"),
                        "model": rec.get("model"),
                        "true": row.get("true"),
                        "pred": row.get("pred"),
                        "count": row.get("count"),
                    }
                )


def _normalization_impact(targets: Sequence[ScheduleTarget]) -> Dict[str, Any]:
    raw_topologies = []
    norm_topologies = []
    raw_timing = []
    norm_timing = []
    changed_topology = 0
    changed_timing = 0
    for t in targets:
        raw_runs = _rle(t.raw_labels)
        raw_top = ">".join(label for label, _ in raw_runs) if raw_runs else "empty"
        raw_dur = ",".join(str(length) for _, length in raw_runs) if raw_runs else "empty"
        raw_topologies.append(raw_top)
        norm_topologies.append(t.topology_key)
        raw_timing.append(raw_dur)
        norm_timing.append(t.timing_key)
        changed_topology += int(raw_top != t.topology_key)
        changed_timing += int(raw_dur != t.timing_key)
    return {
        "n": int(len(targets)),
        "ambiguous_blip_item_count": int(sum(t.ambiguous_count > 0 for t in targets)),
        "ambiguous_blip_total_count": int(sum(t.ambiguous_count for t in targets)),
        "merge_event_total_count": int(sum(t.merge_count for t in targets)),
        "raw_topology_entropy_bits": _entropy_from_values(raw_topologies),
        "normalized_topology_entropy_bits": _entropy_from_values(norm_topologies),
        "raw_timing_entropy_bits": _entropy_from_values(raw_timing),
        "normalized_timing_entropy_bits": _entropy_from_values(norm_timing),
        "topology_changed_by_debounce_count": int(changed_topology),
        "timing_changed_by_debounce_count": int(changed_timing),
    }


def _load_previous_calibration(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"loaded": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    avail = ((data.get("multimodality", {}) or {}).get("available_context", {}) or {})
    return {
        "loaded": True,
        "path": str(path),
        "empirical_available_context_multi_fraction": float(
            avail.get("normalized_multi_signature_bucket_fraction", 0.0)
        ),
        "empirical_available_context_max_signatures": int(avail.get("max_signatures_per_bucket", 0)),
        "expected_available_context_multi_fraction": 0.2917,
        "expected_available_context_max_signatures": 5,
    }


def _select_primary_topology_result(topology_results: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    decisive = [
        r
        for r in topology_results
        if r.get("split") == "contiguous_block" and r.get("model") != "majority"
    ]
    if decisive:
        return sorted(
            decisive,
            key=lambda r: (
                -float(r.get("top3_coverage", 0.0)),
                -float(r.get("top2_coverage", 0.0)),
                -float(r.get("top1_coverage", 0.0)),
                str(r.get("model", "")),
            ),
        )[0]
    decisive = [r for r in topology_results if r.get("split") == "contiguous_block"]
    if decisive:
        return decisive[0]
    return topology_results[0] if topology_results else None


def _architecture_decision(
    topology_results: Sequence[Mapping[str, Any]],
    timing_results: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    primary = _select_primary_topology_result(topology_results)
    if primary is None:
        return {"decision": "data_insufficient", "reason": "no topology result"}
    top1 = float(primary.get("top1_coverage", 0.0))
    top2 = float(primary.get("top2_coverage", 0.0))
    top3 = float(primary.get("top3_coverage", 0.0))
    if top1 >= 0.80 and top2 >= 0.95:
        topology_decision = "deterministic_topology_head"
        topology_reason = "topology top-1 >= 0.80 and top-2 >= 0.95 on primary contiguous-block split"
    elif top3 < 0.80:
        topology_decision = "data_or_condition_insufficient"
        topology_reason = "topology top-3 < 0.80; do not escalate layer-1 to diffusion"
    elif top2 >= 0.80 or top3 >= 0.95:
        topology_decision = "small_categorical_topology_sampler"
        topology_reason = "top-k coverage is high but top-1 is not deterministic"
    else:
        topology_decision = "data_or_condition_insufficient"
        topology_reason = "topology coverage is inconclusive; expand clips or context first"

    primary_timing = [
        r
        for r in timing_results
        if r.get("split") == "contiguous_block"
        and r.get("topology_model") == primary.get("model")
        and r.get("timing_model") == "event_token_nn_continuation"
    ]
    timing_decision = "not_evaluated"
    timing_reason = "no timing row for primary topology model"
    next_step = "hold"
    if primary_timing:
        row = primary_timing[0]
        exact = float(row.get("exact_duration_match_rate", 0.0))
        mae = float(row.get("boundary_mae_frames_mean", 0.0))
        side = float(row.get("support_side_correctness_pass_rate", 0.0))
        proxy = float(row.get("acceptance_proxy_pass_rate", 0.0))
        if topology_decision == "data_or_condition_insufficient":
            timing_decision = "duration_or_ar_event_token_timing_model_diagnostic"
            timing_reason = "timing errors are diagnostic only because topology top-k/data condition is not yet sufficient"
        elif exact < 0.50 or mae > 2.0:
            timing_decision = "duration_or_ar_event_token_timing_model"
            timing_reason = "topology is separable enough, but timing exact match is low or boundary MAE is high"
        else:
            timing_decision = "simple_duration_prototype_may_suffice"
            timing_reason = "duration exact match and boundary MAE are within the preregistered light-baseline band"
        if side >= 0.80 and proxy >= 0.80 and topology_decision != "data_or_condition_insufficient":
            next_step = "schedule_conditioned_deterministic_trajectory_decoder"
        elif topology_decision == "data_or_condition_insufficient":
            next_step = "expand_clip_or_add_missing_context"
        else:
            next_step = "improve_layer1_schedule_predictor_before_decoder"

    predictor_entropy = float(primary.get("predicted_entropy_bits_mean", 0.0))
    empirical_entropy = float(calibration.get("current_topology_bucket_entropy_bits_weighted", 0.0))
    entropy_flag = "ok"
    if empirical_entropy > 0.0 and predictor_entropy > empirical_entropy * 1.5 + 0.25:
        entropy_flag = "underfit_or_missing_condition"

    return {
        "primary_split": primary.get("split"),
        "primary_model": primary.get("model"),
        "primary_topology_top1": top1,
        "primary_topology_top2": top2,
        "primary_topology_top3": top3,
        "topology_decision": topology_decision,
        "topology_reason": topology_reason,
        "timing_decision": timing_decision,
        "timing_reason": timing_reason,
        "next_step": next_step,
        "entropy_calibration_flag": entropy_flag,
        "diffusion_statement": (
            "Layer-1 conclusion must not be 'diffusion required'. Trajectory diffusion is only considered "
            "after fixing schedule and showing residual trajectory multimodality."
        ),
    }


def _contract_payload(example: WindowItem, example_target: ScheduleTarget) -> Dict[str, Any]:
    feat = np.asarray(example.feature_by_tier["available_context"], dtype=np.float32)
    contact = np.asarray(example.seq["contact"], dtype=np.float32)
    bone = np.asarray(example.seq["bone_angvel"], dtype=np.float32)
    labels = np.asarray(example_target.normalized_labels, dtype=object)
    event_tokens = np.asarray(example_target.event_tokens, dtype=np.int64)
    topo_id = np.asarray(0, dtype=np.int64)
    durations = np.asarray(example_target.durations, dtype=np.int64)
    reconstructed = _labels_to_contact(
        _expand_topology(example_target.topology_tokens, example_target.durations, contact.shape[0]),
        contact.shape[0],
    )
    return {
        "available_context_feature": {
            "shape_contract": "[D]",
            "actual_shape": [int(feat.shape[0])],
            "dtype": str(feat.dtype),
            "device": "cpu",
            "finite": bool(np.isfinite(feat).all()),
            "construction": "start ctx [C,281] + commanded cond_dir/yaw cue + soft endpoint cue + start/seam support cue",
        },
        "support_schedule_contact": {
            "shape_contract": "[H,2]",
            "actual_shape": [int(x) for x in contact.shape],
            "dtype": str(contact.dtype),
            "device": "cpu",
        },
        "normalized_label_sequence": {
            "shape_contract": "[H]",
            "actual_shape": [int(labels.shape[0])],
            "dtype": "object/string",
            "device": "cpu",
        },
        "topology_target": {
            "shape_contract": "scalar",
            "actual_shape": [],
            "dtype": str(topo_id.dtype),
            "device": "cpu",
            "meaning": "class id for debounced event topology with durations removed",
        },
        "timing_target_duration": {
            "shape_contract": "[R]",
            "actual_shape": [int(durations.shape[0])],
            "dtype": str(durations.dtype),
            "device": "cpu",
            "meaning": "run durations conditioned on topology",
        },
        "timing_target_event_tokens": {
            "shape_contract": "[H]",
            "actual_shape": [int(event_tokens.shape[0])],
            "dtype": str(event_tokens.dtype),
            "device": "cpu",
            "meaning": "per-frame support event token ids",
        },
        "reconstructed_schedule": {
            "shape_contract": "[H,2]",
            "actual_shape": [int(x) for x in reconstructed.shape],
            "dtype": str(reconstructed.dtype),
            "device": "cpu",
            "construction": "topology tokens + timing durations expanded to contact schedule",
        },
        "bone_angvel_witness": {
            "shape_contract": "[H,138]",
            "actual_shape": [int(x) for x in bone.shape],
            "dtype": str(bone.dtype),
            "device": "cpu",
            "role": "loaded witness only; not a layer-1 prediction target",
            "dim_check": int(bone.shape[1]) == ANGVEL_DIM,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only support schedule predictive baseline.")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--previous-summary", type=Path, default=DEFAULT_PREVIOUS_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--available-bucket-radius", type=float, default=0.035)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--block-gap", type=int, default=None)
    p.add_argument("--low-n-threshold", type=int, default=20)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--include-random-diagnostic", action="store_true")
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
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    baseline_bands = _calibrate_baselines(clips, skeleton, quantile=99.5)
    support_bands, foot_cache = _calibrate_support_side_bands(
        clips,
        skeleton,
        horizon=h,
        min_run_frames=int(args.min_run_frames),
        only_clips=TURN_CLIPS,
    )
    all_items = _build_window_items(
        clips,
        skeleton,
        support_bands,
        foot_cache,
        horizon=h,
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        only_clips=TURN_CLIPS,
        stride=max(1, int(args.stride)),
        include_bone_angvel_witness=True,
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    unmatched_items = [it for it in all_items if it.clip == UNMATCHED_TARGET]
    if not main_items:
        raise RuntimeError("no matched target windows found")
    targets = [_target_from_item(it) for it in main_items]

    topology_id, topology_names = _class_id_maps([t.topology_key for t in targets])
    flat_id, flat_names = _class_id_maps([t.normalized_signature for t in targets])
    topology_values = [t.topology_key for t in targets]
    flat_values = [t.normalized_signature for t in targets]
    timing_values = [t.timing_key for t in targets]

    splits = _build_splits(
        main_items,
        train_fraction=float(args.train_fraction),
        block_gap=block_gap,
        seed=int(args.seed),
        low_n_threshold=int(args.low_n_threshold),
        include_random=bool(args.include_random_diagnostic),
    )

    topology_results: List[Dict[str, Any]] = []
    flat_results: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    timing_results: List[Dict[str, Any]] = []

    for split in splits:
        if not split.train_idx or not split.test_idx:
            continue
        x_train = _features(main_items, split.train_idx)
        x_test = _features(main_items, split.test_idx)
        y_top_train = _ids(targets, split.train_idx, topology_id, "topology_key")
        y_top_test = _ids(targets, split.test_idx, topology_id, "topology_key")
        y_flat_train = _ids(targets, split.train_idx, flat_id, "normalized_signature")
        y_flat_test = _ids(targets, split.test_idx, flat_id, "normalized_signature")

        topology_preds = _classifier_suite(
            x_train,
            y_top_train,
            x_test,
            num_classes=len(topology_names),
            k=int(args.knn_k),
        )
        for pred in topology_preds:
            rec = {
                "split": split.name,
                "split_kind": split.kind,
                "low_n_diagnostic": bool(split.low_n_diagnostic),
                "note": split.note,
                "train_n": int(len(split.train_idx)),
                "test_n": int(len(split.test_idx)),
                **_class_metrics(pred, y_top_test, y_top_train, topology_names),
            }
            topology_results.append(rec)
            rows, summaries = _evaluate_timing_models(
                split=split,
                items=main_items,
                targets=targets,
                topology_pred=pred,
                topology_class_names=topology_names,
                support_bands=support_bands,
                baseline_bands=baseline_bands,
                skeleton=skeleton,
                foot_cache=foot_cache,
                horizon=h,
            )
            timing_rows.extend(rows)
            timing_results.extend(summaries)

        flat_preds = _classifier_suite(
            x_train,
            y_flat_train,
            x_test,
            num_classes=len(flat_names),
            k=int(args.knn_k),
        )
        for pred in flat_preds:
            flat_results.append(
                {
                    "split": split.name,
                    "split_kind": split.kind,
                    "diagnostic_only": True,
                    "train_n": int(len(split.train_idx)),
                    "test_n": int(len(split.test_idx)),
                    **_class_metrics(pred, y_flat_test, y_flat_train, flat_names),
                }
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

    previous = _load_previous_calibration(Path(args.previous_summary))
    bucket_top = _empirical_bucket_entropy(main_items, topology_values, radius=float(args.available_bucket_radius))
    bucket_flat = _empirical_bucket_entropy(main_items, flat_values, radius=float(args.available_bucket_radius))
    bucket_timing = _empirical_bucket_entropy(main_items, timing_values, radius=float(args.available_bucket_radius))
    calibration = {
        **previous,
        "current_topology_bucket_entropy_bits_weighted": bucket_top["entropy_bits_mean_weighted"],
        "current_topology_bucket_entropy": bucket_top,
        "current_flat_signature_bucket_entropy": bucket_flat,
        "current_timing_bucket_entropy": bucket_timing,
    }

    entropy_summary = {
        "flat_signature_entropy_bits": _entropy_from_values(flat_values),
        "topology_entropy_bits": _entropy_from_values(topology_values),
        "timing_entropy_bits_conditional_on_topology": _conditional_entropy(timing_values, topology_values),
        "topology_plus_conditional_timing_entropy_bits": _entropy_from_values(topology_values)
        + _conditional_entropy(timing_values, topology_values),
        "interpretation": (
            "flat signature is diagnostic only; excess flat entropy is treated as mixed topology/timing entropy, "
            "not evidence that layer-1 diffusion is required"
        ),
    }

    arch = _architecture_decision(topology_results, timing_results, calibration)
    norm_impact = _normalization_impact(targets)

    split_summary = [
        {
            "name": s.name,
            "kind": s.kind,
            "train_n": int(len(s.train_idx)),
            "test_n": int(len(s.test_idx)),
            "low_n_diagnostic": bool(s.low_n_diagnostic),
            "note": s.note,
            "train_per_clip": dict(Counter(main_items[i].clip for i in s.train_idx)),
            "test_per_clip": dict(Counter(main_items[i].clip for i in s.test_idx)),
        }
        for s in splits
    ]
    target_counts = Counter(t.topology_key for t in targets)
    duration_counts = Counter((t.topology_key, t.timing_key) for t in targets)
    target_rows = [
        {
            "clip": item.clip,
            "start": int(item.start),
            "end": int(item.end),
            "raw_labels": list(target.raw_labels),
            "normalized_labels": list(target.normalized_labels),
            "topology_tokens": list(target.topology_tokens),
            "duration_vector": [int(x) for x in target.durations],
            "event_tokens": [int(x) for x in target.event_tokens],
            "topology_key": target.topology_key,
            "timing_key": target.timing_key,
            "raw_signature": target.raw_signature,
            "normalized_signature": target.normalized_signature,
            "ambiguous_blip_count": int(target.ambiguous_count),
            "merge_event_count": int(target.merge_count),
        }
        for item, target in zip(main_items, targets)
    ]

    payload = {
        "task": "support_schedule_predictive_baseline",
        "scope": (
            "read-only lightweight sklearn/numpy baseline; no full trajectory generator training; "
            "no production trainer/runtime/gate forward or edit; no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "two_frame_summary": str(args.two_frame_summary),
            "previous_summary": str(args.previous_summary),
            "horizon": h,
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "available_bucket_radius": float(args.available_bucket_radius),
            "train_fraction": float(args.train_fraction),
            "block_gap": int(block_gap),
            "knn_k": int(args.knn_k),
            "seed": int(args.seed),
            "device_policy": "cpu only",
        },
        "input_output_contract": _contract_payload(main_items[0], targets[0]),
        "target_split_contract": {
            "main_target": "topology class id + timing durations/event tokens",
            "forbidden_main_target": "_support_signature or raw/normalized signature string alone",
            "flat_signature_role": "diagnostic baseline only",
            "topology_definition": "debounced support event order with duration removed",
            "timing_definition": "run durations or per-frame event tokens conditioned on topology",
            "label_order": list(LABEL_ORDER),
        },
        "architecture_decision_rules_preregistered": {
            "deterministic_topology": "if topology top-1 >= 0.80 and top-2 >= 0.95, no topology sampler",
            "categorical_topology_sampler": "if top-2/top-3 high but top-1 moderate, use small categorical topology sampler",
            "data_or_condition_insufficient": "if topology top-3 < 0.80, expand clips/context before any diffusion",
            "timing_model": "if topology predictable but duration exact is low or boundary MAE high, use duration/AR event-token timing",
            "decoder_gate": "if reconstructed schedule support_side_correctness is high, next is schedule-conditioned deterministic trajectory decoder",
            "diffusion_limit": "trajectory diffusion is only considered after fixed schedule leaves multimodal trajectory residuals",
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "unmatched_out_of_scope": UNMATCHED_TARGET,
            "main_window_count": int(len(main_items)),
            "unmatched_window_count": int(len(unmatched_items)),
            "per_clip_windows": dict(Counter(it.clip for it in all_items)),
            "topology_class_count": int(len(topology_names)),
            "flat_signature_class_count": int(len(flat_names)),
            "topology_counts": dict(target_counts),
            "topology_duration_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in duration_counts.items()},
        },
        "target_rows": target_rows,
        "splits": split_summary,
        "normalization_impact": norm_impact,
        "empirical_entropy_calibration": calibration,
        "entropy_diagnostic": entropy_summary,
        "topology_results": topology_results,
        "timing_results": timing_results,
        "flat_signature_diagnostic": flat_results,
        "architecture_decision": arch,
        "walk_l_to_r": walk_l_to_r,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_full_generator": False,
            "forwarded_production_runtime_or_trainer": False,
            "modified_production_runtime_trainer_gate": False,
            "used_residual_head": False,
            "used_yaw_as_prediction_target": False,
            "used_diffusion_as_layer1_default": False,
        },
        "artifacts": {
            "summary_json": str(out_dir / "support_schedule_predictive_baseline_summary.json"),
            "summary_md": str(out_dir / "support_schedule_predictive_baseline_summary.md"),
            "rows_csv": str(out_dir / "support_schedule_predictive_baseline_rows.csv"),
            "topology_confusion_csv": str(out_dir / "support_schedule_predictive_baseline_topology_confusion.csv"),
        },
    }

    _dump_json(out_dir / "support_schedule_predictive_baseline_summary.json", payload)
    _write_rows_csv(out_dir / "support_schedule_predictive_baseline_rows.csv", timing_rows)
    _write_confusion_csv(out_dir / "support_schedule_predictive_baseline_topology_confusion.csv", topology_results)

    primary = _select_primary_topology_result(topology_results) or {}
    best_timing = [
        r
        for r in timing_results
        if r.get("split") == "contiguous_block"
        and r.get("topology_model") == primary.get("model")
        and r.get("timing_model") == "event_token_nn_continuation"
    ]
    best_timing_row = best_timing[0] if best_timing else {}
    lines: List[str] = []
    lines.append("# Support Schedule Predictive Baseline")
    lines.append("")
    lines.append("Read-only lightweight baseline. No full trajectory generator training, no production runtime/trainer/gate forward or edit.")
    lines.append("")
    lines.append("## Target Split")
    lines.append("")
    lines.append("- topology: debounced support event order with duration removed.")
    lines.append("- timing: duration vector or per-frame event tokens conditioned on topology.")
    lines.append("- flat normalized signature: diagnostic only, not the layer-1 target.")
    lines.append("")
    lines.append("## Dataset / Splits")
    lines.append("")
    lines.append(f"- main matched windows: `{len(main_items)}` from `{MATCHED_TARGETS}`")
    lines.append(f"- unmatched Walk_L_To_R windows: `{len(unmatched_items)}` diagnostic only")
    for s in split_summary:
        lines.append(
            f"- {s['name']}: train `{s['train_n']}`, test `{s['test_n']}`, "
            f"low_n `{s['low_n_diagnostic']}`; {s['note']}"
        )
    lines.append("")
    lines.append("## Topology Baseline")
    lines.append("")
    lines.append("| split | model | top1 | top2 | top3 | macro | pred entropy | unseen |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in topology_results:
        if r.get("split_kind") == "random_optimistic_diagnostic":
            suffix = " (optimistic)"
        else:
            suffix = ""
        lines.append(
            f"| {r['split']}{suffix} | {r['model']} | {_fmt(r['top1_coverage'])} | "
            f"{_fmt(r['top2_coverage'])} | {_fmt(r['top3_coverage'])} | "
            f"{_fmt(r['macro_accuracy'])} | {_fmt(r['predicted_entropy_bits_mean'])} | "
            f"{r['unseen_true_class_count']} |"
        )
    lines.append("")
    lines.append("## Timing Baseline")
    lines.append("")
    lines.append("| split | topology model | timing model | exact | boundary MAE | token acc | side pass | proxy pass |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in timing_results:
        lines.append(
            f"| {r['split']} | {r['topology_model']} | {r['timing_model']} | "
            f"{_fmt(r['exact_duration_match_rate'])} | {_fmt(r['boundary_mae_frames_mean'])} | "
            f"{_fmt(r['reconstructed_schedule_token_accuracy_mean'])} | "
            f"{_fmt(r['support_side_correctness_pass_rate'])} | {_fmt(r['acceptance_proxy_pass_rate'])} |"
        )
    lines.append("")
    lines.append("## Entropy Calibration")
    lines.append("")
    lines.append(
        f"- previous available_context multi fraction: `{_fmt(previous.get('empirical_available_context_multi_fraction'))}`; "
        f"max signatures: `{previous.get('empirical_available_context_max_signatures')}`"
    )
    lines.append(
        f"- current topology bucket entropy weighted: `{_fmt(bucket_top['entropy_bits_mean_weighted'])}`; "
        f"flat signature bucket entropy weighted: `{_fmt(bucket_flat['entropy_bits_mean_weighted'])}`"
    )
    lines.append(
        f"- flat entropy `{_fmt(entropy_summary['flat_signature_entropy_bits'])}` vs topology "
        f"`{_fmt(entropy_summary['topology_entropy_bits'])}` + timing|topology "
        f"`{_fmt(entropy_summary['timing_entropy_bits_conditional_on_topology'])}`"
    )
    lines.append("")
    lines.append("## Architecture Decision")
    lines.append("")
    lines.append(f"- primary topology model: `{primary.get('model')}` on `{primary.get('split')}`")
    lines.append(
        f"- primary top-k: top1 `{_fmt(primary.get('top1_coverage'))}`, "
        f"top2 `{_fmt(primary.get('top2_coverage'))}`, top3 `{_fmt(primary.get('top3_coverage'))}`"
    )
    if best_timing_row:
        lines.append(
            f"- primary timing: exact `{_fmt(best_timing_row.get('exact_duration_match_rate'))}`, "
            f"boundary MAE `{_fmt(best_timing_row.get('boundary_mae_frames_mean'))}`, "
            f"token acc `{_fmt(best_timing_row.get('reconstructed_schedule_token_accuracy_mean'))}`, "
            f"proxy pass `{_fmt(best_timing_row.get('acceptance_proxy_pass_rate'))}`"
        )
    lines.append(f"- topology decision: `{arch['topology_decision']}`")
    lines.append(f"- timing decision: `{arch['timing_decision']}`")
    lines.append(f"- next step: `{arch['next_step']}`")
    lines.append(f"- entropy flag: `{arch['entropy_calibration_flag']}`")
    lines.append("- layer-1 decision is not `diffusion required`.")
    lines.append("")
    lines.append("## Walk_L_To_R")
    lines.append("")
    lines.append(f"- matched_pair_available: `{bool(walk_l_to_r['matched_pair_available'])}`")
    lines.append(f"- pose_d: `{_fmt(walk_l_to_r['pose_d'])}`")
    lines.append(f"- contact_d: `{_fmt(walk_l_to_r['contact_d'])}`")
    lines.append(f"- seam_support: `{walk_l_to_r['seam_support']}`")
    lines.append(f"- horizon_support: `{walk_l_to_r['horizon_support']}`")
    lines.append(f"- ungroundable_reason: `{walk_l_to_r['ungroundable_reason']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for path in payload["artifacts"].values():
        lines.append(f"- `{path}`")
    _dump_md(out_dir / "support_schedule_predictive_baseline_summary.md", lines)

    print(f"wrote {out_dir / 'support_schedule_predictive_baseline_summary.md'}")
    print(f"wrote {out_dir / 'support_schedule_predictive_baseline_summary.json'}")
    print(f"wrote {out_dir / 'support_schedule_predictive_baseline_rows.csv'}")


if __name__ == "__main__":
    main()
