#!/usr/bin/env python3
"""Granularity/coverage audit for support topology unseen classes.

Read-only audit. No model training, no model forward, no production
trainer/runtime/gate edit, no checkpoint mutation. This script only rebuilds
continuous support-window metadata and reads existing learner-condition
artifacts.
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

from train.data.action_handoff_inbetween import CONTEXT_LEN_C, TURN_CLIPS, WALK_F  # noqa: E402
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _load_clips,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    _context_window,
    _make_sequence,
    _support_contract,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
    UNMATCHED_TARGET,
    _build_splits,
    _target_from_item,
)


DEFAULT_ABLATION_SUMMARY = Path(
    "debug_output/_tmp_action_handoff_support_topology_learner_condition_ablation_20260602/"
    "support_topology_learner_condition_ablation_summary.json"
)
DEFAULT_CONFUSION = Path(
    "debug_output/_tmp_action_handoff_support_topology_learner_condition_ablation_20260602/"
    "support_topology_learner_condition_ablation_confusion.csv"
)
DEFAULT_TIGHTENING_SUMMARY = Path(
    "debug_output/_tmp_action_handoff_support_contract_tightening_20260602/"
    "support_contract_tightening_summary.json"
)
DEFAULT_OUT_DIR = Path(
    "debug_output/_tmp_action_handoff_support_topology_granularity_coverage_audit_20260602"
)

LABELS_WITH_SIDE = {"right", "left", "dual"}
LEFT_HALF_CLIPS = ("Walk_L_To_L", "Walk_L_To_R")


@dataclass
class AuditItem:
    clip: str
    start: int
    end: int
    seq: Dict[str, np.ndarray]
    ctx: np.ndarray
    support_contract: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(key: str) -> Tuple[str, ...]:
    if not key or key == "empty":
        return ("flight_or_unknown",)
    return tuple(str(x) for x in key.split(">") if str(x))


def _key(tokens: Sequence[str]) -> str:
    vals = tuple(str(x) for x in tokens)
    return ">".join(vals) if vals else "empty"


def _rle(labels: Sequence[str]) -> List[Tuple[str, int]]:
    if not labels:
        return []
    out: List[Tuple[str, int]] = []
    cur = str(labels[0])
    n = 1
    for label in labels[1:]:
        label = str(label)
        if label == cur:
            n += 1
            continue
        out.append((cur, n))
        cur = label
        n = 1
    out.append((cur, n))
    return out


def _topology_from_labels(labels: Sequence[str]) -> str:
    return _key([label for label, _ in _rle(labels)])


def _compress(tokens: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for token in tokens:
        token = str(token)
        if out and out[-1] == token:
            continue
        out.append(token)
    return tuple(out)


def _drop_flight(tokens: Sequence[str]) -> Tuple[str, ...]:
    return _compress([x for x in tokens if str(x) != "flight_or_unknown"])


def _main_order(tokens: Sequence[str]) -> Tuple[str, ...]:
    return _drop_flight(tokens)


def _side_swap(tokens: Sequence[str]) -> Tuple[str, ...]:
    swapped = []
    for token in tokens:
        if token == "left":
            swapped.append("right")
        elif token == "right":
            swapped.append("left")
        else:
            swapped.append(token)
    return tuple(swapped)


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    aa = tuple(a)
    bb = tuple(b)
    dp = [[0] * (len(bb) + 1) for _ in range(len(aa) + 1)]
    for i in range(len(aa) + 1):
        dp[i][0] = i
    for j in range(len(bb) + 1):
        dp[0][j] = j
    for i in range(1, len(aa) + 1):
        for j in range(1, len(bb) + 1):
            cost = 0 if aa[i - 1] == bb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return int(dp[-1][-1])


def _nearest_seen(topology: str, seen: Iterable[str]) -> Dict[str, Any]:
    toks = _tokens(topology)
    best: Optional[Tuple[int, str]] = None
    for cand in sorted(set(str(x) for x in seen)):
        d = _edit_distance(toks, _tokens(cand))
        if best is None or (d, cand) < best:
            best = (d, cand)
    if best is None:
        return {"nearest_seen_topology": None, "edit_distance": None}
    return {"nearest_seen_topology": best[1], "edit_distance": int(best[0])}


def _is_flight_insert_delete_only(topology: str, seen: Iterable[str]) -> Tuple[bool, Optional[str]]:
    main = _drop_flight(_tokens(topology))
    if not main:
        return False, None
    for cand in sorted(set(str(x) for x in seen)):
        if main == _drop_flight(_tokens(cand)) and _tokens(topology) != _tokens(cand):
            return True, cand
    return False, None


def _drop_short_run_candidates(labels: Sequence[str], max_len: int) -> List[str]:
    runs = _rle(labels)
    out: List[str] = []
    for idx, (_, length) in enumerate(runs):
        if int(length) > int(max_len):
            continue
        kept = [label for j, (label, _) in enumerate(runs) if j != idx]
        key = _key(_compress(kept))
        if key and key != "empty":
            out.append(key)
    return sorted(set(out))


def _same_main_order_candidate(topology: str, seen: Iterable[str]) -> Tuple[bool, Optional[str]]:
    main = _main_order(_tokens(topology))
    if not main:
        return False, None
    for cand in sorted(set(str(x) for x in seen)):
        if main == _main_order(_tokens(cand)) and topology != cand:
            return True, cand
    return False, None


def _side_reversal_candidate(topology: str, seen: Iterable[str]) -> Tuple[bool, Optional[str]]:
    swapped_key = _key(_side_swap(_tokens(topology)))
    for cand in sorted(set(str(x) for x in seen)):
        if cand == swapped_key and cand != topology:
            return True, cand
    return False, None


def _build_items(
    clips: Mapping[str, Any],
    *,
    horizon: int,
    context_len: int,
    min_run_frames: int,
    stride: int,
) -> List[AuditItem]:
    items: List[AuditItem] = []
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
            items.append(
                AuditItem(
                    clip=name,
                    start=int(start),
                    end=int(start + horizon - 1),
                    seq=seq,
                    ctx=ctx,
                    support_contract=contract,
                )
            )
    return items


def _first_occurrence(items: Sequence[AuditItem], targets: Sequence[Any], topology: str) -> Dict[str, Any]:
    rows = [
        (item.clip, int(item.start), int(item.end))
        for item, target in zip(items, targets)
        if str(target.topology_key) == str(topology)
    ]
    if not rows:
        return {"first_seen_clip": None, "first_seen_start": None, "first_seen_end": None}
    clip_order = {name: i for i, name in enumerate(TURN_CLIPS)}
    clip, start, end = sorted(rows, key=lambda r: (clip_order.get(r[0], 999), r[1], r[2]))[0]
    return {"first_seen_clip": clip, "first_seen_start": int(start), "first_seen_end": int(end)}


def _confusion_by_true(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    if not path.is_file():
        return {}
    counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("feature_tier") != "non_leaky_ctx_history_plus_command":
                continue
            if row.get("learner") != "torch_small_mlp":
                continue
            split = str(row.get("split", ""))
            true = str(row.get("true", ""))
            pred = str(row.get("pred", ""))
            counts[(split, true)][pred] += int(row.get("count") or 0)
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, ctr in counts.items():
        out[key] = [{"pred": pred, "count": int(c)} for pred, c in ctr.most_common(8)]
    return out


def _heldout_from_split(split_name: str) -> Optional[str]:
    if split_name.startswith("leave_clip_out:"):
        return split_name.split(":", 1)[1]
    return None


def _classify(
    *,
    split_support: int,
    global_support: int,
    low_n_threshold: int,
    flight_only: bool,
    blip_all: bool,
    blip_any: bool,
    same_main_order: bool,
    left_half: bool,
    left_half_only_or_heldout: bool,
) -> Tuple[str, str]:
    low_n = split_support < int(low_n_threshold) or global_support < int(low_n_threshold)
    granularity = bool(flight_only or blip_all or same_main_order)
    if granularity:
        bits = []
        if flight_only:
            bits.append("flight_or_unknown insertion/deletion")
        if blip_all:
            bits.append("all samples collapse to seen topology after dropping 1-frame run")
        elif blip_any:
            bits.append("some samples have 1-frame blip collapse evidence")
        if same_main_order:
            bits.append("non-flight main support order matches a seen topology")
        if low_n:
            bits.append(f"low_n={split_support}/{global_support} kept as caution")
        return "granularity_fragment", "; ".join(bits)
    if left_half and left_half_only_or_heldout:
        return "left_domain_coverage_gap", "topology occurs in Walk_L_To_L/Walk_L_To_R left-half coverage"
    if low_n:
        return "ambiguous_low_n", f"support below threshold: split={split_support}, global={global_support}"
    return "true_new_support_mode", "not explained by flight/blip/boundary granularity or left-half coverage"


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "heldout_split",
        "topology_key",
        "topology_tokens",
        "classification",
        "support_count",
        "global_support_count",
        "first_seen_clip",
        "first_seen_start",
        "nearest_seen_topology",
        "edit_distance",
        "flight_insert_delete_only",
        "one_frame_blip_all",
        "one_frame_blip_any",
        "duration_boundary_main_order_same",
        "side_reversal_to_seen",
        "appears_in_walk_l_to_l_or_l_to_r",
        "walk_l_to_r_support_count",
        "candidate_merge_requires_layer2_decoder",
        "primary_confusion_top_pred",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_cls = Counter(str(r["classification"]) for r in rows)
    unique_by_cls: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        unique_by_cls[str(row["classification"])].add(str(row["topology_key"]))
    merge_rows = [r for r in rows if bool(r.get("candidate_merge_requires_layer2_decoder"))]
    return {
        "row_count": int(len(rows)),
        "unique_unseen_topology_count": int(len(set(str(r["topology_key"]) for r in rows))),
        "classification_row_counts": dict(by_cls),
        "classification_unique_topology_counts": {k: int(len(v)) for k, v in sorted(unique_by_cls.items())},
        "candidate_merge_row_count": int(len(merge_rows)),
        "candidate_merge_unique_topologies": sorted(set(str(r["topology_key"]) for r in merge_rows)),
        "topology_granularity_change_allowed": False,
        "granularity_policy": (
            "Do not reduce topology granularity from this audit alone. Candidate merges require layer-2 "
            "oracle-schedule decoder validation that trajectory/contact/FK metrics do not need the distinction."
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ablation = _read_json(args.ablation_summary)
    tightening = _read_json(args.tightening_summary) if args.tightening_summary.is_file() else {}
    cfg = ablation.get("config", {}) or {}
    horizon = int(args.horizon or cfg.get("horizon", DEFAULT_HORIZON))
    context_len = int(args.context_len or cfg.get("context_len", CONTEXT_LEN_C))
    min_run_frames = int(args.min_run_frames or cfg.get("min_run_frames", 2))
    stride = int(args.stride or cfg.get("stride", 1))
    train_fraction = float(args.train_fraction if args.train_fraction is not None else cfg.get("train_fraction", 0.6))
    block_gap = int(args.block_gap if args.block_gap is not None else cfg.get("block_gap", 8))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 20260602))

    clips = _load_clips(args.npz_root, args.z_features)
    all_items = _build_items(
        clips,
        horizon=horizon,
        context_len=context_len,
        min_run_frames=min_run_frames,
        stride=stride,
    )
    all_targets = [_target_from_item(item) for item in all_items]
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    main_targets = [_target_from_item(item) for item in main_items]
    unmatched_items = [it for it in all_items if it.clip == UNMATCHED_TARGET]
    unmatched_targets = [_target_from_item(item) for item in unmatched_items]

    splits = _build_splits(
        main_items,
        train_fraction=train_fraction,
        block_gap=block_gap,
        seed=seed,
        low_n_threshold=int(args.split_low_n_threshold),
        include_random=False,
    )
    split_by_name = {s.name: s for s in splits}
    artifact_splits = [
        s for s in ablation.get("splits", []) or [] if str(s.get("kind")) != "random_optimistic_diagnostic"
    ]
    confusion = _confusion_by_true(args.confusion_csv)

    global_counts = Counter(str(t.topology_key) for t in main_targets)
    global_per_clip: Dict[str, Counter[str]] = defaultdict(Counter)
    for item, target in zip(main_items, main_targets):
        global_per_clip[str(target.topology_key)][item.clip] += 1
    unmatched_counts = Counter(str(t.topology_key) for t in unmatched_targets)

    rows: List[Dict[str, Any]] = []
    for split_rec in artifact_splits:
        split_name = str(split_rec.get("name", ""))
        split = split_by_name.get(split_name)
        train_seen = set(str(x) for x in (split_rec.get("train_topology_support", {}) or {}).keys())
        test_support = {str(k): int(v) for k, v in (split_rec.get("test_topology_support", {}) or {}).items()}
        unseen = [str(x) for x in split_rec.get("unseen_topologies", []) or []]
        heldout = _heldout_from_split(split_name)
        for topology in unseen:
            split_support = int(test_support.get(topology, 0))
            nearest = _nearest_seen(topology, train_seen)
            flight_only, flight_seen = _is_flight_insert_delete_only(topology, train_seen)
            same_main, same_main_seen = _same_main_order_candidate(topology, train_seen)
            side_rev, side_seen = _side_reversal_candidate(topology, train_seen)

            matching_items: List[AuditItem] = []
            matching_targets: List[Any] = []
            if split is not None:
                for idx in split.test_idx:
                    target = main_targets[int(idx)]
                    if str(target.topology_key) == topology:
                        matching_items.append(main_items[int(idx)])
                        matching_targets.append(target)

            collapse_keys_by_sample = []
            ambiguous_run_count = 0
            for item, target in zip(matching_items, matching_targets):
                labels = list(target.normalized_labels)
                collapse = _drop_short_run_candidates(labels, max_len=1)
                collapse_keys_by_sample.append(collapse)
                norm = item.support_contract.get("normalization", {}) or {}
                ambiguous_run_count += len(norm.get("ambiguous_runs", []) or [])
            collapse_seen_by_sample = [
                [key for key in keys if key in train_seen and key != topology] for keys in collapse_keys_by_sample
            ]
            blip_any = any(bool(x) for x in collapse_seen_by_sample)
            blip_all = bool(collapse_seen_by_sample) and all(bool(x) for x in collapse_seen_by_sample)

            first = _first_occurrence(main_items, main_targets, topology)
            per_clip = dict(global_per_clip.get(topology, Counter()))
            appears_left_half = any(int(per_clip.get(clip, 0)) > 0 for clip in LEFT_HALF_CLIPS) or int(
                unmatched_counts.get(topology, 0)
            ) > 0
            left_half_only = bool(per_clip) and all(clip in LEFT_HALF_CLIPS for clip in per_clip)
            left_half_heldout = heldout in LEFT_HALF_CLIPS if heldout else False
            classification, reason = _classify(
                split_support=split_support,
                global_support=int(global_counts.get(topology, 0)),
                low_n_threshold=int(args.topology_low_n_threshold),
                flight_only=flight_only,
                blip_all=blip_all,
                blip_any=blip_any,
                same_main_order=same_main,
                left_half=appears_left_half,
                left_half_only_or_heldout=bool(left_half_only or left_half_heldout),
            )
            conf = confusion.get((split_name, topology), [])
            top_pred = conf[0]["pred"] if conf else None
            candidate_merge = bool(flight_only or blip_any or same_main)
            rows.append(
                {
                    "split": split_name,
                    "split_kind": split_rec.get("kind"),
                    "heldout_split": heldout,
                    "topology_key": topology,
                    "topology_tokens": ">".join(_tokens(topology)),
                    "support_count": split_support,
                    "global_support_count": int(global_counts.get(topology, 0)),
                    "per_clip_support": per_clip,
                    **first,
                    **nearest,
                    "flight_insert_delete_only": bool(flight_only),
                    "flight_insert_delete_seen_topology": flight_seen,
                    "one_frame_blip_any": bool(blip_any),
                    "one_frame_blip_all": bool(blip_all),
                    "one_frame_blip_seen_collapse_keys": sorted(
                        set(key for keys in collapse_seen_by_sample for key in keys)
                    ),
                    "ambiguous_short_run_count": int(ambiguous_run_count),
                    "duration_boundary_main_order_same": bool(same_main),
                    "duration_boundary_seen_topology": same_main_seen,
                    "side_reversal_to_seen": bool(side_rev),
                    "side_reversal_seen_topology": side_seen,
                    "appears_in_walk_l_to_l_or_l_to_r": bool(appears_left_half),
                    "walk_l_to_l_support_count": int(per_clip.get("Walk_L_To_L", 0)),
                    "walk_l_to_r_support_count": int(unmatched_counts.get(topology, 0)),
                    "classification": classification,
                    "classification_reason": reason,
                    "candidate_merge_requires_layer2_decoder": candidate_merge,
                    "candidate_merge_without_layer2_decoder_allowed": False,
                    "primary_confusion_top_pred": top_pred,
                    "primary_confusion_counts": conf,
                }
            )

    summary = _summarize(rows)
    payload = {
        "task": "support_topology_granularity_coverage_audit",
        "scope": (
            "read-only granularity/coverage audit; no training; no model forward; no production "
            "trainer/runtime/gate edit; no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "ablation_summary": str(args.ablation_summary),
            "confusion_csv": str(args.confusion_csv),
            "tightening_summary": str(args.tightening_summary),
            "horizon": horizon,
            "context_len": context_len,
            "min_run_frames": min_run_frames,
            "stride": stride,
            "train_fraction": train_fraction,
            "block_gap": block_gap,
            "seed": seed,
            "topology_low_n_threshold": int(args.topology_low_n_threshold),
            "device_policy": "cpu read-only numpy",
        },
        "input_contract": {
            "ctx": {"shape_contract": "[C,281]", "actual_shape": [context_len, 281], "dtype": "float32", "device": "cpu"},
            "oracle_support_window": {
                "shape_contract": "[H,2]",
                "actual_shape": [horizon, 2],
                "dtype": "float32",
                "device": "cpu",
                "role": "metadata/audit target only, not model input",
            },
            "topology_target": {
                "dtype": "object/string",
                "device": "cpu",
                "definition": "RLE over debounced support labels with durations removed",
            },
        },
        "source_artifact_alignment": {
            "ablation_target_contract": ablation.get("target_contract", {}),
            "ablation_dataset": ablation.get("dataset", {}),
            "tightening_metric_boundary_confirmations": tightening.get("metric_boundary_confirmations", {}),
        },
        "summary": summary,
        "rows": rows,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_any_model": False,
            "forwarded_any_model": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
            "reduced_topology_granularity_for_accuracy": False,
        },
        "artifacts": {
            "summary_json": str(args.out_dir / "support_topology_granularity_coverage_summary.json"),
            "summary_md": str(args.out_dir / "support_topology_granularity_coverage_summary.md"),
            "rows_csv": str(args.out_dir / "support_topology_granularity_coverage_rows.csv"),
        },
    }
    return payload


def _write_md(path: Path, payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Support Topology Granularity / Coverage Audit",
        "",
        "Read-only audit. No training, no model forward, no production trainer/runtime/gate edit, no checkpoint mutation.",
        "",
        "## Summary",
        "",
    ]
    counts = summary.get("classification_row_counts", {})
    for cls in ("granularity_fragment", "left_domain_coverage_gap", "true_new_support_mode", "ambiguous_low_n"):
        lines.append(f"- {cls}: `{int(counts.get(cls, 0))}` split-topology rows")
    lines.extend(
        [
            f"- unique unseen topologies: `{summary.get('unique_unseen_topology_count')}`",
            f"- candidate merge topologies requiring layer-2 decoder validation: `{summary.get('candidate_merge_unique_topologies')}`",
            f"- topology granularity change allowed now: `{summary.get('topology_granularity_change_allowed')}`",
            "",
            "## Rows",
            "",
            "| split | topology | n | nearest seen | edit | flight/blip/boundary | left-half | class |",
            "|---|---|---:|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        flags = ",".join(
            name
            for name, ok in (
                ("flight", row.get("flight_insert_delete_only")),
                ("blip", row.get("one_frame_blip_any")),
                ("boundary", row.get("duration_boundary_main_order_same")),
                ("reversal", row.get("side_reversal_to_seen")),
            )
            if ok
        ) or "-"
        lines.append(
            f"| {row['split']} | `{row['topology_key']}` | {row['support_count']} | "
            f"`{row.get('nearest_seen_topology')}` | {row.get('edit_distance')} | {flags} | "
            f"{row.get('appears_in_walk_l_to_l_or_l_to_r')} | `{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- This audit does not lower topology granularity for accuracy.",
            "- Any candidate merge must be validated by layer-2 oracle-schedule trajectory decoding with realized motion/FK metrics.",
        ]
    )
    _dump_md(path, lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--ablation-summary", type=Path, default=DEFAULT_ABLATION_SUMMARY)
    p.add_argument("--confusion-csv", type=Path, default=DEFAULT_CONFUSION)
    p.add_argument("--tightening-summary", type=Path, default=DEFAULT_TIGHTENING_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--context-len", type=int, default=None)
    p.add_argument("--min-run-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--block-gap", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-low-n-threshold", type=int, default=20)
    p.add_argument("--topology-low-n-threshold", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(args.out_dir / "support_topology_granularity_coverage_summary.json", payload)
    _write_md(args.out_dir / "support_topology_granularity_coverage_summary.md", payload)
    _write_rows_csv(args.out_dir / "support_topology_granularity_coverage_rows.csv", payload["rows"])
    print(f"wrote {args.out_dir / 'support_topology_granularity_coverage_summary.json'}")
    print(f"wrote {args.out_dir / 'support_topology_granularity_coverage_summary.md'}")
    print(f"wrote {args.out_dir / 'support_topology_granularity_coverage_rows.csv'}")


if __name__ == "__main__":
    main()
