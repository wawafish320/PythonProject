#!/usr/bin/env python3
"""Support-schedule oracle feasibility probe for action handoff middle generation.

Read-only probe. No training, no checkpoint mutation, no production runtime/gate
change. This answers a narrower question after acceptance replay calibration:

Given a Walk_F start, commanded yaw/cond path, and soft endpoint cue, is the
remaining bridge freedom mostly support/contact schedule and timing?

It does not generate motion. It replays true continuous target middle windows as
oracle motion, then ablates only the command/support cue channels:

* endpoint_only: target oracle motion with Walk_F command + Walk_F support;
* endpoint_command: target oracle motion with target command + Walk_F support;
* endpoint_command_oracle_support: target oracle motion with target command +
  true target support schedule;
* shuffled_support / wrong_side_support: target oracle motion with target
  command but corrupted support schedules.

If corrupted schedules still pass, the acceptance contract is missing a support
condition. If oracle support passes while endpoint/command-only rows fail on
support, explicit support cue is a strong candidate for the first generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_TOPK,
    TURN_CLIPS,
    WALK_F,
    full_state_align,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    DEFAULT_NPZ_ROOT,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    CONTACT_THRESHOLD,
    ClipData,
    _bridgeability_from_deltas,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _evaluate_sequence,
    _fmt,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
    _step_angvel_rms,
    _step_l2,
    _support_label,
)


DEFAULT_HORIZON = 16
DEFAULT_BASELINE_QUANTILE = 99.5
DEFAULT_BRIDGE_BUDGET_QUANTILE = 95.0
DEFAULT_POSE_BUCKET_RADIUS = 0.030


def _support_labels(contact: np.ndarray) -> List[str]:
    return [_support_label(c) for c in np.asarray(contact, dtype=np.float32)]


def _support_entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = float(len(labels))
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0))


def _support_transition_count(labels: Sequence[str]) -> int:
    return int(sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b))


def _dominant_support(contact: np.ndarray) -> str:
    c = np.asarray(contact, dtype=np.float32)
    right = float(np.sum(c[:, 0] > CONTACT_THRESHOLD))
    left = float(np.sum(c[:, 1] > CONTACT_THRESHOLD))
    if right > left + 1:
        return "right"
    if left > right + 1:
        return "left"
    return "mixed"


def _support_signature(contact: np.ndarray) -> str:
    labels = _support_labels(contact)
    return f"{labels[0]}->{labels[-1]}|dom={_dominant_support(contact)}|trans={_support_transition_count(labels)}"


def _support_bridge_ok(contact: np.ndarray, start_label: str, end_label: str, horizon_bridgeable: bool) -> bool:
    labels = _support_labels(contact)
    if not labels:
        return False
    return bool(horizon_bridgeable and labels[0] == start_label and labels[-1] == end_label)


def _cyclic_walk_indices(start: int, horizon: int, n: int) -> np.ndarray:
    return (np.arange(int(start), int(start) + int(horizon), dtype=np.int64) % int(n)).astype(np.int64)


def _root_shifted_target_sequence(walk: ClipData, target: ClipData, phi: int, onset: int, horizon: int) -> Dict[str, np.ndarray]:
    s = int(onset)
    e = int(onset) + int(horizon)
    root_shift = walk.root_pos[int(phi)] - target.root_pos[s]
    return {
        "rot6d": target.rot6d[s:e].copy(),
        "root_pos": target.root_pos[s:e].copy() + root_shift.reshape(1, 3),
        "root_vel": target.root_vel[s:e].copy(),
        "bone_angvel": target.bone_angvel[s:e].copy(),
        "cond_dir": target.cond_dir[s:e].copy(),
        "contact": target.contact[s:e].copy(),
        "yaw_rate": target.yaw_rate[s:e].copy(),
    }


def _replace_seq(seq: Mapping[str, np.ndarray], **updates: np.ndarray) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v).copy() for k, v in seq.items()}
    for key, value in updates.items():
        out[key] = np.asarray(value).copy()
    return out


def _walk_cue_sequence(walk: ClipData, phi: int, horizon: int) -> Dict[str, np.ndarray]:
    idx = _cyclic_walk_indices(phi, horizon, walk.contact.shape[0])
    return {
        "cond_dir": walk.cond_dir[idx].copy(),
        "contact": walk.contact[idx].copy(),
        "yaw_rate": walk.yaw_rate[idx].copy(),
    }


def _shuffled_contact(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float32).copy()
    if c.shape[0] <= 1:
        return c
    return np.roll(c, shift=max(1, c.shape[0] // 2), axis=0)


def _wrong_side_contact(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float32).copy()
    if c.shape[1] >= 2:
        c = c[:, [1, 0]]
    return c


def _bridge_budgets(clips: Mapping[str, ClipData], quantile: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, clip in clips.items():
        out[name] = {
            "angvel": _safe_percentile(_step_angvel_rms(clip.bone_angvel), quantile),
            "rootvel": _safe_percentile(_step_l2(clip.root_vel), quantile),
            "yaw": _safe_percentile(np.abs(np.diff(clip.yaw_rate)), quantile),
            "contact": _safe_percentile(_step_l2(clip.contact), quantile),
        }
    return out


def _endpoint_bridge(
    walk: ClipData,
    target: ClipData,
    phi: int,
    onset: int,
    budgets: Mapping[str, float],
    *,
    horizon: int,
    groundable: bool,
) -> Dict[str, Any]:
    deltas = {
        "angvel": float(np.sqrt(np.mean((target.bone_angvel[onset] - walk.bone_angvel[phi]) ** 2))),
        "rootvel": float(np.linalg.norm(target.root_vel[onset] - walk.root_vel[phi])),
        "yaw": float(abs(target.yaw_rate[onset] - walk.yaw_rate[phi])),
        "contact": float(np.linalg.norm(target.contact[onset] - walk.contact[phi])),
    }
    bridge = _bridgeability_from_deltas(deltas, budgets, horizon=horizon, groundable=groundable)
    bridge["deltas"] = deltas
    bridge["budgets"] = dict(budgets)
    return bridge


def _evaluate_condition(
    *,
    condition: str,
    target: str,
    seq: Mapping[str, np.ndarray],
    target_bands: Mapping[str, Any],
    skeleton: Any,
    start_phase: str,
    expected_label: str,
    endpoint_bridgeability: bool,
    endpoint_details: Mapping[str, Any],
) -> Dict[str, Any]:
    row = _evaluate_sequence(
        seq,
        target=target,
        target_bands=target_bands,
        skeleton=skeleton,
        case=f"support_oracle:{condition}",
        expected_label=expected_label,
        start_phase=start_phase,
        endpoint_bridgeability=endpoint_bridgeability,
        endpoint_details=endpoint_details,
    )
    labels = _support_labels(seq["contact"])
    row["support_schedule"] = {
        "labels": labels,
        "signature": _support_signature(seq["contact"]),
        "entropy_bits": _support_entropy(labels),
        "transition_count": _support_transition_count(labels),
        "dominant_support": _dominant_support(seq["contact"]),
    }
    return row


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case = str(row.get("case"))
        rec = out.setdefault(case, {"n": 0, "pass_n": 0, "failed_family_counts": {}, "entropy": []})
        rec["n"] += 1
        if row.get("pass"):
            rec["pass_n"] += 1
        failed = str(row.get("failed_family") or "")
        for fam in [x for x in failed.split(",") if x]:
            rec["failed_family_counts"][fam] = rec["failed_family_counts"].get(fam, 0) + 1
        sched = row.get("support_schedule", {})
        if isinstance(sched, Mapping):
            ent = sched.get("entropy_bits")
            if ent is not None and math.isfinite(float(ent)):
                rec["entropy"].append(float(ent))
    for rec in out.values():
        rec["pass_rate"] = float(rec["pass_n"] / max(1, rec["n"]))
        ent = rec.pop("entropy")
        rec["support_entropy_mean_bits"] = float(np.mean(ent)) if ent else None
    return out


def _pose_bucket_multisolution(
    clips: Mapping[str, ClipData],
    *,
    horizon: int,
    radius: float,
    only_clips: Sequence[str],
) -> Dict[str, Any]:
    buckets: List[Dict[str, Any]] = []
    for clip_name in only_clips:
        clip = clips[clip_name]
        max_start = int(clip.rot6d.shape[0]) - int(horizon)
        for start in range(max_start + 1):
            end = start + int(horizon) - 1
            feature = np.concatenate([clip.rot6d[start], clip.rot6d[end]], axis=0).astype(np.float32)
            signature = _support_signature(clip.contact[start : start + int(horizon)])
            assigned = False
            for bucket in buckets:
                d = float(np.linalg.norm(feature - bucket["center"]) / math.sqrt(max(1, feature.shape[0])))
                if d <= float(radius):
                    bucket["items"].append((clip_name, start, signature, d))
                    assigned = True
                    break
            if not assigned:
                buckets.append({"center": feature, "items": [(clip_name, start, signature, 0.0)]})

    multi = []
    for idx, bucket in enumerate(buckets):
        sigs = sorted({item[2] for item in bucket["items"]})
        if len(sigs) > 1:
            multi.append(
                {
                    "bucket": idx,
                    "n": len(bucket["items"]),
                    "signature_count": len(sigs),
                    "signatures": sigs,
                    "examples": [
                        {"clip": item[0], "start": int(item[1]), "signature": item[2], "dist_to_center": float(item[3])}
                        for item in bucket["items"][:8]
                    ],
                }
            )
    return {
        "pose_bucket_radius": float(radius),
        "horizon": int(horizon),
        "n_buckets": int(len(buckets)),
        "multisolution_bucket_count": int(len(multi)),
        "max_signatures_per_bucket": int(max([len({item[2] for item in b["items"]}) for b in buckets] or [0])),
        "multi_examples": multi[:10],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "target",
        "start_phase",
        "expected_label",
        "pass",
        "failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
        "support_signature",
        "support_entropy_bits",
        "support_transition_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            sched = row.get("support_schedule", {})
            writer.writerow(
                {
                    **{k: row.get(k) for k in fields[:12]},
                    "support_signature": sched.get("signature") if isinstance(sched, Mapping) else None,
                    "support_entropy_bits": sched.get("entropy_bits") if isinstance(sched, Mapping) else None,
                    "support_transition_count": sched.get("transition_count") if isinstance(sched, Mapping) else None,
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only support-schedule oracle feasibility probe.")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--baseline-quantile", type=float, default=DEFAULT_BASELINE_QUANTILE)
    p.add_argument("--bridge-budget-quantile", type=float, default=DEFAULT_BRIDGE_BUDGET_QUANTILE)
    p.add_argument("--pose-bucket-radius", type=float, default=DEFAULT_POSE_BUCKET_RADIUS)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_support_schedule_oracle_feasibility_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    budgets = _bridge_budgets(clips, quantile=float(args.bridge_budget_quantile))
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}

    walk = clips[WALK_F]
    rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    oracle_support: Dict[str, Any] = {}
    h = int(args.horizon)

    for target_name in TURN_CLIPS:
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
            endpoint_rows.append(
                {
                    "target": target_name,
                    "groundable": bool(align.groundable),
                    "matched_pair_available": False,
                    "pose_d": float(align.full_state_pose_d),
                    "contact_d": float(align.full_state_contact_d),
                    "decision": "separate_ungroundable_no_oracle_matched_endpoint",
                }
            )
            continue

        phi = int(pair["phi"])
        onset = int(pair["onset"])
        if onset + h > target.rot6d.shape[0]:
            endpoint_rows.append(
                {
                    "target": target_name,
                    "groundable": bool(align.groundable),
                    "matched_pair_available": True,
                    "phi": phi,
                    "onset": onset,
                    "decision": "skip_too_short_for_horizon",
                }
            )
            continue

        bridge = _endpoint_bridge(
            walk,
            target,
            phi,
            onset,
            budgets[target_name],
            horizon=h,
            groundable=bool(align.groundable),
        )
        start_label = _support_label(walk.contact[phi])
        oracle_seq = _root_shifted_target_sequence(walk, target, phi, onset, h)
        target_end_label = _support_label(oracle_seq["contact"][-1])
        walk_cue = _walk_cue_sequence(walk, phi, h)
        endpoint_details = {
            "matched_pair_available": True,
            "phi": phi,
            "onset": onset,
            "pose_d": float(pair.get("current_pose_l2", align.full_state_pose_d)),
            "contact_d": float(align.full_state_contact_d),
            "start_support": start_label,
            "oracle_end_support": target_end_label,
            **bridge,
        }
        endpoint_rows.append({"target": target_name, **endpoint_details})

        oracle_labels = _support_labels(oracle_seq["contact"])
        oracle_support[target_name] = {
            "phi": phi,
            "onset": onset,
            "labels": oracle_labels,
            "signature": _support_signature(oracle_seq["contact"]),
            "entropy_bits": _support_entropy(oracle_labels),
            "transition_count": _support_transition_count(oracle_labels),
            "dominant_support": _dominant_support(oracle_seq["contact"]),
        }

        condition_specs = [
            (
                "endpoint_only",
                _replace_seq(oracle_seq, cond_dir=walk_cue["cond_dir"], yaw_rate=walk_cue["yaw_rate"], contact=walk_cue["contact"]),
                "diagnostic_fail_or_incomplete",
            ),
            (
                "endpoint_command",
                _replace_seq(oracle_seq, contact=walk_cue["contact"]),
                "diagnostic_fail_or_incomplete",
            ),
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
        ]

        for condition, seq, expected in condition_specs:
            schedule_ok = _support_bridge_ok(
                seq["contact"],
                start_label,
                target_end_label,
                bool(bridge.get("horizon_bridgeable")),
            )
            details = dict(endpoint_details)
            details["support_schedule_bridge_ok"] = bool(schedule_ok)
            rows.append(
                _evaluate_condition(
                    condition=condition,
                    target=target_name,
                    seq=seq,
                    target_bands=bands[target_name],
                    skeleton=skeleton,
                    start_phase=f"phi={phi};onset={onset};H={h}",
                    expected_label=expected,
                    endpoint_bridgeability=bool(schedule_ok),
                    endpoint_details=details,
                )
            )

    summary = _summarize_rows(rows)
    groundable_targets = [r["target"] for r in endpoint_rows if r.get("matched_pair_available", True) and r.get("horizon_bridgeable")]
    pose_bucket = _pose_bucket_multisolution(
        clips,
        horizon=h,
        radius=float(args.pose_bucket_radius),
        only_clips=[t for t in TURN_CLIPS if t in clips],
    )
    endpoint_coverage = {
        "matched_targets": int(sum(1 for r in endpoint_rows if r.get("matched_pair_available", False))),
        "horizon_bridgeable": int(sum(1 for r in endpoint_rows if r.get("horizon_bridgeable", False))),
        "one_frame_bridgeable": int(sum(1 for r in endpoint_rows if r.get("one_frame_bridgeable", False))),
        "ungroundable_or_unmatched": [r["target"] for r in endpoint_rows if not r.get("matched_pair_available", True)],
    }
    verdict = {
        "oracle_support_pass_rate": summary.get("support_oracle:endpoint_command_oracle_support", {}).get("pass_rate"),
        "endpoint_only_pass_rate": summary.get("support_oracle:endpoint_only", {}).get("pass_rate"),
        "endpoint_command_pass_rate": summary.get("support_oracle:endpoint_command", {}).get("pass_rate"),
        "shuffled_support_pass_rate": summary.get("support_oracle:endpoint_command_shuffled_support", {}).get("pass_rate"),
        "wrong_side_support_pass_rate": summary.get("support_oracle:endpoint_command_wrong_side_support", {}).get("pass_rate"),
        "support_metric_leak_detected": bool(
            (summary.get("support_oracle:endpoint_command_shuffled_support", {}).get("pass_rate") or 0.0) > 0.0
            or (summary.get("support_oracle:endpoint_command_wrong_side_support", {}).get("pass_rate") or 0.0) > 0.0
        ),
        "groundable_horizon_bridgeable_targets": groundable_targets,
    }

    payload = {
        "task": "support_schedule_oracle_feasibility_probe",
        "scope": "read-only oracle replay; no training; no train owner path edits",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "two_frame_summary": str(args.two_frame_summary),
            "horizon": h,
            "baseline_quantile": float(args.baseline_quantile),
            "bridge_budget_quantile": float(args.bridge_budget_quantile),
            "pose_bucket_radius": float(args.pose_bucket_radius),
        },
        "input_contract": {
            "ctx": "[C,281] float32/cpu implied by Walk_F matched phi; context is not fed to a model",
            "command": "cond_dir/yaw_rate from target or Walk_F cue depending on condition",
            "support_schedule": "[H,2] float32/cpu contact schedule cue",
            "oracle_middle": "[H,281] float32/cpu equivalent channels from true target clip",
            "bone_angvel_witness": "[H,138] float32/cpu from processed npz bone_ang_vel",
        },
        "endpoint_bridgeability": endpoint_rows,
        "oracle_support": oracle_support,
        "rows": rows,
        "summary": summary,
        "endpoint_coverage": endpoint_coverage,
        "pose_bucket_multisolution": pose_bucket,
        "verdict": verdict,
    }
    _dump_json(out_dir / "support_schedule_oracle_feasibility_summary.json", payload)
    _write_csv(out_dir / "support_schedule_oracle_feasibility_rows.csv", rows)

    lines: List[str] = []
    lines.append("# Support-Schedule Oracle Feasibility Probe")
    lines.append("")
    lines.append("Read-only oracle replay. No training, checkpoint mutation, production gate change, or `train/` owner edit.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- oracle support pass rate: `{_fmt(verdict['oracle_support_pass_rate'])}`")
    lines.append(f"- endpoint-only pass rate: `{_fmt(verdict['endpoint_only_pass_rate'])}`")
    lines.append(f"- endpoint+command pass rate: `{_fmt(verdict['endpoint_command_pass_rate'])}`")
    lines.append(f"- shuffled support pass rate: `{_fmt(verdict['shuffled_support_pass_rate'])}`")
    lines.append(f"- wrong-side support pass rate: `{_fmt(verdict['wrong_side_support_pass_rate'])}`")
    lines.append(f"- support metric leak detected: `{bool(verdict['support_metric_leak_detected'])}`")
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    lines.append("| condition | n | pass_rate | entropy_bits | top failed families |")
    lines.append("|---|---:|---:|---:|---|")
    for case, rec in sorted(summary.items()):
        failed = rec.get("failed_family_counts", {}) or {}
        top_failed = ", ".join(f"{k}:{v}" for k, v in sorted(failed.items(), key=lambda kv: (-kv[1], kv[0]))[:4])
        lines.append(
            f"| {case} | {int(rec.get('n', 0))} | {_fmt(rec.get('pass_rate'))} | "
            f"{_fmt(rec.get('support_entropy_mean_bits'))} | {top_failed or '-'} |"
        )
    lines.append("")
    lines.append("## Endpoint Coverage")
    lines.append("")
    lines.append("| target | matched | horizon_bridgeable | one_frame | support | max_needed | pose_d | contact_d |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for row in endpoint_rows:
        support = f"{row.get('start_support', '-') }->{row.get('oracle_end_support', '-')}"
        max_needed = row.get("max_needed_frames")
        max_s = "null" if max_needed is None else ("inf" if int(max_needed) >= int(10**9) else str(max_needed))
        lines.append(
            f"| {row['target']} | {bool(row.get('matched_pair_available', True))} | "
            f"{bool(row.get('horizon_bridgeable', False))} | {bool(row.get('one_frame_bridgeable', False))} | "
            f"{support} | {max_s} | {_fmt(row.get('pose_d'))} | {_fmt(row.get('contact_d'))} |"
        )
    lines.append("")
    lines.append("## Support Multimodality")
    lines.append("")
    lines.append(f"- pose bucket radius: `{_fmt(pose_bucket['pose_bucket_radius'], 4)}`")
    lines.append(f"- buckets: `{pose_bucket['n_buckets']}`")
    lines.append(f"- multi-signature buckets: `{pose_bucket['multisolution_bucket_count']}`")
    lines.append(f"- max signatures per bucket: `{pose_bucket['max_signatures_per_bucket']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{out_dir / 'support_schedule_oracle_feasibility_summary.json'}`")
    lines.append(f"- `{out_dir / 'support_schedule_oracle_feasibility_rows.csv'}`")
    _dump_md(out_dir / "support_schedule_oracle_feasibility_summary.md", lines)

    print(f"wrote {out_dir / 'support_schedule_oracle_feasibility_summary.md'}")
    print(f"wrote {out_dir / 'support_schedule_oracle_feasibility_summary.json'}")


if __name__ == "__main__":
    main()
