#!/usr/bin/env python3
"""Debug-only band audit for action-handoff acceptance metrics.

This tool does not train or forward production Trainer/runtime/gate code. It
audits the adjusted acceptance bands against continuous baseline motion, then
tests each zero-slack relabel through the same debug guard semantics used by
the one-window acceptance artifact.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    full_state_align,
)
from tools.run_action_handoff_adjusted_acceptance_guard import (  # noqa: E402
    DEFAULT_COMMAND_DEMOTION_ROWS,
    _adjusted_eval_sequence,
    _artifact_negative_rows,
    _bridge_budgets,
    _command_demotion_negative_summary,
    _event_masks_from_contact,
    _load_gate_pred_row,
    _summaries_by_case,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_BONE_BRIDGE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_REGIME_BRIDGE,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    _bridgeability_from_deltas,
    _calibrate_baselines,
    _contacted_foot_speeds,
    _dump_json,
    _dump_md,
    _fmt,
    _heading_error_rad,
    _jsonify,
    _load_clips,
    _load_skeleton_meta,
    _make_hard_seam_sequence,
    _make_linear_proxy_sequence,
    _make_one_frame_switch_sequence,
    _rms_rows,
    _safe_percentile,
    _step_angvel_component_p95,
    _step_angvel_rms,
    _step_l2,
    _step_pose_l2,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _guard_path_identity,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_band_audit_20260604")
DEFAULT_DOC_PATH = Path("docs/aperiodic_transition/2026-06-04_band_audit.md")
DEFAULT_FINAL_ONE_WINDOW_PRED = Path(
    "debug_output/"
    "_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_"
    "lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/"
    "causal3_minimax_one_window_pred_raw.npz"
)
EPS = 1e-8


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    family: str
    display: str
    event_aware: bool = False
    target_only: bool = False
    heading_effective: bool = False


METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec("bone_angvel_level_rms", "regime_reached", "bone_angvel_level_rms", target_only=True),
    MetricSpec("angvel_step_rms", "rate_budget", "angvel_step_rms", event_aware=True),
    MetricSpec("angvel_step_component_p95", "rate_budget", "angvel_step_component_p95"),
    MetricSpec("rootvel_step_l2", "rate_budget", "rootvel_step_l2"),
    MetricSpec("yaw_rate_step_abs", "rate_budget", "yaw_rate_step_abs"),
    MetricSpec("contact_step_l2", "support_honesty", "contact_step_l2", event_aware=True),
    MetricSpec("foot_slip_contacted_speed_mps", "support_honesty", "foot_slip_contacted_speed_mps"),
    MetricSpec("heading_error_rad", "command_response", "heading_error_rad", heading_effective=True),
    MetricSpec("pose_step_l2", "pose_continuity", "pose_step_l2"),
)


@dataclass(frozen=True)
class BandOverride:
    target: str
    metric: str
    old_band: float
    new_band: float


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _metric_spec(metric: str) -> MetricSpec:
    for spec in METRIC_SPECS:
        if spec.metric == metric:
            return spec
    raise KeyError(metric)


def _clip_metric_samples(
    clip: Any,
    metric: str,
    skeleton: Any,
    *,
    level_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if metric == "pose_step_l2":
        return _step_pose_l2(clip.rot6d), None
    if metric == "angvel_step_rms":
        return _step_angvel_rms(clip.bone_angvel), _event_masks_from_contact(clip.contact, event_window=1)["step_mask"]
    if metric == "angvel_step_component_p95":
        return _step_angvel_component_p95(clip.bone_angvel), None
    if metric == "rootvel_step_l2":
        return _step_l2(clip.root_vel), None
    if metric == "yaw_rate_step_abs":
        return np.abs(np.diff(np.asarray(clip.yaw_rate, dtype=np.float64).reshape(-1))), None
    if metric == "contact_step_l2":
        return _step_l2(clip.contact), _event_masks_from_contact(clip.contact, event_window=1)["step_mask"]
    if metric == "heading_error_rad":
        return _heading_error_rad(clip.root_vel, clip.cond_dir), None
    if metric == "foot_slip_contacted_speed_mps":
        return _contacted_foot_speeds(clip, skeleton), None
    if metric == "bone_angvel_level_rms":
        center = np.asarray(level_center, dtype=np.float32).reshape(1, ANGVEL_DIM)
        return _rms_rows(np.asarray(clip.bone_angvel, dtype=np.float32).reshape(-1, ANGVEL_DIM) - center), None
    raise KeyError(metric)


def _baseline_samples_for_target(
    clips: Mapping[str, Any],
    target: str,
    metric: str,
    skeleton: Any,
    *,
    level_center: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str]:
    spec = _metric_spec(metric)
    domains = [target] if spec.target_only else [WALK_F, target]
    vals: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for name in domains:
        samples, event_mask = _clip_metric_samples(
            clips[name],
            metric,
            skeleton,
            level_center=level_center if name == target else None,
        )
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
        vals.append(arr)
        if event_mask is None:
            masks.append(np.zeros((arr.size,), dtype=bool))
        else:
            mask = np.asarray(event_mask, dtype=bool).reshape(-1)
            if mask.size != arr.size:
                mask = np.zeros((arr.size,), dtype=bool)
            masks.append(mask)
    if vals:
        all_vals = np.concatenate(vals, axis=0)
        all_masks = np.concatenate(masks, axis=0)
    else:
        all_vals = np.zeros((0,), dtype=np.float64)
        all_masks = np.zeros((0,), dtype=bool)
    return all_vals, all_masks, "+".join(domains)


def _copy_band_tables(
    raw_bands: Mapping[str, Mapping[str, Any]],
    reconstructed_bands: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    return copy.deepcopy(raw_bands), copy.deepcopy(reconstructed_bands)


def _apply_overrides_to_bands(
    raw_bands: Dict[str, Dict[str, Any]],
    reconstructed_bands: Dict[str, Dict[str, Any]],
    overrides: Sequence[BandOverride],
) -> None:
    for override in overrides:
        for table in (raw_bands, reconstructed_bands):
            if override.target not in table:
                continue
            table[override.target][override.metric] = float(override.new_band)
            if override.metric == "foot_slip_contacted_speed_mps":
                table[override.target]["foot_slip_contacted_speed_p95_mps"] = float(override.new_band)
                foot = table[override.target].setdefault("foot_slip", {})
                if isinstance(foot, dict):
                    foot["contacted_speed_p95_mps"] = float(override.new_band)


def _continuous_gt_pass_for_target(rows: Sequence[Mapping[str, Any]], target: str) -> bool:
    subset = [r for r in rows if r.get("target") == target and r.get("current_source") == "reconstructed"]
    return all(
        _finite(r.get("baseline_p95")) <= _finite(r.get("new_or_effective_band"), r.get("effective_current_band")) + EPS
        for r in subset
    )


def _build_metric_rows(
    *,
    clips: Mapping[str, Any],
    skeleton: Any,
    raw_bands: Mapping[str, Mapping[str, Any]],
    reconstructed_bands: Mapping[str, Mapping[str, Any]],
    continuous_quantile: float,
    zero_slack_headroom_ratio: float,
    zero_slack_abs_tol: float,
    heading_tolerance_rad: float,
) -> Tuple[List[Dict[str, Any]], List[BandOverride]]:
    rows: List[Dict[str, Any]] = []
    candidates: List[BandOverride] = []
    for target in TURN_CLIPS:
        target_bands = reconstructed_bands.get(target) or raw_bands.get(target)
        if not target_bands:
            continue
        current_source = "reconstructed" if target in reconstructed_bands else "raw_only"
        for spec in METRIC_SPECS:
            if spec.metric not in target_bands:
                continue
            current = _finite(target_bands.get(spec.metric))
            effective = max(current, float(heading_tolerance_rad)) if spec.heading_effective else current
            level_center = (
                np.asarray(target_bands.get("bone_angvel_level_center"), dtype=np.float32)
                if spec.metric == "bone_angvel_level_rms"
                else None
            )
            samples, event_mask, domain = _baseline_samples_for_target(
                clips,
                target,
                spec.metric,
                skeleton,
                level_center=level_center,
            )
            finite = samples[np.isfinite(samples)]
            if spec.event_aware:
                verdict_samples = samples[np.isfinite(samples) & ~event_mask]
                verdict_basis = "non_event_continuous_samples"
            else:
                verdict_samples = finite
                verdict_basis = "continuous_samples"
            p50 = _safe_percentile(finite, 50.0)
            p95 = _safe_percentile(finite, 95.0)
            p99 = _safe_percentile(finite, 99.0)
            vp50 = _safe_percentile(verdict_samples, 50.0)
            vp95 = _safe_percentile(verdict_samples, 95.0)
            vpq = _safe_percentile(verdict_samples, continuous_quantile)
            headroom_abs = float(effective - vpq)
            headroom_ratio = float(headroom_abs / max(abs(effective), EPS))
            p95_headroom_abs = float(effective - vp95)
            close_tol = max(float(zero_slack_abs_tol), abs(effective) * float(zero_slack_headroom_ratio))
            can_relabel = current_source == "reconstructed"
            zero_slack = bool(
                can_relabel
                and (
                    vpq > effective + float(zero_slack_abs_tol)
                    or p95_headroom_abs <= close_tol
                )
            )
            verdict = "zero-slack" if zero_slack else ("has-slack" if can_relabel else "audit-only")
            relabel_needed = bool(zero_slack and vpq > effective + float(zero_slack_abs_tol))
            new_band = vpq if relabel_needed else effective
            row = {
                "target": target,
                "metric": spec.metric,
                "family": spec.family,
                "current_source": current_source,
                "continuous_domain": domain,
                "sample_count": int(finite.size),
                "verdict_sample_count": int(np.asarray(verdict_samples).reshape(-1).size),
                "current_band": current,
                "effective_current_band": effective,
                "baseline_p50": p50,
                "baseline_p95": p95,
                "baseline_p99": p99,
                "verdict_basis": verdict_basis,
                "verdict_p50": vp50,
                "verdict_p95": vp95,
                "verdict_pq": vpq,
                "continuous_quantile": float(continuous_quantile),
                "headroom_abs": headroom_abs,
                "headroom_ratio": headroom_ratio,
                "verdict": verdict,
                "new_or_effective_band": new_band,
                "relabel_candidate": bool(relabel_needed),
                "relabel_applied": False,
                "relabel_rejected_reason": "",
            }
            rows.append(row)
            if relabel_needed:
                candidates.append(BandOverride(target=target, metric=spec.metric, old_band=effective, new_band=new_band))
    return rows, candidates


def _run_guard_with_bands(
    *,
    args: argparse.Namespace,
    clips: Mapping[str, Any],
    skeleton: Any,
    main_items: Sequence[Any],
    raw_bands: Mapping[str, Mapping[str, Any]],
    reconstructed_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Dict[str, Any]:
    gate_adjusted, gate_original = _load_gate_pred_row(
        args=args,
        main_items=main_items,
        baseline_bands=reconstructed_bands,
        support_bands=support_bands,
        skeleton=skeleton,
    )

    shortcut_rows: List[Dict[str, Any]] = []
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}
    budgets = _bridge_budgets(clips, float(args.bridge_budget_quantile))
    walk = clips[WALK_F]
    for target in TURN_CLIPS:
        if target not in raw_bands:
            continue
        target_clip = clips[target]
        align = full_state_align(
            walk.state281,
            target_clip.state281[0],
            topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        pair = matched_pairs.get(target)
        phi = int(pair["phi"]) if pair else int(align.full_state_phi)
        onset = int(pair["onset"]) if pair else 0
        deltas = {
            "angvel": float(np.sqrt(np.mean((target_clip.bone_angvel[onset] - walk.bone_angvel[phi]) ** 2))),
            "rootvel": float(np.linalg.norm(target_clip.root_vel[onset] - walk.root_vel[phi])),
            "yaw": float(abs(target_clip.yaw_rate[onset] - walk.yaw_rate[phi])),
            "contact": float(np.linalg.norm(target_clip.contact[onset] - walk.contact[phi])),
        }
        bridge = _bridgeability_from_deltas(
            deltas,
            budgets[target],
            horizon=int(args.horizon),
            groundable=bool(align.groundable),
        )
        bridge.update({"phi": phi, "onset": onset, "deltas": deltas, "budgets": budgets[target]})
        if not pair:
            continue
        for case, seq, endpoint_ok in (
            ("negative_control:matched_hard_seam", _make_hard_seam_sequence(walk, target_clip, phi, onset), bool(bridge["one_frame_bridgeable"])),
            (
                "negative_control:one_frame_angvel_root_switch",
                _make_one_frame_switch_sequence(walk, target_clip, phi, onset),
                bool(bridge["one_frame_bridgeable"]),
            ),
            (
                "negative_control:linear_pose_contact_proxy",
                _make_linear_proxy_sequence(walk, target_clip, phi, onset, int(args.horizon)),
                bool(bridge["horizon_bridgeable"]),
            ),
        ):
            shortcut_rows.append(
                _adjusted_eval_sequence(
                    seq,
                    target=target,
                    target_bands=raw_bands[target],
                    skeleton=skeleton,
                    case=case,
                    expected_label="fail",
                    start_phase=f"phi={phi};onset={onset};H={len(seq['yaw_rate'])}",
                    endpoint_bridgeability=endpoint_ok,
                    event_window=int(args.event_window),
                    heading_tolerance_rad=float(args.heading_tolerance_rad),
                    endpoint_details=bridge,
                )
            )
    shortcut_rows.extend(_artifact_negative_rows(args))
    shortcut_summary = _summaries_by_case(shortcut_rows)
    command_summary = _command_demotion_negative_summary(Path(args.command_demotion_rows))
    shortcut_fail = all(float(rec.get("adjusted_pass_rate", 0.0) or 0.0) == 0.0 for rec in shortcut_summary.values())
    command_fail = bool(command_summary.get("available")) and int(command_summary.get("demoted_negative_pass_count", 1)) == 0
    gate_pass = bool(gate_adjusted.get("adjusted_pass", False))
    guard_identity = _guard_path_identity(
        items=main_items,
        idxs=tuple(range(len(main_items))),
        baseline_bands=reconstructed_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=1.0,
    )
    return {
        "label": label,
        "gate_w4096_adjusted": gate_adjusted,
        "gate_w4096_original_row": gate_original,
        "shortcut_negative_summary_by_case": shortcut_summary,
        "command_demotion_negative_summary": command_summary,
        "guard_path_identity": guard_identity,
        "verdict": {
            "gate_w4096_full_family_pass": bool(gate_pass),
            "shortcut_negative_controls_still_fail": bool(shortcut_fail),
            "command_demotion_negative_controls_still_fail": bool(command_fail),
            "guard_path_identity_pass": bool(guard_identity.get("passed", False)),
            "decision": (
                "adjusted_acceptance_guard_passed_ready_for_8window_debug_sweep"
                if gate_pass and shortcut_fail and command_fail and bool(guard_identity.get("passed", False))
                else "adjusted_acceptance_guard_failed_do_not_run_8window"
            ),
        },
    }


def _write_per_metric_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "target",
        "metric",
        "family",
        "current_source",
        "continuous_domain",
        "sample_count",
        "verdict_sample_count",
        "current_band",
        "effective_current_band",
        "baseline_p50",
        "baseline_p95",
        "baseline_p99",
        "verdict_basis",
        "verdict_p50",
        "verdict_p95",
        "verdict_pq",
        "continuous_quantile",
        "headroom_abs",
        "headroom_ratio",
        "verdict",
        "new_or_effective_band",
        "relabel_candidate",
        "relabel_applied",
        "relabel_rejected_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_guard_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "label",
        "target",
        "metric",
        "old_band",
        "new_band",
        "accepted",
        "gate_w4096_full_family_pass",
        "shortcut_negative_controls_still_fail",
        "command_demotion_negative_controls_still_fail",
        "guard_path_identity_pass",
        "reconstructed_gt_acceptance_rate",
        "decoder_path_from_gt_raw_acceptance_rate",
        "max_abs_seq_delta",
        "decision",
        "rejected_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _metric_table_lines(rows: Sequence[Mapping[str, Any]], *, only_candidates: bool = False) -> List[str]:
    lines = [
        "| target | metric | current band | continuous p50/p95/p99 | verdict basis p99 | verdict | new band |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if only_candidates and not bool(row.get("relabel_candidate")):
            continue
        lines.append(
            "| "
            f"{row.get('target')} | `{row.get('metric')}` | "
            f"{_fmt(row.get('effective_current_band'), 8)} | "
            f"{_fmt(row.get('baseline_p50'), 8)} / {_fmt(row.get('baseline_p95'), 8)} / {_fmt(row.get('baseline_p99'), 8)} | "
            f"{_fmt(row.get('verdict_pq'), 8)} | "
            f"{row.get('verdict')} | {_fmt(row.get('new_or_effective_band'), 8)} |"
        )
    return lines


def _guard_table_lines(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| relabel | old band | new band | one-window | shortcut neg | command neg | guard identity | accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('target')}:{row.get('metric')}` | "
            f"{_fmt(row.get('old_band'), 8)} | {_fmt(row.get('new_band'), 8)} | "
            f"{str(row.get('gate_w4096_full_family_pass')).lower()} | "
            f"{str(row.get('shortcut_negative_controls_still_fail')).lower()} | "
            f"{str(row.get('command_demotion_negative_controls_still_fail')).lower()} | "
            f"{str(row.get('guard_path_identity_pass')).lower()} | "
            f"{str(row.get('accepted')).lower()} |"
        )
    return lines


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# Action-Handoff Band Audit",
        "",
        "Debug-only audit. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## Verdict",
        "",
        f"- zero-slack candidates: `{len(payload['candidate_relabels'])}`",
        f"- accepted relabels: `{len(payload['accepted_relabels'])}`",
        f"- rejected relabels: `{len(payload['rejected_relabels'])}`",
        f"- rootvel_zero_slack_contract_hold_resolved: `{str(decision['rootvel_zero_slack_contract_hold_resolved']).lower()}`",
        f"- authorize_8window_after_band_audit: `{str(decision['authorize_8window_after_band_audit']).lower()}`",
        "",
        "## Step 1 - Continuous Baseline Audit",
        "",
        "Continuous rows use `Walk_F + target` samples for rate/support/pose/heading metrics, and target-only samples for the target-regime level metric. Event-aware contact/angvel verdicts use non-event samples for the zero-slack decision.",
        "",
    ]
    lines.extend(_metric_table_lines(payload["metric_rows"]))
    lines.extend(
        [
            "",
            "## Step 2 - Relabel Candidates",
            "",
        ]
    )
    if payload["candidate_relabels"]:
        lines.extend(_metric_table_lines(payload["metric_rows"], only_candidates=True))
    else:
        lines.append("No zero-slack candidate was found under the configured criterion.")
    lines.extend(
        [
            "",
            "## Step 3 - Per-Band Guard",
            "",
        ]
    )
    if payload["guard_rows"]:
        lines.extend(_guard_table_lines(payload["guard_rows"]))
    else:
        lines.append("No per-band guard was required.")
    final_guard = payload.get("final_guard", {}) or {}
    final_verdict = final_guard.get("verdict", {}) or {}
    identity = final_guard.get("guard_path_identity", {}) or {}
    lines.extend(
        [
            "",
            "## Final Combined Guard",
            "",
            f"- gate_w4096_full_family_pass: `{str(final_verdict.get('gate_w4096_full_family_pass')).lower()}`",
            f"- shortcut_negative_controls_still_fail: `{str(final_verdict.get('shortcut_negative_controls_still_fail')).lower()}`",
            f"- command_demotion_negative_controls_still_fail: `{str(final_verdict.get('command_demotion_negative_controls_still_fail')).lower()}`",
            f"- guard_path_identity_pass: `{str(final_verdict.get('guard_path_identity_pass')).lower()}`",
            f"- reconstructed_gt_acceptance_rate: `{_fmt(identity.get('reconstructed_gt_acceptance_rate'), 4)}`",
            f"- decoder_path_from_gt_raw_acceptance_rate: `{_fmt(identity.get('decoder_path_from_gt_raw_acceptance_rate'), 4)}`",
            f"- max_abs_seq_delta: `{_fmt(identity.get('max_abs_seq_delta'), 8)}`",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- per metric csv: `{payload['artifacts']['per_metric_csv']}`",
            f"- guard results csv: `{payload['artifacts']['guard_results_csv']}`",
        ]
    )
    _dump_md(path, lines)


def _write_doc_md(path: Path, payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# Band Audit: Continuous-Percentile Relabel",
        "",
        "Date: 2026-06-04",
        "",
        "Scope: debug-only band audit for action-handoff inbetween acceptance. No production Trainer/runtime/gate/checkpoint path was changed.",
        "",
        "## Conclusion",
        "",
        f"- `rootvel_zero_slack_contract_hold_resolved`: `{str(decision['rootvel_zero_slack_contract_hold_resolved']).lower()}`",
        f"- `authorize_8window_after_band_audit`: `{str(decision['authorize_8window_after_band_audit']).lower()}`",
        f"- accepted relabel count: `{len(payload['accepted_relabels'])}`",
        f"- rejected relabel count: `{len(payload['rejected_relabels'])}`",
        "",
        "## Step 1: Zero-Slack Audit",
        "",
    ]
    lines.extend(_metric_table_lines(payload["metric_rows"]))
    lines.extend(["", "## Step 2: Continuous-Percentile Relabels", ""])
    if payload["accepted_relabels"]:
        lines.extend(
            [
                "| target | metric | old band | new band | basis |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in payload["accepted_relabels"]:
            lines.append(
                f"| {row['target']} | `{row['metric']}` | {_fmt(row['old_band'], 8)} | "
                f"{_fmt(row['new_band'], 8)} | continuous p{_fmt(payload['config']['continuous_quantile'], 1)} with no tightening |"
            )
    else:
        lines.append("No band was relabeled.")
    lines.extend(["", "## Step 3: Guard Results", ""])
    if payload["guard_rows"]:
        lines.extend(_guard_table_lines(payload["guard_rows"]))
    else:
        lines.append("No per-band guard was required.")
    final_guard = payload.get("final_guard", {}) or {}
    final_verdict = final_guard.get("verdict", {}) or {}
    identity = final_guard.get("guard_path_identity", {}) or {}
    lines.extend(
        [
            "",
            "## Final State",
            "",
            f"- one-window full-family pass under accepted relabels: `{str(final_verdict.get('gate_w4096_full_family_pass')).lower()}`",
            f"- shortcut negative controls still fail: `{str(final_verdict.get('shortcut_negative_controls_still_fail')).lower()}`",
            f"- command demotion negative controls still fail: `{str(final_verdict.get('command_demotion_negative_controls_still_fail')).lower()}`",
            f"- reconstructed GT acceptance: `{_fmt(identity.get('reconstructed_gt_acceptance_rate'), 4)}`",
            f"- decoder-path-from-GT acceptance: `{_fmt(identity.get('decoder_path_from_gt_raw_acceptance_rate'), 4)}`",
            f"- `max_abs_seq_delta`: `{_fmt(identity.get('max_abs_seq_delta'), 8)}`",
            "",
            "Artifacts:",
            "",
            f"- `debug_output/_tmp_action_handoff_band_audit_20260604/summary.md`",
            f"- `debug_output/_tmp_action_handoff_band_audit_20260604/per_metric.csv`",
            f"- `debug_output/_tmp_action_handoff_band_audit_20260604/band_audit_summary.json`",
        ]
    )
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    raw_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    reconstructed_bands = _calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )

    metric_rows, candidates = _build_metric_rows(
        clips=clips,
        skeleton=skeleton,
        raw_bands=raw_bands,
        reconstructed_bands=reconstructed_bands,
        continuous_quantile=float(args.continuous_quantile),
        zero_slack_headroom_ratio=float(args.zero_slack_headroom_ratio),
        zero_slack_abs_tol=float(args.zero_slack_abs_tol),
        heading_tolerance_rad=float(args.heading_tolerance_rad),
    )

    accepted: List[BandOverride] = []
    rejected: List[Dict[str, Any]] = []
    guard_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        trial_raw, trial_reconstructed = _copy_band_tables(raw_bands, reconstructed_bands)
        _apply_overrides_to_bands(trial_raw, trial_reconstructed, [candidate])
        guard = _run_guard_with_bands(
            args=args,
            clips=clips,
            skeleton=skeleton,
            main_items=main_items,
            raw_bands=trial_raw,
            reconstructed_bands=trial_reconstructed,
            support_bands=support_bands,
            label=f"{candidate.target}:{candidate.metric}",
        )
        verdict = guard["verdict"]
        accepted_flag = bool(
            verdict.get("gate_w4096_full_family_pass")
            and verdict.get("shortcut_negative_controls_still_fail")
            and verdict.get("command_demotion_negative_controls_still_fail")
            and verdict.get("guard_path_identity_pass")
        )
        reason = "" if accepted_flag else str(verdict.get("decision"))
        if accepted_flag:
            accepted.append(candidate)
        else:
            rejected.append(
                {
                    "target": candidate.target,
                    "metric": candidate.metric,
                    "old_band": candidate.old_band,
                    "new_band": candidate.new_band,
                    "reason": reason,
                }
            )
        identity = guard.get("guard_path_identity", {}) or {}
        guard_rows.append(
            {
                "label": guard.get("label"),
                "target": candidate.target,
                "metric": candidate.metric,
                "old_band": candidate.old_band,
                "new_band": candidate.new_band,
                "accepted": bool(accepted_flag),
                "gate_w4096_full_family_pass": bool(verdict.get("gate_w4096_full_family_pass")),
                "shortcut_negative_controls_still_fail": bool(verdict.get("shortcut_negative_controls_still_fail")),
                "command_demotion_negative_controls_still_fail": bool(
                    verdict.get("command_demotion_negative_controls_still_fail")
                ),
                "guard_path_identity_pass": bool(verdict.get("guard_path_identity_pass")),
                "reconstructed_gt_acceptance_rate": identity.get("reconstructed_gt_acceptance_rate"),
                "decoder_path_from_gt_raw_acceptance_rate": identity.get("decoder_path_from_gt_raw_acceptance_rate"),
                "max_abs_seq_delta": identity.get("max_abs_seq_delta"),
                "decision": verdict.get("decision"),
                "rejected_reason": reason,
            }
        )

    accepted_raw, accepted_reconstructed = _copy_band_tables(raw_bands, reconstructed_bands)
    _apply_overrides_to_bands(accepted_raw, accepted_reconstructed, accepted)
    final_guard = _run_guard_with_bands(
        args=args,
        clips=clips,
        skeleton=skeleton,
        main_items=main_items,
        raw_bands=accepted_raw,
        reconstructed_bands=accepted_reconstructed,
        support_bands=support_bands,
        label="accepted_combined",
    )
    final_verdict = final_guard.get("verdict", {}) or {}
    for row in metric_rows:
        for override in accepted:
            if row.get("target") == override.target and row.get("metric") == override.metric:
                row["relabel_applied"] = True
        for rej in rejected:
            if row.get("target") == rej["target"] and row.get("metric") == rej["metric"]:
                row["relabel_rejected_reason"] = rej["reason"]

    rootvel_candidates = [c for c in candidates if c.metric == "rootvel_step_l2"]
    rootvel_resolved = bool(rootvel_candidates) and all(
        any(a.target == c.target and a.metric == c.metric for a in accepted) for c in rootvel_candidates
    )
    final_ok = bool(
        final_verdict.get("gate_w4096_full_family_pass")
        and final_verdict.get("shortcut_negative_controls_still_fail")
        and final_verdict.get("command_demotion_negative_controls_still_fail")
        and final_verdict.get("guard_path_identity_pass")
        and not rejected
    )
    decision = {
        "rootvel_zero_slack_contract_hold_resolved": bool(rootvel_resolved and final_ok),
        "authorize_8window_after_band_audit": bool(rootvel_resolved and final_ok),
    }

    payload: Dict[str, Any] = {
        "task": "action_handoff_band_audit",
        "scope": "debug-only; no production trainer/runtime/gate/checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "pose_sweep_pred_raw": str(args.pose_sweep_pred_raw),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "baseline_quantile": float(args.baseline_quantile),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "continuous_quantile": float(args.continuous_quantile),
            "zero_slack_headroom_ratio": float(args.zero_slack_headroom_ratio),
            "zero_slack_abs_tol": float(args.zero_slack_abs_tol),
            "event_window": int(args.event_window),
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "audited_turn_clips": list(TURN_CLIPS),
            "guard_matched_targets": list(MATCHED_TARGETS),
            "dtype": "float32 inputs / float64 metric reductions",
            "device": "cpu",
        },
        "metric_rows": metric_rows,
        "candidate_relabels": [c.__dict__ for c in candidates],
        "accepted_relabels": [c.__dict__ for c in accepted],
        "rejected_relabels": rejected,
        "guard_rows": guard_rows,
        "final_guard": final_guard,
        "decision": decision,
        "hard_constraint_confirmations": {
            "debug_only": True,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["artifacts"] = {
        "summary_json": str(args.out_dir / "band_audit_summary.json"),
        "summary_md": str(args.out_dir / "summary.md"),
        "per_metric_csv": str(args.out_dir / "per_metric.csv"),
        "guard_results_csv": str(args.out_dir / "guard_results.csv"),
        "doc_md": str(args.doc_path),
    }
    _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
    _write_per_metric_csv(Path(payload["artifacts"]["per_metric_csv"]), metric_rows)
    _write_guard_csv(Path(payload["artifacts"]["guard_results_csv"]), guard_rows)
    _write_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
    _write_doc_md(Path(payload["artifacts"]["doc_md"]), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--bone-bridge-summary", type=Path, default=DEFAULT_BONE_BRIDGE)
    p.add_argument("--regime-bridge-summary", type=Path, default=DEFAULT_REGIME_BRIDGE)
    p.add_argument("--command-demotion-rows", type=Path, default=DEFAULT_COMMAND_DEMOTION_ROWS)
    p.add_argument("--pose-sweep-pred-raw", type=Path, default=DEFAULT_FINAL_ONE_WINDOW_PRED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--baseline-quantile", type=float, default=99.5)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--continuous-quantile", type=float, default=99.0)
    p.add_argument("--bridge-budget-quantile", type=float, default=95.0)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--event-window", type=int, default=1)
    p.add_argument("--heading-tolerance-rad", type=float, default=1e-5)
    p.add_argument("--zero-slack-headroom-ratio", type=float, default=0.01)
    p.add_argument("--zero-slack-abs-tol", type=float, default=1e-6)
    p.add_argument("--oracle-contact-passthrough", action="store_true", default=False)
    p.add_argument("--command-align-root-vel", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")
    print(json.dumps(_jsonify(payload["decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
