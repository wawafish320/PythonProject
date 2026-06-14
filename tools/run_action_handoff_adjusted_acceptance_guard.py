#!/usr/bin/env python3
"""Adjusted-acceptance guard for action-handoff one-window pose sweep.

Debug-only read-only audit. It does not edit production Trainer/runtime/gate,
does not train a production model, and does not mutate checkpoints. The audit
checks whether the one-window c1 pass remains meaningful after applying the
localization-authorized adjusted acceptance view:

* event-aware contact/angvel step bands at support switches;
* heading tolerance for the known 1e-5 tail;
* original pose/regime/root/yaw/FK/support-side/endpoint checks otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    DEFAULT_BONE_BRIDGE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_REGIME_BRIDGE,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    FPS,
    POSE_DIM,
    _bridgeability_from_deltas,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _fmt,
    _foot_slip_metrics,
    _load_clips,
    _load_skeleton_meta,
    _make_hard_seam_sequence,
    _make_linear_proxy_sequence,
    _make_one_frame_switch_sequence,
    _rows_from_bone_bridge_artifact,
    _rows_from_regime_bridge_artifact,
    _safe_percentile,
    _step_angvel_component_p95,
    _step_angvel_rms,
    _step_l2,
    _step_pose_l2,
    _support_label,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _evaluate_raw_items,
    _reshape_state_aux,
    _seq_from_prediction,
)
from tools.run_action_handoff_support_contract_tightening_probe import _support_contract  # noqa: E402
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_adjusted_acceptance_guard_20260603")
DEFAULT_POSE_SWEEP_PRED = Path(
    "debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/gate_w4096_pred_raw.npz"
)
DEFAULT_COMMAND_DEMOTION_ROWS = Path(
    "debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_rows.csv"
)
EPS = 1e-8


def _jsonify(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonify(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _event_masks_from_contact(contact: np.ndarray, *, event_window: int) -> Dict[str, Any]:
    c = np.asarray(contact, dtype=np.float32).reshape(-1, 2)
    labels = [_support_label(row) for row in c]
    switch_frames: List[int] = []
    contact_bin = c > 0.5
    for t in range(1, c.shape[0]):
        if labels[t] != labels[t - 1] or bool(np.any(contact_bin[t] != contact_bin[t - 1])):
            switch_frames.append(int(t))
    frame_mask = np.zeros((c.shape[0],), dtype=bool)
    radius = max(0, int(event_window))
    for t in switch_frames:
        frame_mask[max(0, t - radius) : min(c.shape[0], t + radius + 1)] = True
    step_mask = np.zeros((max(0, c.shape[0] - 1),), dtype=bool)
    if c.shape[0] > 1:
        step_mask = frame_mask[1:] | frame_mask[:-1]
    return {
        "labels": labels,
        "switch_frames": switch_frames,
        "frame_mask": frame_mask,
        "step_mask": step_mask,
    }


def _step_event_ok(vals: np.ndarray, band: float, step_mask: np.ndarray) -> Tuple[bool, List[int], List[int]]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    mask = np.asarray(step_mask, dtype=bool).reshape(-1)
    over = [int(i + 1) for i, v in enumerate(arr) if float(v) > float(band) + EPS]
    unexcused = [int(i + 1) for i, v in enumerate(arr) if float(v) > float(band) + EPS and not bool(mask[i])]
    return (not unexcused), over, unexcused


def _adjusted_eval_sequence(
    seq: Mapping[str, np.ndarray],
    *,
    target: str,
    target_bands: Mapping[str, Any],
    skeleton: Any,
    case: str,
    expected_label: str,
    start_phase: str,
    endpoint_bridgeability: bool,
    event_window: int,
    heading_tolerance_rad: float,
    support_side_correctness: Optional[bool] = None,
    endpoint_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_DIM)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)
    event = _event_masks_from_contact(contact, event_window=event_window)

    foot = _foot_slip_metrics(rot6d, root_pos, contact, skeleton)
    level_center = np.asarray(target_bands["bone_angvel_level_center"], dtype=np.float32).reshape(1, -1)
    level_rms = float(np.sqrt(np.mean((bone_angvel[-1:].reshape(1, -1) - level_center) ** 2)))
    pose_step = _step_pose_l2(rot6d)
    angvel_step = _step_angvel_rms(bone_angvel)
    angvel_component = _step_angvel_component_p95(bone_angvel)
    rootvel_step = _step_l2(root_vel)
    yaw_step = np.abs(np.diff(yaw_rate))
    contact_step = _step_l2(contact)
    heading = _heading_error_rad(root_vel, cond_dir)
    foot_p95 = float(foot.get("contacted_speed_p95_mps", 0.0) or 0.0)

    pose_p95 = _safe_percentile(pose_step, 95.0)
    angvel_p95 = _safe_percentile(angvel_step, 95.0)
    angvel_component_p95 = _safe_percentile(angvel_component, 95.0)
    rootvel_p95 = _safe_percentile(rootvel_step, 95.0)
    yaw_p95 = _safe_percentile(yaw_step, 95.0)
    contact_p95 = _safe_percentile(contact_step, 95.0)
    heading_p95 = _safe_percentile(heading, 95.0)

    angvel_event_ok, angvel_over, angvel_unexcused = _step_event_ok(
        angvel_step,
        float(target_bands["angvel_step_rms"]),
        event["step_mask"],
    )
    contact_event_ok, contact_over, contact_unexcused = _step_event_ok(
        contact_step,
        float(target_bands["contact_step_l2"]),
        event["step_mask"],
    )
    regime_reached = level_rms <= float(target_bands["bone_angvel_level_rms"]) + EPS
    rate_budget = (
        bool(angvel_event_ok)
        and angvel_component_p95 <= float(target_bands["angvel_step_component_p95"]) + EPS
        and rootvel_p95 <= float(target_bands["rootvel_step_l2"]) + EPS
        and yaw_p95 <= float(target_bands["yaw_rate_step_abs"]) + EPS
    )
    support_honesty = bool(contact_event_ok) and foot_p95 <= float(target_bands["foot_slip_contacted_speed_mps"]) + EPS
    command_response = heading_p95 <= max(float(target_bands["heading_error_rad"]), float(heading_tolerance_rad)) + EPS
    pose_continuity = pose_p95 <= float(target_bands["pose_step_l2"]) + EPS

    families = {
        "regime_reached": bool(regime_reached),
        "rate_budget": bool(rate_budget),
        "support_honesty": bool(support_honesty),
        "command_response": bool(command_response),
        "pose_continuity": bool(pose_continuity),
        "endpoint_bridgeability": bool(endpoint_bridgeability),
    }
    if support_side_correctness is not None:
        families["support_side_correctness"] = bool(support_side_correctness)
    order = [
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
    ]
    failed = [k for k in order if k in families and not bool(families[k])]
    return {
        "case": case,
        "target": target,
        "start_phase": start_phase,
        "expected_label": expected_label,
        "adjusted_pass": bool(not failed),
        "adjusted_failed_family": ",".join(failed),
        **families,
        "event_switch_frames": event["switch_frames"],
        "event_boundary_frames": [int(i) for i, v in enumerate(event["frame_mask"]) if bool(v)],
        "angvel_over_frames": angvel_over,
        "angvel_unexcused_over_frames": angvel_unexcused,
        "contact_over_frames": contact_over,
        "contact_unexcused_over_frames": contact_unexcused,
        "metrics": {
            "pose_step_l2_p95": pose_p95,
            "angvel_step_rms_p95": angvel_p95,
            "angvel_component_p95_p95": angvel_component_p95,
            "rootvel_step_l2_p95": rootvel_p95,
            "yaw_rate_step_abs_p95": yaw_p95,
            "contact_step_l2_p95": contact_p95,
            "heading_error_p95_rad": heading_p95,
            "foot_slip_p95_mps": foot_p95,
            "bone_angvel_level_rms_to_target": level_rms,
        },
        "thresholds": {k: v for k, v in target_bands.items() if k != "bone_angvel_level_center"},
        "endpoint_details": dict(endpoint_details or {}),
    }


def _heading_error_rad(root_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    rv = np.asarray(root_vel, dtype=np.float64).reshape(-1, 2)
    cd = np.asarray(cond_dir, dtype=np.float64).reshape(-1, 2)
    n = min(rv.shape[0], cd.shape[0])
    rv = rv[:n]
    cd = cd[:n]
    speed = np.linalg.norm(rv, axis=1)
    cmd_norm = np.linalg.norm(cd, axis=1)
    valid = (speed > 1e-4) & (cmd_norm > EPS)
    err = np.zeros(n, dtype=np.float64)
    if np.any(valid):
        dot = np.sum(rv[valid] * cd[valid], axis=1) / np.maximum(speed[valid] * cmd_norm[valid], EPS)
        err[valid] = np.arccos(np.clip(dot, -1.0, 1.0))
    return err


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failed = Counter()
    for row in rows:
        for fam in str(row.get("adjusted_failed_family") or "").split(","):
            if fam:
                failed[fam] += 1
    return {
        "n": int(len(rows)),
        "adjusted_pass_rate": float(np.mean([bool(r.get("adjusted_pass", False)) for r in rows])) if rows else 0.0,
        "adjusted_pass_count": int(sum(bool(r.get("adjusted_pass", False)) for r in rows)),
        "failed_family_counts": dict(failed),
    }


def _summaries_by_case(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for case in sorted(set(str(r.get("case")) for r in rows)):
        out[case] = _summarize([r for r in rows if str(r.get("case")) == case])
    return out


def _bridge_budgets(clips: Mapping[str, Any], quantile: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, clip in clips.items():
        out[name] = {
            "angvel": _safe_percentile(_step_angvel_rms(clip.bone_angvel), float(quantile)),
            "rootvel": _safe_percentile(_step_l2(clip.root_vel), float(quantile)),
            "yaw": _safe_percentile(np.abs(np.diff(clip.yaw_rate)), float(quantile)),
            "contact": _safe_percentile(_step_l2(clip.contact), float(quantile)),
        }
    return out


def _artifact_negative_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _rows_from_bone_bridge_artifact(Path(args.bone_bridge_summary)) + _rows_from_regime_bridge_artifact(
        Path(args.regime_bridge_summary)
    ):
        rec = dict(row)
        # Artifact rows do not carry full trajectories. They remain valid
        # negative controls if any non-command family still fails after heading
        # demotion/tolerance.
        failed = [
            fam
            for fam in str(rec.get("failed_family") or "").split(",")
            if fam and fam not in {"command_response"}
        ]
        rec["adjusted_pass"] = bool(not failed)
        rec["adjusted_failed_family"] = ",".join(failed)
        rec["case"] = str(rec.get("case", "")).replace("proxy_replay:", "artifact_proxy:")
        rows.append(rec)
    return rows


def _command_demotion_negative_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("case", "")).startswith("negative_control:"):
                rows.append(row)
    by_case: Dict[str, Any] = {}
    for case in sorted(set(r.get("case", "") for r in rows)):
        subset = [r for r in rows if r.get("case") == case]
        pass_count = sum(str(r.get("demoted_acceptance_pass", "")).lower() == "true" for r in subset)
        by_case[case] = {
            "n": int(len(subset)),
            "demoted_pass_count": int(pass_count),
            "demoted_pass_rate": float(pass_count / max(1, len(subset))),
        }
    return {
        "available": True,
        "rows_csv": str(path),
        "n": int(len(rows)),
        "demoted_negative_pass_count": int(
            sum(str(r.get("demoted_acceptance_pass", "")).lower() == "true" for r in rows)
        ),
        "by_case": by_case,
    }


def _load_gate_pred_row(
    *,
    args: argparse.Namespace,
    main_items: Sequence[Any],
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = np.load(args.pose_sweep_pred_raw, allow_pickle=True)
    pred_raw = np.asarray(data["pred_raw"], dtype=np.float32)
    idxs = [int(x) for x in np.asarray(data["train_indices"]).reshape(-1).tolist()]
    if len(idxs) != 1:
        raise RuntimeError(f"expected one train index in {args.pose_sweep_pred_raw}, got {idxs}")
    item = main_items[idxs[0]]
    base_rows = _evaluate_raw_items(
        variant="gate_w4096",
        split="adjusted_acceptance_guard",
        split_kind="pose_c1c2_sweep_saved_pred",
        partition="train",
        items=main_items,
        idxs=idxs,
        raw=pred_raw,
        horizon=int(args.horizon),
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        calibration_domain="reconstructed_state281",
    )
    if not base_rows:
        raise RuntimeError("saved gate prediction did not evaluate")
    state, aux = _reshape_state_aux(pred_raw, int(args.horizon))
    seq = _seq_from_prediction(
        item,
        state[0],
        aux[0],
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    adjusted = _adjusted_eval_sequence(
        seq,
        target=item.clip,
        target_bands=baseline_bands[item.clip],
        skeleton=skeleton,
        case="positive_saved_pred:gate_w4096",
        expected_label="pass",
        start_phase=f"{item.clip}:{item.start}-{item.end}",
        endpoint_bridgeability=bool(base_rows[0].get("endpoint_bridgeability", False)),
        event_window=int(args.event_window),
        heading_tolerance_rad=float(args.heading_tolerance_rad),
        support_side_correctness=bool(base_rows[0].get("support_side_correctness", False)),
        endpoint_details={
            "source": str(args.pose_sweep_pred_raw),
            "base_failed_family": base_rows[0].get("failed_family", ""),
            "base_acceptance_proxy_pass": bool(base_rows[0].get("acceptance_proxy_pass", False)),
        },
    )
    return adjusted, base_rows[0]


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case",
        "target",
        "start_phase",
        "expected_label",
        "adjusted_pass",
        "adjusted_failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_correctness",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
        "event_switch_frames",
        "angvel_over_frames",
        "angvel_unexcused_over_frames",
        "contact_over_frames",
        "contact_unexcused_over_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    lines = [
        "# Adjusted Acceptance Guard",
        "",
        "Debug-only read-only guard. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## Verdict",
        "",
        f"- adjusted gate_w4096 full-family pass: `{str(verdict['gate_w4096_full_family_pass']).lower()}`",
        f"- shortcut negative controls still fail: `{str(verdict['shortcut_negative_controls_still_fail']).lower()}`",
        f"- command demotion negative controls still fail: `{str(verdict['command_demotion_negative_controls_still_fail']).lower()}`",
        f"- decision: `{verdict['decision']}`",
        "",
        "## Gate W4096",
        "",
        "| metric | value | band/tolerance |",
        "|---|---:|---:|",
    ]
    gate = payload["gate_w4096_adjusted"]
    metrics = gate.get("metrics", {}) or {}
    thresholds = gate.get("thresholds", {}) or {}
    lines.extend(
        [
            f"| pose_step_l2_p95 | {_fmt(metrics.get('pose_step_l2_p95'), 8)} | {_fmt(thresholds.get('pose_step_l2'), 8)} |",
            f"| angvel_step_rms_p95 | {_fmt(metrics.get('angvel_step_rms_p95'), 8)} | event-aware `{_fmt(thresholds.get('angvel_step_rms'), 8)}` |",
            f"| angvel_component_p95_p95 | {_fmt(metrics.get('angvel_component_p95_p95'), 8)} | {_fmt(thresholds.get('angvel_step_component_p95'), 8)} |",
            f"| rootvel_step_l2_p95 | {_fmt(metrics.get('rootvel_step_l2_p95'), 8)} | {_fmt(thresholds.get('rootvel_step_l2'), 8)} |",
            f"| yaw_rate_step_abs_p95 | {_fmt(metrics.get('yaw_rate_step_abs_p95'), 8)} | {_fmt(thresholds.get('yaw_rate_step_abs'), 8)} |",
            f"| contact_step_l2_p95 | {_fmt(metrics.get('contact_step_l2_p95'), 8)} | event-aware `{_fmt(thresholds.get('contact_step_l2'), 8)}` |",
            f"| heading_error_p95_rad | {_fmt(metrics.get('heading_error_p95_rad'), 8)} | {_fmt(payload['config']['heading_tolerance_rad'], 8)} |",
            f"| foot_slip_p95_mps | {_fmt(metrics.get('foot_slip_p95_mps'), 8)} | {_fmt(thresholds.get('foot_slip_contacted_speed_mps'), 8)} |",
        ]
    )
    lines.extend(
        [
            "",
            f"- failed family: `{gate.get('adjusted_failed_family', '')}`",
            f"- event switch frames: `{gate.get('event_switch_frames')}`",
            f"- original reconstructed row failed family: `{payload['gate_w4096_original_row'].get('failed_family', '')}`",
            f"- support-side failure count: `{payload['gate_w4096_original_row'].get('support_side_failure_count')}`",
            f"- support-side failure sample: `{payload['gate_w4096_original_row'].get('support_side_failures', [])[:3]}`",
            "",
            "## Shortcut Negative Controls",
            "",
            "| case | n | adjusted pass rate | failed families |",
            "|---|---:|---:|---|",
        ]
    )
    for case, rec in payload["shortcut_negative_summary_by_case"].items():
        lines.append(
            f"| {case} | {rec['n']} | {_fmt(rec['adjusted_pass_rate'])} | {rec.get('failed_family_counts', {})} |"
        )
    lines.extend(
        [
            "",
            "## Command Demotion Negative Controls",
            "",
            f"- source: `{payload['command_demotion_negative_summary'].get('rows_csv')}`",
            f"- pass count: `{payload['command_demotion_negative_summary'].get('demoted_negative_pass_count')}`",
            "",
            "## Artifacts",
            "",
            f"- summary json: `{payload['artifacts']['summary_json']}`",
            f"- rows csv: `{payload['artifacts']['rows_csv']}`",
        ]
    )
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    raw_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    budgets = _bridge_budgets(clips, float(args.bridge_budget_quantile))

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
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )

    gate_adjusted, gate_original = _load_gate_pred_row(
        args=args,
        main_items=main_items,
        baseline_bands=reconstructed_bands,
        support_bands=reconstructed_support_bands,
        skeleton=skeleton,
    )

    rows: List[Dict[str, Any]] = [gate_adjusted]
    shortcut_rows: List[Dict[str, Any]] = []
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}
    walk = clips[WALK_F]
    for target in TURN_CLIPS:
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
    rows.extend(shortcut_rows)
    shortcut_summary = _summaries_by_case(shortcut_rows)
    command_summary = _command_demotion_negative_summary(Path(args.command_demotion_rows))
    shortcut_fail = all(float(rec.get("adjusted_pass_rate", 0.0) or 0.0) == 0.0 for rec in shortcut_summary.values())
    command_fail = bool(command_summary.get("available")) and int(command_summary.get("demoted_negative_pass_count", 1)) == 0
    gate_pass = bool(gate_adjusted.get("adjusted_pass", False))
    decision = (
        "adjusted_acceptance_guard_passed_ready_for_8window_debug_sweep"
        if gate_pass and shortcut_fail and command_fail
        else "adjusted_acceptance_guard_failed_do_not_run_8window"
    )
    payload = {
        "task": "adjusted_acceptance_guard",
        "scope": "debug-only read-only adjusted acceptance audit; no production gate/runtime/checkpoint mutation",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "pose_sweep_pred_raw": str(args.pose_sweep_pred_raw),
            "horizon": int(args.horizon),
            "event_window": int(args.event_window),
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "baseline_quantile": float(args.baseline_quantile),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "dtype": "float32",
            "device": "cpu",
        },
        "gate_w4096_adjusted": gate_adjusted,
        "gate_w4096_original_row": gate_original,
        "shortcut_negative_summary_by_case": shortcut_summary,
        "command_demotion_negative_summary": command_summary,
        "verdict": {
            "gate_w4096_full_family_pass": gate_pass,
            "shortcut_negative_controls_still_fail": bool(shortcut_fail),
            "command_demotion_negative_controls_still_fail": bool(command_fail),
            "decision": decision,
        },
        "hard_constraint_confirmations": {
            "debug_only": True,
            "trained_production_trainer": False,
            "forwarded_production_runtime_or_gate": False,
            "modified_checkpoint": False,
            "modified_production_runtime_trainer_gate": False,
        },
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["artifacts"] = {
        "summary_json": str(args.out_dir / "adjusted_acceptance_guard_summary.json"),
        "summary_md": str(args.out_dir / "summary.md"),
        "rows_csv": str(args.out_dir / "rows.csv"),
    }
    _dump_json(Path(payload["artifacts"]["summary_json"]), _jsonify(payload))
    _write_rows_csv(Path(payload["artifacts"]["rows_csv"]), rows)
    _write_summary_md(Path(payload["artifacts"]["summary_md"]), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--bone-bridge-summary", type=Path, default=DEFAULT_BONE_BRIDGE)
    p.add_argument("--regime-bridge-summary", type=Path, default=DEFAULT_REGIME_BRIDGE)
    p.add_argument("--command-demotion-rows", type=Path, default=DEFAULT_COMMAND_DEMOTION_ROWS)
    p.add_argument("--pose-sweep-pred-raw", type=Path, default=DEFAULT_POSE_SWEEP_PRED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--baseline-quantile", type=float, default=99.5)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--bridge-budget-quantile", type=float, default=95.0)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--event-window", type=int, default=1)
    p.add_argument("--heading-tolerance-rad", type=float, default=1e-4)
    p.add_argument("--oracle-contact-passthrough", action="store_true", default=False)
    p.add_argument("--command-align-root-vel", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")
    print(json.dumps(_jsonify(payload["verdict"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
