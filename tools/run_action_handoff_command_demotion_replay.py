#!/usr/bin/env python3
"""Replay command-response demotion with anti-gaming negative controls.

Debug-only read-only probe. This does not train a model, does not forward or edit
production Trainer/runtime/gate, and does not mutate checkpoints. It replays the
middle acceptance cases with per-frame command tracking demoted to diagnostic
and a net/integral command-compatibility check added as the hard anti-shortcut
command family.
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
    CONTEXT_LEN_C,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_TOPK,
    TURN_CLIPS,
    WALK_F,
    full_state_align,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    DEFAULT_BONE_BRIDGE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_REGIME_BRIDGE,
    DEFAULT_TWO_FRAME,
    DEFAULT_Z_FEATURES,
    FPS,
    _bridgeability_from_deltas,
    _calibrate_baselines,
    _dump_json,
    _dump_md,
    _evaluate_sequence,
    _fmt,
    _foot_positions,
    _load_clips,
    _load_skeleton_meta,
    _make_hard_seam_sequence,
    _make_linear_proxy_sequence,
    _make_one_frame_switch_sequence,
    _make_sequence,
    _rows_from_bone_bridge_artifact,
    _rows_from_regime_bridge_artifact,
    _safe_percentile,
    _step_angvel_rms,
    _step_l2,
    _support_label,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _evaluate_seq_common,
    _reconstructed_gt_seq,
)
from tools.run_action_handoff_signal_representation_audit import (  # noqa: E402
    NOISE_LEVELS,
    _anchored_state,
    _flat_state,
    _rng_for,
    _root_position_state,
    _state_and_seq_from_state,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    SUPPORT_SIDE_FEATURE_KEYS,
    _evaluate_support_side_correctness,
    _support_contract,
    _support_side_features,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
)


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_command_demotion_replay_20260603")
EPS = 1e-8
COMMANDISH_SUPPORT_KEYS = {
    "yaw_sum_rad",
    "yaw_abs_sum_rad",
    "heading_error_p95_rad",
    "root_speed_mean",
    "root_lateral_mean",
    "support_yaw_product",
    "support_lateral_product",
}
SUPPORT_SIDE_CORE_KEYS = tuple(k for k in SUPPORT_SIDE_FEATURE_KEYS if k not in COMMANDISH_SUPPORT_KEYS)


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


def _angle_path_delta(cond_dir: np.ndarray) -> float:
    cond = np.asarray(cond_dir, dtype=np.float64).reshape(-1, 2)
    if cond.shape[0] < 2:
        return 0.0
    ang = np.arctan2(cond[:, 1], cond[:, 0])
    diff = (np.diff(ang) + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.sum(diff))


def _net_yaw_from_seq(seq: Mapping[str, np.ndarray]) -> float:
    yaw = np.asarray(seq["yaw_rate"], dtype=np.float64).reshape(-1)
    return float(np.sum(yaw) / float(FPS)) if yaw.size else 0.0


def _root_command_alignment(seq: Mapping[str, np.ndarray]) -> Tuple[float, float]:
    root = np.asarray(seq["root_pos"], dtype=np.float64).reshape(-1, 3)
    cond = np.asarray(seq["cond_dir"], dtype=np.float64).reshape(-1, 2)
    if root.shape[0] < 2 or cond.shape[0] == 0:
        return 0.0, 1.0
    disp = root[-1, :2] - root[0, :2]
    mean_cmd = np.mean(cond, axis=0)
    n = float(np.linalg.norm(mean_cmd))
    if n <= EPS:
        return float(np.linalg.norm(disp)), 1.0
    unit = mean_cmd / n
    align = float(np.dot(disp, unit))
    ratio = float(align / max(float(np.linalg.norm(disp)), EPS))
    return align, ratio


def _calibrate_command_bands(
    clips: Mapping[str, Any],
    *,
    horizon: int,
    quantile: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, clip in clips.items():
        diffs: List[float] = []
        align_ratios: List[float] = []
        max_start = int(clip.rot6d.shape[0]) - int(horizon)
        if max_start < 0:
            continue
        for start in range(max_start + 1):
            seq = _make_sequence(clip, start, horizon)
            net_yaw = _net_yaw_from_seq(seq)
            net_cmd = _angle_path_delta(seq["cond_dir"])
            _align, ratio = _root_command_alignment(seq)
            diffs.append(abs(net_yaw - net_cmd))
            align_ratios.append(ratio)
        out[name] = {
            "n_windows": int(len(diffs)),
            "quantile": float(quantile),
            "net_yaw_vs_cond_abs_diff_rad": _safe_percentile(np.asarray(diffs, dtype=np.float64), quantile),
            "root_command_alignment_ratio_min": float(np.min(align_ratios)) if align_ratios else -1.0,
            "definition": "abs(sum(yaw_rate)/FPS - wrapped cond_dir net angle) <= continuous-window quantile; root displacement must not be counter-command",
        }
    return out


def _command_compatibility(
    seq: Mapping[str, np.ndarray],
    band: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    net_yaw = _net_yaw_from_seq(seq)
    net_cmd = _angle_path_delta(seq["cond_dir"])
    align, ratio = _root_command_alignment(seq)
    diff = abs(net_yaw - net_cmd)
    thr = float((band or {}).get("net_yaw_vs_cond_abs_diff_rad", 0.0) or 0.0)
    # Root direction is intentionally weak: it catches counter-command travel but
    # does not require per-frame controller-style heading.
    root_not_counter = bool(align >= -1e-4)
    net_integral_ok = bool(diff <= thr + 1e-6)
    return {
        "command_compatibility": bool(net_integral_ok and root_not_counter),
        "net_integral_ok": bool(net_integral_ok),
        "root_not_counter_command": bool(root_not_counter),
        "net_yaw_rad": float(net_yaw),
        "net_cond_dir_rad": float(net_cmd),
        "net_yaw_cond_abs_diff_rad": float(diff),
        "net_yaw_cond_abs_diff_threshold_rad": float(thr),
        "root_command_alignment_m": float(align),
        "root_command_alignment_ratio": float(ratio),
    }


def _subset_band_check(
    features: Mapping[str, float],
    bands: Mapping[str, Any],
    keys: Sequence[str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    for key in keys:
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


def _support_side_core(
    seq: Mapping[str, np.ndarray],
    labels: Sequence[str],
    skeleton: Any,
    bands: Mapping[str, Any],
) -> Dict[str, Any]:
    foot = _foot_positions(seq["rot6d"], seq["root_pos"], skeleton)
    features = _support_side_features(seq, labels, foot)
    full_ok, full_failures = _evaluate_support_side_correctness(features, bands)
    core_ok, core_failures = _subset_band_check(features, bands, SUPPORT_SIDE_CORE_KEYS)
    return {
        "support_side_core": bool(core_ok),
        "support_side_full_legacy": bool(full_ok),
        "support_side_core_failure_count": int(len(core_failures)),
        "support_side_full_failure_count": int(len(full_failures)),
        "support_side_core_failures": core_failures[:8],
        "support_side_full_failures": full_failures[:8],
    }


def _failed_demoted(row: Mapping[str, Any], *, include_support_side_core: bool) -> List[str]:
    keys = [
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "command_compatibility",
        "pose_continuity",
        "endpoint_bridgeability",
    ]
    if include_support_side_core:
        keys.insert(3, "support_side_core")
    return [k for k in keys if not bool(row.get(k, False))]


def _attach_demoted_acceptance(
    row: Dict[str, Any],
    *,
    include_support_side_core: bool,
) -> Dict[str, Any]:
    failed = _failed_demoted(row, include_support_side_core=include_support_side_core)
    row["demoted_acceptance_pass"] = bool(not failed)
    row["demoted_failed_family"] = ",".join(failed)
    row["legacy_command_response_diagnostic"] = bool(row.get("command_response", False))
    return row


def _evaluate_seq_demoted(
    *,
    case: str,
    target: str,
    expected_label: str,
    start_phase: str,
    seq: Mapping[str, np.ndarray],
    target_bands: Mapping[str, Any],
    command_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    endpoint_bridgeability: bool,
    endpoint_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = _evaluate_sequence(
        seq,
        target=target,
        target_bands=target_bands,
        skeleton=skeleton,
        case=case,
        expected_label=expected_label,
        start_phase=start_phase,
        endpoint_bridgeability=endpoint_bridgeability,
        endpoint_details=endpoint_details,
    )
    cmd = _command_compatibility(seq, command_bands.get(target))
    row.update(cmd)
    row["legacy_acceptance_pass"] = bool(row.get("pass", False))
    row["legacy_failed_family"] = row.get("failed_family", "")
    return _attach_demoted_acceptance(row, include_support_side_core=False)


def _evaluate_signal_seq_demoted(
    *,
    representation: str,
    section: str,
    noise_mse: Optional[float],
    item: Any,
    state: np.ndarray,
    aux: np.ndarray,
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    command_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
) -> Dict[str, Any]:
    state, seq = _state_and_seq_from_state(item, state, aux)
    row = _evaluate_seq_common(
        variant=representation,
        split="command_demotion_replay",
        split_kind="gt_read_only",
        partition=section,
        item=item,
        seq=seq,
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=min_run_frames,
        endpoint_note="reconstructed-domain acceptance path; command_response demoted to diagnostic",
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
        calibration_domain="reconstructed_state281",
    )
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    contract = _support_contract(seq["contact"], min_run_frames=min_run_frames)
    labels = [str(x) for x in contract["normalized_label_sequence"]] or labels
    side = _support_side_core(seq, labels, skeleton, support_bands[item.clip]["feature_bands"])
    cmd = _command_compatibility(seq, command_bands.get(item.clip))
    root_err = np.linalg.norm(
        np.asarray(seq["root_pos"], dtype=np.float64) - np.asarray(baseline_seq["root_pos"], dtype=np.float64),
        axis=1,
    )
    row.update(side)
    row.update(cmd)
    row["case"] = f"signal_reconstruct:{representation}" if section == "reconstructability" else f"signal_perturb:{representation}"
    row["target"] = item.clip
    row["expected_label"] = "pass" if representation != "support_anchor_drop_inter_anchor" else "fail"
    row["start_phase"] = f"{item.clip}[{item.start}:{item.end}]"
    row["noise_mse"] = noise_mse
    row["legacy_acceptance_pass"] = bool(row.get("acceptance_proxy_pass", False))
    row["legacy_failed_family"] = row.get("failed_family", "")
    row["max_abs_state_delta"] = float(
        np.max(np.abs(np.asarray(state, dtype=np.float64) - np.asarray(true_state, dtype=np.float64)))
    )
    row["root_path_error_p95_m"] = _safe_percentile(root_err, 95.0)
    row["root_path_error_max_m"] = float(np.max(root_err)) if root_err.size else 0.0
    row["heading_error_p95_rad"] = float(row.get("metrics", {}).get("heading_error_p95_rad", 0.0) or 0.0)
    row["foot_slip_p95_to_band_ratio"] = float(row.get("foot_slip_p95_to_band_ratio", 0.0) or 0.0)
    return _attach_demoted_acceptance(row, include_support_side_core=True)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    failed = Counter()
    for row in rows:
        for fam in str(row.get("demoted_failed_family") or "").split(","):
            if fam:
                failed[fam] += 1

    def rate(key: str) -> float:
        return float(np.mean([bool(r.get(key, False)) for r in rows]))

    def mean(key: str) -> float:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n": int(len(rows)),
        "legacy_pass_rate": rate("legacy_acceptance_pass"),
        "demoted_pass_rate": rate("demoted_acceptance_pass"),
        "command_compatibility_pass_rate": rate("command_compatibility"),
        "legacy_command_response_pass_rate": rate("legacy_command_response_diagnostic"),
        "support_side_core_pass_rate": rate("support_side_core") if any("support_side_core" in r for r in rows) else None,
        "support_side_full_legacy_pass_rate": rate("support_side_full_legacy")
        if any("support_side_full_legacy" in r for r in rows)
        else None,
        "support_honesty_pass_rate": rate("support_honesty"),
        "rate_budget_pass_rate": rate("rate_budget"),
        "pose_continuity_pass_rate": rate("pose_continuity"),
        "endpoint_bridgeability_pass_rate": rate("endpoint_bridgeability"),
        "net_yaw_cond_abs_diff_rad_mean": mean("net_yaw_cond_abs_diff_rad"),
        "heading_error_p95_rad_mean": mean("heading_error_p95_rad"),
        "foot_slip_p95_to_band_ratio_mean": mean("foot_slip_p95_to_band_ratio"),
        "root_path_error_p95_m_mean": mean("root_path_error_p95_m"),
        "failed_family_counts": dict(failed),
    }


def _summaries_by(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for value in sorted(set(str(r.get(key)) for r in rows)):
        out[value] = _summarize([r for r in rows if str(r.get(key)) == value])
    return out


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "target",
        "start_phase",
        "expected_label",
        "noise_mse",
        "legacy_acceptance_pass",
        "legacy_failed_family",
        "demoted_acceptance_pass",
        "demoted_failed_family",
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "support_side_core",
        "support_side_full_legacy",
        "command_compatibility",
        "legacy_command_response_diagnostic",
        "pose_continuity",
        "endpoint_bridgeability",
        "net_integral_ok",
        "root_not_counter_command",
        "net_yaw_rad",
        "net_cond_dir_rad",
        "net_yaw_cond_abs_diff_rad",
        "net_yaw_cond_abs_diff_threshold_rad",
        "root_command_alignment_m",
        "root_command_alignment_ratio",
        "heading_error_p95_rad",
        "foot_slip_p95_to_band_ratio",
        "root_path_error_p95_m",
        "max_abs_state_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Command Demotion Replay")
    lines.append("")
    lines.append("Date: 2026-06-03")
    lines.append("")
    lines.append(
        "Debug-only read-only replay. Per-frame `command_response` is demoted to diagnostic; "
        "hard command check is replaced by net/integral `command_compatibility`."
    )
    lines.append("")
    lines.append("## Guard Verdict")
    verdict = payload["verdict"]
    lines.append(f"- negative controls still fail: `{str(verdict['negative_controls_still_fail']).lower()}`")
    lines.append(f"- linear proxy demoted pass rate: `{_fmt(verdict['linear_proxy_demoted_pass_rate'])}`")
    lines.append(f"- anchored keep reconstruct demoted pass rate: `{_fmt(verdict['anchored_keep_reconstruct_demoted_pass_rate'])}`")
    lines.append(f"- support-side core vs full legacy split exposed: `{str(verdict['support_side_core_split_exposed']).lower()}`")
    lines.append(f"- decision: `{verdict['decision']}`")
    lines.append("")
    lines.append("## Negative Controls")
    lines.append("")
    lines.append("| case | n | legacy pass | demoted pass | command compat | failed families |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for case, rec in sorted(payload["negative_control_summary_by_case"].items()):
        lines.append(
            f"| {case} | {int(rec.get('n', 0))} | {_fmt(rec.get('legacy_pass_rate'))} | "
            f"{_fmt(rec.get('demoted_pass_rate'))} | {_fmt(rec.get('command_compatibility_pass_rate'))} | "
            f"{rec.get('failed_family_counts', {})} |"
        )
    lines.append("")
    lines.append("## Signal Reconstructability")
    lines.append("")
    lines.append(
        "| representation | n | legacy pass | demoted pass | command compat | support core | support full legacy | support honest | rate | heading err mean | foot ratio | root p95 err |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case, rec in sorted(payload["signal_reconstruct_summary_by_case"].items()):
        rep = case.replace("signal_reconstruct:", "")
        lines.append(
            f"| {rep} | {int(rec.get('n', 0))} | {_fmt(rec.get('legacy_pass_rate'))} | "
            f"{_fmt(rec.get('demoted_pass_rate'))} | {_fmt(rec.get('command_compatibility_pass_rate'))} | "
            f"{_fmt(rec.get('support_side_core_pass_rate'))} | {_fmt(rec.get('support_side_full_legacy_pass_rate'))} | "
            f"{_fmt(rec.get('support_honesty_pass_rate'))} | {_fmt(rec.get('rate_budget_pass_rate'))} | "
            f"{_fmt(rec.get('heading_error_p95_rad_mean'), 4)} | {_fmt(rec.get('foot_slip_p95_to_band_ratio_mean'), 4)} | "
            f"{_fmt(rec.get('root_path_error_p95_m_mean'), 6)} |"
        )
    lines.append("")
    lines.append("## Perturbation")
    lines.append("")
    lines.append(
        "| representation/noise | n | demoted pass | command compat | support core | support honest | rate | heading err mean | foot ratio | root p95 err |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, rec in sorted(payload["signal_perturb_summary_by_case_noise"].items()):
        lines.append(
            f"| {key} | {int(rec.get('n', 0))} | {_fmt(rec.get('demoted_pass_rate'))} | "
            f"{_fmt(rec.get('command_compatibility_pass_rate'))} | {_fmt(rec.get('support_side_core_pass_rate'))} | "
            f"{_fmt(rec.get('support_honesty_pass_rate'))} | {_fmt(rec.get('rate_budget_pass_rate'))} | "
            f"{_fmt(rec.get('heading_error_p95_rad_mean'), 4)} | {_fmt(rec.get('foot_slip_p95_to_band_ratio_mean'), 4)} | "
            f"{_fmt(rec.get('root_path_error_p95_m_mean'), 6)} |"
        )
    lines.append("")
    lines.append("## Perturbation Noise Caveat")
    lines.append("")
    lines.append(
        "The perturbation rows use per-frame independent Gaussian noise. This is a high-frequency "
        "noise model: flat velocity-to-position reconstruction integrates it, while lifted "
        "position-to-velocity reconstruction finite-differences it. These numbers diagnose "
        "high-frequency sensitivity and are not a fair anchored-vs-flat conditioning verdict."
    )
    lines.append("")
    lines.append(
        "A fair conditioning gate must use native-space correlated/bias noise, calibrate amplitudes "
        "to equal reconstructed-state281 MSE, and report both position-side and velocity-side metrics."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- summary json: `{payload['artifacts']['summary_json']}`")
    lines.append(f"- rows csv: `{payload['artifacts']['rows_csv']}`")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    raw_bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))
    command_bands = _calibrate_command_bands(clips, horizon=int(args.horizon), quantile=float(args.command_quantile))

    h = int(args.horizon)
    rows: List[Dict[str, Any]] = []

    # Positive continuous rows for calibration sanity.
    for clip_name, clip in clips.items():
        max_start = int(clip.rot6d.shape[0]) - h
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, int(args.stride))):
            seq = _make_sequence(clip, start, h)
            rows.append(
                _evaluate_seq_demoted(
                    case="positive_oracle:real_continuous",
                    target=clip_name,
                    expected_label="pass",
                    start_phase=f"{clip_name}[{start}:{start + h}]",
                    seq=seq,
                    target_bands=raw_bands[clip_name],
                    command_bands=command_bands,
                    skeleton=skeleton,
                    endpoint_bridgeability=True,
                    endpoint_details={"source": "raw continuous clip"},
                )
            )

    bridge_budgets: Dict[str, Dict[str, float]] = {}
    for name, clip in clips.items():
        bridge_budgets[name] = {
            "angvel": _safe_percentile(_step_angvel_rms(clip.bone_angvel), float(args.bridge_budget_quantile)),
            "rootvel": _safe_percentile(_step_l2(clip.root_vel), float(args.bridge_budget_quantile)),
            "yaw": _safe_percentile(np.abs(np.diff(clip.yaw_rate)), float(args.bridge_budget_quantile)),
            "contact": _safe_percentile(_step_l2(clip.contact), float(args.bridge_budget_quantile)),
        }

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
        if pair:
            phi = int(pair["phi"])
            onset = int(pair["onset"])
        else:
            phi = int(align.full_state_phi)
            onset = 0
        deltas = {
            "angvel": float(np.sqrt(np.mean((target_clip.bone_angvel[onset] - walk.bone_angvel[phi]) ** 2))),
            "rootvel": float(np.linalg.norm(target_clip.root_vel[onset] - walk.root_vel[phi])),
            "yaw": float(abs(target_clip.yaw_rate[onset] - walk.yaw_rate[phi])),
            "contact": float(np.linalg.norm(target_clip.contact[onset] - walk.contact[phi])),
        }
        bridge = _bridgeability_from_deltas(
            deltas,
            bridge_budgets[target],
            horizon=h,
            groundable=bool(align.groundable),
        )
        bridge.update(
            {
                "phi": phi,
                "onset": onset,
                "pose_d": float(align.full_state_pose_d),
                "contact_d": float(align.full_state_contact_d),
                "support_start": _support_label(walk.contact[phi]),
                "support_end": _support_label(target_clip.contact[onset]),
                "deltas": deltas,
                "budget_quantile": float(args.bridge_budget_quantile),
                "budgets": bridge_budgets[target],
            }
        )
        if not pair:
            row = {
                "case": "endpoint_bridgeability:ungroundable_candidate",
                "target": target,
                "start_phase": f"phi={phi};onset=0",
                "expected_label": "separate_not_groundable",
                "regime_reached": False,
                "rate_budget": False,
                "support_honesty": False,
                "command_response": False,
                "command_compatibility": False,
                "net_integral_ok": False,
                "root_not_counter_command": False,
                "pose_continuity": bool(align.full_state_pose_d <= float(args.ground_pose_thr)),
                "endpoint_bridgeability": False,
                "legacy_acceptance_pass": False,
                "legacy_failed_family": "regime_reached,rate_budget,support_honesty,command_response,endpoint_bridgeability",
            }
            rows.append(_attach_demoted_acceptance(row, include_support_side_core=False))
            continue
        specs = [
            ("negative_control:matched_hard_seam", _make_hard_seam_sequence(walk, target_clip, phi, onset), bool(bridge["one_frame_bridgeable"])),
            (
                "negative_control:one_frame_angvel_root_switch",
                _make_one_frame_switch_sequence(walk, target_clip, phi, onset),
                bool(bridge["one_frame_bridgeable"]),
            ),
            (
                "negative_control:linear_pose_contact_proxy",
                _make_linear_proxy_sequence(walk, target_clip, phi, onset, h),
                bool(bridge["horizon_bridgeable"]),
            ),
        ]
        for case, seq, endpoint_ok in specs:
            rows.append(
                _evaluate_seq_demoted(
                    case=case,
                    target=target,
                    expected_label="fail",
                    start_phase=f"phi={phi};onset={onset};H={len(seq['yaw_rate'])}",
                    seq=seq,
                    target_bands=raw_bands[target],
                    command_bands=command_bands,
                    skeleton=skeleton,
                    endpoint_bridgeability=endpoint_ok,
                    endpoint_details=bridge,
                )
            )

    # Artifact-backed direct/lambda/proxy rows do not all carry reconstructable
    # sequences, so they keep legacy non-command failures and demote only the
    # old command family.
    for artifact_row in (
        _rows_from_bone_bridge_artifact(Path(args.bone_bridge_summary))
        + _rows_from_regime_bridge_artifact(Path(args.regime_bridge_summary))
    ):
        row = dict(artifact_row)
        row["legacy_acceptance_pass"] = bool(row.get("pass", False))
        row["legacy_failed_family"] = row.get("failed_family", "")
        row["command_compatibility"] = False if "command_response" in str(row.get("failed_family", "")) else True
        row["net_integral_ok"] = row["command_compatibility"]
        row["root_not_counter_command"] = row["command_compatibility"]
        rows.append(_attach_demoted_acceptance(row, include_support_side_core=False))

    # Lifted/anchored signal reconstructability and perturbation rows.
    all_items = _build_items(
        clips,
        horizon=h,
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    reconstructed_baseline_bands = _calibrate_reconstructed_baseline_bands(
        all_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    reconstructed_support_bands = _calibrate_reconstructed_support_side_bands(
        all_items,
        skeleton,
        horizon=h,
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    for item_i, item in enumerate(main_items):
        baseline_seq = _reconstructed_gt_seq(item, oracle_contact_passthrough=True, command_align_root_vel=False)
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(h, -1)
        aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(h, -1)
        states = {
            "flat_state281": _flat_state(item, noise_scale=0.0, rng=_rng_for(args.seed, item_i, "flat", 0)),
            "root_position_lifted": _root_position_state(
                item, baseline_seq, noise_scale=0.0, rng=_rng_for(args.seed, item_i, "root", 0)
            ),
            "support_anchor_keep_inter_anchor": _anchored_state(
                item,
                skeleton,
                baseline_seq,
                keep_inter_anchor=True,
                noise_scale=0.0,
                rng=_rng_for(args.seed, item_i, "anchor_keep", 0),
            ),
            "support_anchor_drop_inter_anchor": _anchored_state(
                item,
                skeleton,
                baseline_seq,
                keep_inter_anchor=False,
                noise_scale=0.0,
                rng=_rng_for(args.seed, item_i, "anchor_drop", 0),
            ),
        }
        for rep, state in states.items():
            rows.append(
                _evaluate_signal_seq_demoted(
                    representation=rep,
                    section="reconstructability",
                    noise_mse=None,
                    item=item,
                    state=state,
                    aux=aux,
                    baseline_seq=baseline_seq,
                    true_state=true_state,
                    baseline_bands=reconstructed_baseline_bands,
                    support_bands=reconstructed_support_bands,
                    command_bands=command_bands,
                    skeleton=skeleton,
                    min_run_frames=int(args.min_run_frames),
                )
            )
        for level in args.noise_levels:
            noise_scale = math.sqrt(float(level))
            key = f"{float(level):.0e}"
            for trial in range(int(args.noise_trials)):
                perturb_states = {
                    "flat_velocity_state281": _flat_state(
                        item, noise_scale=noise_scale, rng=_rng_for(args.seed, item_i, "flat_velocity_state281", key, trial)
                    ),
                    "root_position_lifted": _root_position_state(
                        item,
                        baseline_seq,
                        noise_scale=noise_scale,
                        rng=_rng_for(args.seed, item_i, "root_position_lifted", key, trial),
                    ),
                    "support_anchor_keep_inter_anchor": _anchored_state(
                        item,
                        skeleton,
                        baseline_seq,
                        keep_inter_anchor=True,
                        noise_scale=noise_scale,
                        rng=_rng_for(args.seed, item_i, "support_anchor_keep_inter_anchor", key, trial),
                    ),
                    "support_anchor_drop_inter_anchor": _anchored_state(
                        item,
                        skeleton,
                        baseline_seq,
                        keep_inter_anchor=False,
                        noise_scale=noise_scale,
                        rng=_rng_for(args.seed, item_i, "support_anchor_drop_inter_anchor", key, trial),
                    ),
                }
                for rep, state in perturb_states.items():
                    rows.append(
                        _evaluate_signal_seq_demoted(
                            representation=rep,
                            section="perturbation_sensitivity",
                            noise_mse=float(level),
                            item=item,
                            state=state,
                            aux=aux,
                            baseline_seq=baseline_seq,
                            true_state=true_state,
                            baseline_bands=reconstructed_baseline_bands,
                            support_bands=reconstructed_support_bands,
                            command_bands=command_bands,
                            skeleton=skeleton,
                            min_run_frames=int(args.min_run_frames),
                        )
                    )

    negative_rows = [r for r in rows if str(r.get("case", "")).startswith("negative_control:")]
    signal_reconstruct_rows = [r for r in rows if str(r.get("case", "")).startswith("signal_reconstruct:")]
    signal_perturb_rows = [r for r in rows if str(r.get("case", "")).startswith("signal_perturb:")]

    neg_by_case = _summaries_by(negative_rows, "case")
    signal_recon_by_case = _summaries_by(signal_reconstruct_rows, "case")

    by_case_noise: Dict[str, List[Mapping[str, Any]]] = {}
    for row in signal_perturb_rows:
        key = f"{row.get('case')}|noise={row.get('noise_mse')}"
        by_case_noise.setdefault(key, []).append(row)
    perturb_summary = {k: _summarize(v) for k, v in sorted(by_case_noise.items())}

    linear_pass = float(neg_by_case.get("negative_control:linear_pose_contact_proxy", {}).get("demoted_pass_rate", 0.0) or 0.0)
    all_neg_fail = all(float(rec.get("demoted_pass_rate", 0.0) or 0.0) == 0.0 for rec in neg_by_case.values())
    keep = signal_recon_by_case.get("signal_reconstruct:support_anchor_keep_inter_anchor", {})
    keep_demoted = float(keep.get("demoted_pass_rate", 0.0) or 0.0)
    keep_core = float(keep.get("support_side_core_pass_rate", 0.0) or 0.0)
    keep_full = float(keep.get("support_side_full_legacy_pass_rate", 0.0) or 0.0)
    support_split = keep_core > keep_full + 0.25
    if not all_neg_fail:
        decision = "do_not_demote_command_metric_negative_control_leak"
    elif keep_demoted >= float(args.reconstruct_pass_threshold):
        decision = "command_demotion_guard_passed_review_perturbation_tradeoff_before_decoder"
    else:
        decision = "command_demotion_guard_ok_but_reconstructability_still_blocked"

    payload = {
        "task": "command_response_demotion_replay",
        "scope": (
            "debug-only read-only acceptance replay; per-frame command_response demoted; net/integral "
            "command_compatibility added; no training/runtime/checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "horizon": h,
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "baseline_quantile": float(args.baseline_quantile),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "command_quantile": float(args.command_quantile),
            "noise_levels_mse": [float(x) for x in args.noise_levels],
            "noise_trials": int(args.noise_trials),
            "dtype": "float32",
            "device": "cpu",
        },
        "support_side_core_keys": list(SUPPORT_SIDE_CORE_KEYS),
        "support_side_demoted_keys": sorted(COMMANDISH_SUPPORT_KEYS),
        "command_bands": command_bands,
        "negative_control_summary_by_case": neg_by_case,
        "signal_reconstruct_summary_by_case": signal_recon_by_case,
        "signal_perturb_summary_by_case_noise": perturb_summary,
        "perturbation_noise_caveat": {
            "current_noise_model": "per-frame independent Gaussian in the debug candidate variable",
            "interpretation": (
                "high-frequency noise is low-passed by flat velocity integration and amplified by "
                "lifted position finite-difference; current perturbation rows are not a fair "
                "anchored-vs-flat conditioning verdict"
            ),
            "required_fair_gate": (
                "native-space correlated/bias noise, equal reconstructed-state281 MSE calibration, "
                "and dual position-side / velocity-side metrics"
            ),
        },
        "summary_by_case": _summaries_by(rows, "case"),
        "verdict": {
            "negative_controls_still_fail": bool(all_neg_fail),
            "linear_proxy_demoted_pass_rate": linear_pass,
            "anchored_keep_reconstruct_demoted_pass_rate": keep_demoted,
            "anchored_keep_support_side_core_pass_rate": keep_core,
            "anchored_keep_support_side_full_legacy_pass_rate": keep_full,
            "support_side_core_split_exposed": bool(support_split),
            "decision": decision,
        },
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_new_model": False,
            "forwarded_production_runtime_or_trainer": False,
            "edited_production_runtime_trainer_gate": False,
            "mutated_checkpoint": False,
        },
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "command_demotion_replay_summary.json"
    rows_csv = args.out_dir / "command_demotion_replay_rows.csv"
    summary_md = args.out_dir / "command_demotion_replay_summary.md"
    payload["artifacts"] = {
        "summary_json": str(summary_json),
        "rows_csv": str(rows_csv),
        "summary_md": str(summary_md),
    }
    _dump_json(summary_json, payload)
    _write_rows_csv(rows_csv, rows)
    _write_summary_md(summary_md, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--bone-bridge-summary", type=Path, default=DEFAULT_BONE_BRIDGE)
    p.add_argument("--regime-bridge-summary", type=Path, default=DEFAULT_REGIME_BRIDGE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--baseline-quantile", type=float, default=99.5)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--bridge-budget-quantile", type=float, default=95.0)
    p.add_argument("--command-quantile", type=float, default=100.0)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    p.add_argument("--noise-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--reconstruct-pass-threshold", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"wrote {payload['artifacts']['summary_md']}")
    print(f"wrote {payload['artifacts']['summary_json']}")
    print(f"wrote {payload['artifacts']['rows_csv']}")
    print(json.dumps(_jsonify(payload["verdict"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
