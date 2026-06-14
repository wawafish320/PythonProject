#!/usr/bin/env python3
"""Debug-only 8-window preflight bandfix audit and rerun.

This runner keeps the production trainer/runtime/gate/checkpoint untouched. It
reuses the 8-window debug decoder harness, audits the GT/decoder/stage1
preflight failures, relabels the three known zero-slack classes through guarded
debug bands, and only then lets the existing Stage2 minimax tail run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import TURN_CLIPS, full_state_align  # noqa: E402
from tools import run_action_handoff_inbetween_8window_train_fit_debug as fit  # noqa: E402
from tools.run_action_handoff_adjusted_acceptance_guard import (  # noqa: E402
    DEFAULT_POSE_SWEEP_PRED,
    _bridge_budgets,
)
from tools.run_action_handoff_band_audit import (  # noqa: E402
    _baseline_samples_for_target,
    _run_guard_with_bands,
)
from tools.run_action_handoff_command_demotion_replay import (  # noqa: E402
    _calibrate_command_bands,
    _evaluate_seq_demoted,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    TURN_CLIPS as RAW_TURN_CLIPS,
    WALK_F,
    _bridgeability_from_deltas,
    _calibrate_baselines,
    _fmt,
    _heading_error_rad,
    _make_hard_seam_sequence,
    _make_linear_proxy_sequence,
    _make_one_frame_switch_sequence,
    _safe_percentile,
    _step_angvel_rms,
    _step_l2,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    _guard_path_identity,
    _reconstructed_gt_seq,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


RUN_DATE = "20260605"
DEFAULT_OUT_DIR = Path(f"debug_output/_tmp_action_handoff_8window_preflight_bandfix_{RUN_DATE}")
DEFAULT_DOC_PATH = Path(f"docs/aperiodic_transition/2026-06-05_8window_preflight_bandfix_review.md")
DEFAULT_SOURCE_STAGE1_DIR = Path("debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605")

CLASS_B_FEATURES: Tuple[str, ...] = (
    "root_lateral_mean",
    "right_rel_x_mean",
    "right_rel_y_mean",
    "right_rel_z_mean",
    "right_rel_norm_p95",
    "claimed_support_slip_mean_mps",
    "claimed_support_slip_p95_mps",
)
CLASS_C_UPPER: Mapping[str, str] = {
    "yaw_rate_step_abs_p95": "yaw_rate_step_abs",
    "angvel_component_p95_p95": "angvel_step_component_p95",
}
HEADING_METRICS: Tuple[str, ...] = ("heading_error_p95_rad", "support_side.heading_error_p95_rad")


def _fit_default_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0] if old_argv else "run_action_handoff_inbetween_8window_train_fit_debug.py"]
        return fit.parse_args()
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    defaults = _fit_default_args()
    default_heading_tolerance_rad = float(defaults.heading_tolerance_rad)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=defaults.npz_root)
    p.add_argument("--z-features", type=Path, default=defaults.z_features)
    p.add_argument("--checkpoint", type=Path, default=defaults.checkpoint)
    p.add_argument("--bundle", type=Path, default=defaults.bundle)
    p.add_argument("--pretrain-template", type=Path, default=defaults.pretrain_template)
    p.add_argument("--encoder-bundle", type=Path, default=defaults.encoder_bundle)
    p.add_argument("--band-audit-summary", type=Path, default=defaults.band_audit_summary)
    p.add_argument("--source-stage1-dir", type=Path, default=DEFAULT_SOURCE_STAGE1_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    p.add_argument("--stage1-supervised-epochs", type=int, default=defaults.stage1_supervised_epochs)
    p.add_argument("--stage2-tail-epochs", type=int, default=defaults.stage2_tail_epochs)
    p.add_argument("--instrument-step-log-stride", type=int, default=defaults.instrument_step_log_stride)
    p.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--stage2-minimax-tail-lr", type=float, default=defaults.stage2_minimax_tail_lr)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--torch-num-threads", type=int, default=defaults.torch_num_threads)
    p.add_argument("--heading-tolerance-rad", type=float, default=None)
    p.add_argument("--heading-tolerance-mode", choices=("auto", "fixed"), default="auto")
    p.add_argument("--gt-zero-slack-normalized-threshold", type=float, default=1e-6)
    p.add_argument("--oracle-contact-passthrough", action="store_true", default=defaults.oracle_contact_passthrough)
    p.add_argument("--command-align-root-vel", action="store_true", default=defaults.command_align_root_vel)
    p.add_argument("--skip-stage2", action="store_true", default=False)
    cli = p.parse_args()

    args = defaults
    for key, value in vars(cli).items():
        setattr(args, key, value)
    args.requested_heading_tolerance_rad = cli.heading_tolerance_rad
    if args.heading_tolerance_rad is None:
        args.heading_tolerance_rad = default_heading_tolerance_rad
    _install_guard_aliases(args)
    return args


def _install_guard_aliases(args: argparse.Namespace) -> None:
    args.two_frame_summary = getattr(args, "guard_two_frame_summary")
    args.bone_bridge_summary = getattr(args, "guard_bone_bridge_summary")
    args.regime_bridge_summary = getattr(args, "guard_regime_bridge_summary")
    args.command_demotion_rows = getattr(args, "guard_command_demotion_rows")
    args.pose_sweep_pred_raw = DEFAULT_POSE_SWEEP_PRED
    args.pose_topk = getattr(args, "guard_pose_topk")
    args.ground_contact_thr = getattr(args, "guard_ground_contact_thr")
    args.ground_pose_thr = getattr(args, "guard_ground_pose_thr")
    args.bridge_budget_quantile = 95.0


def _jsonify(v: Any) -> Any:
    return fit._jsonify(v)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fit._write_csv(path, rows)


def _dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    fit._dump_json(path, payload)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    fit._dump_md(path, lines)


def _load_stage1_npz(path: Path, selected_idxs: Sequence[int]) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing source stage1 npz: {path}")
    data = np.load(path)
    required = ("stage1_pred_raw", "true_raw", "train_indices")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"{path} missing keys: {missing}")
    train_indices = np.asarray(data["train_indices"], dtype=np.int64).reshape(-1)
    expected = np.asarray(selected_idxs, dtype=np.int64).reshape(-1)
    if train_indices.shape != expected.shape or not np.array_equal(train_indices, expected):
        raise RuntimeError(
            f"source stage1 train_indices mismatch: got {train_indices.tolist()}, expected {expected.tolist()}"
        )
    return {
        "stage1_pred_raw": np.asarray(data["stage1_pred_raw"], dtype=np.float32),
        "true_raw": np.asarray(data["true_raw"], dtype=np.float32),
        "train_indices": train_indices,
    }


def _score(
    *,
    stage: str,
    raw: np.ndarray,
    idxs: Sequence[int],
    items: Sequence[Any],
    original_bands: Mapping[str, Mapping[str, Any]],
    accepted_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    band_metric_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    skeleton: Any,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return fit._score_raw(
        stage=stage,
        raw=np.asarray(raw, dtype=np.float32),
        idxs=idxs,
        items=items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_metric_map,
        skeleton=skeleton,
        args=args,
    )


def _metric_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return str(row.get("clip")), int(row.get("start")), str(row.get("metric"))


def _raw_slack(row: Mapping[str, Any]) -> float:
    value = float(row.get("raw_value", 0.0) or 0.0)
    if str(row.get("band_kind")) == "interval":
        lo = float(row.get("accepted_p99_band_min", 0.0) or 0.0)
        hi = float(row.get("accepted_p99_band_max", 0.0) or 0.0)
        return float(min(value - lo, hi - value))
    band = float(row.get("accepted_p99_band", 0.0) or 0.0)
    return float(band - value)


def _metric_class(metric: str) -> str:
    if metric in HEADING_METRICS:
        return "A_heading"
    if metric.startswith("support_side.") and metric[len("support_side.") :] in CLASS_B_FEATURES:
        return "B_interval_gt_edge"
    if metric in CLASS_C_UPPER:
        return "C_sibling_window_upper"
    return "other"


def _summarize_window_pass(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "n": int(len(rows)),
        "accepted_p99_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in rows)),
        "p95_shadow_clean_pass_count": int(sum(bool(r.get("p95_shadow_clean_pass", False)) for r in rows)),
        "fail_count": int(sum(str(r.get("window_state")) == "fail" for r in rows)),
        "p99_only_count": int(sum(str(r.get("window_state")) == "p99-only" for r in rows)),
    }


def _build_e1_rows(
    *,
    gt_metric: Sequence[Mapping[str, Any]],
    dec_metric: Sequence[Mapping[str, Any]],
    stage1_metric: Sequence[Mapping[str, Any]],
    zero_slack_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_by_key = {_metric_key(r): r for r in gt_metric}
    dec_by_key = {_metric_key(r): r for r in dec_metric}
    out: List[Dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for row in stage1_metric:
        state = str(row.get("state"))
        metric = str(row.get("metric"))
        cls = _metric_class(metric)
        gt = gt_by_key.get(_metric_key(row), {})
        dec = dec_by_key.get(_metric_key(row), {})
        include = state == "fail" or cls != "other"
        if not include:
            continue
        gt_slack = float(gt.get("normalized_slack", 0.0) or 0.0)
        gt_raw_slack = _raw_slack(gt) if gt else 0.0
        if state != "fail":
            bucket = "non_fail_reference"
        elif (not bool(gt.get("accepted_pass", False))) or abs(gt_slack) <= float(zero_slack_threshold):
            bucket = "band_too_tight_for_cross_window_GT"
        else:
            bucket = "fit_residual_amplification"
        if state == "fail":
            bucket_counts[bucket] += 1
            class_counts[cls] += 1
        out.append(
            {
                "clip": row.get("clip"),
                "start": int(row.get("start")),
                "end": int(row.get("end")),
                "metric": metric,
                "class": cls,
                "bucket": bucket,
                "stage1_state": state,
                "stage1_raw_value": float(row.get("raw_value", 0.0) or 0.0),
                "stage1_normalized_slack": float(row.get("normalized_slack", 0.0) or 0.0),
                "gt_raw_value": float(gt.get("raw_value", 0.0) or 0.0),
                "gt_normalized_slack": gt_slack,
                "gt_raw_slack": gt_raw_slack,
                "gt_accepted_pass": bool(gt.get("accepted_pass", False)),
                "decoder_raw_value": float(dec.get("raw_value", 0.0) or 0.0),
                "decoder_normalized_slack": float(dec.get("normalized_slack", 0.0) or 0.0),
                "decoder_accepted_pass": bool(dec.get("accepted_pass", False)),
                "band_kind": row.get("band_kind"),
                "accepted_p99_band": row.get("accepted_p99_band"),
                "accepted_p99_band_min": row.get("accepted_p99_band_min", ""),
                "accepted_p99_band_max": row.get("accepted_p99_band_max", ""),
            }
        )
    summary = {
        "bucket_counts": dict(bucket_counts),
        "fail_class_counts": dict(class_counts),
        "stage1_fail_count": int(sum(1 for r in stage1_metric if str(r.get("state")) == "fail")),
    }
    return out, summary


def _selected_relabel_pairs(stage1_metric: Sequence[Mapping[str, Any]]) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    upper_pairs: set[Tuple[str, str]] = set()
    support_pairs: set[Tuple[str, str]] = set()
    for row in stage1_metric:
        if str(row.get("state")) != "fail":
            continue
        metric = str(row.get("metric"))
        target = str(row.get("clip"))
        if metric in CLASS_C_UPPER:
            upper_pairs.add((target, metric))
        if metric.startswith("support_side."):
            feature = metric[len("support_side.") :]
            if feature in CLASS_B_FEATURES:
                support_pairs.add((target, feature))
    return upper_pairs, support_pairs


def _support_feature_values(
    *,
    items: Sequence[Any],
    target: str,
    feature: str,
    skeleton: Any,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, str]:
    vals: List[float] = []
    for item in items:
        if item.clip not in {WALK_F, target}:
            continue
        seq = _reconstructed_gt_seq(
            item,
            oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
            command_align_root_vel=bool(args.command_align_root_vel),
        )
        contract = fit._support_contract(seq["contact"], min_run_frames=int(args.min_run_frames))
        foot = fit._foot_positions(seq["rot6d"], seq["root_pos"], skeleton)
        feats = fit._support_side_features(seq, contract["normalized_label_sequence"], foot)
        vals.append(float(feats.get(feature, 0.0)))
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr, f"{WALK_F}+{target}:reconstructed_state281_windows"


def _apply_support_relabel(
    support_bands: Dict[str, Dict[str, Any]],
    *,
    target: str,
    feature: str,
    new_min: float,
    new_max: float,
    p01: float,
    p99: float,
) -> None:
    band = support_bands[target]["feature_bands"][feature]
    band["min"] = float(new_min)
    band["max"] = float(new_max)
    band["p01"] = float(min(float(band.get("p01", p01)), float(p01)))
    band["p99"] = float(max(float(band.get("p99", p99)), float(p99)))
    band["bandfix_basis"] = "pooled_walkf_target_p01_p99_no_tightening"


def _apply_upper_relabel(
    bands: Dict[str, Dict[str, Any]],
    *,
    target: str,
    band_key: str,
    new_band: float,
) -> None:
    bands[target][band_key] = float(new_band)
    if band_key == "foot_slip_contacted_speed_mps":
        bands[target]["foot_slip_contacted_speed_p95_mps"] = float(new_band)
        foot = bands[target].setdefault("foot_slip", {})
        if isinstance(foot, dict):
            foot["contacted_speed_p95_mps"] = float(new_band)


def _build_class_bc_relabels(
    *,
    clips: Mapping[str, Any],
    all_items: Sequence[Any],
    skeleton: Any,
    raw_bands: Mapping[str, Mapping[str, Any]],
    accepted_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    upper_pairs: set[Tuple[str, str]],
    support_pairs: set[Tuple[str, str]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw_out = copy.deepcopy(raw_bands)
    accepted_out = copy.deepcopy(accepted_bands)
    support_out = copy.deepcopy(support_bands)
    rows: List[Dict[str, Any]] = []

    for target, metric in sorted(upper_pairs):
        band_key = CLASS_C_UPPER[metric]
        samples, event_mask, domain = _baseline_samples_for_target(
            clips,
            target,
            band_key,
            skeleton,
            level_center=None,
        )
        del event_mask
        finite = np.asarray(samples, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        p99 = _safe_percentile(finite, 99.0)
        old = float(accepted_out[target].get(band_key, 0.0) or 0.0)
        new = max(old, p99)
        _apply_upper_relabel(raw_out, target=target, band_key=band_key, new_band=new)
        _apply_upper_relabel(accepted_out, target=target, band_key=band_key, new_band=new)
        rows.append(
            {
                "class": "C_sibling_window_upper",
                "target": target,
                "metric": metric,
                "band_key": band_key,
                "old_band": old,
                "new_band": new,
                "pooled_p99": p99,
                "basis": f"{domain}:continuous_p99",
                "sample_count": int(finite.size),
                "tightened": bool(new < old),
                "widened": bool(new > old),
                "guard_shortcut_still_fail": "",
                "guard_command_demotion_still_fail": "",
                "guard_identity_pass": "",
            }
        )

    for target, feature in sorted(support_pairs):
        vals, domain = _support_feature_values(
            items=all_items,
            target=target,
            feature=feature,
            skeleton=skeleton,
            args=args,
        )
        p01 = float(np.percentile(vals, 1)) if vals.size else 0.0
        p99 = float(np.percentile(vals, 99)) if vals.size else 0.0
        old_band = support_out[target]["feature_bands"][feature]
        old_min = float(old_band.get("min", 0.0) or 0.0)
        old_max = float(old_band.get("max", 0.0) or 0.0)
        new_min = min(old_min, p01)
        new_max = max(old_max, p99)
        _apply_support_relabel(
            support_out,
            target=target,
            feature=feature,
            new_min=new_min,
            new_max=new_max,
            p01=p01,
            p99=p99,
        )
        rows.append(
            {
                "class": "B_interval_gt_edge",
                "target": target,
                "metric": f"support_side.{feature}",
                "band_key": feature,
                "old_band": f"[{old_min},{old_max}]",
                "new_band": f"[{new_min},{new_max}]",
                "old_min": old_min,
                "old_max": old_max,
                "new_min": new_min,
                "new_max": new_max,
                "pooled_p01": p01,
                "pooled_p99": p99,
                "basis": domain,
                "sample_count": int(vals.size),
                "tightened": bool(new_min > old_min or new_max < old_max),
                "widened": bool(new_min < old_min or new_max > old_max),
                "guard_shortcut_still_fail": "",
                "guard_command_demotion_still_fail": "",
                "guard_identity_pass": "",
            }
        )
    return rows, raw_out, accepted_out, support_out


def _support_heading_floor(support_bands: Dict[str, Dict[str, Any]], tol: float) -> None:
    for target, rec in support_bands.items():
        feature_bands = rec.get("feature_bands", {})
        band = feature_bands.get("heading_error_p95_rad") if isinstance(feature_bands, Mapping) else None
        if not isinstance(band, dict):
            continue
        old_max = float(band.get("max", 0.0) or 0.0)
        band["max"] = float(max(old_max, float(tol)))
        band["p99"] = float(max(float(band.get("p99", old_max) or old_max), float(tol)))
        band["bandfix_basis"] = "shared_heading_tolerance_floor"


def _heading_value(seq: Mapping[str, np.ndarray]) -> float:
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    return float(_safe_percentile(_heading_error_rad(root_vel, cond_dir), 95.0))


def _shortcut_sequences(
    *,
    clips: Mapping[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}
    budgets = _bridge_budgets(clips, float(args.bridge_budget_quantile))
    walk = clips[WALK_F]
    for target in RAW_TURN_CLIPS:
        target_clip = clips[target]
        align = full_state_align(
            walk.state281,
            target_clip.state281[0],
            topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        pair = matched_pairs.get(target)
        if not pair:
            continue
        phi = int(pair["phi"])
        onset = int(pair["onset"])
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
        specs = (
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
        )
        for case, seq, endpoint_ok in specs:
            rows.append(
                {
                    "case": case,
                    "target": target,
                    "start_phase": f"phi={phi};onset={onset};H={len(seq['yaw_rate'])}",
                    "endpoint_bridgeability": bool(endpoint_ok),
                    "seq": seq,
                }
            )
    return rows


def _heading_seam_audit(
    *,
    gt_metric: Sequence[Mapping[str, Any]],
    dec_metric: Sequence[Mapping[str, Any]],
    stage1_metric: Sequence[Mapping[str, Any]],
    clips: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    rows: List[Dict[str, Any]] = []

    def add_metric_rows(source: str, metric_rows: Sequence[Mapping[str, Any]], *, role: str) -> None:
        for row in metric_rows:
            metric = str(row.get("metric"))
            if metric not in HEADING_METRICS:
                continue
            rows.append(
                {
                    "source": source,
                    "role": role,
                    "case": "selected_8window",
                    "target": row.get("clip"),
                    "start": int(row.get("start")),
                    "end": int(row.get("end")),
                    "metric": metric,
                    "heading_available": True,
                    "heading_error_p95_rad": float(row.get("raw_value", 0.0) or 0.0),
                    "accepted_pass_before_floor": bool(row.get("accepted_pass", False)),
                    "failed_family": row.get("failed_family", ""),
                }
            )

    add_metric_rows("exact_gt_true_raw", gt_metric, role="decoder_side")
    add_metric_rows("decoder_replay_from_gt_raw", dec_metric, role="decoder_side")
    add_metric_rows("stage1_best_pred", stage1_metric, role="stage1_reference")

    command_rows_by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    with Path(args.command_demotion_rows).open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            if str(rec.get("case", "")).startswith("negative_control:"):
                command_rows_by_case[str(rec.get("case"))].append(rec)

    for rec in _shortcut_sequences(clips=clips, args=args):
        heading = _heading_value(rec["seq"])
        rows.append(
            {
                "source": "shortcut_sequence",
                "role": "negative_control",
                "case": rec["case"],
                "target": rec["target"],
                "start": "",
                "end": "",
                "metric": "heading_error_p95_rad",
                "heading_available": True,
                "heading_error_p95_rad": heading,
                "accepted_pass_before_floor": "",
                "failed_family": "",
                "start_phase": rec["start_phase"],
            }
        )
        rows.append(
            {
                "source": "command_demotion_reconstructed_sequence",
                "role": "negative_control",
                "case": rec["case"],
                "target": rec["target"],
                "start": "",
                "end": "",
                "metric": "heading_error_p95_rad",
                "heading_available": True,
                "heading_error_p95_rad": heading,
                "accepted_pass_before_floor": "",
                "failed_family": "",
                "start_phase": rec["start_phase"],
            }
        )

    for case, subset in sorted(command_rows_by_case.items()):
        if case in {"negative_control:matched_hard_seam", "negative_control:one_frame_angvel_root_switch", "negative_control:linear_pose_contact_proxy"}:
            continue
        for rec in subset:
            rows.append(
                {
                    "source": "command_demotion_artifact_only",
                    "role": "negative_control_artifact_only",
                    "case": case,
                    "target": rec.get("target"),
                    "start": "",
                    "end": "",
                    "metric": "heading_error_p95_rad",
                    "heading_available": False,
                    "heading_error_p95_rad": "",
                    "accepted_pass_before_floor": "",
                    "failed_family": rec.get("demoted_failed_family", ""),
                    "start_phase": rec.get("start_phase", ""),
                }
            )

    decoder_vals = [
        float(r["heading_error_p95_rad"])
        for r in rows
        if r.get("role") == "decoder_side" and bool(r.get("heading_available", False))
    ]
    negative_vals = [
        float(r["heading_error_p95_rad"])
        for r in rows
        if r.get("role") == "negative_control" and bool(r.get("heading_available", False))
    ]
    stage1_vals = [
        float(r["heading_error_p95_rad"])
        for r in rows
        if r.get("role") == "stage1_reference" and bool(r.get("heading_available", False))
    ]
    max_decoder = max(decoder_vals) if decoder_vals else float("nan")
    min_negative = min(negative_vals) if negative_vals else float("nan")
    clean = bool(math.isfinite(max_decoder) and math.isfinite(min_negative) and max_decoder < min_negative)
    if not clean:
        min_examples = []
        for r in rows:
            if r.get("role") != "negative_control" or not bool(r.get("heading_available", False)):
                continue
            value = float(r["heading_error_p95_rad"])
            if math.isfinite(min_negative) and abs(value - min_negative) <= 1e-12:
                min_examples.append(
                    {
                        "source": r.get("source"),
                        "case": r.get("case"),
                        "target": r.get("target"),
                        "start_phase": r.get("start_phase", ""),
                        "heading_error_p95_rad": value,
                    }
                )
        summary = {
            "clean_seam": False,
            "max_decoder_side_heading": max_decoder,
            "min_negative_heading": min_negative,
            "negative_heading_available_count": int(len(negative_vals)),
            "negative_heading_unavailable_artifact_only_count": int(
                sum(r.get("role") == "negative_control_artifact_only" for r in rows)
            ),
            "min_negative_examples": min_examples[:8],
            "decision": "no_clean_heading_seam_stop",
        }
        return rows, summary, float("nan")
    auto_tol = 0.5 * (max_decoder + min_negative)
    if args.heading_tolerance_mode == "fixed":
        if getattr(args, "requested_heading_tolerance_rad", None) is None:
            raise RuntimeError("--heading-tolerance-mode fixed requires --heading-tolerance-rad")
        tol = float(args.requested_heading_tolerance_rad)
    elif getattr(args, "requested_heading_tolerance_rad", None) is not None:
        tol = float(args.requested_heading_tolerance_rad)
    else:
        tol = float(auto_tol)
    in_seam = bool(max_decoder < tol < min_negative)
    stage1_max = max(stage1_vals) if stage1_vals else float("nan")
    summary = {
        "clean_seam": clean,
        "max_decoder_side_heading": max_decoder,
        "min_negative_heading": min_negative,
        "auto_tolerance_rad": auto_tol,
        "selected_tolerance_rad": tol,
        "selected_tolerance_in_seam": in_seam,
        "stage1_best_heading_max_reference": stage1_max,
        "stage1_best_heading_cleared_by_selected_tolerance": bool(math.isfinite(stage1_max) and stage1_max <= tol + fit.EPS),
        "negative_heading_available_count": int(len(negative_vals)),
        "negative_heading_unavailable_artifact_only_count": int(
            sum(r.get("role") == "negative_control_artifact_only" for r in rows)
        ),
        "decision": "heading_tolerance_selected" if in_seam else "selected_heading_tolerance_outside_seam_stop",
    }
    return rows, summary, tol


def _guard_relabels(
    *,
    rows: List[Dict[str, Any]],
    clips: Mapping[str, Any],
    skeleton: Any,
    main_items: Sequence[Any],
    raw_base: Mapping[str, Mapping[str, Any]],
    accepted_base: Mapping[str, Mapping[str, Any]],
    support_base: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    for row in rows:
        trial_raw = copy.deepcopy(raw_base)
        trial_accepted = copy.deepcopy(accepted_base)
        trial_support = copy.deepcopy(support_base)
        if row["class"] == "C_sibling_window_upper":
            _apply_upper_relabel(
                trial_raw,
                target=str(row["target"]),
                band_key=str(row["band_key"]),
                new_band=float(row["new_band"]),
            )
            _apply_upper_relabel(
                trial_accepted,
                target=str(row["target"]),
                band_key=str(row["band_key"]),
                new_band=float(row["new_band"]),
            )
        else:
            _apply_support_relabel(
                trial_support,
                target=str(row["target"]),
                feature=str(row["band_key"]),
                new_min=float(row["new_min"]),
                new_max=float(row["new_max"]),
                p01=float(row["pooled_p01"]),
                p99=float(row["pooled_p99"]),
            )
        guard = _run_guard_with_bands(
            args=args,
            clips=clips,
            skeleton=skeleton,
            main_items=main_items,
            raw_bands=trial_raw,
            reconstructed_bands=trial_accepted,
            support_bands=trial_support,
            label=f"{row['target']}:{row['metric']}",
        )
        verdict = guard.get("verdict", {}) or {}
        row["guard_shortcut_still_fail"] = bool(verdict.get("shortcut_negative_controls_still_fail", False))
        row["guard_command_demotion_still_fail"] = bool(verdict.get("command_demotion_negative_controls_still_fail", False))
        row["guard_identity_pass"] = bool(verdict.get("guard_path_identity_pass", False))
        row["guard_gate_w4096_full_family_pass"] = bool(verdict.get("gate_w4096_full_family_pass", False))
        row["guard_decision"] = verdict.get("decision", "")


def _command_demotion_sequence_guard(
    *,
    clips: Mapping[str, Any],
    raw_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    command_bands = _calibrate_command_bands(clips, horizon=int(args.horizon), quantile=99.5)
    rows: List[Dict[str, Any]] = []
    for rec in _shortcut_sequences(clips=clips, args=args):
        row = _evaluate_seq_demoted(
            case=str(rec["case"]),
            target=str(rec["target"]),
            expected_label="fail",
            start_phase=str(rec["start_phase"]),
            seq=rec["seq"],
            target_bands=raw_bands[str(rec["target"])],
            command_bands=command_bands,
            skeleton=skeleton,
            endpoint_bridgeability=bool(rec["endpoint_bridgeability"]),
            endpoint_details={"source": "bandfix_recomputed_sequence_guard"},
        )
        metrics = row.get("metrics", {}) or {}
        row["heading_error_p95_rad"] = float(metrics.get("heading_error_p95_rad", 0.0) or 0.0)
        rows.append(row)
    return rows


def _stage2_stall_rows(
    *,
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    step_log: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_stage1 = {(str(r["clip"]), int(r["start"])): r for r in stage1_rows}
    first_gate = None
    last_gate = None
    gate_vals = [
        float(r.get("loss_refactor_true_window_metric_hard_max_gate", 0.0) or 0.0)
        for r in step_log
        if not bool(r.get("skipped", False))
    ]
    if gate_vals:
        first_gate = gate_vals[0]
        last_gate = gate_vals[-1]
    out: List[Dict[str, Any]] = []
    for row in stage2_rows:
        key = (str(row["clip"]), int(row["start"]))
        s1 = by_stage1.get(key, {})
        final_fail = [str(x) for x in row.get("fail_metrics", [])]
        final_p99_only = [str(x) for x in row.get("p99_only_metrics", [])]
        if bool(row.get("accepted_p99_pass", False)) and bool(row.get("p95_shadow_clean_pass", False)):
            cls = "accepted_clean_pass"
            evidence = "accepted p99 and p95-shadow both pass"
        elif bool(row.get("accepted_p99_pass", False)):
            cls = "accepted_p99_pass_p95_shadow_risk"
            evidence = f"accepted p99 pass; p95 shadow misses {len(final_p99_only)} metrics"
        elif bool(s1.get("accepted_p99_pass", False)):
            if first_gate is not None and last_gate is not None and last_gate < first_gate:
                cls = "optimization_capacity_residual"
                evidence = "Stage1 preflight passed; minimax hard gate improved but final accepted band still fails"
            else:
                cls = "conflict_window_candidate_unresolved"
                evidence = "Stage1 preflight passed; minimax did not show a monotone hard-gate improvement"
        else:
            cls = "preflight_not_unlocked"
            evidence = "Stage1 still failed preflight under relabeled bands"
        out.append(
            {
                "clip": row["clip"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "stage1_accepted_p99_pass": bool(s1.get("accepted_p99_pass", False)),
                "stage2_accepted_p99_pass": bool(row.get("accepted_p99_pass", False)),
                "stage2_p95_shadow_clean_pass": bool(row.get("p95_shadow_clean_pass", False)),
                "stage2_window_state": row.get("window_state"),
                "classification": cls,
                "evidence": evidence,
                "stage2_fail_metrics": final_fail,
                "stage2_p99_only_metrics": final_p99_only,
                "hard_gate_initial": first_gate,
                "hard_gate_final": last_gate,
            }
        )
    return out


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    lines = [
        "# 8-Window Preflight Bandfix",
        "",
        "Debug-only run. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## Verdict",
        "",
        f"- E1 GT pass: `{verdict['e1_gt_pass_count']}/8`; decoder pass: `{verdict['e1_decoder_pass_count']}/8`.",
        f"- Class B/C relabel count: `{verdict['class_bc_relabel_count']}`.",
        f"- heading tolerance: `{_fmt(verdict['heading_tolerance_rad'], 8)}` rad; in seam: `{str(verdict['heading_tolerance_in_seam']).lower()}`.",
        f"- relabeled Stage1 accepted pass: `{verdict['stage1_relabel_pass_count']}/8`.",
        f"- Stage2 skipped: `{str(verdict['stage2_skipped']).lower()}`.",
        f"- Stage2 accepted pass: `{verdict['stage2_pass_count']}/8`; p95 clean: `{verdict['stage2_p95_clean_count']}/8`.",
        f"- final guard shortcut fail: `{str(verdict['shortcut_negative_controls_still_fail']).lower()}`; command-demotion fail: `{str(verdict['command_demotion_negative_controls_still_fail']).lower()}`.",
        f"- guard identity `max_abs_seq_delta`: `{_fmt(verdict['guard_identity_max_abs_seq_delta'], 8)}`.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    _dump_md(path, lines)


def _write_doc_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    e1 = payload["e1"]
    heading = payload["heading_seam_summary"]
    final_guard = payload["final_guard"].get("verdict", {}) or {}
    lines = [
        "# 8-Window Preflight Bandfix Review",
        "",
        "Scope: debug-only fixed-oracle-schedule band/preflight audit. No production Trainer/runtime/gate/checkpoint mutation.",
        "",
        "## 1. E1 full-family GT/decoder preflight",
        "",
        f"- exact `true_raw [8,6704] float32 CPU NumPy` accepted pass: `{verdict['e1_gt_pass_count']}/8`.",
        f"- decoder-replay-from-GT accepted pass: `{verdict['e1_decoder_pass_count']}/8`.",
        f"- Stage1-best fail buckets: `{e1['bucket_summary'].get('bucket_counts', {})}`.",
        f"- Stage1-best fail classes: `{e1['bucket_summary'].get('fail_class_counts', {})}`.",
        "",
        "## 2. E2 Class B/C relabels",
        "",
        f"- relabel rows: `{verdict['class_bc_relabel_count']}`; source: pooled `Walk_F + target` baseline, no tightening.",
        f"- per-row table: `{payload['artifacts']['band_relabel_classBC_csv']}`.",
        "",
        "## 3. E3 heading seam",
        "",
        f"- max decoder-side heading: `{_fmt(heading.get('max_decoder_side_heading'), 8)}` rad.",
        f"- min available negative-control heading: `{_fmt(heading.get('min_negative_heading'), 8)}` rad.",
        f"- selected tolerance: `{_fmt(heading.get('selected_tolerance_rad'), 8)}` rad.",
        f"- selected tolerance in seam: `{str(heading.get('selected_tolerance_in_seam')).lower()}`.",
        f"- Stage1-best max heading reference: `{_fmt(heading.get('stage1_best_heading_max_reference'), 8)}` rad.",
        f"- support-side heading floor applied to command and support metrics: `true`.",
        "",
        "## 4. E4 re-guard and Stage2",
        "",
        f"- one-window full-family guard pass: `{str(final_guard.get('gate_w4096_full_family_pass')).lower()}`.",
        f"- shortcut negative controls still fail: `{str(final_guard.get('shortcut_negative_controls_still_fail')).lower()}`.",
        f"- command-demotion negative controls still fail: `{str(final_guard.get('command_demotion_negative_controls_still_fail')).lower()}`.",
        f"- guard identity pass: `{str(final_guard.get('guard_path_identity_pass')).lower()}`; `max_abs_seq_delta={_fmt(verdict['guard_identity_max_abs_seq_delta'], 8)}`.",
        f"- relabeled Stage1 accepted pass: `{verdict['stage1_relabel_pass_count']}/8`.",
        f"- Stage2 accepted pass: `{verdict['stage2_pass_count']}/8`; p95 clean: `{verdict['stage2_p95_clean_count']}/8`.",
        "",
        "## 5. Stage2 stall classification",
        "",
        "| window | p99 | p95 | classification | fail metrics | p99-only metrics |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in payload["stall_classification"]:
        lines.append(
            f"| {row['clip']}:{row['start']}-{row['end']} | "
            f"{str(row['stage2_accepted_p99_pass']).lower()} | "
            f"{str(row['stage2_p95_shadow_clean_pass']).lower()} | "
            f"{row['classification']} | {row['stage2_fail_metrics']} | {row['stage2_p99_only_metrics']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Answer status",
            "",
            (
                "- The 8-window minimax generalization question is now executable under guarded bands, "
                "but deterministic fixed-schedule Stage2 still is not a sampling/multimodality proof."
            ),
        ]
    )
    _dump_md(path, lines)


def _write_stopped_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    heading = payload["heading_seam_summary"]
    lines = [
        "# 8-Window Preflight Bandfix",
        "",
        "Debug-only stopped run. Production Trainer/runtime/gate/checkpoint are unchanged.",
        "",
        "## Stop",
        "",
        f"- reason: `{verdict['reason']}`",
        f"- E1 GT pass: `{verdict['e1_gt_pass_count']}/8`; decoder pass: `{verdict['e1_decoder_pass_count']}/8`.",
        f"- Stage1-best fail buckets: `{payload['e1']['bucket_summary'].get('bucket_counts', {})}`.",
        f"- Class B/C relabel rows: `{verdict['class_bc_relabel_count']}`.",
        f"- Class B/C widened rows: `{verdict.get('class_bc_widened_count', 0)}`.",
        f"- heading max decoder: `{_fmt(heading.get('max_decoder_side_heading'), 8)}` rad.",
        f"- heading min available negative: `{_fmt(heading.get('min_negative_heading'), 8)}` rad.",
        "- Stage2 was not run.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    _dump_md(path, lines)


def _write_stopped_doc_md(path: Path, payload: Mapping[str, Any]) -> None:
    heading = payload["heading_seam_summary"]
    e1 = payload["e1"]
    lines = [
        "# 8-Window Preflight Bandfix Review",
        "",
        "Scope: debug-only stopped preflight audit. No production Trainer/runtime/gate/checkpoint mutation.",
        "",
        "## 1. Stop condition",
        "",
        "- Stage2 was not run because E3 did not find a strict decoder-vs-negative heading seam.",
        f"- max decoder-side heading: `{_fmt(heading.get('max_decoder_side_heading'), 8)}` rad.",
        f"- min available negative-control heading: `{_fmt(heading.get('min_negative_heading'), 8)}` rad.",
        "- Per prompt, this means heading is non-discriminative for the available negative controls and must be treated as a passenger/gross-violation check unless the negative-control set is partitioned.",
        f"- min-heading negative examples: `{heading.get('min_negative_examples', [])}`.",
        "",
        "## 2. E1 full-family preflight",
        "",
        f"- exact GT accepted pass: `{payload['verdict']['e1_gt_pass_count']}/8`.",
        f"- decoder-replay-from-GT accepted pass: `{payload['verdict']['e1_decoder_pass_count']}/8`.",
        f"- Stage1-best fail buckets: `{e1['bucket_summary'].get('bucket_counts', {})}`.",
        f"- Stage1-best fail classes: `{e1['bucket_summary'].get('fail_class_counts', {})}`.",
        "",
        "## 3. E2 Class B/C relabel attempt",
        "",
        f"- relabel rows audited: `{payload['verdict']['class_bc_relabel_count']}`.",
        f"- rows that actually widened: `{payload['verdict'].get('class_bc_widened_count', 0)}`.",
        "- Under the implemented `pooled p1/p99 + no tightening` rule, B/C rows did not widen beyond the existing inclusive GT min/max bands; see `band_relabel_classBC.csv`.",
        "- That is a discrepancy against the intended 'GT gets genuine slack' criterion and should be resolved before any Stage2 rerun.",
        "",
        "## 4. Artifacts",
        "",
    ]
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = fit._load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = fit._load_skeleton_meta(Path(args.npz_root), fit.WALK_F)
    raw_bands = _calibrate_baselines(clips, skeleton, quantile=99.5)
    all_items = fit._build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [it for it in all_items if it.clip in MATCHED_TARGETS]
    if len(main_items) != 188:
        raise RuntimeError(f"expected 188 matched windows, got {len(main_items)}")
    selected_idxs, selected_windows = fit._select_stratified_windows(
        main_items,
        horizon=int(args.horizon),
        event_window=int(args.event_window),
        n=8,
    )

    original_bands = fit._calibrate_reconstructed_baseline_bands(
        main_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    accepted_bands = fit._copy_bands(original_bands)
    support_bands = fit._calibrate_reconstructed_support_side_bands(
        main_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
    )
    band_audit = fit._load_band_audit(Path(args.band_audit_summary))
    fit._apply_accepted_relabels(accepted_bands, band_audit["accepted_relabels"])
    fit._apply_accepted_relabels(raw_bands, band_audit["accepted_relabels"])

    source_npz = _load_stage1_npz(Path(args.source_stage1_dir) / "pred_raw.npz", selected_idxs)
    true_raw = source_npz["true_raw"]
    stage1_pred_raw = source_npz["stage1_pred_raw"]

    gt_window, gt_metric = _score(
        stage="exact_gt_true_raw",
        raw=true_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )
    dec_window, dec_metric = _score(
        stage="decoder_replay_from_gt_raw",
        raw=true_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )
    stage1_window_before, stage1_metric_before = _score(
        stage="stage1_best_pred_before_bandfix",
        raw=stage1_pred_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )
    e1_rows, e1_summary = _build_e1_rows(
        gt_metric=gt_metric,
        dec_metric=dec_metric,
        stage1_metric=stage1_metric_before,
        zero_slack_threshold=float(args.gt_zero_slack_normalized_threshold),
    )

    upper_pairs, support_pairs = _selected_relabel_pairs(stage1_metric_before)
    relabel_rows, raw_relabeled, accepted_relabeled, support_relabeled = _build_class_bc_relabels(
        clips=clips,
        all_items=all_items,
        skeleton=skeleton,
        raw_bands=raw_bands,
        accepted_bands=accepted_bands,
        support_bands=support_bands,
        upper_pairs=upper_pairs,
        support_pairs=support_pairs,
        args=args,
    )
    _guard_relabels(
        rows=relabel_rows,
        clips=clips,
        skeleton=skeleton,
        main_items=main_items,
        raw_base=raw_bands,
        accepted_base=accepted_bands,
        support_base=support_bands,
        args=args,
    )

    heading_rows, heading_summary, heading_tol = _heading_seam_audit(
        gt_metric=gt_metric,
        dec_metric=dec_metric,
        stage1_metric=stage1_metric_before,
        clips=clips,
        args=args,
    )
    if not bool(heading_summary.get("selected_tolerance_in_seam", False)):
        artifacts = {
            "gt_decoder_fullfamily_preflight_csv": str(out_dir / "gt_decoder_fullfamily_preflight.csv"),
            "gt_decoder_fullfamily_preflight_json": str(out_dir / "gt_decoder_fullfamily_preflight.json"),
            "band_relabel_classBC_csv": str(out_dir / "band_relabel_classBC.csv"),
            "heading_seam_audit_csv": str(out_dir / "heading_seam_audit.csv"),
            "heading_seam_audit_json": str(out_dir / "heading_seam_audit.json"),
            "summary_json": str(out_dir / "summary.json"),
            "summary_md": str(out_dir / "summary.md"),
            "doc_md": str(args.doc_path),
        }
        verdict = {
            "stopped": True,
            "reason": heading_summary.get("decision"),
            "e1_gt_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in gt_window)),
            "e1_decoder_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in dec_window)),
            "class_bc_relabel_count": int(len(relabel_rows)),
            "class_bc_widened_count": int(sum(bool(r.get("widened", False)) for r in relabel_rows)),
            "class_bc_tightened_count": int(sum(bool(r.get("tightened", False)) for r in relabel_rows)),
            "stage2_skipped": True,
            "debug_only": True,
            "modified_production_runtime_trainer_gate": False,
            "modified_checkpoint": False,
        }
        payload = {
            "task": "8window_preflight_bandfix",
            "selected_windows": selected_windows,
            "e1": {"bucket_rows": e1_rows, "bucket_summary": e1_summary},
            "class_bc_relabels": relabel_rows,
            "heading_seam_summary": heading_summary,
            "verdict": verdict,
            "artifacts": artifacts,
        }
        _write_csv(Path(artifacts["gt_decoder_fullfamily_preflight_csv"]), gt_metric + dec_metric + stage1_metric_before + e1_rows)
        _dump_json(Path(artifacts["gt_decoder_fullfamily_preflight_json"]), payload["e1"])
        _write_csv(Path(artifacts["band_relabel_classBC_csv"]), relabel_rows)
        _write_csv(Path(artifacts["heading_seam_audit_csv"]), heading_rows)
        _dump_json(Path(artifacts["heading_seam_audit_json"]), heading_summary)
        _dump_json(Path(artifacts["summary_json"]), payload)
        _write_stopped_summary_md(Path(artifacts["summary_md"]), payload)
        _write_stopped_doc_md(Path(args.doc_path), payload)
        return payload

    args.heading_tolerance_rad = float(heading_tol)
    _support_heading_floor(support_relabeled, float(heading_tol))

    gt_window_after, gt_metric_after = _score(
        stage="exact_gt_true_raw_after_bandfix",
        raw=true_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_relabeled,
        support_bands=support_relabeled,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )
    dec_window_after, dec_metric_after = _score(
        stage="decoder_replay_from_gt_raw_after_bandfix",
        raw=true_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_relabeled,
        support_bands=support_relabeled,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )
    stage1_window_after, stage1_metric_after = _score(
        stage="stage1_best_pred_after_bandfix",
        raw=stage1_pred_raw,
        idxs=selected_idxs,
        items=main_items,
        original_bands=original_bands,
        accepted_bands=accepted_relabeled,
        support_bands=support_relabeled,
        band_metric_map=band_audit["metric_map"],
        skeleton=skeleton,
        args=args,
    )

    final_guard = _run_guard_with_bands(
        args=args,
        clips=clips,
        skeleton=skeleton,
        main_items=main_items,
        raw_bands=raw_relabeled,
        reconstructed_bands=accepted_relabeled,
        support_bands=support_relabeled,
        label="final_A_B_C_relabels",
    )
    guard_identity_selected = _guard_path_identity(
        items=main_items,
        idxs=selected_idxs,
        baseline_bands=accepted_relabeled,
        support_bands=support_relabeled,
        skeleton=skeleton,
        min_run_frames=int(args.min_run_frames),
        horizon=int(args.horizon),
        oracle_contact_passthrough=bool(args.oracle_contact_passthrough),
        command_align_root_vel=bool(args.command_align_root_vel),
        pass_rate_threshold=1.0,
    )
    command_demoted_rows = _command_demotion_sequence_guard(
        clips=clips,
        raw_bands=raw_relabeled,
        skeleton=skeleton,
        args=args,
    )

    base = fit._build_base_operator(args, Path(args.npz_root), device)
    if bool(args.skip_stage2):
        raise RuntimeError("--skip-stage2 is only for local debugging; full E4 requires Stage2")
    result = fit._train_debug_decoder(
        idxs=selected_idxs,
        items=main_items,
        base=base,
        skeleton=skeleton,
        original_bands=original_bands,
        accepted_bands=accepted_relabeled,
        support_bands=support_relabeled,
        band_metric_map=band_audit["metric_map"],
        args=args,
        device=device,
    )
    per_window_rows = result["stage1_window_rows"] + result["stage2_window_rows"]
    per_metric_rows = result["stage1_metric_rows"] + result["stage2_metric_rows"]
    stall_rows = _stage2_stall_rows(
        stage1_rows=result["stage1_window_rows"],
        stage2_rows=result["stage2_window_rows"],
        metric_rows=per_metric_rows,
        step_log=result["stage2_step_log"],
    )

    pred_path = out_dir / "pred_raw.npz"
    np.savez_compressed(
        pred_path,
        stage2_pred_raw=np.asarray(result["stage2_raw"], dtype=np.float32),
        stage1_pred_raw=np.asarray(result["stage1_raw"], dtype=np.float32),
        true_raw=np.asarray(result["true_raw"], dtype=np.float32),
        train_indices=np.asarray(selected_idxs, dtype=np.int64),
        clip=np.asarray([main_items[int(i)].clip for i in selected_idxs]),
        start=np.asarray([int(main_items[int(i)].start) for i in selected_idxs], dtype=np.int64),
        end=np.asarray([int(main_items[int(i)].end) for i in selected_idxs], dtype=np.int64),
    )
    state_path = out_dir / "decoder_state.pt"
    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "x_scaler": {"mean": result["x_scaler"].mean, "std": result["x_scaler"].std},
            "y_scaler": {"mean": result["y_scaler"].mean, "std": result["y_scaler"].std},
            "selected_idxs": [int(x) for x in selected_idxs],
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
        },
        state_path,
    )

    artifacts = {
        "gt_decoder_fullfamily_preflight_csv": str(out_dir / "gt_decoder_fullfamily_preflight.csv"),
        "gt_decoder_fullfamily_preflight_json": str(out_dir / "gt_decoder_fullfamily_preflight.json"),
        "band_relabel_classBC_csv": str(out_dir / "band_relabel_classBC.csv"),
        "heading_seam_audit_csv": str(out_dir / "heading_seam_audit.csv"),
        "heading_seam_audit_json": str(out_dir / "heading_seam_audit.json"),
        "negative_control_guard_csv": str(out_dir / "negative_control_guard.csv"),
        "per_window_csv": str(out_dir / "per_window.csv"),
        "per_metric_csv": str(out_dir / "per_metric.csv"),
        "stage2_minimax_step_log_csv": str(out_dir / "stage2_minimax_step_log.csv"),
        "stall_classification_csv": str(out_dir / "stall_classification.csv"),
        "pred_raw_npz": str(pred_path),
        "decoder_state_pt": str(state_path),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
        "doc_md": str(args.doc_path),
    }
    stage1_summary = _summarize_window_pass(result["stage1_window_rows"])
    stage2_summary = _summarize_window_pass(result["stage2_window_rows"])
    final_guard_verdict = final_guard.get("verdict", {}) or {}
    final_identity = final_guard.get("guard_path_identity", {}) or {}
    verdict = {
        "e1_gt_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in gt_window)),
        "e1_decoder_pass_count": int(sum(bool(r.get("accepted_p99_pass", False)) for r in dec_window)),
        "class_bc_relabel_count": int(len(relabel_rows)),
        "heading_tolerance_rad": float(args.heading_tolerance_rad),
        "heading_tolerance_in_seam": bool(heading_summary.get("selected_tolerance_in_seam", False)),
        "stage1_relabel_pass_count": int(stage1_summary["accepted_p99_pass_count"]),
        "stage2_skipped": bool(result.get("stage2_skipped", False)),
        "stage2_pass_count": int(stage2_summary["accepted_p99_pass_count"]),
        "stage2_p95_clean_count": int(stage2_summary["p95_shadow_clean_pass_count"]),
        "shortcut_negative_controls_still_fail": bool(final_guard_verdict.get("shortcut_negative_controls_still_fail", False)),
        "command_demotion_negative_controls_still_fail": bool(
            final_guard_verdict.get("command_demotion_negative_controls_still_fail", False)
        ),
        "recomputed_command_demotion_sequence_pass_count": int(
            sum(bool(r.get("demoted_acceptance_pass", False)) for r in command_demoted_rows)
        ),
        "guard_identity_pass": bool(final_guard_verdict.get("guard_path_identity_pass", False)),
        "guard_identity_max_abs_seq_delta": float(final_identity.get("max_abs_seq_delta", 0.0) or 0.0),
        "selected_guard_identity_max_abs_seq_delta": float(guard_identity_selected.get("max_abs_seq_delta", 0.0) or 0.0),
        "debug_only": True,
        "modified_production_runtime_trainer_gate": False,
        "modified_checkpoint": False,
    }
    payload: Dict[str, Any] = {
        "task": "action_handoff_inbetween_8window_preflight_bandfix",
        "scope": "debug-only fixed-oracle-schedule band audit and Stage2 rerun",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "checkpoint": str(args.checkpoint),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stage1_supervised_epochs": int(args.stage1_supervised_epochs),
            "stage2_tail_epochs": int(args.stage2_tail_epochs),
            "device": "cpu",
            "dtype": "float32",
            "heading_tolerance_rad": float(args.heading_tolerance_rad),
            "source_stage1_dir": str(args.source_stage1_dir),
        },
        "input_output_contract": {
            "decoder_input": {"shape": [8, result["input_dim"]], "dtype": "float32", "device": "cpu"},
            "middle_state_output": {"shape": [8, int(args.horizon), fit.STATE_DIM], "dtype": "float32", "device": "cpu"},
            "bone_angvel_aux_output": {"shape": [8, int(args.horizon), fit.ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
            "saved_pred_raw": {"shape": [8, int(args.horizon) * (fit.STATE_DIM + fit.ANGVEL_DIM)], "dtype": "float32", "device": "cpu numpy"},
        },
        "selected_windows": selected_windows,
        "e1": {
            "gt_window_summary_before": _summarize_window_pass(gt_window),
            "decoder_window_summary_before": _summarize_window_pass(dec_window),
            "stage1_window_summary_before": _summarize_window_pass(stage1_window_before),
            "gt_window_summary_after": _summarize_window_pass(gt_window_after),
            "decoder_window_summary_after": _summarize_window_pass(dec_window_after),
            "stage1_window_summary_after": _summarize_window_pass(stage1_window_after),
            "bucket_rows": e1_rows,
            "bucket_summary": e1_summary,
        },
        "class_bc_relabels": relabel_rows,
        "heading_seam_summary": heading_summary,
        "final_guard": final_guard,
        "selected_guard_identity": guard_identity_selected,
        "stage1": {
            "window_summary": stage1_summary,
            "loss_metrics": result["stage1_loss_metrics"],
            "best_supervised_epoch": int(result.get("best_supervised_epoch", -1)),
            "best_supervised_flat_standardized_mse": float(result.get("best_supervised_flat_standardized_mse", 0.0)),
        },
        "stage2": {
            "window_summary": stage2_summary,
            "loss_metrics": result["stage2_loss_metrics"],
            "skipped": bool(result.get("stage2_skipped", False)),
            "skip_reason": str(result.get("stage2_skip_reason", "")),
            "step_log_row_count": int(len(result["stage2_step_log"])),
        },
        "stall_classification": stall_rows,
        "negative_control_recomputed_command_demotion_sequences": command_demoted_rows,
        "verdict": verdict,
        "artifacts": artifacts,
    }

    preflight_rows = (
        gt_metric
        + dec_metric
        + stage1_metric_before
        + gt_metric_after
        + dec_metric_after
        + stage1_metric_after
        + e1_rows
    )
    _write_csv(Path(artifacts["gt_decoder_fullfamily_preflight_csv"]), preflight_rows)
    _dump_json(Path(artifacts["gt_decoder_fullfamily_preflight_json"]), payload["e1"])
    _write_csv(Path(artifacts["band_relabel_classBC_csv"]), relabel_rows)
    _write_csv(Path(artifacts["heading_seam_audit_csv"]), heading_rows)
    _dump_json(Path(artifacts["heading_seam_audit_json"]), heading_summary)
    _write_csv(Path(artifacts["negative_control_guard_csv"]), command_demoted_rows)
    _write_csv(Path(artifacts["per_window_csv"]), per_window_rows)
    _write_csv(Path(artifacts["per_metric_csv"]), per_metric_rows)
    _write_csv(Path(artifacts["stage2_minimax_step_log_csv"]), result["stage2_step_log"])
    _write_csv(Path(artifacts["stall_classification_csv"]), stall_rows)
    _dump_json(Path(artifacts["summary_json"]), payload)
    _write_summary_md(Path(artifacts["summary_md"]), payload)
    _write_doc_md(Path(args.doc_path), payload)
    return payload


def main() -> None:
    payload = run(parse_args())
    print(f"[OK] wrote {payload['artifacts']['summary_md']}")
    print(json.dumps(_jsonify(payload["verdict"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
