#!/usr/bin/env python3
"""Layer-2 action-handoff debug harness.

Consolidated read-only harness for oracle-schedule Layer-2 audits. It does not
train a decoder, does not forward production Trainer/runtime/gate, does not
mutate checkpoints, and does not attach any production path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import CONTEXT_LEN_C, EGO_VEL_SLICE, STATE_DIM  # noqa: E402
from tools.run_action_handoff_command_demotion_replay import _calibrate_command_bands  # noqa: E402
from tools.run_action_handoff_lifted_contract_exactness_repair import (  # noqa: E402
    CORE_ACCEPTANCE_KEYS,
    FLOAT32_FOOT_SLIP_ABS_EPS_MPS,
    FLOAT32_FOOT_SLIP_REL_EPS,
    _evaluate_variant_seq,
    _jsonify,
    _seq_from_components,
    _state_from_world_root_vel,
    _summarize_variant,
    _variant_sequences,
    _world_fd_central_endpoint,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _dump_json,
    _dump_md,
    _fmt,
    _load_clips,
    _load_skeleton_meta,
    _safe_percentile,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    DecoderItem,
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _reconstructed_gt_seq,
)
from tools.run_action_handoff_signal_representation_audit import (  # noqa: E402
    _anchor_root_path,
    _rng_for,
    _state_and_seq_from_state,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import DEFAULT_HORIZON  # noqa: E402
from tools.run_action_handoff_support_schedule_predictive_baseline import MATCHED_TARGETS  # noqa: E402


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_layer2_harness_20260603")
DEFAULT_TARGET_STATE_MSE = (1e-6,)
DEFAULT_TARGET_ROOT_P95_M = (1e-3,)
DEFAULT_ARMS = ("flat_velocity_state281", "endpoint_consistent_fd_native", "endpoint_consistent_fd_roundtrip")
DEFAULT_MODES = ("data_line", "fair_perturbation")
DEFAULT_EQUALIZATION_MODES = ("state_mse", "root_path_p95")
EPS = 1e-12


def _finite_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _correlated_noise(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    *,
    kind: str,
    rho: float,
) -> np.ndarray:
    h, d = int(shape[0]), int(shape[1])
    if kind == "bias":
        return np.repeat(rng.normal(0.0, 1.0, size=(1, d)), h, axis=0).astype(np.float32)
    if kind != "correlated":
        raise ValueError(f"unsupported noise kind {kind!r}")
    rho = float(np.clip(rho, 0.0, 0.999))
    out = np.zeros((h, d), dtype=np.float32)
    out[0] = rng.normal(0.0, 1.0, size=(d,)).astype(np.float32)
    innovation_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
    for t in range(1, h):
        out[t] = rho * out[t - 1] + innovation_scale * rng.normal(0.0, 1.0, size=(d,)).astype(np.float32)
    return out.astype(np.float32, copy=False)


def _state_mse(state: np.ndarray, true_state: np.ndarray) -> float:
    delta = np.asarray(state, dtype=np.float64) - np.asarray(true_state, dtype=np.float64)
    return float(np.mean(delta * delta)) if delta.size else 0.0


def _calibrate_scale_to_state_mse(
    build: Callable[[float], Tuple[np.ndarray, Dict[str, np.ndarray]]],
    true_state: np.ndarray,
    *,
    target_mse: float,
    max_iter: int,
) -> Tuple[float, np.ndarray, Dict[str, np.ndarray], float, str]:
    target = float(target_mse)
    state0, seq0 = build(0.0)
    mse0 = _state_mse(state0, true_state)
    if target <= 0.0 or mse0 >= target:
        return 0.0, state0, seq0, mse0, "zero_or_baseline_exceeds_target"

    lo = 0.0
    hi = 1.0
    state_hi, seq_hi = build(hi)
    mse_hi = _state_mse(state_hi, true_state)
    grow = 0
    while mse_hi < target and grow < 24:
        lo = hi
        hi *= 2.0
        state_hi, seq_hi = build(hi)
        mse_hi = _state_mse(state_hi, true_state)
        grow += 1
    if mse_hi < target:
        return 0.0, state0, seq0, mse0, "target_unreachable"

    best_scale = hi
    best_state = state_hi
    best_seq = seq_hi
    best_mse = mse_hi
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        state_mid, seq_mid = build(mid)
        mse_mid = _state_mse(state_mid, true_state)
        best_scale, best_state, best_seq, best_mse = mid, state_mid, seq_mid, mse_mid
        if mse_mid < target:
            lo = mid
        else:
            hi = mid
    return best_scale, best_state, best_seq, best_mse, "ok"


def _root_path_error_p95(seq: Mapping[str, np.ndarray], baseline_seq: Mapping[str, np.ndarray]) -> float:
    err = np.linalg.norm(
        np.asarray(seq["root_pos"], dtype=np.float64) - np.asarray(baseline_seq["root_pos"], dtype=np.float64),
        axis=1,
    )
    return _safe_percentile(err, 95.0)


def _calibrate_scale_to_metric(
    build: Callable[[float], Tuple[np.ndarray, Dict[str, np.ndarray]]],
    metric: Callable[[np.ndarray, Dict[str, np.ndarray]], float],
    *,
    target_value: float,
    max_iter: int,
) -> Tuple[float, np.ndarray, Dict[str, np.ndarray], float, str]:
    target = float(target_value)
    state0, seq0 = build(0.0)
    val0 = float(metric(state0, seq0))
    if target <= 0.0 or val0 >= target:
        return 0.0, state0, seq0, val0, "zero_or_baseline_exceeds_target"

    lo = 0.0
    hi = 1.0
    state_hi, seq_hi = build(hi)
    val_hi = float(metric(state_hi, seq_hi))
    grow = 0
    while val_hi < target and grow < 24:
        lo = hi
        hi *= 2.0
        state_hi, seq_hi = build(hi)
        val_hi = float(metric(state_hi, seq_hi))
        grow += 1
    if val_hi < target:
        return 0.0, state0, seq0, val0, "target_unreachable"

    best_scale = hi
    best_state = state_hi
    best_seq = seq_hi
    best_val = val_hi
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        state_mid, seq_mid = build(mid)
        val_mid = float(metric(state_mid, seq_mid))
        best_scale, best_state, best_seq, best_val = mid, state_mid, seq_mid, val_mid
        if val_mid < target:
            lo = mid
        else:
            hi = mid
    return best_scale, best_state, best_seq, best_val, "ok"


def _support_signature(item: DecoderItem) -> str:
    labels = [str(x) for x in item.support_contract.get("normalized_label_sequence", [])]
    if not labels:
        return "empty"
    runs: List[str] = []
    start = 0
    for idx in range(1, len(labels)):
        if labels[idx] != labels[start]:
            runs.append(f"{labels[start]}:{idx - start}")
            start = idx
    runs.append(f"{labels[start]}:{len(labels) - start}")
    return ">".join(runs)


def _data_line_summary(items: Sequence[DecoderItem]) -> Dict[str, Any]:
    signatures = Counter(_support_signature(item) for item in items)
    switch_count = int(sum(int(item.support_contract.get("normalized_transition_count", 0)) >= 1 for item in items))
    return {
        "matched_window_count": int(len(items)),
        "matched_targets": list(MATCHED_TARGETS),
        "per_clip_windows": dict(Counter(item.clip for item in items)),
        "switch_window_count": switch_count,
        "support_signature_unique_count": int(len(signatures)),
        "support_signature_top10": dict(signatures.most_common(10)),
        "data_line_status": "started_read_only_window_and_support_signature_line",
    }


def _flat_perturb_builder(
    *,
    item: DecoderItem,
    true_state: np.ndarray,
    aux: np.ndarray,
    noise: np.ndarray,
) -> Callable[[float], Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    def build(scale: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        state = np.asarray(true_state, dtype=np.float32).copy()
        state[:, EGO_VEL_SLICE] += float(scale) * np.asarray(noise, dtype=np.float32)
        return _state_and_seq_from_state(item, state, aux)

    return build


def _endpoint_perturb_builder(
    *,
    item: DecoderItem,
    skeleton: Any,
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    aux: np.ndarray,
    noise: np.ndarray,
    seed: int,
    item_i: int,
    roundtrip: bool,
) -> Callable[[float], Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    anchored_root = _anchor_root_path(
        item,
        skeleton,
        baseline_seq,
        keep_inter_anchor=True,
        noise_scale=0.0,
        rng=_rng_for(seed, item_i, "layer2_harness_endpoint_root"),
    )

    def build(scale: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        root = np.asarray(anchored_root, dtype=np.float32).copy()
        root[:, :2] += float(scale) * np.asarray(noise, dtype=np.float32)
        root_vel = _world_fd_central_endpoint(root)
        state = _state_from_world_root_vel(true_state, root_vel, np.asarray(baseline_seq["cond_dir"], dtype=np.float32))
        if roundtrip:
            return _state_and_seq_from_state(item, state, aux)
        seq = _seq_from_components(base_seq=baseline_seq, root_pos=root, root_vel=root_vel)
        return state, seq

    return build


def _exact_endpoint_reference(
    *,
    item_i: int,
    item: DecoderItem,
    skeleton: Any,
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    aux: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    return _variant_sequences(
        item_i=item_i,
        item=item,
        skeleton=skeleton,
        baseline_seq=baseline_seq,
        true_state=true_state,
        aux=aux,
        seed=seed,
    )["endpoint_consistent_fd"][:2]


def _sensitivity_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = _summarize_variant(rows)

    def arr(key: str) -> np.ndarray:
        return np.asarray([_finite_float(r.get(key)) for r in rows], dtype=np.float64)

    achieved = arr("achieved_state281_mse")
    achieved_equalization = arr("achieved_equalization_value")
    scale = arr("native_noise_scale")
    target = arr("target_state281_mse")
    target_equalization = arr("target_equalization_value")
    state_denom_source = np.where(target > 0.0, target, achieved)
    sqrt_target = np.sqrt(np.maximum(state_denom_source, EPS))
    equalization_denom = np.maximum(achieved_equalization, EPS)
    root = arr("root_path_error_p95_m")
    disp = arr("support_foot_world_displacement_p95_m")
    heading = arr("heading_error_p95_rad")
    valid = np.asarray([bool(r.get("calibration_valid", True)) for r in rows], dtype=np.float64)
    status_counts = Counter(str(r.get("calibration_status", "ok")) for r in rows)
    valid_rows = [r for r in rows if bool(r.get("calibration_valid", True))]

    def valid_rate(key: str) -> float:
        return float(np.mean([bool(r.get(key, False)) for r in valid_rows])) if valid_rows else 0.0

    def valid_mean(key: str) -> float:
        vals = np.asarray([_finite_float(r.get(key)) for r in valid_rows], dtype=np.float64)
        return float(np.mean(vals)) if vals.size else 0.0

    base.update(
        {
            "valid_n": int(len(valid_rows)),
            "calibration_valid_rate": float(np.mean(valid)) if valid.size else 0.0,
            "calibration_status_counts": dict(status_counts),
            "valid_demoted_acceptance_pass_rate": valid_rate("demoted_acceptance_pass"),
            "valid_float32_precision_tolerant_demoted_pass_rate": valid_rate(
                "float32_precision_tolerant_demoted_acceptance_pass"
            ),
            "valid_rate_budget_pass_rate": valid_rate("rate_budget"),
            "valid_support_side_core_pass_rate": valid_rate("support_side_core"),
            "valid_root_path_error_p95_m_mean": valid_mean("root_path_error_p95_m"),
            "valid_support_foot_world_displacement_p95_m_mean": valid_mean(
                "support_foot_world_displacement_p95_m"
            ),
            "valid_heading_error_p95_rad_mean": valid_mean("heading_error_p95_rad"),
            "target_state281_mse_mean": float(np.mean(target)) if target.size else 0.0,
            "achieved_state281_mse_mean": float(np.mean(achieved)) if achieved.size else 0.0,
            "achieved_state281_mse_p95": _safe_percentile(achieved, 95.0),
            "target_equalization_value_mean": float(np.mean(target_equalization)) if target_equalization.size else 0.0,
            "achieved_equalization_value_mean": float(np.mean(achieved_equalization)) if achieved_equalization.size else 0.0,
            "achieved_equalization_value_p95": _safe_percentile(achieved_equalization, 95.0),
            "native_noise_scale_mean": float(np.mean(scale)) if scale.size else 0.0,
            "native_noise_scale_p95": _safe_percentile(scale, 95.0),
            "root_path_error_per_sqrt_state_mse_mean": float(np.mean(root / sqrt_target)) if root.size else 0.0,
            "support_disp_per_sqrt_state_mse_mean": float(np.mean(disp / sqrt_target)) if disp.size else 0.0,
            "heading_error_per_sqrt_state_mse_mean": float(np.mean(heading / sqrt_target)) if heading.size else 0.0,
            "root_path_error_per_equalization_value_mean": float(np.mean(root / equalization_denom)) if root.size else 0.0,
            "support_disp_per_equalization_value_mean": float(np.mean(disp / equalization_denom)) if disp.size else 0.0,
            "heading_error_per_equalization_value_mean": float(np.mean(heading / equalization_denom)) if heading.size else 0.0,
            "gradient_diagnostic": "finite_difference_native_noise_to_acceptance_metric_sensitivity_proxy",
        }
    )
    return base


def _noise_configs(noise_kinds: Sequence[str], noise_rhos: Sequence[float]) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    for kind in noise_kinds:
        kind = str(kind)
        if kind == "bias":
            out.append((kind, None))
            continue
        for rho in noise_rhos:
            out.append((kind, float(rho)))
    return out


def _targets_for_equalization(args: argparse.Namespace, mode: str) -> Sequence[float]:
    if mode == "state_mse":
        return [float(x) for x in args.target_state_mse]
    if mode == "root_path_p95":
        return [float(x) for x in args.target_root_p95_m]
    raise ValueError(f"unsupported equalization mode {mode!r}")


def _run_fair_perturbation(
    *,
    args: argparse.Namespace,
    items: Sequence[DecoderItem],
    skeleton: Any,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    command_bands: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    selected_items = list(items)
    if int(args.max_windows) > 0:
        selected_items = selected_items[: int(args.max_windows)]
    for item_i, item in enumerate(selected_items):
        baseline_seq = _reconstructed_gt_seq(item, oracle_contact_passthrough=True, command_align_root_vel=False)
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(int(args.horizon), STATE_DIM)
        aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(int(args.horizon), ANGVEL_DIM)
        for arm in args.arms:
            arm = str(arm)
            arm_alias = "endpoint_consistent_fd_native" if arm == "endpoint_consistent_fd" else arm
            for equalization_mode in args.equalization_modes:
                equalization_mode = str(equalization_mode)
                for target_value in _targets_for_equalization(args, equalization_mode):
                    for noise_kind, noise_rho in _noise_configs(args.noise_kinds, args.noise_rhos):
                        for trial in range(int(args.trials)):
                            rng = _rng_for(
                                args.seed,
                                "fair_perturbation",
                                item_i,
                                arm_alias,
                                equalization_mode,
                                target_value,
                                trial,
                                noise_kind,
                                noise_rho,
                            )
                            noise = _correlated_noise(
                                rng,
                                (int(args.horizon), 2),
                                kind=str(noise_kind),
                                rho=0.0 if noise_rho is None else float(noise_rho),
                            )
                            if arm_alias == "flat_velocity_state281":
                                build = _flat_perturb_builder(item=item, true_state=true_state, aux=aux, noise=noise)
                            elif arm_alias in {"endpoint_consistent_fd_native", "endpoint_consistent_fd_roundtrip"}:
                                build = _endpoint_perturb_builder(
                                    item=item,
                                    skeleton=skeleton,
                                    baseline_seq=baseline_seq,
                                    true_state=true_state,
                                    aux=aux,
                                    noise=noise,
                                    seed=int(args.seed),
                                    item_i=item_i,
                                    roundtrip=arm_alias == "endpoint_consistent_fd_roundtrip",
                                )
                            else:
                                raise ValueError(f"unsupported arm {arm!r}")

                            if equalization_mode == "state_mse":
                                scale, state, seq, achieved_equalization, calibration_status = _calibrate_scale_to_state_mse(
                                    build,
                                    true_state,
                                    target_mse=float(target_value),
                                    max_iter=int(args.scale_calibration_iters),
                                )
                            elif equalization_mode == "root_path_p95":
                                scale, state, seq, achieved_equalization, calibration_status = _calibrate_scale_to_metric(
                                    build,
                                    lambda _state, built_seq: _root_path_error_p95(built_seq, baseline_seq),
                                    target_value=float(target_value),
                                    max_iter=int(args.scale_calibration_iters),
                                )
                            else:
                                raise ValueError(f"unsupported equalization mode {equalization_mode!r}")
                            if float(scale) > float(args.max_native_noise_scale):
                                state, seq = build(0.0)
                                achieved_equalization = (
                                    _state_mse(state, true_state)
                                    if equalization_mode == "state_mse"
                                    else _root_path_error_p95(seq, baseline_seq)
                                )
                                scale = 0.0
                                calibration_status = "scale_exceeds_cap"
                            achieved_state_mse = _state_mse(state, true_state)

                            row = _evaluate_variant_seq(
                                variant=arm_alias,
                                acceptance_keys=CORE_ACCEPTANCE_KEYS,
                                item=item,
                                state=state,
                                seq=seq,
                                baseline_seq=baseline_seq,
                                true_state=true_state,
                                baseline_bands=baseline_bands,
                                support_bands=support_bands,
                                command_bands=command_bands,
                                skeleton=skeleton,
                                min_run_frames=int(args.min_run_frames),
                                note=(
                                    "Layer-2 harness fair perturbation: native-space "
                                    f"{noise_kind} noise calibrated by {equalization_mode}"
                                ),
                            )
                            row.update(
                                {
                                    "harness_mode": "fair_perturbation",
                                    "arm": arm_alias,
                                    "target_state281_mse": float(target_value)
                                    if equalization_mode == "state_mse"
                                    else None,
                                    "target_root_path_error_p95_m": float(target_value)
                                    if equalization_mode == "root_path_p95"
                                    else None,
                                    "equalization_mode": equalization_mode,
                                    "target_equalization_value": float(target_value),
                                    "achieved_equalization_value": float(achieved_equalization),
                                    "achieved_state281_mse": float(achieved_state_mse),
                                    "native_noise_scale": float(scale),
                                    "calibration_status": calibration_status,
                                    "calibration_valid": bool(calibration_status == "ok"),
                                    "noise_kind": str(noise_kind),
                                    "noise_rho": None if noise_rho is None else float(noise_rho),
                                    "trial": int(trial),
                                }
                            )
                            rows.append(row)
    summary: Dict[str, Any] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rho = row.get("noise_rho")
        rho_key = "bias" if rho is None else f"rho={float(rho):.2g}"
        target_key = (
            f"mse={float(row['target_equalization_value']):.0e}"
            if row.get("equalization_mode") == "state_mse"
            else f"rootp95={float(row['target_equalization_value']):.0e}m"
        )
        key = f"{row['arm']}|eq={row['equalization_mode']}|{target_key}|noise={row['noise_kind']}|{rho_key}"
        grouped[key].append(row)
    for key, group_rows in sorted(grouped.items()):
        summary[key] = _sensitivity_summary(group_rows)
    return summary, rows


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "harness_mode",
        "arm",
        "variant",
        "clip",
        "start",
        "end",
        "target_state281_mse",
        "target_root_path_error_p95_m",
        "equalization_mode",
        "target_equalization_value",
        "achieved_equalization_value",
        "achieved_state281_mse",
        "native_noise_scale",
        "calibration_status",
        "calibration_valid",
        "noise_kind",
        "noise_rho",
        "trial",
        "demoted_acceptance_pass",
        "float32_precision_tolerant_demoted_acceptance_pass",
        "demoted_failed_family",
        "float32_precision_tolerant_failed_family",
        "rate_budget",
        "support_honesty",
        "float32_precision_tolerant_support_honesty",
        "support_side_core",
        "command_compatibility",
        "pose_continuity",
        "endpoint_bridgeability",
        "root_path_error_p95_m",
        "support_foot_world_displacement_p95_m",
        "foot_slip_p95_to_band_ratio",
        "heading_error_p95_rad",
        "max_abs_state_delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Layer-2 Action-Handoff Harness")
    lines.append("")
    lines.append("Date: 2026-06-03")
    lines.append("")
    lines.append(
        "Debug-only consolidated harness. No training, no production Trainer/runtime/gate forward, "
        "no checkpoint mutation, and no decoder toy smoke."
    )
    lines.append("")
    lines.append("## Contract")
    lines.append("")
    lines.append("- committed lifted reconstruction contract: `endpoint_consistent_fd`")
    lines.append("- `copied_gt_root_vel` is excluded as an oracle-only upper bound.")
    lines.append(
        f"- float32 support-slip tolerance: abs `{FLOAT32_FOOT_SLIP_ABS_EPS_MPS:g} m/s`, "
        f"rel `{FLOAT32_FOOT_SLIP_REL_EPS:g}`."
    )
    lines.append("")
    lines.append("## Data Line")
    dl = payload.get("data_line", {})
    if dl:
        lines.append(f"- matched windows: `{dl.get('matched_window_count')}`")
        lines.append(f"- switch windows: `{dl.get('switch_window_count')}`")
        lines.append(f"- per clip: `{dl.get('per_clip_windows')}`")
        lines.append(f"- unique support signatures: `{dl.get('support_signature_unique_count')}`")
    else:
        lines.append("- not run")
    lines.append("")
    lines.append("## Fair Perturbation")
    fp = payload.get("fair_perturbation", {})
    if fp:
        lines.append("| arm/noise | n/valid | cal valid | valid pass | achieved eq | achieved state mse | native scale | root p95 mean | support disp mean | heading mean | failed families |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for key, rec in fp.items():
            lines.append(
                f"| {key} | {int(rec.get('n', 0))}/{int(rec.get('valid_n', 0))} | "
                f"{_fmt(rec.get('calibration_valid_rate'))} | "
                f"{_fmt(rec.get('valid_float32_precision_tolerant_demoted_pass_rate'))} | "
                f"{_fmt(rec.get('achieved_equalization_value_mean'), 8)} | "
                f"{_fmt(rec.get('achieved_state281_mse_mean'), 8)} | "
                f"{_fmt(rec.get('native_noise_scale_mean'), 8)} | "
                f"{_fmt(rec.get('valid_root_path_error_p95_m_mean'), 8)} | "
                f"{_fmt(rec.get('valid_support_foot_world_displacement_p95_m_mean'), 8)} | "
                f"{_fmt(rec.get('valid_heading_error_p95_rad_mean'), 8)} | "
                f"{rec.get('float32_precision_tolerant_failed_family_counts', rec.get('failed_family_counts', {}))} |"
            )
    else:
        lines.append("- not run")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- summary json: `{payload['artifacts']['summary_json']}`")
    lines.append(f"- rows csv: `{payload['artifacts']['rows_csv']}`")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root))
    all_items = _build_items(
        clips,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
        min_run_frames=int(args.min_run_frames),
        stride=int(args.stride),
    )
    main_items = [item for item in all_items if item.clip in MATCHED_TARGETS]
    baseline_bands = _calibrate_reconstructed_baseline_bands(
        all_items,
        skeleton,
        quantile=float(args.reconstructed_baseline_quantile),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    support_bands = _calibrate_reconstructed_support_side_bands(
        all_items,
        skeleton,
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    command_bands = _calibrate_command_bands(
        clips,
        horizon=int(args.horizon),
        quantile=float(args.command_quantile),
    )

    modes = {str(x) for x in args.modes}
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "task": "layer2_action_handoff_harness",
        "scope": (
            "debug-only consolidated Layer-2 harness; no training, no production Trainer/runtime/gate "
            "forward or edit, no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "out_dir": str(args.out_dir),
            "modes": sorted(modes),
            "arms": list(args.arms),
            "equalization_modes": list(args.equalization_modes),
            "target_state281_mse": [float(x) for x in args.target_state_mse],
            "target_root_path_error_p95_m": [float(x) for x in args.target_root_p95_m],
            "trials": int(args.trials),
            "noise_kinds": list(args.noise_kinds),
            "noise_rhos": [float(x) for x in args.noise_rhos],
            "max_windows": int(args.max_windows),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "command_quantile": float(args.command_quantile),
            "scale_calibration_iters": int(args.scale_calibration_iters),
            "max_native_noise_scale": float(args.max_native_noise_scale),
            "seed": int(args.seed),
            "dtype": "float32",
            "device": "cpu",
        },
        "committed_contract": {
            "lifted_reconstruction": "endpoint_consistent_fd",
            "oracle_upper_bound_excluded": "copied_gt_root_vel",
            "yaw_cond_dir_role": "commanded cue only",
        },
        "input_output_contract": {
            "state281": {"shape": [int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "root_vel": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "bone_angvel": {"shape": [int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
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
            "decoder_toy_smoke": False,
        },
    }
    if "data_line" in modes:
        payload["data_line"] = _data_line_summary(main_items)
    if "fair_perturbation" in modes:
        fair_summary, fair_rows = _run_fair_perturbation(
            args=args,
            items=main_items,
            skeleton=skeleton,
            baseline_bands=baseline_bands,
            support_bands=support_bands,
            command_bands=command_bands,
        )
        payload["fair_perturbation"] = fair_summary
        rows.extend(fair_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "layer2_harness_summary.json"
    rows_csv = args.out_dir / "layer2_harness_rows.csv"
    summary_md = args.out_dir / "layer2_harness_summary.md"
    payload["artifacts"] = {
        "summary_json": str(summary_json),
        "rows_csv": str(rows_csv),
        "summary_md": str(summary_md),
    }
    payload["rows"] = rows
    _dump_json(summary_json, payload)
    _write_rows_csv(rows_csv, rows)
    _write_summary_md(summary_md, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=("data_line", "fair_perturbation"))
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_ARMS),
        choices=(
            "flat_velocity_state281",
            "endpoint_consistent_fd",
            "endpoint_consistent_fd_native",
            "endpoint_consistent_fd_roundtrip",
        ),
    )
    p.add_argument(
        "--equalization-modes",
        nargs="+",
        default=list(DEFAULT_EQUALIZATION_MODES),
        choices=("state_mse", "root_path_p95"),
    )
    p.add_argument("--target-state-mse", type=float, nargs="+", default=list(DEFAULT_TARGET_STATE_MSE))
    p.add_argument("--target-root-p95-m", type=float, nargs="+", default=list(DEFAULT_TARGET_ROOT_P95_M))
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--noise-kinds", nargs="+", choices=("correlated", "bias"), default=["correlated"])
    p.add_argument("--noise-rhos", type=float, nargs="+", default=[0.9])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--command-quantile", type=float, default=100.0)
    p.add_argument("--scale-calibration-iters", type=int, default=24)
    p.add_argument("--max-native-noise-scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260603)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"wrote {payload['artifacts']['summary_md']}")
    print(f"wrote {payload['artifacts']['summary_json']}")
    print(f"wrote {payload['artifacts']['rows_csv']}")
    print(json.dumps(_jsonify({k: payload.get(k) for k in ("data_line", "fair_perturbation")}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
