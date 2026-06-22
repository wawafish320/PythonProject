#!/usr/bin/env python3
"""Signal expressibility and perturbation-sensitivity audit.

Debug-only GT/read-only probe for action-handoff inbetween representation
questions. This script does not train a generator, does not forward production
Trainer/runtime/gate, does not mutate checkpoints, and does not attach any
runtime path. It reuses the existing reconstructed-domain acceptance helpers
from the action-handoff debug probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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
    YAW_RATE_SLICE,
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
    _evaluate_seq_common,
    _foot_positions,
    _reconstructed_gt_seq,
    _seq_from_prediction,
    _summarize_rows,
)
from tools.run_action_handoff_support_contract_tightening_probe import (  # noqa: E402
    _label_has_side,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)
from tools.run_action_handoff_support_schedule_predictive_baseline import (  # noqa: E402
    MATCHED_TARGETS,
    UNMATCHED_TARGET,
)


DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_signal_representation_audit_20260603")
NOISE_LEVELS = (1e-4, 1e-3, 1e-2)
EPS = 1e-8


@dataclass(frozen=True)
class EvalRecord:
    row: Dict[str, Any]
    root_path_error_p95_m: float
    root_path_error_max_m: float
    support_foot_world_displacement_p95_m: float
    support_foot_world_displacement_max_m: float
    max_abs_state_delta: float


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


def _rng_for(seed: int, *parts: Any) -> np.random.Generator:
    h = int(seed) & 0xFFFFFFFF
    for part in parts:
        for ch in str(part):
            h = ((h * 131) + ord(ch)) & 0xFFFFFFFF
    return np.random.default_rng(h)


def _finite_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _ego_from_world_vel(world_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    vel = np.asarray(world_vel, dtype=np.float32).reshape(-1, 2)
    cmd = np.asarray(cond_dir, dtype=np.float32).reshape(-1, 2)
    n = min(int(vel.shape[0]), int(cmd.shape[0]))
    vel = vel[:n]
    cmd = cmd[:n]
    norm = np.maximum(np.linalg.norm(cmd, axis=1, keepdims=True), EPS)
    fwd = cmd / norm
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    ego_fwd = np.sum(vel * fwd, axis=1)
    ego_lat = np.sum(vel * lat, axis=1)
    return np.stack([ego_fwd, ego_lat], axis=1).astype(np.float32)


def _state_from_root_path(base_seq: Mapping[str, np.ndarray], root_pos: np.ndarray) -> np.ndarray:
    root = np.asarray(root_pos, dtype=np.float32).reshape(-1, 3)
    h = int(root.shape[0])
    world_vel = np.zeros((h, 2), dtype=np.float32)
    if h > 1:
        world_vel[:-1] = (root[1:, :2] - root[:-1, :2]) * float(FPS)
        world_vel[-1] = world_vel[-2]
    ego = _ego_from_world_vel(world_vel, np.asarray(base_seq["cond_dir"], dtype=np.float32))
    state = np.concatenate(
        [
            np.asarray(base_seq["rot6d"], dtype=np.float32).reshape(h, POSE_SLICE.stop - POSE_SLICE.start),
            ego.reshape(h, 2),
            np.asarray(base_seq["yaw_rate"], dtype=np.float32).reshape(h, 1),
            np.asarray(base_seq["contact"], dtype=np.float32).reshape(h, 2),
        ],
        axis=1,
    )
    if state.shape != (h, STATE_DIM):
        raise RuntimeError(f"reconstructed state shape mismatch: {state.shape} != {(h, STATE_DIM)}")
    return state.astype(np.float32, copy=False)


def _state_and_seq_from_state(
    item: DecoderItem,
    state: np.ndarray,
    aux: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    state = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    aux = np.asarray(aux, dtype=np.float32).reshape(state.shape[0], ANGVEL_DIM)
    seq = _seq_from_prediction(
        item,
        state,
        aux,
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )
    return state, seq


def _candidate_side_labels(item: DecoderItem) -> List[str]:
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    contact = np.asarray(item.seq["contact"], dtype=np.float32).reshape(len(labels), 2)
    sides: List[str] = []
    prev: Optional[str] = None
    for i, label in enumerate(labels):
        if label == "right":
            side = "right"
        elif label == "left":
            side = "left"
        elif label == "dual":
            side = "right" if float(contact[i, 0]) >= float(contact[i, 1]) else "left"
        elif prev in {"right", "left"}:
            side = prev
        else:
            side = "right" if float(contact[i, 0]) >= float(contact[i, 1]) else "left"
        sides.append(side)
        prev = side
    return sides


def _runs_from_sides(sides: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not sides:
        return []
    runs: List[Tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(sides)):
        if str(sides[i]) != str(sides[start]):
            runs.append((start, i, str(sides[start])))
            start = i
    runs.append((start, len(sides), str(sides[start])))
    return runs


def _anchor_root_path(
    item: DecoderItem,
    skeleton: Any,
    base_seq: Mapping[str, np.ndarray],
    *,
    keep_inter_anchor: bool,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h = int(np.asarray(base_seq["rot6d"]).shape[0])
    sides = _candidate_side_labels(item)
    runs = _runs_from_sides(sides)
    if not runs:
        return np.asarray(base_seq["root_pos"], dtype=np.float32).reshape(h, 3).copy()

    world_foot = _foot_positions(
        np.asarray(base_seq["rot6d"], dtype=np.float32),
        np.asarray(base_seq["root_pos"], dtype=np.float32),
        skeleton,
    )
    if world_foot is None:
        return np.asarray(base_seq["root_pos"], dtype=np.float32).reshape(h, 3).copy()

    base_root = np.asarray(base_seq["root_pos"], dtype=np.float32).reshape(h, 3)
    original_anchors: List[np.ndarray] = []
    for start, _end, side in runs:
        anchor = np.asarray(world_foot[side][start], dtype=np.float32).reshape(3).copy()
        original_anchors.append(anchor)

    eval_anchors = [x.copy() for x in original_anchors]
    if not keep_inter_anchor and eval_anchors:
        eval_anchors = [eval_anchors[0].copy() for _ in eval_anchors]

    root = np.zeros((h, 3), dtype=np.float32)
    for run_i, (start, end, side) in enumerate(runs):
        del side
        original_anchor = original_anchors[run_i]
        eval_anchor = eval_anchors[run_i]
        root_rel = base_root[start:end] - original_anchor.reshape(1, 3)
        if noise_scale > 0.0:
            root_rel = root_rel.copy()
            root_rel[:, :2] += rng.normal(0.0, float(noise_scale), size=root_rel[:, :2].shape).astype(np.float32)
        root[start:end] = eval_anchor.reshape(1, 3) + root_rel
    return root.astype(np.float32, copy=False)


def _support_foot_world_displacement(
    seq: Mapping[str, np.ndarray],
    baseline_seq: Mapping[str, np.ndarray],
    labels: Sequence[str],
    skeleton: Any,
) -> Dict[str, float]:
    foot = _foot_positions(
        np.asarray(seq["rot6d"], dtype=np.float32),
        np.asarray(seq["root_pos"], dtype=np.float32),
        skeleton,
    )
    base = _foot_positions(
        np.asarray(baseline_seq["rot6d"], dtype=np.float32),
        np.asarray(baseline_seq["root_pos"], dtype=np.float32),
        skeleton,
    )
    if foot is None or base is None:
        return {"count": 0.0, "mean_m": 0.0, "p95_m": 0.0, "max_m": 0.0}
    vals: List[float] = []
    for i, label in enumerate(labels):
        for side in ("right", "left"):
            if not _label_has_side(str(label), side):
                continue
            if side not in foot or side not in base:
                continue
            vals.append(float(np.linalg.norm(np.asarray(foot[side][i]) - np.asarray(base[side][i]))))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0.0, "mean_m": 0.0, "p95_m": 0.0, "max_m": 0.0}
    return {
        "count": float(arr.size),
        "mean_m": float(np.mean(arr)),
        "p95_m": _safe_percentile(arr, 95.0),
        "max_m": float(np.max(arr)),
    }


def _evaluate_candidate_state(
    *,
    representation: str,
    section: str,
    noise_mse: Optional[float],
    noise_trial: Optional[int],
    item: DecoderItem,
    state: np.ndarray,
    aux: np.ndarray,
    baseline_seq: Mapping[str, np.ndarray],
    true_state: np.ndarray,
    baseline_bands: Mapping[str, Mapping[str, Any]],
    support_bands: Mapping[str, Mapping[str, Any]],
    skeleton: Any,
    min_run_frames: int,
) -> EvalRecord:
    state, seq = _state_and_seq_from_state(item, state, aux)
    row = _evaluate_seq_common(
        variant=representation,
        split="signal_representation_audit",
        split_kind="gt_read_only",
        partition=section,
        item=item,
        seq=seq,
        baseline_bands=baseline_bands,
        support_bands=support_bands,
        skeleton=skeleton,
        min_run_frames=min_run_frames,
        endpoint_note="reconstructed-domain acceptance path; oracle contact schedule retained",
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
        calibration_domain="reconstructed_state281",
    )
    labels = [str(x) for x in item.support_contract["normalized_label_sequence"]]
    root_err = np.linalg.norm(
        np.asarray(seq["root_pos"], dtype=np.float64) - np.asarray(baseline_seq["root_pos"], dtype=np.float64),
        axis=1,
    )
    disp = _support_foot_world_displacement(seq, baseline_seq, labels, skeleton)
    row["representation"] = representation
    row["audit_section"] = section
    row["noise_mse"] = noise_mse
    row["noise_trial"] = noise_trial
    row["switch_window"] = bool(int(item.support_contract.get("normalized_transition_count", 0)) >= 1)
    row["root_path_error_p95_m"] = _safe_percentile(root_err, 95.0)
    row["root_path_error_max_m"] = float(np.max(root_err)) if root_err.size else 0.0
    row["support_foot_world_displacement_p95_m"] = float(disp["p95_m"])
    row["support_foot_world_displacement_max_m"] = float(disp["max_m"])
    row["support_foot_world_displacement_count"] = int(disp["count"])
    row["max_abs_state_delta"] = float(
        np.max(np.abs(np.asarray(state, dtype=np.float64) - np.asarray(true_state, dtype=np.float64)))
    )
    return EvalRecord(
        row=row,
        root_path_error_p95_m=float(row["root_path_error_p95_m"]),
        root_path_error_max_m=float(row["root_path_error_max_m"]),
        support_foot_world_displacement_p95_m=float(row["support_foot_world_displacement_p95_m"]),
        support_foot_world_displacement_max_m=float(row["support_foot_world_displacement_max_m"]),
        max_abs_state_delta=float(row["max_abs_state_delta"]),
    )


def _flat_state(item: DecoderItem, *, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(-1, STATE_DIM).copy()
    if noise_scale > 0.0:
        state[:, EGO_VEL_SLICE] += rng.normal(
            0.0,
            float(noise_scale),
            size=state[:, EGO_VEL_SLICE].shape,
        ).astype(np.float32)
    return state


def _root_position_state(
    item: DecoderItem,
    base_seq: Mapping[str, np.ndarray],
    *,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    del item
    root = np.asarray(base_seq["root_pos"], dtype=np.float32).reshape(-1, 3).copy()
    if noise_scale > 0.0:
        root[:, :2] += rng.normal(0.0, float(noise_scale), size=root[:, :2].shape).astype(np.float32)
    return _state_from_root_path(base_seq, root)


def _anchored_state(
    item: DecoderItem,
    skeleton: Any,
    base_seq: Mapping[str, np.ndarray],
    *,
    keep_inter_anchor: bool,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    root = _anchor_root_path(
        item,
        skeleton,
        base_seq,
        keep_inter_anchor=keep_inter_anchor,
        noise_scale=noise_scale,
        rng=rng,
    )
    return _state_from_root_path(base_seq, root)


def _aggregate_records(records: Sequence[EvalRecord]) -> Dict[str, Any]:
    rows = [r.row for r in records]
    summary = _summarize_rows(rows)
    switch_rows = [r.row for r in records if bool(r.row.get("switch_window", False))]

    def vals(attr: str) -> np.ndarray:
        return np.asarray([float(getattr(r, attr)) for r in records], dtype=np.float64)

    def max_row(key: str) -> float:
        arr = np.asarray([_finite_float(r.row.get(key)) for r in records], dtype=np.float64)
        return float(np.max(arr)) if arr.size else 0.0

    root_p95 = vals("root_path_error_p95_m")
    disp_p95 = vals("support_foot_world_displacement_p95_m")
    state_delta = vals("max_abs_state_delta")
    summary.update(
        {
            "switch_window_count": int(len(switch_rows)),
            "switch_acceptance_proxy_pass_rate": _summarize_rows(switch_rows).get("acceptance_proxy_pass_rate", 0.0)
            if switch_rows
            else 0.0,
            "max_abs_reconstruct_delta": float(np.max(state_delta)) if state_delta.size else 0.0,
            "root_path_error_p95_m_mean": float(np.mean(root_p95)) if root_p95.size else 0.0,
            "root_path_error_p95_m_p95": _safe_percentile(root_p95, 95.0),
            "root_path_error_max_m": max_row("root_path_error_max_m"),
            "support_foot_world_displacement_p95_m_mean": float(np.mean(disp_p95)) if disp_p95.size else 0.0,
            "support_foot_world_displacement_p95_m_p95": _safe_percentile(disp_p95, 95.0),
            "support_foot_world_displacement_max_m": max_row("support_foot_world_displacement_max_m"),
        }
    )
    return summary


def _signal_ledger(n_windows: int, switch_windows: int, horizon: int) -> List[Dict[str, str]]:
    coverage_note = (
        "predictable-but-coverage-bound: topology audit has 16 granularity_fragment rows, "
        "12 unique unseen topologies, true_new_support_mode=0; learner train top1 mostly "
        "0.9915..1.0000 but blocked/leave-clip top1 remains low and decision is "
        "data_coverage_insufficient_expand_clips_no_generator; not diffusion required"
    )
    return [
        {
            "signal": "soft contact",
            "source": "GT state281 contact channels / oracle schedule condition",
            "shape_dtype_device": f"[B,H,2] float32 cpu; audited B={n_windows}, H={horizon}",
            "causal_availability": "available as commanded/oracle schedule in this GT probe; runtime prediction remains layer-1 coverage-bound",
            "derived_from_what": "state281[:,279:281] soft contact",
            "acceptance_role": "support_honesty, support token equality, endpoint bridgeability proxy",
            "leakage_risk": "oracle if future contact is consumed as decoder input; must not be conflated with predicted topology",
            "schema_status": "canonical state281 field",
            "reconstructability_expectation": "acceptance-grade when passed through reconstructed_state281 guard",
            "perturbation_sensitivity_expectation": "threshold crossings can change support labels and foot-slip masks",
        },
        {
            "signal": "support label / topology / timing",
            "source": "debug support contract over soft contact",
            "shape_dtype_device": f"labels [B,H] token/object or one-hot [B,H,4] float32 cpu; switch windows={switch_windows}",
            "causal_availability": coverage_note,
            "derived_from_what": "soft contact > support contract normalization with min_run_frames",
            "acceptance_role": "support_side_correctness and cross-switch event contract",
            "leakage_risk": "oracle future schedule leaks event timing if used as model input without layer-1 proof",
            "schema_status": "debug contract / candidate layer-1 output, not state281 field",
            "reconstructability_expectation": "must preserve first/last and switch timing for acceptance-grade replay",
            "perturbation_sensitivity_expectation": "timing off-by-one can flip support side or footstep placement",
        },
        {
            "signal": "FK foot world pos",
            "source": "train.geometry FK through existing debug helper",
            "shape_dtype_device": f"[B,H,2,3] float32 cpu; feet=(right,left), H={horizon}",
            "causal_availability": "diagnostic-only from predicted/reconstructed pose+root; not a commanded cue",
            "derived_from_what": "rot6d [B,H,276] + root_pos [B,H,3] + skeleton offsets",
            "acceptance_role": "foot slip p95/ratio and support-side feature bands",
            "leakage_risk": "low if derived after prediction; high if future FK target is fed as condition",
            "schema_status": "derived diagnostic, not schema field",
            "reconstructability_expectation": "must match reconstructed-domain FK path, not raw-only MSE",
            "perturbation_sensitivity_expectation": "root/pose errors directly move declared planted foot",
        },
        {
            "signal": "support-foot anchor transform",
            "source": "debug candidate representation from declared support side and FK foot world",
            "shape_dtype_device": "[B,R,3] anchor positions + [B,H] side tokens, float32/int64 cpu",
            "causal_availability": "oracle in this audit; predictable-but-coverage-bound if produced by layer-1 topology/timing",
            "derived_from_what": "support side labels + FK foot world position at support runs",
            "acceptance_role": "binds root path to declared support foot for support_honesty",
            "leakage_risk": "future footstep placement is oracle unless predicted/conditioned causally",
            "schema_status": "candidate lifted representation contract",
            "reconstructability_expectation": "acceptance-grade only if switch anchors are carried across runs",
            "perturbation_sensitivity_expectation": "root-relative perturbation can remain as sensitive as direct root-position perturbation unless the decoder contract adds smoothing or foot-locking",
        },
        {
            "signal": "inter-anchor / footstep placement",
            "source": "debug candidate representation anchor deltas between support runs",
            "shape_dtype_device": "[B,R-1,3] float32 cpu",
            "causal_availability": "oracle in this audit; future decoder contract must predict or condition it explicitly",
            "derived_from_what": "successive support-foot anchor positions",
            "acceptance_role": "cross-switch root path continuity and foot placement",
            "leakage_risk": "high if copied from future GT; required field does not imply causal solved",
            "schema_status": "must-enter candidate contract if drop-arm fails",
            "reconstructability_expectation": "dropping it should fail cross-switch root path reconstruction",
            "perturbation_sensitivity_expectation": "drop-arm should expose cross-switch root displacement even before decoder training",
        },
        {
            "signal": "root pos / root vel / root-relative-to-anchor",
            "source": "raw processed root_pos/root_vel and candidate lifted transforms",
            "shape_dtype_device": "[B,H,3] root_pos, [B,H,2] root_vel/ego_vel, [B,H,3] root-relative-to-anchor float32 cpu",
            "causal_availability": "root_vel is state281 output; root_pos/root-relative-to-anchor are lifted candidate variables",
            "derived_from_what": "state281 ego_vel + cond_dir integration, or anchor + local FK offset",
            "acceptance_role": "command_response, rate_budget, support_honesty through FK",
            "leakage_risk": "root path copied from GT is oracle; root-relative variables still need causal conditioning",
            "schema_status": "root_vel canonical; root_pos/root-relative candidate lifted schema",
            "reconstructability_expectation": "state/seq reconstruction must pass acceptance, not only root MSE",
            "perturbation_sensitivity_expectation": "per-frame position noise creates high derivative/foot-slip sensitivity",
        },
        {
            "signal": "local pose rot6d / pose delta",
            "source": "state281 pose prefix",
            "shape_dtype_device": "[B,H,276] and [B,H-1,276] float32 cpu",
            "causal_availability": "decoder output variable; GT-only in this audit",
            "derived_from_what": "bone_rot6d.reshape(H,46*6)",
            "acceptance_role": "pose_continuity and FK local foot offset",
            "leakage_risk": "low as output target; high if future pose frames condition the decoder",
            "schema_status": "canonical state281 field",
            "reconstructability_expectation": "rot6d convention remains existing repo convention; finite FK path required",
            "perturbation_sensitivity_expectation": "pose noise can move FK support foot unless root is anchored to compensate",
        },
        {
            "signal": "event phase",
            "source": "debug run-phase features over support labels",
            "shape_dtype_device": "[B,H,2] sin/cos float32 cpu or scalar phase [B,H,1]",
            "causal_availability": "oracle if derived from future schedule; predictable-but-coverage-bound with layer-1 schedule",
            "derived_from_what": "normalized support runs and timing",
            "acceptance_role": "conditions controlled discontinuity at support switches",
            "leakage_risk": "future event timing leak unless schedule is provided causally",
            "schema_status": "candidate condition feature, not success metric",
            "reconstructability_expectation": "must align with switch windows, not single-support-only phases",
            "perturbation_sensitivity_expectation": "phase/timing shifts can cause wrong support-foot anchor selection",
        },
        {
            "signal": "bone_angvel",
            "source": "raw processed bone_ang_vel",
            "shape_dtype_device": f"[B,H,138] float32 cpu; audited H={horizon}",
            "causal_availability": "diagnostic/aux witness; not a state281 field",
            "derived_from_what": "processed skeleton angular velocity artifact",
            "acceptance_role": "regime/rate witness and rate_budget calibration",
            "leakage_risk": "future dynamics leak if used as condition; safe as derived/eval target",
            "schema_status": "optional aux/witness output, not handoff state schema",
            "reconstructability_expectation": "needed for same acceptance helper family; not standalone success",
            "perturbation_sensitivity_expectation": "rate spikes can fail rate_budget even when pose MSE is low",
        },
        {
            "signal": "GRU hidden/carry / latent",
            "source": "model internals or prior z/hidden diagnostic artifacts",
            "shape_dtype_device": "runtime-dependent, e.g. [layers,B,D] or [B,D] float32 device",
            "causal_availability": "diagnostic-only unless proven causal + stable + runtime-recoverable",
            "derived_from_what": "production model recurrent state or latent probes",
            "acceptance_role": "witness only; never acceptance success metric in this audit",
            "leakage_risk": "can package hidden collapse/proximity as fake motion success",
            "schema_status": "not representation contract for middle motion",
            "reconstructability_expectation": "cannot reconstruct state/seq without an explicit decoder contract",
            "perturbation_sensitivity_expectation": "undefined for gate; exclude from pass/fail claims",
        },
    ]


def _decision(
    reconstruct: Mapping[str, Mapping[str, Any]],
    perturb: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    flat_rec = reconstruct.get("flat_state281", {})
    anchor_rec = reconstruct.get("support_anchor_keep_inter_anchor", {})
    drop_rec = reconstruct.get("support_anchor_drop_inter_anchor", {})
    flat_small = perturb.get("flat_velocity_state281", {}).get("1e-04", {})
    anchor_small = perturb.get("support_anchor_keep_inter_anchor", {}).get("1e-04", {})
    anchor_mid = perturb.get("support_anchor_keep_inter_anchor", {}).get("1e-03", {})
    flat_mid = perturb.get("flat_velocity_state281", {}).get("1e-03", {})

    flat_gt_ok = _finite_float(flat_rec.get("acceptance_proxy_pass_rate")) >= 0.999
    anchor_gt_ok = _finite_float(anchor_rec.get("acceptance_proxy_pass_rate")) >= 0.999
    drop_fails = _finite_float(drop_rec.get("acceptance_proxy_pass_rate")) < 0.999
    flat_mid_ratio = _finite_float(flat_mid.get("foot_slip_p95_to_band_ratio_mean"))
    anchor_mid_ratio = _finite_float(anchor_mid.get("foot_slip_p95_to_band_ratio_mean"))
    flat_small_disp = _finite_float(flat_small.get("support_foot_world_displacement_p95_m_mean"))
    anchor_small_disp = _finite_float(anchor_small.get("support_foot_world_displacement_p95_m_mean"))
    fair_perturbation_gate_completed = False
    allow = bool(flat_gt_ok and anchor_gt_ok and drop_fails and fair_perturbation_gate_completed)
    reasons: List[str] = []
    if not flat_gt_ok:
        reasons.append("flat_state281 reconstructability is not acceptance-grade")
    if not anchor_gt_ok:
        reasons.append("support_anchor_keep_inter_anchor reconstructability is not acceptance-grade")
    if not drop_fails:
        reasons.append("dropping inter-anchor placement did not fail, so field necessity is not proven")
    if not fair_perturbation_gate_completed:
        reasons.append(
            "current independent high-frequency perturbation rows are diagnostic-only; "
            "fair correlated/equal-state-MSE conditioning gate has not been run"
        )
    if allow:
        reasons.append("flat and anchored reconstruct GT; inter-anchor drop fails; fair perturbation gate passes")
    return {
        "allow_anchored_lifted_decoder_toy_smoke": allow,
        "reason": "; ".join(reasons),
        "flat_gt_acceptance": _finite_float(flat_rec.get("acceptance_proxy_pass_rate")),
        "anchor_gt_acceptance": _finite_float(anchor_rec.get("acceptance_proxy_pass_rate")),
        "drop_inter_anchor_gt_acceptance": _finite_float(drop_rec.get("acceptance_proxy_pass_rate")),
        "flat_mid_foot_slip_ratio": flat_mid_ratio,
        "anchor_mid_foot_slip_ratio": anchor_mid_ratio,
        "flat_small_support_disp_m": flat_small_disp,
        "anchor_small_support_disp_m": anchor_small_disp,
        "support_topology_availability": "coverage/granularity-bound; layer 1 not solved and not diffusion-required",
        "fair_perturbation_gate_completed": bool(fair_perturbation_gate_completed),
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "section",
        "representation",
        "noise_mse",
        "noise_scale",
        "n_windows",
        "switch_window_count",
        "acceptance_proxy_pass_rate",
        "switch_acceptance_proxy_pass_rate",
        "support_side_correctness_pass_rate",
        "support_honesty_pass_rate",
        "command_response_pass_rate",
        "pose_continuity_pass_rate",
        "rate_budget_pass_rate",
        "support_token_accuracy_mean",
        "foot_slip_p95_to_band_ratio_mean",
        "root_path_error_p95_m_mean",
        "root_path_error_p95_m_p95",
        "support_foot_world_displacement_p95_m_mean",
        "support_foot_world_displacement_p95_m_p95",
        "max_abs_reconstruct_delta",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_md(
    path: Path,
    *,
    payload: Mapping[str, Any],
    csv_path: Path,
    json_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Signal Representation Audit")
    lines.append("")
    lines.append("Date: 2026-06-03")
    lines.append("")
    lines.append(
        "Debug-only GT/read-only probe. No model training, no production Trainer/runtime/gate forward, "
        "no checkpoint mutation, no residual head, and no endpoint/yaw/discriminator continuation."
    )
    lines.append("")
    lines.append("## Current Judgment")
    lines.append("")
    lines.extend(
        [
            "- `flat_only` one-window exact plus acceptance pass shows optimizer/standardizer/MLP are not the root cause.",
            "- `flat + root_vel/root_pos/foot_vel` single-term failures localize conflict to root/foot grounding derived paths.",
            "- This is not an expressibility failure: flat and anchored are both theoretically lossless reparameterizations.",
            "- The perturbation rows are high-frequency sensitivity diagnostics, not an anchored-vs-flat conditioning verdict.",
            "- Reconstructability is tested as acceptance-grade: signal -> reconstructed state/seq -> existing reconstructed-domain acceptance path.",
            "- Anchored candidates are evaluated on switch windows and require explicit inter-anchor / footstep placement.",
            "- Causal availability is coverage-bound for topology/timing; yaw/cond_dir remains a commanded cue only.",
            "- Hidden/carry/latent is witness-only and is excluded from success metrics.",
        ]
    )
    lines.append("")
    lines.append("## Dataset / Contract")
    cfg = payload["config"]
    ds = payload["dataset"]
    lines.append(f"- matched windows: `{ds['matched_window_count']}` from `{ds['matched_targets']}`")
    lines.append(f"- switch windows: `{ds['switch_window_count']}`")
    lines.append(f"- horizon: `{cfg['horizon']}`, dtype/device: `float32/cpu`")
    lines.append("- acceptance domain: `reconstructed_state281` using existing debug helper path")
    lines.append("")
    lines.append("## Signal Ledger")
    lines.append("")
    lines.append(
        "| signal | source | shape / dtype / device | causal availability | acceptance role | leakage risk | schema status | reconstructability expectation | perturbation sensitivity expectation |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in payload["signal_ledger"]:
        lines.append(
            f"| {row['signal']} | {row['source']} | {row['shape_dtype_device']} | "
            f"{row['causal_availability']} | {row['acceptance_role']} | {row['leakage_risk']} | "
            f"{row['schema_status']} | {row['reconstructability_expectation']} | "
            f"{row['perturbation_sensitivity_expectation']} |"
        )
    lines.append("")
    lines.append("## Acceptance-grade Reconstructability")
    lines.append("")
    lines.append(
        "| representation | n | switch n | max abs state delta | accept | switch accept | side | support honest | command | pose | rate | foot ratio | root p95 err m | support foot disp p95 m |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rec in payload["reconstructability"].items():
        lines.append(
            f"| {name} | {int(rec.get('n', 0))} | {int(rec.get('switch_window_count', 0))} | "
            f"{_fmt(rec.get('max_abs_reconstruct_delta'), 6)} | {_fmt(rec.get('acceptance_proxy_pass_rate'))} | "
            f"{_fmt(rec.get('switch_acceptance_proxy_pass_rate'))} | {_fmt(rec.get('support_side_correctness_pass_rate'))} | "
            f"{_fmt(rec.get('support_honesty_pass_rate'))} | {_fmt(rec.get('command_response_pass_rate'))} | "
            f"{_fmt(rec.get('pose_continuity_pass_rate'))} | {_fmt(rec.get('rate_budget_pass_rate'))} | "
            f"{_fmt(rec.get('foot_slip_p95_to_band_ratio_mean'), 4)} | "
            f"{_fmt(rec.get('root_path_error_p95_m_mean'), 6)} | "
            f"{_fmt(rec.get('support_foot_world_displacement_p95_m_mean'), 6)} |"
        )
    lines.append("")
    lines.append("## Perturbation Sensitivity")
    lines.append("")
    lines.append(
        "Noise levels are per-channel MSE levels; injected Gaussian std is `sqrt(level)`. "
        "Flat perturbation targets state281 ego velocity, root-position lifted perturbation targets per-frame root XY, "
        "and anchored perturbation targets per-frame root-relative-to-anchor XY while keeping inter-anchor placement unless marked dropped."
    )
    lines.append("")
    lines.append(
        "| representation | noise mse | std | accept | side | support honest | command | pose | rate | foot ratio | root p95 err m | support foot disp p95 m |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, by_noise in payload["perturbation_sensitivity"].items():
        for key, rec in by_noise.items():
            lines.append(
                f"| {name} | {key} | {_fmt(rec.get('noise_scale'), 6)} | "
                f"{_fmt(rec.get('acceptance_proxy_pass_rate'))} | "
                f"{_fmt(rec.get('support_side_correctness_pass_rate'))} | "
                f"{_fmt(rec.get('support_honesty_pass_rate'))} | "
                f"{_fmt(rec.get('command_response_pass_rate'))} | "
                f"{_fmt(rec.get('pose_continuity_pass_rate'))} | "
                f"{_fmt(rec.get('rate_budget_pass_rate'))} | "
                f"{_fmt(rec.get('foot_slip_p95_to_band_ratio_mean'), 4)} | "
                f"{_fmt(rec.get('root_path_error_p95_m_mean'), 6)} | "
                f"{_fmt(rec.get('support_foot_world_displacement_p95_m_mean'), 6)} |"
            )
    lines.append("")
    lines.append("## Cross-switch / Inter-anchor Requirement")
    drop = payload["reconstructability"].get("support_anchor_drop_inter_anchor", {})
    keep = payload["reconstructability"].get("support_anchor_keep_inter_anchor", {})
    lines.append(
        f"- Drop-inter-anchor negative control: accept `{_fmt(drop.get('acceptance_proxy_pass_rate'))}`, "
        f"switch accept `{_fmt(drop.get('switch_acceptance_proxy_pass_rate'))}`, "
        f"root p95 error `{_fmt(drop.get('root_path_error_p95_m_mean'), 6)}` m."
    )
    if float(keep.get("acceptance_proxy_pass_rate", 0.0) or 0.0) >= 0.999:
        lines.append(
            "- Keep-inter-anchor passes while drop-inter-anchor fails, so inter-anchor / footstep placement is a representation-contract field, not an optional visualization."
        )
    else:
        lines.append(
            "- Keep-inter-anchor has near-zero root-path error but still fails command/support-side acceptance after root-position-to-root-velocity reconstruction; inter-anchor necessity is not yet a clean pass/fail contract until that exactness issue is resolved."
        )
    lines.append("")
    lines.append("## Decision")
    dec = payload["decision"]
    lines.append(f"- allow anchored/lifted decoder toy smoke: `{str(dec['allow_anchored_lifted_decoder_toy_smoke']).lower()}`")
    lines.append(f"- reason: {dec['reason']}")
    lines.append("- support topology/timing remains coverage/granularity-bound; Layer 1 is not solved and this is not diffusion evidence.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- summary json: `{json_path}`")
    lines.append(f"- rows csv: `{csv_path}`")
    _dump_md(path, lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
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
    switch_count = int(
        sum(int(it.support_contract.get("normalized_transition_count", 0)) >= 1 for it in main_items)
    )
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
        horizon=int(args.horizon),
        min_run_frames=int(args.min_run_frames),
        oracle_contact_passthrough=True,
        command_align_root_vel=False,
    )

    reconstruct_specs = (
        "flat_state281",
        "root_position_lifted",
        "support_anchor_keep_inter_anchor",
        "support_anchor_drop_inter_anchor",
    )
    reconstruct_records: Dict[str, List[EvalRecord]] = {name: [] for name in reconstruct_specs}
    perturb_records: Dict[str, Dict[str, List[EvalRecord]]] = {
        "flat_velocity_state281": {},
        "root_position_lifted": {},
        "support_anchor_keep_inter_anchor": {},
        "support_anchor_drop_inter_anchor": {},
    }

    for item_i, item in enumerate(main_items):
        baseline_seq = _reconstructed_gt_seq(
            item,
            oracle_contact_passthrough=True,
            command_align_root_vel=False,
        )
        true_state = np.asarray(item.seq["state281"], dtype=np.float32).reshape(int(args.horizon), STATE_DIM)
        aux = np.asarray(item.seq["bone_angvel"], dtype=np.float32).reshape(int(args.horizon), ANGVEL_DIM)

        state_by_repr = {
            "flat_state281": _flat_state(item, noise_scale=0.0, rng=_rng_for(args.seed, item_i, "flat", 0)),
            "root_position_lifted": _root_position_state(
                item,
                baseline_seq,
                noise_scale=0.0,
                rng=_rng_for(args.seed, item_i, "root", 0),
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
        for repr_name, state in state_by_repr.items():
            reconstruct_records[repr_name].append(
                _evaluate_candidate_state(
                    representation=repr_name,
                    section="reconstructability",
                    noise_mse=None,
                    noise_trial=None,
                    item=item,
                    state=state,
                    aux=aux,
                    baseline_seq=baseline_seq,
                    true_state=true_state,
                    baseline_bands=reconstructed_baseline_bands,
                    support_bands=reconstructed_support_bands,
                    skeleton=skeleton,
                    min_run_frames=int(args.min_run_frames),
                )
            )

        for level in args.noise_levels:
            key = f"{float(level):.0e}"
            noise_scale = math.sqrt(float(level))
            for name in perturb_records:
                perturb_records[name].setdefault(key, [])
            for trial in range(int(args.noise_trials)):
                perturb_states = {
                    "flat_velocity_state281": _flat_state(
                        item,
                        noise_scale=noise_scale,
                        rng=_rng_for(args.seed, item_i, "flat_velocity_state281", key, trial),
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
                for repr_name, state in perturb_states.items():
                    perturb_records[repr_name][key].append(
                        _evaluate_candidate_state(
                            representation=repr_name,
                            section="perturbation_sensitivity",
                            noise_mse=float(level),
                            noise_trial=int(trial),
                            item=item,
                            state=state,
                            aux=aux,
                            baseline_seq=baseline_seq,
                            true_state=true_state,
                            baseline_bands=reconstructed_baseline_bands,
                            support_bands=reconstructed_support_bands,
                            skeleton=skeleton,
                            min_run_frames=int(args.min_run_frames),
                        )
                    )

    reconstruct_summary = {
        name: _aggregate_records(records) for name, records in reconstruct_records.items()
    }
    perturb_summary: Dict[str, Dict[str, Any]] = {}
    for name, by_noise in perturb_records.items():
        perturb_summary[name] = {}
        for noise_key, records in by_noise.items():
            rec = _aggregate_records(records)
            rec["noise_mse"] = float(noise_key)
            rec["noise_scale"] = math.sqrt(float(noise_key))
            rec["noise_trials"] = int(args.noise_trials)
            perturb_summary[name][noise_key] = rec

    metric_rows: List[Dict[str, Any]] = []
    for name, rec in reconstruct_summary.items():
        metric_rows.append(
            {
                "section": "reconstructability",
                "representation": name,
                "noise_mse": "",
                "noise_scale": "",
                "n_windows": rec.get("n", 0),
                "support_token_accuracy_mean": rec.get("oracle_support_token_accuracy_mean", 0.0),
                "note": "GT signal -> reconstructed state/seq -> reconstructed-domain acceptance",
                **rec,
            }
        )
    for name, by_noise in perturb_summary.items():
        for noise_key, rec in by_noise.items():
            metric_rows.append(
                {
                    "section": "perturbation_sensitivity",
                    "representation": name,
                    "noise_mse": noise_key,
                    "noise_scale": rec.get("noise_scale", 0.0),
                    "n_windows": rec.get("n", 0),
                    "support_token_accuracy_mean": rec.get("oracle_support_token_accuracy_mean", 0.0),
                    "note": "Gaussian perturbation on representation-specific root/support variable",
                    **rec,
                }
            )

    ledger = _signal_ledger(len(main_items), switch_count, int(args.horizon))
    decision = _decision(reconstruct_summary, perturb_summary)
    payload = {
        "task": "action_handoff_signal_representation_audit",
        "scope": (
            "GT/read-only signal expressibility and perturbation-sensitivity audit; no training, "
            "no production Trainer/runtime/gate forward or edit, no checkpoint mutation"
        ),
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "out_dir": str(args.out_dir),
            "horizon": int(args.horizon),
            "context_len": int(args.context_len),
            "stride": int(args.stride),
            "min_run_frames": int(args.min_run_frames),
            "reconstructed_baseline_quantile": float(args.reconstructed_baseline_quantile),
            "noise_levels_mse": [float(x) for x in args.noise_levels],
            "noise_trials": int(args.noise_trials),
            "seed": int(args.seed),
            "dtype": "float32",
            "device": "cpu",
        },
        "dataset": {
            "matched_targets": list(MATCHED_TARGETS),
            "unmatched_out_of_scope": UNMATCHED_TARGET,
            "matched_window_count": int(len(main_items)),
            "switch_window_count": int(switch_count),
            "per_clip_windows": dict(Counter(it.clip for it in main_items)),
        },
        "input_output_contract": {
            "flat_state281": {"shape": [int(args.horizon), STATE_DIM], "dtype": "float32", "device": "cpu"},
            "soft_contact": {"shape": [int(args.horizon), 2], "dtype": "float32", "device": "cpu"},
            "fk_foot_world_pos": {"shape": [int(args.horizon), 2, 3], "dtype": "float32", "device": "cpu"},
            "bone_angvel": {"shape": [int(args.horizon), ANGVEL_DIM], "dtype": "float32", "device": "cpu"},
        },
        "signal_ledger": ledger,
        "reconstructability": reconstruct_summary,
        "perturbation_sensitivity": perturb_summary,
        "decision": decision,
        "hard_constraint_confirmations": {
            "committed": False,
            "pushed": False,
            "stashed": False,
            "cleaned_or_reverted_dirty_untracked": False,
            "trained_new_model": False,
            "forwarded_production_runtime_or_trainer": False,
            "edited_production_runtime_trainer_gate": False,
            "mutated_checkpoint": False,
            "residual_head": False,
            "endpoint_yaw_discriminator_continuation": False,
        },
        "_metric_rows": metric_rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "signal_representation_audit_summary.json"
    md_path = args.out_dir / "signal_representation_audit_summary.md"
    csv_path = args.out_dir / "signal_representation_audit_rows.csv"
    _dump_json(json_path, payload)
    _write_rows_csv(csv_path, metric_rows)
    _write_summary_md(md_path, payload=payload, csv_path=csv_path, json_path=json_path)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_C)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-run-frames", type=int, default=2)
    p.add_argument("--reconstructed-baseline-quantile", type=float, default=100.0)
    p.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    p.add_argument("--noise-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260603)
    return p.parse_args()


def main() -> None:
    payload = run(parse_args())
    out_dir = Path(payload["config"]["out_dir"])
    print(f"wrote {out_dir / 'signal_representation_audit_summary.md'}")
    print(f"wrote {out_dir / 'signal_representation_audit_summary.json'}")
    print(f"wrote {out_dir / 'signal_representation_audit_rows.csv'}")
    print(json.dumps(_jsonify(payload["decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
