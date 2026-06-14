#!/usr/bin/env python3
"""Replay the middle-generator acceptance contract on existing artifacts.

Read-only probe. No training, no checkpoint mutation, no production runtime/gate
change. The purpose is to check whether the v0 acceptance families can separate:

* real continuous motion windows, which should pass;
* matched hard seams and one-frame switches, which should fail;
* ramp/mapping/direct/lambda proxies, which may be diagnostic but must not be
  misreported as accepted middle motion.

The probe writes per-case rows; aggregate pass rates are only summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_TOPK,
    RAW_COND_DIR_SLICE,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    full_state_align,
    load_clip_states,
)
from train.geometry import fk_positions_from_rot6d  # noqa: E402


DEFAULT_NPZ_ROOT = Path("raw_data/processed_data")
DEFAULT_Z_FEATURES = Path("debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz")
DEFAULT_TWO_FRAME = Path(
    "debug_output/_tmp_action_handoff_two_frame_dynamics_check_20260601/"
    "two_frame_dynamics_check_summary.json"
)
DEFAULT_BONE_BRIDGE = Path(
    "debug_output/_tmp_action_handoff_bone_angvel_bridge_probe_20260601_v1/"
    "bone_angvel_bridge_probe_summary.json"
)
DEFAULT_REGIME_BRIDGE = Path(
    "debug_output/_tmp_action_handoff_regime_bridge_probe_20260601_v2/"
    "regime_bridge_probe_summary.json"
)

LOCKED_CLIPS = (WALK_F, *TURN_CLIPS)
FPS = 60.0
POSE_DIM = 276
ANGVEL_DIM = 138
CONTACT_THRESHOLD = 0.5
ROOT_SPEED_EPS = 1e-4
EPS = 1e-8


@dataclass(frozen=True)
class ClipData:
    name: str
    state281: np.ndarray
    rot6d: np.ndarray
    root_pos: np.ndarray
    root_vel: np.ndarray
    bone_angvel: np.ndarray
    cond_dir: np.ndarray
    contact: np.ndarray
    yaw_rate: np.ndarray


@dataclass(frozen=True)
class SkeletonMeta:
    bone_names: List[str]
    parents: List[int]
    offsets: np.ndarray
    right_foot_idx: Optional[int]
    left_foot_idx: Optional[int]


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "null"
    if not math.isfinite(x):
        return "null"
    return f"{x:.{digits}f}"


def _finite_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _npz_json_scalar(v: Any) -> Dict[str, Any]:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, str):
        return json.loads(v)
    if isinstance(v, Mapping):
        return dict(v)
    raise TypeError(f"cannot parse npz JSON scalar type={type(v).__name__}")


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


def _dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_percentile(vals: np.ndarray, q: float, default: float = 0.0) -> float:
    x = np.asarray(vals, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, float(q)))


def _rms_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.sqrt(np.mean(arr * arr, axis=1))


def _step_pose_l2(rot6d: np.ndarray) -> np.ndarray:
    if rot6d.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    d = np.diff(rot6d.reshape(rot6d.shape[0], -1), axis=0)
    return np.linalg.norm(d, axis=1) / math.sqrt(max(1, d.shape[1]))


def _step_angvel_rms(angvel: np.ndarray) -> np.ndarray:
    if angvel.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return _rms_rows(np.diff(angvel.reshape(angvel.shape[0], -1), axis=0))


def _step_angvel_component_p95(angvel: np.ndarray) -> np.ndarray:
    if angvel.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.percentile(np.abs(np.diff(angvel.reshape(angvel.shape[0], -1), axis=0)), 95, axis=1)


def _step_l2(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(arr.reshape(arr.shape[0], -1), axis=0), axis=1)


def _heading_error_rad(root_vel: np.ndarray, cond_dir: np.ndarray) -> np.ndarray:
    rv = np.asarray(root_vel, dtype=np.float64).reshape(-1, 2)
    cd = np.asarray(cond_dir, dtype=np.float64).reshape(-1, 2)
    n = min(rv.shape[0], cd.shape[0])
    rv = rv[:n]
    cd = cd[:n]
    speed = np.linalg.norm(rv, axis=1)
    cmd_norm = np.linalg.norm(cd, axis=1)
    valid = (speed > ROOT_SPEED_EPS) & (cmd_norm > EPS)
    err = np.zeros(n, dtype=np.float64)
    if np.any(valid):
        dot = np.sum(rv[valid] * cd[valid], axis=1) / np.maximum(speed[valid] * cmd_norm[valid], EPS)
        err[valid] = np.arccos(np.clip(dot, -1.0, 1.0))
    return err


def _support_label(contact_frame: np.ndarray) -> str:
    c = np.asarray(contact_frame, dtype=np.float64).reshape(-1)
    right = bool(c.size > 0 and c[0] > CONTACT_THRESHOLD)
    left = bool(c.size > 1 and c[1] > CONTACT_THRESHOLD)
    if right and left:
        return "dual"
    if right:
        return "right"
    if left:
        return "left"
    return "flight_or_unknown"


def _load_skeleton_meta(npz_root: Path, clip: str = WALK_F) -> SkeletonMeta:
    path = npz_root / f"{clip}.npz"
    with np.load(path, allow_pickle=True) as z:
        meta = _npz_json_scalar(z["meta_json"])
    sk = meta.get("skeleton", {}) if isinstance(meta, Mapping) else {}
    names = [str(x) for x in sk.get("bone_names", [])]
    parents = [int(x) for x in sk.get("parents", [])]
    offsets = np.asarray(sk.get("ref_local_offsets_m", []), dtype=np.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise RuntimeError(f"{path}: invalid skeleton.ref_local_offsets_m shape={tuple(offsets.shape)}")

    def _idx(side: str) -> Optional[int]:
        for name in (f"ball_{side}", f"toe_{side}", f"foot_{side}"):
            if name in names:
                return int(names.index(name))
        return None

    return SkeletonMeta(
        bone_names=names,
        parents=parents,
        offsets=offsets,
        right_foot_idx=_idx("r"),
        left_foot_idx=_idx("l"),
    )


def _foot_slip_metrics(
    rot6d: np.ndarray,
    root_pos: np.ndarray,
    contact: np.ndarray,
    skeleton: SkeletonMeta,
) -> Dict[str, Any]:
    if rot6d.shape[0] < 2:
        return {
            "status": "short",
            "contacted_speed_count": 0,
            "contacted_speed_mean_mps": 0.0,
            "contacted_speed_p95_mps": 0.0,
            "contacted_speed_max_mps": 0.0,
        }
    if skeleton.right_foot_idx is None and skeleton.left_foot_idx is None:
        return {"status": "unavailable", "reason": "no foot joints"}
    try:
        rot = torch.as_tensor(rot6d.reshape(rot6d.shape[0], -1, 6), dtype=torch.float32)
        root = torch.as_tensor(root_pos.reshape(root_pos.shape[0], 3), dtype=torch.float32)
        pos = fk_positions_from_rot6d(
            rot,
            skeleton.parents,
            torch.as_tensor(skeleton.offsets, dtype=torch.float32),
            root_pos=root,
        ).detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover - diagnostic should report, not hide.
        return {"status": "unavailable", "reason": f"fk failed: {type(exc).__name__}: {exc}"}

    c = np.asarray(contact, dtype=np.float64)
    vals: List[float] = []

    def _collect(ch_idx: int, joint_idx: Optional[int]) -> Dict[str, Any]:
        if joint_idx is None or c.shape[1] <= ch_idx:
            return {"n": 0, "mean_mps": None, "p95_mps": None, "max_mps": None}
        mask = (c[:-1, ch_idx] > CONTACT_THRESHOLD) & (c[1:, ch_idx] > CONTACT_THRESHOLD)
        speed = np.linalg.norm(pos[1:, joint_idx] - pos[:-1, joint_idx], axis=1) * FPS
        picked = speed[mask]
        vals.extend(float(x) for x in picked.tolist())
        if picked.size == 0:
            return {"n": 0, "mean_mps": None, "p95_mps": None, "max_mps": None}
        return {
            "n": int(picked.size),
            "mean_mps": float(np.mean(picked)),
            "p95_mps": float(np.percentile(picked, 95)),
            "max_mps": float(np.max(picked)),
        }

    right = _collect(0, skeleton.right_foot_idx)
    left = _collect(1, skeleton.left_foot_idx)
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "status": "ok",
        "contact_order": "rl",
        "right": right,
        "left": left,
        "contacted_speed_count": int(arr.size),
        "contacted_speed_mean_mps": float(np.mean(arr)) if arr.size else 0.0,
        "contacted_speed_p95_mps": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "contacted_speed_max_mps": float(np.max(arr)) if arr.size else 0.0,
    }


def _load_clips(npz_root: Path, z_features: Path) -> Dict[str, ClipData]:
    states = load_clip_states(z_features, npz_root, clips=LOCKED_CLIPS)
    out: Dict[str, ClipData] = {}
    for name in LOCKED_CLIPS:
        path = npz_root / f"{name}.npz"
        with np.load(path, allow_pickle=True) as z:
            state = np.asarray(states[name], dtype=np.float32)
            n = int(state.shape[0])
            rot6d = np.asarray(z["bone_rot6d"], dtype=np.float32).reshape(-1, POSE_DIM)[:n]
            root_pos = np.asarray(z["root_pos"], dtype=np.float32).reshape(-1, 3)[:n]
            root_vel = np.asarray(z["root_vel"], dtype=np.float32).reshape(-1, 2)[:n]
            bone_angvel = np.asarray(z["bone_ang_vel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)[:n]
            cond = np.asarray(z["cond_in"], dtype=np.float32)[:n]
        n = min(n, rot6d.shape[0], root_pos.shape[0], root_vel.shape[0], bone_angvel.shape[0], cond.shape[0])
        if n < 2:
            raise RuntimeError(f"{name}: aligned frame count too small ({n})")
        out[name] = ClipData(
            name=name,
            state281=state[:n],
            rot6d=rot6d[:n],
            root_pos=root_pos[:n],
            root_vel=root_vel[:n],
            bone_angvel=bone_angvel[:n],
            cond_dir=cond[:n, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]],
            contact=state[:n, CONTACT_SLICE],
            yaw_rate=state[:n, YAW_RATE_SLICE].reshape(-1),
        )
    return out


def _make_sequence(clip: ClipData, start: int, horizon: int) -> Dict[str, np.ndarray]:
    s = int(start)
    e = int(start) + int(horizon)
    return {
        "rot6d": clip.rot6d[s:e],
        "root_pos": clip.root_pos[s:e],
        "root_vel": clip.root_vel[s:e],
        "bone_angvel": clip.bone_angvel[s:e],
        "cond_dir": clip.cond_dir[s:e],
        "contact": clip.contact[s:e],
        "yaw_rate": clip.yaw_rate[s:e],
    }


def _make_hard_seam_sequence(walk: ClipData, target: ClipData, phi: int, onset: int) -> Dict[str, np.ndarray]:
    target_root_shift = walk.root_pos[int(phi)] - target.root_pos[int(onset)]
    return {
        "rot6d": np.stack([walk.rot6d[int(phi)], target.rot6d[int(onset)]], axis=0),
        "root_pos": np.stack(
            [walk.root_pos[int(phi)], target.root_pos[int(onset)] + target_root_shift],
            axis=0,
        ),
        "root_vel": np.stack([walk.root_vel[int(phi)], target.root_vel[int(onset)]], axis=0),
        "bone_angvel": np.stack([walk.bone_angvel[int(phi)], target.bone_angvel[int(onset)]], axis=0),
        "cond_dir": np.stack([walk.cond_dir[int(phi)], target.cond_dir[int(onset)]], axis=0),
        "contact": np.stack([walk.contact[int(phi)], target.contact[int(onset)]], axis=0),
        "yaw_rate": np.asarray([walk.yaw_rate[int(phi)], target.yaw_rate[int(onset)]], dtype=np.float32),
    }


def _make_one_frame_switch_sequence(walk: ClipData, target: ClipData, phi: int, onset: int) -> Dict[str, np.ndarray]:
    j = (int(phi) + 1) % int(walk.rot6d.shape[0])
    return {
        "rot6d": np.stack([walk.rot6d[int(phi)], walk.rot6d[j]], axis=0),
        "root_pos": np.stack([walk.root_pos[int(phi)], walk.root_pos[j]], axis=0),
        "root_vel": np.stack([walk.root_vel[int(phi)], target.root_vel[int(onset)]], axis=0),
        "bone_angvel": np.stack([walk.bone_angvel[int(phi)], target.bone_angvel[int(onset)]], axis=0),
        "cond_dir": np.stack([walk.cond_dir[int(phi)], target.cond_dir[int(onset)]], axis=0),
        "contact": np.stack([walk.contact[int(phi)], walk.contact[j]], axis=0),
        "yaw_rate": np.asarray([walk.yaw_rate[int(phi)], target.yaw_rate[int(onset)]], dtype=np.float32),
    }


def _make_linear_proxy_sequence(
    walk: ClipData,
    target: ClipData,
    phi: int,
    onset: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    h = int(horizon)
    alpha = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    end = min(int(onset) + h - 1, int(target.rot6d.shape[0]) - 1)
    target_root_shift = walk.root_pos[int(phi)] - target.root_pos[int(onset)]

    def _lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (1.0 - alpha) * a.reshape(1, -1) + alpha * b.reshape(1, -1)

    return {
        "rot6d": _lerp(walk.rot6d[int(phi)], target.rot6d[end]).astype(np.float32),
        "root_pos": _lerp(walk.root_pos[int(phi)], target.root_pos[end] + target_root_shift).astype(np.float32),
        "root_vel": _lerp(walk.root_vel[int(phi)], target.root_vel[end]).astype(np.float32),
        "bone_angvel": _lerp(walk.bone_angvel[int(phi)], target.bone_angvel[end]).astype(np.float32),
        "cond_dir": _lerp(walk.cond_dir[int(phi)], target.cond_dir[end]).astype(np.float32),
        "contact": _lerp(walk.contact[int(phi)], target.contact[end]).astype(np.float32),
        "yaw_rate": _lerp(np.asarray([walk.yaw_rate[int(phi)]]), np.asarray([target.yaw_rate[end]])).reshape(-1).astype(np.float32),
    }


def _calibrate_baselines(
    clips: Mapping[str, ClipData],
    skeleton: SkeletonMeta,
    *,
    quantile: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, clip in clips.items():
        foot = _foot_slip_metrics(clip.rot6d, clip.root_pos, clip.contact, skeleton)
        level_center = np.mean(clip.bone_angvel, axis=0)
        level_dist = _rms_rows(clip.bone_angvel - level_center.reshape(1, -1))
        out[name] = {
            "quantile": float(quantile),
            "pose_step_l2": _safe_percentile(_step_pose_l2(clip.rot6d), quantile),
            "angvel_step_rms": _safe_percentile(_step_angvel_rms(clip.bone_angvel), quantile),
            "angvel_step_component_p95": _safe_percentile(_step_angvel_component_p95(clip.bone_angvel), quantile),
            "rootvel_step_l2": _safe_percentile(_step_l2(clip.root_vel), quantile),
            "yaw_rate_step_abs": _safe_percentile(np.abs(np.diff(clip.yaw_rate)), quantile),
            "contact_step_l2": _safe_percentile(_step_l2(clip.contact), quantile),
            "heading_error_rad": _safe_percentile(_heading_error_rad(clip.root_vel, clip.cond_dir), quantile),
            "bone_angvel_level_rms": _safe_percentile(level_dist, quantile),
            "bone_angvel_level_center": level_center.astype(np.float32),
            "foot_slip_contacted_speed_p95_mps": _safe_percentile(
                np.asarray([foot.get("contacted_speed_p95_mps", 0.0)], dtype=np.float64),
                100.0,
            ),
            "foot_slip": foot,
        }
        # Per-contacted-step threshold is more stable than a single aggregate p95.
        # Recompute the full contacted speed pool for the threshold when possible.
        speeds = _contacted_foot_speeds(clip, skeleton)
        out[name]["foot_slip_contacted_speed_mps"] = _safe_percentile(speeds, quantile)
    return out


def _contacted_foot_speeds(clip: ClipData, skeleton: SkeletonMeta) -> np.ndarray:
    foot = _foot_positions(clip.rot6d, clip.root_pos, skeleton)
    if foot is None:
        return np.zeros((0,), dtype=np.float64)
    vals: List[float] = []
    for ch_idx, side in ((0, "right"), (1, "left")):
        if side not in foot or clip.contact.shape[1] <= ch_idx:
            continue
        mask = (clip.contact[:-1, ch_idx] > CONTACT_THRESHOLD) & (clip.contact[1:, ch_idx] > CONTACT_THRESHOLD)
        speed = np.linalg.norm(foot[side][1:] - foot[side][:-1], axis=1) * FPS
        vals.extend(float(x) for x in speed[mask].tolist())
    return np.asarray(vals, dtype=np.float64)


def _foot_positions(
    rot6d: np.ndarray,
    root_pos: np.ndarray,
    skeleton: SkeletonMeta,
) -> Optional[Dict[str, np.ndarray]]:
    if skeleton.right_foot_idx is None and skeleton.left_foot_idx is None:
        return None
    try:
        rot = torch.as_tensor(rot6d.reshape(rot6d.shape[0], -1, 6), dtype=torch.float32)
        root = torch.as_tensor(root_pos.reshape(root_pos.shape[0], 3), dtype=torch.float32)
        pos = fk_positions_from_rot6d(
            rot,
            skeleton.parents,
            torch.as_tensor(skeleton.offsets, dtype=torch.float32),
            root_pos=root,
        ).detach().cpu().numpy()
    except Exception:
        return None
    out: Dict[str, np.ndarray] = {}
    if skeleton.right_foot_idx is not None:
        out["right"] = pos[:, skeleton.right_foot_idx]
    if skeleton.left_foot_idx is not None:
        out["left"] = pos[:, skeleton.left_foot_idx]
    return out


def _bridgeability_from_deltas(
    deltas: Mapping[str, float],
    budgets: Mapping[str, float],
    *,
    horizon: int,
    groundable: bool,
) -> Dict[str, Any]:
    needed: Dict[str, Optional[int]] = {}
    for key, delta in deltas.items():
        budget = float(budgets.get(key, 0.0) or 0.0)
        if budget <= EPS:
            needed[key] = None if abs(float(delta)) <= EPS else int(10**9)
        else:
            needed[key] = int(max(1, math.ceil(abs(float(delta)) / budget)))
    finite_needed = [int(v) for v in needed.values() if v is not None and int(v) < int(10**9)]
    has_impossible = any(v is not None and int(v) >= int(10**9) for v in needed.values())
    max_needed = max(finite_needed) if finite_needed else (0 if not has_impossible else int(10**9))
    return {
        "groundable": bool(groundable),
        "needed_frames": needed,
        "max_needed_frames": int(max_needed),
        "one_frame_bridgeable": bool(groundable and not has_impossible and max_needed <= 1),
        "horizon_bridgeable": bool(groundable and not has_impossible and max_needed <= int(horizon)),
    }


def _evaluate_sequence(
    seq: Mapping[str, np.ndarray],
    *,
    target: str,
    target_bands: Mapping[str, Any],
    skeleton: SkeletonMeta,
    case: str,
    expected_label: str,
    start_phase: str,
    endpoint_bridgeability: bool,
    endpoint_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rot6d = np.asarray(seq["rot6d"], dtype=np.float32).reshape(-1, POSE_DIM)
    root_pos = np.asarray(seq["root_pos"], dtype=np.float32).reshape(-1, 3)
    root_vel = np.asarray(seq["root_vel"], dtype=np.float32).reshape(-1, 2)
    bone_angvel = np.asarray(seq["bone_angvel"], dtype=np.float32).reshape(-1, ANGVEL_DIM)
    cond_dir = np.asarray(seq["cond_dir"], dtype=np.float32).reshape(-1, 2)
    contact = np.asarray(seq["contact"], dtype=np.float32).reshape(-1, 2)
    yaw_rate = np.asarray(seq["yaw_rate"], dtype=np.float32).reshape(-1)

    foot = _foot_slip_metrics(rot6d, root_pos, contact, skeleton)
    level_center = np.asarray(target_bands["bone_angvel_level_center"], dtype=np.float32).reshape(1, -1)
    level_rms = float(_rms_rows(bone_angvel[-1:].reshape(1, -1) - level_center)[0])

    pose_step_p95 = _safe_percentile(_step_pose_l2(rot6d), 95)
    angvel_step_rms_p95 = _safe_percentile(_step_angvel_rms(bone_angvel), 95)
    angvel_component_p95_p95 = _safe_percentile(_step_angvel_component_p95(bone_angvel), 95)
    rootvel_step_l2_p95 = _safe_percentile(_step_l2(root_vel), 95)
    yaw_rate_step_abs_p95 = _safe_percentile(np.abs(np.diff(yaw_rate)), 95)
    contact_step_l2_p95 = _safe_percentile(_step_l2(contact), 95)
    heading_error_p95 = _safe_percentile(_heading_error_rad(root_vel, cond_dir), 95)
    foot_p95 = _finite_float(foot.get("contacted_speed_p95_mps")) or 0.0

    regime_reached = level_rms <= float(target_bands["bone_angvel_level_rms"]) + EPS
    rate_budget = (
        angvel_step_rms_p95 <= float(target_bands["angvel_step_rms"]) + EPS
        and angvel_component_p95_p95 <= float(target_bands["angvel_step_component_p95"]) + EPS
        and rootvel_step_l2_p95 <= float(target_bands["rootvel_step_l2"]) + EPS
        and yaw_rate_step_abs_p95 <= float(target_bands["yaw_rate_step_abs"]) + EPS
    )
    support_honesty = (
        contact_step_l2_p95 <= float(target_bands["contact_step_l2"]) + EPS
        and foot_p95 <= float(target_bands["foot_slip_contacted_speed_mps"]) + EPS
    )
    command_response = heading_error_p95 <= float(target_bands["heading_error_rad"]) + EPS
    pose_continuity = pose_step_p95 <= float(target_bands["pose_step_l2"]) + EPS

    families = {
        "regime_reached": bool(regime_reached),
        "rate_budget": bool(rate_budget),
        "support_honesty": bool(support_honesty),
        "command_response": bool(command_response),
        "pose_continuity": bool(pose_continuity),
        "endpoint_bridgeability": bool(endpoint_bridgeability),
    }
    failed = [k for k, v in families.items() if not v]
    return {
        "case": case,
        "target": target,
        "start_phase": start_phase,
        "expected_label": expected_label,
        "pass": bool(not failed),
        "failed_family": ",".join(failed) if failed else "",
        **families,
        "metrics": {
            "horizon": int(rot6d.shape[0]),
            "pose_step_l2_p95": pose_step_p95,
            "angvel_step_rms_p95": angvel_step_rms_p95,
            "angvel_component_p95_p95": angvel_component_p95_p95,
            "rootvel_step_l2_p95": rootvel_step_l2_p95,
            "yaw_rate_step_abs_p95": yaw_rate_step_abs_p95,
            "contact_step_l2_p95": contact_step_l2_p95,
            "heading_error_p95_rad": heading_error_p95,
            "bone_angvel_level_rms_to_target": level_rms,
            "foot_slip_p95_mps": foot_p95,
            "support_start": _support_label(contact[0]),
            "support_end": _support_label(contact[-1]),
        },
        "thresholds": {k: v for k, v in target_bands.items() if k != "bone_angvel_level_center"},
        "endpoint_details": dict(endpoint_details or {}),
    }


def _artifact_row(
    *,
    case: str,
    target: str,
    start_phase: str,
    expected_label: str,
    families: Mapping[str, bool],
    metrics: Mapping[str, Any],
    endpoint_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    failed = [k for k in (
        "regime_reached",
        "rate_budget",
        "support_honesty",
        "command_response",
        "pose_continuity",
        "endpoint_bridgeability",
    ) if not bool(families.get(k, False))]
    return {
        "case": case,
        "target": target,
        "start_phase": start_phase,
        "expected_label": expected_label,
        "pass": bool(not failed),
        "failed_family": ",".join(failed) if failed else "",
        "regime_reached": bool(families.get("regime_reached", False)),
        "rate_budget": bool(families.get("rate_budget", False)),
        "support_honesty": bool(families.get("support_honesty", False)),
        "command_response": bool(families.get("command_response", False)),
        "pose_continuity": bool(families.get("pose_continuity", False)),
        "endpoint_bridgeability": bool(families.get("endpoint_bridgeability", False)),
        "metrics": dict(metrics),
        "thresholds": {},
        "endpoint_details": dict(endpoint_details or {}),
    }


def _rows_from_bone_bridge_artifact(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []

    motion = data.get("B_motion_safe_v2", {})
    for target, block in (motion.get("per_target", {}) or {}).items():
        for row in block.get("rows", []) or []:
            variant = str(row.get("variant", ""))
            if not (
                variant.startswith("bone_angvel_ramp")
                or variant.startswith("bone_angvel_rootvel_cmdyaw_ramp")
            ):
                continue
            checks = row.get("motion_safe_v2_checks", {}) or {}
            rows.append(
                _artifact_row(
                    case=f"proxy_replay:{variant}",
                    target=str(target),
                    start_phase=f"phi={block.get('phi')};onset={block.get('onset')}",
                    expected_label="diagnostic_only_fail",
                    families={
                        "regime_reached": True,
                        "rate_budget": bool(checks.get("bone_angvel_delta_rms_rad_s"))
                        and bool(checks.get("ego_velocity_delta_l2")),
                        "support_honesty": bool(checks.get("fk_foot_slip_mean_mps"))
                        and bool(checks.get("contact_delta_l2")),
                        "command_response": bool(checks.get("realized_yaw_rate_deviation_mean_deg_s")),
                        "pose_continuity": bool(checks.get("pose_delta_rms_rot6d")),
                        "endpoint_bridgeability": False,
                    },
                    metrics={k: row.get(k) for k in (
                        "bone_angvel_delta_rms_rad_s",
                        "bone_angvel_delta_p95_rad_s",
                        "fk_foot_slip_mean_mps",
                        "fk_foot_slip_p95_mps",
                        "realized_yaw_rate_deviation_mean_deg_s",
                        "ego_velocity_delta_l2",
                        "contact_delta_l2",
                    )},
                    endpoint_details={"source": str(path), "artifact_motion_safe_v2_pass": row.get("motion_safe_v2_pass")},
                )
            )

    mapping = data.get("C_mapping_probe", {}).get("readout_motion_safe_v2", {})
    for row in mapping.get("rows", []) or []:
        variant = str(row.get("variant", "mapping_state281"))
        checks = row.get("motion_safe_v2_checks", {}) or {}
        rows.append(
            _artifact_row(
                case=f"proxy_replay:{variant}",
                target=str(row.get("target")),
                start_phase="artifact_mapping_pair",
                expected_label="diagnostic_only_fail",
                families={
                    "regime_reached": True,
                    "rate_budget": bool(checks.get("bone_angvel_delta_rms_rad_s"))
                    and bool(checks.get("ego_velocity_delta_l2")),
                    "support_honesty": bool(checks.get("fk_foot_slip_mean_mps"))
                    and bool(checks.get("contact_delta_l2")),
                    "command_response": bool(checks.get("realized_yaw_rate_deviation_mean_deg_s")),
                    "pose_continuity": bool(checks.get("pose_delta_rms_rot6d")),
                    "endpoint_bridgeability": False,
                },
                metrics={k: row.get(k) for k in (
                    "bone_angvel_delta_rms_rad_s",
                    "bone_angvel_delta_p95_rad_s",
                    "fk_foot_slip_mean_mps",
                    "fk_foot_slip_p95_mps",
                    "realized_yaw_rate_deviation_mean_deg_s",
                    "ego_velocity_delta_l2",
                    "contact_delta_l2",
                    "hidden_pre_collapse_fraction_to_walk",
                )},
                endpoint_details={"source": str(path), "artifact_motion_safe_v2_pass": row.get("motion_safe_v2_pass")},
            )
        )
    return rows


def _rows_from_regime_bridge_artifact(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    direct = data.get("C_direct_motion_honesty", {})
    for target, block in (direct.get("per_target", {}) or {}).items():
        for mode, row in (block.get("modes", {}) or {}).items():
            state = row.get("state_space", {}) or {}
            foot = row.get("foot_slip", {}) or {}
            pop_safe = bool(state.get("pop_safe", False))
            rows.append(
                _artifact_row(
                    case=f"negative_control:{mode}",
                    target=str(target),
                    start_phase=f"phi={block.get('phi')};onset={block.get('onset')}",
                    expected_label="fail",
                    families={
                        "regime_reached": bool(state.get("reached_proxy", False)),
                        "rate_budget": False,
                        "support_honesty": bool((foot.get("status") == "ok") and _finite_float(foot.get("mean_mps_over_sides")) is not None),
                        "command_response": False,
                        "pose_continuity": pop_safe,
                        "endpoint_bridgeability": False,
                    },
                    metrics={
                        "best_pose_d": state.get("best_pose_d"),
                        "pop": state.get("pop"),
                        "pop_safe": state.get("pop_safe"),
                        "foot_slip_mean_mps": foot.get("mean_mps_over_sides"),
                        "foot_slip_max_mps": foot.get("max_mps_over_sides"),
                        "lambda_mean": row.get("lambda_mean"),
                    },
                    endpoint_details={"source": str(path), "note": "direct/lambda proxy; pop_safe false is binding"},
                )
            )
    return rows


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
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_case: Dict[str, Dict[str, Any]] = {}
    by_case_target: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case = str(row.get("case"))
        target = str(row.get("target"))
        key = f"{case}|{target}"
        for table, table_key in ((by_case, case), (by_case_target, key)):
            rec = table.setdefault(table_key, {"n": 0, "pass_n": 0, "failed_family_counts": {}})
            rec["n"] += 1
            if row.get("pass"):
                rec["pass_n"] += 1
            failed = str(row.get("failed_family") or "")
            for fam in [x for x in failed.split(",") if x]:
                rec["failed_family_counts"][fam] = rec["failed_family_counts"].get(fam, 0) + 1
    for table in (by_case, by_case_target):
        for rec in table.values():
            rec["pass_rate"] = float(rec["pass_n"] / max(1, rec["n"]))
    return {"by_case": by_case, "by_case_target": by_case_target}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Metric-only replay for middle-generator acceptance v0.")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    p.add_argument("--two-frame-summary", type=Path, default=DEFAULT_TWO_FRAME)
    p.add_argument("--bone-bridge-summary", type=Path, default=DEFAULT_BONE_BRIDGE)
    p.add_argument("--regime-bridge-summary", type=Path, default=DEFAULT_REGIME_BRIDGE)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--baseline-quantile", type=float, default=99.5)
    p.add_argument("--bridge-budget-quantile", type=float, default=95.0)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_middle_acceptance_replay_probe_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_clips(Path(args.npz_root), Path(args.z_features))
    skeleton = _load_skeleton_meta(Path(args.npz_root), WALK_F)
    bands = _calibrate_baselines(clips, skeleton, quantile=float(args.baseline_quantile))

    bridge_budgets: Dict[str, Dict[str, float]] = {}
    for name, clip in clips.items():
        bridge_budgets[name] = {
            "angvel": _safe_percentile(_step_angvel_rms(clip.bone_angvel), float(args.bridge_budget_quantile)),
            "rootvel": _safe_percentile(_step_l2(clip.root_vel), float(args.bridge_budget_quantile)),
            "yaw": _safe_percentile(np.abs(np.diff(clip.yaw_rate)), float(args.bridge_budget_quantile)),
            "contact": _safe_percentile(_step_l2(clip.contact), float(args.bridge_budget_quantile)),
        }

    rows: List[Dict[str, Any]] = []
    h = int(args.horizon)
    stride = max(1, int(args.stride))

    # Positive oracle: real continuous windows must mostly pass under their own regime.
    for clip_name, clip in clips.items():
        max_start = int(clip.rot6d.shape[0]) - h
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            rows.append(
                _evaluate_sequence(
                    _make_sequence(clip, start, h),
                    target=clip_name,
                    target_bands=bands[clip_name],
                    skeleton=skeleton,
                    case="positive_oracle:real_continuous",
                    expected_label="pass",
                    start_phase=f"{clip_name}[{start}:{start + h}]",
                    endpoint_bridgeability=True,
                    endpoint_details={"source": "raw continuous clip"},
                )
            )

    # Matched seams and synthetic one-frame switches.
    two_frame = json.loads(Path(args.two_frame_summary).read_text(encoding="utf-8"))
    matched_pairs = two_frame.get("matched_pairs", {}) or {}
    walk = clips[WALK_F]
    endpoint_rows: List[Dict[str, Any]] = []
    for target in TURN_CLIPS:
        target_clip = clips[target]
        align = full_state_align(
            clips[WALK_F].state281,
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
        endpoint_rows.append({"target": target, **bridge})

        if not pair:
            rows.append(
                _artifact_row(
                    case="endpoint_bridgeability:ungroundable_candidate",
                    target=target,
                    start_phase=f"phi={phi};onset=0",
                    expected_label="separate_not_groundable",
                    families={
                        "regime_reached": False,
                        "rate_budget": False,
                        "support_honesty": False,
                        "command_response": False,
                        "pose_continuity": bool(align.full_state_pose_d <= float(args.ground_pose_thr)),
                        "endpoint_bridgeability": False,
                    },
                    metrics={"pose_d": align.full_state_pose_d, "contact_d": align.full_state_contact_d},
                    endpoint_details=bridge,
                )
            )
            continue

        rows.append(
            _evaluate_sequence(
                _make_hard_seam_sequence(walk, target_clip, phi, onset),
                target=target,
                target_bands=bands[target],
                skeleton=skeleton,
                case="negative_control:matched_hard_seam",
                expected_label="fail",
                start_phase=f"phi={phi};onset={onset}",
                endpoint_bridgeability=bool(bridge["one_frame_bridgeable"]),
                endpoint_details=bridge,
            )
        )
        rows.append(
            _evaluate_sequence(
                _make_one_frame_switch_sequence(walk, target_clip, phi, onset),
                target=target,
                target_bands=bands[target],
                skeleton=skeleton,
                case="negative_control:one_frame_angvel_root_switch",
                expected_label="fail",
                start_phase=f"phi={phi};onset={onset}",
                endpoint_bridgeability=bool(bridge["one_frame_bridgeable"]),
                endpoint_details=bridge,
            )
        )
        rows.append(
            _evaluate_sequence(
                _make_linear_proxy_sequence(walk, target_clip, phi, onset, h),
                target=target,
                target_bands=bands[target],
                skeleton=skeleton,
                case="negative_control:linear_pose_contact_proxy",
                expected_label="fail",
                start_phase=f"phi={phi};onset={onset};H={h}",
                endpoint_bridgeability=bool(bridge["horizon_bridgeable"]),
                endpoint_details=bridge,
            )
        )

    rows.extend(_rows_from_bone_bridge_artifact(Path(args.bone_bridge_summary)))
    rows.extend(_rows_from_regime_bridge_artifact(Path(args.regime_bridge_summary)))

    summary = _summarize_rows(rows)
    verdict = {
        "real_continuous_pass_rate": summary["by_case"].get("positive_oracle:real_continuous", {}).get("pass_rate"),
        "matched_hard_seam_pass_rate": summary["by_case"].get("negative_control:matched_hard_seam", {}).get("pass_rate"),
        "one_frame_switch_pass_rate": summary["by_case"].get("negative_control:one_frame_angvel_root_switch", {}).get("pass_rate"),
        "linear_proxy_pass_rate": summary["by_case"].get("negative_control:linear_pose_contact_proxy", {}).get("pass_rate"),
        "direct_lambda_pass_rates": {
            key: rec.get("pass_rate")
            for key, rec in summary["by_case"].items()
            if key.startswith("negative_control:") and key not in {
                "negative_control:matched_hard_seam",
                "negative_control:one_frame_angvel_root_switch",
                "negative_control:linear_pose_contact_proxy",
            }
        },
        "proxy_replay_pass_rates": {
            key: rec.get("pass_rate") for key, rec in summary["by_case"].items() if key.startswith("proxy_replay:")
        },
        "walk_l_to_r_reported_separately": any(
            r.get("target") == "Walk_L_To_R" and str(r.get("case", "")).startswith("endpoint_bridgeability")
            for r in rows
        ),
    }

    payload = {
        "task": "middle_acceptance_replay_probe",
        "scope": "read-only metric replay; no training; no train owner path edits",
        "config": {
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "horizon": h,
            "stride": stride,
            "baseline_quantile": float(args.baseline_quantile),
            "bridge_budget_quantile": float(args.bridge_budget_quantile),
            "contact_threshold": CONTACT_THRESHOLD,
        },
        "input_contract": {
            "handoff_state": "[T,281] float32/cpu from load_clip_states",
            "base_state": "[T,419] float32/cpu available in processed npz",
            "bone_angvel": "[T,138] float32/cpu from bone_ang_vel",
            "contact": "[T,2] float32/cpu from z-feature future_desc contact slice",
        },
        "baseline_bands": bands,
        "bridge_budgets": bridge_budgets,
        "endpoint_bridgeability": endpoint_rows,
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
    }

    _dump_json(out_dir / "middle_acceptance_replay_summary.json", payload)
    _write_csv(out_dir / "middle_acceptance_replay_rows.csv", rows)

    lines: List[str] = []
    lines.append("# Middle Acceptance Replay Probe")
    lines.append("")
    lines.append("Read-only metric replay. No training, checkpoint mutation, production gate change, or `train/` owner edit.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- real continuous pass rate: `{_fmt(verdict['real_continuous_pass_rate'])}`")
    lines.append(f"- matched hard seam pass rate: `{_fmt(verdict['matched_hard_seam_pass_rate'])}`")
    lines.append(f"- one-frame angvel/root switch pass rate: `{_fmt(verdict['one_frame_switch_pass_rate'])}`")
    lines.append(f"- linear pose/contact proxy pass rate: `{_fmt(verdict['linear_proxy_pass_rate'])}`")
    lines.append(f"- Walk_L_To_R separately reported: `{bool(verdict['walk_l_to_r_reported_separately'])}`")
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    lines.append("| case | n | pass_rate | top failed families |")
    lines.append("|---|---:|---:|---|")
    for case, rec in sorted(summary["by_case"].items()):
        failed_counts = rec.get("failed_family_counts", {}) or {}
        top_failed = ", ".join(f"{k}:{v}" for k, v in sorted(failed_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4])
        lines.append(f"| {case} | {int(rec.get('n', 0))} | {_fmt(rec.get('pass_rate'))} | {top_failed or '-'} |")
    lines.append("")
    lines.append("## Endpoint Bridgeability")
    lines.append("")
    lines.append("| target | groundable | one_frame | horizon | max_needed | support | pose_d | contact_d |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for row in endpoint_rows:
        support = f"{row.get('support_start')}->{row.get('support_end')}"
        max_needed = row.get("max_needed_frames")
        max_s = "inf" if max_needed is not None and int(max_needed) >= int(10**9) else str(max_needed)
        lines.append(
            f"| {row['target']} | {bool(row.get('groundable'))} | {bool(row.get('one_frame_bridgeable'))} | "
            f"{bool(row.get('horizon_bridgeable'))} | {max_s} | {support} | "
            f"{_fmt(row.get('pose_d'))} | {_fmt(row.get('contact_d'))} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{out_dir / 'middle_acceptance_replay_summary.json'}`")
    lines.append(f"- `{out_dir / 'middle_acceptance_replay_rows.csv'}`")
    _dump_md(out_dir / "middle_acceptance_replay_summary.md", lines)

    print(f"wrote {out_dir / 'middle_acceptance_replay_summary.md'}")
    print(f"wrote {out_dir / 'middle_acceptance_replay_summary.json'}")


if __name__ == "__main__":
    main()
