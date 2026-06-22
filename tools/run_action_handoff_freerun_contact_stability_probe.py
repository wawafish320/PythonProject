#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.data.action_handoff_inbetween import CONTACT_SLICE, WALK_F, load_clip_states  # noqa: E402
from train.geometry import geodesic_R, rot6d_to_matrix  # noqa: E402
from train.history import advance_pose_hist_state_with_tail  # noqa: E402
from train.validate.run_freerun_cycles import (  # noqa: E402
    FreeRunCycleRunner,
    _init_eval_pose_hist_state,
    _resolve_eval_pose_hist_input,
)


DATE_TAG = "20260606"
DEFAULT_STEM = "20260606_freerun_contact_stability_probe"
CALIBRATION_PATH = ROOT / "debug_output" / "20260606_gap_selection_goal_contract_calibration.py"
CONTACT_FOOT_NAMES = ("right", "left")
WITNESS_SIGNAL = "plan_contact"
CARRY_ECHO_SIGNAL = "carry_contact_echo"
SOFT_SIGNALS = ("contacts_meas", "dist_score", "vz_score", "vxy_score", WITNESS_SIGNAL, CARRY_ECHO_SIGNAL)
SOFT_COMPONENT_SIGNALS = ("dist_score", "vz_score", "vxy_score")
SOFT_DECAY_EPS = 0.02
GAP_BUCKET_WIDTH = 12
LAYER1_SIGNALS = (
    "pose_step_geo_deg",
    "pose_manifold_z_rms",
    "pose_knn1_z_rms",
    "angvel_abs_rms",
    "angvel_z_rms",
    "angvel_step_rms",
    "rootvel_norm",
    "rootpos_step_norm",
    "rootpos_from_start_norm",
)


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    case_kind: str
    source_clip: str
    source_frame: int
    target_clip: str
    target_frame: int
    independent_unit: str
    source_region_key: str
    target_region_key: str
    start_label: str
    phase_bin: str
    cycle_bin8: int
    goal_max_gap: int


@dataclass(frozen=True)
class Layer1Baseline:
    pose_mu: np.ndarray
    pose_std: np.ndarray
    pose_z: np.ndarray
    angvel_mu: Optional[np.ndarray]
    angvel_std: Optional[np.ndarray]
    z_std_floor: float
    pose_raw_std_min: float
    pose_raw_std_p50: float
    pose_raw_std_p95: float
    pose_std_floored_n: int
    angvel_raw_std_min: Optional[float]
    angvel_raw_std_p50: Optional[float]
    angvel_raw_std_p95: Optional[float]
    angvel_std_floored_n: Optional[int]
    walk_frames: int
    pose_dim: int
    joint_count: int
    columns: tuple[str, str]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _gap_bucket_label(gap: int, *, min_gap: int = 12, width: int = GAP_BUCKET_WIDTH) -> str:
    g = int(gap)
    w = max(1, int(width))
    lo = int(min_gap) + ((g - int(min_gap)) // w) * w
    hi = lo + w - 1
    return f"{lo:02d}-{hi:02d}"


def _coerce_pair_series(value: Any, *, expected_rows: Optional[int] = None, expected_cols: int = 2) -> np.ndarray:
    rows = int(expected_rows) if expected_rows is not None else None
    cols = max(1, int(expected_cols))
    arr = np.asarray(value if value is not None else [], dtype=np.float64)
    if arr.size == 0:
        return np.full((max(0, rows or 0), cols), np.nan, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % cols == 0:
            arr = arr.reshape(-1, cols)
        else:
            tmp = np.full((1, cols), np.nan, dtype=np.float64)
            n = min(cols, int(arr.size))
            tmp[0, :n] = arr[:n]
            arr = tmp
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        arr = np.full((max(0, rows or 0), cols), np.nan, dtype=np.float64)
    if arr.shape[1] != cols:
        tmp = np.full((arr.shape[0], cols), np.nan, dtype=np.float64)
        n = min(cols, int(arr.shape[1]))
        if n > 0:
            tmp[:, :n] = arr[:, :n]
        arr = tmp
    if rows is not None and arr.shape[0] != rows:
        tmp = np.full((max(0, rows), cols), np.nan, dtype=np.float64)
        n = min(int(tmp.shape[0]), int(arr.shape[0]))
        if n > 0:
            tmp[:n] = arr[:n]
        arr = tmp
    return arr.astype(np.float64, copy=False)


def _finite_values(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _mean_array(value: Any) -> float:
    vals = _finite_values(value)
    return float(vals.mean()) if vals.size else float("nan")


def _percentile_array(value: Any, q: float) -> float:
    vals = _finite_values(value)
    return float(np.percentile(vals, float(q))) if vals.size else float("nan")


def _mean_by_step(arr: np.ndarray) -> list[float]:
    out: list[float] = []
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0:
        val = float(a)
        return [val if math.isfinite(val) else float("nan")]
    if a.ndim == 1:
        for val in a:
            fv = float(val)
            out.append(fv if math.isfinite(fv) else float("nan"))
        return out
    for row in a:
        row_arr = np.asarray(row, dtype=np.float64).reshape(-1)
        vals = row_arr[np.isfinite(row_arr)]
        out.append(float(vals.mean()) if vals.size else float("nan"))
    return out


def _series_json(arr: np.ndarray) -> str:
    return json.dumps(_jsonable(np.asarray(arr, dtype=np.float64)), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _json_slice(value: Optional[slice]) -> str:
    if not isinstance(value, slice):
        return "null"
    return json.dumps([value.start, value.stop, value.step], separators=(",", ":"))


def _slice_width_checked(value: Optional[slice], *, name: str) -> int:
    if not isinstance(value, slice) or value.start is None or value.stop is None:
        raise ValueError(f"{name} must be a concrete slice")
    width = int(value.stop) - int(value.start)
    if width <= 0:
        raise ValueError(f"{name} has non-positive width: {value}")
    return width


def _series_from_row_any(row: Mapping[str, Any], signal: str) -> np.ndarray:
    raw = row.get(f"{signal}_series", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    return np.asarray(raw, dtype=np.float64)


def _stat_series(prefix: str, arr: Any) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        f"{prefix}_mean": _mean_array(a),
        f"{prefix}_terminal_mean": _terminal_mean(a),
        f"{prefix}_first_quarter_mean": _window_mean(a, tail=False),
        f"{prefix}_last_quarter_mean": _window_mean(a, tail=True),
        f"{prefix}_min": _percentile_array(a, 0),
        f"{prefix}_p05": _percentile_array(a, 5),
        f"{prefix}_p50": _percentile_array(a, 50),
        f"{prefix}_p95": _percentile_array(a, 95),
        f"{prefix}_max": _percentile_array(a, 100),
        f"{prefix}_slope_per_frame": _linear_slope(a),
        f"{prefix}_finite_n": int(np.isfinite(a).sum()),
        f"{prefix}_series": _series_json(a),
    }


def _as_batch(sample: Mapping[str, torch.Tensor], key: str, device: torch.device) -> Optional[torch.Tensor]:
    value = sample.get(key)
    if not torch.is_tensor(value):
        return None
    return value.unsqueeze(0).to(device=device, dtype=torch.float32)


def _squeeze_time(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value):
        return None
    out = value.detach()
    if out.dim() == 3 and out.shape[1] == 1:
        out = out[:, 0]
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


def _compose_raw(trainer: Any, ret: Mapping[str, Any], y_prev_raw: torch.Tensor, step_idx: int, total_steps: int, apply_lambda: bool) -> torch.Tensor:
    y_inc_raw = trainer._compose_delta_to_raw(
        y_prev_raw,
        ret["out"],
        omega_hat=ret.get("omega_hat", None) if bool(getattr(trainer, "so3_corr_apply", False)) else None,
        so3_gate=getattr(trainer, "so3_corr_gate_force", None),
        so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
    )
    if bool(apply_lambda) and torch.is_tensor(ret.get("out_direct", None)) and torch.is_tensor(ret.get("lambda_fusion", None)):
        lam_eff, _ = trainer._lambda_fusion_apply_reliability(
            ret["lambda_fusion"],
            step_idx=int(step_idx),
            total_steps=int(total_steps),
            ret=dict(ret),
        )
        return trainer._apply_lambda_fusion_to_raw(y_inc_raw, direct_norm=ret["out_direct"], lambda_fusion=lam_eff)
    return y_inc_raw


def _state_angvel(motion: torch.Tensor, cfg: Any) -> Optional[torch.Tensor]:
    sl = getattr(cfg, "angvel_x_slice", None)
    if isinstance(sl, slice):
        return motion[..., sl].detach()
    return None


def _time_inputs(motion: torch.Tensor, step: int, total_steps: int) -> tuple[int, torch.Tensor]:
    denom = max(1, int(total_steps) - 1)
    rollout_step = torch.full((motion.shape[0], 1, 1), float(step) / float(denom), device=motion.device, dtype=motion.dtype)
    return int(step), rollout_step


def _initial_state_raw_from_sample(trainer: Any, sample: Mapping[str, torch.Tensor], device: torch.device) -> np.ndarray:
    state_seq = _as_batch(sample, "motion", device)
    if state_seq is None or state_seq.shape[1] <= 0:
        return np.zeros((0,), dtype=np.float64)
    raw = trainer.normalizer.denorm_x(state_seq[:, 0].detach())
    return raw.detach().cpu().numpy()[0].astype(np.float64, copy=False)


def _series_from_row(row: Mapping[str, Any], signal: str) -> np.ndarray:
    raw = row.get(f"{signal}_series", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    expected = int(row.get("frames", 0) or 0)
    return _coerce_pair_series(raw, expected_rows=expected, expected_cols=2)


def _window_mean(arr: np.ndarray, *, tail: bool) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[0] <= 0:
        return float("nan")
    n = max(1, int(math.ceil(float(a.shape[0]) * 0.25)))
    return _mean_array(a[-n:] if tail else a[:n])


def _terminal_mean(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[0] <= 0:
        return float("nan")
    return _mean_array(a[-1:])


def _linear_slope(arr: np.ndarray) -> float:
    vals = np.asarray(_mean_by_step(arr), dtype=np.float64)
    ok = np.isfinite(vals)
    if int(ok.sum()) < 2:
        return float("nan")
    x = np.arange(vals.shape[0], dtype=np.float64)[ok]
    y = vals[ok]
    x = x - float(x.mean())
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y - float(y.mean())) / denom)


def _soft_signal_series(seq: Mapping[str, Any]) -> dict[str, np.ndarray]:
    contacts = _coerce_pair_series(seq.get("contacts"), expected_cols=2)
    frames = int(contacts.shape[0])
    comps = seq.get("contact_score_components", {})
    if not isinstance(comps, Mapping):
        comps = {}
    plan_contact = _coerce_pair_series(seq.get("plan_probs"), expected_rows=frames, expected_cols=2)
    state_raw = np.asarray(seq.get("state_raw", []), dtype=np.float64)
    if state_raw.ndim == 2 and state_raw.shape[1] >= int(CONTACT_SLICE.stop or 0):
        carry_echo = _coerce_pair_series(state_raw[:, CONTACT_SLICE], expected_rows=frames, expected_cols=2)
    else:
        carry_echo = np.full((frames, 2), np.nan, dtype=np.float64)
    return {
        "contacts_meas": contacts,
        "dist_score": _coerce_pair_series(comps.get("dist_score"), expected_rows=frames, expected_cols=2),
        "vz_score": _coerce_pair_series(comps.get("vz_score"), expected_rows=frames, expected_cols=2),
        "vxy_score": _coerce_pair_series(comps.get("vxy_score"), expected_rows=frames, expected_cols=2),
        WITNESS_SIGNAL: plan_contact,
        CARRY_ECHO_SIGNAL: carry_echo,
    }


def _soft_stats(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    a = _coerce_pair_series(arr, expected_cols=2)
    stats: dict[str, Any] = {
        f"{prefix}_mean": _mean_array(a),
        f"{prefix}_terminal_mean": _terminal_mean(a),
        f"{prefix}_first_quarter_mean": _window_mean(a, tail=False),
        f"{prefix}_last_quarter_mean": _window_mean(a, tail=True),
        f"{prefix}_min": _percentile_array(a, 0),
        f"{prefix}_p05": _percentile_array(a, 5),
        f"{prefix}_p50": _percentile_array(a, 50),
        f"{prefix}_p95": _percentile_array(a, 95),
        f"{prefix}_max": _percentile_array(a, 100),
        f"{prefix}_slope_per_frame": _linear_slope(a),
        f"{prefix}_finite_n": int(np.isfinite(a).sum()),
        f"{prefix}_series": _series_json(a),
    }
    for idx, foot in enumerate(CONTACT_FOOT_NAMES):
        stats[f"{prefix}_{foot}_mean"] = _mean_array(a[:, idx]) if a.shape[1] > idx else float("nan")
        stats[f"{prefix}_{foot}_terminal"] = _mean_array(a[-1:, idx]) if a.shape[0] > 0 and a.shape[1] > idx else float("nan")
    return stats


def _run_sequence_layer1_only(
    *,
    trainer: Any,
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    mode: str,
    apply_lambda: bool,
) -> dict[str, Any]:
    state_seq = _as_batch(sample, "motion", device)
    gt_seq = _as_batch(sample, "gt_motion", device)
    cond_seq = _as_batch(sample, "cond_in", device)
    cond_raw_seq = _as_batch(sample, "cond_tgt_raw", device)
    pose_hist_seq = _as_batch(sample, "pose_hist", device)
    if state_seq is None or gt_seq is None or cond_seq is None or cond_raw_seq is None:
        raise RuntimeError("sample missing required tensors")
    total = int(state_seq.shape[1])
    cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    if not (isinstance(cfg.rot6d_x_slice, slice) and isinstance(cfg.rot6d_y_slice, slice)):
        raise RuntimeError("free carry rot6d slices missing")
    if not (isinstance(cfg.rootvel_x_slice, slice) and isinstance(cfg.rootpos_x_slice, slice)):
        raise RuntimeError("free carry root slices missing")

    motion = state_seq[:, 0].detach()
    motion_raw = trainer.normalizer.denorm_x(motion)
    initial_state_raw = motion_raw.detach().cpu().numpy()[0].astype(np.float64, copy=False)
    y_prev_raw = trainer._denorm(gt_seq[:, 0])
    pose_hist_state = _init_eval_pose_hist_state(
        trainer,
        ref_tensor=state_seq,
        pose_hist_seq=pose_hist_seq,
        step=0,
        device=device,
        dtype=state_seq.dtype,
    )
    plan_z = None
    y_raw_out: list[np.ndarray] = []
    state_raw_out: list[np.ndarray] = []
    plan_probs: list[np.ndarray] = []
    plan_norms: list[float] = []
    direct_norms: list[float] = []
    lambda_means: list[float] = []
    rootvel_y: list[np.ndarray] = []
    ang_step_vals: list[float] = []
    contact_dim = int(getattr(model, "contact_dim", 2) or 2)

    model.eval()
    with torch.no_grad():
        for t in range(total - 1):
            if mode == "teacher":
                motion = state_seq[:, t].detach()
                motion_raw = trainer.normalizer.denorm_x(motion)
                y_prev_raw = trainer._denorm(gt_seq[:, t])
                pose_hist_t = _resolve_eval_pose_hist_input(
                    state=pose_hist_state,
                    pose_hist_seq=pose_hist_seq,
                    idx=t,
                    source="seq",
                    batch_size=int(motion.shape[0]),
                    device=device,
                    dtype=motion.dtype,
                )
            else:
                pose_hist_t = _resolve_eval_pose_hist_input(
                    state=pose_hist_state,
                    pose_hist_seq=pose_hist_seq,
                    idx=t,
                    source="buffer",
                    batch_size=int(motion.shape[0]),
                    device=device,
                    dtype=motion.dtype,
                )

            ti, rs = _time_inputs(motion, t, total - 1)
            ret = model(
                motion,
                cond_seq[:, t],
                contacts=None,
                angvel=_state_angvel(motion, cfg),
                pose_history=pose_hist_t,
                plan_z=plan_z,
                phase_z=None,
                phase_event_age=None,
                meas_logits_prev=None,
                time_index=ti,
                rollout_step=rs,
            )
            if not isinstance(ret, dict) or not torch.is_tensor(ret.get("out", None)):
                raise RuntimeError("model forward did not return ret['out']")
            y_used_raw = _compose_raw(trainer, ret, y_prev_raw, t, total - 1, apply_lambda)
            cond_next_raw = cond_raw_seq[:, min(t, cond_raw_seq.shape[1] - 1)]
            next_raw = _rollout_kernel.apply_free_carry_raw(
                x_prev=motion_raw,
                y_next_raw=y_used_raw,
                cond_next_raw=cond_next_raw,
                rot6d_x_slice=cfg.rot6d_x_slice,
                rot6d_y_slice=cfg.rot6d_y_slice,
                angvel_x_slice=cfg.angvel_x_slice,
                rootvel_x_slice=cfg.rootvel_x_slice,
                rootpos_x_slice=cfg.rootpos_x_slice,
                bone_hz=cfg.bone_hz,
                columns=cfg.columns,
            )

            cp = _squeeze_time(ret.get("contacts_plan", None))
            if torch.is_tensor(cp):
                plan_probs.append(cp.cpu().numpy()[0].astype(float))
            else:
                plan_probs.append(np.full((contact_dim,), np.nan, dtype=float))
            pn = _squeeze_time(ret.get("plan_z_next", None))
            plan_norms.append(float(pn.norm(dim=-1).mean().item()) if torch.is_tensor(pn) else float("nan"))
            od = _squeeze_time(ret.get("out_direct", None))
            direct_norms.append(float(od.norm(dim=-1).mean().item()) if torch.is_tensor(od) else float("nan"))
            lf = _squeeze_time(ret.get("lambda_fusion", None))
            lambda_means.append(float(lf.mean().item()) if torch.is_tensor(lf) else float("nan"))
            y_raw_out.append(y_used_raw.detach().cpu().numpy()[0].astype(float))
            state_raw_out.append(next_raw.detach().cpu().numpy()[0].astype(float))
            y_root_sl = getattr(trainer, "rootvel_slice", None)
            if isinstance(y_root_sl, slice):
                rootvel_y.append(y_used_raw[..., y_root_sl].detach().cpu().numpy()[0].astype(float))
            if isinstance(cfg.angvel_x_slice, slice):
                prev_av = motion_raw[..., cfg.angvel_x_slice]
                next_av = next_raw[..., cfg.angvel_x_slice]
                d_av = (next_av - prev_av).reshape(next_av.shape[0], -1)
                ang_step_vals.append(float(torch.sqrt(torch.mean(d_av * d_av)).item()))

            plan_z_next = ret.get("plan_z_next", None)
            plan_z = plan_z_next.detach() if torch.is_tensor(plan_z_next) else None
            if mode == "free":
                y_prev_raw = y_used_raw.detach()
                motion_raw = next_raw.detach()
                motion = trainer._diag_norm_x(motion_raw).detach()
                pose_hist_state = advance_pose_hist_state_with_tail(
                    pose_hist_state,
                    rot_tail_raw=y_used_raw[..., cfg.rot6d_y_slice],
                )

    frames = max(0, total - 1)
    nan_contacts = np.full((frames, contact_dim), np.nan, dtype=np.float64)
    return {
        "contacts": nan_contacts,
        "markers": np.zeros((0,), dtype=np.float64),
        "y_raw": np.asarray(y_raw_out, dtype=np.float64),
        "state_raw": np.asarray(state_raw_out, dtype=np.float64),
        "initial_state_raw": np.asarray(initial_state_raw, dtype=np.float64),
        "contact_score_components": {
            "dist_score": nan_contacts.copy(),
            "vz_score": nan_contacts.copy(),
            "vxy_score": nan_contacts.copy(),
        },
        "plan_probs": np.asarray(plan_probs, dtype=np.float64),
        "plan_norms": np.asarray(plan_norms, dtype=np.float64),
        "direct_norms": np.asarray(direct_norms, dtype=np.float64),
        "lambda_means": np.asarray(lambda_means, dtype=np.float64),
        "rootvel_y": np.asarray(rootvel_y, dtype=np.float64),
        "ang_step_rms": np.asarray(ang_step_vals, dtype=np.float64),
        "tensor_meta": {
            "cond": {
                "shape": [int(x) for x in cond_seq.shape],
                "dtype": str(cond_seq.dtype).replace("torch.", ""),
                "device": str(cond_seq.device),
                "finite": bool(torch.isfinite(cond_seq).all().item()),
            },
            "state_raw": {
                "shape": [int(x) for x in np.asarray(state_raw_out, dtype=np.float64).shape],
                "dtype": "float64",
                "device": "cpu",
                "finite": bool(np.isfinite(np.asarray(state_raw_out, dtype=np.float64)).all()) if state_raw_out else True,
            },
            "layer1_no_fk": True,
        },
    }


def _std_stat(std: np.ndarray, q: float) -> float:
    vals = np.asarray(std, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, float(q))) if vals.size else float("nan")


def _build_layer1_baseline(walk_raw_state: np.ndarray, cfg: Any, *, z_std_floor: float) -> Layer1Baseline:
    rot_sl = getattr(cfg, "rot6d_x_slice", None)
    rot_width = _slice_width_checked(rot_sl, name="rot6d_x_slice")
    if rot_width % 6 != 0:
        raise ValueError(f"rot6d_x_slice width must be divisible by 6, got {rot_width}")
    walk = np.asarray(walk_raw_state, dtype=np.float64)
    if walk.ndim != 2 or walk.shape[1] < int(rot_sl.stop or 0):
        raise ValueError(f"walk_raw_state must be [T,Dx] with rot6d_x_slice available, got {walk.shape}")
    floor = max(float(z_std_floor), 1e-8)
    pose = np.asarray(walk[:, rot_sl], dtype=np.float64)
    pose_mu = np.nanmean(pose, axis=0)
    pose_std_raw = np.nanstd(pose, axis=0)
    pose_std = np.nan_to_num(pose_std_raw, nan=floor, posinf=floor, neginf=floor)
    pose_std_floored_n = int((pose_std < floor).sum())
    pose_std = np.clip(pose_std, floor, None)
    pose_mu = np.nan_to_num(pose_mu, nan=0.0, posinf=0.0, neginf=0.0)
    pose_z = np.nan_to_num((pose - pose_mu) / pose_std, nan=0.0, posinf=0.0, neginf=0.0)

    ang_mu = None
    ang_std = None
    ang_raw_std_min = None
    ang_raw_std_p50 = None
    ang_raw_std_p95 = None
    ang_std_floored_n = None
    av_sl = getattr(cfg, "angvel_x_slice", None)
    if isinstance(av_sl, slice):
        _slice_width_checked(av_sl, name="angvel_x_slice")
        av = np.asarray(walk[:, av_sl], dtype=np.float64)
        ang_mu = np.nanmean(av, axis=0)
        ang_std_raw = np.nanstd(av, axis=0)
        ang_raw_std_min = _std_stat(ang_std_raw, 0)
        ang_raw_std_p50 = _std_stat(ang_std_raw, 50)
        ang_raw_std_p95 = _std_stat(ang_std_raw, 95)
        ang_std = np.nan_to_num(ang_std_raw, nan=floor, posinf=floor, neginf=floor)
        ang_std_floored_n = int((ang_std < floor).sum())
        ang_std = np.clip(ang_std, floor, None)
        ang_mu = np.nan_to_num(ang_mu, nan=0.0, posinf=0.0, neginf=0.0)

    cols_raw = tuple(getattr(cfg, "columns", ("X", "Z")))
    columns = (str(cols_raw[0]).upper(), str(cols_raw[1]).upper()) if len(cols_raw) >= 2 else ("X", "Z")
    return Layer1Baseline(
        pose_mu=pose_mu.astype(np.float64, copy=False),
        pose_std=pose_std.astype(np.float64, copy=False),
        pose_z=pose_z.astype(np.float64, copy=False),
        angvel_mu=None if ang_mu is None else ang_mu.astype(np.float64, copy=False),
        angvel_std=None if ang_std is None else ang_std.astype(np.float64, copy=False),
        z_std_floor=float(floor),
        pose_raw_std_min=_std_stat(pose_std_raw, 0),
        pose_raw_std_p50=_std_stat(pose_std_raw, 50),
        pose_raw_std_p95=_std_stat(pose_std_raw, 95),
        pose_std_floored_n=int(pose_std_floored_n),
        angvel_raw_std_min=ang_raw_std_min,
        angvel_raw_std_p50=ang_raw_std_p50,
        angvel_raw_std_p95=ang_raw_std_p95,
        angvel_std_floored_n=ang_std_floored_n,
        walk_frames=int(walk.shape[0]),
        pose_dim=int(pose.shape[1]),
        joint_count=int(rot_width // 6),
        columns=(columns[0], columns[1]),
    )


def _denorm_walk_raw_state(trainer: Any, clip: Any, device: torch.device) -> np.ndarray:
    x_norm = np.asarray(getattr(clip, "X"), dtype=np.float32)
    with torch.no_grad():
        raw = trainer.normalizer.denorm_x(torch.from_numpy(x_norm).to(device=device, dtype=torch.float32))
    return raw.detach().cpu().numpy().astype(np.float64, copy=False)


def _pose_step_geodesic_deg(pose_seq_flat: np.ndarray, *, joint_count: int, columns: Sequence[str]) -> np.ndarray:
    pose_seq = np.asarray(pose_seq_flat, dtype=np.float64)
    if pose_seq.ndim != 2 or pose_seq.shape[0] < 2 or pose_seq.shape[1] != int(joint_count) * 6:
        return np.full((0, int(joint_count)), np.nan, dtype=np.float64)
    with torch.no_grad():
        rot = torch.as_tensor(pose_seq.reshape(pose_seq.shape[0], int(joint_count), 6), dtype=torch.float64)
        mats = rot6d_to_matrix(rot, columns=tuple(columns))
        geo = geodesic_R(mats[1:], mats[:-1], reduce=None) * (180.0 / math.pi)
    return geo.detach().cpu().numpy().astype(np.float64, copy=False)


def _nearest_z_rms(query_z: np.ndarray, baseline_z: np.ndarray) -> np.ndarray:
    q = np.asarray(query_z, dtype=np.float64)
    b = np.asarray(baseline_z, dtype=np.float64)
    if q.ndim != 2 or b.ndim != 2 or q.shape[1] != b.shape[1] or q.shape[0] <= 0 or b.shape[0] <= 0:
        return np.full((max(0, q.shape[0] if q.ndim >= 1 else 0),), np.nan, dtype=np.float64)
    out: list[float] = []
    for row in q:
        diff = b - row.reshape(1, -1)
        dist = np.sqrt(np.mean(diff * diff, axis=1))
        vals = dist[np.isfinite(dist)]
        out.append(float(vals.min()) if vals.size else float("nan"))
    return np.asarray(out, dtype=np.float64)


def _layer1_series(seq: Mapping[str, Any], cfg: Any, baseline: Layer1Baseline) -> dict[str, np.ndarray]:
    state_raw = np.asarray(seq.get("state_raw", []), dtype=np.float64)
    if state_raw.ndim == 1:
        state_raw = state_raw.reshape(1, -1)
    initial = np.asarray(seq.get("initial_state_raw", []), dtype=np.float64).reshape(-1)
    if initial.size == state_raw.shape[-1] and state_raw.ndim == 2:
        full_state = np.concatenate([initial.reshape(1, -1), state_raw], axis=0)
    else:
        full_state = state_raw
    frames = int(state_raw.shape[0]) if state_raw.ndim == 2 else 0

    rot_sl = getattr(cfg, "rot6d_x_slice", None)
    pose = state_raw[:, rot_sl] if frames > 0 and isinstance(rot_sl, slice) else np.zeros((0, baseline.pose_dim), dtype=np.float64)
    full_pose = full_state[:, rot_sl] if full_state.ndim == 2 and isinstance(rot_sl, slice) else pose
    pose_z = np.nan_to_num((pose - baseline.pose_mu) / baseline.pose_std, nan=0.0, posinf=0.0, neginf=0.0)
    pose_manifold = np.sqrt(np.mean(pose_z * pose_z, axis=1)) if pose_z.size else np.full((frames,), np.nan, dtype=np.float64)
    pose_knn = _nearest_z_rms(pose_z, baseline.pose_z)
    pose_step = _pose_step_geodesic_deg(full_pose, joint_count=baseline.joint_count, columns=baseline.columns)

    av_sl = getattr(cfg, "angvel_x_slice", None)
    if isinstance(av_sl, slice) and frames > 0:
        av = np.asarray(state_raw[:, av_sl], dtype=np.float64)
        ang_abs = np.sqrt(np.mean(av * av, axis=1)) if av.size else np.full((frames,), np.nan, dtype=np.float64)
        if baseline.angvel_mu is not None and baseline.angvel_std is not None and av.shape[1] == baseline.angvel_mu.shape[0]:
            av_z = np.nan_to_num((av - baseline.angvel_mu) / baseline.angvel_std, nan=0.0, posinf=0.0, neginf=0.0)
            ang_z = np.sqrt(np.mean(av_z * av_z, axis=1))
        else:
            ang_z = np.full((frames,), np.nan, dtype=np.float64)
    else:
        ang_abs = np.full((frames,), np.nan, dtype=np.float64)
        ang_z = np.full((frames,), np.nan, dtype=np.float64)
    ang_step = np.asarray(seq.get("ang_step_rms", []), dtype=np.float64).reshape(-1)
    if ang_step.shape[0] != frames:
        tmp = np.full((frames,), np.nan, dtype=np.float64)
        n = min(frames, int(ang_step.shape[0]))
        if n > 0:
            tmp[:n] = ang_step[:n]
        ang_step = tmp

    rv_sl = getattr(cfg, "rootvel_x_slice", None)
    if isinstance(rv_sl, slice) and frames > 0:
        rv = np.asarray(state_raw[:, rv_sl], dtype=np.float64)
        rootvel_norm = np.linalg.norm(rv, axis=1)
    else:
        rootvel_norm = np.full((frames,), np.nan, dtype=np.float64)

    rp_sl = getattr(cfg, "rootpos_x_slice", None)
    if isinstance(rp_sl, slice) and full_state.ndim == 2 and full_state.shape[0] >= 2:
        rp = np.asarray(full_state[:, rp_sl], dtype=np.float64)
        rootpos_step = np.linalg.norm(np.diff(rp, axis=0), axis=1)
        rootpos_from_start = np.linalg.norm(rp[1:] - rp[:1], axis=1)
    else:
        rootpos_step = np.full((frames,), np.nan, dtype=np.float64)
        rootpos_from_start = np.full((frames,), np.nan, dtype=np.float64)

    return {
        "pose_step_geo_deg": pose_step,
        "pose_manifold_z_rms": pose_manifold,
        "pose_knn1_z_rms": pose_knn,
        "angvel_abs_rms": ang_abs,
        "angvel_z_rms": ang_z,
        "angvel_step_rms": ang_step,
        "rootvel_norm": rootvel_norm,
        "rootpos_step_norm": rootpos_step,
        "rootpos_from_start_norm": rootpos_from_start,
    }


def _layer1_stats(seq: Mapping[str, Any], cfg: Any, baseline: Layer1Baseline) -> dict[str, Any]:
    series = _layer1_series(seq, cfg, baseline)
    out: dict[str, Any] = {}
    for signal in LAYER1_SIGNALS:
        out.update(_stat_series(signal, series[signal]))
    state_raw = np.asarray(seq.get("state_raw", []), dtype=np.float64)
    out.update(
        {
            "layer1_no_fk": int(bool(seq.get("tensor_meta", {}).get("layer1_no_fk", False)) if isinstance(seq.get("tensor_meta", {}), Mapping) else 0),
            "layer1_realized_state_raw_shape": json.dumps([int(x) for x in state_raw.shape], separators=(",", ":")),
            "layer1_realized_state_raw_dtype": "float64",
            "layer1_realized_state_raw_device": "cpu",
            "layer1_realized_state_raw_finite": int(bool(np.isfinite(state_raw).all())) if state_raw.size else 1,
            "layer1_initial_state_raw_shape": json.dumps([int(x) for x in np.asarray(seq.get("initial_state_raw", []), dtype=np.float64).shape], separators=(",", ":")),
        }
    )
    return out


def _idx_window(length: int, start: int, horizon: int) -> np.ndarray:
    if int(length) <= 0:
        raise ValueError("cannot build wrapped window from empty clip")
    return (np.arange(int(start), int(start) + int(horizon), dtype=np.int64) % int(length)).astype(np.int64, copy=False)


def _normalize_cond_window(ds: Any, c_in: np.ndarray, c_tgt_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    c_in = np.asarray(c_in, dtype=np.float32).copy()
    c_tgt = np.asarray(c_tgt_raw, dtype=np.float32).copy()
    if not bool(getattr(ds, "normalize_c", True)) or c_in.shape[1] <= 0:
        return c_in, c_tgt, None, None
    mu, std = ds._robust_mean_std(c_in)
    std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    c_in = (c_in - mu) / std
    c_tgt = (c_tgt - mu) / std
    np.nan_to_num(c_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(c_tgt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(c_in, -6.0, 6.0, out=c_in)
    np.clip(c_tgt, -6.0, 6.0, out=c_tgt)
    return c_in.astype(np.float32, copy=False), c_tgt.astype(np.float32, copy=False), mu.astype(np.float32), std.astype(np.float32)


def _build_wrapped_window_sample(ds: Any, clip: Any, start: int, horizon: int) -> dict[str, torch.Tensor]:
    h = int(horizon)
    idx = _idx_window(int(clip.X.shape[0]), int(start), h)
    cond_idx = _idx_window(int(clip.C.shape[0]), int(start), h)
    cond_next_idx = _idx_window(int(clip.C.shape[0]), int(start) + 1, h)
    x_v = np.asarray(clip.X[idx], dtype=np.float32)
    y_v = np.asarray(clip.Y[idx], dtype=np.float32)
    c_raw = np.asarray(clip.C, dtype=np.float32)
    c_in, c_tgt, c_mu, c_std = _normalize_cond_window(ds, c_raw[cond_idx], c_raw[cond_next_idx])
    contact_dim = int(getattr(ds, "contact_dim", 2) or 2)
    sample: dict[str, torch.Tensor] = {
        "motion": torch.from_numpy(x_v.copy()).float(),
        "gt_motion": torch.from_numpy(y_v.copy()).float(),
        "cond_in": torch.from_numpy(c_in).float(),
        "cond_tgt": torch.from_numpy(c_tgt).float(),
        "cond_tgt_raw": torch.from_numpy(c_raw[cond_next_idx].astype(np.float32, copy=False)).float(),
        "clip_id": torch.tensor(0, dtype=torch.int64),
        "start": torch.tensor(int(start), dtype=torch.int64),
        "clip_len": torch.tensor(int(clip.X.shape[0]), dtype=torch.int64),
    }
    if c_mu is not None and c_std is not None and c_mu.size == c_in.shape[1]:
        sample["cond_norm_mu"] = torch.from_numpy(c_mu).float()
        sample["cond_norm_std"] = torch.from_numpy(c_std).float()
    if getattr(clip, "contacts", None) is not None:
        sample["contacts"] = torch.from_numpy(np.asarray(clip.contacts[idx], dtype=np.float32)).float()
    else:
        sample["contacts"] = torch.zeros((h, contact_dim), dtype=torch.float32)
    if getattr(clip, "angvel_norm", None) is not None:
        sample["angvel"] = torch.from_numpy(np.asarray(clip.angvel_norm[idx], dtype=np.float32)).float()
    else:
        sample["angvel"] = torch.zeros((h, int(getattr(ds, "angvel_dim", 0) or 0)), dtype=torch.float32)
    if getattr(clip, "pose_hist_norm", None) is not None:
        sample["pose_hist"] = torch.from_numpy(np.asarray(clip.pose_hist_norm[idx], dtype=np.float32)).float()
    else:
        sample["pose_hist"] = torch.zeros((h, int(getattr(ds, "pose_hist_dim", 0) or 0)), dtype=torch.float32)
    return sample


def _phase_cases(calib: Any, labels: Sequence[str], phase_bins: int) -> list[ProbeCase]:
    n = int(len(labels))
    out: list[ProbeCase] = []
    seen_frames: set[int] = set()
    for b in range(max(1, int(phase_bins))):
        frame = int(math.floor((float(b) + 0.5) * float(max(1, n)) / float(max(1, int(phase_bins)))))
        frame = max(0, min(n - 1, frame))
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        ph = calib._phase_at(labels, frame, isolation_frames=34)
        out.append(
            ProbeCase(
                case_id=f"walkf_phase:q{b}:s{frame}",
                case_kind="walkf_phase",
                source_clip=WALK_F,
                source_frame=int(frame),
                target_clip=WALK_F,
                target_frame=int(frame),
                independent_unit=f"{WALK_F}:phase_bin:{b}",
                source_region_key=f"{WALK_F}:{ph['region_key_suffix']}",
                target_region_key=f"{WALK_F}:{ph['region_key_suffix']}",
                start_label=str(ph["label"]),
                phase_bin=str(ph["phase_bin"]),
                cycle_bin8=int(ph["cycle_bin8"]),
                goal_max_gap=0,
            )
        )
    return out


def _cross_cases(calib: Any, labels_by_clip: Mapping[str, Sequence[str]], state281: Mapping[str, np.ndarray], *, min_gap: int, max_gap: int, seam_len: int) -> list[ProbeCase]:
    grounded = calib._grounded_pairs(state281)
    cases = calib._build_cross_cases(
        labels_by_clip=labels_by_clip,
        grounded=grounded,
        min_gap=int(min_gap),
        max_gap_cap=int(max_gap),
        seam_len=int(seam_len),
        isolation_frames=34,
    )
    out: list[ProbeCase] = []
    for c in cases:
        out.append(
            ProbeCase(
                case_id=f"goal_cross:{c.target_clip}:src{int(c.source_frame)}:tgt{int(c.target_frame)}",
                case_kind="goal_cross",
                source_clip=str(c.source_clip),
                source_frame=int(c.source_frame),
                target_clip=str(c.target_clip),
                target_frame=int(c.target_frame),
                independent_unit=str(c.independent_unit),
                source_region_key=str(c.source_region_key),
                target_region_key=str(c.target_region_key),
                start_label=str(c.start_label),
                phase_bin=str(c.phase_bin),
                cycle_bin8=int(c.cycle_bin8),
                goal_max_gap=int(c.max_gap),
            )
        )
    return out


def _metric_row(
    *,
    arm: str,
    case: ProbeCase,
    gap: int,
    seq: Mapping[str, Any],
    cfg: Any,
    layer1_baseline: Layer1Baseline,
    contact_vxy_mode: str,
    tensor_state_shape: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    series = _soft_signal_series(seq)
    frames = int(series["contacts_meas"].shape[0])
    cond_meta = seq.get("tensor_meta", {}).get("cond", {}) if isinstance(seq.get("tensor_meta", {}), Mapping) else {}
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "arm": arm,
        "valid": 1,
        "invalid_reason": "",
        "gap": int(gap),
        "gap_bucket": _gap_bucket_label(int(gap)),
        "frames": frames,
        "contact_feet": 2,
        "contact_vxy_mode": str(contact_vxy_mode),
        "source_clip": case.source_clip,
        "target_clip": case.target_clip,
        "source_frame": int(case.source_frame),
        "target_frame": int(case.target_frame),
        "independent_unit": case.independent_unit,
        "source_region_key": case.source_region_key,
        "target_region_key": case.target_region_key,
        "start_label": case.start_label,
        "phase_bin": case.phase_bin,
        "cycle_bin8": int(case.cycle_bin8),
        "soft_metric_axis": "continuous",
        "soft_metric_no_binary_labels": 1,
        "layer1_metric_axis": "raw_state_realized_motion_feature_space",
        "layer1_delta_direction": "free_minus_teacher; positive means free-run metric larger than same-start teacher",
        "layer1_main_signal": "pose_step_geo_deg + pose_manifold_z_rms + pose_knn1_z_rms; root is command-driven companion only",
        "rot6d_x_slice": _json_slice(getattr(cfg, "rot6d_x_slice", None)),
        "angvel_x_slice": _json_slice(getattr(cfg, "angvel_x_slice", None)),
        "rootvel_x_slice": _json_slice(getattr(cfg, "rootvel_x_slice", None)),
        "rootpos_x_slice": _json_slice(getattr(cfg, "rootpos_x_slice", None)),
        "rot6d_columns": json.dumps(list(layer1_baseline.columns), separators=(",", ":")),
        "rot6d_joint_count": int(layer1_baseline.joint_count),
        "pose_baseline_clip": WALK_F,
        "pose_baseline_frames": int(layer1_baseline.walk_frames),
        "pose_baseline_dim": int(layer1_baseline.pose_dim),
        "pose_baseline_dtype": "float64",
        "pose_baseline_device": "cpu",
        "layer1_z_std_floor": float(layer1_baseline.z_std_floor),
        "pose_baseline_raw_std_min": float(layer1_baseline.pose_raw_std_min),
        "pose_baseline_raw_std_p50": float(layer1_baseline.pose_raw_std_p50),
        "pose_baseline_raw_std_p95": float(layer1_baseline.pose_raw_std_p95),
        "pose_baseline_std_floored_n": int(layer1_baseline.pose_std_floored_n),
        "angvel_baseline_raw_std_min": layer1_baseline.angvel_raw_std_min if layer1_baseline.angvel_raw_std_min is not None else float("nan"),
        "angvel_baseline_raw_std_p50": layer1_baseline.angvel_raw_std_p50 if layer1_baseline.angvel_raw_std_p50 is not None else float("nan"),
        "angvel_baseline_raw_std_p95": layer1_baseline.angvel_raw_std_p95 if layer1_baseline.angvel_raw_std_p95 is not None else float("nan"),
        "angvel_baseline_std_floored_n": layer1_baseline.angvel_std_floored_n if layer1_baseline.angvel_std_floored_n is not None else "",
        "state_shape": json.dumps([int(x) for x in tensor_state_shape], separators=(",", ":")),
        "state_dtype": "float32",
        "state_device": str(device),
        "cond_shape": json.dumps(cond_meta.get("shape", []), separators=(",", ":")),
        "cond_dtype": cond_meta.get("dtype", ""),
        "cond_device": cond_meta.get("device", ""),
        "cond_finite": int(bool(cond_meta.get("finite", False))),
    }
    for signal in SOFT_SIGNALS:
        row.update(_soft_stats(signal, series[signal]))
    row.update(_layer1_stats(seq, cfg, layer1_baseline))
    for signal in (WITNESS_SIGNAL, CARRY_ECHO_SIGNAL):
        vals = series[signal][np.isfinite(series[signal])]
        row[f"{signal}_out_of_range_n"] = int(((vals < 0.0) | (vals > 1.0)).sum()) if vals.size else 0
    return row


def _invalid_row(*, arm: str, case: ProbeCase, gap: int, reason: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "arm": arm,
        "valid": 0,
        "invalid_reason": reason,
        "gap": int(gap),
        "gap_bucket": _gap_bucket_label(int(gap)),
        "source_clip": case.source_clip,
        "target_clip": case.target_clip,
        "source_frame": int(case.source_frame),
        "target_frame": int(case.target_frame),
        "independent_unit": case.independent_unit,
        "source_region_key": case.source_region_key,
        "target_region_key": case.target_region_key,
        "start_label": case.start_label,
        "phase_bin": case.phase_bin,
        "cycle_bin8": int(case.cycle_bin8),
    }


def _row_float(row: Mapping[str, Any], key: str) -> float:
    return _finite_float(row.get(key), float("nan"))


def _arm_counter(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    sub = [r for r in rows if str(r.get("arm")) == arm]
    valid = [r for r in sub if int(r.get("valid", 0) or 0) == 1]
    out: dict[str, Any] = {
        "rows": len(sub),
        "valid": len(valid),
        "invalid": len(sub) - len(valid),
        "effective_independent_n": int(len({str(r.get("independent_unit")) for r in valid})),
        "gap_min": min([int(r.get("gap", 0) or 0) for r in valid], default=None),
        "gap_max": max([int(r.get("gap", 0) or 0) for r in valid], default=None),
        f"{WITNESS_SIGNAL}_out_of_range_n": int(sum(int(r.get(f"{WITNESS_SIGNAL}_out_of_range_n", 0) or 0) for r in valid)),
        f"{CARRY_ECHO_SIGNAL}_out_of_range_n": int(sum(int(r.get(f"{CARRY_ECHO_SIGNAL}_out_of_range_n", 0) or 0) for r in valid)),
    }
    for signal in SOFT_SIGNALS:
        out[f"{signal}_mean"] = _mean_array([_row_float(r, f"{signal}_mean") for r in valid])
        out[f"{signal}_terminal_mean"] = _mean_array([_row_float(r, f"{signal}_terminal_mean") for r in valid])
        out[f"{signal}_last_quarter_mean"] = _mean_array([_row_float(r, f"{signal}_last_quarter_mean") for r in valid])
    return out


def _first_decay_step(curve: Sequence[float], *, eps: float = SOFT_DECAY_EPS) -> Optional[int]:
    for idx, val in enumerate(curve):
        fv = _finite_float(val, float("nan"))
        if math.isfinite(fv) and fv > float(eps):
            return int(idx)
    return None


def _pair_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row.get("valid", 0) or 0) != 1:
            continue
        grouped[(str(row.get("case_id")), int(row.get("gap", -1)))][str(row.get("arm"))] = row
    records: list[dict[str, Any]] = []
    for (case_id, gap), arms in sorted(grouped.items()):
        for base, free_arm, teacher_arm in (
            ("native", "native_free", "native_teacher"),
            ("goal", "goal_free", "goal_teacher"),
        ):
            if free_arm not in arms or teacher_arm not in arms:
                continue
            free = arms[free_arm]
            teacher = arms[teacher_arm]
            rec: dict[str, Any] = {
                "base": base,
                "case_id": case_id,
                "case_kind": free.get("case_kind"),
                "gap": int(gap),
                "gap_bucket": str(free.get("gap_bucket") or _gap_bucket_label(int(gap))),
                "independent_unit": free.get("independent_unit"),
                "source_region_key": free.get("source_region_key"),
                "target_region_key": free.get("target_region_key"),
                "source_clip": free.get("source_clip"),
                "target_clip": free.get("target_clip"),
            }
            for signal in SOFT_SIGNALS:
                f_arr = _series_from_row(free, signal)
                t_arr = _series_from_row(teacher, signal)
                n = min(int(f_arr.shape[0]), int(t_arr.shape[0]))
                f_arr = f_arr[:n]
                t_arr = t_arr[:n]
                decay = t_arr - f_arr
                curve = _mean_by_step(decay)
                rec[f"{signal}_free_mean"] = _mean_array(f_arr)
                rec[f"{signal}_teacher_mean"] = _mean_array(t_arr)
                rec[f"{signal}_decay_mean"] = _mean_array(decay)
                rec[f"{signal}_decay_p50"] = _percentile_array(decay, 50)
                rec[f"{signal}_decay_p95"] = _percentile_array(decay, 95)
                rec[f"{signal}_decay_terminal_mean"] = _terminal_mean(decay)
                rec[f"{signal}_decay_last_quarter_mean"] = _window_mean(decay, tail=True)
                rec[f"{signal}_decay_first_step_gt_eps"] = _first_decay_step(curve)
                rec[f"{signal}_decay_curve"] = curve
            component_decays = {
                signal: _finite_float(rec.get(f"{signal}_decay_last_quarter_mean"), float("nan"))
                for signal in SOFT_COMPONENT_SIGNALS
            }
            finite_components = {k: v for k, v in component_decays.items() if math.isfinite(v)}
            if finite_components:
                driver, driver_value = max(finite_components.items(), key=lambda item: item[1])
                rec["component_driver_tail"] = driver if driver_value > SOFT_DECAY_EPS else "none"
                rec["component_driver_tail_decay"] = float(driver_value)
            else:
                rec["component_driver_tail"] = "unknown"
                rec["component_driver_tail_decay"] = float("nan")
            fk_decay = _finite_float(rec.get("contacts_meas_decay_last_quarter_mean"), float("nan"))
            witness_decay = _finite_float(rec.get(f"{WITNESS_SIGNAL}_decay_last_quarter_mean"), float("nan"))
            rec["fk_minus_witness_decay_last_quarter_mean"] = (
                float(fk_decay - witness_decay) if math.isfinite(fk_decay) and math.isfinite(witness_decay) else float("nan")
            )
            if math.isfinite(fk_decay) and fk_decay > SOFT_DECAY_EPS and math.isfinite(witness_decay) and witness_decay > SOFT_DECAY_EPS:
                consistency = "fk_drop_witness_drop"
            elif math.isfinite(fk_decay) and fk_decay > SOFT_DECAY_EPS and (not math.isfinite(witness_decay) or witness_decay <= SOFT_DECAY_EPS):
                consistency = "fk_drop_witness_not_drop"
            elif (not math.isfinite(fk_decay) or fk_decay <= SOFT_DECAY_EPS) and math.isfinite(witness_decay) and witness_decay > SOFT_DECAY_EPS:
                consistency = "witness_drop_fk_not_drop"
            else:
                consistency = "no_soft_drop"
            rec["fk_witness_consistency_tail"] = consistency
            records.append(rec)
    return records


def _curve_mean(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    curves = [list(r.get(key, [])) for r in records if isinstance(r.get(key, []), list)]
    max_len = max([len(c) for c in curves], default=0)
    out: list[float] = []
    for idx in range(max_len):
        vals = [_finite_float(c[idx], float("nan")) for c in curves if idx < len(c)]
        out.append(_mean_array(vals))
    return out


def _record_mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean_array([_finite_float(r.get(key), float("nan")) for r in records])


def _onset_stats(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    vals = [int(v) for r in records for v in [r.get(key)] if v is not None and str(v) != ""]
    return {
        "n": int(len(vals)),
        "min": min(vals) if vals else None,
        "mean": float(sum(vals) / len(vals)) if vals else None,
        "p50": float(np.percentile(np.asarray(vals, dtype=np.float64), 50)) if vals else None,
    }


def _summarize_pair_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_pairs": int(len(records)),
        "effective_independent_n": int(len({str(r.get("independent_unit")) for r in records})),
        "case_id_n": int(len({str(r.get("case_id")) for r in records})),
        "gap_min": min([int(r.get("gap", 0) or 0) for r in records], default=None),
        "gap_max": max([int(r.get("gap", 0) or 0) for r in records], default=None),
        "component_driver_tail_counts": dict(Counter(str(r.get("component_driver_tail")) for r in records)),
        "fk_witness_consistency_tail_counts": dict(Counter(str(r.get("fk_witness_consistency_tail")) for r in records)),
        "fk_minus_witness_decay_last_quarter_mean": _record_mean(records, "fk_minus_witness_decay_last_quarter_mean"),
    }
    for signal in SOFT_SIGNALS:
        out[f"{signal}_free_mean"] = _record_mean(records, f"{signal}_free_mean")
        out[f"{signal}_teacher_mean"] = _record_mean(records, f"{signal}_teacher_mean")
        out[f"{signal}_decay_mean"] = _record_mean(records, f"{signal}_decay_mean")
        out[f"{signal}_decay_p50"] = _record_mean(records, f"{signal}_decay_p50")
        out[f"{signal}_decay_p95"] = _record_mean(records, f"{signal}_decay_p95")
        out[f"{signal}_decay_terminal_mean"] = _record_mean(records, f"{signal}_decay_terminal_mean")
        out[f"{signal}_decay_last_quarter_mean"] = _record_mean(records, f"{signal}_decay_last_quarter_mean")
        out[f"{signal}_decay_first_step_gt_eps"] = _onset_stats(records, f"{signal}_decay_first_step_gt_eps")
        out[f"{signal}_decay_curve_mean"] = _curve_mean(records, f"{signal}_decay_curve")
    return out


def _soft_pair_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = _pair_records(rows)
    by_base: dict[str, Any] = {}
    for base in sorted({str(r.get("base")) for r in records}):
        by_base[base] = _summarize_pair_group([r for r in records if str(r.get("base")) == base])
    by_gap_bucket: list[dict[str, Any]] = []
    for base in sorted({str(r.get("base")) for r in records}):
        buckets = sorted({str(r.get("gap_bucket")) for r in records if str(r.get("base")) == base})
        for bucket in buckets:
            group = [r for r in records if str(r.get("base")) == base and str(r.get("gap_bucket")) == bucket]
            rec = {"base": base, "gap_bucket": bucket}
            rec.update(_summarize_pair_group(group))
            by_gap_bucket.append(rec)
    by_gap: list[dict[str, Any]] = []
    for base in sorted({str(r.get("base")) for r in records}):
        gaps = sorted({int(r.get("gap")) for r in records if str(r.get("base")) == base})
        for gap in gaps:
            group = [r for r in records if str(r.get("base")) == base and int(r.get("gap")) == int(gap)]
            rec = {"base": base, "gap": int(gap)}
            rec.update(_summarize_pair_group(group))
            by_gap.append(rec)
    return {
        "pair_n": int(len(records)),
        "decay_definition": "teacher_minus_free; positive means free-run soft contact is lower than same-start teacher",
        "decay_eps_for_onset_and_consistency_only": float(SOFT_DECAY_EPS),
        "overall": _summarize_pair_group(records),
        "by_base": by_base,
        "by_gap_bucket": by_gap_bucket,
        "by_gap": by_gap,
    }


def _layer1_pair_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row.get("valid", 0) or 0) != 1:
            continue
        grouped[(str(row.get("case_id")), int(row.get("gap", -1)))][str(row.get("arm"))] = row
    records: list[dict[str, Any]] = []
    for (case_id, gap), arms in sorted(grouped.items()):
        for base, free_arm, teacher_arm in (
            ("native", "native_free", "native_teacher"),
            ("goal", "goal_free", "goal_teacher"),
        ):
            if free_arm not in arms or teacher_arm not in arms:
                continue
            free = arms[free_arm]
            teacher = arms[teacher_arm]
            rec: dict[str, Any] = {
                "base": base,
                "case_id": case_id,
                "case_kind": free.get("case_kind"),
                "gap": int(gap),
                "gap_bucket": str(free.get("gap_bucket") or _gap_bucket_label(int(gap))),
                "independent_unit": free.get("independent_unit"),
                "source_region_key": free.get("source_region_key"),
                "target_region_key": free.get("target_region_key"),
                "source_clip": free.get("source_clip"),
                "target_clip": free.get("target_clip"),
            }
            for signal in LAYER1_SIGNALS:
                f_arr = _series_from_row_any(free, signal)
                t_arr = _series_from_row_any(teacher, signal)
                if f_arr.ndim == 0:
                    f_arr = f_arr.reshape(1)
                if t_arr.ndim == 0:
                    t_arr = t_arr.reshape(1)
                n = min(int(f_arr.shape[0]), int(t_arr.shape[0]))
                f_aligned = f_arr[:n]
                t_aligned = t_arr[:n]
                delta = f_aligned - t_aligned
                rec[f"{signal}_free_mean"] = _mean_array(f_aligned)
                rec[f"{signal}_teacher_mean"] = _mean_array(t_aligned)
                rec[f"{signal}_delta_mean"] = _mean_array(delta)
                rec[f"{signal}_delta_p50"] = _percentile_array(delta, 50)
                rec[f"{signal}_delta_p95"] = _percentile_array(delta, 95)
                rec[f"{signal}_delta_terminal_mean"] = _terminal_mean(delta)
                rec[f"{signal}_delta_last_quarter_mean"] = _window_mean(delta, tail=True)
                rec[f"{signal}_delta_curve"] = _mean_by_step(delta)
            records.append(rec)
    return records


def _summarize_layer1_pair_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = sorted({str(r.get("independent_unit")) for r in records})
    out: dict[str, Any] = {
        "n_pairs": int(len(records)),
        "effective_independent_n": int(len(units)),
        "effective_independent_units": units,
        "case_id_n": int(len({str(r.get("case_id")) for r in records})),
        "gap_min": min([int(r.get("gap", 0) or 0) for r in records], default=None),
        "gap_max": max([int(r.get("gap", 0) or 0) for r in records], default=None),
    }
    for signal in LAYER1_SIGNALS:
        out[f"{signal}_free_mean"] = _record_mean(records, f"{signal}_free_mean")
        out[f"{signal}_teacher_mean"] = _record_mean(records, f"{signal}_teacher_mean")
        out[f"{signal}_delta_mean"] = _record_mean(records, f"{signal}_delta_mean")
        out[f"{signal}_delta_p50"] = _record_mean(records, f"{signal}_delta_p50")
        out[f"{signal}_delta_p95"] = _record_mean(records, f"{signal}_delta_p95")
        out[f"{signal}_delta_terminal_mean"] = _record_mean(records, f"{signal}_delta_terminal_mean")
        out[f"{signal}_delta_last_quarter_mean"] = _record_mean(records, f"{signal}_delta_last_quarter_mean")
        out[f"{signal}_delta_curve_mean"] = _curve_mean(records, f"{signal}_delta_curve")
    return out


def _layer1_pair_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = _layer1_pair_records(rows)
    bases = sorted({str(r.get("base")) for r in records})
    by_base: dict[str, Any] = {}
    for base in bases:
        by_base[base] = _summarize_layer1_pair_group([r for r in records if str(r.get("base")) == base])

    by_gap_bucket: list[dict[str, Any]] = []
    for base in bases:
        buckets = sorted({str(r.get("gap_bucket")) for r in records if str(r.get("base")) == base})
        for bucket in buckets:
            group = [r for r in records if str(r.get("base")) == base and str(r.get("gap_bucket")) == bucket]
            rec = {"base": base, "gap_bucket": bucket}
            rec.update(_summarize_layer1_pair_group(group))
            by_gap_bucket.append(rec)

    by_gap: list[dict[str, Any]] = []
    for base in bases:
        gaps = sorted({int(r.get("gap")) for r in records if str(r.get("base")) == base})
        for gap in gaps:
            group = [r for r in records if str(r.get("base")) == base and int(r.get("gap")) == int(gap)]
            rec = {"base": base, "gap": int(gap)}
            rec.update(_summarize_layer1_pair_group(group))
            by_gap.append(rec)

    by_independent_unit: list[dict[str, Any]] = []
    for base in bases:
        units = sorted({str(r.get("independent_unit")) for r in records if str(r.get("base")) == base})
        for unit in units:
            group = [r for r in records if str(r.get("base")) == base and str(r.get("independent_unit")) == unit]
            rec = {"base": base, "independent_unit": unit}
            rec.update(_summarize_layer1_pair_group(group))
            by_independent_unit.append(rec)

    return {
        "pair_n": int(len(records)),
        "delta_definition": "free_minus_teacher; positive means free-run metric is larger than same-start teacher",
        "fk_free": True,
        "main_pose_signals": ["pose_step_geo_deg", "pose_manifold_z_rms", "pose_knn1_z_rms"],
        "root_caveat": "rootvel/rootpos are reported only as companions because apply_free_carry_raw writes them from cond_speed/cond_dir",
        "overall": _summarize_layer1_pair_group(records),
        "by_base": by_base,
        "by_gap_bucket": by_gap_bucket,
        "by_gap": by_gap,
        "by_independent_unit": by_independent_unit,
    }


def _summary(rows: Sequence[Mapping[str, Any]], cases: Sequence[ProbeCase], *, gaps: Sequence[int]) -> dict[str, Any]:
    valid = [r for r in rows if int(r.get("valid", 0) or 0) == 1]
    arms = sorted({str(r.get("arm")) for r in rows})
    return {
        "rows": int(len(rows)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(rows) - len(valid)),
        "arms": {arm: _arm_counter(rows, arm) for arm in arms},
        "case_n": int(len(cases)),
        "case_kind_counts": dict(Counter(c.case_kind for c in cases)),
        "phase_bin_n": int(len({c.phase_bin for c in cases})),
        "cycle_bin8_n": int(len({c.cycle_bin8 for c in cases})),
        "source_frame_n": int(len({(c.source_clip, c.source_frame) for c in cases})),
        "independent_unit_n": int(len({c.independent_unit for c in cases})),
        "gaps": [int(x) for x in gaps],
        "metric_axis": "layer1_realized_motion_feature_space_fk_free_with_soft_contact_reference_columns",
        "soft_pair_deltas": _soft_pair_summary(rows),
        "layer1_pair_deltas": _layer1_pair_summary(rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only FR-vs-TF-vs-native contact stability diagnostic for action handoff support failures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--pretrain-template", type=Path, default=None)
    parser.add_argument("--encoder-bundle", type=Path, default=None)
    parser.add_argument("--npz-root", type=Path, default=None)
    parser.add_argument("--z-features", type=Path, default=None)
    parser.add_argument("--bands", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("debug_output") / f"_{DEFAULT_STEM}")
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--min-gap", type=int, default=12)
    parser.add_argument("--max-gap", type=int, default=84)
    parser.add_argument("--gap-step", type=int, default=12)
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--seam-len", type=int, default=6)
    parser.add_argument("--contact-vxy-mode", choices=("abs", "root_rel"), default="abs")
    parser.add_argument(
        "--layer1-z-std-floor",
        type=float,
        default=0.05,
        help="Floor for GT Walk_F per-channel std in Layer-1 z-score distances.",
    )
    parser.add_argument(
        "--include-whitebox-soft",
        action="store_true",
        help="Also call the legacy FK whitebox contact measurement. Default is FK-free Layer-1-only rollout logging.",
    )
    parser.add_argument("--limit-phase-cases", type=int, default=0)
    parser.add_argument("--limit-goal-cases", type=int, default=0)
    parser.add_argument("--include-goal-cross", action="store_true", default=True)
    parser.add_argument("--no-goal-cross", dest="include_goal_cross", action="store_false")
    parser.add_argument("--print-full-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    calib = _load_module(CALIBRATION_PATH, "freerun_contact_stability_calibration")
    helper = calib._load_helper()
    args.checkpoint = Path(args.checkpoint or calib.CKPT)
    args.bundle = Path(args.bundle or calib.BUNDLE)
    args.pretrain_template = Path(args.pretrain_template or calib.PRETRAIN_TEMPLATE)
    args.encoder_bundle = Path(args.encoder_bundle or calib.ENCODER_BUNDLE)
    args.npz_root = Path(args.npz_root or calib.NPZ_ROOT)
    args.z_features = Path(args.z_features or calib.Z_FEATURES)
    args.bands = Path(args.bands or calib.BANDS)

    state281 = load_clip_states(args.z_features, args.npz_root)
    labels_by_clip = {name: calib._labels_from_contacts(arr[:, CONTACT_SLICE]) for name, arr in state281.items()}
    gaps = list(range(int(args.min_gap), int(args.max_gap) + 1, max(1, int(args.gap_step))))

    phase_cases = _phase_cases(calib, labels_by_clip[WALK_F], int(args.phase_bins))
    if int(args.limit_phase_cases) > 0:
        phase_cases = phase_cases[: int(args.limit_phase_cases)]
    goal_cases: list[ProbeCase] = []
    if bool(args.include_goal_cross):
        goal_cases = _cross_cases(
            calib,
            labels_by_clip,
            state281,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            seam_len=int(args.seam_len),
        )
        if int(args.limit_goal_cases) > 0:
            goal_cases = goal_cases[: int(args.limit_goal_cases)]
    cases = phase_cases + goal_cases

    runner = FreeRunCycleRunner(calib._runner_args(args))
    ds_by_clip: dict[str, Any] = {}
    clip_by_name: dict[str, Any] = {}
    for clip_name in sorted(labels_by_clip.keys()):
        ds = runner._build_dataset(args.npz_root / f"{clip_name}.npz", seq_len=max(2, int(args.max_gap) + 1))
        runner._ensure_model_ready(ds)
        ds_by_clip[clip_name] = ds
        clip_by_name[clip_name] = ds.clips[0]
    if runner.trainer is None or runner.model is None:
        raise RuntimeError("runner did not initialize model")
    trainer = runner.trainer
    model = runner.model
    device = runner.device
    contact_vxy_mode = str(args.contact_vxy_mode)
    setattr(trainer, "contact_meas_vxy_mode", contact_vxy_mode)
    cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    walk_raw_state = _denorm_walk_raw_state(trainer, clip_by_name[WALK_F], device)
    layer1_baseline = _build_layer1_baseline(walk_raw_state, cfg, z_std_floor=float(args.layer1_z_std_floor))
    include_whitebox_soft = bool(args.include_whitebox_soft)

    rows: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases, start=1):
        for gap in gaps:
            total = int(gap) + 1
            native_sample = _build_wrapped_window_sample(
                ds_by_clip[WALK_F],
                clip_by_name[WALK_F],
                int(case.source_frame),
                total,
            )
            for arm, mode in (("native_free", "free"), ("native_teacher", "teacher")):
                if include_whitebox_soft:
                    seq = helper._run_sequence(
                        trainer=trainer,
                        model=model,
                        sample=native_sample,
                        device=device,
                        mode=mode,
                        apply_lambda=True,
                    )
                    if "initial_state_raw" not in seq:
                        seq = dict(seq)
                        seq["initial_state_raw"] = _initial_state_raw_from_sample(trainer, native_sample, device)
                else:
                    seq = _run_sequence_layer1_only(
                        trainer=trainer,
                        model=model,
                        sample=native_sample,
                        device=device,
                        mode=mode,
                        apply_lambda=True,
                    )
                rows.append(
                    _metric_row(
                        arm=arm,
                        case=case,
                        gap=int(gap),
                        seq=seq,
                        cfg=cfg,
                        layer1_baseline=layer1_baseline,
                        contact_vxy_mode=contact_vxy_mode,
                        tensor_state_shape=native_sample["motion"].unsqueeze(0).shape,
                        device=device,
                    )
                )

            if case.case_kind != "goal_cross":
                continue
            if int(gap) > int(case.goal_max_gap):
                for arm in ("goal_free", "goal_teacher"):
                    rows.append(
                        _invalid_row(
                            arm=arm,
                            case=case,
                            gap=int(gap),
                            reason=f"goal_window_exceeds_target_clip_max_gap:{int(case.goal_max_gap)}",
                        )
                    )
                continue
            goal_sample = calib._prepare_cross_sample(
                helper=helper,
                trainer=trainer,
                ds_target=ds_by_clip[case.target_clip],
                hub_clip=clip_by_name[WALK_F],
                target_clip=clip_by_name[case.target_clip],
                source_frame=int(case.source_frame),
                target_frame=int(case.target_frame),
                total_frames=total,
            )
            for arm, mode in (("goal_free", "free"), ("goal_teacher", "teacher")):
                if include_whitebox_soft:
                    seq = helper._run_sequence(
                        trainer=trainer,
                        model=model,
                        sample=goal_sample,
                        device=device,
                        mode=mode,
                        apply_lambda=True,
                    )
                    if "initial_state_raw" not in seq:
                        seq = dict(seq)
                        seq["initial_state_raw"] = _initial_state_raw_from_sample(trainer, goal_sample, device)
                else:
                    seq = _run_sequence_layer1_only(
                        trainer=trainer,
                        model=model,
                        sample=goal_sample,
                        device=device,
                        mode=mode,
                        apply_lambda=True,
                    )
                rows.append(
                    _metric_row(
                        arm=arm,
                        case=case,
                        gap=int(gap),
                        seq=seq,
                        cfg=cfg,
                        layer1_baseline=layer1_baseline,
                        contact_vxy_mode=contact_vxy_mode,
                        tensor_state_shape=goal_sample["motion"].unsqueeze(0).shape,
                        device=device,
                    )
                )
        print(f"[case] {case_idx}/{len(cases)} {case.case_kind} {case.case_id}", flush=True)

    out_dir = Path(args.out_dir)
    rows_csv = out_dir / f"{args.stem}_rows.csv"
    summary_json = out_dir / f"{args.stem}_summary.json"
    payload = {
        "meta": {
            "task": "fr_vs_tf_vs_native_contact_stability_probe",
            "checkpoint": str(args.checkpoint),
            "bundle": str(args.bundle),
            "pretrain_template": str(args.pretrain_template),
            "encoder_bundle": str(args.encoder_bundle),
            "npz_root": str(args.npz_root),
            "z_features": str(args.z_features),
            "bands": str(args.bands),
            "read_only_forward": True,
            "no_training": True,
            "no_weight_write": True,
            "fk_called": bool(include_whitebox_soft),
            "include_whitebox_soft": bool(include_whitebox_soft),
            "gap_range": [int(args.min_gap), int(args.max_gap), int(args.gap_step)],
            "phase_bins_requested": int(args.phase_bins),
            "contact_vxy_mode": str(args.contact_vxy_mode),
            "metric_axis": "Layer-1 realized raw-state motion features; optional soft contact reference is disabled unless --include-whitebox-soft",
            "layer1_scope": (
                "FK-free realized motion diagnostics from next_raw/state_raw: rot6d pose geodesic step, pose z-score manifold distance, "
                "angvel channel stats, and commanded-root companion stats"
            ),
            "layer1_delta_definition": "free_minus_teacher; positive means free-run metric is larger than same-start teacher",
            "common_mode_baseline": "free-run decay is teacher_minus_free for the same case_id/gap/start; positive means free-run soft contact is lower",
            "whitebox_scope": (
                "contacts_meas/dist_score/vz_score/vxy_score are NaN placeholders in the default FK-free run; "
                "if --include-whitebox-soft is set they are existing compute_contact_meas_whitebox outputs and remain biased reference only"
            ),
            "witness_scope": "plan_contact is ret['contacts_plan'] from the model, reported as a diagnostic witness only",
            "carry_echo_scope": (
                "carry_contact_echo is denormalized state_raw[..., CONTACT_SLICE]; apply_free_carry_raw does not write it, "
                "so it is retained only to audit the old frozen-channel witness failure"
            ),
            "case_selection_note": (
                "case inventory still reuses the prior calibration helper's CONTACT_SLICE labels to reproduce the full-gap case set; "
                "those labels are not used as metric/success rows"
            ),
            "tensor_contract": {
                "state": "[1,H,Dx] float32 device",
                "cond": "[1,H,Dc] float32 device",
                "realized_state_raw": "[H-1,Dx] float64 cpu next_raw generated by apply_free_carry_raw",
                "initial_state_raw": "[Dx] float64 cpu seed state before first next_raw",
                "pose_step_geo_deg": "[H-1,J] float64 cpu, rot6d_to_matrix + geodesic_R on adjacent realized pose frames; no FK",
                "pose_manifold_z_rms": "[H-1] float64 cpu, per-channel z-score RMS against Walk_F raw rot6d distribution",
                "pose_knn1_z_rms": "[H-1] float64 cpu, nearest Walk_F raw rot6d frame distance in the same z-space",
                "angvel_abs_rms": "[H-1] float64 cpu from state_raw angvel_x_slice",
                "root_companion": "rootvel/rootpos are state_raw slices but are command-written by apply_free_carry_raw",
                "contacts_meas": "[H-1,2] float64 cpu NaN unless --include-whitebox-soft",
                "contact_score_components": "{dist_score,vz_score,vxy_score}: [H-1,2] float64 cpu NaN unless --include-whitebox-soft",
                "plan_contact": "[H-1,2] float64 cpu from ret['contacts_plan']",
                "carry_contact_echo": "[H-1,2] float64 cpu from state_raw[..., CONTACT_SLICE]",
                "layer1_baseline": {
                    "clip": WALK_F,
                    "state_space": "denormalized model raw state clip.X, not the 281-d action-handoff sampler state",
                    "frames": int(layer1_baseline.walk_frames),
                    "pose_dim": int(layer1_baseline.pose_dim),
                    "joint_count": int(layer1_baseline.joint_count),
                    "columns": list(layer1_baseline.columns),
                    "z_std_floor": float(layer1_baseline.z_std_floor),
                    "pose_raw_std_min": float(layer1_baseline.pose_raw_std_min),
                    "pose_raw_std_p50": float(layer1_baseline.pose_raw_std_p50),
                    "pose_raw_std_p95": float(layer1_baseline.pose_raw_std_p95),
                    "pose_std_floored_n": int(layer1_baseline.pose_std_floored_n),
                    "angvel_raw_std_min": layer1_baseline.angvel_raw_std_min,
                    "angvel_raw_std_p50": layer1_baseline.angvel_raw_std_p50,
                    "angvel_raw_std_p95": layer1_baseline.angvel_raw_std_p95,
                    "angvel_std_floored_n": layer1_baseline.angvel_std_floored_n,
                },
            },
        },
        "summary": _summary(rows, cases, gaps=gaps),
        "cases": [_jsonable(c.__dict__) for c in cases],
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_json": str(summary_json),
        },
    }
    _write_csv(rows_csv, rows)
    _write_json(summary_json, payload)
    if bool(args.print_full_summary):
        console_payload = {"summary": payload["summary"], "artifacts": payload["artifacts"]}
    else:
        layer1 = payload["summary"].get("layer1_pair_deltas", {})
        console_payload = {
            "summary": {
                "rows": payload["summary"].get("rows"),
                "valid_rows": payload["summary"].get("valid_rows"),
                "invalid_rows": payload["summary"].get("invalid_rows"),
                "case_n": payload["summary"].get("case_n"),
                "gaps_n": len(payload["summary"].get("gaps", [])),
                "layer1_pair_n": layer1.get("pair_n"),
                "fk_called": payload["meta"].get("fk_called"),
            },
            "artifacts": payload["artifacts"],
        }
    print(json.dumps(console_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
