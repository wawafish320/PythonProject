#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (  # noqa: E402
    DEFAULT_BASELINE_CKPT,
    DEFAULT_BASELINE_EVAL,
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    _load_case,
)
from train.validate import run_freerun_cycles as rfc  # noqa: E402


RUN_DATE = "20260404"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_main_trunk_drift_audit_{RUN_DATE}" / "summary.json"
)
PRIMARY_SIGNALS: Tuple[str, ...] = (
    "out",
    "y_inc_raw",
    "motion_in",
    "pose_history_in",
)
APPENDIX_SIGNALS: Tuple[str, ...] = (
    "y_used_raw",
    "motion_raw_after_carry",
    "pose_hist_write_raw",
)
DEPTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("d0_9", 0, 9),
    ("d10_20", 10, 20),
    ("d21_43", 21, 43),
)
SIC_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("sic0_10", 0, 10),
    ("sic11_21", 11, 21),
    ("sic22_43", 22, 43),
)
OFFSET_SPECS: Tuple[Tuple[str, int], ...] = (
    ("0", 0),
    ("5", 5),
    ("20", 20),
)
GROWTH_SPECS: Tuple[Tuple[str, int, int], ...] = (
    ("0_to_5", 0, 5),
    ("5_to_20", 5, 20),
    ("0_to_20", 0, 20),
)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in vals:
        fv = _safe_float(value)
        if math.isfinite(fv):
            out.append(fv)
    return np.asarray(out, dtype=np.float64)


def _summary(vals: np.ndarray) -> Dict[str, float]:
    if vals.size <= 0:
        return {
            "samples": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "samples": int(vals.size),
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


def _tensor_to_mean_vec(x: Any) -> Optional[np.ndarray]:
    if not torch.is_tensor(x):
        return None
    vec = x.detach()
    if vec.ndim == 3 and int(vec.shape[1]) == 1:
        vec = vec[:, 0]
    if vec.ndim == 2:
        vec = vec.mean(dim=0)
    elif vec.ndim > 2:
        vec = vec.reshape(-1)
    else:
        vec = vec.reshape(-1)
    try:
        arr = vec.to(device="cpu", dtype=torch.float32).numpy().astype(np.float64, copy=False)
    except Exception:
        return None
    return arr.reshape(-1)


def _trace_metric(
    freerun_vecs: Sequence[Optional[np.ndarray]],
    teacher_vecs: Sequence[Optional[np.ndarray]],
) -> Dict[str, Any]:
    total = int(min(len(freerun_vecs), len(teacher_vecs)))
    norm_l2: List[Optional[float]] = []
    cosine_distance: List[Optional[float]] = []
    mean_abs: List[Optional[float]] = []
    dims: List[Optional[int]] = []
    valid: List[int] = []
    for idx in range(total):
        vf = freerun_vecs[idx]
        vt = teacher_vecs[idx]
        if vf is None or vt is None:
            norm_l2.append(None)
            cosine_distance.append(None)
            mean_abs.append(None)
            dims.append(None)
            valid.append(0)
            continue
        if tuple(vf.shape) != tuple(vt.shape):
            raise RuntimeError(
                f"trace shape mismatch at step {idx}: freerun={tuple(vf.shape)} teacher={tuple(vt.shape)}"
            )
        diff = vf - vt
        dim = int(vf.size)
        denom = math.sqrt(float(max(1, dim)))
        norm_l2.append(float(np.linalg.norm(diff) / denom))
        mean_abs.append(float(np.mean(np.abs(diff))))
        vf_norm = float(np.linalg.norm(vf))
        vt_norm = float(np.linalg.norm(vt))
        if vf_norm <= 1e-12 or vt_norm <= 1e-12:
            cosine_distance.append(float("nan"))
        else:
            cos_sim = float(np.dot(vf, vt) / (vf_norm * vt_norm))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            cosine_distance.append(float(1.0 - cos_sim))
        dims.append(dim)
        valid.append(1)
    return {
        "steps": total,
        "norm_l2": norm_l2,
        "cosine_distance": cosine_distance,
        "mean_abs": mean_abs,
        "dim": dims,
        "valid": valid,
    }


def _pair_metric(
    lhs: Sequence[Optional[np.ndarray]],
    rhs: Sequence[Optional[np.ndarray]],
) -> Dict[str, Any]:
    trace = _trace_metric(lhs, rhs)
    return {
        "norm_l2": _summary(_finite(trace["norm_l2"])),
        "mean_abs": _summary(_finite(trace["mean_abs"])),
        "cosine_distance": _summary(_finite(trace["cosine_distance"])),
        "steps": int(trace["steps"]),
    }


def _mask_rows(
    per_step: Sequence[Mapping[str, Any]],
    *,
    depth_lo: int,
    depth_hi: int,
    sic_lo: Optional[int] = None,
    sic_hi: Optional[int] = None,
    drop_wrap: bool = False,
) -> List[int]:
    rows: List[int] = []
    for idx, rec in enumerate(per_step):
        if int(idx) < int(depth_lo) or int(idx) > int(depth_hi):
            continue
        if bool(drop_wrap) and bool(rec.get("wrap_boundary_step", False)):
            continue
        if sic_lo is not None or sic_hi is not None:
            try:
                sic = int(rec.get("step_in_cycle", -1) or -1)
            except Exception:
                sic = -1
            if sic_lo is not None and sic < int(sic_lo):
                continue
            if sic_hi is not None and sic > int(sic_hi):
                continue
        rows.append(int(idx))
    return rows


def _selected_steps(per_step: Sequence[Mapping[str, Any]]) -> List[int]:
    rows: List[int] = []
    total = int(len(per_step))
    for idx, rec in enumerate(per_step):
        if idx + 20 >= total:
            continue
        if idx < 10:
            continue
        if bool(rec.get("wrap_boundary_step", False)):
            continue
        try:
            sic = int(rec.get("step_in_cycle", -1) or -1)
        except Exception:
            sic = -1
        if 11 <= sic <= 43:
            rows.append(int(idx))
    return rows


def _offset_growth_summary(
    *,
    rows: Sequence[int],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    offsets: Dict[str, Any] = {}
    for label, off in OFFSET_SPECS:
        vals_l2: List[float] = []
        vals_abs: List[float] = []
        vals_cos: List[float] = []
        for base in rows:
            pos = int(base) + int(off)
            if pos >= int(trace["steps"]) or not trace["valid"][pos]:
                continue
            v_l2 = trace["norm_l2"][pos]
            v_abs = trace["mean_abs"][pos]
            v_cos = trace["cosine_distance"][pos]
            if v_l2 is not None and math.isfinite(float(v_l2)):
                vals_l2.append(float(v_l2))
            if v_abs is not None and math.isfinite(float(v_abs)):
                vals_abs.append(float(v_abs))
            if v_cos is not None and math.isfinite(float(v_cos)):
                vals_cos.append(float(v_cos))
        offsets[str(label)] = {
            "norm_l2": _summary(_finite(vals_l2)),
            "mean_abs": _summary(_finite(vals_abs)),
            "cosine_distance": _summary(_finite(vals_cos)),
        }
    growth: Dict[str, Any] = {}
    for label, off_a, off_b in GROWTH_SPECS:
        vals_l2: List[float] = []
        vals_abs: List[float] = []
        vals_cos: List[float] = []
        for base in rows:
            pa = int(base) + int(off_a)
            pb = int(base) + int(off_b)
            if pb >= int(trace["steps"]):
                continue
            if not (trace["valid"][pa] and trace["valid"][pb]):
                continue
            va_l2 = trace["norm_l2"][pa]
            vb_l2 = trace["norm_l2"][pb]
            va_abs = trace["mean_abs"][pa]
            vb_abs = trace["mean_abs"][pb]
            va_cos = trace["cosine_distance"][pa]
            vb_cos = trace["cosine_distance"][pb]
            if va_l2 is not None and vb_l2 is not None and math.isfinite(float(va_l2)) and math.isfinite(float(vb_l2)):
                vals_l2.append(float(vb_l2) - float(va_l2))
            if va_abs is not None and vb_abs is not None and math.isfinite(float(va_abs)) and math.isfinite(float(vb_abs)):
                vals_abs.append(float(vb_abs) - float(va_abs))
            if va_cos is not None and vb_cos is not None and math.isfinite(float(va_cos)) and math.isfinite(float(vb_cos)):
                vals_cos.append(float(vb_cos) - float(va_cos))
        growth[str(label)] = {
            "norm_l2_delta": _summary(_finite(vals_l2)),
            "mean_abs_delta": _summary(_finite(vals_abs)),
            "cosine_distance_delta": _summary(_finite(vals_cos)),
        }
    return {
        "rows": int(len(rows)),
        "offsets": offsets,
        "growth": growth,
    }


def _bucket_signal_summary(rows: Sequence[int], trace: Mapping[str, Any]) -> Dict[str, Any]:
    vals_l2 = [trace["norm_l2"][i] for i in rows if i < int(trace["steps"]) and trace["valid"][i]]
    vals_abs = [trace["mean_abs"][i] for i in rows if i < int(trace["steps"]) and trace["valid"][i]]
    vals_cos = [trace["cosine_distance"][i] for i in rows if i < int(trace["steps"]) and trace["valid"][i]]
    return {
        "steps": int(len(rows)),
        "norm_l2": _summary(_finite(vals_l2)),
        "mean_abs": _summary(_finite(vals_abs)),
        "cosine_distance": _summary(_finite(vals_cos)),
    }


def _bucket_gap_summary(rows: Sequence[int], per_step: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    vals = [_safe_float(per_step[i].get(key)) for i in rows if i < len(per_step)]
    return _summary(_finite(vals))


def _timing_tables(
    *,
    per_step: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    depth_out: Dict[str, Any] = {}
    for label, lo, hi in DEPTH_BUCKETS:
        rows = _mask_rows(per_step, depth_lo=int(lo), depth_hi=int(min(int(hi), max(0, len(per_step) - 1))))
        depth_out[label] = {
            "rows": int(len(rows)),
            "GeoLocalDeg": _bucket_gap_summary(rows, per_step, "GeoLocalDeg"),
            "BlendGeoLocalDeg": _bucket_gap_summary(rows, per_step, "BlendGeoLocalDeg"),
            "DirectGeoLocalDeg": _bucket_gap_summary(rows, per_step, "DirectGeoLocalDeg"),
            "signals": {
                name: _bucket_signal_summary(rows, trace)
                for name, trace in traces.items()
            },
        }
    sic_out: Dict[str, Any] = {}
    for label, lo, hi in SIC_BUCKETS:
        rows = _mask_rows(
            per_step,
            depth_lo=0,
            depth_hi=max(0, len(per_step) - 1),
            sic_lo=int(lo),
            sic_hi=int(hi),
            drop_wrap=True,
        )
        sic_out[label] = {
            "rows": int(len(rows)),
            "GeoLocalDeg": _bucket_gap_summary(rows, per_step, "GeoLocalDeg"),
            "BlendGeoLocalDeg": _bucket_gap_summary(rows, per_step, "BlendGeoLocalDeg"),
            "DirectGeoLocalDeg": _bucket_gap_summary(rows, per_step, "DirectGeoLocalDeg"),
            "signals": {
                name: _bucket_signal_summary(rows, trace)
                for name, trace in traces.items()
            },
        }
    return {
        "depth_buckets": depth_out,
        "sic_buckets": sic_out,
    }


@dataclass
class TraceRecorder:
    motion_in: List[Optional[np.ndarray]] = field(default_factory=list)
    pose_history_in: List[Optional[np.ndarray]] = field(default_factory=list)
    out: List[Optional[np.ndarray]] = field(default_factory=list)
    y_inc_raw: List[Optional[np.ndarray]] = field(default_factory=list)
    y_used_raw: List[Optional[np.ndarray]] = field(default_factory=list)
    motion_raw_after_carry: List[Optional[np.ndarray]] = field(default_factory=list)
    pose_hist_write_raw: List[Optional[np.ndarray]] = field(default_factory=list)


def _capture_run(
    case: Mapping[str, Any],
    *,
    rounds: int,
    teacher_conditioned: bool,
) -> Dict[str, Any]:
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError("trainer.model missing")
    runner = case["runner"]
    recorder = TraceRecorder()

    orig_forward = model.forward
    orig_compose = trainer._compose_delta_to_raw
    orig_apply_free_carry_raw = rfc._rollout_kernel.apply_free_carry_raw
    orig_hist_advance = rfc.advance_pose_hist_state_with_tail

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        motion = args[0] if len(args) > 0 else kwargs.get("state")
        pose_hist = kwargs.get("pose_history")
        if pose_hist is None and len(args) > 4:
            pose_hist = args[4]
        ret = orig_forward(*args, **kwargs)
        if isinstance(ret, dict) and ret.get("out") is not None:
            recorder.motion_in.append(_tensor_to_mean_vec(motion))
            recorder.pose_history_in.append(_tensor_to_mean_vec(pose_hist))
            recorder.out.append(_tensor_to_mean_vec(ret.get("out")))
        return ret

    def wrapped_compose(*args: Any, **kwargs: Any) -> Any:
        ret = orig_compose(*args, **kwargs)
        recorder.y_inc_raw.append(_tensor_to_mean_vec(ret))
        return ret

    def wrapped_apply_free_carry_raw(*args: Any, **kwargs: Any) -> Any:
        y_next_raw = kwargs.get("y_next_raw")
        if y_next_raw is None and len(args) > 1:
            y_next_raw = args[1]
        ret = orig_apply_free_carry_raw(*args, **kwargs)
        recorder.y_used_raw.append(_tensor_to_mean_vec(y_next_raw))
        recorder.motion_raw_after_carry.append(_tensor_to_mean_vec(ret))
        return ret

    def wrapped_hist_advance(state: Any, *, rot_tail_raw: Optional[torch.Tensor]) -> Any:
        recorder.pose_hist_write_raw.append(_tensor_to_mean_vec(rot_tail_raw))
        return orig_hist_advance(state, rot_tail_raw=rot_tail_raw)

    model.forward = wrapped_forward
    trainer._compose_delta_to_raw = wrapped_compose
    rfc._rollout_kernel.apply_free_carry_raw = wrapped_apply_free_carry_raw
    rfc.advance_pose_hist_state_with_tail = wrapped_hist_advance
    try:
        metrics_per_round, per_step, extra = rfc._run_freerun_cycles(
            trainer=trainer,
            sample=case["sample"],
            rounds=int(rounds),
            device=runner.device,
            time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
            lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            pose_hist_source=("seq" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_source"])),
            pose_hist_update_source=("gt" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_update_source"])),
            freerun_x_gt=bool(teacher_conditioned),
        )
    finally:
        model.forward = orig_forward
        trainer._compose_delta_to_raw = orig_compose
        rfc._rollout_kernel.apply_free_carry_raw = orig_apply_free_carry_raw
        rfc.advance_pose_hist_state_with_tail = orig_hist_advance

    expected = int(len(per_step))
    for name in ("motion_in", "pose_history_in", "out", "y_inc_raw", "y_used_raw", "motion_raw_after_carry"):
        got = len(getattr(recorder, name))
        if got != expected:
            raise RuntimeError(f"{case['case_name']} {name} length mismatch: trace={got} per_step={expected}")
    if len(recorder.pose_hist_write_raw) not in (0, expected):
        raise RuntimeError(
            f"{case['case_name']} pose_hist_write_raw length mismatch: trace={len(recorder.pose_hist_write_raw)} per_step={expected}"
        )

    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "signals": {
            "motion_in": recorder.motion_in,
            "pose_history_in": recorder.pose_history_in,
            "out": recorder.out,
            "y_inc_raw": recorder.y_inc_raw,
            "y_used_raw": recorder.y_used_raw,
            "motion_raw_after_carry": recorder.motion_raw_after_carry,
            "pose_hist_write_raw": recorder.pose_hist_write_raw,
        },
    }


def _series_from_per_step(per_step: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    return [_safe_float(rec.get(key)) for rec in per_step]


def _blend_equals_geo(per_step: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diffs: List[float] = []
    for rec in per_step:
        a = _safe_float(rec.get("GeoLocalDeg"))
        b = _safe_float(rec.get("BlendGeoLocalDeg"))
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(abs(a - b))
    arr = _finite(diffs)
    return {
        "samples": int(arr.size),
        "max_abs_diff": float(arr.max()) if arr.size > 0 else float("nan"),
        "mean_abs_diff": float(arr.mean()) if arr.size > 0 else float("nan"),
        "allclose_1e_9": bool(arr.size > 0 and float(arr.max()) <= 1e-9),
    }


def _nonnull_count(per_step: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(1 for rec in per_step if rec.get(key) is not None))


def _build_case_report(
    case: Mapping[str, Any],
    *,
    rounds: int,
) -> Dict[str, Any]:
    freerun = _capture_run(case, rounds=int(rounds), teacher_conditioned=False)
    teacher_run = _capture_run(case, rounds=int(rounds), teacher_conditioned=True)
    if len(freerun["per_step"]) != len(teacher_run["per_step"]):
        raise RuntimeError(
            f"{case['case_name']} per_step length mismatch: freerun={len(freerun['per_step'])} teacher={len(teacher_run['per_step'])}"
        )

    traces: Dict[str, Any] = {}
    for signal in PRIMARY_SIGNALS + APPENDIX_SIGNALS:
        traces[str(signal)] = _trace_metric(
            freerun["signals"].get(signal, []),
            teacher_run["signals"].get(signal, []),
        )

    per_step = freerun["per_step"]
    selected = _selected_steps(per_step)
    primary_selected = {
        signal: _offset_growth_summary(rows=selected, trace=traces[signal])
        for signal in PRIMARY_SIGNALS
    }

    y_used_vs_y_inc = _pair_metric(
        freerun["signals"]["y_used_raw"],
        freerun["signals"]["y_inc_raw"],
    )
    rot_slice = case["rot_slice"]
    y_used_rot = []
    for vec in freerun["signals"]["y_used_raw"]:
        if vec is None or not isinstance(rot_slice, slice):
            y_used_rot.append(None)
            continue
        y_used_rot.append(vec[int(rot_slice.start):int(rot_slice.stop)])
    pose_hist_write_vs_y_used_rot = _pair_metric(
        freerun["signals"]["pose_hist_write_raw"],
        y_used_rot,
    )

    timing = _timing_tables(
        per_step=per_step,
        traces={signal: traces[signal] for signal in PRIMARY_SIGNALS},
    )

    return {
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "eval_json_path": case["eval_json_path"],
        "teacher_path": case["teacher_path"],
        "runtime_overrides": case["runtime_overrides"],
        "selection": {
            "rounds": int(rounds),
            "selected_rows": int(len(selected)),
            "selected_definition": "depth>=10, step_in_cycle in [11,43], drop wrap, and require +20 horizon",
        },
        "fusion_runtime_status": {
            "lambda_fusion_apply_flag": bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            "lambda_mean_nonnull_steps": _nonnull_count(per_step, "LambdaMean"),
            "lambda_eff_mean_nonnull_steps": _nonnull_count(per_step, "LambdaEffMean"),
            "blend_equals_geo_local": _blend_equals_geo(per_step),
        },
        "trace_series": {
            "steps": int(len(per_step)),
            "step_meta": [
                {
                    "step": int(i),
                    "cycle": int(rec.get("cycle", 0) or 0),
                    "step_in_cycle": int(rec.get("step_in_cycle", -1) or -1),
                    "wrap_boundary_step": bool(rec.get("wrap_boundary_step", False)),
                }
                for i, rec in enumerate(per_step)
            ],
            "signals": traces,
            "GeoLocalDeg": _series_from_per_step(per_step, "GeoLocalDeg"),
            "BlendGeoLocalDeg": _series_from_per_step(per_step, "BlendGeoLocalDeg"),
            "DirectGeoLocalDeg": _series_from_per_step(per_step, "DirectGeoLocalDeg"),
        },
        "summary": {
            "selected_window": primary_selected,
            "timing": timing,
            "appendix": {
                "y_used_raw_drift": _offset_growth_summary(rows=selected, trace=traces["y_used_raw"]),
                "motion_raw_after_carry_drift": _offset_growth_summary(rows=selected, trace=traces["motion_raw_after_carry"]),
                "pose_hist_write_raw_drift": _offset_growth_summary(rows=selected, trace=traces["pose_hist_write_raw"]),
                "freerun_internal_checks": {
                    "y_used_raw_vs_y_inc_raw": y_used_vs_y_inc,
                    "pose_hist_write_raw_vs_y_used_rot_slice": pose_hist_write_vs_y_used_rot,
                },
            },
        },
    }


def _selected_compare(
    tail_case: Mapping[str, Any],
    base_case: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for signal in PRIMARY_SIGNALS:
        tail_sig = ((tail_case.get("summary", {}) or {}).get("selected_window", {}) or {}).get(signal, {})
        base_sig = ((base_case.get("summary", {}) or {}).get("selected_window", {}) or {}).get(signal, {})
        out[signal] = {
            "offsets": {},
            "growth": {},
        }
        for label, _off in OFFSET_SPECS:
            t_mean = _safe_float((((tail_sig.get("offsets", {}) or {}).get(label, {}) or {}).get("norm_l2", {}) or {}).get("mean"))
            b_mean = _safe_float((((base_sig.get("offsets", {}) or {}).get(label, {}) or {}).get("norm_l2", {}) or {}).get("mean"))
            out[signal]["offsets"][label] = {
                "tailk7_mean": t_mean,
                "baseline_mean": b_mean,
                "tail_minus_base": t_mean - b_mean,
            }
        for label, _a, _b in GROWTH_SPECS:
            t_mean = _safe_float(
                ((((tail_sig.get("growth", {}) or {}).get(label, {}) or {}).get("norm_l2_delta", {}) or {}).get("mean"))
            )
            b_mean = _safe_float(
                ((((base_sig.get("growth", {}) or {}).get(label, {}) or {}).get("norm_l2_delta", {}) or {}).get("mean"))
            )
            out[signal]["growth"][label] = {
                "tailk7_mean": t_mean,
                "baseline_mean": b_mean,
                "tail_minus_base": t_mean - b_mean,
            }
    return out


def _bucket_compare(
    tail_case: Mapping[str, Any],
    base_case: Mapping[str, Any],
    *,
    family: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tail_buckets = (((tail_case.get("summary", {}) or {}).get("timing", {}) or {}).get(family, {}) or {})
    base_buckets = (((base_case.get("summary", {}) or {}).get("timing", {}) or {}).get(family, {}) or {})
    labels = sorted(set(tail_buckets.keys()) | set(base_buckets.keys()))
    for label in labels:
        tail_row = tail_buckets.get(label, {}) or {}
        base_row = base_buckets.get(label, {}) or {}
        entry: Dict[str, Any] = {
            "rows": {
                "tailk7": int(tail_row.get("rows", 0) or 0),
                "baseline": int(base_row.get("rows", 0) or 0),
            },
            "GeoLocalDeg": {
                "tailk7_mean": _safe_float(((tail_row.get("GeoLocalDeg", {}) or {}).get("mean"))),
                "baseline_mean": _safe_float(((base_row.get("GeoLocalDeg", {}) or {}).get("mean"))),
            },
            "signals": {},
        }
        entry["GeoLocalDeg"]["tail_minus_base"] = (
            _safe_float(entry["GeoLocalDeg"]["tailk7_mean"]) - _safe_float(entry["GeoLocalDeg"]["baseline_mean"])
        )
        for signal in PRIMARY_SIGNALS:
            tail_sig = (((tail_row.get("signals", {}) or {}).get(signal, {}) or {}).get("norm_l2", {}) or {})
            base_sig = (((base_row.get("signals", {}) or {}).get(signal, {}) or {}).get("norm_l2", {}) or {})
            tail_mean = _safe_float(tail_sig.get("mean"))
            base_mean = _safe_float(base_sig.get("mean"))
            entry["signals"][signal] = {
                "tailk7_mean": tail_mean,
                "baseline_mean": base_mean,
                "tail_minus_base": tail_mean - base_mean,
            }
        out[label] = entry
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Main trunk drift audit for cp015 tailk7 current control vs baseline replace.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    teacher = args.teacher.expanduser().resolve()
    tail_ckpt = args.tail_ckpt.expanduser().resolve()
    tail_eval = args.tail_eval.expanduser().resolve()
    base_ckpt = args.baseline_ckpt.expanduser().resolve()
    base_eval = args.baseline_eval.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    for path in (teacher, tail_ckpt, tail_eval, base_ckpt, base_eval):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    tail_case = _load_case(
        case_name="tailk7_current_control",
        ckpt_path=tail_ckpt,
        eval_json_path=tail_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )
    base_case = _load_case(
        case_name="baseline_replace",
        ckpt_path=base_ckpt,
        eval_json_path=base_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )

    tail_report = _build_case_report(tail_case, rounds=int(args.rounds))
    base_report = _build_case_report(base_case, rounds=int(args.rounds))

    payload = {
        "analysis": "main_trunk_drift_audit",
        "teacher_batch": str(teacher),
        "definitions": {
            "teacher_conditioned_runtime": "Same multicycle _run_freerun_cycles path, but pose_hist_source='seq', pose_hist_update_source='gt', freerun_x_gt=True.",
            "freerun_runtime": "Eval json runtime overrides with pose_hist_source='buffer', pose_hist_update_source='pred', lambda_fusion_apply flag preserved.",
            "drift_metric": {
                "norm_l2": "||free - teacher||_2 / sqrt(D)",
                "mean_abs": "mean(abs(free - teacher))",
                "cosine_distance": "1 - cosine_similarity(free, teacher)",
            },
            "main_trunk_control_metric": "GeoLocalDeg from the incremental/main-trunk path (predY -> pred_raw_full -> geo_local_full -> per_step[*].GeoLocalDeg).",
            "direct_metric_note": "DirectGeoLocalDeg is exported separately from predY_direct and is not treated as the actual rollout-used metric in this audit.",
        },
        "cases": {
            "tailk7_current_control": tail_report,
            "baseline_replace": base_report,
        },
        "comparison": {
            "selected_window_primary_signals": _selected_compare(tail_report, base_report),
            "depth_buckets": _bucket_compare(tail_report, base_report, family="depth_buckets"),
            "sic_buckets": _bucket_compare(tail_report, base_report, family="sic_buckets"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
