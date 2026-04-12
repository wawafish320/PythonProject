#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (  # noqa: E402
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    GROUP_KEYS,
    _load_case,
    _summary,
)
from tools.analyze_cp015_tailk7_single_step_rescue import _select_target_steps  # noqa: E402
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260404"
OFFSETS: Tuple[int, ...] = (0, 1, 5, 20)
HORIZONS: Tuple[int, ...] = (5, 20)
LEAD_LAG_MAX = 20
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_hfinal_drift_audit_{RUN_DATE}" / "summary.json"
)


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in vals:
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            out.append(fv)
    return np.asarray(out, dtype=np.float64)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = _finite(x)
    ya = _finite(y)
    n = int(min(xa.size, ya.size))
    if n < 2:
        return float("nan")
    xa = xa[:n]
    ya = ya[:n]
    if float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _future_mean(series: Sequence[float], start: int, horizon: int) -> Optional[float]:
    end = int(start) + int(horizon)
    if int(start) < 0 or end > len(series):
        return None
    arr = _finite(series[int(start):end])
    if arr.size <= 0:
        return None
    return float(np.mean(arr))


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
        out = vec.to(device="cpu", dtype=torch.float32).numpy().astype(np.float64, copy=False)
    except Exception:
        return None
    return out.reshape(-1)


def _trace_metric(
    freerun_vecs: Sequence[Optional[np.ndarray]],
    teacher_vecs: Sequence[Optional[np.ndarray]],
) -> Dict[str, Any]:
    total = int(min(len(freerun_vecs), len(teacher_vecs)))
    norm_l2: List[Optional[float]] = []
    cosine_distance: List[Optional[float]] = []
    dims: List[Optional[int]] = []
    valid: List[int] = []
    for idx in range(total):
        vf = freerun_vecs[idx]
        vt = teacher_vecs[idx]
        if vf is None or vt is None:
            norm_l2.append(None)
            cosine_distance.append(None)
            dims.append(None)
            valid.append(0)
            continue
        if tuple(vf.shape) != tuple(vt.shape):
            raise RuntimeError(
                f"trace shape mismatch at step {idx}: freerun={tuple(vf.shape)} teacher={tuple(vt.shape)}"
            )
        dim = int(vf.size)
        diff = vf - vt
        denom = math.sqrt(float(max(1, dim)))
        norm_l2.append(float(np.linalg.norm(diff) / denom))
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
        "dim": dims,
        "valid": valid,
    }


def _capture_hfinal_trace(
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
    captured: List[Optional[np.ndarray]] = []

    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        captured.append(_tensor_to_mean_vec(output))
        return output

    handle = model.coupling_norm.register_forward_hook(_hook)
    try:
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=trainer,
            sample=case["sample"],
            rounds=int(rounds),
            device=runner.device,
            time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
            lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            export_plan_state_series=True,
            pose_hist_source=("seq" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_source"])),
            pose_hist_update_source=("gt" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_update_source"])),
            freerun_x_gt=bool(teacher_conditioned),
        )
    finally:
        handle.remove()
    if len(captured) != len(per_step):
        raise RuntimeError(f"h_final trace length mismatch: hook={len(captured)} per_step={len(per_step)}")
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "h_final_vecs": captured,
    }


def _plan_z_trace(extra: Mapping[str, Any]) -> List[Optional[np.ndarray]]:
    payload = (extra or {}).get("plan_state_series", {})
    series = (payload or {}).get("series", {})
    plan = (series or {}).get("plan_z_in", {})
    data = plan.get("data", [])
    valid = plan.get("valid", [])
    out: List[Optional[np.ndarray]] = []
    for idx, row in enumerate(data if isinstance(data, list) else []):
        ok = bool(valid[idx]) if idx < len(valid) else False
        if not ok:
            out.append(None)
            continue
        try:
            arr = np.asarray(row, dtype=np.float64).reshape(-1)
        except Exception:
            arr = None
        out.append(arr if arr is not None and arr.size > 0 else None)
    return out


def _group_error_series(
    case: Mapping[str, Any],
    per_step: Sequence[Mapping[str, Any]],
) -> Dict[str, List[float]]:
    group_bones = {
        group: [case["bone_names"][int(i)] for i in case["groups"][group]]
        for group in GROUP_KEYS
    }
    out: Dict[str, List[float]] = {group: [] for group in GROUP_KEYS}
    for idx, rec in enumerate(per_step):
        keybone = rec.get("KeyBoneGeoLocalDeg", None)
        if not isinstance(keybone, Mapping):
            raise RuntimeError(f"missing KeyBoneGeoLocalDeg at step {idx}")
        for group in GROUP_KEYS:
            vals = _finite(keybone.get(str(name)) for name in group_bones[group])
            out[group].append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
    return out


def _offset_summary(
    *,
    selected_steps: Sequence[int],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for offset in OFFSETS:
        vals_l2: List[float] = []
        vals_cos: List[float] = []
        for step_idx in selected_steps:
            pos = int(step_idx) + int(offset)
            if pos >= int(trace["steps"]):
                continue
            valid = trace["valid"][pos]
            if not valid:
                continue
            v_l2 = trace["norm_l2"][pos]
            v_cos = trace["cosine_distance"][pos]
            if v_l2 is not None and math.isfinite(float(v_l2)):
                vals_l2.append(float(v_l2))
            if v_cos is not None and math.isfinite(float(v_cos)):
                vals_cos.append(float(v_cos))
        out[str(offset)] = {
            "samples": int(len(vals_l2)),
            "norm_l2": _summary(_finite(vals_l2)),
            "cosine_distance": _summary(_finite(vals_cos)),
        }
    return out


def _growth_summary(
    *,
    selected_steps: Sequence[int],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    spans = {
        "0_to_5": (0, 5),
        "5_to_20": (5, 20),
        "0_to_20": (0, 20),
    }
    out: Dict[str, Any] = {}
    for name, (off_a, off_b) in spans.items():
        vals_l2: List[float] = []
        vals_cos: List[float] = []
        for step_idx in selected_steps:
            pa = int(step_idx) + int(off_a)
            pb = int(step_idx) + int(off_b)
            if pb >= int(trace["steps"]):
                continue
            if not (trace["valid"][pa] and trace["valid"][pb]):
                continue
            va = trace["norm_l2"][pa]
            vb = trace["norm_l2"][pb]
            ca = trace["cosine_distance"][pa]
            cb = trace["cosine_distance"][pb]
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):
                vals_l2.append(float(vb) - float(va))
            if ca is not None and cb is not None and math.isfinite(float(ca)) and math.isfinite(float(cb)):
                vals_cos.append(float(cb) - float(ca))
        out[name] = {
            "samples": int(len(vals_l2)),
            "norm_l2_delta": _summary(_finite(vals_l2)),
            "cosine_distance_delta": _summary(_finite(vals_cos)),
        }
    return out


def _association_summary(
    *,
    selected_steps: Sequence[int],
    trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for offset in OFFSETS:
        offset_key = str(offset)
        out[offset_key] = {}
        for horizon in HORIZONS:
            h_key = f"horizon_{int(horizon)}"
            out[offset_key][h_key] = {}
            for group in GROUP_KEYS:
                xs_l2: List[float] = []
                xs_cos: List[float] = []
                ys_l2: List[float] = []
                ys_cos: List[float] = []
                series = group_errors[group]
                for step_idx in selected_steps:
                    pos = int(step_idx) + int(offset)
                    if pos >= int(trace["steps"]):
                        continue
                    if not trace["valid"][pos]:
                        continue
                    fut = _future_mean(series, pos, int(horizon))
                    if fut is None or not math.isfinite(float(fut)):
                        continue
                    v_l2 = trace["norm_l2"][pos]
                    v_cos = trace["cosine_distance"][pos]
                    if v_l2 is not None and math.isfinite(float(v_l2)):
                        xs_l2.append(float(v_l2))
                        ys_l2.append(float(fut))
                    if v_cos is not None and math.isfinite(float(v_cos)):
                        xs_cos.append(float(v_cos))
                        ys_cos.append(float(fut))
                out[offset_key][h_key][group] = {
                    "samples_norm_l2": int(min(len(xs_l2), len(ys_l2))),
                    "samples_cosine_distance": int(min(len(xs_cos), len(ys_cos))),
                    "pearson_r_norm_l2": _pearson(xs_l2, ys_l2),
                    "pearson_r_cosine_distance": _pearson(xs_cos, ys_cos),
                    "future_error_mean": _summary(_finite(ys_l2 if ys_l2 else ys_cos)),
                }
    return out


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


def _subset_summary(
    rows: Sequence[int],
    trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
    *,
    label: str,
) -> Dict[str, Any]:
    drift_vals = [trace["norm_l2"][i] for i in rows if i < int(trace["steps"]) and trace["valid"][i]]
    cos_vals = [trace["cosine_distance"][i] for i in rows if i < int(trace["steps"]) and trace["valid"][i]]
    growth_vals = [
        float(trace["norm_l2"][i]) - float(trace["norm_l2"][i - 1])
        for i in rows
        if i > 0
        and i < int(trace["steps"])
        and trace["valid"][i]
        and trace["valid"][i - 1]
        and trace["norm_l2"][i] is not None
        and trace["norm_l2"][i - 1] is not None
    ]
    out = {
        "label": str(label),
        "steps": int(len(rows)),
        "norm_l2": _summary(_finite(drift_vals)),
        "cosine_distance": _summary(_finite(cos_vals)),
        "norm_l2_growth_per_step": _summary(_finite(growth_vals)),
        "groups": {},
    }
    for group in GROUP_KEYS:
        vals = [group_errors[group][i] for i in rows if i < len(group_errors[group])]
        out["groups"][group] = _summary(_finite(vals))
    return out


def _timing_summaries(
    per_step: Sequence[Mapping[str, Any]],
    hfinal_trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    depth_specs = (
        ("d0_9", 0, 9),
        ("d10_20", 10, 20),
        ("d21_43", 21, 43),
        ("d44_86", 44, 86),
        ("d87_433", 87, max(87, len(per_step) - 1)),
    )
    sic_specs = (
        ("sic0_10", 0, 10),
        ("sic11_43", 11, 43),
        ("sic44_86", 44, 86),
    )
    depth_out: Dict[str, Any] = {}
    for label, lo, hi in depth_specs:
        rows = _mask_rows(per_step, depth_lo=int(lo), depth_hi=int(hi))
        depth_out[label] = _subset_summary(rows, hfinal_trace, group_errors, label=label)
    sic_out: Dict[str, Any] = {}
    for label, lo, hi in sic_specs:
        rows = _mask_rows(per_step, depth_lo=0, depth_hi=max(0, len(per_step) - 1), sic_lo=int(lo), sic_hi=int(hi), drop_wrap=True)
        sic_out[label] = _subset_summary(rows, hfinal_trace, group_errors, label=label)
    return {
        "depth_buckets": depth_out,
        "sic_buckets": sic_out,
    }


def _lead_lag_summary(
    plan_trace: Mapping[str, Any],
    hfinal_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    plan = [float(v) if v is not None and math.isfinite(float(v)) else float("nan") for v in plan_trace["norm_l2"]]
    hfin = [float(v) if v is not None and math.isfinite(float(v)) else float("nan") for v in hfinal_trace["norm_l2"]]
    plan_g = [float("nan")]
    hfin_g = [float("nan")]
    for idx in range(1, min(len(plan), len(hfin))):
        plan_g.append(plan[idx] - plan[idx - 1] if math.isfinite(plan[idx]) and math.isfinite(plan[idx - 1]) else float("nan"))
        hfin_g.append(hfin[idx] - hfin[idx - 1] if math.isfinite(hfin[idx]) and math.isfinite(hfin[idx - 1]) else float("nan"))

    lag_corr: Dict[str, float] = {}
    best_lag = 0
    best_corr = -float("inf")
    for lag in range(-LEAD_LAG_MAX, LEAD_LAG_MAX + 1):
        xs: List[float] = []
        ys: List[float] = []
        for idx in range(len(plan_g)):
            j = idx + int(lag)
            if j < 0 or j >= len(hfin_g):
                continue
            pv = plan_g[idx]
            hv = hfin_g[j]
            if not (math.isfinite(pv) and math.isfinite(hv)):
                continue
            xs.append(float(pv))
            ys.append(float(hv))
        corr = _pearson(xs, ys)
        lag_corr[str(lag)] = corr
        if math.isfinite(float(corr)) and float(corr) > float(best_corr):
            best_corr = float(corr)
            best_lag = int(lag)
    return {
        "metric": "first_difference_norm_l2",
        "lag_definition": "positive lag means plan_z drift growth leads h_final drift growth by that many steps",
        "lag_range": [-int(LEAD_LAG_MAX), int(LEAD_LAG_MAX)],
        "best_lag": int(best_lag),
        "best_pearson_r": float(best_corr) if math.isfinite(float(best_corr)) else float("nan"),
        "lag_to_pearson_r": lag_corr,
    }


def _selected_window_cycle_offsets(
    selected_meta: Sequence[Any],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cycles = sorted({int(getattr(meta, "cycle", 0)) for meta in selected_meta})
    for cycle in cycles:
        rows = [int(getattr(meta, "step_idx")) for meta in selected_meta if int(getattr(meta, "cycle", 0)) == int(cycle)]
        row_out: Dict[str, Any] = {"samples": int(len(rows))}
        for offset in (0, 5, 20):
            vals = [
                float(trace["norm_l2"][pos])
                for base in rows
                for pos in [int(base) + int(offset)]
                if pos < int(trace["steps"])
                and trace["valid"][pos]
                and trace["norm_l2"][pos] is not None
                and math.isfinite(float(trace["norm_l2"][pos]))
            ]
            row_out[f"offset_{int(offset)}_norm_l2"] = _summary(_finite(vals))
        out[str(cycle)] = row_out
    return out


def _selected_window_bucket_summary(
    *,
    selected_meta: Sequence[Any],
    trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
    key: str,
    specs: Sequence[Tuple[str, int, int]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label, lo, hi in specs:
        rows = [meta for meta in selected_meta if int(lo) <= int(getattr(meta, key)) <= int(hi)]
        offset0: List[float] = []
        offset5: List[float] = []
        offset20: List[float] = []
        growth05: List[float] = []
        growth520: List[float] = []
        future_h20: Dict[str, List[float]] = {group: [] for group in GROUP_KEYS}
        for meta in rows:
            t = int(getattr(meta, "step_idx"))
            if t + 20 >= int(trace["steps"]):
                continue
            if not (trace["valid"][t] and trace["valid"][t + 5] and trace["valid"][t + 20]):
                continue
            v0 = trace["norm_l2"][t]
            v5 = trace["norm_l2"][t + 5]
            v20 = trace["norm_l2"][t + 20]
            if v0 is None or v5 is None or v20 is None:
                continue
            if not (math.isfinite(float(v0)) and math.isfinite(float(v5)) and math.isfinite(float(v20))):
                continue
            offset0.append(float(v0))
            offset5.append(float(v5))
            offset20.append(float(v20))
            growth05.append(float(v5) - float(v0))
            growth520.append(float(v20) - float(v5))
            for group in GROUP_KEYS:
                fut = _future_mean(group_errors[group], t, 20)
                if fut is not None and math.isfinite(float(fut)):
                    future_h20[group].append(float(fut))
        out[str(label)] = {
            "samples": int(len(offset0)),
            "offset_0_norm_l2": _summary(_finite(offset0)),
            "offset_5_norm_l2": _summary(_finite(offset5)),
            "offset_20_norm_l2": _summary(_finite(offset20)),
            "growth_0_to_5_norm_l2": _summary(_finite(growth05)),
            "growth_5_to_20_norm_l2": _summary(_finite(growth520)),
            "future_horizon20_error": {
                group: _summary(_finite(vals))
                for group, vals in future_h20.items()
            },
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Teacher-vs-freerun h_final drift audit for cp015 tailk7 control.")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth-min", type=int, default=10)
    ap.add_argument("--sic-lo", type=int, default=11)
    ap.add_argument("--sic-hi", type=int, default=43)
    ap.add_argument("--drop-wrap", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = args.ckpt.expanduser().resolve()
    eval_json = args.eval.expanduser().resolve()
    teacher = args.teacher.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    for path in (ckpt, eval_json, teacher):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    case = _load_case(
        case_name="tailk7_factorized_control",
        ckpt_path=ckpt,
        eval_json_path=eval_json,
        teacher_path=teacher,
        device_pref=str(args.device),
    )
    selected_meta = _select_target_steps(
        eval_json,
        depth_min=int(args.depth_min),
        sic_lo=int(args.sic_lo),
        sic_hi=int(args.sic_hi),
        drop_wrap=bool(args.drop_wrap),
    )
    if not selected_meta:
        raise SystemExit("[FATAL] no target steps selected")
    selected_steps = [int(m.step_idx) for m in selected_meta]

    freerun = _capture_hfinal_trace(case, rounds=int(args.rounds), teacher_conditioned=False)
    teacher_run = _capture_hfinal_trace(case, rounds=int(args.rounds), teacher_conditioned=True)
    if len(freerun["per_step"]) != len(teacher_run["per_step"]):
        raise RuntimeError(
            f"per_step length mismatch: freerun={len(freerun['per_step'])} teacher={len(teacher_run['per_step'])}"
        )

    hfinal_trace = _trace_metric(freerun["h_final_vecs"], teacher_run["h_final_vecs"])
    plan_trace = _trace_metric(_plan_z_trace(freerun["extra"]), _plan_z_trace(teacher_run["extra"]))
    group_errors = _group_error_series(case, freerun["per_step"])

    payload: Dict[str, Any] = {
        "analysis": "h_final_drift_audit",
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "eval_json_path": case["eval_json_path"],
        "teacher_path": case["teacher_path"],
        "runtime_overrides": case["runtime_overrides"],
        "selection": {
            "depth_min": int(args.depth_min),
            "sic_range": [int(args.sic_lo), int(args.sic_hi)],
            "drop_wrap": bool(args.drop_wrap),
            "selected_steps": int(len(selected_steps)),
        },
        "metric_definition": {
            "hidden_drift": {
                "norm_l2": "||free - teacher||_2 / sqrt(D)",
                "cosine_distance": "1 - cosine_similarity(free, teacher)",
            },
            "error": "used_local_geo_deg approximated from metrics_per_step[*].KeyBoneGeoLocalDeg and grouped by arm/all_ex_root/leg.",
            "offsets": [int(v) for v in OFFSETS],
            "horizons": [int(v) for v in HORIZONS],
            "note": "Both traces come from the same multicycle runtime path via _run_freerun_cycles; h_final is captured from model.coupling_norm, and plan_z uses exported plan_state_series.plan_z_in.",
        },
        "trace_series": {
            "steps": int(hfinal_trace["steps"]),
            "step_meta": [
                {
                    "step": int(i),
                    "cycle": int(rec.get("cycle", 0) or 0),
                    "step_in_cycle": int(rec.get("step_in_cycle", -1) or -1),
                    "wrap_boundary_step": bool(rec.get("wrap_boundary_step", False)),
                }
                for i, rec in enumerate(freerun["per_step"])
            ],
            "h_final": hfinal_trace,
            "plan_z": plan_trace,
            "used_local_geo_deg": {group: list(vals) for group, vals in group_errors.items()},
        },
        "summary": {
            "h_final_offsets": _offset_summary(selected_steps=selected_steps, trace=hfinal_trace),
            "h_final_growth": _growth_summary(selected_steps=selected_steps, trace=hfinal_trace),
            "h_final_future_error_association": _association_summary(
                selected_steps=selected_steps,
                trace=hfinal_trace,
                group_errors=group_errors,
            ),
            "selected_window_cycle_offsets": _selected_window_cycle_offsets(selected_meta, hfinal_trace),
            "selected_window_depth_buckets": _selected_window_bucket_summary(
                selected_meta=selected_meta,
                trace=hfinal_trace,
                group_errors=group_errors,
                key="step_idx",
                specs=(
                    ("d10_20", 10, 20),
                    ("d21_43", 21, 43),
                    ("d87_173", 87, 173),
                    ("d174_433", 174, 433),
                ),
            ),
            "selected_window_sic_buckets": _selected_window_bucket_summary(
                selected_meta=selected_meta,
                trace=hfinal_trace,
                group_errors=group_errors,
                key="step_in_cycle",
                specs=(
                    ("sic11_21", 11, 21),
                    ("sic22_43", 22, 43),
                ),
            ),
            "timing": _timing_summaries(
                freerun["per_step"],
                hfinal_trace,
                group_errors,
            ),
            "plan_z_offsets": _offset_summary(selected_steps=selected_steps, trace=plan_trace),
            "plan_z_growth": _growth_summary(selected_steps=selected_steps, trace=plan_trace),
            "plan_z_lead_lag_vs_h_final": _lead_lag_summary(plan_trace, hfinal_trace),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
