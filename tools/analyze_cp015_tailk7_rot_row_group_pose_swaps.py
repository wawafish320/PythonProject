#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

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
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260405"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_rot_row_group_pose_swaps_{RUN_DATE}" / "summary.json"
)

PRIMARY_METRICS: tuple[str, ...] = (
    "Rot6dLocalL2",
    "Rot6dLocalL2Weighted",
    "GeoDeg",
    "KeyBoneGeoDegMean",
    "KeyBoneGeoLocalDegMean",
)
SECONDARY_METRICS: tuple[str, ...] = ("GeoLocalDeg",)
APPENDIX_METRICS: tuple[str, ...] = ("DirectGeoLocalDeg", "RootPosErr")
ALL_METRICS: tuple[str, ...] = PRIMARY_METRICS + SECONDARY_METRICS + APPENDIX_METRICS

DEPTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("d0_9", 0, 9),
    ("d10_20", 10, 20),
    ("d21_43", 21, 43),
)
SIC_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("sic0_10", 0, 10),
    ("sic11_21", 11, 21),
    ("sic22_43", 22, 43),
)
FOCUS_BUCKETS: tuple[str, ...] = ("d10_20", "d21_43", "sic11_21", "sic22_43")
ALPHA_VALUES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class RowGroupSpec:
    key: str
    label: str
    description: str
    joint_names: List[str]
    joint_indices: List[int]
    row_indices: List[int]
    row_ranges: List[List[int]]
    group_kind: str


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


def _summary(vals: Iterable[Any]) -> Dict[str, float]:
    arr = _finite(vals)
    if arr.size <= 0:
        return {
            "samples": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "samples": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _rows_for_depth(total: int, lo: int, hi: int) -> List[int]:
    lo_i = max(0, int(lo))
    hi_i = min(total - 1, int(hi))
    if hi_i < lo_i:
        return []
    return list(range(lo_i, hi_i + 1))


def _rows_for_sic(per_step_rows: Sequence[Mapping[str, Any]], lo: int, hi: int) -> List[int]:
    rows: List[int] = []
    for idx, rec in enumerate(per_step_rows):
        if bool(rec.get("wrap_boundary_step", False)):
            continue
        try:
            sic = int(rec.get("step_in_cycle", -1) or -1)
        except Exception:
            sic = -1
        if int(lo) <= sic <= int(hi):
            rows.append(int(idx))
    return rows


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _last_linear(module: nn.Module) -> nn.Linear:
    last: Optional[nn.Linear] = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last = child
    if last is None:
        raise RuntimeError("failed to find final Linear in motion_head")
    return last


def _zero_bone_adapters(model: nn.Module) -> None:
    adapters = list(getattr(model, "_bone_adapters", None) or [])
    with torch.no_grad():
        for adapter in adapters:
            alpha = getattr(adapter, "alpha", None)
            if torch.is_tensor(alpha):
                alpha.zero_()


def _replace_selected_rows(
    dst_model: nn.Module,
    src_model: nn.Module,
    *,
    row_indices: Sequence[int],
) -> None:
    dst_last = _last_linear(dst_model.motion_head)
    src_last = _last_linear(src_model.motion_head)
    if tuple(dst_last.weight.shape) != tuple(src_last.weight.shape):
        raise RuntimeError(
            f"final motion_head Linear shape mismatch: dst={tuple(dst_last.weight.shape)} src={tuple(src_last.weight.shape)}"
        )
    rows = sorted({int(i) for i in row_indices})
    if not rows:
        return
    if rows[0] < 0 or rows[-1] >= int(dst_last.weight.shape[0]):
        raise RuntimeError(f"row index out of range: min={rows[0]} max={rows[-1]} shape={tuple(dst_last.weight.shape)}")
    row_t = torch.as_tensor(rows, dtype=torch.long, device=dst_last.weight.device)
    with torch.no_grad():
        dst_last.weight.index_copy_(0, row_t, src_last.weight.detach().to(device=dst_last.weight.device).index_select(0, row_t))
        if dst_last.bias is not None and src_last.bias is not None:
            dst_last.bias.index_copy_(0, row_t, src_last.bias.detach().to(device=dst_last.bias.device).index_select(0, row_t))


def _alpha_blend_selected_rows(
    dst_model: nn.Module,
    src_model: nn.Module,
    *,
    row_indices: Sequence[int],
    alpha: float,
) -> None:
    dst_last = _last_linear(dst_model.motion_head)
    src_last = _last_linear(src_model.motion_head)
    if tuple(dst_last.weight.shape) != tuple(src_last.weight.shape):
        raise RuntimeError(
            f"final motion_head Linear shape mismatch: dst={tuple(dst_last.weight.shape)} src={tuple(src_last.weight.shape)}"
        )
    rows = sorted({int(i) for i in row_indices})
    if not rows:
        return
    a = float(alpha)
    if a <= 0.0:
        return
    if a >= 1.0:
        _replace_selected_rows(dst_model, src_model, row_indices=rows)
        return
    row_t = torch.as_tensor(rows, dtype=torch.long, device=dst_last.weight.device)
    with torch.no_grad():
        cur_w = dst_last.weight.index_select(0, row_t)
        src_w = src_last.weight.detach().to(device=dst_last.weight.device).index_select(0, row_t)
        dst_last.weight.index_copy_(0, row_t, ((1.0 - a) * cur_w) + (a * src_w))
        if dst_last.bias is not None and src_last.bias is not None:
            cur_b = dst_last.bias.index_select(0, row_t)
            src_b = src_last.bias.detach().to(device=dst_last.bias.device).index_select(0, row_t)
            dst_last.bias.index_copy_(0, row_t, ((1.0 - a) * cur_b) + (a * src_b))


def _run_case(case: Mapping[str, Any], *, rounds: int) -> Dict[str, Any]:
    runner = case["runner"]
    metrics_per_round, per_step, extra = _run_freerun_cycles(
        trainer=case["trainer"],
        sample=case["sample"],
        rounds=int(rounds),
        device=runner.device,
        time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
        lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
        pose_hist_source=str(case["runtime_overrides"]["pose_hist_source"]),
        pose_hist_update_source=str(case["runtime_overrides"]["pose_hist_update_source"]),
        debug_rot_gain=False,
        debug_so3_corr=False,
        export_joint_direct_geolocal_series=False,
        export_plan_state_series=False,
    )
    return {
        "metrics_per_round": metrics_per_round,
        "metrics_per_step": per_step,
        "extra": extra,
    }


def _row_ranges_from_rows(rows: Sequence[int]) -> List[List[int]]:
    uniq = sorted({int(x) for x in rows})
    if not uniq:
        return []
    out: List[List[int]] = []
    start = uniq[0]
    prev = uniq[0]
    for row in uniq[1:]:
        if row == prev + 1:
            prev = row
            continue
        out.append([int(start), int(prev) + 1])
        start = row
        prev = row
    out.append([int(start), int(prev) + 1])
    return out


def _joint_rows(rot_slice: slice, joint_idx: int) -> List[int]:
    start = int(rot_slice.start or 0) + (6 * int(joint_idx))
    return list(range(start, start + 6))


def _group_spec_from_joint_names(
    *,
    key: str,
    label: str,
    description: str,
    bone_names: Sequence[str],
    rot_slice: slice,
    joint_names: Sequence[str],
    group_kind: str,
) -> RowGroupSpec:
    name_to_idx = {str(name): int(i) for i, name in enumerate(bone_names)}
    missing = [str(name) for name in joint_names if str(name) not in name_to_idx]
    if missing:
        raise RuntimeError(f"group {key} missing joint names: {missing}")
    joint_indices = [name_to_idx[str(name)] for name in joint_names]
    row_indices: List[int] = []
    for idx in joint_indices:
        row_indices.extend(_joint_rows(rot_slice, idx))
    return RowGroupSpec(
        key=key,
        label=label,
        description=description,
        joint_names=[str(x) for x in joint_names],
        joint_indices=[int(x) for x in joint_indices],
        row_indices=sorted({int(x) for x in row_indices}),
        row_ranges=_row_ranges_from_rows(row_indices),
        group_kind=group_kind,
    )


def _build_group_specs(case: Mapping[str, Any]) -> Dict[str, RowGroupSpec]:
    bone_names = list(case["bone_names"])
    rot_slice = case["rot_slice"]
    full_rows = list(range(int(rot_slice.start or 0), int(rot_slice.stop or 0)))

    leg_broad_names = [
        str(name)
        for name in bone_names
        if any(token in str(name).lower() for token in ("thigh", "calf", "foot", "ball"))
    ]
    non_leg_names = [str(name) for name in bone_names if str(name) not in set(leg_broad_names)]

    specs = {
        "all_rot_rows": RowGroupSpec(
            key="all_rot_rows",
            label="All final rot rows [0:276]",
            description="Swap every final motion_head row that emits rot[0:276].",
            joint_names=list(bone_names),
            joint_indices=list(range(len(bone_names))),
            row_indices=full_rows,
            row_ranges=_row_ranges_from_rows(full_rows),
            group_kind="anchor",
        ),
        "thigh_pair": _group_spec_from_joint_names(
            key="thigh_pair",
            label="thigh_l + thigh_r",
            description="Swap only the exact `thigh_l` and `thigh_r` rot rows.",
            bone_names=bone_names,
            rot_slice=rot_slice,
            joint_names=("thigh_l", "thigh_r"),
            group_kind="user_group",
        ),
        "calf_pair": _group_spec_from_joint_names(
            key="calf_pair",
            label="calf_l + calf_r",
            description="Swap only the exact `calf_l` and `calf_r` rot rows.",
            bone_names=bone_names,
            rot_slice=rot_slice,
            joint_names=("calf_l", "calf_r"),
            group_kind="user_group",
        ),
        "foot_pair": _group_spec_from_joint_names(
            key="foot_pair",
            label="foot_l + foot_r",
            description="Swap only the exact `foot_l` and `foot_r` rot rows; `ball_*` stays tail.",
            bone_names=bone_names,
            rot_slice=rot_slice,
            joint_names=("foot_l", "foot_r"),
            group_kind="user_group",
        ),
        "all_leg_rows": _group_spec_from_joint_names(
            key="all_leg_rows",
            label="All leg rows",
            description=(
                "Swap all rot rows whose joint name contains any of `thigh|calf|foot|ball`; "
                "this includes calf twist rows."
            ),
            bone_names=bone_names,
            rot_slice=rot_slice,
            joint_names=leg_broad_names,
            group_kind="user_group",
        ),
        "non_leg_rows": _group_spec_from_joint_names(
            key="non_leg_rows",
            label="Non-leg rows",
            description="Swap the complement of `all_leg_rows` within rot[0:276].",
            bone_names=bone_names,
            rot_slice=rot_slice,
            joint_names=non_leg_names,
            group_kind="user_group",
        ),
    }
    return specs


def _mean_keybone_blob(blob: Any, *, root_name: str, exclude_root: bool) -> float:
    if not isinstance(blob, Mapping):
        return float("nan")
    vals: List[float] = []
    for name, value in blob.items():
        if exclude_root and str(name) == str(root_name):
            continue
        fv = _safe_float(value)
        if math.isfinite(fv):
            vals.append(fv)
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _step_metric_value(step: Mapping[str, Any], *, metric: str, root_name: str) -> float:
    if metric == "KeyBoneGeoDegMean":
        return _mean_keybone_blob(step.get("KeyBoneGeoDeg"), root_name=root_name, exclude_root=False)
    if metric == "KeyBoneGeoLocalDegMean":
        return _mean_keybone_blob(step.get("KeyBoneGeoLocalDeg"), root_name=root_name, exclude_root=True)
    return _safe_float(step.get(metric))


def _round_metric_value(round_row: Mapping[str, Any], *, metric: str) -> float:
    return _safe_float(round_row.get(metric))


def _metric_bucket_summary(
    per_step: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    root_name: str,
    row_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    if row_indices is None:
        values = [_step_metric_value(step, metric=metric, root_name=root_name) for step in per_step]
    else:
        values = [
            _step_metric_value(per_step[int(i)], metric=metric, root_name=root_name)
            for i in row_indices
            if 0 <= int(i) < len(per_step)
        ]
    return _summary(values)


def _round_metric_summary(metrics_per_round: Sequence[Mapping[str, Any]], *, metric: str) -> Dict[str, float]:
    return _summary(_round_metric_value(row, metric=metric) for row in metrics_per_round)


def _bucket_summaries(
    per_step: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    root_name: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in DEPTH_BUCKETS:
        out[name] = _metric_bucket_summary(
            per_step,
            metric=metric,
            root_name=root_name,
            row_indices=_rows_for_depth(len(per_step), lo, hi),
        )
    for name, lo, hi in SIC_BUCKETS:
        out[name] = _metric_bucket_summary(
            per_step,
            metric=metric,
            root_name=root_name,
            row_indices=_rows_for_sic(per_step, lo, hi),
        )
    return out


def _gap_closed(current_value: Any, variant_value: Any, anchor_value: Any) -> float:
    cur = _safe_float(current_value)
    var = _safe_float(variant_value)
    anc = _safe_float(anchor_value)
    denom = cur - anc
    if (not math.isfinite(cur)) or (not math.isfinite(var)) or (not math.isfinite(anc)) or abs(denom) <= 1e-12:
        return float("nan")
    return float((cur - var) / denom)


def _relative_improvement(current_value: Any, variant_value: Any) -> float:
    cur = _safe_float(current_value)
    var = _safe_float(variant_value)
    denom = abs(cur)
    if (not math.isfinite(cur)) or (not math.isfinite(var)) or denom <= 1e-12:
        return float("nan")
    return float((cur - var) / denom)


def _case_summary(run_payload: Mapping[str, Any], *, root_name: str) -> Dict[str, Any]:
    rounds = list(run_payload["metrics_per_round"])
    per_step = list(run_payload["metrics_per_step"])
    metrics: Dict[str, Any] = {}
    for metric in ALL_METRICS:
        metrics[metric] = {
            "rounds": _round_metric_summary(rounds, metric=metric),
            "steps": _metric_bucket_summary(per_step, metric=metric, root_name=root_name),
            "buckets": _bucket_summaries(per_step, metric=metric, root_name=root_name),
        }
    return {
        "rounds": int(len(rounds)),
        "steps": int(len(per_step)),
        "metrics": metrics,
    }


def _variant_metric_row(
    case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    variant_key: str,
    anchor_key: str,
    metric: str,
) -> Dict[str, Any]:
    variant = case_summaries[variant_key]
    tail = case_summaries["tail_current_control"]
    anchor = case_summaries[anchor_key]
    row: Dict[str, Any] = {
        "variant": variant_key,
        f"{metric}_mean": variant["metrics"][metric]["steps"]["mean"],
        f"{metric}_delta_vs_tail_current": (
            variant["metrics"][metric]["steps"]["mean"] - tail["metrics"][metric]["steps"]["mean"]
        ),
        f"{metric}_relative_improvement_vs_tail_current": _relative_improvement(
            tail["metrics"][metric]["steps"]["mean"],
            variant["metrics"][metric]["steps"]["mean"],
        ),
        f"{metric}_delta_vs_{anchor_key}": (
            variant["metrics"][metric]["steps"]["mean"] - anchor["metrics"][metric]["steps"]["mean"]
        ),
        f"{metric}_gap_closed_vs_{anchor_key}": _gap_closed(
            tail["metrics"][metric]["steps"]["mean"],
            variant["metrics"][metric]["steps"]["mean"],
            anchor["metrics"][metric]["steps"]["mean"],
        ),
    }
    for bucket in FOCUS_BUCKETS:
        row[bucket] = ((variant["metrics"].get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}).get("mean")
    return row


def _variant_primary_overview(
    case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    variant_key: str,
    anchor_key: str,
    include_metrics: Sequence[str],
) -> Dict[str, Any]:
    tail = case_summaries["tail_current_control"]
    anchor = case_summaries[anchor_key]
    variant = case_summaries[variant_key]

    improvement_rows: Dict[str, float] = {}
    gap_rows: Dict[str, float] = {}
    win_count = 0
    finite_count = 0
    for metric in include_metrics:
        cur = tail["metrics"][metric]["steps"]["mean"]
        var = variant["metrics"][metric]["steps"]["mean"]
        anc = anchor["metrics"][metric]["steps"]["mean"]
        improvement = _relative_improvement(cur, var)
        gap = _gap_closed(cur, var, anc)
        improvement_rows[metric] = improvement
        gap_rows[metric] = gap
        if math.isfinite(improvement):
            finite_count += 1
        if math.isfinite(_safe_float(cur)) and math.isfinite(_safe_float(var)) and _safe_float(var) < _safe_float(cur):
            win_count += 1

    focus_improvement_rows: Dict[str, float] = {}
    focus_gap_rows: Dict[str, float] = {}
    for bucket in FOCUS_BUCKETS:
        per_metric_improvement: List[float] = []
        per_metric: List[float] = []
        for metric in include_metrics:
            cur = ((tail["metrics"].get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}).get("mean")
            var = ((variant["metrics"].get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}).get("mean")
            anc = ((anchor["metrics"].get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}).get("mean")
            improvement = _relative_improvement(cur, var)
            gap = _gap_closed(cur, var, anc)
            if math.isfinite(improvement):
                per_metric_improvement.append(improvement)
            if math.isfinite(gap):
                per_metric.append(gap)
        focus_improvement_rows[bucket] = (
            float(np.mean(np.asarray(per_metric_improvement, dtype=np.float64))) if per_metric_improvement else float("nan")
        )
        focus_gap_rows[bucket] = float(np.mean(np.asarray(per_metric, dtype=np.float64))) if per_metric else float("nan")

    improvement_arr = _finite(improvement_rows.values())
    gap_arr = _finite(gap_rows.values())
    focus_improvement_arr = _finite(focus_improvement_rows.values())
    focus_arr = _finite(focus_gap_rows.values())
    return {
        "variant": variant_key,
        "primary_metric_relative_improvement_vs_tail": improvement_rows,
        "primary_metric_gap_closed": gap_rows,
        "primary_relative_improvement_mean": (
            float(np.mean(improvement_arr)) if improvement_arr.size > 0 else float("nan")
        ),
        "primary_relative_improvement_min": (
            float(np.min(improvement_arr)) if improvement_arr.size > 0 else float("nan")
        ),
        "primary_gap_closed_mean": float(np.mean(gap_arr)) if gap_arr.size > 0 else float("nan"),
        "primary_gap_closed_min": float(np.min(gap_arr)) if gap_arr.size > 0 else float("nan"),
        "primary_win_count_vs_tail": int(win_count),
        "primary_finite_metric_count": int(finite_count),
        "focus_bucket_relative_improvement_mean": (
            float(np.mean(focus_improvement_arr)) if focus_improvement_arr.size > 0 else float("nan")
        ),
        "focus_bucket_relative_improvement_vs_tail": focus_improvement_rows,
        "focus_bucket_gap_closed_mean": float(np.mean(focus_arr)) if focus_arr.size > 0 else float("nan"),
        "focus_bucket_gap_closed": focus_gap_rows,
        "secondary_GeoLocalDeg_relative_improvement_vs_tail": _relative_improvement(
            tail["metrics"]["GeoLocalDeg"]["steps"]["mean"],
            variant["metrics"]["GeoLocalDeg"]["steps"]["mean"],
        ),
        "secondary_GeoLocalDeg_gap_closed_vs_anchor": _gap_closed(
            tail["metrics"]["GeoLocalDeg"]["steps"]["mean"],
            variant["metrics"]["GeoLocalDeg"]["steps"]["mean"],
            anchor["metrics"]["GeoLocalDeg"]["steps"]["mean"],
        ),
    }


def _variant_ranking_table(
    case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    variant_keys: Sequence[str],
    anchor_key: str,
) -> List[Dict[str, Any]]:
    rows = [
        _variant_primary_overview(case_summaries, variant_key=key, anchor_key=anchor_key, include_metrics=PRIMARY_METRICS)
        for key in variant_keys
    ]
    rows.sort(
        key=lambda row: (
            _safe_float(row["primary_relative_improvement_mean"]),
            _safe_float(row["focus_bucket_relative_improvement_mean"]),
            int(row["primary_win_count_vs_tail"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _run_variant_with_restore(
    *,
    tail_model: nn.Module,
    baseline_model: nn.Module,
    tail_case: Mapping[str, Any],
    tail_original_state: Mapping[str, torch.Tensor],
    key: str,
    description: str,
    rounds: int,
    apply_fn: Callable[[nn.Module, nn.Module], None],
) -> Dict[str, Any]:
    print(f"[pose-swaps] running {key}")
    start = time.perf_counter()
    tail_model.load_state_dict(copy.deepcopy(dict(tail_original_state)), strict=True)
    tail_model.eval()
    apply_fn(tail_model, baseline_model)
    run_payload = _run_case(tail_case, rounds=int(rounds))
    elapsed = time.perf_counter() - start
    return {
        "description": description,
        "run_payload": run_payload,
        "elapsed_sec": float(elapsed),
    }


def _alpha_variant_key(group_key: str, alpha: float) -> str:
    return f"alpha_blend_{group_key}_a{int(round(alpha * 100)):03d}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="No-train pose-primary final-rot-row causal swaps on cp015 tailk7.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--alpha-top-k", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tail_case = _load_case(
        case_name="tail_current_control",
        ckpt_path=args.tail_ckpt,
        eval_json_path=args.tail_eval,
        teacher_path=args.teacher,
        device_pref=args.device,
    )
    baseline_case = _load_case(
        case_name="baseline_row_source",
        ckpt_path=args.baseline_ckpt,
        eval_json_path=args.baseline_eval,
        teacher_path=args.teacher,
        device_pref=args.device,
    )

    tail_model = tail_case["trainer"].model
    baseline_model = baseline_case["trainer"].model
    if tail_model is None or baseline_model is None:
        raise RuntimeError("failed to reconstruct tail/baseline models")

    rot_slice = tail_case["rot_slice"]
    if rot_slice != baseline_case["rot_slice"]:
        raise RuntimeError(f"rot_slice mismatch: tail={rot_slice} baseline={baseline_case['rot_slice']}")
    if list(tail_case["bone_names"]) != list(baseline_case["bone_names"]):
        raise RuntimeError("bone_names mismatch between tail and baseline row sources")

    root_name = str(tail_case["bone_names"][int(tail_case["root_idx"])])
    group_specs = _build_group_specs(tail_case)
    tail_original_state = _clone_state_dict(tail_model)

    variant_runs: Dict[str, Dict[str, Any]] = {}
    variant_meta: Dict[str, Dict[str, Any]] = {}

    base_variants: List[tuple[str, str, Callable[[nn.Module, nn.Module], None]]] = [
        (
            "tail_current_control",
            "Tail current control without any row swap.",
            lambda tail_m, base_m: None,
        ),
        (
            "swap_final_rot_rows_all",
            "Swap all final rot rows [0:276] from baseline into tail current control.",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["all_rot_rows"].row_indices),
        ),
        (
            "swap_final_rot_rows_thigh_pair",
            "Swap only `thigh_l + thigh_r` final rot rows.",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["thigh_pair"].row_indices),
        ),
        (
            "swap_final_rot_rows_calf_pair",
            "Swap only `calf_l + calf_r` final rot rows.",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["calf_pair"].row_indices),
        ),
        (
            "swap_final_rot_rows_foot_pair",
            "Swap only `foot_l + foot_r` final rot rows.",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["foot_pair"].row_indices),
        ),
        (
            "swap_final_rot_rows_all_leg_rows",
            "Swap all leg rows, including calf twist / foot / ball rows.",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["all_leg_rows"].row_indices),
        ),
        (
            "swap_final_rot_rows_non_leg_rows",
            "Swap the complement of leg rows inside rot[0:276].",
            lambda tail_m, base_m: _replace_selected_rows(tail_m, base_m, row_indices=group_specs["non_leg_rows"].row_indices),
        ),
        (
            "tail_current_zero_adapters",
            "Appendix control: keep tail head, zero all adapters.",
            lambda tail_m, base_m: _zero_bone_adapters(tail_m),
        ),
        (
            "swap_final_rot_rows_all_zero_adapters",
            "Appendix control: swap all rot rows then zero adapters.",
            lambda tail_m, base_m: (
                _replace_selected_rows(tail_m, base_m, row_indices=group_specs["all_rot_rows"].row_indices),
                _zero_bone_adapters(tail_m),
            ),
        ),
    ]

    for key, description, apply_fn in base_variants:
        result = _run_variant_with_restore(
            tail_model=tail_model,
            baseline_model=baseline_model,
            tail_case=tail_case,
            tail_original_state=tail_original_state,
            key=key,
            description=description,
            rounds=int(args.rounds),
            apply_fn=apply_fn,
        )
        variant_runs[key] = result
        variant_meta[key] = {
            "description": description,
            "elapsed_sec": result["elapsed_sec"],
        }

    case_summaries = {
        key: _case_summary(value["run_payload"], root_name=root_name)
        for key, value in variant_runs.items()
    }

    user_group_variant_keys = [
        "swap_final_rot_rows_thigh_pair",
        "swap_final_rot_rows_calf_pair",
        "swap_final_rot_rows_foot_pair",
        "swap_final_rot_rows_all_leg_rows",
        "swap_final_rot_rows_non_leg_rows",
    ]
    ranking = _variant_ranking_table(
        case_summaries,
        variant_keys=user_group_variant_keys,
        anchor_key="swap_final_rot_rows_all",
    )

    alpha_variant_meta: Dict[str, Dict[str, Any]] = {}
    alpha_selected_groups: List[Dict[str, Any]] = []
    alpha_candidates = [row for row in ranking if _safe_float(row["primary_relative_improvement_mean"]) > 0.0]
    alpha_groups = alpha_candidates[: max(0, int(args.alpha_top_k))]
    group_key_lookup = {
        "swap_final_rot_rows_thigh_pair": "thigh_pair",
        "swap_final_rot_rows_calf_pair": "calf_pair",
        "swap_final_rot_rows_foot_pair": "foot_pair",
        "swap_final_rot_rows_all_leg_rows": "all_leg_rows",
        "swap_final_rot_rows_non_leg_rows": "non_leg_rows",
    }
    for row in alpha_groups:
        base_variant_key = str(row["variant"])
        spec_key = group_key_lookup[base_variant_key]
        spec = group_specs[spec_key]
        alpha_selected_groups.append(
            {
                "variant": base_variant_key,
                "group_spec_key": spec_key,
                "label": spec.label,
                "primary_relative_improvement_mean": row["primary_relative_improvement_mean"],
                "focus_bucket_relative_improvement_mean": row["focus_bucket_relative_improvement_mean"],
                "primary_gap_closed_mean": row["primary_gap_closed_mean"],
                "focus_bucket_gap_closed_mean": row["focus_bucket_gap_closed_mean"],
            }
        )
        for alpha in ALPHA_VALUES:
            alpha_key = _alpha_variant_key(spec_key, alpha)
            if abs(float(alpha) - 1.0) <= 1e-12:
                case_summaries[alpha_key] = case_summaries[base_variant_key]
                alpha_variant_meta[alpha_key] = {
                    "source_variant": base_variant_key,
                    "group_spec_key": spec_key,
                    "alpha": float(alpha),
                    "description": f"Reuse exact swap result for {spec.label} at alpha=1.0.",
                    "elapsed_sec": 0.0,
                    "reused_existing_case": True,
                }
                continue
            result = _run_variant_with_restore(
                tail_model=tail_model,
                baseline_model=baseline_model,
                tail_case=tail_case,
                tail_original_state=tail_original_state,
                key=alpha_key,
                description=f"Alpha blend baseline rows into {spec.label} with alpha={alpha:.2f}.",
                rounds=int(args.rounds),
                apply_fn=lambda tail_m, base_m, rows=spec.row_indices, a=float(alpha): _alpha_blend_selected_rows(
                    tail_m,
                    base_m,
                    row_indices=rows,
                    alpha=a,
                ),
            )
            variant_runs[alpha_key] = result
            case_summaries[alpha_key] = _case_summary(result["run_payload"], root_name=root_name)
            alpha_variant_meta[alpha_key] = {
                "source_variant": base_variant_key,
                "group_spec_key": spec_key,
                "alpha": float(alpha),
                "description": f"Alpha blend baseline rows into {spec.label} with alpha={alpha:.2f}.",
                "elapsed_sec": result["elapsed_sec"],
                "reused_existing_case": False,
            }

    overall_metric_tables = {
        metric: [
            _variant_metric_row(case_summaries, variant_key=key, anchor_key="swap_final_rot_rows_all", metric=metric)
            for key in (
                "tail_current_control",
                "swap_final_rot_rows_all",
                *user_group_variant_keys,
                "tail_current_zero_adapters",
                "swap_final_rot_rows_all_zero_adapters",
            )
        ]
        for metric in ALL_METRICS
    }

    alpha_tables: Dict[str, List[Dict[str, Any]]] = {}
    for selected in alpha_selected_groups:
        spec_key = str(selected["group_spec_key"])
        rows: List[Dict[str, Any]] = []
        for alpha in ALPHA_VALUES:
            alpha_key = _alpha_variant_key(spec_key, alpha)
            if alpha_key not in case_summaries:
                continue
            row: Dict[str, Any] = {
                "variant": alpha_key,
                "group_spec_key": spec_key,
                "alpha": float(alpha),
            }
            for metric in PRIMARY_METRICS + SECONDARY_METRICS:
                row[metric] = case_summaries[alpha_key]["metrics"][metric]["steps"]["mean"]
                row[f"{metric}_delta_vs_tail_current"] = (
                    case_summaries[alpha_key]["metrics"][metric]["steps"]["mean"]
                    - case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"]
                )
                row[f"{metric}_relative_improvement_vs_tail_current"] = _relative_improvement(
                    case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                    case_summaries[alpha_key]["metrics"][metric]["steps"]["mean"],
                )
                row[f"{metric}_gap_closed_vs_full_swap"] = _gap_closed(
                    case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                    case_summaries[alpha_key]["metrics"][metric]["steps"]["mean"],
                    case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"],
                )
            rows.append(row)
        alpha_tables[spec_key] = rows

    all_leg_vs_non_leg = {
        metric: {
            "all_leg_mean": case_summaries["swap_final_rot_rows_all_leg_rows"]["metrics"][metric]["steps"]["mean"],
            "non_leg_mean": case_summaries["swap_final_rot_rows_non_leg_rows"]["metrics"][metric]["steps"]["mean"],
            "tail_mean": case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
            "full_swap_mean": case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"],
            "all_leg_gap_closed_vs_full_swap": _gap_closed(
                case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_all_leg_rows"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"],
            ),
            "all_leg_relative_improvement_vs_tail_current": _relative_improvement(
                case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_all_leg_rows"]["metrics"][metric]["steps"]["mean"],
            ),
            "non_leg_gap_closed_vs_full_swap": _gap_closed(
                case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_non_leg_rows"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"],
            ),
            "non_leg_relative_improvement_vs_tail_current": _relative_improvement(
                case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
                case_summaries["swap_final_rot_rows_non_leg_rows"]["metrics"][metric]["steps"]["mean"],
            ),
        }
        for metric in PRIMARY_METRICS + SECONDARY_METRICS
    }

    appendix_adapter_control = {
        metric: {
            "tail_current_control": case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"],
            "tail_current_zero_adapters": case_summaries["tail_current_zero_adapters"]["metrics"][metric]["steps"]["mean"],
            "swap_final_rot_rows_all": case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"],
            "swap_final_rot_rows_all_zero_adapters": case_summaries["swap_final_rot_rows_all_zero_adapters"]["metrics"][metric]["steps"]["mean"],
        }
        for metric in PRIMARY_METRICS + SECONDARY_METRICS + APPENDIX_METRICS
    }

    payload = {
        "analysis": "cp015_tailk7_pose_primary_final_rot_row_group_causal_swaps",
        "script_path": str(Path(__file__).resolve()),
        "summary_path": str(args.out.resolve()),
        "teacher_batch": str(args.teacher.resolve()),
        "parameters": {
            "rounds": int(args.rounds),
            "device": str(args.device),
            "alpha_top_k": int(args.alpha_top_k),
            "alpha_values": [float(x) for x in ALPHA_VALUES],
        },
        "metric_policy": {
            "primary_metrics": list(PRIMARY_METRICS),
            "secondary_metrics": list(SECONDARY_METRICS),
            "appendix_only_metrics": list(APPENDIX_METRICS),
            "primary_sort_rule": (
                "Rank user-requested row-group swaps by mean relative improvement vs `tail_current_control` across primary metrics only. "
                "Gap-closure vs `swap_final_rot_rows_all` is kept only as a secondary reference because the full-row swap is not monotonic "
                "on every primary metric. `DirectGeoLocalDeg` and `RootPosErr` are reported only in appendix tables."
            ),
        },
        "code_facts": {
            "forward_site": (
                "All variants start from the same tail current control checkpoint and only mutate the final `motion_head` "
                "Linear rows that emit rot[0:276] before calling `_run_freerun_cycles`."
            ),
            "rot_slice": [int(rot_slice.start or 0), int(rot_slice.stop or 0)],
            "joint_count": int((int(rot_slice.stop or 0) - int(rot_slice.start or 0)) // 6),
            "root_joint": {
                "index": int(tail_case["root_idx"]),
                "name": root_name,
            },
            "bone_names": list(tail_case["bone_names"]),
            "row_group_definition_note": (
                "`all_leg_rows` is the broad leg complement inside rot[0:276], defined by joint-name substring match "
                "on `thigh|calf|foot|ball`, so it includes calf twist rows. "
                "`thigh_pair`, `calf_pair`, and `foot_pair` are exact-name swaps only."
            ),
            "motion_head_last_linear_shape_tail": list(_last_linear(tail_model.motion_head).weight.shape),
            "motion_head_last_linear_shape_baseline": list(_last_linear(baseline_model.motion_head).weight.shape),
            "adapter_zero_method": "Set each `_bone_adapters[i].alpha := 0` in-memory before evaluation.",
        },
        "references": {
            "tail_current_control": {
                "ckpt_path": str(args.tail_ckpt.resolve()),
                "eval_json_path": str(args.tail_eval.resolve()),
                "runtime_overrides": dict(tail_case["runtime_overrides"]),
            },
            "baseline_row_source": {
                "ckpt_path": str(args.baseline_ckpt.resolve()),
                "eval_json_path": str(args.baseline_eval.resolve()),
                "runtime_overrides": dict(baseline_case["runtime_overrides"]),
            },
        },
        "row_group_specs": {
            key: {
                "label": spec.label,
                "description": spec.description,
                "group_kind": spec.group_kind,
                "joint_names": list(spec.joint_names),
                "joint_indices": list(spec.joint_indices),
                "row_indices": list(spec.row_indices),
                "row_ranges": list(spec.row_ranges),
                "row_count": int(len(spec.row_indices)),
            }
            for key, spec in group_specs.items()
        },
        "variant_meta": {
            **variant_meta,
            **alpha_variant_meta,
        },
        "cases": case_summaries,
        "tables": {
            "overall_metric_tables": overall_metric_tables,
            "user_group_primary_ranking": ranking,
            "all_leg_vs_non_leg": all_leg_vs_non_leg,
            "appendix_adapter_controls": appendix_adapter_control,
            "alpha_selected_groups": alpha_selected_groups,
            "alpha_tables": alpha_tables,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    stdout_summary = {
        "summary_path": str(args.out.resolve()),
        "tail_current_primary": {
            metric: case_summaries["tail_current_control"]["metrics"][metric]["steps"]["mean"] for metric in PRIMARY_METRICS
        },
        "swap_all_rot_rows_primary": {
            metric: case_summaries["swap_final_rot_rows_all"]["metrics"][metric]["steps"]["mean"] for metric in PRIMARY_METRICS
        },
        "top_user_groups": ranking[:3],
        "alpha_selected_groups": alpha_selected_groups,
    }
    print(json.dumps(stdout_summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
