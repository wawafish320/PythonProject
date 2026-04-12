#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
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
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_motion_head_causal_swaps_{RUN_DATE}" / "summary.json"
)

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
FOCUS_DEPTHS: tuple[str, ...] = ("d10_20", "d21_43")
FOCUS_SICS: tuple[str, ...] = ("sic11_21", "sic22_43")


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


def _swap_full_motion_head(dst_model: nn.Module, src_model: nn.Module) -> None:
    dst_model.motion_head.load_state_dict(_clone_state_dict(src_model.motion_head), strict=True)


def _swap_final_rot_rows(dst_model: nn.Module, src_model: nn.Module, *, rot_stop: int) -> None:
    dst_last = _last_linear(dst_model.motion_head)
    src_last = _last_linear(src_model.motion_head)
    if tuple(dst_last.weight.shape) != tuple(src_last.weight.shape):
        raise RuntimeError(
            f"final motion_head Linear shape mismatch: dst={tuple(dst_last.weight.shape)} src={tuple(src_last.weight.shape)}"
        )
    if int(rot_stop) > int(dst_last.weight.shape[0]):
        raise RuntimeError(f"rot_stop {rot_stop} exceeds final head rows {dst_last.weight.shape[0]}")
    with torch.no_grad():
        dst_last.weight[: int(rot_stop)].copy_(src_last.weight[: int(rot_stop)])
        if dst_last.bias is not None and src_last.bias is not None:
            dst_last.bias[: int(rot_stop)].copy_(src_last.bias[: int(rot_stop)])


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


def _metric_bucket_summary(
    per_step: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    row_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    if row_indices is None:
        values = [step.get(metric) for step in per_step]
    else:
        values = [
            per_step[int(i)].get(metric)
            for i in row_indices
            if 0 <= int(i) < len(per_step)
        ]
    return _summary(values)


def _round_metric_summary(metrics_per_round: Sequence[Mapping[str, Any]], *, metric: str) -> Dict[str, float]:
    return _summary(row.get(metric) for row in metrics_per_round)


def _bucket_summaries(per_step: Sequence[Mapping[str, Any]], *, metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in DEPTH_BUCKETS:
        out[name] = _metric_bucket_summary(per_step, metric=metric, row_indices=_rows_for_depth(len(per_step), lo, hi))
    for name, lo, hi in SIC_BUCKETS:
        out[name] = _metric_bucket_summary(per_step, metric=metric, row_indices=_rows_for_sic(per_step, lo, hi))
    return out


def _gap_closed_fraction(current_value: Any, variant_value: Any, baseline_value: Any) -> float:
    cur = _safe_float(current_value)
    var = _safe_float(variant_value)
    base = _safe_float(baseline_value)
    denom = cur - base
    if (not math.isfinite(cur)) or (not math.isfinite(var)) or (not math.isfinite(base)) or abs(denom) <= 1e-12:
        return float("nan")
    return float((cur - var) / denom)


def _case_summary(run_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rounds = list(run_payload["metrics_per_round"])
    per_step = list(run_payload["metrics_per_step"])
    metrics = {}
    for metric in (
        "GeoDeg",
        "GeoLocalDeg",
        "DirectGeoLocalDeg",
        "Rot6dLocalL2",
        "Rot6dLocalL2Weighted",
        "DirectRot6dLocalL2",
        "RootPosErr",
    ):
        metrics[metric] = {
            "rounds": _round_metric_summary(rounds, metric=metric),
            "steps": _metric_bucket_summary(per_step, metric=metric),
            "buckets": _bucket_summaries(per_step, metric=metric),
        }
    return {
        "rounds": int(len(rounds)),
        "steps": int(len(per_step)),
        "metrics": metrics,
    }


def _make_overall_table(case_summaries: Mapping[str, Mapping[str, Any]], *, current_key: str, baseline_key: str) -> List[Dict[str, Any]]:
    current = case_summaries[current_key]
    baseline = case_summaries[baseline_key]
    out: List[Dict[str, Any]] = []
    for key, case in case_summaries.items():
        geo = case["metrics"]["GeoLocalDeg"]["steps"]["mean"]
        direct_geo = case["metrics"]["DirectGeoLocalDeg"]["steps"]["mean"]
        out.append(
            {
                "variant": key,
                "GeoLocalDeg_mean": geo,
                "DirectGeoLocalDeg_mean": direct_geo,
                "GeoLocalDeg_gap_closed_vs_tail_to_baseline": _gap_closed_fraction(
                    current["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                    geo,
                    baseline["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                ),
                "DirectGeoLocalDeg_gap_closed_vs_tail_to_baseline": _gap_closed_fraction(
                    current["metrics"]["DirectGeoLocalDeg"]["steps"]["mean"],
                    direct_geo,
                    baseline["metrics"]["DirectGeoLocalDeg"]["steps"]["mean"],
                ),
            }
        )
    return out


def _make_overall_metric_table(
    case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    current_key: str,
    baseline_key: str,
    metric: str,
) -> List[Dict[str, Any]]:
    current = case_summaries[current_key]
    baseline = case_summaries[baseline_key]
    out: List[Dict[str, Any]] = []
    field_name = f"{metric}_mean"
    gap_name = f"{metric}_gap_closed_vs_tail_to_baseline"
    for key, case in case_summaries.items():
        metric_value = case["metrics"][metric]["steps"]["mean"]
        out.append(
            {
                "variant": key,
                field_name: metric_value,
                gap_name: _gap_closed_fraction(
                    current["metrics"][metric]["steps"]["mean"],
                    metric_value,
                    baseline["metrics"][metric]["steps"]["mean"],
                ),
            }
        )
    return out


def _make_focus_table(
    case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    bucket_names: Sequence[str],
    metric: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, case in case_summaries.items():
        row: Dict[str, Any] = {"variant": key}
        for bucket in bucket_names:
            row[bucket] = ((case["metrics"].get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}).get("mean")
        out.append(row)
    return out


def _variant_specs(rot_stop: int) -> Dict[str, Dict[str, Any]]:
    return {
        "tail_current_control": {
            "label": "Tail current control",
            "apply": lambda tail_model, base_model: None,
            "description": "Tail current model unchanged.",
        },
        "swap_full_head_keep_adapters": {
            "label": "Tail current + baseline full motion_head + tail adapters",
            "apply": lambda tail_model, base_model: _swap_full_motion_head(tail_model, base_model),
            "description": "Replace the full motion_head with baseline weights, keep tail adapters.",
        },
        "swap_final_rot_rows_keep_adapters": {
            "label": "Tail current + baseline final rot rows [0:276] + rest tail",
            "apply": lambda tail_model, base_model: _swap_final_rot_rows(tail_model, base_model, rot_stop=int(rot_stop)),
            "description": "Only replace the final motion_head Linear rows that emit rot[0:276], keep tail earlier head layers and tail adapters.",
        },
        "zero_adapters_tail_head": {
            "label": "Tail current + tail motion_head + adapters zeroed",
            "apply": lambda tail_model, base_model: _zero_bone_adapters(tail_model),
            "description": "Keep tail motion_head, zero all bone adapter outputs by setting alpha=0.",
        },
        "swap_full_head_zero_adapters": {
            "label": "Tail current + baseline full motion_head + adapters zeroed",
            "apply": lambda tail_model, base_model: (_swap_full_motion_head(tail_model, base_model), _zero_bone_adapters(tail_model)),
            "description": "Replace full motion_head with baseline weights and zero all tail adapters.",
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="No-train motion_head causal swaps on cp015 tailk7 current control.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
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
        case_name="baseline_reference",
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
    rot_stop = int(rot_slice.stop or 0)
    tail_original_state = _clone_state_dict(tail_model)
    variant_specs = _variant_specs(rot_stop)

    print("[swap] running baseline_reference")
    baseline_run = _run_case(baseline_case, rounds=int(args.rounds))
    case_runs: Dict[str, Mapping[str, Any]] = {
        "baseline_reference": baseline_run,
    }

    for key, spec in variant_specs.items():
        print(f"[swap] running {key}")
        tail_model.load_state_dict(copy.deepcopy(tail_original_state), strict=True)
        tail_model.eval()
        spec["apply"](tail_model, baseline_model)
        case_runs[key] = _run_case(tail_case, rounds=int(args.rounds))

    case_summaries = {key: _case_summary(run_payload) for key, run_payload in case_runs.items()}

    payload = {
        "analysis": "motion_head_causal_swaps",
        "script_path": str(Path(__file__).resolve()),
        "summary_path": str(args.out.resolve()),
        "teacher_batch": str(args.teacher.resolve()),
        "parameters": {
            "rounds": int(args.rounds),
            "device": str(args.device),
        },
        "code_facts": {
            "tail_base_model": "All swap variants start from tail current control state_dict, then apply an in-memory no-train readout swap before _run_freerun_cycles.",
            "rot_rows": [0, int(rot_stop)],
            "motion_head_last_linear_shape_tail": list(_last_linear(tail_model.motion_head).weight.shape),
            "motion_head_last_linear_shape_baseline": list(_last_linear(baseline_model.motion_head).weight.shape),
            "bone_adapter_names": list(getattr(tail_model, "_bone_adapter_names", None) or []),
            "adapter_zero_method": "Set each _bone_adapters[i].alpha := 0, which makes adapter_i(h_final) identically 0 at inference.",
        },
        "references": {
            "tail_current_control": {
                "ckpt_path": str(args.tail_ckpt.resolve()),
                "eval_json_path": str(args.tail_eval.resolve()),
                "runtime_overrides": dict(tail_case["runtime_overrides"]),
            },
            "baseline_reference": {
                "ckpt_path": str(args.baseline_ckpt.resolve()),
                "eval_json_path": str(args.baseline_eval.resolve()),
                "runtime_overrides": dict(baseline_case["runtime_overrides"]),
            },
        },
        "variant_definitions": {
            key: {"label": str(spec["label"]), "description": str(spec["description"])}
            for key, spec in variant_specs.items()
        },
        "cases": case_summaries,
        "tables": {
            "overall": _make_overall_table(
                case_summaries,
                current_key="tail_current_control",
                baseline_key="baseline_reference",
            ),
            "Rot6dLocalL2_overall": _make_overall_metric_table(
                case_summaries,
                current_key="tail_current_control",
                baseline_key="baseline_reference",
                metric="Rot6dLocalL2",
            ),
            "Rot6dLocalL2_depth_focus": _make_focus_table(
                case_summaries,
                bucket_names=FOCUS_DEPTHS,
                metric="Rot6dLocalL2",
            ),
            "Rot6dLocalL2_sic_focus": _make_focus_table(
                case_summaries,
                bucket_names=FOCUS_SICS,
                metric="Rot6dLocalL2",
            ),
            "GeoLocalDeg_depth_focus": _make_focus_table(case_summaries, bucket_names=FOCUS_DEPTHS, metric="GeoLocalDeg"),
            "DirectGeoLocalDeg_depth_focus": _make_focus_table(
                case_summaries,
                bucket_names=FOCUS_DEPTHS,
                metric="DirectGeoLocalDeg",
            ),
            "GeoLocalDeg_sic_focus": _make_focus_table(case_summaries, bucket_names=FOCUS_SICS, metric="GeoLocalDeg"),
            "DirectGeoLocalDeg_sic_focus": _make_focus_table(
                case_summaries,
                bucket_names=FOCUS_SICS,
                metric="DirectGeoLocalDeg",
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.out.resolve()),
                "tail_current_Rot6dLocalL2_mean": payload["cases"]["tail_current_control"]["metrics"]["Rot6dLocalL2"]["steps"]["mean"],
                "baseline_Rot6dLocalL2_mean": payload["cases"]["baseline_reference"]["metrics"]["Rot6dLocalL2"]["steps"]["mean"],
                "tail_current_GeoLocalDeg_mean": payload["cases"]["tail_current_control"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                "baseline_GeoLocalDeg_mean": payload["cases"]["baseline_reference"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                "swap_full_head_keep_adapters_GeoLocalDeg_mean": payload["cases"]["swap_full_head_keep_adapters"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                "swap_final_rot_rows_keep_adapters_GeoLocalDeg_mean": payload["cases"]["swap_final_rot_rows_keep_adapters"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                "zero_adapters_tail_head_GeoLocalDeg_mean": payload["cases"]["zero_adapters_tail_head"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
                "swap_full_head_zero_adapters_GeoLocalDeg_mean": payload["cases"]["swap_full_head_zero_adapters"]["metrics"]["GeoLocalDeg"]["steps"]["mean"],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
