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
    DEFAULT_BASELINE_CKPT,
    DEFAULT_BASELINE_EVAL,
    DEFAULT_TAIL_CKPT,
    DEFAULT_TAIL_EVAL,
    DEFAULT_TEACHER,
    GROUP_KEYS,
    _load_case,
    _summary,
)
from tools.analyze_cp015_tailk7_hfinal_drift import (  # noqa: E402
    _growth_summary,
    _mask_rows,
    _offset_summary,
    _selected_window_bucket_summary,
    _selected_window_cycle_offsets,
    _subset_summary,
    _trace_metric,
)
from tools.analyze_cp015_tailk7_single_step_rescue import _select_target_steps  # noqa: E402
from tools.phasea_group_summary import _pick_group_indices  # noqa: E402
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260404"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_plan_meas_drift_audit_{RUN_DATE}" / "summary.json"
)
DEPTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("d0_9", 0, 9),
    ("d10_20", 10, 20),
    ("d21_43", 21, 43),
    ("d44_86", 44, 86),
    ("d87_433", 87, 433),
)
SIC_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("sic0_10", 0, 10),
    ("sic11_21", 11, 21),
    ("sic22_43", 22, 43),
    ("sic44_86", 44, 86),
)
SELECTED_DEPTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("d10_20", 10, 20),
    ("d21_43", 21, 43),
    ("d87_173", 87, 173),
    ("d174_433", 174, 433),
)
SELECTED_SIC_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("sic11_21", 11, 21),
    ("sic22_43", 22, 43),
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


def _vec_from_per_c(row: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    raw = row.get(key, None)
    if raw is None:
        return None
    try:
        vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if vec.size <= 0 or not np.isfinite(vec).all():
        return None
    return vec


def _contact_series(per_step: Sequence[Mapping[str, Any]], key: str) -> List[Optional[np.ndarray]]:
    return [_vec_from_per_c(rec, key) for rec in per_step]


def _contact_trace(
    freerun_per_step: Sequence[Mapping[str, Any]],
    teacher_per_step: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    freerun_vecs = _contact_series(freerun_per_step, key)
    teacher_vecs = _contact_series(teacher_per_step, key)
    trace = _trace_metric(freerun_vecs, teacher_vecs)
    mean_abs: List[Optional[float]] = []
    total = int(min(len(freerun_vecs), len(teacher_vecs)))
    for idx in range(total):
        vf = freerun_vecs[idx]
        vt = teacher_vecs[idx]
        if vf is None or vt is None or tuple(vf.shape) != tuple(vt.shape):
            mean_abs.append(None)
            continue
        mean_abs.append(float(np.mean(np.abs(vf - vt))))
    trace["mean_abs"] = mean_abs
    return trace


def _group_error_series(eval_obj: Mapping[str, Any]) -> Dict[str, List[float]]:
    per = eval_obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(per, Mapping):
        raise RuntimeError("missing per_step_direct_geolocal_deg in freerun eval json")
    names = per.get("bone_names", [])
    mat = per.get("DirectGeoLocalDeg", [])
    if not isinstance(names, list) or not isinstance(mat, list) or not names or not mat:
        raise RuntimeError("invalid per_step_direct_geolocal_deg payload")
    try:
        root_idx = int(per.get("root_idx", 0) or 0)
    except Exception:
        root_idx = 0
    group_idx = _pick_group_indices([str(x) for x in names], root_idx)
    out: Dict[str, List[float]] = {group: [] for group in GROUP_KEYS}
    for idx, row in enumerate(mat):
        if not isinstance(row, list):
            raise RuntimeError(f"invalid DirectGeoLocalDeg row at step {idx}")
        for group in GROUP_KEYS:
            vals = _finite(row[int(i)] for i in group_idx[group] if int(i) < len(row))
            out[group].append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
    return out


def _timing_summaries(
    per_step: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    depth_out: Dict[str, Any] = {}
    for label, lo, hi in DEPTH_BUCKETS:
        rows = _mask_rows(per_step, depth_lo=int(lo), depth_hi=int(min(int(hi), max(0, len(per_step) - 1))))
        depth_out[label] = _subset_summary(rows, trace, group_errors, label=label)
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
        sic_out[label] = _subset_summary(rows, trace, group_errors, label=label)
    return {
        "depth_buckets": depth_out,
        "sic_buckets": sic_out,
    }


def _comparison_offset_growth(signal_a: Mapping[str, Any], signal_b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"offsets": {}, "growth": {}}
    for offset in ("0", "5", "20"):
        a_mean = _safe_float(((signal_a.get("offsets", {}) or {}).get(offset, {}) or {}).get("norm_l2", {}).get("mean"))
        b_mean = _safe_float(((signal_b.get("offsets", {}) or {}).get(offset, {}) or {}).get("norm_l2", {}).get("mean"))
        out["offsets"][offset] = {
            "tailk7_mean": a_mean,
            "baseline_mean": b_mean,
            "tail_minus_base": a_mean - b_mean,
        }
    for span in ("0_to_5", "5_to_20", "0_to_20"):
        a_mean = _safe_float(
            ((signal_a.get("growth", {}) or {}).get(span, {}) or {}).get("norm_l2_delta", {}).get("mean")
        )
        b_mean = _safe_float(
            ((signal_b.get("growth", {}) or {}).get(span, {}) or {}).get("norm_l2_delta", {}).get("mean")
        )
        out["growth"][span] = {
            "tailk7_mean": a_mean,
            "baseline_mean": b_mean,
            "tail_minus_base": a_mean - b_mean,
        }
    return out


def _compare_bucket_family(
    tail_signal: Mapping[str, Any],
    base_signal: Mapping[str, Any],
    tail_errors: Mapping[str, Any],
    base_errors: Mapping[str, Any],
) -> Dict[str, Any]:
    labels = sorted(set(tail_signal.keys()) | set(base_signal.keys()) | set(tail_errors.keys()) | set(base_errors.keys()))
    out: Dict[str, Any] = {}
    for label in labels:
        row: Dict[str, Any] = {}
        tail_drift = _safe_float(((tail_signal.get(label, {}) or {}).get("norm_l2", {}) or {}).get("mean"))
        base_drift = _safe_float(((base_signal.get(label, {}) or {}).get("norm_l2", {}) or {}).get("mean"))
        row["drift_norm_l2"] = {
            "tailk7_mean": tail_drift,
            "baseline_mean": base_drift,
            "tail_minus_base": tail_drift - base_drift,
        }
        for group in GROUP_KEYS:
            tail_err = _safe_float((((tail_errors.get(label, {}) or {}).get("groups", {}) or {}).get(group, {}) or {}).get("mean"))
            base_err = _safe_float((((base_errors.get(label, {}) or {}).get("groups", {}) or {}).get(group, {}) or {}).get("mean"))
            row[f"{group}_used_local_geo_deg"] = {
                "tailk7_mean": tail_err,
                "baseline_mean": base_err,
                "tail_minus_base": tail_err - base_err,
            }
        out[label] = row
    return out


def _source_applied_values(per_step: Sequence[Mapping[str, Any]]) -> List[str]:
    values = sorted({str(rec.get("ContactsMeasSourceApplied")) for rec in per_step if rec.get("ContactsMeasSourceApplied") is not None})
    return values


def _nonnull_override_steps(per_step: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(1 for rec in per_step if rec.get(key) is not None))


def _capture_teacher_contacts(case: Mapping[str, Any], *, rounds: int) -> Dict[str, Any]:
    trainer = case["trainer"]
    runner = case["runner"]
    prev_log_contacts = bool(getattr(trainer, "log_contacts", False))
    prev_direct_meas = getattr(trainer, "direct_pose_meas_source", None)
    prev_direct_plan = getattr(trainer, "direct_pose_plan_source", None)
    prev_contacts_meas = getattr(trainer, "contacts_meas_source", None)
    try:
        trainer.log_contacts = True
        trainer.direct_pose_meas_source = str(case["runtime_overrides"]["direct_pose_meas_source"])
        trainer.direct_pose_plan_source = str(case["runtime_overrides"]["direct_pose_plan_source"])
        trainer.contacts_meas_source = str(case["runtime_overrides"]["contacts_meas_source"])
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=trainer,
            sample=case["sample"],
            rounds=int(rounds),
            device=runner.device,
            time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
            lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            pose_hist_source="seq",
            pose_hist_update_source="gt",
            freerun_x_gt=True,
        )
    finally:
        trainer.log_contacts = prev_log_contacts
        trainer.direct_pose_meas_source = prev_direct_meas
        trainer.direct_pose_plan_source = prev_direct_plan
        trainer.contacts_meas_source = prev_contacts_meas
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
    }


def _signal_summary(
    *,
    selected_meta: Sequence[Any],
    trace: Mapping[str, Any],
    group_errors: Mapping[str, Sequence[float]],
    per_step: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected_steps = [int(meta.step_idx) for meta in selected_meta]
    return {
        "offsets": _offset_summary(selected_steps=selected_steps, trace=trace),
        "growth": _growth_summary(selected_steps=selected_steps, trace=trace),
        "selected_window_cycle_offsets": _selected_window_cycle_offsets(selected_meta, trace),
        "selected_window_depth_buckets": _selected_window_bucket_summary(
            selected_meta=selected_meta,
            trace=trace,
            group_errors=group_errors,
            key="step_idx",
            specs=SELECTED_DEPTH_BUCKETS,
        ),
        "selected_window_sic_buckets": _selected_window_bucket_summary(
            selected_meta=selected_meta,
            trace=trace,
            group_errors=group_errors,
            key="step_in_cycle",
            specs=SELECTED_SIC_BUCKETS,
        ),
        "timing": _timing_summaries(per_step, trace, group_errors),
    }


def _run_case(
    *,
    case_name: str,
    ckpt_path: Path,
    eval_json_path: Path,
    teacher_path: Path,
    device_pref: str,
    rounds: int,
    depth_min: int,
    sic_lo: int,
    sic_hi: int,
    drop_wrap: bool,
) -> Dict[str, Any]:
    ckpt_payload = torch.load(str(ckpt_path), map_location="cpu")
    model_state = ckpt_payload.get("model", {}) if isinstance(ckpt_payload, Mapping) else {}
    legacy_phase_keys = sorted(
        str(k)
        for k in (model_state.keys() if isinstance(model_state, Mapping) else [])
        if str(k).startswith("contact_plan_phase_head.") or str(k).startswith("contact_phase_state_")
    )
    case = _load_case(
        case_name=case_name,
        ckpt_path=ckpt_path,
        eval_json_path=eval_json_path,
        teacher_path=teacher_path,
        device_pref=device_pref,
    )
    freerun_obj = json.loads(eval_json_path.read_text())
    freerun_per_step = freerun_obj.get("metrics_per_step", [])
    if not isinstance(freerun_per_step, list) or not freerun_per_step:
        raise RuntimeError(f"{case_name}: invalid freerun eval json: {eval_json_path}")

    teacher_run = _capture_teacher_contacts(case, rounds=int(rounds))
    teacher_per_step = teacher_run["per_step"]
    if not isinstance(teacher_per_step, list) or not teacher_per_step:
        raise RuntimeError(f"{case_name}: teacher-conditioned run returned no per_step records")

    steps = int(min(len(freerun_per_step), len(teacher_per_step)))
    freerun_per_step = freerun_per_step[:steps]
    teacher_per_step = teacher_per_step[:steps]
    selected_meta = _select_target_steps(
        eval_json_path,
        depth_min=int(depth_min),
        sic_lo=int(sic_lo),
        sic_hi=int(sic_hi),
        drop_wrap=bool(drop_wrap),
    )
    selected_meta = [meta for meta in selected_meta if int(meta.step_idx) + 20 < steps]
    if not selected_meta:
        raise RuntimeError(f"{case_name}: no selected steps after length alignment")

    group_errors = _group_error_series(freerun_obj)
    plan_trace = _contact_trace(freerun_per_step, teacher_per_step, "ContactPlanPerC")
    meas_trace = _contact_trace(freerun_per_step, teacher_per_step, "ContactMeasPerC")
    model = case["trainer"].model

    return {
        "case_name": case_name,
        "ckpt_path": str(ckpt_path),
        "eval_json_path": str(eval_json_path),
        "teacher_path": str(teacher_path),
        "runtime_overrides": case["runtime_overrides"],
        "runtime_facts": {
            "model": {
                "direct_pose_feat_source": getattr(model, "direct_pose_feat_source", None),
                "direct_pose_meas_mode": getattr(model, "direct_pose_meas_mode", None),
                "direct_pose_detach_plan": bool(getattr(model, "direct_pose_detach_plan", False)),
                "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
                "contact_meas_head_present": getattr(model, "contact_meas_head", None) is not None,
                "contact_meas_enable": getattr(model, "contact_meas_enable", None),
                "legacy_phase_key_count_in_ckpt": int(len(legacy_phase_keys)),
                "legacy_phase_key_examples": legacy_phase_keys[:8],
            },
            "freerun_eval_artifact": {
                "steps": int(len(freerun_per_step)),
                "contacts_meas_source_applied_values": _source_applied_values(freerun_per_step),
                "direct_meas_override_nonnull_steps": _nonnull_override_steps(freerun_per_step, "DirectMeasOverridePerC"),
                "direct_plan_override_nonnull_steps": _nonnull_override_steps(freerun_per_step, "DirectPlanOverridePerC"),
            },
            "teacher_conditioned_runtime": {
                "steps": int(len(teacher_per_step)),
                "contacts_meas_source_applied_values": _source_applied_values(teacher_per_step),
                "direct_meas_override_nonnull_steps": _nonnull_override_steps(teacher_per_step, "DirectMeasOverridePerC"),
                "direct_plan_override_nonnull_steps": _nonnull_override_steps(teacher_per_step, "DirectPlanOverridePerC"),
            },
            "consumed_tensor_definition": {
                "contacts_plan": "Under direct_pose_plan_source=model with no DirectPlanOverridePerC and eval-mode dropout/noise disabled, direct head consumes plan_in=contacts_plan (detach is value-preserving). Main tables therefore use ContactPlanPerC.",
                "contacts_meas": "Under direct_pose_meas_source=model with no DirectMeasOverridePerC and eval-mode dropout/noise disabled, direct head consumes meas_in=clamp(contacts_meas,0,1). Main tables therefore use ContactMeasPerC, which already lies in [0,1] in the audited traces.",
            },
        },
        "selection": {
            "depth_min": int(depth_min),
            "sic_range": [int(sic_lo), int(sic_hi)],
            "drop_wrap": bool(drop_wrap),
            "selected_steps": int(len(selected_meta)),
        },
        "metric_definition": {
            "drift": {
                "norm_l2": "||free - teacher||_2 / sqrt(D)",
                "cosine_distance": "1 - cosine_similarity(free, teacher)",
                "mean_abs": "mean(abs(free - teacher)) across contact channels",
            },
            "error": "used_local_geo_deg from freerun_eval_json.per_step_direct_geolocal_deg['DirectGeoLocalDeg'], grouped exactly like tools/phasea_group_summary.py into arm/all_ex_root/leg.",
            "note": "freerun side comes from the fixed reference eval JSON; teacher side is a fresh teacher-conditioned _run_freerun_cycles pass with the same runtime overrides and log_contacts enabled.",
        },
        "trace_series": {
            "steps": int(steps),
            "step_meta": [
                {
                    "step": int(i),
                    "cycle": int(rec.get("cycle", 0) or 0),
                    "step_in_cycle": int(rec.get("step_in_cycle", -1) or -1),
                    "wrap_boundary_step": bool(rec.get("wrap_boundary_step", False)),
                }
                for i, rec in enumerate(freerun_per_step)
            ],
            "contacts_plan_drift": plan_trace,
            "contacts_meas_drift": meas_trace,
            "used_local_geo_deg": {group: list(vals) for group, vals in group_errors.items()},
        },
        "summary": {
            "contacts_plan": _signal_summary(
                selected_meta=selected_meta,
                trace=plan_trace,
                group_errors=group_errors,
                per_step=freerun_per_step,
            ),
            "contacts_meas": _signal_summary(
                selected_meta=selected_meta,
                trace=meas_trace,
                group_errors=group_errors,
                per_step=freerun_per_step,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan/meas drift audit for cp015 tailk7 vs baseline replace.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
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
    teacher = args.teacher.expanduser().resolve()
    tail_ckpt = args.tail_ckpt.expanduser().resolve()
    tail_eval = args.tail_eval.expanduser().resolve()
    base_ckpt = args.baseline_ckpt.expanduser().resolve()
    base_eval = args.baseline_eval.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    for path in (teacher, tail_ckpt, tail_eval, base_ckpt, base_eval):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    tail = _run_case(
        case_name="tailk7_current_control",
        ckpt_path=tail_ckpt,
        eval_json_path=tail_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
        rounds=int(args.rounds),
        depth_min=int(args.depth_min),
        sic_lo=int(args.sic_lo),
        sic_hi=int(args.sic_hi),
        drop_wrap=bool(args.drop_wrap),
    )
    baseline = _run_case(
        case_name="baseline_replace",
        ckpt_path=base_ckpt,
        eval_json_path=base_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
        rounds=int(args.rounds),
        depth_min=int(args.depth_min),
        sic_lo=int(args.sic_lo),
        sic_hi=int(args.sic_hi),
        drop_wrap=bool(args.drop_wrap),
    )

    payload = {
        "analysis": "plan_meas_drift_audit",
        "teacher_batch": str(teacher),
        "cases": {
            "tailk7_current_control": tail,
            "baseline_replace": baseline,
        },
        "comparison": {
            "contacts_plan": _comparison_offset_growth(
                tail["summary"]["contacts_plan"],
                baseline["summary"]["contacts_plan"],
            ),
            "contacts_meas": _comparison_offset_growth(
                tail["summary"]["contacts_meas"],
                baseline["summary"]["contacts_meas"],
            ),
            "pattern_vs_error": {
                "contacts_plan": {
                    "depth_buckets": _compare_bucket_family(
                        tail["summary"]["contacts_plan"]["timing"]["depth_buckets"],
                        baseline["summary"]["contacts_plan"]["timing"]["depth_buckets"],
                        tail["summary"]["contacts_plan"]["timing"]["depth_buckets"],
                        baseline["summary"]["contacts_plan"]["timing"]["depth_buckets"],
                    ),
                    "sic_buckets": _compare_bucket_family(
                        tail["summary"]["contacts_plan"]["timing"]["sic_buckets"],
                        baseline["summary"]["contacts_plan"]["timing"]["sic_buckets"],
                        tail["summary"]["contacts_plan"]["timing"]["sic_buckets"],
                        baseline["summary"]["contacts_plan"]["timing"]["sic_buckets"],
                    ),
                },
                "contacts_meas": {
                    "depth_buckets": _compare_bucket_family(
                        tail["summary"]["contacts_meas"]["timing"]["depth_buckets"],
                        baseline["summary"]["contacts_meas"]["timing"]["depth_buckets"],
                        tail["summary"]["contacts_meas"]["timing"]["depth_buckets"],
                        baseline["summary"]["contacts_meas"]["timing"]["depth_buckets"],
                    ),
                    "sic_buckets": _compare_bucket_family(
                        tail["summary"]["contacts_meas"]["timing"]["sic_buckets"],
                        baseline["summary"]["contacts_meas"]["timing"]["sic_buckets"],
                        tail["summary"]["contacts_meas"]["timing"]["sic_buckets"],
                        baseline["summary"]["contacts_meas"]["timing"]["sic_buckets"],
                    ),
                },
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
