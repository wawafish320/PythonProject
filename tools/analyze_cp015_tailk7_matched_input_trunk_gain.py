#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
    _direct_local_geo_deg,
    _load_case,
)
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260405"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_matched_input_trunk_gain_audit_{RUN_DATE}" / "summary.json"
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
CHANNEL_SPECS: tuple[tuple[str, bool, bool], ...] = (
    ("pose_history_only", False, True),
    ("motion_only", True, False),
    ("motion_plus_pose_history", True, True),
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


def _clone_tensor(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _flatten_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    return x.detach().reshape(-1).to(dtype=torch.float32)


def _norm_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if (not torch.is_tensor(a)) or (not torch.is_tensor(b)):
        return None
    va = _flatten_tensor(a)
    vb = _flatten_tensor(b)
    if va is None or vb is None:
        return None
    if tuple(va.shape) != tuple(vb.shape):
        return None
    if int(va.numel()) <= 0:
        return None
    diff = va - vb
    denom = math.sqrt(float(max(1, int(diff.numel()))))
    return float(torch.linalg.vector_norm(diff).item() / denom)


def _self_norm_l2(x: Optional[torch.Tensor]) -> Optional[float]:
    if not torch.is_tensor(x):
        return None
    vec = _flatten_tensor(x)
    if vec is None or int(vec.numel()) <= 0:
        return None
    denom = math.sqrt(float(max(1, int(vec.numel()))))
    return float(torch.linalg.vector_norm(vec).item() / denom)


def _gain(response: Optional[float], input_norm: Optional[float]) -> Optional[float]:
    rv = _safe_float(response)
    iv = _safe_float(input_norm)
    if (not math.isfinite(rv)) or (not math.isfinite(iv)) or iv <= 1e-12:
        return None
    return float(rv / iv)


def _clone_arg(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return x


def _rows_for_depth(per_step: Sequence[Mapping[str, Any]], lo: int, hi: int) -> List[int]:
    total = int(len(per_step))
    lo_i = max(0, int(lo))
    hi_i = min(total - 1, int(hi))
    if hi_i < lo_i:
        return []
    return list(range(lo_i, hi_i + 1))


def _rows_for_sic(per_step: Sequence[Mapping[str, Any]], lo: int, hi: int) -> List[int]:
    rows: List[int] = []
    for idx, rec in enumerate(per_step):
        if bool(rec.get("wrap_boundary_step", False)):
            continue
        try:
            sic = int(rec.get("step_in_cycle", -1) or -1)
        except Exception:
            sic = -1
        if int(lo) <= sic <= int(hi):
            rows.append(int(idx))
    return rows


def _aggregate_case_rows(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    return {
        "response_h_final": _summary(r.get(f"{prefix}_response_h_final") for r in rows),
        "response_out": _summary(r.get(f"{prefix}_response_out") for r in rows),
        "response_y_inc_raw": _summary(r.get(f"{prefix}_response_y_inc_raw") for r in rows),
        "gain_h_final": _summary(r.get(f"{prefix}_gain_h_final") for r in rows),
        "gain_out": _summary(r.get(f"{prefix}_gain_out") for r in rows),
        "gain_y_inc_raw": _summary(r.get(f"{prefix}_gain_y_inc_raw") for r in rows),
        "geo_local_deg_clean": _summary(r.get(f"{prefix}_geo_local_deg_clean") for r in rows),
        "geo_local_deg_perturbed": _summary(r.get(f"{prefix}_geo_local_deg_perturbed") for r in rows),
        "geo_local_deg_delta": _summary(r.get(f"{prefix}_geo_local_deg_delta") for r in rows),
        "geo_local_deg_abs_delta": _summary(r.get(f"{prefix}_geo_local_deg_abs_delta") for r in rows),
        "gain_geo_local_abs": _summary(r.get(f"{prefix}_gain_geo_local_abs") for r in rows),
    }


def _ratio_of_means(lhs: Mapping[str, Any], rhs: Mapping[str, Any], key: str) -> float:
    lv = _safe_float(((lhs.get(key, {}) or {}).get("mean")))
    rv = _safe_float(((rhs.get(key, {}) or {}).get("mean")))
    if (not math.isfinite(lv)) or (not math.isfinite(rv)) or abs(rv) <= 1e-12:
        return float("nan")
    return float(lv / rv)


def _aggregate_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    tail = _aggregate_case_rows(rows, "tail")
    base = _aggregate_case_rows(rows, "baseline")
    out = {
        "rows": int(len(rows)),
        "input_norm": _summary(r.get("input_norm") for r in rows),
        "delta_motion_norm": _summary(r.get("delta_motion_norm") for r in rows),
        "delta_pose_history_norm": _summary(r.get("delta_pose_history_norm") for r in rows),
        "freerun_geo_local_deg": {
            "tail": _summary(r.get("tail_freerun_geo_local_deg") for r in rows),
            "baseline": _summary(r.get("baseline_freerun_geo_local_deg") for r in rows),
        },
        "tail": tail,
        "baseline": base,
        "compare": {
            "gain_h_final_ratio_of_means": _ratio_of_means(tail, base, "gain_h_final"),
            "gain_out_ratio_of_means": _ratio_of_means(tail, base, "gain_out"),
            "gain_y_inc_raw_ratio_of_means": _ratio_of_means(tail, base, "gain_y_inc_raw"),
            "response_h_final_ratio_of_means": _ratio_of_means(tail, base, "response_h_final"),
            "response_out_ratio_of_means": _ratio_of_means(tail, base, "response_out"),
            "response_y_inc_raw_ratio_of_means": _ratio_of_means(tail, base, "response_y_inc_raw"),
            "gain_geo_local_abs_ratio_of_means": _ratio_of_means(tail, base, "gain_geo_local_abs"),
        },
    }
    return out


def _bucket_report(
    *,
    per_step_rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
) -> Dict[str, Any]:
    selected = [per_step_rows[int(i)] for i in row_indices if 0 <= int(i) < len(per_step_rows)]
    return _aggregate_rows(rows=selected)


def _repeat_gt_raw(case: Mapping[str, Any], rounds: int) -> List[torch.Tensor]:
    trainer = case["trainer"]
    gt_base = case["sample"]["gt_motion"].unsqueeze(0).to(case["runner"].device)
    gt_tiled = gt_base.repeat(1, int(rounds), 1)
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for idx in range(int(gt_tiled.shape[1])):
            out.append(trainer._denorm(gt_tiled[:, idx]).detach().clone())
    return out


def _geo_local_deg_from_raw(
    *,
    case: Mapping[str, Any],
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
) -> float:
    geo = _direct_local_geo_deg(
        pred_raw=pred_raw.detach().cpu(),
        gt_raw=gt_raw.detach().cpu(),
        rot_slice=case["rot_slice"],
        root_idx=int(case["root_idx"]),
        columns=case["columns"],
    ).detach().cpu()
    if int(geo.ndim) != 2:
        return float("nan")
    joint_mask = [idx for idx in range(int(geo.shape[1])) if int(idx) != int(case["root_idx"])]
    if not joint_mask:
        return 0.0
    vals = geo[:, joint_mask]
    return float(vals.mean().item())


def _resolve_counterfactual_contacts(
    *,
    case: Mapping[str, Any],
    motion: torch.Tensor,
    pose_history: Optional[torch.Tensor],
    fallback_contacts: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    trainer = case["trainer"]
    model = trainer.model
    if model is None or not bool(getattr(model, "contact_plan_enable", False)):
        return fallback_contacts
    try:
        contacts = trainer._predict_pretrain_contacts_from_frozen(
            motion_step_t=motion,
            pose_hist_step_t=pose_history,
        )
    except Exception:
        contacts = None
    if contacts is None:
        return fallback_contacts
    return contacts.detach()


def _resolve_counterfactual_angvel(
    *,
    case: Mapping[str, Any],
    motion: torch.Tensor,
    fallback_angvel: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    trainer = case["trainer"]
    angvel_slice = getattr(trainer, "angvel_x_slice", None)
    if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(angvel_slice, slice):
        try:
            return motion[..., angvel_slice].detach()
        except Exception:
            return fallback_angvel
    return fallback_angvel


def _resolve_y_inc_raw(
    *,
    case: Mapping[str, Any],
    y_prev_raw: Optional[torch.Tensor],
    out: torch.Tensor,
    ret: Mapping[str, Any],
) -> torch.Tensor:
    trainer = case["trainer"]
    if y_prev_raw is None:
        return trainer._denorm(out).detach()
    try:
        so3_gate = getattr(trainer, "so3_corr_gate_force", None)
        y_inc_raw = trainer._compose_delta_to_raw(
            y_prev_raw,
            out,
            omega_hat=ret.get("omega_hat", None) if bool(case["runtime_overrides"].get("so3_corr_apply", False)) else None,
            so3_gate=so3_gate,
            so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
            omega_detach=True,
        )
    except Exception:
        y_inc_raw = trainer._denorm(out)
    return y_inc_raw.detach()


def _counterfactual_one_step(
    *,
    case: Mapping[str, Any],
    clean_record: Mapping[str, Any],
    delta_motion: Optional[torch.Tensor],
    delta_pose_history: Optional[torch.Tensor],
    use_motion: bool,
    use_pose_history: bool,
    alpha: float,
    gt_raw: torch.Tensor,
) -> Dict[str, Any]:
    model = case["trainer"].model
    trainer = case["trainer"]
    if model is None:
        raise RuntimeError("trainer.model missing")

    base_motion = clean_record["motion_in"]
    if not torch.is_tensor(base_motion):
        raise RuntimeError("clean record missing motion_in tensor")
    motion = base_motion.detach().clone()
    pose_history = clean_record.get("pose_history_in", None)
    if torch.is_tensor(pose_history):
        pose_history = pose_history.detach().clone()

    dm = None
    dp = None
    if use_motion:
        if not torch.is_tensor(delta_motion):
            raise RuntimeError("motion channel requested but delta_motion is missing")
        if tuple(delta_motion.shape) != tuple(motion.shape):
            raise RuntimeError(
                f"motion delta shape mismatch: delta={tuple(delta_motion.shape)} base={tuple(motion.shape)}"
            )
        dm = (float(alpha) * delta_motion).detach()
        motion = motion + dm
    if use_pose_history:
        if not torch.is_tensor(delta_pose_history):
            raise RuntimeError("pose_history channel requested but delta_pose_history is missing")
        if not torch.is_tensor(pose_history):
            raise RuntimeError("pose_history channel requested but base pose_history is missing")
        if tuple(delta_pose_history.shape) != tuple(pose_history.shape):
            raise RuntimeError(
                f"pose_history delta shape mismatch: delta={tuple(delta_pose_history.shape)} base={tuple(pose_history.shape)}"
            )
        dp = (float(alpha) * delta_pose_history).detach()
        pose_history = pose_history + dp

    contacts = _resolve_counterfactual_contacts(
        case=case,
        motion=motion,
        pose_history=pose_history if torch.is_tensor(pose_history) else None,
        fallback_contacts=clean_record.get("contacts_in", None),
    )
    angvel = _resolve_counterfactual_angvel(
        case=case,
        motion=motion,
        fallback_angvel=clean_record.get("angvel_in", None),
    )

    with torch.no_grad():
        ret = model(
            motion,
            clean_record.get("cond_input", None),
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_history,
            plan_z=clean_record.get("plan_z_in", None),
            phase_event_age=clean_record.get("phase_event_age_in", None),
            meas_logits_prev=clean_record.get("meas_logits_prev_in", None),
            time_index=clean_record.get("time_index_in", None),
            rollout_step=clean_record.get("rollout_step_in", None),
        )
    if not isinstance(ret, Mapping):
        raise RuntimeError("counterfactual forward must return dict")
    out = ret.get("out", None)
    h_final = ret.get("h_final", None)
    if not torch.is_tensor(out) or not torch.is_tensor(h_final):
        raise RuntimeError("counterfactual forward missing h_final/out tensors")

    y_inc_raw = _resolve_y_inc_raw(
        case=case,
        y_prev_raw=clean_record.get("y_prev_raw", None),
        out=out,
        ret=ret,
    )
    geo_local_deg = _geo_local_deg_from_raw(case=case, pred_raw=y_inc_raw, gt_raw=gt_raw)

    input_parts = []
    if dm is not None:
        input_parts.append(dm)
    if dp is not None:
        input_parts.append(dp)
    input_vec = None
    if input_parts:
        input_vec = torch.cat([part.reshape(-1) for part in input_parts], dim=0)

    clean_h = clean_record.get("h_final", None)
    clean_out = clean_record.get("out", None)
    clean_y = clean_record.get("y_inc_raw", None)
    clean_geo = _safe_float(clean_record.get("geo_local_deg"))

    input_norm = _self_norm_l2(input_vec)
    response_h = _norm_l2(h_final, clean_h)
    response_out = _norm_l2(out, clean_out)
    response_y = _norm_l2(y_inc_raw, clean_y)
    geo_delta = float(geo_local_deg - clean_geo) if math.isfinite(clean_geo) else float("nan")
    geo_abs_delta = abs(geo_delta) if math.isfinite(geo_delta) else float("nan")

    return {
        "delta_motion_norm": _self_norm_l2(dm),
        "delta_pose_history_norm": _self_norm_l2(dp),
        "input_norm": input_norm,
        "response_h_final": response_h,
        "response_out": response_out,
        "response_y_inc_raw": response_y,
        "gain_h_final": _gain(response_h, input_norm),
        "gain_out": _gain(response_out, input_norm),
        "gain_y_inc_raw": _gain(response_y, input_norm),
        "geo_local_deg_clean": clean_geo,
        "geo_local_deg_perturbed": geo_local_deg,
        "geo_local_deg_delta": geo_delta,
        "geo_local_deg_abs_delta": geo_abs_delta,
        "gain_geo_local_abs": _gain(geo_abs_delta, input_norm),
    }


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
    records: List[Dict[str, Any]] = []

    orig_forward = model.forward
    orig_compose = trainer._compose_delta_to_raw

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        ret = orig_forward(*args, **kwargs)
        if isinstance(ret, Mapping) and ret.get("out") is not None:
            record = {
                "motion_in": _clone_tensor(args[0] if len(args) > 0 else kwargs.get("state", None)),
                "cond_input": _clone_tensor(args[1] if len(args) > 1 else kwargs.get("cond", None)),
                "contacts_in": _clone_tensor(kwargs.get("contacts", None)),
                "angvel_in": _clone_tensor(kwargs.get("angvel", None)),
                "pose_history_in": _clone_tensor(kwargs.get("pose_history", None)),
                "plan_z_in": _clone_tensor(kwargs.get("plan_z", None)),
                "phase_event_age_in": _clone_tensor(kwargs.get("phase_event_age", None)),
                "meas_logits_prev_in": _clone_tensor(kwargs.get("meas_logits_prev", None)),
                "time_index_in": _clone_arg(kwargs.get("time_index", None)),
                "rollout_step_in": _clone_tensor(kwargs.get("rollout_step", None)),
                "h_final": _clone_tensor(ret.get("h_final", None)),
                "out": _clone_tensor(ret.get("out", None)),
            }
            records.append(record)
        return ret

    def wrapped_compose(*args: Any, **kwargs: Any) -> Any:
        ret = orig_compose(*args, **kwargs)
        if records:
            rec = records[-1]
            rec["y_prev_raw"] = _clone_tensor(args[0] if len(args) > 0 else kwargs.get("y_raw_prev", None))
            rec["y_inc_raw"] = _clone_tensor(ret)
        return ret

    model.forward = wrapped_forward
    trainer._compose_delta_to_raw = wrapped_compose
    try:
        metrics_per_round, per_step, extra = _run_freerun_cycles(
            trainer=trainer,
            sample=case["sample"],
            rounds=int(rounds),
            device=runner.device,
            time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
            lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
            pose_hist_source=("seq" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_source"])),
            pose_hist_update_source=("gt" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_update_source"])),
            freerun_x_gt=bool(teacher_conditioned),
            debug_rot_gain=False,
            debug_so3_corr=False,
            export_plan_state_series=False,
            export_joint_direct_geolocal_series=False,
        )
    finally:
        model.forward = orig_forward
        trainer._compose_delta_to_raw = orig_compose

    if len(records) != len(per_step):
        raise RuntimeError(
            f"{case['case_name']} capture length mismatch: forward_records={len(records)} per_step={len(per_step)}"
        )

    gt_raw_series = _repeat_gt_raw(case, rounds=int(rounds))
    if len(gt_raw_series) < len(records):
        raise RuntimeError(
            f"{case['case_name']} gt_raw length mismatch: gt={len(gt_raw_series)} records={len(records)}"
        )

    for idx, rec in enumerate(records):
        rec["step"] = int(idx)
        rec["cycle"] = int((per_step[idx] or {}).get("cycle", 0) or 0)
        rec["step_in_cycle"] = int((per_step[idx] or {}).get("step_in_cycle", -1) or -1)
        rec["wrap_boundary_step"] = bool((per_step[idx] or {}).get("wrap_boundary_step", False))
        rec["gt_raw"] = gt_raw_series[idx]
        rec["geo_local_deg"] = _geo_local_deg_from_raw(
            case=case,
            pred_raw=rec["y_inc_raw"],
            gt_raw=gt_raw_series[idx],
        )
        rec["freerun_geo_local_deg"] = _safe_float((per_step[idx] or {}).get("GeoLocalDeg"))
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "records": records,
    }


def _build_channel_rows(
    *,
    tail_teacher: Mapping[str, Any],
    tail_free: Mapping[str, Any],
    base_teacher: Mapping[str, Any],
    base_free: Mapping[str, Any],
    tail_case: Mapping[str, Any],
    base_case: Mapping[str, Any],
    use_motion: bool,
    use_pose_history: bool,
    alpha: float,
) -> List[Dict[str, Any]]:
    tail_teacher_records = list(tail_teacher["records"])
    tail_free_records = list(tail_free["records"])
    base_teacher_records = list(base_teacher["records"])
    base_free_records = list(base_free["records"])

    total = min(
        len(tail_teacher_records),
        len(tail_free_records),
        len(base_teacher_records),
        len(base_free_records),
    )
    rows: List[Dict[str, Any]] = []
    for idx in range(total):
        t_clean = tail_teacher_records[idx]
        t_free = tail_free_records[idx]
        b_clean = base_teacher_records[idx]
        b_free = base_free_records[idx]

        delta_motion = None
        delta_pose = None
        if torch.is_tensor(t_clean.get("motion_in")) and torch.is_tensor(t_free.get("motion_in")):
            if tuple(t_clean["motion_in"].shape) != tuple(t_free["motion_in"].shape):
                raise RuntimeError(f"tail motion_in shape mismatch at step {idx}")
            delta_motion = t_free["motion_in"] - t_clean["motion_in"]
        if torch.is_tensor(t_clean.get("pose_history_in")) and torch.is_tensor(t_free.get("pose_history_in")):
            if tuple(t_clean["pose_history_in"].shape) != tuple(t_free["pose_history_in"].shape):
                raise RuntimeError(f"tail pose_history_in shape mismatch at step {idx}")
            delta_pose = t_free["pose_history_in"] - t_clean["pose_history_in"]

        tail_cf = _counterfactual_one_step(
            case=tail_case,
            clean_record=t_clean,
            delta_motion=delta_motion,
            delta_pose_history=delta_pose,
            use_motion=bool(use_motion),
            use_pose_history=bool(use_pose_history),
            alpha=float(alpha),
            gt_raw=t_clean["gt_raw"],
        )
        base_cf = _counterfactual_one_step(
            case=base_case,
            clean_record=b_clean,
            delta_motion=delta_motion,
            delta_pose_history=delta_pose,
            use_motion=bool(use_motion),
            use_pose_history=bool(use_pose_history),
            alpha=float(alpha),
            gt_raw=b_clean["gt_raw"],
        )

        row = {
            "step": int(idx),
            "cycle": int(t_clean.get("cycle", 0) or 0),
            "step_in_cycle": int(t_clean.get("step_in_cycle", -1) or -1),
            "wrap_boundary_step": bool(t_clean.get("wrap_boundary_step", False)),
            "input_norm": tail_cf["input_norm"],
            "delta_motion_norm": tail_cf["delta_motion_norm"],
            "delta_pose_history_norm": tail_cf["delta_pose_history_norm"],
            "tail_freerun_geo_local_deg": _safe_float(t_free.get("freerun_geo_local_deg")),
            "baseline_freerun_geo_local_deg": _safe_float(b_free.get("freerun_geo_local_deg")),
        }
        for prefix, payload in (("tail", tail_cf), ("baseline", base_cf)):
            for key, value in payload.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def _build_channel_report(
    *,
    channel_name: str,
    per_step_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    depth = {
        label: _bucket_report(
            per_step_rows=per_step_rows,
            row_indices=_rows_for_depth(per_step_rows, lo, hi),
        )
        for label, lo, hi in DEPTH_BUCKETS
    }
    sic = {
        label: _bucket_report(
            per_step_rows=per_step_rows,
            row_indices=_rows_for_sic(per_step_rows, lo, hi),
        )
        for label, lo, hi in SIC_BUCKETS
    }
    overall = _aggregate_rows(rows=per_step_rows)
    return {
        "channel": str(channel_name),
        "overall": overall,
        "depth_buckets": depth,
        "sic_buckets": sic,
        "per_step": list(per_step_rows),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Matched-input one-step trunk gain audit for cp015 tailk7 vs baseline.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
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

    rounds = int(args.rounds)
    alpha = float(args.alpha)

    tail_teacher = _capture_run(tail_case, rounds=rounds, teacher_conditioned=True)
    tail_free = _capture_run(tail_case, rounds=rounds, teacher_conditioned=False)
    base_teacher = _capture_run(base_case, rounds=rounds, teacher_conditioned=True)
    base_free = _capture_run(base_case, rounds=rounds, teacher_conditioned=False)

    channel_reports: Dict[str, Any] = {}
    for channel_name, use_motion, use_pose in CHANNEL_SPECS:
        per_step_rows = _build_channel_rows(
            tail_teacher=tail_teacher,
            tail_free=tail_free,
            base_teacher=base_teacher,
            base_free=base_free,
            tail_case=tail_case,
            base_case=base_case,
            use_motion=bool(use_motion),
            use_pose_history=bool(use_pose),
            alpha=alpha,
        )
        channel_reports[channel_name] = _build_channel_report(
            channel_name=channel_name,
            per_step_rows=per_step_rows,
        )

    payload = {
        "analysis": "matched_input_trunk_gain_audit",
        "teacher_batch": str(teacher),
        "summary_path": str(out_path),
        "script_path": str((ROOT / "tools" / "analyze_cp015_tailk7_matched_input_trunk_gain.py").resolve()),
        "parameters": {
            "rounds": rounds,
            "alpha": alpha,
            "device": str(args.device),
        },
        "code_facts": {
            "runtime_model_call": "validate._run_freerun_cycles calls model(motion, cond_input, contacts, angvel, pose_history, plan_z, phase_event_age, meas_logits_prev, time_index, rollout_step).",
            "shared_trunk_direct_concat": "EventMotionModel.forward builds x = concat([state, cond, plan_feat_for_inject]) where current ckpts use contact_plan_inject='plan_z'.",
            "pose_history_indirect_path": "pose_history does not concatenate directly into shared trunk x; it affects h_final through the frozen contact/period side path and the contact-plan/event-clock path.",
            "h_final_definition": "Exact tensor from EventMotionModel.forward: h_final = coupling_norm(...), then result['h_final'] = h_final.",
            "out_definition": "Exact tensor from EventMotionModel.forward: out = motion_head(h_final).",
            "y_inc_raw_definition": "validate._run_freerun_cycles composes y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, out, ...).",
            "geo_proxy_definition": "One-step main-trunk local geo is recomputed from y_inc_raw vs tiled gtY using the same root-relative SO(3) local geodesic convention as eval GeoLocalDeg; current eval contract already established y_used_raw == y_inc_raw for these runs.",
            "matched_input_definition": "Per-step delta source is tail freerun input minus tail teacher-conditioned input at the same global step; the same delta is injected into both tail and baseline teacher-conditioned base states.",
            "counterfactual_fixed_inputs": [
                "cond_input",
                "plan_z",
                "phase_event_age",
                "meas_logits_prev",
                "time_index",
                "rollout_step",
            ],
            "counterfactual_rederived_inputs": [
                "angvel_t from injected motion when use_freerun_state_sync=True",
                "contacts_in_t from trainer._predict_pretrain_contacts_from_frozen(injected motion, injected pose_history)",
            ],
        },
        "cases": {
            "tailk7_current_control": {
                "ckpt_path": tail_case["ckpt_path"],
                "eval_json_path": tail_case["eval_json_path"],
                "runtime_overrides": tail_case["runtime_overrides"],
            },
            "baseline_replace": {
                "ckpt_path": base_case["ckpt_path"],
                "eval_json_path": base_case["eval_json_path"],
                "runtime_overrides": base_case["runtime_overrides"],
            },
        },
        "channels": channel_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
