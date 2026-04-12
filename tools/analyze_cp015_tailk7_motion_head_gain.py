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
import torch.nn as nn
import torch.nn.functional as F

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
from tools.analyze_cp015_tailk7_matched_input_trunk_gain import (  # noqa: E402
    _capture_run,
    _geo_local_deg_from_raw,
    _resolve_counterfactual_angvel,
    _resolve_counterfactual_contacts,
    _resolve_y_inc_raw,
)


RUN_DATE = "20260405"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_motion_head_gain_audit_{RUN_DATE}" / "summary.json"
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


def _tensor_l2(x: Optional[torch.Tensor]) -> Optional[float]:
    if not torch.is_tensor(x):
        return None
    vec = x.detach().reshape(-1).to(dtype=torch.float32)
    if int(vec.numel()) <= 0:
        return None
    return float(torch.linalg.vector_norm(vec).item())


def _diff_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor], sl: Optional[slice] = None) -> Optional[float]:
    if (not torch.is_tensor(a)) or (not torch.is_tensor(b)):
        return None
    xa = a.detach()
    xb = b.detach()
    if sl is not None:
        xa = xa[..., sl]
        xb = xb[..., sl]
    if tuple(xa.shape) != tuple(xb.shape):
        return None
    return _tensor_l2(xa - xb)


def _gain(response: Optional[float], inp: Optional[float]) -> Optional[float]:
    rv = _safe_float(response)
    iv = _safe_float(inp)
    if (not math.isfinite(rv)) or (not math.isfinite(iv)) or iv <= 1e-12:
        return None
    return float(rv / iv)


def _ratio(lhs: Any, rhs: Any) -> float:
    lv = _safe_float(lhs)
    rv = _safe_float(rhs)
    if (not math.isfinite(lv)) or (not math.isfinite(rv)) or abs(rv) <= 1e-12:
        return float("nan")
    return float(lv / rv)


def _flatten_named_modules(module: nn.Module, prefix: str) -> List[tuple[str, nn.Module]]:
    out: List[tuple[str, nn.Module]] = []
    for name, child in module.named_modules():
        key = prefix if name == "" else f"{prefix}.{name}"
        out.append((key, child))
    return out


def _linear_sigma(linear: nn.Linear) -> float:
    w = linear.weight.detach().to(dtype=torch.float32)
    return float(torch.linalg.svdvals(w).max().item())


def _effective_adapter_alpha(adapter: nn.Module) -> float:
    alpha = getattr(adapter, "alpha", None)
    mode = str(getattr(adapter, "alpha_mode", "linear") or "linear").lower()
    if not torch.is_tensor(alpha):
        return 1.0
    val = float(alpha.detach().item())
    if mode == "tanh":
        return float(math.tanh(val))
    return float(val)


def _joint_names_for_slice(case: Mapping[str, Any], sl: slice) -> List[str]:
    rot_slice = case["rot_slice"]
    if not isinstance(rot_slice, slice):
        return []
    rot_start = int(rot_slice.start or 0)
    rot_stop = int(rot_slice.stop or rot_start)
    if int(sl.start or 0) < rot_start or int(sl.stop or 0) > rot_stop:
        return []
    if ((int(sl.start or 0) - rot_start) % 6) != 0 or ((int(sl.stop or 0) - rot_start) % 6) != 0:
        return []
    lo = (int(sl.start or 0) - rot_start) // 6
    hi = (int(sl.stop or 0) - rot_start) // 6
    names = list(case.get("bone_names", []))
    return [str(names[idx]) for idx in range(lo, min(hi, len(names)))]


def _output_slice_specs(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    trainer = case["trainer"]
    out_dim = int(getattr(trainer.model, "out_motion_dim", 0))
    loss_fn = getattr(trainer, "loss_fn", None)
    group_slices = dict(getattr(loss_fn, "group_slices", {}) or {})

    specs: List[Dict[str, Any]] = []
    rot_slice = group_slices.get("BoneRotations6D")
    if isinstance(rot_slice, slice):
        specs.append(
            {
                "key": "rot",
                "label": "BoneRotations6D",
                "slice": rot_slice,
                "width": int(rot_slice.stop - rot_slice.start),
                "available": True,
            }
        )
    else:
        specs.append({"key": "rot", "label": "BoneRotations6D", "slice": None, "width": 0, "available": False})

    rootvel_slice = group_slices.get("RootVelocity")
    if isinstance(rootvel_slice, slice):
        specs.append(
            {
                "key": "root_vel",
                "label": "RootVelocity",
                "slice": rootvel_slice,
                "width": int(rootvel_slice.stop - rootvel_slice.start),
                "available": True,
            }
        )
    else:
        specs.append({"key": "root_vel", "label": "RootVelocity", "slice": None, "width": 0, "available": False})

    angvel_slice = group_slices.get("BoneAngularVelocities")
    specs.append(
        {
            "key": "angvel",
            "label": "BoneAngularVelocities",
            "slice": angvel_slice if isinstance(angvel_slice, slice) else None,
            "width": int(angvel_slice.stop - angvel_slice.start) if isinstance(angvel_slice, slice) else 0,
            "available": isinstance(angvel_slice, slice),
        }
    )

    contacts_slice = group_slices.get("Contacts")
    specs.append(
        {
            "key": "contacts",
            "label": "Contacts",
            "slice": contacts_slice if isinstance(contacts_slice, slice) else None,
            "width": int(contacts_slice.stop - contacts_slice.start) if isinstance(contacts_slice, slice) else 0,
            "available": isinstance(contacts_slice, slice),
        }
    )

    rootpos_slice = group_slices.get("RootPosition")
    specs.append(
        {
            "key": "root_pos",
            "label": "RootPosition",
            "slice": rootpos_slice if isinstance(rootpos_slice, slice) else None,
            "width": int(rootpos_slice.stop - rootpos_slice.start) if isinstance(rootpos_slice, slice) else 0,
            "available": isinstance(rootpos_slice, slice),
        }
    )

    occupied = np.zeros(max(out_dim, 0), dtype=np.bool_)
    for spec in specs:
        sl = spec.get("slice")
        if isinstance(sl, slice):
            st = max(0, int(sl.start or 0))
            ed = min(out_dim, int(sl.stop or st))
            occupied[st:ed] = True
    other_ranges: List[slice] = []
    start = None
    for idx in range(out_dim):
        if not occupied[idx] and start is None:
            start = idx
        if occupied[idx] and start is not None:
            other_ranges.append(slice(start, idx))
            start = None
    if start is not None:
        other_ranges.append(slice(start, out_dim))
    if len(other_ranges) == 1:
        other_slice = other_ranges[0]
    else:
        other_slice = None
    specs.append(
        {
            "key": "other",
            "label": "Other",
            "slice": other_slice,
            "width": int(other_slice.stop - other_slice.start) if isinstance(other_slice, slice) else 0,
            "available": isinstance(other_slice, slice),
        }
    )
    return specs


def _readout_from_h(case: Mapping[str, Any], h: torch.Tensor) -> torch.Tensor:
    model = case["trainer"].model
    out = model.motion_head(h)
    adapters = getattr(model, "_bone_adapters", None)
    adapter_slices = getattr(model, "_bone_adapter_slices", None)
    if adapters and adapter_slices:
        add_terms: List[torch.Tensor] = []
        out_dim = int(out.shape[-1])
        for sl, adapter in zip(adapter_slices, adapters):
            if not isinstance(sl, slice):
                continue
            part = adapter(h)
            st = int(sl.start or 0)
            ed = int(sl.stop or st)
            pad = (st, max(0, out_dim - ed))
            add_terms.append(F.pad(part, pad))
        if add_terms:
            out = out + torch.stack(add_terms, dim=0).sum(dim=0)
    return out


def _head_static_audit(case: Mapping[str, Any]) -> Dict[str, Any]:
    model = case["trainer"].model
    motion_head = model.motion_head
    head_layers: List[Dict[str, Any]] = []
    head_upper = 1.0
    for name, mod in _flatten_named_modules(motion_head, "motion_head"):
        if isinstance(mod, nn.Linear):
            sigma = _linear_sigma(mod)
            head_upper *= sigma
            head_layers.append(
                {
                    "module": name,
                    "type": type(mod).__name__,
                    "weight_shape": list(mod.weight.shape),
                    "sigma_max": sigma,
                }
            )

    adapter_rows: List[Dict[str, Any]] = []
    adapter_sum_bound = 0.0
    adapters = getattr(model, "_bone_adapters", None) or []
    adapter_slices = getattr(model, "_bone_adapter_slices", None) or []
    for idx, (sl, adapter) in enumerate(zip(adapter_slices, adapters)):
        layers: List[Dict[str, Any]] = []
        branch_upper = abs(_effective_adapter_alpha(adapter))
        for name, mod in _flatten_named_modules(adapter, f"bone_adapter[{idx}]"):
            if isinstance(mod, nn.Linear):
                sigma = _linear_sigma(mod)
                branch_upper *= sigma
                layers.append(
                    {
                        "module": name,
                        "type": type(mod).__name__,
                        "weight_shape": list(mod.weight.shape),
                        "sigma_max": sigma,
                    }
                )
        adapter_sum_bound += branch_upper
        adapter_rows.append(
            {
                "index": int(idx),
                "slice": [int(sl.start or 0), int(sl.stop or 0)] if isinstance(sl, slice) else None,
                "joint_names": _joint_names_for_slice(case, sl) if isinstance(sl, slice) else [],
                "alpha_effective": _effective_adapter_alpha(adapter),
                "layers": layers,
                "upper_bound": float(branch_upper),
            }
        )

    return {
        "motion_head_layers": head_layers,
        "motion_head_upper_bound": float(head_upper),
        "bone_adapters": adapter_rows,
        "bone_adapter_upper_bound_sum": float(adapter_sum_bound),
        "full_head_conservative_upper_bound": float(head_upper + adapter_sum_bound),
    }


def _compute_motion_only_hidden_delta(
    *,
    case: Mapping[str, Any],
    clean_record: Mapping[str, Any],
    freerun_record: Mapping[str, Any],
) -> Dict[str, Any]:
    model = case["trainer"].model
    if model is None:
        raise RuntimeError("trainer.model missing")

    base_motion = clean_record.get("motion_in")
    free_motion = freerun_record.get("motion_in")
    if not torch.is_tensor(base_motion) or not torch.is_tensor(free_motion):
        raise RuntimeError("motion_in tensors missing from capture")
    if tuple(base_motion.shape) != tuple(free_motion.shape):
        raise RuntimeError(
            f"motion_in shape mismatch: clean={tuple(base_motion.shape)} freerun={tuple(free_motion.shape)}"
        )
    delta_motion = (free_motion - base_motion).detach()
    motion = (base_motion + delta_motion).detach()
    pose_history = clean_record.get("pose_history_in")
    if torch.is_tensor(pose_history):
        pose_history = pose_history.detach().clone()

    contacts = _resolve_counterfactual_contacts(
        case=case,
        motion=motion,
        pose_history=pose_history if torch.is_tensor(pose_history) else None,
        fallback_contacts=clean_record.get("contacts_in"),
    )
    angvel = _resolve_counterfactual_angvel(
        case=case,
        motion=motion,
        fallback_angvel=clean_record.get("angvel_in"),
    )

    with torch.no_grad():
        ret = model(
            motion,
            clean_record.get("cond_input"),
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_history,
            plan_z=clean_record.get("plan_z_in"),
            phase_event_age=clean_record.get("phase_event_age_in"),
            meas_logits_prev=clean_record.get("meas_logits_prev_in"),
            time_index=clean_record.get("time_index_in"),
            rollout_step=clean_record.get("rollout_step_in"),
        )
    if not isinstance(ret, Mapping):
        raise RuntimeError("counterfactual forward must return dict")
    h_clean = clean_record.get("h_final")
    h_pert = ret.get("h_final")
    if not torch.is_tensor(h_clean) or not torch.is_tensor(h_pert):
        raise RuntimeError("motion-only counterfactual missing h_final tensors")
    delta_h = (h_pert - h_clean).detach()
    return {
        "delta_motion": delta_motion,
        "delta_motion_norm": _tensor_l2(delta_motion),
        "delta_h_final": delta_h,
        "delta_h_final_norm": _tensor_l2(delta_h),
    }


def _slice_metrics(
    *,
    clean_out: torch.Tensor,
    pert_out: torch.Tensor,
    delta_h_norm: Optional[float],
    slice_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in slice_specs:
        key = str(spec["key"])
        sl = spec.get("slice")
        if not isinstance(sl, slice):
            out[key] = {
                "available": False,
                "width": int(spec.get("width", 0) or 0),
                "response_out": None,
                "gain_out": None,
            }
            continue
        resp = _diff_l2(pert_out, clean_out, sl)
        out[key] = {
            "available": True,
            "width": int(spec.get("width", 0) or 0),
            "response_out": resp,
            "gain_out": _gain(resp, delta_h_norm),
        }
    return out


def _local_dir_metrics(
    *,
    case: Mapping[str, Any],
    clean_record: Mapping[str, Any],
    delta_h: torch.Tensor,
    slice_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    delta_h_norm = _tensor_l2(delta_h)
    if delta_h_norm is None or delta_h_norm <= 1e-12:
        return {
            "overall": None,
            "by_slice": {
                str(spec["key"]): {
                    "available": bool(spec.get("available", False)),
                    "width": int(spec.get("width", 0) or 0),
                    "gain_out": None,
                }
                for spec in slice_specs
            },
        }

    h_clean = clean_record.get("h_final")
    if not torch.is_tensor(h_clean):
        raise RuntimeError("clean record missing h_final")
    unit = (delta_h / float(delta_h_norm)).detach()
    h_base = h_clean.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        _, tangent = torch.autograd.functional.jvp(
            lambda x: _readout_from_h(case, x),
            h_base,
            unit,
            create_graph=False,
            strict=False,
        )

    by_slice: Dict[str, Dict[str, Any]] = {}
    for spec in slice_specs:
        key = str(spec["key"])
        sl = spec.get("slice")
        if not isinstance(sl, slice):
            by_slice[key] = {
                "available": False,
                "width": int(spec.get("width", 0) or 0),
                "gain_out": None,
            }
            continue
        by_slice[key] = {
            "available": True,
            "width": int(spec.get("width", 0) or 0),
            "gain_out": _tensor_l2(tangent[..., sl]),
        }

    return {
        "overall": _tensor_l2(tangent),
        "by_slice": by_slice,
    }


def _head_side_response(
    *,
    case: Mapping[str, Any],
    clean_record: Mapping[str, Any],
    delta_h: torch.Tensor,
    gt_raw: torch.Tensor,
    slice_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    h_clean = clean_record.get("h_final")
    clean_out = clean_record.get("out")
    clean_y = clean_record.get("y_inc_raw")
    if not torch.is_tensor(h_clean) or not torch.is_tensor(clean_out) or not torch.is_tensor(clean_y):
        raise RuntimeError("clean record missing head-side tensors")

    delta_h_norm = _tensor_l2(delta_h)
    h_injected = h_clean.detach() + delta_h.detach()
    with torch.no_grad():
        out = _readout_from_h(case, h_injected)
    y_inc_raw = _resolve_y_inc_raw(
        case=case,
        y_prev_raw=clean_record.get("y_prev_raw"),
        out=out,
        ret={},
    )
    geo_local_deg = _geo_local_deg_from_raw(case=case, pred_raw=y_inc_raw, gt_raw=gt_raw)
    clean_geo = _safe_float(clean_record.get("geo_local_deg"))
    geo_delta = float(geo_local_deg - clean_geo) if math.isfinite(clean_geo) else float("nan")

    resp_out = _diff_l2(out, clean_out)
    resp_y = _diff_l2(y_inc_raw, clean_y)
    slices = _slice_metrics(
        clean_out=clean_out,
        pert_out=out,
        delta_h_norm=delta_h_norm,
        slice_specs=slice_specs,
    )
    local_dir = _local_dir_metrics(
        case=case,
        clean_record=clean_record,
        delta_h=delta_h,
        slice_specs=slice_specs,
    )
    return {
        "delta_h_final_norm": delta_h_norm,
        "response_out": resp_out,
        "response_y_inc_raw": resp_y,
        "gain_out": _gain(resp_out, delta_h_norm),
        "gain_y_inc_raw": _gain(resp_y, delta_h_norm),
        "geo_local_deg_clean": clean_geo,
        "geo_local_deg_perturbed": geo_local_deg,
        "geo_local_deg_delta": geo_delta,
        "geo_local_deg_abs_delta": abs(geo_delta) if math.isfinite(geo_delta) else float("nan"),
        "slice_response": slices,
        "local_dir_gain_out": local_dir["overall"],
        "local_dir_slice_gain_out": local_dir["by_slice"],
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


def _aggregate_case(rows: Sequence[Mapping[str, Any]], prefix: str, slice_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "delta_h_final_norm": _summary(r.get("delta_h_final_norm") for r in rows),
        "response_out": _summary(r.get(f"{prefix}_response_out") for r in rows),
        "response_y_inc_raw": _summary(r.get(f"{prefix}_response_y_inc_raw") for r in rows),
        "gain_out": _summary(r.get(f"{prefix}_gain_out") for r in rows),
        "gain_y_inc_raw": _summary(r.get(f"{prefix}_gain_y_inc_raw") for r in rows),
        "local_dir_gain_out": _summary(r.get(f"{prefix}_local_dir_gain_out") for r in rows),
        "geo_local_deg_clean": _summary(r.get(f"{prefix}_geo_local_deg_clean") for r in rows),
        "geo_local_deg_perturbed": _summary(r.get(f"{prefix}_geo_local_deg_perturbed") for r in rows),
        "geo_local_deg_abs_delta": _summary(r.get(f"{prefix}_geo_local_deg_abs_delta") for r in rows),
        "freerun_geo_local_deg": _summary(r.get(f"{prefix}_freerun_geo_local_deg") for r in rows),
        "slices": {},
    }
    for spec in slice_specs:
        key = str(spec["key"])
        out["slices"][key] = {
            "available": bool(spec.get("available", False)),
            "width": int(spec.get("width", 0) or 0),
            "gain_out": _summary(r.get(f"{prefix}_slice_gain_out_{key}") for r in rows),
            "local_dir_gain_out": _summary(r.get(f"{prefix}_local_dir_slice_gain_out_{key}") for r in rows),
        }
    return out


def _aggregate_compare(
    rows: Sequence[Mapping[str, Any]],
    slice_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    tail = _aggregate_case(rows, "tail", slice_specs)
    base = _aggregate_case(rows, "baseline", slice_specs)
    compare: Dict[str, Any] = {
        "rows": int(len(rows)),
        "delta_h_final_norm": _summary(r.get("delta_h_final_norm") for r in rows),
        "delta_motion_norm_source": _summary(r.get("delta_motion_norm_source") for r in rows),
        "tail": tail,
        "baseline": base,
        "ratio_of_means": {
            "gain_out": _ratio(tail["gain_out"]["mean"], base["gain_out"]["mean"]),
            "gain_y_inc_raw": _ratio(tail["gain_y_inc_raw"]["mean"], base["gain_y_inc_raw"]["mean"]),
            "local_dir_gain_out": _ratio(tail["local_dir_gain_out"]["mean"], base["local_dir_gain_out"]["mean"]),
            "response_out": _ratio(tail["response_out"]["mean"], base["response_out"]["mean"]),
            "response_y_inc_raw": _ratio(tail["response_y_inc_raw"]["mean"], base["response_y_inc_raw"]["mean"]),
            "geo_local_abs_delta": _ratio(
                tail["geo_local_deg_abs_delta"]["mean"], base["geo_local_deg_abs_delta"]["mean"]
            ),
        },
        "slices": {},
    }
    for spec in slice_specs:
        key = str(spec["key"])
        compare["slices"][key] = {
            "available": bool(spec.get("available", False)),
            "width": int(spec.get("width", 0) or 0),
            "gain_out_ratio_of_means": _ratio(
                tail["slices"][key]["gain_out"]["mean"], base["slices"][key]["gain_out"]["mean"]
            ),
            "local_dir_gain_out_ratio_of_means": _ratio(
                tail["slices"][key]["local_dir_gain_out"]["mean"],
                base["slices"][key]["local_dir_gain_out"]["mean"],
            ),
        }
    return compare


def _selected_window_rows(per_step_rows: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    bucket_map = {name: (lo, hi) for name, lo, hi in DEPTH_BUCKETS + SIC_BUCKETS}
    for bucket_name in selected:
        lo, hi = bucket_map[bucket_name]
        if bucket_name.startswith("d"):
            rows = _rows_for_depth(len(per_step_rows), lo, hi)
            kind = "depth"
        else:
            rows = _rows_for_sic(per_step_rows, lo, hi)
            kind = "step_in_cycle"
        tables.append({"bucket": bucket_name, "kind": kind, "row_indices": rows})
    return tables


def _build_rows(
    *,
    tail_case: Mapping[str, Any],
    base_case: Mapping[str, Any],
    tail_teacher: Mapping[str, Any],
    tail_free: Mapping[str, Any],
    base_teacher: Mapping[str, Any],
    base_free: Mapping[str, Any],
    slice_specs: Sequence[Mapping[str, Any]],
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

        observed = _compute_motion_only_hidden_delta(case=tail_case, clean_record=t_clean, freerun_record=t_free)
        delta_h = observed["delta_h_final"]
        t_resp = _head_side_response(
            case=tail_case,
            clean_record=t_clean,
            delta_h=delta_h,
            gt_raw=t_clean["gt_raw"],
            slice_specs=slice_specs,
        )
        b_resp = _head_side_response(
            case=base_case,
            clean_record=b_clean,
            delta_h=delta_h,
            gt_raw=b_clean["gt_raw"],
            slice_specs=slice_specs,
        )

        row: Dict[str, Any] = {
            "step": int(idx),
            "cycle": int(t_clean.get("cycle", 0) or 0),
            "step_in_cycle": int(t_clean.get("step_in_cycle", -1) or -1),
            "wrap_boundary_step": bool(t_clean.get("wrap_boundary_step", False)),
            "delta_motion_norm_source": observed["delta_motion_norm"],
            "delta_h_final_norm": observed["delta_h_final_norm"],
            "tail_freerun_geo_local_deg": _safe_float(t_free.get("freerun_geo_local_deg")),
            "baseline_freerun_geo_local_deg": _safe_float(b_free.get("freerun_geo_local_deg")),
        }
        for prefix, payload in (("tail", t_resp), ("baseline", b_resp)):
            row[f"{prefix}_response_out"] = payload["response_out"]
            row[f"{prefix}_response_y_inc_raw"] = payload["response_y_inc_raw"]
            row[f"{prefix}_gain_out"] = payload["gain_out"]
            row[f"{prefix}_gain_y_inc_raw"] = payload["gain_y_inc_raw"]
            row[f"{prefix}_geo_local_deg_clean"] = payload["geo_local_deg_clean"]
            row[f"{prefix}_geo_local_deg_perturbed"] = payload["geo_local_deg_perturbed"]
            row[f"{prefix}_geo_local_deg_abs_delta"] = payload["geo_local_deg_abs_delta"]
            row[f"{prefix}_local_dir_gain_out"] = payload["local_dir_gain_out"]
            for spec in slice_specs:
                key = str(spec["key"])
                row[f"{prefix}_slice_gain_out_{key}"] = (
                    (payload["slice_response"].get(key) or {}).get("gain_out")
                )
                row[f"{prefix}_local_dir_slice_gain_out_{key}"] = (
                    (payload["local_dir_slice_gain_out"].get(key) or {}).get("gain_out")
                )
        rows.append(row)
    return rows


def _bucket_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    slice_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected = [rows[int(i)] for i in row_indices if 0 <= int(i) < len(rows)]
    return _aggregate_compare(selected, slice_specs)


def _window_alignment_table(
    *,
    rows: Sequence[Mapping[str, Any]],
    slice_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _selected_window_rows(rows, ("d10_20", "d21_43", "sic11_21", "sic22_43")):
        bucket = _bucket_report(rows=rows, row_indices=item["row_indices"], slice_specs=slice_specs)
        out.append(
            {
                "bucket": item["bucket"],
                "kind": item["kind"],
                "rows": int(bucket["rows"]),
                "tail_gain_out_mean": bucket["tail"]["gain_out"]["mean"],
                "baseline_gain_out_mean": bucket["baseline"]["gain_out"]["mean"],
                "gain_out_ratio": bucket["ratio_of_means"]["gain_out"],
                "tail_gain_y_mean": bucket["tail"]["gain_y_inc_raw"]["mean"],
                "baseline_gain_y_mean": bucket["baseline"]["gain_y_inc_raw"]["mean"],
                "gain_y_ratio": bucket["ratio_of_means"]["gain_y_inc_raw"],
                "tail_local_dir_gain_out_mean": bucket["tail"]["local_dir_gain_out"]["mean"],
                "baseline_local_dir_gain_out_mean": bucket["baseline"]["local_dir_gain_out"]["mean"],
                "local_dir_gain_out_ratio": bucket["ratio_of_means"]["local_dir_gain_out"],
                "tail_freerun_geo_local_deg_mean": bucket["tail"]["freerun_geo_local_deg"]["mean"],
                "baseline_freerun_geo_local_deg_mean": bucket["baseline"]["freerun_geo_local_deg"]["mean"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher-conditioned motion-head gain audit with matched hidden perturbations.")
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    tail_case = _load_case(
        case_name="tailk7_current_control",
        ckpt_path=args.tail_ckpt,
        eval_json_path=args.tail_eval,
        teacher_path=args.teacher,
        device_pref=args.device,
    )
    base_case = _load_case(
        case_name="baseline_replace",
        ckpt_path=args.baseline_ckpt,
        eval_json_path=args.baseline_eval,
        teacher_path=args.teacher,
        device_pref=args.device,
    )
    slice_specs = _output_slice_specs(tail_case)

    tail_teacher = _capture_run(tail_case, rounds=int(args.rounds), teacher_conditioned=True)
    tail_free = _capture_run(tail_case, rounds=int(args.rounds), teacher_conditioned=False)
    base_teacher = _capture_run(base_case, rounds=int(args.rounds), teacher_conditioned=True)
    base_free = _capture_run(base_case, rounds=int(args.rounds), teacher_conditioned=False)

    rows = _build_rows(
        tail_case=tail_case,
        base_case=base_case,
        tail_teacher=tail_teacher,
        tail_free=tail_free,
        base_teacher=base_teacher,
        base_free=base_free,
        slice_specs=slice_specs,
    )

    depth_tables = {
        name: _bucket_report(
            rows=rows,
            row_indices=_rows_for_depth(len(rows), lo, hi),
            slice_specs=slice_specs,
        )
        for name, lo, hi in DEPTH_BUCKETS
    }
    sic_tables = {
        name: _bucket_report(
            rows=rows,
            row_indices=_rows_for_sic(rows, lo, hi),
            slice_specs=slice_specs,
        )
        for name, lo, hi in SIC_BUCKETS
    }

    tail_static = _head_static_audit(tail_case)
    base_static = _head_static_audit(base_case)
    overall = _aggregate_compare(rows, slice_specs)

    payload = {
        "analysis": "motion_head_gain_audit",
        "script_path": str(Path(__file__).resolve()),
        "summary_path": str(args.out.resolve()),
        "teacher_batch": str(args.teacher.resolve()),
        "parameters": {
            "rounds": int(args.rounds),
            "device": str(args.device),
        },
        "code_facts": {
            "motion_head_definition": (
                "EventMotionModel.__init__ sets motion_head = build_mlp(hidden_dim, hidden_dim, num_layers=1, "
                "activation=ReLU, dropout=dropout, final_dim=out_motion_dim)."
            ),
            "h_final_to_out_path": (
                "EventMotionModel.forward computes h_final = coupling_norm(...), then out = motion_head(h_final), "
                "then adds optional _bone_adapters on configured output slices before returning result['out'] and result['h_final']."
            ),
            "output_layout": {
                "tail": dict(tail_case["trainer"]._y_layout),
                "baseline": dict(base_case["trainer"]._y_layout),
            },
            "output_slice_availability": [
                {
                    "key": str(spec["key"]),
                    "label": str(spec["label"]),
                    "available": bool(spec["available"]),
                    "width": int(spec["width"]),
                }
                for spec in slice_specs
            ],
            "y_inc_raw_path": (
                "validate.run_freerun_cycles uses trainer._compose_delta_to_raw(y_raw_prev, out, ...) and then "
                "y_used_raw = y_inc_raw when lambda fusion is inactive/effective identity for these runs."
            ),
            "head_side_injection": (
                "Observed motion-only delta_h_final is measured on tail teacher-conditioned clean state, then the exact same "
                "delta_h_final is added to clean h_final for both tail and baseline before applying the deployed head readout."
            ),
            "local_jacobian_method": (
                "Directional JVP on the exact deployed head-side readout out(h_final), not full input-side rollout. "
                "This isolates the local Jacobian of h_final -> out including bone adapters."
            ),
        },
        "cases": {
            "tailk7_current_control": {
                "ckpt_path": str(args.tail_ckpt.resolve()),
                "eval_json_path": str(args.tail_eval.resolve()),
                "runtime_overrides": dict(tail_case["runtime_overrides"]),
                "static_head": tail_static,
            },
            "baseline_replace": {
                "ckpt_path": str(args.baseline_ckpt.resolve()),
                "eval_json_path": str(args.baseline_eval.resolve()),
                "runtime_overrides": dict(base_case["runtime_overrides"]),
                "static_head": base_static,
            },
        },
        "static_compare": {
            "motion_head_upper_bound_ratio": _ratio(
                tail_static["motion_head_upper_bound"], base_static["motion_head_upper_bound"]
            ),
            "full_head_conservative_upper_bound_ratio": _ratio(
                tail_static["full_head_conservative_upper_bound"],
                base_static["full_head_conservative_upper_bound"],
            ),
            "bone_adapter_upper_bound_sum_ratio": _ratio(
                tail_static["bone_adapter_upper_bound_sum"],
                base_static["bone_adapter_upper_bound_sum"],
            ),
        },
        "overall": overall,
        "depth_buckets": depth_tables,
        "step_in_cycle_buckets": sic_tables,
        "window_alignment": _window_alignment_table(rows=rows, slice_specs=slice_specs),
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(
        {
            "summary_path": str(args.out.resolve()),
            "overall_gain_out_ratio": payload["overall"]["ratio_of_means"]["gain_out"],
            "overall_gain_y_ratio": payload["overall"]["ratio_of_means"]["gain_y_inc_raw"],
            "overall_local_dir_gain_out_ratio": payload["overall"]["ratio_of_means"]["local_dir_gain_out"],
        },
        indent=2,
        allow_nan=True,
    ))


if __name__ == "__main__":
    main()
