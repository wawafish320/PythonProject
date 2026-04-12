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
    _resolve_counterfactual_angvel,
    _resolve_counterfactual_contacts,
)


RUN_DATE = "20260405"
DEFAULT_OUT = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_rot_readout_decomposition_{RUN_DATE}" / "summary.json"
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


def _safe_sub(lhs: Any, rhs: Any) -> float:
    lv = _safe_float(lhs)
    rv = _safe_float(rhs)
    if (not math.isfinite(lv)) or (not math.isfinite(rv)):
        return float("nan")
    return float(lv - rv)


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


def _slice_bounds(sl: Optional[slice]) -> Optional[tuple[int, int]]:
    if not isinstance(sl, slice):
        return None
    st = int(sl.start or 0)
    ed = int(sl.stop or st)
    return (st, ed)


def _slice_list(sl: Optional[slice]) -> Optional[List[int]]:
    bounds = _slice_bounds(sl)
    if bounds is None:
        return None
    return [int(bounds[0]), int(bounds[1])]


def _joint_names_for_slice(case: Mapping[str, Any], sl: slice) -> List[str]:
    rot_slice = case["rot_slice"]
    rot_bounds = _slice_bounds(rot_slice)
    cur_bounds = _slice_bounds(sl)
    if rot_bounds is None or cur_bounds is None:
        return []
    rot_start, rot_stop = rot_bounds
    st, ed = cur_bounds
    if st < rot_start or ed > rot_stop:
        return []
    if ((st - rot_start) % 6) != 0 or ((ed - rot_start) % 6) != 0:
        return []
    lo = (st - rot_start) // 6
    hi = (ed - rot_start) // 6
    names = list(case.get("bone_names", []))
    return [str(names[idx]) for idx in range(lo, min(hi, len(names)))]


def _output_slices(case: Mapping[str, Any]) -> Dict[str, Optional[List[int]]]:
    trainer = case["trainer"]
    loss_fn = getattr(trainer, "loss_fn", None)
    group_slices = dict(getattr(loss_fn, "group_slices", {}) or {})
    return {
        "rot": _slice_list(case.get("rot_slice")),
        "root_vel": _slice_list(group_slices.get("RootVelocity")),
        "angvel": _slice_list(group_slices.get("BoneAngularVelocities")),
        "contacts": _slice_list(group_slices.get("Contacts")),
        "root_pos": _slice_list(group_slices.get("RootPosition")),
    }


def _adapter_meta(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = case["trainer"].model
    adapters = list(getattr(model, "_bone_adapters", None) or [])
    adapter_slices = list(getattr(model, "_bone_adapter_slices", None) or [])
    adapter_names = list(getattr(model, "_bone_adapter_names", None) or [])
    rot_bounds = _slice_bounds(case["rot_slice"])
    out: List[Dict[str, Any]] = []
    for idx, (sl, _adapter) in enumerate(zip(adapter_slices, adapters)):
        cur_bounds = _slice_bounds(sl)
        if cur_bounds is None:
            continue
        st, ed = cur_bounds
        name = str(adapter_names[idx]) if idx < len(adapter_names) else f"adapter_{idx}"
        joint_names = _joint_names_for_slice(case, sl)
        meta = {
            "index": int(idx),
            "name": name,
            "key": f"adapter_{name}",
            "slice": [int(st), int(ed)],
            "joint_names": joint_names,
            "width": int(ed - st),
            "within_rot": False,
            "rot_slice": None,
        }
        if rot_bounds is not None:
            rot_start, rot_stop = rot_bounds
            if st >= rot_start and ed <= rot_stop:
                meta["within_rot"] = True
                meta["rot_slice"] = [int(st - rot_start), int(ed - rot_start)]
        out.append(meta)
    return out


def _part_specs(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "key": "total",
            "label": "rot_total",
            "kind": "aggregate",
            "joint_names": [],
            "slice": _slice_list(case["rot_slice"]),
        },
        {
            "key": "main",
            "label": "rot_main",
            "kind": "aggregate",
            "joint_names": [],
            "slice": _slice_list(case["rot_slice"]),
        },
        {
            "key": "adapter_total",
            "label": "rot_adapter_total",
            "kind": "aggregate",
            "joint_names": [],
            "slice": _slice_list(case["rot_slice"]),
        },
    ]
    for meta in _adapter_meta(case):
        if not bool(meta.get("within_rot", False)):
            continue
        specs.append(
            {
                "key": str(meta["key"]),
                "label": f"rot_adapter[{meta['name']}]",
                "kind": "adapter",
                "joint_names": list(meta.get("joint_names", [])),
                "slice": list(meta.get("slice") or []),
            }
        )
    return specs


def _assert_compatible_specs(lhs: Sequence[Mapping[str, Any]], rhs: Sequence[Mapping[str, Any]]) -> None:
    left = [
        (
            str(item.get("key")),
            str(item.get("kind")),
            tuple(item.get("joint_names", []) or []),
            tuple(item.get("slice", []) or []),
        )
        for item in lhs
    ]
    right = [
        (
            str(item.get("key")),
            str(item.get("kind")),
            tuple(item.get("joint_names", []) or []),
            tuple(item.get("slice", []) or []),
        )
        for item in rhs
    ]
    if left != right:
        raise RuntimeError(f"tail/base part specs mismatch:\nleft={left}\nright={right}")


def _static_decomposition(case: Mapping[str, Any]) -> Dict[str, Any]:
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

    meta_rows = _adapter_meta(case)
    adapters = list(getattr(model, "_bone_adapters", None) or [])
    adapter_rows: List[Dict[str, Any]] = []
    adapter_sum_bound = 0.0
    for meta, adapter in zip(meta_rows, adapters):
        layers: List[Dict[str, Any]] = []
        branch_upper = abs(_effective_adapter_alpha(adapter))
        for name, mod in _flatten_named_modules(adapter, f"bone_adapter[{meta['index']}]"):
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
                **meta,
                "alpha_effective": _effective_adapter_alpha(adapter),
                "upper_bound": float(branch_upper),
                "layers": layers,
            }
        )

    return {
        "output_slices": _output_slices(case),
        "motion_head_layers": head_layers,
        "motion_head_upper_bound": float(head_upper),
        "bone_adapters": adapter_rows,
        "bone_adapter_upper_bound_sum": float(adapter_sum_bound),
        "full_head_conservative_upper_bound": float(head_upper + adapter_sum_bound),
    }


def _decompose_rot_outputs(case: Mapping[str, Any], h: torch.Tensor) -> Dict[str, Any]:
    model = case["trainer"].model
    rot_bounds = _slice_bounds(case["rot_slice"])
    if rot_bounds is None:
        raise RuntimeError("rot slice missing")
    rot_start, rot_stop = rot_bounds

    out_main_full = model.motion_head(h)
    rot_main = out_main_full[..., rot_start:rot_stop]
    rot_adapter_total = torch.zeros_like(rot_main)

    adapter_parts: List[Dict[str, Any]] = []
    adapters = list(getattr(model, "_bone_adapters", None) or [])
    adapter_slices = list(getattr(model, "_bone_adapter_slices", None) or [])
    adapter_meta = _adapter_meta(case)
    for meta, sl, adapter in zip(adapter_meta, adapter_slices, adapters):
        cur_bounds = _slice_bounds(sl)
        if cur_bounds is None:
            continue
        st, ed = cur_bounds
        local = adapter(h)
        rot_full = rot_main.new_zeros(rot_main.shape)
        ov_st = max(st, rot_start)
        ov_ed = min(ed, rot_stop)
        if ov_ed > ov_st:
            src_lo = ov_st - st
            src_hi = ov_ed - st
            dst_lo = ov_st - rot_start
            dst_hi = ov_ed - rot_start
            rot_full[..., dst_lo:dst_hi] = local[..., src_lo:src_hi]
        rot_adapter_total = rot_adapter_total + rot_full
        adapter_parts.append({**meta, "rot_tensor": rot_full})

    return {
        "main": rot_main,
        "adapter_total": rot_adapter_total,
        "total": rot_main + rot_adapter_total,
        "adapter_parts": adapter_parts,
    }


def _rot_part_tensor_map(
    case: Mapping[str, Any],
    h: torch.Tensor,
    part_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, torch.Tensor]:
    dec = _decompose_rot_outputs(case, h)
    out: Dict[str, torch.Tensor] = {
        "total": dec["total"],
        "main": dec["main"],
        "adapter_total": dec["adapter_total"],
    }
    for item in dec["adapter_parts"]:
        out[str(item["key"])] = item["rot_tensor"]
    missing = [str(spec["key"]) for spec in part_specs if str(spec["key"]) not in out]
    if missing:
        raise RuntimeError(f"missing part tensors for keys: {missing}")
    return out


def _rot_tuple_from_h(
    case: Mapping[str, Any],
    h: torch.Tensor,
    part_specs: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, ...]:
    part_map = _rot_part_tensor_map(case, h, part_specs)
    return tuple(part_map[str(spec["key"])] for spec in part_specs)


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
        "delta_motion_norm": _tensor_l2(delta_motion),
        "delta_h_final": delta_h,
        "delta_h_final_norm": _tensor_l2(delta_h),
    }


def _local_directional_metrics(
    *,
    case: Mapping[str, Any],
    clean_h: torch.Tensor,
    delta_h: torch.Tensor,
    part_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Optional[float]]:
    delta_h_norm = _tensor_l2(delta_h)
    if delta_h_norm is None or delta_h_norm <= 1e-12:
        return {str(spec["key"]): None for spec in part_specs}

    unit = (delta_h / float(delta_h_norm)).detach()
    h_base = clean_h.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        _, tangents = torch.autograd.functional.jvp(
            lambda x: _rot_tuple_from_h(case, x, part_specs),
            h_base,
            unit,
            create_graph=False,
            strict=False,
        )
    if not isinstance(tangents, tuple):
        tangents = (tangents,)
    return {
        str(spec["key"]): _tensor_l2(tangent)
        for spec, tangent in zip(part_specs, tangents)
    }


def _rot_decomposition_response(
    *,
    case: Mapping[str, Any],
    clean_record: Mapping[str, Any],
    delta_h: torch.Tensor,
    part_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    h_clean = clean_record.get("h_final")
    if not torch.is_tensor(h_clean):
        raise RuntimeError("clean record missing h_final")

    delta_h_norm = _tensor_l2(delta_h)
    h_injected = h_clean.detach() + delta_h.detach()
    with torch.no_grad():
        clean_parts = _rot_part_tensor_map(case, h_clean.detach(), part_specs)
        injected_parts = _rot_part_tensor_map(case, h_injected, part_specs)

    responses: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in part_specs:
        key = str(spec["key"])
        resp = _tensor_l2(injected_parts[key] - clean_parts[key])
        responses[key] = {
            "response_rot": resp,
            "gain_rot": _gain(resp, delta_h_norm),
        }

    local_dir = _local_directional_metrics(
        case=case,
        clean_h=h_clean,
        delta_h=delta_h,
        part_specs=part_specs,
    )
    return {
        "delta_h_final_norm": delta_h_norm,
        "parts": responses,
        "local_dir": local_dir,
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


def _build_rows(
    *,
    tail_case: Mapping[str, Any],
    base_case: Mapping[str, Any],
    tail_teacher: Mapping[str, Any],
    tail_free: Mapping[str, Any],
    base_teacher: Mapping[str, Any],
    base_free: Mapping[str, Any],
    part_specs: Sequence[Mapping[str, Any]],
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
        t_resp = _rot_decomposition_response(
            case=tail_case,
            clean_record=t_clean,
            delta_h=delta_h,
            part_specs=part_specs,
        )
        b_resp = _rot_decomposition_response(
            case=base_case,
            clean_record=b_clean,
            delta_h=delta_h,
            part_specs=part_specs,
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
            for spec in part_specs:
                key = str(spec["key"])
                part_payload = payload["parts"][key]
                row[f"{prefix}_response_rot_{key}"] = part_payload["response_rot"]
                row[f"{prefix}_gain_rot_{key}"] = part_payload["gain_rot"]
                row[f"{prefix}_local_dir_rot_{key}"] = payload["local_dir"][key]
        rows.append(row)
    return rows


def _aggregate_case(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
    part_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "delta_h_final_norm": _summary(r.get("delta_h_final_norm") for r in rows),
        "freerun_geo_local_deg": _summary(r.get(f"{prefix}_freerun_geo_local_deg") for r in rows),
        "parts": {},
    }
    for spec in part_specs:
        key = str(spec["key"])
        out["parts"][key] = {
            "response_rot": _summary(r.get(f"{prefix}_response_rot_{key}") for r in rows),
            "gain_rot": _summary(r.get(f"{prefix}_gain_rot_{key}") for r in rows),
            "local_dir_rot": _summary(r.get(f"{prefix}_local_dir_rot_{key}") for r in rows),
        }
    return out


def _aggregate_compare(rows: Sequence[Mapping[str, Any]], part_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tail = _aggregate_case(rows, "tail", part_specs)
    base = _aggregate_case(rows, "baseline", part_specs)
    compare: Dict[str, Any] = {
        "rows": int(len(rows)),
        "delta_h_final_norm": _summary(r.get("delta_h_final_norm") for r in rows),
        "tail": tail,
        "baseline": base,
        "parts": {},
    }
    for spec in part_specs:
        key = str(spec["key"])
        compare["parts"][key] = {
            "meta": {
                "key": key,
                "label": str(spec.get("label", key)),
                "kind": str(spec.get("kind", "")),
                "joint_names": list(spec.get("joint_names", []) or []),
                "slice": list(spec.get("slice", []) or []),
            },
            "response_rot_ratio_of_means": _ratio(
                tail["parts"][key]["response_rot"]["mean"],
                base["parts"][key]["response_rot"]["mean"],
            ),
            "gain_rot_ratio_of_means": _ratio(
                tail["parts"][key]["gain_rot"]["mean"],
                base["parts"][key]["gain_rot"]["mean"],
            ),
            "local_dir_rot_ratio_of_means": _ratio(
                tail["parts"][key]["local_dir_rot"]["mean"],
                base["parts"][key]["local_dir_rot"]["mean"],
            ),
            "gain_rot_excess_mean": _safe_sub(
                tail["parts"][key]["gain_rot"]["mean"],
                base["parts"][key]["gain_rot"]["mean"],
            ),
            "local_dir_rot_excess_mean": _safe_sub(
                tail["parts"][key]["local_dir_rot"]["mean"],
                base["parts"][key]["local_dir_rot"]["mean"],
            ),
        }
    return compare


def _bucket_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    part_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected = [rows[int(i)] for i in row_indices if 0 <= int(i) < len(rows)]
    return _aggregate_compare(selected, part_specs)


def _part_overview(compare: Mapping[str, Any], part_specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tail = compare["tail"]
    base = compare["baseline"]
    for spec in part_specs:
        key = str(spec["key"])
        comp = compare["parts"][key]
        out.append(
            {
                "key": key,
                "label": str(spec.get("label", key)),
                "kind": str(spec.get("kind", "")),
                "joint_names": list(spec.get("joint_names", []) or []),
                "slice": list(spec.get("slice", []) or []),
                "tail_response_rot_mean": tail["parts"][key]["response_rot"]["mean"],
                "baseline_response_rot_mean": base["parts"][key]["response_rot"]["mean"],
                "response_rot_ratio_of_means": comp["response_rot_ratio_of_means"],
                "tail_gain_rot_mean": tail["parts"][key]["gain_rot"]["mean"],
                "baseline_gain_rot_mean": base["parts"][key]["gain_rot"]["mean"],
                "gain_rot_ratio_of_means": comp["gain_rot_ratio_of_means"],
                "gain_rot_excess_mean": comp["gain_rot_excess_mean"],
                "tail_local_dir_rot_mean": tail["parts"][key]["local_dir_rot"]["mean"],
                "baseline_local_dir_rot_mean": base["parts"][key]["local_dir_rot"]["mean"],
                "local_dir_rot_ratio_of_means": comp["local_dir_rot_ratio_of_means"],
                "local_dir_rot_excess_mean": comp["local_dir_rot_excess_mean"],
            }
        )
    return out


def _bucket_overview(
    bucket_map: Mapping[str, Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bucket, compare in bucket_map.items():
        row: Dict[str, Any] = {
            "bucket": str(bucket),
            "rows": int(compare["rows"]),
            "tail_freerun_geo_local_deg_mean": compare["tail"]["freerun_geo_local_deg"]["mean"],
            "baseline_freerun_geo_local_deg_mean": compare["baseline"]["freerun_geo_local_deg"]["mean"],
        }
        for key in keys:
            row[f"tail_gain_rot_{key}_mean"] = compare["tail"]["parts"][key]["gain_rot"]["mean"]
            row[f"baseline_gain_rot_{key}_mean"] = compare["baseline"]["parts"][key]["gain_rot"]["mean"]
            row[f"gain_rot_{key}_ratio"] = compare["parts"][key]["gain_rot_ratio_of_means"]
            row[f"tail_local_dir_rot_{key}_mean"] = compare["tail"]["parts"][key]["local_dir_rot"]["mean"]
            row[f"baseline_local_dir_rot_{key}_mean"] = compare["baseline"]["parts"][key]["local_dir_rot"]["mean"]
            row[f"local_dir_rot_{key}_ratio"] = compare["parts"][key]["local_dir_rot_ratio_of_means"]
        out.append(row)
    return out


def _window_alignment_table(
    rows: Sequence[Mapping[str, Any]],
    part_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    bucket_defs = {
        "d10_20": ("depth", _rows_for_depth(len(rows), 10, 20)),
        "d21_43": ("depth", _rows_for_depth(len(rows), 21, 43)),
        "sic11_21": ("step_in_cycle", _rows_for_sic(rows, 11, 21)),
        "sic22_43": ("step_in_cycle", _rows_for_sic(rows, 22, 43)),
    }
    out: List[Dict[str, Any]] = []
    for bucket, (kind, idxs) in bucket_defs.items():
        compare = _bucket_report(rows=rows, row_indices=idxs, part_specs=part_specs)
        out.append(
            {
                "bucket": bucket,
                "kind": kind,
                "rows": int(compare["rows"]),
                "tail_freerun_geo_local_deg_mean": compare["tail"]["freerun_geo_local_deg"]["mean"],
                "baseline_freerun_geo_local_deg_mean": compare["baseline"]["freerun_geo_local_deg"]["mean"],
                "gain_rot_total_ratio": compare["parts"]["total"]["gain_rot_ratio_of_means"],
                "gain_rot_main_ratio": compare["parts"]["main"]["gain_rot_ratio_of_means"],
                "gain_rot_adapter_ratio": compare["parts"]["adapter_total"]["gain_rot_ratio_of_means"],
                "local_dir_rot_total_ratio": compare["parts"]["total"]["local_dir_rot_ratio_of_means"],
                "local_dir_rot_main_ratio": compare["parts"]["main"]["local_dir_rot_ratio_of_means"],
                "local_dir_rot_adapter_ratio": compare["parts"]["adapter_total"]["local_dir_rot_ratio_of_means"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Matched-hidden rot readout main-vs-adapter decomposition.")
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

    tail_specs = _part_specs(tail_case)
    base_specs = _part_specs(base_case)
    _assert_compatible_specs(tail_specs, base_specs)
    part_specs = tail_specs

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
        part_specs=part_specs,
    )

    depth_tables = {
        name: _bucket_report(
            rows=rows,
            row_indices=_rows_for_depth(len(rows), lo, hi),
            part_specs=part_specs,
        )
        for name, lo, hi in DEPTH_BUCKETS
    }
    sic_tables = {
        name: _bucket_report(
            rows=rows,
            row_indices=_rows_for_sic(rows, lo, hi),
            part_specs=part_specs,
        )
        for name, lo, hi in SIC_BUCKETS
    }

    tail_static = _static_decomposition(tail_case)
    base_static = _static_decomposition(base_case)
    overall = _aggregate_compare(rows, part_specs)

    main_keys = ("total", "main", "adapter_total")
    adapter_keys = [str(spec["key"]) for spec in part_specs if str(spec.get("kind")) == "adapter"]

    payload = {
        "analysis": "rot_readout_main_vs_adapter_decomposition",
        "script_path": str(Path(__file__).resolve()),
        "summary_path": str(args.out.resolve()),
        "teacher_batch": str(args.teacher.resolve()),
        "parameters": {
            "rounds": int(args.rounds),
            "device": str(args.device),
        },
        "code_facts": {
            "forward_site": (
                "EventMotionModel.forward computes h_final = coupling_norm(...), then out = motion_head(h_final), "
                "then for each (slice_i, adapter_i) in zip(_bone_adapter_slices, _bone_adapters) it writes "
                "delta_full[..., slice_i] = adapter_i(h_final) and returns out_total = out_main + delta_full."
            ),
            "decomposition_definition": {
                "out_main": "motion_head(h_final)",
                "out_adapter_total": "sum_i scatter(adapter_i(h_final), slice_i -> full out dim)",
                "out_total": "out_main + out_adapter_total",
            },
            "rot_slice": _slice_list(tail_case["rot_slice"]),
            "output_slices": {
                "tail": tail_static["output_slices"],
                "baseline": base_static["output_slices"],
            },
            "adapter_slice_to_joint_tail": tail_static["bone_adapters"],
            "adapter_slice_to_joint_baseline": base_static["bone_adapters"],
            "adapter_slices_all_within_rot": all(
                bool(item.get("within_rot", False)) for item in tail_static["bone_adapters"]
            ),
            "matched_hidden_method": (
                "Observed motion-only delta_h_final is measured on tail teacher-conditioned clean state, then the exact same "
                "delta_h_final is added to clean h_final for both tail and baseline before applying only the head-side rot readout."
            ),
            "response_definition": (
                "For each part p in {total, main, adapter_total, adapter_i}, "
                "Delta out_rot_p = out_rot_p(h_clean + Delta h_final) - out_rot_p(h_clean), "
                "gain_rot_p = ||Delta out_rot_p||_2 / ||Delta h_final||_2."
            ),
            "local_jacobian_definition": (
                "With u = Delta h_final / ||Delta h_final||, directional local Jacobian metric is ||J_p,rot(h_clean) u||_2. "
                "A single torch.autograd.functional.jvp call returns total/main/adapter_total/adapter_i tangents together."
            ),
        },
        "cases": {
            "tailk7_current_control": {
                "ckpt_path": str(args.tail_ckpt.resolve()),
                "eval_json_path": str(args.tail_eval.resolve()),
                "runtime_overrides": dict(tail_case["runtime_overrides"]),
                "static_decomposition": tail_static,
            },
            "baseline_replace": {
                "ckpt_path": str(args.baseline_ckpt.resolve()),
                "eval_json_path": str(args.baseline_eval.resolve()),
                "runtime_overrides": dict(base_case["runtime_overrides"]),
                "static_decomposition": base_static,
            },
        },
        "part_specs": list(part_specs),
        "static_compare": {
            "motion_head_upper_bound_ratio": _ratio(
                tail_static["motion_head_upper_bound"],
                base_static["motion_head_upper_bound"],
            ),
            "bone_adapter_upper_bound_sum_ratio": _ratio(
                tail_static["bone_adapter_upper_bound_sum"],
                base_static["bone_adapter_upper_bound_sum"],
            ),
            "full_head_conservative_upper_bound_ratio": _ratio(
                tail_static["full_head_conservative_upper_bound"],
                base_static["full_head_conservative_upper_bound"],
            ),
            "static_main_vs_adapter": [
                {
                    "component": "motion_head",
                    "tail": tail_static["motion_head_upper_bound"],
                    "baseline": base_static["motion_head_upper_bound"],
                    "tail_base_ratio": _ratio(
                        tail_static["motion_head_upper_bound"],
                        base_static["motion_head_upper_bound"],
                    ),
                },
                {
                    "component": "bone_adapters_sum",
                    "tail": tail_static["bone_adapter_upper_bound_sum"],
                    "baseline": base_static["bone_adapter_upper_bound_sum"],
                    "tail_base_ratio": _ratio(
                        tail_static["bone_adapter_upper_bound_sum"],
                        base_static["bone_adapter_upper_bound_sum"],
                    ),
                },
                {
                    "component": "full_head_conservative_upper",
                    "tail": tail_static["full_head_conservative_upper_bound"],
                    "baseline": base_static["full_head_conservative_upper_bound"],
                    "tail_base_ratio": _ratio(
                        tail_static["full_head_conservative_upper_bound"],
                        base_static["full_head_conservative_upper_bound"],
                    ),
                },
            ],
            "static_adapter_wise": [
                {
                    "joint": str(tail_item["name"]),
                    "slice": list(tail_item["slice"]),
                    "tail_alpha_effective": tail_item["alpha_effective"],
                    "tail_upper_bound": tail_item["upper_bound"],
                    "baseline_alpha_effective": base_item["alpha_effective"],
                    "baseline_upper_bound": base_item["upper_bound"],
                    "tail_base_ratio": _ratio(tail_item["upper_bound"], base_item["upper_bound"]),
                }
                for tail_item, base_item in zip(tail_static["bone_adapters"], base_static["bone_adapters"])
            ],
        },
        "overall": overall,
        "overall_tables": {
            "main_vs_adapter": _part_overview(overall, [spec for spec in part_specs if str(spec["key"]) in main_keys]),
            "adapter_wise": _part_overview(overall, [spec for spec in part_specs if str(spec["key"]) in adapter_keys]),
        },
        "depth_buckets": depth_tables,
        "depth_overview": _bucket_overview(depth_tables, keys=main_keys),
        "step_in_cycle_buckets": sic_tables,
        "step_in_cycle_overview": _bucket_overview(sic_tables, keys=main_keys),
        "window_alignment": _window_alignment_table(rows, part_specs),
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.out.resolve()),
                "rows": len(rows),
                "overall_gain_rot_total_ratio": payload["overall"]["parts"]["total"]["gain_rot_ratio_of_means"],
                "overall_gain_rot_main_ratio": payload["overall"]["parts"]["main"]["gain_rot_ratio_of_means"],
                "overall_gain_rot_adapter_ratio": payload["overall"]["parts"]["adapter_total"]["gain_rot_ratio_of_means"],
                "overall_local_dir_rot_total_ratio": payload["overall"]["parts"]["total"]["local_dir_rot_ratio_of_means"],
                "overall_local_dir_rot_main_ratio": payload["overall"]["parts"]["main"]["local_dir_rot_ratio_of_means"],
                "overall_local_dir_rot_adapter_ratio": payload["overall"]["parts"]["adapter_total"]["local_dir_rot_ratio_of_means"],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
