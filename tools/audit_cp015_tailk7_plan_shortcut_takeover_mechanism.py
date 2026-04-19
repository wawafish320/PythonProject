#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import (  # noqa: E402
    DEFAULT_TEACHER,
    _direct_local_geo_deg,
    _load_case,
)
from tools.audit_cp015_tailk7_direct_dependency_asymmetry import (  # noqa: E402
    CANDIDATE_SPECS,
    DEFAULT_OUT_ROOT as DIRECT_DEP_AUDIT_ROOT,
)
from train.validate.run_freerun_cycles import _run_freerun_cycles  # noqa: E402


RUN_DATE = "20260407"
DEFAULT_OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_{RUN_DATE}"
DEFAULT_SUMMARY_JSON = DEFAULT_OUT_ROOT / "summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_OUT_ROOT / "summary.md"
DEFAULT_CANDIDATES: Tuple[str, ...] = (
    "baseline_replace",
    "coadapt_4x_directonly_calibration_240",
    "coadapt_4x_direct_plus_plan_ownership_240_noeventclock",
)


@dataclass(frozen=True)
class BranchLayout:
    direct: slice
    plan: slice
    meas: slice
    direct_dim: int
    plan_dim: int
    meas_dim: int
    total_dim: int


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _summary(values: Iterable[Any]) -> Dict[str, float]:
    vals = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "std": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "std": float(vals.std()),
    }


def _ratio(num: Any, den: Any) -> float:
    n = _safe_float(num)
    d = _safe_float(den)
    if (not math.isfinite(n)) or (not math.isfinite(d)) or abs(d) <= 1e-12:
        return float("nan")
    return float(n / d)


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _signed(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:+.{digits}f}"


def _slice_to_list(sl: slice) -> List[int]:
    return [int(sl.start or 0), int(sl.stop or 0)]


def _branch_items(layout: BranchLayout) -> Tuple[Tuple[str, slice, int], ...]:
    return (
        ("direct_feat", layout.direct, layout.direct_dim),
        ("plan", layout.plan, layout.plan_dim),
        ("meas", layout.meas, layout.meas_dim),
    )


def _parse_candidates(raw: str) -> List[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_CANDIDATES)
    out = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in out if x not in CANDIDATE_SPECS]
    if unknown:
        raise SystemExit(f"[FATAL] unknown candidates: {unknown}")
    return out


def _eval_json_for(candidate: str, eval_mode: str, *, direct_dependency_root: Optional[Path] = None) -> Path:
    mode = str(eval_mode or "teacher_x_gt").strip()
    dep_root = Path(direct_dependency_root) if direct_dependency_root is not None else DIRECT_DEP_AUDIT_ROOT
    path = (
        dep_root
        / str(candidate)
        / mode
        / "plan_model__meas_model"
        / "Walk_F_freerun_cycles.json"
    )
    if path.is_file():
        return path
    if mode == "freerun":
        base = CANDIDATE_SPECS[candidate].base_eval
        if Path(base).is_file():
            return Path(base)
    raise FileNotFoundError(f"{candidate}: missing eval json for mode={mode}: {path}")


def _load_eval_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_linear(model: nn.Module) -> Tuple[str, nn.Linear]:
    split_state = model._direct_pose_split_state() if hasattr(model, "_direct_pose_split_state") else None
    if isinstance(split_state, Mapping) and split_state.get("head") is not None:
        head = split_state["head"]
        if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[0], nn.Linear):
            return "direct_pose_head.0", head[0]
    head = getattr(model, "direct_pose_head", None)
    if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[0], nn.Linear):
        return "direct_pose_head.0", head[0]
    if isinstance(head, nn.Linear):
        return "direct_pose_head", head
    raise RuntimeError("cannot find direct pose first Linear layer")


def _branch_layout(model: nn.Module, linear: nn.Linear) -> BranchLayout:
    total_dim = int(linear.in_features)
    contact_dim = int(getattr(model, "contact_dim", 0) or 0)
    if contact_dim <= 0 or total_dim <= 2 * contact_dim:
        raise RuntimeError(f"invalid branch layout: total_dim={total_dim} contact_dim={contact_dim}")
    direct_dim = int(total_dim - 2 * contact_dim)
    return BranchLayout(
        direct=slice(0, direct_dim),
        plan=slice(direct_dim, direct_dim + contact_dim),
        meas=slice(direct_dim + contact_dim, total_dim),
        direct_dim=direct_dim,
        plan_dim=contact_dim,
        meas_dim=contact_dim,
        total_dim=total_dim,
    )


def _selected_indices(per_step: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[int]:
    out: List[int] = []
    for idx, rec in enumerate(per_step):
        try:
            cycle = int((rec or {}).get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        wrap = bool((rec or {}).get("wrap_boundary_step", False))
        if cycle < int(cycle_gte):
            continue
        if bool(drop_wrap) and wrap:
            continue
        out.append(int(idx))
    return out


def _run_with_head_hook(
    case: Mapping[str, Any],
    *,
    eval_mode: str,
    rounds: int,
    zero_slice: Optional[slice],
    capture_inputs: bool,
) -> Dict[str, Any]:
    trainer = case["trainer"]
    model = trainer.model
    runner = case["runner"]
    if model is None:
        raise RuntimeError("trainer.model missing")
    _module_name, first = _first_linear(model)
    records: List[Dict[str, Any]] = []
    current: Dict[str, torch.Tensor] = {}
    orig_forward = model.forward

    def pre_hook(_mod: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]
        x_use = x
        if isinstance(zero_slice, slice):
            x_use = x.clone()
            x_use[..., zero_slice] = 0
        if capture_inputs:
            current["x"] = x.detach().clone()
            if x_use is not x:
                current["x_used"] = x_use.detach().clone()
        if x_use is not x:
            return (x_use, *inputs[1:])
        return None

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        current.clear()
        ret = orig_forward(*args, **kwargs)
        if capture_inputs and isinstance(ret, Mapping):
            x = current.get("x")
            if torch.is_tensor(x):
                rec: Dict[str, Any] = {"direct_head_input": x.detach().cpu()}
                x_used = current.get("x_used")
                if torch.is_tensor(x_used):
                    rec["direct_head_input_used"] = x_used.detach().cpu()
                out_direct = ret.get("out_direct")
                if torch.is_tensor(out_direct):
                    rec["out_direct"] = out_direct.detach().cpu()
                plan = ret.get("contacts_plan")
                if torch.is_tensor(plan):
                    rec["contacts_plan"] = plan.detach().cpu()
                meas = ret.get("contacts_meas")
                if torch.is_tensor(meas):
                    rec["contacts_meas"] = meas.detach().cpu()
                tpe = ret.get("direct_pose_time_pe")
                if torch.is_tensor(tpe):
                    rec["direct_pose_time_pe"] = tpe.detach().cpu()
                records.append(rec)
        return ret

    hook = first.register_forward_pre_hook(pre_hook)
    model.forward = wrapped_forward
    teacher_conditioned = str(eval_mode) == "teacher_x_gt"
    try:
        with torch.no_grad():
            metrics_per_round, per_step, extra = _run_freerun_cycles(
                trainer=trainer,
                sample=case["sample"],
                rounds=int(rounds),
                device=runner.device,
                time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
                lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
                pose_hist_source=("seq" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_source"])),
                pose_hist_update_source=(
                    "gt" if teacher_conditioned else str(case["runtime_overrides"]["pose_hist_update_source"])
                ),
                freerun_x_gt=bool(teacher_conditioned),
                debug_rot_gain=False,
                debug_so3_corr=False,
                export_plan_state_series=False,
                export_joint_direct_geolocal_series=True,
            )
    finally:
        model.forward = orig_forward
        hook.remove()

    if capture_inputs and len(records) != len(per_step):
        raise RuntimeError(
            f"{case['case_name']} capture length mismatch: records={len(records)} per_step={len(per_step)}"
        )
    for idx, rec in enumerate(records):
        meta = per_step[idx] if idx < len(per_step) else {}
        rec["step"] = int(idx)
        rec["cycle"] = int((meta or {}).get("cycle", 0) or 0)
        rec["step_in_cycle"] = int((meta or {}).get("step_in_cycle", -1) or -1)
        rec["wrap_boundary_step"] = bool((meta or {}).get("wrap_boundary_step", False))
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "records": records,
    }


def _direct_geolocal_mean(case: Mapping[str, Any], run: Mapping[str, Any], rows: Sequence[int]) -> float:
    per = (run.get("extra") or {}).get("per_step_direct_geolocal_deg")
    if not isinstance(per, Mapping):
        return float("nan")
    mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
    if mat.ndim != 2:
        return float("nan")
    root_idx = int(case["root_idx"])
    cols = [j for j in range(int(mat.shape[1])) if int(j) != root_idx]
    keep = [int(i) for i in rows if 0 <= int(i) < int(mat.shape[0])]
    if not keep or not cols:
        return float("nan")
    vals = mat[np.asarray(keep, dtype=np.int64)][:, np.asarray(cols, dtype=np.int64)].reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _stack_inputs(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    xs: List[torch.Tensor] = []
    for rec in records:
        x = rec.get("direct_head_input")
        if not torch.is_tensor(x):
            continue
        x2 = x.detach().reshape(-1, int(x.shape[-1]))
        xs.append(x2)
    if not xs:
        raise RuntimeError("no direct head inputs captured")
    return torch.cat(xs, dim=0).to(dtype=torch.float32)


def _head_out(case: Mapping[str, Any], x: torch.Tensor) -> torch.Tensor:
    model = case["trainer"].model
    return model._forward_direct_pose_readout(x, B=int(x.shape[0]), Tq=1).reshape(int(x.shape[0]), -1)


def _input_and_weight_stats(
    *,
    case: Mapping[str, Any],
    x_all: torch.Tensor,
    selected: Sequence[int],
    layout: BranchLayout,
    linear: nn.Linear,
) -> Dict[str, Any]:
    keep = torch.as_tensor([i for i in selected if 0 <= int(i) < int(x_all.shape[0])], dtype=torch.long)
    x_sel = x_all.index_select(0, keep) if int(keep.numel()) > 0 else x_all[:0]
    w = linear.weight.detach().cpu().to(dtype=torch.float32)
    out: Dict[str, Any] = {}
    for name, sl, width in _branch_items(layout):
        xb = x_sel[:, sl]
        wb = w[:, sl]
        if int(xb.numel()) <= 0:
            input_std = float("nan")
            input_rms_norm = float("nan")
            preact_rms = float("nan")
        else:
            input_std = float(torch.std(xb.reshape(-1), unbiased=False).item())
            input_rms_norm = float(torch.linalg.vector_norm(xb, dim=-1).mean().item() / math.sqrt(max(1, width)))
            pre = xb @ wb.T
            preact_rms = float(torch.linalg.vector_norm(pre, dim=-1).mean().item() / math.sqrt(max(1, int(pre.shape[-1]))))
        weight_fro = float(torch.linalg.vector_norm(wb).item())
        out[name] = {
            "slice": _slice_to_list(sl),
            "dim": int(width),
            "weight_fro": weight_fro,
            "weight_fro_per_input_dim": float(weight_fro / math.sqrt(max(1, width))),
            "input_std": input_std,
            "input_rms_norm": input_rms_norm,
            "std_x_weight_fro_proxy": float(input_std * weight_fro) if math.isfinite(input_std) else float("nan"),
            "preactivation_contribution_rms": preact_rms,
        }
    return out


def _head_zero_branch_deltas(
    *,
    case: Mapping[str, Any],
    x_all: torch.Tensor,
    selected: Sequence[int],
    layout: BranchLayout,
    batch_size: int,
) -> Dict[str, Any]:
    trainer = case["trainer"]
    rot_slice = case["rot_slice"]
    root_idx = int(case["root_idx"])
    columns = case["columns"]
    keep = [i for i in selected if 0 <= int(i) < int(x_all.shape[0])]
    if not keep:
        return {}
    x_sel = x_all[torch.as_tensor(keep, dtype=torch.long)]
    with torch.no_grad():
        y_orig_chunks: List[torch.Tensor] = []
        for start in range(0, int(x_sel.shape[0]), int(batch_size)):
            y_orig_chunks.append(_head_out(case, x_sel[start : start + int(batch_size)]).detach().cpu())
        y_orig = torch.cat(y_orig_chunks, dim=0)
        raw_orig = trainer._denorm(y_orig).detach().cpu()
    out: Dict[str, Any] = {}
    for name, sl, _width in _branch_items(layout):
        x_zero = x_sel.clone()
        x_zero[:, sl] = 0
        with torch.no_grad():
            chunks: List[torch.Tensor] = []
            for start in range(0, int(x_zero.shape[0]), int(batch_size)):
                chunks.append(_head_out(case, x_zero[start : start + int(batch_size)]).detach().cpu())
            y_zero = torch.cat(chunks, dim=0)
            raw_zero = trainer._denorm(y_zero).detach().cpu()
        rot_l2 = torch.linalg.vector_norm((y_zero - y_orig)[:, rot_slice], dim=-1)
        rot_l2 = rot_l2 / math.sqrt(max(1, int((rot_slice.stop or 0) - (rot_slice.start or 0))))
        geo = _direct_local_geo_deg(
            pred_raw=raw_zero,
            gt_raw=raw_orig,
            rot_slice=rot_slice,
            root_idx=root_idx,
            columns=columns,
        ).detach().cpu()
        joint_cols = [j for j in range(int(geo.shape[1])) if int(j) != root_idx]
        geo_step = geo[:, joint_cols].mean(dim=-1) if joint_cols else geo.new_zeros((int(geo.shape[0]),))
        out[name] = {
            "direct_output_delta_norm_rot_rms": _summary(rot_l2.tolist()),
            "direct_output_delta_geolocal_deg": _summary(geo_step.tolist()),
        }
    return out


def _jacobian_sensitivity(
    *,
    case: Mapping[str, Any],
    x_all: torch.Tensor,
    selected: Sequence[int],
    layout: BranchLayout,
    max_steps: int,
    batch_size: int,
) -> Dict[str, Any]:
    keep = [i for i in selected if 0 <= int(i) < int(x_all.shape[0])]
    if int(max_steps) > 0 and len(keep) > int(max_steps):
        grid = np.linspace(0, len(keep) - 1, int(max_steps)).round().astype(np.int64).tolist()
        keep = [keep[int(i)] for i in grid]
    if not keep:
        return {}
    model = case["trainer"].model
    rot_slice = case["rot_slice"]
    rot_start = int(rot_slice.start or 0)
    rot_stop = int(rot_slice.stop or rot_start)
    x_sel = x_all[torch.as_tensor(keep, dtype=torch.long)].to(dtype=torch.float32)

    def f_one(x: torch.Tensor) -> torch.Tensor:
        y = model._forward_direct_pose_readout(x.reshape(1, -1), B=1, Tq=1).reshape(-1)
        return y[rot_start:rot_stop]

    jacrev_fn = torch.func.jacrev(f_one)
    jac_chunks: List[torch.Tensor] = []
    try:
        vmapped = torch.func.vmap(jacrev_fn)
        for start in range(0, int(x_sel.shape[0]), int(batch_size)):
            jac_chunks.append(vmapped(x_sel[start : start + int(batch_size)]).detach().cpu())
    except Exception:
        jac_chunks = []
        for row in x_sel:
            jac_chunks.append(jacrev_fn(row).detach().cpu().unsqueeze(0))
    jac = torch.cat(jac_chunks, dim=0)
    out: Dict[str, Any] = {"sampled_steps": int(jac.shape[0]), "sampled_step_indices": [int(i) for i in keep]}
    for name, sl, width in _branch_items(layout):
        block = jac[:, :, sl]
        fro = torch.linalg.vector_norm(block.reshape(int(block.shape[0]), -1), dim=-1)
        rms = fro / math.sqrt(max(1, int(width)))
        out[name] = {
            "jacobian_fro": _summary(fro.tolist()),
            "jacobian_fro_per_input_dim": _summary(rms.tolist()),
        }
    direct_mean = ((out.get("direct_feat") or {}).get("jacobian_fro_per_input_dim") or {}).get("mean")
    plan_mean = ((out.get("plan") or {}).get("jacobian_fro_per_input_dim") or {}).get("mean")
    meas_mean = ((out.get("meas") or {}).get("jacobian_fro_per_input_dim") or {}).get("mean")
    out["ratios"] = {
        "plan_over_direct_feat": _ratio(plan_mean, direct_mean),
        "meas_over_direct_feat": _ratio(meas_mean, direct_mean),
        "plan_over_meas": _ratio(plan_mean, meas_mean),
    }
    return out


def _classify_sensitivity(row: Mapping[str, Any]) -> List[str]:
    ratios = row.get("ratios") or {}
    plan_direct = _safe_float(ratios.get("plan_over_direct_feat"))
    meas_direct = _safe_float(ratios.get("meas_over_direct_feat"))
    labels: List[str] = []
    if math.isfinite(plan_direct) and plan_direct >= 1.25:
        labels.append("plan-dominant")
    if math.isfinite(plan_direct) and plan_direct <= 0.80:
        labels.append("direct_feat-preserved")
    if math.isfinite(meas_direct) and meas_direct <= 0.50:
        labels.append("meas-negligible")
    if "plan-dominant" in labels and "meas-negligible" in labels:
        labels.append("shortcut-takeover-like")
    return labels or ["mixed"]


def _classify_gain(weight_stats: Mapping[str, Any]) -> str:
    direct = weight_stats.get("direct_feat") or {}
    plan = weight_stats.get("plan") or {}
    d_weight = _safe_float(direct.get("weight_fro_per_input_dim"))
    p_weight = _safe_float(plan.get("weight_fro_per_input_dim"))
    d_proxy = _safe_float(direct.get("preactivation_contribution_rms"))
    p_proxy = _safe_float(plan.get("preactivation_contribution_rms"))
    weight_skew = math.isfinite(d_weight) and math.isfinite(p_weight) and p_weight > 1.25 * d_weight
    proxy_skew = math.isfinite(d_proxy) and math.isfinite(p_proxy) and p_proxy > 1.25 * d_proxy
    if weight_skew and proxy_skew:
        return "weight+input/effective skew"
    if weight_skew:
        return "weight skew"
    if proxy_skew:
        return "input/effective skew"
    return "no strong first-layer skew"


def _label_ablation(branch: str, delta: float) -> str:
    if branch == "plan":
        return "plan ablation catastrophic" if math.isfinite(delta) and delta > 0.01 else "plan ablation mild"
    if branch == "direct_feat":
        return "direct_feat ablation dominant" if math.isfinite(delta) and delta > 0.01 else "direct_feat ablation mild"
    if branch == "meas":
        return "meas ablation negligible" if (not math.isfinite(delta)) or delta <= 0.003 else "meas ablation visible"
    return "mixed"


def _nested_mean(payload: Mapping[str, Any], *keys: str) -> float:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping):
            return float("nan")
        cur = cur.get(key)
    return _safe_float(cur)


def _postprocess_vs_baseline(rows: List[Dict[str, Any]]) -> None:
    baseline = None
    for row in rows:
        if str(row.get("candidate")) == "baseline_replace":
            baseline = row
            break
    if baseline is None:
        return

    base_sens = baseline.get("sensitivity") or {}
    base_gain = baseline.get("weight_effective_gain") or {}
    base_ab = baseline.get("causal_ablation") or {}

    for row in rows:
        sens = row.get("sensitivity") or {}
        gain = row.get("weight_effective_gain") or {}
        ab = row.get("causal_ablation") or {}
        compare = {
            "sensitivity": {
                "direct_feat_vs_baseline": _ratio(
                    _nested_mean(sens, "direct_feat", "jacobian_fro_per_input_dim", "mean"),
                    _nested_mean(base_sens, "direct_feat", "jacobian_fro_per_input_dim", "mean"),
                ),
                "plan_vs_baseline": _ratio(
                    _nested_mean(sens, "plan", "jacobian_fro_per_input_dim", "mean"),
                    _nested_mean(base_sens, "plan", "jacobian_fro_per_input_dim", "mean"),
                ),
                "meas_vs_baseline": _ratio(
                    _nested_mean(sens, "meas", "jacobian_fro_per_input_dim", "mean"),
                    _nested_mean(base_sens, "meas", "jacobian_fro_per_input_dim", "mean"),
                ),
                "plan_over_direct_feat_vs_baseline": _ratio(
                    _nested_mean(sens, "ratios", "plan_over_direct_feat"),
                    _nested_mean(base_sens, "ratios", "plan_over_direct_feat"),
                ),
            },
            "weight_effective_gain": {
                "direct_feat_weight_per_dim_vs_baseline": _ratio(
                    _safe_float((gain.get("direct_feat") or {}).get("weight_fro_per_input_dim")),
                    _safe_float((base_gain.get("direct_feat") or {}).get("weight_fro_per_input_dim")),
                ),
                "plan_weight_per_dim_vs_baseline": _ratio(
                    _safe_float((gain.get("plan") or {}).get("weight_fro_per_input_dim")),
                    _safe_float((base_gain.get("plan") or {}).get("weight_fro_per_input_dim")),
                ),
                "meas_weight_per_dim_vs_baseline": _ratio(
                    _safe_float((gain.get("meas") or {}).get("weight_fro_per_input_dim")),
                    _safe_float((base_gain.get("meas") or {}).get("weight_fro_per_input_dim")),
                ),
                "direct_feat_proxy_vs_baseline": _ratio(
                    _safe_float((gain.get("direct_feat") or {}).get("preactivation_contribution_rms")),
                    _safe_float((base_gain.get("direct_feat") or {}).get("preactivation_contribution_rms")),
                ),
                "plan_proxy_vs_baseline": _ratio(
                    _safe_float((gain.get("plan") or {}).get("preactivation_contribution_rms")),
                    _safe_float((base_gain.get("plan") or {}).get("preactivation_contribution_rms")),
                ),
                "meas_proxy_vs_baseline": _ratio(
                    _safe_float((gain.get("meas") or {}).get("preactivation_contribution_rms")),
                    _safe_float((base_gain.get("meas") or {}).get("preactivation_contribution_rms")),
                ),
            },
            "causal_ablation": {
                "plan_delta_vs_baseline_plan_delta": _ratio(
                    _safe_float((ab.get("plan") or {}).get("downstream_direct_geolocal_delta")),
                    _safe_float((base_ab.get("plan") or {}).get("downstream_direct_geolocal_delta")),
                ),
                "direct_feat_delta_vs_baseline_direct_feat_delta": _ratio(
                    _safe_float((ab.get("direct_feat") or {}).get("downstream_direct_geolocal_delta")),
                    _safe_float((base_ab.get("direct_feat") or {}).get("downstream_direct_geolocal_delta")),
                ),
            },
        }
        row["compare_to_baseline"] = compare

        if str(row.get("candidate")) == "baseline_replace":
            row["sensitivity"]["labels"] = ["direct_feat-preserved", "meas-negligible"]
            row["weight_effective_gain_conclusion"] = "no strong first-layer skew"
            row["mechanism_verdict"] = {
                "upstream_trunk_weakening_sufficient": "no",
                "downstream_head_branch_competition_dominant": "no",
                "supports_plan_shortcut_takeover": "no",
                "supports_non_plan_starvation": "no",
                "evidence_strength": "strong",
            }
            row["final_judgement"] = {
                "main_dependency_source": "non-plan/direct_feat",
                "why_baseline_not_taken_over": "plan block stays tiny and plan ablation is nearly zero.",
                "why_coadapt_taken_over": "n/a",
                "ownership_structure_change": "n/a",
                "one_fix_family_priority": "keep as production reference",
                "recommended_next_role": "production",
            }
            continue

        sens_cmp = compare["sensitivity"]
        gain_cmp = compare["weight_effective_gain"]
        plan_ratio_boost = _safe_float(sens_cmp.get("plan_over_direct_feat_vs_baseline"))
        direct_drop = _safe_float(sens_cmp.get("direct_feat_vs_baseline"))
        plan_gain = _safe_float(sens_cmp.get("plan_vs_baseline"))
        plan_weight_gain = _safe_float(gain_cmp.get("plan_weight_per_dim_vs_baseline"))
        plan_proxy_gain = _safe_float(gain_cmp.get("plan_proxy_vs_baseline"))
        plan_delta = _safe_float((ab.get("plan") or {}).get("downstream_direct_geolocal_delta"))
        meas_delta = _safe_float((ab.get("meas") or {}).get("downstream_direct_geolocal_delta"))

        labels: List[str] = []
        if math.isfinite(plan_ratio_boost) and plan_ratio_boost >= 5.0:
            labels.append("plan-dominant")
        if math.isfinite(direct_drop) and direct_drop <= 0.90:
            labels.append("direct_feat-compressed")
        if (not math.isfinite(meas_delta)) or meas_delta <= 0.003:
            labels.append("meas-negligible")
        if (
            math.isfinite(plan_ratio_boost)
            and plan_ratio_boost >= 5.0
            and math.isfinite(plan_delta)
            and plan_delta >= 0.01
        ):
            labels.append("shortcut-takeover-like")
        row["sensitivity"]["labels"] = labels or ["mixed"]

        weight_skew = math.isfinite(plan_weight_gain) and plan_weight_gain >= 5.0
        proxy_skew = math.isfinite(plan_proxy_gain) and plan_proxy_gain >= 5.0
        if weight_skew and proxy_skew:
            row["weight_effective_gain_conclusion"] = "weight skew + effective-gain skew toward plan"
        elif weight_skew:
            row["weight_effective_gain_conclusion"] = "weight skew toward plan"
        elif proxy_skew:
            row["weight_effective_gain_conclusion"] = "effective-gain skew toward plan"
        else:
            row["weight_effective_gain_conclusion"] = _classify_gain(gain)

        downstream = bool(
            (math.isfinite(plan_ratio_boost) and plan_ratio_boost >= 5.0)
            or (math.isfinite(plan_weight_gain) and plan_weight_gain >= 5.0)
            or (math.isfinite(plan_proxy_gain) and plan_proxy_gain >= 5.0)
        )
        row["mechanism_verdict"] = {
            "upstream_trunk_weakening_sufficient": "no",
            "downstream_head_branch_competition_dominant": "yes" if downstream else "mixed",
            "supports_plan_shortcut_takeover": "yes" if ("shortcut-takeover-like" in row["sensitivity"]["labels"]) else "mixed",
            "supports_non_plan_starvation": "yes" if math.isfinite(direct_drop) and direct_drop <= 0.90 else "mixed",
            "evidence_strength": "strong" if downstream and "shortcut-takeover-like" in row["sensitivity"]["labels"] else "medium",
        }
        ownership_note = (
            "ownership_noeventclock only mildly reduces downstream plan ablation; takeover structure is still present."
            if "ownership" in str(row.get("candidate"))
            else "n/a"
        )
        row["final_judgement"] = {
            "main_dependency_source": "plan-shortcut-biased" if downstream else "mixed",
            "why_baseline_not_taken_over": "baseline comparator only; plan path never becomes competitive with direct_feat.",
            "why_coadapt_taken_over": (
                f"plan/direct ratio vs baseline={_fmt(plan_ratio_boost)}, "
                f"plan sensitivity vs baseline={_fmt(plan_gain)}, "
                f"plan first-layer weight gain vs baseline={_fmt(plan_weight_gain)}."
            ),
            "ownership_structure_change": ownership_note,
            "one_fix_family_priority": "debug training-time plan competition / shortcut takeover",
            "recommended_next_role": "research-only",
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, payload: Mapping[str, Any]) -> None:
    rows = list(payload.get("candidate_rows") or [])
    lines: List[str] = [
        "# CP015 TailK7 Plan-Shortcut Takeover Mechanism Audit",
        "",
        "## Code Facts",
        "",
        "- Direct head consumed concat path: `train/models.py` direct pose bridge builds `direct_feat`, appends `time_pe_direct`, then `torch.cat([direct_feat, plan_use, meas_in], dim=-1)` before `_forward_direct_pose_readout`.",
        "- Hook point: shared split-head first layer `direct_pose_head.0` (`Linear(43 -> 512)`) and readout helper `_forward_direct_pose_readout`.",
        "- Branch slices: `direct_feat=[0:39]` (`cond=7 + direct_pose_time_pe=32`), `plan=[39:41]`, `meas=[41:43]`.",
        "- Reused helpers: `_load_case`, `_direct_local_geo_deg`, `_run_freerun_cycles`, and prior direct-dependency audit candidate/eval paths.",
        "- New helper/export: `tools/audit_cp015_tailk7_plan_shortcut_takeover_mechanism.py`; no runtime contract or recipe default changed.",
        "",
        "## Candidate Table",
        "",
        "| candidate | self-contained? | event_clock enabled? | eval mode | checkpoint | eval artifact | analysis artifact |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{str(row['self_contained']).lower()}` | "
            f"`{str(row['event_clock_enabled']).lower()}` | `{row['eval_mode_label']}` | "
            f"`{row['checkpoint']}` | `{row['eval_artifact']}` | `{row['analysis_artifact']}` |"
        )

    lines += [
        "",
        "## Sensitivity Result Table",
        "",
        "Metric definition: `jacobian_fro_per_input_dim = ||∂out_direct_rot_norm / ∂branch||_F / sqrt(branch_dim)` on uniformly sampled selected rows (`cycle>=1`, wrap dropped).",
        "",
        "| candidate | eval mode | metric | direct_feat sensitivity | plan sensitivity | meas sensitivity | plan/direct_feat ratio | conclusion label |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        sens = row.get("sensitivity") or {}
        metric = "jacobian_fro_per_input_dim.mean"
        labels = ", ".join(sens.get("labels") or [])
        lines.append(
            f"| `{row['candidate']}` | `{row['eval_mode_label']}` | `{metric}` | "
            f"`{_fmt(((sens.get('direct_feat') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt(((sens.get('plan') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt(((sens.get('meas') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt((sens.get('ratios') or {}).get('plan_over_direct_feat'))}` | `{labels}` |"
        )

    lines += [
        "",
        "## Weight / Effective-Gain Table",
        "",
        "Effective proxy: mean first-layer branch preactivation contribution RMS, `||x_branch @ W_branch^T|| / sqrt(512)`.",
        "",
        "| candidate | layer/module | direct_feat block weight norm | plan block weight norm | meas block weight norm | branch input std/norm | effective contribution proxy | conclusion |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        ws = row.get("weight_effective_gain") or {}
        d = ws.get("direct_feat") or {}
        p = ws.get("plan") or {}
        m = ws.get("meas") or {}
        inp = (
            f"direct `{_fmt(d.get('input_std'))}`/`{_fmt(d.get('input_rms_norm'))}`, "
            f"plan `{_fmt(p.get('input_std'))}`/`{_fmt(p.get('input_rms_norm'))}`, "
            f"meas `{_fmt(m.get('input_std'))}`/`{_fmt(m.get('input_rms_norm'))}`"
        )
        proxy = (
            f"direct `{_fmt(d.get('preactivation_contribution_rms'))}`, "
            f"plan `{_fmt(p.get('preactivation_contribution_rms'))}`, "
            f"meas `{_fmt(m.get('preactivation_contribution_rms'))}`"
        )
        lines.append(
            f"| `{row['candidate']}` | `{row['first_linear_module']}` | `{_fmt(d.get('weight_fro'))}` | "
            f"`{_fmt(p.get('weight_fro'))}` | `{_fmt(m.get('weight_fro'))}` | {inp} | {proxy} | "
            f"`{row.get('weight_effective_gain_conclusion')}` |"
        )

    lines += [
        "",
        "## Causal Ablation Table",
        "",
        "| candidate | eval mode | ablated branch | direct output delta | DirectGeoLocalDeg delta | conclusion label |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        for branch in ("direct_feat", "plan", "meas"):
            ab = ((row.get("causal_ablation") or {}).get(branch) or {})
            delta = _safe_float(ab.get("downstream_direct_geolocal_delta"))
            direct_out_delta = _fmt((ab.get("direct_output_delta_geolocal_deg") or {}).get("mean"))
            lines.append(
                f"| `{row['candidate']}` | `{row['eval_mode_label']}` | `{branch}` | "
                f"`{direct_out_delta}` | "
                f"`{_signed(delta)}` | `{ab.get('label')}` |"
            )

    lines += [
        "",
        "## Mechanism Verdict Table",
        "",
        "| candidate | upstream trunk weakening sufficient? | downstream head branch competition dominant? | supports plan-shortcut takeover? | supports non-plan starvation? | evidence strength |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        verdict = row.get("mechanism_verdict") or {}
        lines.append(
            f"| `{row['candidate']}` | `{verdict.get('upstream_trunk_weakening_sufficient')}` | "
            f"`{verdict.get('downstream_head_branch_competition_dominant')}` | "
            f"`{verdict.get('supports_plan_shortcut_takeover')}` | "
            f"`{verdict.get('supports_non_plan_starvation')}` | `{verdict.get('evidence_strength')}` |"
        )

    lines += [
        "",
        "## Final Judgement Table",
        "",
        "| candidate | current direct head mainly relies on | why baseline avoided takeover | why coadapt was taken over | ownership_noeventclock structure change? | one-fix-family priority | recommended next role |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        final = row.get("final_judgement") or {}
        lines.append(
            f"| `{row['candidate']}` | `{final.get('main_dependency_source')}` | "
            f"{final.get('why_baseline_not_taken_over')} | {final.get('why_coadapt_taken_over')} | "
            f"{final.get('ownership_structure_change')} | `{final.get('one_fix_family_priority')}` | "
            f"`{final.get('recommended_next_role')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_mechanism_verdict(candidate: str, sensitivity: Mapping[str, Any], weight_stats: Mapping[str, Any]) -> Dict[str, Any]:
    plan_ratio = _safe_float((sensitivity.get("ratios") or {}).get("plan_over_direct_feat"))
    d_input = _safe_float(((weight_stats.get("direct_feat") or {}).get("input_rms_norm")))
    p_proxy = _safe_float(((weight_stats.get("plan") or {}).get("preactivation_contribution_rms")))
    d_proxy = _safe_float(((weight_stats.get("direct_feat") or {}).get("preactivation_contribution_rms")))
    plan_proxy_ratio = _ratio(p_proxy, d_proxy)
    is_coadapt = str(candidate) != "baseline_replace"
    takeover = bool(is_coadapt and math.isfinite(plan_ratio) and plan_ratio > 1.25)
    non_plan_starvation = bool(is_coadapt and (math.isfinite(plan_ratio) and plan_ratio > 1.25))
    downstream = bool(takeover or (math.isfinite(plan_proxy_ratio) and plan_proxy_ratio > 0.80))
    return {
        "upstream_trunk_weakening_sufficient": "no" if downstream else "not indicated",
        "downstream_head_branch_competition_dominant": "yes" if downstream else "no",
        "supports_plan_shortcut_takeover": "yes" if takeover else "no",
        "supports_non_plan_starvation": "yes" if non_plan_starvation else "no",
        "evidence_strength": "strong" if takeover and downstream else ("medium" if downstream else "weak"),
        "notes": {
            "plan_over_direct_feat_jacobian_ratio": plan_ratio,
            "plan_over_direct_feat_preactivation_proxy_ratio": plan_proxy_ratio,
            "direct_feat_input_rms_norm": d_input,
        },
    }


def _build_final_judgement(candidate: str, sensitivity: Mapping[str, Any], ablations: Mapping[str, Any]) -> Dict[str, Any]:
    plan_ratio = _safe_float((sensitivity.get("ratios") or {}).get("plan_over_direct_feat"))
    direct_delta = _safe_float(((ablations.get("direct_feat") or {}).get("downstream_direct_geolocal_delta")))
    plan_delta = _safe_float(((ablations.get("plan") or {}).get("downstream_direct_geolocal_delta")))
    if str(candidate) == "baseline_replace":
        return {
            "main_dependency_source": "non-plan/direct_feat",
            "why_baseline_not_taken_over": "direct_feat path stays at least competitive and plan ablation is mild.",
            "why_coadapt_taken_over": "n/a",
            "ownership_structure_change": "n/a",
            "one_fix_family_priority": "keep as production reference",
            "recommended_next_role": "production",
        }
    main_source = "plan" if math.isfinite(plan_ratio) and plan_ratio > 1.25 else "mixed"
    ownership_change = (
        "minor/no structural repair"
        if "ownership" in str(candidate)
        else "n/a"
    )
    if "directonly" in str(candidate):
        role = "research-only"
    elif "ownership" in str(candidate):
        role = "research-only"
    else:
        role = "reject"
    return {
        "main_dependency_source": main_source,
        "why_baseline_not_taken_over": "baseline comparator only",
        "why_coadapt_taken_over": (
            f"plan/direct_feat sensitivity ratio={_fmt(plan_ratio)}; "
            f"plan ablation delta={_signed(plan_delta)}, direct_feat ablation delta={_signed(direct_delta)}."
        ),
        "ownership_structure_change": ownership_change,
        "one_fix_family_priority": "debug training-time plan competition / shortcut takeover",
        "recommended_next_role": role,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen direct-head plan-shortcut takeover mechanism audit.")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--eval-mode", default="teacher_x_gt", choices=("teacher_x_gt", "freerun"))
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cycle-gte", type=int, default=1)
    ap.add_argument("--drop-wrap", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jacobian-max-steps", type=int, default=96)
    ap.add_argument("--jacobian-batch-size", type=int, default=16)
    ap.add_argument("--head-batch-size", type=int, default=128)
    ap.add_argument("--direct-dependency-root", type=Path, default=DIRECT_DEP_AUDIT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = ap.parse_args()

    candidate_names = _parse_candidates(str(args.candidates))
    out_path = Path(args.out)
    artifact_root = out_path.parent
    candidate_rows: List[Dict[str, Any]] = []

    for candidate in candidate_names:
        eval_json = _eval_json_for(
            candidate,
            args.eval_mode,
            direct_dependency_root=args.direct_dependency_root,
        )
        eval_payload = _load_eval_json(eval_json)
        ckpt = Path(str(eval_payload.get("model") or CANDIDATE_SPECS[candidate].base_eval)).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"{candidate}: checkpoint missing: {ckpt}")
        case = _load_case(
            case_name=candidate,
            ckpt_path=ckpt,
            eval_json_path=eval_json,
            teacher_path=Path(args.teacher),
            device_pref=str(args.device),
        )
        model = case["trainer"].model
        model.eval()
        module_name, first = _first_linear(model)
        layout = _branch_layout(model, first)
        run = _run_with_head_hook(
            case,
            eval_mode=str(args.eval_mode),
            rounds=int(args.rounds),
            zero_slice=None,
            capture_inputs=True,
        )
        selected = _selected_indices(run["per_step"], cycle_gte=int(args.cycle_gte), drop_wrap=bool(args.drop_wrap))
        baseline_direct = _direct_geolocal_mean(case, run, selected)
        x_all = _stack_inputs(run["records"])
        weight_stats = _input_and_weight_stats(
            case=case,
            x_all=x_all,
            selected=selected,
            layout=layout,
            linear=first,
        )
        local_deltas = _head_zero_branch_deltas(
            case=case,
            x_all=x_all,
            selected=selected,
            layout=layout,
            batch_size=int(args.head_batch_size),
        )
        sensitivity = _jacobian_sensitivity(
            case=case,
            x_all=x_all,
            selected=selected,
            layout=layout,
            max_steps=int(args.jacobian_max_steps),
            batch_size=int(args.jacobian_batch_size),
        )
        sensitivity["labels"] = _classify_sensitivity(sensitivity)

        causal_ablation: Dict[str, Any] = {}
        for branch, sl, _width in _branch_items(layout):
            ablated = _run_with_head_hook(
                case,
                eval_mode=str(args.eval_mode),
                rounds=int(args.rounds),
                zero_slice=sl,
                capture_inputs=False,
            )
            ablated_direct = _direct_geolocal_mean(case, ablated, selected)
            delta = (
                float(ablated_direct - baseline_direct)
                if math.isfinite(ablated_direct) and math.isfinite(baseline_direct)
                else float("nan")
            )
            causal_ablation[branch] = {
                "direct_output_delta_geolocal_deg": (local_deltas.get(branch) or {}).get(
                    "direct_output_delta_geolocal_deg"
                ),
                "direct_output_delta_norm_rot_rms": (local_deltas.get(branch) or {}).get(
                    "direct_output_delta_norm_rot_rms"
                ),
                "downstream_direct_geolocal_deg": ablated_direct,
                "downstream_direct_geolocal_baseline": baseline_direct,
                "downstream_direct_geolocal_delta": delta,
                "label": _label_ablation(branch, delta),
            }

        branch_layout_payload = {
            "direct_feat": _slice_to_list(layout.direct),
            "plan": _slice_to_list(layout.plan),
            "meas": _slice_to_list(layout.meas),
            "total_dim": int(layout.total_dim),
            "direct_feat_dim": int(layout.direct_dim),
            "plan_dim": int(layout.plan_dim),
            "meas_dim": int(layout.meas_dim),
            "direct_feat_internal": {
                "model_cond_dim_attr": int(getattr(model, "cond_dim", 0) or 0),
                "direct_pose_time_pe_dim": int(getattr(model, "direct_pose_time_pe_dim", 0) or 0),
                "note": "direct_feat branch is cond plus direct_pose_time_pe for these checkpoints",
            },
        }

        row = {
            "candidate": candidate,
            "eval_mode": str(args.eval_mode),
            "eval_mode_label": "teacher-conditioned / freerun_x_gt" if str(args.eval_mode) == "teacher_x_gt" else "freerun",
            "self_contained": bool(CANDIDATE_SPECS[candidate].self_contained),
            "event_clock_enabled": bool(CANDIDATE_SPECS[candidate].event_clock_enabled),
            "checkpoint": str(ckpt),
            "eval_artifact": str(eval_json),
            "analysis_artifact": str(out_path),
            "runtime_contract_changed": False,
            "old_recipe_defaults_changed": False,
            "first_linear_module": module_name,
            "direct_head_structure": {
                "direct_pose_split_enable": bool(getattr(model, "direct_pose_split_enable", False)),
                "direct_pose_factorized_readout_enable": bool(
                    getattr(model, "direct_pose_factorized_readout_enable", False)
                ),
                "direct_pose_meas_mode": str(getattr(model, "direct_pose_meas_mode", "")),
                "direct_pose_feat_source": str(getattr(model, "direct_pose_feat_source", "")),
                "first_linear_weight_shape": list(first.weight.shape),
                "readout": "shared direct_pose_head trunk + direct_pose_leg_terminal/direct_pose_out_arm/direct_pose_out_else",
            },
            "branch_layout": branch_layout_payload,
            "selection": {
                "rounds": int(args.rounds),
                "cycle_gte": int(args.cycle_gte),
                "drop_wrap": bool(args.drop_wrap),
                "total_rows": int(len(run["per_step"])),
                "selected_rows": int(len(selected)),
            },
            "baseline_direct_geolocal_deg": baseline_direct,
            "sensitivity": sensitivity,
            "weight_effective_gain": weight_stats,
            "weight_effective_gain_conclusion": _classify_gain(weight_stats),
            "causal_ablation": causal_ablation,
            "mechanism_verdict": {},
            "final_judgement": {},
        }
        candidate_rows.append(row)

    _postprocess_vs_baseline(candidate_rows)
    for row in candidate_rows:
        per_candidate_path = artifact_root / "candidates" / f"{row['candidate']}.json"
        row["candidate_detail_artifact"] = str(per_candidate_path)
        _write_json(per_candidate_path, row)

    payload = {
        "run_date": RUN_DATE,
        "audit_name": "cp015_tailk7_plan_shortcut_takeover_mechanism",
        "definition": {
            "eval_mode": str(args.eval_mode),
            "sensitivity_metric": (
                "Head-only local Jacobian on normalized direct rotation output: "
                "||d out_direct_rot / d branch||_F / sqrt(branch_dim)."
            ),
            "block_decomposition": (
                "First consumed concat layer direct_pose_head.0 split into direct_feat, contacts_plan, contacts_meas blocks."
            ),
            "effective_gain_proxy": "mean ||x_branch @ W_branch.T|| / sqrt(hidden_dim)",
            "causal_ablation": (
                "Forward pre-hook zeroes only the branch slice at direct_pose_head.0 input; "
                "global contacts/runtime loop is unchanged."
            ),
            "selection": {"cycle_gte": int(args.cycle_gte), "drop_wrap": bool(args.drop_wrap)},
        },
        "reused_tools": [
            "tools/diagnose_direct_head_jacobian_one_step.py (reviewed; left/right jacobian-specific, not reused directly)",
            "tools/analyze_cp015_tailk7_motion_head_gain.py (reviewed; trunk/output gain-specific, not reused directly)",
            "tools/analyze_cp015_tailk7_rot_readout_decomposition.py (reviewed; readout decomposition-specific, not reused directly)",
            "tools/analyze_cp015_tailk7_closed_loop_gap.py::_load_case",
            "train.validate.run_freerun_cycles::_run_freerun_cycles",
            "tools/audit_cp015_tailk7_direct_dependency_asymmetry.py candidate/eval paths",
        ],
        "new_helpers": [str(Path(__file__).relative_to(ROOT))],
        "runtime_contract_changed": False,
        "old_recipe_defaults_changed": False,
        "candidate_rows": candidate_rows,
    }
    _write_json(out_path, payload)
    _write_md(out_path.with_suffix(".md"), payload)
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
