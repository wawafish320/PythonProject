#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _build_step_ctx,
    _case_bundle,
    _norm_l2,
    _prepare_fixed_offset_context,
    _restore_weight_swap,
    _run_single_step,
    _temporary_weight_swap,
)
from tools.audit_cp015_tailk7_plan_shortcut_takeover_mechanism import _first_linear  # noqa: E402
from tools.run_cp015_tailk7_upstream_replace_transferability_e0 import (  # noqa: E402
    BASELINE_REPLACE_CKPT,
    BASELINE_REPLACE_CONFIG,
    BASELINE_REPLACE_EVAL,
    COADAPT_HOST_CKPT,
    COADAPT_HOST_CONFIG,
    COADAPT_HOST_EVAL,
    DEFAULT_OFFSET,
    DEFAULT_TEACHER,
    DIRECT_BRANCH_MODULES,
    _add_closure,
    _safe_float,
    _tensor_metric_gaps,
)
from tools.run_cp015_tailk_curriculum_e2a import E2A_70A_CKPT, E2A_70A_EVAL  # noqa: E402
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP3_70A_CKPT,
    TOP3_70A_EVAL,
)
from train import posttrain  # noqa: E402


RUN_DATE = "20260409"
RUN_NAME = "direct_head_input_guard_a1s4"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_direct_head_input_guard_a1s4_record.md"

A1S1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_partial_transplant_boundary_a1s1_20260409" / "summary.json"
A1S1_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_partial_transplant_boundary_a1s1_record.md"
A1S2_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_mixed_contract_a1s2_20260409" / "summary.json"
A1S2_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_mixed_contract_a1s2_record.md"
A1S3_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_replace_absorb_boundary_a1s3_20260409" / "summary.json"
A1S3_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_replace_absorb_boundary_a1s3_record.md"

SHORT_HORIZON_STEPS = 12
MOMENT_MATCH_EPS = 1e-6
CLEAR_AGG_MARGIN = 0.05
LAYER0_NONDISC_L2 = 0.01
LAYER0_NONDISC_COS = 0.01

E1_MIX_MODULES: tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_leg_head",
    "direct_pose_out_leg",
)
E2A_MIX_MODULES: tuple[str, ...] = (
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_out_arm",
    "direct_pose_out_else",
)

INHERITED_CONCLUSIONS: list[str] = [
    "root cause not in planner semantics mainline",
    "root cause not in replace-entry external rollout state",
    "root cause not in contacts_in_t",
    "earliest semantic split at direct_pose_head boundary",
    "direct_pose_head is earliest boundary / necessary anchor but not standalone sufficient",
    "baseline 7-module direct branch can transfer into coadapt context",
    "E1-top3 is the only clearly effective upstream intervention so far",
    "all late/full top7 variants are worse than E1-top3",
    "E3A-RF further argues allocation ordering is not a sufficient lever",
    "current normality probe is non-discriminative and not a main criterion",
]

A1S1_DIRECT_INHERITED: list[str] = [
    "A1S1-anchor_only 不比 E2A-R full7 更 replace-transferable",
    "A1-S1 更像 Case 3",
    "更像 shared head 本身 already compromised",
    "anchor_plus_nonleg residual retention 明显好于 anchor_plus_leg",
]

A1S2_DIRECT_INHERITED: list[str] = [
    "A1S2-mix-nonleg 比 E2A-R full7 更好但不够 clear-win",
    "A1S2-mix-nonleg aggregate 上接近 E1-top3 full7",
    "A1S2-mix-nonleg 同时改善了相对 E2A-R full7 的 dir_nonleg 和 dir_leg",
    "plain mixed transplant 还不足以支持 replace-side absorb-expansion 已 solved",
]

A1S3_DIRECT_INHERITED: list[str] = [
    "两个 A1-S3 split arms 都不优于 A1S2-mix-nonleg",
    "plain replace-side split 仍不足以成为 decisive absorb 路线",
    "若强行二选一，只有 weak lean toward host nonleg out side；当前推荐仍偏向更早 boundary / stronger boundary guard",
]

DIAGNOSTIC_ARM_ORDER: tuple[str, ...] = (
    "target-full7",
    "E1-top3-full7",
    "E2A-R-full7",
    "A1S2-mix-nonleg",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _fmt(value: Any) -> str:
    val = _safe_float(value)
    return "nan" if not math.isfinite(val) else f"{val:.6f}"


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _tensor_shape(x: Optional[torch.Tensor]) -> List[int]:
    if not torch.is_tensor(x):
        return []
    return [int(v) for v in x.shape]


def _flatten_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    return x.detach().reshape(-1).to(dtype=torch.float32, device="cpu")


def _cosine_similarity(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    va = _flatten_tensor(a)
    vb = _flatten_tensor(b)
    if va is None or vb is None or tuple(va.shape) != tuple(vb.shape) or int(va.numel()) <= 0:
        return float("nan")
    na = float(torch.linalg.vector_norm(va).item())
    nb = float(torch.linalg.vector_norm(vb).item())
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(torch.dot(va, vb).item() / (na * nb))


def _cosine_distance(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    sim = _cosine_similarity(a, b)
    if not math.isfinite(sim):
        return float("nan")
    return float(max(0.0, 1.0 - sim))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray([_safe_float(v) for v in xs], dtype=np.float64)
    y = np.asarray([_safe_float(v) for v in ys], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _stack_rows(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.reshape(1, -1).detach().cpu().to(dtype=torch.float32)
    return x.detach().reshape(-1, int(x.shape[-1])).cpu().to(dtype=torch.float32)


def _channel_stats(x: torch.Tensor) -> Dict[str, Any]:
    rows = _stack_rows(x)
    mean_vec = rows.mean(dim=0)
    std_vec = rows.std(dim=0, unbiased=False)
    return {
        "rows": rows,
        "shape": [int(v) for v in x.shape],
        "rows_count": int(rows.shape[0]),
        "feature_dim": int(rows.shape[-1]),
        "mean_vector": mean_vec.tolist(),
        "std_vector": std_vec.tolist(),
        "mean_abs": float(rows.abs().mean().item()),
        "rms_norm_per_row": float(
            torch.linalg.vector_norm(rows, dim=-1).mean().item() / math.sqrt(max(1, int(rows.shape[-1])))
        ),
    }


def _extract_raw_step_metric(
    accum_ctx: Mapping[str, List[torch.Tensor]],
    key: str,
    prev_len: int,
    step_weight: float,
) -> float:
    seq = accum_ctx.get(key) or []
    if len(seq) <= int(prev_len):
        return float("nan")
    val = seq[-1]
    if not torch.is_tensor(val):
        return float("nan")
    weighted = float(val.detach().cpu().item())
    if abs(step_weight) <= 1e-12:
        return float("nan")
    return float(weighted / step_weight)


def _transfer_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, Any]:
    gap_keys = (
        "out_direct_gap",
        "dir_base_gap",
        "dir_leg_gap",
        "dir_nonleg_gap",
    )
    closure_keys = (
        "out_direct_closure_ratio",
        "dir_base_closure_ratio",
        "dir_leg_closure_ratio",
        "dir_nonleg_closure_ratio",
        "aggregate_transfer_score",
    )
    return {
        "gap_delta_candidate_minus_reference": {
            key: float(_safe_float(candidate.get(key)) - _safe_float(reference.get(key))) for key in gap_keys
        },
        "closure_delta_candidate_minus_reference": {
            key: float(_safe_float(candidate.get(key)) - _safe_float(reference.get(key))) for key in closure_keys
        },
    }


def _arm_module_groups(
    *,
    arm_name: str,
    baseline_bundle: Mapping[str, Any],
    e1_bundle: Mapping[str, Any],
    e2a_bundle: Mapping[str, Any],
) -> List[tuple[Mapping[str, Any], List[str]]]:
    if arm_name == "target-full7":
        return [(baseline_bundle, list(DIRECT_BRANCH_MODULES))]
    if arm_name == "E1-top3-full7":
        return [(e1_bundle, list(DIRECT_BRANCH_MODULES))]
    if arm_name == "E2A-R-full7":
        return [(e2a_bundle, list(DIRECT_BRANCH_MODULES))]
    if arm_name == "A1S2-mix-nonleg":
        return [
            (e1_bundle, list(E1_MIX_MODULES)),
            (e2a_bundle, list(E2A_MIX_MODULES)),
        ]
    raise KeyError(f"unknown arm: {arm_name}")


def _swap_modules(
    *,
    target_model: nn.Module,
    donor_groups: Sequence[tuple[Mapping[str, Any], Sequence[str]]],
) -> List[tuple[nn.Module, Dict[str, Any]]]:
    backups: List[tuple[nn.Module, Dict[str, Any]]] = []
    for donor_bundle, modules in donor_groups:
        if not modules:
            continue
        donor_model = donor_bundle["case"]["trainer"].model
        if donor_model is None:
            raise RuntimeError("donor model missing")
        backups.extend(
            _temporary_weight_swap(
                target_model=target_model,
                donor_model=donor_model,
                module_names=list(modules),
            )
        )
    return backups


def _guard_payload(
    *,
    name: str,
    base_arm: str,
    reference_arm: str,
    source_stats: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
    eps: float,
) -> Dict[str, Any]:
    mu_src = torch.tensor(source_stats["channel_mean_vector"], dtype=torch.float32)
    std_src = torch.tensor(source_stats["channel_std_vector"], dtype=torch.float32)
    mu_ref = torch.tensor(reference_stats["channel_mean_vector"], dtype=torch.float32)
    std_ref = torch.tensor(reference_stats["channel_std_vector"], dtype=torch.float32)
    scale = std_ref / (std_src + float(eps))
    return {
        "name": str(name),
        "base_arm": str(base_arm),
        "reference_arm": str(reference_arm),
        "mu_src": mu_src,
        "std_src": std_src,
        "mu_ref": mu_ref,
        "std_ref": std_ref,
        "eps": float(eps),
        "summary": {
            "base_arm": str(base_arm),
            "reference_arm": str(reference_arm),
            "formula": "x_hat = (x - mu_src) / (std_src + eps) * std_ref + mu_ref",
            "stats_scope": "Layer1 short-horizon rows from the same host / same entry / same offset",
            "stats_source_arm": str(base_arm),
            "stats_reference_arm": str(reference_arm),
            "stats_estimation_rows": int(source_stats["rows_total"]),
            "feature_dim": int(source_stats["feature_dim"]),
            "eps": float(eps),
            "source_low_variance_channels_le_eps": int(np.sum(np.asarray(source_stats["channel_std_vector"]) <= float(eps))),
            "reference_low_variance_channels_le_eps": int(
                np.sum(np.asarray(reference_stats["channel_std_vector"]) <= float(eps))
            ),
            "scale_summary": _summary_stats(scale.detach().cpu().numpy().tolist()),
            "source_rows_total": int(source_stats["rows_total"]),
            "reference_rows_total": int(reference_stats["rows_total"]),
        },
    }


def _apply_guard(x: torch.Tensor, guard: Mapping[str, Any]) -> torch.Tensor:
    dims = [1] * max(0, int(x.dim()) - 1)
    mu_src = guard["mu_src"].to(device=x.device, dtype=x.dtype).view(*dims, -1)
    std_src = guard["std_src"].to(device=x.device, dtype=x.dtype).view(*dims, -1)
    mu_ref = guard["mu_ref"].to(device=x.device, dtype=x.dtype).view(*dims, -1)
    std_ref = guard["std_ref"].to(device=x.device, dtype=x.dtype).view(*dims, -1)
    eps = float(guard["eps"])
    return ((x - mu_src) / (std_src + eps)) * std_ref + mu_ref


def _run_single_step_with_hook(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    donor_groups: Sequence[tuple[Mapping[str, Any], Sequence[str]]],
    guard: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model = host_bundle["case"]["trainer"].model
    if model is None:
        raise RuntimeError("host model missing")
    hook_name, first_linear = _first_linear(model)
    capture: Dict[str, torch.Tensor] = {}
    handle: Optional[Any] = None
    backups: List[tuple[nn.Module, Dict[str, Any]]] = []
    try:
        backups = _swap_modules(target_model=model, donor_groups=donor_groups)

        def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            x = inputs[0]
            capture["pre_input"] = x.detach().clone().cpu()
            x_use = _apply_guard(x, guard) if guard is not None else x
            capture["used_input"] = x_use.detach().clone().cpu()
            if x_use is x:
                return None
            if len(inputs) == 1:
                return (x_use,)
            return (x_use, *inputs[1:])

        handle = first_linear.register_forward_pre_hook(_pre_hook)
        result = _run_single_step(host_bundle, prep_host, fixed_contacts=fixed_contacts)
    finally:
        if handle is not None:
            handle.remove()
        if backups:
            _restore_weight_swap(list(reversed(backups)))

    result["hook_point"] = {
        "module_name": str(hook_name),
        "module_class": type(first_linear).__name__,
        "resolved_as": "direct_pose_head first Linear forward pre-hook input",
    }
    result["direct_head_input_pre"] = capture.get("pre_input")
    result["direct_head_input_used"] = capture.get("used_input", capture.get("pre_input"))
    return result


def _run_short_horizon_capture(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts_first: torch.Tensor,
    donor_groups: Sequence[tuple[Mapping[str, Any], Sequence[str]]],
    window_steps: int,
    guard: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model = host_bundle["case"]["trainer"].model
    if model is None:
        raise RuntimeError("host model missing")
    hook_name, first_linear = _first_linear(model)
    backups: List[tuple[nn.Module, Dict[str, Any]]] = []
    handle: Optional[Any] = None
    orig_prepare_contacts = posttrain._prepare_rollout_contacts_input
    hook_capture: Dict[str, Any] = {}
    contact_calls = {"count": 0}
    records: List[Dict[str, Any]] = []
    used_inputs: List[torch.Tensor] = []
    pre_inputs: List[torch.Tensor] = []
    try:
        backups = _swap_modules(target_model=model, donor_groups=donor_groups)

        def _prepare_contacts_override(
            trainer_: Any,
            model_: Any,
            *,
            motion_t: torch.Tensor,
            pose_hist_t: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if int(contact_calls["count"]) == 0:
                contact_calls["count"] += 1
                return fixed_contacts_first.detach().clone().to(device=motion_t.device, dtype=motion_t.dtype)
            contact_calls["count"] += 1
            return orig_prepare_contacts(trainer_, model_, motion_t=motion_t, pose_hist_t=pose_hist_t)

        def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            x = inputs[0]
            hook_capture["pre_input"] = x.detach().clone().cpu()
            x_use = _apply_guard(x, guard) if guard is not None else x
            hook_capture["used_input"] = x_use.detach().clone().cpu()
            if x_use is x:
                return None
            if len(inputs) == 1:
                return (x_use,)
            return (x_use, *inputs[1:])

        handle = first_linear.register_forward_pre_hook(_pre_hook)
        posttrain._prepare_rollout_contacts_input = _prepare_contacts_override

        ctx = _build_step_ctx(host_bundle, prep_host)
        total_steps = min(int(window_steps), int(ctx["runtime"]["total_steps"]))
        for step_idx in range(total_steps):
            hook_capture.clear()
            prev_lens = {
                "loss_terms": len(ctx["accum"]["loss_terms"]),
                "inc_terms": len(ctx["accum"]["inc_terms"]),
                "dir_base_terms": len(ctx["accum"]["dir_base_terms"]),
                "dir_leg_base_terms": len(ctx["accum"]["dir_leg_base_terms"]),
                "dir_nonleg_base_terms": len(ctx["accum"]["dir_nonleg_base_terms"]),
            }
            posttrain._lambda_rollout_unroll_single_step(t=int(step_idx), ctx=ctx)
            step_weight = float(ctx["data"]["step_weights"][step_idx].detach().cpu().item())
            pre_input = hook_capture.get("pre_input")
            used_input = hook_capture.get("used_input", pre_input)
            if not torch.is_tensor(pre_input) or not torch.is_tensor(used_input):
                raise RuntimeError(f"failed to capture direct_pose_head input at short-horizon step {step_idx}")
            pre_stats = _channel_stats(pre_input)
            used_stats = _channel_stats(used_input)
            pre_inputs.append(pre_input.detach().clone().cpu())
            used_inputs.append(used_input.detach().clone().cpu())
            records.append(
                {
                    "step": int(step_idx),
                    "step_weight": float(step_weight),
                    "input_shape": list(pre_stats["shape"]),
                    "pre_input_mean_abs": float(pre_stats["mean_abs"]),
                    "used_input_mean_abs": float(used_stats["mean_abs"]),
                    "pre_input_rms_norm_per_row": float(pre_stats["rms_norm_per_row"]),
                    "used_input_rms_norm_per_row": float(used_stats["rms_norm_per_row"]),
                    "pre_input_mean_vector": list(pre_stats["mean_vector"]),
                    "pre_input_std_vector": list(pre_stats["std_vector"]),
                    "used_input_mean_vector": list(used_stats["mean_vector"]),
                    "used_input_std_vector": list(used_stats["std_vector"]),
                    "blend_error": _extract_raw_step_metric(ctx["accum"], "loss_terms", prev_lens["loss_terms"], step_weight),
                    "inc_error": _extract_raw_step_metric(ctx["accum"], "inc_terms", prev_lens["inc_terms"], step_weight),
                    "dir_base_error": _extract_raw_step_metric(
                        ctx["accum"], "dir_base_terms", prev_lens["dir_base_terms"], step_weight
                    ),
                    "dir_leg_error": _extract_raw_step_metric(
                        ctx["accum"], "dir_leg_base_terms", prev_lens["dir_leg_base_terms"], step_weight
                    ),
                    "dir_nonleg_error": _extract_raw_step_metric(
                        ctx["accum"], "dir_nonleg_base_terms", prev_lens["dir_nonleg_base_terms"], step_weight
                    ),
                    "fixed_contacts_override_applied": bool(step_idx == 0),
                }
            )
    finally:
        if handle is not None:
            handle.remove()
        posttrain._prepare_rollout_contacts_input = orig_prepare_contacts
        if backups:
            _restore_weight_swap(list(reversed(backups)))

    all_rows = torch.cat([_stack_rows(x) for x in used_inputs], dim=0)
    global_mean = all_rows.mean(dim=0)
    global_std = all_rows.std(dim=0, unbiased=False)
    return {
        "hook_point": {
            "module_name": str(hook_name),
            "module_class": type(first_linear).__name__,
            "resolved_as": "direct_pose_head first Linear forward pre-hook input",
        },
        "window_steps": int(len(records)),
        "fixed_contacts_mode": "baseline replace native contacts_in_t only on rollout first forward",
        "step_records": records,
        "global_stats": {
            "rows_total": int(all_rows.shape[0]),
            "feature_dim": int(all_rows.shape[-1]),
            "channel_mean_vector": global_mean.tolist(),
            "channel_std_vector": global_std.tolist(),
            "channel_mean_abs_summary": _summary_stats(all_rows.abs().mean(dim=0).tolist()),
            "channel_std_summary": _summary_stats(global_std.tolist()),
            "rms_norm_per_row_summary": _summary_stats(
                (
                    torch.linalg.vector_norm(all_rows, dim=-1) / math.sqrt(max(1, int(all_rows.shape[-1])))
                ).tolist()
            ),
        },
        "_used_inputs": used_inputs,
        "_pre_inputs": pre_inputs,
    }


def _pairwise_table_from_step_results(
    arm_to_inputs: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    keys = list(arm_to_inputs.keys())
    for a in keys:
        out[a] = {}
        for b in keys:
            out[a][b] = {
                "l2_rms": _norm_l2(arm_to_inputs[a], arm_to_inputs[b]),
                "cosine_similarity": _cosine_similarity(arm_to_inputs[a], arm_to_inputs[b]),
                "cosine_distance": _cosine_distance(arm_to_inputs[a], arm_to_inputs[b]),
            }
    return out


def _pairwise_curves(
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
) -> List[Dict[str, float]]:
    a_inputs = run_a["_used_inputs"]
    b_inputs = run_b["_used_inputs"]
    steps = min(len(a_inputs), len(b_inputs))
    rows: List[Dict[str, float]] = []
    for step_idx in range(steps):
        a_t = a_inputs[step_idx]
        b_t = b_inputs[step_idx]
        rows.append(
            {
                "step": int(step_idx),
                "l2_rms": _norm_l2(a_t, b_t),
                "cosine_similarity": _cosine_similarity(a_t, b_t),
                "cosine_distance": _cosine_distance(a_t, b_t),
            }
        )
    return rows


def _curve_summary(curve: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    l2s = [_safe_float(row.get("l2_rms")) for row in curve]
    cos_d = [_safe_float(row.get("cosine_distance")) for row in curve]
    return {
        "steps": int(len(curve)),
        "l2_rms": _summary_stats(l2s),
        "cosine_distance": _summary_stats(cos_d),
    }


def _reference_divergence_payload(
    *,
    arm_name: str,
    arm_run: Mapping[str, Any],
    ref_name: str,
    ref_run: Mapping[str, Any],
) -> Dict[str, Any]:
    curve = _pairwise_curves(arm_run, ref_run)
    return {
        "arm": str(arm_name),
        "reference": str(ref_name),
        "curve": curve,
        "aggregate": _curve_summary(curve),
    }


def _correlation_payload(
    *,
    arm_run: Mapping[str, Any],
    divergence_curve: Sequence[Mapping[str, Any]],
    reference_name: str,
) -> Dict[str, Any]:
    step_rows = arm_run["step_records"]
    l2_curve = [_safe_float(row.get("l2_rms")) for row in divergence_curve]
    cos_curve = [_safe_float(row.get("cosine_distance")) for row in divergence_curve]
    dir_leg = [_safe_float(row.get("dir_leg_error")) for row in step_rows[: len(divergence_curve)]]
    dir_nonleg = [_safe_float(row.get("dir_nonleg_error")) for row in step_rows[: len(divergence_curve)]]
    dir_base = [_safe_float(row.get("dir_base_error")) for row in step_rows[: len(divergence_curve)]]
    return {
        "reference": str(reference_name),
        "divergence_metric_definition": "direct_pose_head first-linear used-input flattened per-step activation",
        "step_error_definition": (
            "per-step raw direct errors recovered from posttrain accum terms divided by rollout step weight "
            "(dir_leg_error / dir_nonleg_error / dir_base_error)"
        ),
        "pearson": {
            "l2_rms_vs_dir_leg_error": _pearson(l2_curve, dir_leg),
            "l2_rms_vs_dir_nonleg_error": _pearson(l2_curve, dir_nonleg),
            "l2_rms_vs_dir_base_error": _pearson(l2_curve, dir_base),
            "cosine_distance_vs_dir_leg_error": _pearson(cos_curve, dir_leg),
            "cosine_distance_vs_dir_nonleg_error": _pearson(cos_curve, dir_nonleg),
            "cosine_distance_vs_dir_base_error": _pearson(cos_curve, dir_base),
        },
    }


def _mean_pairwise_to_refs(
    *,
    arm_name: str,
    layer1_divergence: Mapping[str, Any],
    refs: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ref in refs:
        payload = ((layer1_divergence.get(arm_name) or {}).get(ref) or {})
        agg = payload.get("aggregate") or {}
        out[str(ref)] = float(_safe_float(((agg.get("l2_rms") or {}).get("mean"))))
    return out


def _layer2_short_horizon_delta(
    *,
    base_run: Mapping[str, Any],
    guarded_run: Mapping[str, Any],
    reference_run: Mapping[str, Any],
) -> Dict[str, Any]:
    base_curve = _pairwise_curves(base_run, reference_run)
    guarded_curve = _pairwise_curves(guarded_run, reference_run)
    steps = min(len(base_curve), len(guarded_curve))
    deltas: List[Dict[str, Any]] = []
    for idx in range(steps):
        deltas.append(
            {
                "step": int(idx),
                "l2_rms_delta_guarded_minus_base": float(
                    _safe_float(guarded_curve[idx]["l2_rms"]) - _safe_float(base_curve[idx]["l2_rms"])
                ),
                "cosine_distance_delta_guarded_minus_base": float(
                    _safe_float(guarded_curve[idx]["cosine_distance"]) - _safe_float(base_curve[idx]["cosine_distance"])
                ),
            }
        )
    return {
        "base_vs_reference": {"curve": base_curve, "aggregate": _curve_summary(base_curve)},
        "guarded_vs_reference": {"curve": guarded_curve, "aggregate": _curve_summary(guarded_curve)},
        "delta_curve_guarded_minus_base": deltas,
        "delta_summary": {
            "l2_rms_delta_guarded_minus_base": _summary_stats(
                [row["l2_rms_delta_guarded_minus_base"] for row in deltas]
            ),
            "cosine_distance_delta_guarded_minus_base": _summary_stats(
                [row["cosine_distance_delta_guarded_minus_base"] for row in deltas]
            ),
        },
    }


def _transfer_improvement_status(
    *,
    guarded_transfer: Mapping[str, Any],
    base_transfer: Mapping[str, Any],
) -> str:
    agg_delta = float(_safe_float(guarded_transfer.get("aggregate_transfer_score")) - _safe_float(base_transfer.get("aggregate_transfer_score")))
    if agg_delta > CLEAR_AGG_MARGIN:
        return "clear_win"
    if agg_delta > 0.0:
        return "partial_positive"
    if agg_delta < -0.01:
        return "negative"
    return "flat"


def _judge_guard_direction(
    *,
    layer0_pairwise: Mapping[str, Any],
    layer1_divergence: Mapping[str, Any],
    guarded_transfer: Mapping[str, Any],
    mix_transfer: Mapping[str, Any],
) -> Dict[str, Any]:
    layer0_l2: List[float] = []
    layer0_cos: List[float] = []
    for left in layer0_pairwise.values():
        for payload in left.values():
            layer0_l2.append(_safe_float(payload.get("l2_rms")))
            layer0_cos.append(_safe_float(payload.get("cosine_distance")))
    max_layer0_l2 = max([v for v in layer0_l2 if math.isfinite(v)], default=float("nan"))
    max_layer0_cos = max([v for v in layer0_cos if math.isfinite(v)], default=float("nan"))

    a1s2_to_e1 = float(
        _safe_float((((layer1_divergence.get("A1S2-mix-nonleg") or {}).get("E1-top3-full7") or {}).get("aggregate") or {}).get("l2_rms", {}).get("mean"))
    )
    a1s2_to_target = float(
        _safe_float((((layer1_divergence.get("A1S2-mix-nonleg") or {}).get("target-full7") or {}).get("aggregate") or {}).get("l2_rms", {}).get("mean"))
    )
    e2a_to_e1 = float(
        _safe_float((((layer1_divergence.get("E2A-R-full7") or {}).get("E1-top3-full7") or {}).get("aggregate") or {}).get("l2_rms", {}).get("mean"))
    )
    divergence_present = bool(
        (
            math.isfinite(a1s2_to_e1)
            and math.isfinite(max_layer0_l2)
            and a1s2_to_e1 > max(max_layer0_l2 * 2.0, LAYER0_NONDISC_L2)
        )
        or (
            math.isfinite(a1s2_to_target)
            and math.isfinite(max_layer0_l2)
            and a1s2_to_target > max(max_layer0_l2 * 2.0, LAYER0_NONDISC_L2)
        )
        or (
            math.isfinite(e2a_to_e1)
            and math.isfinite(max_layer0_l2)
            and e2a_to_e1 > max(max_layer0_l2 * 2.0, LAYER0_NONDISC_L2)
        )
    )
    improvement_status = _transfer_improvement_status(guarded_transfer=guarded_transfer, base_transfer=mix_transfer)
    if divergence_present and improvement_status == "clear_win":
        case_label = "Case A"
        interpretation = "input-side distribution shift is useful signal and simplest affine guard already shows clear positive transfer signal"
        recommend_mainline = True
        next_step = "stronger affine guard or learned lightweight adapter"
    elif divergence_present and improvement_status == "partial_positive":
        case_label = "Case B"
        interpretation = "direction looks real, but first/second-order moment matching is not enough for a decisive win"
        recommend_mainline = False
        next_step = "learned affine adapter or finer channel/group-wise guard"
    elif divergence_present:
        case_label = "Case C"
        interpretation = "divergence exists, but pure moment matching at direct_pose_head input has low ceiling here"
        recommend_mainline = False
        next_step = "learned adapter or training-side / earlier-boundary constraint"
    else:
        case_label = "Case D"
        interpretation = "direct_pose_head input is not the strongest intervention lever under this assay"
        recommend_mainline = False
        next_step = "head-internal/downstream work, learned adapter, or earlier boundary"
    return {
        "layer0_max_l2_rms": float(max_layer0_l2),
        "layer0_max_cosine_distance": float(max_layer0_cos),
        "short_horizon_divergence_present": bool(divergence_present),
        "guard_transfer_status_vs_A1S2_mix": str(improvement_status),
        "case_label": str(case_label),
        "interpretation": str(interpretation),
        "recommend_input_side_boundary_guard_as_mainline": bool(recommend_mainline),
        "next_step": str(next_step),
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_record(summary: Mapping[str, Any]) -> None:
    layer0 = summary["layer0_sanity"]
    layer1 = summary["layer1_short_horizon"]
    layer2 = summary["layer2_moment_matching"]

    layer0_rows: List[List[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, row in (layer0.get("pairwise_table") or {}).items():
        for right, payload in row.items():
            pair = tuple(sorted((str(left), str(right))))
            if left == right or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            layer0_rows.append(
                [
                    str(left),
                    str(right),
                    _fmt(payload.get("l2_rms")),
                    _fmt(payload.get("cosine_distance")),
                ]
            )

    layer1_rows: List[List[str]] = []
    layer1_pairs = layer1.get("pairwise_aggregate") or {}
    for pair_name, payload in layer1_pairs.items():
        layer1_rows.append(
            [
                str(pair_name),
                _fmt(((payload.get("l2_rms") or {}).get("mean"))),
                _fmt(((payload.get("cosine_distance") or {}).get("mean"))),
                _fmt(((payload.get("l2_rms") or {}).get("max"))),
            ]
        )

    corr_rows: List[List[str]] = []
    for arm_name, per_ref in (layer1.get("step_level_correlations") or {}).items():
        for ref_name, payload in per_ref.items():
            pearson = payload.get("pearson") or {}
            corr_rows.append(
                [
                    str(arm_name),
                    str(ref_name),
                    _fmt(pearson.get("l2_rms_vs_dir_leg_error")),
                    _fmt(pearson.get("l2_rms_vs_dir_nonleg_error")),
                    _fmt(pearson.get("cosine_distance_vs_dir_leg_error")),
                    _fmt(pearson.get("cosine_distance_vs_dir_nonleg_error")),
                ]
            )

    transfer_rows = []
    base_transfer = summary["reused_references"]["A1S2_mix_nonleg"]["transfer"]
    e1_transfer = summary["reused_references"]["E1_top3_full7"]["transfer"]
    e2a_transfer = summary["reused_references"]["E2A_R_full7"]["transfer"]
    target_transfer = summary["reused_references"]["transplant_compatible_target"]["transfer"]
    guard_transfer = layer2["fixed_transfer"]["A1S4-mm-E1ref-on-A1S2mix"]["transfer"]
    for label, transfer in (
        ("target-full7", target_transfer),
        ("E1-top3-full7", e1_transfer),
        ("E2A-R-full7", e2a_transfer),
        ("A1S2-mix-nonleg", base_transfer),
        ("A1S4-mm-E1ref-on-A1S2mix", guard_transfer),
    ):
        transfer_rows.append(
            [
                str(label),
                _fmt(transfer.get("out_direct_gap")),
                _fmt(transfer.get("dir_base_gap")),
                _fmt(transfer.get("dir_leg_gap")),
                _fmt(transfer.get("dir_nonleg_gap")),
                _fmt(transfer.get("aggregate_transfer_score")),
            ]
        )

    judgement = summary["judgement"]
    explicit = summary["explicit_answers"]
    mm_cfg = layer2["moment_matching_config"]["A1S4-mm-E1ref-on-A1S2mix"]

    lines: List[str] = []
    lines.append("# 2026-04-09 Direct Head Input Guard A1-S4")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("- Scope: A1-S4 direct_pose_head input merged diagnostic + moment-matching affine guard, no-train.")
    lines.append("- Host assay remains fixed to `coadapt_allrot_interface_bestlr_longer_4x_20260406`, offset `45`, same entry, same teacher clip.")
    lines.append("- Fixed first-forward contacts remain baseline replace native same-entry `contacts_in_t`.")
    for item in summary["inherited_conclusions"]:
        lines.append(f"- Inherited mainline: {item}.")
    for item in summary["a1s1_inherited"]["direct_conclusions"]:
        lines.append(f"- A1-S1 inherit: {item}.")
    for item in summary["a1s2_inherited"]["direct_conclusions"]:
        lines.append(f"- A1-S2 inherit: {item}.")
    for item in summary["a1s3_inherited"]["direct_conclusions"]:
        lines.append(f"- A1-S3 inherit: {item}.")
    lines.append("")
    lines.append("## 2. Why A1-S4 after A1-S3")
    lines.append("")
    lines.append("- A1-S3 still did not beat `A1S2-mix-nonleg`, so the next minimal lever is a stronger boundary guard at the earliest usable anchor.")
    lines.append("- This round stays at `direct_pose_head` input only, with runtime affine guard and no learned adapter / no retraining.")
    lines.append("")
    lines.append("## 3. Host / donor / target inventory")
    lines.append("")
    lines.append(f"- Host ckpt: `{summary['host']['ckpt']}`.")
    lines.append(f"- E1 donor ckpt: `{summary['anchor_donor']['ckpt']}`.")
    lines.append(f"- E2A donor ckpt: `{summary['expansion_donor']['ckpt']}`.")
    lines.append(f"- Baseline replace donor ckpt: `{summary['baseline_replace']['ckpt']}`.")
    lines.append("")
    lines.append("## 4. Hook point definition")
    lines.append("")
    lines.append(f"- Resolved hook: `{summary['hook_point']['module_name']}` ({summary['hook_point']['module_class']}).")
    lines.append("- Meaning: `direct_pose_head` first Linear forward pre-hook input activation, not output / not weights / not trunk hidden.")
    lines.append("")
    lines.append("## 5. Layer 0 sanity table")
    lines.append("")
    lines.append(_markdown_table(["left", "right", "l2_rms", "cosine_distance"], layer0_rows))
    lines.append("")
    lines.append(
        f"- Call: {layer0['interpretation']['call']} (max l2=`{_fmt(layer0['interpretation']['max_pairwise_l2_rms'])}`, "
        f"max cosdist=`{_fmt(layer0['interpretation']['max_pairwise_cosine_distance'])}`)."
    )
    lines.append("")
    lines.append("## 6. Layer 1 divergence table")
    lines.append("")
    lines.append(_markdown_table(["pair", "mean_l2_rms", "mean_cosdist", "max_l2_rms"], layer1_rows))
    lines.append("")
    lines.append("- `A1S2-mix-nonleg` mean l2 to refs:")
    for ref_name, val in (layer1["a1s2_mean_l2_to_refs"] or {}).items():
        lines.append(f"  - vs {ref_name}: `{_fmt(val)}`")
    lines.append("")
    lines.append("## 7. Step-level correlation summary")
    lines.append("")
    lines.append(
        _markdown_table(
            ["arm", "reference", "l2~dir_leg", "l2~dir_nonleg", "cos~dir_leg", "cos~dir_nonleg"],
            corr_rows,
        )
    )
    lines.append("")
    lines.append("## 8. Moment-matching transform definition")
    lines.append("")
    lines.append(f"- Base arm: `{mm_cfg.get('base_arm') or mm_cfg.get('stats_source_arm')}`.")
    lines.append(f"- Reference stats arm: `{mm_cfg.get('reference_arm') or mm_cfg.get('stats_reference_arm')}`.")
    lines.append(f"- Formula: `{mm_cfg['formula']}`.")
    lines.append(f"- Estimation rows: `{mm_cfg['stats_estimation_rows']}`, feature_dim=`{mm_cfg['feature_dim']}`, eps=`{_fmt(mm_cfg['eps'])}`.")
    lines.append(
        f"- Scale summary mean=`{_fmt(mm_cfg['scale_summary']['mean'])}`, p90=`{_fmt(mm_cfg['scale_summary']['p90'])}`, max=`{_fmt(mm_cfg['scale_summary']['max'])}`."
    )
    lines.append("")
    lines.append("## 9. Layer 2 fixed transfer assay table")
    lines.append("")
    lines.append(
        _markdown_table(
            ["arm", "out_gap", "dir_base_gap", "dir_leg_gap", "dir_nonleg_gap", "agg_score"],
            transfer_rows,
        )
    )
    lines.append("")
    lines.append(
        f"- `A1S4-mm-E1ref-on-A1S2mix` minus `A1S2-mix-nonleg` aggregate = "
        f"`{_fmt(layer2['fixed_transfer']['A1S4-mm-E1ref-on-A1S2mix']['delta_vs_A1S2_mix_nonleg']['closure_delta_candidate_minus_reference']['aggregate_transfer_score'])}`."
    )
    lines.append("")
    lines.append("## 10. Boundary-guard interpretation")
    lines.append("")
    lines.append(f"- Case: `{judgement['case_label']}`.")
    lines.append(f"- Interpretation: {judgement['interpretation']}.")
    lines.append(
        f"- Input-side boundary guard mainline? `{str(judgement['recommend_input_side_boundary_guard_as_mainline']).lower()}`."
    )
    lines.append("")
    lines.append("## 11. Next-step recommendation")
    lines.append("")
    lines.append(f"- Recommended next step: {judgement['next_step']}.")
    lines.append(f"- Q1: {explicit['q1_same_host_fixed_t0_non_discriminative']['answer']}.")
    lines.append(f"- Q2: {explicit['q2_a1s2_short_horizon_closest_reference']['answer']}.")
    lines.append(f"- Q3: {explicit['q3_guard_clear_transfer_win_over_a1s2']['answer']}.")
    lines.append(f"- Q4: {explicit['q4_input_side_boundary_guard_mainline']['answer']}.")
    lines.append(f"- Q5: {explicit['q5_next_step_priority']['answer']}.")
    lines.append("")

    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    a1s1_summary = _load_json(A1S1_SUMMARY_JSON)
    a1s2_summary = _load_json(A1S2_SUMMARY_JSON)
    a1s3_summary = _load_json(A1S3_SUMMARY_JSON)
    reused_refs = dict(a1s2_summary.get("reused_references") or {})
    host_native_reference = dict((reused_refs.get("host_native_bad_reference") or {}).get("transfer") or {})
    target_transfer_reference = dict((reused_refs.get("transplant_compatible_target") or {}).get("transfer") or {})
    e1_top3_reference = dict((reused_refs.get("E1_top3_full7") or {}).get("transfer") or {})
    e2a_full7_reference = dict((reused_refs.get("E2A_R_full7") or {}).get("transfer") or {})
    a1s2_mix_transfer = dict(
        ((a1s2_summary.get("mixed_assays") or {}).get("A1S2-mix-nonleg") or {}).get("transfer") or {}
    )

    teacher = DEFAULT_TEACHER.resolve()
    baseline_bundle = _case_bundle(
        case_name="baseline_replace",
        ckpt_path=BASELINE_REPLACE_CKPT,
        eval_json_path=BASELINE_REPLACE_EVAL,
        teacher_path=teacher,
        config_path=BASELINE_REPLACE_CONFIG,
        device_pref="cpu",
    )
    host_bundle = _case_bundle(
        case_name="coadapt_host",
        ckpt_path=COADAPT_HOST_CKPT,
        eval_json_path=COADAPT_HOST_EVAL,
        teacher_path=teacher,
        config_path=COADAPT_HOST_CONFIG,
        device_pref="cpu",
    )
    e1_bundle = _case_bundle(
        case_name="E1_top3_anchor_donor",
        ckpt_path=TOP3_70A_CKPT,
        eval_json_path=TOP3_70A_EVAL,
        teacher_path=teacher,
        config_path=STAGE70A_CONFIG,
        device_pref="cpu",
    )
    e2a_bundle = _case_bundle(
        case_name="E2A_R_expansion_donor",
        ckpt_path=E2A_70A_CKPT,
        eval_json_path=E2A_70A_EVAL,
        teacher_path=teacher,
        config_path=STAGE70A_CONFIG,
        device_pref="cpu",
    )

    prep_base = _prepare_fixed_offset_context(baseline_bundle, offset=DEFAULT_OFFSET)
    prep_host = _prepare_fixed_offset_context(host_bundle, offset=DEFAULT_OFFSET)

    baseline_native = _run_single_step(baseline_bundle, prep_base, fixed_contacts=None)
    fixed_contacts = baseline_native["inputs"]["contacts"]
    if not torch.is_tensor(fixed_contacts):
        raise RuntimeError("failed to materialize baseline native fixed contacts")

    target_result = _run_single_step(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        weight_swap_modules=DIRECT_BRANCH_MODULES,
        donor_bundle=baseline_bundle,
    )

    host_model = host_bundle["case"]["trainer"].model
    if host_model is None:
        raise RuntimeError("host model missing")
    hook_module_name, hook_linear = _first_linear(host_model)
    hook_point_definition = {
        "module_name": str(hook_module_name),
        "module_class": type(hook_linear).__name__,
        "resolved_definition": "direct_pose_head first Linear forward pre-hook input",
        "not_this": [
            "not direct_pose_head output",
            "not trunk_hidden",
            "not direct_pose_head.0 weight statistics",
        ],
    }

    layer0_runs: Dict[str, Any] = {}
    for arm_name in DIAGNOSTIC_ARM_ORDER:
        donor_groups = _arm_module_groups(
            arm_name=arm_name,
            baseline_bundle=baseline_bundle,
            e1_bundle=e1_bundle,
            e2a_bundle=e2a_bundle,
        )
        layer0_runs[arm_name] = _run_single_step_with_hook(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            donor_groups=donor_groups,
            guard=None,
        )

    layer0_pairwise = _pairwise_table_from_step_results(
        {
            arm_name: arm_result["direct_head_input_used"]
            for arm_name, arm_result in layer0_runs.items()
        }
    )
    layer0_max_l2 = max(
        (
            _safe_float(payload.get("l2_rms"))
            for row in layer0_pairwise.values()
            for payload in row.values()
            if math.isfinite(_safe_float(payload.get("l2_rms")))
        ),
        default=float("nan"),
    )
    layer0_max_cos = max(
        (
            _safe_float(payload.get("cosine_distance"))
            for row in layer0_pairwise.values()
            for payload in row.values()
            if math.isfinite(_safe_float(payload.get("cosine_distance")))
        ),
        default=float("nan"),
    )
    layer0_non_discriminative = bool(
        math.isfinite(layer0_max_l2)
        and math.isfinite(layer0_max_cos)
        and layer0_max_l2 <= LAYER0_NONDISC_L2
        and layer0_max_cos <= LAYER0_NONDISC_COS
    )

    layer1_runs: Dict[str, Any] = {}
    for arm_name in DIAGNOSTIC_ARM_ORDER:
        donor_groups = _arm_module_groups(
            arm_name=arm_name,
            baseline_bundle=baseline_bundle,
            e1_bundle=e1_bundle,
            e2a_bundle=e2a_bundle,
        )
        layer1_runs[arm_name] = _run_short_horizon_capture(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts_first=fixed_contacts,
            donor_groups=donor_groups,
            window_steps=SHORT_HORIZON_STEPS,
            guard=None,
        )

    layer1_pairwise_by_step: Dict[str, Any] = {}
    layer1_pairwise_aggregate: Dict[str, Any] = {}
    for idx, left in enumerate(DIAGNOSTIC_ARM_ORDER):
        for right in DIAGNOSTIC_ARM_ORDER[idx + 1 :]:
            pair_name = f"{left}__vs__{right}"
            curve = _pairwise_curves(layer1_runs[left], layer1_runs[right])
            layer1_pairwise_by_step[pair_name] = curve
            layer1_pairwise_aggregate[pair_name] = _curve_summary(curve)

    layer1_divergence: Dict[str, Any] = {}
    layer1_correlations: Dict[str, Any] = {}
    for arm_name in ("E2A-R-full7", "A1S2-mix-nonleg"):
        layer1_divergence[arm_name] = {}
        layer1_correlations[arm_name] = {}
        for ref_name in ("E1-top3-full7", "target-full7"):
            payload = _reference_divergence_payload(
                arm_name=arm_name,
                arm_run=layer1_runs[arm_name],
                ref_name=ref_name,
                ref_run=layer1_runs[ref_name],
            )
            layer1_divergence[arm_name][ref_name] = payload
            layer1_correlations[arm_name][ref_name] = _correlation_payload(
                arm_run=layer1_runs[arm_name],
                divergence_curve=payload["curve"],
                reference_name=ref_name,
            )

    a1s2_stats = layer1_runs["A1S2-mix-nonleg"]["global_stats"]
    e1_stats = layer1_runs["E1-top3-full7"]["global_stats"]
    target_stats = layer1_runs["target-full7"]["global_stats"]
    mm_e1_guard = _guard_payload(
        name="A1S4-mm-E1ref-on-A1S2mix",
        base_arm="A1S2-mix-nonleg",
        reference_arm="E1-top3-full7",
        source_stats=a1s2_stats,
        reference_stats=e1_stats,
        eps=MOMENT_MATCH_EPS,
    )

    guard_single = _run_single_step_with_hook(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        donor_groups=_arm_module_groups(
            arm_name="A1S2-mix-nonleg",
            baseline_bundle=baseline_bundle,
            e1_bundle=e1_bundle,
            e2a_bundle=e2a_bundle,
        ),
        guard=mm_e1_guard,
    )
    guard_transfer = _add_closure(
        _tensor_metric_gaps(
            host_case=host_bundle["case"],
            target_result=target_result,
            candidate_result=guard_single,
        ),
        host_native_reference,
    )

    guard_short = _run_short_horizon_capture(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts_first=fixed_contacts,
        donor_groups=_arm_module_groups(
            arm_name="A1S2-mix-nonleg",
            baseline_bundle=baseline_bundle,
            e1_bundle=e1_bundle,
            e2a_bundle=e2a_bundle,
        ),
        window_steps=SHORT_HORIZON_STEPS,
        guard=mm_e1_guard,
    )

    layer2_divergence = {
        "vs_E1_top3_full7": _layer2_short_horizon_delta(
            base_run=layer1_runs["A1S2-mix-nonleg"],
            guarded_run=guard_short,
            reference_run=layer1_runs["E1-top3-full7"],
        ),
        "vs_target_full7": _layer2_short_horizon_delta(
            base_run=layer1_runs["A1S2-mix-nonleg"],
            guarded_run=guard_short,
            reference_run=layer1_runs["target-full7"],
        ),
    }
    layer2_correlations = {
        "E1-top3-full7": _correlation_payload(
            arm_run=guard_short,
            divergence_curve=layer2_divergence["vs_E1_top3_full7"]["guarded_vs_reference"]["curve"],
            reference_name="E1-top3-full7",
        ),
        "target-full7": _correlation_payload(
            arm_run=guard_short,
            divergence_curve=layer2_divergence["vs_target_full7"]["guarded_vs_reference"]["curve"],
            reference_name="target-full7",
        ),
    }

    layer1_a1s2_mean_l2_to_refs = {
        "E1-top3-full7": float(
            _safe_float(
                ((layer1_divergence["A1S2-mix-nonleg"]["E1-top3-full7"]["aggregate"]["l2_rms"] or {}).get("mean"))
            )
        ),
        "target-full7": float(
            _safe_float(
                ((layer1_divergence["A1S2-mix-nonleg"]["target-full7"]["aggregate"]["l2_rms"] or {}).get("mean"))
            )
        ),
        "E2A-R-full7": float(
            _safe_float(_curve_summary(_pairwise_curves(layer1_runs["A1S2-mix-nonleg"], layer1_runs["E2A-R-full7"]))["l2_rms"]["mean"])
        ),
    }

    finite_ref_rows = [
        (str(name), float(_safe_float(val)))
        for name, val in layer1_a1s2_mean_l2_to_refs.items()
        if math.isfinite(float(_safe_float(val)))
    ]
    if not finite_ref_rows:
        closest_ref = "unresolved"
    else:
        min_val = min(val for _, val in finite_ref_rows)
        tied = [name for name, val in finite_ref_rows if abs(val - min_val) <= 1e-9]
        closest_ref = (
            tied[0]
            if len(tied) == 1
            else "tie:" + ",".join(sorted(tied))
        )

    judgement = _judge_guard_direction(
        layer0_pairwise=layer0_pairwise,
        layer1_divergence=layer1_divergence,
        guarded_transfer=guard_transfer,
        mix_transfer=a1s2_mix_transfer,
    )

    aggregate_delta_vs_a1s2 = float(
        _safe_float(guard_transfer.get("aggregate_transfer_score")) - _safe_float(a1s2_mix_transfer.get("aggregate_transfer_score"))
    )
    explicit_answers = {
        "q1_same_host_fixed_t0_non_discriminative": {
            "answer": "yes" if layer0_non_discriminative else "no",
            "reason": (
                "same-host fixed t=0 pairwise divergence stays very small, so this view is close to non-discriminative"
                if layer0_non_discriminative
                else "same-host fixed t=0 already shows material pairwise spread"
            ),
            "max_pairwise_l2_rms": float(layer0_max_l2),
            "max_pairwise_cosine_distance": float(layer0_max_cos),
        },
        "q2_a1s2_short_horizon_closest_reference": {
            "answer": str(closest_ref),
            "mean_l2_to_E1_top3_full7": float(layer1_a1s2_mean_l2_to_refs["E1-top3-full7"]),
            "mean_l2_to_E2A_R_full7": float(layer1_a1s2_mean_l2_to_refs["E2A-R-full7"]),
            "mean_l2_to_target_full7": float(layer1_a1s2_mean_l2_to_refs["target-full7"]),
        },
        "q3_guard_clear_transfer_win_over_a1s2": {
            "answer": "yes" if aggregate_delta_vs_a1s2 > CLEAR_AGG_MARGIN else "no",
            "aggregate_delta_vs_A1S2_mix_nonleg": float(aggregate_delta_vs_a1s2),
            "status": str(judgement["guard_transfer_status_vs_A1S2_mix"]),
        },
        "q4_input_side_boundary_guard_mainline": {
            "answer": "yes" if bool(judgement["recommend_input_side_boundary_guard_as_mainline"]) else "no",
            "case_label": str(judgement["case_label"]),
        },
        "q5_next_step_priority": {
            "answer": str(judgement["next_step"]),
            "case_label": str(judgement["case_label"]),
        },
    }

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "A1-S4 direct-head-input guard assay (probe + moment matching, no-train)",
            "mode": "merged same-hook diagnostic + runtime affine boundary-guard assay",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "offset": DEFAULT_OFFSET,
            "teacher": str(teacher),
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "layer0_mode": "same-host fixed-contacts deterministic first-forward t=0 sanity diagnostic",
            "layer1_mode": "same-host same-entry same-offset short-horizon rollout diagnostic",
            "layer2_mode": "runtime pre-hook per-channel moment-matching affine guard only",
            "short_horizon_steps": SHORT_HORIZON_STEPS,
            "constraints": [
                "no new training",
                "no learned adapter",
                "no architecture redesign",
                "no E0/E1/E2-A/E2-C/E3-A/A1-S1/A1-S2/A1-S3 reruns",
                "no full grid expansion",
            ],
            "goal": (
                "test whether direct_pose_head input carries usable distribution-shift signal under short horizon, "
                "and whether the simplest affine moment-matching guard can move A1S2-mix-nonleg toward the "
                "E1-top3 / target replace-compatible basin"
            ),
        },
        "inherited_conclusions": INHERITED_CONCLUSIONS,
        "a1s1_inherited": {
            "summary_json": str(A1S1_SUMMARY_JSON),
            "record_md": str(A1S1_RECORD_MD),
            "direct_conclusions": A1S1_DIRECT_INHERITED,
            "boundary_interpretation": dict(a1s1_summary.get("boundary_interpretation") or {}),
        },
        "a1s2_inherited": {
            "summary_json": str(A1S2_SUMMARY_JSON),
            "record_md": str(A1S2_RECORD_MD),
            "direct_conclusions": A1S2_DIRECT_INHERITED,
            "judgement": dict(a1s2_summary.get("judgement") or {}),
        },
        "a1s3_inherited": {
            "summary_json": str(A1S3_SUMMARY_JSON),
            "record_md": str(A1S3_RECORD_MD),
            "direct_conclusions": A1S3_DIRECT_INHERITED,
            "judgement": dict(a1s3_summary.get("judgement") or {}),
        },
        "host": {
            "label": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "ckpt": str(COADAPT_HOST_CKPT),
            "config": str(COADAPT_HOST_CONFIG),
            "eval_json": str(COADAPT_HOST_EVAL),
        },
        "anchor_donor": {
            "label": "E1-top3 final70a",
            "ckpt": str(TOP3_70A_CKPT),
            "config": str(STAGE70A_CONFIG),
            "eval_json": str(TOP3_70A_EVAL),
        },
        "expansion_donor": {
            "label": "E2A-R final70a",
            "ckpt": str(E2A_70A_CKPT),
            "config": str(STAGE70A_CONFIG),
            "eval_json": str(E2A_70A_EVAL),
        },
        "baseline_replace": {
            "label": "baseline replace donor for transplant-compatible target",
            "ckpt": str(BASELINE_REPLACE_CKPT),
            "config": str(BASELINE_REPLACE_CONFIG),
            "eval_json": str(BASELINE_REPLACE_EVAL),
        },
        "hook_point": hook_point_definition,
        "reused_references": {
            "host_native_bad_reference": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": host_native_reference,
            },
            "transplant_compatible_target": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": target_transfer_reference,
            },
            "E1_top3_full7": dict(reused_refs.get("E1_top3_full7") or {}),
            "E2A_R_full7": dict(reused_refs.get("E2A_R_full7") or {}),
            "A1S2_mix_nonleg": {
                "source": str(A1S2_SUMMARY_JSON),
                "transfer": a1s2_mix_transfer,
            },
        },
        "layer0_sanity": {
            "arms": {
                arm_name: {
                    "hook_point": dict(layer0_runs[arm_name]["hook_point"]),
                    "input_shape": _tensor_shape(layer0_runs[arm_name]["direct_head_input_used"]),
                    "used_input_mean_abs": float(
                        _stack_rows(layer0_runs[arm_name]["direct_head_input_used"]).abs().mean().item()
                    ),
                }
                for arm_name in DIAGNOSTIC_ARM_ORDER
            },
            "pairwise_table": layer0_pairwise,
            "interpretation": {
                "max_pairwise_l2_rms": float(layer0_max_l2),
                "max_pairwise_cosine_distance": float(layer0_max_cos),
                "call": "non_discriminative_or_near_zero" if layer0_non_discriminative else "not_trivially_zero",
                "note": (
                    "same-host fixed t=0 divergence is near zero and should not be over-interpreted"
                    if layer0_non_discriminative
                    else "same-host fixed t=0 already shows some spread"
                ),
            },
        },
        "layer1_short_horizon": {
            "window_steps": SHORT_HORIZON_STEPS,
            "arms": {
                arm_name: {
                    "hook_point": dict(layer1_runs[arm_name]["hook_point"]),
                    "fixed_contacts_mode": str(layer1_runs[arm_name]["fixed_contacts_mode"]),
                    "global_stats": dict(layer1_runs[arm_name]["global_stats"]),
                    "step_records": list(layer1_runs[arm_name]["step_records"]),
                }
                for arm_name in DIAGNOSTIC_ARM_ORDER
            },
            "pairwise_by_step": layer1_pairwise_by_step,
            "pairwise_aggregate": layer1_pairwise_aggregate,
            "divergence_to_references": layer1_divergence,
            "step_level_correlations": layer1_correlations,
            "a1s2_mean_l2_to_refs": layer1_a1s2_mean_l2_to_refs,
        },
        "layer2_moment_matching": {
            "moment_matching_config": {
                "A1S4-mm-E1ref-on-A1S2mix": dict(mm_e1_guard["summary"]),
            },
            "fixed_transfer": {
                "A1S4-mm-E1ref-on-A1S2mix": {
                    "transfer": guard_transfer,
                    "delta_vs_A1S2_mix_nonleg": _transfer_delta(guard_transfer, a1s2_mix_transfer),
                    "delta_vs_E1_top3_full7": _transfer_delta(guard_transfer, e1_top3_reference),
                    "delta_vs_E2A_R_full7": _transfer_delta(guard_transfer, e2a_full7_reference),
                    "delta_vs_target_full7": _transfer_delta(guard_transfer, target_transfer_reference),
                }
            },
            "short_horizon_support": {
                "A1S4-mm-E1ref-on-A1S2mix": {
                    "global_stats": dict(guard_short["global_stats"]),
                    "step_records": list(guard_short["step_records"]),
                    "vs_E1_top3_full7": layer2_divergence["vs_E1_top3_full7"],
                    "vs_target_full7": layer2_divergence["vs_target_full7"],
                    "step_level_correlations": layer2_correlations,
                    "stats_gap_pre_vs_post": {
                        "to_E1_top3_full7": {
                            "base_mean_gap_l2": _norm_l2(
                                torch.tensor(a1s2_stats["channel_mean_vector"]),
                                torch.tensor(e1_stats["channel_mean_vector"]),
                            ),
                            "guarded_mean_gap_l2": _norm_l2(
                                torch.tensor(guard_short["global_stats"]["channel_mean_vector"]),
                                torch.tensor(e1_stats["channel_mean_vector"]),
                            ),
                            "base_std_gap_l2": _norm_l2(
                                torch.tensor(a1s2_stats["channel_std_vector"]),
                                torch.tensor(e1_stats["channel_std_vector"]),
                            ),
                            "guarded_std_gap_l2": _norm_l2(
                                torch.tensor(guard_short["global_stats"]["channel_std_vector"]),
                                torch.tensor(e1_stats["channel_std_vector"]),
                            ),
                        },
                        "to_target_full7": {
                            "base_mean_gap_l2": _norm_l2(
                                torch.tensor(a1s2_stats["channel_mean_vector"]),
                                torch.tensor(target_stats["channel_mean_vector"]),
                            ),
                            "guarded_mean_gap_l2": _norm_l2(
                                torch.tensor(guard_short["global_stats"]["channel_mean_vector"]),
                                torch.tensor(target_stats["channel_mean_vector"]),
                            ),
                            "base_std_gap_l2": _norm_l2(
                                torch.tensor(a1s2_stats["channel_std_vector"]),
                                torch.tensor(target_stats["channel_std_vector"]),
                            ),
                            "guarded_std_gap_l2": _norm_l2(
                                torch.tensor(guard_short["global_stats"]["channel_std_vector"]),
                                torch.tensor(target_stats["channel_std_vector"]),
                            ),
                        },
                    },
                }
            },
        },
        "judgement": judgement,
        "explicit_answers": explicit_answers,
    }

    SUMMARY_JSON.write_text(json.dumps(_to_serializable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_record(summary)
    print(f"[done] wrote {SUMMARY_JSON}")
    print(f"[done] wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
