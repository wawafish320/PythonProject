#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
)
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
from tools.run_direct_head_input_guard_a1s4 import (  # noqa: E402
    _apply_guard,
    _arm_module_groups,
    _channel_stats,
    _cosine_distance,
    _cosine_similarity,
    _extract_raw_step_metric,
    _fmt,
    _load_json,
    _markdown_table,
    _pairwise_table_from_step_results,
    _pearson,
    _stack_rows,
    _summary_stats,
    _swap_modules,
    _tensor_shape,
    _to_serializable,
    _transfer_delta,
)
from train import posttrain  # noqa: E402


RUN_DATE = "20260409"
RUN_NAME = "direct_head_output_guard_a1s5"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DOC_PATH = ROOT / "docs" / "train_design" / "2026-04-09_direct_head_output_guard_a1s5_record.md"

A1S1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_partial_transplant_boundary_a1s1_20260409" / "summary.json"
A1S1_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_partial_transplant_boundary_a1s1_record.md"
A1S2_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_mixed_contract_a1s2_20260409" / "summary.json"
A1S2_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_mixed_contract_a1s2_record.md"
A1S3_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_replace_absorb_boundary_a1s3_20260409" / "summary.json"
A1S3_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_replace_absorb_boundary_a1s3_record.md"
A1S4_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_direct_head_input_guard_a1s4_20260409" / "summary.json"
A1S4_RECORD_MD = ROOT / "docs" / "train_design" / "2026-04-09_direct_head_input_guard_a1s4_record.md"

SHORT_HORIZON_STEPS = 12
AFFINE_EPS = 1e-6
CLEAR_AGG_MARGIN = 0.05
LAYER0_NONDISC_L2 = 0.01
LAYER0_NONDISC_COS = 0.01
SPILLOVER_LEG_DELTA_WARN = 0.02

DIAGNOSTIC_ARM_ORDER: tuple[str, ...] = (
    "target-full7",
    "E1-top3-full7",
    "E2A-R-full7",
    "A1S2-mix-nonleg",
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
    "若强行二选一，只有 weak lean toward host nonleg out side",
    "当前推荐仍偏向更早 boundary / stronger boundary guard",
]

A1S4_DIRECT_INHERITED: list[str] = [
    "direct_pose_head input pairwise divergence = 0",
    "same host + weight transplant only 口径下，所有 arms 在 direct_pose_head.0 input 收到完全相同的 activation",
    "A1-S4 moment-matching input guard 只有数值噪声级别效果，不是有效 lever",
    "因此 bottleneck 不在 head 上游，而更像在 head 内部 / head output → downstream expansion contract",
]


def _global_stats_from_tensors(tensors: Sequence[torch.Tensor]) -> Dict[str, Any]:
    rows = torch.cat([_stack_rows(t) for t in tensors], dim=0)
    std = rows.std(dim=0, unbiased=False)
    return {
        "rows_total": int(rows.shape[0]),
        "feature_dim": int(rows.shape[-1]),
        "channel_mean_vector": rows.mean(dim=0).tolist(),
        "channel_std_vector": std.tolist(),
        "channel_mean_abs_summary": _summary_stats(rows.abs().mean(dim=0).tolist()),
        "channel_std_summary": _summary_stats(std.tolist()),
        "rms_norm_per_row_summary": _summary_stats(
            (torch.linalg.vector_norm(rows, dim=-1) / math.sqrt(max(1, int(rows.shape[-1])))).tolist()
        ),
    }


def _closest_reference(values: Mapping[str, Any]) -> str:
    finite = [(str(k), float(_safe_float(v))) for k, v in values.items() if math.isfinite(float(_safe_float(v)))]
    if not finite:
        return "unresolved"
    min_val = min(val for _, val in finite)
    tied = [name for name, val in finite if abs(val - min_val) <= 1e-9]
    return tied[0] if len(tied) == 1 else "tie:" + ",".join(sorted(tied))


def _resolve_direct_head_output_module(model: nn.Module) -> Dict[str, Any]:
    split_state = model._direct_pose_split_state() if hasattr(model, "_direct_pose_split_state") else None
    head = None
    if isinstance(split_state, Mapping):
        head = split_state.get("head")
    if head is None:
        head = getattr(model, "direct_pose_head", None)
    if not isinstance(head, nn.Module):
        raise RuntimeError("cannot resolve direct_pose_head module for output hook")
    return {
        "module_name": "direct_pose_head",
        "module_class": type(head).__name__,
        "resolved_as": "direct_pose_head forward output activation (shared trunk hidden before split consumers)",
        "module": head,
        "why_this_boundary": (
            "A1-S4 already ruled out head input; this hook captures the earliest downstream-visible shared hidden "
            "that feeds leg/nonleg expansion."
        ),
    }


def _resolve_nonleg_consumer_modules(model: nn.Module) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for side in ("arm", "else"):
        proj = getattr(model, f"direct_pose_{side}_proj", None)
        out = getattr(model, f"direct_pose_out_{side}", None)
        if isinstance(proj, nn.Module):
            resolved.append(
                {
                    "module_name": f"direct_pose_{side}_proj",
                    "module_class": type(proj).__name__,
                    "resolved_as": f"preferred {side} nonleg consumer entry pre-hook",
                    "fallback_used": False,
                    "module": proj,
                }
            )
        elif isinstance(out, nn.Module):
            resolved.append(
                {
                    "module_name": f"direct_pose_out_{side}",
                    "module_class": type(out).__name__,
                    "resolved_as": f"fallback {side} nonleg consumer entry pre-hook because direct_pose_{side}_proj is absent",
                    "fallback_used": True,
                    "module": out,
                }
            )
        else:
            raise RuntimeError(f"cannot resolve nonleg consumer entry for side={side}")
    return resolved


def _combine_consumer_inputs(
    tensor_map: Mapping[str, torch.Tensor],
    module_order: Sequence[str],
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for module_name in module_order:
        tensor = tensor_map.get(module_name)
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"missing captured consumer tensor for {module_name}")
        rows.append(_stack_rows(tensor))
    return torch.cat(rows, dim=0)


def _module_shape_map(tensor_map: Mapping[str, Any], module_order: Sequence[str]) -> Dict[str, List[int]]:
    return {str(name): _tensor_shape(tensor_map.get(name)) for name in module_order}


def _consumer_global_stats(
    *,
    per_module_inputs: Mapping[str, Sequence[torch.Tensor]],
    combined_inputs: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    return {
        "combined": _global_stats_from_tensors(combined_inputs),
        "per_module": {
            str(module_name): _global_stats_from_tensors(inputs)
            for module_name, inputs in per_module_inputs.items()
        },
    }


def _make_module_guard(
    *,
    module_name: str,
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
        "module_name": str(module_name),
        "base_arm": str(base_arm),
        "reference_arm": str(reference_arm),
        "mu_src": mu_src,
        "std_src": std_src,
        "mu_ref": mu_ref,
        "std_ref": std_ref,
        "eps": float(eps),
        "summary": {
            "module_name": str(module_name),
            "base_arm": str(base_arm),
            "reference_arm": str(reference_arm),
            "formula": "x_hat = (x - mu_src) / (std_src + eps) * std_ref + mu_ref",
            "stats_scope": "Layer1 short-horizon nonleg consumer-entry rows from the same host / same entry / same offset",
            "stats_source_arm": str(base_arm),
            "stats_reference_arm": str(reference_arm),
            "rows_total_source": int(source_stats["rows_total"]),
            "rows_total_reference": int(reference_stats["rows_total"]),
            "feature_dim": int(source_stats["feature_dim"]),
            "eps": float(eps),
            "source_low_variance_channels_le_eps": int(
                sum(1 for v in source_stats["channel_std_vector"] if float(_safe_float(v)) <= float(eps))
            ),
            "reference_low_variance_channels_le_eps": int(
                sum(1 for v in reference_stats["channel_std_vector"] if float(_safe_float(v)) <= float(eps))
            ),
            "scale_summary": _summary_stats(scale.detach().cpu().tolist()),
        },
    }


def _build_guard_bundle(
    *,
    name: str,
    base_arm: str,
    reference_arm: str,
    module_order: Sequence[str],
    source_stats_by_module: Mapping[str, Any],
    reference_stats_by_module: Mapping[str, Any],
    eps: float,
) -> Dict[str, Any]:
    guards: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    for module_name in module_order:
        guard = _make_module_guard(
            module_name=module_name,
            base_arm=base_arm,
            reference_arm=reference_arm,
            source_stats=source_stats_by_module[module_name],
            reference_stats=reference_stats_by_module[module_name],
            eps=eps,
        )
        guards[module_name] = guard
        summaries[module_name] = dict(guard["summary"])
    return {
        "name": str(name),
        "base_arm": str(base_arm),
        "reference_arm": str(reference_arm),
        "module_order": [str(x) for x in module_order],
        "module_guards": guards,
        "summary": {
            "name": str(name),
            "base_arm": str(base_arm),
            "reference_arm": str(reference_arm),
            "formula": "x_hat = (x - mu_src) / (std_src + eps) * std_ref + mu_ref",
            "stats_scope": "Layer1 short-horizon nonleg consumer-entry activation, estimated separately per consumer module",
            "module_order": [str(x) for x in module_order],
            "combined_representation": "consumer-entry rows concatenated in fixed module order for reporting only",
            "eps": float(eps),
            "per_module": summaries,
        },
    }


def _run_single_step_boundary_capture(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    donor_groups: Sequence[Tuple[Mapping[str, Any], Sequence[str]]],
    guard_bundle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model = host_bundle["case"]["trainer"].model
    if model is None:
        raise RuntimeError("host model missing")
    diagnostic_hook = _resolve_direct_head_output_module(model)
    intervention_hooks = _resolve_nonleg_consumer_modules(model)
    consumer_order = [str(row["module_name"]) for row in intervention_hooks]
    capture: Dict[str, Any] = {
        "head_output": None,
        "consumer_pre": {},
        "consumer_used": {},
    }
    handles: List[Any] = []
    backups: List[Tuple[nn.Module, Dict[str, Any]]] = []
    try:
        backups = _swap_modules(target_model=model, donor_groups=donor_groups)

        def _head_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise RuntimeError("direct_pose_head output hook received non-tensor")
            capture["head_output"] = output.detach().clone().cpu()

        handles.append(diagnostic_hook["module"].register_forward_hook(_head_hook))

        for hook_row in intervention_hooks:
            module_name = str(hook_row["module_name"])

            def _make_pre_hook(name: str):
                def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    capture["consumer_pre"][name] = x.detach().clone().cpu()
                    guard = None
                    if guard_bundle is not None:
                        guard = (guard_bundle.get("module_guards") or {}).get(name)
                    x_use = _apply_guard(x, guard) if guard is not None else x
                    capture["consumer_used"][name] = x_use.detach().clone().cpu()
                    if x_use is x:
                        return None
                    if len(inputs) == 1:
                        return (x_use,)
                    return (x_use, *inputs[1:])

                return _pre_hook

            handles.append(hook_row["module"].register_forward_pre_hook(_make_pre_hook(module_name)))

        result = _run_single_step(host_bundle, prep_host, fixed_contacts=fixed_contacts)
    finally:
        for handle in reversed(handles):
            handle.remove()
        if backups:
            _restore_weight_swap(list(reversed(backups)))

    head_output = capture["head_output"]
    if not torch.is_tensor(head_output):
        raise RuntimeError("failed to capture direct_pose_head output at single step")
    consumer_pre = capture["consumer_pre"]
    consumer_used = capture["consumer_used"]
    combined_pre = _combine_consumer_inputs(consumer_pre, consumer_order)
    combined_used = _combine_consumer_inputs(consumer_used, consumer_order)
    result["diagnostic_hook"] = {
        "module_name": str(diagnostic_hook["module_name"]),
        "module_class": str(diagnostic_hook["module_class"]),
        "resolved_as": str(diagnostic_hook["resolved_as"]),
        "why_this_boundary": str(diagnostic_hook["why_this_boundary"]),
    }
    result["intervention_hook"] = {
        "resolved_modules": [
            {
                "module_name": str(row["module_name"]),
                "module_class": str(row["module_class"]),
                "resolved_as": str(row["resolved_as"]),
                "fallback_used": bool(row["fallback_used"]),
            }
            for row in intervention_hooks
        ],
        "combined_representation": "row-concat of consumer-entry tensors in fixed module order",
        "why_this_boundary": (
            "guard acts only on nonleg downstream consumer entry, so preserved leg side does not receive a global shared-hidden rewrite"
        ),
    }
    result["direct_head_output"] = head_output
    result["nonleg_consumer_entry_pre"] = consumer_pre
    result["nonleg_consumer_entry_used"] = consumer_used
    result["nonleg_consumer_entry_pre_combined"] = combined_pre
    result["nonleg_consumer_entry_used_combined"] = combined_used
    return result


def _run_short_horizon_boundary_capture(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts_first: torch.Tensor,
    donor_groups: Sequence[Tuple[Mapping[str, Any], Sequence[str]]],
    window_steps: int,
    guard_bundle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model = host_bundle["case"]["trainer"].model
    if model is None:
        raise RuntimeError("host model missing")
    diagnostic_hook = _resolve_direct_head_output_module(model)
    intervention_hooks = _resolve_nonleg_consumer_modules(model)
    consumer_order = [str(row["module_name"]) for row in intervention_hooks]
    backups: List[Tuple[nn.Module, Dict[str, Any]]] = []
    handles: List[Any] = []
    orig_prepare_contacts = posttrain._prepare_rollout_contacts_input
    contact_calls = {"count": 0}
    step_capture: Dict[str, Any] = {"head_output": None, "consumer_pre": {}, "consumer_used": {}}
    head_outputs: List[torch.Tensor] = []
    consumer_pre_combined: List[torch.Tensor] = []
    consumer_used_combined: List[torch.Tensor] = []
    consumer_pre_by_module: Dict[str, List[torch.Tensor]] = {name: [] for name in consumer_order}
    consumer_used_by_module: Dict[str, List[torch.Tensor]] = {name: [] for name in consumer_order}
    records: List[Dict[str, Any]] = []
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

        def _head_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise RuntimeError("direct_pose_head output hook received non-tensor")
            step_capture["head_output"] = output.detach().clone().cpu()

        handles.append(diagnostic_hook["module"].register_forward_hook(_head_hook))

        for hook_row in intervention_hooks:
            module_name = str(hook_row["module_name"])

            def _make_pre_hook(name: str):
                def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    step_capture["consumer_pre"][name] = x.detach().clone().cpu()
                    guard = None
                    if guard_bundle is not None:
                        guard = (guard_bundle.get("module_guards") or {}).get(name)
                    x_use = _apply_guard(x, guard) if guard is not None else x
                    step_capture["consumer_used"][name] = x_use.detach().clone().cpu()
                    if x_use is x:
                        return None
                    if len(inputs) == 1:
                        return (x_use,)
                    return (x_use, *inputs[1:])

                return _pre_hook

            handles.append(hook_row["module"].register_forward_pre_hook(_make_pre_hook(module_name)))

        posttrain._prepare_rollout_contacts_input = _prepare_contacts_override
        ctx = _build_step_ctx(host_bundle, prep_host)
        total_steps = min(int(window_steps), int(ctx["runtime"]["total_steps"]))

        for step_idx in range(total_steps):
            step_capture["head_output"] = None
            step_capture["consumer_pre"] = {}
            step_capture["consumer_used"] = {}
            prev_lens = {
                "loss_terms": len(ctx["accum"]["loss_terms"]),
                "inc_terms": len(ctx["accum"]["inc_terms"]),
                "dir_base_terms": len(ctx["accum"]["dir_base_terms"]),
                "dir_leg_base_terms": len(ctx["accum"]["dir_leg_base_terms"]),
                "dir_nonleg_base_terms": len(ctx["accum"]["dir_nonleg_base_terms"]),
            }
            posttrain._lambda_rollout_unroll_single_step(t=int(step_idx), ctx=ctx)
            step_weight = float(ctx["data"]["step_weights"][step_idx].detach().cpu().item())

            head_output = step_capture["head_output"]
            if not torch.is_tensor(head_output):
                raise RuntimeError(f"failed to capture direct_pose_head output at short-horizon step {step_idx}")
            combined_pre = _combine_consumer_inputs(step_capture["consumer_pre"], consumer_order)
            combined_used = _combine_consumer_inputs(step_capture["consumer_used"], consumer_order)

            head_outputs.append(head_output)
            consumer_pre_combined.append(combined_pre)
            consumer_used_combined.append(combined_used)

            head_stats = _channel_stats(head_output)
            combined_stats = _channel_stats(combined_used)
            per_module_payload: Dict[str, Any] = {}
            for module_name in consumer_order:
                pre_t = step_capture["consumer_pre"][module_name]
                used_t = step_capture["consumer_used"][module_name]
                consumer_pre_by_module[module_name].append(pre_t)
                consumer_used_by_module[module_name].append(used_t)
                pre_stats = _channel_stats(pre_t)
                used_stats = _channel_stats(used_t)
                per_module_payload[module_name] = {
                    "pre_shape": list(pre_stats["shape"]),
                    "used_shape": list(used_stats["shape"]),
                    "pre_mean_abs": float(pre_stats["mean_abs"]),
                    "used_mean_abs": float(used_stats["mean_abs"]),
                    "pre_rms_norm_per_row": float(pre_stats["rms_norm_per_row"]),
                    "used_rms_norm_per_row": float(used_stats["rms_norm_per_row"]),
                }

            records.append(
                {
                    "step": int(step_idx),
                    "step_weight": float(step_weight),
                    "direct_head_output_shape": list(head_stats["shape"]),
                    "direct_head_output_mean_abs": float(head_stats["mean_abs"]),
                    "direct_head_output_rms_norm_per_row": float(head_stats["rms_norm_per_row"]),
                    "direct_head_output_mean_vector": list(head_stats["mean_vector"]),
                    "direct_head_output_std_vector": list(head_stats["std_vector"]),
                    "nonleg_consumer_entry_combined_shape": list(combined_stats["shape"]),
                    "nonleg_consumer_entry_combined_mean_abs": float(combined_stats["mean_abs"]),
                    "nonleg_consumer_entry_combined_rms_norm_per_row": float(combined_stats["rms_norm_per_row"]),
                    "nonleg_consumer_entry_combined_mean_vector": list(combined_stats["mean_vector"]),
                    "nonleg_consumer_entry_combined_std_vector": list(combined_stats["std_vector"]),
                    "nonleg_consumer_entry_modules": per_module_payload,
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
        for handle in reversed(handles):
            handle.remove()
        posttrain._prepare_rollout_contacts_input = orig_prepare_contacts
        if backups:
            _restore_weight_swap(list(reversed(backups)))

    return {
        "diagnostic_hook": {
            "module_name": str(diagnostic_hook["module_name"]),
            "module_class": str(diagnostic_hook["module_class"]),
            "resolved_as": str(diagnostic_hook["resolved_as"]),
            "why_this_boundary": str(diagnostic_hook["why_this_boundary"]),
        },
        "intervention_hook": {
            "resolved_modules": [
                {
                    "module_name": str(row["module_name"]),
                    "module_class": str(row["module_class"]),
                    "resolved_as": str(row["resolved_as"]),
                    "fallback_used": bool(row["fallback_used"]),
                }
                for row in intervention_hooks
            ],
            "combined_representation": "row-concat of consumer-entry tensors in fixed module order",
            "why_this_boundary": (
                "guard acts only on nonleg downstream consumer entry, so preserved leg side does not receive a global shared-hidden rewrite"
            ),
        },
        "window_steps": int(len(records)),
        "fixed_contacts_mode": "baseline replace native contacts_in_t only on rollout first forward",
        "step_records": records,
        "head_output_global_stats": _global_stats_from_tensors(head_outputs),
        "nonleg_consumer_entry_global_stats": _consumer_global_stats(
            per_module_inputs=consumer_used_by_module,
            combined_inputs=consumer_used_combined,
        ),
        "_head_outputs": head_outputs,
        "_consumer_pre_combined": consumer_pre_combined,
        "_consumer_used_combined": consumer_used_combined,
    }


def _pairwise_curves_from_series(
    series_a: Sequence[torch.Tensor],
    series_b: Sequence[torch.Tensor],
) -> List[Dict[str, float]]:
    steps = min(len(series_a), len(series_b))
    rows: List[Dict[str, float]] = []
    for step_idx in range(steps):
        a_t = series_a[step_idx]
        b_t = series_b[step_idx]
        rows.append(
            {
                "step": int(step_idx),
                "l2_rms": _norm_l2(a_t, b_t),
                "cosine_similarity": _cosine_similarity(a_t, b_t),
                "cosine_distance": _cosine_distance(a_t, b_t),
            }
        )
    return rows


def _reference_divergence_payload_from_series(
    *,
    arm_name: str,
    reference_name: str,
    arm_series: Sequence[torch.Tensor],
    reference_series: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    curve = _pairwise_curves_from_series(arm_series, reference_series)
    return {
        "arm": str(arm_name),
        "reference": str(reference_name),
        "curve": curve,
        "aggregate": {
            "steps": int(len(curve)),
            "l2_rms": _summary_stats([row["l2_rms"] for row in curve]),
            "cosine_distance": _summary_stats([row["cosine_distance"] for row in curve]),
        },
    }


def _correlation_payload_from_series(
    *,
    step_records: Sequence[Mapping[str, Any]],
    divergence_curve: Sequence[Mapping[str, Any]],
    reference_name: str,
    divergence_metric_definition: str,
) -> Dict[str, Any]:
    l2_curve = [_safe_float(row.get("l2_rms")) for row in divergence_curve]
    cos_curve = [_safe_float(row.get("cosine_distance")) for row in divergence_curve]
    dir_leg = [_safe_float(row.get("dir_leg_error")) for row in step_records[: len(divergence_curve)]]
    dir_nonleg = [_safe_float(row.get("dir_nonleg_error")) for row in step_records[: len(divergence_curve)]]
    dir_base = [_safe_float(row.get("dir_base_error")) for row in step_records[: len(divergence_curve)]]
    return {
        "reference": str(reference_name),
        "divergence_metric_definition": str(divergence_metric_definition),
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


def _layer2_short_horizon_delta_from_series(
    *,
    base_series: Sequence[torch.Tensor],
    guarded_series: Sequence[torch.Tensor],
    reference_series: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    base_curve = _pairwise_curves_from_series(base_series, reference_series)
    guarded_curve = _pairwise_curves_from_series(guarded_series, reference_series)
    steps = min(len(base_curve), len(guarded_curve))
    delta_curve: List[Dict[str, Any]] = []
    for idx in range(steps):
        delta_curve.append(
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
        "base_vs_reference": {
            "curve": base_curve,
            "aggregate": {
                "steps": int(len(base_curve)),
                "l2_rms": _summary_stats([row["l2_rms"] for row in base_curve]),
                "cosine_distance": _summary_stats([row["cosine_distance"] for row in base_curve]),
            },
        },
        "guarded_vs_reference": {
            "curve": guarded_curve,
            "aggregate": {
                "steps": int(len(guarded_curve)),
                "l2_rms": _summary_stats([row["l2_rms"] for row in guarded_curve]),
                "cosine_distance": _summary_stats([row["cosine_distance"] for row in guarded_curve]),
            },
        },
        "delta_curve_guarded_minus_base": delta_curve,
        "delta_summary": {
            "l2_rms_delta_guarded_minus_base": _summary_stats(
                [row["l2_rms_delta_guarded_minus_base"] for row in delta_curve]
            ),
            "cosine_distance_delta_guarded_minus_base": _summary_stats(
                [row["cosine_distance_delta_guarded_minus_base"] for row in delta_curve]
            ),
        },
    }


def _stats_gap_payload(
    *,
    base_stats: Mapping[str, Any],
    guarded_stats: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    base_mean_gap = _norm_l2(
        torch.tensor(base_stats["channel_mean_vector"], dtype=torch.float32),
        torch.tensor(reference_stats["channel_mean_vector"], dtype=torch.float32),
    )
    guarded_mean_gap = _norm_l2(
        torch.tensor(guarded_stats["channel_mean_vector"], dtype=torch.float32),
        torch.tensor(reference_stats["channel_mean_vector"], dtype=torch.float32),
    )
    base_std_gap = _norm_l2(
        torch.tensor(base_stats["channel_std_vector"], dtype=torch.float32),
        torch.tensor(reference_stats["channel_std_vector"], dtype=torch.float32),
    )
    guarded_std_gap = _norm_l2(
        torch.tensor(guarded_stats["channel_std_vector"], dtype=torch.float32),
        torch.tensor(reference_stats["channel_std_vector"], dtype=torch.float32),
    )
    return {
        "base_mean_gap_l2": float(base_mean_gap),
        "guarded_mean_gap_l2": float(guarded_mean_gap),
        "delta_mean_gap_guarded_minus_base": float(_safe_float(guarded_mean_gap) - _safe_float(base_mean_gap)),
        "base_std_gap_l2": float(base_std_gap),
        "guarded_std_gap_l2": float(guarded_std_gap),
        "delta_std_gap_guarded_minus_base": float(_safe_float(guarded_std_gap) - _safe_float(base_std_gap)),
    }


def _transfer_status(
    *,
    guarded_transfer: Mapping[str, Any],
    base_transfer: Mapping[str, Any],
) -> str:
    agg_delta = float(
        _safe_float(guarded_transfer.get("aggregate_transfer_score"))
        - _safe_float(base_transfer.get("aggregate_transfer_score"))
    )
    if agg_delta > CLEAR_AGG_MARGIN:
        return "clear_win"
    if agg_delta > 0.0:
        return "partial_positive"
    if agg_delta < -0.01:
        return "negative"
    return "flat"


def _judge_boundary_direction(
    *,
    layer0_head_pairwise: Mapping[str, Any],
    head_mean_l2_to_refs: Mapping[str, Any],
    consumer_mean_l2_to_refs: Mapping[str, Any],
    guarded_transfer: Mapping[str, Any],
    base_transfer: Mapping[str, Any],
    layer2_support_vs_e2a: Mapping[str, Any],
) -> Dict[str, Any]:
    layer0_l2 = [
        _safe_float(payload.get("l2_rms"))
        for row in layer0_head_pairwise.values()
        for payload in row.values()
        if math.isfinite(_safe_float(payload.get("l2_rms")))
    ]
    max_layer0_l2 = max(layer0_l2, default=float("nan"))
    divergence_threshold = max(
        LAYER0_NONDISC_L2,
        float(_safe_float(max_layer0_l2)) * 2.0 if math.isfinite(float(_safe_float(max_layer0_l2))) else 0.0,
    )
    finite_divergence_values = [
        float(_safe_float(v))
        for v in list(head_mean_l2_to_refs.values()) + list(consumer_mean_l2_to_refs.values())
        if math.isfinite(float(_safe_float(v)))
    ]
    divergence_present = any(val > divergence_threshold for val in finite_divergence_values)
    transfer_status = _transfer_status(guarded_transfer=guarded_transfer, base_transfer=base_transfer)
    e2a_l2_delta_mean = float(
        _safe_float(
            ((layer2_support_vs_e2a.get("delta_summary") or {}).get("l2_rms_delta_guarded_minus_base") or {}).get("mean")
        )
    )
    support_improves_e2a = math.isfinite(e2a_l2_delta_mean) and e2a_l2_delta_mean < 0.0
    agg_delta = float(
        _safe_float(guarded_transfer.get("aggregate_transfer_score"))
        - _safe_float(base_transfer.get("aggregate_transfer_score"))
    )
    leg_delta = float(
        _safe_float(guarded_transfer.get("dir_leg_closure_ratio"))
        - _safe_float(base_transfer.get("dir_leg_closure_ratio"))
    )
    if divergence_present and transfer_status == "clear_win":
        case_label = "Case A"
        interpretation = "head-to-expansion contract mismatch carries useful signal and the simplest affine guard already improves fixed transfer clearly"
        recommend_boundary_mainline = True
        next_step = "stronger guard or learned lightweight adapter at the same head-to-expansion boundary"
    elif divergence_present and (transfer_status == "partial_positive" or support_improves_e2a):
        case_label = "Case B"
        interpretation = "direction looks real, but first/second-order affine matching is only partially effective"
        recommend_boundary_mainline = True
        next_step = "upgrade directly to learned affine / small adapter at the same boundary; do not claim solved"
    elif divergence_present:
        case_label = "Case C"
        interpretation = "divergence exists, but pure affine moment matching has little transfer effect; mismatch looks more structural than simple moment shift"
        recommend_boundary_mainline = True
        next_step = "skip stronger pure-stat guards and move to learned adapter or training-side contract constraint at the same boundary"
    else:
        case_label = "Case D"
        interpretation = "this boundary is not strongly discriminative under the current assay, and the affine guard is not useful"
        recommend_boundary_mainline = False
        next_step = "shift effort to more-downstream contract work or training-side recipe constraints"
    if abs(leg_delta) > SPILLOVER_LEG_DELTA_WARN:
        spillover_note = "leg-side spillover is material and should be treated as a warning"
    else:
        spillover_note = "leg-side spillover stays limited under the consumer-only guard"
    return {
        "layer0_max_head_output_l2_rms": float(max_layer0_l2),
        "divergence_threshold_l2_rms": float(divergence_threshold),
        "short_horizon_divergence_present": bool(divergence_present),
        "guard_transfer_status_vs_A1S2_mix": str(transfer_status),
        "aggregate_delta_vs_A1S2_mix_nonleg": float(agg_delta),
        "support_mean_l2_delta_to_E2A_R_full7": float(e2a_l2_delta_mean),
        "dir_leg_closure_delta_vs_A1S2_mix_nonleg": float(leg_delta),
        "leg_spillover_note": str(spillover_note),
        "case_label": str(case_label),
        "interpretation": str(interpretation),
        "recommend_head_to_expansion_boundary_as_mainline": bool(recommend_boundary_mainline),
        "next_step": str(next_step),
    }


def _write_record(summary: Mapping[str, Any]) -> None:
    layer0 = summary["layer0_sanity"]
    layer1 = summary["layer1_short_horizon"]
    layer2 = summary["layer2_affine_guard"]
    judgement = summary["judgement"]
    explicit = summary["explicit_answers"]

    layer0_rows: List[List[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, row in (layer0.get("head_output_pairwise_table") or {}).items():
        for right, payload in row.items():
            pair = tuple(sorted((str(left), str(right))))
            if left == right or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            layer0_rows.append(
                [str(left), str(right), _fmt(payload.get("l2_rms")), _fmt(payload.get("cosine_distance"))]
            )

    layer1_rows: List[List[str]] = []
    for tap_name, pair_table in (layer1.get("pairwise_distance_tables") or {}).items():
        for pair_name, payload in (pair_table.get("aggregate") or {}).items():
            layer1_rows.append(
                [
                    str(tap_name),
                    str(pair_name),
                    _fmt(((payload.get("l2_rms") or {}).get("mean"))),
                    _fmt(((payload.get("cosine_distance") or {}).get("mean"))),
                    _fmt(((payload.get("l2_rms") or {}).get("max"))),
                ]
            )

    corr_rows: List[List[str]] = []
    for tap_name, tap_payload in ((layer1.get("step_level_correlations") or {}).get("A1S2-mix-nonleg") or {}).items():
        for ref_name, payload in tap_payload.items():
            pearson = payload.get("pearson") or {}
            corr_rows.append(
                [
                    str(tap_name),
                    str(ref_name),
                    _fmt(pearson.get("l2_rms_vs_dir_leg_error")),
                    _fmt(pearson.get("l2_rms_vs_dir_nonleg_error")),
                    _fmt(pearson.get("cosine_distance_vs_dir_leg_error")),
                    _fmt(pearson.get("cosine_distance_vs_dir_nonleg_error")),
                ]
            )

    transfer_rows: List[List[str]] = []
    refs = summary["reused_references"]
    guard_key = "A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer"
    guard_transfer = (layer2["fixed_transfer"][guard_key] or {}).get("transfer") or {}
    for label, transfer in (
        ("target-full7", refs["transplant_compatible_target"]["transfer"]),
        ("E1-top3-full7", refs["E1_top3_full7"]["transfer"]),
        ("E2A-R-full7", refs["E2A_R_full7"]["transfer"]),
        ("A1S2-mix-nonleg", refs["A1S2_mix_nonleg"]["transfer"]),
        (guard_key, guard_transfer),
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

    guard_cfg = layer2["affine_guard_config"][guard_key]
    lines: List[str] = []
    lines.append("# 2026-04-09 direct-head-output guard A1-S5 record")
    lines.append("")
    lines.append("## 1. Scope / inherited conclusions")
    lines.append("")
    lines.append("- Scope: A1-S5 direct_pose_head output / nonleg consumer-entry merged diagnostic + runtime affine guard, no-train.")
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
    for item in summary["a1s4_inherited"]["direct_conclusions"]:
        lines.append(f"- A1-S4 inherit: {item}.")
    lines.append("")
    lines.append("## 2. Why A1-S5 after A1-S4")
    lines.append("")
    lines.append("- A1-S4 already showed `direct_pose_head` input pairwise divergence = 0, so upstream input-side guard is not the lever.")
    lines.append("- This round therefore moves one boundary later: shared head output for diagnosis, nonleg consumer entry for intervention.")
    lines.append("")
    lines.append("## 3. Host / donor / target inventory")
    lines.append("")
    lines.append(f"- Host ckpt: `{summary['host']['ckpt']}`.")
    lines.append(f"- E1 donor ckpt: `{summary['anchor_donor']['ckpt']}`.")
    lines.append(f"- E2A donor ckpt: `{summary['expansion_donor']['ckpt']}`.")
    lines.append(f"- Baseline replace donor ckpt: `{summary['baseline_replace']['ckpt']}`.")
    lines.append("")
    lines.append("## 4. Diagnostic hook definition")
    lines.append("")
    lines.append(f"- Module: `{summary['diagnostic_hook_definition']['module_name']}` ({summary['diagnostic_hook_definition']['module_class']}).")
    lines.append(f"- Meaning: `{summary['diagnostic_hook_definition']['resolved_as']}`.")
    lines.append(f"- Why here: {summary['diagnostic_hook_definition']['why_this_boundary']}.")
    lines.append("")
    lines.append("## 5. Intervention hook definition")
    lines.append("")
    for row in summary["intervention_hook_definition"]["resolved_modules"]:
        lines.append(
            f"- `{row['module_name']}` ({row['module_class']}) — {row['resolved_as']}."
        )
    lines.append(f"- Boundary rationale: {summary['intervention_hook_definition']['why_this_boundary']}.")
    lines.append("")
    lines.append("## 6. Layer 0 sanity table")
    lines.append("")
    lines.append(_markdown_table(["left", "right", "l2_rms", "cosine_distance"], layer0_rows))
    lines.append("")
    lines.append(
        f"- Call: `{layer0['interpretation']['call']}` "
        f"(max l2=`{_fmt(layer0['interpretation']['max_pairwise_l2_rms'])}`, "
        f"max cosdist=`{_fmt(layer0['interpretation']['max_pairwise_cosine_distance'])}`)."
    )
    lines.append("")
    lines.append("## 7. Layer 1 divergence table")
    lines.append("")
    lines.append(_markdown_table(["tap", "pair", "mean_l2_rms", "mean_cosdist", "max_l2_rms"], layer1_rows))
    lines.append("")
    lines.append("- `A1S2-mix-nonleg` mean l2 to refs by tap:")
    for tap_name, vals in (layer1["a1s2_mean_l2_to_refs"] or {}).items():
        lines.append(
            f"  - {tap_name}: E1=`{_fmt(vals.get('E1-top3-full7'))}`, "
            f"E2A=`{_fmt(vals.get('E2A-R-full7'))}`, target=`{_fmt(vals.get('target-full7'))}`"
        )
    lines.append("")
    lines.append("## 8. Step-level correlation summary")
    lines.append("")
    lines.append(
        _markdown_table(
            ["tap", "reference", "l2~dir_leg", "l2~dir_nonleg", "cos~dir_leg", "cos~dir_nonleg"],
            corr_rows,
        )
    )
    lines.append("")
    lines.append("## 9. Affine transform definition")
    lines.append("")
    lines.append(f"- Guard arm: `{guard_cfg['name']}`.")
    lines.append(f"- Base arm: `{guard_cfg['base_arm']}`.")
    lines.append(f"- Reference arm: `{guard_cfg['reference_arm']}`.")
    lines.append(f"- Formula: `{guard_cfg['formula']}`.")
    lines.append(f"- Eps: `{_fmt(guard_cfg['eps'])}`.")
    for module_name, module_cfg in (guard_cfg.get("per_module") or {}).items():
        lines.append(
            f"- `{module_name}` rows={module_cfg['rows_total_source']}, feat_dim={module_cfg['feature_dim']}, "
            f"scale_mean=`{_fmt((module_cfg['scale_summary'] or {}).get('mean'))}`."
        )
    lines.append("")
    lines.append("## 10. Layer 2 fixed transfer assay table")
    lines.append("")
    lines.append(
        _markdown_table(
            ["arm", "out_gap", "dir_base_gap", "dir_leg_gap", "dir_nonleg_gap", "agg_score"],
            transfer_rows,
        )
    )
    lines.append("")
    lines.append(
        f"- Guard minus `A1S2-mix-nonleg` aggregate = "
        f"`{_fmt(((layer2['fixed_transfer'][guard_key] or {}).get('delta_vs_A1S2_mix_nonleg') or {}).get('closure_delta_candidate_minus_reference', {}).get('aggregate_transfer_score'))}`."
    )
    lines.append("")
    lines.append("## 11. Boundary interpretation")
    lines.append("")
    lines.append(f"- Case: `{judgement['case_label']}`.")
    lines.append(f"- Interpretation: {judgement['interpretation']}.")
    lines.append(f"- Spillover: {judgement['leg_spillover_note']}.")
    lines.append("")
    lines.append("## 12. Next-step recommendation")
    lines.append("")
    lines.append(f"- Recommended next step: {judgement['next_step']}.")
    lines.append(f"- Q1: {explicit['q1_direct_head_output_same_host_fixed_t0_nonzero_divergence']['answer']}.")
    lines.append(f"- Q2: {explicit['q2_a1s2_short_horizon_closest_reference']['answer']}.")
    lines.append(f"- Q3: {explicit['q3_a1s5_affine_guard_clear_transfer_win_over_a1s2']['answer']}.")
    lines.append(f"- Q4: {explicit['q4_head_to_expansion_boundary_mainline']['answer']}.")
    lines.append(f"- Q5: {explicit['q5_next_step_priority']['answer']}.")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    a1s1_summary = _load_json(A1S1_SUMMARY_JSON)
    a1s2_summary = _load_json(A1S2_SUMMARY_JSON)
    a1s3_summary = _load_json(A1S3_SUMMARY_JSON)
    a1s4_summary = _load_json(A1S4_SUMMARY_JSON)

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
    diagnostic_hook_definition = _resolve_direct_head_output_module(host_model)
    intervention_hook_definition = _resolve_nonleg_consumer_modules(host_model)

    layer0_runs: Dict[str, Any] = {}
    for arm_name in DIAGNOSTIC_ARM_ORDER:
        donor_groups = _arm_module_groups(
            arm_name=arm_name,
            baseline_bundle=baseline_bundle,
            e1_bundle=e1_bundle,
            e2a_bundle=e2a_bundle,
        )
        layer0_runs[arm_name] = _run_single_step_boundary_capture(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            donor_groups=donor_groups,
            guard_bundle=None,
        )

    layer0_head_pairwise = _pairwise_table_from_step_results(
        {arm_name: run["direct_head_output"] for arm_name, run in layer0_runs.items()}
    )
    layer0_max_l2 = max(
        (
            _safe_float(payload.get("l2_rms"))
            for row in layer0_head_pairwise.values()
            for payload in row.values()
            if math.isfinite(_safe_float(payload.get("l2_rms")))
        ),
        default=float("nan"),
    )
    layer0_max_cos = max(
        (
            _safe_float(payload.get("cosine_distance"))
            for row in layer0_head_pairwise.values()
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
        layer1_runs[arm_name] = _run_short_horizon_boundary_capture(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts_first=fixed_contacts,
            donor_groups=donor_groups,
            window_steps=SHORT_HORIZON_STEPS,
            guard_bundle=None,
        )

    pairwise_tables: Dict[str, Any] = {
        "head_output": {"by_step": {}, "aggregate": {}},
        "nonleg_consumer_entry": {"by_step": {}, "aggregate": {}},
    }
    for idx, left in enumerate(DIAGNOSTIC_ARM_ORDER):
        for right in DIAGNOSTIC_ARM_ORDER[idx + 1 :]:
            pair_name = f"{left}__vs__{right}"
            head_curve = _pairwise_curves_from_series(layer1_runs[left]["_head_outputs"], layer1_runs[right]["_head_outputs"])
            cons_curve = _pairwise_curves_from_series(
                layer1_runs[left]["_consumer_used_combined"],
                layer1_runs[right]["_consumer_used_combined"],
            )
            pairwise_tables["head_output"]["by_step"][pair_name] = head_curve
            pairwise_tables["head_output"]["aggregate"][pair_name] = {
                "steps": int(len(head_curve)),
                "l2_rms": _summary_stats([row["l2_rms"] for row in head_curve]),
                "cosine_distance": _summary_stats([row["cosine_distance"] for row in head_curve]),
            }
            pairwise_tables["nonleg_consumer_entry"]["by_step"][pair_name] = cons_curve
            pairwise_tables["nonleg_consumer_entry"]["aggregate"][pair_name] = {
                "steps": int(len(cons_curve)),
                "l2_rms": _summary_stats([row["l2_rms"] for row in cons_curve]),
                "cosine_distance": _summary_stats([row["cosine_distance"] for row in cons_curve]),
            }

    a1s2_divergence: Dict[str, Any] = {"head_output": {}, "nonleg_consumer_entry": {}}
    a1s2_correlations: Dict[str, Any] = {"head_output": {}, "nonleg_consumer_entry": {}}
    for ref_name in ("E1-top3-full7", "E2A-R-full7", "target-full7"):
        head_payload = _reference_divergence_payload_from_series(
            arm_name="A1S2-mix-nonleg",
            reference_name=ref_name,
            arm_series=layer1_runs["A1S2-mix-nonleg"]["_head_outputs"],
            reference_series=layer1_runs[ref_name]["_head_outputs"],
        )
        cons_payload = _reference_divergence_payload_from_series(
            arm_name="A1S2-mix-nonleg",
            reference_name=ref_name,
            arm_series=layer1_runs["A1S2-mix-nonleg"]["_consumer_used_combined"],
            reference_series=layer1_runs[ref_name]["_consumer_used_combined"],
        )
        a1s2_divergence["head_output"][ref_name] = head_payload
        a1s2_divergence["nonleg_consumer_entry"][ref_name] = cons_payload
        a1s2_correlations["head_output"][ref_name] = _correlation_payload_from_series(
            step_records=layer1_runs["A1S2-mix-nonleg"]["step_records"],
            divergence_curve=head_payload["curve"],
            reference_name=ref_name,
            divergence_metric_definition="direct_pose_head output activation flattened per-step",
        )
        a1s2_correlations["nonleg_consumer_entry"][ref_name] = _correlation_payload_from_series(
            step_records=layer1_runs["A1S2-mix-nonleg"]["step_records"],
            divergence_curve=cons_payload["curve"],
            reference_name=ref_name,
            divergence_metric_definition="nonleg consumer-entry activation flattened after module-order row concat",
        )

    a1s2_mean_l2_to_refs = {
        "head_output": {
            ref_name: float(
                _safe_float((((a1s2_divergence["head_output"][ref_name] or {}).get("aggregate") or {}).get("l2_rms") or {}).get("mean"))
            )
            for ref_name in ("E1-top3-full7", "E2A-R-full7", "target-full7")
        },
        "nonleg_consumer_entry": {
            ref_name: float(
                _safe_float(
                    (((a1s2_divergence["nonleg_consumer_entry"][ref_name] or {}).get("aggregate") or {}).get("l2_rms") or {}).get("mean")
                )
            )
            for ref_name in ("E1-top3-full7", "E2A-R-full7", "target-full7")
        },
    }
    closest_head_ref = _closest_reference(a1s2_mean_l2_to_refs["head_output"])
    closest_consumer_ref = _closest_reference(a1s2_mean_l2_to_refs["nonleg_consumer_entry"])

    consumer_order = [
        str(row["module_name"])
        for row in layer1_runs["A1S2-mix-nonleg"]["intervention_hook"]["resolved_modules"]
    ]
    guard_bundle = _build_guard_bundle(
        name="A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer",
        base_arm="A1S2-mix-nonleg",
        reference_arm="E2A-R-full7",
        module_order=consumer_order,
        source_stats_by_module=(layer1_runs["A1S2-mix-nonleg"]["nonleg_consumer_entry_global_stats"] or {}).get("per_module") or {},
        reference_stats_by_module=(layer1_runs["E2A-R-full7"]["nonleg_consumer_entry_global_stats"] or {}).get("per_module") or {},
        eps=AFFINE_EPS,
    )

    donor_groups_mix = _arm_module_groups(
        arm_name="A1S2-mix-nonleg",
        baseline_bundle=baseline_bundle,
        e1_bundle=e1_bundle,
        e2a_bundle=e2a_bundle,
    )
    guard_single = _run_single_step_boundary_capture(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        donor_groups=donor_groups_mix,
        guard_bundle=guard_bundle,
    )
    guard_transfer = _add_closure(
        _tensor_metric_gaps(
            host_case=host_bundle["case"],
            target_result=target_result,
            candidate_result=guard_single,
        ),
        host_native_reference,
    )
    guard_short = _run_short_horizon_boundary_capture(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts_first=fixed_contacts,
        donor_groups=donor_groups_mix,
        window_steps=SHORT_HORIZON_STEPS,
        guard_bundle=guard_bundle,
    )

    layer2_support = {
        "vs_E2A_R_full7": _layer2_short_horizon_delta_from_series(
            base_series=layer1_runs["A1S2-mix-nonleg"]["_consumer_used_combined"],
            guarded_series=guard_short["_consumer_used_combined"],
            reference_series=layer1_runs["E2A-R-full7"]["_consumer_used_combined"],
        ),
        "vs_E1_top3_full7": _layer2_short_horizon_delta_from_series(
            base_series=layer1_runs["A1S2-mix-nonleg"]["_consumer_used_combined"],
            guarded_series=guard_short["_consumer_used_combined"],
            reference_series=layer1_runs["E1-top3-full7"]["_consumer_used_combined"],
        ),
        "vs_target_full7": _layer2_short_horizon_delta_from_series(
            base_series=layer1_runs["A1S2-mix-nonleg"]["_consumer_used_combined"],
            guarded_series=guard_short["_consumer_used_combined"],
            reference_series=layer1_runs["target-full7"]["_consumer_used_combined"],
        ),
    }

    base_consumer_stats = layer1_runs["A1S2-mix-nonleg"]["nonleg_consumer_entry_global_stats"]
    guarded_consumer_stats = guard_short["nonleg_consumer_entry_global_stats"]
    e2a_consumer_stats = layer1_runs["E2A-R-full7"]["nonleg_consumer_entry_global_stats"]
    e1_consumer_stats = layer1_runs["E1-top3-full7"]["nonleg_consumer_entry_global_stats"]
    target_consumer_stats = layer1_runs["target-full7"]["nonleg_consumer_entry_global_stats"]
    stats_gap_pre_vs_post = {
        "combined": {
            "to_E2A_R_full7": _stats_gap_payload(
                base_stats=base_consumer_stats["combined"],
                guarded_stats=guarded_consumer_stats["combined"],
                reference_stats=e2a_consumer_stats["combined"],
            ),
            "to_E1_top3_full7": _stats_gap_payload(
                base_stats=base_consumer_stats["combined"],
                guarded_stats=guarded_consumer_stats["combined"],
                reference_stats=e1_consumer_stats["combined"],
            ),
            "to_target_full7": _stats_gap_payload(
                base_stats=base_consumer_stats["combined"],
                guarded_stats=guarded_consumer_stats["combined"],
                reference_stats=target_consumer_stats["combined"],
            ),
        },
        "per_module_to_E2A_R_full7": {
            module_name: _stats_gap_payload(
                base_stats=base_consumer_stats["per_module"][module_name],
                guarded_stats=guarded_consumer_stats["per_module"][module_name],
                reference_stats=e2a_consumer_stats["per_module"][module_name],
            )
            for module_name in consumer_order
        },
    }

    judgement = _judge_boundary_direction(
        layer0_head_pairwise=layer0_head_pairwise,
        head_mean_l2_to_refs=a1s2_mean_l2_to_refs["head_output"],
        consumer_mean_l2_to_refs=a1s2_mean_l2_to_refs["nonleg_consumer_entry"],
        guarded_transfer=guard_transfer,
        base_transfer=a1s2_mix_transfer,
        layer2_support_vs_e2a=layer2_support["vs_E2A_R_full7"],
    )

    q1_answer = "no"
    if math.isfinite(layer0_max_l2) and layer0_max_l2 > 0.0:
        q1_answer = "yes_but_near_zero" if layer0_non_discriminative else "yes"
    aggregate_delta_vs_a1s2 = float(
        _safe_float(guard_transfer.get("aggregate_transfer_score"))
        - _safe_float(a1s2_mix_transfer.get("aggregate_transfer_score"))
    )
    if judgement["case_label"] == "Case A":
        q4_answer = "yes"
    elif judgement["case_label"] == "Case B":
        q4_answer = "yes_but_upgrade_to_learned_adapter"
    elif judgement["case_label"] == "Case C":
        q4_answer = "yes_for_same_boundary_only_if_moving_directly_to_learned_adapter_or_training_constraint"
    else:
        q4_answer = "no"

    explicit_answers = {
        "q1_direct_head_output_same_host_fixed_t0_nonzero_divergence": {
            "answer": str(q1_answer),
            "max_pairwise_l2_rms": float(layer0_max_l2),
            "max_pairwise_cosine_distance": float(layer0_max_cos),
            "note": (
                "non-zero exists but remains near-zero / non-discriminative"
                if q1_answer == "yes_but_near_zero"
                else "same-host fixed t=0 already shows non-trivial pairwise spread"
                if q1_answer == "yes"
                else "pairwise divergence stays exactly zero under this metric"
            ),
        },
        "q2_a1s2_short_horizon_closest_reference": {
            "answer": {
                "head_output": str(closest_head_ref),
                "nonleg_consumer_entry": str(closest_consumer_ref),
            },
            "head_output_mean_l2_to_refs": a1s2_mean_l2_to_refs["head_output"],
            "nonleg_consumer_entry_mean_l2_to_refs": a1s2_mean_l2_to_refs["nonleg_consumer_entry"],
        },
        "q3_a1s5_affine_guard_clear_transfer_win_over_a1s2": {
            "answer": "yes" if aggregate_delta_vs_a1s2 > CLEAR_AGG_MARGIN else "no",
            "aggregate_delta_vs_A1S2_mix_nonleg": float(aggregate_delta_vs_a1s2),
            "status": str(judgement["guard_transfer_status_vs_A1S2_mix"]),
        },
        "q4_head_to_expansion_boundary_mainline": {
            "answer": str(q4_answer),
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
            "experiment": "A1-S5 direct-head-output / nonleg-consumer contract assay (probe + affine guard, no-train)",
            "mode": "merged diagnostic + runtime affine consumer-entry guard",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "offset": DEFAULT_OFFSET,
            "teacher": str(teacher),
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "layer0_mode": "same-host fixed-contacts deterministic first-forward head-output sanity diagnostic",
            "layer1_mode": "same-host same-entry same-offset short-horizon rollout on head output and nonleg consumer entry",
            "layer2_mode": "runtime per-module affine guard only on nonleg consumer entry",
            "short_horizon_steps": SHORT_HORIZON_STEPS,
            "constraints": [
                "no new training",
                "no architecture redesign",
                "no E0/E1/E2-A/E2-C/E3-A/A1-S1/A1-S2/A1-S3/A1-S4 reruns",
                "no full grid expansion",
            ],
            "goal": (
                "test whether direct_pose_head output / nonleg consumer entry carries useful contract-mismatch signal, "
                "and whether the simplest runtime affine guard can move A1S2-mix-nonleg toward a better replace-compatible basin"
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
        "a1s4_inherited": {
            "summary_json": str(A1S4_SUMMARY_JSON),
            "record_md": str(A1S4_RECORD_MD),
            "direct_conclusions": A1S4_DIRECT_INHERITED,
            "judgement": dict(a1s4_summary.get("judgement") or {}),
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
        "diagnostic_hook_definition": {
            "module_name": str(diagnostic_hook_definition["module_name"]),
            "module_class": str(diagnostic_hook_definition["module_class"]),
            "resolved_as": str(diagnostic_hook_definition["resolved_as"]),
            "why_this_boundary": str(diagnostic_hook_definition["why_this_boundary"]),
            "shape_t0_by_arm": {
                arm_name: _tensor_shape(layer0_runs[arm_name]["direct_head_output"]) for arm_name in DIAGNOSTIC_ARM_ORDER
            },
        },
        "intervention_hook_definition": {
            "resolved_modules": [
                {
                    "module_name": str(row["module_name"]),
                    "module_class": str(row["module_class"]),
                    "resolved_as": str(row["resolved_as"]),
                    "fallback_used": bool(row["fallback_used"]),
                }
                for row in intervention_hook_definition
            ],
            "why_this_boundary": (
                "intervention stays on nonleg downstream consumer entry so shared leg side is preserved as much as possible"
            ),
            "shape_t0_by_arm": {
                arm_name: {
                    "combined": _tensor_shape(layer0_runs[arm_name]["nonleg_consumer_entry_used_combined"]),
                    "per_module": _module_shape_map(
                        layer0_runs[arm_name]["nonleg_consumer_entry_used"],
                        [str(row["module_name"]) for row in intervention_hook_definition],
                    ),
                }
                for arm_name in DIAGNOSTIC_ARM_ORDER
            },
        },
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
                    "diagnostic_hook": dict(layer0_runs[arm_name]["diagnostic_hook"]),
                    "intervention_hook": dict(layer0_runs[arm_name]["intervention_hook"]),
                    "head_output_shape": _tensor_shape(layer0_runs[arm_name]["direct_head_output"]),
                    "head_output_mean_abs": float(
                        _stack_rows(layer0_runs[arm_name]["direct_head_output"]).abs().mean().item()
                    ),
                    "nonleg_consumer_entry_combined_shape": _tensor_shape(
                        layer0_runs[arm_name]["nonleg_consumer_entry_used_combined"]
                    ),
                }
                for arm_name in DIAGNOSTIC_ARM_ORDER
            },
            "head_output_pairwise_table": layer0_head_pairwise,
            "interpretation": {
                "max_pairwise_l2_rms": float(layer0_max_l2),
                "max_pairwise_cosine_distance": float(layer0_max_cos),
                "call": "non_discriminative_or_near_zero" if layer0_non_discriminative else "not_trivially_zero",
                "note": (
                    "same-host fixed t=0 head output divergence is near zero and should not be over-interpreted"
                    if layer0_non_discriminative
                    else "same-host fixed t=0 head output already shows some spread"
                ),
            },
        },
        "layer1_short_horizon": {
            "window_steps": SHORT_HORIZON_STEPS,
            "arms": {
                arm_name: {
                    "diagnostic_hook": dict(layer1_runs[arm_name]["diagnostic_hook"]),
                    "intervention_hook": dict(layer1_runs[arm_name]["intervention_hook"]),
                    "fixed_contacts_mode": str(layer1_runs[arm_name]["fixed_contacts_mode"]),
                    "head_output_global_stats": dict(layer1_runs[arm_name]["head_output_global_stats"]),
                    "nonleg_consumer_entry_global_stats": dict(layer1_runs[arm_name]["nonleg_consumer_entry_global_stats"]),
                    "step_records": list(layer1_runs[arm_name]["step_records"]),
                }
                for arm_name in DIAGNOSTIC_ARM_ORDER
            },
            "pairwise_distance_tables": pairwise_tables,
            "divergence_to_references": {
                "A1S2-mix-nonleg": a1s2_divergence,
            },
            "step_level_correlations": {
                "A1S2-mix-nonleg": a1s2_correlations,
            },
            "a1s2_mean_l2_to_refs": a1s2_mean_l2_to_refs,
        },
        "layer2_affine_guard": {
            "affine_guard_config": {
                "A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer": dict(guard_bundle["summary"]),
            },
            "fixed_transfer": {
                "A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer": {
                    "transfer": guard_transfer,
                    "delta_vs_A1S2_mix_nonleg": _transfer_delta(guard_transfer, a1s2_mix_transfer),
                    "delta_vs_E1_top3_full7": _transfer_delta(guard_transfer, e1_top3_reference),
                    "delta_vs_E2A_R_full7": _transfer_delta(guard_transfer, e2a_full7_reference),
                    "delta_vs_target_full7": _transfer_delta(guard_transfer, target_transfer_reference),
                }
            },
            "short_horizon_support": {
                "A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer": {
                    "global_stats": dict(guard_short["nonleg_consumer_entry_global_stats"]),
                    "step_records": list(guard_short["step_records"]),
                    "vs_E2A_R_full7": layer2_support["vs_E2A_R_full7"],
                    "vs_E1_top3_full7": layer2_support["vs_E1_top3_full7"],
                    "vs_target_full7": layer2_support["vs_target_full7"],
                    "stats_gap_pre_vs_post": stats_gap_pre_vs_post,
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
