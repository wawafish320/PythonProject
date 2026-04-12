#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _build_step_ctx,
    _case_bundle,
    _prepare_fixed_offset_context,
    _restore_weight_swap,
    _run_single_step,
    _temporary_weight_swap,
)
from tools.audit_cp015_tailk7_plan_shortcut_takeover_mechanism import (  # noqa: E402
    _branch_layout,
    _classify_sensitivity,
    _first_linear,
    _head_zero_branch_deltas,
    _jacobian_sensitivity,
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
    _direct_head_proxy,
    _safe_float,
    _tensor_metric_gaps,
)
from train import posttrain  # noqa: E402


RUN_DATE = "20260408"
RUN_NAME = "cp015_tailk_support_scope_isolation_e1"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"

TOP7_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json"
)
TOP7_BASETRAIN_EPOCH014 = (
    ROOT
    / "models"
    / "cp015_phasecd_tailk_probe_20260331"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401"
    / "ckpt_epoch_014.pth"
)
TOP7_STAGE6_TAILFIX_CONFIG = (
    ROOT
    / "config"
    / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json"
)
TOP7_STAGE6_TAILFIX_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage6_tailfix_20260401"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth"
)
TOP7_STAGE6_TAILFIX_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage6_tailfix_20260401"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.json"
)
STAGE70A_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_70a_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70a_lr3e4_from_ep014center_20260328.json"
)
TOP7_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
TOP7_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.json"
)
TOP7_70A_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "eval_model_source"
    / "Walk_F_freerun_cycles.json"
)

TOP3_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json"
)
TOP3_BASETRAIN_EPOCH014 = (
    ROOT
    / "models"
    / "cp015_phasecd_tailk_probe_20260331"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408"
    / "ckpt_epoch_014.pth"
)
TOP3_STAGE6_TAILFIX_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk3_rankmix_tw020_e1_20260408.pth"
)
TOP3_STAGE6_TAILFIX_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk3_rankmix_tw020_e1_20260408.json"
)
TOP3_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth"
)
TOP3_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.json"
)
TOP3_70A_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408"
    / "eval_model_source"
    / "Walk_F_freerun_cycles.json"
)


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    support_scope: str
    provenance: str
    basetrain_config: Path
    basetrain_epoch014_ckpt: Path
    stage6_tailfix_config: Path
    stage6_tailfix_ckpt: Path
    stage6_tailfix_log: Path
    stage70a_config: Path
    stage70a_ckpt: Path
    stage70a_log: Path
    stage70a_eval: Path
    notes: str


ARMS: Tuple[ArmSpec, ...] = (
    ArmSpec(
        arm="top7",
        support_scope="top7",
        provenance="reuse_existing",
        basetrain_config=TOP7_BASETRAIN_CONFIG,
        basetrain_epoch014_ckpt=TOP7_BASETRAIN_EPOCH014,
        stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
        stage6_tailfix_ckpt=TOP7_STAGE6_TAILFIX_CKPT,
        stage6_tailfix_log=TOP7_STAGE6_TAILFIX_LOG,
        stage70a_config=STAGE70A_CONFIG,
        stage70a_ckpt=TOP7_70A_CKPT,
        stage70a_log=TOP7_70A_LOG,
        stage70a_eval=TOP7_70A_EVAL,
        notes="Current canonical top7 support chain reused as E1-A.",
    ),
    ArmSpec(
        arm="top3",
        support_scope="top3",
        provenance="new_matched_control",
        basetrain_config=TOP3_BASETRAIN_CONFIG,
        basetrain_epoch014_ckpt=TOP3_BASETRAIN_EPOCH014,
        stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
        stage6_tailfix_ckpt=TOP3_STAGE6_TAILFIX_CKPT,
        stage6_tailfix_log=TOP3_STAGE6_TAILFIX_LOG,
        stage70a_config=STAGE70A_CONFIG,
        stage70a_ckpt=TOP3_70A_CKPT,
        stage70a_log=TOP3_70A_LOG,
        stage70a_eval=TOP3_70A_EVAL,
        notes="Matched-control arm: top7 rankmix tw020 pipeline cloned, support scope only reduced to top3.",
    ),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory_row(arm: ArmSpec) -> Dict[str, Any]:
    return {
        "arm": arm.arm,
        "support_scope": arm.support_scope,
        "provenance": arm.provenance,
        "basetrain_config": str(arm.basetrain_config),
        "basetrain_epoch014_ckpt": str(arm.basetrain_epoch014_ckpt),
        "stage6_tailfix_config": str(arm.stage6_tailfix_config),
        "stage6_tailfix_ckpt": str(arm.stage6_tailfix_ckpt),
        "stage6_tailfix_log": str(arm.stage6_tailfix_log),
        "stage70a_config": str(arm.stage70a_config),
        "stage70a_ckpt": str(arm.stage70a_ckpt),
        "stage70a_log": str(arm.stage70a_log),
        "stage70a_eval": str(arm.stage70a_eval),
        "exists": {
            "basetrain_config": arm.basetrain_config.is_file(),
            "basetrain_epoch014_ckpt": arm.basetrain_epoch014_ckpt.is_file(),
            "stage6_tailfix_config": arm.stage6_tailfix_config.is_file(),
            "stage6_tailfix_ckpt": arm.stage6_tailfix_ckpt.is_file(),
            "stage6_tailfix_log": arm.stage6_tailfix_log.is_file(),
            "stage70a_config": arm.stage70a_config.is_file(),
            "stage70a_ckpt": arm.stage70a_ckpt.is_file(),
            "stage70a_log": arm.stage70a_log.is_file(),
            "stage70a_eval": arm.stage70a_eval.is_file(),
        },
        "notes": arm.notes,
    }


def _config_diff_rows(top7_cfg: Mapping[str, Any], top3_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "field": "rot_local_tail_k",
            "top7": int(top7_cfg["rot_local_tail_k"]),
            "top3": int(top3_cfg["rot_local_tail_k"]),
            "same_except_scope": True,
        },
        {
            "field": "freerun_stage_schedule.phase_b.core.rot_local_tail_k",
            "top7": int(top7_cfg["freerun_stage_schedule"][1]["loss_groups"]["core"]["rot_local_tail_k"]),
            "top3": int(top3_cfg["freerun_stage_schedule"][1]["loss_groups"]["core"]["rot_local_tail_k"]),
            "same_except_scope": True,
        },
        {
            "field": "freerun_stage_schedule.phase_c.core.rot_local_tail_k",
            "top7": int(top7_cfg["freerun_stage_schedule"][2]["loss_groups"]["core"]["rot_local_tail_k"]),
            "top3": int(top3_cfg["freerun_stage_schedule"][2]["loss_groups"]["core"]["rot_local_tail_k"]),
            "same_except_scope": True,
        },
        {
            "field": "freerun_stage_schedule.phase_d.core.rot_local_tail_k",
            "top7": int(top7_cfg["freerun_stage_schedule"][3]["loss_groups"]["core"]["rot_local_tail_k"]),
            "top3": int(top3_cfg["freerun_stage_schedule"][3]["loss_groups"]["core"]["rot_local_tail_k"]),
            "same_except_scope": True,
        },
        {
            "field": "rot_local_tail_reduce",
            "top7": str(top7_cfg["rot_local_tail_reduce"]),
            "top3": str(top3_cfg["rot_local_tail_reduce"]),
            "same_except_scope": False,
        },
        {
            "field": "rot_local_tail_uniform_mix",
            "top7": float(top7_cfg["rot_local_tail_uniform_mix"]),
            "top3": float(top3_cfg["rot_local_tail_uniform_mix"]),
            "same_except_scope": False,
        },
        {
            "field": "rot_local_tail_rank_mix",
            "top7": float(top7_cfg["rot_local_tail_rank_mix"]),
            "top3": float(top3_cfg["rot_local_tail_rank_mix"]),
            "same_except_scope": False,
        },
        {
            "field": "save_fit_ckpt_epochs",
            "top7": str(top7_cfg["save_fit_ckpt_epochs"]),
            "top3": str(top3_cfg["save_fit_ckpt_epochs"]),
            "same_except_scope": False,
        },
    ]


def _capture_direct_head_input(
    bundle: Mapping[str, Any],
    prep_ctx: Mapping[str, Any],
    *,
    fixed_contacts: Optional[torch.Tensor],
    donor_bundle: Optional[Mapping[str, Any]] = None,
    weight_swap_modules: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    case = bundle["case"]
    trainer = case["trainer"]
    model = trainer.model
    if model is None:
        raise RuntimeError(f"{case['case_name']}: missing model")
    donor_model = donor_bundle["case"]["trainer"].model if donor_bundle is not None else None
    if weight_swap_modules and donor_model is None:
        raise RuntimeError("weight swap requested but donor bundle/model missing")

    linear_name, linear = _first_linear(model)
    captured: Dict[str, Any] = {}
    orig_prepare_contacts = posttrain._prepare_rollout_contacts_input
    backups: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []

    def _prepare_contacts_override(
        trainer_: Any,
        model_: Any,
        *,
        motion_t: torch.Tensor,
        pose_hist_t: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        _ = trainer_, model_, motion_t, pose_hist_t
        if fixed_contacts is None:
            return orig_prepare_contacts(trainer_, model_, motion_t=motion_t, pose_hist_t=pose_hist_t)
        return fixed_contacts.detach().clone().to(device=motion_t.device, dtype=motion_t.dtype)

    def _pre_hook(_mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0].detach().clone().reshape(-1, int(inputs[0].shape[-1])).cpu()
        captured["direct_head_input"] = x

    handle = linear.register_forward_pre_hook(_pre_hook)
    try:
        posttrain._prepare_rollout_contacts_input = _prepare_contacts_override
        if weight_swap_modules:
            backups = _temporary_weight_swap(
                target_model=model,
                donor_model=donor_model,
                module_names=weight_swap_modules,
            )
        ctx = _build_step_ctx(bundle, prep_ctx)
        with torch.no_grad():
            posttrain._lambda_rollout_unroll_single_step(t=0, ctx=ctx)
    finally:
        handle.remove()
        posttrain._prepare_rollout_contacts_input = orig_prepare_contacts
        if backups:
            _restore_weight_swap(backups)

    if "direct_head_input" not in captured:
        raise RuntimeError("failed to capture direct_pose_head input")
    return {
        "linear_name": linear_name,
        "layout": _branch_layout(model, linear),
        "x_all": captured["direct_head_input"],
    }


def _lower_is_better_closure(candidate: Any, bad: Any, good: Any) -> float:
    cand = _safe_float(candidate)
    bad_v = _safe_float(bad)
    good_v = _safe_float(good)
    if not math.isfinite(cand) or not math.isfinite(bad_v) or not math.isfinite(good_v):
        return float("nan")
    denom = bad_v - good_v
    if abs(denom) <= 1e-12:
        return float("nan")
    return float((bad_v - cand) / denom)


def _normality_label(plan_ratio: float, plan_delta: float) -> str:
    if math.isfinite(plan_ratio) and plan_ratio <= 0.80 and math.isfinite(plan_delta) and plan_delta <= 0.01:
        return "nonplan_owned"
    if (math.isfinite(plan_ratio) and plan_ratio >= 1.25) or (math.isfinite(plan_delta) and plan_delta > 0.01):
        return "plan_compensatory"
    return "mixed"


def _normality_readout(
    *,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    donor_bundle: Mapping[str, Any],
    host_bad_reference: Optional[Mapping[str, Any]] = None,
    target_reference: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    captured = _capture_direct_head_input(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        donor_bundle=donor_bundle,
        weight_swap_modules=DIRECT_BRANCH_MODULES,
    )
    x_all = captured["x_all"]
    layout = captured["layout"]
    selected = [0]
    jacobian = _jacobian_sensitivity(
        case=host_bundle["case"],
        x_all=x_all,
        selected=selected,
        layout=layout,
        max_steps=1,
        batch_size=1,
    )
    zero_deltas = _head_zero_branch_deltas(
        case=host_bundle["case"],
        x_all=x_all,
        selected=selected,
        layout=layout,
        batch_size=1,
    )
    plan_ratio = _safe_float(((jacobian.get("ratios") or {}).get("plan_over_direct_feat")))
    meas_ratio = _safe_float(((jacobian.get("ratios") or {}).get("meas_over_direct_feat")))
    plan_delta = _safe_float((((zero_deltas.get("plan") or {}).get("direct_output_delta_geolocal_deg") or {}).get("mean")))
    direct_delta = _safe_float(
        (((zero_deltas.get("direct_feat") or {}).get("direct_output_delta_geolocal_deg") or {}).get("mean"))
    )
    meas_delta = _safe_float((((zero_deltas.get("meas") or {}).get("direct_output_delta_geolocal_deg") or {}).get("mean")))
    out = {
        "assay_mode": "deterministic single-step first-forward in fixed replace host",
        "linear_name": captured["linear_name"],
        "jacobian": jacobian,
        "zero_branch_deltas": zero_deltas,
        "plan_over_direct_sensitivity": plan_ratio,
        "meas_over_direct_sensitivity": meas_ratio,
        "plan_zero_delta_geolocal_deg": plan_delta,
        "direct_zero_delta_geolocal_deg": direct_delta,
        "meas_zero_delta_geolocal_deg": meas_delta,
        "sensitivity_labels": _classify_sensitivity(jacobian),
        "conclusion_label": _normality_label(plan_ratio, plan_delta),
    }
    if host_bad_reference is not None and target_reference is not None:
        bad_ratio = _safe_float(host_bad_reference.get("plan_over_direct_sensitivity"))
        good_ratio = _safe_float(target_reference.get("plan_over_direct_sensitivity"))
        bad_delta = _safe_float(host_bad_reference.get("plan_zero_delta_geolocal_deg"))
        good_delta = _safe_float(target_reference.get("plan_zero_delta_geolocal_deg"))
        out["normality_closure"] = {
            "plan_over_direct_sensitivity": _lower_is_better_closure(plan_ratio, bad_ratio, good_ratio),
            "plan_zero_delta_geolocal_deg": _lower_is_better_closure(plan_delta, bad_delta, good_delta),
        }
        out["aggregate_normality_score"] = _safe_float(
            (
                _safe_float(out["normality_closure"]["plan_over_direct_sensitivity"])
                + _safe_float(out["normality_closure"]["plan_zero_delta_geolocal_deg"])
            )
            / 2.0
        )
    return out


def _normality_delta(top3: Mapping[str, Any], top7: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "plan_over_direct_sensitivity": _safe_float(top3.get("plan_over_direct_sensitivity"))
        - _safe_float(top7.get("plan_over_direct_sensitivity")),
        "plan_zero_delta_geolocal_deg": _safe_float(top3.get("plan_zero_delta_geolocal_deg"))
        - _safe_float(top7.get("plan_zero_delta_geolocal_deg")),
        "aggregate_normality_score": _safe_float(top3.get("aggregate_normality_score"))
        - _safe_float(top7.get("aggregate_normality_score")),
        "label_top7": top7.get("conclusion_label"),
        "label_top3": top3.get("conclusion_label"),
    }


def _transfer_delta(top3: Mapping[str, Any], top7: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "out_direct_gap",
        "dir_base_gap",
        "dir_leg_gap",
        "dir_nonleg_gap",
        "out_direct_closure_ratio",
        "dir_base_closure_ratio",
        "dir_leg_closure_ratio",
        "dir_nonleg_closure_ratio",
        "aggregate_transfer_score",
    )
    return {key: _safe_float(top3.get(key)) - _safe_float(top7.get(key)) for key in keys}


def _proxy_delta(top3: Mapping[str, Any], top7: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "plan_norm_per_dim": _safe_float(top3["blocks"]["plan"]["norm_per_dim"])
        - _safe_float(top7["blocks"]["plan"]["norm_per_dim"]),
        "direct_norm_per_dim": _safe_float(top3["blocks"]["direct"]["norm_per_dim"])
        - _safe_float(top7["blocks"]["direct"]["norm_per_dim"]),
        "meas_norm_per_dim": _safe_float(top3["blocks"]["meas"]["norm_per_dim"])
        - _safe_float(top7["blocks"]["meas"]["norm_per_dim"]),
        "plan_over_direct": _safe_float(top3["ratios"]["plan_over_direct"])
        - _safe_float(top7["ratios"]["plan_over_direct"]),
        "plan_over_meas": _safe_float(top3["ratios"]["plan_over_meas"])
        - _safe_float(top7["ratios"]["plan_over_meas"]),
        "plan_over_direct_plus_meas": _safe_float(top3["ratios"]["plan_over_direct_plus_meas"])
        - _safe_float(top7["ratios"]["plan_over_direct_plus_meas"]),
    }


def _improves_transfer(delta: Mapping[str, Any]) -> bool:
    closure_improved = sum(
        1
        for key in (
            "out_direct_closure_ratio",
            "dir_base_closure_ratio",
            "dir_leg_closure_ratio",
            "dir_nonleg_closure_ratio",
        )
        if _safe_float(delta.get(key)) > 0.03
    )
    return closure_improved >= 3 and _safe_float(delta.get("aggregate_transfer_score")) > 0.08


def _label_rank(label: Any) -> int:
    order = {
        "plan_compensatory": 0,
        "mixed": 1,
        "nonplan_owned": 2,
    }
    return order.get(str(label), -1)


def _improves_normality(delta: Mapping[str, Any]) -> bool:
    return (
        _label_rank(delta.get("label_top3")) > _label_rank(delta.get("label_top7"))
        or _safe_float(delta.get("aggregate_normality_score")) > 0.10
        or (
            _safe_float(delta.get("plan_over_direct_sensitivity")) < -0.10
            and _safe_float(delta.get("plan_zero_delta_geolocal_deg")) < -0.002
        )
    )


def _lever_judgement(transfer_delta: Mapping[str, Any], normality_delta: Mapping[str, Any]) -> Dict[str, Any]:
    transfer_improved = _improves_transfer(transfer_delta)
    normality_improved = _improves_normality(normality_delta)
    if transfer_improved and normality_improved:
        label = "strong lever / causal contributor"
        next_step = "continue_to_E2_curriculum_path_shaping"
        why = "top3 improves both fixed transferability and replace-entry normality under matched conditions"
    elif transfer_improved or normality_improved:
        label = "partial lever"
        next_step = "continue_to_E2_curriculum_path_shaping"
        why = "support scope helps, but the improvement is incomplete along at least one primary axis"
    else:
        label = "insufficient"
        next_step = "direct_to_E2_curriculum_path_shaping"
        why = "scope isolation alone does not materially repair transferability or replace-entry normality"
    return {
        "support_scope_judgement": label,
        "transfer_improved": transfer_improved,
        "normality_improved": normality_improved,
        "next_step": next_step,
        "why": why,
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    required = [
        DEFAULT_TEACHER,
        BASELINE_REPLACE_CONFIG,
        BASELINE_REPLACE_CKPT,
        BASELINE_REPLACE_EVAL,
        COADAPT_HOST_CONFIG,
        COADAPT_HOST_CKPT,
        COADAPT_HOST_EVAL,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("[FATAL] missing fixed-context artifact(s):\n" + "\n".join(missing))

    arm_inventory = [_inventory_row(arm) for arm in ARMS]
    arm_missing = []
    for row in arm_inventory:
        for key, exists in row["exists"].items():
            if not exists:
                arm_missing.append(f"{row['arm']}::{key}::{row[key]}")
    if arm_missing:
        raise SystemExit("[FATAL] missing arm artifact(s):\n" + "\n".join(arm_missing))

    top7_cfg = _load_json(TOP7_BASETRAIN_CONFIG)
    top3_cfg = _load_json(TOP3_BASETRAIN_CONFIG)
    config_diff = _config_diff_rows(top7_cfg, top3_cfg)

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

    prep_base = _prepare_fixed_offset_context(baseline_bundle, offset=DEFAULT_OFFSET)
    prep_host = _prepare_fixed_offset_context(host_bundle, offset=DEFAULT_OFFSET)
    baseline_native = _run_single_step(baseline_bundle, prep_base, fixed_contacts=None)
    fixed_contacts = baseline_native["inputs"]["contacts"]
    host_native = _run_single_step(host_bundle, prep_host, fixed_contacts=fixed_contacts)
    target_result = _run_single_step(
        host_bundle,
        prep_host,
        fixed_contacts=fixed_contacts,
        weight_swap_modules=DIRECT_BRANCH_MODULES,
        donor_bundle=baseline_bundle,
    )

    host_gaps = _tensor_metric_gaps(
        host_case=host_bundle["case"],
        target_result=target_result,
        candidate_result=host_native,
    )

    host_native_normality = _normality_readout(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        donor_bundle=host_bundle,
    )
    target_normality = _normality_readout(
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        donor_bundle=baseline_bundle,
    )

    arm_results: Dict[str, Any] = {}
    for arm in ARMS:
        donor_bundle = {
            "case": _case_bundle(
                case_name=f"{arm.arm}_raw70a",
                ckpt_path=arm.stage70a_ckpt,
                eval_json_path=arm.stage70a_eval,
                teacher_path=teacher,
                config_path=arm.stage70a_config,
                device_pref="cpu",
            )["case"]
        }
        candidate_result = _run_single_step(
            host_bundle,
            prep_host,
            fixed_contacts=fixed_contacts,
            weight_swap_modules=DIRECT_BRANCH_MODULES,
            donor_bundle=donor_bundle,
        )
        transfer = _add_closure(
            _tensor_metric_gaps(
                host_case=host_bundle["case"],
                target_result=target_result,
                candidate_result=candidate_result,
            ),
            host_gaps,
        )
        proxy = _direct_head_proxy(donor_bundle["case"]["trainer"].model)
        normality = _normality_readout(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            donor_bundle=donor_bundle,
            host_bad_reference=host_native_normality,
            target_reference=target_normality,
        )
        arm_results[arm.arm] = {
            "arm": arm.arm,
            "support_scope": arm.support_scope,
            "stage70a_ckpt": str(arm.stage70a_ckpt),
            "stage70a_eval": str(arm.stage70a_eval),
            "stage70a_log": str(arm.stage70a_log),
            "transfer": transfer,
            "replace_normality": normality,
            "proxy_telemetry": proxy,
        }

    top7_result = arm_results["top7"]
    top3_result = arm_results["top3"]
    transfer_delta = _transfer_delta(top3_result["transfer"], top7_result["transfer"])
    normality_delta = _normality_delta(top3_result["replace_normality"], top7_result["replace_normality"])
    proxy_delta = _proxy_delta(top3_result["proxy_telemetry"], top7_result["proxy_telemetry"])
    judgement = _lever_judgement(transfer_delta, normality_delta)

    proxy_value_add = "supportive_readout"
    if abs(_safe_float(proxy_delta["plan_over_direct"])) < 0.003 and abs(
        _safe_float(normality_delta["aggregate_normality_score"])
    ) > 0.10:
        proxy_value_add = "almost_no_incremental_information"

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "E1 support scope isolation",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "coadapt host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic single-step first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "strict_constraints": [
                "no new attribution mainline",
                "no planner semantics mainline",
                "no new large sweep",
                "top7 vs top3 only",
            ],
        },
        "inherited_conclusions": [
            "root cause not in planner semantics mainline",
            "root cause not in replace-entry external rollout state",
            "root cause not in contacts_in_t",
            "earliest semantic split at direct_pose_head boundary",
            "first-step split most resembles whole direct-branch contract mismatch",
            "direct_pose_head is earliest boundary / necessary anchor but not standalone sufficient",
            "7-module direct-branch joint contract is required for high closure",
            "baseline 7-module direct branch can transfer into coadapt context",
            "current top7 path is already off by stage6 exact epoch013",
            "epoch014 and epoch015 are better than final tailfix/70a on transferability",
            "largest observed top7 deterioration is epoch015 -> stage6 tailfix",
            "direct_pose_head.0 input-block allocation proxy is only a coarse concurrent readout",
        ],
        "strict_comparability_note": {
            "existing_top3_control_reused": False,
            "why_not": "Existing top3 control_denseckpt artifacts use default rot_local_tail_reduce=flat, so they are not strict matches for the top7 rankmix tw020 lane.",
            "matched_top3_control_added": True,
        },
        "arm_inventory": arm_inventory,
        "key_config_diff_top7_vs_top3": config_diff,
        "fixed_context_reference": {
            "baseline_replace_native": {
                "config": str(BASELINE_REPLACE_CONFIG),
                "ckpt": str(BASELINE_REPLACE_CKPT),
                "eval": str(BASELINE_REPLACE_EVAL),
            },
            "coadapt_host": {
                "config": str(COADAPT_HOST_CONFIG),
                "ckpt": str(COADAPT_HOST_CKPT),
                "eval": str(COADAPT_HOST_EVAL),
            },
            "host_gap_to_target": host_gaps,
            "host_native_replace_normality": host_native_normality,
            "target_replace_normality": target_normality,
        },
        "final_70a_results": arm_results,
        "delta_summary_top3_minus_top7": {
            "transfer": transfer_delta,
            "replace_normality": normality_delta,
            "proxy_telemetry": proxy_delta,
        },
        "judgement": {
            **judgement,
            "direct_pose_head0_proxy_value": proxy_value_add,
            "proxy_role": "supportive readout" if proxy_value_add == "supportive_readout" else "almost no added information",
        },
        "explicit_answers": {
            "q1_top3_more_replace_transferable_final_ckpt": bool(
                _safe_float(top3_result["transfer"]["aggregate_transfer_score"])
                > _safe_float(top7_result["transfer"]["aggregate_transfer_score"]) + 0.08
            ),
            "q2_top3_enters_replace_more_normally": bool(
                _improves_normality(normality_delta)
            ),
            "q3_support_scope_judgement": judgement["support_scope_judgement"],
            "q4_next_best_step": (
                "directly_do_E2_curriculum_path_shaping"
                if judgement["next_step"].endswith("E2_curriculum_path_shaping")
                else "continue_finer_E1_isolation"
            ),
        },
        "preliminary_next_step_recommendation": {
            "recommended": "E2 curriculum/path-shaping"
            if judgement["next_step"].endswith("E2_curriculum_path_shaping")
            else "continue finer E1 isolation",
            "why": judgement["why"],
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
