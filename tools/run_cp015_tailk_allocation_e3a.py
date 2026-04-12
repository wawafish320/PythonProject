#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _case_bundle,
    _prepare_fixed_offset_context,
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
    _direct_head_proxy,
    _safe_float,
    _tensor_metric_gaps,
)
from tools.run_cp015_tailk_curriculum_e2a import (  # noqa: E402
    ArmSpec,
    _arm_stage_result,
    _inventory_row,
    _normality_delta,
    _normality_improved,
    _normality_probe_discriminative,
    _proxy_delta,
    _transfer_delta,
)
from tools.run_cp015_tailk_legfirst_e2c import (  # noqa: E402
    ARMS as PRIOR_ARMS,
    _closure_gain,
    _gap_reduction,
    _retention_ratio,
)
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP7_STAGE6_TAILFIX_CONFIG,
)


RUN_DATE = "20260408"
RUN_NAME = "cp015_tailk_allocation_e3a"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"

E3A_BASERUN = (
    "exp_phase_DirectBranch_v1_d1_cp015_tailk7_e3a_rf_readoutfirst_rankmix_tw020_"
    "corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408"
)
E3A_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_e3a_rf_readoutfirst_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json"
)
E3A_BASETRAIN_EPOCH014 = ROOT / "models" / "cp015_phasecd_tailk_probe_20260331" / E3A_BASERUN / "ckpt_epoch_014.pth"
E3A_STAGE6_TAILFIX_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_e3a_rf_stage6_tailfix_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_e3a_rf_20260408.pth"
)
E3A_STAGE6_TAILFIX_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_e3a_rf_stage6_tailfix_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_e3a_rf_20260408.json"
)
E3A_STAGE6_TAILFIX_EVAL = OUT_ROOT / "stage6_tailfix" / "stage6_freerun" / "Walk_F_freerun_cycles.json"
E3A_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_e3a_rf_stage70a_from_tailfix_20260408"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_e3a_rf_stage6tailfix_20260408.pth"
)
E3A_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_e3a_rf_stage70a_from_tailfix_20260408"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_e3a_rf_stage6tailfix_20260408.json"
)
E3A_70A_EVAL = OUT_ROOT / "stage70a" / "eval_model_source" / "Walk_F_freerun_cycles.json"

E3A_ARM = ArmSpec(
    arm="E3A-RF",
    provenance="new_allocation_arm",
    support_schedule="7 -> 7 -> 7 (readout-first allocation)",
    basetrain_config=E3A_BASETRAIN_CONFIG,
    basetrain_epoch014_ckpt=E3A_BASETRAIN_EPOCH014,
    stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
    stage6_tailfix_ckpt=E3A_STAGE6_TAILFIX_CKPT,
    stage6_tailfix_log=E3A_STAGE6_TAILFIX_LOG,
    stage6_tailfix_eval=E3A_STAGE6_TAILFIX_EVAL,
    stage70a_config=STAGE70A_CONFIG,
    stage70a_ckpt=E3A_70A_CKPT,
    stage70a_log=E3A_70A_LOG,
    stage70a_eval=E3A_70A_EVAL,
    notes=(
        "Matched E3-A arm: keep top7 support and loss family fixed, but train direct readout slice first, "
        "then restore late hidden block, then return to full direct-head co-adaptation."
    ),
)

ARMS = tuple(list(PRIOR_ARMS) + [E3A_ARM])

TRANSFER_FIELDS = (
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

DIRECT_BRANCH_FAMILY_MAPPING: Dict[str, Any] = {
    "basetrain_shared_head_top7_matched": {
        "head": {
            "modules": ["direct_pose_head"],
            "parameter_prefixes": ["direct_pose_head.0", "direct_pose_head.3"],
            "meaning": "shared hidden layers / main head block",
        },
        "adapters": {
            "modules": [],
            "parameter_prefixes": [],
            "meaning": "not instantiated in canonical matched basetrain config",
        },
        "readouts": {
            "modules": ["direct_pose_head"],
            "parameter_prefixes": ["direct_pose_head.6"],
            "meaning": "monolithic direct head final linear readout",
        },
        "notes": [
            "E3A-RF basetrain uses the canonical shared-head direct branch; no standalone adapters are instantiated.",
            "phase_a trainable subset is therefore effectively readout-only inside direct_pose_head.",
            "phase_b restores the late hidden block (direct_pose_head.3.*) before full unfreeze in phase_c/phase_d.",
        ],
    },
    "stage6_stage70a_transfer_contract": {
        "head": {
            "modules": ["direct_pose_head"],
            "parameter_prefixes": ["direct_pose_head."],
            "meaning": "shared split-head trunk / earliest anchor",
        },
        "adapters": {
            "modules": ["direct_pose_arm_proj", "direct_pose_else_proj"],
            "parameter_prefixes": ["direct_pose_arm_proj.", "direct_pose_else_proj."],
            "meaning": "nonleg branch adapters / amplifiers inside the 7-module contract",
        },
        "readouts": {
            "modules": ["direct_pose_out_leg", "direct_pose_out_arm", "direct_pose_out_else", "direct_pose_leg_head"],
            "parameter_prefixes": [
                "direct_pose_out_leg.",
                "direct_pose_out_arm.",
                "direct_pose_out_else.",
                "direct_pose_leg_head.",
            ],
            "meaning": "leg/nonleg direct readout heads inside the 7-module contract",
        },
        "full_7module_set": list(DIRECT_BRANCH_MODULES),
    },
}


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_core(cfg: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    stages = cfg["freerun_stage_schedule"]
    loss_groups = stages[idx].get("loss_groups") or {}
    core = loss_groups.get("core") if isinstance(loss_groups, Mapping) else None
    return core if isinstance(core, Mapping) else {}


def _stage_params(cfg: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    stage = cfg["freerun_stage_schedule"][idx]
    params = stage.get("params")
    return params if isinstance(params, Mapping) else {}


def _config_diff_rows(base_cfg: Mapping[str, Any], e3a_cfg: Mapping[str, Any]) -> list[Dict[str, Any]]:
    base_phase_b = _stage_core(base_cfg, 1)
    base_phase_c = _stage_core(base_cfg, 2)
    base_phase_d = _stage_core(base_cfg, 3)
    phase_a_params = _stage_params(e3a_cfg, 0)
    phase_b_params = _stage_params(e3a_cfg, 1)
    phase_c_params = _stage_params(e3a_cfg, 2)
    phase_d_params = _stage_params(e3a_cfg, 3)
    return [
        {
            "field": "rot_local_tail_k",
            "baseline_top7": int(base_cfg["rot_local_tail_k"]),
            "E3A-RF": int(e3a_cfg["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_b.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_b["rot_local_tail_k"]),
            "E3A-RF": int(_stage_core(e3a_cfg, 1)["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_c.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_c["rot_local_tail_k"]),
            "E3A-RF": int(_stage_core(e3a_cfg, 2)["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_d.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_d["rot_local_tail_k"]),
            "E3A-RF": int(_stage_core(e3a_cfg, 3)["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_a.params.direct_pose_head_train_scope",
            "baseline_top7": None,
            "E3A-RF": phase_a_params.get("direct_pose_head_train_scope"),
            "meaning": "early freeze main head / readout-first allocation",
        },
        {
            "field": "freerun_stage_schedule.phase_b.params.direct_pose_head_train_scope",
            "baseline_top7": None,
            "E3A-RF": phase_b_params.get("direct_pose_head_train_scope"),
            "meaning": "mid ramp / late-head restore",
        },
        {
            "field": "freerun_stage_schedule.phase_c.params.direct_pose_head_train_scope",
            "baseline_top7": None,
            "E3A-RF": phase_c_params.get("direct_pose_head_train_scope"),
            "meaning": "return to full direct-branch co-adaptation",
        },
        {
            "field": "freerun_stage_schedule.phase_d.params.direct_pose_head_train_scope",
            "baseline_top7": None,
            "E3A-RF": phase_d_params.get("direct_pose_head_train_scope"),
            "meaning": "late full direct-branch target",
        },
        {
            "field": "save_fit_ckpt_epochs",
            "baseline_top7": str(base_cfg["save_fit_ckpt_epochs"]),
            "E3A-RF": str(e3a_cfg["save_fit_ckpt_epochs"]),
            "meaning": "kept fixed",
        },
    ]


def _strictly_higher(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return _safe_float(candidate.get("aggregate_transfer_score")) > _safe_float(reference.get("aggregate_transfer_score"))


def _clearly_higher(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, margin: float = 0.03) -> bool:
    return _safe_float(candidate.get("aggregate_transfer_score")) > _safe_float(reference.get("aggregate_transfer_score")) + margin


def _stage_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    return {key: _safe_float(candidate.get(key)) - _safe_float(reference.get(key)) for key in TRANSFER_FIELDS}


def _best_prior_score(*rows: Mapping[str, Any]) -> float:
    vals = [_safe_float(row.get("aggregate_transfer_score")) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else float("nan")


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

    inventory = [_inventory_row(arm) for arm in ARMS]
    arm_missing: list[str] = []
    for row in inventory:
        for key, exists in row["exists"].items():
            if exists is False:
                arm_missing.append(f"{row['arm']}::{key}::{row[key]}")
    if arm_missing:
        raise SystemExit("[FATAL] missing arm artifact(s):\n" + "\n".join(arm_missing))

    top7_cfg = _load_json(PRIOR_ARMS[0].basetrain_config)
    e3a_cfg = _load_json(E3A_BASETRAIN_CONFIG)
    config_diff = _config_diff_rows(top7_cfg, e3a_cfg)

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
    host_native_normality = _arm_stage_result(
        name="host_native_bad_reference",
        ckpt_path=COADAPT_HOST_CKPT,
        eval_json_path=COADAPT_HOST_EVAL,
        config_path=COADAPT_HOST_CONFIG,
        teacher_path=teacher,
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        target_result=target_result,
        host_gaps=host_gaps,
        host_native_normality=None,
        target_normality=None,
        include_normality=True,
    )["replace_normality"]
    target_normality = _arm_stage_result(
        name="baseline_transplant_target",
        ckpt_path=BASELINE_REPLACE_CKPT,
        eval_json_path=BASELINE_REPLACE_EVAL,
        config_path=BASELINE_REPLACE_CONFIG,
        teacher_path=teacher,
        host_bundle=host_bundle,
        prep_host=prep_host,
        fixed_contacts=fixed_contacts,
        target_result=target_result,
        host_gaps=host_gaps,
        host_native_normality=None,
        target_normality=None,
        include_normality=True,
    )["replace_normality"]

    final_70a_results: Dict[str, Any] = {}
    for arm in ARMS:
        final_70a_results[arm.arm] = _arm_stage_result(
            name=f"{arm.arm}_70a",
            ckpt_path=arm.stage70a_ckpt,
            eval_json_path=arm.stage70a_eval,
            config_path=arm.stage70a_config,
            teacher_path=teacher,
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            target_result=target_result,
            host_gaps=host_gaps,
            host_native_normality=host_native_normality,
            target_normality=target_normality,
            include_normality=True,
        )

    stage6_results = {
        "E3A-RF": _arm_stage_result(
            name="E3A-RF_stage6_tailfix",
            ckpt_path=E3A_STAGE6_TAILFIX_CKPT,
            eval_json_path=E3A_STAGE6_TAILFIX_EVAL,
            config_path=TOP7_STAGE6_TAILFIX_CONFIG,
            teacher_path=teacher,
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            target_result=target_result,
            host_gaps=host_gaps,
            host_native_normality=None,
            target_normality=None,
            include_normality=False,
        )
    }

    top7 = final_70a_results["E1-top7"]
    top3 = final_70a_results["E1-top3"]
    e2a = final_70a_results["E2A-R"]
    e2c = final_70a_results["E2C-L"]
    e3a = final_70a_results["E3A-RF"]
    e3a_stage6 = stage6_results["E3A-RF"]

    normality_probe = _normality_probe_discriminative(
        {
            "host_native_bad_reference": host_native_normality,
            "baseline_transplant_target": target_normality,
            "E1-top7": top7["replace_normality"],
            "E1-top3": top3["replace_normality"],
            "E2A-R": e2a["replace_normality"],
            "E2C-L": e2c["replace_normality"],
            "E3A-RF": e3a["replace_normality"],
        }
    )

    delta_summary = {
        "E3A-RF_minus_E1-top7": {
            "transfer": _transfer_delta(e3a["transfer"], top7["transfer"]),
            "replace_normality": _normality_delta(e3a["replace_normality"], top7["replace_normality"]),
            "proxy_telemetry": _proxy_delta(e3a["proxy_telemetry"], top7["proxy_telemetry"]),
        },
        "E3A-RF_minus_E1-top3": {
            "transfer": _transfer_delta(e3a["transfer"], top3["transfer"]),
            "replace_normality": _normality_delta(e3a["replace_normality"], top3["replace_normality"]),
            "proxy_telemetry": _proxy_delta(e3a["proxy_telemetry"], top3["proxy_telemetry"]),
        },
        "E3A-RF_minus_E2A-R": {
            "transfer": _transfer_delta(e3a["transfer"], e2a["transfer"]),
            "replace_normality": _normality_delta(e3a["replace_normality"], e2a["replace_normality"]),
            "proxy_telemetry": _proxy_delta(e3a["proxy_telemetry"], e2a["proxy_telemetry"]),
        },
        "E3A-RF_minus_E2C-L": {
            "transfer": _transfer_delta(e3a["transfer"], e2c["transfer"]),
            "replace_normality": _normality_delta(e3a["replace_normality"], e2c["replace_normality"]),
            "proxy_telemetry": _proxy_delta(e3a["proxy_telemetry"], e2c["proxy_telemetry"]),
        },
        "E3A-RF_final70a_minus_stage6_tailfix_final": {
            "transfer": _stage_delta(e3a["transfer"], e3a_stage6["transfer"]),
        },
    }

    leg_gain = {
        "gap_reduction_vs_E1-top7": _gap_reduction(e3a["transfer"], top7["transfer"], "dir_leg_gap"),
        "gap_reduction_vs_E1-top3": _gap_reduction(e3a["transfer"], top3["transfer"], "dir_leg_gap"),
        "gap_reduction_vs_E2A-R": _gap_reduction(e3a["transfer"], e2a["transfer"], "dir_leg_gap"),
        "gap_reduction_vs_E2C-L": _gap_reduction(e3a["transfer"], e2c["transfer"], "dir_leg_gap"),
        "closure_gain_vs_E1-top7": _closure_gain(e3a["transfer"], top7["transfer"], "dir_leg_closure_ratio"),
        "closure_gain_vs_E1-top3": _closure_gain(e3a["transfer"], top3["transfer"], "dir_leg_closure_ratio"),
        "closure_gain_vs_E2A-R": _closure_gain(e3a["transfer"], e2a["transfer"], "dir_leg_closure_ratio"),
        "closure_gain_vs_E2C-L": _closure_gain(e3a["transfer"], e2c["transfer"], "dir_leg_closure_ratio"),
        "not_worse_than_E2A-R_by_gap": _safe_float(e3a["transfer"].get("dir_leg_gap"))
        <= _safe_float(e2a["transfer"].get("dir_leg_gap")) + 1e-12,
        "not_worse_than_E2C-L_by_gap": _safe_float(e3a["transfer"].get("dir_leg_gap"))
        <= _safe_float(e2c["transfer"].get("dir_leg_gap")) + 1e-12,
    }

    nonleg_retention = {
        "dir_base_closure_retention_vs_E1-top3": _retention_ratio(e3a["transfer"], top3["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_retention_vs_E1-top3": _retention_ratio(e3a["transfer"], top3["transfer"], "dir_nonleg_closure_ratio"),
        "dir_base_closure_retention_vs_E2A-R": _retention_ratio(e3a["transfer"], e2a["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_retention_vs_E2A-R": _retention_ratio(e3a["transfer"], e2a["transfer"], "dir_nonleg_closure_ratio"),
        "dir_base_closure_delta_vs_E1-top3": _closure_gain(e3a["transfer"], top3["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_delta_vs_E1-top3": _closure_gain(e3a["transfer"], top3["transfer"], "dir_nonleg_closure_ratio"),
        "dir_base_closure_delta_vs_E2A-R": _closure_gain(e3a["transfer"], e2a["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_delta_vs_E2A-R": _closure_gain(e3a["transfer"], e2a["transfer"], "dir_nonleg_closure_ratio"),
        "dir_base_closure_delta_vs_E2C-L": _closure_gain(e3a["transfer"], e2c["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_delta_vs_E2C-L": _closure_gain(e3a["transfer"], e2c["transfer"], "dir_nonleg_closure_ratio"),
    }

    higher_than_top7 = _strictly_higher(e3a["transfer"], top7["transfer"])
    higher_than_top3 = _strictly_higher(e3a["transfer"], top3["transfer"])
    higher_than_e2a = _strictly_higher(e3a["transfer"], e2a["transfer"])
    higher_than_e2c = _strictly_higher(e3a["transfer"], e2c["transfer"])

    clearly_higher_than_top7 = _clearly_higher(e3a["transfer"], top7["transfer"])
    clearly_higher_than_top3 = _clearly_higher(e3a["transfer"], top3["transfer"])
    clearly_higher_than_e2a = _clearly_higher(e3a["transfer"], e2a["transfer"])
    clearly_higher_than_e2c = _clearly_higher(e3a["transfer"], e2c["transfer"])

    meaningful_leg_gain = any(
        _safe_float(leg_gain[key]) > 0.08
        for key in (
            "closure_gain_vs_E1-top7",
            "closure_gain_vs_E1-top3",
            "closure_gain_vs_E2A-R",
            "closure_gain_vs_E2C-L",
        )
    )
    partial_leg_gain = any(
        _safe_float(leg_gain[key]) > 0.03
        for key in (
            "closure_gain_vs_E1-top7",
            "closure_gain_vs_E1-top3",
            "closure_gain_vs_E2A-R",
            "closure_gain_vs_E2C-L",
        )
    )
    unacceptable_nonleg_giveback = (
        (_safe_float(nonleg_retention["dir_base_closure_retention_vs_E1-top3"]) < 0.75)
        or (_safe_float(nonleg_retention["dir_nonleg_closure_retention_vs_E1-top3"]) < 0.75)
    )

    normality_improvements = (
        "normality_probe_non_discriminative"
        if normality_probe["normality_probe_non_discriminative"]
        else {
            "vs_E1-top7": bool(_normality_improved(e3a["replace_normality"], top7["replace_normality"])),
            "vs_E1-top3": bool(_normality_improved(e3a["replace_normality"], top3["replace_normality"])),
            "vs_E2A-R": bool(_normality_improved(e3a["replace_normality"], e2a["replace_normality"])),
            "vs_E2C-L": bool(_normality_improved(e3a["replace_normality"], e2c["replace_normality"])),
        }
    )

    stage6_signal = {
        "stage6_transfer_score": _safe_float(e3a_stage6["transfer"].get("aggregate_transfer_score")),
        "final70a_transfer_score": _safe_float(e3a["transfer"].get("aggregate_transfer_score")),
        "stage6_minus_final70a_transfer_score": _safe_float(e3a_stage6["transfer"].get("aggregate_transfer_score"))
        - _safe_float(e3a["transfer"].get("aggregate_transfer_score")),
        "stage6_minus_final70a_dir_leg_closure": _safe_float(e3a_stage6["transfer"].get("dir_leg_closure_ratio"))
        - _safe_float(e3a["transfer"].get("dir_leg_closure_ratio")),
        "best_prior_final70a_transfer_score": _best_prior_score(
            top7["transfer"], top3["transfer"], e2a["transfer"], e2c["transfer"]
        ),
    }
    stage6_suggests_path_right_but_late_degrades = (
        _safe_float(stage6_signal["stage6_transfer_score"]) > _safe_float(stage6_signal["best_prior_final70a_transfer_score"]) + 0.02
        and _safe_float(stage6_signal["stage6_minus_final70a_transfer_score"]) > 0.05
    ) or (_safe_float(stage6_signal["stage6_minus_final70a_dir_leg_closure"]) > 0.08)

    judgement_case = "Case 3"
    next_step = "E3-B"
    allocation_missing_lever = False
    top7_viable = False
    interpretation = (
        "first staged allocation arm does not yet prove a better transfer-compatible top7 basin; the next priority remains another allocation direction"
    )
    if (
        clearly_higher_than_top7
        and clearly_higher_than_top3
        and clearly_higher_than_e2a
        and clearly_higher_than_e2c
        and meaningful_leg_gain
        and (not unacceptable_nonleg_giveback)
    ):
        judgement_case = "Case 1"
        next_step = "E3_confirm_or_exploit"
        allocation_missing_lever = True
        top7_viable = True
        interpretation = (
            "E3A-RF clearly beats every prior arm, finally improves dir_leg, and does not give back most dir_base/dir_nonleg gains; staged co-adaptation allocation is the missing lever for this top7 lane"
        )
    elif (
        clearly_higher_than_top7
        and clearly_higher_than_e2a
        and clearly_higher_than_e2c
        and partial_leg_gain
    ):
        judgement_case = "Case 2"
        next_step = "E3-B"
        interpretation = (
            "allocation helps and does better than the weaker prior top7-style arms, but it still does not cleanly dominate E1-top3 and/or it gives back too much nonleg closure"
        )
    else:
        judgement_case = "Case 3"
        if stage6_suggests_path_right_but_late_degrades:
            next_step = "E4"
            interpretation = (
                "E3A-RF shows stronger signal earlier in the chain than it preserves at final 70a, so local optimization dynamics now look more likely than another first-order allocation axis"
            )
        else:
            next_step = "E3-B"
            interpretation = (
                "E3A-RF mostly repeats the prior nonleg-biased pattern or still fails to move dir_leg enough, so the next best lever is another allocation ordering rather than second-order tuning"
            )
    if normality_probe["normality_probe_non_discriminative"]:
        interpretation += "; normality probe remains non-discriminative, so fixed transferability carries the evidential load"

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "E3-A freeze head, train readouts/adapters first",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "coadapt host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic single-step first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "strict_constraints": [
                "single new arm only",
                "reuse E1/E2-A/E2-C anchors",
                "no E0/E1/E2 rerun",
                "no attribution side quest",
                "no planner semantics mainline",
                "no support-width change",
                "no loss-family / architecture / optimizer-family change",
                "stage6/70a configs reused except ckpt_in/out_dir/run_name",
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
            "direct_pose_head.0 input-block allocation proxy is useful but not a leading indicator",
            "E1-top3 and E2A-R mainly improve dir_base/dir_nonleg while leaving dir_leg largely unimproved",
            "E2C-L still fails to clearly lift dir_leg and gives back nonleg closure",
            "replace-normality probe has been non-discriminative under the current fixed readout",
        ],
        "degraded_e3a_variant": False,
        "arm_inventory": inventory,
        "E3A_RF_key_config_diff": config_diff,
        "direct_branch_module_family_mapping": DIRECT_BRANCH_FAMILY_MAPPING,
        "artifact_paths": {
            "E3A-RF": {
                "basetrain_config": str(E3A_BASETRAIN_CONFIG),
                "basetrain_epoch014_ckpt": str(E3A_BASETRAIN_EPOCH014),
                "stage6_tailfix_config": str(TOP7_STAGE6_TAILFIX_CONFIG),
                "stage6_tailfix_ckpt": str(E3A_STAGE6_TAILFIX_CKPT),
                "stage6_tailfix_log": str(E3A_STAGE6_TAILFIX_LOG),
                "stage6_tailfix_eval": str(E3A_STAGE6_TAILFIX_EVAL),
                "stage70a_config": str(STAGE70A_CONFIG),
                "stage70a_ckpt": str(E3A_70A_CKPT),
                "stage70a_log": str(E3A_70A_LOG),
                "stage70a_eval": str(E3A_70A_EVAL),
            }
        },
        "basetrain_allocation_schedule": {
            "top_level_rot_local_tail_k": int(e3a_cfg["rot_local_tail_k"]),
            "phase_a": {
                "epochs": [1, 5],
                "allocation_mode": "freeze main head / readout-first",
                "direct_pose_head_train_scope": _stage_params(e3a_cfg, 0).get("direct_pose_head_train_scope"),
                "effective_trainable_families": ["readouts"],
                "effective_notes": "canonical matched basetrain has no instantiated direct adapters, so early trainable set is readout-only inside direct_pose_head",
            },
            "phase_b": {
                "epochs": [6, 9],
                "allocation_mode": "restore late hidden block",
                "direct_pose_head_train_scope": _stage_params(e3a_cfg, 1).get("direct_pose_head_train_scope"),
                "effective_trainable_families": ["head_late_block", "readouts"],
            },
            "phase_c": {
                "epochs": [10, 11],
                "allocation_mode": "full direct-head co-adaptation",
                "direct_pose_head_train_scope": _stage_params(e3a_cfg, 2).get("direct_pose_head_train_scope"),
                "effective_trainable_families": ["head", "readouts"],
            },
            "phase_d": {
                "epochs": [12, 15],
                "allocation_mode": "late full top7 target",
                "direct_pose_head_train_scope": _stage_params(e3a_cfg, 3).get("direct_pose_head_train_scope"),
                "effective_trainable_families": ["head", "readouts"],
            },
        },
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
        },
        "stage6_tailfix_final": stage6_results,
        "final_70a_results": final_70a_results,
        "replace_normality_readout": {
            "host_native_bad_reference": host_native_normality,
            "baseline_transplant_target": target_normality,
            "E1-top7": top7["replace_normality"],
            "E1-top3": top3["replace_normality"],
            "E2A-R": e2a["replace_normality"],
            "E2C-L": e2c["replace_normality"],
            "E3A-RF": e3a["replace_normality"],
        },
        "direct_pose_head0_proxy_telemetry": {
            "E1-top7": top7["proxy_telemetry"],
            "E1-top3": top3["proxy_telemetry"],
            "E2A-R": e2a["proxy_telemetry"],
            "E2C-L": e2c["proxy_telemetry"],
            "E3A-RF": e3a["proxy_telemetry"],
        },
        "delta_summary": delta_summary,
        "dir_leg_delta_closure_summary": {
            **leg_gain,
            "meaningful_leg_gain": bool(meaningful_leg_gain),
            "partial_leg_gain": bool(partial_leg_gain),
        },
        "nonleg_retention_giveback_summary": {
            **nonleg_retention,
            "unacceptable_nonleg_giveback": bool(unacceptable_nonleg_giveback),
        },
        "normality_probe_assessment": normality_probe,
        "judgement": {
            "case": judgement_case,
            "normality_probe_non_discriminative": bool(normality_probe["normality_probe_non_discriminative"]),
            "co_adaptation_allocation_is_missing_lever": bool(allocation_missing_lever),
            "top7_viable_under_staged_allocation_compatible_coadaptation": bool(top7_viable),
            "next_step_recommendation": next_step,
            "interpretation": interpretation,
        },
        "explicit_answers": {
            "q1_E3A_RF_better_than_E1_top7_final70a": bool(higher_than_top7),
            "q2_E3A_RF_better_than_E1_top3_final70a": bool(higher_than_top3),
            "q3_E3A_RF_better_than_E2A_R_final70a": bool(higher_than_e2a),
            "q4_E3A_RF_better_than_E2C_L_final70a": bool(higher_than_e2c),
            "q5_E3A_RF_clearly_lifts_dir_leg": bool(meaningful_leg_gain),
            "q5b_E3A_RF_at_least_not_worse_than_E2_family_on_dir_leg": bool(
                leg_gain["not_worse_than_E2A-R_by_gap"] and leg_gain["not_worse_than_E2C-L_by_gap"]
            ),
            "q6_leg_gain_has_unacceptable_nonleg_giveback": bool(unacceptable_nonleg_giveback),
            "q7_co_adaptation_allocation_is_missing_lever": bool(allocation_missing_lever),
            "q7_top7_viable_under_staged_allocation_compatible_coadaptation": bool(top7_viable),
            "q8_next_step": next_step,
            "normality_vs_priors": normality_improvements,
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
