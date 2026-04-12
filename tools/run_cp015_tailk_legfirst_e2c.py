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
    ARMS as BASE_ARMS,
    ArmSpec,
    E2A_70A_CKPT,
    E2A_70A_EVAL,
    E2A_70A_LOG,
    E2A_STAGE6_TAILFIX_CKPT,
    E2A_STAGE6_TAILFIX_EVAL,
    E2A_STAGE6_TAILFIX_LOG,
    _arm_stage_result,
    _inventory_row,
    _normality_delta,
    _normality_improved,
    _normality_probe_discriminative,
    _proxy_delta,
    _transfer_better,
    _transfer_delta,
)
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP7_STAGE6_TAILFIX_CONFIG,
)


RUN_DATE = "20260408"
RUN_NAME = "cp015_tailk_legfirst_e2c"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"

E2C_BASERUN = (
    "exp_phase_DirectBranch_v1_d1_cp015_tailk7_legfirst_nonlegexp_rankmix_tw020_"
    "corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408"
)
E2C_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_legfirst_nonlegexp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json"
)
E2C_BASETRAIN_EPOCH014 = ROOT / "models" / "cp015_phasecd_tailk_probe_20260331" / E2C_BASERUN / "ckpt_epoch_014.pth"
E2C_STAGE6_TAILFIX_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_legfirst_stage6_tailfix_e2c_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_legfirst_e2c_20260408.pth"
)
E2C_STAGE6_TAILFIX_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_legfirst_stage6_tailfix_e2c_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_legfirst_e2c_20260408.json"
)
E2C_STAGE6_TAILFIX_EVAL = OUT_ROOT / "stage6_tailfix" / "stage6_freerun" / "Walk_F_freerun_cycles.json"
E2C_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_legfirst_stage70a_from_tailfix_e2c_20260408"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_legfirst_stage6tailfix_e2c_20260408.pth"
)
E2C_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_legfirst_stage70a_from_tailfix_e2c_20260408"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_legfirst_stage6tailfix_e2c_20260408.json"
)
E2C_70A_EVAL = OUT_ROOT / "stage70a" / "eval_model_source" / "Walk_F_freerun_cycles.json"

E2C_ARM = ArmSpec(
    arm="E2C-L",
    provenance="new_legfirst_arm",
    support_schedule="7 -> 7 -> 7 (leg-first -> nonleg expansion)",
    basetrain_config=E2C_BASETRAIN_CONFIG,
    basetrain_epoch014_ckpt=E2C_BASETRAIN_EPOCH014,
    stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
    stage6_tailfix_ckpt=E2C_STAGE6_TAILFIX_CKPT,
    stage6_tailfix_log=E2C_STAGE6_TAILFIX_LOG,
    stage6_tailfix_eval=E2C_STAGE6_TAILFIX_EVAL,
    stage70a_config=STAGE70A_CONFIG,
    stage70a_ckpt=E2C_70A_CKPT,
    stage70a_log=E2C_70A_LOG,
    stage70a_eval=E2C_70A_EVAL,
    notes="Matched E2-C arm: fixed top7 support with stage-wise leg-first -> nonleg expansion on the direct-pose objective.",
)

ARMS = tuple(list(BASE_ARMS) + [E2C_ARM])


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_core(cfg: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    stages = cfg["freerun_stage_schedule"]
    loss_groups = stages[idx].get("loss_groups") or {}
    core = loss_groups.get("core") if isinstance(loss_groups, Mapping) else None
    return core if isinstance(core, Mapping) else {}


def _config_diff_rows(base_cfg: Mapping[str, Any], e2c_cfg: Mapping[str, Any]) -> list[Dict[str, Any]]:
    phase_a = _stage_core(e2c_cfg, 0)
    phase_b = _stage_core(e2c_cfg, 1)
    phase_c = _stage_core(e2c_cfg, 2)
    phase_d = _stage_core(e2c_cfg, 3)
    base_phase_b = _stage_core(base_cfg, 1)
    base_phase_c = _stage_core(base_cfg, 2)
    base_phase_d = _stage_core(base_cfg, 3)
    return [
        {
            "field": "rot_local_tail_k",
            "baseline_top7": int(base_cfg["rot_local_tail_k"]),
            "E2C-L": int(e2c_cfg["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_b.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_b["rot_local_tail_k"]),
            "E2C-L": int(phase_b["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_c.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_c["rot_local_tail_k"]),
            "E2C-L": int(phase_c["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_d.core.rot_local_tail_k",
            "baseline_top7": int(base_phase_d["rot_local_tail_k"]),
            "E2C-L": int(phase_d["rot_local_tail_k"]),
            "meaning": "kept fixed at top7",
        },
        {
            "field": "freerun_stage_schedule.phase_a.core.direct_pose_loss_leg_split",
            "baseline_top7": None,
            "E2C-L": bool(phase_a["direct_pose_loss_leg_split"]),
            "meaning": "early leg-first split objective",
        },
        {
            "field": "freerun_stage_schedule.phase_a.core.direct_pose_loss_group_norm_w_nonleg",
            "baseline_top7": None,
            "E2C-L": float(phase_a["direct_pose_loss_group_norm_w_nonleg"]),
            "meaning": "leg-only warmup",
        },
        {
            "field": "freerun_stage_schedule.phase_b.core.direct_pose_loss_group_norm_w_nonleg",
            "baseline_top7": None,
            "E2C-L": float(phase_b["direct_pose_loss_group_norm_w_nonleg"]),
            "meaning": "early nonleg expansion / leg-dominant",
        },
        {
            "field": "freerun_stage_schedule.phase_c.core.direct_pose_loss_group_norm_w_nonleg",
            "baseline_top7": None,
            "E2C-L": float(phase_c["direct_pose_loss_group_norm_w_nonleg"]),
            "meaning": "full nonleg restored inside split objective",
        },
        {
            "field": "freerun_stage_schedule.phase_d.core.direct_pose_loss_leg_split",
            "baseline_top7": None,
            "E2C-L": bool(phase_d["direct_pose_loss_leg_split"]),
            "meaning": "late return to full-branch baseline objective",
        },
        {
            "field": "save_fit_ckpt_epochs",
            "baseline_top7": str(base_cfg["save_fit_ckpt_epochs"]),
            "E2C-L": str(e2c_cfg["save_fit_ckpt_epochs"]),
            "meaning": "kept fixed",
        },
    ]


def _gap_reduction(candidate: Mapping[str, Any], reference: Mapping[str, Any], key: str) -> float:
    return _safe_float(reference.get(key)) - _safe_float(candidate.get(key))


def _closure_gain(candidate: Mapping[str, Any], reference: Mapping[str, Any], key: str) -> float:
    return _safe_float(candidate.get(key)) - _safe_float(reference.get(key))


def _retention_ratio(candidate: Mapping[str, Any], reference: Mapping[str, Any], key: str) -> float:
    denom = _safe_float(reference.get(key))
    if not math.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return _safe_float(candidate.get(key)) / denom


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

    top7_cfg = _load_json(BASE_ARMS[0].basetrain_config)
    e2c_cfg = _load_json(E2C_BASETRAIN_CONFIG)
    config_diff = _config_diff_rows(top7_cfg, e2c_cfg)

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
        "E2A-R": _arm_stage_result(
            name="E2A-R_stage6_tailfix",
            ckpt_path=E2A_STAGE6_TAILFIX_CKPT,
            eval_json_path=E2A_STAGE6_TAILFIX_EVAL,
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
        ),
        "E2C-L": _arm_stage_result(
            name="E2C-L_stage6_tailfix",
            ckpt_path=E2C_STAGE6_TAILFIX_CKPT,
            eval_json_path=E2C_STAGE6_TAILFIX_EVAL,
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
        ),
    }

    top7 = final_70a_results["E1-top7"]
    top3 = final_70a_results["E1-top3"]
    e2a = final_70a_results["E2A-R"]
    e2c = final_70a_results["E2C-L"]

    normality_probe = _normality_probe_discriminative(
        {
            "host_native_bad_reference": host_native_normality,
            "baseline_transplant_target": target_normality,
            "E1-top7": top7["replace_normality"],
            "E1-top3": top3["replace_normality"],
            "E2A-R": e2a["replace_normality"],
            "E2C-L": e2c["replace_normality"],
        }
    )

    delta_vs_top7 = {
        "transfer": _transfer_delta(e2c["transfer"], top7["transfer"]),
        "replace_normality": _normality_delta(e2c["replace_normality"], top7["replace_normality"]),
        "proxy_telemetry": _proxy_delta(e2c["proxy_telemetry"], top7["proxy_telemetry"]),
    }
    delta_vs_top3 = {
        "transfer": _transfer_delta(e2c["transfer"], top3["transfer"]),
        "replace_normality": _normality_delta(e2c["replace_normality"], top3["replace_normality"]),
        "proxy_telemetry": _proxy_delta(e2c["proxy_telemetry"], top3["proxy_telemetry"]),
    }
    delta_vs_e2a = {
        "transfer": _transfer_delta(e2c["transfer"], e2a["transfer"]),
        "replace_normality": _normality_delta(e2c["replace_normality"], e2a["replace_normality"]),
        "proxy_telemetry": _proxy_delta(e2c["proxy_telemetry"], e2a["proxy_telemetry"]),
    }

    better_than_top7 = _transfer_better(e2c["transfer"], top7["transfer"], margin=0.08)
    better_than_top3 = _transfer_better(e2c["transfer"], top3["transfer"], margin=0.08)
    better_than_e2a = _transfer_better(e2c["transfer"], e2a["transfer"], margin=0.08)
    normality_improved_vs_top7 = (
        False if normality_probe["normality_probe_non_discriminative"] else _normality_improved(e2c["replace_normality"], top7["replace_normality"])
    )
    normality_improved_vs_top3 = (
        False if normality_probe["normality_probe_non_discriminative"] else _normality_improved(e2c["replace_normality"], top3["replace_normality"])
    )
    normality_improved_vs_e2a = (
        False if normality_probe["normality_probe_non_discriminative"] else _normality_improved(e2c["replace_normality"], e2a["replace_normality"])
    )

    leg_gain = {
        "gap_reduction_vs_E1-top7": _gap_reduction(e2c["transfer"], top7["transfer"], "dir_leg_gap"),
        "gap_reduction_vs_E1-top3": _gap_reduction(e2c["transfer"], top3["transfer"], "dir_leg_gap"),
        "gap_reduction_vs_E2A-R": _gap_reduction(e2c["transfer"], e2a["transfer"], "dir_leg_gap"),
        "closure_gain_vs_E1-top7": _closure_gain(e2c["transfer"], top7["transfer"], "dir_leg_closure_ratio"),
        "closure_gain_vs_E1-top3": _closure_gain(e2c["transfer"], top3["transfer"], "dir_leg_closure_ratio"),
        "closure_gain_vs_E2A-R": _closure_gain(e2c["transfer"], e2a["transfer"], "dir_leg_closure_ratio"),
    }
    nonleg_retention = {
        "dir_base_closure_retention_vs_E1-top3": _retention_ratio(
            e2c["transfer"], top3["transfer"], "dir_base_closure_ratio"
        ),
        "dir_nonleg_closure_retention_vs_E1-top3": _retention_ratio(
            e2c["transfer"], top3["transfer"], "dir_nonleg_closure_ratio"
        ),
        "dir_base_closure_delta_vs_E1-top3": _closure_gain(e2c["transfer"], top3["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_delta_vs_E1-top3": _closure_gain(e2c["transfer"], top3["transfer"], "dir_nonleg_closure_ratio"),
        "dir_base_closure_delta_vs_E2A-R": _closure_gain(e2c["transfer"], e2a["transfer"], "dir_base_closure_ratio"),
        "dir_nonleg_closure_delta_vs_E2A-R": _closure_gain(e2c["transfer"], e2a["transfer"], "dir_nonleg_closure_ratio"),
    }

    meaningful_leg_gain = (
        _safe_float(leg_gain["closure_gain_vs_E1-top7"]) > 0.10
        or _safe_float(leg_gain["closure_gain_vs_E2A-R"]) > 0.10
    )
    partial_leg_gain = (
        _safe_float(leg_gain["closure_gain_vs_E1-top7"]) > 0.03
        or _safe_float(leg_gain["closure_gain_vs_E2A-R"]) > 0.03
    )
    unacceptable_nonleg_giveback = (
        _safe_float(nonleg_retention["dir_base_closure_retention_vs_E1-top3"]) < 0.60
        or _safe_float(nonleg_retention["dir_nonleg_closure_retention_vs_E1-top3"]) < 0.60
    )

    case_label = "Case 3"
    missing_lever = False
    top7_viable = False
    next_step = "E3"
    interpretation = (
        "current leg-first basetrain path does not yet show enough leg-specific closure improvement to claim a transfer-compatible top7 path"
    )
    if better_than_top7 and better_than_top3 and better_than_e2a and meaningful_leg_gain and (not unacceptable_nonleg_giveback):
        case_label = "Case 1"
        missing_lever = True
        top7_viable = True
        next_step = "leg-targeted_E2-B_confirm_or_exploit"
        interpretation = (
            "E2C-L beats all prior arms while finally lifting dir_leg without giving back most dir_base/dir_nonleg gains, so leg-targeted path-shaping appears to be the missing lever"
        )
    elif better_than_top7 and better_than_e2a and partial_leg_gain:
        case_label = "Case 2"
        missing_lever = False
        top7_viable = False
        next_step = "leg-targeted_E2-B"
        interpretation = (
            "leg-targeted path-shaping helps relative to E1-top7 and E2A-R, but the gain is still partial and does not cleanly dominate E1-top3 and/or preserve all nonleg gains"
        )
    else:
        case_label = "Case 3"
        next_step = "E3"
        interpretation = (
            "this basetrain path-shaping family still mostly reproduces the prior nonleg-biased pattern or fails to improve dir_leg enough, so the next lever is closer to E3"
        )

    if normality_probe["normality_probe_non_discriminative"]:
        interpretation += "; the current replace-normality probe remains non-discriminative, so fixed transferability carries the conclusion"

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "E2-C leg-first -> nonleg expansion",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "coadapt host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic single-step first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "strict_constraints": [
                "single new arm only",
                "reuse E1 and E2-A anchors",
                "no E0 archaeology rerun",
                "no new attribution mainline",
                "no planner semantics mainline",
                "no support-width sweep",
                "no optimizer / loss family / architecture change",
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
            "top3 and E2A-R mainly improve dir_base/dir_nonleg while leaving dir_leg largely unimproved",
            "replace-normality probe has been non-discriminative under the current fixed readout",
        ],
        "degraded_e2c_variant": False,
        "arm_inventory": inventory,
        "E2C_L_key_config_diff": config_diff,
        "artifact_paths": {
            "E2C-L": {
                "basetrain_config": str(E2C_BASETRAIN_CONFIG),
                "basetrain_epoch014_ckpt": str(E2C_BASETRAIN_EPOCH014),
                "stage6_tailfix_config": str(TOP7_STAGE6_TAILFIX_CONFIG),
                "stage6_tailfix_ckpt": str(E2C_STAGE6_TAILFIX_CKPT),
                "stage6_tailfix_log": str(E2C_STAGE6_TAILFIX_LOG),
                "stage6_tailfix_eval": str(E2C_STAGE6_TAILFIX_EVAL),
                "stage70a_config": str(STAGE70A_CONFIG),
                "stage70a_ckpt": str(E2C_70A_CKPT),
                "stage70a_log": str(E2C_70A_LOG),
                "stage70a_eval": str(E2C_70A_EVAL),
            }
        },
        "basetrain_legfirst_schedule": {
            "top_level_rot_local_tail_k": int(e2c_cfg["rot_local_tail_k"]),
            "phase_a": {
                "epochs": [1, 5],
                "objective": "leg_only",
                "direct_pose_loss_leg_split": True,
                "direct_pose_loss_group_norm_enable": True,
                "direct_pose_loss_group_norm_w_leg": 1.0,
                "direct_pose_loss_group_norm_w_nonleg": 0.0,
            },
            "phase_b": {
                "epochs": [6, 9],
                "objective": "leg_dominant",
                "direct_pose_loss_leg_split": True,
                "direct_pose_loss_group_norm_enable": True,
                "direct_pose_loss_group_norm_w_leg": 1.0,
                "direct_pose_loss_group_norm_w_nonleg": 0.25,
            },
            "phase_c": {
                "epochs": [10, 11],
                "objective": "split_full_nonleg",
                "direct_pose_loss_leg_split": True,
                "direct_pose_loss_group_norm_enable": True,
                "direct_pose_loss_group_norm_w_leg": 1.0,
                "direct_pose_loss_group_norm_w_nonleg": 1.0,
            },
            "phase_d": {
                "epochs": [12, 15],
                "objective": "full_branch_target",
                "direct_pose_loss_leg_split": False,
                "direct_pose_loss_group_norm_enable": False,
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
        },
        "proxy_telemetry": {
            "E1-top7": top7["proxy_telemetry"],
            "E1-top3": top3["proxy_telemetry"],
            "E2A-R": e2a["proxy_telemetry"],
            "E2C-L": e2c["proxy_telemetry"],
        },
        "delta_summary": {
            "E2C-L_minus_E1-top7": delta_vs_top7,
            "E2C-L_minus_E1-top3": delta_vs_top3,
            "E2C-L_minus_E2A-R": delta_vs_e2a,
        },
        "dir_leg_delta_closure_summary": leg_gain,
        "nonleg_retention_giveback_summary": {
            **nonleg_retention,
            "unacceptable_nonleg_giveback": bool(unacceptable_nonleg_giveback),
        },
        "normality_probe_assessment": normality_probe,
        "judgement": {
            "case": case_label,
            "normality_probe_non_discriminative": bool(normality_probe["normality_probe_non_discriminative"]),
            "leg_targeted_path_shaping_is_missing_lever": bool(missing_lever),
            "top7_viable_under_leg_targeted_transfer_compatible_path": bool(top7_viable),
            "next_step_recommendation": next_step,
            "interpretation": interpretation,
        },
        "explicit_answers": {
            "q1_E2C_L_better_than_E1_top7_final70a": bool(better_than_top7),
            "q2_E2C_L_better_than_E1_top3_final70a": bool(better_than_top3),
            "q3_E2C_L_better_than_E2A_R_final70a": bool(better_than_e2a),
            "q4_E2C_L_clearly_lifts_dir_leg": bool(meaningful_leg_gain),
            "q5_leg_gain_has_unacceptable_nonleg_giveback": bool(unacceptable_nonleg_giveback),
            "q6_leg_targeted_path_shaping_is_missing_lever": bool(missing_lever),
            "q6_top7_viable_under_leg_targeted_transfer_compatible_path": bool(top7_viable),
            "q7_next_step": next_step,
            "normality_vs_priors": (
                "normality_probe_non_discriminative"
                if normality_probe["normality_probe_non_discriminative"]
                else {
                    "vs_E1-top7": bool(normality_improved_vs_top7),
                    "vs_E1-top3": bool(normality_improved_vs_top3),
                    "vs_E2A-R": bool(normality_improved_vs_e2a),
                }
            ),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
