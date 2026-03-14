#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        TEACHER,
        create_replace_zerophase_warmstart,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        mean,
        parse_reference_payload,
        run_70r_promote,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )
    from run_stage6_plantransplant_compare import extract_stage6_init, resolve_model_state, run_cmd
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        TEACHER,
        create_replace_zerophase_warmstart,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        mean,
        parse_reference_payload,
        run_70r_promote,
        run_eval,
        run_posttrain_stage,
        safe_float,
        window_group_stats,
        write_json,
    )
    from tools.run_stage6_plantransplant_compare import extract_stage6_init, resolve_model_state, run_cmd


RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_oldplan_component_ablation_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_oldplan_component_ablation_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
STAGE6_ROOT = OUT_ROOT / "stage6"
STAGE6_MODEL_ROOT = MODEL_ROOT / "stage6"
DOWNSTREAM_ROOT = OUT_ROOT / "downstream"
DOWNSTREAM_MODEL_ROOT = MODEL_ROOT / "downstream"

BASELINE_COMPARE_JSON = ROOT / "debug_output" / "_tmp_stage6_basetrain_compare_20260313" / "compare_summary.json"
PLANTRANSPLANT_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_stage6_plantransplant_20260314" / "summary.json"
OLDPLAN_CHAIN_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_oldplan_downstream_chain_20260314" / "summary.json"
STAGE6_CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"

SOURCE_BASELINE_CASE = "cp015_bestfree"
CONTROL_STAGE6_CASE = "cp015_with_old_planstack"


@dataclass(frozen=True)
class GroupSpec:
    name: str
    label: str
    prefixes: Tuple[str, ...]
    exact_keys: Tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class LaneSpec:
    code: str
    name: str
    label: str
    rollback_groups: Tuple[str, ...]
    note: str


GROUP_SPECS: Tuple[GroupSpec, ...] = (
    GroupSpec(
        name="plan_head",
        label="A. plan head rollback",
        prefixes=("contact_plan_head.", "contact_plan_time_head."),
        exact_keys=(),
        rationale=(
            "Readout layer from plan_z to contacts_plan logits, including the additive time-PE bias head."
        ),
    ),
    GroupSpec(
        name="plan_init_state",
        label="B. plan init-state rollback",
        prefixes=("contact_plan_init_head.",),
        exact_keys=("contact_plan_init_z", "contact_phase_state_init"),
        rationale=(
            "Cold-start state for plan_z / phase-state, including learnable init vectors and obs-conditioned init head."
        ),
    ),
    GroupSpec(
        name="planner_core",
        label="C. planner-core rollback",
        prefixes=("contact_plan_cell.", "event_clock_gate.", "event_clock_corrector."),
        exact_keys=(),
        rationale=(
            "Core recurrent planner latent plus Event-Clock gate/corrector that rescales and corrects plan_z inside the loop."
        ),
    ),
    GroupSpec(
        name="phase_contact_input",
        label="D. phase/contact input-side rollback",
        prefixes=("contact_plan_phase_head.", "contact_phase_state_delta_head."),
        exact_keys=(),
        rationale=(
            "Phase/contact side inputs into the planner: phase residual on logits and phase-state update head driven by cond/meas/delta_meas."
        ),
    ),
)


LANE_SPECS: Tuple[LaneSpec, ...] = (
    LaneSpec(
        code="A",
        name="rollback_plan_head",
        label="A. plan head rollback",
        rollback_groups=("plan_head",),
        note="Rollback only the plan readout/time-bias heads to cp015; keep old-plan core/init/phase side.",
    ),
    LaneSpec(
        code="B",
        name="rollback_plan_init_state",
        label="B. plan init-state rollback",
        rollback_groups=("plan_init_state",),
        note="Rollback only init-state parameters to cp015; keep old-plan core/head/phase side.",
    ),
    LaneSpec(
        code="C",
        name="rollback_planner_core",
        label="C. planner-core rollback",
        rollback_groups=("planner_core",),
        note="Rollback planner latent core and Event-Clock corrector/gate to cp015; keep old-plan head/init/phase side.",
    ),
    LaneSpec(
        code="D",
        name="rollback_phase_contact_input",
        label="D. phase/contact input-side rollback",
        rollback_groups=("phase_contact_input",),
        note="Rollback phase/contact-side input modules to cp015; keep old-plan core/head/init intact.",
    ),
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def delta_block(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, float]:
    keys = sorted(set(cur.keys()) | set(ref.keys()))
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def enrich_init_section(section: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(section)
    dir_leg = safe_float(payload.get("dir_leg_base"))
    dir_nonleg = safe_float(payload.get("dir_nonleg_base"))
    grad_arm = safe_float(payload.get("direct_grad_norm_out_arm"))
    grad_else = safe_float(payload.get("direct_grad_norm_out_else"))
    if not math.isfinite(safe_float(payload.get("leg_over_nonleg"))) and math.isfinite(dir_nonleg) and dir_nonleg != 0.0:
        payload["leg_over_nonleg"] = float(dir_leg / dir_nonleg)
    if not math.isfinite(safe_float(payload.get("grad_arm_over_else"))) and math.isfinite(grad_else) and grad_else != 0.0:
        payload["grad_arm_over_else"] = float(grad_arm / grad_else)
    return payload


def enrich_stage6_init(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out["step1"] = enrich_init_section(payload.get("step1", {}))
    out["head20_mean"] = enrich_init_section(payload.get("head20_mean", {}))
    return out


def is_finite(x: Any) -> bool:
    return math.isfinite(safe_float(x))


def improvement(base: Any, cur: Any) -> float:
    a = safe_float(base)
    b = safe_float(cur)
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return float(a - b)


def ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def stage6_paths(name: str) -> Dict[str, Path]:
    lane_root = STAGE6_ROOT / name
    model_dir = STAGE6_MODEL_ROOT / name
    run_name = f"WalkF_stage6_{name}_{RUN_DATE}"
    return {
        "lane_root": lane_root,
        "lane_log": lane_root / "lane.log",
        "surgery_ckpt": model_dir / "surgery" / f"{name}.pth",
        "surgery_report": lane_root / "rollback_report.json",
        "stage6_model_dir": model_dir / "posttrain",
        "stage6_log_json": model_dir / "posttrain" / f"posttrain_log_{run_name}.json",
        "stage6_ckpt": model_dir / "posttrain" / f"ckpt_last_{run_name}.pth",
        "stage6_init_json": lane_root / "posttrain_stage6_init_stats.json",
        "stage6_eval_dir": lane_root / "stage6_freerun",
        "stage6_eval_json": lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        "stage6_group_json": lane_root / "stage6_group_summary.json",
        "status_json": lane_root / "status.json",
        "run_name": Path(run_name),
    }


def downstream_paths(name: str) -> Dict[str, Path]:
    out_root = DOWNSTREAM_ROOT / name
    model_root = DOWNSTREAM_MODEL_ROOT / name
    return {
        "lane_root": out_root,
        "lane_log": out_root / "lane.log",
        "status_json": out_root / "status.json",
        "cfg_70b_replace": CONFIG_ROOT / f"{name}_70b_replace.json",
        "cfg_70r": CONFIG_ROOT / f"{name}_70R.json",
        "warmstart_ckpt": model_root / "warmstart" / f"ckpt_last_{name}_replace_zerophase_{RUN_DATE}.pth",
        "warmstart_report": out_root / "warmstart" / "replace_zerophase_report.json",
        "ckpt_70a": model_root / "70a" / f"ckpt_last_WalkF_stage7_70a_from_{name}_{RUN_DATE}.pth",
        "ckpt_70b": model_root / "70b" / f"ckpt_last_WalkF_stage7_70b_from_{name}_{RUN_DATE}.pth",
        "ckpt_70b_replace": model_root / "70b_replace" / f"ckpt_last_WalkF_stage7_70b_replace_from_{name}_{RUN_DATE}.pth",
        "ckpt_70r": model_root / "70R" / f"ckpt_last_WalkF_stage7_70R_from_{name}_{RUN_DATE}.pth",
        "ckpt_71": model_root / "71" / f"ckpt_last_WalkF_stage7_71_from_{name}_{RUN_DATE}.pth",
        "ckpt_72": model_root / "72" / f"ckpt_last_WalkF_stage7_72_from_{name}_{RUN_DATE}.pth",
        "ckpt_lambda": model_root / "lambda" / f"ckpt_last_WalkF_stage7_lambda_from_{name}_{RUN_DATE}.pth",
        "eval_strict_dir": out_root / "eval_pretrain_contact",
        "eval_strict_json": out_root / "eval_pretrain_contact" / "Walk_F_freerun_cycles.json",
        "eval_strict_group": out_root / "eval_pretrain_contact_group_summary.json",
        "eval_model_dir": out_root / "eval_model_source",
        "eval_model_json": out_root / "eval_model_source" / "Walk_F_freerun_cycles.json",
        "eval_model_group": out_root / "eval_model_source_group_summary.json",
    }


def load_baseline_rows() -> Dict[str, Dict[str, Any]]:
    payload = load_json(BASELINE_COMPARE_JSON)
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        raise RuntimeError(f"invalid lanes in {BASELINE_COMPARE_JSON}")
    rows = {str(row["name"]): row for row in lanes if isinstance(row, dict) and "name" in row}
    if SOURCE_BASELINE_CASE not in rows:
        raise RuntimeError(f"missing {SOURCE_BASELINE_CASE} in {BASELINE_COMPARE_JSON}")
    return rows


def load_stage6_case(name: str) -> Dict[str, Any]:
    payload = load_json(PLANTRANSPLANT_SUMMARY_JSON)
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError(f"invalid cases in {PLANTRANSPLANT_SUMMARY_JSON}")
    for case in cases:
        if isinstance(case, dict) and str(case.get("name")) == name:
            return case
    raise RuntimeError(f"missing case {name} in {PLANTRANSPLANT_SUMMARY_JSON}")


def build_group_catalog(control_report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prefix_map = control_report.get("recipient_prefix_map", {})
    if not isinstance(prefix_map, Mapping):
        raise RuntimeError("control transplant report missing recipient_prefix_map")
    exact_keys = {str(x) for x in control_report.get("exact_keys", [])}
    report_keys = [str(x) for x in control_report.get("keys", [])]

    groups: List[Dict[str, Any]] = []
    covered: List[str] = []
    for spec in GROUP_SPECS:
        keys: List[str] = []
        for prefix in spec.prefixes:
            matches = prefix_map.get(prefix, None)
            if not isinstance(matches, list) or not matches:
                raise RuntimeError(f"control transplant report missing keys for prefix {prefix}")
            keys.extend(str(x) for x in matches)
        for key in spec.exact_keys:
            if key not in exact_keys:
                raise RuntimeError(f"control transplant report missing exact key {key}")
            keys.append(key)
        keys = ordered_unique(keys)
        groups.append(
            {
                "name": spec.name,
                "label": spec.label,
                "prefixes": list(spec.prefixes),
                "exact_keys_requested": list(spec.exact_keys),
                "exact_keys": [key for key in keys if key in exact_keys],
                "keys": keys,
                "key_count": int(len(keys)),
                "rationale": spec.rationale,
            }
        )
        covered.extend(keys)

    covered = ordered_unique(covered)
    if set(covered) != set(report_keys):
        raise RuntimeError(
            "group coverage mismatch:\n"
            f"missing={sorted(set(report_keys) - set(covered))}\n"
            f"extra={sorted(set(covered) - set(report_keys))}"
        )
    return groups


def group_index(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        out[str(group["name"])] = dict(group)
    return out


def build_rollback_report(
    *,
    lane: LaneSpec,
    rollback_keys: Sequence[str],
    group_catalog: Mapping[str, Mapping[str, Any]],
    control_case: Mapping[str, Any],
    control_report: Mapping[str, Any],
    effective_changed_keys: Sequence[str],
) -> Dict[str, Any]:
    control_paths = control_case.get("paths", {})
    groups = [group_catalog[name] for name in lane.rollback_groups]
    return {
        "lane": lane.name,
        "lane_code": lane.code,
        "lane_label": lane.label,
        "note": lane.note,
        "rollback_groups": [
            {
                "name": group["name"],
                "label": group["label"],
                "prefixes": group["prefixes"],
                "exact_keys": group["exact_keys"],
                "keys": group["keys"],
                "key_count": group["key_count"],
                "rationale": group["rationale"],
            }
            for group in groups
        ],
        "rollback_keys": list(rollback_keys),
        "rollback_key_count": int(len(rollback_keys)),
        "effective_changed_keys": list(effective_changed_keys),
        "effective_changed_key_count": int(len(effective_changed_keys)),
        "parent_control_case": str(control_case.get("name")),
        "parent_control_stage6_ckpt": str(control_paths.get("stage6_ckpt")),
        "parent_control_surgery_ckpt": str(control_paths.get("surgery_ckpt")),
        "control_transplant_report": str(control_paths.get("surgery_report")),
        "recipient_ckpt": str(control_report.get("recipient_ckpt")),
        "donor_ckpt": str(control_report.get("donor_ckpt")),
        "source_baseline_case": SOURCE_BASELINE_CASE,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def ensure_rollback_surgery(
    lane: LaneSpec,
    *,
    control_case: Mapping[str, Any],
    control_report: Mapping[str, Any],
    group_catalog: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    paths = stage6_paths(lane.name)
    if paths["surgery_ckpt"].is_file() and paths["surgery_report"].is_file():
        return load_json(paths["surgery_report"])

    control_surgery_ckpt = Path(str(control_case["paths"]["surgery_ckpt"]))
    recipient_ckpt = Path(str(control_report["recipient_ckpt"]))
    if not control_surgery_ckpt.is_file():
        raise RuntimeError(f"missing control surgery ckpt: {control_surgery_ckpt}")
    if not recipient_ckpt.is_file():
        raise RuntimeError(f"missing recipient ckpt: {recipient_ckpt}")

    rollback_keys = ordered_unique(
        key
        for group_name in lane.rollback_groups
        for key in group_catalog[group_name]["keys"]
    )

    control_obj = torch.load(control_surgery_ckpt, map_location="cpu")
    recipient_obj = torch.load(recipient_ckpt, map_location="cpu")
    control_state = resolve_model_state(control_obj, control_surgery_ckpt)
    recipient_state = resolve_model_state(recipient_obj, recipient_ckpt)
    out_obj = copy.deepcopy(control_obj)
    out_state = resolve_model_state(out_obj, control_surgery_ckpt)

    per_key: List[Dict[str, Any]] = []
    effective_changed: List[str] = []
    for key in rollback_keys:
        if key not in control_state or key not in recipient_state:
            raise RuntimeError(f"missing rollback key {key}")
        control_value = control_state[key]
        recipient_value = recipient_state[key]
        if torch.is_tensor(control_value) != torch.is_tensor(recipient_value):
            raise RuntimeError(f"type mismatch for rollback key {key}")
        if torch.is_tensor(control_value):
            if tuple(control_value.shape) != tuple(recipient_value.shape):
                raise RuntimeError(f"shape mismatch for rollback key {key}")
            same_before = bool(torch.equal(control_value, recipient_value))
            out_state[key] = recipient_value.clone()
        else:
            same_before = control_value == recipient_value
            out_state[key] = copy.deepcopy(recipient_value)
        if not same_before:
            effective_changed.append(key)
        per_key.append(
            {
                "key": key,
                "same_as_recipient_before": bool(same_before),
            }
        )

    paths["surgery_ckpt"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, paths["surgery_ckpt"])

    reloaded = torch.load(paths["surgery_ckpt"], map_location="cpu")
    reloaded_state = resolve_model_state(reloaded, paths["surgery_ckpt"])
    verify_count = 0
    for key in rollback_keys:
        recipient_value = recipient_state[key]
        after_value = reloaded_state[key]
        if torch.is_tensor(recipient_value):
            if torch.equal(recipient_value, after_value):
                verify_count += 1
        elif recipient_value == after_value:
            verify_count += 1

    report = build_rollback_report(
        lane=lane,
        rollback_keys=rollback_keys,
        group_catalog=group_catalog,
        control_case=control_case,
        control_report=control_report,
        effective_changed_keys=effective_changed,
    )
    report["surgery_ckpt"] = str(paths["surgery_ckpt"])
    report["verify_same_as_recipient_after_count"] = int(verify_count)
    report["verified_all_after"] = bool(verify_count == len(rollback_keys))
    report["per_key"] = per_key
    if not bool(report["verified_all_after"]):
        raise RuntimeError(f"rollback verification failed for {lane.name}")
    write_json(paths["surgery_report"], report)
    return report


def ensure_stage6_lane(lane: LaneSpec) -> None:
    paths = stage6_paths(lane.name)
    run_name = str(paths["run_name"])

    if not paths["stage6_ckpt"].is_file() or not paths["stage6_log_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(STAGE6_CONFIG),
                "--ckpt_in",
                str(paths["surgery_ckpt"]),
                "--out_dir",
                str(paths["stage6_model_dir"]),
                "--run_name",
                run_name,
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--encoder_bundle",
                str(ENCODER_BUNDLE),
                "--posttrain_contacts_pretrain_affine_stats",
                str(AFFINE_STATS),
            ],
            log_file=paths["lane_log"],
        )

    if not paths["stage6_init_json"].is_file():
        extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])

    if not paths["stage6_group_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(TEACHER),
                "--model",
                str(paths["stage6_ckpt"]),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "pretrain_contact",
                "--contacts_meas_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--contacts_meas_pretrain_affine_stats",
                str(AFFINE_STATS),
                "--encoder-bundle",
                str(ENCODER_BUNDLE),
                "--export_joint_direct_geolocal_series",
                "--out",
                str(paths["stage6_eval_dir"]),
                "--force",
            ],
            log_file=paths["lane_log"],
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "tools" / "phasea_group_summary.py"),
                str(paths["stage6_eval_json"]),
                "--cycle_gte",
                "1",
                "--drop_wrap",
                "--out",
                str(paths["stage6_group_json"]),
            ],
            log_file=paths["lane_log"],
        )

    write_json(
        paths["status_json"],
        {
            "lane": lane.name,
            "surgery_ckpt": str(paths["surgery_ckpt"]),
            "stage6_ckpt": str(paths["stage6_ckpt"]),
            "stage6_group_summary": str(paths["stage6_group_json"]),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def build_control_stage6_entry(
    *,
    control_case: Mapping[str, Any],
    control_report: Mapping[str, Any],
    source_baseline_row: Mapping[str, Any],
    group_catalog: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    stage6_exit = dict(control_case["stage6_exit"])
    stage6_init = enrich_stage6_init(control_case["stage6_init"])
    source_stage6 = {
        "all_ex_root": safe_float(source_baseline_row["stage6_exit"]["all_ex_root_mean"]),
        "leg": safe_float(source_baseline_row["stage6_exit"]["leg_mean"]),
        "nonleg": safe_float(source_baseline_row["stage6_exit"]["nonleg_mean"]),
        "arm": safe_float(source_baseline_row["stage6_exit"]["arm_mean"]),
        "else": safe_float(source_baseline_row["stage6_exit"]["else_mean"]),
    }
    return {
        "code": "E",
        "name": str(control_case["name"]),
        "label": "E. full oldplan control",
        "case_type": "control_reused",
        "rollback_groups": [group_catalog[name]["label"] for name in group_catalog.keys()],
        "rollback_group_names": list(group_catalog.keys()),
        "rollback_key_count": int(sum(int(group_catalog[name]["key_count"]) for name in group_catalog.keys())),
        "rollback_effective_changed_key_count": int(control_case["transplant_report"]["changed_key_count"]),
        "note": "Reused existing full old-plan transplant result as control lane; no retrain/rerun here.",
        "stage6_exit": stage6_exit,
        "stage6_init": stage6_init,
        "delta_vs_full_oldplan_control": {key: 0.0 for key in stage6_exit.keys()},
        "delta_vs_cp015_source_baseline": {
            key: diff(stage6_exit.get(key), source_stage6.get(key))
            for key in stage6_exit.keys()
        },
        "improvement_vs_cp015_source_baseline": {
            key: improvement(source_stage6.get(key), stage6_exit.get(key))
            for key in stage6_exit.keys()
        },
        "qualifies_threshold": True,
        "selection_status": "control_reused",
        "paths": dict(control_case["paths"]),
        "source_paths": {
            "source_baseline_ckpt": str(source_baseline_row["ckpt"]),
            "recipient_ckpt": str(control_report["recipient_ckpt"]),
            "donor_ckpt": str(control_report["donor_ckpt"]),
            "control_surgery_ckpt": str(control_case["paths"]["surgery_ckpt"]),
            "control_transplant_report": str(control_case["paths"]["surgery_report"]),
        },
    }


def build_stage6_entry(
    lane: LaneSpec,
    *,
    source_baseline_row: Mapping[str, Any],
    control_case: Mapping[str, Any],
    group_catalog: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    paths = stage6_paths(lane.name)
    rollback_report = load_json(paths["surgery_report"])
    stage6_exit = group_metrics(paths["stage6_group_json"])
    stage6_init = enrich_stage6_init(load_json(paths["stage6_init_json"]))
    control_stage6 = control_case["stage6_exit"]
    source_stage6 = {
        "all_ex_root": safe_float(source_baseline_row["stage6_exit"]["all_ex_root_mean"]),
        "leg": safe_float(source_baseline_row["stage6_exit"]["leg_mean"]),
        "nonleg": safe_float(source_baseline_row["stage6_exit"]["nonleg_mean"]),
        "arm": safe_float(source_baseline_row["stage6_exit"]["arm_mean"]),
        "else": safe_float(source_baseline_row["stage6_exit"]["else_mean"]),
    }
    delta_vs_control = {
        key: diff(stage6_exit.get(key), control_stage6.get(key))
        for key in stage6_exit.keys()
    }
    qualifies = (
        safe_float(delta_vs_control["all_ex_root"]) <= 0.02
        and safe_float(delta_vs_control["leg"]) <= 0.05
    )
    entry = {
        "code": lane.code,
        "name": lane.name,
        "label": lane.label,
        "case_type": "rollback",
        "rollback_groups": [group_catalog[name]["label"] for name in lane.rollback_groups],
        "rollback_group_names": list(lane.rollback_groups),
        "rollback_key_count": int(rollback_report["rollback_key_count"]),
        "rollback_effective_changed_key_count": int(rollback_report["effective_changed_key_count"]),
        "note": lane.note,
        "stage6_exit": stage6_exit,
        "stage6_init": stage6_init,
        "delta_vs_full_oldplan_control": delta_vs_control,
        "delta_vs_cp015_source_baseline": {
            key: diff(stage6_exit.get(key), source_stage6.get(key))
            for key in stage6_exit.keys()
        },
        "improvement_vs_cp015_source_baseline": {
            key: improvement(source_stage6.get(key), stage6_exit.get(key))
            for key in stage6_exit.keys()
        },
        "init_delta_vs_full_oldplan_control": {
            "step1": delta_block(stage6_init["step1"], control_case["stage6_init"]["step1"]),
            "head20_mean": delta_block(stage6_init["head20_mean"], control_case["stage6_init"]["head20_mean"]),
        },
        "init_delta_vs_cp015_source_baseline": {
            "step1": delta_block(stage6_init["step1"], source_baseline_row["stage6_init"]["step1"]),
            "head20_mean": delta_block(stage6_init["head20_mean"], source_baseline_row["stage6_init"]["head20_mean"]),
        },
        "qualifies_threshold": bool(qualifies),
        "selection_status": "screened",
        "paths": {
            "lane_root": str(paths["lane_root"]),
            "surgery_ckpt": str(paths["surgery_ckpt"]),
            "surgery_report": str(paths["surgery_report"]),
            "stage6_ckpt": str(paths["stage6_ckpt"]),
            "stage6_init_stats": str(paths["stage6_init_json"]),
            "stage6_group_summary": str(paths["stage6_group_json"]),
        },
        "source_paths": {
            "source_baseline_ckpt": str(source_baseline_row["ckpt"]),
            "recipient_ckpt": str(rollback_report["recipient_ckpt"]),
            "donor_ckpt": str(rollback_report["donor_ckpt"]),
            "parent_control_surgery_ckpt": str(rollback_report["parent_control_surgery_ckpt"]),
        },
        "rollback_report": rollback_report,
    }
    return entry


def stage6_sort_key(entry: Mapping[str, Any]) -> Tuple[float, float, float]:
    delta = entry["delta_vs_full_oldplan_control"]
    return (
        safe_float(delta.get("all_ex_root")),
        safe_float(delta.get("leg")),
        safe_float(delta.get("nonleg")),
    )


def select_promoted_lanes(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rollback_entries = [entry for entry in entries if str(entry.get("case_type")) == "rollback"]
    ranked = sorted(rollback_entries, key=stage6_sort_key)
    qualified = [entry for entry in ranked if bool(entry.get("qualifies_threshold"))]
    promoted = qualified[:2]
    reason = "threshold_pass"
    note = "Promote lanes whose Stage6 gap stays within +0.02 all_ex_root and +0.05 leg vs full oldplan control."
    if not promoted and ranked:
        promoted = ranked[:1]
        reason = "top1_confirm_only"
        note = (
            "No rollback lane met the Stage6 threshold; promote only top-1 for downstream confirmation. "
            "This lane is not treated as promotion-qualified by Stage6."
        )
    promoted_names = [str(entry["name"]) for entry in promoted]
    return {
        "ranked_names": [str(entry["name"]) for entry in ranked],
        "qualified_names": [str(entry["name"]) for entry in qualified],
        "promoted_names": promoted_names,
        "promote_reason": reason,
        "note": note,
        "thresholds": {
            "all_ex_root_max_delta_vs_control": 0.02,
            "leg_max_delta_vs_control": 0.05,
        },
    }


def ensure_downstream_lane(
    lane_name: str,
    *,
    stage6_ckpt: Path,
) -> Dict[str, Any]:
    paths = downstream_paths(lane_name)
    run_name_70a = f"WalkF_stage7_70a_from_{lane_name}_{RUN_DATE}"
    run_name_70b = f"WalkF_stage7_70b_from_{lane_name}_{RUN_DATE}"
    run_name_70b_replace = f"WalkF_stage7_70b_replace_from_{lane_name}_{RUN_DATE}"
    run_name_70r = f"WalkF_stage7_70R_from_{lane_name}_{RUN_DATE}"
    run_name_71 = f"WalkF_stage7_71_from_{lane_name}_{RUN_DATE}"
    run_name_72 = f"WalkF_stage7_72_from_{lane_name}_{RUN_DATE}"
    run_name_lambda = f"WalkF_stage7_lambda_from_{lane_name}_{RUN_DATE}"

    log(f"=== downstream {lane_name}: 70a ===")
    ckpt_70a = run_posttrain_stage(
        config=CONFIG_70A,
        ckpt_in=stage6_ckpt,
        out_dir=paths["ckpt_70a"].parent,
        run_name=run_name_70a,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: 70b ===")
    ckpt_70b = run_posttrain_stage(
        config=CONFIG_70B,
        ckpt_in=ckpt_70a,
        out_dir=paths["ckpt_70b"].parent,
        run_name=run_name_70b,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: replace zerophase warmstart ===")
    create_replace_zerophase_warmstart(
        src_ckpt=ckpt_70a,
        dst_ckpt=paths["warmstart_ckpt"],
        report_json=paths["warmstart_report"],
    )

    log(f"=== downstream {lane_name}: new70b_replace ===")
    cfg_70b_replace = make_generated_config(
        CONFIG_70B,
        paths["cfg_70b_replace"],
        {
            "ckpt_in": str(paths["warmstart_ckpt"]),
            "out_dir": str(paths["ckpt_70b_replace"].parent),
            "run_name": run_name_70b_replace,
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70b_replace = run_posttrain_stage(
        config=cfg_70b_replace,
        ckpt_in=paths["warmstart_ckpt"],
        out_dir=paths["ckpt_70b_replace"].parent,
        run_name=run_name_70b_replace,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: 70R(s180) ===")
    cfg_70r = make_generated_config(
        CONFIG_70R,
        paths["cfg_70r"],
        {
            "ckpt_in": str(ckpt_70b_replace),
            "out_dir": str(paths["ckpt_70r"].parent),
            "run_name": run_name_70r,
            "lr": 3e-4,
            "epochs": 1,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70r = run_70r_promote(
        config_json=cfg_70r,
        out_dir=paths["ckpt_70r"].parent,
        run_name=run_name_70r,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: 71 ===")
    ckpt_71 = run_posttrain_stage(
        config=CONFIG_71,
        ckpt_in=ckpt_70r,
        out_dir=paths["ckpt_71"].parent,
        run_name=run_name_71,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: 72 ===")
    ckpt_72 = run_posttrain_stage(
        config=CONFIG_72,
        ckpt_in=ckpt_71,
        out_dir=paths["ckpt_72"].parent,
        run_name=run_name_72,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: lambda final ===")
    ckpt_lambda = run_posttrain_stage(
        config=CONFIG_LAMBDA,
        ckpt_in=ckpt_72,
        out_dir=paths["ckpt_lambda"].parent,
        run_name=run_name_lambda,
        log_file=paths["lane_log"],
    )

    log(f"=== downstream {lane_name}: strict eval ===")
    eval_strict_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_strict_dir"],
        contacts_source="pretrain_contact",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_strict_json, paths["eval_strict_group"], log_file=paths["lane_log"])

    log(f"=== downstream {lane_name}: model-source eval ===")
    eval_model_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_model_dir"],
        contacts_source="model",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_model_json, paths["eval_model_group"], log_file=paths["lane_log"])

    status = {
        "lane": lane_name,
        "stage_ckpts": {
            "70a": str(ckpt_70a),
            "70b": str(ckpt_70b),
            "warmstart": str(paths["warmstart_ckpt"]),
            "70b_replace": str(ckpt_70b_replace),
            "70R": str(ckpt_70r),
            "71": str(ckpt_71),
            "72": str(ckpt_72),
            "lambda": str(ckpt_lambda),
        },
        "evals": {
            "strict_pretrain_contact": str(eval_strict_json),
            "model_source": str(eval_model_json),
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(paths["status_json"], status)
    return status


def build_downstream_entry(
    lane_entry: Mapping[str, Any],
    *,
    control_chain: Mapping[str, Any],
    accepted_refs: Mapping[str, Any],
) -> Dict[str, Any]:
    lane_name = str(lane_entry["name"])
    paths = downstream_paths(lane_name)
    strict_masked = masked_metric_means(paths["eval_strict_json"])
    strict_group = group_metrics(paths["eval_strict_group"])
    strict_windows = window_group_stats(paths["eval_strict_json"])
    model_masked = masked_metric_means(paths["eval_model_json"])
    model_group = group_metrics(paths["eval_model_group"])
    model_windows = window_group_stats(paths["eval_model_json"])

    control_strict = control_chain["final_evals"]["strict_pretrain_contact"]
    control_model = control_chain["final_evals"]["model_source"]
    accepted_old = accepted_refs["accepted_old_baseline_r5"]
    accepted_final = accepted_refs["accepted_final_model_source"]

    cmp_model_control = {
        "masked_means_delta": {
            key: diff(model_masked.get(key), control_model["masked_means"].get(key))
            for key in (
                "DirectGeoLocalDeg",
                "DirectGeoLocalDegWeighted",
                "BlendGeoLocalDeg",
                "BlendGeoLocalDegWeighted",
                "GeoLocalDeg",
                "GeoLocalDegWeighted",
            )
        },
        "direct_group_delta": {
            key: diff(model_group.get(key), control_model["direct_group_summary"].get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
    }
    cmp_strict_control = {
        "masked_means_delta": {
            key: diff(strict_masked.get(key), control_strict["masked_means"].get(key))
            for key in (
                "DirectGeoLocalDeg",
                "DirectGeoLocalDegWeighted",
                "BlendGeoLocalDeg",
                "BlendGeoLocalDegWeighted",
                "GeoLocalDeg",
                "GeoLocalDegWeighted",
            )
        },
        "direct_group_delta": {
            key: diff(strict_group.get(key), control_strict["direct_group_summary"].get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
    }
    cmp_model_accepted = {
        "masked_means_delta": {
            key: diff(model_masked.get(key), accepted_final["masked_means"].get(key))
            for key in (
                "DirectGeoLocalDeg",
                "DirectGeoLocalDegWeighted",
                "BlendGeoLocalDeg",
                "BlendGeoLocalDegWeighted",
                "GeoLocalDeg",
                "GeoLocalDegWeighted",
            )
        },
        "direct_group_delta": {
            key: diff(model_group.get(key), accepted_final["direct_group_summary"].get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
    }
    cmp_model_old = {
        "masked_means_delta": {
            key: diff(model_masked.get(key), accepted_old["masked_means"].get(key))
            for key in (
                "DirectGeoLocalDeg",
                "DirectGeoLocalDegWeighted",
                "BlendGeoLocalDeg",
                "BlendGeoLocalDegWeighted",
                "GeoLocalDeg",
                "GeoLocalDegWeighted",
            )
        },
        "direct_group_delta": {
            key: diff(model_group.get(key), accepted_old["direct_group_summary"].get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
    }
    cmp_strict_old = {
        "masked_means_delta": {
            key: diff(strict_masked.get(key), accepted_old["masked_means"].get(key))
            for key in (
                "DirectGeoLocalDeg",
                "DirectGeoLocalDegWeighted",
                "BlendGeoLocalDeg",
                "BlendGeoLocalDegWeighted",
                "GeoLocalDeg",
                "GeoLocalDegWeighted",
            )
        },
        "direct_group_delta": {
            key: diff(strict_group.get(key), accepted_old["direct_group_summary"].get(key))
            for key in ("all_ex_root", "leg", "nonleg")
        },
    }

    strict_better_than_control = safe_float(cmp_strict_control["direct_group_delta"]["all_ex_root"]) < 0.0
    model_better_than_control = safe_float(cmp_model_control["direct_group_delta"]["all_ex_root"]) < 0.0
    strict_beats_accepted = safe_float(cmp_strict_old["direct_group_delta"]["all_ex_root"]) < 0.0
    model_beats_accepted = safe_float(cmp_model_old["direct_group_delta"]["all_ex_root"]) < 0.0

    return {
        "name": lane_name,
        "label": lane_entry["label"],
        "selection_status": lane_entry["selection_status"],
        "selection_note": lane_entry.get("selection_note"),
        "stage6_exit": lane_entry["stage6_exit"],
        "stage_paths": {
            "transplant_stage6": str(lane_entry["paths"]["stage6_ckpt"]),
            "70a": str(paths["ckpt_70a"]),
            "70b": str(paths["ckpt_70b"]),
            "new70b_replace": str(paths["ckpt_70b_replace"]),
            "70R": str(paths["ckpt_70r"]),
            "71": str(paths["ckpt_71"]),
            "72": str(paths["ckpt_72"]),
            "lambda_final": str(paths["ckpt_lambda"]),
            "warmstart": str(paths["warmstart_ckpt"]),
        },
        "config_paths": {
            "70a": str(CONFIG_70A),
            "70b": str(CONFIG_70B),
            "new70b_replace": str(paths["cfg_70b_replace"]),
            "70R": str(paths["cfg_70r"]),
            "71": str(CONFIG_71),
            "72": str(CONFIG_72),
            "lambda_final": str(CONFIG_LAMBDA),
        },
        "final_evals": {
            "strict_pretrain_contact": {
                "masked_means": strict_masked,
                "direct_group_summary": strict_group,
                "window_summary": strict_windows,
                "paths": {
                    "eval_json": str(paths["eval_strict_json"]),
                    "group_summary": str(paths["eval_strict_group"]),
                },
            },
            "model_source": {
                "masked_means": model_masked,
                "direct_group_summary": model_group,
                "window_summary": model_windows,
                "paths": {
                    "eval_json": str(paths["eval_model_json"]),
                    "group_summary": str(paths["eval_model_group"]),
                },
            },
        },
        "comparisons": {
            "model_source_vs_full_oldplan_control": cmp_model_control,
            "strict_pretrain_contact_vs_full_oldplan_control": cmp_strict_control,
            "model_source_vs_accepted_final_model_source": cmp_model_accepted,
            "model_source_vs_accepted_old_baseline_r5": cmp_model_old,
            "strict_pretrain_contact_vs_accepted_old_baseline_r5": cmp_strict_old,
        },
        "consistency": {
            "strict_vs_model_both_beat_old_baseline": bool(strict_beats_accepted and model_beats_accepted),
            "strict_vs_model_both_beat_full_oldplan_control": bool(strict_better_than_control and model_better_than_control),
            "strict_model_control_sign_match": bool(strict_better_than_control == model_better_than_control),
        },
    }


def apply_selection(
    stage6_entries: Sequence[Dict[str, Any]],
    selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    promote_set = set(str(x) for x in selection["promoted_names"])
    promote_reason = str(selection["promote_reason"])
    out: List[Dict[str, Any]] = []
    for entry in stage6_entries:
        entry = copy.deepcopy(entry)
        if str(entry.get("name")) in promote_set:
            entry["selection_status"] = "promoted"
            if promote_reason == "top1_confirm_only":
                entry["selection_note"] = (
                    "No Stage6-qualified rollback lane; promoted only as confirm-only downstream check."
                )
            else:
                entry["selection_note"] = (
                    "Within Stage6 threshold vs full oldplan control; promoted to downstream chain."
                )
        elif str(entry.get("case_type")) == "rollback":
            entry["selection_status"] = "screened_out"
        out.append(entry)
    return out


def build_answers(
    *,
    stage6_entries: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    downstream_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rollback_entries = [entry for entry in stage6_entries if str(entry.get("case_type")) == "rollback"]
    ranked = sorted(rollback_entries, key=stage6_sort_key)
    worst = sorted(
        rollback_entries,
        key=lambda entry: (
            -safe_float(entry["delta_vs_full_oldplan_control"]["all_ex_root"]),
            -safe_float(entry["delta_vs_full_oldplan_control"]["leg"]),
            -safe_float(entry["delta_vs_full_oldplan_control"]["nonleg"]),
        ),
    )[0]
    best = ranked[0] if ranked else None
    best_name = str(best["name"]) if best is not None else None
    smaller_exists = bool(
        best is not None
        and bool(best.get("qualifies_threshold"))
    )

    promoted_best = downstream_entries[0] if downstream_entries else None
    penetrates = False
    better_than_full_control_overall = False
    cleaner_but_mixed_vs_full_control = False
    better_than_accepted = False
    if promoted_best is not None:
        cmp_old = promoted_best["comparisons"]["model_source_vs_accepted_old_baseline_r5"]
        cmp_ctrl = promoted_best["comparisons"]["model_source_vs_full_oldplan_control"]
        cmp_ctrl_strict = promoted_best["comparisons"]["strict_pretrain_contact_vs_full_oldplan_control"]
        cmp_final = promoted_best["comparisons"]["model_source_vs_accepted_final_model_source"]
        penetrates = (
            safe_float(cmp_old["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
            and safe_float(cmp_old["direct_group_delta"]["all_ex_root"]) < 0.0
        )
        better_than_full_control_overall = (
            safe_float(cmp_ctrl["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
            and safe_float(cmp_ctrl["direct_group_delta"]["all_ex_root"]) < 0.0
        )
        cleaner_but_mixed_vs_full_control = (
            better_than_full_control_overall
            and (
                safe_float(cmp_ctrl["direct_group_delta"]["leg"]) > 0.0
                or safe_float(cmp_ctrl_strict["direct_group_delta"]["leg"]) > 0.0
            )
        )
        better_than_accepted = (
            safe_float(cmp_final["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
            and safe_float(cmp_final["direct_group_delta"]["all_ex_root"]) < 0.0
        )

    return {
        "q1_most_critical_group": {
            "lane": str(worst["name"]),
            "label": str(worst["label"]),
            "delta_vs_control": dict(worst["delta_vs_full_oldplan_control"]),
            "interpretation": "Largest rollback damage vs full oldplan control at Stage6 is treated as the most critical group.",
        },
        "q2_has_smaller_stage6_safe_version": {
            "value": bool(smaller_exists),
            "best_stage6_lane": best_name,
            "selection_reason": str(selection["promote_reason"]),
        },
        "q3_smaller_version_penetrates_to_lambda_final": {
            "value": bool(penetrates),
            "lane": str(promoted_best["name"]) if promoted_best is not None else None,
            "note": "Measured against the accepted old-baseline r5 anchor, same as the previous round carry claim.",
        },
        "q4_smaller_version_beats_full_oldplan_chain": {
            "value": bool(better_than_full_control_overall and not cleaner_but_mixed_vs_full_control),
            "lane": str(promoted_best["name"]) if promoted_best is not None else None,
            "note": (
                "Overall direct/all_ex_root improves vs full oldplan control, but leg regresses, so treat this as a mixed tradeoff rather than a clean win."
                if cleaner_but_mixed_vs_full_control
                else "Requires a clean overall win without the leg tradeoff."
            ),
        },
        "q5_smaller_version_beats_current_accepted_final": {
            "value": bool(better_than_accepted),
            "lane": str(promoted_best["name"]) if promoted_best is not None else None,
        },
        "q6_can_propose_new_cleaner_challenger": {
            "value": bool(smaller_exists),
            "lane": best_name,
            "note": (
                "Cleaner challenger/control lane only; baseline switch still requires clearing the current accepted final."
                if smaller_exists
                else "No rollback-safe simplification cleared the Stage6 screening bar."
            ),
        },
        "q7_should_pause_further_simplification": {
            "value": bool(not smaller_exists),
            "note": (
                "If no Stage6-safe simplification survives, the next step should be direct planner mechanism validation."
            ),
        },
    }


def build_summary(
    *,
    group_catalog: Sequence[Mapping[str, Any]],
    source_baseline_row: Mapping[str, Any],
    control_case: Mapping[str, Any],
    control_chain: Mapping[str, Any],
    accepted_refs: Mapping[str, Any],
    stage6_entries: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    downstream_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    answers = build_answers(
        stage6_entries=stage6_entries,
        selection=selection,
        downstream_entries=downstream_entries,
    )
    return {
        "run_date": RUN_DATE,
        "policy": {
            "baseline_case": SOURCE_BASELINE_CASE,
            "control_stage6_case": CONTROL_STAGE6_CASE,
            "baseline_compare_summary": str(BASELINE_COMPARE_JSON),
            "stage6_plantransplant_summary": str(PLANTRANSPLANT_SUMMARY_JSON),
            "full_oldplan_downstream_summary": str(OLDPLAN_CHAIN_SUMMARY_JSON),
            "stage6_config": str(STAGE6_CONFIG),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "strict_eval": {
                "contacts_meas_source": "pretrain_contact",
                "contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
                "contacts_meas_pretrain_affine_stats": str(AFFINE_STATS),
                "encoder_bundle": str(ENCODER_BUNDLE),
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
            },
            "model_source_eval": {
                "contacts_meas_source": "model",
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
            },
        },
        "references": {
            "source_baseline": {
                "name": SOURCE_BASELINE_CASE,
                "ckpt": str(source_baseline_row["ckpt"]),
                "stage6_exit": {
                    "all_ex_root": safe_float(source_baseline_row["stage6_exit"]["all_ex_root_mean"]),
                    "leg": safe_float(source_baseline_row["stage6_exit"]["leg_mean"]),
                    "nonleg": safe_float(source_baseline_row["stage6_exit"]["nonleg_mean"]),
                    "arm": safe_float(source_baseline_row["stage6_exit"]["arm_mean"]),
                    "else": safe_float(source_baseline_row["stage6_exit"]["else_mean"]),
                },
            },
            "full_oldplan_stage6_control": control_case,
            "full_oldplan_downstream_control": control_chain,
            "accepted_refs": accepted_refs,
        },
        "key_groups": list(group_catalog),
        "stage6_screening": {
            "selection": dict(selection),
            "lanes": list(stage6_entries),
        },
        "downstream": {
            "control_reused": {
                "summary_json": str(OLDPLAN_CHAIN_SUMMARY_JSON),
                "stage_paths": dict(control_chain["stages"]),
                "final_evals": dict(control_chain["final_evals"]),
            },
            "promoted_lanes": list(downstream_entries),
        },
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    stage6_lanes = summary["stage6_screening"]["lanes"]
    stage6_rollbacks = [lane for lane in stage6_lanes if lane["case_type"] == "rollback"]
    downstream_lanes = summary["downstream"]["promoted_lanes"]
    answers = summary["answers"]
    accepted_final = summary["references"]["accepted_refs"]["accepted_final_model_source"]
    control_chain = summary["references"]["full_oldplan_downstream_control"]

    lines.append("# cp015 oldplan component ablation")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- baseline: `{summary['policy']['baseline_case']}` (kept as source baseline only)")
    lines.append(f"- control: `{summary['policy']['control_stage6_case']}` (reused full old-plan challenger lane)")
    lines.append(f"- stage6 config: `{summary['policy']['stage6_config']}`")
    lines.append("")
    lines.append("## Key groups")
    lines.append("")
    lines.append("| group | prefixes / exact | key_count | rationale |")
    lines.append("|---|---|---:|---|")
    for group in summary["key_groups"]:
        parts = list(group["prefixes"]) + list(group["exact_keys"])
        lines.append(
            f"| {group['label']} | `{', '.join(parts)}` | {group['key_count']} | {group['rationale']} |"
        )
    lines.append("")
    lines.append("## Stage6 screening")
    lines.append("")
    lines.append("| lane | rollback groups | all_ex_root | leg | nonleg | arm | else | delta vs control all_ex_root | delta vs control leg | delta vs cp015 | status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for lane in stage6_lanes:
        d_ctrl = lane["delta_vs_full_oldplan_control"]
        d_src = lane["delta_vs_cp015_source_baseline"]
        lines.append(
            f"| {lane['name']} | {', '.join(lane['rollback_group_names'])} | "
            f"{fmt(lane['stage6_exit']['all_ex_root'])} | {fmt(lane['stage6_exit']['leg'])} | {fmt(lane['stage6_exit']['nonleg'])} | "
            f"{fmt(lane['stage6_exit']['arm'])} | {fmt(lane['stage6_exit']['else'])} | "
            f"{fmt(d_ctrl['all_ex_root'])} | {fmt(d_ctrl['leg'])} | {fmt(d_src['all_ex_root'])} | {lane['selection_status']} |"
        )
    lines.append("")
    lines.append("## Stage6 ranking")
    lines.append("")
    lines.append("| rank | lane | delta all_ex_root | delta leg | delta nonleg | rollback_effective_changed_keys |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for idx, lane in enumerate(sorted(stage6_rollbacks, key=stage6_sort_key), start=1):
        d = lane["delta_vs_full_oldplan_control"]
        lines.append(
            f"| {idx} | {lane['name']} | {fmt(d['all_ex_root'])} | {fmt(d['leg'])} | {fmt(d['nonleg'])} | {lane['rollback_effective_changed_key_count']} |"
        )
    lines.append("")
    sel = summary["stage6_screening"]["selection"]
    lines.append(
        f"- promote_reason: `{sel['promote_reason']}`; promoted: `{', '.join(sel['promoted_names']) if sel['promoted_names'] else 'none'}`"
    )
    lines.append(f"- note: {sel['note']}")
    lines.append("")
    if downstream_lanes:
        lines.append("## Downstream")
        lines.append("")
        lines.append("| lane | DirectGeoLocalDeg(model) | BlendGeoLocalDeg(model) | GeoLocalDeg(model) | all_ex_root(model) | leg(model) | nonleg(model) | delta vs full oldplan all_ex_root | delta vs accepted final all_ex_root |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lane in downstream_lanes:
            model_eval = lane["final_evals"]["model_source"]
            cmp_ctrl = lane["comparisons"]["model_source_vs_full_oldplan_control"]
            cmp_final = lane["comparisons"]["model_source_vs_accepted_final_model_source"]
            lines.append(
                f"| {lane['name']} | {fmt(model_eval['masked_means']['DirectGeoLocalDeg'])} | "
                f"{fmt(model_eval['masked_means']['BlendGeoLocalDeg'])} | {fmt(model_eval['masked_means']['GeoLocalDeg'])} | "
                f"{fmt(model_eval['direct_group_summary']['all_ex_root'])} | {fmt(model_eval['direct_group_summary']['leg'])} | "
                f"{fmt(model_eval['direct_group_summary']['nonleg'])} | {fmt(cmp_ctrl['direct_group_delta']['all_ex_root'])} | "
                f"{fmt(cmp_final['direct_group_delta']['all_ex_root'])} |"
            )
        lines.append("")
        lines.append("| lane | strict DirectGeoLocalDeg | strict all_ex_root | strict leg | strict nonleg | strict delta vs full oldplan all_ex_root |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for lane in downstream_lanes:
            strict_eval = lane["final_evals"]["strict_pretrain_contact"]
            cmp_ctrl = lane["comparisons"]["strict_pretrain_contact_vs_full_oldplan_control"]
            lines.append(
                f"| {lane['name']} | {fmt(strict_eval['masked_means']['DirectGeoLocalDeg'])} | "
                f"{fmt(strict_eval['direct_group_summary']['all_ex_root'])} | {fmt(strict_eval['direct_group_summary']['leg'])} | "
                f"{fmt(strict_eval['direct_group_summary']['nonleg'])} | {fmt(cmp_ctrl['direct_group_delta']['all_ex_root'])} |"
            )
        lines.append("")
        lines.append("## Promoted lane paths")
        lines.append("")
        lines.append("| lane | stage6 | 70a | 70b | new70b_replace | 70R | 71 | 72 | lambda |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for lane in downstream_lanes:
            paths = lane["stage_paths"]
            lines.append(
                f"| {lane['name']} | `{paths['transplant_stage6']}` | `{paths['70a']}` | `{paths['70b']}` | "
                f"`{paths['new70b_replace']}` | `{paths['70R']}` | `{paths['71']}` | `{paths['72']}` | `{paths['lambda_final']}` |"
            )
        lines.append("")
        lines.append("## Control refs")
        lines.append("")
        lines.append(
            f"- full oldplan control strict all_ex_root={fmt(control_chain['final_evals']['strict_pretrain_contact']['direct_group_summary']['all_ex_root'])}, "
            f"leg={fmt(control_chain['final_evals']['strict_pretrain_contact']['direct_group_summary']['leg'])}, "
            f"nonleg={fmt(control_chain['final_evals']['strict_pretrain_contact']['direct_group_summary']['nonleg'])}"
        )
        lines.append(
            f"- full oldplan control model all_ex_root={fmt(control_chain['final_evals']['model_source']['direct_group_summary']['all_ex_root'])}, "
            f"leg={fmt(control_chain['final_evals']['model_source']['direct_group_summary']['leg'])}, "
            f"nonleg={fmt(control_chain['final_evals']['model_source']['direct_group_summary']['nonleg'])}"
        )
        lines.append(
            f"- accepted final model-source all_ex_root={fmt(accepted_final['direct_group_summary']['all_ex_root'])}, "
            f"leg={fmt(accepted_final['direct_group_summary']['leg'])}, "
            f"nonleg={fmt(accepted_final['direct_group_summary']['nonleg'])}"
        )
        lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"1. Most critical rollback group: `{answers['q1_most_critical_group']['lane']}` "
        f"(delta all_ex_root={fmt(answers['q1_most_critical_group']['delta_vs_control']['all_ex_root'])}, "
        f"leg={fmt(answers['q1_most_critical_group']['delta_vs_control']['leg'])})."
    )
    lines.append(
        f"2. Has smaller Stage6-safe version: `{str(bool(answers['q2_has_smaller_stage6_safe_version']['value'])).lower()}`."
    )
    lines.append(
        f"3. Smaller version penetrates to lambda final: `{str(bool(answers['q3_smaller_version_penetrates_to_lambda_final']['value'])).lower()}`."
    )
    lines.append(
        f"4. Smaller version beats full oldplan chain: `{str(bool(answers['q4_smaller_version_beats_full_oldplan_chain']['value'])).lower()}` "
        f"({answers['q4_smaller_version_beats_full_oldplan_chain']['note']})"
    )
    lines.append(
        f"5. Smaller version beats current accepted final: `{str(bool(answers['q5_smaller_version_beats_current_accepted_final']['value'])).lower()}`."
    )
    lines.append(
        f"6. Can propose cleaner challenger lane: `{str(bool(answers['q6_can_propose_new_cleaner_challenger']['value'])).lower()}` "
        f"({answers['q6_can_propose_new_cleaner_challenger']['note']})"
    )
    lines.append(
        f"7. Should pause further simplification: `{str(bool(answers['q7_should_pause_further_simplification']['value'])).lower()}`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        BASELINE_COMPARE_JSON,
        PLANTRANSPLANT_SUMMARY_JSON,
        OLDPLAN_CHAIN_SUMMARY_JSON,
        STAGE6_CONFIG,
        TEACHER,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_baseline_rows()
    source_baseline_row = baseline_rows[SOURCE_BASELINE_CASE]
    control_case = load_stage6_case(CONTROL_STAGE6_CASE)
    control_report = load_json(Path(str(control_case["paths"]["surgery_report"])))
    control_chain = load_json(OLDPLAN_CHAIN_SUMMARY_JSON)
    accepted_refs = parse_reference_payload()

    groups = build_group_catalog(control_report)
    group_map = group_index(groups)

    stage6_entries: List[Dict[str, Any]] = [
        build_control_stage6_entry(
            control_case=control_case,
            control_report=control_report,
            source_baseline_row=source_baseline_row,
            group_catalog=group_map,
        )
    ]

    for lane in LANE_SPECS:
        log(f"=== Stage6 lane {lane.name}: build rollback surgery ===")
        ensure_rollback_surgery(
            lane,
            control_case=control_case,
            control_report=control_report,
            group_catalog=group_map,
        )
        log(f"=== Stage6 lane {lane.name}: run/eval ===")
        ensure_stage6_lane(lane)
        stage6_entries.append(
            build_stage6_entry(
                lane,
                source_baseline_row=source_baseline_row,
                control_case=control_case,
                group_catalog=group_map,
            )
        )

    selection = select_promoted_lanes(stage6_entries)
    stage6_entries = apply_selection(stage6_entries, selection)

    downstream_entries: List[Dict[str, Any]] = []
    promote_set = set(selection["promoted_names"])
    for lane_entry in stage6_entries:
        if str(lane_entry["name"]) not in promote_set:
            continue
        stage6_ckpt = Path(str(lane_entry["paths"]["stage6_ckpt"]))
        ensure_downstream_lane(str(lane_entry["name"]), stage6_ckpt=stage6_ckpt)
        downstream_entries.append(
            build_downstream_entry(
                lane_entry,
                control_chain=control_chain,
                accepted_refs=accepted_refs,
            )
        )

    summary = build_summary(
        group_catalog=groups,
        source_baseline_row=source_baseline_row,
        control_case=control_case,
        control_chain=control_chain,
        accepted_refs=accepted_refs,
        stage6_entries=stage6_entries,
        selection=selection,
        downstream_entries=downstream_entries,
    )
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
