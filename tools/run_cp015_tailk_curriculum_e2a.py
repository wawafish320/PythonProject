#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
from tools.run_cp015_tailk_support_scope_isolation_e1 import (  # noqa: E402
    STAGE70A_CONFIG,
    TOP3_70A_CKPT,
    TOP3_70A_EVAL,
    TOP3_70A_LOG,
    TOP3_BASETRAIN_CONFIG,
    TOP3_BASETRAIN_EPOCH014,
    TOP3_STAGE6_TAILFIX_CKPT,
    TOP3_STAGE6_TAILFIX_LOG,
    TOP7_70A_CKPT,
    TOP7_70A_EVAL,
    TOP7_70A_LOG,
    TOP7_BASETRAIN_CONFIG,
    TOP7_BASETRAIN_EPOCH014,
    TOP7_STAGE6_TAILFIX_CONFIG,
    TOP7_STAGE6_TAILFIX_CKPT,
    TOP7_STAGE6_TAILFIX_LOG,
    _normality_readout,
)


RUN_DATE = "20260408"
RUN_NAME = "cp015_tailk_curriculum_e2a"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"

E2A_BASERUN = (
    "exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_"
    "corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408"
)
E2A_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json"
)
E2A_BASETRAIN_EPOCH014 = ROOT / "models" / "cp015_phasecd_tailk_probe_20260331" / E2A_BASERUN / "ckpt_epoch_014.pth"
E2A_STAGE6_TAILFIX_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk357ramp_stage6_tailfix_e2a_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk357ramp_e2a_20260408.pth"
)
E2A_STAGE6_TAILFIX_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk357ramp_stage6_tailfix_e2a_20260408"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk357ramp_e2a_20260408.json"
)
E2A_STAGE6_TAILFIX_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk_curriculum_e2a_20260408"
    / "stage6_tailfix"
    / "stage6_freerun"
    / "Walk_F_freerun_cycles.json"
)
E2A_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth"
)
E2A_70A_LOG = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408"
    / "posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.json"
)
E2A_70A_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk_curriculum_e2a_20260408"
    / "stage70a"
    / "eval_model_source"
    / "Walk_F_freerun_cycles.json"
)


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    provenance: str
    support_schedule: str
    basetrain_config: Path
    basetrain_epoch014_ckpt: Path
    stage6_tailfix_config: Path
    stage6_tailfix_ckpt: Path
    stage6_tailfix_log: Path
    stage6_tailfix_eval: Path | None
    stage70a_config: Path
    stage70a_ckpt: Path
    stage70a_log: Path
    stage70a_eval: Path
    notes: str


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm="E1-top7",
        provenance="reuse_existing",
        support_schedule="7 -> 7 -> 7",
        basetrain_config=TOP7_BASETRAIN_CONFIG,
        basetrain_epoch014_ckpt=TOP7_BASETRAIN_EPOCH014,
        stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
        stage6_tailfix_ckpt=TOP7_STAGE6_TAILFIX_CKPT,
        stage6_tailfix_log=TOP7_STAGE6_TAILFIX_LOG,
        stage6_tailfix_eval=(
            ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_stage6_tailfix_20260401"
            / "lr3e4_e8x60_wd1e4_reinit1"
            / "stage6_freerun"
            / "Walk_F_freerun_cycles.json"
        ),
        stage70a_config=STAGE70A_CONFIG,
        stage70a_ckpt=TOP7_70A_CKPT,
        stage70a_log=TOP7_70A_LOG,
        stage70a_eval=TOP7_70A_EVAL,
        notes="Canonical top7 arm reused from E1.",
    ),
    ArmSpec(
        arm="E1-top3",
        provenance="reuse_existing",
        support_schedule="3 -> 3 -> 3",
        basetrain_config=TOP3_BASETRAIN_CONFIG,
        basetrain_epoch014_ckpt=TOP3_BASETRAIN_EPOCH014,
        stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
        stage6_tailfix_ckpt=TOP3_STAGE6_TAILFIX_CKPT,
        stage6_tailfix_log=TOP3_STAGE6_TAILFIX_LOG,
        stage6_tailfix_eval=None,
        stage70a_config=STAGE70A_CONFIG,
        stage70a_ckpt=TOP3_70A_CKPT,
        stage70a_log=TOP3_70A_LOG,
        stage70a_eval=TOP3_70A_EVAL,
        notes="Matched top3 control reused from E1.",
    ),
    ArmSpec(
        arm="E2A-R",
        provenance="new_curriculum_arm",
        support_schedule="3 -> 5 -> 7",
        basetrain_config=E2A_BASETRAIN_CONFIG,
        basetrain_epoch014_ckpt=E2A_BASETRAIN_EPOCH014,
        stage6_tailfix_config=TOP7_STAGE6_TAILFIX_CONFIG,
        stage6_tailfix_ckpt=E2A_STAGE6_TAILFIX_CKPT,
        stage6_tailfix_log=E2A_STAGE6_TAILFIX_LOG,
        stage6_tailfix_eval=E2A_STAGE6_TAILFIX_EVAL,
        stage70a_config=STAGE70A_CONFIG,
        stage70a_ckpt=E2A_70A_CKPT,
        stage70a_log=E2A_70A_LOG,
        stage70a_eval=E2A_70A_EVAL,
        notes="Matched curriculum arm: top3 warmup, top5 ramp, top7 late target.",
    ),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory_row(arm: ArmSpec) -> Dict[str, Any]:
    return {
        "arm": arm.arm,
        "provenance": arm.provenance,
        "support_schedule": arm.support_schedule,
        "basetrain_config": str(arm.basetrain_config),
        "basetrain_epoch014_ckpt": str(arm.basetrain_epoch014_ckpt),
        "stage6_tailfix_config": str(arm.stage6_tailfix_config),
        "stage6_tailfix_ckpt": str(arm.stage6_tailfix_ckpt),
        "stage6_tailfix_log": str(arm.stage6_tailfix_log),
        "stage6_tailfix_eval": (str(arm.stage6_tailfix_eval) if arm.stage6_tailfix_eval is not None else None),
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
            "stage6_tailfix_eval": (arm.stage6_tailfix_eval.is_file() if arm.stage6_tailfix_eval is not None else None),
            "stage70a_config": arm.stage70a_config.is_file(),
            "stage70a_ckpt": arm.stage70a_ckpt.is_file(),
            "stage70a_log": arm.stage70a_log.is_file(),
            "stage70a_eval": arm.stage70a_eval.is_file(),
        },
        "notes": arm.notes,
    }


def _config_diff_rows(base_cfg: Mapping[str, Any], ramp_cfg: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "field": "rot_local_tail_k",
            "baseline_top7": int(base_cfg["rot_local_tail_k"]),
            "E2A-R": int(ramp_cfg["rot_local_tail_k"]),
            "meaning": "top-level / early effective support",
        },
        {
            "field": "freerun_stage_schedule.phase_b.core.rot_local_tail_k",
            "baseline_top7": int(base_cfg["freerun_stage_schedule"][1]["loss_groups"]["core"]["rot_local_tail_k"]),
            "E2A-R": int(ramp_cfg["freerun_stage_schedule"][1]["loss_groups"]["core"]["rot_local_tail_k"]),
            "meaning": "early warmup",
        },
        {
            "field": "freerun_stage_schedule.phase_c.core.rot_local_tail_k",
            "baseline_top7": int(base_cfg["freerun_stage_schedule"][2]["loss_groups"]["core"]["rot_local_tail_k"]),
            "E2A-R": int(ramp_cfg["freerun_stage_schedule"][2]["loss_groups"]["core"]["rot_local_tail_k"]),
            "meaning": "mid ramp",
        },
        {
            "field": "freerun_stage_schedule.phase_d.core.rot_local_tail_k",
            "baseline_top7": int(base_cfg["freerun_stage_schedule"][3]["loss_groups"]["core"]["rot_local_tail_k"]),
            "E2A-R": int(ramp_cfg["freerun_stage_schedule"][3]["loss_groups"]["core"]["rot_local_tail_k"]),
            "meaning": "late target",
        },
        {
            "field": "rot_local_tail_reduce",
            "baseline_top7": str(base_cfg["rot_local_tail_reduce"]),
            "E2A-R": str(ramp_cfg["rot_local_tail_reduce"]),
            "meaning": "kept fixed",
        },
        {
            "field": "rot_local_tail_uniform_mix",
            "baseline_top7": float(base_cfg["rot_local_tail_uniform_mix"]),
            "E2A-R": float(ramp_cfg["rot_local_tail_uniform_mix"]),
            "meaning": "kept fixed",
        },
        {
            "field": "rot_local_tail_rank_mix",
            "baseline_top7": float(base_cfg["rot_local_tail_rank_mix"]),
            "E2A-R": float(ramp_cfg["rot_local_tail_rank_mix"]),
            "meaning": "kept fixed",
        },
        {
            "field": "save_fit_ckpt_epochs",
            "baseline_top7": str(base_cfg["save_fit_ckpt_epochs"]),
            "E2A-R": str(ramp_cfg["save_fit_ckpt_epochs"]),
            "meaning": "kept fixed",
        },
    ]


def _delta(a: Mapping[str, Any], b: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    return {key: _safe_float(a.get(key)) - _safe_float(b.get(key)) for key in keys}


def _label_rank(label: Any) -> int:
    order = {
        "plan_compensatory": 0,
        "mixed": 1,
        "nonplan_owned": 2,
    }
    return order.get(str(label), -1)


def _normality_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "plan_over_direct_sensitivity": _safe_float(candidate.get("plan_over_direct_sensitivity"))
        - _safe_float(reference.get("plan_over_direct_sensitivity")),
        "plan_zero_delta_geolocal_deg": _safe_float(candidate.get("plan_zero_delta_geolocal_deg"))
        - _safe_float(reference.get("plan_zero_delta_geolocal_deg")),
        "direct_zero_delta_geolocal_deg": _safe_float(candidate.get("direct_zero_delta_geolocal_deg"))
        - _safe_float(reference.get("direct_zero_delta_geolocal_deg")),
        "meas_zero_delta_geolocal_deg": _safe_float(candidate.get("meas_zero_delta_geolocal_deg"))
        - _safe_float(reference.get("meas_zero_delta_geolocal_deg")),
        "aggregate_normality_score": _safe_float(candidate.get("aggregate_normality_score"))
        - _safe_float(reference.get("aggregate_normality_score")),
        "label_candidate": candidate.get("conclusion_label"),
        "label_reference": reference.get("conclusion_label"),
    }


def _normality_improved(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    delta = _normality_delta(candidate, reference)
    return (
        _label_rank(delta["label_candidate"]) > _label_rank(delta["label_reference"])
        or _safe_float(delta["aggregate_normality_score"]) > 0.10
        or (
            _safe_float(delta["plan_over_direct_sensitivity"]) < -0.10
            and _safe_float(delta["plan_zero_delta_geolocal_deg"]) < -0.002
        )
    )


def _transfer_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    return _delta(
        candidate,
        reference,
        (
            "out_direct_gap",
            "dir_base_gap",
            "dir_leg_gap",
            "dir_nonleg_gap",
            "out_direct_closure_ratio",
            "dir_base_closure_ratio",
            "dir_leg_closure_ratio",
            "dir_nonleg_closure_ratio",
            "aggregate_transfer_score",
        ),
    )


def _proxy_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "plan_norm_per_dim": _safe_float(candidate["blocks"]["plan"]["norm_per_dim"])
        - _safe_float(reference["blocks"]["plan"]["norm_per_dim"]),
        "direct_norm_per_dim": _safe_float(candidate["blocks"]["direct"]["norm_per_dim"])
        - _safe_float(reference["blocks"]["direct"]["norm_per_dim"]),
        "meas_norm_per_dim": _safe_float(candidate["blocks"]["meas"]["norm_per_dim"])
        - _safe_float(reference["blocks"]["meas"]["norm_per_dim"]),
        "plan_over_direct": _safe_float(candidate["ratios"]["plan_over_direct"])
        - _safe_float(reference["ratios"]["plan_over_direct"]),
        "plan_over_meas": _safe_float(candidate["ratios"]["plan_over_meas"])
        - _safe_float(reference["ratios"]["plan_over_meas"]),
        "plan_over_direct_plus_meas": _safe_float(candidate["ratios"]["plan_over_direct_plus_meas"])
        - _safe_float(reference["ratios"]["plan_over_direct_plus_meas"]),
    }


def _transfer_better(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, margin: float) -> bool:
    return _safe_float(candidate.get("aggregate_transfer_score")) > _safe_float(reference.get("aggregate_transfer_score")) + margin


def _case_signature(readout: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": readout.get("conclusion_label"),
        "plan_over_direct_sensitivity": _safe_float(readout.get("plan_over_direct_sensitivity")),
        "plan_zero_delta_geolocal_deg": _safe_float(readout.get("plan_zero_delta_geolocal_deg")),
        "direct_zero_delta_geolocal_deg": _safe_float(readout.get("direct_zero_delta_geolocal_deg")),
        "meas_zero_delta_geolocal_deg": _safe_float(readout.get("meas_zero_delta_geolocal_deg")),
    }


def _normality_probe_discriminative(readouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    signatures = {name: _case_signature(row) for name, row in readouts.items()}
    labels = sorted({str(row["label"]) for row in signatures.values()})
    spans: Dict[str, float] = {}
    for key in (
        "plan_over_direct_sensitivity",
        "plan_zero_delta_geolocal_deg",
        "direct_zero_delta_geolocal_deg",
        "meas_zero_delta_geolocal_deg",
    ):
        vals = [_safe_float(row[key]) for row in signatures.values()]
        good = [v for v in vals if math.isfinite(v)]
        spans[key] = float(max(good) - min(good)) if good else float("nan")
    thresholds = {
        "plan_over_direct_sensitivity": 0.005,
        "plan_zero_delta_geolocal_deg": 0.01,
        "direct_zero_delta_geolocal_deg": 0.05,
        "meas_zero_delta_geolocal_deg": 0.01,
    }
    discriminative = len(labels) > 1 or any(
        math.isfinite(_safe_float(spans[key])) and _safe_float(spans[key]) > float(thresholds[key]) for key in thresholds
    )
    return {
        "discriminative": bool(discriminative),
        "normality_probe_non_discriminative": not bool(discriminative),
        "signatures": signatures,
        "labels": labels,
        "spans": spans,
        "thresholds": thresholds,
    }


def _arm_stage_result(
    *,
    name: str,
    ckpt_path: Path,
    eval_json_path: Path,
    config_path: Path,
    teacher_path: Path,
    host_bundle: Mapping[str, Any],
    prep_host: Mapping[str, Any],
    fixed_contacts: torch.Tensor,
    target_result: Mapping[str, Any],
    host_gaps: Mapping[str, Any],
    host_native_normality: Mapping[str, Any] | None,
    target_normality: Mapping[str, Any] | None,
    include_normality: bool,
) -> Dict[str, Any]:
    donor_bundle = {
        "case": _case_bundle(
            case_name=name,
            ckpt_path=ckpt_path,
            eval_json_path=eval_json_path,
            teacher_path=teacher_path,
            config_path=config_path,
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
    out = {
        "ckpt": str(ckpt_path),
        "eval_json": str(eval_json_path),
        "transfer": transfer,
    }
    if include_normality:
        out["replace_normality"] = _normality_readout(
            host_bundle=host_bundle,
            prep_host=prep_host,
            fixed_contacts=fixed_contacts,
            donor_bundle=donor_bundle,
            host_bad_reference=host_native_normality,
            target_reference=target_normality,
        )
        out["proxy_telemetry"] = _direct_head_proxy(donor_bundle["case"]["trainer"].model)
    return out


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

    top7_cfg = _load_json(TOP7_BASETRAIN_CONFIG)
    e2a_cfg = _load_json(E2A_BASETRAIN_CONFIG)
    config_diff = _config_diff_rows(top7_cfg, e2a_cfg)

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

    e2a_stage6_result = _arm_stage_result(
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
    )

    top7 = final_70a_results["E1-top7"]
    top3 = final_70a_results["E1-top3"]
    e2a = final_70a_results["E2A-R"]

    normality_probe = _normality_probe_discriminative(
        {
            "host_native_bad_reference": host_native_normality,
            "baseline_transplant_target": target_normality,
            "E1-top7": top7["replace_normality"],
            "E1-top3": top3["replace_normality"],
            "E2A-R": e2a["replace_normality"],
        }
    )

    delta_vs_top7 = {
        "transfer": _transfer_delta(e2a["transfer"], top7["transfer"]),
        "replace_normality": _normality_delta(e2a["replace_normality"], top7["replace_normality"]),
        "proxy_telemetry": _proxy_delta(e2a["proxy_telemetry"], top7["proxy_telemetry"]),
    }
    delta_vs_top3 = {
        "transfer": _transfer_delta(e2a["transfer"], top3["transfer"]),
        "replace_normality": _normality_delta(e2a["replace_normality"], top3["replace_normality"]),
        "proxy_telemetry": _proxy_delta(e2a["proxy_telemetry"], top3["proxy_telemetry"]),
    }

    better_than_top7 = _transfer_better(e2a["transfer"], top7["transfer"], margin=0.08)
    better_than_top3 = _transfer_better(e2a["transfer"], top3["transfer"], margin=0.08)
    normality_improved_vs_top7 = (
        False if normality_probe["normality_probe_non_discriminative"] else _normality_improved(e2a["replace_normality"], top7["replace_normality"])
    )
    normality_improved_vs_top3 = (
        False if normality_probe["normality_probe_non_discriminative"] else _normality_improved(e2a["replace_normality"], top3["replace_normality"])
    )

    case_label = "Case 4"
    top7_viable = False
    next_step = "E2-B/C"
    interpretation = (
        "normality probe is non-discriminative in the current fixed readout, so replace-entry normality remains inconclusive and fixed transferability carries most of the evidential load"
    )
    if not normality_probe["normality_probe_non_discriminative"]:
        if better_than_top7 and better_than_top3 and (normality_improved_vs_top7 or normality_improved_vs_top3):
            case_label = "Case 1"
            top7_viable = True
            next_step = "E2-B/C_confirm_or_exploit"
            interpretation = "E2A-R beats both E1-top7 and E1-top3 while also improving replace-entry normality, so top7 appears viable under a transfer-compatible path"
        elif better_than_top7 and (not better_than_top3 or not (normality_improved_vs_top7 or normality_improved_vs_top3)):
            case_label = "Case 2"
            top7_viable = False
            next_step = "E2-B/C"
            interpretation = "curriculum helps relative to E1-top7 but is still not enough to clearly surpass E1-top3 and/or normalize replace entry"
        else:
            case_label = "Case 3"
            top7_viable = False
            next_step = "E3"
            interpretation = "support ramp alone is insufficient; widening still falls back toward an incompatible basin"
    else:
        if better_than_top7 and better_than_top3:
            case_label = "Case 4 leaning Case 1 on transfer only"
            top7_viable = True
            next_step = "E2-B/C_confirm_or_exploit"
            interpretation = (
                "transferability alone supports a viable path for top7, but the normality probe remains non-discriminative so the replace-entry claim stays conservative"
            )
        elif better_than_top7:
            case_label = "Case 4 leaning Case 2"
            top7_viable = False
            next_step = "E2-B/C"
            interpretation = (
                "curriculum/path shaping helps on fixed transferability, but without a discriminative normality probe or a clear win over E1-top3 this remains partial"
            )
        else:
            case_label = "Case 4 leaning Case 3"
            top7_viable = False
            next_step = "E3"
            interpretation = (
                "transfer does not clearly recover under ramping, and the current normality probe does not rescue the claim"
            )

    summary = {
        "analysis": RUN_NAME,
        "scope": {
            "experiment": "E2-A top3 warmup -> top7 ramp",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "coadapt host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic single-step first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "strict_constraints": [
                "reuse E1 anchors",
                "single new arm only",
                "no E0 archaeology rerun",
                "no new attribution mainline",
                "no planner semantics mainline",
                "no new sweep",
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
            "support scope is a partial lever but insufficient alone",
            "direct_pose_head.0 proxy remains supportive readout rather than root-cause evidence",
        ],
        "degraded_e2a_variant": False,
        "arm_inventory": inventory,
        "E2A_R_key_config_diff": config_diff,
        "basetrain_support_schedule": {
            "top_level_rot_local_tail_k": int(e2a_cfg["rot_local_tail_k"]),
            "phase_b_rot_local_tail_k": int(e2a_cfg["freerun_stage_schedule"][1]["loss_groups"]["core"]["rot_local_tail_k"]),
            "phase_c_rot_local_tail_k": int(e2a_cfg["freerun_stage_schedule"][2]["loss_groups"]["core"]["rot_local_tail_k"]),
            "phase_d_rot_local_tail_k": int(e2a_cfg["freerun_stage_schedule"][3]["loss_groups"]["core"]["rot_local_tail_k"]),
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
        "stage6_tailfix_final": {
            "E2A-R": e2a_stage6_result,
        },
        "final_70a_results": final_70a_results,
        "replace_normality_readout": {
            "host_native_bad_reference": host_native_normality,
            "baseline_transplant_target": target_normality,
            "E1-top7": top7["replace_normality"],
            "E1-top3": top3["replace_normality"],
            "E2A-R": e2a["replace_normality"],
        },
        "proxy_telemetry": {
            "E1-top7": top7["proxy_telemetry"],
            "E1-top3": top3["proxy_telemetry"],
            "E2A-R": e2a["proxy_telemetry"],
        },
        "delta_summary": {
            "E2A-R_minus_E1-top7": delta_vs_top7,
            "E2A-R_minus_E1-top3": delta_vs_top3,
        },
        "normality_probe_assessment": normality_probe,
        "judgement": {
            "case": case_label,
            "top7_viable_under_transfer_compatible_path": bool(top7_viable),
            "next_step_recommendation": next_step,
            "interpretation": interpretation,
        },
        "explicit_answers": {
            "q1_E2A_R_better_than_E1_top7_final70a": bool(better_than_top7),
            "q2_E2A_R_better_than_E1_top3_final70a": bool(better_than_top3),
            "q3_E2A_R_enters_replace_more_normally": (
                "inconclusive_normality_probe_non_discriminative"
                if normality_probe["normality_probe_non_discriminative"]
                else bool(normality_improved_vs_top7 or normality_improved_vs_top3)
            ),
            "q4_top7_viable_under_transfer_compatible_path": bool(top7_viable),
            "q5_next_step": next_step,
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
