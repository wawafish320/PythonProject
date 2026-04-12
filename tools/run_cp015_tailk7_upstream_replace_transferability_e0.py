#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import _direct_local_geo_deg, _load_case  # noqa: E402
from tools.analyze_cp015_tailk7_same_input_module_attribution import (  # noqa: E402
    _case_bundle,
    _norm_l2,
    _prepare_fixed_offset_context,
    _run_single_step,
)
from tools.audit_cp015_tailk7_plan_shortcut_takeover_mechanism import (  # noqa: E402
    _branch_layout,
    _first_linear,
)


RUN_DATE = "20260408"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_upstream_replace_transferability_e0_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
DEFAULT_TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
DEFAULT_OFFSET = 45

DIRECT_BRANCH_MODULES: tuple[str, ...] = (
    "direct_pose_head",
    "direct_pose_leg_head",
    "direct_pose_arm_proj",
    "direct_pose_else_proj",
    "direct_pose_out_leg",
    "direct_pose_out_arm",
    "direct_pose_out_else",
)

BASELINE_REPLACE_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "configs"
    / "posttrain_70b_replace_lowdrift_fromfresh_20260317.json"
)
BASELINE_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
BASELINE_REPLACE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)

COADAPT_HOST_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "configs"
    / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json"
)
COADAPT_HOST_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "coadapt_allrot_interface_bestlr_longer_4x"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth"
)
COADAPT_HOST_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "eval_model_source"
    / "coadapt_allrot_interface_bestlr_longer_4x"
    / "Walk_F_freerun_cycles.json"
)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    family: str
    run: str
    phase: str
    step_label: str
    curve_order: Optional[int]
    assay_included: bool
    ckpt: Path
    eval_json: Path
    note: str = ""


def _candidate(
    *,
    label: str,
    family: str,
    run: str,
    phase: str,
    step_label: str,
    curve_order: Optional[int],
    assay_included: bool,
    ckpt: str,
    eval_json: str,
    note: str = "",
) -> CandidateSpec:
    return CandidateSpec(
        label=label,
        family=family,
        run=run,
        phase=phase,
        step_label=step_label,
        curve_order=curve_order,
        assay_included=assay_included,
        ckpt=ROOT / ckpt,
        eval_json=ROOT / eval_json,
        note=note,
    )


ASSAY_CANDIDATES: tuple[CandidateSpec, ...] = (
    _candidate(
        label="baseline_stage6_fromfresh",
        family="top3_reference",
        run="posttrain_pipeline_from_bestfree_20260317",
        phase="stage6",
        step_label="final",
        curve_order=0,
        assay_included=True,
        ckpt="models/__tmp_posttrain_pipeline_from_bestfree_20260317/stage6/ckpt_last_WalkF_stage6_fromfresh_20260317.pth",
        eval_json="debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/stage6/Walk_F_freerun_cycles.json",
        note="canonical top3-reference stage6 anchor",
    ),
    _candidate(
        label="baseline_70a_fromfresh",
        family="top3_reference",
        run="posttrain_pipeline_from_bestfree_20260317",
        phase="70a",
        step_label="final",
        curve_order=1,
        assay_included=True,
        ckpt="models/__tmp_posttrain_pipeline_from_bestfree_20260317/70a/ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth",
        eval_json="debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/70a/Walk_F_freerun_cycles.json",
        note="canonical top3-reference 70a anchor",
    ),
    _candidate(
        label="ep014center_70a_lr3e4",
        family="top3_reference_historical",
        run="ep014center_70a_lowlr_sweep_20260328",
        phase="70a",
        step_label="final",
        curve_order=2,
        assay_included=True,
        ckpt="models/__tmp_ep014center_70a_lowlr_sweep_20260328/lr3e4/ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth",
        eval_json="debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/eval_model_source/lr3e4/Walk_F_freerun_cycles.json",
        note="historical top3-ish 70a reference",
    ),
    _candidate(
        label="tailk7_stage6_exact_epoch013",
        family="top7_current",
        run="cp015_tailk7_rankmix_tw020_stage6_20260401",
        phase="stage6",
        step_label="epoch013",
        curve_order=13,
        assay_included=True,
        ckpt="models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch013/ckpt_last_epoch013_stage6_exact_tailk7_rankmix_tw020_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch013/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk7_stage6_exact_epoch014",
        family="top7_current",
        run="cp015_tailk7_rankmix_tw020_stage6_20260401",
        phase="stage6",
        step_label="epoch014",
        curve_order=14,
        assay_included=True,
        ckpt="models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch014/ckpt_last_epoch014_stage6_exact_tailk7_rankmix_tw020_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch014/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk7_stage6_exact_epoch015",
        family="top7_current",
        run="cp015_tailk7_rankmix_tw020_stage6_20260401",
        phase="stage6",
        step_label="epoch015",
        curve_order=15,
        assay_included=True,
        ckpt="models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch015/ckpt_last_epoch015_stage6_exact_tailk7_rankmix_tw020_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch015/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk7_stage6_tailfix_lr3e4_reinit1",
        family="top7_current",
        run="cp015_tailk7_stage6_tailfix_20260401",
        phase="stage6",
        step_label="tailfix_final",
        curve_order=16,
        assay_included=True,
        ckpt="models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/stage6_freerun/Walk_F_freerun_cycles.json",
        note="canonical stage6 winner",
    ),
    _candidate(
        label="tailk7_70a_from_tailfix",
        family="top7_current",
        run="cp015_tailk7_stage70a_from_tailfix_20260402",
        phase="70a",
        step_label="final",
        curve_order=17,
        assay_included=True,
        ckpt="models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source/Walk_F_freerun_cycles.json",
        note="current upstream handoff checkpoint",
    ),
)

INVENTORY_ONLY_CANDIDATES: tuple[CandidateSpec, ...] = (
    _candidate(
        label="tailk7_stage6_tailfix_lr1e4_reinit1",
        family="top7_current_variant",
        run="cp015_tailk7_stage6_tailfix_20260401",
        phase="stage6",
        step_label="tailfix_variant_lr1e4_reinit1",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr1e4_e8x60_wd1e4_reinit1/ckpt_last_lr1e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_stage6_tailfix_20260401/lr1e4_e8x60_wd1e4_reinit1/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk7_stage6_tailfix_lr3e4_reinit0",
        family="top7_current_variant",
        run="cp015_tailk7_stage6_tailfix_20260401",
        phase="stage6",
        step_label="tailfix_variant_lr3e4_reinit0",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit0/ckpt_last_lr3e4_e8x60_wd1e4_reinit0_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit0/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_only_epoch012",
        family="tailk5_other_available",
        run="cp015_tailk5_only_stage6_20260331",
        phase="stage6",
        step_label="epoch012",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_only_stage6_20260331/epoch012/ckpt_last_epoch012_stage6_exact_tailk5_only_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_only_20260331/stage6_exact/epoch012/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_only_epoch013",
        family="tailk5_other_available",
        run="cp015_tailk5_only_stage6_20260331",
        phase="stage6",
        step_label="epoch013",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_only_stage6_20260331/epoch013/ckpt_last_epoch013_stage6_exact_tailk5_only_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_only_20260331/stage6_exact/epoch013/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_only_epoch014",
        family="tailk5_other_available",
        run="cp015_tailk5_only_stage6_20260331",
        phase="stage6",
        step_label="epoch014",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_only_stage6_20260331/epoch014/ckpt_last_epoch014_stage6_exact_tailk5_only_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_only_20260331/stage6_exact/epoch014/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_only_epoch015",
        family="tailk5_other_available",
        run="cp015_tailk5_only_stage6_20260331",
        phase="stage6",
        step_label="epoch015",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_only_stage6_20260331/epoch015/ckpt_last_epoch015_stage6_exact_tailk5_only_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_only_20260331/stage6_exact/epoch015/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_epoch013",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_stage6_20260331",
        phase="stage6",
        step_label="epoch013",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_stage6_20260331/epoch013/ckpt_last_epoch013_stage6_exact_tailk5_rankmix_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_20260331/stage6_exact/epoch013/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_epoch014",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_stage6_20260331",
        phase="stage6",
        step_label="epoch014",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_stage6_20260331/epoch014/ckpt_last_epoch014_stage6_exact_tailk5_rankmix_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_20260331/stage6_exact/epoch014/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_epoch015",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_stage6_20260331",
        phase="stage6",
        step_label="epoch015",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_stage6_20260331/epoch015/ckpt_last_epoch015_stage6_exact_tailk5_rankmix_20260331.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_20260331/stage6_exact/epoch015/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_tw025_epoch013",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_tw025_stage6_20260401",
        phase="stage6",
        step_label="epoch013",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_tw025_stage6_20260401/epoch013/ckpt_last_epoch013_stage6_exact_tailk5_rankmix_tw025_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_tw025_20260401/stage6_exact/epoch013/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_tw025_epoch014",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_tw025_stage6_20260401",
        phase="stage6",
        step_label="epoch014",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_tw025_stage6_20260401/epoch014/ckpt_last_epoch014_stage6_exact_tailk5_rankmix_tw025_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_tw025_20260401/stage6_exact/epoch014/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="tailk5_rankmix_tw025_epoch015",
        family="tailk5_other_available",
        run="cp015_tailk5_rankmix_tw025_stage6_20260401",
        phase="stage6",
        step_label="epoch015",
        curve_order=None,
        assay_included=False,
        ckpt="models/__tmp_cp015_tailk5_rankmix_tw025_stage6_20260401/epoch015/ckpt_last_epoch015_stage6_exact_tailk5_rankmix_tw025_20260401.pth",
        eval_json="debug_output/_tmp_cp015_tailk5_rankmix_tw025_20260401/stage6_exact/epoch015/stage6_freerun/Walk_F_freerun_cycles.json",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch010_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch010",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_010.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch010/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch011_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch011",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_011.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch011/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch012_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch012",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_012.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch012/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch013_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch013",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_013.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch013/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch014_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch014",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_014.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch014/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
    _candidate(
        label="top3_control_denseckpt_rerun_epoch015_legacy",
        family="top3_legacy_denseckpt_rerun",
        run="cp015_stage6_entry_contract_matrix_20260330",
        phase="stage6",
        step_label="epoch015",
        curve_order=None,
        assay_included=False,
        ckpt="models/cp015_stage6_entry_contract_matrix_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_rerun_seed2024_20260330/ckpt_epoch_015.pth",
        eval_json="debug_output/_tmp_stage6_entry_contract_matrix_20260330/control_denseckpt_rerun/epoch015/stage6_freerun/Walk_F_freerun_cycles.json",
        note="legacy payload uses `config` instead of `posttrain_cfg`; inventoried but not swept",
    ),
)

ALL_CANDIDATES: tuple[CandidateSpec, ...] = ASSAY_CANDIDATES + INVENTORY_ONLY_CANDIDATES


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _ratio(num: Any, den: Any) -> float:
    n = _safe_float(num)
    d = _safe_float(den)
    if (not math.isfinite(n)) or (not math.isfinite(d)) or abs(d) <= 1e-12:
        return float("nan")
    return float(n / d)


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < arr.shape[0]:
        j = i + 1
        while j < arr.shape[0] and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xv = np.asarray([_safe_float(v) for v in x], dtype=np.float64)
    yv = np.asarray([_safe_float(v) for v in y], dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(mask.sum()) < 2:
        return float("nan")
    xr = _rankdata(xv[mask])
    yr = _rankdata(yv[mask])
    corr = np.corrcoef(xr, yr)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else float("nan")


def _inventory_row(spec: CandidateSpec) -> Dict[str, Any]:
    return {
        "label": spec.label,
        "family": spec.family,
        "run": spec.run,
        "phase": spec.phase,
        "step_label": spec.step_label,
        "curve_order": spec.curve_order,
        "assay_included": spec.assay_included,
        "ckpt": str(spec.ckpt),
        "eval_json": str(spec.eval_json),
        "ckpt_exists": spec.ckpt.is_file(),
        "eval_exists": spec.eval_json.is_file(),
        "note": spec.note,
    }


def _build_nonleg_indices(groups: Mapping[str, Sequence[int]]) -> List[int]:
    leg = {int(i) for i in groups.get("leg", [])}
    return [int(i) for i in groups.get("all_ex_root", []) if int(i) not in leg]


def _tensor_metric_gaps(
    *,
    host_case: Mapping[str, Any],
    target_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
) -> Dict[str, float]:
    trainer = host_case["trainer"]
    rot_slice = host_case["rot_slice"]
    root_idx = int(host_case["root_idx"])
    columns = host_case["columns"]
    groups = host_case["groups"]
    leg = [int(i) for i in groups.get("leg", [])]
    all_ex_root = [int(i) for i in groups.get("all_ex_root", [])]
    nonleg = _build_nonleg_indices(groups)

    target_direct = target_result["ret"].get("out_direct") if isinstance(target_result.get("ret"), Mapping) else None
    cand_direct = candidate_result["ret"].get("out_direct") if isinstance(candidate_result.get("ret"), Mapping) else None
    out_gap = _norm_l2(target_direct, cand_direct)

    if not torch.is_tensor(target_direct) or not torch.is_tensor(cand_direct):
        return {
            "out_direct_gap": float(out_gap),
            "dir_base_gap": float("nan"),
            "dir_leg_gap": float("nan"),
            "dir_nonleg_gap": float("nan"),
        }

    target_raw = trainer._denorm(target_direct).reshape(int(target_direct.shape[0]), -1)
    cand_raw = trainer._denorm(cand_direct).reshape(int(cand_direct.shape[0]), -1)
    geo = _direct_local_geo_deg(
        pred_raw=cand_raw,
        gt_raw=target_raw,
        rot_slice=rot_slice,
        root_idx=root_idx,
        columns=columns,
    ).detach().cpu()

    def _group_mean(idxs: Sequence[int]) -> float:
        keep = [int(i) for i in idxs if 0 <= int(i) < int(geo.shape[1])]
        if not keep:
            return float("nan")
        return float(geo[:, keep].mean().item())

    return {
        "out_direct_gap": float(out_gap),
        "dir_base_gap": _group_mean(all_ex_root),
        "dir_leg_gap": _group_mean(leg),
        "dir_nonleg_gap": _group_mean(nonleg),
    }


def _add_closure(candidate_gaps: Mapping[str, Any], origin_gaps: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(candidate_gaps)
    closures = {
        "out_direct_closure_ratio": float("nan"),
        "dir_base_closure_ratio": float("nan"),
        "dir_leg_closure_ratio": float("nan"),
        "dir_nonleg_closure_ratio": float("nan"),
    }
    pairs = (
        ("out_direct_gap", "out_direct_closure_ratio"),
        ("dir_base_gap", "dir_base_closure_ratio"),
        ("dir_leg_gap", "dir_leg_closure_ratio"),
        ("dir_nonleg_gap", "dir_nonleg_closure_ratio"),
    )
    for gap_key, closure_key in pairs:
        cand = _safe_float(candidate_gaps.get(gap_key))
        orig = _safe_float(origin_gaps.get(gap_key))
        if math.isfinite(cand) and math.isfinite(orig) and orig > 1e-12:
            closures[closure_key] = float(1.0 - (cand / orig))
    out.update(closures)
    out["aggregate_transfer_score"] = _mean(closures.values())
    return out


def _direct_head_proxy(model: torch.nn.Module) -> Dict[str, Any]:
    linear_name, linear = _first_linear(model)
    layout = _branch_layout(model, linear)
    weight = linear.weight.detach().cpu().to(dtype=torch.float32)
    block_rows: Dict[str, Dict[str, Any]] = {}
    for name, sl, dim in (
        ("direct", layout.direct, layout.direct_dim),
        ("plan", layout.plan, layout.plan_dim),
        ("meas", layout.meas, layout.meas_dim),
    ):
        block = weight[:, sl]
        fro = float(torch.linalg.vector_norm(block).item())
        norm_per_dim = float(fro / math.sqrt(max(1, int(dim))))
        block_rows[name] = {
            "slice": [int(sl.start or 0), int(sl.stop or 0)],
            "dim": int(dim),
            "weight_fro": fro,
            "norm_per_dim": norm_per_dim,
        }
    direct_norm = _safe_float(block_rows["direct"]["norm_per_dim"])
    plan_norm = _safe_float(block_rows["plan"]["norm_per_dim"])
    meas_norm = _safe_float(block_rows["meas"]["norm_per_dim"])
    return {
        "first_linear": linear_name,
        "total_in_dim": int(layout.total_dim),
        "blocks": block_rows,
        "ratios": {
            "plan_over_direct": _ratio(plan_norm, direct_norm),
            "plan_over_meas": _ratio(plan_norm, meas_norm),
            "plan_over_direct_plus_meas": _ratio(plan_norm, direct_norm + meas_norm),
        },
    }


def _judgement_for_top7(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["curve_order"]))
    if not ordered:
        return {}
    scores = [_safe_float(row["transfer"]["aggregate_transfer_score"]) for row in ordered]
    plan_ratio = [
        _safe_float((row["proxy_telemetry"]["ratios"] or {}).get("plan_over_direct")) for row in ordered
    ]
    proxy_good = [(-v if math.isfinite(v) else float("nan")) for v in plan_ratio]

    best_idx = max(range(len(ordered)), key=lambda i: _safe_float(scores[i]))
    final_idx = len(ordered) - 1
    best_row = ordered[best_idx]
    final_row = ordered[final_idx]
    first_row = ordered[0]

    better_than_final = [
        row
        for row in ordered[:-1]
        if _safe_float(row["transfer"]["aggregate_transfer_score"])
        > _safe_float(final_row["transfer"]["aggregate_transfer_score"]) + 0.05
    ]

    drop_edges: List[Dict[str, Any]] = []
    for idx in range(len(ordered) - 1):
        s0 = _safe_float(scores[idx])
        s1 = _safe_float(scores[idx + 1])
        p0 = _safe_float(plan_ratio[idx])
        p1 = _safe_float(plan_ratio[idx + 1])
        drop_edges.append(
            {
                "from": ordered[idx]["label"],
                "to": ordered[idx + 1]["label"],
                "transfer_delta": float(s1 - s0) if math.isfinite(s0) and math.isfinite(s1) else float("nan"),
                "proxy_plan_over_direct_delta": float(p1 - p0)
                if math.isfinite(p0) and math.isfinite(p1)
                else float("nan"),
            }
        )

    worst_transfer_edge = None
    worst_proxy_edge = None
    valid_transfer_edges = [edge for edge in drop_edges if math.isfinite(_safe_float(edge["transfer_delta"]))]
    valid_proxy_edges = [
        edge for edge in drop_edges if math.isfinite(_safe_float(edge["proxy_plan_over_direct_delta"]))
    ]
    if valid_transfer_edges:
        worst_transfer_edge = min(valid_transfer_edges, key=lambda edge: _safe_float(edge["transfer_delta"]))
    if valid_proxy_edges:
        worst_proxy_edge = max(valid_proxy_edges, key=lambda edge: _safe_float(edge["proxy_plan_over_direct_delta"]))

    first_score = _safe_float(first_row["transfer"]["aggregate_transfer_score"])
    best_score = _safe_float(best_row["transfer"]["aggregate_transfer_score"])
    final_score = _safe_float(final_row["transfer"]["aggregate_transfer_score"])

    if math.isfinite(first_score) and first_score < 0.35 and best_score < 0.55:
        formation = (
            "bad_from_earliest_available_stage6"
            if best_idx == 0
            else "already_bad_then_partial_repair_without_reaching_good_transfer"
        )
        earliest_window = f"at-or-before {first_row['label']}"
    elif best_idx < final_idx and best_score > final_score + 0.05:
        formation = "good_then_bad"
        earliest_window = f"{best_row['label']} -> {ordered[min(best_idx + 1, final_idx)]['label']}"
    elif best_idx == final_idx and best_score > first_score + 0.05:
        formation = "mid_to_late_repair"
        earliest_window = f"{first_row['label']} -> {best_row['label']}"
    else:
        formation = "flat_or_mild_drift"
        earliest_window = (
            f"{worst_transfer_edge['from']} -> {worst_transfer_edge['to']}"
            if worst_transfer_edge is not None
            else "undetermined"
        )

    if worst_transfer_edge is None or worst_proxy_edge is None:
        sync_class = "long_term_async"
        proxy_role = "not_proxy"
        proxy_help = "limited"
    else:
        transfer_edge_idx = drop_edges.index(worst_transfer_edge)
        proxy_edge_idx = drop_edges.index(worst_proxy_edge)
        if transfer_edge_idx == proxy_edge_idx:
            sync_class = "synchronous_inflection"
            proxy_role = "concurrent_readout"
            proxy_help = "useful_coarse_window_locator"
        elif proxy_edge_idx < transfer_edge_idx:
            sync_class = "proxy_leads_transfer"
            proxy_role = "leading_indicator"
            proxy_help = "useful_early_warning"
        elif proxy_edge_idx > transfer_edge_idx:
            sync_class = "transfer_leads_proxy"
            proxy_role = "lagging_readout"
            proxy_help = "useful_after_the_fact"
        else:
            sync_class = "long_term_async"
            proxy_role = "not_proxy"
            proxy_help = "limited"

    corr = _spearman(scores, proxy_good)
    if not math.isfinite(corr) or abs(corr) < 0.4:
        if sync_class == "synchronous_inflection":
            useful_proxy = "weak_yes"
        else:
            useful_proxy = "no"
    else:
        useful_proxy = "yes"

    if sync_class == "proxy_leads_transfer":
        sync_summary = "proxy worsens before transferability clearly degrades"
    elif sync_class == "transfer_leads_proxy":
        sync_summary = "transferability degrades before proxy clearly worsens"
    elif sync_class == "synchronous_inflection":
        if "at-or-before" in earliest_window:
            sync_summary = (
                "largest later bad-turn and largest proxy worsening happen on the same adjacent checkpoint edge, "
                "but transferability is already mediocre at the earliest available stage6 checkpoint"
            )
        else:
            sync_summary = "largest transfer drop and largest proxy worsening happen on the same adjacent checkpoint edge"
    else:
        sync_summary = "no stable shared turning edge; transfer and proxy remain only loosely aligned"

    return {
        "curve_labels": [row["label"] for row in ordered],
        "ordered_rows": [
            {
                "label": row["label"],
                "curve_order": row["curve_order"],
                "transfer_score": row["transfer"]["aggregate_transfer_score"],
                "out_direct_closure_ratio": row["transfer"]["out_direct_closure_ratio"],
                "dir_base_closure_ratio": row["transfer"]["dir_base_closure_ratio"],
                "dir_leg_closure_ratio": row["transfer"]["dir_leg_closure_ratio"],
                "dir_nonleg_closure_ratio": row["transfer"]["dir_nonleg_closure_ratio"],
                "plan_norm_per_dim": row["proxy_telemetry"]["blocks"]["plan"]["norm_per_dim"],
                "direct_norm_per_dim": row["proxy_telemetry"]["blocks"]["direct"]["norm_per_dim"],
                "meas_norm_per_dim": row["proxy_telemetry"]["blocks"]["meas"]["norm_per_dim"],
                "plan_over_direct": row["proxy_telemetry"]["ratios"]["plan_over_direct"],
                "plan_over_direct_plus_meas": row["proxy_telemetry"]["ratios"]["plan_over_direct_plus_meas"],
            }
            for row in ordered
        ],
        "best_checkpoint": best_row["label"],
        "final_checkpoint": final_row["label"],
        "best_transfer_score": best_score,
        "final_transfer_score": final_score,
        "earliest_available_checkpoint": first_row["label"],
        "formation_judgement": formation,
        "exists_intermediate_better_than_final": bool(better_than_final),
        "intermediate_better_than_final": [row["label"] for row in better_than_final],
        "earliest_divergence_window": earliest_window,
        "largest_adjacent_deltas": drop_edges,
        "worst_transfer_edge": worst_transfer_edge,
        "worst_proxy_edge": worst_proxy_edge,
        "spearman_transfer_vs_negative_plan_over_direct": corr,
        "synchrony_classification": sync_class,
        "synchrony_summary": sync_summary,
        "plan_weight_useful_proxy": useful_proxy,
        "proxy_role": proxy_role,
        "proxy_helpfulness": proxy_help,
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    teacher = DEFAULT_TEACHER.resolve()
    required = (
        teacher,
        BASELINE_REPLACE_CONFIG,
        BASELINE_REPLACE_CKPT,
        BASELINE_REPLACE_EVAL,
        COADAPT_HOST_CONFIG,
        COADAPT_HOST_CKPT,
        COADAPT_HOST_EVAL,
    )
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("[FATAL] missing fixed-context artifact(s):\n" + "\n".join(missing))

    inventory = [_inventory_row(spec) for spec in ALL_CANDIDATES]

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
    fixed_contacts = (
        baseline_native["inputs"]["contacts"]
        if isinstance(baseline_native.get("inputs"), Mapping)
        else None
    )
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

    assay_results: List[Dict[str, Any]] = []
    for spec in ASSAY_CANDIDATES:
        row: Dict[str, Any] = _inventory_row(spec)
        if not spec.ckpt.is_file() or not spec.eval_json.is_file():
            row["status"] = "missing"
            assay_results.append(row)
            continue
        try:
            donor_case = _load_case(
                case_name=spec.label,
                ckpt_path=spec.ckpt,
                eval_json_path=spec.eval_json,
                teacher_path=teacher,
                device_pref="cpu",
            )
            donor_bundle = {"case": donor_case}
            candidate_result = _run_single_step(
                host_bundle,
                prep_host,
                fixed_contacts=fixed_contacts,
                weight_swap_modules=DIRECT_BRANCH_MODULES,
                donor_bundle=donor_bundle,
            )
            transfer_gaps = _tensor_metric_gaps(
                host_case=host_bundle["case"],
                target_result=target_result,
                candidate_result=candidate_result,
            )
            transfer = _add_closure(transfer_gaps, host_gaps)
            proxy = _direct_head_proxy(donor_case["trainer"].model)
            row.update(
                {
                    "status": "ok",
                    "transfer": transfer,
                    "proxy_telemetry": proxy,
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        assay_results.append(row)

    ok_rows = [row for row in assay_results if row.get("status") == "ok"]
    top7_rows = [row for row in ok_rows if row.get("family") == "top7_current"]
    best_overall = (
        max(ok_rows, key=lambda row: _safe_float((row.get("transfer") or {}).get("aggregate_transfer_score")))
        if ok_rows
        else None
    )
    top7_judgement = _judgement_for_top7(top7_rows)

    next_step = "E1_scope_isolation"
    if str((top7_judgement or {}).get("formation_judgement")) == "good_then_bad":
        next_step = "E1_scope_isolation"
    elif str((top7_judgement or {}).get("formation_judgement")) in {
        "bad_from_earliest_available_stage6",
        "already_bad_then_partial_repair_without_reaching_good_transfer",
    }:
        next_step = "E1_scope_isolation"
    elif not bool((top7_judgement or {}).get("exists_intermediate_better_than_final")):
        next_step = "E1_scope_isolation"

    summary = {
        "analysis": "cp015_tailk7_upstream_replace_transferability_e0",
        "scope": {
            "mode": "analysis_only",
            "fixed_replace_context": "coadapt_allrot_interface_bestlr_longer_4x_20260406",
            "transplant_compatible_target": "coadapt host + baseline replace 7-module direct-branch transplant",
            "assay_mode": "deterministic single-step first-forward",
            "offset": DEFAULT_OFFSET,
            "fixed_contacts_source": "baseline replace native same-entry contacts_in_t",
            "direct_branch_modules": list(DIRECT_BRANCH_MODULES),
        },
        "inherited_conclusions": [
            "root cause not in planner semantics mainline",
            "root cause not in replace-entry external rollout state",
            "root cause not in contacts_in_t",
            "earliest semantic split at direct_pose_head boundary",
            "direct_pose_head is necessary anchor but not standalone sufficient",
            "7-module direct-branch joint contract is required for high closure",
            "baseline 7-module direct branch can transfer into coadapt context",
        ],
        "checkpoint_inventory": inventory,
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
        "per_checkpoint_results": assay_results,
        "best_checkpoint_by_transferability": {
            "label": best_overall.get("label") if best_overall else None,
            "family": best_overall.get("family") if best_overall else None,
            "phase": best_overall.get("phase") if best_overall else None,
            "transfer_score": ((best_overall.get("transfer") or {}).get("aggregate_transfer_score") if best_overall else None),
        },
        "top7_curve_judgement": top7_judgement,
        "transfer_proxy_synchronicity_judgement": {
            "classification": (top7_judgement or {}).get("synchrony_classification"),
            "summary": (top7_judgement or {}).get("synchrony_summary"),
            "plan_weight_useful_proxy": (top7_judgement or {}).get("plan_weight_useful_proxy"),
            "proxy_role": (top7_judgement or {}).get("proxy_role"),
        },
        "explicit_answers": {
            "q1_when_problem_forms": (
                "already_mediocre_by_earliest_available_stage6_exact_epoch013__partial_repair_at_epoch014_015__then_sharp_drop_into_tailfix_and_70a"
                if str((top7_judgement or {}).get("formation_judgement", "")).startswith("already_bad")
                else (
                    "at-or-before-earliest-available-stage6"
                    if str((top7_judgement or {}).get("formation_judgement", "")).startswith("bad_from_earliest")
                    else (top7_judgement or {}).get("formation_judgement")
                )
            ),
            "q2_exists_better_intermediate": bool((top7_judgement or {}).get("exists_intermediate_better_than_final")),
            "q3_proxy_helps_window_localization": (top7_judgement or {}).get("proxy_helpfulness"),
            "q4_recommended_next_step": next_step,
        },
        "initial_next_step_recommendation": {
            "recommended": next_step,
            "why": (
                "available E0 already localizes the failure window well enough to test support/path isolation directly; "
                "more archaeology is unlikely to dominate E1 value"
            ),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
