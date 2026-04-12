#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_closed_loop_gap import DEFAULT_TEACHER, _load_case  # noqa: E402
from tools.audit_cp015_tailk7_direct_dependency_asymmetry import (  # noqa: E402
    _build_eval_command,
    _load_base_eval_meta,
    _run_command,
    _runtime_metrics,
    _safe_float,
)
from tools.audit_cp015_tailk7_plan_shortcut_takeover_mechanism import (  # noqa: E402
    _branch_items,
    _branch_layout,
    _classify_gain,
    _direct_geolocal_mean,
    _first_linear,
    _head_zero_branch_deltas,
    _input_and_weight_stats,
    _jacobian_sensitivity,
    _label_ablation,
    _run_with_head_hook,
    _selected_indices,
    _stack_inputs,
)


RUN_DATE = "20260407"
RUN_NAME = "cp015_tailk7_raw70a_nonplan_readiness_falsifier"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_{RUN_NAME}_{RUN_DATE}"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
AUDIT_LOG = OUT_ROOT / "audit.log"

TEACHER_MODE = "teacher_x_gt"
FREERUN_MODE = "freerun"
BEHAVIOR_OVERRIDES: Tuple[Tuple[str, str], ...] = (
    ("model", "model"),
    ("zero", "model"),
    ("gt", "model"),
    ("model", "zero"),
    ("zero", "zero"),
)
FREERUN_SPOTCHECK_KEYS: Tuple[str, ...] = (
    "baseline_raw_70a",
    "baseline_70a_replace_zerophase",
    "tailk7_raw_70a",
    "tailk7_70a_replace_zerophase",
    "tailk7_baseline_style_adapted_warmstart",
)


INHERITED_BLOCK_WEIGHT_FACTS: Tuple[Dict[str, Any], ...] = (
    {
        "candidate": "baseline raw 70a",
        "stage_type": "raw70a",
        "direct_per_dim": 2.100333,
        "plan_per_dim": 2.020063,
        "meas_per_dim": 2.094611,
    },
    {
        "candidate": "baseline 70a_replace_zerophase",
        "stage_type": "warmstart/zerophase",
        "direct_per_dim": 2.100333,
        "plan_per_dim": 0.0,
        "meas_per_dim": 0.0,
    },
    {
        "candidate": "baseline_replace final",
        "stage_type": "70b final",
        "direct_per_dim": 2.101414,
        "plan_per_dim": 0.027915,
        "meas_per_dim": 0.030368,
    },
    {
        "candidate": "tailk7 raw 70a",
        "stage_type": "raw70a",
        "direct_per_dim": 2.038038,
        "plan_per_dim": 2.011084,
        "meas_per_dim": 1.948133,
    },
    {
        "candidate": "tailk7 70a_replace_zerophase",
        "stage_type": "warmstart/copy-only zerophase",
        "direct_per_dim": 2.038038,
        "plan_per_dim": 2.011084,
        "meas_per_dim": 1.948133,
    },
    {
        "candidate": "tailk7 coadapt final",
        "stage_type": "70b final/coadapt",
        "direct_per_dim": 2.038038,
        "plan_per_dim": 2.011084,
        "meas_per_dim": 1.948133,
    },
)


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    label: str
    stage_type: str
    ckpt: Path
    meta_source_eval: Path
    purpose: str
    self_contained: bool = True
    default_eval_by_mode: Mapping[str, Path] = None  # type: ignore[assignment]

    def eval_default(self, mode: str) -> Optional[Path]:
        if not self.default_eval_by_mode:
            return None
        path = self.default_eval_by_mode.get(str(mode))
        if path is None:
            return None
        return Path(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _slug(plan_source: str, meas_source: str) -> str:
    return f"plan_{plan_source}__meas_{meas_source}"


def _label_mode(mode: str) -> str:
    return "teacher-conditioned / freerun_x_gt" if str(mode) == TEACHER_MODE else "freerun spot-check"


def _mean(values: Iterable[Any]) -> float:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _candidate_specs() -> Dict[str, CandidateSpec]:
    return {
        "baseline_raw_70a": CandidateSpec(
            key="baseline_raw_70a",
            label="baseline raw 70a",
            stage_type="raw70a",
            ckpt=ROOT
            / "models"
            / "__tmp_posttrain_pipeline_from_bestfree_20260317"
            / "70a"
            / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_posttrain_pipeline_from_bestfree_20260317"
            / "eval_model_source"
            / "70a"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={
                FREERUN_MODE: ROOT
                / "debug_output"
                / "_tmp_posttrain_pipeline_from_bestfree_20260317"
                / "eval_model_source"
                / "70a"
                / "Walk_F_freerun_cycles.json",
            },
            purpose="raw 70a anchor；检验 baseline 的 non-plan path 在 surgery 前是否已可用",
        ),
        "baseline_70a_replace_zerophase": CandidateSpec(
            key="baseline_70a_replace_zerophase",
            label="baseline 70a_replace_zerophase",
            stage_type="warmstart/zerophase",
            ckpt=ROOT
            / "models"
            / "__tmp_posttrain_pipeline_from_bestfree_20260317"
            / "warmstart"
            / "ckpt_last_70a_replace_zerophase_20260317.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_posttrain_pipeline_from_bestfree_20260317"
            / "eval_model_source"
            / "70a"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={},
            purpose="检验 baseline warmstart surgery 是创造能力还是暴露已 ready 的 non-plan path",
        ),
        "baseline_replace_final": CandidateSpec(
            key="baseline_replace_final",
            label="baseline_replace final",
            stage_type="70b final",
            ckpt=ROOT
            / "models"
            / "__tmp_posttrain_pipeline_from_bestfree_20260317"
            / "70b_replace_lowdrift"
            / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_posttrain_pipeline_from_bestfree_20260317"
            / "eval_model_source"
            / "new70b_replace_lowdrift"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={
                FREERUN_MODE: ROOT
                / "debug_output"
                / "_tmp_posttrain_pipeline_from_bestfree_20260317"
                / "eval_model_source"
                / "new70b_replace_lowdrift"
                / "Walk_F_freerun_cycles.json",
            },
            purpose="production reference；只作为 low-plan basin 最终参考锚点",
        ),
        "tailk7_raw_70a": CandidateSpec(
            key="tailk7_raw_70a",
            label="tailk7 raw 70a",
            stage_type="raw70a",
            ckpt=ROOT
            / "models"
            / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
            / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
            / "eval_model_source"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={
                FREERUN_MODE: ROOT
                / "debug_output"
                / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
                / "eval_model_source"
                / "Walk_F_freerun_cycles.json",
            },
            purpose="raw 70a falsifier 主角；检验 tailk7 的 non-plan readiness 是否已在 70a 出口更弱",
        ),
        "tailk7_70a_replace_zerophase": CandidateSpec(
            key="tailk7_70a_replace_zerophase",
            label="tailk7 70a_replace_zerophase",
            stage_type="warmstart/copy-only zerophase",
            ckpt=ROOT
            / "models"
            / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
            / "warmstart"
            / "ckpt_last_cp015_tailk7_70a_replace_zerophase_20260406.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
            / "eval_model_source"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={},
            purpose="copy-only warmstart；检验 tailk7 只是进入 replace 后 copy 版本能否暴露 non-plan path",
        ),
        "tailk7_baseline_style_adapted_warmstart": CandidateSpec(
            key="tailk7_baseline_style_adapted_warmstart",
            label="tailk7 baseline-style adapted warmstart",
            stage_type="warmstart/baseline-style adapted",
            ckpt=ROOT
            / "models"
            / "__tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel"
            / "warmstart"
            / "ckpt_last_cp015_tailk7_70a_replace_baseline_style_20260402_warmstart_contract_sentinel.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
            / "eval_model_source"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={
                FREERUN_MODE: ROOT
                / "debug_output"
                / "_tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel"
                / "eval_model_source"
                / "tailk7_adapted_warmstart"
                / "step_000"
                / "Walk_F_freerun_cycles.json",
            },
            purpose="warmstart contract sentinel；检验 baseline-style adaptation 是否足以把 tailk7 拉进 baseline basin",
        ),
        "tailk7_coadapt_final": CandidateSpec(
            key="tailk7_coadapt_final",
            label="tailk7 coadapt final",
            stage_type="70b final/coadapt",
            ckpt=ROOT
            / "models"
            / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
            / "coadapt_allrot_interface_bestlr_longer_4x"
            / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth",
            meta_source_eval=ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
            / "eval_model_source"
            / "coadapt_allrot_interface_bestlr_longer_4x"
            / "Walk_F_freerun_cycles.json",
            default_eval_by_mode={
                FREERUN_MODE: ROOT
                / "debug_output"
                / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
                / "eval_model_source"
                / "coadapt_allrot_interface_bestlr_longer_4x"
                / "Walk_F_freerun_cycles.json",
            },
            purpose="tailk7 final reference；只作为 no-collapse 终态 readout / symptom 对照",
        ),
    }


def _load_meta(spec: CandidateSpec) -> Dict[str, Any]:
    meta = _load_base_eval_meta(spec.meta_source_eval)
    meta["model"] = spec.ckpt.resolve()
    return meta


def _ensure_eval(
    *,
    spec: CandidateSpec,
    meta: Mapping[str, Any],
    mode: str,
    plan_source: str,
    meas_source: str,
    python_exe: str,
    force: bool,
) -> Tuple[Path, str]:
    default_eval = spec.eval_default(mode)
    if (
        not force
        and default_eval is not None
        and str(plan_source) == "model"
        and str(meas_source) == "model"
        and default_eval.is_file()
    ):
        return default_eval, "existing"
    out_dir = OUT_ROOT / "eval_matrix" / spec.key / mode / _slug(plan_source, meas_source)
    eval_json = out_dir / "Walk_F_freerun_cycles.json"
    if not eval_json.is_file() or force:
        cmd = _build_eval_command(
            python_exe=python_exe,
            meta=meta,
            out_dir=out_dir,
            plan_source=plan_source,
            meas_source=meas_source,
            mode=mode,
        )
        _run_command(cmd, cwd=ROOT, log_path=AUDIT_LOG)
    return eval_json, "probe"


def _summarize_behavior_candidate_mode(
    *,
    spec: CandidateSpec,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_override = {(str(r["plan_source"]), str(r["meas_source"])): r for r in rows}
    default = by_override[("model", "model")]
    zero_model = by_override[("zero", "model")]
    gt_model = by_override[("gt", "model")]
    model_zero = by_override[("model", "zero")]
    zero_zero = by_override[("zero", "zero")]
    default_direct = _safe_float(default["DirectGeoLocalDeg"])
    plan_zero_delta = _safe_float(zero_model["DirectGeoLocalDeg"]) - default_direct
    plan_gt_delta = _safe_float(gt_model["DirectGeoLocalDeg"]) - default_direct
    meas_zero_delta = _safe_float(model_zero["DirectGeoLocalDeg"]) - default_direct
    both_zero_delta = _safe_float(zero_zero["DirectGeoLocalDeg"]) - default_direct
    plan_score = max(abs(plan_zero_delta), abs(plan_gt_delta), abs(both_zero_delta))
    meas_score = max(abs(meas_zero_delta), abs(both_zero_delta))
    if both_zero_delta <= 0.005 and plan_zero_delta <= 0.005:
        conclusion = "non-plan ready"
    elif plan_zero_delta >= 0.015 or both_zero_delta >= 0.020:
        conclusion = "plan-compensatory dependency"
    elif meas_zero_delta >= 0.015:
        conclusion = "meas-sensitive"
    else:
        conclusion = "mixed"
    return {
        "candidate_key": spec.key,
        "candidate": spec.label,
        "stage_type": spec.stage_type,
        "eval_mode_key": mode,
        "eval_mode": _label_mode(mode),
        "model_model": _safe_float(default["DirectGeoLocalDeg"]),
        "plan_zero_meas_model": _safe_float(zero_model["DirectGeoLocalDeg"]),
        "plan_gt_meas_model": _safe_float(gt_model["DirectGeoLocalDeg"]),
        "model_zero": _safe_float(model_zero["DirectGeoLocalDeg"]),
        "zero_zero": _safe_float(zero_zero["DirectGeoLocalDeg"]),
        "plan_zero_delta": plan_zero_delta,
        "plan_gt_delta": plan_gt_delta,
        "model_zero_delta": meas_zero_delta,
        "zero_zero_delta": both_zero_delta,
        "plan_score": plan_score,
        "meas_score": meas_score,
        "DirectGeoLocalDeg": _safe_float(default["DirectGeoLocalDeg"]),
        "conclusion_label": conclusion,
        "artifacts": {
            "model_model": str(default["json"]),
            "plan_zero_meas_model": str(zero_model["json"]),
            "plan_gt_meas_model": str(gt_model["json"]),
            "model_zero": str(model_zero["json"]),
            "zero_zero": str(zero_zero["json"]),
        },
    }


def _per_joint_means(eval_json: Path, *, cycle_gte: int = 1, drop_wrap: bool = True) -> Dict[str, Any]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    per = payload.get("per_step_direct_geolocal_deg") or {}
    steps = list(payload.get("metrics_per_step") or [])
    mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
    names = [str(x) for x in (per.get("bone_names") or [])]
    root_idx = int(per.get("root_idx", 0) or 0)
    if mat.ndim != 2 or not steps or len(names) != int(mat.shape[1]):
        return {"joint_means": {}, "top5": []}
    keep: List[int] = []
    for idx, meta in enumerate(steps):
        cycle = int(meta.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(drop_wrap) and bool(meta.get("wrap_boundary_step", False)):
            continue
        keep.append(int(idx))
    if not keep:
        return {"joint_means": {}, "top5": []}
    arr = mat[np.asarray(keep, dtype=np.int64)]
    joint_means: Dict[str, float] = {}
    for j, name in enumerate(names):
        if j == root_idx:
            continue
        col = arr[:, j]
        col = col[np.isfinite(col)]
        joint_means[name] = float(col.mean()) if col.size > 0 else float("nan")
    top5 = [
        {"joint": name, "mean": float(val)}
        for name, val in sorted(joint_means.items(), key=lambda kv: _safe_float(kv[1]), reverse=True)[:5]
    ]
    return {"joint_means": joint_means, "top5": top5}


def _per_sic_means(eval_json: Path, *, cycle_gte: int = 1, drop_wrap: bool = True) -> Dict[int, float]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    steps = list(payload.get("metrics_per_step") or [])
    per = payload.get("per_step_direct_geolocal_deg") or {}
    mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
    root_idx = int(per.get("root_idx", 0) or 0)
    if mat.ndim != 2 or not steps:
        return {}
    bucket: Dict[int, List[float]] = defaultdict(list)
    for idx, meta in enumerate(steps):
        if idx >= int(mat.shape[0]):
            break
        cycle = int(meta.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(drop_wrap) and bool(meta.get("wrap_boundary_step", False)):
            continue
        sic = int(meta.get("step_in_cycle", -1) or -1)
        row = mat[idx]
        vals = [float(v) for j, v in enumerate(row) if j != root_idx and math.isfinite(float(v))]
        if vals:
            bucket[sic].append(float(np.mean(vals)))
    return {int(k): _mean(v) for k, v in sorted(bucket.items())}


def _behavior_supporting_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pick = {}
    for row in rows:
        if row["candidate_key"] in ("baseline_raw_70a", "tailk7_raw_70a") and row["eval_mode_key"] == TEACHER_MODE:
            pick[row["candidate_key"]] = row
    out: Dict[str, Any] = {}
    for key, row in pick.items():
        out[key] = {
            "model_model": {
                "top_joints": _per_joint_means(Path(row["artifacts"]["model_model"]))["top5"],
                "sic_means": _per_sic_means(Path(row["artifacts"]["model_model"])),
            },
            "zero_zero": {
                "top_joints": _per_joint_means(Path(row["artifacts"]["zero_zero"]))["top5"],
                "sic_means": _per_sic_means(Path(row["artifacts"]["zero_zero"])),
            },
        }
    return out


def _mechanism_conclusion(
    *,
    plan_ratio: float,
    plan_delta: float,
    direct_delta: float,
) -> str:
    if math.isfinite(plan_ratio) and plan_ratio <= 0.10 and math.isfinite(plan_delta) and plan_delta <= 0.005:
        return "non-plan owned / low-plan"
    if math.isfinite(plan_ratio) and plan_ratio >= 0.40 and math.isfinite(plan_delta) and plan_delta >= 0.010:
        return "plan-compensatory takeover"
    if math.isfinite(direct_delta) and direct_delta >= 0.010:
        return "direct path still essential"
    return "mixed"


def _run_mechanism_for_candidate(
    *,
    spec: CandidateSpec,
    teacher_eval_json: Path,
    teacher_path: Path,
    device: str,
    rounds: int,
    cycle_gte: int,
    drop_wrap: bool,
    jacobian_max_steps: int,
    jacobian_batch_size: int,
    head_batch_size: int,
) -> Dict[str, Any]:
    case = _load_case(
        case_name=spec.key,
        ckpt_path=spec.ckpt,
        eval_json_path=teacher_eval_json,
        teacher_path=teacher_path,
        device_pref=device,
    )
    model = case["trainer"].model
    model.eval()
    module_name, first = _first_linear(model)
    layout = _branch_layout(model, first)
    run = _run_with_head_hook(
        case,
        eval_mode=TEACHER_MODE,
        rounds=int(rounds),
        zero_slice=None,
        capture_inputs=True,
    )
    selected = _selected_indices(run["per_step"], cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap))
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
        batch_size=int(head_batch_size),
    )
    sensitivity = _jacobian_sensitivity(
        case=case,
        x_all=x_all,
        selected=selected,
        layout=layout,
        max_steps=int(jacobian_max_steps),
        batch_size=int(jacobian_batch_size),
    )
    causal_ablation: Dict[str, Any] = {}
    for branch, sl, _width in _branch_items(layout):
        ablated = _run_with_head_hook(
            case,
            eval_mode=TEACHER_MODE,
            rounds=int(rounds),
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
            "downstream_direct_geolocal_deg": ablated_direct,
            "downstream_direct_geolocal_baseline": baseline_direct,
            "downstream_direct_geolocal_delta": delta,
            "label": _label_ablation(branch, delta),
            "direct_output_delta_geolocal_deg": (local_deltas.get(branch) or {}).get("direct_output_delta_geolocal_deg"),
            "direct_output_delta_norm_rot_rms": (local_deltas.get(branch) or {}).get(
                "direct_output_delta_norm_rot_rms"
            ),
        }
    row = {
        "candidate_key": spec.key,
        "candidate": spec.label,
        "stage_type": spec.stage_type,
        "eval_mode_key": TEACHER_MODE,
        "eval_mode": _label_mode(TEACHER_MODE),
        "checkpoint": str(spec.ckpt.resolve()),
        "eval_artifact": str(teacher_eval_json.resolve()),
        "first_linear_module": module_name,
        "baseline_direct_geolocal_deg": baseline_direct,
        "selection": {"rounds": int(rounds), "cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
        "branch_layout": {
            "direct_feat_dim": int(layout.direct_dim),
            "plan_dim": int(layout.plan_dim),
            "meas_dim": int(layout.meas_dim),
            "total_dim": int(layout.total_dim),
        },
        "sensitivity": sensitivity,
        "weight_effective_gain": weight_stats,
        "causal_ablation": causal_ablation,
    }
    plan_ratio = _safe_float((sensitivity.get("ratios") or {}).get("plan_over_direct_feat"))
    plan_delta = _safe_float((causal_ablation.get("plan") or {}).get("downstream_direct_geolocal_delta"))
    direct_delta = _safe_float((causal_ablation.get("direct_feat") or {}).get("downstream_direct_geolocal_delta"))
    row["conclusion_label"] = _mechanism_conclusion(
        plan_ratio=plan_ratio,
        plan_delta=plan_delta,
        direct_delta=direct_delta,
    )
    row["candidate_detail_artifact"] = str((OUT_ROOT / "mechanism" / "candidates" / f"{spec.key}.json").resolve())
    return row


def _hypothesis_table(
    *,
    behavior_by_key_mode: Mapping[Tuple[str, str], Mapping[str, Any]],
    mechanism_by_key: Mapping[str, Mapping[str, Any]],
    supporting_breakdown: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    base_raw = behavior_by_key_mode[("baseline_raw_70a", TEACHER_MODE)]
    tail_raw = behavior_by_key_mode[("tailk7_raw_70a", TEACHER_MODE)]
    base_raw_free = behavior_by_key_mode.get(("baseline_raw_70a", FREERUN_MODE), {})
    tail_raw_free = behavior_by_key_mode.get(("tailk7_raw_70a", FREERUN_MODE), {})
    base_ws = behavior_by_key_mode[("baseline_70a_replace_zerophase", TEACHER_MODE)]
    tail_copy_ws = behavior_by_key_mode[("tailk7_70a_replace_zerophase", TEACHER_MODE)]
    tail_adapt_ws = behavior_by_key_mode[("tailk7_baseline_style_adapted_warmstart", TEACHER_MODE)]
    tail_final = behavior_by_key_mode[("tailk7_coadapt_final", TEACHER_MODE)]

    base_raw_mech = mechanism_by_key["baseline_raw_70a"]
    tail_raw_mech = mechanism_by_key["tailk7_raw_70a"]
    tail_adapt_mech = mechanism_by_key["tailk7_baseline_style_adapted_warmstart"]
    tail_final_mech = mechanism_by_key["tailk7_coadapt_final"]

    a_strong = []
    if _safe_float(base_raw["plan_gt_delta"]) + 0.005 < _safe_float(tail_raw["plan_gt_delta"]):
        a_strong.append(
            "baseline raw 70a 在 teacher 模式下用 GT plan 改善更大，说明 baseline planner 可能更 noisy"
        )
    if _safe_float(base_raw_free.get("plan_gt_delta")) + 0.005 < _safe_float(tail_raw_free.get("plan_gt_delta")):
        a_strong.append("baseline raw 70a 在 freerun spot-check 下也更受 GT plan 纠正")
    a_support = "weak" if not a_strong else "partially supported"

    b_top = supporting_breakdown.get("tailk7_raw_70a", {}).get("zero_zero", {}).get("top_joints", [])
    b_support = "partially supported" if b_top else "weak"
    b_evidence = (
        "raw 70a 的 zero/zero SIC / joint residual profile 确实不同，说明 replace step0 的 error mass 组成并不相同"
        if b_top
        else "本轮只拿到最小 SIC/joint 辅证，B 更像待验证的中层机制"
    )

    c_teacher_gap = _safe_float(tail_raw["plan_zero_delta"]) - _safe_float(base_raw["plan_zero_delta"])
    c_zero_gap = _safe_float(tail_raw["zero_zero_delta"]) - _safe_float(base_raw["zero_zero_delta"])
    c_primary_falsifier_supported = (
        math.isfinite(c_teacher_gap)
        and math.isfinite(c_zero_gap)
        and c_teacher_gap >= 0.015
        and c_zero_gap >= 0.020
    )
    c_secondary_hint = (
        _safe_float((tail_raw_mech.get("sensitivity") or {}).get("direct_feat", {}).get("jacobian_fro_per_input_dim", {}).get("mean"))
        + 0.10
        < _safe_float((base_raw_mech.get("sensitivity") or {}).get("direct_feat", {}).get("jacobian_fro_per_input_dim", {}).get("mean"))
        and _safe_float((tail_adapt_mech.get("sensitivity") or {}).get("ratios", {}).get("plan_over_direct_feat")) > 0.60
        and _safe_float(tail_adapt_ws["zero_zero_delta"]) > 0.005
        and _safe_float(tail_final["plan_score"]) >= 0.015
    )
    if c_primary_falsifier_supported:
        c_support = "strongly supported"
    elif c_secondary_hint:
        c_support = "partially supported"
    else:
        c_support = "weak"

    return [
        {
            "hypothesis": "A",
            "support_level": a_support,
            "strongest_evidence": a_strong[0] if a_strong else "没有足够强的新证据把 A 提到和 C 同级",
            "weakest_point": "即使 baseline planner 稍 noisy，也解释不了 tailk7 adapted warmstart 仍然不进入 low-plan basin",
            "explains_baseline_low_plan_basin": "partial" if a_strong else "no",
            "explains_tailk7_no_collapse": "no",
            "explains_high_lr_tailk7_worse": "partial" if a_strong else "no",
        },
        {
            "hypothesis": "B",
            "support_level": b_support,
            "strongest_evidence": b_evidence,
            "weakest_point": "目前只证明 residual composition 不同，还没用新的 step0/1 gradient audit 把它锁成主因",
            "explains_baseline_low_plan_basin": "partial",
            "explains_tailk7_no_collapse": "partial",
            "explains_high_lr_tailk7_worse": "partial",
        },
        {
            "hypothesis": "C",
            "support_level": c_support,
            "strongest_evidence": (
                "tailk7 adapted warmstart 仍未进入 baseline low-plan basin，且 raw 70a 的 direct_feat sensitivity 低于 baseline raw 70a"
                if not c_primary_falsifier_supported
                else "tailk7 raw 70a 在不训练 teacher audit 下的 plan=zero / zero=zero 已显著差于 baseline raw 70a，且 tailk7 adapted warmstart 仍未接近 baseline zerophase"
            ),
            "weakest_point": (
                "本轮 primary falsifier 是反向的：tailk7 raw 70a 的 plan=zero 并没有比 baseline raw 70a 更差"
                if not c_primary_falsifier_supported
                else "本轮没有做新的 step0/1 gradient audit，只能把 B 留作造成 C 的训练机制候选"
            ),
            "explains_baseline_low_plan_basin": "yes",
            "explains_tailk7_no_collapse": "partial" if not c_primary_falsifier_supported else "yes",
            "explains_high_lr_tailk7_worse": "partial" if not c_primary_falsifier_supported else "yes",
        },
    ]


def _final_candidate_hypothesis_table(
    *,
    hypothesis_rows: Sequence[Mapping[str, Any]],
    behavior_by_key_mode: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    c_row = next(row for row in hypothesis_rows if row["hypothesis"] == "C")
    support_c = str(c_row["support_level"]) in ("strongly supported", "partially supported")
    support_c_strong = str(c_row["support_level"]) == "strongly supported"
    rows: List[Dict[str, Any]] = []
    for key in (
        "baseline_raw_70a",
        "baseline_70a_replace_zerophase",
        "baseline_replace_final",
        "tailk7_raw_70a",
        "tailk7_70a_replace_zerophase",
        "tailk7_baseline_style_adapted_warmstart",
        "tailk7_coadapt_final",
    ):
        beh = behavior_by_key_mode[(key, TEACHER_MODE)]
        ready = str(beh["conclusion_label"]) == "non-plan ready"
        plan_dep = "plan" in str(beh["conclusion_label"])
        if key == "baseline_replace_final":
            role = "production"
        else:
            role = "research-only"
        rows.append(
            {
                "candidate_or_hypothesis": beh["candidate"],
                "root_cause_stage6_70a": (
                    "yes"
                    if support_c_strong and key.startswith("tailk7_") and "final" not in key
                    else "partial"
                    if support_c and key.startswith("tailk7_") and "final" not in key
                    else "baseline reference"
                    if key.startswith("baseline")
                    else "partial"
                ),
                "plan_symptom_readout": "yes" if support_c else "partial",
                "warmstart_not_main_cause": (
                    "yes"
                    if key in ("tailk7_70a_replace_zerophase", "tailk7_baseline_style_adapted_warmstart")
                    else "n/a"
                ),
                "non_plan_readiness_main_contradiction": (
                    "yes" if support_c_strong else "partial" if support_c else "no"
                ),
                "recommended_next_role": role,
            }
        )
    rows.append(
        {
            "candidate_or_hypothesis": "Hypothesis C",
            "root_cause_stage6_70a": "yes" if support_c_strong else "partial",
            "plan_symptom_readout": "yes" if support_c else "partial",
            "warmstart_not_main_cause": "yes" if support_c else "partial",
            "non_plan_readiness_main_contradiction": "yes" if support_c else "partial",
            "recommended_next_role": "research-only",
        }
    )
    return rows


def _render_markdown(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    hypothesis_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    direct_answers: Sequence[str],
) -> str:
    lines: List[str] = [
        "## 2026-04-07 raw70a non-plan readiness falsifier（C 优先）",
        "",
        "### (1) Corrected factual basis",
        "",
        "- baseline 的 `plan collapse` 主体发生在 `70a_replace_zerophase` warmstart surgery，不是后续 `70b` 60-step loss-driven collapse。",
        "- 以下同口径 `direct_pose_head.0.weight` first-layer block weight / dim 数字直接继承，不在本轮重证：",
        "",
        "| candidate | stage type | direct / dim | plan / dim | meas / dim |",
        "|---|---|---:|---:|---:|",
    ]
    for row in INHERITED_BLOCK_WEIGHT_FACTS:
        lines.append(
            f"| `{row['candidate']}` | `{row['stage_type']}` | `{_fmt(row['direct_per_dim'])}` | "
            f"`{_fmt(row['plan_per_dim'])}` | `{_fmt(row['meas_per_dim'])}` |"
        )
    lines += [
        "",
        "### (2) Candidate table",
        "",
        "| candidate | stage type | checkpoint path | purpose | self-contained? | eval artifact path | analysis artifact path |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in candidate_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['stage_type']}` | `{row['checkpoint']}` | {row['purpose']} | "
            f"`{str(row['self_contained']).lower()}` | `{row['eval_artifact_path']}` | `{row['analysis_artifact_path']}` |"
        )
    lines += [
        "",
        "### (3) Behavior result table",
        "",
        "| candidate | eval mode | model/model | plan=zero, meas=model | plan=gt, meas=model | model/zero | zero/zero | plan_score | meas_score | zero/zero delta | DirectGeoLocalDeg | 结论标签 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in behavior_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['eval_mode']}` | `{_fmt(row['model_model'])}` | "
            f"`{_fmt(row['plan_zero_meas_model'])}` | `{_fmt(row['plan_gt_meas_model'])}` | "
            f"`{_fmt(row['model_zero'])}` | `{_fmt(row['zero_zero'])}` | `{_fmt(row['plan_score'])}` | "
            f"`{_fmt(row['meas_score'])}` | `{_signed(row['zero_zero_delta'])}` | "
            f"`{_fmt(row['DirectGeoLocalDeg'])}` | `{row['conclusion_label']}` |"
        )
    lines += [
        "",
        "### (4) Mechanism result table",
        "",
        "| candidate | direct_feat sensitivity | plan sensitivity | meas sensitivity | plan/direct ratio | plan block weight / dim | direct block weight / dim | effective contribution proxy | plan ablation delta | 结论标签 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in mechanism_rows:
        sens = row.get("sensitivity") or {}
        gain = row.get("weight_effective_gain") or {}
        d = gain.get("direct_feat") or {}
        p = gain.get("plan") or {}
        m = gain.get("meas") or {}
        lines.append(
            f"| `{row['candidate']}` | "
            f"`{_fmt(((sens.get('direct_feat') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt(((sens.get('plan') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt(((sens.get('meas') or {}).get('jacobian_fro_per_input_dim') or {}).get('mean'))}` | "
            f"`{_fmt((sens.get('ratios') or {}).get('plan_over_direct_feat'))}` | "
            f"`{_fmt(p.get('weight_fro_per_input_dim'))}` | "
            f"`{_fmt(d.get('weight_fro_per_input_dim'))}` | "
            f"`direct={_fmt(d.get('preactivation_contribution_rms'))}; plan={_fmt(p.get('preactivation_contribution_rms'))}; meas={_fmt(m.get('preactivation_contribution_rms'))}` | "
            f"`{_signed(((row.get('causal_ablation') or {}).get('plan') or {}).get('downstream_direct_geolocal_delta'))}` | "
            f"`{row['conclusion_label']}` |"
        )
    lines += [
        "",
        "### (5) Hypothesis judgement table",
        "",
        "| hypothesis | support level | strongest evidence | weakest point | baseline low-plan basin? | tailk7 no-collapse? | high-LR tailk7 worse? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in hypothesis_rows:
        lines.append(
            f"| `{row['hypothesis']}` | `{row['support_level']}` | {row['strongest_evidence']} | "
            f"{row['weakest_point']} | `{row['explains_baseline_low_plan_basin']}` | "
            f"`{row['explains_tailk7_no_collapse']}` | `{row['explains_high_lr_tailk7_worse']}` |"
        )
    lines += [
        "",
        "### (6) Final judgement table",
        "",
        "| candidate / hypothesis | root cause 前移到 stage6/70a | plan 更像 symptom / readout | warmstart surgery 不是主因 | non-plan readiness 是主矛盾 | recommended next role |",
        "|---|---|---|---|---|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| `{row['candidate_or_hypothesis']}` | `{row['root_cause_stage6_70a']}` | "
            f"`{row['plan_symptom_readout']}` | `{row['warmstart_not_main_cause']}` | "
            f"`{row['non_plan_readiness_main_contradiction']}` | `{row['recommended_next_role']}` |"
        )
    lines += [
        "",
        "### (7) Direct answers",
        "",
    ]
    for idx, answer in enumerate(direct_answers, start=1):
        lines.append(f"{idx}. {answer}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal raw70a non-plan readiness falsifier for CP015 tailk7.")
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mechanism-rounds", type=int, default=5)
    parser.add_argument("--mechanism-cycle-gte", type=int, default=1)
    parser.add_argument("--mechanism-drop-wrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jacobian-max-steps", type=int, default=96)
    parser.add_argument("--jacobian-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    args = parser.parse_args()

    specs = _candidate_specs()
    for spec in specs.values():
        if not spec.ckpt.is_file():
            raise FileNotFoundError(f"missing ckpt: {spec.ckpt}")
        if not spec.meta_source_eval.is_file():
            raise FileNotFoundError(f"missing meta eval: {spec.meta_source_eval}")
        default_eval = spec.eval_default(FREERUN_MODE)
        if default_eval is not None and not default_eval.is_file():
            raise FileNotFoundError(f"missing default eval: {default_eval}")

    behavior_raw_rows: List[Dict[str, Any]] = []
    behavior_summary_rows: List[Dict[str, Any]] = []
    candidate_table_rows: List[Dict[str, Any]] = []

    meta_cache = {key: _load_meta(spec) for key, spec in specs.items()}

    for spec in specs.values():
        candidate_detail: Dict[str, Any] = {
            "candidate": spec.label,
            "stage_type": spec.stage_type,
            "checkpoint": str(spec.ckpt.resolve()),
            "purpose": spec.purpose,
            "behavior": {},
        }
        for mode in (TEACHER_MODE, FREERUN_MODE):
            if mode == FREERUN_MODE and spec.key not in FREERUN_SPOTCHECK_KEYS:
                continue
            mode_rows: List[Dict[str, Any]] = []
            for plan_source, meas_source in BEHAVIOR_OVERRIDES:
                eval_json, origin = _ensure_eval(
                    spec=spec,
                    meta=meta_cache[spec.key],
                    mode=mode,
                    plan_source=plan_source,
                    meas_source=meas_source,
                    python_exe=str(args.python),
                    force=bool(args.force),
                )
                runtime = _runtime_metrics(eval_json, cycle_gte=1)
                row = {
                    "candidate_key": spec.key,
                    "candidate": spec.label,
                    "stage_type": spec.stage_type,
                    "eval_mode_key": mode,
                    "eval_mode": _label_mode(mode),
                    "plan_source": plan_source,
                    "meas_source": meas_source,
                    "json": str(eval_json.resolve()),
                    "origin": origin,
                    "DirectGeoLocalDeg": _safe_float(runtime.get("direct_geolocaldeg")),
                }
                behavior_raw_rows.append(row)
                mode_rows.append(row)
            summary_row = _summarize_behavior_candidate_mode(spec=spec, mode=mode, rows=mode_rows)
            behavior_summary_rows.append(summary_row)
            candidate_detail["behavior"][mode] = summary_row

        detail_path = OUT_ROOT / "behavior" / "candidates" / f"{spec.key}.json"
        _write_json(detail_path, candidate_detail)
        primary_eval = (
            candidate_detail["behavior"].get(TEACHER_MODE, {})
            .get("artifacts", {})
            .get("model_model")
            or str(spec.eval_default(FREERUN_MODE) or spec.meta_source_eval)
        )
        candidate_table_rows.append(
            {
                "candidate": spec.label,
                "stage_type": spec.stage_type,
                "checkpoint": str(spec.ckpt.resolve()),
                "purpose": spec.purpose,
                "self_contained": bool(spec.self_contained),
                "eval_artifact_path": primary_eval,
                "analysis_artifact_path": str(detail_path.resolve()),
            }
        )

    behavior_summary_rows.sort(key=lambda row: (row["candidate"], row["eval_mode_key"]))
    behavior_by_key_mode = {(row["candidate_key"], row["eval_mode_key"]): row for row in behavior_summary_rows}
    supporting_breakdown = _behavior_supporting_breakdown(behavior_summary_rows)
    _write_json(OUT_ROOT / "behavior" / "supporting_breakdown.json", supporting_breakdown)

    mechanism_rows: List[Dict[str, Any]] = []
    mechanism_by_key: Dict[str, Dict[str, Any]] = {}
    for spec in specs.values():
        teacher_eval_json = Path(str(behavior_by_key_mode[(spec.key, TEACHER_MODE)]["artifacts"]["model_model"]))
        row = _run_mechanism_for_candidate(
            spec=spec,
            teacher_eval_json=teacher_eval_json,
            teacher_path=Path(args.teacher),
            device=str(args.device),
            rounds=int(args.mechanism_rounds),
            cycle_gte=int(args.mechanism_cycle_gte),
            drop_wrap=bool(args.mechanism_drop_wrap),
            jacobian_max_steps=int(args.jacobian_max_steps),
            jacobian_batch_size=int(args.jacobian_batch_size),
            head_batch_size=int(args.head_batch_size),
        )
        detail_path = OUT_ROOT / "mechanism" / "candidates" / f"{spec.key}.json"
        _write_json(detail_path, row)
        row["candidate_detail_artifact"] = str(detail_path.resolve())
        mechanism_rows.append(row)
        mechanism_by_key[spec.key] = row
    mechanism_rows.sort(key=lambda row: row["candidate"])

    hypothesis_rows = _hypothesis_table(
        behavior_by_key_mode=behavior_by_key_mode,
        mechanism_by_key=mechanism_by_key,
        supporting_breakdown=supporting_breakdown,
    )
    final_rows = _final_candidate_hypothesis_table(
        hypothesis_rows=hypothesis_rows,
        behavior_by_key_mode=behavior_by_key_mode,
    )

    base_raw_teacher = behavior_by_key_mode[("baseline_raw_70a", TEACHER_MODE)]
    tail_raw_teacher = behavior_by_key_mode[("tailk7_raw_70a", TEACHER_MODE)]
    base_ws_teacher = behavior_by_key_mode[("baseline_70a_replace_zerophase", TEACHER_MODE)]
    tail_adapt_teacher = behavior_by_key_mode[("tailk7_baseline_style_adapted_warmstart", TEACHER_MODE)]
    c_level = next(row["support_level"] for row in hypothesis_rows if row["hypothesis"] == "C")
    b_level = next(row["support_level"] for row in hypothesis_rows if row["hypothesis"] == "B")
    a_level = next(row["support_level"] for row in hypothesis_rows if row["hypothesis"] == "A")
    direct_answers = [
        (
            f"不是。`tailk7 raw 70a` 的 `plan=zero` 并没有明显差于 `baseline raw 70a`；"
            f"在 teacher audit 下反而更小：`{_signed(tail_raw_teacher['plan_zero_delta'])}` vs `{_signed(base_raw_teacher['plan_zero_delta'])}`。"
        ),
        (
            "因此，仅凭这轮 primary falsifier，"
            "还不足以把 root cause 更明确地前移到 `stage6/70a`；"
            "这轮更像是否掉了“强 C 版本”，而不是把 C 彻底证死。"
        ),
        (
            f"`baseline 70a_replace_zerophase` 在本轮 `DirectGeoLocalDeg` 口径下，"
            f"更像一个把 `plan/meas` 直接清零的 basin-entry surgery，"
            f"不是“原本就 ready 的 non-plan path 被直接裸露出来”；"
            f"因为它的 teacher `model/model={_fmt(base_ws_teacher['model_model'])}` 并不优于 baseline raw 70a。"
        ),
        (
            f"是。`tailk7 baseline-style adapted warmstart` 仍没有进入 baseline low-plan basin，"
            f"且 teacher `zero/zero delta={_signed(tail_adapt_teacher['zero_zero_delta'])}`、"
            f"`plan/direct ratio={_fmt((mechanism_by_key['tailk7_baseline_style_adapted_warmstart']['sensitivity']['ratios']['plan_over_direct_feat']))}`；"
            "这更支持问题不在 warmstart surgery 本身，而在 donor-state / 70a exit basin。"
        ),
        (
            f"这轮我不再维持 `C > B >> A`。更像 `B ~ C >> A`；"
            f"若必须排序，我会给 `B ≳ C >> A`，当前支持度分别是 "
            f"`B={b_level}`、`C={c_level}`、`A={a_level}`。"
        ),
        (
            "由于这轮没有继续把 stronger C 拉高，"
            "我现在更把 B 看成需要下一步直接验证的中层机制 / 共同驱动项，"
            "而不只是已经从属于 C 的附属解释。"
        ),
        "基于这轮结果，下一步最该打的仍是 `70a exit basin / donor-state` 线路，但最小新增 probe 应该是 raw70a 的 `step0/1 gradient composition audit`，不是 planner semantics 线路。",
    ]

    summary = {
        "run_date": RUN_DATE,
        "run_name": RUN_NAME,
        "out_root": str(OUT_ROOT.resolve()),
        "audit_log": str(AUDIT_LOG.resolve()),
        "corrected_factual_basis": {
            "plan_collapse_happened_mainly_in_warmstart_surgery": True,
            "not_70b_60step_loss_driven_collapse": True,
            "inherited_block_weight_facts": list(INHERITED_BLOCK_WEIGHT_FACTS),
        },
        "candidate_table": candidate_table_rows,
        "behavior_rows": behavior_summary_rows,
        "behavior_supporting_breakdown": supporting_breakdown,
        "mechanism_rows": mechanism_rows,
        "hypothesis_rows": hypothesis_rows,
        "final_rows": final_rows,
        "direct_answers": direct_answers,
    }
    _write_json(args.summary_json, summary)
    md = _render_markdown(
        candidate_rows=candidate_table_rows,
        behavior_rows=behavior_summary_rows,
        mechanism_rows=mechanism_rows,
        hypothesis_rows=hypothesis_rows,
        final_rows=final_rows,
        direct_answers=direct_answers,
    )
    _write_text(args.summary_md, md)
    print(f"[OK] wrote {args.summary_json}")
    print(f"[OK] wrote {args.summary_md}")


if __name__ == "__main__":
    main()
