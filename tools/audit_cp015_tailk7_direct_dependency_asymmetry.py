#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_cp015_tailk7_replace_freerun_table import (  # noqa: E402
    _metrics_for_eval_json,
    _safe_float,
)


RUN_DATE = "20260407"
DEFAULT_OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_{RUN_DATE}"
DEFAULT_SUMMARY_JSON = DEFAULT_OUT_ROOT / "summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_OUT_ROOT / "summary.md"
DEFAULT_AUDIT_LOG = DEFAULT_OUT_ROOT / "audit.log"
DEFAULT_PYTHON = sys.executable or "python3"
DEFAULT_DEPTH = 3
PLAN_DROP_PROBE_OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_plan_drop_competition_probe_{RUN_DATE}"
DEFAULT_OVERRIDES: Tuple[Tuple[str, str], ...] = (
    ("model", "model"),
    ("zero", "model"),
    ("gt", "model"),
    ("model", "zero"),
    ("model", "gt"),
    ("zero", "zero"),
    ("gt", "gt"),
)
DEFAULT_MODES: Tuple[str, ...] = ("freerun", "teacher_x_gt")
DEFAULT_CANDIDATE_NAMES: Tuple[str, ...] = (
    "baseline_replace",
    "coadapt_4x_directonly_calibration_240",
    "coadapt_4x_direct_plus_plan_ownership_240_noeventclock",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    base_eval: Path
    self_contained: bool
    event_clock_enabled: bool
    existing_override_evals: Mapping[Tuple[str, str], Path]


CANDIDATE_SPECS: Dict[str, CandidateSpec] = {
    "baseline_replace": CandidateSpec(
        name="baseline_replace",
        base_eval=ROOT
        / "debug_output"
        / "_tmp_posttrain_pipeline_from_bestfree_20260317"
        / "eval_model_source"
        / "new70b_replace_lowdrift"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={},
    ),
    "coadapt_4x_directonly_calibration_240": CandidateSpec(
        name="coadapt_4x_directonly_calibration_240",
        base_eval=ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406"
        / "eval_model_source"
        / "coadapt_4x_directonly_calibration_240"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={
            ("gt", "model"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_gt__meas_model"
            / "Walk_F_freerun_cycles.json",
            ("model", "gt"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_model__meas_gt"
            / "Walk_F_freerun_cycles.json",
            ("gt", "gt"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_gt__meas_gt"
            / "Walk_F_freerun_cycles.json",
            ("zero", "model"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_zero__meas_model"
            / "Walk_F_freerun_cycles.json",
            ("model", "zero"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_model__meas_zero"
            / "Walk_F_freerun_cycles.json",
            ("zero", "zero"): ROOT
            / "debug_output"
            / "_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406"
            / "coadapt_4x_directonly_calibration_240"
            / "plan_zero__meas_zero"
            / "Walk_F_freerun_cycles.json",
        },
    ),
    "coadapt_plan_drop_0p3": CandidateSpec(
        name="coadapt_plan_drop_0p3",
        base_eval=PLAN_DROP_PROBE_OUT_ROOT
        / "eval_model_source"
        / "coadapt_plan_drop_0p3"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={},
    ),
    "coadapt_plan_drop_0p5": CandidateSpec(
        name="coadapt_plan_drop_0p5",
        base_eval=PLAN_DROP_PROBE_OUT_ROOT
        / "eval_model_source"
        / "coadapt_plan_drop_0p5"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={},
    ),
    "coadapt_plan_drop_sched_1p0_to_0p3_240": CandidateSpec(
        name="coadapt_plan_drop_sched_1p0_to_0p3_240",
        base_eval=PLAN_DROP_PROBE_OUT_ROOT
        / "eval_model_source"
        / "coadapt_plan_drop_sched_1p0_to_0p3_240"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={},
    ),
    "coadapt_plan_drop_sched_1p0_to_0p0_240": CandidateSpec(
        name="coadapt_plan_drop_sched_1p0_to_0p0_240",
        base_eval=PLAN_DROP_PROBE_OUT_ROOT
        / "eval_model_source"
        / "coadapt_plan_drop_sched_1p0_to_0p0_240"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=True,
        existing_override_evals={},
    ),
    "coadapt_4x_direct_plus_plan_ownership_240_noeventclock": CandidateSpec(
        name="coadapt_4x_direct_plus_plan_ownership_240_noeventclock",
        base_eval=ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_plan_ownership_calibration_20260406"
        / "eval_model_source"
        / "coadapt_4x_direct_plus_plan_ownership_240_noeventclock"
        / "Walk_F_freerun_cycles.json",
        self_contained=True,
        event_clock_enabled=False,
        existing_override_evals={},
    ),
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


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
    return "teacher-conditioned" if str(mode) == "teacher_x_gt" else "freerun"


def _parse_candidate_names(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_CANDIDATE_NAMES)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in items if name not in CANDIDATE_SPECS]
    if unknown:
        raise SystemExit(f"[FATAL] unknown candidates: {unknown}")
    return items


def _parse_modes(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_MODES)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in items if name not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"[FATAL] unknown modes: {unknown}")
    return items


def _parse_overrides(raw: str) -> List[Tuple[str, str]]:
    if not raw.strip():
        return list(DEFAULT_OVERRIDES)
    pairs: List[Tuple[str, str]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "/" not in token:
            raise SystemExit(f"[FATAL] invalid override token {token!r}; expected plan/meas.")
        plan_source, meas_source = [part.strip() for part in token.split("/", 1)]
        pairs.append((plan_source, meas_source))
    return pairs


def _load_base_eval_meta(base_eval: Path) -> Dict[str, Any]:
    payload = _load_json(base_eval)
    model = payload.get("model")
    teacher_json = payload.get("teacher_json")
    bundle = payload.get("bundle")
    pretrain_template = payload.get("pretrain_template")
    encoder_bundle = payload.get("encoder_bundle")
    if not model or not teacher_json:
        raise SystemExit(f"[FATAL] missing model/teacher_json in {base_eval}")
    return {
        "model": Path(str(model)).expanduser().resolve(),
        "teacher_json": Path(str(teacher_json)).expanduser().resolve(),
        "bundle": Path(str(bundle)).expanduser().resolve() if bundle else None,
        "pretrain_template": Path(str(pretrain_template)).expanduser().resolve() if pretrain_template else None,
        "encoder_bundle": Path(str(encoder_bundle)).expanduser().resolve() if encoder_bundle else None,
        "rounds": int(payload.get("rounds", 5) or 5),
        "time_index_mode": str(payload.get("time_index_mode", "cycle") or "cycle"),
        "event_clock": str(payload.get("event_clock", "auto") or "auto"),
        "phase_reset_source": str(payload.get("phase_reset_source", "none") or "none"),
        "contacts_meas_source": str(payload.get("contacts_meas_source", "model") or "model"),
        "depth": int(payload.get("depth", DEFAULT_DEPTH) or DEFAULT_DEPTH),
    }


def _build_eval_command(
    *,
    python_exe: str,
    meta: Mapping[str, Any],
    out_dir: Path,
    plan_source: str,
    meas_source: str,
    mode: str,
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(meta["teacher_json"]),
        "--model",
        str(meta["model"]),
        "--rounds",
        str(int(meta["rounds"])),
        "--depth",
        str(int(meta["depth"])),
        "--time-index-mode",
        str(meta["time_index_mode"]),
        "--event_clock",
        str(meta["event_clock"]),
        "--phase_reset_source",
        str(meta["phase_reset_source"]),
        "--contacts_meas_source",
        str(meta["contacts_meas_source"]),
        "--direct_pose_plan_source",
        str(plan_source),
        "--direct_pose_meas_source",
        str(meas_source),
        "--lambda_fusion_apply",
        "--log_contacts",
        "--export_joint_direct_geolocal_series",
        "--out",
        str(out_dir),
        "--force",
    ]
    if meta.get("bundle"):
        cmd.extend(["--bundle", str(meta["bundle"])])
    if meta.get("pretrain_template"):
        cmd.extend(["--pretrain-template", str(meta["pretrain_template"])])
    if meta.get("encoder_bundle"):
        cmd.extend(["--encoder-bundle", str(meta["encoder_bundle"])])
    if str(mode) == "teacher_x_gt":
        cmd.append("--freerun_x_gt")
    return cmd


def _run_command(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log_file.write(f"[exit={proc.returncode}]\n\n")
        log_file.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit={proc.returncode}: {' '.join(str(part) for part in cmd)}")


def _extract_step_series(eval_json: Path, *, cycle_gte: int = 1) -> List[Dict[str, Any]]:
    payload = _load_json(eval_json)
    steps = list(payload.get("metrics_per_step", []) or [])
    per = dict(payload.get("per_step_direct_geolocal_deg", {}) or {})
    if not per:
        out_simple: List[Dict[str, Any]] = []
        for step_i, meta in enumerate(steps):
            cycle = int(meta.get("cycle", 0) or 0)
            if cycle < int(cycle_gte):
                continue
            if bool(meta.get("wrap_boundary_step", False)):
                continue
            out_simple.append(
                {
                    "step_index": step_i,
                    "cycle": cycle,
                    "step_in_cycle": int(meta.get("step_in_cycle", 0) or 0),
                    "direct_mean": _safe_float(meta.get("DirectGeoLocalDeg")),
                }
            )
        return out_simple
    mat = list(per.get("DirectGeoLocalDeg", []) or [])
    root_idx = int(per.get("root_idx", 0) or 0)
    out: List[Dict[str, Any]] = []
    for step_i, row in enumerate(mat):
        if step_i >= len(steps):
            break
        meta = steps[step_i]
        cycle = int(meta.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(meta.get("wrap_boundary_step", False)):
            continue
        if not isinstance(row, list):
            continue
        vals: List[float] = []
        for joint_idx, value in enumerate(row):
            if int(joint_idx) == int(root_idx):
                continue
            val = _safe_float(value)
            if math.isfinite(val):
                vals.append(val)
        out.append(
            {
                "step_index": step_i,
                "cycle": cycle,
                "step_in_cycle": int(meta.get("step_in_cycle", 0) or 0),
                "direct_mean": _mean(vals),
            }
        )
    return out


def _runtime_metrics(eval_json: Path, *, cycle_gte: int = 1) -> Dict[str, Any]:
    payload = _load_json(eval_json)
    if "per_step_direct_geolocal_deg" in payload:
        return _metrics_for_eval_json(eval_json, cycle_gte=cycle_gte)

    steps = list(payload.get("metrics_per_step", []) or [])
    direct_vals: List[float] = []
    geo_vals: List[float] = []
    blend_vals: List[float] = []
    lambda_mean_vals: List[float] = []
    lambda_eff_vals: List[float] = []
    for step in steps:
        cycle = int(step.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(step.get("wrap_boundary_step", False)):
            continue
        direct = _safe_float(step.get("DirectGeoLocalDeg"))
        geo = _safe_float(step.get("GeoLocalDeg"))
        blend = _safe_float(step.get("BlendGeoLocalDeg"))
        lam = _safe_float(step.get("LambdaMean"))
        lam_eff = _safe_float(step.get("LambdaEffMean"))
        if math.isfinite(direct):
            direct_vals.append(direct)
        if math.isfinite(geo):
            geo_vals.append(geo)
        if math.isfinite(blend):
            blend_vals.append(blend)
        if math.isfinite(lam):
            lambda_mean_vals.append(lam)
        if math.isfinite(lam_eff):
            lambda_eff_vals.append(lam_eff)

    metrics_per_round = list(payload.get("metrics_per_round", []) or [])
    direct_round = _safe_float((metrics_per_round[-1] if metrics_per_round else {}).get("DirectGeoLocalDeg"))
    return {
        "source": str(eval_json),
        "direct_geolocaldeg": direct_round if math.isfinite(direct_round) else _mean(direct_vals),
        "geo_localdeg_mean": _mean(geo_vals),
        "blend_geolocaldeg_mean": _mean(blend_vals),
        "direct_geolocaldeg_mean_steps": _mean(direct_vals),
        "lambda_mean_avg": _mean(lambda_mean_vals),
        "lambda_eff_avg": _mean(lambda_eff_vals),
        "lambda_present": bool(lambda_mean_vals or lambda_eff_vals),
    }


def _profile_delta(base_json: Path, probe_json: Path) -> Dict[str, Any]:
    base_series = _extract_step_series(base_json, cycle_gte=1)
    probe_series = _extract_step_series(probe_json, cycle_gte=1)
    count = min(len(base_series), len(probe_series))
    delta_by_cycle: Dict[int, List[float]] = defaultdict(list)
    delta_by_sic: Dict[int, List[float]] = defaultdict(list)
    for idx in range(count):
        base = base_series[idx]
        probe = probe_series[idx]
        delta = _safe_float(probe.get("direct_mean")) - _safe_float(base.get("direct_mean"))
        if not math.isfinite(delta):
            continue
        delta_by_cycle[int(base["cycle"])].append(float(delta))
        delta_by_sic[int(base["step_in_cycle"])].append(float(delta))
    cycle_mean_deltas = {str(cycle): _mean(vals) for cycle, vals in sorted(delta_by_cycle.items())}
    sic_mean_deltas = {int(sic): _mean(vals) for sic, vals in delta_by_sic.items()}
    top_abs_sics = [
        {
            "sic": int(sic),
            "delta": float(delta),
        }
        for sic, delta in sorted(
            sic_mean_deltas.items(),
            key=lambda item: abs(_safe_float(item[1])),
            reverse=True,
        )[:5]
    ]
    return {
        "count": int(count),
        "cycle_mean_deltas": cycle_mean_deltas,
        "top_abs_sics": top_abs_sics,
    }


def _classify_delta(*, plan_source: str, meas_source: str, delta: float) -> str:
    if plan_source == "model" and meas_source == "model":
        return "default"
    if abs(delta) <= 0.003:
        return "non-plan path robust"
    if plan_source == "zero" and meas_source in ("model", "zero") and delta >= 0.015:
        return "collapsed without plan"
    if plan_source == "gt" and delta <= -0.010:
        return "plan-sensitive"
    if meas_source == "gt" and delta <= -0.010:
        return "meas-sensitive"
    if meas_source == "zero" and delta >= 0.010:
        return "collapsed without meas"
    return "mild sensitivity"


def _dominant_dependency(plan_score: float, meas_score: float) -> str:
    if plan_score >= meas_score + 0.005:
        return "plan"
    if meas_score >= plan_score + 0.005:
        return "meas"
    return "mixed"


def _summarize_candidate_mode(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_override = {
        (str(row["plan_source"]), str(row["meas_source"])): row
        for row in rows
    }
    default_row = by_override[("model", "model")]
    default_direct = _safe_float(default_row["DirectGeoLocalDeg"])
    plan_gt_delta = _safe_float(by_override[("gt", "model")]["delta_vs_default"])
    meas_gt_delta = _safe_float(by_override[("model", "gt")]["delta_vs_default"])
    plan_zero_delta = _safe_float(by_override[("zero", "model")]["delta_vs_default"])
    meas_zero_delta = _safe_float(by_override[("model", "zero")]["delta_vs_default"])
    both_zero_delta = _safe_float(by_override[("zero", "zero")]["delta_vs_default"])
    both_gt_delta = _safe_float(by_override[("gt", "gt")]["delta_vs_default"])
    plan_score = max(abs(plan_gt_delta), abs(plan_zero_delta), abs(both_gt_delta))
    meas_score = max(abs(meas_gt_delta), abs(meas_zero_delta))
    return {
        "candidate": str(default_row["candidate"]),
        "eval_mode": str(default_row["eval_mode"]),
        "eval_mode_key": str(default_row["eval_mode_key"]),
        "default_direct": default_direct,
        "plan_gt_delta": plan_gt_delta,
        "meas_gt_delta": meas_gt_delta,
        "plan_zero_delta": plan_zero_delta,
        "meas_zero_delta": meas_zero_delta,
        "both_zero_delta": both_zero_delta,
        "both_gt_delta": both_gt_delta,
        "plan_score": plan_score,
        "meas_score": meas_score,
        "dominant_dependency": _dominant_dependency(plan_score, meas_score),
        "non_plan_path_robust": bool(abs(both_zero_delta) <= 0.005),
        "collapsed_without_plan": bool(plan_zero_delta >= 0.015 or both_zero_delta >= 0.020),
        "collapsed_without_meas": bool(meas_zero_delta >= 0.015),
        "plan_shortcut_supported": bool(plan_score >= 0.015 and plan_score >= meas_score + 0.005),
    }


def _summarize_candidate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_mode = {str(row["eval_mode_key"]): row for row in rows}
    freerun = by_mode["freerun"]
    teacher = by_mode["teacher_x_gt"]
    freerun_sens = max(_safe_float(freerun["plan_score"]), _safe_float(freerun["meas_score"]))
    teacher_sens = max(_safe_float(teacher["plan_score"]), _safe_float(teacher["meas_score"]))
    if teacher_sens >= freerun_sens + 0.005:
        primary_issue_mode = "teacher-conditioned"
    elif freerun_sens >= teacher_sens + 0.005:
        primary_issue_mode = "freerun"
    else:
        primary_issue_mode = "similar"
    return {
        "candidate": str(freerun["candidate"]),
        "freerun": freerun,
        "teacher_x_gt": teacher,
        "non_plan_direct_path_strength": (
            "strong"
            if bool(freerun["non_plan_path_robust"]) and bool(teacher["non_plan_path_robust"])
            else "weak"
            if bool(freerun["collapsed_without_plan"]) or bool(teacher["collapsed_without_plan"])
            else "mixed"
        ),
        "plan_dependency_dominant": bool(
            str(freerun["dominant_dependency"]) == "plan" and str(teacher["dominant_dependency"]) == "plan"
        ),
        "meas_dependency_dominant": bool(
            str(freerun["dominant_dependency"]) == "meas" and str(teacher["dominant_dependency"]) == "meas"
        ),
        "plan_shortcut_supported": bool(
            bool(freerun["plan_shortcut_supported"]) or bool(teacher["plan_shortcut_supported"])
        ),
        "primary_issue_mode": primary_issue_mode,
    }


def _render_candidate_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate / run | self-contained? | event_clock enabled? | eval mode | override mode | eval artifact path |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {'yes' if row['self_contained'] else 'no'} | "
            f"{'yes' if row['event_clock_enabled'] else 'no'} | {row['eval_mode']} | "
            f"{row['plan_source']}/{row['meas_source']} | {row['json']} |"
        )
    return lines


def _render_dependency_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | eval mode | plan/meas override | DirectGeoLocalDeg | delta vs default | conclusion label |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['eval_mode']} | {row['plan_source']}/{row['meas_source']} | "
            f"{_fmt(row['DirectGeoLocalDeg'])} | {_signed(row['delta_vs_default'])} | {row['label']} |"
        )
    return lines


def _render_asymmetry_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | baseline-vs-coadapt差异 | non-plan direct path 是否强 | plan 依赖是否主导 | meas 依赖是否主导 | 主要问题发生在 teacher 还是 freerun | 是否支持 `plan` 过强捷径 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        diff_text = (
            "baseline reference"
            if row["candidate"] == "baseline_replace"
            else "weaker non-plan path than baseline"
            if row["non_plan_direct_path_strength"] == "weak"
            else "closer to baseline but still asymmetric"
            if row["non_plan_direct_path_strength"] == "mixed"
            else "close to baseline"
        )
        lines.append(
            f"| {row['candidate']} | {diff_text} | {row['non_plan_direct_path_strength']} | "
            f"{'yes' if row['plan_dependency_dominant'] else 'no'} | "
            f"{'yes' if row['meas_dependency_dominant'] else 'no'} | {row['primary_issue_mode']} | "
            f"{'yes' if row['plan_shortcut_supported'] else 'no'} |"
        )
    return lines


def _render_supporting_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | eval mode | override | cycle mean deltas | top |Δ| SICs |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        cycle_text = ", ".join(
            f"c{cycle}:{_signed(delta, digits=4)}"
            for cycle, delta in row["profile"]["cycle_mean_deltas"].items()
        ) or "n/a"
        sic_text = ", ".join(
            f"{int(item['sic'])}:{_signed(item['delta'], digits=4)}"
            for item in row["profile"]["top_abs_sics"]
        ) or "n/a"
        lines.append(
            f"| {row['candidate']} | {row['eval_mode']} | {row['override']} | {cycle_text} | {sic_text} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal baseline-vs-coadapt direct dependency asymmetry audit.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATE_NAMES))
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--overrides",
        type=str,
        default=",".join(f"{plan}/{meas}" for plan, meas in DEFAULT_OVERRIDES),
    )
    parser.add_argument("--force", action="store_true", help="Re-run outputs already present under --out-root.")
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Ignore historical eval JSONs and run every matrix cell under --out-root.",
    )
    args = parser.parse_args()

    candidate_names = _parse_candidate_names(args.candidates)
    modes = _parse_modes(args.modes)
    overrides = _parse_overrides(args.overrides)

    rows: List[Dict[str, Any]] = []
    meta_cache: Dict[str, Dict[str, Any]] = {}

    for candidate_name in candidate_names:
        spec = CANDIDATE_SPECS[candidate_name]
        meta = _load_base_eval_meta(spec.base_eval)
        meta_cache[candidate_name] = meta
        for mode in modes:
            for plan_source, meas_source in overrides:
                override_key = (str(plan_source), str(meas_source))
                reuse_json: Optional[Path] = None
                origin = "probe"
                if not args.no_reuse_existing and str(mode) == "freerun":
                    if override_key == ("model", "model"):
                        reuse_json = spec.base_eval
                        origin = "existing"
                    elif override_key in spec.existing_override_evals:
                        reuse_json = spec.existing_override_evals[override_key]
                        origin = "existing"

                if reuse_json is None:
                    out_dir = args.out_root / candidate_name / mode / _slug(plan_source, meas_source)
                    eval_json = out_dir / "Walk_F_freerun_cycles.json"
                    if not eval_json.is_file() or args.force:
                        cmd = _build_eval_command(
                            python_exe=args.python,
                            meta=meta,
                            out_dir=out_dir,
                            plan_source=plan_source,
                            meas_source=meas_source,
                            mode=mode,
                        )
                        _run_command(cmd, cwd=ROOT, log_path=args.audit_log)
                    reuse_json = eval_json
                runtime = _runtime_metrics(reuse_json, cycle_gte=1)
                rows.append(
                    {
                        "candidate": candidate_name,
                        "self_contained": bool(spec.self_contained),
                        "event_clock_enabled": bool(spec.event_clock_enabled),
                        "eval_mode": _label_mode(mode),
                        "eval_mode_key": mode,
                        "plan_source": str(plan_source),
                        "meas_source": str(meas_source),
                        "json": str(reuse_json.resolve()),
                        "origin": origin,
                        "DirectGeoLocalDeg": _safe_float(runtime.get("direct_geolocaldeg")),
                        "GeoLocalDegMean": _safe_float(runtime.get("geo_localdeg_mean")),
                        "BlendGeoLocalDegMean": _safe_float(runtime.get("blend_geolocaldeg_mean")),
                        "lambda_present": bool(runtime.get("lambda_present")),
                    }
                )

    rows.sort(key=lambda row: (row["candidate"], row["eval_mode_key"], row["plan_source"], row["meas_source"]))

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["eval_mode_key"]))].append(row)

    candidate_mode_summaries: List[Dict[str, Any]] = []
    for key, group_rows in grouped.items():
        default_direct = None
        for row in group_rows:
            if (row["plan_source"], row["meas_source"]) == ("model", "model"):
                default_direct = _safe_float(row["DirectGeoLocalDeg"])
                break
        if default_direct is None:
            raise SystemExit(f"[FATAL] missing model/model baseline for {key}")
        for row in group_rows:
            row["delta_vs_default"] = _safe_float(row["DirectGeoLocalDeg"]) - default_direct
            row["label"] = _classify_delta(
                plan_source=str(row["plan_source"]),
                meas_source=str(row["meas_source"]),
                delta=_safe_float(row["delta_vs_default"]),
            )
        candidate_mode_summaries.append(_summarize_candidate_mode(group_rows))

    candidate_mode_summaries.sort(key=lambda row: (row["candidate"], row["eval_mode"]))
    candidate_summaries: List[Dict[str, Any]] = []
    for candidate_name in candidate_names:
        mode_rows = [row for row in candidate_mode_summaries if row["candidate"] == candidate_name]
        candidate_summaries.append(_summarize_candidate(mode_rows))

    profile_focus_rows: List[Dict[str, Any]] = []
    for summary in candidate_mode_summaries:
        candidate_name = str(summary["candidate"])
        mode_key = "teacher_x_gt" if str(summary["eval_mode"]) == "teacher-conditioned" else "freerun"
        group_rows = grouped[(candidate_name, mode_key)]
        by_override = {
            (str(row["plan_source"]), str(row["meas_source"])): row
            for row in group_rows
        }
        base_json = Path(str(by_override[("model", "model")]["json"]))
        for override_key in (("gt", "model"), ("model", "gt"), ("zero", "model"), ("zero", "zero")):
            profile_focus_rows.append(
                {
                    "candidate": candidate_name,
                    "eval_mode": str(summary["eval_mode"]),
                    "override": f"{override_key[0]}/{override_key[1]}",
                    "profile": _profile_delta(base_json, Path(str(by_override[override_key]["json"]))),
                }
            )

    candidate_rows_for_table: List[Dict[str, Any]] = []
    for candidate_name in candidate_names:
        spec = CANDIDATE_SPECS[candidate_name]
        for mode in modes:
            group_rows = grouped[(candidate_name, mode)]
            for plan_source, meas_source in DEFAULT_OVERRIDES:
                row = next(
                    (
                        item
                        for item in group_rows
                        if item["plan_source"] == plan_source and item["meas_source"] == meas_source
                    ),
                    None,
                )
                if row is None:
                    continue
                candidate_rows_for_table.append(
                    {
                        "candidate": candidate_name,
                        "self_contained": bool(spec.self_contained),
                        "event_clock_enabled": bool(spec.event_clock_enabled),
                        "eval_mode": str(row["eval_mode"]),
                        "plan_source": plan_source,
                        "meas_source": meas_source,
                        "json": str(row["json"]),
                    }
                )

    candidate_table_lines = _render_candidate_table(candidate_rows_for_table)
    dependency_table_lines = _render_dependency_table(rows)
    asymmetry_table_lines = _render_asymmetry_table(candidate_summaries)
    supporting_table_lines = _render_supporting_table(profile_focus_rows)

    summary = {
        "out_root": str(args.out_root.resolve()),
        "audit_log": str(args.audit_log.resolve()),
        "candidates": {
            name: {
                "base_eval": str(CANDIDATE_SPECS[name].base_eval.resolve()),
                "model": str(meta_cache[name]["model"]),
                "teacher_json": str(meta_cache[name]["teacher_json"]),
                "self_contained": bool(CANDIDATE_SPECS[name].self_contained),
                "event_clock_enabled": bool(CANDIDATE_SPECS[name].event_clock_enabled),
            }
            for name in candidate_names
        },
        "eval_modes": {
            "freerun": {
                "runtime_path": "train.validate.run_freerun_cycles",
                "teacher_conditioned": False,
            },
            "teacher_x_gt": {
                "runtime_path": "train.validate.run_freerun_cycles --freerun_x_gt",
                "teacher_conditioned": True,
            },
        },
        "rows": rows,
        "candidate_mode_summaries": candidate_mode_summaries,
        "candidate_summaries": candidate_summaries,
        "supporting_profiles": profile_focus_rows,
        "tables": {
            "candidate_table": candidate_table_lines,
            "dependency_table": dependency_table_lines,
            "asymmetry_table": asymmetry_table_lines,
            "supporting_table": supporting_table_lines,
        },
    }
    _write_json(args.summary_json, summary)

    md_lines: List[str] = []
    md_lines.append("# Direct dependency asymmetry audit")
    md_lines.append("")
    md_lines.append("## Candidate table")
    md_lines.extend(candidate_table_lines)
    md_lines.append("")
    md_lines.append("## Dependency result table")
    md_lines.extend(dependency_table_lines)
    md_lines.append("")
    md_lines.append("## Asymmetry table")
    md_lines.extend(asymmetry_table_lines)
    md_lines.append("")
    md_lines.append("## Supporting profile table")
    md_lines.extend(supporting_table_lines)
    md_lines.append("")
    _write_text(args.summary_md, "\n".join(md_lines).rstrip() + "\n")

    print(f"[OK] wrote summary json: {args.summary_json}")
    print(f"[OK] wrote summary md: {args.summary_md}")


if __name__ == "__main__":
    main()
