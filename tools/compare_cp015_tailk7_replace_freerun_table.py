#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_rot_row_group_pose_swaps import (  # noqa: E402
    PRIMARY_METRICS,
    _case_summary,
    _relative_improvement,
)

try:
    from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
except Exception:
    DEFAULT_DIRECT_POSE_LEG_BONES = (
        "thigh_r",
        "calf_r",
        "foot_r",
        "ball_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "ball_l",
    )
    STAGE6_3WAY_ARMCHAIN_BONES = (
        "clavicle_l",
        "upperarm_l",
        "RUpArmTwist_l_01",
        "RUpArmTwist_l_02",
        "lowerarm_l",
        "L_ForeTwist_01",
        "L_ForeTwist_02",
        "hand_l",
        "index_01_l",
        "middle_01_l",
        "ring_01_l",
        "pinky_01_l",
        "thumb_01_l",
        "clavicle_r",
        "upperarm_r",
        "RUpArmTwist_r_01",
        "RUpArmTwist_r_02",
        "lowerarm_r",
        "R_ForeTwist_01",
        "R_ForeTwist_02",
        "hand_r",
        "index_01_r",
        "middle_01_r",
        "ring_01_r",
        "pinky_01_r",
        "thumb_01_r",
    )


RUN_DATE = "20260406"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_freerun_table_{RUN_DATE}"
DEFAULT_SUMMARY_JSON = OUT_ROOT / "summary.json"
DEFAULT_SUMMARY_MD = OUT_ROOT / "summary.md"

DEFAULT_CASES: Tuple[Tuple[str, Path], ...] = (
    (
        "current_frozen_trunk_replace_control",
        ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
        / "eval_model_source"
        / "e3x60_adapter_factorized"
        / "Walk_F_freerun_cycles.json",
    ),
    (
        "baseline_replace",
        ROOT
        / "debug_output"
        / "_tmp_posttrain_pipeline_from_bestfree_20260317"
        / "eval_model_source"
        / "new70b_replace_lowdrift"
        / "Walk_F_freerun_cycles.json",
    ),
    (
        "coadapt_allrot_interface_lrscale_0p04",
        ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"
        / "eval_model_source"
        / "coadapt_allrot_interface_lrscale_0p04"
        / "Walk_F_freerun_cycles.json",
    ),
    (
        "coadapt_allrot_interface_bestlr_longer_2x",
        ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"
        / "eval_model_source"
        / "coadapt_allrot_interface_bestlr_longer_2x"
        / "Walk_F_freerun_cycles.json",
    ),
    (
        "coadapt_allrot_interface_bestlr_longer_3x",
        ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
        / "eval_model_source"
        / "coadapt_allrot_interface_bestlr_longer_3x"
        / "Walk_F_freerun_cycles.json",
    ),
    (
        "coadapt_allrot_interface_bestlr_longer_4x",
        ROOT
        / "debug_output"
        / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
        / "eval_model_source"
        / "coadapt_allrot_interface_bestlr_longer_4x"
        / "Walk_F_freerun_cycles.json",
    ),
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _ratio(num: Any, den: Any) -> float:
    num_v = _safe_float(num)
    den_v = _safe_float(den)
    if not math.isfinite(num_v) or not math.isfinite(den_v) or abs(den_v) <= 1e-12:
        return float("nan")
    return float(num_v / den_v)


def _fmt(x: Any, digits: int = 6) -> str:
    value = _safe_float(x)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _pick_group_indices(names: Sequence[str], root_idx: int) -> Dict[str, List[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    idx_nonleg = [i for i in idx_all if i not in set(idx_leg)]
    idx_else = [i for i in idx_nonleg if i not in set(idx_arm)]
    return {
        "all_ex_root": idx_all,
        "leg": idx_leg,
        "nonleg": idx_nonleg,
        "arm": idx_arm,
        "else": idx_else,
    }


def _build_step_mask(
    steps: Sequence[Mapping[str, Any]],
    *,
    cycle_gte: int,
    drop_wrap: bool,
) -> np.ndarray:
    keep: List[bool] = []
    for rec in steps:
        cycle = int(rec.get("cycle", 0) or 0)
        wrap = bool(rec.get("wrap_boundary_step", False))
        ok = cycle >= int(cycle_gte)
        if drop_wrap and wrap:
            ok = False
        keep.append(bool(ok))
    return np.asarray(keep, dtype=bool)


def _collect_values(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    indices: Sequence[int],
) -> List[float]:
    vals: List[float] = []
    for step_i, row in enumerate(mat):
        if step_i >= len(steps) or step_i >= len(mask) or not bool(mask[step_i]):
            continue
        if not isinstance(row, list):
            continue
        for idx in indices:
            if idx >= len(row):
                continue
            val = _safe_float(row[idx])
            if math.isfinite(val):
                vals.append(float(val))
    return vals


def _window_mean(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    indices: Sequence[int],
    sic_lo: int,
    sic_hi: int,
) -> float:
    vals: List[float] = []
    for step_i, row in enumerate(mat):
        if step_i >= len(steps) or step_i >= len(mask) or not bool(mask[step_i]):
            continue
        if not isinstance(row, list):
            continue
        sic = int(steps[step_i].get("step_in_cycle", 0) or 0)
        if sic < int(sic_lo) or sic > int(sic_hi):
            continue
        for idx in indices:
            if idx >= len(row):
                continue
            val = _safe_float(row[idx])
            if math.isfinite(val):
                vals.append(float(val))
    return _mean(vals)


def _metrics_for_eval_json(eval_json: Path, *, cycle_gte: int = 1) -> Dict[str, Any]:
    payload = _load_json(eval_json)
    per = payload["per_step_direct_geolocal_deg"]
    names = [str(x) for x in per["bone_names"]]
    root_idx = int(per.get("root_idx", 0) or 0)
    mat = per["DirectGeoLocalDeg"]
    steps = payload["metrics_per_step"]
    groups = _pick_group_indices(names, root_idx)
    name_to_idx = {name: i for i, name in enumerate(names)}
    mask = _build_step_mask(steps, cycle_gte=cycle_gte, drop_wrap=True)

    leg_idx = groups["leg"]
    foot_idx = [name_to_idx["foot_l"], name_to_idx["ball_l"]]
    calf_idx = [name_to_idx["calf_r"]]

    leg_mean = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=leg_idx))
    leg_sic12_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=12, sic_hi=24)
    leg_sic20_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=20, sic_hi=24)
    leg_sic49_52 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=49, sic_hi=52)
    leg_sic57_70 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=57, sic_hi=70)
    calf_r_sic2_4 = _window_mean(mat=mat, steps=steps, mask=mask, indices=calf_idx, sic_lo=2, sic_hi=4)

    geo_vals: List[float] = []
    blend_vals: List[float] = []
    direct_vals: List[float] = []
    lambda_mean_vals: List[float] = []
    lambda_eff_vals: List[float] = []
    blend_geo_absdiff: List[float] = []
    for step_i, step in enumerate(steps):
        if step_i >= len(mask) or not bool(mask[step_i]):
            continue
        geo = _safe_float(step.get("GeoLocalDeg"))
        blend = _safe_float(step.get("BlendGeoLocalDeg"))
        direct = _safe_float(step.get("DirectGeoLocalDeg"))
        lam = _safe_float(step.get("LambdaMean"))
        lam_eff = _safe_float(step.get("LambdaEffMean"))
        if math.isfinite(geo):
            geo_vals.append(geo)
        if math.isfinite(blend):
            blend_vals.append(blend)
        if math.isfinite(direct):
            direct_vals.append(direct)
        if math.isfinite(lam):
            lambda_mean_vals.append(lam)
        if math.isfinite(lam_eff):
            lambda_eff_vals.append(lam_eff)
        if math.isfinite(geo) and math.isfinite(blend):
            blend_geo_absdiff.append(abs(blend - geo))

    return {
        "source": str(eval_json),
        "mask": {
            "cycle_gte": int(cycle_gte),
            "drop_wrap": True,
            "kept_steps": int(mask.sum()),
            "total_steps": int(mask.shape[0]),
        },
        "direct_geolocaldeg": _safe_float((((payload.get("metrics_per_round", []) or [{}])[-1]).get("DirectGeoLocalDeg"))),
        "geo_localdeg_mean": _mean(geo_vals),
        "blend_geolocaldeg_mean": _mean(blend_vals),
        "direct_geolocaldeg_mean_steps": _mean(direct_vals),
        "blend_geo_max_abs_diff": max(blend_geo_absdiff) if blend_geo_absdiff else float("nan"),
        "lambda_mean_avg": _mean(lambda_mean_vals),
        "lambda_eff_avg": _mean(lambda_eff_vals),
        "lambda_present": bool(lambda_mean_vals or lambda_eff_vals),
        "all_ex_root": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["all_ex_root"])),
        "leg": leg_mean,
        "nonleg": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["nonleg"])),
        "arm": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["arm"])),
        "else": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["else"])),
        "calf_r_over_leg": _ratio(_mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=calf_idx)), leg_mean),
        "ratio12_24_over_57_70": _ratio(leg_sic12_24, leg_sic57_70),
        "ratio20_24_plus_49_52_over_57_70": _ratio(leg_sic20_24 + leg_sic49_52, leg_sic57_70),
        "foot_l_ball_l_sic12_15": _window_mean(mat=mat, steps=steps, mask=mask, indices=foot_idx, sic_lo=12, sic_hi=15),
        "calf_r_sic2_4": calf_r_sic2_4,
        "leg_sic57_70": leg_sic57_70,
        "leg_sic12_24": leg_sic12_24,
        "leg_sic20_24": leg_sic20_24,
        "leg_sic49_52": leg_sic49_52,
    }


def _pose_case_summary(eval_json: Path) -> Dict[str, Any]:
    payload = _load_json(eval_json)
    return _case_summary(
        {
            "metrics_per_round": list(payload.get("metrics_per_round", []) or []),
            "metrics_per_step": list(payload.get("metrics_per_step", []) or []),
        },
        root_name="pelvis",
    )


def _primary_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        var = (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _primary_bucket_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any], bucket: str) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (
            (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        var = (
            (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _parse_case_specs(case_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in case_specs:
        if "=" not in str(item):
            raise SystemExit(f"[FATAL] expected --case label=/path/to/Walk_F_freerun_cycles.json, got {item!r}")
        label, raw_path = str(item).split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not label:
            raise SystemExit(f"[FATAL] empty label in --case {item!r}")
        out.append((label, path))
    return out


def _render_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else | calf_r/leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | foot_l/ball_l@SIC12-15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['direct_geolocaldeg'])} | {_fmt(row['all_ex_root'])} | {_fmt(row['leg'])} | "
            f"{_fmt(row['nonleg'])} | {_fmt(row['arm'])} | {_fmt(row['else'])} | {_fmt(row['calf_r_over_leg'])} | "
            f"{_fmt(row['ratio12_24_over_57_70'])} | {_fmt(row['ratio20_24_plus_49_52_over_57_70'])} | {_fmt(row['foot_l_ball_l_sic12_15'])} |"
        )
    return lines


def _render_delta_table(rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> List[str]:
    lines = [
        f"| compare_vs_{base['candidate']} | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else | d_calf_r/leg | d_ratio12_24/57_70 | d_ratio20_24+49_52/57_70 | d_foot_l/ball_l@SIC12-15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["candidate"] == base["candidate"]:
            continue
        lines.append(
            f"| {row['candidate']} | {_fmt(row['direct_geolocaldeg'] - base['direct_geolocaldeg'])} | {_fmt(row['all_ex_root'] - base['all_ex_root'])} | "
            f"{_fmt(row['leg'] - base['leg'])} | {_fmt(row['nonleg'] - base['nonleg'])} | {_fmt(row['arm'] - base['arm'])} | "
            f"{_fmt(row['else'] - base['else'])} | {_fmt(row['calf_r_over_leg'] - base['calf_r_over_leg'])} | "
            f"{_fmt(row['ratio12_24_over_57_70'] - base['ratio12_24_over_57_70'])} | "
            f"{_fmt(row['ratio20_24_plus_49_52_over_57_70'] - base['ratio20_24_plus_49_52_over_57_70'])} | "
            f"{_fmt(row['foot_l_ball_l_sic12_15'] - base['foot_l_ball_l_sic12_15'])} |"
        )
    return lines


def _render_pose_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs first row | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['Rot6dLocalL2'])} | {_fmt(row['Rot6dLocalL2Weighted'])} | {_fmt(row['GeoDeg'])} | "
            f"{_fmt(row['KeyBoneGeoDegMean'])} | {_fmt(row['KeyBoneGeoLocalDegMean'])} | {_fmt(row['GeoLocalDeg'])} | "
            f"{100.0 * _safe_float(row['primary_rel_vs_anchor']):+.2f}% | {100.0 * _safe_float(row['d0_9']):+.2f}% | "
            f"{100.0 * _safe_float(row['d10_20']):+.2f}% | {100.0 * _safe_float(row['d21_43']):+.2f}% | "
            f"{100.0 * _safe_float(row['sic0_10']):+.2f}% | {100.0 * _safe_float(row['sic11_21']):+.2f}% | "
            f"{100.0 * _safe_float(row['sic22_43']):+.2f}% |"
        )
    return lines


def _render_runtime_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | GeoLocalDeg mean | BlendGeoLocalDeg mean | DirectGeoLocalDeg mean(step) | DirectGeoLocalDeg round | blend-geo max_abs_diff | lambda_present | lambda_mean_avg | lambda_eff_avg |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['geo_localdeg_mean'])} | {_fmt(row['blend_geolocaldeg_mean'])} | "
            f"{_fmt(row['direct_geolocaldeg_mean_steps'])} | {_fmt(row['direct_geolocaldeg'])} | {_fmt(row['blend_geo_max_abs_diff'])} | "
            f"{str(bool(row['lambda_present'])).lower()} | {_fmt(row['lambda_mean_avg'])} | {_fmt(row['lambda_eff_avg'])} |"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare cp015 tailk7 replace freerun candidates with a compact DirectGeoLocalDeg group/probe table."
    )
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional label=/abs/or/rel/path/to/Walk_F_freerun_cycles.json. Replaces the built-in default case list.",
    )
    ap.add_argument("--cycle-gte", type=int, default=1, help="Keep only steps with cycle >= this value. Default: 1")
    ap.add_argument("--out-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    ap.add_argument("--out-md", type=str, default=str(DEFAULT_SUMMARY_MD))
    args = ap.parse_args()

    cases = _parse_case_specs(args.case) if args.case else list(DEFAULT_CASES)
    if not cases:
        raise SystemExit("[FATAL] no cases to compare")

    rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []
    for label, eval_json in cases:
        if not eval_json.is_file():
            raise FileNotFoundError(f"missing freerun json: {eval_json}")
        metrics = _metrics_for_eval_json(eval_json, cycle_gte=int(args.cycle_gte))
        metrics["candidate"] = str(label)
        rows.append(metrics)
        pose_summary = _pose_case_summary(eval_json)
        pose_rows.append(
            {
                "candidate": str(label),
                "pose_summary": pose_summary,
                "Rot6dLocalL2": _safe_float((((pose_summary.get("metrics", {}) or {}).get("Rot6dLocalL2", {}) or {}).get("steps", {}) or {}).get("mean")),
                "Rot6dLocalL2Weighted": _safe_float((((pose_summary.get("metrics", {}) or {}).get("Rot6dLocalL2Weighted", {}) or {}).get("steps", {}) or {}).get("mean")),
                "GeoDeg": _safe_float((((pose_summary.get("metrics", {}) or {}).get("GeoDeg", {}) or {}).get("steps", {}) or {}).get("mean")),
                "KeyBoneGeoDegMean": _safe_float((((pose_summary.get("metrics", {}) or {}).get("KeyBoneGeoDegMean", {}) or {}).get("steps", {}) or {}).get("mean")),
                "KeyBoneGeoLocalDegMean": _safe_float((((pose_summary.get("metrics", {}) or {}).get("KeyBoneGeoLocalDegMean", {}) or {}).get("steps", {}) or {}).get("mean")),
                "GeoLocalDeg": _safe_float((((pose_summary.get("metrics", {}) or {}).get("GeoLocalDeg", {}) or {}).get("steps", {}) or {}).get("mean")),
            }
        )

    if pose_rows:
        anchor_summary = pose_rows[0]["pose_summary"]
        for row in pose_rows:
            pose_summary = row["pose_summary"]
            row["primary_rel_vs_anchor"] = _primary_relative_mean(pose_summary, anchor_summary)
            for bucket in ("d0_9", "d10_20", "d21_43", "sic0_10", "sic11_21", "sic22_43"):
                row[bucket] = _primary_bucket_relative_mean(pose_summary, anchor_summary, bucket)
            del row["pose_summary"]

    summary = {
        "cycle_gte": int(args.cycle_gte),
        "legacy_direct_probe_cases": rows,
        "pose_primary_cases": pose_rows,
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    _write_json(out_json, summary)

    lines: List[str] = [
        "# cp015 tailk7 replace freerun compare table",
        "",
        f"- cycle_gte: {int(args.cycle_gte)}",
        "- metric family A: legacy DirectGeoLocalDeg group means + intuitive hotspot / ratio probes",
        "- metric family B: pose-side primary metrics + bucket improvements relative to the first row",
        "",
        "## Legacy Direct Probe Table",
        "",
    ]
    lines.extend(_render_table(rows))
    if rows:
        lines.extend(["", "## Delta Vs First Row", ""])
        lines.extend(_render_delta_table(rows, rows[0]))
    lines.extend(
        [
            "",
            "## Runtime Path Status",
            "",
            "> This table shows whether the final freerun path is actually changing through direct / lambda / blend.",
            "",
        ]
    )
    lines.extend(_render_runtime_table(rows))
    lines.extend(
        [
            "",
            "## Pose-Side Primary Table",
            "",
            "> Note: for the current replace-stage co-adapt question, this table is the actually decision-relevant one.",
            "> The legacy direct probe table above can stay nearly flat because it reads the direct path, while this round mainly changes the incremental path.",
            "",
        ]
    )
    lines.extend(_render_pose_table(pose_rows))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    print("LEGACY_DIRECT_PROBE")
    for line in _render_table(rows):
        print(line)
    print("RUNTIME_PATH_STATUS")
    for line in _render_runtime_table(rows):
        print(line)
    print("POSE_PRIMARY")
    for line in _render_pose_table(pose_rows):
        print(line)


if __name__ == "__main__":
    main()
