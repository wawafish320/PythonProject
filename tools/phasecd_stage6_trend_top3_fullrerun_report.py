#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from phasea_group_summary import _pick_group_indices


OLD_ENTRY = ROOT / "debug_output/_tmp_stage6_basetrain_compare_20260313/old_bestfree/basetrain_freerun/Walk_F_freerun_cycles.json"
OLD_EXIT = ROOT / "debug_output/_tmp_stage6_basetrain_compare_20260313/old_bestfree/stage6_freerun/Walk_F_freerun_cycles.json"
OUT_ROOT = ROOT / "debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330"
MODEL_ROOT = ROOT / "models/__tmp_phasecd_stage6_trend_top3_fullrerun_20260330"

CANDIDATES = {
    "cplus2_keepd_final": {
        "ckpt_in": ROOT / "models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus2_keepd_seed2024_20260330/ckpt_epoch_017.pth",
        "entry_json": ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/handoff/cplus2_keepd/epoch017/handoff_eval/Walk_F_freerun_cycles.json",
    },
    "control_denseckpt_final": {
        "ckpt_in": ROOT / "models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth",
        "entry_json": ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/handoff/control_denseckpt/epoch015/handoff_eval/Walk_F_freerun_cycles.json",
    },
    "dplus1_orig_final": {
        "ckpt_in": ROOT / "models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330/ckpt_epoch_016.pth",
        "entry_json": ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/handoff/dplus1_orig/epoch016/handoff_eval/Walk_F_freerun_cycles.json",
    },
}

REPORT_MD = OUT_ROOT / "stage6_trend_top3_fullrerun_summary_20260330.md"
REPORT_JSON = OUT_ROOT / "stage6_trend_top3_fullrerun_summary_20260330.json"


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def _fmt(value: Any, digits: int = 6) -> str:
    out = _safe_float(value)
    return f"{out:.{digits}f}" if math.isfinite(out) else "missing"


def _build_step_mask(
    steps: Sequence[Mapping[str, Any]],
    *,
    cycle_gte: int,
    drop_wrap: bool,
    exclude_sic01: bool,
) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        keep = cycle >= int(cycle_gte)
        if keep and drop_wrap and bool(step.get("wrap_boundary_step", False)):
            keep = False
        if keep and exclude_sic01:
            try:
                sic = int(step.get("step_in_cycle", step.get("sic", -1)))
            except Exception:
                sic = -1
            if sic in (0, 1):
                keep = False
        mask.append(bool(keep))
    return mask


def _get_layout(payload: Mapping[str, Any]) -> Dict[str, Any]:
    per = payload["per_step_direct_geolocal_deg"]
    names = [str(x) for x in per["bone_names"]]
    root_idx = int(per.get("root_idx", 0) or 0)
    mat = per["DirectGeoLocalDeg"]
    groups = _pick_group_indices(names, root_idx)
    return {
        "steps": payload["metrics_per_step"],
        "mat": mat,
        "names": names,
        "root_idx": root_idx,
        "groups": groups,
        "name_to_idx": {name: i for i, name in enumerate(names)},
    }


def _collect_values(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
    sic_lo: int | None = None,
    sic_hi: int | None = None,
) -> List[float]:
    values: List[float] = []
    for keep, step, row in zip(mask, steps, mat):
        if not keep or not isinstance(row, list):
            continue
        try:
            sic = int(step.get("step_in_cycle", step.get("sic")))
        except Exception:
            sic = None
        if sic_lo is not None and sic_hi is not None:
            if sic is None or sic < int(sic_lo) or sic > int(sic_hi):
                continue
        for idx in indices:
            if idx >= len(row):
                continue
            value = _safe_float(row[idx])
            if math.isfinite(value):
                values.append(value)
    return values


def _window_mean(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
    sic_lo: int,
    sic_hi: int,
) -> float:
    return _mean(
        _collect_values(
            mat=mat,
            steps=steps,
            mask=mask,
            indices=indices,
            sic_lo=sic_lo,
            sic_hi=sic_hi,
        )
    )


def _group_curve_by_sic(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for keep, step, row in zip(mask, steps, mat):
        if not keep or not isinstance(row, list):
            continue
        try:
            sic = int(step.get("step_in_cycle", step.get("sic")))
        except Exception:
            continue
        bucket = buckets.setdefault(sic, [])
        for idx in indices:
            if idx >= len(row):
                continue
            value = _safe_float(row[idx])
            if math.isfinite(value):
                bucket.append(value)
    return {sic: _mean(values) for sic, values in sorted(buckets.items())}


def _curve_l1(curve_a: Mapping[int, float], curve_b: Mapping[int, float]) -> float:
    keys = sorted(set(curve_a) & set(curve_b))
    if not keys:
        return float("nan")
    return _mean(abs(_safe_float(curve_a[key]) - _safe_float(curve_b[key])) for key in keys)


def _metrics_for_payload(payload: Mapping[str, Any], *, cycle_gte: int, exclude_sic01: bool) -> Dict[str, Any]:
    layout = _get_layout(payload)
    steps = layout["steps"]
    mat = layout["mat"]
    groups = layout["groups"]
    name_to_idx = layout["name_to_idx"]
    mask = _build_step_mask(steps, cycle_gte=cycle_gte, drop_wrap=True, exclude_sic01=exclude_sic01)

    leg_idx = groups["leg"]
    foot_idx = [name_to_idx["foot_l"], name_to_idx["ball_l"]]
    calf_idx = [name_to_idx["calf_r"]]

    leg_mean = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=leg_idx))
    leg_12_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=12, sic_hi=24)
    leg_20_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=20, sic_hi=24)
    leg_49_52 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=49, sic_hi=52)
    leg_57_70 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=57, sic_hi=70)
    calf_2_4 = _window_mean(mat=mat, steps=steps, mask=mask, indices=calf_idx, sic_lo=2, sic_hi=4)

    out = {
        "mask": {
            "cycle_gte": int(cycle_gte),
            "drop_wrap": True,
            "exclude_sic01": bool(exclude_sic01),
            "kept_steps": int(sum(mask)),
            "total_steps": int(len(mask)),
        },
        "all_ex_root": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["all_ex_root"])),
        "leg": leg_mean,
        "nonleg": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["nonleg"])),
        "arm": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["arm"])),
        "else": _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["else"])),
        "foot_l_ball_l_sic12_15": _window_mean(mat=mat, steps=steps, mask=mask, indices=foot_idx, sic_lo=12, sic_hi=15),
        "calf_r_sic2_4": calf_2_4,
        "calf_over_leg": _ratio(_mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=calf_idx)), leg_mean),
        "ratio12_24_57_70": _ratio(leg_12_24, leg_57_70),
        "ratio20_24_plus_49_52_57_70": _ratio(leg_20_24 + leg_49_52, leg_57_70),
        "leg_broad_mean": leg_mean,
        "calf_r_sic2_4_over_leg": _ratio(calf_2_4, leg_mean),
        "leg_sic57_70": leg_57_70,
        "leg_sic12_24": leg_12_24,
        "leg_sic20_24": leg_20_24,
        "leg_sic49_52": leg_49_52,
        "leg_curve": _group_curve_by_sic(mat=mat, steps=steps, mask=mask, indices=leg_idx),
        "all_ex_root_curve": _group_curve_by_sic(mat=mat, steps=steps, mask=mask, indices=groups["all_ex_root"]),
    }
    return out


def _group_summary_means(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    groups = payload.get("groups", {})
    out: Dict[str, float] = {}
    for key in ("all_ex_root", "leg", "nonleg", "arm", "else"):
        group = groups.get(key, {})
        out[key] = _safe_float(group.get("mean"))
    return out


def _trend_flags(entry: Mapping[str, Any], final: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "calf_global_not_more_concentrated": _safe_float(final["calf_over_leg"]) <= _safe_float(entry["calf_over_leg"]),
        "early_calf_not_more_concentrated": _safe_float(final["calf_r_sic2_4_over_leg"]) <= _safe_float(entry["calf_r_sic2_4_over_leg"]),
        "late_tail_not_thinner": _safe_float(final["leg_sic57_70"]) >= _safe_float(entry["leg_sic57_70"]),
        "early_mid_not_more_forward": _safe_float(final["ratio12_24_57_70"]) <= _safe_float(entry["ratio12_24_57_70"]),
        "mid_tail_not_more_forward": _safe_float(final["ratio20_24_plus_49_52_57_70"]) <= _safe_float(entry["ratio20_24_plus_49_52_57_70"]),
    }


def _score_row(
    *,
    entry: Mapping[str, Any],
    final: Mapping[str, Any],
    old_exit: Mapping[str, Any],
) -> Dict[str, float]:
    leg_l1 = _curve_l1(final["leg_curve"], old_exit["leg_curve"])
    all_l1 = _curve_l1(final["all_ex_root_curve"], old_exit["all_ex_root_curve"])
    blended_l1 = (
        0.7 * leg_l1 + 0.3 * all_l1
        if math.isfinite(leg_l1) and math.isfinite(all_l1)
        else float("nan")
    )
    return {
        "delta_calf_over_leg": _safe_float(final["calf_over_leg"]) - _safe_float(entry["calf_over_leg"]),
        "delta_calf_r_sic2_4_over_leg": _safe_float(final["calf_r_sic2_4_over_leg"]) - _safe_float(entry["calf_r_sic2_4_over_leg"]),
        "delta_leg_sic57_70": _safe_float(final["leg_sic57_70"]) - _safe_float(entry["leg_sic57_70"]),
        "delta_ratio12_24_57_70": _safe_float(final["ratio12_24_57_70"]) - _safe_float(entry["ratio12_24_57_70"]),
        "delta_ratio20_24_plus_49_52_57_70": _safe_float(final["ratio20_24_plus_49_52_57_70"]) - _safe_float(entry["ratio20_24_plus_49_52_57_70"]),
        "old_exit_leg_l1": leg_l1,
        "old_exit_all_l1": all_l1,
        "old_exit_blended_l1": blended_l1,
    }


def _rank_candidates(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    def key_fn(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            _safe_float(row["delta_ratio20_24_plus_49_52_57_70"]),
            _safe_float(row["delta_ratio12_24_57_70"]),
            -_safe_float(row["delta_leg_sic57_70"]),
            _safe_float(row["delta_calf_r_sic2_4_over_leg"]),
            _safe_float(row["delta_calf_over_leg"]),
            _safe_float(row["old_exit_blended_l1"]),
        )

    valid = [row for row in rows if not row.get("missing")]
    valid.sort(key=key_fn)
    return [str(row["candidate"]) for row in valid]


def _render_run_status(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | train_ckpt_in | stage6_ckpt | freerun_json | group_summary | missing |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['ckpt_in']} | {row['stage6_ckpt']} | {row['stage6_json']} | "
            f"{row['group_summary']} | {', '.join(row['missing']) if row['missing'] else ''} |"
        )
    return lines


def _render_entry_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | entry_json | leg broad mean | calf_r@SIC2-4 / leg | leg SIC57-70 | ratio12_24/57_70 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['entry_json']} | {_fmt(row['leg_broad_mean'])} | "
            f"{_fmt(row['calf_r_sic2_4_over_leg'])} | {_fmt(row['leg_sic57_70'])} | {_fmt(row['ratio12_24_57_70'])} |"
        )
    return lines


def _render_final_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | stage6_ckpt | stage6_group_summary | all_ex_root | leg | nonleg | arm | else | calf_r/leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | foot_l/ball_l@SIC12-15 | leg broad mean | calf_r@SIC2-4 / leg | leg SIC57-70 | missing |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['stage6_ckpt']} | {row['group_summary']} | {_fmt(row.get('all_ex_root'))} | {_fmt(row.get('leg'))} | "
            f"{_fmt(row.get('nonleg'))} | {_fmt(row.get('arm'))} | {_fmt(row.get('else'))} | {_fmt(row.get('calf_over_leg'))} | "
            f"{_fmt(row.get('ratio12_24_57_70'))} | {_fmt(row.get('ratio20_24_plus_49_52_57_70'))} | {_fmt(row.get('foot_l_ball_l_sic12_15'))} | "
            f"{_fmt(row.get('leg_broad_mean'))} | {_fmt(row.get('calf_r_sic2_4_over_leg'))} | {_fmt(row.get('leg_sic57_70'))} | "
            f"{', '.join(row['missing']) if row['missing'] else ''} |"
        )
    return lines


def main() -> int:
    old_entry_metrics = _metrics_for_payload(_load_json(OLD_ENTRY), cycle_gte=1, exclude_sic01=False)
    old_exit_metrics = _metrics_for_payload(_load_json(OLD_EXIT), cycle_gte=1, exclude_sic01=False)

    run_rows: List[Dict[str, Any]] = []
    entry_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for name, meta in CANDIDATES.items():
        run_name = f"{name}_stage6_trend_fullrerun_20260330"
        stage6_ckpt = MODEL_ROOT / name / f"ckpt_last_{run_name}.pth"
        stage6_json = OUT_ROOT / name / "stage6_freerun/Walk_F_freerun_cycles.json"
        group_summary = OUT_ROOT / name / "stage6_group_summary.json"

        entry_payload = _load_json(meta["entry_json"]) if Path(meta["entry_json"]).exists() else None
        entry_metrics = _metrics_for_payload(entry_payload, cycle_gte=1, exclude_sic01=False) if entry_payload else {}
        entry_rows.append(
            {
                "candidate": name,
                "entry_json": str(meta["entry_json"]),
                **entry_metrics,
            }
        )

        missing: List[str] = []
        if not Path(meta["ckpt_in"]).exists():
            missing.append("ckpt_in")
        if not stage6_ckpt.exists():
            missing.append("stage6_ckpt")
        if not stage6_json.exists():
            missing.append("stage6_freerun_json")
        if not group_summary.exists():
            missing.append("stage6_group_summary")

        run_rows.append(
            {
                "candidate": name,
                "ckpt_in": str(meta["ckpt_in"]),
                "stage6_ckpt": str(stage6_ckpt),
                "stage6_json": str(stage6_json),
                "group_summary": str(group_summary),
                "missing": missing,
            }
        )

        if stage6_json.exists():
            final_payload = _load_json(stage6_json)
            final_metrics = _metrics_for_payload(final_payload, cycle_gte=1, exclude_sic01=False)
            summary_means = _group_summary_means(group_summary)
            for key, value in summary_means.items():
                if math.isfinite(_safe_float(value)):
                    final_metrics[key] = _safe_float(value)
            trend = _trend_flags(entry_metrics, final_metrics) if entry_metrics else {}
            score = _score_row(entry=entry_metrics, final=final_metrics, old_exit=old_exit_metrics) if entry_metrics else {}
        else:
            final_metrics = {}
            trend = {}
            score = {}

        final_rows.append(
            {
                "candidate": name,
                "stage6_ckpt": str(stage6_ckpt),
                "group_summary": str(group_summary),
                "stage6_json": str(stage6_json),
                "missing": missing,
                **final_metrics,
                **trend,
                **score,
            }
        )

    ranking = _rank_candidates(final_rows)

    best_name = ranking[0] if ranking else "missing"
    thin_tail = [
        row["candidate"]
        for row in final_rows
        if not row.get("missing") and _safe_float(row.get("delta_leg_sic57_70")) < 0.0
    ]
    forward_shift = [
        row["candidate"]
        for row in final_rows
        if not row.get("missing")
        and (
            _safe_float(row.get("delta_ratio12_24_57_70")) > 0.0
            or _safe_float(row.get("delta_ratio20_24_plus_49_52_57_70")) > 0.0
        )
    ]
    worth_next = [
        row["candidate"]
        for row in final_rows
        if not row.get("missing")
        and row.get("candidate") == best_name
        and _safe_float(row.get("delta_ratio12_24_57_70")) <= 0.0
        and _safe_float(row.get("delta_ratio20_24_plus_49_52_57_70")) <= 0.0
        and _safe_float(row.get("delta_leg_sic57_70")) >= 0.0
    ]

    md_lines: List[str] = []
    md_lines.append("# Stage6 top3 full rerun summary")
    md_lines.append("")
    md_lines.append("## A. run status")
    md_lines.append("")
    md_lines.extend(_render_run_status(run_rows))
    md_lines.append("")
    md_lines.append("## B. entry trend table")
    md_lines.append("")
    md_lines.extend(_render_entry_table(entry_rows))
    md_lines.append("")
    md_lines.append(
        f"Reference old entry: `{OLD_ENTRY}` | leg broad mean={_fmt(old_entry_metrics['leg_broad_mean'])}, "
        f"calf_r@SIC2-4/leg={_fmt(old_entry_metrics['calf_r_sic2_4_over_leg'])}, "
        f"leg SIC57-70={_fmt(old_entry_metrics['leg_sic57_70'])}, ratio12_24/57_70={_fmt(old_entry_metrics['ratio12_24_57_70'])}."
    )
    md_lines.append("")
    md_lines.append("## C. final Stage6 exit table")
    md_lines.append("")
    md_lines.extend(_render_final_table(final_rows))
    md_lines.append("")
    md_lines.append(
        f"Reference old Stage6 exit: `{OLD_EXIT}` | all_ex_root={_fmt(old_exit_metrics['all_ex_root'])}, "
        f"leg={_fmt(old_exit_metrics['leg'])}, nonleg={_fmt(old_exit_metrics['nonleg'])}, arm={_fmt(old_exit_metrics['arm'])}, else={_fmt(old_exit_metrics['else'])}, "
        f"calf_r/leg={_fmt(old_exit_metrics['calf_over_leg'])}, ratio12_24/57_70={_fmt(old_exit_metrics['ratio12_24_57_70'])}, "
        f"ratio20_24+49_52/57_70={_fmt(old_exit_metrics['ratio20_24_plus_49_52_57_70'])}, foot_l/ball_l@SIC12-15={_fmt(old_exit_metrics['foot_l_ball_l_sic12_15'])}, "
        f"leg broad mean={_fmt(old_exit_metrics['leg_broad_mean'])}, calf_r@SIC2-4/leg={_fmt(old_exit_metrics['calf_r_sic2_4_over_leg'])}, leg SIC57-70={_fmt(old_exit_metrics['leg_sic57_70'])}."
    )
    md_lines.append("")
    md_lines.append("## D. trend conclusion")
    md_lines.append("")
    if best_name != "missing":
        best_row = next(row for row in final_rows if row["candidate"] == best_name)
        md_lines.append(
            f"- Most downstream-friendly continuation: `{best_name}` | "
            f"Δratio12_24/57_70={_fmt(best_row.get('delta_ratio12_24_57_70'))}, "
            f"Δratio20_24+49_52/57_70={_fmt(best_row.get('delta_ratio20_24_plus_49_52_57_70'))}, "
            f"Δleg SIC57-70={_fmt(best_row.get('delta_leg_sic57_70'))}, "
            f"Δcalf_r@SIC2-4/leg={_fmt(best_row.get('delta_calf_r_sic2_4_over_leg'))}, "
            f"old-exit blended L1={_fmt(best_row.get('old_exit_blended_l1'))}."
        )
    else:
        md_lines.append("- Most downstream-friendly continuation: missing.")
    md_lines.append(
        f"- Still keeps moving forward after full Stage6: {', '.join(forward_shift) if forward_shift else 'none'}."
    )
    md_lines.append(
        f"- Still keeps thinning the late tail after full Stage6: {', '.join(thin_tail) if thin_tail else 'none'}."
    )
    md_lines.append(
        f"- Worth pushing to later stage based on these Stage6-only signals: {', '.join(worth_next) if worth_next else 'none'}."
    )

    REPORT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    payload = {
        "old_entry": str(OLD_ENTRY),
        "old_exit": str(OLD_EXIT),
        "run_status": run_rows,
        "entry_rows": entry_rows,
        "final_rows": final_rows,
        "ranking": ranking,
        "report_md": str(REPORT_MD),
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REPORT_MD)
    print(REPORT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
