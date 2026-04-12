#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_cp015_tailk7_rot_row_group_pose_swaps import PRIMARY_METRICS  # noqa: E402
from tools.compare_cp015_tailk7_replace_freerun_table import (  # noqa: E402
    _metrics_for_eval_json,
    _pose_case_summary,
    _safe_float,
)


RUN_DATE = "20260406"
DEFAULT_BRIDGE_SUMMARY = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_direct_recovery_bridge_{RUN_DATE}" / "summary.json"
)
DEFAULT_OUT_JSON = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_direct_recovery_bridge_{RUN_DATE}" / "baseline_replacement_gate.json"
)
DEFAULT_OUT_MD = (
    ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_direct_recovery_bridge_{RUN_DATE}" / "baseline_replacement_gate.md"
)
DIRECT_GATE_MARGIN = 0.01


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt(x: Any, digits: int = 6) -> str:
    value = _safe_float(x)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _bool_str(x: Any) -> str:
    return "yes" if bool(x) else "no"


def _primary_metrics(eval_json: Path) -> Dict[str, Any]:
    pose_summary = _pose_case_summary(eval_json)
    out: Dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        out[metric] = _safe_float(
            ((((pose_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean"))
        )
    out["GeoLocalDeg"] = _safe_float(
        ((((pose_summary.get("metrics", {}) or {}).get("GeoLocalDeg", {}) or {}).get("steps", {}) or {}).get("mean"))
    )
    return out


def _all_primary_better(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    for metric in PRIMARY_METRICS:
        cur = _safe_float(candidate.get(metric))
        ref = _safe_float(baseline.get(metric))
        if (not math.isfinite(cur)) or (not math.isfinite(ref)) or cur >= ref:
            return False
    return True


def _collect_case_row(case: Mapping[str, Any]) -> Dict[str, Any]:
    eval_json = Path(str(case["eval"]))
    runtime = _metrics_for_eval_json(eval_json, cycle_gte=1)
    pose = _primary_metrics(eval_json)
    return {
        "candidate": str(case["name"]),
        "kind": str(case["kind"]),
        "direct_source": str(case.get("direct_source", case["name"])),
        "pose_source": str(case.get("pose_source", case["name"])),
        "runtime": runtime,
        "pose": pose,
        "swap_mode": str(case.get("swap_mode", case["kind"])),
        "steps_per_epoch": case.get("steps_per_epoch"),
        "swap_report": case.get("swap_report"),
    }


def _render_final_compare(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | DirectGeoLocalDeg | delta vs baseline gate | GeoLocalDeg mean | BlendGeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    baseline_direct = _safe_float(next(row["runtime"]["direct_geolocaldeg"] for row in rows if row["candidate"] == "baseline_replace"))
    gate_threshold = baseline_direct + float(DIRECT_GATE_MARGIN)
    for row in rows:
        runtime = row["runtime"]
        pose = row["pose"]
        lines.append(
            f"| {row['candidate']} | {_fmt(runtime['direct_geolocaldeg'])} | {_fmt(_safe_float(runtime['direct_geolocaldeg']) - gate_threshold)} | {_fmt(runtime['geo_localdeg_mean'])} | "
            f"{_fmt(runtime['blend_geolocaldeg_mean'])} | {str(bool(runtime['lambda_present'])).lower()} | "
            f"{_fmt(pose['Rot6dLocalL2'])} | {_fmt(pose['Rot6dLocalL2Weighted'])} | {_fmt(pose['GeoDeg'])} | "
            f"{_fmt(pose['KeyBoneGeoDegMean'])} | {_fmt(pose['KeyBoneGeoLocalDegMean'])} |"
        )
    return lines


def _render_direct_swap_table(rows: Sequence[Mapping[str, Any]], *, baseline_direct: float, coadapt_direct: float) -> List[str]:
    lines = [
        "| candidate | direct source | pose source | DirectGeoLocalDeg | delta vs baseline_replace | delta vs coadapt_4x |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        cur = _safe_float(row["runtime"]["direct_geolocaldeg"])
        lines.append(
            f"| {row['candidate']} | {row['direct_source']} | {row['pose_source']} | {_fmt(cur)} | "
            f"{_fmt(cur - baseline_direct)} | {_fmt(cur - coadapt_direct)} |"
        )
    return lines


def _render_gate_table(rows: Sequence[Mapping[str, Any]], *, baseline_pose: Mapping[str, Any], baseline_direct: float) -> List[str]:
    lines = [
        "| candidate | pose better than baseline? | direct non-regression vs baseline? | replace baseline eligible? |",
        "|---|---|---|---|",
    ]
    for row in rows:
        pose_better = _all_primary_better(row["pose"], baseline_pose)
        direct_delta = _safe_float(row["runtime"]["direct_geolocaldeg"]) - float(baseline_direct)
        direct_ok = bool(math.isfinite(direct_delta) and direct_delta <= float(DIRECT_GATE_MARGIN))
        lines.append(
            f"| {row['candidate']} | {_bool_str(pose_better)} | {_bool_str(direct_ok)} | {_bool_str(pose_better and direct_ok)} |"
        )
    return lines


def _render_calibration_trajectory(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_direct: float,
    coadapt_direct: float,
) -> List[str]:
    lines = [
        "| candidate | steps | DirectGeoLocalDeg | delta vs baseline_replace | delta vs coadapt_4x | pose preserved vs coadapt_4x? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        cur = _safe_float(row["runtime"]["direct_geolocaldeg"])
        lines.append(
            f"| {row['candidate']} | {int(row.get('steps_per_epoch') or 0)} | {_fmt(cur)} | "
            f"{_fmt(cur - baseline_direct)} | {_fmt(cur - coadapt_direct)} | "
            f"{_bool_str(bool(row.get('pose_preserved_vs_coadapt4x')))} |"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize cp015 tailk7 replace direct recovery bridge with baseline replacement hard gate.")
    ap.add_argument("--bridge-summary", type=Path, default=DEFAULT_BRIDGE_SUMMARY)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    bridge_summary = json.loads(args.bridge_summary.read_text(encoding="utf-8"))
    case_rows = [_collect_case_row(case) for case in bridge_summary["cases"]]
    by_name = {row["candidate"]: row for row in case_rows}

    baseline = by_name["baseline_replace"]
    coadapt = by_name["coadapt_allrot_interface_bestlr_longer_4x"]
    direct_only_rows = [
        {
            **row,
            "pose_preserved_vs_coadapt4x": (
                case.get("pose_preservation_vs_coadapt4x", {}).get("within_rel_tol", False)
                if isinstance(case, dict)
                else False
            ),
            "trajectory_order": case.get("trajectory_order"),
            "warmstart_case": case.get("warmstart_case"),
            "warmstart_ckpt": case.get("warmstart_ckpt"),
        }
        for case in bridge_summary["cases"]
        for row in [_collect_case_row(case)]
        if str(case.get("kind")) == "directonly_calibration"
    ]
    direct_only_rows.sort(
        key=lambda row: (
            int(row.get("trajectory_order")) if row.get("trajectory_order") is not None else 10**9,
            int(row.get("steps_per_epoch") or 0),
        )
    )
    swap_rows = [
        by_name["coadapt_4x_plus_baseline_directpose_swap"],
        by_name["coadapt_4x_plus_control_directpose_swap"],
        by_name["baseline_plus_coadapt4x_directpose_swap"],
    ]
    gate_rows: List[Dict[str, Any]] = [
        baseline,
        by_name["current_frozen_trunk_replace_control"],
        by_name["coadapt_allrot_interface_bestlr_longer_4x"],
        by_name["coadapt_4x_plus_baseline_directpose_swap"],
        by_name["coadapt_4x_plus_control_directpose_swap"],
    ]
    gate_rows.extend(direct_only_rows)

    final_compare_rows = [
        by_name["current_frozen_trunk_replace_control"],
        baseline,
        coadapt,
        by_name["coadapt_4x_plus_baseline_directpose_swap"],
        by_name["coadapt_4x_plus_control_directpose_swap"],
        by_name["baseline_plus_coadapt4x_directpose_swap"],
    ]
    final_compare_rows.extend(direct_only_rows)

    baseline_direct = _safe_float(baseline["runtime"]["direct_geolocaldeg"])
    coadapt_direct = _safe_float(coadapt["runtime"]["direct_geolocaldeg"])
    baseline_gate_threshold = baseline_direct + float(DIRECT_GATE_MARGIN)
    trainable_direct_only_rows = [
        row for row in direct_only_rows
        if _all_primary_better(row["pose"], baseline["pose"])
        and (_safe_float(row["runtime"]["direct_geolocaldeg"]) - baseline_direct) <= float(DIRECT_GATE_MARGIN)
    ]
    best_direct_only_row = min(
        direct_only_rows,
        key=lambda row: _safe_float(row["runtime"]["direct_geolocaldeg"]),
    ) if direct_only_rows else None

    judgements = {
        "coadapt_4x_plus_baseline_directpose_repairs_direct": bool(
            _safe_float(by_name["coadapt_4x_plus_baseline_directpose_swap"]["runtime"]["direct_geolocaldeg"]) - baseline_direct
            <= float(DIRECT_GATE_MARGIN)
        ),
        "coadapt_4x_plus_baseline_directpose_preserves_pose_primary": bool(
            _all_primary_better(by_name["coadapt_4x_plus_baseline_directpose_swap"]["pose"], baseline["pose"])
        ),
        "direct_regression_mainly_direct_pose_stale_untrained": bool(
            (_safe_float(by_name["coadapt_4x_plus_baseline_directpose_swap"]["runtime"]["direct_geolocaldeg"]) - baseline_direct)
            <= float(DIRECT_GATE_MARGIN)
            and (_safe_float(by_name["baseline_plus_coadapt4x_directpose_swap"]["runtime"]["direct_geolocaldeg"]) - baseline_direct)
            > float(DIRECT_GATE_MARGIN)
        ),
        "contact_plan_or_lambda_required_for_direct_repair": False,
        "has_pose_better_and_direct_non_regressed_candidate": any(
            _all_primary_better(row["pose"], baseline["pose"])
            and ((_safe_float(row["runtime"]["direct_geolocaldeg"]) - baseline_direct) <= float(DIRECT_GATE_MARGIN))
            for row in gate_rows
        ),
        "baseline_direct_gate_threshold": baseline_gate_threshold,
        "any_direct_only_case_clears_gate": any(
            (_safe_float(row["runtime"]["direct_geolocaldeg"]) - baseline_direct) <= float(DIRECT_GATE_MARGIN)
            for row in direct_only_rows
        ),
        "trainable_direct_only_candidate_exists": bool(trainable_direct_only_rows),
        "best_trainable_direct_only_candidate": (
            None
            if not trainable_direct_only_rows
            else min(
                trainable_direct_only_rows,
                key=lambda row: _safe_float(row["runtime"]["direct_geolocaldeg"]),
            )["candidate"]
        ),
        "best_direct_only_candidate": None if best_direct_only_row is None else best_direct_only_row["candidate"],
        "best_direct_only_direct_geolocaldeg": (
            float("nan")
            if best_direct_only_row is None
            else _safe_float(best_direct_only_row["runtime"]["direct_geolocaldeg"])
        ),
    }
    judgements["coadapt_still_cannot_replace_baseline"] = not bool(
        judgements["has_pose_better_and_direct_non_regressed_candidate"]
    )
    judgements["coadapt_still_cannot_replace_baseline_with_trainable_candidate"] = not bool(
        judgements["trainable_direct_only_candidate_exists"]
    )
    judgements["next_priority_direct_only_calibration_not_basetrain"] = bool(
        bridge_summary["judgements"]["proceed_to_direct_only_calibration"]
    )
    judgements["adapter_not_next_priority"] = True

    out = {
        "bridge_summary": str(args.bridge_summary),
        "direct_gate_margin": float(DIRECT_GATE_MARGIN),
        "baseline_direct_gate_threshold": baseline_gate_threshold,
        "tables": {
            "final_intuitive_compare": final_compare_rows,
            "direct_swap_table": swap_rows,
            "baseline_replacement_gate": gate_rows,
            "calibration_trajectory": direct_only_rows,
        },
        "judgements": judgements,
    }
    _write_json(args.out_json, out)

    lines: List[str] = [
        "# cp015 tailk7 replace baseline replacement gate",
        "",
        "## Final Intuitive Compare",
        "",
    ]
    lines.extend(_render_final_compare(final_compare_rows))
    lines.extend(["", "## Direct Swap Table", ""])
    lines.extend(
        _render_direct_swap_table(
            swap_rows,
            baseline_direct=baseline_direct,
            coadapt_direct=coadapt_direct,
        )
    )
    lines.extend(["", "## Baseline Replacement Gate", ""])
    lines.extend(
        _render_gate_table(
            gate_rows,
            baseline_pose=baseline["pose"],
            baseline_direct=baseline_direct,
        )
    )
    lines.extend(["", "## Calibration Trajectory", ""])
    lines.extend(
        _render_calibration_trajectory(
            direct_only_rows,
            baseline_direct=baseline_direct,
            coadapt_direct=coadapt_direct,
        )
    )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            f"- `coadapt_4x + baseline direct_pose_*` repairs direct gate: {judgements['coadapt_4x_plus_baseline_directpose_repairs_direct']}",
            f"- pose-side primary metrics stay better than baseline after that swap: {judgements['coadapt_4x_plus_baseline_directpose_preserves_pose_primary']}",
            f"- direct regression is mainly a stale/untrained `direct_pose_*` ownership issue: {judgements['direct_regression_mainly_direct_pose_stale_untrained']}",
            f"- contact-plan / lambda path is required for direct repair: {judgements['contact_plan_or_lambda_required_for_direct_repair']}",
            f"- there exists a candidate that beats baseline on pose-side primary metrics and clears the direct hard gate: {judgements['has_pose_better_and_direct_non_regressed_candidate']}",
            f"- there exists a trainable direct-only candidate that beats baseline on pose-side primary metrics and clears the direct hard gate: {judgements['trainable_direct_only_candidate_exists']}",
            f"- if no such candidate existed, co-adapt still could not replace baseline: {judgements['coadapt_still_cannot_replace_baseline']}",
            f"- if no trainable such candidate existed, co-adapt still could not replace baseline with direct-only calibration: {judgements['coadapt_still_cannot_replace_baseline_with_trainable_candidate']}",
            f"- next priority stays direct-only calibration instead of basetrain / 70a: {judgements['next_priority_direct_only_calibration_not_basetrain']}",
            f"- adapter is still not the next first-priority move: {judgements['adapter_not_next_priority']}",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
