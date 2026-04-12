#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
except Exception:
    DEFAULT_DIRECT_POSE_LEG_BONES = (
        "thigh_r", "calf_r", "foot_r", "ball_r", "thigh_l", "calf_l", "foot_l", "ball_l",
    )
    STAGE6_3WAY_ARMCHAIN_BONES = (
        "clavicle_l", "upperarm_l", "RUpArmTwist_l_01", "RUpArmTwist_l_02", "lowerarm_l", "L_ForeTwist_01",
        "L_ForeTwist_02", "hand_l", "index_01_l", "middle_01_l", "ring_01_l", "pinky_01_l", "thumb_01_l",
        "clavicle_r", "upperarm_r", "RUpArmTwist_r_01", "RUpArmTwist_r_02", "lowerarm_r", "R_ForeTwist_01",
        "R_ForeTwist_02", "hand_r", "index_01_r", "middle_01_r", "ring_01_r", "pinky_01_r", "thumb_01_r",
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _finite(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _mean(values: Sequence[Any]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _quantile(values: Sequence[Any], q: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    idx = int(round(max(0.0, min(1.0, float(q))) * (len(vals) - 1)))
    return float(vals[idx])


def _vector_norm(values: Sequence[Any]) -> float:
    acc = 0.0
    used = False
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if not math.isfinite(x):
            continue
        used = True
        acc += x * x
    if not used:
        return float("nan")
    return float(math.sqrt(acc))


def _build_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        if cycle < int(cycle_gte):
            mask.append(False)
            continue
        if bool(drop_wrap) and bool(step.get("wrap_boundary_step", False)):
            mask.append(False)
            continue
        mask.append(True)
    return mask


def _pick_group_indices(names: Sequence[str], root_idx: int) -> Dict[str, List[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    idx_nonleg = [i for i in idx_all if i not in set(idx_leg)]
    idx_else = [i for i in idx_nonleg if i not in set(idx_arm)]
    return {
        "leg": idx_leg,
        "nonleg": idx_nonleg,
        "arm": idx_arm,
        "else": idx_else,
        "all_ex_root": idx_all,
    }


def _masked_direct_geolocal_rows(
    obj: Mapping[str, Any], *, cycle_gte: int, drop_wrap: bool
) -> Tuple[List[Tuple[int, int, int]], List[Mapping[str, Any]], List[List[float]], List[str], int]:
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise SystemExit("[FATAL] missing metrics_per_step/per_step_direct_geolocal_deg")
    names = per.get("bone_names", [])
    matrix = per.get("DirectGeoLocalDeg", [])
    if not isinstance(names, list) or not isinstance(matrix, list):
        raise SystemExit("[FATAL] invalid per_step_direct_geolocal_deg payload")
    try:
        root_idx = int(per.get("root_idx", 0) or 0)
    except Exception:
        root_idx = 0
    mask = _build_mask(steps, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap))
    keys: List[Tuple[int, int, int]] = []
    kept_steps: List[Mapping[str, Any]] = []
    kept_rows: List[List[float]] = []
    for step_i, keep in enumerate(mask):
        if not keep or step_i >= len(matrix):
            continue
        step = steps[step_i]
        row = matrix[step_i]
        if not isinstance(step, Mapping) or not isinstance(row, list):
            continue
        try:
            key = (
                int(step.get("cycle", 0) or 0),
                int(step.get("step_in_cycle", 0) or 0),
                int(step.get("step", step_i) or step_i),
            )
        except Exception:
            key = (0, 0, int(step_i))
        values: List[float] = []
        for value in row:
            try:
                x = float(value)
            except Exception:
                x = float("nan")
            values.append(x)
        keys.append(key)
        kept_steps.append(step)
        kept_rows.append(values)
    return keys, kept_steps, kept_rows, [str(x) for x in names], root_idx


def _align_cases(
    baseline_obj: Mapping[str, Any], candidate_obj: Mapping[str, Any], *, cycle_gte: int, drop_wrap: bool
) -> Dict[str, Any]:
    base_keys, base_steps, base_rows, base_names, base_root = _masked_direct_geolocal_rows(
        baseline_obj, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap)
    )
    cand_keys, cand_steps, cand_rows, cand_names, cand_root = _masked_direct_geolocal_rows(
        candidate_obj, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap)
    )
    if base_names != cand_names:
        raise SystemExit("[FATAL] baseline/candidate bone_names mismatch")
    if int(base_root) != int(cand_root):
        raise SystemExit("[FATAL] baseline/candidate root_idx mismatch")

    base_map = {key: (step, row) for key, step, row in zip(base_keys, base_steps, base_rows)}
    cand_map = {key: (step, row) for key, step, row in zip(cand_keys, cand_steps, cand_rows)}
    shared_keys = [key for key in cand_keys if key in base_map]
    if not shared_keys:
        raise SystemExit("[FATAL] no shared masked steps between baseline/candidate")

    return {
        "keys": shared_keys,
        "baseline_steps": [base_map[key][0] for key in shared_keys],
        "candidate_steps": [cand_map[key][0] for key in shared_keys],
        "baseline_rows": [base_map[key][1] for key in shared_keys],
        "candidate_rows": [cand_map[key][1] for key in shared_keys],
        "bone_names": base_names,
        "root_idx": int(base_root),
    }


def _step_group_stats(
    aligned: Mapping[str, Any], group_idx: Mapping[str, Sequence[int]]
) -> Dict[str, List[Dict[str, Any]]]:
    baseline_steps = aligned["baseline_steps"]
    baseline_rows = aligned["baseline_rows"]
    candidate_rows = aligned["candidate_rows"]
    step_values: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        group_name: {} for group_name in group_idx.keys()
    }
    for step, base_row, cand_row in zip(baseline_steps, baseline_rows, candidate_rows):
        step_in_cycle = int(step.get("step_in_cycle", 0) or 0)
        for group_name, indices in group_idx.items():
            bucket = step_values[group_name].setdefault(step_in_cycle, {"baseline": [], "candidate": []})
            for joint_i in indices:
                if joint_i >= len(base_row) or joint_i >= len(cand_row):
                    continue
                bucket["baseline"].append(base_row[joint_i])
                bucket["candidate"].append(cand_row[joint_i])

    out: Dict[str, List[Dict[str, Any]]] = {}
    for group_name, by_step in step_values.items():
        rows: List[Dict[str, Any]] = []
        for step_in_cycle in sorted(by_step.keys()):
            baseline = by_step[step_in_cycle]["baseline"]
            candidate = by_step[step_in_cycle]["candidate"]
            rows.append(
                {
                    "step_in_cycle": int(step_in_cycle),
                    "samples": int(len(_finite(candidate))),
                    "baseline_mean": _mean(baseline),
                    "candidate_mean": _mean(candidate),
                    "delta_mean": _mean(candidate) - _mean(baseline),
                    "baseline_p90": _quantile(baseline, 0.90),
                    "candidate_p90": _quantile(candidate, 0.90),
                    "delta_p90": _quantile(candidate, 0.90) - _quantile(baseline, 0.90),
                    "baseline_p95": _quantile(baseline, 0.95),
                    "candidate_p95": _quantile(candidate, 0.95),
                    "delta_p95": _quantile(candidate, 0.95) - _quantile(baseline, 0.95),
                }
            )
        out[group_name] = rows
    return out


def _arm_bone_p95(aligned: Mapping[str, Any], arm_indices: Sequence[int]) -> List[Dict[str, Any]]:
    names = aligned["bone_names"]
    baseline_rows = aligned["baseline_rows"]
    candidate_rows = aligned["candidate_rows"]
    rows: List[Dict[str, Any]] = []
    for joint_i in arm_indices:
        baseline = [row[joint_i] for row in baseline_rows if joint_i < len(row)]
        candidate = [row[joint_i] for row in candidate_rows if joint_i < len(row)]
        rows.append(
            {
                "bone": str(names[joint_i]),
                "baseline_p95": _quantile(baseline, 0.95),
                "candidate_p95": _quantile(candidate, 0.95),
                "delta_p95": _quantile(candidate, 0.95) - _quantile(baseline, 0.95),
            }
        )
    rows.sort(key=lambda row: (float(row["delta_p95"]), str(row["bone"])))
    return rows


def _probe_feature_norms(obj: Mapping[str, Any], *, features: Sequence[str]) -> Dict[str, Dict[str, float]]:
    probe = obj.get("direct_arm_probe", {})
    steps = probe.get("steps", []) if isinstance(probe, Mapping) else []
    stats: Dict[str, Dict[str, float]] = {}
    for feature_name in features:
        norms: List[float] = []
        for row in steps:
            if not isinstance(row, Mapping):
                continue
            feat = row.get("features", {})
            if not isinstance(feat, Mapping):
                continue
            values = feat.get(feature_name, None)
            if not isinstance(values, list):
                continue
            norm = _vector_norm(values)
            if math.isfinite(norm):
                norms.append(norm)
        stats[feature_name] = {
            "samples": int(len(norms)),
            "mean": _mean(norms),
            "p90": _quantile(norms, 0.90),
            "p95": _quantile(norms, 0.95),
            "max": _quantile(norms, 1.00),
        }
    return stats


def _probe_norm_delta(
    baseline_obj: Mapping[str, Any], candidate_obj: Mapping[str, Any], *, features: Sequence[str]
) -> List[Dict[str, Any]]:
    baseline = _probe_feature_norms(baseline_obj, features=features)
    candidate = _probe_feature_norms(candidate_obj, features=features)
    rows: List[Dict[str, Any]] = []
    for feature_name in features:
        b = baseline.get(feature_name, {})
        c = candidate.get(feature_name, {})
        rows.append(
            {
                "feature": str(feature_name),
                "baseline_samples": int(b.get("samples", 0) or 0),
                "candidate_samples": int(c.get("samples", 0) or 0),
                "baseline_mean": float(b.get("mean", float("nan"))),
                "candidate_mean": float(c.get("mean", float("nan"))),
                "delta_mean": float(c.get("mean", float("nan"))) - float(b.get("mean", float("nan"))),
                "baseline_p90": float(b.get("p90", float("nan"))),
                "candidate_p90": float(c.get("p90", float("nan"))),
                "delta_p90": float(c.get("p90", float("nan"))) - float(b.get("p90", float("nan"))),
                "baseline_p95": float(b.get("p95", float("nan"))),
                "candidate_p95": float(c.get("p95", float("nan"))),
                "delta_p95": float(c.get("p95", float("nan"))) - float(b.get("p95", float("nan"))),
                "baseline_max": float(b.get("max", float("nan"))),
                "candidate_max": float(c.get("max", float("nan"))),
                "delta_max": float(c.get("max", float("nan"))) - float(b.get("max", float("nan"))),
            }
        )
    return rows


def build_report(
    baseline_path: Path,
    candidate_path: Path,
    *,
    cycle_gte: int,
    drop_wrap: bool,
    topk: int,
    probe_features: Sequence[str],
) -> Dict[str, Any]:
    baseline_obj = _load_json(baseline_path)
    candidate_obj = _load_json(candidate_path)
    aligned = _align_cases(baseline_obj, candidate_obj, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap))
    group_idx = _pick_group_indices(aligned["bone_names"], int(aligned["root_idx"]))
    step_group = _step_group_stats(aligned, group_idx)
    arm_bones = _arm_bone_p95(aligned, group_idx["arm"])
    probe_norms = _probe_norm_delta(baseline_obj, candidate_obj, features=probe_features)

    topk = max(1, int(topk))
    top_step: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for group_name, rows in step_group.items():
        top_step[group_name] = {
            "best_p95": sorted(rows, key=lambda row: (float(row["delta_p95"]), int(row["step_in_cycle"])))[:topk],
            "worst_p95": sorted(rows, key=lambda row: (float(row["delta_p95"]), int(row["step_in_cycle"])), reverse=True)[:topk],
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "mask": {
            "cycle_gte": int(cycle_gte),
            "drop_wrap": bool(drop_wrap),
            "shared_steps": int(len(aligned["keys"])),
        },
        "groups": {
            "arm": [aligned["bone_names"][i] for i in group_idx["arm"]],
            "leg": [aligned["bone_names"][i] for i in group_idx["leg"]],
            "else": [aligned["bone_names"][i] for i in group_idx["else"]],
        },
        "step_in_cycle_group_delta": step_group,
        "top_step_in_cycle_delta": top_step,
        "arm_bone_p95_delta": arm_bones,
        "top_arm_bone_p95_delta": {
            "best": arm_bones[:topk],
            "worst": list(reversed(arm_bones[-topk:])),
        },
        "direct_arm_probe_norm_delta": probe_norms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--candidate", type=str, required=True)
    ap.add_argument("--cycle_gte", type=int, default=1)
    ap.add_argument("--drop_wrap", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument(
        "--probe_features",
        type=str,
        default="direct_in,trunk_hidden,proj_pre0,out_in,arm_out",
        help="comma-separated features from direct_arm_probe.steps[].features",
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probe_features = [str(x).strip() for x in str(args.probe_features).split(",") if str(x).strip()]
    payload = build_report(
        baseline_path,
        candidate_path,
        cycle_gte=int(args.cycle_gte),
        drop_wrap=bool(args.drop_wrap),
        topk=int(args.topk),
        probe_features=probe_features,
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"[followup] wrote {out_path}")
    print("[followup] top arm step_in_cycle p95 improvements:")
    for row in payload["top_step_in_cycle_delta"]["arm"]["best_p95"]:
        print(
            f"  step={int(row['step_in_cycle'])} "
            f"base_p95={float(row['baseline_p95']):.6f} cand_p95={float(row['candidate_p95']):.6f} "
            f"delta={float(row['delta_p95']):+.6f}"
        )
    print("[followup] top arm-bone p95 improvements:")
    for row in payload["top_arm_bone_p95_delta"]["best"]:
        print(
            f"  bone={str(row['bone'])} "
            f"base_p95={float(row['baseline_p95']):.6f} cand_p95={float(row['candidate_p95']):.6f} "
            f"delta={float(row['delta_p95']):+.6f}"
        )
    print("[followup] probe norm delta:")
    for row in payload["direct_arm_probe_norm_delta"]:
        print(
            f"  feature={str(row['feature'])} "
            f"delta_mean={float(row['delta_mean']):+.6f} "
            f"delta_p95={float(row['delta_p95']):+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
