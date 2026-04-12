#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


RUN_DATE = "20260328"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_ep014center_70a_vs_baseline_sic_profile_{RUN_DATE}"

BASELINE_70A_EVAL = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "eval_model_source" / "70a" / "Walk_F_freerun_cycles.json"
BASELINE_REPLACE_EVAL = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317" / "eval_model_source" / "new70b_replace_lowdrift" / "Walk_F_freerun_cycles.json"
NEW_70A_EVAL = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "eval_model_source" / "lr3e4" / "Walk_F_freerun_cycles.json"
NEW_REPLACE_EVAL = ROOT / "debug_output" / "_tmp_ep014center_replace_lowlr_sweep_20260328" / "eval_model_source" / "lr5e5" / "Walk_F_freerun_cycles.json"

GROUPS: Tuple[str, ...] = ("all_ex_root", "leg", "nonleg", "arm", "else")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def fmt(x: Any, digits: int = 6) -> str:
    value = safe_float(x)
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


def _per_sic_group_profile(path: Path, *, cycle_gte: int, drop_wrap: bool) -> Dict[str, Any]:
    obj = load_json(path)
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise SystemExit(f"[FATAL] invalid freerun json: {path}")
    names = [str(x) for x in per.get("bone_names", [])]
    mat = per.get("DirectGeoLocalDeg", [])
    if not names or not isinstance(mat, list):
        raise SystemExit(f"[FATAL] missing direct geolocal matrix in {path}")
    root_idx = int(per.get("root_idx", 0) or 0)
    cycle_len = int(obj.get("cycle_len", 0) or 0)
    if cycle_len <= 0:
        raise SystemExit(f"[FATAL] invalid cycle_len in {path}")
    group_idx = _pick_group_indices(names, root_idx)
    grouped: Dict[str, Dict[int, List[float]]] = {
        group: {sic: [] for sic in range(cycle_len)}
        for group in group_idx
    }
    for step_i, step in enumerate(steps):
        cycle = int(step.get("cycle", 0) or 0)
        if cycle < int(cycle_gte):
            continue
        if bool(drop_wrap) and bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", 0) or 0)
        if step_i >= len(mat):
            continue
        row = mat[step_i]
        if not isinstance(row, list):
            continue
        for group_name, indices in group_idx.items():
            values: List[float] = []
            for joint_i in indices:
                if joint_i >= len(row):
                    continue
                value = safe_float(row[joint_i])
                if math.isfinite(value):
                    values.append(value)
            if values:
                grouped[group_name][sic].append(float(np.mean(np.asarray(values, dtype=np.float64))))

    profiles: Dict[str, List[float]] = {}
    for group_name, sic_map in grouped.items():
        arr = [
            float(np.mean(np.asarray(sic_map[sic], dtype=np.float64))) if sic_map[sic] else float("nan")
            for sic in range(cycle_len)
        ]
        profiles[group_name] = arr

    return {
        "source": str(path),
        "clip": obj.get("clip"),
        "cycle_len": int(cycle_len),
        "mask": {"cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
        "bone_names": names,
        "root_idx": int(root_idx),
        "group_indices": group_idx,
        "profiles": profiles,
    }


def _profile_stats(arr: Sequence[float], *, topk: int = 8) -> Dict[str, Any]:
    raw = np.asarray(list(arr), dtype=np.float64)
    valid_mask = np.isfinite(raw)
    valid_idx = np.nonzero(valid_mask)[0]
    vals = raw[valid_mask]
    if vals.size <= 0:
        return {}
    median = float(np.median(vals))
    excess = np.maximum(vals - median, 0.0)
    order = np.argsort(-excess)
    topk_use = min(int(topk), int(vals.size))
    top_idx = [int(valid_idx[i]) for i in order[:topk_use]]
    top_share = float(excess[order[:topk_use]].sum() / excess.sum()) if float(excess.sum()) > 0.0 else float("nan")
    max_pos = int(valid_idx[int(np.argmax(vals))])
    min_pos = int(valid_idx[int(np.argmin(vals))])
    top_mask = np.zeros_like(vals, dtype=bool)
    top_mask[order[:topk_use]] = True
    rest_mask = ~top_mask
    return {
        "mean": float(np.mean(vals)),
        "median": median,
        "min": float(np.min(vals)),
        "min_sic": min_pos,
        "max": float(np.max(vals)),
        "max_sic": max_pos,
        "sic_std": float(np.std(vals)),
        "max_minus_mean": float(np.max(vals) - np.mean(vals)),
        "max_over_mean": float(np.max(vals) / np.mean(vals)) if float(np.mean(vals)) > 0.0 else float("nan"),
        "top_sics": top_idx,
        "top_excess_share_over_median": top_share,
        "top_mean": float(np.mean(vals[top_mask])) if np.any(top_mask) else float("nan"),
        "rest_mean": float(np.mean(vals[rest_mask])) if np.any(rest_mask) else float("nan"),
        "top_minus_rest": float(np.mean(vals[top_mask]) - np.mean(vals[rest_mask])) if np.any(top_mask) and np.any(rest_mask) else float("nan"),
    }


def _delta_summary(before: Sequence[float], after: Sequence[float], *, topk: int = 8) -> Dict[str, Any]:
    before_arr = np.asarray(list(before), dtype=np.float64)
    after_arr = np.asarray(list(after), dtype=np.float64)
    valid = np.isfinite(before_arr) & np.isfinite(after_arr)
    before_vals = before_arr[valid]
    after_vals = after_arr[valid]
    delta = after_vals - before_vals
    valid_idx = np.nonzero(valid)[0]
    order = np.argsort(delta)
    topk_use = min(int(topk), int(delta.size))
    own_top_before = np.argsort(-before_vals)[:topk_use]
    return {
        "mean_delta": float(np.mean(delta)),
        "best_improve_sics": [
            {"sic": int(valid_idx[i]), "delta": float(delta[i])}
            for i in order[:topk_use]
        ],
        "worst_regress_sics": [
            {"sic": int(valid_idx[i]), "delta": float(delta[i])}
            for i in order[-topk_use:][::-1]
        ],
        "mean_delta_on_own_top_before": float(np.mean(delta[own_top_before])) if topk_use > 0 else float("nan"),
        "mean_delta_on_rest": float(np.mean(np.delete(delta, own_top_before))) if delta.size > topk_use else float("nan"),
    }


def _compare_profile_stats(profile_a: Sequence[float], profile_b: Sequence[float]) -> Dict[str, Any]:
    arr_a = np.asarray(list(profile_a), dtype=np.float64)
    arr_b = np.asarray(list(profile_b), dtype=np.float64)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    vals_a = arr_a[valid]
    vals_b = arr_b[valid]
    idx = np.nonzero(valid)[0]
    delta = vals_a - vals_b
    order = np.argsort(-np.abs(delta))
    return {
        "mean_delta": float(np.mean(delta)),
        "largest_abs_delta_sics": [
            {"sic": int(idx[i]), "delta": float(delta[i]), "a": float(vals_a[i]), "b": float(vals_b[i])}
            for i in order[:8]
        ],
    }


def build_summary() -> Dict[str, Any]:
    baseline_70a = _per_sic_group_profile(BASELINE_70A_EVAL, cycle_gte=1, drop_wrap=True)
    baseline_replace = _per_sic_group_profile(BASELINE_REPLACE_EVAL, cycle_gte=1, drop_wrap=True)
    new_70a = _per_sic_group_profile(NEW_70A_EVAL, cycle_gte=1, drop_wrap=True)
    new_replace = _per_sic_group_profile(NEW_REPLACE_EVAL, cycle_gte=1, drop_wrap=True)

    summary: Dict[str, Any] = {
        "run_date": RUN_DATE,
        "policy": {
            "mask": {"cycle_gte": 1, "drop_wrap": True},
            "question": "Does baseline 70a have a larger/more removable phase-locked error structure than new 70a at replace entry?",
        },
        "artifacts": {
            "baseline_70a_eval": str(BASELINE_70A_EVAL),
            "baseline_replace_eval": str(BASELINE_REPLACE_EVAL),
            "new_70a_eval": str(NEW_70A_EVAL),
            "new_replace_eval": str(NEW_REPLACE_EVAL),
        },
        "profiles": {
            "baseline_70a": baseline_70a,
            "baseline_replace": baseline_replace,
            "new_70a": new_70a,
            "new_replace": new_replace,
        },
        "group_summaries": {},
    }

    for group_name in GROUPS:
        base_before = baseline_70a["profiles"][group_name]
        base_after = baseline_replace["profiles"][group_name]
        new_before = new_70a["profiles"][group_name]
        new_after = new_replace["profiles"][group_name]
        summary["group_summaries"][group_name] = {
            "baseline_70a": _profile_stats(base_before),
            "new_70a": _profile_stats(new_before),
            "baseline_70a_minus_new_70a": _compare_profile_stats(base_before, new_before),
            "baseline_replace_delta": _delta_summary(base_before, base_after),
            "new_replace_delta": _delta_summary(new_before, new_after),
        }
    return summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# ep014center 70a vs baseline SIC profile",
        "",
        "- mask: `cycle>=1`, `drop_wrap=true`",
        "- no training; uses existing `70a` / `replace` freerun eval JSONs",
        "",
        "## Entry Profile Stats",
        "",
        "| group | baseline mean | baseline max@sic | baseline top-rest | new mean | new max@sic | new top-rest |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name in GROUPS:
        row = summary["group_summaries"][group_name]
        base = row["baseline_70a"]
        new = row["new_70a"]
        lines.append(
            f"| {group_name} | {fmt(base['mean'])} | {fmt(base['max'])}@{int(base['max_sic'])} | {fmt(base['top_minus_rest'])} | "
            f"{fmt(new['mean'])} | {fmt(new['max'])}@{int(new['max_sic'])} | {fmt(new['top_minus_rest'])} |"
        )
    lines.extend(
        [
            "",
            "## Replace Delta On Own Top-8 SICs",
            "",
            "| group | baseline top8 d | baseline rest d | new top8 d | new rest d |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_name in GROUPS:
        row = summary["group_summaries"][group_name]
        b = row["baseline_replace_delta"]
        n = row["new_replace_delta"]
        lines.append(
            f"| {group_name} | {fmt(b['mean_delta_on_own_top_before'])} | {fmt(b['mean_delta_on_rest'])} | "
            f"{fmt(n['mean_delta_on_own_top_before'])} | {fmt(n['mean_delta_on_rest'])} |"
        )
    lines.extend(["", "## Top SIC Bins", ""])
    for group_name in ("all_ex_root", "leg", "nonleg", "arm"):
        row = summary["group_summaries"][group_name]
        lines.append(f"### {group_name}")
        lines.append(f"- baseline top SICs: {row['baseline_70a']['top_sics']}")
        lines.append(f"- new top SICs: {row['new_70a']['top_sics']}")
        lines.append(
            f"- baseline replace best-improve SICs: {[item['sic'] for item in row['baseline_replace_delta']['best_improve_sics'][:8]]}"
        )
        lines.append(
            f"- new replace best-improve SICs: {[item['sic'] for item in row['new_replace_delta']['best_improve_sics'][:8]]}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    required = [BASELINE_70A_EVAL, BASELINE_REPLACE_EVAL, NEW_70A_EVAL, NEW_REPLACE_EVAL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("=== build SIC profile summary ===")
    summary = build_summary()
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
