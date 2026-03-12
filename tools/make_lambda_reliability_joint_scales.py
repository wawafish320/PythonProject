#!/usr/bin/env python3
"""
Generate per-joint warmup scales for `lambda_reliability_warmup_joint_scales`.

This script reads an existing freerun_cycles diagnostic JSON that contains:
  - per_joint_geolocal.GeoLocalDegEarlyMean
  - per_joint_geolocal.DirectGeoLocalDegEarlyMean

Then it builds a "reliability-based" scale per joint:
  scale_j = clamp( (inc_early_j / direct_early_j) ** alpha, min_scale, max_scale )

Intuition:
  - If direct is better early (direct_early < inc_early) => scale > 1 (warmup faster).
  - If direct is worse early => scale < 1 (warmup slower).

The output JSON is compatible with:
  --lambda_reliability_warmup_joint_scales /path/to/output.json
because `run_freerun_cycles.py` accepts either a list or a dict with key "scales".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _topk_by_abs_delta(
    names: List[str],
    inc_early: List[float],
    direct_early: List[float],
    scales: List[float],
    *,
    k: int = 8,
) -> List[Tuple[str, int, float, float, float]]:
    rows: List[Tuple[str, int, float, float, float]] = []
    for j, (n, a, b, s) in enumerate(zip(names, inc_early, direct_early, scales)):
        rows.append((str(n), int(j), float(a), float(b), float(s)))
    rows.sort(key=lambda r: abs(r[2] - r[3]), reverse=True)
    return rows[: max(0, int(k))]


def build_scales_from_diag(
    diag: Dict[str, Any],
    *,
    alpha: float,
    min_scale: float,
    max_scale: float,
    eps: float,
    normalize_mean_to_1: bool,
) -> Dict[str, Any]:
    per = diag.get("per_joint_geolocal")
    if not isinstance(per, dict):
        raise KeyError("Input JSON missing 'per_joint_geolocal' dict.")

    names = per.get("bone_names")
    inc_early = per.get("GeoLocalDegEarlyMean")
    direct_early = per.get("DirectGeoLocalDegEarlyMean")
    root_idx = per.get("root_idx", 0)
    analysis_steps = per.get("analysis_steps", None)

    if not isinstance(names, list) or not names:
        raise KeyError("per_joint_geolocal missing 'bone_names' list.")
    if not isinstance(inc_early, list) or not inc_early:
        raise KeyError("per_joint_geolocal missing 'GeoLocalDegEarlyMean' list.")
    if not isinstance(direct_early, list) or not direct_early:
        raise KeyError("per_joint_geolocal missing 'DirectGeoLocalDegEarlyMean' list.")
    if len(inc_early) != len(direct_early) or len(names) != len(inc_early):
        raise ValueError(
            f"Length mismatch: bone_names={len(names)}, inc_early={len(inc_early)}, direct_early={len(direct_early)}"
        )

    J = len(names)
    root_idx = int(root_idx) if root_idx is not None else 0
    root_idx = max(0, min(J - 1, root_idx))

    scales: List[float] = []
    for j in range(J):
        if j == root_idx:
            scales.append(1.0)
            continue
        a = float(inc_early[j])
        b = float(direct_early[j])
        ratio = (a + eps) / (b + eps)
        s = ratio ** float(alpha)
        scales.append(_clamp(float(s), float(min_scale), float(max_scale)))

    if normalize_mean_to_1:
        vals = [scales[j] for j in range(J) if j != root_idx]
        mean = sum(vals) / max(1, len(vals))
        if mean > 1e-12:
            scales = [scales[j] if j == root_idx else _clamp(scales[j] / mean, min_scale, max_scale) for j in range(J)]
            # Keep the hard safety guarantee: if direct_early > inc_early, never accelerate (scale>1).
            for j in range(J):
                if j == root_idx:
                    continue
                if float(direct_early[j]) > float(inc_early[j]):
                    scales[j] = min(scales[j], 1.0)

    # Summary for quick inspection.
    num_gt_1 = sum(1 for j in range(J) if j != root_idx and scales[j] > 1.0 + 1e-8)
    num_lt_1 = sum(1 for j in range(J) if j != root_idx and scales[j] < 1.0 - 1e-8)
    mean_ex_root = (
        sum(scales[j] for j in range(J) if j != root_idx) / max(1, (J - 1))
        if J > 1
        else 1.0
    )
    min_ex_root = min((scales[j] for j in range(J) if j != root_idx), default=1.0)
    max_ex_root = max((scales[j] for j in range(J) if j != root_idx), default=1.0)

    top_abs = _topk_by_abs_delta(names, inc_early, direct_early, scales, k=10)

    return {
        "scales": [float(x) for x in scales],
        "meta": {
            "method": "direct_vs_inc_geolocal_early_ratio",
            "alpha": float(alpha),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "eps": float(eps),
            "normalize_mean_to_1": bool(normalize_mean_to_1),
            "analysis_steps": int(analysis_steps) if analysis_steps is not None else None,
            "root_idx": int(root_idx),
            "stats": {
                "J": int(J),
                "mean_ex_root": float(mean_ex_root),
                "min_ex_root": float(min_ex_root),
                "max_ex_root": float(max_ex_root),
                "num_gt_1": int(num_gt_1),
                "num_lt_1": int(num_lt_1),
            },
            "top_abs_delta_examples": [
                {"name": n, "idx": j, "inc_early": a, "direct_early": b, "scale": s} for (n, j, a, b, s) in top_abs
            ],
        },
        "bone_names": [str(x) for x in names],
        "src_diag": {
            "clip": diag.get("clip"),
            "teacher_json": diag.get("teacher_json"),
            "model": diag.get("model"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reliability-based per-joint warmup scales from a freerun_cycles JSON.")
    ap.add_argument("--in_diag", type=str, required=True, help="Input *freerun_cycles.json (must contain per_joint_geolocal).")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path (dict with key 'scales').")
    ap.add_argument("--alpha", type=float, default=0.5, help="Exponent for (inc/direct)^alpha (sqrt by default).")
    ap.add_argument("--min_scale", type=float, default=0.25, help="Clamp lower bound.")
    ap.add_argument("--max_scale", type=float, default=4.0, help="Clamp upper bound.")
    ap.add_argument("--eps", type=float, default=1e-8, help="Epsilon to avoid divide-by-zero.")
    ap.add_argument(
        "--normalize_mean_to_1",
        action="store_true",
        help="Normalize non-root scales to mean=1, then enforce (direct>inc)=>scale<=1 safety.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_diag).expanduser()
    out_path = Path(args.out).expanduser()

    diag = _load_json(in_path)
    if not isinstance(diag, dict):
        raise TypeError(f"Expected dict JSON, got {type(diag)}")

    out = build_scales_from_diag(
        diag,
        alpha=float(args.alpha),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        eps=float(args.eps),
        normalize_mean_to_1=bool(args.normalize_mean_to_1),
    )
    _save_json(out_path, out)

    stats = out.get("meta", {}).get("stats", {})
    clip = out.get("src_diag", {}).get("clip", None)
    print(f"[OK] wrote {out_path} (clip={clip}, J={stats.get('J')}, mean_ex_root={stats.get('mean_ex_root'):.4f})")
    print(f"     num_gt_1={stats.get('num_gt_1')} num_lt_1={stats.get('num_lt_1')} min={stats.get('min_ex_root'):.4f} max={stats.get('max_ex_root'):.4f}")


if __name__ == "__main__":
    main()
