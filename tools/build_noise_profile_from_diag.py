"""
Utility script to build an empirical rotation-noise profile from freerun
diagnostic .pt files.

Usage (from project root):
    python -m tools.build_noise_profile_from_diag \\
        --glob 'debug_output/freerun_diag_ep002_*.pt' \\
        --out  'debug_output/noise_profile_empirical.json'

This reads the `metrics` dict in each .pt, looks for:
    - 'GeoDegCurve' (per-step geodesic error, degrees)
    - 'KeyBoneDetails' / 'FreeRun/KeyBoneDetails' (per-body-part stats)

and writes a JSON payload of the form:
{
  "global": {
    "num_samples": ...,
    "mean_deg": ...,
    "p50_deg": ...,
    "p80_deg": ...,
    "p95_deg": ...
  },
  "buckets": [
    {"prob": 0.40, "min_deg": 0.0, "max_deg":  p50},
    {"prob": 0.35, "min_deg": p50, "max_deg": p80},
    {"prob": 0.25, "min_deg": p80, "max_deg": p95}
  ],
  "keybone": {
    "pelvis": {...},
    "upperarm_l": {...},
    ...
  }
}

You can then copy `buckets` into `input_noise_deg_mix` or
`contraction_noise_deg_mix` in the training config, instead of the
hand-written ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import torch


def _gather_metrics(paths: List[Path]) -> Dict[str, Any]:
    """Aggregate GeoDeg samples and keybone stats from a list of diag .pt files."""
    geo_samples: List[float] = []
    keybone_stats: Dict[str, List[float]] = {}

    for p in paths:
        try:
            payload = torch.load(p, map_location="cpu")
        except Exception:
            continue
        metrics = payload.get("metrics") or {}
        if not metrics:
            continue

        # 1) Global per-step GeoDegCurve (already aggregated over joints).
        curve = metrics.get("GeoDegCurve") or metrics.get("FreeRun/GeoDegCurve")
        if curve is not None:
            try:
                vals = [float(x) for x in curve]
                geo_samples.extend(vals)
            except Exception:
                pass

        # 2) Per-bone details (already pre-aggregated at trainer side).
        kb = metrics.get("FreeRun/KeyBoneDetails") or metrics.get("KeyBoneDetails") or {}
        if isinstance(kb, dict):
            for name, entry in kb.items():
                if not isinstance(entry, dict):
                    continue
                val = entry.get("GeoDeg") or entry.get("GeoLocalDeg")
                if val is None:
                    continue
                try:
                    v = float(val)
                except Exception:
                    continue
                keybone_stats.setdefault(name, []).append(v)

    return {"geo_samples": geo_samples, "keybone_stats": keybone_stats}


def _quantiles(xs: List[float], qs: List[float]) -> List[float]:
    if not xs:
        return [0.0 for _ in qs]
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    out: List[float] = []
    for q in qs:
        if n == 1:
            out.append(xs_sorted[0])
            continue
        q = min(max(q, 0.0), 1.0)
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        val = xs_sorted[lo] * (1.0 - frac) + xs_sorted[hi] * frac
        out.append(float(val))
    return out


def build_profile(glob_pattern: str, out_path: Path) -> None:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise SystemExit(f"no diag files matched pattern: {glob_pattern}")

    agg = _gather_metrics(paths)
    geo_samples: List[float] = agg["geo_samples"]
    keybone_stats: Dict[str, List[float]] = agg["keybone_stats"]

    if not geo_samples:
        raise SystemExit("no GeoDegCurve samples found in diag files.")

    mean_deg = float(sum(geo_samples) / len(geo_samples))
    p50, p80, p95 = _quantiles(geo_samples, [0.50, 0.80, 0.95])

    # Build three buckets from empirical quantiles.
    buckets = [
        {"prob": 0.40, "min_deg": 0.0, "max_deg": p50},
        {"prob": 0.35, "min_deg": p50, "max_deg": p80},
        {"prob": 0.25, "min_deg": p80, "max_deg": p95},
    ]

    keybone_summary: Dict[str, Dict[str, float]] = {}
    for name, vals in keybone_stats.items():
        if not vals:
            continue
        m = float(sum(vals) / len(vals))
        q50, q80, q95 = _quantiles(vals, [0.50, 0.80, 0.95])
        keybone_summary[name] = {
            "num_samples": float(len(vals)),
            "mean_deg": m,
            "p50_deg": q50,
            "p80_deg": q80,
            "p95_deg": q95,
        }

    payload = {
        "source_glob": glob_pattern,
        "global": {
            "num_samples": float(len(geo_samples)),
            "mean_deg": mean_deg,
            "p50_deg": p50,
            "p80_deg": p80,
            "p95_deg": p95,
        },
        "buckets": buckets,
        "keybone": keybone_summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[NoiseProfile] wrote empirical profile to {out_path}")
    print("[NoiseProfile] suggested input_noise_deg_mix buckets:")
    for b in buckets:
        print(
            f"  - prob: {b['prob']:.2f}, "
            f"min_deg: {b['min_deg']:.2f}, max_deg: {b['max_deg']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an empirical rotation-noise profile from freerun diag .pt files."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="debug_output/freerun_diag_ep002_*.pt",
        help="Glob pattern for diag .pt files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug_output/noise_profile_empirical.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()
    build_profile(args.glob, Path(args.out))


if __name__ == "__main__":
    main()

