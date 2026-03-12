#!/usr/bin/env python3
"""
Diagnose lowerarm regression source via checkpoint-swap A/B:
  - H1: old trunk + B2 non-leg readout rows
  - H2: B2 trunk + old non-leg readout rows

Reads four freerun JSON files and writes:
  - lowerarm_source_diag_metrics.csv
  - lowerarm_source_diag_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np


LEG8 = {"thigh_r", "calf_r", "foot_r", "ball_r", "thigh_l", "calf_l", "foot_l", "ball_l"}


def _load_metrics(path: Path) -> Dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    per = d["per_step_direct_geolocal_deg"]
    names = list(per["bone_names"])
    root = int(per.get("root_idx", 0) or 0)
    mat = np.asarray(per["DirectGeoLocalDeg"], dtype=np.float64)
    steps = d["metrics_per_step"]

    mask = np.zeros((len(steps),), dtype=bool)
    for i, st in enumerate(steps):
        cyc = int(st.get("cycle", 0) or 0)
        if cyc < 1:
            continue
        if bool(st.get("wrap_boundary_step", False)):
            continue
        mask[i] = True

    idx_all = [i for i in range(len(names)) if i != root]
    idx_nonleg = [i for i, n in enumerate(names) if i != root and n not in LEG8]

    def stats_for_idx(j: int) -> Dict[str, float]:
        v = mat[mask, j]
        v = v[np.isfinite(v)]
        return {
            "mean": float(np.mean(v)),
            "p95": float(np.quantile(v, 0.95)),
            "max": float(np.max(v)),
        }

    def mean_for_idxs(idxs) -> float:
        x = mat[mask][:, idxs]
        v = x[np.isfinite(x)]
        return float(np.mean(v))

    out: Dict[str, float] = {
        "global_mean": mean_for_idxs(idx_all),
        "nonleg_mean": mean_for_idxs(idx_nonleg),
    }
    for bone in ("lowerarm_l", "lowerarm_r", "upperarm_l", "thumb_01_l", "head"):
        if bone in names:
            st = stats_for_idx(names.index(bone))
            out[f"{bone}_mean"] = st["mean"]
            out[f"{bone}_p95"] = st["p95"]
            out[f"{bone}_max"] = st["max"]
    out["lowerarm_pair_mean"] = 0.5 * (out["lowerarm_l_mean"] + out["lowerarm_r_mean"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-json", type=str, required=True)
    ap.add_argument("--b2-json", type=str, required=True)
    ap.add_argument("--h1-json", type=str, required=True)
    ap.add_argument("--h2-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    cases = {
        "OLD_nosplit": _load_metrics(Path(args.old_json)),
        "B2_split": _load_metrics(Path(args.b2_json)),
        "H1_oldTrunk_b2Nonleg": _load_metrics(Path(args.h1_json)),
        "H2_b2Trunk_oldNonleg": _load_metrics(Path(args.h2_json)),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "case",
        "global_mean",
        "nonleg_mean",
        "lowerarm_l_mean",
        "lowerarm_r_mean",
        "lowerarm_pair_mean",
        "upperarm_l_mean",
        "thumb_01_l_mean",
        "head_mean",
        "lowerarm_l_p95",
        "lowerarm_r_p95",
        "lowerarm_l_max",
        "lowerarm_r_max",
    ]
    csv_path = out_dir / "lowerarm_source_diag_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in ("OLD_nosplit", "B2_split", "H1_oldTrunk_b2Nonleg", "H2_b2Trunk_oldNonleg"):
            r = cases[k]
            w.writerow([k] + [f"{r[c]:.6f}" for c in cols[1:]])

    old = cases["OLD_nosplit"]
    b2 = cases["B2_split"]
    h1 = cases["H1_oldTrunk_b2Nonleg"]
    h2 = cases["H2_b2Trunk_oldNonleg"]

    a = abs(h1["lowerarm_pair_mean"] - b2["lowerarm_pair_mean"])
    b = abs(h2["lowerarm_pair_mean"] - b2["lowerarm_pair_mean"])
    if b > a * 1.5:
        verdict = "out_nonleg-dominant"
    elif a > b * 1.5:
        verdict = "trunk-dominant"
    else:
        verdict = "mixed"

    lines = []
    lines.append("# Lowerarm source diagnosis: trunk vs out_nonleg")
    lines.append("")
    lines.append("- Mask: cycle>=1 + drop_wrap + exclude root")
    lines.append("- H1 = old trunk + B2 non-leg readout rows")
    lines.append("- H2 = B2 trunk + old non-leg readout rows")
    lines.append("")
    lines.append("## Core metrics")
    lines.append("")
    lines.append("| case | global_mean | nonleg_mean | lowerarm_l | lowerarm_r | lowerarm_pair |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in ("OLD_nosplit", "B2_split", "H1_oldTrunk_b2Nonleg", "H2_b2Trunk_oldNonleg"):
        r = cases[k]
        lines.append(
            f"| {k} | {r['global_mean']:.6f} | {r['nonleg_mean']:.6f} | "
            f"{r['lowerarm_l_mean']:.6f} | {r['lowerarm_r_mean']:.6f} | {r['lowerarm_pair_mean']:.6f} |"
        )
    lines.append("")
    lines.append("## Swap effects (focus: lowerarm_pair_mean)")
    lines.append("")
    lines.append(f"- OLD -> H1 (only nonleg head from B2): {h1['lowerarm_pair_mean'] - old['lowerarm_pair_mean']:+.6f}")
    lines.append(f"- OLD -> H2 (only trunk from B2): {h2['lowerarm_pair_mean'] - old['lowerarm_pair_mean']:+.6f}")
    lines.append(f"- B2 -> H1 (replace trunk with OLD): {h1['lowerarm_pair_mean'] - b2['lowerarm_pair_mean']:+.6f}")
    lines.append(f"- B2 -> H2 (replace nonleg head with OLD): {h2['lowerarm_pair_mean'] - b2['lowerarm_pair_mean']:+.6f}")
    lines.append("")
    lines.append(f"- Attribution heuristic: **{verdict}** (|B2->H1|={a:.6f}, |B2->H2|={b:.6f})")

    md_path = out_dir / "lowerarm_source_diag_summary.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(csv_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
