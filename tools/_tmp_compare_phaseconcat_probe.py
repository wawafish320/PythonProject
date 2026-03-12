#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(obj):
    per = obj["per_step_direct_geolocal_deg"]
    names = list(per["bone_names"])
    root = int(per.get("root_idx", 0))
    mat = np.asarray(per["DirectGeoLocalDeg"], dtype=np.float64)
    steps = obj["metrics_per_step"]
    cyc = np.asarray([int(s.get("cycle", 0)) if isinstance(s, dict) else 0 for s in steps], dtype=np.int64)
    wrap = np.asarray([bool(s.get("wrap_boundary_step", False)) if isinstance(s, dict) else False for s in steps], dtype=bool)
    sic = np.asarray(
        [int(s.get("step_in_cycle", s.get("sic", i))) if isinstance(s, dict) else i for i, s in enumerate(steps)],
        dtype=np.int64,
    )

    mask = (cyc >= 1) & (~wrap)
    idx_all = [i for i in range(len(names)) if i != root]
    foot_idx = [names.index("foot_l"), names.index("ball_l")]
    calf_idx = [names.index("calf_r")]

    def mean(mask_step, idx):
        if mask_step.sum() == 0:
            return float("nan")
        sub = mat[mask_step][:, idx]
        v = sub[np.isfinite(sub)]
        return float(v.mean()) if v.size else float("nan")

    return {
        "global": mean(mask, idx_all),
        "sic12_15_footl_balll": mean(mask & np.isin(sic, [12, 13, 14, 15]), foot_idx),
        "calf_r_global": mean(mask, calf_idx),
        "calf_r_sic2_4": mean(mask & np.isin(sic, [2, 3, 4]), calf_idx),
    }


old_path = Path("debug_output/verify_stage7_rerun_from_stage5_20260225/stagewise_old_new/old/s70R/Walk_F_freerun_cycles.json")
new_path = Path("debug_output/verify_stage7_rerun_from_stage5_20260225/stagewise_old_new/new/s70R/Walk_F_freerun_cycles.json")
probe_path = Path(
    "debug_output/verify_stage7_rerun_from_stage5_20260225_phaseconcat_probe/s70R/Walk_F_freerun_cycles.json/Walk_F_freerun_cycles.json"
)
new70b_path = Path("debug_output/verify_stage7_rerun_from_stage5_20260225/stagewise_old_new/new/s70b/Walk_F_freerun_cycles.json")

old = metrics(load_json(old_path))
new = metrics(load_json(new_path))
probe = metrics(load_json(probe_path))
new70b = metrics(load_json(new70b_path))

out = Path("debug_output/verify_stage7_rerun_from_stage5_20260225_phaseconcat_probe/phaseconcat_probe_summary.md")
lines = []
lines.append("# Phase-concat probe at 7.0R (start from new 7.0b)")
lines.append("")
lines.append("| Metric | old 7.0R | new 7.0R (replace_contacts) | probe 7.0R (concat) | new 7.0b |")
lines.append("|---|---:|---:|---:|---:|")
for k, lbl in [
    ("global", "Global"),
    ("sic12_15_footl_balll", "SIC12-15 foot_l/ball_l"),
    ("calf_r_global", "calf_r global"),
    ("calf_r_sic2_4", "calf_r SIC2-4"),
]:
    lines.append(f"| {lbl} | {old[k]:.6f} | {new[k]:.6f} | {probe[k]:.6f} | {new70b[k]:.6f} |")

lines.append("")
for k, lbl in [
    ("global", "Global"),
    ("sic12_15_footl_balll", "SIC12-15 foot_l/ball_l"),
    ("calf_r_global", "calf_r global"),
    ("calf_r_sic2_4", "calf_r SIC2-4"),
]:
    lines.append(
        f"- {lbl}: Δ(new-old)={new[k]-old[k]:+.6f}, "
        f"Δ(probe-old)={probe[k]-old[k]:+.6f}, "
        f"Δ(probe-new)={probe[k]-new[k]:+.6f}, "
        f"Δ(probe-70b)={probe[k]-new70b[k]:+.6f}"
    )

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[OK] wrote", out)
