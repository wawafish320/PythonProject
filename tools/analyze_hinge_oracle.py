#!/usr/bin/env python3
"""
Oracle upper bound for 1D joint-local hinge correction.

Expected input JSON: produced by
  python -m train.validate.run_freerun_cycles --export_keybone_omega --export_keybone_omega_series

We rely on:
  keybone_omega.series.branches.{inc|direct|blend}.omega_deg_xyz[bone][t] = [wx, wy, wz]

where omega is the *standard* axis-angle log vector (deg) for:
  R_err = R_pred^T @ R_gt

Given a hinge axis a (joint-local) and range delta ∈ [-max, max], we solve per-step:
  delta* = argmin angle( exp(-delta * a) @ R_err )

Interpretation:
  - If oracle hinge can drive large errors (e.g. ~22°) down to ~few degrees, the 1D hinge mechanism is expressive
    enough; the bottleneck is predicting delta (features/conditioning/supervision/optimization).
  - If oracle hinge saturates around ~15° (similar to learned hinge), 1D hinge is likely insufficient (need higher-DOF residual).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so add project root for `import train.*`.
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.geometry import so3_exp_map  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _mean_std(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in xs]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _fmt_float(x: Optional[float], *, prec: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return "NA"


def _fmt_phase_mean(ph: Optional[Dict[str, Any]], *, prec: int = 2) -> str:
    if not isinstance(ph, dict):
        return "NA"
    p = ph.get("phase", None)
    v = ph.get("mean", ph.get("mean_deg", None))
    try:
        return f"{int(p)}:{float(v):.{prec}f}"
    except Exception:
        return "NA"


def _angle_from_R(R: torch.Tensor) -> torch.Tensor:
    """
    Geodesic angle of a rotation matrix R (radians), in [0, pi].
    Vectorized over leading dims.
    """
    # Equivalent to geodesic_R(I, R) but without materializing identity.
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    skew = R - R.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    sin = vec.norm(dim=-1)
    return torch.atan2(sin, cos)


def _iter_len_safe(*maybe_lists: Any) -> Optional[int]:
    n: Optional[int] = None
    for xs in maybe_lists:
        if isinstance(xs, list):
            n = len(xs) if n is None else min(n, len(xs))
    return n


def _select_step_indices(
    *,
    steps: Optional[List[Dict[str, Any]]],
    cycle_len: int,
    min_len: int,
    min_cycle: int,
    contact_key: str,
    contact_idx: Optional[int],
    contact_value: Optional[int],
    contact_thresh: float,
) -> List[int]:
    sel: List[int] = []
    for i in range(int(min_len)):
        # Cycle filter (R1+ etc).
        cy: Optional[int] = None
        if steps is not None and i < len(steps):
            cy_raw = steps[i].get("cycle", None)
            if isinstance(cy_raw, int):
                cy = int(cy_raw)
        if cy is None and cycle_len > 0:
            cy = int(i // cycle_len)
        cy = int(cy or 0)
        if cy < int(min_cycle):
            continue

        # Optional: contact filter (swing/stance) driven by per-step Contact*PerC.
        if contact_value is not None and contact_idx is not None and steps is not None and i < len(steps):
            c = steps[i].get(contact_key, None)
            if not isinstance(c, list) or contact_idx < 0 or contact_idx >= len(c):
                continue
            try:
                v = float(c[contact_idx])
            except Exception:
                continue
            state = 1 if v >= contact_thresh else 0
            if int(state) != int(contact_value):
                continue

        sel.append(int(i))
    return sel


def _phase_of_step(i: int, *, steps: Optional[List[Dict[str, Any]]], cycle_len: int) -> Optional[int]:
    if steps is not None and i < len(steps):
        si = steps[i].get("step_in_cycle", None)
        if isinstance(si, int):
            return int(si)
    if cycle_len > 0:
        return int(i % cycle_len)
    return None


def _compute_phase_stats(
    vals: Sequence[float], step_idx: Sequence[int], *, steps: Optional[List[Dict[str, Any]]], cycle_len: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[float]]:
    phase_bins: Dict[int, List[float]] = {}
    for v, i in zip(vals, step_idx):
        ph = _phase_of_step(int(i), steps=steps, cycle_len=int(cycle_len))
        if ph is None:
            continue
        phase_bins.setdefault(int(ph), []).append(float(v))

    if not phase_bins:
        return None, None, None
    ph_means = {p: float(mean(vs)) for p, vs in phase_bins.items() if vs}
    if not ph_means:
        return None, None, None
    pmax = max(ph_means, key=lambda p: ph_means[p])
    pmin = min(ph_means, key=lambda p: ph_means[p])
    vmax = float(ph_means[pmax])
    vmin = float(ph_means[pmin])
    return {"phase": int(pmax), "mean": vmax}, {"phase": int(pmin), "mean": vmin}, float(vmax - vmin)


def _as_axis_vec(axis: str) -> torch.Tensor:
    a = str(axis).strip().lower()
    if a not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    idx = {"x": 0, "y": 1, "z": 2}[a]
    v = torch.zeros(3, dtype=torch.float32)
    v[idx] = 1.0
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle upper bound for 1D hinge correction using keybone_omega.series.")
    ap.add_argument("--json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument(
        "--branch",
        type=str,
        default="direct",
        choices=("inc", "direct", "blend"),
        help="Which branch to read under keybone_omega.series.branches (default: direct).",
    )
    ap.add_argument("--bones", nargs="*", default=None, help="Subset of bones to analyze (default: all in JSON).")
    ap.add_argument(
        "--axis",
        type=str,
        default=None,
        help="Hinge axis in joint-local frame (x/y/z). Default: direct_pose_hinge_axis in JSON, else 'z'.",
    )
    ap.add_argument(
        "--max-deg",
        type=float,
        default=None,
        help="Search range for delta in degrees. Default: direct_pose_hinge_max_deg in JSON, else 45.",
    )
    ap.add_argument(
        "--grid-step-deg",
        type=float,
        default=0.25,
        help="Grid resolution for delta* search in degrees (default: 0.25).",
    )
    ap.add_argument("--exclude-round0", action="store_true", help="Use cycles>=1 only.")
    ap.add_argument(
        "--min-cycle",
        type=int,
        default=None,
        help="Include only steps with cycle >= this value (overrides --exclude-round0 if set).",
    )
    ap.add_argument(
        "--contact-source",
        type=str,
        default="gt",
        choices=("gt", "plan", "meas"),
        help="Filter steps by per-step contact source from metrics_per_step (default: gt).",
    )
    ap.add_argument(
        "--contact-side",
        type=str,
        default=None,
        choices=("l", "r"),
        help="Contact channel side for filtering (l/r). If omitted, uses --contact-index.",
    )
    ap.add_argument(
        "--contact-index",
        type=int,
        default=None,
        help="Contact channel index for filtering (in Contact*PerC list).",
    )
    ap.add_argument(
        "--contact-value",
        type=int,
        default=None,
        help="Filter steps where binarized contact == 0 (swing) or 1 (stance).",
    )
    ap.add_argument(
        "--contact-thresh",
        type=float,
        default=0.5,
        help="Threshold to binarize contact floats into {0,1} (default: 0.5).",
    )
    ap.add_argument(
        "--angle-thresh",
        type=float,
        default=None,
        help="Tail threshold in degrees for P(Ang>th) and delta* phase stats. Default: keybone_omega.deg_thresh.",
    )
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    obj = _load_json(path)
    ko = obj.get("keybone_omega", None)
    if not isinstance(ko, dict):
        raise SystemExit("Missing keybone_omega in JSON (re-run with --export_keybone_omega).")
    series = ko.get("series", None)
    if not isinstance(series, dict):
        raise SystemExit("Missing keybone_omega.series in JSON (re-run with --export_keybone_omega_series).")
    sbranches = series.get("branches", None)
    if not isinstance(sbranches, dict):
        raise SystemExit("Invalid keybone_omega.series: missing branches dict.")
    bdat = sbranches.get(str(args.branch), None)
    if not isinstance(bdat, dict):
        raise SystemExit(f"Invalid keybone_omega.series: branches.{args.branch} missing/empty.")

    omega_xyz_map = bdat.get("omega_deg_xyz", None)
    if not isinstance(omega_xyz_map, dict):
        raise SystemExit(
            "Missing keybone_omega.series.branches.*.omega_deg_xyz in JSON. "
            "Re-run freerun after updating train/validate/run_freerun_cycles.py to export omega_deg_xyz."
        )

    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        steps = None
    cycle_len = int(obj.get("cycle_len", 0) or 0)

    # Determine a safe length shared across bones (and aligned to metrics_per_step if present).
    min_len = _iter_len_safe(*omega_xyz_map.values())
    if min_len is None:
        raise SystemExit("Invalid keybone_omega.series: omega_deg_xyz lists are missing.")
    if steps is not None:
        min_len = min(min_len, len(steps))
    min_len = int(min_len)
    if min_len <= 0:
        raise SystemExit("Empty keybone_omega.series (min_len<=0).")

    min_cycle = int(args.min_cycle) if args.min_cycle is not None else (1 if bool(args.exclude_round0) else 0)
    contact_key = {
        "gt": "ContactGTPerC",
        "plan": "ContactPlanPerC",
        "meas": "ContactMeasPerC",
    }.get(str(args.contact_source).strip().lower(), "ContactGTPerC")

    contact_value = args.contact_value
    if contact_value is not None and int(contact_value) not in (0, 1):
        raise SystemExit("--contact-value must be 0 or 1.")
    contact_value = int(contact_value) if contact_value is not None else None

    contact_thresh = float(args.contact_thresh)
    if not (contact_thresh > 0.0):
        raise SystemExit("--contact-thresh must be > 0.")

    contact_idx: Optional[int] = None
    if args.contact_index is not None:
        contact_idx = int(args.contact_index)
    elif args.contact_side is not None:
        contact_idx = 0 if str(args.contact_side).strip().lower() == "l" else 1
    elif contact_value is not None:
        # Match analyze_freerun_keybone_omega.py: default to right foot if filtering without specifying channel.
        contact_idx = 1
    if contact_value is not None and steps is None:
        raise SystemExit("Contact filtering requires metrics_per_step in JSON.")

    sel_idx = _select_step_indices(
        steps=steps,
        cycle_len=cycle_len,
        min_len=min_len,
        min_cycle=min_cycle,
        contact_key=contact_key,
        contact_idx=contact_idx,
        contact_value=contact_value,
        contact_thresh=contact_thresh,
    )
    if not sel_idx:
        raise SystemExit("No steps selected (check --min-cycle/--contact-* filters).")

    # Tail threshold used for P(Ang>th) and delta stats.
    use_th = float(args.angle_thresh) if args.angle_thresh is not None else float(ko.get("deg_thresh", 0.0) or 0.0)

    # Hinge axis/range defaults from JSON (if available).
    axis = args.axis
    if axis is None:
        axis = obj.get("direct_pose_hinge_axis", None) or "z"
    axis = str(axis).strip().lower()
    axis_vec = _as_axis_vec(axis)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

    max_deg = args.max_deg
    if max_deg is None:
        max_deg = obj.get("direct_pose_hinge_max_deg", None)
    max_deg = float(max_deg) if max_deg is not None else 45.0
    max_deg = abs(float(max_deg))

    step_deg = float(args.grid_step_deg)
    if not (step_deg > 0.0):
        raise SystemExit("--grid-step-deg must be > 0.")

    # Grid includes endpoints, so saturation can be detected as (idx==0 or idx==D-1).
    D = int(round((2.0 * max_deg) / step_deg)) + 1
    delta_deg_grid = torch.linspace(-max_deg, max_deg, D, dtype=torch.float32)
    delta_rad_grid = delta_deg_grid * (math.pi / 180.0)

    # Precompute correction rotations exp(-delta * axis).
    with torch.no_grad():
        R_corr = so3_exp_map((-delta_rad_grid[:, None]) * axis_vec[None, :])  # (D,3,3)

    bones = list(series.get("bones", []))
    if args.bones:
        want = set(args.bones)
        bones = [b for b in bones if b in want]
    if not bones:
        bones = [str(b) for b in omega_xyz_map.keys()]

    contact_desc = "none"
    if contact_value is not None:
        contact_desc = f"{contact_key}[{contact_idx}] @ thr={contact_thresh} -> {contact_value}"
    print(f"[JSON] {path}")
    print(
        f"[OracleHinge] branch={args.branch} axis={axis} max_deg={max_deg} grid_step_deg={step_deg} "
        f"ang_deg_thresh={use_th} min_cycle={min_cycle} contact={contact_desc}"
    )
    print()

    headers = [
        "Bone",
        "N",
        "MeanAng",
        "P(Ang>th)",
        "MeanAngOracle",
        "P(AngOracle>th)",
        "MeanDelta*@Ang>th",
        "StdDelta*@Ang>th",
        "P(|Delta*|=max)@Ang>th",
        "PhaseMaxMeanDelta*@Ang>th",
        "PhaseMinMeanDelta*@Ang>th",
        "PhaseAmpMeanDelta*@Ang>th",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

    for bone in bones:
        wdeg = omega_xyz_map.get(bone, None)
        if not isinstance(wdeg, list) or not wdeg:
            continue

        # Align length + apply selection indices.
        T = min(len(wdeg), min_len)
        idx = [i for i in sel_idx if i < T]
        if not idx:
            continue

        # (N,3) in degrees.
        w_sel_deg: List[List[float]] = []
        for i in idx:
            v = wdeg[i]
            if not (isinstance(v, list) and len(v) == 3):
                continue
            try:
                w_sel_deg.append([float(v[0]), float(v[1]), float(v[2])])
            except Exception:
                continue
        if not w_sel_deg:
            continue

        w = torch.tensor(w_sel_deg, dtype=torch.float32)  # (N,3) degrees
        ang_deg = w.norm(dim=-1)  # (N,)
        w_rad = w * (math.pi / 180.0)

        with torch.no_grad():
            R_err = so3_exp_map(w_rad)  # (N,3,3)
            # (D,N,3,3)
            R_new = torch.matmul(R_corr[:, None, :, :], R_err[None, :, :, :])
            ang_new_rad = _angle_from_R(R_new)  # (D,N)
            ang_new_deg = ang_new_rad * (180.0 / math.pi)
            best_idx = ang_new_deg.argmin(dim=0)  # (N,)
            best_ang_deg = ang_new_deg[best_idx, torch.arange(ang_new_deg.shape[1])]
            best_delta_deg = delta_deg_grid[best_idx]

        # Angle stats
        n_tot = int(ang_deg.numel())
        mean_ang = float(ang_deg.mean().item()) if n_tot > 0 else None
        p_sel = float((ang_deg > use_th).to(dtype=torch.float32).mean().item()) if n_tot > 0 else None
        mean_ang_oracle = float(best_ang_deg.mean().item()) if n_tot > 0 else None
        p_sel_oracle = (
            float((best_ang_deg > use_th).to(dtype=torch.float32).mean().item()) if n_tot > 0 else None
        )

        # Tail (based on original angle).
        tail_mask = ang_deg > use_th
        tail_delta = best_delta_deg[tail_mask]
        tail_idx = [int(i) for (i, m) in zip(idx, tail_mask.tolist()) if bool(m)]
        tail_delta_list = [float(x) for x in tail_delta.tolist()] if int(tail_delta.numel()) > 0 else []
        m_d, s_d = _mean_std(tail_delta_list)
        sat = None
        if tail_delta_list:
            # Saturation: hits grid endpoints exactly.
            sat = float(((tail_delta.abs() >= (max_deg - 1e-6)).to(dtype=torch.float32).mean().item()))

        ph_max, ph_min, ph_amp = _compute_phase_stats(
            tail_delta_list, tail_idx, steps=steps, cycle_len=cycle_len
        )

        print(
            "| "
            + " | ".join(
                [
                    str(bone),
                    str(n_tot),
                    _fmt_float(mean_ang, prec=2),
                    _fmt_float(p_sel, prec=3),
                    _fmt_float(mean_ang_oracle, prec=2),
                    _fmt_float(p_sel_oracle, prec=3),
                    _fmt_float(m_d, prec=2),
                    _fmt_float(s_d, prec=2),
                    _fmt_float(sat, prec=3),
                    _fmt_phase_mean(ph_max, prec=2),
                    _fmt_phase_mean(ph_min, prec=2),
                    _fmt_float(ph_amp, prec=2),
                ]
            )
            + " |"
        )

        # Extra: quick sanity of how delta relates to omega along the hinge axis (on tail frames).
        if tail_delta_list:
            omega_axis = w[tail_mask, int(axis_idx)]
            omega_axis_list = [float(x) for x in omega_axis.tolist()]
            m_om, s_om = _mean_std(omega_axis_list)
            m_res, s_res = _mean_std([float(a - d) for a, d in zip(omega_axis_list, tail_delta_list)])
            print(
                f"  [Tail:{bone}] omega_{axis}_deg@Ang>th mean={_fmt_float(m_om, prec=2)} std={_fmt_float(s_om, prec=2)} | "
                f"(omega_{axis}_deg - delta*) mean={_fmt_float(m_res, prec=2)} std={_fmt_float(s_res, prec=2)}"
            )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
