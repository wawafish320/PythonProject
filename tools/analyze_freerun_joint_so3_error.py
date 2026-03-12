#!/usr/bin/env python3
"""
Analyze per-step per-joint SO(3) error vectors exported by:

  python -m train.validate.run_freerun_cycles --export_joint_so3_error_series

Expected JSON fields:
  - per_step_joint_so3_error:
      - bone_names, root_idx
      - branches.{inc|direct|blend}.{body|world}.{rotvec_deg_xyz, ang_deg}
  - metrics_per_step[t]: cycle, step_in_cycle, wrap_boundary_step

This tool supports Experiment 1.1 style summaries:
  - per-joint bias (mean epsilon) and variance (std epsilon)
  - axis dominance (dominant |component| fraction) under angle thresholding
  - phase-locked explained variance by step_in_cycle (sic)
  - drift/diffusion on increments via delta_E = E_t^T @ E_{t+1}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.geometry import so3_exp_map, so3_log_map  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _mask_indices(
    steps: Optional[List[Dict[str, Any]]],
    *,
    T: int,
    min_cycle: int,
    exclude_wrap: bool,
) -> np.ndarray:
    idx: List[int] = []
    for t in range(int(T)):
        if steps is None or t >= len(steps):
            cy = 0
            wrap = False
        else:
            s = steps[t]
            cy = int(s.get("cycle", 0) or 0)
            wrap = bool(s.get("wrap_boundary_step", False))
        if cy < int(min_cycle):
            continue
        if exclude_wrap and wrap:
            continue
        idx.append(int(t))
    return np.asarray(idx, dtype=np.int64)


def _sic_vector(
    steps: Optional[List[Dict[str, Any]]],
    *,
    T: int,
    idx: np.ndarray,
) -> Optional[np.ndarray]:
    if steps is None:
        return None
    sic = np.full((int(T),), -1, dtype=np.int64)
    for t in range(min(int(T), len(steps))):
        v = steps[t].get("step_in_cycle", None)
        if isinstance(v, int):
            sic[int(t)] = int(v)
    out = sic[idx]
    if not np.any(out >= 0):
        return None
    return out


def _dominant_axis_frac(u_abs: np.ndarray) -> np.ndarray:
    """u_abs: (N,3) absolute unit directions -> frac over {x,y,z}."""
    if u_abs.ndim != 2 or u_abs.shape[1] != 3:
        raise ValueError(f"u_abs must be (N,3), got {u_abs.shape}")
    if u_abs.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64)
    dom = np.argmax(u_abs, axis=1)
    return np.asarray([(dom == 0).mean(), (dom == 1).mean(), (dom == 2).mean()], dtype=np.float64)


def _explained_var_by_sic(
    eps: np.ndarray,  # (N,3)
    sic: Optional[np.ndarray],  # (N,) or None
    *,
    eps0: float = 1e-8,
) -> Optional[float]:
    if sic is None:
        return None
    if eps.ndim != 2 or eps.shape[1] != 3:
        raise ValueError(f"eps must be (N,3), got {eps.shape}")
    if sic.shape[0] != eps.shape[0]:
        raise ValueError("sic/eps length mismatch")

    m = np.mean(eps, axis=0, keepdims=True)
    var_total = float(np.mean(np.sum((eps - m) ** 2, axis=1)))
    if not (var_total > eps0):
        return 0.0

    uniq = sorted({int(s) for s in sic.tolist() if int(s) >= 0})
    if not uniq:
        return None

    mu_by: Dict[int, np.ndarray] = {}
    for s in uniq:
        mask = sic == int(s)
        if not np.any(mask):
            continue
        mu_by[int(s)] = np.mean(eps[mask], axis=0)
    if not mu_by:
        return None

    resid = eps.copy()
    for i in range(resid.shape[0]):
        mu = mu_by.get(int(sic[i]), None)
        if mu is not None:
            resid[i] -= mu
    var_resid = float(np.mean(np.sum(resid**2, axis=1)))
    return float(1.0 - (var_resid / (var_total + eps0)))


def _delta_stats_from_eps(
    eps_deg: np.ndarray,  # (N,3) in degrees
    *,
    max_ang_deg: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute drift/diffusion on increments:
      E_t = Exp(eps_t)
      delta_E_t = E_t^T @ E_{t+1}
      delta = Log(delta_E)

    Returns:
      mean_delta_deg_xyz, std_delta_deg_xyz  (each (3,)) or (None,None) if insufficient.
    """
    if eps_deg.shape[0] < 2:
        return None, None
    ang = np.linalg.norm(eps_deg, axis=1)
    keep = np.isfinite(ang) & (ang <= float(max_ang_deg))
    if int(np.sum(keep)) < 2:
        return None, None

    x = eps_deg[keep].astype(np.float32, copy=False)
    x_rad = x * (math.pi / 180.0)

    import torch

    with torch.no_grad():
        E = so3_exp_map(torch.from_numpy(x_rad))  # (M,3,3)
        dE = torch.matmul(E[:-1].transpose(-1, -2), E[1:])  # (M-1,3,3)
        d = so3_log_map(dE)  # (M-1,3) rad
        d_deg = (d * (180.0 / math.pi)).cpu().numpy().astype(np.float64)

    if d_deg.shape[0] <= 0:
        return None, None
    return np.mean(d_deg, axis=0), np.std(d_deg, axis=0, ddof=0)


def _fmt_vec3(v: Sequence[float], prec: int = 3) -> str:
    return "[" + ", ".join(f"{float(x): .{prec}f}" for x in v) + "]"


def _fmt_dom(dom: Optional[np.ndarray]) -> str:
    if dom is None:
        return "NA"
    return "[x={:.2f}, y={:.2f}, z={:.2f}]".format(float(dom[0]), float(dom[1]), float(dom[2]))


@dataclass
class JointRow:
    name: str
    j: int
    bias_norm: float
    bias_xyz: np.ndarray  # (3,)
    std_xyz: np.ndarray  # (3,)
    dom_frac: Optional[np.ndarray]  # (3,)
    explained_by_sic: Optional[float]
    drift_norm: Optional[float]
    drift_xyz: Optional[np.ndarray]
    diff_std_xyz: Optional[np.ndarray]
    diff_norm: Optional[float]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze per_step_joint_so3_error from freerun_cycles JSON.")
    ap.add_argument("--json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--space", type=str, default="body", choices=("body", "world"))
    ap.add_argument("--min-cycle", type=int, default=1, help="Include steps with cycle >= this (default: 1).")
    ap.add_argument("--exclude-wrap", action="store_true", help="Exclude wrap_boundary_step frames.")
    ap.add_argument("--exclude-root", action="store_true", help="Exclude root joint (root_idx) from reporting.")
    ap.add_argument(
        "--angle-thresh-deg",
        type=float,
        default=1.0,
        help="Angle threshold (deg) for axis dominance stats (default: 1.0).",
    )
    ap.add_argument(
        "--max-inc-angle-deg",
        type=float,
        default=120.0,
        help="Max |eps| (deg) used when computing increment drift/diffusion (default: 120).",
    )
    ap.add_argument("--no-increments", action="store_true", help="Skip drift/diffusion stats on increments.")
    ap.add_argument("--no-sic", action="store_true", help="Skip explained variance by sic.")
    ap.add_argument("--report-topk", type=int, default=20, help="Print top-k joints by bias norm (default: 20).")
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    obj = _load_json(path)
    block = obj.get("per_step_joint_so3_error", None)
    if not isinstance(block, dict):
        raise SystemExit("Missing per_step_joint_so3_error (re-run with --export_joint_so3_error_series).")

    names = block.get("bone_names", None)
    root_idx = int(block.get("root_idx", 0) or 0)
    if not isinstance(names, list) or not names:
        raise SystemExit("Invalid per_step_joint_so3_error: missing bone_names.")

    branches = block.get("branches", None)
    if not isinstance(branches, dict):
        raise SystemExit("Invalid per_step_joint_so3_error: missing branches.")
    b = branches.get(str(args.branch), None)
    if not isinstance(b, dict):
        raise SystemExit(f"Missing branch '{args.branch}' under per_step_joint_so3_error.branches.")
    sp = b.get(str(args.space), None)
    if not isinstance(sp, dict):
        raise SystemExit(f"Missing space '{args.space}' for branch '{args.branch}'. Re-run with space=both.")

    eps_all = _as_np(sp.get("rotvec_deg_xyz", None))  # (T,J,3)
    if eps_all.ndim != 3 or eps_all.shape[2] != 3:
        raise SystemExit(f"rotvec_deg_xyz must be (T,J,3), got {eps_all.shape}")
    T, J = int(eps_all.shape[0]), int(eps_all.shape[1])
    if len(names) < J:
        names = names + [f"joint_{i}" for i in range(len(names), J)]
    names = names[:J]

    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        steps = None

    idx_t = _mask_indices(
        steps,
        T=T,
        min_cycle=int(args.min_cycle),
        exclude_wrap=bool(args.exclude_wrap),
    )
    if idx_t.size <= 0:
        raise SystemExit("Mask produced empty step set.")
    sic = _sic_vector(steps, T=T, idx=idx_t) if (steps is not None and not bool(args.no_sic)) else None

    angle_thresh = float(args.angle_thresh_deg)
    max_inc_ang = float(args.max_inc_angle_deg)

    rows: List[JointRow] = []
    dom_counts = np.zeros((3,), dtype=np.int64)
    dom_total = 0
    expl_vals: List[float] = []

    for j in range(J):
        if bool(args.exclude_root) and int(j) == int(root_idx):
            continue
        eps_j = eps_all[idx_t, j, :]  # (N,3)
        if eps_j.shape[0] <= 0:
            continue
        mu = np.mean(eps_j, axis=0)
        sd = np.std(eps_j, axis=0, ddof=0)
        bias_norm = float(np.linalg.norm(mu))

        # Axis dominance stats (only for sufficiently large angles).
        ang = np.linalg.norm(eps_j, axis=1)
        mask_ang = np.isfinite(ang) & (ang > angle_thresh)
        dom_frac = None
        if int(np.sum(mask_ang)) > 0:
            u = eps_j[mask_ang] / (ang[mask_ang].reshape(-1, 1) + 1e-9)
            u_abs = np.abs(u)
            dom_frac = _dominant_axis_frac(u_abs)
            dom = np.argmax(u_abs, axis=1)
            for k in (0, 1, 2):
                dom_counts[k] += int(np.sum(dom == k))
            dom_total += int(dom.shape[0])

        explained = None
        if sic is not None:
            try:
                explained = _explained_var_by_sic(eps_j, sic)
                if explained is not None and math.isfinite(float(explained)):
                    expl_vals.append(float(explained))
            except Exception:
                explained = None

        drift_norm = None
        drift_xyz = None
        diff_std_xyz = None
        diff_norm = None
        if not bool(args.no_increments):
            try:
                dmu, dsd = _delta_stats_from_eps(eps_j, max_ang_deg=max_inc_ang)
                if dmu is not None and dsd is not None:
                    drift_xyz = dmu
                    diff_std_xyz = dsd
                    drift_norm = float(np.linalg.norm(dmu))
                    diff_norm = float(np.linalg.norm(dsd))
            except Exception:
                pass

        rows.append(
            JointRow(
                name=str(names[j]),
                j=int(j),
                bias_norm=bias_norm,
                bias_xyz=mu,
                std_xyz=sd,
                dom_frac=dom_frac,
                explained_by_sic=explained,
                drift_norm=drift_norm,
                drift_xyz=drift_xyz,
                diff_std_xyz=diff_std_xyz,
                diff_norm=diff_norm,
            )
        )

    rows.sort(key=lambda r: float(r.bias_norm), reverse=True)

    print(f"[joint_so3_error] json={path}")
    print(f"  branch={args.branch} space={args.space} T={T} J={J} root_idx={root_idx}")
    print(f"  mask: cycle>={args.min_cycle}, exclude_wrap={bool(args.exclude_wrap)}, steps_used={int(idx_t.size)}")
    print(f"  axis_stats: angle_thresh_deg={angle_thresh}")
    if not bool(args.no_increments):
        print(f"  inc_stats: max_inc_angle_deg={max_inc_ang} (filter on |eps| before building increments)")
    if sic is not None:
        m = float(np.mean(expl_vals)) if expl_vals else float("nan")
        print(f"  sic_explained: mean={m:.3f} over {len(expl_vals)} joints")

    topk = int(max(0, args.report_topk))
    if topk <= 0:
        return

    print("\nTop joints by ||bias|| (deg):")
    hdr = "  {:>3s} {:<18s} {:>8s}  {:<24s}  {:<24s}  {:<18s}  {:>8s}  {:>8s}  {:>8s}".format(
        "idx",
        "joint",
        "|mu|",
        "mu_xyz(deg)",
        "std_xyz(deg)",
        "dom_frac",
        "sic_expl",
        "|drift|",
        "|diff|",
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in rows[:topk]:
        sic_s = "NA" if r.explained_by_sic is None else f"{float(r.explained_by_sic):.3f}"
        drift_s = "NA" if r.drift_norm is None else f"{float(r.drift_norm):.3f}"
        diff_s = "NA" if r.diff_norm is None else f"{float(r.diff_norm):.3f}"
        print(
            "  {:3d} {:<18s} {:8.3f}  {:<24s}  {:<24s}  {:<18s}  {:>8s}  {:>8s}  {:>8s}".format(
                int(r.j),
                str(r.name)[:18],
                float(r.bias_norm),
                _fmt_vec3(r.bias_xyz, prec=3),
                _fmt_vec3(r.std_xyz, prec=3),
                _fmt_dom(r.dom_frac),
                sic_s,
                drift_s,
                diff_s,
            )
        )

    if dom_total > 0:
        frac = dom_counts.astype(np.float64) / float(dom_total)
        print("\nGlobal dominant-axis fraction (aggregated over joints/time where |eps|>thr):")
        print(f"  x={float(frac[0]):.3f} y={float(frac[1]):.3f} z={float(frac[2]):.3f} (N={int(dom_total)})")


if __name__ == "__main__":
    main()
