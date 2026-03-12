#!/usr/bin/env python3
"""
Report: overlay SIC hotspots (phase-locked SO(3) error bias) with GT joint angular velocity.

This is designed for diagnosing: "why the model makes a systematic same-direction error at specific sic".

Inputs:
  - freerun_cycles JSON (must contain):
      - per_step_joint_so3_error.branches[branch][space].rotvec_deg_xyz  (T_total,J,3) in degrees
      - metrics_per_step[t]: cycle, step_in_cycle, wrap_boundary_step
  - processed clip NPZ (raw_data/processed_data/*.npz) with:
      - bone_names
      - bone_ang_vel (T,B,3) in rad/s (preferred) OR bone_rot6d (T,B,6) to recompute
      - FPS

Outputs:
  - a markdown report (tables + links)
  - a PNG figure (angvel profiles + vertical hotspot markers + error magnitude overlay)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _fmt_vec3(v: Sequence[float], prec: int = 3) -> str:
    return "[" + ", ".join(f"{float(x): .{prec}f}" for x in v) + "]"


def _dom_axis_sign(v: np.ndarray) -> str:
    a = int(np.argmax(np.abs(v)))
    ax = "xyz"[a]
    s = "+" if float(v[a]) >= 0.0 else "-"
    return ax + s


def _mask_indices(
    steps: Sequence[Dict[str, Any]],
    *,
    T: int,
    min_cycle: int,
    exclude_wrap: bool,
) -> np.ndarray:
    idx: List[int] = []
    for t in range(int(T)):
        s = steps[t] if t < len(steps) else {}
        cy = int(s.get("cycle", 0) or 0)
        wrap = bool(s.get("wrap_boundary_step", False))
        sic = s.get("step_in_cycle", None)
        sic_i = int(sic) if isinstance(sic, int) else -1
        if cy < int(min_cycle):
            continue
        if exclude_wrap and wrap:
            continue
        if sic_i < 0:
            continue
        idx.append(int(t))
    return np.asarray(idx, dtype=np.int64)


def _sic_vec(steps: Sequence[Dict[str, Any]], *, T: int) -> np.ndarray:
    out = np.full((int(T),), -1, dtype=np.int64)
    for t in range(min(int(T), len(steps))):
        v = steps[t].get("step_in_cycle", None)
        if isinstance(v, int):
            out[int(t)] = int(v)
    return out


def _infer_cycle_len(steps: Sequence[Dict[str, Any]]) -> Optional[int]:
    # Prefer first cycle (cycle==0).
    sic0: List[int] = []
    for s in steps:
        cy = int(s.get("cycle", 0) or 0)
        if cy != 0:
            continue
        v = s.get("step_in_cycle", None)
        if isinstance(v, int) and v >= 0:
            sic0.append(int(v))
    if sic0:
        return int(max(sic0) + 1)
    # Fallback: global max sic + 1.
    sic_all: List[int] = []
    for s in steps:
        v = s.get("step_in_cycle", None)
        if isinstance(v, int) and v >= 0:
            sic_all.append(int(v))
    if sic_all:
        return int(max(sic_all) + 1)
    return None


def _load_npz_angvel(npz_path: Path) -> Tuple[List[str], float, np.ndarray]:
    """
    Returns:
      names, fps, omega_rad_s (T,B,3) float64
    """
    npz = np.load(str(npz_path), allow_pickle=True)
    names = npz.get("bone_names", None)
    if names is None:
        raise ValueError(f"{npz_path}: missing bone_names")
    names_list = names.tolist()
    fps = float(npz.get("FPS", 0.0) or 0.0)
    if "bone_ang_vel" in npz:
        w = np.asarray(npz["bone_ang_vel"], dtype=np.float64)
        if w.ndim != 3 or w.shape[-1] != 3:
            raise ValueError(f"{npz_path}: bone_ang_vel must be (T,B,3), got {w.shape}")
        return names_list, fps, w

    # Fallback: recompute from bone_rot6d if bone_ang_vel missing.
    if "bone_rot6d" not in npz:
        raise ValueError(f"{npz_path}: missing bone_ang_vel and bone_rot6d (cannot compute omega).")

    if fps <= 0:
        raise ValueError(f"{npz_path}: invalid FPS={fps} (needed for omega recompute).")

    import torch
    from train.geometry import reproject_rot6d, rot6d_to_matrix, so3_log_map

    rot6d = np.asarray(npz["bone_rot6d"], dtype=np.float32)
    if rot6d.ndim != 3 or rot6d.shape[-1] != 6:
        raise ValueError(f"{npz_path}: bone_rot6d must be (T,B,6), got {rot6d.shape}")
    T, B, _ = rot6d.shape

    with torch.no_grad():
        x = torch.from_numpy(rot6d)
        x = reproject_rot6d(x)  # (T,B,6)
        R = rot6d_to_matrix(x.view(1, T, B, 6))[0]  # (T,B,3,3)
        dR = torch.matmul(R[1:], R[:-1].transpose(-1, -2))  # (T-1,B,3,3)
        phi = so3_log_map(dR)  # (T-1,B,3) rad
        omega = phi * float(fps)  # (T-1,B,3) rad/s
        omega = torch.cat([omega, omega[-1:]], dim=0)  # pad to T
        w = omega.cpu().numpy().astype(np.float64, copy=False)
    return names_list, fps, w


@dataclass
class GlobalHotspot:
    sic: int
    N: int
    mean_mu_norm: float
    max_mu_norm: float
    top_joint: str
    top_mu_xyz: np.ndarray  # (3,)
    top_dom: str


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]
    if a.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Fit y = a + b*x on finite pairs.

    Returns:
      a, b, r2, n
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return a, b, r2, n


def _parse_int_csv(text: str) -> List[int]:
    vals: List[int] = []
    for part in str(text or "").split(","):
        tok = part.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    out: List[int] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(int(v))
    return out


def _compute_dt_stats_for_joint(
    *,
    mu_xyz_deg: np.ndarray,
    omega_xyz_deg_s: np.ndarray,
    axis_i: int,
    mu_thr_deg: float,
    omega_min_deg_s: float,
    fps: float,
) -> Dict[str, float]:
    mu_xyz = np.asarray(mu_xyz_deg, dtype=np.float64)
    omg_xyz = np.asarray(omega_xyz_deg_s, dtype=np.float64)
    T = int(min(mu_xyz.shape[0], omg_xyz.shape[0]))
    if T <= 0:
        return {
            "align_frac": float("nan"),
            "N_mu": 0,
            "N_dt": 0,
            "dt_median": float("nan"),
            "dt_iqr": [float("nan"), float("nan")],
        }
    mu_xyz = mu_xyz[:T]
    omg_xyz = omg_xyz[:T]
    mu_axis = mu_xyz[:, int(axis_i)]
    mu_norm = np.linalg.norm(mu_xyz, axis=1)
    omg_axis = omg_xyz[:, int(axis_i)]
    mask_mu = np.isfinite(mu_axis) & np.isfinite(mu_norm) & np.isfinite(omg_axis) & (mu_norm >= float(mu_thr_deg))
    align = float(np.mean((mu_axis[mask_mu] * omg_axis[mask_mu]) > 0.0)) if np.any(mask_mu) else float("nan")
    dt = np.full((T,), np.nan, dtype=np.float64)
    ok = mask_mu & (np.abs(omg_axis) >= float(omega_min_deg_s))
    dt[ok] = (mu_axis[ok] / omg_axis[ok]) * float(fps)
    vals = dt[np.isfinite(dt)]
    if vals.size > 0:
        dt_median = float(np.median(vals))
        dt_iqr = [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]
    else:
        dt_median = float("nan")
        dt_iqr = [float("nan"), float("nan")]
    return {
        "align_frac": align,
        "N_mu": int(np.sum(mask_mu)),
        "N_dt": int(vals.size),
        "dt_median": dt_median,
        "dt_iqr": dt_iqr,
    }


def _dt_pair_summary(dt_by_joint: Dict[str, Dict[str, float]], joints: Sequence[str]) -> Dict[str, Any]:
    med = {str(k): float(v.get("dt_median", float("nan"))) for k, v in dt_by_joint.items()}
    if len(joints) >= 2:
        j0 = str(joints[0])
        j1 = str(joints[1])
        m0 = med.get(j0, float("nan"))
        m1 = med.get(j1, float("nan"))
        both = np.isfinite(m0) and np.isfinite(m1)
        common = float(0.5 * (m0 + m1)) if both else float("nan")
        asym = float(0.5 * abs(m0 - m1)) if both else float("nan")
        same = bool((m0 >= 0 and m1 >= 0) or (m0 <= 0 and m1 <= 0)) if both else False
    else:
        common = float("nan")
        asym = float("nan")
        same = False
    return {
        "dt_medians": med,
        "common_dt": common,
        "asym_dt": asym,
        "same_sign": same,
    }


def _compute_gt_symmetry_sanity(
    *,
    npz_path: Path,
    gt_names: Sequence[str],
    omega_deg_s: np.ndarray,
    joints: Sequence[str],
    axis_i: int,
    mu_thr_deg: float,
    omega_min_deg_s: float,
    fps: float,
    shifts: Sequence[int],
) -> Dict[str, Any]:
    npz = np.load(str(npz_path), allow_pickle=True)
    if "bone_rot6d" not in npz:
        return {"available": False, "reason": "missing bone_rot6d in npz"}

    rot6d = np.asarray(npz["bone_rot6d"], dtype=np.float32)
    if rot6d.ndim != 3 or int(rot6d.shape[-1]) != 6:
        return {"available": False, "reason": f"invalid bone_rot6d shape: {tuple(rot6d.shape)}"}
    T, J, _ = rot6d.shape
    if T <= 1 or J <= 0:
        return {"available": False, "reason": f"bone_rot6d too short: T={T}, J={J}"}

    npz_names = [str(x) for x in np.asarray(npz["bone_names"]).tolist()]
    name_to_rot = {str(n): int(i) for i, n in enumerate(npz_names)}
    name_to_w = {str(n): int(i) for i, n in enumerate(gt_names)}

    use_joints: List[str] = []
    for n in joints:
        if str(n) in name_to_rot and str(n) in name_to_w:
            use_joints.append(str(n))
    if not use_joints:
        return {"available": False, "reason": "no requested joints resolved in npz/omega"}

    import torch
    from train.geometry import reproject_rot6d, rot6d_to_matrix, so3_log_map

    with torch.no_grad():
        x = torch.from_numpy(rot6d.astype(np.float32))
        x = reproject_rot6d(x)
        R_gt = rot6d_to_matrix(x.view(1, int(T), int(J), 6))[0]  # (T,J,3,3)

    shift_rows: List[Dict[str, Any]] = []
    for k in shifts:
        kk = int(k)
        if kk == 0:
            R_pred = R_gt
        elif kk > 0:
            pad = R_gt[:1].expand(min(kk, T), -1, -1, -1)
            R_pred = torch.cat([pad, R_gt[: max(0, T - kk)]], dim=0)
        else:
            take = min(-kk, T)
            pad = R_gt[T - 1 : T].expand(take, -1, -1, -1)
            R_pred = torch.cat([R_gt[take:], pad], dim=0)

        with torch.no_grad():
            R_err = torch.matmul(R_pred.transpose(-1, -2), R_gt)
            mu = so3_log_map(R_err) * (180.0 / math.pi)  # (T,J,3) deg
            mu_np = mu.cpu().numpy().astype(np.float64, copy=False)

        by_joint: Dict[str, Dict[str, float]] = {}
        for name in use_joints:
            jr = int(name_to_rot[name])
            jw = int(name_to_w[name])
            w_xyz = np.asarray(omega_deg_s[:T, jw, :], dtype=np.float64)
            m_xyz = np.asarray(mu_np[:T, jr, :], dtype=np.float64)
            by_joint[name] = _compute_dt_stats_for_joint(
                mu_xyz_deg=m_xyz,
                omega_xyz_deg_s=w_xyz,
                axis_i=int(axis_i),
                mu_thr_deg=float(mu_thr_deg),
                omega_min_deg_s=float(omega_min_deg_s),
                fps=float(fps),
            )
        pair = _dt_pair_summary(by_joint, joints=use_joints)
        shift_rows.append(
            {
                "k": int(kk),
                "joints": by_joint,
                "common_dt": float(pair["common_dt"]),
                "asym_dt": float(pair["asym_dt"]),
                "same_sign": bool(pair["same_sign"]),
            }
        )

    omega_snapshot: Dict[str, Any] = {"axis": "xyz"[int(axis_i)], "joints": {}}
    for name in use_joints[:2]:
        j = int(name_to_w[name])
        wa = np.asarray(omega_deg_s[:T, j, int(axis_i)], dtype=np.float64)
        omega_snapshot["joints"][name] = {
            "mean_abs_omega_axis_deg_s": float(np.nanmean(np.abs(wa))),
            "std_abs_omega_axis_deg_s": float(np.nanstd(np.abs(wa))),
        }

    if len(use_joints) >= 2:
        a = np.asarray(omega_deg_s[:T, int(name_to_w[use_joints[0]]), int(axis_i)], dtype=np.float64)
        b = np.asarray(omega_deg_s[:T, int(name_to_w[use_joints[1]]), int(axis_i)], dtype=np.float64)
        best_corr = float("nan")
        best_shift = 0
        for s in range(int(T)):
            corr = _safe_corr(a, np.roll(b, int(s)))
            if not math.isfinite(corr):
                continue
            if not math.isfinite(best_corr) or corr > best_corr:
                best_corr = float(corr)
                best_shift = int(s)
        omega_snapshot["lr_axis_roll_corr"] = {
            "joint_a": str(use_joints[0]),
            "joint_b": str(use_joints[1]),
            "best_shift": int(best_shift),
            "best_corr": float(best_corr),
            "cycle_len": int(T),
        }

    return {
        "available": True,
        "axis": "xyz"[int(axis_i)],
        "joints": [str(x) for x in use_joints],
        "mu_thr_deg": float(mu_thr_deg),
        "omega_min_deg_s": float(omega_min_deg_s),
        "fps": float(fps),
        "shifts": [int(x) for x in shifts],
        "shift_rows": shift_rows,
        "omega_snapshot": omega_snapshot,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SIC hotspots vs GT angular-velocity overlay report.")
    ap.add_argument("--freerun-json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument("--npz", type=str, required=True, help="Path to processed clip NPZ (e.g. raw_data/processed_data/Walk_F.npz)")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--space", type=str, default="body", choices=("body", "world"))
    ap.add_argument("--min-cycle", type=int, default=1)
    ap.add_argument("--exclude-wrap", action="store_true")
    ap.add_argument("--exclude-root", action="store_true")
    ap.add_argument("--axis", type=str, default="z", choices=("x", "y", "z"), help="Which GT omega component to plot.")
    ap.add_argument(
        "--joints",
        type=str,
        default="calf_l,calf_r",
        help="Comma-separated joints to plot and tabulate (default: calf_l,calf_r).",
    )
    ap.add_argument("--mu-dom-thresh-deg", type=float, default=0.5)
    ap.add_argument("--omega-min-deg-s", type=float, default=30.0, help="Min |omega_axis| for lag estimate (deg/s).")
    ap.add_argument(
        "--projection-diag",
        action="store_true",
        help="Enable projection-based lag diagnostics (dt*, r_perp, H1/H2 fit).",
    )
    ap.add_argument(
        "--projection-mu-max-deg",
        type=float,
        default=5.0,
        help="Projection diag gate: max ||mu|| in degrees (default: 5).",
    )
    ap.add_argument(
        "--projection-unreliable-r-perp",
        type=float,
        default=0.5,
        help="Mark dt* as unreliable when r_perp(median) exceeds this threshold.",
    )
    ap.add_argument("--hotspot-topk", type=int, default=15, help="Top-K sic by global max_j ||mu|| to highlight.")
    ap.add_argument(
        "--out-md",
        type=str,
        default="docs/Problems/active/2026-02-11_WalkF_stage7_sic_hotspots_vs_gt_knee_angvel.md",
        help="Output markdown path.",
    )
    ap.add_argument(
        "--out-fig",
        type=str,
        default="docs/Problems/active/assets/2026-02-11_WalkF_sic_vs_gt_knee_angvel.png",
        help="Output figure path (png).",
    )
    ap.add_argument(
        "--gt-symmetry-sanity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GT-only dt sanity using synthetic frame shifts `pred(t)=gt(t-k)`.",
    )
    ap.add_argument(
        "--gt-shifts",
        type=str,
        default="1,2,3,-1,-2,-3",
        help="Comma-separated synthetic shifts (frames) for GT-only dt sanity.",
    )
    ap.add_argument(
        "--gt-sym-joints",
        type=str,
        default="",
        help="Optional joints CSV for GT-only sanity (default: reuse --joints resolved list).",
    )
    args = ap.parse_args()

    fr_path = Path(args.freerun_json).expanduser()
    npz_path = Path(args.npz).expanduser()
    out_md = Path(args.out_md).expanduser()
    out_fig = Path(args.out_fig).expanduser()

    obj = _load_json(fr_path)
    blk = obj.get("per_step_joint_so3_error", None)
    if not isinstance(blk, dict):
        raise SystemExit("Missing per_step_joint_so3_error (re-run freerun with --export_joint_so3_error_series).")

    names = blk.get("bone_names", None)
    if not isinstance(names, list) or not names:
        raise SystemExit("Invalid per_step_joint_so3_error: missing bone_names.")
    root_idx = int(blk.get("root_idx", 0) or 0)

    branches = blk.get("branches", None)
    if not isinstance(branches, dict):
        raise SystemExit("Invalid per_step_joint_so3_error: missing branches.")
    b = branches.get(str(args.branch), None)
    if not isinstance(b, dict):
        raise SystemExit(f"Missing branch '{args.branch}'.")
    sp = b.get(str(args.space), None)
    if not isinstance(sp, dict):
        raise SystemExit(f"Missing space '{args.space}' under branch '{args.branch}'. Re-run with space=both.")

    eps_all = _as_np(sp.get("rotvec_deg_xyz", None))  # (T_total,J,3) degrees
    if eps_all.ndim != 3 or eps_all.shape[2] != 3:
        raise SystemExit(f"rotvec_deg_xyz must be (T,J,3), got {eps_all.shape}")
    T_total, J, _ = eps_all.shape

    bone_names = list(names)
    if len(bone_names) < int(J):
        bone_names = bone_names + [f"joint_{i}" for i in range(len(bone_names), int(J))]
    bone_names = bone_names[: int(J)]

    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        raise SystemExit("Missing metrics_per_step.")

    idx_t = _mask_indices(
        steps,
        T=int(T_total),
        min_cycle=int(args.min_cycle),
        exclude_wrap=bool(args.exclude_wrap),
    )
    if idx_t.size <= 0:
        raise SystemExit("Mask produced empty step set.")

    sic_all = _sic_vec(steps, T=int(T_total))
    sics = sorted({int(s) for s in sic_all[idx_t].tolist() if int(s) >= 0})
    if not sics:
        raise SystemExit("No valid sic under mask.")

    # Infer cycle length from the unmasked metrics (should be stable).
    cycle_len = _infer_cycle_len(steps) or (max(sics) + 1)
    cycle_len = int(max(1, cycle_len))

    # mu_by_sic[s] = (J,3) in degrees (body/world depends on args.space)
    mu_by_sic: Dict[int, np.ndarray] = {}
    for s in sics:
        t_idx = idx_t[sic_all[idx_t] == int(s)]
        mu = np.mean(eps_all[t_idx], axis=0)
        if bool(args.exclude_root) and 0 <= int(root_idx) < int(J):
            mu[int(root_idx)] = 0.0
        mu_by_sic[int(s)] = mu

    # Global hotspots (per sic): max over joints of ||mu||.
    global_rows: List[GlobalHotspot] = []
    mu_dom_thr = float(args.mu_dom_thresh_deg)
    for s in sics:
        mu = mu_by_sic[int(s)]
        n = np.linalg.norm(mu, axis=1)
        if bool(args.exclude_root) and 0 <= int(root_idx) < int(J):
            n[int(root_idx)] = np.nan
        mean_mu = float(np.nanmean(n))
        jtop = int(np.nanargmax(n))
        max_mu = float(n[int(jtop)])
        global_rows.append(
            GlobalHotspot(
                sic=int(s),
                N=int(np.sum(sic_all[idx_t] == int(s))),
                mean_mu_norm=mean_mu,
                max_mu_norm=max_mu,
                top_joint=str(bone_names[jtop]),
                top_mu_xyz=mu[int(jtop)].copy(),
                top_dom=_dom_axis_sign(mu[int(jtop)]),
            )
        )
    global_rows.sort(key=lambda r: float(r.max_mu_norm), reverse=True)
    top_global = global_rows[: int(max(0, args.hotspot_topk))]
    global_hot_sics = [int(r.sic) for r in top_global]

    # Load GT angvel.
    gt_names, gt_fps, omega_rad_s = _load_npz_angvel(npz_path)  # (T_npz,B,3)
    T_npz = int(omega_rad_s.shape[0])
    cycle_len = min(cycle_len, T_npz)
    omega_deg_s = omega_rad_s * (180.0 / math.pi)

    # Resolve joints.
    want_joints = [s.strip() for s in str(args.joints or "").split(",") if s.strip()]
    if not want_joints:
        want_joints = ["calf_l", "calf_r"]
    name_to_idx_gt = {str(n): int(i) for i, n in enumerate(gt_names)}
    name_to_idx_mu = {str(n): int(i) for i, n in enumerate(bone_names)}

    joints: List[Tuple[str, int, int]] = []  # (name, j_mu, j_gt)
    for n in want_joints:
        if n not in name_to_idx_gt or n not in name_to_idx_mu:
            continue
        joints.append((str(n), int(name_to_idx_mu[n]), int(name_to_idx_gt[n])))
    if not joints:
        raise SystemExit(f"No joints resolved from --joints={args.joints!r}.")

    ax_i = {"x": 0, "y": 1, "z": 2}[str(args.axis)]
    omega_min = float(args.omega_min_deg_s)
    fps = float(gt_fps) if gt_fps > 0 else 60.0

    x = np.arange(cycle_len, dtype=np.int64)

    # Per-joint series.
    per_joint: Dict[str, Dict[str, np.ndarray]] = {}
    for name, j_mu, j_gt in joints:
        mu_axis = np.full((cycle_len,), np.nan, dtype=np.float64)
        mu_norm = np.full((cycle_len,), np.nan, dtype=np.float64)
        mu_xyz = np.full((cycle_len, 3), np.nan, dtype=np.float64)
        for s in sics:
            if 0 <= int(s) < int(cycle_len):
                v = mu_by_sic[int(s)][int(j_mu)]
                mu_axis[int(s)] = float(v[int(ax_i)])
                mu_norm[int(s)] = float(np.linalg.norm(v))
                mu_xyz[int(s)] = v

        w = omega_deg_s[:cycle_len, int(j_gt), :]  # (T,3)
        w_axis = w[:, int(ax_i)].copy()
        w_mag = np.linalg.norm(w, axis=1)

        # Lag estimate (frames): dt = mu_axis / w_axis (sec), frames = dt * fps.
        dt_frames = np.full((cycle_len,), np.nan, dtype=np.float64)
        ok = np.isfinite(mu_axis) & np.isfinite(w_axis) & (np.abs(w_axis) >= omega_min)
        dt_frames[ok] = (mu_axis[ok] / w_axis[ok]) * fps
        d: Dict[str, np.ndarray] = {
            "mu_axis_deg": mu_axis,
            "mu_norm_deg": mu_norm,
            "mu_xyz_deg": mu_xyz,
            "omega_xyz_deg_s": w,
            "omega_axis_deg_s": w_axis,
            "omega_mag_deg_s": w_mag,
            "dt_frames": dt_frames,
        }

        if bool(args.projection_diag):
            dot_mu_omega = np.sum(mu_xyz * w, axis=1)  # deg*deg/s
            omega_sq = np.sum(w * w, axis=1)  # (deg/s)^2
            omega_norm = np.sqrt(omega_sq)
            safe_omega = np.isfinite(omega_sq) & (omega_sq > 1e-9)

            dt_star = np.full((cycle_len,), np.nan, dtype=np.float64)
            mu_parallel_scalar = np.full((cycle_len,), np.nan, dtype=np.float64)
            mu_parallel_xyz = np.full((cycle_len, 3), np.nan, dtype=np.float64)
            mu_perp_xyz = np.full((cycle_len, 3), np.nan, dtype=np.float64)
            r_perp = np.full((cycle_len,), np.nan, dtype=np.float64)

            ok_dt = np.isfinite(dot_mu_omega) & safe_omega
            dt_star[ok_dt] = (dot_mu_omega[ok_dt] / omega_sq[ok_dt]) * fps

            ok_para = np.isfinite(dot_mu_omega) & np.isfinite(omega_norm) & (omega_norm > 1e-9)
            mu_parallel_scalar[ok_para] = dot_mu_omega[ok_para] / omega_norm[ok_para]

            mu_parallel_xyz[ok_dt] = (
                w[ok_dt] * (dot_mu_omega[ok_dt] / omega_sq[ok_dt]).reshape(-1, 1)
            )
            mu_perp_xyz[ok_dt] = mu_xyz[ok_dt] - mu_parallel_xyz[ok_dt]

            mu_perp_norm = np.linalg.norm(mu_perp_xyz, axis=1)
            ok_r = np.isfinite(mu_norm) & (mu_norm > 1e-9) & np.isfinite(mu_perp_norm)
            r_perp[ok_r] = mu_perp_norm[ok_r] / mu_norm[ok_r]

            proj_gate = (
                np.isfinite(dot_mu_omega)
                & np.isfinite(mu_norm)
                & np.isfinite(omega_norm)
                & (dot_mu_omega > 0.0)
                & (mu_norm < float(args.projection_mu_max_deg))
                & (omega_norm >= float(omega_min))
            )

            d.update(
                {
                    "dt_star_frames": dt_star,
                    "r_perp": r_perp,
                    "mu_parallel_scalar_deg": mu_parallel_scalar,
                    "dot_mu_omega": dot_mu_omega,
                    "projection_gate": proj_gate,
                }
            )

        per_joint[name] = d

    # Quick summary stats: sign alignment and lag distribution on "hot" sics.
    quick: Dict[str, Dict[str, float]] = {}
    quick_projection: Dict[str, Dict[str, float]] = {}
    for name, _j_mu, _j_gt in joints:
        s = per_joint[name]
        mu_axis = s["mu_axis_deg"]
        mu_norm = s["mu_norm_deg"]
        w_axis = s["omega_axis_deg_s"]
        dt_frames = s["dt_frames"]

        mask_mu = np.isfinite(mu_norm) & (mu_norm >= float(mu_dom_thr)) & np.isfinite(mu_axis) & np.isfinite(w_axis)
        if int(np.sum(mask_mu)) > 0:
            align_frac = float(np.mean((mu_axis[mask_mu] * w_axis[mask_mu]) > 0.0))
        else:
            align_frac = float("nan")

        mask_dt = np.isfinite(dt_frames) & mask_mu
        if int(np.sum(mask_dt)) > 0:
            dtv = dt_frames[mask_dt]
            dt_med = float(np.median(dtv))
            dt_p25 = float(np.percentile(dtv, 25))
            dt_p75 = float(np.percentile(dtv, 75))
            dt_n = int(dtv.size)
        else:
            dt_med = float("nan")
            dt_p25 = float("nan")
            dt_p75 = float("nan")
            dt_n = 0

        quick[str(name)] = {
            "mu_thr_deg": float(mu_dom_thr),
            "omega_min_deg_s": float(omega_min),
            "mu_count": int(np.sum(mask_mu)),
            "align_frac": align_frac,
            "dt_median": dt_med,
            "dt_p25": dt_p25,
            "dt_p75": dt_p75,
            "dt_count": dt_n,
        }

        if bool(args.projection_diag):
            dt_star = s["dt_star_frames"]
            r_perp = s["r_perp"]
            mu_para = s["mu_parallel_scalar_deg"]
            gate = np.asarray(s["projection_gate"], dtype=bool)
            omega_mag = s["omega_mag_deg_s"]
            valid = np.isfinite(mu_norm) & np.isfinite(omega_mag)

            n_valid = int(np.sum(valid))
            n_gate = int(np.sum(gate))
            gate_frac = float(n_gate / n_valid) if n_valid > 0 else float("nan")

            dt_star_g = dt_star[gate & np.isfinite(dt_star)]
            rp_g = r_perp[gate & np.isfinite(r_perp)]
            mu_para_g = mu_para[gate & np.isfinite(mu_para)]
            om_g = omega_mag[gate & np.isfinite(mu_para) & np.isfinite(omega_mag)]

            if dt_star_g.size > 0:
                dt_star_med = float(np.median(dt_star_g))
                dt_star_p25 = float(np.percentile(dt_star_g, 25))
                dt_star_p75 = float(np.percentile(dt_star_g, 75))
            else:
                dt_star_med = float("nan")
                dt_star_p25 = float("nan")
                dt_star_p75 = float("nan")

            if rp_g.size > 0:
                r_perp_med = float(np.median(rp_g))
                r_perp_p90 = float(np.percentile(rp_g, 90))
            else:
                r_perp_med = float("nan")
                r_perp_p90 = float("nan")

            if mu_para_g.size > 0:
                mu_para_med = float(np.median(mu_para_g))
                mu_para_std = float(np.std(mu_para_g))
            else:
                mu_para_med = float("nan")
                mu_para_std = float("nan")

            fit_a, fit_b, fit_r2, fit_n = _fit_linear(om_g, mu_para_g)

            quick_projection[str(name)] = {
                "n_valid": float(n_valid),
                "n_gate": float(n_gate),
                "gate_frac": gate_frac,
                "dt_star_median": dt_star_med,
                "dt_star_p25": dt_star_p25,
                "dt_star_p75": dt_star_p75,
                "r_perp_median": r_perp_med,
                "r_perp_p90": r_perp_p90,
                "mu_parallel_median": mu_para_med,
                "mu_parallel_std": mu_para_std,
                "fit_a": fit_a,
                "fit_b": fit_b,
                "fit_r2": fit_r2,
                "fit_n": float(fit_n),
            }

    gt_symmetry: Optional[Dict[str, Any]] = None
    if bool(args.gt_symmetry_sanity):
        gt_sym_joints = [s.strip() for s in str(args.gt_sym_joints or "").split(",") if s.strip()]
        if not gt_sym_joints:
            gt_sym_joints = [str(name) for name, _j_mu, _j_gt in joints]
        gt_shifts = _parse_int_csv(str(args.gt_shifts))
        if not gt_shifts:
            gt_shifts = [1, -1]
        try:
            gt_symmetry = _compute_gt_symmetry_sanity(
                npz_path=npz_path,
                gt_names=gt_names,
                omega_deg_s=omega_deg_s,
                joints=gt_sym_joints,
                axis_i=int(ax_i),
                mu_thr_deg=float(mu_dom_thr),
                omega_min_deg_s=float(omega_min),
                fps=float(fps),
                shifts=gt_shifts,
            )
        except Exception as ex:
            gt_symmetry = {"available": False, "reason": f"error: {type(ex).__name__}: {ex}"}

    # ---- Plot ----------------------------------------------------------------
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(len(joints), 1, figsize=(11, 3.8 * len(joints)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (name, _j_mu, _j_gt) in zip(axes, joints):
        s = per_joint[name]
        w_axis = s["omega_axis_deg_s"]
        w_mag = s["omega_mag_deg_s"]
        mu_n = s["mu_norm_deg"]

        ax.plot(x, w_axis, color="C0", lw=1.6, label=f"{name}: omega_{args.axis} (deg/s)")
        ax.plot(x, w_mag, color="C0", lw=1.0, ls="--", alpha=0.45, label=f"{name}: |omega| (deg/s)")
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.25)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("GT angvel (deg/s)")

        ax2 = ax.twinx()
        ax2.plot(x, mu_n, color="C3", lw=1.5, label=f"{name}: ||mu_sic|| (deg)")
        ax2.set_ylabel("||mu_sic|| (deg)")

        for sic_h in global_hot_sics:
            if 0 <= int(sic_h) < int(cycle_len):
                ax.axvline(int(sic_h), color="0.35", lw=0.8, ls=":", alpha=0.30)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
        ax.set_title(f"SIC overlay: {name}  (branch={args.branch}, space={args.space})")

    axes[-1].set_xlabel("step_in_cycle (sic)")
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    projection_figs: Dict[str, Path] = {}
    if bool(args.projection_diag):
        scatter_dir = out_fig.parent / f"{out_fig.stem}_projection"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        for name, _j_mu, _j_gt in joints:
            s = per_joint[name]
            gate = np.asarray(s.get("projection_gate", np.zeros((cycle_len,), dtype=bool)), dtype=bool)
            x_omega = s["omega_mag_deg_s"]
            y_mu_para = s["mu_parallel_scalar_deg"]
            sic_plot = x
            m = gate & np.isfinite(x_omega) & np.isfinite(y_mu_para) & np.isfinite(sic_plot)
            if int(np.sum(m)) <= 0:
                continue

            fit_a, fit_b, fit_r2, fit_n = _fit_linear(x_omega[m], y_mu_para[m])
            xx = np.linspace(float(np.nanmin(x_omega[m])), float(np.nanmax(x_omega[m])), 128)
            yy = fit_a + fit_b * xx

            fig_s, ax_s = plt.subplots(figsize=(6.2, 4.8))
            sc = ax_s.scatter(
                x_omega[m],
                y_mu_para[m],
                c=sic_plot[m],
                cmap="viridis",
                s=34,
                alpha=0.90,
                edgecolors="none",
            )
            ax_s.plot(xx, yy, color="C3", lw=1.4, label=f"fit: y={fit_a:.3f}+{fit_b:.3f}x (R^2={fit_r2:.3f}, N={fit_n})")
            ax_s.set_xlabel("||omega|| (deg/s)")
            ax_s.set_ylabel("mu_parallel_scalar (deg)")
            ax_s.grid(True, alpha=0.25)
            ax_s.set_title(f"Projection diag: {name} (gated subset)")
            ax_s.legend(loc="best", fontsize=8)
            cb = fig_s.colorbar(sc, ax=ax_s)
            cb.set_label("sic")
            fig_s.tight_layout()

            out_sc = scatter_dir / f"{name}_mu_parallel_vs_omega.png"
            fig_s.savefig(out_sc, dpi=170)
            plt.close(fig_s)
            projection_figs[str(name)] = out_sc

    # ---- Markdown ------------------------------------------------------------
    lines: List[str] = []
    lines.append("# Walk_F / Stage7: SIC hotspots vs GT knee angular-velocity overlay")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append("")
    lines.append("Goal: relate phase-locked SO(3) error bias at specific sic to GT angular-velocity peaks/sign flips.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- freerun json: `{fr_path}`")
    lines.append(f"- npz: `{npz_path}` (FPS={fps:.1f})")
    lines.append(f"- branch/space: `{args.branch}` / `{args.space}`")
    lines.append("")
    lines.append("## Mask / protocol")
    lines.append("")
    lines.append(f"- cycle >= {int(args.min_cycle)}")
    lines.append(f"- exclude_wrap = {bool(args.exclude_wrap)}")
    lines.append(f"- exclude_root = {bool(args.exclude_root)} (root_idx={root_idx})")
    lines.append(f"- projection_diag = {bool(args.projection_diag)}")
    if bool(args.projection_diag):
        lines.append(
            "- projection gate: sign(mu*omega)>0, ||mu||<{m:.1f} deg, ||omega||>={w:.1f} deg/s".format(
                m=float(args.projection_mu_max_deg),
                w=float(omega_min),
            )
        )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(f"- png: `{out_fig}`")
    if bool(args.projection_diag):
        if projection_figs:
            lines.append("- projection scatter(s):")
            for name, p in projection_figs.items():
                lines.append(f"  - {name}: `{p}`")
        else:
            lines.append("- projection scatter(s): NA (no gated samples)")
    lines.append("")

    lines.append("## Quick summary")
    lines.append("")
    lines.append(f"Thresholds: `||mu_sic|| >= {mu_dom_thr:.3f} deg`, `|omega_{args.axis}| >= {omega_min:.1f} deg/s`.")
    lines.append("")
    for name, _j_mu, _j_gt in joints:
        q = quick.get(str(name), {})
        lines.append(f"- {name}:")
        lines.append(f"  - sign(mu_{args.axis} * omega_{args.axis}) > 0 fraction = {q.get('align_frac', float('nan')):.3f} (N_mu={int(q.get('mu_count', 0))})")
        if int(q.get("dt_count", 0)) > 0 and math.isfinite(float(q.get("dt_median", float("nan")))):
            lines.append(
                "  - dt_frames = (mu_{ax}/omega_{ax}) * FPS: median={med:.3f}, IQR=[{p25:.3f}, {p75:.3f}] (N_dt={n})".format(
                    ax=str(args.axis),
                    med=float(q["dt_median"]),
                    p25=float(q["dt_p25"]),
                    p75=float(q["dt_p75"]),
                    n=int(q["dt_count"]),
                )
            )
        else:
            lines.append("  - dt_frames = NA (insufficient points under |omega| threshold)")
        if bool(args.projection_diag):
            qp = quick_projection.get(str(name), {})
            n_valid = int(qp.get("n_valid", 0))
            n_gate = int(qp.get("n_gate", 0))
            gate_frac = float(qp.get("gate_frac", float("nan")))
            lines.append(
                "  - proj_gate(sign(mu*omega)>0, ||mu||<{m:.1f}deg, ||omega||>={w:.1f}deg/s): N={n}/{v} ({f:.3f})".format(
                    m=float(args.projection_mu_max_deg),
                    w=float(omega_min),
                    n=n_gate,
                    v=n_valid,
                    f=gate_frac,
                )
            )

            if n_gate > 0 and math.isfinite(float(qp.get("dt_star_median", float("nan")))):
                dt_star_med = float(qp.get("dt_star_median", float("nan")))
                dt_star_p25 = float(qp.get("dt_star_p25", float("nan")))
                dt_star_p75 = float(qp.get("dt_star_p75", float("nan")))
                r_med = float(qp.get("r_perp_median", float("nan")))
                r_p90 = float(qp.get("r_perp_p90", float("nan")))
                mu_p_med = float(qp.get("mu_parallel_median", float("nan")))
                mu_p_std = float(qp.get("mu_parallel_std", float("nan")))
                fit_a = float(qp.get("fit_a", float("nan")))
                fit_b = float(qp.get("fit_b", float("nan")))
                fit_r2 = float(qp.get("fit_r2", float("nan")))
                fit_n = int(qp.get("fit_n", 0))
                unreliable_tag = " [unreliable]" if (math.isfinite(r_med) and r_med > float(args.projection_unreliable_r_perp)) else ""

                lines.append(
                    "  - dt* = (mu*omega/||omega||^2)*FPS: median={med:.3f}, IQR=[{p25:.3f}, {p75:.3f}] (N={n}){tag}".format(
                        med=dt_star_med,
                        p25=dt_star_p25,
                        p75=dt_star_p75,
                        n=n_gate,
                        tag=unreliable_tag,
                    )
                )
                lines.append(f"  - r_perp = ||mu_perp||/||mu||: median={r_med:.3f}, p90={r_p90:.3f}")
                lines.append(f"  - mu_parallel_scalar = mu*omega_hat: median={mu_p_med:.3f} deg, std={mu_p_std:.3f}")
                lines.append(f"  - H1/H2 fit (mu_parallel=a+b*||omega||): a={fit_a:.4f}, b={fit_b:.4f}, R^2={fit_r2:.3f} (N={fit_n})")
            else:
                lines.append("  - dt* / r_perp / H1-H2 fit = NA (no samples in projection gate)")
    lines.append("")
    lines.append(
        "Interpretation note: if the small-angle approximation `mu ~= omega * dt` holds, "
        "then `dt_frames>0` suggests phase-lag (pred behind GT along motion direction), "
        "while `dt_frames<0` suggests phase-lead / opposite-sign bias."
    )
    if bool(args.projection_diag):
        lines.append(
            "Projection note: `dt*` is only interpreted on the gated subset; "
            "when `r_perp(median)` is high, most error energy is orthogonal to omega and lag interpretation is weak."
        )
    lines.append("")

    lines.append("## GT symmetry sanity (data-only)")
    lines.append("")
    if bool(args.gt_symmetry_sanity):
        if not isinstance(gt_symmetry, dict) or not bool(gt_symmetry.get("available", False)):
            reason = ""
            if isinstance(gt_symmetry, dict):
                reason = str(gt_symmetry.get("reason", "") or "")
            lines.append(f"- NA: {reason or 'failed'}")
        else:
            gt_joints = [str(x) for x in (gt_symmetry.get("joints") or [])]
            lines.append("- Method: synthetic lag on GT (`pred(t)=gt(t-k)` with edge padding), then apply the same dt estimator.")
            lines.append(
                "- thresholds: `||mu||>={m:.3f} deg`, `|omega_{ax}|>={w:.1f} deg/s`".format(
                    m=float(gt_symmetry.get("mu_thr_deg", mu_dom_thr)),
                    ax=str(gt_symmetry.get("axis", args.axis)),
                    w=float(gt_symmetry.get("omega_min_deg_s", omega_min)),
                )
            )
            if gt_joints:
                lines.append(f"- joints: `{','.join(gt_joints)}`")
            if len(gt_joints) >= 2:
                j0, j1 = gt_joints[0], gt_joints[1]
                lines.append("")
                lines.append(f"|k (frames)|{j0} dt_med|{j1} dt_med|{j0} align|{j1} align|same_sign|")
                lines.append("|---:|---:|---:|---:|---:|:---:|")
                for row in gt_symmetry.get("shift_rows", []):
                    if not isinstance(row, dict):
                        continue
                    k = int(row.get("k", 0))
                    jj = row.get("joints", {}) if isinstance(row.get("joints", {}), dict) else {}
                    s0 = jj.get(j0, {}) if isinstance(jj.get(j0, {}), dict) else {}
                    s1 = jj.get(j1, {}) if isinstance(jj.get(j1, {}), dict) else {}

                    def _fnum(v: Any, p: int = 3) -> str:
                        try:
                            x = float(v)
                        except Exception:
                            return "NA"
                        return f"{x:.{p}f}" if math.isfinite(x) else "NA"

                    lines.append(
                        "|{k}|{d0}|{d1}|{a0}|{a1}|{sgn}|".format(
                            k=k,
                            d0=_fnum(s0.get("dt_median", float("nan")), 3),
                            d1=_fnum(s1.get("dt_median", float("nan")), 3),
                            a0=_fnum(s0.get("align_frac", float("nan")), 3),
                            a1=_fnum(s1.get("align_frac", float("nan")), 3),
                            sgn=("Y" if bool(row.get("same_sign", False)) else "N"),
                        )
                    )
            omega_snap = gt_symmetry.get("omega_snapshot", {})
            if isinstance(omega_snap, dict):
                j_stats = omega_snap.get("joints", {})
                if isinstance(j_stats, dict):
                    for jn in gt_joints[:2]:
                        jv = j_stats.get(jn, {})
                        if not isinstance(jv, dict):
                            continue
                        try:
                            mabs = float(jv.get("mean_abs_omega_axis_deg_s", float("nan")))
                        except Exception:
                            mabs = float("nan")
                        if math.isfinite(mabs):
                            lines.append(f"- mean(|omega_{gt_symmetry.get('axis', args.axis)}|) {jn} = {mabs:.3f} deg/s")
                lr_corr = omega_snap.get("lr_axis_roll_corr", {})
                if isinstance(lr_corr, dict):
                    try:
                        corr = float(lr_corr.get("best_corr", float("nan")))
                        shift = int(lr_corr.get("best_shift", 0))
                        cyc = int(lr_corr.get("cycle_len", 0))
                        ja = str(lr_corr.get("joint_a", "joint_a"))
                        jb = str(lr_corr.get("joint_b", "joint_b"))
                    except Exception:
                        corr = float("nan")
                        shift = 0
                        cyc = 0
                        ja = "joint_a"
                        jb = "joint_b"
                    if math.isfinite(corr):
                        lines.append(f"- best corr(omega_{ja}, roll(omega_{jb}, s)) = {corr:.3f} at s={shift} (cycle len={cyc})")
    else:
        lines.append("- disabled by `--no-gt-symmetry-sanity`")
    lines.append("")

    if bool(args.projection_diag):
        lines.append("## Projection diagnostics summary (gated subset)")
        lines.append("")
        lines.append(
            "|joint|N_gate/N_valid|dt* median (IQR)|r_perp median/p90|mu_parallel median+/-std (deg)|fit a|fit b|R^2|reliability|"
        )
        lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|:---|")
        for name, _j_mu, _j_gt in joints:
            qp = quick_projection.get(str(name), {})
            n_valid = int(qp.get("n_valid", 0))
            n_gate = int(qp.get("n_gate", 0))

            dt_med = float(qp.get("dt_star_median", float("nan")))
            dt_p25 = float(qp.get("dt_star_p25", float("nan")))
            dt_p75 = float(qp.get("dt_star_p75", float("nan")))
            dt_text = "NA"
            if math.isfinite(dt_med):
                dt_text = f"{dt_med:.3f} [{dt_p25:.3f}, {dt_p75:.3f}]"

            r_med = float(qp.get("r_perp_median", float("nan")))
            r_p90 = float(qp.get("r_perp_p90", float("nan")))
            r_text = "NA" if not math.isfinite(r_med) else f"{r_med:.3f}/{r_p90:.3f}"

            mu_med = float(qp.get("mu_parallel_median", float("nan")))
            mu_std = float(qp.get("mu_parallel_std", float("nan")))
            mu_text = "NA" if not math.isfinite(mu_med) else f"{mu_med:.3f}+/-{mu_std:.3f}"

            fit_a = float(qp.get("fit_a", float("nan")))
            fit_b = float(qp.get("fit_b", float("nan")))
            fit_r2 = float(qp.get("fit_r2", float("nan")))
            fit_a_text = "NA" if not math.isfinite(fit_a) else f"{fit_a:.4f}"
            fit_b_text = "NA" if not math.isfinite(fit_b) else f"{fit_b:.4f}"
            fit_r2_text = "NA" if not math.isfinite(fit_r2) else f"{fit_r2:.3f}"

            reliability = "unknown"
            if math.isfinite(r_med):
                reliability = "unreliable" if r_med > float(args.projection_unreliable_r_perp) else "usable"

            lines.append(
                "|{name}|{n_gate}/{n_valid}|{dt}|{r}|{mu}|{a}|{b}|{r2}|{rel}|".format(
                    name=str(name),
                    n_gate=n_gate,
                    n_valid=n_valid,
                    dt=dt_text,
                    r=r_text,
                    mu=mu_text,
                    a=fit_a_text,
                    b=fit_b_text,
                    r2=fit_r2_text,
                    rel=reliability,
                )
            )
        lines.append("")

    # Global hotspots table
    lines.append(f"## Global SIC hotspots (Top {len(top_global)} by max_j ||mu_sic,j||)")
    lines.append("")
    lines.append("|rank|sic|N|mean||mu|| (deg)|max||mu|| (deg)|top_joint|top mu_xyz (deg)|dom|")
    lines.append("|---:|---:|---:|---:|---:|:---|:---|:---:|")
    for rank, r in enumerate(top_global, start=1):
        lines.append(
            "|{rank}|{sic}|{N}|{mean:.3f}|{mx:.3f}|{j}|`{mu}`|{dom}|".format(
                rank=int(rank),
                sic=int(r.sic),
                N=int(r.N),
                mean=float(r.mean_mu_norm),
                mx=float(r.max_mu_norm),
                j=str(r.top_joint),
                mu=_fmt_vec3(r.top_mu_xyz, prec=3),
                dom=str(r.top_dom),
            )
        )
    lines.append("")

    # Per-joint tables
    lines.append("## Per-joint SIC tables (mu vs omega)")
    lines.append("")
    for name, j_mu, j_gt in joints:
        s = per_joint[name]
        mu_axis = s["mu_axis_deg"]
        mu_norm = s["mu_norm_deg"]
        w_axis = s["omega_axis_deg_s"]
        w_mag = s["omega_mag_deg_s"]
        dt_frames = s["dt_frames"]
        dt_star = s.get("dt_star_frames", np.full((cycle_len,), np.nan, dtype=np.float64))
        r_perp = s.get("r_perp", np.full((cycle_len,), np.nan, dtype=np.float64))
        mu_para = s.get("mu_parallel_scalar_deg", np.full((cycle_len,), np.nan, dtype=np.float64))

        corr_mu_axis = _safe_corr(mu_axis[np.isfinite(mu_axis)], w_axis[np.isfinite(mu_axis)])
        corr_mu_norm = _safe_corr(mu_norm[np.isfinite(mu_norm)], w_mag[np.isfinite(mu_norm)])

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- corr(mu_{args.axis}, omega_{args.axis}) = {corr_mu_axis:.3f}")
        lines.append(f"- corr(||mu||, |omega|) = {corr_mu_norm:.3f}")
        lines.append(f"- lag estimate uses |omega_{args.axis}| >= {omega_min:.1f} deg/s; dt_frames = (mu_{args.axis}/omega_{args.axis}) * FPS")
        lines.append("")
        if bool(args.projection_diag):
            lines.append(
                "|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|dt* (frames)|r_perp|mu_parallel (deg)|sign(mu_axis*omega_axis)|"
            )
            lines.append("|---:|---:|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|")
        else:
            lines.append("|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|sign(mu_axis*omega_axis)|")
            lines.append("|---:|---:|---:|---:|:---|:---:|---:|---:|---:|:---:|")

        order = np.argsort(-np.nan_to_num(mu_norm, nan=-1.0))
        kk = 0
        for sic_i in order.tolist():
            if kk >= 15:
                break
            if not np.isfinite(mu_norm[int(sic_i)]) or float(mu_norm[int(sic_i)]) < 0:
                continue
            if int(sic_i) not in mu_by_sic:
                continue
            mu_j = mu_by_sic[int(sic_i)][int(j_mu)]
            mu_dom = _dom_axis_sign(mu_j)
            a = float(mu_axis[int(sic_i)])
            o = float(w_axis[int(sic_i)])
            sgn = "0" if (abs(a * o) < 1e-9) else ("+" if (a * o) > 0 else "-")
            dtf = dt_frames[int(sic_i)]
            dtf_s = "NA" if not np.isfinite(dtf) else f"{dtf:.3f}"
            if bool(args.projection_diag):
                dts = dt_star[int(sic_i)]
                dts_s = "NA" if not np.isfinite(dts) else f"{dts:.3f}"
                rp = r_perp[int(sic_i)]
                rp_s = "NA" if not np.isfinite(rp) else f"{rp:.3f}"
                mp = mu_para[int(sic_i)]
                mp_s = "NA" if not np.isfinite(mp) else f"{mp:.3f}"
                lines.append(
                    "|{rank}|{sic}|{mn:.3f}|{ma:.3f}|`{mxyz}`|{dom}|{oa:.1f}|{om:.1f}|{dt}|{dt_star}|{r_perp}|{mu_p}|{sgn}|".format(
                        rank=int(kk + 1),
                        sic=int(sic_i),
                        mn=float(mu_norm[int(sic_i)]),
                        ma=float(mu_axis[int(sic_i)]),
                        mxyz=_fmt_vec3(mu_j, prec=3),
                        dom=mu_dom,
                        oa=float(w_axis[int(sic_i)]),
                        om=float(w_mag[int(sic_i)]),
                        dt=dtf_s,
                        dt_star=dts_s,
                        r_perp=rp_s,
                        mu_p=mp_s,
                        sgn=sgn,
                    )
                )
            else:
                lines.append(
                    "|{rank}|{sic}|{mn:.3f}|{ma:.3f}|`{mxyz}`|{dom}|{oa:.1f}|{om:.1f}|{dt}|{sgn}|".format(
                        rank=int(kk + 1),
                        sic=int(sic_i),
                        mn=float(mu_norm[int(sic_i)]),
                        ma=float(mu_axis[int(sic_i)]),
                        mxyz=_fmt_vec3(mu_j, prec=3),
                        dom=mu_dom,
                        oa=float(w_axis[int(sic_i)]),
                        om=float(w_mag[int(sic_i)]),
                        dt=dtf_s,
                        sgn=sgn,
                    )
                )
            kk += 1
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {out_fig}")
    if bool(args.projection_diag):
        if projection_figs:
            print(f"[OK] wrote {len(projection_figs)} projection scatter figure(s) under {next(iter(projection_figs.values())).parent}")
        else:
            print("[OK] projection diagnostics requested, but no gated samples were found.")


if __name__ == "__main__":
    main()
