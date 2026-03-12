#!/usr/bin/env python3
"""
Experiment 2: delta* predictability (offline regressors).

We treat per-step oracle hinge correction delta* (deg) as a label and fit simple
linear regressors under different feature sets:
  - const                     : per-bone bias
  - phase                     : sin/cos phase (optionally multiple harmonics)
  - cond                      : teacher cond features
  - cond+phase                : cond + phase
  - plan+meas+phase           : (ContactPlanPerC, ContactMeasPerC) + phase
  - cond+plan+meas+phase      : cond + (plan,meas) + phase

Evaluation is done in terms of:
  - delta error: MAE/RMSE vs delta*
  - angle after applying delta_hat: mean ang, P(ang>th) on the selected subset

Input freerun JSON must include:
  keybone_omega.series.branches.{branch}.omega_deg_xyz[bone][t] = [wx, wy, wz]  (deg)
and metrics_per_step for cycle/contact filtering.

Optional (recommended) additional context features (exported by run_freerun_cycles.py):
  - plan_state_series.series.plan_z_in / phase_z_in / phase_event_age_in
  - keybone_state.series.branches.{inc|direct|blend}.pred_rotvec_deg_xyz

For cond features we load the teacher JSON pointed by `teacher_json` in freerun JSON.
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
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.geometry import so3_exp_map  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_axis_vec(axis: str) -> np.ndarray:
    a = str(axis).strip().lower()
    if a not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    idx = {"x": 0, "y": 1, "z": 2}[a]
    v = np.zeros(3, dtype=np.float32)
    v[idx] = 1.0
    return v


def _angle_from_R(R: torch.Tensor) -> torch.Tensor:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    skew = R - R.transpose(-1, -2)
    vec = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) * 0.5
    sin = vec.norm(dim=-1)
    return torch.atan2(sin, cos)


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

        # Optional: contact filter.
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


def _ridge_solve(X: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    """
    Solve linear regression (ridge) for y ~ X w.
    - X: (N,D), y: (N,)
    - returns w: (D,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    y = y.reshape(-1)
    N, D = X.shape
    if y.shape[0] != N:
        raise ValueError(f"X/y length mismatch: {X.shape} vs {y.shape}")

    if not (alpha > 0.0):
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return w.astype(np.float32, copy=False)

    XtX = X.T @ X
    Xty = X.T @ y
    reg = float(alpha) * np.eye(D, dtype=np.float32)
    w = np.linalg.solve(XtX + reg, Xty)
    return w.astype(np.float32, copy=False)


@dataclass
class SampleSet:
    w_deg_xyz: np.ndarray  # (N,3)
    ang_deg: np.ndarray  # (N,)
    delta_star_deg: np.ndarray  # (N,)
    # Indices into the original per-step arrays (metrics_per_step / exported series).
    step_idx: np.ndarray  # (N,)
    step_in_cycle: np.ndarray  # (N,)
    cycle_len: int
    phasez_sincos: Optional[np.ndarray]  # (N,2) from metrics_per_step.PhaseZSinCosPerC[{l|r}]
    cond: Optional[np.ndarray]  # (N,Dc)
    plan: Optional[np.ndarray]  # (N,C)
    meas: Optional[np.ndarray]  # (N,C)
    # Optional: internal state/context exports (mean-over-batch).
    plan_z_in: Optional[np.ndarray]  # (N,Dz)
    phase_z_in: Optional[np.ndarray]  # (N,Dp)
    phase_event_age_in: Optional[np.ndarray]  # (N,Da)
    # Optional: per-step predicted joint-local pose state exports (flattened rotvec_deg_xyz).
    # Shapes: (N, 3*K) where K == len(keybone_state_bones).
    keybone_state_bones: Optional[List[str]]
    keybone_state_rotvec_deg_xyz: Optional[Dict[str, np.ndarray]]  # branch -> (N, 3*K)


def _resolve_teacher_path(freerun_json: Path, teacher_json: str) -> Optional[Path]:
    cand: List[Path] = []
    t = Path(teacher_json)
    if t.is_absolute():
        cand.append(t)
    else:
        cand.append((_PROJECT_ROOT / t).resolve())
        cand.append((Path.cwd() / t).resolve())
        cand.append((freerun_json.parent / t).resolve())
    for p in cand:
        if p.is_file():
            return p
    return None


def _load_teacher_cond(teacher_path: Path) -> np.ndarray:
    obj = _load_json(teacher_path)
    t = obj.get("teacher", None)
    if not isinstance(t, dict):
        raise ValueError(f"{teacher_path}: invalid teacher json (missing teacher dict)")
    cond = np.asarray(t.get("cond", []), dtype=np.float32)
    if cond.ndim != 2:
        raise ValueError(f"{teacher_path}: teacher.cond must be 2D, got {cond.shape}")
    return cond


def _stack_contact(xs: List[List[float]]) -> Optional[np.ndarray]:
    if not xs:
        return None
    m = max((len(v) for v in xs), default=0)
    if m <= 0:
        return None
    arr = np.zeros((len(xs), m), dtype=np.float32)
    for i, v in enumerate(xs):
        if not v:
            continue
        vv = np.asarray(v, dtype=np.float32).reshape(-1)
        arr[i, : min(m, vv.shape[0])] = vv[:m]
    return arr


def _build_samples(
    path: Path,
    *,
    bone: str,
    branch: str,
    axis: str,
    max_deg: float,
    grid_step_deg: float,
    min_cycle: int,
    contact_source: str,
    contact_side: Optional[str],
    contact_value: Optional[int],
    contact_thresh: float,
) -> SampleSet:
    obj = _load_json(path)

    ko = obj.get("keybone_omega", None)
    if not isinstance(ko, dict):
        raise ValueError(f"{path}: missing keybone_omega (run freerun with --export_keybone_omega)")
    series = ko.get("series", None)
    if not isinstance(series, dict):
        raise ValueError(f"{path}: missing keybone_omega.series (run with --export_keybone_omega_series)")
    sbranches = series.get("branches", None)
    if not isinstance(sbranches, dict):
        raise ValueError(f"{path}: invalid keybone_omega.series (missing branches)")
    bdat = sbranches.get(str(branch), None)
    if not isinstance(bdat, dict):
        raise ValueError(f"{path}: missing keybone_omega.series.branches.{branch}")
    omega_xyz_map = bdat.get("omega_deg_xyz", None)
    if not isinstance(omega_xyz_map, dict):
        raise ValueError(f"{path}: missing omega_deg_xyz (re-run freerun after enabling omega_deg_xyz export)")
    wdeg = omega_xyz_map.get(str(bone), None)
    if not isinstance(wdeg, list) or not wdeg:
        raise ValueError(f"{path}: missing omega_deg_xyz for bone={bone!r} in branch={branch!r}")

    steps = obj.get("metrics_per_step", None)
    if not isinstance(steps, list) or not steps:
        steps = None
    cycle_len = int(obj.get("cycle_len", 0) or 0)

    min_len = min(len(wdeg), len(steps) if steps is not None else len(wdeg))

    contact_key = {
        "gt": "ContactGTPerC",
        "plan": "ContactPlanPerC",
        "meas": "ContactMeasPerC",
    }.get(str(contact_source).strip().lower(), "ContactGTPerC")

    contact_idx: Optional[int] = None
    if contact_side is not None:
        contact_idx = 0 if str(contact_side).strip().lower() == "l" else 1
    elif contact_value is not None:
        contact_idx = 1

    sel_idx = _select_step_indices(
        steps=steps,
        cycle_len=cycle_len,
        min_len=min_len,
        min_cycle=int(min_cycle),
        contact_key=contact_key,
        contact_idx=contact_idx,
        contact_value=int(contact_value) if contact_value is not None else None,
        contact_thresh=float(contact_thresh),
    )
    if not sel_idx:
        raise ValueError(f"{path}: no steps selected (check filters)")

    w_sel: List[List[float]] = []
    step_idx: List[int] = []
    step_in_cycle: List[int] = []
    phasez_sincos: List[List[float]] = []
    plan: List[List[float]] = []
    meas: List[List[float]] = []

    for i in sel_idx:
        v = wdeg[i]
        if not (isinstance(v, list) and len(v) == 3):
            continue
        try:
            w_sel.append([float(v[0]), float(v[1]), float(v[2])])
        except Exception:
            continue
        step_idx.append(int(i))

        ph = None
        if steps is not None and i < len(steps):
            si = steps[i].get("step_in_cycle", None)
            if isinstance(si, int):
                ph = int(si)
        if ph is None and cycle_len > 0:
            ph = int(i % cycle_len)
        step_in_cycle.append(int(ph or 0))

        # Optional: PhaseZ sin/cos per contact channel (often more semantically consistent than step_in_cycle across clips).
        pz_sc = None
        if steps is not None and i < len(steps) and contact_idx is not None:
            try:
                pz = steps[i].get("PhaseZSinCosPerC", None)
                if isinstance(pz, list) and 0 <= int(contact_idx) < len(pz):
                    v = pz[int(contact_idx)]
                    if isinstance(v, list) and len(v) == 2:
                        pz_sc = [float(v[0]), float(v[1])]
            except Exception:
                pz_sc = None
        if pz_sc is None:
            # Unknown / missing => leave zeros (caller can decide to use/ignore).
            pz_sc = [0.0, 0.0]
        phasez_sincos.append(pz_sc)

        if steps is not None and i < len(steps):
            for key, out in [("ContactPlanPerC", plan), ("ContactMeasPerC", meas)]:
                vv = steps[i].get(key, None)
                if isinstance(vv, list) and vv:
                    try:
                        out.append([float(x) for x in vv])
                    except Exception:
                        out.append([])
                else:
                    out.append([])

    if not w_sel:
        raise ValueError(f"{path}: omega_deg_xyz entries are invalid/empty after filtering")

    # Base angle from reconstructed R_err (more faithful than ||w|| if log-map was clipped).
    w = torch.tensor(w_sel, dtype=torch.float32)
    w_rad = w * (math.pi / 180.0)
    with torch.no_grad():
        R_err = so3_exp_map(w_rad)  # (N,3,3)
        ang_base_deg = (_angle_from_R(R_err) * (180.0 / math.pi)).cpu().numpy().astype(np.float32)

    # Oracle delta* via grid search: exp(-delta*axis) @ R_err.
    axis_vec_np = _as_axis_vec(axis)
    axis_vec = torch.tensor(axis_vec_np, dtype=torch.float32)

    D = int(round((2.0 * float(max_deg)) / float(grid_step_deg))) + 1
    delta_deg_grid = torch.linspace(-float(max_deg), float(max_deg), D, dtype=torch.float32)
    delta_rad_grid = delta_deg_grid * (math.pi / 180.0)
    with torch.no_grad():
        R_corr = so3_exp_map((-delta_rad_grid[:, None]) * axis_vec[None, :])  # (D,3,3)
        R_new = torch.matmul(R_corr[:, None, :, :], R_err[None, :, :, :])  # (D,N,3,3)
        ang_new_deg = _angle_from_R(R_new) * (180.0 / math.pi)  # (D,N)
        best_idx = ang_new_deg.argmin(dim=0)  # (N,)
        best_delta_deg = delta_deg_grid[best_idx].cpu().numpy().astype(np.float32)

    # cond features from teacher_json (optional).
    cond_feat: Optional[np.ndarray] = None
    teacher_json = obj.get("teacher_json", None)
    if isinstance(teacher_json, str) and teacher_json:
        tpath = _resolve_teacher_path(path, teacher_json)
        if tpath is not None:
            cond_cycle = _load_teacher_cond(tpath)  # (L,Dc)
            ph = np.asarray(step_in_cycle, dtype=np.int64)
            ph = np.clip(ph, 0, cond_cycle.shape[0] - 1)
            cond_feat = cond_cycle[ph].astype(np.float32, copy=False)

    # Optional: plan/phase internal state exports (mean-over-batch per step).
    plan_z_in: Optional[np.ndarray] = None
    phase_z_in: Optional[np.ndarray] = None
    phase_event_age_in: Optional[np.ndarray] = None
    try:
        ps = obj.get("plan_state_series", None)
        if isinstance(ps, dict):
            series = ps.get("series", None)
            if isinstance(series, dict) and step_idx:
                N = int(len(step_idx))

                def _extract_pack(key: str) -> Optional[np.ndarray]:
                    pack = series.get(key, None)
                    if not isinstance(pack, dict):
                        return None
                    data = pack.get("data", None)
                    valid = pack.get("valid", None)
                    dim = int(pack.get("dim", 0) or 0)
                    if dim <= 0 or not isinstance(data, list) or not isinstance(valid, list):
                        return None
                    out = np.zeros((N, dim), dtype=np.float32)
                    for r, t in enumerate(step_idx):
                        if t < 0 or t >= len(data) or t >= len(valid):
                            continue
                        if int(valid[t]) != 1:
                            continue
                        v = data[t]
                        if not (isinstance(v, list) and len(v) == dim):
                            continue
                        out[r] = np.asarray(v, dtype=np.float32)
                    return out

                plan_z_in = _extract_pack("plan_z_in")
                phase_z_in = _extract_pack("phase_z_in")
                phase_event_age_in = _extract_pack("phase_event_age_in")
    except Exception:
        plan_z_in = None
        phase_z_in = None
        phase_event_age_in = None

    # Optional: keybone predicted pose state exports (per branch, selected bones).
    keybone_state_bones: Optional[List[str]] = None
    keybone_state_rotvec_deg_xyz: Optional[Dict[str, np.ndarray]] = None
    try:
        ks = obj.get("keybone_state", None)
        if isinstance(ks, dict):
            s = ks.get("series", None)
            if isinstance(s, dict):
                bones = s.get("bones", None)
                branches = s.get("branches", None)
                if isinstance(bones, list) and isinstance(branches, dict) and bones and step_idx:
                    bones = [str(b) for b in bones]
                    K = int(len(bones))
                    N = int(len(step_idx))
                    out_map: Dict[str, np.ndarray] = {}
                    for br, br_dat in branches.items():
                        if not isinstance(br_dat, dict):
                            continue
                        rot = br_dat.get("pred_rotvec_deg_xyz", None)
                        if not isinstance(rot, dict):
                            continue
                        X = np.zeros((N, 3 * K), dtype=np.float32)
                        for bi, bname in enumerate(bones):
                            seq = rot.get(bname, None)
                            if not isinstance(seq, list):
                                continue
                            for r, t in enumerate(step_idx):
                                if t < 0 or t >= len(seq):
                                    continue
                                v = seq[t]
                                if not (isinstance(v, list) and len(v) == 3):
                                    continue
                                try:
                                    X[r, 3 * bi + 0] = float(v[0])
                                    X[r, 3 * bi + 1] = float(v[1])
                                    X[r, 3 * bi + 2] = float(v[2])
                                except Exception:
                                    continue
                        out_map[str(br)] = X
                    if out_map:
                        keybone_state_bones = bones
                        keybone_state_rotvec_deg_xyz = out_map
    except Exception:
        keybone_state_bones = None
        keybone_state_rotvec_deg_xyz = None

    return SampleSet(
        w_deg_xyz=np.asarray(w_sel, dtype=np.float32),
        ang_deg=ang_base_deg,
        delta_star_deg=best_delta_deg,
        step_idx=np.asarray(step_idx, dtype=np.int64),
        step_in_cycle=np.asarray(step_in_cycle, dtype=np.int64),
        cycle_len=int(cycle_len),
        phasez_sincos=np.asarray(phasez_sincos, dtype=np.float32) if phasez_sincos else None,
        cond=cond_feat,
        plan=_stack_contact(plan),
        meas=_stack_contact(meas),
        plan_z_in=plan_z_in,
        phase_z_in=phase_z_in,
        phase_event_age_in=phase_event_age_in,
        keybone_state_bones=keybone_state_bones,
        keybone_state_rotvec_deg_xyz=keybone_state_rotvec_deg_xyz,
    )


def _phase_sincos(step_in_cycle: np.ndarray, cycle_len: int, *, harmonics: int) -> np.ndarray:
    if cycle_len <= 0:
        raise ValueError(f"cycle_len must be >0 for phase features, got {cycle_len}")
    p = step_in_cycle.astype(np.float32) / float(cycle_len)
    feats: List[np.ndarray] = [np.ones((p.shape[0], 1), dtype=np.float32)]
    for k in range(1, int(harmonics) + 1):
        ang = (2.0 * math.pi * float(k)) * p
        feats.append(np.sin(ang).reshape(-1, 1).astype(np.float32))
        feats.append(np.cos(ang).reshape(-1, 1).astype(np.float32))
    return np.concatenate(feats, axis=1)


def _phase_sincos_from_sincos(phase_sincos: np.ndarray, *, harmonics: int) -> np.ndarray:
    """
    Build [1, sin(k*theta), cos(k*theta)] features from provided [sin(theta), cos(theta)].
    """
    sc = np.asarray(phase_sincos, dtype=np.float32)
    if sc.ndim != 2 or sc.shape[1] != 2:
        raise ValueError(f"phase_sincos must be (N,2), got {sc.shape}")
    sin = sc[:, 0]
    cos = sc[:, 1]
    # Normalize onto unit circle to stabilize atan2 when values drift.
    n = np.sqrt(sin * sin + cos * cos) + 1e-8
    sin = sin / n
    cos = cos / n
    theta = np.arctan2(sin, cos)  # (-pi, pi]
    feats: List[np.ndarray] = [np.ones((theta.shape[0], 1), dtype=np.float32)]
    for k in range(1, int(harmonics) + 1):
        ang = float(k) * theta
        feats.append(np.sin(ang).reshape(-1, 1).astype(np.float32))
        feats.append(np.cos(ang).reshape(-1, 1).astype(np.float32))
    return np.concatenate(feats, axis=1)


def _make_features(samples: SampleSet, *, feature_set: str, phase_harmonics: int) -> np.ndarray:
    fs = str(feature_set).strip().lower()
    if fs == "const":
        return np.ones((samples.delta_star_deg.shape[0], 1), dtype=np.float32)

    tokens = {t.strip().lower() for t in fs.replace(" ", "").split("+") if t.strip()}
    Xs: List[np.ndarray] = []
    if "phasez" in fs:
        if samples.phasez_sincos is None:
            raise ValueError("phasez features requested but PhaseZSinCosPerC is missing.")
        Xs.append(_phase_sincos_from_sincos(samples.phasez_sincos, harmonics=int(phase_harmonics)))
    elif "phase" in fs:
        Xs.append(_phase_sincos(samples.step_in_cycle, samples.cycle_len, harmonics=int(phase_harmonics)))
    else:
        Xs.append(np.ones((samples.delta_star_deg.shape[0], 1), dtype=np.float32))

    if "cyclelen" in fs:
        N = int(samples.delta_star_deg.shape[0])
        L = float(samples.cycle_len or 0)
        inv = 0.0 if L <= 0.0 else 1.0 / L
        # Provide both a scaled length and an inverse length (frequency proxy).
        Xs.append(np.tile(np.asarray([L / 100.0, inv], dtype=np.float32), (N, 1)))

    if "cond" in fs:
        if samples.cond is None:
            raise ValueError("cond features requested but teacher_json/teacher.cond is missing.")
        Xs.append(samples.cond.astype(np.float32, copy=False))

    if "plan" in fs:
        if samples.plan is None:
            raise ValueError("plan features requested but ContactPlanPerC is missing.")
        Xs.append(samples.plan.astype(np.float32, copy=False))

    if "meas" in fs:
        if samples.meas is None:
            raise ValueError("meas features requested but ContactMeasPerC is missing.")
        Xs.append(samples.meas.astype(np.float32, copy=False))

    # New: internal plan/phase state exports (from freerun JSON).
    if "planz" in tokens:
        if samples.plan_z_in is None:
            raise ValueError("planz requested but plan_state_series.series.plan_z_in is missing (re-run freerun with --export_plan_state_series).")
        Xs.append(samples.plan_z_in.astype(np.float32, copy=False))
    if "phasez_state" in tokens:
        if samples.phase_z_in is None:
            raise ValueError(
                "phasez_state requested but plan_state_series.series.phase_z_in is missing (re-run freerun with --export_plan_state_series)."
            )
        Xs.append(samples.phase_z_in.astype(np.float32, copy=False))
    if "phaseage" in tokens:
        if samples.phase_event_age_in is None:
            raise ValueError(
                "phaseage requested but plan_state_series.series.phase_event_age_in is missing (re-run freerun with --export_plan_state_series)."
            )
        Xs.append(samples.phase_event_age_in.astype(np.float32, copy=False))

    # New: per-step predicted joint-local pose state (from freerun JSON keybone_state export).
    want_br: List[str] = []
    if "kstate" in tokens or "kstate_direct" in tokens:
        want_br.append("direct")
    if "kstateinc" in tokens or "kstate_inc" in tokens:
        want_br.append("inc")
    if "kstateblend" in tokens or "kstate_blend" in tokens:
        want_br.append("blend")
    if want_br:
        if samples.keybone_state_rotvec_deg_xyz is None or samples.keybone_state_bones is None:
            raise ValueError(
                "kstate* requested but keybone_state is missing (re-run freerun with --export_keybone_state_series)."
            )
        for br in want_br:
            Xb = samples.keybone_state_rotvec_deg_xyz.get(str(br), None)
            if Xb is None:
                raise ValueError(f"kstate branch={br!r} requested but keybone_state.series.branches.{br} is missing.")
            Xs.append(Xb.astype(np.float32, copy=False))

    return np.concatenate(Xs, axis=1)


def _apply_delta(w_deg_xyz: np.ndarray, delta_deg: np.ndarray, *, axis: str) -> np.ndarray:
    w = torch.tensor(w_deg_xyz, dtype=torch.float32)
    w_rad = w * (math.pi / 180.0)
    with torch.no_grad():
        R_err = so3_exp_map(w_rad)  # (N,3,3)
        axis_vec = torch.tensor(_as_axis_vec(axis), dtype=torch.float32)
        d = torch.tensor(delta_deg, dtype=torch.float32).reshape(-1, 1)
        R_corr = so3_exp_map((-d * (math.pi / 180.0)) * axis_vec[None, :])  # (N,3,3)
        R_new = torch.matmul(R_corr, R_err)
        ang = _angle_from_R(R_new) * (180.0 / math.pi)
    return ang.cpu().numpy().astype(np.float32)


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2: delta* predictability (offline regressors).")
    ap.add_argument("--train-json", nargs="+", required=True, help="Train freerun_cycles JSON(s).")
    ap.add_argument("--test-json", nargs="+", required=True, help="Test freerun_cycles JSON(s).")
    ap.add_argument("--bone", type=str, default="calf_r", help="Bone name (default: calf_r).")
    ap.add_argument("--branch", type=str, default="direct", choices=("inc", "direct", "blend"))
    ap.add_argument("--axis", type=str, default="z", help="Hinge axis (x/y/z).")
    ap.add_argument("--max-deg", type=float, default=45.0, help="Oracle delta search range in degrees.")
    ap.add_argument("--grid-step-deg", type=float, default=0.25, help="Oracle delta search step in degrees.")
    ap.add_argument("--angle-thresh", type=float, default=20.0, help="Tail threshold in degrees.")
    ap.add_argument("--min-cycle", type=int, default=1, help="Filter: keep cycles>=min_cycle (default: 1 == R1+).")
    ap.add_argument("--contact-source", type=str, default="gt", choices=("gt", "plan", "meas"))
    ap.add_argument("--contact-side", type=str, default="r", choices=("l", "r"))
    ap.add_argument("--contact-value", type=int, default=0, help="Contact binarized value: 0 swing / 1 stance.")
    ap.add_argument("--contact-thresh", type=float, default=0.5)

    ap.add_argument(
        "--feature-set",
        type=str,
        default="phase",
        choices=(
            "const",
            "phase",
            "phasez",
            "cond",
            "cond+phase",
            "cond+phasez",
            "plan+meas+phase",
            "plan+meas+phasez",
            "plan+meas+phase+cyclelen",
            "plan+meas+phasez+cyclelen",
            "cond+plan+meas+phase",
            "cond+plan+meas+phasez",
            "cond+plan+meas+phase+cyclelen",
            "cond+plan+meas+phasez+cyclelen",
        ),
    )
    ap.add_argument("--phase-harmonics", type=int, default=1, help="Harmonics for phase sin/cos (default: 1).")
    ap.add_argument("--ridge", type=float, default=1e-3, help="Ridge alpha for linear regressor (default: 1e-3).")
    ap.add_argument("--train-tail-only", action="store_true", help="Train only on frames with ang_deg>angle_thresh.")
    args = ap.parse_args()

    max_deg = abs(float(args.max_deg))
    step_deg = float(args.grid_step_deg)
    if not (step_deg > 0.0):
        raise SystemExit("--grid-step-deg must be > 0.")

    def _load_many(paths: Sequence[str]) -> List[SampleSet]:
        out: List[SampleSet] = []
        for p in paths:
            out.append(
                _build_samples(
                    Path(p).expanduser(),
                    bone=str(args.bone),
                    branch=str(args.branch),
                    axis=str(args.axis),
                    max_deg=max_deg,
                    grid_step_deg=step_deg,
                    min_cycle=int(args.min_cycle),
                    contact_source=str(args.contact_source),
                    contact_side=str(args.contact_side) if args.contact_side is not None else None,
                    contact_value=int(args.contact_value) if args.contact_value is not None else None,
                    contact_thresh=float(args.contact_thresh),
                )
            )
        return out

    train_sets = _load_many(args.train_json)
    test_sets = _load_many(args.test_json)

    X_train = np.concatenate(
        [_make_features(s, feature_set=str(args.feature_set), phase_harmonics=int(args.phase_harmonics)) for s in train_sets],
        axis=0,
    )
    y_train = np.concatenate([s.delta_star_deg for s in train_sets], axis=0)
    ang_train = np.concatenate([s.ang_deg for s in train_sets], axis=0)

    if bool(args.train_tail_only):
        m = ang_train > float(args.angle_thresh)
        if not bool(np.any(m)):
            raise SystemExit("train-tail-only selected but no tail frames found.")
        X_train = X_train[m]
        y_train = y_train[m]

    w = _ridge_solve(X_train.astype(np.float32), y_train.astype(np.float32), alpha=float(args.ridge))

    def _eval(name: str, sets: Sequence[SampleSet]) -> None:
        X = np.concatenate(
            [_make_features(s, feature_set=str(args.feature_set), phase_harmonics=int(args.phase_harmonics)) for s in sets],
            axis=0,
        )
        y = np.concatenate([s.delta_star_deg for s in sets], axis=0)
        ang0 = np.concatenate([s.ang_deg for s in sets], axis=0)
        delta_hat = (X @ w).astype(np.float32)
        delta_hat = np.clip(delta_hat, -max_deg, max_deg)
        wxyz = np.concatenate([s.w_deg_xyz for s in sets], axis=0)
        ang1 = _apply_delta(wxyz, delta_hat, axis=str(args.axis))

        th = float(args.angle_thresh)
        tail = ang0 > th
        print(f"[{name}] N={int(ang0.shape[0])} tail_frac={float(np.mean(tail)):.3f} (th={th})")
        print(
            f"  angle: mean={float(np.mean(ang0)):.2f} -> {float(np.mean(ang1)):.2f} | "
            f"P(>th)={float(np.mean(ang0 > th)):.3f} -> {float(np.mean(ang1 > th)):.3f}"
        )
        print(f"  delta*: MAE={_mae(delta_hat, y):.2f} RMSE={_rmse(delta_hat, y):.2f} (all)")
        if bool(np.any(tail)):
            print(f"  delta*@tail: MAE={_mae(delta_hat[tail], y[tail]):.2f} RMSE={_rmse(delta_hat[tail], y[tail]):.2f}")

    print(
        f"[Config] bone={args.bone} branch={args.branch} axis={args.axis} max_deg={max_deg} grid_step_deg={step_deg} "
        f"min_cycle={args.min_cycle} contact={args.contact_source}:{args.contact_side}=={args.contact_value} "
        f"feature_set={args.feature_set} phase_harmonics={args.phase_harmonics} ridge={args.ridge} "
        f"train_tail_only={bool(args.train_tail_only)}"
    )
    _eval("TRAIN", train_sets)
    _eval("TEST", test_sets)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
