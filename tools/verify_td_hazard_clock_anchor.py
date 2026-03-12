#!/usr/bin/env python3
"""
Verify that --phase_reset_source=td_hazard is a *usable* clock-anchor by checking
TD-hazard events against TTC (ttc_gt) touchdown events.

This is meant to be run after a hazard-only posttrain (train_contact_td_hazard=true).

What it checks (roughly matching docs/Problems/... Section 9.3.2):
  1) Capability/config: meas JSON must have phase_reset_source_applied == td_hazard
  2) Loop stability: per-(cycle,ch) TDHazardEventPerC should be exactly 1 (strict),
     period stats should be stable, and TDHazardProbPerC mass/cycle should be ~1.
  3) Event timing: strict alignment dt (=step_meas - step_gt) should have full coverage
     and max|dt| <= dt_max (default: 3 frames).

Usage:
  python tools/verify_td_hazard_clock_anchor.py \\
    --meas debug_output/.../C1_td_hazard/Walk_F_freerun_cycles.json \\
    --gt   debug_output/.../B1_ttc_gt/Walk_F_freerun_cycles.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Event:
    cycle: int
    ch: int
    step: int
    sic: int


def _fail(msg: str) -> None:
    raise SystemExit(f"[FATAL] {msg}")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        _fail(f"Failed to read JSON: {path} ({e})")
    if not isinstance(obj, dict):
        _fail(f"Expected dict JSON at {path}, got {type(obj)}")
    return obj


def _as_int(x: Any, *, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, *, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        return v
    except Exception:
        return default


def _extract_events(
    obj: Dict[str, Any],
    *,
    key: str,
    thr: float,
    cycle_gte: int,
) -> Dict[Tuple[int, int], List[Event]]:
    steps = obj.get("metrics_per_step")
    if not isinstance(steps, list) or not steps:
        _fail("Missing/empty metrics_per_step in JSON.")
    groups: Dict[Tuple[int, int], List[Event]] = {}
    for i, rec in enumerate(steps):
        if not isinstance(rec, dict):
            continue
        cyc = _as_int(rec.get("cycle"), default=None)
        if cyc is None or cyc < int(cycle_gte):
            continue
        step = _as_int(rec.get("step"), default=i)
        sic = _as_int(rec.get("step_in_cycle"), default=-1) or -1
        ev = rec.get(key)
        if not isinstance(ev, (list, tuple)) or not ev:
            continue
        for ch, v in enumerate(ev):
            fv = _as_float(v, default=None)
            if fv is None:
                continue
            if float(fv) >= float(thr):
                k = (int(cyc), int(ch))
                groups.setdefault(k, []).append(Event(cycle=int(cyc), ch=int(ch), step=int(step), sic=int(sic)))
    # sort each group by step
    for k in list(groups.keys()):
        groups[k].sort(key=lambda e: e.step)
    return groups


def _extract_mass(
    obj: Dict[str, Any],
    *,
    key: str,
    cycle_gte: int,
) -> Dict[Tuple[int, int], float]:
    steps = obj.get("metrics_per_step")
    if not isinstance(steps, list) or not steps:
        _fail("Missing/empty metrics_per_step in JSON.")
    mass: Dict[Tuple[int, int], float] = {}
    for i, rec in enumerate(steps):
        if not isinstance(rec, dict):
            continue
        cyc = _as_int(rec.get("cycle"), default=None)
        if cyc is None or cyc < int(cycle_gte):
            continue
        prob = rec.get(key)
        if not isinstance(prob, (list, tuple)) or not prob:
            continue
        for ch, v in enumerate(prob):
            fv = _as_float(v, default=None)
            if fv is None:
                continue
            k = (int(cyc), int(ch))
            mass[k] = float(mass.get(k, 0.0) + float(fv))
    return mass


def _dt_stats(dts: Sequence[int]) -> Dict[str, Optional[float]]:
    if not dts:
        return {"mean": None, "median": None, "std": None, "max_abs": None}
    dd = [float(x) for x in dts]
    return {
        "mean": mean(dd),
        "median": median(dd),
        "std": pstdev(dd) if len(dd) >= 2 else 0.0,
        "max_abs": float(max(abs(x) for x in dts)),
    }


def _fmt(x: Optional[float], *, digits: int = 2) -> str:
    if x is None or (not math.isfinite(float(x))):
        return "-"
    return f"{float(x):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify td_hazard clock-anchor alignment vs TTC GT touchdown events.")
    ap.add_argument("--meas", type=str, required=True, help="C1 td_hazard *_freerun_cycles.json (measured).")
    ap.add_argument("--gt", type=str, required=True, help="B1 ttc_gt *_freerun_cycles.json (ground truth).")
    ap.add_argument("--meas-event-key", type=str, default="TDHazardEventPerC")
    ap.add_argument("--meas-prob-key", type=str, default="TDHazardProbPerC")
    ap.add_argument("--gt-event-key", type=str, default="TTCEventPerC")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cycle-gte", type=int, default=0)
    ap.add_argument("--require-phase-reset", type=str, default="td_hazard")
    ap.add_argument("--dt-max", type=int, default=3, help="Max allowed |dt| (frames) for strict alignment.")
    ap.add_argument("--coverage-min", type=float, default=1.0, help="Min required strict coverage (0..1).")
    ap.add_argument("--period-mean-tol", type=float, default=2.0, help="|period_mean - cycle_len| tolerance (frames).")
    ap.add_argument("--period-std-max", type=float, default=2.0, help="Max allowed period std (frames).")
    ap.add_argument("--mass-mean-tol", type=float, default=0.10, help="|mass_mean - mass_target| tolerance.")
    ap.add_argument("--mass-std-max", type=float, default=0.05, help="Max allowed mass std.")
    args = ap.parse_args()

    meas_p = Path(args.meas).expanduser()
    gt_p = Path(args.gt).expanduser()

    meas_obj = _load_json(meas_p)
    gt_obj = _load_json(gt_p)

    # 1) Capability/config consistency.
    phase_applied = str(meas_obj.get("phase_reset_source_applied") or meas_obj.get("phase_reset_source") or "")
    want_phase = str(args.require_phase_reset or "").strip()
    if want_phase and phase_applied != want_phase:
        _fail(f"meas phase_reset_source_applied={phase_applied!r} != required {want_phase!r} ({meas_p})")

    cycle_len = _as_int(meas_obj.get("cycle_len"), default=None)
    if cycle_len is None:
        cycle_len = _as_int(gt_obj.get("cycle_len"), default=None)
    if cycle_len is None or int(cycle_len) <= 0:
        _fail("Could not infer cycle_len from JSONs.")
    cycle_len = int(cycle_len)

    # 2) Extract events/mass tables.
    thr = float(args.thr)
    gt_groups = _extract_events(gt_obj, key=str(args.gt_event_key), thr=thr, cycle_gte=int(args.cycle_gte))
    if not gt_groups:
        _fail(f"No GT events extracted (key={args.gt_event_key}, thr={thr}) from {gt_p}")
    expected_keys = sorted(gt_groups.keys())
    bad_gt = [(k, len(v)) for k, v in gt_groups.items() if len(v) != 1]
    if bad_gt:
        ex = "; ".join(f"c{c}/ch{ch}:n={n}" for (c, ch), n in bad_gt[:10])
        _fail(f"GT is not a clean 1-event table for strict alignment (examples: {ex})")

    meas_groups = _extract_events(meas_obj, key=str(args.meas_event_key), thr=thr, cycle_gte=int(args.cycle_gte))
    meas_mass = _extract_mass(meas_obj, key=str(args.meas_prob_key), cycle_gte=int(args.cycle_gte))

    # Loop stability: per-key event cardinality.
    missing: List[Tuple[int, int]] = []
    multi: List[Tuple[int, int, List[int]]] = []
    for cyc, ch in expected_keys:
        evs = meas_groups.get((cyc, ch), [])
        if len(evs) == 0:
            missing.append((cyc, ch))
        elif len(evs) != 1:
            multi.append((cyc, ch, [e.step for e in evs]))
    if missing or multi:
        msg_parts = []
        if missing:
            msg_parts.append("missing=" + ",".join(f"c{c}/ch{ch}" for c, ch in missing[:10]))
        if multi:
            msg_parts.append("multi=" + ";".join(f"c{c}/ch{ch}:{steps}" for c, ch, steps in multi[:5]))
        _fail(f"TDHazardEventPerC is not 1-per-(cycle,ch) (strict). " + " | ".join(msg_parts))

    # Period stats (per ch): computed on global step diffs between consecutive cycles.
    for ch in (0, 1):
        ev_steps = [meas_groups[(cyc, ch)][0].step for (cyc, cc) in expected_keys if cc == ch]
        if len(ev_steps) >= 2:
            diffs = [int(b - a) for a, b in zip(ev_steps, ev_steps[1:])]
            m = mean([float(x) for x in diffs])
            s = pstdev([float(x) for x in diffs]) if len(diffs) >= 2 else 0.0
            if abs(float(m) - float(cycle_len)) > float(args.period_mean_tol):
                _fail(f"period mean drift too large on ch{ch}: mean={m:.2f} vs cycle_len={cycle_len}")
            if float(s) > float(args.period_std_max):
                _fail(f"period jitter too large on ch{ch}: std={s:.2f} > {float(args.period_std_max):.2f}")

    # Mass/cycle stats: compare to GT target mass (usually 1.0 per cycle/ch).
    mass_vals: List[float] = []
    for cyc, ch in expected_keys:
        pred = meas_mass.get((cyc, ch), None)
        if pred is None:
            _fail(f"Missing {args.meas_prob_key} mass for c{cyc}/ch{ch} (did you log TDHazardProbPerC?)")
        mass_vals.append(float(pred))
    mass_mean = mean(mass_vals)
    mass_std = pstdev(mass_vals) if len(mass_vals) >= 2 else 0.0
    mass_target = 1.0
    if abs(float(mass_mean) - float(mass_target)) > float(args.mass_mean_tol):
        _fail(
            f"mass/cycle mean off: mean={mass_mean:.3f} (target={mass_target:.3f}, tol={float(args.mass_mean_tol):.3f})"
        )
    if float(mass_std) > float(args.mass_std_max):
        _fail(f"mass/cycle std too large: std={mass_std:.3f} > {float(args.mass_std_max):.3f}")

    # 3) Strict alignment dt.
    aligned: List[Tuple[int, int, int]] = []  # (cycle,ch,dt)
    for cyc, ch in expected_keys:
        m_step = meas_groups[(cyc, ch)][0].step
        g_step = gt_groups[(cyc, ch)][0].step
        aligned.append((cyc, ch, int(m_step - g_step)))
    dts = [dt for (_c, _ch, dt) in aligned]
    cov = len(aligned)
    exp = len(expected_keys)
    cov_frac = float(cov) / float(exp) if exp > 0 else 0.0
    if cov_frac < float(args.coverage_min):
        _fail(f"strict coverage {cov}/{exp}={cov_frac:.3f} < {float(args.coverage_min):.3f}")
    max_abs = max(abs(dt) for dt in dts) if dts else None
    if max_abs is None:
        _fail("strict alignment produced no dt rows.")
    if int(max_abs) > int(args.dt_max):
        # Show worst offenders for debugging.
        top = sorted(aligned, key=lambda t: -abs(int(t[2])))[:10]
        ex = "; ".join(f"c{c}/ch{ch}:dt={dt}" for c, ch, dt in top)
        _fail(f"max|dt|={max_abs} > {int(args.dt_max)} frames (worst: {ex})")

    # Pretty summary.
    dts0 = [dt for (_c, ch, dt) in aligned if ch == 0]
    dts1 = [dt for (_c, ch, dt) in aligned if ch == 1]
    s0 = _dt_stats(dts0)
    s1 = _dt_stats(dts1)
    sa = _dt_stats(dts)
    print(f"[OK] td_hazard clock-anchor verified.")
    print(f"  meas: {meas_p}")
    print(f"  gt  : {gt_p}")
    print(f"  strict coverage: {cov}/{exp} ({cov_frac:.3f})")
    print(
        "  dt=step_meas-step_gt:\n"
        f"    ch0 mean/med/std/max|.| = {_fmt(s0['mean'])}/{_fmt(s0['median'])}/{_fmt(s0['std'])}/{_fmt(s0['max_abs'])}\n"
        f"    ch1 mean/med/std/max|.| = {_fmt(s1['mean'])}/{_fmt(s1['median'])}/{_fmt(s1['std'])}/{_fmt(s1['max_abs'])}\n"
        f"    all mean/med/std/max|.| = {_fmt(sa['mean'])}/{_fmt(sa['median'])}/{_fmt(sa['std'])}/{_fmt(sa['max_abs'])}"
    )
    print(f"  mass/cycle mean±std (all cycles/ch): {mass_mean:.3f}±{mass_std:.3f} (target={mass_target:.3f})")


if __name__ == "__main__":
    main()
