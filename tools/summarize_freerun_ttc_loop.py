#!/usr/bin/env python3
"""
Summarize TTC-anchored loop stability from `train.validate.run_freerun_cycles` JSON outputs.

This script reads per-step TTC event logs produced by run_freerun_cycles:
  - TTCEventPerC: per-contact event indicator (float list; typically 0/1)
  - Optionally: a per-step probability mass key (e.g. TDHazardProbPerC) to summarize mass/cycle

Example:
  python tools/summarize_freerun_ttc_loop.py \\
    --json debug_output/_tmp_b0_ttc_gt/*_freerun_cycles.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _expand_json_specs(specs: Sequence[str], *, pattern: str = "*_freerun_cycles.json") -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        if not spec:
            continue
        s = os.path.expanduser(str(spec))
        matches: List[Path] = []
        if any(ch in s for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(s)]
        else:
            p = Path(s)
            if p.is_dir():
                matches = sorted(p.glob(pattern))
            elif p.is_file():
                matches = [p]
        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _as_float_list(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list) or not x:
        return None
    out: List[float] = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def _load_step_series(data: Dict[str, Any], *, key: str) -> np.ndarray:
    steps = data.get("metrics_per_step", None)
    rows: List[List[float]] = []
    if isinstance(steps, list) and steps:
        for st in steps:
            if not isinstance(st, dict):
                continue
            v = _as_float_list(st.get(key))
            if v is not None:
                rows.append(v)
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 0), dtype=np.float64)


def _events_lr(
    ev: np.ndarray, *, thr: float
) -> Tuple[Tuple[int, int], Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
    if ev.ndim != 2 or ev.shape[1] < 2:
        return (0, 0), (None, None), (None, None)
    counts: List[int] = []
    pmeans: List[Optional[float]] = []
    pstds: List[Optional[float]] = []
    for c in (0, 1):
        idx = np.where(ev[:, c] > float(thr))[0].astype(np.int64)
        counts.append(int(idx.size))
        if idx.size >= 2:
            d = np.diff(idx)
            pmeans.append(float(d.mean()))
            pstds.append(float(d.std(ddof=0)))
        else:
            pmeans.append(None)
            pstds.append(None)
    return (counts[0], counts[1]), (pmeans[0], pmeans[1]), (pstds[0], pstds[1])


def _mass_lr(prob: np.ndarray, *, cycle_len: int, rounds: int) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
    if prob.ndim != 2 or prob.shape[1] < 2:
        return (None, None), (None, None)
    cycle_len = int(cycle_len or 0)
    if cycle_len <= 0:
        return (None, None), (None, None)
    ncy = int(prob.shape[0] // cycle_len)
    if rounds > 0:
        ncy = min(ncy, int(rounds))
    if ncy <= 0:
        return (None, None), (None, None)
    x = prob[: ncy * cycle_len, :2].reshape(ncy, cycle_len, 2).sum(axis=1)  # (ncy,2)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    return (float(mean[0]), float(mean[1])), (float(std[0]), float(std[1]))


def _fmt(x: Optional[float], *, digits: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _s(x: Any) -> str:
    return "-" if x is None else str(x)


def _r14_mean(metrics_per_round: Any, key: str) -> Optional[float]:
    if not isinstance(metrics_per_round, list) or len(metrics_per_round) < 5:
        return None
    try:
        vals = [float(metrics_per_round[i][key]) for i in range(1, 5)]
    except Exception:
        return None
    return float(sum(vals) / 4.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize TTC loop stability from freerun_cycles JSON outputs.")
    ap.add_argument("--json", nargs="+", required=True, help="Paths/dirs/globs to *_freerun_cycles.json.")
    ap.add_argument("--event_key", type=str, default="TTCEventPerC", help="metrics_per_step key to use as event indicator.")
    ap.add_argument(
        "--prob_key",
        type=str,
        default=None,
        help=(
            "Optional metrics_per_step key to use as per-step probability mass for mass/cycle stats "
            "(e.g. TDHazardProbPerC). If omitted, the script tries to infer it by replacing "
            "'EventPerC' -> 'ProbPerC' in --event_key."
        ),
    )
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold on event_key values to count an event.")
    args = ap.parse_args()

    paths = _expand_json_specs(args.json)
    if not paths:
        raise SystemExit("[FATAL] No *_freerun_cycles.json matched the provided specs.")

    key = str(args.event_key or "TTCEventPerC")
    prob_key = str(args.prob_key).strip() if args.prob_key is not None else ""
    if not prob_key:
        if key.endswith("EventPerC"):
            prob_key = key.replace("EventPerC", "ProbPerC")

    print(
        f"| run | phase_reset | meas_src | rounds | cycle_len | BlendGeoLocalDeg R1-4 | {key} events (L/R) | {key} events/cycle (L/R) | {key} period mean±std (L/R) | {prob_key or '-'} mass/cycle mean±std (L/R) |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        rounds = int(data.get("rounds", 0) or 0)
        cycle_len = int(data.get("cycle_len", 0) or 0)

        ev = _load_step_series(data, key=key)
        counts, pmeans, pstds = _events_lr(ev, thr=float(args.thr))

        if prob_key:
            prob = _load_step_series(data, key=prob_key)
            mass_mean, mass_std = _mass_lr(prob, cycle_len=cycle_len, rounds=rounds)
        else:
            mass_mean, mass_std = (None, None), (None, None)

        evpc = (
            (float(counts[0]) / float(rounds), float(counts[1]) / float(rounds)) if rounds > 0 else (None, None)
        )

        run = p.parent.name or p.name
        out = (
            f"| {run} | {_s(data.get('phase_reset_source_applied') or data.get('phase_reset_source'))} | "
            f"{_s(data.get('contacts_meas_source'))} | {int(data.get('rounds', 0) or 0)} | {int(data.get('cycle_len', 0) or 0)} | "
            f"{_fmt(_r14_mean(data.get('metrics_per_round'), 'BlendGeoLocalDeg'), digits=3)} | "
            f"{counts[0]}/{counts[1]} | "
            f"{_fmt(evpc[0])}/{_fmt(evpc[1])} | "
            f"{(_fmt(pmeans[0]) + '±' + _fmt(pstds[0])) if pmeans[0] is not None else '-'}"
            f"/{(_fmt(pmeans[1]) + '±' + _fmt(pstds[1])) if pmeans[1] is not None else '-'} | "
            f"{(_fmt(mass_mean[0], digits=2) + '±' + _fmt(mass_std[0], digits=2)) if mass_mean[0] is not None else '-'}"
            f"/{(_fmt(mass_mean[1], digits=2) + '±' + _fmt(mass_std[1], digits=2)) if mass_mean[1] is not None else '-'} |"
        )
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
