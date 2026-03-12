#!/usr/bin/env python3
"""
Summarize per-foot white-box signal "touchdown peak" quality from freerun_cycles JSON outputs.

Historical note (2026-03-09):
  The whitebox runtime/validate lane and `--log_contacts_whitebox*` flags were removed
  from current mainline. This tool is kept only for archived JSON produced on historical
  snapshots; do not treat the command examples below as current execution guidance.

Goal:
  Given runs produced by `python -m train.validate.run_freerun_cycles` with:
    --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps <enough>
  this script quantifies whether candidate anchor signals (dist / vxy / vz) are:
    - aligned to touchdown (peak alignment)
    - single-valley around touchdown (unimodality)
    - free of multi-valley ambiguity (multi-peak count)

Touchdown anchor:
  - Default uses GT contact rising edge from ContactGTPerC (analysis-only threshold crossing).
  - Optionally use TTCEventPerC if present.

Example:
  python tools/summarize_freerun_contact_whitebox_peaks.py \\
    --json debug_output/bridge_table_resid_20260110_contactlog_condauto_ttcgt/*/Walk_F_freerun_cycles.json \\
    --cycles 1-4 \\
    --signals DistCmMean VxyAbsCmpsMean VzCmpsMean
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _parse_cycles(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        if "-" in s:
            a_s, b_s = s.split("-", 1)
            a = int(a_s.strip())
            b = int(b_s.strip())
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(b, a + 1)))
        else:
            out.append(int(s))
    uniq: List[int] = []
    seen = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


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


def _infer_contact_dim(steps: Sequence[Dict[str, Any]]) -> int:
    for rec in steps:
        if not isinstance(rec, dict):
            continue
        for k in ("ContactGTPerC", "ContactMeasPerC", "ContactPlanPerC", "TTCEventPerC", "TTCPredPerC"):
            v = _as_float_list(rec.get(k))
            if v is not None:
                return int(len(v))
    return 0


def _touchdown_indices_from_contact(
    contact: np.ndarray, *, thr: float, min_interval: int = 0
) -> np.ndarray:
    contact = np.asarray(contact, dtype=np.float64).reshape(-1)
    if contact.size < 2:
        return np.zeros((0,), dtype=np.int64)
    prev = contact[:-1]
    cur = contact[1:]
    m = (prev < float(thr)) & (cur >= float(thr))
    ev = (np.where(m)[0] + 1).astype(np.int64)
    if int(min_interval or 0) > 0 and ev.size > 1:
        kept: List[int] = []
        last = -10**9
        for e in ev.tolist():
            e_i = int(e)
            if e_i - last >= int(min_interval):
                kept.append(e_i)
                last = e_i
        ev = np.asarray(kept, dtype=np.int64)
    return ev


def _touchdown_indices_from_ttc_event(ttc_event: np.ndarray, *, thr: float) -> np.ndarray:
    ttc_event = np.asarray(ttc_event, dtype=np.float64).reshape(-1)
    if ttc_event.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.where(ttc_event > float(thr))[0].astype(np.int64)


def _count_segments(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return 0
    # count contiguous True runs
    diff = np.diff(mask.astype(np.int8), prepend=0)
    return int(np.sum(diff == 1))


def _best_valley_center(vals: np.ndarray, *, frac: float = 0.2) -> Tuple[Optional[int], int]:
    """
    Return (center_idx, valley_count) for valleys defined as vals <= thr.
    - center_idx is the midpoint of the "best" valley segment.
    - valley_count is number of contiguous valley segments in vals.

    Uses a percentile-based adaptive threshold:
      thr = min(vals) + frac * (p90(vals) - min(vals))
    """
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    if v.size == 0 or (not np.isfinite(v).any()):
        return None, 0
    if not np.isfinite(v).all():
        return None, 0
    vmin = float(np.min(v))
    p90 = float(np.percentile(v, 90))
    thr = vmin + float(frac) * (p90 - vmin)
    low = v <= thr
    valley_count = _count_segments(low)
    if valley_count <= 0:
        return None, 0

    # Enumerate segments [start,end] where low==True.
    segs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, b in enumerate(low.tolist()):
        if b and start is None:
            start = i
        elif (not b) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, int(v.size) - 1))

    # Pick the segment with lowest min; tie-break by closeness to center.
    center_ref = int(v.size // 2)
    best_seg = segs[0]
    best_min = float(np.min(v[best_seg[0] : best_seg[1] + 1]))
    best_tie = abs(int((best_seg[0] + best_seg[1]) // 2) - center_ref)
    for s0, s1 in segs[1:]:
        seg_min = float(np.min(v[s0 : s1 + 1]))
        tie = abs(int((s0 + s1) // 2) - center_ref)
        if seg_min < best_min - 1e-12:
            best_seg = (s0, s1)
            best_min = seg_min
            best_tie = tie
        elif abs(seg_min - best_min) <= 1e-12 and tie < best_tie:
            best_seg = (s0, s1)
            best_tie = tie

    center_idx = int((best_seg[0] + best_seg[1]) // 2)
    return center_idx, int(valley_count)


@dataclass(frozen=True)
class _FootAgg:
    n_cycles: int
    align_abs_mean: Optional[float]
    align_mean: Optional[float]
    single_rate: Optional[float]
    multi_cycles: int


def _aggregate_per_foot(
    cycle_to_ti_to_val: Dict[int, Dict[int, float]],
    *,
    cycle_len: int,
    td_by_cycle: Dict[int, int],
    window: int,
) -> Tuple[_FootAgg, float]:
    """
    Returns (agg, coverage) where coverage is fraction of (cycle,ti) within selected cycles that had a value.
    """
    if cycle_len <= 0:
        return _FootAgg(0, None, None, None, 0), 0.0
    cycles = sorted(td_by_cycle.keys())
    if not cycles:
        return _FootAgg(0, None, None, None, 0), 0.0

    total_slots = int(len(cycles) * int(cycle_len))
    have_slots = 0
    for c in cycles:
        have_slots += int(len(cycle_to_ti_to_val.get(c, {})))
    coverage = float(have_slots) / float(total_slots) if total_slots > 0 else 0.0

    aligns: List[int] = []
    valley_counts: List[int] = []
    for c in cycles:
        td = int(td_by_cycle[c])
        series: List[float] = []
        idxs: List[int] = []
        for dt in range(-int(window), int(window) + 1):
            ti = int((td + dt) % int(cycle_len))
            v = cycle_to_ti_to_val.get(c, {}).get(ti, None)
            if v is None:
                series = []
                break
            series.append(float(v))
            idxs.append(ti)
        if not series:
            continue
        center_idx, valley_count = _best_valley_center(np.asarray(series, dtype=np.float64), frac=0.2)
        if center_idx is None or valley_count <= 0:
            continue
        align = int(center_idx) - int(window)
        aligns.append(int(align))
        valley_counts.append(int(valley_count))

    n = int(len(aligns))
    if n <= 0:
        return _FootAgg(0, None, None, None, 0), coverage
    a = np.asarray(aligns, dtype=np.float64)
    vc = np.asarray(valley_counts, dtype=np.int64)
    single = (vc == 1).astype(np.float64)
    multi = int(np.sum(vc > 1))

    return (
        _FootAgg(
            n_cycles=n,
            align_abs_mean=float(np.mean(np.abs(a))),
            align_mean=float(np.mean(a)),
            single_rate=float(np.mean(single)),
            multi_cycles=multi,
        ),
        coverage,
    )


def _extract_wb_signal_per_c(step: Dict[str, Any], *, wb_key: str, contact_dim: int) -> Optional[List[float]]:
    wb = step.get("ContactMeasWhitebox", None)
    if not isinstance(wb, dict):
        return None
    v = wb.get(wb_key, None)
    vv = _as_float_list(v)
    if vv is None or int(len(vv)) < int(contact_dim):
        return None
    return vv[: int(contact_dim)]


def _as_int(x: Any, *, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize touchdown peak quality from ContactMeasWhitebox logs.")
    ap.add_argument("--json", nargs="+", required=True, help="Paths/dirs/globs to *_freerun_cycles.json.")
    ap.add_argument(
        "--cycles",
        type=str,
        default="1-4",
        help="Which cycles to analyze, e.g. '1-4' (default) or '0-4' or '0,1'.",
    )
    ap.add_argument(
        "--anchor",
        type=str,
        default="gt_touchdown",
        choices=("gt_touchdown", "ttc_event"),
        help="Touchdown anchor source (ttc_event requires TTCEventPerC to be present).",
    )
    ap.add_argument("--contact-thr", type=float, default=0.5, help="Threshold for GT touchdown (analysis only).")
    ap.add_argument("--ttc-thr", type=float, default=0.5, help="Threshold for TTCEventPerC to count an event.")
    ap.add_argument("--window", type=int, default=10, help="Half-window (frames) around touchdown for peak analysis.")
    ap.add_argument(
        "--signals",
        nargs="+",
        default=["DistCmMean", "VxyAbsCmpsMean", "VzCmpsMean"],
        help="Whitebox keys to analyze (per-foot lists), e.g. DistCmMean VxyAbsCmpsMean VzCmpsMean.",
    )
    args = ap.parse_args()

    paths = _expand_json_specs(args.json)
    if not paths:
        raise SystemExit("[FATAL] No *_freerun_cycles.json matched the provided specs.")

    cycles = _parse_cycles(args.cycles)
    if not cycles:
        raise SystemExit("[FATAL] --cycles parsed to empty.")
    cycle_set = {int(c) for c in cycles}

    header = (
        "| run | phase_reset | cycles | wb_cov% | signal | |Δti| mean (L/R) | single-valley rate (L/R) | multi-valley cycles (L/R) |"
    )
    print(header)
    print("|---|---|---:|---:|---|---:|---:|---:|")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        steps = data.get("metrics_per_step", None)
        if not isinstance(steps, list) or not steps:
            continue
        steps = [s for s in steps if isinstance(s, dict)]
        if not steps:
            continue

        cycle_len = int(data.get("cycle_len", 0) or 0)
        if cycle_len <= 0:
            try:
                cycle_len = 1 + max(int(s.get("step_in_cycle", 0) or 0) for s in steps)
            except Exception:
                cycle_len = 0
        if cycle_len <= 0:
            continue

        contact_dim = _infer_contact_dim(steps)
        if contact_dim <= 0:
            continue
        if contact_dim < 2:
            # This script assumes L/R at indices 0/1 for reporting.
            continue

        # Collect per-cycle touchdown index (ti) per foot.
        td_by_cycle_by_foot: List[Dict[int, int]] = [{}, {}]
        for c in sorted(cycle_set):
            # Build GT contact series for this cycle.
            gt_series = [np.full((cycle_len,), np.nan, dtype=np.float64) for _ in range(2)]
            ttc_ev_series = [np.full((cycle_len,), np.nan, dtype=np.float64) for _ in range(2)]
            have_any = False
            for rec in steps:
                cyc_i = _as_int(rec.get("cycle", None), default=None)
                if cyc_i is None or int(cyc_i) != int(c):
                    continue
                ti_i = _as_int(rec.get("step_in_cycle", None), default=None)
                if ti_i is None or not (0 <= int(ti_i) < int(cycle_len)):
                    continue
                gt = _as_float_list(rec.get("ContactGTPerC"))
                if gt is not None and len(gt) >= 2:
                    gt_series[0][int(ti_i)] = float(gt[0])
                    gt_series[1][int(ti_i)] = float(gt[1])
                    have_any = True
                ev = _as_float_list(rec.get("TTCEventPerC"))
                if ev is not None and len(ev) >= 2:
                    ttc_ev_series[0][int(ti_i)] = float(ev[0])
                    ttc_ev_series[1][int(ti_i)] = float(ev[1])
            if not have_any:
                continue

            for foot in (0, 1):
                if args.anchor == "ttc_event" and np.isfinite(ttc_ev_series[foot]).any():
                    ev_idx = _touchdown_indices_from_ttc_event(ttc_ev_series[foot], thr=float(args.ttc_thr))
                    if ev_idx.size > 0:
                        td_by_cycle_by_foot[foot][int(c)] = int(ev_idx[0])
                        continue
                # Fallback/default: GT threshold crossing.
                if not np.isfinite(gt_series[foot]).all():
                    continue
                ev_idx = _touchdown_indices_from_contact(gt_series[foot], thr=float(args.contact_thr), min_interval=0)
                if ev_idx.size > 0:
                    td_by_cycle_by_foot[foot][int(c)] = int(ev_idx[0])

        for sig in args.signals:
            sig = str(sig)
            # per foot: cycle -> ti -> value
            sig_by_cycle_by_foot: List[Dict[int, Dict[int, float]]] = [{}, {}]
            for rec in steps:
                c = rec.get("cycle", None)
                ti = rec.get("step_in_cycle", None)
                if c is None or ti is None:
                    continue
                c_i = int(c)
                if c_i not in cycle_set:
                    continue
                ti_i = int(ti)
                if not (0 <= ti_i < int(cycle_len)):
                    continue
                vv = _extract_wb_signal_per_c(rec, wb_key=sig, contact_dim=contact_dim)
                if vv is None:
                    continue
                for foot in (0, 1):
                    sig_by_cycle_by_foot[foot].setdefault(c_i, {})[ti_i] = float(vv[foot])

            agg0, cov0 = _aggregate_per_foot(
                sig_by_cycle_by_foot[0],
                cycle_len=cycle_len,
                td_by_cycle=td_by_cycle_by_foot[0],
                window=int(args.window),
            )
            agg1, cov1 = _aggregate_per_foot(
                sig_by_cycle_by_foot[1],
                cycle_len=cycle_len,
                td_by_cycle=td_by_cycle_by_foot[1],
                window=int(args.window),
            )
            cov = 100.0 * float(0.5 * (cov0 + cov1))

            def _fmt(x: Optional[float], *, digits: int = 2) -> str:
                if x is None or (not np.isfinite(float(x))):
                    return "-"
                return f"{float(x):.{digits}f}"

            def _fmt_rate(x: Optional[float]) -> str:
                if x is None or (not np.isfinite(float(x))):
                    return "-"
                return f"{100.0*float(x):.1f}%"

            def _fmt_int_or_dash(v: int, *, ok: bool) -> str:
                return str(int(v)) if ok else "-"

            run = p.parent.name or p.name
            phase = data.get("phase_reset_source_applied") or data.get("phase_reset_source")
            out = (
                f"| {run} | {phase} | {args.cycles} | {cov:.1f} | {sig} | "
                f"{_fmt(agg0.align_abs_mean)}/{_fmt(agg1.align_abs_mean)} | "
                f"{_fmt_rate(agg0.single_rate)}/{_fmt_rate(agg1.single_rate)} | "
                f"{_fmt_int_or_dash(agg0.multi_cycles, ok=agg0.n_cycles > 0)}/{_fmt_int_or_dash(agg1.multi_cycles, ok=agg1.n_cycles > 0)} |"
            )
            print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
