#!/usr/bin/env python3
"""
Diagnose touchdown-anchor signals and TTC-pred jitter from freerun_cycles JSON outputs.

Historical note (2026-03-09):
  The whitebox runtime/validate lane and `--log_contacts_whitebox*` flags were removed
  from current mainline. This tool is kept only for archived JSON produced on historical
  snapshots; do not treat the command examples below as current execution guidance.

This is an offline analysis tool (no inference knobs):
  - Uses GT contact rising edge (ContactGTPerC) as touchdown anchor (analysis-only threshold).
  - Reads ContactMeasWhitebox signals (e.g., VxyAbsCmpsMean) logged by:
      run_freerun_cycles --log_contacts --log_contacts_whitebox --log_contacts_whitebox_first_steps 500
  - Quantifies:
      (A) Anchor signal quality: signed offset, single/multi-valley around touchdown (±window).
      (B) ttc_pred jitter: round() increments / reset-jump rate, and where TTCEventPerC fires vs GT touchdown.

Example:
  python tools/diagnose_freerun_ttd_anchor.py \\
    --json debug_output/bridge_table_resid_20260110_contactlog_condauto_whitebox_ttcpred/*/Walk_F_freerun_cycles.json \\
    --cycles 1-4 --window 10 \\
    --signals VxyAbsCmpsMean DistCmMean \\
    --ttc-event-thr 0.5
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


def _as_int(x: Any, *, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _infer_contact_dim(steps: Sequence[Dict[str, Any]]) -> int:
    for rec in steps:
        if not isinstance(rec, dict):
            continue
        for k in ("ContactGTPerC", "ContactMeasPerC", "ContactPlanPerC", "TTCEventPerC", "TTCPredPerC"):
            v = _as_float_list(rec.get(k))
            if v is not None:
                return int(len(v))
    return 0


def _touchdown_idx_from_contact(contact: np.ndarray, *, thr: float) -> Optional[int]:
    contact = np.asarray(contact, dtype=np.float64).reshape(-1)
    if contact.size < 2:
        return None
    prev = contact[:-1]
    cur = contact[1:]
    finite = np.isfinite(prev) & np.isfinite(cur)
    idx = np.where(finite & (prev < float(thr)) & (cur >= float(thr)))[0]
    return int(idx[0] + 1) if idx.size > 0 else None


def _count_segments(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return 0
    diff = np.diff(mask.astype(np.int8), prepend=0)
    return int(np.sum(diff == 1))


def _best_valley_center(vals: np.ndarray, *, frac: float) -> Tuple[Optional[int], int, Optional[float]]:
    """
    Return (center_idx, valley_count, thr) for valleys defined as vals <= thr.
    thr uses percentile-based adaptive threshold:
      thr = min(vals) + frac * (p90(vals) - min(vals))
    """
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    if v.size == 0 or (not np.isfinite(v).all()):
        return None, 0, None
    vmin = float(np.min(v))
    p90 = float(np.percentile(v, 90))
    thr = vmin + float(frac) * (p90 - vmin)
    low = v <= thr
    valley_count = _count_segments(low)
    if valley_count <= 0:
        return None, 0, thr

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
    return center_idx, int(valley_count), thr


def _signed_delta(mod_delta: int, *, cycle_len: int) -> int:
    d = int(mod_delta) % int(cycle_len)
    if d > int(cycle_len) // 2:
        d -= int(cycle_len)
    return int(d)


@dataclass(frozen=True)
class _AnchorAgg:
    n_cycles: int
    offset_mean: Optional[float]
    offset_std: Optional[float]
    abs_offset_mean: Optional[float]
    single_rate: Optional[float]
    multi_cycles: int


def _agg_offsets(offsets: Sequence[int], valley_counts: Sequence[int]) -> _AnchorAgg:
    if not offsets:
        return _AnchorAgg(0, None, None, None, None, 0)
    off = np.asarray(list(offsets), dtype=np.float64)
    vc = np.asarray(list(valley_counts), dtype=np.int64)
    single = (vc == 1).astype(np.float64)
    return _AnchorAgg(
        n_cycles=int(off.size),
        offset_mean=float(off.mean()),
        offset_std=float(off.std(ddof=0)) if off.size > 0 else None,
        abs_offset_mean=float(np.abs(off).mean()),
        single_rate=float(single.mean()) if single.size > 0 else None,
        multi_cycles=int(np.sum(vc > 1)),
    )


def _fmt(x: Optional[float], *, digits: int = 2) -> str:
    if x is None or (not np.isfinite(float(x))):
        return "-"
    return f"{float(x):.{digits}f}"


def _fmt_rate(x: Optional[float]) -> str:
    if x is None or (not np.isfinite(float(x))):
        return "-"
    return f"{100.0*float(x):.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose touchdown anchor signals and TTC jitter.")
    ap.add_argument("--json", nargs="+", required=True, help="Paths/dirs/globs to *_freerun_cycles.json.")
    ap.add_argument("--cycles", type=str, default="1-4", help="Cycles to analyze, e.g. '1-4' or '0-4'.")
    ap.add_argument("--window", type=int, default=10, help="Half-window (frames) around touchdown.")
    ap.add_argument("--contact-thr", type=float, default=0.5, help="GT contact threshold for touchdown (analysis only).")
    ap.add_argument(
        "--signals",
        nargs="+",
        default=["VxyAbsCmpsMean", "VxyRelCmpsMean", "DistCmMean"],
        help="ContactMeasWhitebox keys to analyze.",
    )
    ap.add_argument(
        "--valley-frac",
        type=float,
        default=0.2,
        help="Valley threshold fraction: thr = min + frac*(p90-min).",
    )
    ap.add_argument("--ttc-event-thr", type=float, default=0.5, help="Threshold to count TTCEventPerC as an event.")
    args = ap.parse_args()

    paths = _expand_json_specs(args.json)
    if not paths:
        raise SystemExit("[FATAL] No *_freerun_cycles.json matched the provided specs.")

    cycles = _parse_cycles(args.cycles)
    if not cycles:
        raise SystemExit("[FATAL] --cycles parsed to empty.")
    cycle_set = {int(c) for c in cycles}

    print("## Anchor Signal Diagnostics (GT touchdown anchor)")
    print(
        "| run | phase_reset | signal | abs(|Δti|) mean (L/R) | Δti mean±std (L/R) | single-valley rate (L/R) | multi-valley cycles (L/R) |"
    )
    print("|---|---|---|---:|---:|---:|---:|")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        steps_any = data.get("metrics_per_step", None)
        if not isinstance(steps_any, list) or not steps_any:
            continue
        steps = [s for s in steps_any if isinstance(s, dict)]
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
        if contact_dim < 2:
            continue

        # Index records for fast window lookup.
        by_ct: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for rec in steps:
            cyc_i = _as_int(rec.get("cycle", None), default=None)
            ti_i = _as_int(rec.get("step_in_cycle", None), default=None)
            if cyc_i is None or ti_i is None:
                continue
            by_ct[(int(cyc_i), int(ti_i))] = rec

        # Touchdown ti per cycle per foot from GT contact.
        td_by_cycle: List[Dict[int, int]] = [{}, {}]
        for cyc in sorted(cycle_set):
            gt_series = [np.full((cycle_len,), np.nan, dtype=np.float64) for _ in range(2)]
            for ti in range(cycle_len):
                rec = by_ct.get((int(cyc), int(ti)))
                if rec is None:
                    continue
                gt = _as_float_list(rec.get("ContactGTPerC"))
                if gt is None or len(gt) < 2:
                    continue
                gt_series[0][ti] = float(gt[0])
                gt_series[1][ti] = float(gt[1])
            for foot in (0, 1):
                td = _touchdown_idx_from_contact(gt_series[foot], thr=float(args.contact_thr))
                if td is not None:
                    td_by_cycle[foot][int(cyc)] = int(td)

        phase = data.get("phase_reset_source_applied") or data.get("phase_reset_source")
        run = p.parent.name or p.name

        for sig in args.signals:
            sig = str(sig)
            offsets: List[List[int]] = [[], []]
            vcounts: List[List[int]] = [[], []]
            for cyc in sorted(cycle_set):
                for foot in (0, 1):
                    td = td_by_cycle[foot].get(int(cyc))
                    if td is None:
                        continue
                    series: List[float] = []
                    for dt in range(-int(args.window), int(args.window) + 1):
                        ti = int((int(td) + int(dt)) % int(cycle_len))
                        rec = by_ct.get((int(cyc), int(ti)))
                        if rec is None:
                            series = []
                            break
                        wb = rec.get("ContactMeasWhitebox", None)
                        if not isinstance(wb, dict):
                            series = []
                            break
                        vv = _as_float_list(wb.get(sig))
                        if vv is None or len(vv) < 2:
                            series = []
                            break
                        series.append(float(vv[foot]))
                    if not series:
                        continue
                    center, vc, _thr = _best_valley_center(np.asarray(series, dtype=np.float64), frac=float(args.valley_frac))
                    if center is None or vc <= 0:
                        continue
                    off = int(center) - int(args.window)
                    offsets[foot].append(int(off))
                    vcounts[foot].append(int(vc))

            aggL = _agg_offsets(offsets[0], vcounts[0])
            aggR = _agg_offsets(offsets[1], vcounts[1])
            out = (
                f"| {run} | {phase} | {sig} | "
                f"{_fmt(aggL.abs_offset_mean)}/{_fmt(aggR.abs_offset_mean)} | "
                f"{_fmt(aggL.offset_mean)}±{_fmt(aggL.offset_std)}/{_fmt(aggR.offset_mean)}±{_fmt(aggR.offset_std)} | "
                f"{_fmt_rate(aggL.single_rate)}/{_fmt_rate(aggR.single_rate)} | "
                f"{aggL.multi_cycles if aggL.n_cycles else '-'}/{aggR.multi_cycles if aggR.n_cycles else '-'} |"
            )
            print(out)

    print()
    print("## ttc_pred Jitter Diagnostics (round() / reset-jump)")
    print(
        "| run | phase_reset | cycles | TTCPred median (L/R) | round inc% (L/R) | reset_jump count (L/R) | "
        "TTCEvent count (L/R) | events/cycle mean (L/R) | event mean|Δti| (L/R) | within±10 of GT TD (L/R) | within±2 of GT TD (L/R) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        steps_any = data.get("metrics_per_step", None)
        if not isinstance(steps_any, list) or not steps_any:
            continue
        steps = [s for s in steps_any if isinstance(s, dict)]
        if not steps:
            continue
        cycle_len = int(data.get("cycle_len", 0) or 0)
        if cycle_len <= 0:
            continue
        contact_dim = _infer_contact_dim(steps)
        if contact_dim < 2:
            continue

        phase = data.get("phase_reset_source_applied") or data.get("phase_reset_source")
        run = p.parent.name or p.name

        # Build per-cycle GT touchdown ti.
        by_ct: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for rec in steps:
            cyc_i = _as_int(rec.get("cycle", None), default=None)
            ti_i = _as_int(rec.get("step_in_cycle", None), default=None)
            if cyc_i is None or ti_i is None:
                continue
            by_ct[(int(cyc_i), int(ti_i))] = rec

        td_by_cycle: List[Dict[int, int]] = [{}, {}]
        for cyc in sorted(cycle_set):
            gt_series = [np.full((cycle_len,), np.nan, dtype=np.float64) for _ in range(2)]
            for ti in range(cycle_len):
                rec = by_ct.get((int(cyc), int(ti)))
                if rec is None:
                    continue
                gt = _as_float_list(rec.get("ContactGTPerC"))
                if gt is None or len(gt) < 2:
                    continue
                gt_series[0][ti] = float(gt[0])
                gt_series[1][ti] = float(gt[1])
            for foot in (0, 1):
                td = _touchdown_idx_from_contact(gt_series[foot], thr=float(args.contact_thr))
                if td is not None:
                    td_by_cycle[foot][int(cyc)] = int(td)

        # Collect TTCPredPerC and TTCEventPerC within selected cycles.
        ttc_pred: List[List[float]] = [[], []]
        ttc_ev: List[List[int]] = [[], []]  # signed delta to GT touchdown for events
        ttc_ev_count: List[int] = [0, 0]
        ttc_ev_within10: List[int] = [0, 0]
        ttc_ev_within2: List[int] = [0, 0]
        ev_by_cycle: List[Dict[int, int]] = [{}, {}]

        for rec in steps:
            cyc = _as_int(rec.get("cycle", None), default=None)
            ti = _as_int(rec.get("step_in_cycle", None), default=None)
            if cyc is None or ti is None or int(cyc) not in cycle_set:
                continue

            pred = _as_float_list(rec.get("TTCPredPerC"))
            if pred is not None and len(pred) >= 2:
                ttc_pred[0].append(float(pred[0]))
                ttc_pred[1].append(float(pred[1]))

            ev = _as_float_list(rec.get("TTCEventPerC"))
            if ev is not None and len(ev) >= 2:
                for foot in (0, 1):
                    if float(ev[foot]) <= float(args.ttc_event_thr):
                        continue
                    td = td_by_cycle[foot].get(int(cyc))
                    if td is None:
                        continue
                    ttc_ev_count[foot] += 1
                    d = _signed_delta(int(ti) - int(td), cycle_len=cycle_len)
                    ttc_ev[foot].append(int(d))
                    if abs(int(d)) <= 10:
                        ttc_ev_within10[foot] += 1
                    if abs(int(d)) <= 2:
                        ttc_ev_within2[foot] += 1
                    ev_by_cycle[foot][int(cyc)] = int(ev_by_cycle[foot].get(int(cyc), 0)) + 1

        if not ttc_pred[0] and not ttc_pred[1] and not any(ttc_ev_count):
            continue

        def _round_inc_rate(xs: List[float]) -> Tuple[Optional[float], int]:
            if len(xs) < 2:
                return None, 0
            r = np.round(np.asarray(xs, dtype=np.float64)).astype(np.int64)
            inc = int(np.sum(r[1:] > r[:-1]))
            return float(inc) / float(r.size - 1), inc

        med = [None, None]
        inc_rate = [None, None]
        inc_cnt = [0, 0]
        for foot in (0, 1):
            if ttc_pred[foot]:
                arr = np.asarray(ttc_pred[foot], dtype=np.float64)
                med[foot] = float(np.median(arr))
                inc_rate[foot], inc_cnt[foot] = _round_inc_rate(ttc_pred[foot])

        def _fmt_pct(x: Optional[float]) -> str:
            if x is None or (not np.isfinite(float(x))):
                return "-"
            return f"{100.0*float(x):.1f}%"

        def _pair(a: Any, b: Any, *, digits: int = 2) -> str:
            return f"{_fmt(a, digits=digits)}/{_fmt(b, digits=digits)}"

        mean_abs_delta = [None, None]
        ev_per_cycle_mean = [None, None]
        for foot in (0, 1):
            if ttc_ev[foot]:
                arr = np.asarray(ttc_ev[foot], dtype=np.float64)
                mean_abs_delta[foot] = float(np.mean(np.abs(arr)))
            cyc_counts = list(ev_by_cycle[foot].values())
            if cyc_counts:
                ev_per_cycle_mean[foot] = float(np.mean(np.asarray(cyc_counts, dtype=np.float64)))

        out = (
            f"| {run} | {phase} | {args.cycles} | {_pair(med[0], med[1], digits=3)} | "
            f"{_fmt_pct(inc_rate[0])}/{_fmt_pct(inc_rate[1])} | "
            f"{inc_cnt[0]}/{inc_cnt[1]} | "
            f"{ttc_ev_count[0]}/{ttc_ev_count[1]} | "
            f"{_pair(ev_per_cycle_mean[0], ev_per_cycle_mean[1], digits=2)} | "
            f"{_pair(mean_abs_delta[0], mean_abs_delta[1], digits=2)} | "
            f"{ttc_ev_within10[0]}/{ttc_ev_within10[1]} | "
            f"{ttc_ev_within2[0]}/{ttc_ev_within2[1]} |"
        )
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
