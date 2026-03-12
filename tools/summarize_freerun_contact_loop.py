#!/usr/bin/env python3
"""
Summarize contact-loop stability from `train.validate.run_freerun_cycles` JSON outputs.

This is meant for quick ablation comparisons like:
  - contacts_meas_source={model,gt,zero,pretrain_contact}
  - event_clock={on,off}

Notes:
  - Requires `--log_contacts` in `run_freerun_cycles` to populate Contact*PerC in metrics_per_step.

Example:
  python tools/summarize_freerun_contact_loop.py \\
    --json debug_output/_tmp_ablate_meas_resid20260108/*/Walk_F_freerun_cycles.json \\
    --thr 0.5 --event-kind touchdown --min-interval 0
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


def _pick_contacts_from_step(step: Dict[str, Any], *, source: str) -> Optional[List[float]]:
    source = str(source).strip().lower()
    if source in ("gt", "contactgt"):
        return _as_float_list(step.get("ContactGTPerC"))
    if source in ("meas", "contactmeas"):
        return _as_float_list(step.get("ContactMeasPerC"))
    if source in ("plan", "contactplan"):
        return _as_float_list(step.get("ContactPlanPerC"))
    raise ValueError(f"Unknown contacts source {source!r} (expected gt/meas/plan).")


def _detect_events(x: np.ndarray, *, thr: float, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.zeros((0,), dtype=np.int64)
    prev = x[:-1]
    cur = x[1:]
    kind = str(kind).strip().lower()
    if kind in ("touchdown", "td", "rise", "up"):
        m = (prev < thr) & (cur >= thr)
    elif kind in ("liftoff", "lo", "fall", "down"):
        m = (prev >= thr) & (cur < thr)
    elif kind in ("both", "any"):
        m = ((prev < thr) & (cur >= thr)) | ((prev >= thr) & (cur < thr))
    else:
        raise ValueError(f"Unknown event kind {kind!r}.")
    return (np.where(m)[0] + 1).astype(np.int64)


def _suppress_events(events: np.ndarray, *, min_interval: int) -> np.ndarray:
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    min_interval = int(min_interval or 0)
    if events.size == 0 or min_interval <= 0:
        return events
    kept: List[int] = []
    last = -10**9
    for e in events.tolist():
        e = int(e)
        if e - last >= min_interval:
            kept.append(e)
            last = e
    return np.asarray(kept, dtype=np.int64)


def _period_mean(events: np.ndarray) -> Optional[float]:
    ev = np.asarray(events, dtype=np.int64).reshape(-1)
    if ev.size < 2:
        return None
    p = np.diff(ev)
    if p.size == 0:
        return None
    return float(p.mean())


def _r14_mean(metrics_per_round: Any, key: str) -> Optional[float]:
    if not isinstance(metrics_per_round, list) or len(metrics_per_round) < 5:
        return None
    try:
        vals = [float(metrics_per_round[i][key]) for i in range(1, 5)]
    except Exception:
        return None
    return float(sum(vals) / 4.0)


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _s(x: Any) -> str:
    if x is None:
        return "-"
    return str(x)


def _events_lr(
    arr: np.ndarray, *, thr: float, kind: str, min_interval: int
) -> Tuple[Tuple[int, int], Tuple[Optional[float], Optional[float]]]:
    if arr.ndim != 2 or arr.shape[1] < 2:
        return (0, 0), (None, None)
    counts: List[int] = []
    pmeans: List[Optional[float]] = []
    for c in (0, 1):
        ev = _detect_events(arr[:, c], thr=float(thr), kind=str(kind))
        ev = _suppress_events(ev, min_interval=int(min_interval))
        counts.append(int(ev.size))
        pmeans.append(_period_mean(ev))
    return (counts[0], counts[1]), (pmeans[0], pmeans[1])


def _load_contacts(data: Dict[str, Any], *, source: str) -> np.ndarray:
    steps = data.get("metrics_per_step", None)
    rows: List[List[float]] = []
    if isinstance(steps, list) and steps:
        for st in steps:
            if not isinstance(st, dict):
                continue
            v = _pick_contacts_from_step(st, source=source)
            if v is not None:
                rows.append(v)
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 0), dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize contact-loop stability from freerun_cycles JSON outputs.")
    ap.add_argument("--json", nargs="+", required=True, help="Paths/dirs/globs to *_freerun_cycles.json.")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for event detection.")
    ap.add_argument(
        "--event-kind",
        type=str,
        default="touchdown",
        choices=("touchdown", "liftoff", "both"),
        help="Event definition via threshold crossing.",
    )
    ap.add_argument(
        "--min-interval",
        type=int,
        default=0,
        help="Cooldown (frames) to suppress duplicate events after threshold crossing (0 disables).",
    )
    args = ap.parse_args()

    paths = _expand_json_specs(args.json)
    if not paths:
        raise SystemExit("[FATAL] No *_freerun_cycles.json matched the provided specs.")

    print(
        "| run | meas_src | ec | reproj | wb_ground | vxy | BlendGeoLocalDeg R1-4 | GT TD (L/R) | Meas TD (L/R) | Meas period mean (L/R) |"
    )
    print("|---|---|---|---|---|---|---:|---:|---:|---:|")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        gt = _load_contacts(data, source="gt")
        meas = _load_contacts(data, source="meas")

        gt_events, _ = _events_lr(gt, thr=float(args.thr), kind=str(args.event_kind), min_interval=int(args.min_interval))
        meas_events, meas_pmean = _events_lr(
            meas, thr=float(args.thr), kind=str(args.event_kind), min_interval=int(args.min_interval)
        )

        gt_lr = f"{gt_events[0]}/{gt_events[1]}"
        meas_lr = f"{meas_events[0]}/{meas_events[1]}"
        pmean_lr = f"{_fmt(meas_pmean[0], digits=1)}/{_fmt(meas_pmean[1], digits=1)}"

        run = p.parent.name or p.name
        out = (
            f"| {run} | {_s(data.get('contacts_meas_source'))} | {_s(data.get('event_clock'))} | "
            f"{_s(data.get('cond_reprojection'))} | {_s(data.get('contact_meas_ground_z_mode'))} | "
            f"{_s(data.get('contact_meas_vxy_mode'))} | {_fmt(_r14_mean(data.get('metrics_per_round'), 'BlendGeoLocalDeg'), digits=3)} | "
            f"{gt_lr} | {meas_lr} | {pmean_lr} |"
        )
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
