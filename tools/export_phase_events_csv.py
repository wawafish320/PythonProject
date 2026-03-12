#!/usr/bin/env python3
"""
Export per-cycle / per-foot phase-reset event timestamps from a freerun_cycles JSON.

This is meant for diagnosing event extraction jitter / lag / double-trigger issues, e.g.:
  - TTCEventPerC (ground-truth TTC events when --phase_reset_source=ttc_gt)
  - phase_event_age_in == 0 (reset-applied frames; useful proxy for detected events)
  - TDHazardEventPerC (integrate-to-1 TD-hazard events when --phase_reset_source=td_hazard)

Example:
  # 1) Build the fixed GT event schedule from TTCEventPerC
  python tools/export_phase_events_csv.py \\
    --json debug_output/.../B1__meas_model__reset_ttc_gt/Walk_F_freerun_cycles.json \\
    --source ttc_event \\
    --out events_ttc_gt.csv

  # 2) Export reset events inferred from phase_event_age_in
  python tools/export_phase_events_csv.py \\
    --json debug_output/.../A0_dbg/Walk_F_freerun_cycles.json \\
    --source phase_age0 \\
    --out events_meas.csv

  # 3) Export TD-hazard events (integrate-to-1 deterministic events)
  python tools/export_phase_events_csv.py \\
    --json debug_output/.../C1__reset_td_hazard/Walk_F_freerun_cycles.json \\
    --source td_hazard_event \\
    --out events_td_hazard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(obj)}")
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
        return float(x)
    except Exception:
        return default


def _series_get_at(series_item: Any, i: int) -> Tuple[Optional[Sequence[float]], int]:
    """
    Return (data, valid) at index i for a plan_state_series item of shape:
      {"data": [...], "valid": [...], "dim": K}
    """
    if not isinstance(series_item, dict):
        return None, 0
    data = series_item.get("data")
    valid = series_item.get("valid")
    if not isinstance(data, list) or not isinstance(valid, list):
        return None, 0
    if i < 0 or i >= len(data) or i >= len(valid):
        return None, 0
    v = 1 if int(valid[i] or 0) != 0 else 0
    row = data[i]
    if not isinstance(row, (list, tuple)):
        return None, v
    out: List[float] = []
    for x in row:
        fx = _as_float(x, default=None)
        out.append(float("nan") if fx is None else float(fx))
    return out, v


def _infer_contact_dim_from_steps(steps: List[Dict[str, Any]], *, key: str) -> int:
    for rec in steps:
        v = rec.get(key)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return int(len(v))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Export phase-reset events (ttc_event / td_hazard_event / phase_age0) to CSV.")
    ap.add_argument("--json", type=str, required=True, help="Input *_freerun_cycles.json.")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ap.add_argument(
        "--source",
        type=str,
        required=True,
        choices=("ttc_event", "td_hazard_event", "phase_age0"),
        help=(
            "Event source: "
            "'ttc_event' uses TTCEventPerC (thresholded by --thr). "
            "'td_hazard_event' uses TDHazardEventPerC (thresholded by --thr). "
            "'phase_age0' uses plan_state_series.series.phase_event_age_in == 0 (valid==1)."
        ),
    )
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for event indicator (ttc_event / td_hazard_event).")
    ap.add_argument("--kind", type=str, default="touchdown", help="Fallback kind label if per-step kind is missing.")
    args = ap.parse_args()

    in_json = Path(args.json).expanduser()
    out_csv = Path(args.out).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    obj = _load_json(in_json)
    steps = obj.get("metrics_per_step")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"[FATAL] Missing/empty metrics_per_step in {in_json}")
    steps = [s for s in steps if isinstance(s, dict)]
    if not steps:
        raise SystemExit(f"[FATAL] metrics_per_step has no dict entries in {in_json}")

    rows: List[Dict[str, Any]] = []
    if args.source in ("ttc_event", "td_hazard_event"):
        if args.source == "ttc_event":
            key = "TTCEventPerC"
        else:
            key = "TDHazardEventPerC"

        contact_dim = _infer_contact_dim_from_steps(steps, key=key)
        if contact_dim <= 0:
            raise SystemExit(f"[FATAL] {key} not found in metrics_per_step in {in_json}")
        thr = float(args.thr)
        for rec in steps:
            cyc = _as_int(rec.get("cycle"), default=None)
            step = _as_int(rec.get("step"), default=None)
            sic = _as_int(rec.get("step_in_cycle"), default=None)
            ev = rec.get(key)
            if cyc is None or step is None or sic is None:
                continue
            if not isinstance(ev, (list, tuple)) or len(ev) < contact_dim:
                continue
            kind_key = "TTCEventKind" if args.source == "ttc_event" else "TDHazardEventKind"
            kind = str(rec.get(kind_key) or args.kind)
            for ch in range(contact_dim):
                v = _as_float(ev[ch], default=None)
                if v is None:
                    continue
                if float(v) >= thr:
                    rows.append({"cycle": cyc, "step": step, "sic": sic, "ch": ch, "kind": kind})
    elif args.source == "phase_age0":
        ps = obj.get("plan_state_series")
        if not isinstance(ps, dict):
            raise SystemExit(f"[FATAL] plan_state_series missing in {in_json} (need --export_plan_state_series).")
        series = ps.get("series")
        if not isinstance(series, dict):
            raise SystemExit(f"[FATAL] plan_state_series.series missing in {in_json}.")
        pea = series.get("phase_event_age_in")
        if pea is None:
            raise SystemExit(f"[FATAL] plan_state_series.series.phase_event_age_in missing in {in_json}.")
        # Infer dim from series['dim'] if present, else from first row.
        dim = 0
        if isinstance(pea, dict):
            dim = _as_int(pea.get("dim"), default=0) or 0
        if dim <= 0:
            for i in range(len(steps)):
                row, valid = _series_get_at(pea, i)
                if valid and row is not None:
                    dim = len(row)
                    break
        if dim <= 0:
            raise SystemExit(f"[FATAL] Could not infer phase_event_age_in dim in {in_json}.")
        if len(getattr(pea, "get", lambda _k, _d=None: [])("data", [])) != len(steps):
            # Best-effort: still index by i, but warn the user.
            print(f"[warn] phase_event_age_in length != metrics_per_step length in {in_json}")

        for i, rec in enumerate(steps):
            cyc = _as_int(rec.get("cycle"), default=None)
            step = _as_int(rec.get("step"), default=None)
            sic = _as_int(rec.get("step_in_cycle"), default=None)
            if cyc is None or step is None or sic is None:
                continue
            age_row, valid = _series_get_at(pea, i)
            if not valid or age_row is None or len(age_row) < dim:
                continue
            kind = str(args.kind)
            for ch in range(dim):
                age = _as_float(age_row[ch], default=None)
                if age is None:
                    continue
                if float(age) == 0.0:
                    rows.append({"cycle": cyc, "step": step, "sic": sic, "ch": ch, "kind": kind, "age": age})
    else:
        raise SystemExit(f"[FATAL] Unsupported --source {args.source}")

    if not rows:
        raise SystemExit(f"[FATAL] No events extracted from {in_json} (source={args.source}).")

    # Deterministic sort: cycle, ch, step.
    rows.sort(key=lambda r: (int(r.get("cycle", 0)), int(r.get("ch", 0)), int(r.get("step", 0))))

    fieldnames = ["cycle", "step", "sic", "ch", "kind"]
    if args.source == "phase_age0":
        fieldnames.append("age")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"[ok] wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
