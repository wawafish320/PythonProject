#!/usr/bin/env python3
"""
Tabulate per-step_in_cycle statistics over selected cycles for a freerun_cycles JSON.

Typical use:
  - Inspect a seam / worst-window region like "83-0" (wrap) over steady cycles 1-4.
  - Compare inc/direct/blend and gate signals (LambdaEffMean, EventClockLambdaCorrMean).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(obj)}")
    return obj


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
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
                out.append(a)
                out.append(b)
        else:
            out.append(int(s))
    return out


def _parse_cycles(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
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


def _parse_ti_window(spec: str, *, cycle_len: int) -> List[int]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    out: List[int] = []
    for token in parts:
        if "-" in token:
            a_s, b_s = token.split("-", 1)
            start = int(a_s.strip())
            end = int(b_s.strip())
            if cycle_len > 0:
                start %= int(cycle_len)
                end %= int(cycle_len)
            if cycle_len > 0 and start > end:
                out.extend(list(range(start, int(cycle_len))))
                out.extend(list(range(0, end + 1)))
            else:
                out.extend(list(range(start, end + 1)))
        else:
            ti = int(token)
            if cycle_len > 0:
                ti %= int(cycle_len)
            out.append(ti)
    uniq: List[int] = []
    seen = set()
    for ti in out:
        if ti not in seen:
            seen.add(ti)
            uniq.append(ti)
    return uniq


def _mean_list_of_lists(vals: Sequence[Sequence[float]]) -> Optional[List[float]]:
    arr = np.asarray([list(v) for v in vals], dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return [float(x) for x in arr.mean(axis=0)]


def _collect_steps(
    steps: Sequence[Dict[str, Any]],
    *,
    cycles: Iterable[int],
    tis: Iterable[int],
    cycle_len: int,
) -> List[Dict[str, Any]]:
    cset = {int(c) for c in cycles}
    tset = {int(t) for t in tis}
    out: List[Dict[str, Any]] = []
    for rec in steps:
        try:
            cyc = rec.get("cycle", None)
            ti = rec.get("step_in_cycle", None)
            if cyc is None or ti is None:
                continue
            cyc_i = int(cyc)
            ti_i = int(ti)
            if cycle_len > 0:
                ti_i %= int(cycle_len)
            if cyc_i not in cset or ti_i not in tset:
                continue
            out.append(rec)
        except Exception:
            continue
    return out


def _mean_scalar(records: Sequence[Dict[str, Any]], key: str) -> Tuple[int, Optional[float]]:
    vals: List[float] = []
    for rec in records:
        v = rec.get(key, None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return (0, None)
    return (len(vals), float(np.mean(np.asarray(vals, dtype=np.float64))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to *_freerun_cycles.json")
    ap.add_argument("--cycles", type=str, default="1-4", help="Which cycles to average, e.g. '1-4' or '0,1'")
    ap.add_argument("--window", type=str, default="83-0", help="step_in_cycle window, supports wrap like '83-0'")
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    args = ap.parse_args()

    src = Path(args.json).expanduser().resolve()
    data = _load_json(src)
    steps = data.get("metrics_per_step")
    if not isinstance(steps, list):
        raise SystemExit(f"[FATAL] {src} missing metrics_per_step list")

    cycle_len = int(data.get("cycle_len", 0) or 0)
    if cycle_len <= 0:
        try:
            cycle_len = 1 + max(int(s.get("step_in_cycle", 0) or 0) for s in steps if isinstance(s, dict))
        except Exception:
            cycle_len = 0

    cycles = _parse_cycles(args.cycles)
    tis = _parse_ti_window(args.window, cycle_len=cycle_len)
    if not cycles:
        raise SystemExit("[FATAL] --cycles parsed to empty")
    if not tis:
        raise SystemExit("[FATAL] --window parsed to empty")

    metrics = (
        ("GeoLocalDeg", "inc"),
        ("DirectGeoLocalDeg", "direct"),
        ("BlendGeoLocalDeg", "blend"),
        ("LambdaEffMean", "lambda_eff"),
        ("LambdaRelMean", "lambda_rel"),
        ("EventClockLambdaCorrMean", "lambda_corr"),
    )

    per_ti: List[Dict[str, Any]] = []
    for ti in tis:
        recs = _collect_steps(steps, cycles=cycles, tis=(ti,), cycle_len=cycle_len)
        row: Dict[str, Any] = {"ti": int(ti), "n": int(len(recs))}
        for key, out_key in metrics:
            n_key, mean = _mean_scalar(recs, key)
            row[f"{out_key}_mean"] = mean
            row[f"{out_key}_n"] = int(n_key)

        # Optional contact debug fields (requires --log_contacts in the run).
        for ck, out_key in (
            ("ContactGTPerC", "contact_gt_per_c_mean"),
            ("ContactPlanPerC", "contact_plan_per_c_mean"),
            ("ContactMeasPerC", "contact_meas_per_c_mean"),
            ("ContactErrPerC", "contact_err_per_c_mean"),
        ):
            vals: List[Sequence[float]] = []
            for rec in recs:
                v = rec.get(ck, None)
                if isinstance(v, (list, tuple)) and v:
                    try:
                        vals.append([float(x) for x in v])
                    except Exception:
                        continue
            row[out_key] = _mean_list_of_lists(vals) if vals else None

        wb_vals: List[Sequence[float]] = []
        for rec in recs:
            wb = rec.get("ContactMeasWhitebox", None)
            if not isinstance(wb, dict):
                continue
            v = wb.get("MeasMean", None)
            if isinstance(v, (list, tuple)) and v:
                try:
                    wb_vals.append([float(x) for x in v])
                except Exception:
                    continue
        row["contact_meas_whitebox_per_c_mean"] = _mean_list_of_lists(wb_vals) if wb_vals else None

        per_ti.append(row)

    def _group_mean(ti_set: Sequence[int]) -> Dict[str, Any]:
        recs = _collect_steps(steps, cycles=cycles, tis=ti_set, cycle_len=cycle_len)
        out: Dict[str, Any] = {"tis": [int(t) for t in ti_set], "n": int(len(recs))}
        for key, out_key in metrics:
            _, mean = _mean_scalar(recs, key)
            out[f"{out_key}_mean"] = mean
        return out

    group_ti0 = _group_mean([0]) if 0 in tis else None
    group_non0 = _group_mean([t for t in tis if int(t) != 0]) if any(int(t) != 0 for t in tis) else None

    out_obj: Dict[str, Any] = {
        "source": str(src),
        "clip": data.get("clip"),
        "model": data.get("model"),
        "cycle_len": int(cycle_len),
        "cycles": [int(c) for c in cycles],
        "window": [int(t) for t in tis],
        "per_ti": per_ti,
        "group_ti0": group_ti0,
        "group_non0": group_non0,
    }

    out_path = Path(args.out).expanduser() if args.out else src.with_name(src.stem + "_window_table.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt(x: Optional[float]) -> str:
        if x is None or not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.3f}"

    print(f"[OK] wrote: {out_path}")
    print("ti | n | inc | direct | blend | lambda_eff | lambda_corr")
    for row in per_ti:
        print(
            f"{row['ti']:>2d} | {row['n']:>3d} | {_fmt(row.get('inc_mean'))} | {_fmt(row.get('direct_mean'))} | {_fmt(row.get('blend_mean'))} | {_fmt(row.get('lambda_eff_mean'))} | {_fmt(row.get('lambda_corr_mean'))}"
        )
    if group_ti0 is not None and group_non0 is not None:
        print("group | n | inc | direct | blend | lambda_eff | lambda_corr")
        print(
            f"ti==0 | {group_ti0['n']:>3d} | {_fmt(group_ti0.get('inc_mean'))} | {_fmt(group_ti0.get('direct_mean'))} | {_fmt(group_ti0.get('blend_mean'))} | {_fmt(group_ti0.get('lambda_eff_mean'))} | {_fmt(group_ti0.get('lambda_corr_mean'))}"
        )
        print(
            f"ti!=0 | {group_non0['n']:>3d} | {_fmt(group_non0.get('inc_mean'))} | {_fmt(group_non0.get('direct_mean'))} | {_fmt(group_non0.get('blend_mean'))} | {_fmt(group_non0.get('lambda_eff_mean'))} | {_fmt(group_non0.get('lambda_corr_mean'))}"
        )


if __name__ == "__main__":
    main()
