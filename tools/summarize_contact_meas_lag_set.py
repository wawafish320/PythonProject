#!/usr/bin/env python3
"""
Summarize `contact_meas_head` lag diagnostics over a set of `*_teacher_pred.json` files.

This is meant to answer: "Is the R-fall long-tail lag (e.g., +15 frames) systemic across clips,
or just a single clip artifact (e.g., Walk_F)?"

Workflow:
  1) Generate teacher-rollout predictions:
       python train/validate/run_teacher_rollout.py \
         --model <ckpt.pth> --teacher validate/teacher_batches/*.json \
         --encoder-bundle models/motion_encoder_equiv_stageA.pt --depth 3 \
         --out debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all --force --quiet

  2) Summarize lag:
       python tools/summarize_contact_meas_lag_set.py \
         --pred debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all \
         --max-lag 30 --on-th 0.8 --off-th 0.1 --out debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from tools.analyze_contact_meas_lag import analyze as analyze_contact_lag  # type: ignore
except Exception:  # pragma: no cover
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so import sibling directly.
    from analyze_contact_meas_lag import analyze as analyze_contact_lag  # type: ignore


def _expand_pred_specs(specs: Sequence[str], *, pattern: str = "*_teacher_pred.json") -> List[Path]:
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


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_int(x: Optional[int]) -> str:
    return "-" if x is None else str(int(x))


def _fmt_float(x: Optional[float], *, digits: int = 3) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _nested_get(d: Any, keys: Sequence[str]) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols: List[str] = [
        "clip",
        "T",
        "pose_hist_source",
        "pose_hist_ablation",
        "pose_hist_keep_last",
        "R_best_lag",
        "R_best_corr",
        "R_delta_best_lag",
        "R_delta_best_corr",
        "R_rise_slope_med",
        "R_fall_slope_med",
        "R_fall_pred_at_med",
        "R_fall_time_to_mid_med",
        "R_fall_time_to_on_med",
        "L_rise_slope_med",
        "L_fall_slope_med",
        "L_fall_pred_at_med",
        "L_fall_time_to_mid_med",
        "R_vs_GT_L_best_lag",
        "R_vs_GT_L_best_corr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in cols})


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "clip",
        "T",
        "pose_hist",
        "R_best_lag",
        "R_pred@GT_fall",
        "R_dt<=mid",
        "P(R_vs_GT_L)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|\n")
        for r in rows:
            pose = f"{r.get('pose_hist_source','?')}/{r.get('pose_hist_ablation','?')}"
            rvsl = f"{_fmt_float(_as_float(r.get('R_vs_GT_L_best_corr')), digits=3)}@{_fmt_int(_as_int(r.get('R_vs_GT_L_best_lag')))}"
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r.get("clip", "")),
                        str(r.get("T", "")),
                        pose,
                        _fmt_int(_as_int(r.get("R_best_lag"))),
                        _fmt_float(_as_float(r.get("R_fall_pred_at_med")), digits=3),
                        _fmt_float(_as_float(r.get("R_fall_time_to_mid_med")), digits=1),
                        rvsl,
                    ]
                )
                + " |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize contact_meas lag diagnostics over teacher-rollout outputs.")
    ap.add_argument(
        "--pred",
        nargs="+",
        required=True,
        help="Paths / dirs / globs to `*_teacher_pred.json` (dir implies '*_teacher_pred.json').",
    )
    ap.add_argument("--max-lag", type=int, default=30, help="Search lag in [-max_lag,+max_lag] for correlation.")
    ap.add_argument("--on-th", type=float, default=0.8, help="Event ON threshold for edge-based lag.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Event OFF threshold for edge-based lag.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory (writes contact_meas_lag_summary.{json,csv,md}).",
    )
    args = ap.parse_args()

    files = _expand_pred_specs(args.pred)
    if not files:
        raise SystemExit("[FATAL] --pred expanded to empty file list.")

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for p in files:
        s = analyze_contact_lag(p, max_lag=int(args.max_lag), on_th=float(args.on_th), off_th=float(args.off_th))
        summaries.append(s)

        corr_r_lag = _nested_get(s, ("corr", "R", "best_lag"))
        corr_r_corr = _nested_get(s, ("corr", "R", "best_corr"))
        corr_rd_lag = _nested_get(s, ("corr", "R_delta", "best_lag"))
        corr_rd_corr = _nested_get(s, ("corr", "R_delta", "best_corr"))
        r_rise_slope_med = _nested_get(s, ("event", "R", "rising_slope", "median_lag"))
        r_fall_slope_med = _nested_get(s, ("event", "R", "falling_slope", "median_lag"))
        l_rise_slope_med = _nested_get(s, ("event", "L", "rising_slope", "median_lag"))
        l_fall_slope_med = _nested_get(s, ("event", "L", "falling_slope", "median_lag"))
        r_fall_pred_at_med = _nested_get(s, ("event", "R", "falling_time", "pred_at_gt", "median"))
        r_fall_dt_mid_med = _nested_get(s, ("event", "R", "falling_time", "time_to_le_mid", "median"))
        r_fall_dt_on_med = _nested_get(s, ("event", "R", "falling_time", "time_to_le_on", "median"))
        l_fall_pred_at_med = _nested_get(s, ("event", "L", "falling_time", "pred_at_gt", "median"))
        l_fall_dt_mid_med = _nested_get(s, ("event", "L", "falling_time", "time_to_le_mid", "median"))
        rvsl_lag = _nested_get(s, ("corr", "R_vs_GT_L", "best_lag"))
        rvsl_corr = _nested_get(s, ("corr", "R_vs_GT_L", "best_corr"))

        data = json.loads(p.read_text(encoding="utf-8"))
        ab = data.get("ablation", {}) if isinstance(data.get("ablation"), dict) else {}
        rows.append(
            {
                "clip": s.get("clip"),
                "T": s.get("T"),
                "pose_hist_source": ab.get("pose_hist_source"),
                "pose_hist_ablation": ab.get("pose_hist_ablation"),
                "pose_hist_keep_last": ab.get("pose_hist_keep_last"),
                "R_best_lag": corr_r_lag,
                "R_best_corr": corr_r_corr,
                "R_delta_best_lag": corr_rd_lag,
                "R_delta_best_corr": corr_rd_corr,
                "R_rise_slope_med": r_rise_slope_med,
                "R_fall_slope_med": r_fall_slope_med,
                "R_fall_pred_at_med": r_fall_pred_at_med,
                "R_fall_time_to_mid_med": r_fall_dt_mid_med,
                "R_fall_time_to_on_med": r_fall_dt_on_med,
                "L_rise_slope_med": l_rise_slope_med,
                "L_fall_slope_med": l_fall_slope_med,
                "L_fall_pred_at_med": l_fall_pred_at_med,
                "L_fall_time_to_mid_med": l_fall_dt_mid_med,
                "R_vs_GT_L_best_lag": rvsl_lag,
                "R_vs_GT_L_best_corr": rvsl_corr,
                "json": str(p),
            }
        )

    # Sort by robust post-fall time-to-mid (descending); fallback to slope metric.
    def _sort_key(r: Dict[str, Any]) -> float:
        v = _as_float(r.get("R_fall_time_to_mid_med"))
        if v is not None:
            return float(v)
        v = _as_float(r.get("R_fall_slope_med"))
        if v is None:
            return float("-inf")
        return float(v)

    rows_sorted = sorted(rows, key=_sort_key, reverse=True)

    # Print a compact console table
    print(f"[LagSet] n={len(rows_sorted)} max_lag={int(args.max_lag)} on_th={float(args.on_th)} off_th={float(args.off_th)}")
    for r in rows_sorted:
        print(
            f"  {str(r.get('clip')):18s} "
            f"R_best_lag={_fmt_int(_as_int(r.get('R_best_lag'))):>4s} "
            f"R_fall_dt_mid={_fmt_float(_as_float(r.get('R_fall_time_to_mid_med')), digits=1):>6s} "
            f"R_pred@fall={_fmt_float(_as_float(r.get('R_fall_pred_at_med')), digits=3):>6s} "
            f"pose_hist={r.get('pose_hist_source')}/{r.get('pose_hist_ablation')}"
        )

    # Aggregate: how many clips show a large post-fall time-to-mid?
    fall_vals = [_as_float(r.get("R_fall_time_to_mid_med")) for r in rows_sorted]
    fall_vals = [v for v in fall_vals if v is not None]
    n = len(rows_sorted)
    n_fall = len(fall_vals)
    n_ge10 = sum(1 for v in fall_vals if v >= 10.0)
    n_ge15 = sum(1 for v in fall_vals if v >= 15.0)
    print(f"[LagSet] R_fall_time_to_mid_med valid={n_fall}/{n}  >=10: {n_ge10}  >=15: {n_ge15}")

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "contact_meas_lag_summary.json").write_text(
            json.dumps({"rows": rows_sorted, "summaries": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(out_dir / "contact_meas_lag_summary.csv", rows_sorted)
        _write_md(out_dir / "contact_meas_lag_summary.md", rows_sorted)
        print(f"[Wrote] {out_dir / 'contact_meas_lag_summary.md'}")


if __name__ == "__main__":
    main()
