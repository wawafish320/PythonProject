#!/usr/bin/env python3
"""
Pad-aware analysis for `pose_hist_time_shift` sweeps.

Why:
  In `train/validate/run_teacher_rollout.py`, the debug knob `--pose_hist_time_shift`
  is implemented as:

    new[t] = old[clip(t - shift)]

  where `clip()` clamps indices to [0, T-1]. For large |shift|, this introduces
  a *padded region* where the shifted `pose_hist` becomes constant (head pad for
  shift>0, tail pad for shift<0). That padding can heavily bias metrics like
  P(pred_L>pred_R | Lsup), especially when most L-support frames lie near the tail.

This script re-computes regime metrics split by:
  - nonpad: indices where (t - shift) stays within [0, T-1]
  - pad:    indices where (t - shift) is clipped

Inputs:
  A sweep directory produced by:
    tools/sweep_contact_meas_time_shift_set.py
  which contains a `sweep_summary.json` with per-shift per-clip `*_teacher_pred.json`.

Outputs:
  - `sweep_summary_pad.json` alongside the original sweep (or under --out).
  - A compact console table with nonpad-weighted probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_path(x: Any) -> Optional[Path]:
    if not x:
        return None
    try:
        return Path(str(x)).expanduser()
    except Exception:
        return None


def _weighted_mean(pairs: List[Tuple[int, Optional[float]]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for w, v in pairs:
        w = int(w or 0)
        if w <= 0 or v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    return (num / den) if den > 0.0 else None


def _pad_masks(T: int, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nonpad_mask, pad_mask) for time index t in [0..T-1]."""
    T = int(T)
    shift = int(shift)
    if T <= 0:
        z = np.zeros((0,), dtype=bool)
        return z, z
    t = np.arange(T, dtype=np.int64)
    idx = t - shift
    idx_clip = np.clip(idx, 0, T - 1)
    pad = idx != idx_clip
    return (~pad), pad


def _support_masks(gt: np.ndarray, *, on_th: float, off_th: float) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise ValueError(f"contacts gt must be (T,2+), got {gt.shape}")
    on = float(on_th)
    off = float(off_th)
    left = (gt[:, 0] > on) & (gt[:, 1] < off)
    right = (gt[:, 1] > on) & (gt[:, 0] < off)
    return left, right


def _p_order(pred: np.ndarray, mask: np.ndarray, *, which: str) -> Optional[float]:
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if pred.ndim != 2 or pred.shape[1] < 2 or mask.ndim != 1:
        return None
    if mask.size != pred.shape[0]:
        return None
    n = int(mask.sum())
    if n <= 0:
        return None
    if which == "L_gt_R":
        return float((pred[mask, 0] > pred[mask, 1]).mean())
    if which == "R_gt_L":
        return float((pred[mask, 1] > pred[mask, 0]).mean())
    raise ValueError(f"unknown which={which}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute pad/nonpad metrics for pose_hist_time_shift sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Sweep tag directory (contains sweep_summary.json and shift*/ outputs).",
    )
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: --root).")
    ap.add_argument("--on-th", type=float, default=0.8, help="Support ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Support OFF threshold.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[FATAL] --root is not a directory: {root}")
    out_dir = Path(args.out).expanduser().resolve() if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = root / "sweep_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"[FATAL] Missing sweep_summary.json: {summary_path}")

    sweep = _load_json(summary_path)
    meta = sweep.get("meta", {}) if isinstance(sweep.get("meta"), dict) else {}
    rows_in = sweep.get("rows", [])
    if not isinstance(rows_in, list) or not rows_in:
        raise SystemExit(f"[FATAL] Bad sweep_summary.json format: {summary_path}")

    rows_out: List[Dict[str, Any]] = []
    for r in rows_in:
        if not isinstance(r, dict):
            continue
        shift = int(r.get("shift", 0) or 0)
        clips = r.get("clips", [])
        if not isinstance(clips, list):
            clips = []

        clip_rows: List[Dict[str, Any]] = []
        for c in clips:
            if not isinstance(c, dict):
                continue
            pred_path = _as_path(c.get("json"))
            if pred_path is None or not pred_path.is_file():
                continue
            d = _load_json(pred_path)
            gt = np.asarray(d.get("aux_inputs", {}).get("contacts", []), dtype=np.float64)
            pred = np.asarray(d.get("contacts_pred", {}).get("contacts_meas", []), dtype=np.float64)
            if gt.ndim != 2 or pred.ndim != 2 or gt.shape != pred.shape or gt.shape[1] < 2:
                continue
            T = int(gt.shape[0])

            left, right = _support_masks(gt, on_th=float(args.on_th), off_th=float(args.off_th))
            nonpad, pad = _pad_masks(T, shift)

            l_all = left
            l_np = left & nonpad
            l_pd = left & pad
            r_all = right
            r_np = right & nonpad
            r_pd = right & pad

            def _n(x: np.ndarray) -> int:
                return int(np.asarray(x, dtype=bool).sum())

            l_n_all, l_n_np, l_n_pd = _n(l_all), _n(l_np), _n(l_pd)
            r_n_all, r_n_np, r_n_pd = _n(r_all), _n(r_np), _n(r_pd)

            row = dict(c)
            row.update(
                {
                    "T": T,
                    "pad_n": int(pad.sum()),
                    "nonpad_n": int(nonpad.sum()),
                    "left_support_n_nonpad": int(l_n_np),
                    "left_support_n_pad": int(l_n_pd),
                    "left_support_p_nonpad": _p_order(pred, l_np, which="L_gt_R"),
                    "left_support_p_pad": _p_order(pred, l_pd, which="L_gt_R"),
                    "left_support_pad_frac": float(l_n_pd / max(1, l_n_all)) if l_n_all > 0 else None,
                    "right_support_n_nonpad": int(r_n_np),
                    "right_support_n_pad": int(r_n_pd),
                    "right_support_p_nonpad": _p_order(pred, r_np, which="R_gt_L"),
                    "right_support_p_pad": _p_order(pred, r_pd, which="R_gt_L"),
                    "right_support_pad_frac": float(r_n_pd / max(1, r_n_all)) if r_n_all > 0 else None,
                }
            )
            clip_rows.append(row)

        # Aggregate by nonpad regime counts.
        w_left_np = [(int(x.get("left_support_n_nonpad") or 0), x.get("left_support_p_nonpad")) for x in clip_rows]
        w_right_np = [(int(x.get("right_support_n_nonpad") or 0), x.get("right_support_p_nonpad")) for x in clip_rows]
        total_left_np = sum(int(x.get("left_support_n_nonpad") or 0) for x in clip_rows)
        total_right_np = sum(int(x.get("right_support_n_nonpad") or 0) for x in clip_rows)

        rows_out.append(
            {
                "shift": int(shift),
                "out_dir": r.get("out_dir"),
                "total_left_support_n": r.get("total_left_support_n"),
                "total_right_support_n": r.get("total_right_support_n"),
                "weighted_p_L_gt_R": r.get("weighted_p_L_gt_R"),
                "weighted_p_R_gt_L": r.get("weighted_p_R_gt_L"),
                "total_left_support_n_nonpad": int(total_left_np),
                "total_right_support_n_nonpad": int(total_right_np),
                "weighted_p_L_gt_R_nonpad": _weighted_mean(w_left_np),
                "weighted_p_R_gt_L_nonpad": _weighted_mean(w_right_np),
                "clips": clip_rows,
            }
        )

    out = {"meta": meta, "rows": rows_out}
    out_path = out_dir / "sweep_summary_pad.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SweepPad] root={root}")
    print("| shift | N(Lsup_np) | P(L>R|Lsup)_np | N(Rsup_np) | P(R>L|Rsup)_np |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows_out:
        s = int(r.get("shift", 0) or 0)
        nl = int(r.get("total_left_support_n_nonpad") or 0)
        nr = int(r.get("total_right_support_n_nonpad") or 0)
        pl = r.get("weighted_p_L_gt_R_nonpad", None)
        pr = r.get("weighted_p_R_gt_L_nonpad", None)
        pl_s = "-" if pl is None else f"{float(pl):.3f}"
        pr_s = "-" if pr is None else f"{float(pr):.3f}"
        print(f"| {s} | {nl} | {pl_s} | {nr} | {pr_s} |")
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

