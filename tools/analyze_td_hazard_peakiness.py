#!/usr/bin/env python3
"""
Analyze touchdown-hazard (TDHazard*) time-shape from run_freerun_cycles JSON outputs.

Motivation:
  - Detect the "constant-rate" failure mode: hazard prob is ~flat across time so integrate-to-1 fires too often.
  - Quantify peakiness via: mass/cycle, CV, softmax(logit) entropy vs log(T), and (optional) CE-to-GT touchdown.

Usage (example):
  python tools/analyze_td_hazard_peakiness.py \
    --pred-json debug_output/_smoke200_td_hazard_rollout_alignfix_20260116/Walk_F_freerun_cycles.json \
    --gt-json   debug_output/_smoke200_td_hazard_rollout_alignfix_20260116_ttcgt/Walk_F_freerun_cycles.json \
    --smoothing 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _log_softmax(a: np.ndarray, axis: int) -> np.ndarray:
    lse = _logsumexp(a, axis=axis)
    return a - np.expand_dims(lse, axis=axis)


def _group_by_cycle(steps: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for s in steps:
        out.setdefault(int(s["cycle"]), []).append(s)
    return out


def _stack_key(
    steps: List[Dict[str, Any]],
    key: str,
    *,
    C: int,
    allow_missing: bool = False,
) -> Optional[np.ndarray]:
    rows: List[np.ndarray] = []
    for s in steps:
        v = s.get(key, None)
        if v is None:
            if allow_missing:
                return None
            raise KeyError(f"Missing key={key!r} in per-step JSON entry.")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            # Scalar -> broadcast to per-contact for robustness.
            arr = np.full((C,), float(arr), dtype=np.float64)
        if int(arr.shape[-1]) != int(C):
            raise ValueError(f"{key} has shape {arr.shape}, expected (C,) with C={C}.")
        rows.append(arr)
    return np.stack(rows, axis=0)  # (T,C)


def _fmt_vec(x: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):7.3f}" for v in x.tolist()) + "]"


def _cycle_metrics(
    *,
    p: np.ndarray,            # (T,C) hazard prob
    logit: np.ndarray,        # (T,C) hazard logit
    mask: np.ndarray,         # (T,C) valid mask (True = included)
    gt_event: Optional[np.ndarray],  # (T,C) one-hot-ish touchdown, optional
    smoothing: float,
) -> Dict[str, np.ndarray]:
    if p.shape != logit.shape or p.shape != mask.shape:
        raise ValueError(f"Shape mismatch: p{p.shape}, logit{logit.shape}, mask{mask.shape}.")

    # Mass/cycle (sum of hazard prob across time).
    mass = (p * mask).sum(axis=0)
    denom = np.maximum(mask.sum(axis=0), 1.0)
    mean_p = (p * mask).sum(axis=0) / denom
    std_p = np.sqrt(((p - mean_p) ** 2 * mask).sum(axis=0) / denom)
    cv = std_p / np.maximum(mean_p, 1e-9)

    # Peakiness on prob scale.
    p_masked = np.where(mask, p, -np.inf)
    peak = np.max(p_masked, axis=0)
    peak_ratio = peak / np.maximum(mean_p, 1e-9)
    peak_idx = np.argmax(p_masked, axis=0).astype(np.int64)

    # Peakiness on softmax(logit_over_time) scale.
    logit_masked = np.where(mask, logit, -1e9)
    logp = _log_softmax(logit_masked, axis=0)  # over time
    sp = np.exp(logp)
    entropy = -(sp * logp).sum(axis=0)  # (C,)

    out: Dict[str, np.ndarray] = {
        "mass": mass,
        "mean_p": mean_p,
        "cv": cv,
        "peak_ratio": peak_ratio,
        "peak_idx": peak_idx.astype(np.float64),  # keep numeric for aggregation
        "entropy": entropy,
    }

    if gt_event is not None:
        if gt_event.shape != p.shape:
            raise ValueError(f"gt_event shape {gt_event.shape} mismatch with p {p.shape}.")
        q = gt_event.copy()
        q[~mask] = 0.0
        qsum = q.sum(axis=0, keepdims=True)
        qnorm = q / np.maximum(qsum, 1e-12)
        if smoothing > 0.0:
            unif = mask.astype(np.float64) / np.maximum(mask.sum(axis=0, keepdims=True), 1.0)
            qnorm = (1.0 - smoothing) * qnorm + smoothing * unif
        ce = -(qnorm * logp).sum(axis=0)

        gt_idx = np.full((p.shape[1],), np.nan, dtype=np.float64)
        for c in range(p.shape[1]):
            idx = np.where((gt_event[:, c] > 0.5) & mask[:, c])[0]
            if idx.size > 0:
                gt_idx[c] = float(idx[0])
        peak_dt = peak_idx.astype(np.float64) - gt_idx

        out.update({"ce": ce, "gt_idx": gt_idx, "peak_dt": peak_dt})

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", type=str, required=True, help="freerun_cycles.json to analyze (TDHazard* from model).")
    ap.add_argument(
        "--gt-json",
        type=str,
        default=None,
        help="Optional freerun_cycles.json that contains TTCEventPerC/TTCGTValidPerC for GT touchdown alignment.",
    )
    ap.add_argument("--smoothing", type=float, default=0.05, help="Label smoothing for CE alignment (time axis).")
    ap.add_argument("--no-per-cycle", action="store_true", help="Only print aggregate stats (skip per-cycle lines).")
    args = ap.parse_args()

    pred_path = Path(args.pred_json)
    gt_path = Path(args.gt_json) if args.gt_json else None
    smoothing = float(args.smoothing)
    if smoothing < 0.0:
        smoothing = 0.0
    if smoothing > 0.99:
        smoothing = 0.99

    pred = json.loads(pred_path.read_text())
    pred_steps = pred.get("metrics_per_step", None)
    if not isinstance(pred_steps, list) or not pred_steps:
        raise RuntimeError(f"{pred_path} has no metrics_per_step list.")

    C = len(pred_steps[0].get("TDHazardProbPerC", []))
    if C <= 0:
        raise RuntimeError(f"{pred_path} missing TDHazardProbPerC or empty.")

    gt_steps: Optional[List[Dict[str, Any]]] = None
    if gt_path is not None:
        gt = json.loads(gt_path.read_text())
        gt_steps = gt.get("metrics_per_step", None)
        if not isinstance(gt_steps, list) or not gt_steps:
            raise RuntimeError(f"{gt_path} has no metrics_per_step list.")
        if len(gt_steps) != len(pred_steps):
            print(f"[WARN] step count mismatch: pred={len(pred_steps)} gt={len(gt_steps)} (will align by cycle+step_in_cycle).")

    pred_by = _group_by_cycle(pred_steps)
    gt_by = _group_by_cycle(gt_steps) if gt_steps is not None else None

    print(f"pred_json: {pred_path}")
    print(f"  phase_reset_source_applied={pred.get('phase_reset_source_applied')!r} cycle_len={pred.get('cycle_len')} rounds={pred.get('rounds')}")
    if gt_path is not None and gt_steps is not None:
        gt = json.loads(gt_path.read_text())
        print(f"gt_json:   {gt_path}")
        print(f"  phase_reset_source_applied={gt.get('phase_reset_source_applied')!r} cycle_len={gt.get('cycle_len')} rounds={gt.get('rounds')}")
    print(f"C={C}  log(T=87)={float(np.log(87.0)):.6f}")

    keys = ["mass", "cv", "entropy", "peak_ratio"]
    if gt_by is not None:
        keys += ["ce", "peak_dt"]

    agg: Dict[str, List[np.ndarray]] = {k: [] for k in keys}

    for cyc in sorted(pred_by.keys()):
        pred_c = pred_by[cyc]
        p = _stack_key(pred_c, "TDHazardProbPerC", C=C)
        logit = _stack_key(pred_c, "TDHazardLogitPerC", C=C)
        mask = np.ones_like(p, dtype=bool)

        gt_event = None
        if gt_by is not None:
            gt_c = gt_by.get(cyc, None)
            if gt_c is None:
                raise RuntimeError(f"gt_json missing cycle={cyc}.")
            gt_event = _stack_key(gt_c, "TTCEventPerC", C=C)
            gt_valid = _stack_key(gt_c, "TTCGTValidPerC", C=C)
            mask = (gt_valid > 0.5)

        m = _cycle_metrics(p=p, logit=logit, mask=mask, gt_event=gt_event, smoothing=smoothing)

        if not bool(args.no_per_cycle):
            print(f"\ncycle {cyc} (T={p.shape[0]})")
            print(f"  mass      {_fmt_vec(m['mass'])}   mean_p {_fmt_vec(m['mean_p'])}")
            print(f"  cv        {_fmt_vec(m['cv'])}   peak/mean {_fmt_vec(m['peak_ratio'])}")
            print(f"  entropy   {_fmt_vec(m['entropy'])}")
            if gt_event is not None:
                print(f"  CE(s={smoothing}) {_fmt_vec(m['ce'])}   peak_dt {m['peak_dt'].tolist()}")

        for k in keys:
            agg[k].append(m[k])

    print("\n-- aggregate over cycles --")
    for k in keys:
        arr = np.stack(agg[k], axis=0)
        print(f"{k:10s} mean {np.nanmean(arr, axis=0)}  std {np.nanstd(arr, axis=0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

