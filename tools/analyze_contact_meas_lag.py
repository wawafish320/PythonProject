#!/usr/bin/env python3
"""
Lag / cross-correlation diagnostics for contact_meas_head teacher-rollout outputs.

This complements `tools/analyze_contact_meas_head.py` by quantifying temporal lag:
we search for the lag (in frames) that maximizes Pearson correlation between
`contacts_pred.contacts_meas` and `aux_inputs.contacts`.

Lag convention:
  corr(lag) = corr(pred[t], gt[t + lag]) over the overlap.
So:
  - best_lag < 0 means pred is delayed (matches earlier GT)
  - best_lag > 0 means pred is advanced (matches later GT)

Example:
  python tools/analyze_contact_meas_lag.py \
    --json debug_output/_tmp_teacher_debug/teacher_rollout_measdiag_all/Walk_F_teacher_pred.json \
    --max-lag 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    sa = float(np.sqrt((a * a).sum()))
    sb = float(np.sqrt((b * b).sum()))
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    return float((a * b).sum() / (sa * sb))


def _corr_over_lags(pred: np.ndarray, gt: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    T = int(min(pred.shape[0], gt.shape[0]))
    pred = pred[:T]
    gt = gt[:T]
    max_lag = int(max(0, max_lag))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    corrs = np.full(lags.shape, np.nan, dtype=np.float64)
    for i, lag in enumerate(lags.tolist()):
        if lag >= 0:
            a = pred[: T - lag]
            b = gt[lag:]
        else:
            a = pred[-lag:]
            b = gt[: T + lag]
        corrs[i] = _pearson_corr(a, b)
    return lags, corrs


def _schmitt_state(x: np.ndarray, *, on_th: float, off_th: float) -> np.ndarray:
    """
    Convert a soft contact signal to a binary state using hysteresis (Schmitt trigger).

    State update:
      - x[t] > on_th  => 1
      - x[t] < off_th => 0
      - otherwise keep previous
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.astype(np.int64)
    on_th = float(on_th)
    off_th = float(off_th)
    state = np.zeros_like(x, dtype=np.int64)
    state[0] = 1 if (np.isfinite(x[0]) and x[0] > on_th) else 0
    for t in range(1, x.size):
        v = x[t]
        prev = int(state[t - 1])
        if not np.isfinite(v):
            state[t] = prev
        elif v > on_th:
            state[t] = 1
        elif v < off_th:
            state[t] = 0
        else:
            state[t] = prev
    return state


def _edge_times(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rising_idxs, falling_idxs) from a binary state sequence."""
    s = np.asarray(state, dtype=np.int64).reshape(-1)
    if s.size < 2:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    prev = s[:-1]
    cur = s[1:]
    rising = np.where((prev == 0) & (cur == 1))[0] + 1
    falling = np.where((prev == 1) & (cur == 0))[0] + 1
    return rising.astype(np.int64), falling.astype(np.int64)


def _match_edges(gt_edges: np.ndarray, pred_edges: np.ndarray, *, max_shift: int) -> Tuple[np.ndarray, int, int]:
    """
    Match each GT edge to the nearest unused pred edge within +/- max_shift.

    Returns:
      lags: pred_t - gt_t for matched edges
      n_gt: len(gt_edges)
      n_pred: len(pred_edges)
    """
    gt_edges = np.asarray(gt_edges, dtype=np.int64).reshape(-1)
    pred_edges = np.asarray(pred_edges, dtype=np.int64).reshape(-1)
    max_shift = int(max(0, max_shift))
    if gt_edges.size == 0 or pred_edges.size == 0:
        return np.zeros((0,), dtype=np.int64), int(gt_edges.size), int(pred_edges.size)

    used = np.zeros((pred_edges.size,), dtype=bool)
    lags: list[int] = []
    for gt_t in gt_edges.tolist():
        best_j = None
        best_abs = None
        best_lag = None
        for j, pred_t in enumerate(pred_edges.tolist()):
            if used[j]:
                continue
            lag = int(pred_t) - int(gt_t)
            a = abs(lag)
            if a > max_shift:
                continue
            if best_abs is None or a < best_abs:
                best_abs = a
                best_j = j
                best_lag = lag
        if best_j is not None and best_lag is not None:
            used[best_j] = True
            lags.append(int(best_lag))
    return np.asarray(lags, dtype=np.int64), int(gt_edges.size), int(pred_edges.size)


def _edge_slope_match_lags(gt_edges: np.ndarray, pred_signal: np.ndarray, *, max_shift: int, kind: str) -> np.ndarray:
    """
    Match each GT edge to the strongest slope in the pred signal within +/- max_shift.

    This works even when the pred signal never crosses the Schmitt thresholds.
      - kind='rising': maximize d_pred
      - kind='falling': minimize d_pred
    """
    gt_edges = np.asarray(gt_edges, dtype=np.int64).reshape(-1)
    x = np.asarray(pred_signal, dtype=np.float64).reshape(-1)
    T = int(x.size)
    if gt_edges.size == 0 or T <= 0:
        return np.zeros((0,), dtype=np.int64)
    d = np.diff(x, prepend=x[:1])
    max_shift = int(max(0, max_shift))
    lags: list[int] = []
    for gt_t in gt_edges.tolist():
        s = max(0, int(gt_t) - max_shift)
        e = min(T, int(gt_t) + max_shift + 1)
        seg = d[s:e]
        if seg.size == 0:
            continue
        if kind == "falling":
            j = int(np.argmin(seg))
        else:
            j = int(np.argmax(seg))
        pred_t = s + j
        lags.append(int(pred_t) - int(gt_t))
    return np.asarray(lags, dtype=np.int64)


def _time_to_threshold(
    pred_signal: np.ndarray,
    gt_edges: np.ndarray,
    *,
    threshold: float,
    max_steps: int,
    direction: str,
) -> Tuple[list[float], list[Optional[int]]]:
    """
    For each GT edge time t0, compute:
      - pred_at_gt: pred[t0]
      - dt: min dt>=0 such that pred[t0+dt] crosses threshold within max_steps.

    direction:
      - 'le': find first pred <= threshold (falling)
      - 'ge': find first pred >= threshold (rising)
    """
    x = np.asarray(pred_signal, dtype=np.float64).reshape(-1)
    edges = np.asarray(gt_edges, dtype=np.int64).reshape(-1)
    max_steps = int(max(0, max_steps))
    thr = float(threshold)
    T = int(x.size)
    pred_at: list[float] = []
    dts: list[Optional[int]] = []
    if edges.size == 0 or T <= 0:
        return pred_at, dts

    for t0 in edges.tolist():
        t0 = int(t0)
        if t0 < 0 or t0 >= T:
            continue
        pred_at.append(float(x[t0]))
        end = min(T, t0 + max_steps + 1)
        seg = x[t0:end]
        if seg.size == 0:
            dts.append(None)
            continue
        if direction == "ge":
            idx = np.where(seg >= thr)[0]
        else:
            idx = np.where(seg <= thr)[0]
        dts.append(int(idx[0]) if idx.size else None)
    return pred_at, dts


def _stats_optional_int(values: list[Optional[int]]) -> Dict[str, object]:
    clean = [int(v) for v in values if v is not None]
    if not clean:
        return {"median": None, "mean": None, "values": values}
    a = np.asarray(clean, dtype=np.float64)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "values": values}


def _stats_float(values: list[float]) -> Dict[str, object]:
    if not values:
        return {"median": None, "mean": None, "values": values}
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "values": values}


def analyze(
    path: Path,
    *,
    max_lag: int = 30,
    on_th: float = 0.8,
    off_th: float = 0.1,
    event_max_shift: Optional[int] = None,
    mid_th: float = 0.55,
    time_window: Optional[int] = None,
) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gt = np.asarray(data.get("aux_inputs", {}).get("contacts", []), dtype=np.float64)
    pred = np.asarray(data.get("contacts_pred", {}).get("contacts_meas", []), dtype=np.float64)
    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"Expected gt/pred to be 2D arrays, got gt={gt.shape}, pred={pred.shape}.")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch gt={gt.shape} pred={pred.shape}.")
    if gt.shape[1] < 2:
        raise ValueError(f"Expected contact_dim>=2, got C={gt.shape[1]}.")

    T = int(gt.shape[0])
    out: Dict[str, object] = {
        "clip": data.get("clip"),
        "json": str(path),
        "T": T,
        "C": int(gt.shape[1]),
        "max_lag": int(max_lag),
    }

    results = {}
    for ch, name in [(0, "L"), (1, "R")]:
        lags, corrs = _corr_over_lags(pred[:, ch], gt[:, ch], max_lag=max_lag)
        idx = int(np.nanargmax(corrs))
        results[name] = {
            "best_lag": int(lags[idx]),
            "best_corr": float(corrs[idx]),
            "lags": lags.tolist(),
            "corrs": corrs.tolist(),
        }

        # Edge/transition correlation (diff) to highlight hysteresis.
        d_pred = np.diff(pred[:, ch], prepend=pred[:1, ch])
        d_gt = np.diff(gt[:, ch], prepend=gt[:1, ch])
        l2, c2 = _corr_over_lags(d_pred, d_gt, max_lag=max_lag)
        idx2 = int(np.nanargmax(c2))
        results[f"{name}_delta"] = {
            "best_lag": int(l2[idx2]),
            "best_corr": float(c2[idx2]),
            "lags": l2.tolist(),
            "corrs": c2.tolist(),
        }

    # Cross-channel lagged correlation (helps detect L/R confusion with phase shift).
    for (pred_ch, gt_ch, key) in [
        (0, 1, "L_vs_GT_R"),
        (1, 0, "R_vs_GT_L"),
    ]:
        lags, corrs = _corr_over_lags(pred[:, pred_ch], gt[:, gt_ch], max_lag=max_lag)
        idx = int(np.nanargmax(corrs))
        results[key] = {
            "best_lag": int(lags[idx]),
            "best_corr": float(corrs[idx]),
            "lags": lags.tolist(),
            "corrs": corrs.tolist(),
        }

        d_pred = np.diff(pred[:, pred_ch], prepend=pred[:1, pred_ch])
        d_gt = np.diff(gt[:, gt_ch], prepend=gt[:1, gt_ch])
        l2, c2 = _corr_over_lags(d_pred, d_gt, max_lag=max_lag)
        idx2 = int(np.nanargmax(c2))
        results[f"{key}_delta"] = {
            "best_lag": int(l2[idx2]),
            "best_corr": float(c2[idx2]),
            "lags": l2.tolist(),
            "corrs": c2.tolist(),
        }

    out["corr"] = results

    # Event-based lag (threshold crossings) using hysteresis thresholds.
    # This better separates "true delay" from "phase ambiguity".
    on_th = float(on_th)
    off_th = float(off_th)
    max_shift = int(max_lag if event_max_shift is None else event_max_shift)
    mid_th = float(mid_th)
    time_window_i = int(max_shift if time_window is None else time_window)
    event: Dict[str, object] = {"on_th": on_th, "off_th": off_th, "max_shift": int(max_shift)}
    for ch, name in [(0, "L"), (1, "R")]:
        s_gt = _schmitt_state(gt[:, ch], on_th=on_th, off_th=off_th)
        s_pr = _schmitt_state(pred[:, ch], on_th=on_th, off_th=off_th)
        gt_rise, gt_fall = _edge_times(s_gt)
        pr_rise, pr_fall = _edge_times(s_pr)
        rise_lags, n_gt_r, n_pr_r = _match_edges(gt_rise, pr_rise, max_shift=max_shift)
        fall_lags, n_gt_f, n_pr_f = _match_edges(gt_fall, pr_fall, max_shift=max_shift)
        rise_slope_lags = _edge_slope_match_lags(gt_rise, pred[:, ch], max_shift=max_shift, kind="rising")
        fall_slope_lags = _edge_slope_match_lags(gt_fall, pred[:, ch], max_shift=max_shift, kind="falling")

        # Robust post-event "time-to-threshold" metrics (avoids slope ambiguity).
        # - falling: how long after GT_fall until pred <= threshold?
        # - rising:  how long after GT_rise until pred >= threshold?
        rise_pred_at, rise_to_on = _time_to_threshold(pred[:, ch], gt_rise, threshold=on_th, max_steps=time_window_i, direction="ge")
        _, rise_to_mid = _time_to_threshold(pred[:, ch], gt_rise, threshold=mid_th, max_steps=time_window_i, direction="ge")
        fall_pred_at, fall_to_on = _time_to_threshold(pred[:, ch], gt_fall, threshold=on_th, max_steps=time_window_i, direction="le")
        _, fall_to_mid = _time_to_threshold(pred[:, ch], gt_fall, threshold=mid_th, max_steps=time_window_i, direction="le")
        _, fall_to_off = _time_to_threshold(pred[:, ch], gt_fall, threshold=off_th, max_steps=time_window_i, direction="le")

        event[name] = {
            "gt_edges": {"rising": gt_rise.tolist(), "falling": gt_fall.tolist()},
            "rising": {
                "n_gt": int(n_gt_r),
                "n_pred": int(n_pr_r),
                "n_matched": int(rise_lags.size),
                "median_lag": float(np.median(rise_lags)) if rise_lags.size else None,
                "mean_lag": float(rise_lags.mean()) if rise_lags.size else None,
                "lags": rise_lags.tolist(),
            },
            "falling": {
                "n_gt": int(n_gt_f),
                "n_pred": int(n_pr_f),
                "n_matched": int(fall_lags.size),
                "median_lag": float(np.median(fall_lags)) if fall_lags.size else None,
                "mean_lag": float(fall_lags.mean()) if fall_lags.size else None,
                "lags": fall_lags.tolist(),
            },
            "rising_slope": {
                "n_gt": int(gt_rise.size),
                "median_lag": float(np.median(rise_slope_lags)) if rise_slope_lags.size else None,
                "mean_lag": float(rise_slope_lags.mean()) if rise_slope_lags.size else None,
                "lags": rise_slope_lags.tolist(),
            },
            "falling_slope": {
                "n_gt": int(gt_fall.size),
                "median_lag": float(np.median(fall_slope_lags)) if fall_slope_lags.size else None,
                "mean_lag": float(fall_slope_lags.mean()) if fall_slope_lags.size else None,
                "lags": fall_slope_lags.tolist(),
            },
            "rising_time": {
                "window": int(time_window_i),
                "mid_th": float(mid_th),
                "pred_at_gt": _stats_float(rise_pred_at),
                "time_to_ge_on": _stats_optional_int(rise_to_on),
                "time_to_ge_mid": _stats_optional_int(rise_to_mid),
            },
            "falling_time": {
                "window": int(time_window_i),
                "mid_th": float(mid_th),
                "pred_at_gt": _stats_float(fall_pred_at),
                "time_to_le_on": _stats_optional_int(fall_to_on),
                "time_to_le_mid": _stats_optional_int(fall_to_mid),
                "time_to_le_off": _stats_optional_int(fall_to_off),
            },
        }
    out["event"] = event
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to *_teacher_pred.json from run_teacher_rollout.")
    ap.add_argument("--max-lag", type=int, default=30, help="Search lag in [-max_lag, +max_lag].")
    ap.add_argument("--on-th", type=float, default=0.8, help="Event ON threshold for edge-based lag.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Event OFF threshold for edge-based lag.")
    ap.add_argument(
        "--mid-th",
        type=float,
        default=0.55,
        help="Mid threshold used for time-to metrics (e.g., time to pred<=mid_th after GT fall).",
    )
    ap.add_argument(
        "--event-max-shift",
        type=int,
        default=None,
        help="Max shift (frames) for matching GT/pred edges; default uses --max-lag.",
    )
    ap.add_argument(
        "--time-window",
        type=int,
        default=None,
        help="Window (frames) for time-to-threshold metrics; default uses --event-max-shift/--max-lag.",
    )
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    path = Path(args.json)
    summary = analyze(
        path,
        max_lag=int(args.max_lag),
        on_th=float(args.on_th),
        off_th=float(args.off_th),
        mid_th=float(args.mid_th),
        event_max_shift=(None if args.event_max_shift is None else int(args.event_max_shift)),
        time_window=(None if args.time_window is None else int(args.time_window)),
    )
    clip = summary.get("clip")
    print(f"[ContactMeasLag] clip={clip} json={path} T={summary.get('T')} max_lag={summary.get('max_lag')}")

    corr = summary.get("corr", {})
    for key in ("L", "R", "L_delta", "R_delta", "L_vs_GT_R", "R_vs_GT_L"):
        block = corr.get(key) if isinstance(corr, dict) else None
        if not isinstance(block, dict):
            continue
        print(f"  {key}: best_lag={block.get('best_lag')} best_corr={block.get('best_corr'):.6f}")

    ev = summary.get("event", {})
    if isinstance(ev, dict):
        for ch in ("L", "R"):
            b = ev.get(ch)
            if not isinstance(b, dict):
                continue
            for edge in ("rising", "falling"):
                e = b.get(edge)
                if not isinstance(e, dict):
                    continue
                print(
                    f"  {ch}_{edge}: n_matched={e.get('n_matched')} "
                    f"median_lag={e.get('median_lag')} mean_lag={e.get('mean_lag')}"
                )
            for edge in ("rising_slope", "falling_slope"):
                e = b.get(edge)
                if not isinstance(e, dict):
                    continue
                print(
                    f"  {ch}_{edge}: n_gt={e.get('n_gt')} "
                    f"median_lag={e.get('median_lag')} mean_lag={e.get('mean_lag')}"
                )
            # Robust post-event time-to threshold metrics
            for edge in ("rising_time", "falling_time"):
                e = b.get(edge)
                if not isinstance(e, dict):
                    continue
                if edge == "falling_time":
                    tt = e.get("time_to_le_mid", {})
                    pa = e.get("pred_at_gt", {})
                    if isinstance(tt, dict) and isinstance(pa, dict):
                        print(
                            f"  {ch}_fall_time_to<={e.get('mid_th')}: median_dt={tt.get('median')} "
                            f"pred_at_fall_med={pa.get('median')}"
                        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
