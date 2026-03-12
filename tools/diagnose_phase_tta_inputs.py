#!/usr/bin/env python3
"""
Diagnose whether a "prev_phase / prev_TTA" recurrent input can be derived cleanly
from the contact signal in a `run_freerun_cycles` JSON.

This is meant for debugging off-by-one / mis-definition issues BEFORE wiring a
Learned-Phase-Advance (LPA) or Time-To-Arrival (TTA) state into the model.

The script:
  - loads per-step soft contacts (default: ContactGTPerC)
  - detects events (touchdown / liftoff) by threshold crossing
  - derives per-foot phase (within event-to-event intervals) and per-foot TTA
  - checks basic consistency (e.g. TTA should decrement by 1 each step between resets)

Example
-------
python tools/diagnose_phase_tta_inputs.py \
  --json debug_output/_tmp_contacts_override/meas_gt/Walk_F_freerun_cycles.json \
  --source gt --event-kind touchdown --thr 0.5 --show 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


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


def _as_float_matrix(x: Any) -> Optional[List[List[float]]]:
    if not isinstance(x, list) or not x:
        return None
    out: List[List[float]] = []
    for row in x:
        v = _as_float_list(row)
        if v is None:
            return None
        out.append(v)
    return out


def _pick_contacts_from_step(step: Dict[str, Any], *, source: str) -> Optional[List[float]]:
    source = str(source).strip().lower()
    if source in ("gt", "contactgt"):
        return _as_float_list(step.get("ContactGTPerC"))
    if source in ("meas", "contactmeas"):
        return _as_float_list(step.get("ContactMeasPerC"))
    if source in ("plan", "contactplan"):
        return _as_float_list(step.get("ContactPlanPerC"))
    raise ValueError(f"Unknown --source={source!r} (expected gt/meas/plan).")


def _detect_events(x: np.ndarray, *, thr: float, kind: str) -> np.ndarray:
    """
    Detect threshold crossings for a 1D soft contact signal x[t] in [0,1].
    Returns indices e where the event is considered to occur at frame e.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.zeros((0,), dtype=np.int64)
    prev = x[:-1]
    cur = x[1:]
    kind = str(kind).strip().lower()
    if kind in ("touchdown", "td", "up", "rise"):
        m = (prev < thr) & (cur >= thr)
    elif kind in ("liftoff", "lo", "down", "fall"):
        m = (prev >= thr) & (cur < thr)
    elif kind in ("both", "any"):
        m = ((prev < thr) & (cur >= thr)) | ((prev >= thr) & (cur < thr))
    else:
        raise ValueError(f"Unknown event kind {kind!r}.")
    return (np.where(m)[0] + 1).astype(np.int64)


def _suppress_events(events: np.ndarray, *, min_interval: int) -> np.ndarray:
    """
    Simple cooldown filter for event indices.

    Keeps the first event in each cluster and drops any subsequent events that occur
    within `min_interval` frames of the last kept event.
    """
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


def _tta_to_next_event(events: np.ndarray, T: int) -> np.ndarray:
    """
    tta[t] = next_event_index - t, for the next event index >= t.
    NaN when no future event exists.
    """
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    t = np.arange(int(T), dtype=np.int64)
    if events.size == 0:
        return np.full((T,), np.nan, dtype=np.float64)
    j = np.searchsorted(events, t, side="left")
    tta = np.full((T,), np.nan, dtype=np.float64)
    ok = j < events.size
    tta[ok] = (events[j[ok]] - t[ok]).astype(np.float64)
    return tta


def _phase_between_events(events: np.ndarray, T: int) -> np.ndarray:
    """
    Build a normalized phase in [0,1) between consecutive events:
      phase[t] = (t - e_k) / (e_{k+1} - e_k)  for t in [e_k, e_{k+1})
    NaN outside covered intervals.
    """
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    phase = np.full((T,), np.nan, dtype=np.float64)
    if events.size < 2:
        return phase
    for a, b in zip(events[:-1].tolist(), events[1:].tolist()):
        a = int(a)
        b = int(b)
        if b <= a:
            continue
        lo = max(0, a)
        hi = min(int(T), b)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi, dtype=np.int64)
        phase[idx] = (idx - a) / float(b - a)
    return phase


def _phase_vec(phase01: np.ndarray) -> np.ndarray:
    phase01 = np.asarray(phase01, dtype=np.float64)
    ang = 2.0 * np.pi * phase01
    out = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
    out[~np.isfinite(out)] = np.nan
    return out


@dataclass
class FootStats:
    name: str
    events: np.ndarray
    periods: np.ndarray
    tta: np.ndarray
    phase01: np.ndarray


def _summarize_periods(periods: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = np.asarray(periods, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return None, None, None
    return float(p.min()), float(p.mean()), float(p.max())


def _check_tta_consistency(tta: np.ndarray, events: np.ndarray) -> Tuple[int, int, float]:
    """
    Between events, tta should decrement by 1 each frame; at event frames, it resets.
    This is a sanity check of the derived TTA targets.
    """
    tta = np.asarray(tta, dtype=np.float64).reshape(-1)
    T = int(tta.size)
    if T < 2:
        return 0, 0, float("nan")
    ev = set(int(x) for x in np.asarray(events, dtype=np.int64).reshape(-1).tolist())
    bad = 0
    total = 0
    for t in range(1, T):
        if not (np.isfinite(tta[t]) and np.isfinite(tta[t - 1])):
            continue
        # Ignore reset steps (event at t or t-1).
        if t in ev or (t - 1) in ev:
            continue
        total += 1
        if abs((tta[t] - tta[t - 1]) + 1.0) > 1e-6:
            bad += 1
    frac = (bad / total) if total > 0 else float("nan")
    return int(bad), int(total), float(frac)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose prev_phase / prev_TTA signals from contacts in freerun JSON.")
    ap.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to *_freerun_cycles.json (metrics_per_step/Contact*PerC) or *_teacher_pred.json (contacts_pred/aux_inputs).",
    )
    ap.add_argument("--source", type=str, default="gt", choices=("gt", "meas", "plan"), help="Which contacts to use.")
    ap.add_argument(
        "--event-kind",
        type=str,
        default="touchdown",
        choices=("touchdown", "liftoff", "both"),
        help="Event definition via threshold crossing.",
    )
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for event detection.")
    ap.add_argument(
        "--min-interval",
        type=int,
        default=0,
        help="Optional cooldown (frames) to suppress duplicate events after threshold crossing (0 disables).",
    )
    ap.add_argument("--show", type=int, default=6, help="How many events to print per foot.")
    ap.add_argument("--preview-rows", type=int, default=14, help="How many initial rows to print.")
    args = ap.parse_args()

    path = Path(args.json).expanduser()
    obj = _load_json(path)
    steps = obj.get("metrics_per_step", None)

    contacts: Optional[List[List[float]]] = None
    if isinstance(steps, list) and steps:
        picked: List[List[float]] = []
        for st in steps:
            v = _pick_contacts_from_step(st, source=args.source)
            if v is None:
                continue
            picked.append(v)
        contacts = picked
    else:
        # Fallback: teacher rollout JSON produced by train.validate.run_teacher_rollout
        src = str(args.source).strip().lower()
        if src in ("plan", "contactplan"):
            raise SystemExit("Teacher-pred JSON has no contacts_plan; use --source gt/meas.")
        if src in ("gt", "contactgt"):
            contacts = _as_float_matrix(obj.get("aux_inputs", {}).get("contacts"))
        elif src in ("meas", "contactmeas"):
            contacts = _as_float_matrix(obj.get("contacts_pred", {}).get("contacts_meas"))
        else:
            raise SystemExit(f"Unknown --source={args.source!r}.")

    if not isinstance(contacts, list) or not contacts:
        raise SystemExit("Invalid JSON: no usable contacts found (need metrics_per_step or contacts_pred/aux_inputs).")

    C = len(contacts[0])
    a = np.asarray(contacts, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != C:
        raise SystemExit("Internal error: contact array shape mismatch.")

    T = int(a.shape[0])
    names = [f"C{c}" for c in range(C)]
    if C == 2:
        names = ["L", "R"]

    feet: List[FootStats] = []
    for c in range(C):
        x = a[:, c]
        ev = _detect_events(x, thr=float(args.thr), kind=str(args.event_kind))
        ev = _suppress_events(ev, min_interval=int(args.min_interval))
        periods = np.diff(ev).astype(np.int64) if ev.size >= 2 else np.zeros((0,), dtype=np.int64)
        tta = _tta_to_next_event(ev, T)
        phase01 = _phase_between_events(ev, T)
        feet.append(FootStats(name=names[c], events=ev, periods=periods, tta=tta, phase01=phase01))

    print(f"[JSON] {path}")
    print(f"[Contacts] source={args.source} T={T} C={C} event_kind={args.event_kind} thr={float(args.thr):.3f}")
    print(f"[Contacts] mean_per_c={a.mean(axis=0).tolist()} std_per_c={a.std(axis=0).tolist()}")

    print()
    for ft in feet:
        pmin, pmean, pmax = _summarize_periods(ft.periods)
        bad, total, frac = _check_tta_consistency(ft.tta, ft.events)
        print(
            f"[{ft.name}] events={int(ft.events.size)} "
            f"period[min/mean/max]={pmin},{pmean},{pmax}  "
            f"TTA_consistency_bad/total={bad}/{total} (frac={frac})"
        )
        if ft.events.size > 0:
            show = min(int(args.show), int(ft.events.size))
            print(f"  first_events={ft.events[:show].tolist()}")

    # Preview table (first N frames)
    print()
    n = max(0, min(int(args.preview_rows), T))
    if n > 0:
        # Use foot0 as phase proxy for the preview (still print all contacts).
        phase0 = feet[0].phase01
        tta0 = feet[0].tta
        d = np.zeros_like(a)
        d[1:] = a[1:] - a[:-1]
        print("[Preview] t | contacts | d_contacts | phase0 | tta0")
        for t in range(n):
            c_str = ",".join(f"{v:.3f}" for v in a[t].tolist())
            d_str = ",".join(f"{v:+.3f}" for v in d[t].tolist())
            ph = phase0[t]
            tt = tta0[t]
            ph_s = "NA" if not np.isfinite(ph) else f"{ph:.3f}"
            tt_s = "NA" if not np.isfinite(tt) else f"{tt:.1f}"
            print(f"  {t:03d} | [{c_str}] | [{d_str}] | {ph_s:>6} | {tt_s:>6}")

    # Output a small derived array block for easy integration into model code later.
    # (These are "current" signals; "prev_*" is simply a 1-step shift at runtime.)
    phase0_vec = _phase_vec(feet[0].phase01)  # (T,2)
    print()
    ok = np.isfinite(phase0_vec).all(axis=1)
    print(f"[Derived] phase0_vec finite frames: {int(ok.sum())}/{T}")
    if int(ok.sum()) > 0:
        idx = int(np.where(ok)[0][0])
        print(f"[Derived] first finite phase0_vec @t={idx}: {phase0_vec[idx].tolist()}")
    print("[Note] If you feed prev_phase, it should be phase_vec[t-1] (shifted by 1) with a well-defined init at t=0.")


if __name__ == "__main__":
    main()
