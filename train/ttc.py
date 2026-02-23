from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import numpy as np

EventKind = Literal["touchdown", "liftoff", "both"]
EventSelect = Literal["all", "first", "last", "longest_run"]


def detect_contact_events_np(
    contacts: np.ndarray,
    *,
    thr: float = 0.5,
    kind: EventKind = "touchdown",
    hyst: float = 0.0,
) -> np.ndarray:
    """
    Detect per-channel contact events by threshold crossing (optionally with hysteresis),
    matching EventMotionModel's phase reset logic:

    - touchdown: prev < (thr-hyst) and cur >= thr
    - liftoff:  prev >= (thr+hyst) and cur < thr
    - both:     either of the above

    Args:
        contacts: (T, C) float array in [0, 1] (soft contact score).
        thr: crossing threshold.
        kind: event kind.
        hyst: hysteresis (0 disables), clamped to [0, 0.49].

    Returns:
        events: (T, C) bool array. events[t, c] = True means an event occurs at frame t for channel c.
                Note: events[0, :] is always False (no prev frame in non-cyclic definition).
    """
    c = np.asarray(contacts, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"contacts must be 2D (T,C), got shape={c.shape}")
    T, C = int(c.shape[0]), int(c.shape[1])
    if T <= 0 or C <= 0:
        return np.zeros((max(0, T), max(0, C)), dtype=bool)

    thr = float(thr)
    if not math.isfinite(thr):
        thr = 0.5
    thr = max(0.0, min(1.0, thr))

    hyst = float(hyst or 0.0)
    if not math.isfinite(hyst):
        hyst = 0.0
    hyst = max(0.0, min(0.49, hyst))

    thr_low = max(0.0, thr - hyst)
    thr_high = min(1.0, thr + hyst)

    prev = c[:-1]
    cur = c[1:]

    kind_n = str(kind).strip().lower()
    if kind_n in ("td", "touch", "touchdown", "rise", "up"):
        m = (prev < thr_low) & (cur >= thr)
    elif kind_n in ("lo", "lift", "liftoff", "fall", "down"):
        m = (prev >= thr_high) & (cur < thr)
    elif kind_n in ("both", "any"):
        m = ((prev < thr_low) & (cur >= thr)) | ((prev >= thr_high) & (cur < thr))
    else:
        raise ValueError(f"Unknown event kind {kind!r} (expected touchdown/liftoff/both).")

    out = np.zeros((T, C), dtype=bool)
    out[1:] = m
    return out


def _detect_contact_events_cyclic_np(
    contacts: np.ndarray,
    *,
    thr: float,
    kind: EventKind,
    hyst: float,
) -> np.ndarray:
    """
    Cyclic (wrap-around) version of detect_contact_events_np.

    - events[0] is computed by comparing prev=contacts[-1] and cur=contacts[0].
    - other indices follow the standard prev/cur definition.
    """
    c = np.asarray(contacts, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"contacts must be 2D (T,C), got shape={c.shape}")
    T, C = int(c.shape[0]), int(c.shape[1])
    if T <= 0 or C <= 0:
        return np.zeros((max(0, T), max(0, C)), dtype=bool)
    if T == 1:
        return np.zeros((1, C), dtype=bool)

    base = detect_contact_events_np(c, thr=float(thr), kind=kind, hyst=float(hyst))
    # Compute index0 event using cyclic prev=last, cur=first.
    thr = float(thr)
    thr = max(0.0, min(1.0, thr)) if math.isfinite(thr) else 0.5
    hyst = float(hyst or 0.0)
    hyst = max(0.0, min(0.49, hyst)) if math.isfinite(hyst) else 0.0
    thr_low = max(0.0, thr - hyst)
    thr_high = min(1.0, thr + hyst)
    prev0 = c[-1]
    cur0 = c[0]
    kind_n = str(kind).strip().lower()
    if kind_n in ("td", "touch", "touchdown", "rise", "up"):
        m0 = (prev0 < thr_low) & (cur0 >= thr)
    elif kind_n in ("lo", "lift", "liftoff", "fall", "down"):
        m0 = (prev0 >= thr_high) & (cur0 < thr)
    elif kind_n in ("both", "any"):
        m0 = ((prev0 < thr_low) & (cur0 >= thr)) | ((prev0 >= thr_high) & (cur0 < thr))
    else:
        raise ValueError(f"Unknown event kind {kind!r} (expected touchdown/liftoff/both).")
    base[0] = m0
    return base


def _select_single_event_per_channel_cyclic(
    contacts: np.ndarray,
    events: np.ndarray,
    *,
    thr: float,
    kind: EventKind,
    hyst: float,
    select: EventSelect,
) -> np.ndarray:
    """
    Select at most one event per channel, within a single cycle (cyclic semantics).

    This is used to make TTC a stable clock anchor when soft_contact_score has
    multiple threshold crossings inside one cycle.

    Selection modes:
    - all: keep all events (no change)
    - first/last: keep the earliest/latest event index
    - longest_run: keep the event preceded by the longest opposite-state run
      (touchdown: longest OFF run; liftoff: longest ON run). For kind=both, falls back to 'first'.
    """
    c = np.asarray(contacts, dtype=np.float64)
    ev = np.asarray(events, dtype=bool)
    if c.ndim != 2 or ev.ndim != 2 or c.shape != ev.shape:
        raise ValueError(f"contacts/events shape mismatch: contacts={c.shape}, events={ev.shape}")
    T, C = int(c.shape[0]), int(c.shape[1])
    if T <= 0 or C <= 0:
        return ev
    sel = str(select).strip().lower()
    if sel in ("", "all", "none"):
        return ev
    out = np.zeros_like(ev, dtype=bool)

    thr = float(thr)
    thr = max(0.0, min(1.0, thr)) if math.isfinite(thr) else 0.5
    hyst = float(hyst or 0.0)
    hyst = max(0.0, min(0.49, hyst)) if math.isfinite(hyst) else 0.0
    thr_low = max(0.0, thr - hyst)
    thr_high = min(1.0, thr + hyst)

    kind_n = str(kind).strip().lower()
    if kind_n in ("td", "touch", "touchdown", "rise", "up"):
        kind_n = "touchdown"
    elif kind_n in ("lo", "lift", "liftoff", "fall", "down"):
        kind_n = "liftoff"
    elif kind_n in ("both", "any"):
        kind_n = "both"
    else:
        kind_n = "touchdown"

    for ch in range(C):
        idx = np.where(ev[:, ch])[0].astype(np.int64)
        if idx.size == 0:
            continue
        if sel == "first":
            out[int(idx.min()), ch] = True
            continue
        if sel == "last":
            out[int(idx.max()), ch] = True
            continue

        if sel == "longest_run":
            if kind_n == "touchdown":
                mask = c[:, ch] < thr_low
            elif kind_n == "liftoff":
                mask = c[:, ch] >= thr_high
            else:
                # Ambiguous; keep the first event (stable & deterministic).
                out[int(idx.min()), ch] = True
                continue

            best_i = int(idx[0])
            best_len = -1
            best_peak = -1.0
            for i in idx.tolist():
                i = int(i)
                run = 0
                # Count preceding run length (cyclic).
                for k in range(1, T + 1):
                    j = (i - k) % T
                    if bool(mask[j]):
                        run += 1
                    else:
                        break
                peak = float(c[i, ch])
                if (run > best_len) or (run == best_len and peak > best_peak):
                    best_len = run
                    best_peak = peak
                    best_i = i
            out[best_i, ch] = True
            continue

        # Unknown selection mode: default to keeping all.
        out[:, ch] = ev[:, ch]
    return out


def ttc_to_next_event_np(
    contacts: np.ndarray,
    *,
    thr: float = 0.5,
    kind: EventKind = "touchdown",
    hyst: float = 0.0,
    ttc_max: Optional[int] = None,
    cyclic: bool = False,
    select: EventSelect = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frame TTC (time-to-next-event) from a soft contact sequence.

    Args:
        contacts: (T, C) float in [0,1]
        thr/kind/hyst: event definition (see detect_contact_events_np)
        ttc_max: optional censoring cap (frames). When set, ttc is clipped to [0, ttc_max].
        cyclic: if True, treat the sequence as one cycle and wrap-around when searching for the next event.
        select: event selection mode. In cyclic mode, this can be used to keep a single stable event per channel.

    Returns:
        ttc: (T, C) float32: frames until next event (0 at event frame).
        valid: (T, C) bool: True if a next event exists within the provided sequence.
        events: (T, C) bool: detected event frames.
    """
    c = np.asarray(contacts, dtype=np.float32)
    if c.ndim != 2:
        raise ValueError(f"contacts must be 2D (T,C), got shape={c.shape}")
    T, C = int(c.shape[0]), int(c.shape[1])
    if T <= 0 or C <= 0:
        z_ttc = np.zeros((max(0, T), max(0, C)), dtype=np.float32)
        z_valid = np.zeros_like(z_ttc, dtype=bool)
        z_ev = np.zeros_like(z_ttc, dtype=bool)
        return z_ttc, z_valid, z_ev

    if bool(cyclic):
        events = _detect_contact_events_cyclic_np(c, thr=float(thr), kind=kind, hyst=float(hyst))
        events = _select_single_event_per_channel_cyclic(
            c, events, thr=float(thr), kind=kind, hyst=float(hyst), select=select
        )
        next_idx = np.full((T, C), -1, dtype=np.int64)
        t_idx = np.arange(T, dtype=np.int64)
        for ch in range(C):
            idx = np.where(events[:, ch])[0].astype(np.int64)
            if idx.size == 0:
                continue
            first = int(idx.min())
            nxt = -1
            for t in range(T - 1, -1, -1):
                if bool(events[t, ch]):
                    nxt = int(t)
                next_idx[t, ch] = int(nxt)
            # Wrap-around for positions after the last event.
            for t in range(T):
                if next_idx[t, ch] < 0:
                    next_idx[t, ch] = first + T
        valid = next_idx >= 0
        ttc = (next_idx - t_idx[:, None]).astype(np.float32)
        ttc[~valid] = 0.0
    else:
        events = detect_contact_events_np(c, thr=float(thr), kind=kind, hyst=float(hyst))
        next_idx = np.full((T, C), -1, dtype=np.int64)
        t_idx = np.arange(T, dtype=np.int64)
        for ch in range(C):
            nxt = -1
            for t in range(T - 1, -1, -1):
                if bool(events[t, ch]):
                    nxt = int(t)
                next_idx[t, ch] = int(nxt)

        valid = next_idx >= 0
        ttc = (next_idx - t_idx[:, None]).astype(np.float32)
        ttc[~valid] = 0.0

    if ttc_max is not None:
        ttc_max_i = int(ttc_max)
        if ttc_max_i >= 0:
            np.clip(ttc, 0.0, float(ttc_max_i), out=ttc)
    return ttc, valid, events
