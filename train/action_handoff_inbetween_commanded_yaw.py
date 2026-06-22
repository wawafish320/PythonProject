from __future__ import annotations

"""Zero-training helpers for the F4 commanded-yaw formalization probe.

Scope is intentionally narrow:
  - replace only the yaw-rate channel of a generated 281-d rollout;
  - classify the held-out outcome into pre-committed F4 verdict buckets.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from train.data.action_handoff_inbetween import STATE_DIM, YAW_RATE_SLICE


def replace_yaw_rate_slice(rollout: np.ndarray, commanded_yaw_rate: np.ndarray) -> np.ndarray:
    """Return a copy of `rollout` with ONLY `YAW_RATE_SLICE` replaced.

    Args:
        rollout: [H, 281] state rollout.
        commanded_yaw_rate: [H] or [H,1] yaw-rate series (rad/s).
    """
    roll = np.asarray(rollout)
    if roll.ndim != 2 or int(roll.shape[1]) != int(STATE_DIM):
        raise ValueError(
            f"rollout must be [H,{STATE_DIM}], got shape={tuple(np.shape(rollout))}"
        )

    cmd = np.asarray(commanded_yaw_rate)
    if cmd.ndim == 2 and cmd.shape[1] == 1:
        cmd = cmd.reshape(-1)
    if cmd.ndim != 1:
        raise ValueError(
            "commanded_yaw_rate must be rank-1 [H] or rank-2 [H,1], "
            f"got shape={tuple(np.shape(commanded_yaw_rate))}"
        )
    if int(cmd.shape[0]) != int(roll.shape[0]):
        raise ValueError(
            f"length mismatch: rollout H={int(roll.shape[0])}, "
            f"commanded H={int(cmd.shape[0])}"
        )

    out = np.array(roll, copy=True)
    out[:, YAW_RATE_SLICE] = cmd.reshape(-1, 1).astype(out.dtype, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("non-finite values after yaw-rate replacement")
    return out


@dataclass
class F4CommandedYawDecision:
    decision: str  # F4_CONTROL_CONFIRMED | F4_MIXED_WITH_POP | F4_NOT_EXPLAINED | GATE_INVALID
    reason: str
    gate_valid: bool
    clips_with_negative_baseline_yaw: List[str]
    clips_yaw_fixed_by_commanded: List[str]
    clips_pop_improved: List[str]
    clips_pop_still_failed: List[str]
    mean_pop_safe_delta: float


def classify_f4_commanded_yaw(
    *,
    held_out_clips: Sequence[str],
    baseline_rows: Mapping[str, Mapping[str, float]],
    commanded_rows: Mapping[str, Mapping[str, float]],
    tau_yaw_rad: float,
    gate_valid: bool,
    pop_improve_eps: float = 0.10,
) -> F4CommandedYawDecision:
    """Classify the F4 probe outcome using pre-committed branch semantics."""
    held = [str(c) for c in held_out_clips]
    neg = [
        c
        for c in held
        if np.isfinite(float(baseline_rows[c].get("yaw_corr", float("nan"))))
        and float(baseline_rows[c].get("yaw_corr", float("nan"))) < 0.0
    ]

    fixed = [
        c
        for c in neg
        if (
            np.isfinite(float(commanded_rows[c].get("yaw_corr", float("nan"))))
            and float(commanded_rows[c].get("yaw_corr", float("nan"))) > 0.0
            and np.isfinite(float(commanded_rows[c].get("heading_mae_rad", float("nan"))))
            and float(commanded_rows[c].get("heading_mae_rad", float("nan"))) < float(tau_yaw_rad)
        )
    ]

    pop_deltas: Dict[str, float] = {}
    pop_improved: List[str] = []
    pop_failed: List[str] = []
    for c in held:
        b = float(baseline_rows[c].get("pop_safe_rate", float("nan")))
        m = float(commanded_rows[c].get("pop_safe_rate", float("nan")))
        d = float("nan") if (not np.isfinite(b) or not np.isfinite(m)) else (m - b)
        pop_deltas[c] = d
        if np.isfinite(d) and d > float(pop_improve_eps):
            pop_improved.append(c)
        if (not np.isfinite(m)) or (m <= 0.0):
            pop_failed.append(c)

    finite_deltas = [v for v in pop_deltas.values() if np.isfinite(v)]
    mean_delta = float(np.mean(finite_deltas)) if finite_deltas else float("nan")

    if not bool(gate_valid):
        return F4CommandedYawDecision(
            decision="GATE_INVALID",
            reason="target yaw positive control failed; fix alignment/mapping/metric before concluding",
            gate_valid=False,
            clips_with_negative_baseline_yaw=neg,
            clips_yaw_fixed_by_commanded=fixed,
            clips_pop_improved=pop_improved,
            clips_pop_still_failed=pop_failed,
            mean_pop_safe_delta=mean_delta,
        )

    if not neg:
        return F4CommandedYawDecision(
            decision="F4_NOT_EXPLAINED",
            reason="no held-out clip had negative baseline yaw_corr; cannot test F4 reversal condition",
            gate_valid=True,
            clips_with_negative_baseline_yaw=neg,
            clips_yaw_fixed_by_commanded=fixed,
            clips_pop_improved=pop_improved,
            clips_pop_still_failed=pop_failed,
            mean_pop_safe_delta=mean_delta,
        )

    yaw_fixed_all = len(fixed) == len(neg)
    if not yaw_fixed_all:
        return F4CommandedYawDecision(
            decision="F4_NOT_EXPLAINED",
            reason="commanded yaw did not flip all negative-yaw held-out clips to positive corr under tau_yaw",
            gate_valid=True,
            clips_with_negative_baseline_yaw=neg,
            clips_yaw_fixed_by_commanded=fixed,
            clips_pop_improved=pop_improved,
            clips_pop_still_failed=pop_failed,
            mean_pop_safe_delta=mean_delta,
        )

    if pop_failed:
        return F4CommandedYawDecision(
            decision="F4_CONTROL_CONFIRMED",
            reason="yaw reversal fixed by commanded yaw, while pop_safe still fails on held-out clips",
            gate_valid=True,
            clips_with_negative_baseline_yaw=neg,
            clips_yaw_fixed_by_commanded=fixed,
            clips_pop_improved=pop_improved,
            clips_pop_still_failed=pop_failed,
            mean_pop_safe_delta=mean_delta,
        )

    return F4CommandedYawDecision(
        decision="F4_MIXED_WITH_POP",
        reason="yaw reversal fixed and pop_safe no longer fails on held-out clips",
        gate_valid=True,
        clips_with_negative_baseline_yaw=neg,
        clips_yaw_fixed_by_commanded=fixed,
        clips_pop_improved=pop_improved,
        clips_pop_still_failed=pop_failed,
        mean_pop_safe_delta=mean_delta,
    )
