from __future__ import annotations

"""Action-handoff in-betweening — soft-endpoint re-anchor caliper (zero-training).

Formalizes the `action-handoff in-betweening` reframe (design note
`2026-05-31_action_handoff_inbetween_soft_endpoint_reframe.md`) into a small set of PURE
functions used by the zero-training falsifiability probe
(`tools/run_action_handoff_inbetween_soft_endpoint_probe.py`). It owns ONLY the scoring /
decision half of the probe; it does not train, unfreeze, or inject anything.

The reframe in one line: the switch signal selects a turn *regime region* (not a landing
frame), the endpoint is *soft / emergent* (the downstream clip re-anchors to wherever the
bridge lands, at an arbitrary in-regime phase), and motion consistency (realized-yaw / pop /
contact) is the bridge's VERIFICATION, not the endpoint's definition.

Operationally this module changes exactly ONE knob relative to the W1d precise caliper: the
**resume-frame candidate set** that the downstream re-anchor is allowed to pick from.

  - PRECISE caliper  : candidate set = the fixed K-frame seam window `target[g0 : g0+K]`
                       (the "match a fixed seam frame" semantics the reframe corrects).
  - SOFT caliper     : candidate set = the turn-*regime* span (frames at/after g0 whose
                       |yaw_rate| is elevated), i.e. the latent region expressed in 281-d
                       state space; re-anchor = best-pose frame within that region.

CRITICAL red lines baked in here (so a "revival" cannot be faked):
  - **Soft ≠ relaxing a threshold.** The pop / pose / yaw thresholds are IDENTICAL between
    calipers; only the resume candidate set widens. (W1a relapse guard.)
  - **Region is the turn regime, not the post-turn walk return.** Turn clips return to the
    Walk_F loop pose at their tail (re-entry resolver), so a straight-walk bridge could
    pose-match that walk-like tail with low pop. The regime mask (elevated |yaw_rate|)
    excludes that tail so soft re-anchor cannot cheat by landing on the walk return.
  - **Latent / region membership is READ from the generated motion (downstream), never
    injected.** In 281-d state space there is no latent to inject; region-entry is a read
    on the bridge's own states.
  - **Realized-yaw is caliper-invariant.** Re-anchor only moves the pose-match + pop; the
    heading-ramp verification (yaw_corr / heading_mae) is unchanged and must still hold, so
    a clip whose bridge ramps heading the wrong way (yaw_corr<0) can never be revived.

All thresholds are PROVISIONAL (inherited from the W1d gate); this module sets none of its
own beyond the regime-mask fraction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from train.action_handoff_inbetween_model import (
    GateThresholds,
    SEAM_C1_SLICES,
    evaluate_rollout_state_space,
)
from train.data.action_handoff_inbetween import (
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)

PRECISE = "precise"
SOFT = "soft"
CALIPERS = (PRECISE, SOFT)

# [PROVISIONAL] regime mask: a turn-regime frame has |yaw_rate| >= frac * max|yaw_rate|
# over the candidate tail. Walk_F yaw_rate ~ 0 and turn onsets ramp to ~1.2 rad/s
# (spec §1.1), so a moderate fraction isolates the genuinely-turning span.
DEFAULT_REGIME_FRAC = 0.25
DEFAULT_SOFT_MIN_SPAN = 4  # if the regime mask is too thin, widen to this many frames

EPS = 1e-8


def _validate_state(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != STATE_DIM:
        raise ValueError(f"{name}: expected state [T,{STATE_DIM}], got {tuple(np.shape(arr))}")
    if a.shape[0] < 1:
        raise ValueError(f"{name}: empty state (T=0)")
    return a


def seam_start(target_len: int, goal_horizon: int, seam_len: int) -> int:
    """The fixed seam offset g0 used by the W1d precise caliper (clamped to fit K)."""
    return int(min(int(goal_horizon), int(target_len) - int(seam_len)))


def turn_regime_indices(
    target: np.ndarray,
    *,
    start: int,
    regime_frac: float = DEFAULT_REGIME_FRAC,
    min_span: int = DEFAULT_SOFT_MIN_SPAN,
) -> np.ndarray:
    """Contiguous span [start, last_in_regime] — the soft re-anchor region.

    The soft region = the latent turn-regime expressed in 281-d state space. A frame is
    in-regime if |yaw_rate| >= regime_frac * max(|yaw_rate| over the tail). We return the
    CONTIGUOUS span from `start` (inclusive, = the precise seam g0) up to the LAST in-regime
    frame, deliberately:

      - starting at `start` guarantees the precise K-window is a SUBSET of the soft region
        (so soft gives the re-anchor strictly MORE freedom, never less — the reframe's best
        fair chance; soft can only improve pose-match, never worsen it);
      - stopping at the last in-regime frame EXCLUDES the post-turn walk-return tail (turn
        clips return to the Walk_F loop pose at their end — re-entry resolver), so a
        straight-walk bridge cannot cheat by pose-matching that walk-like tail.

    Falls back to the first `min_span` frames if the tail carries no turn signal — this only
    SHRINKS the soft region toward precise, never grants a free pass.
    """
    t = _validate_state(target, "turn_regime target").shape[0]
    start = int(max(0, min(start, t - 1)))
    tail = np.asarray(target, dtype=np.float64)[start:, YAW_RATE_SLICE].reshape(-1)
    if tail.size == 0:
        return np.asarray([start], dtype=np.int64)
    peak = float(np.max(np.abs(tail)))
    if peak <= EPS:
        # No turn signal in the tail at all → degenerate; use the minimal span.
        end = min(start + int(min_span), t)
        return np.arange(start, max(end, start + 1), dtype=np.int64)
    thr = float(regime_frac) * peak
    in_regime = np.nonzero(np.abs(tail) >= thr)[0]
    last = int(start + int(in_regime[-1])) if in_regime.size else start
    end = max(int(last) + 1, start + int(min_span))
    end = min(end, t)
    return np.arange(start, max(end, start + 1), dtype=np.int64)


@dataclass
class ResumeRegion:
    """The resume-frame candidate set a downstream re-anchor may pick from."""

    caliper: str
    indices: np.ndarray  # int64 frame indices into the target clip
    frames: np.ndarray  # [M, 281] candidate states
    n_candidates: int
    seam_start: int


def resume_region(
    target: np.ndarray,
    caliper: str,
    *,
    goal_horizon: int,
    seam_len: int,
    regime_frac: float = DEFAULT_REGIME_FRAC,
    min_span: int = DEFAULT_SOFT_MIN_SPAN,
) -> ResumeRegion:
    """Build the resume candidate set for the precise or soft caliper.

    PRECISE: the fixed K-frame seam window `target[g0:g0+K]` (W1d semantics).
    SOFT   : the turn-regime span at/after g0 (re-anchor to an arbitrary in-regime phase).
    """
    t = _validate_state(target, "resume_region target").shape[0]
    g0 = seam_start(t, goal_horizon, seam_len)
    if caliper == PRECISE:
        idx = np.arange(g0, min(g0 + int(seam_len), t), dtype=np.int64)
    elif caliper == SOFT:
        idx = turn_regime_indices(target, start=g0, regime_frac=regime_frac, min_span=min_span)
    else:
        raise ValueError(f"unknown caliper: {caliper!r} (expected one of {CALIPERS})")
    if idx.size == 0:
        idx = np.asarray([g0], dtype=np.int64)
    return ResumeRegion(
        caliper=str(caliper),
        indices=idx,
        frames=np.asarray(target, dtype=np.float64)[idx].astype(np.float32),
        n_candidates=int(idx.size),
        seam_start=int(g0),
    )


def region_entry_min_dist(roll_raw: np.ndarray, region_raw: np.ndarray, std: np.ndarray) -> float:
    """Min group-std distance from any rollout frame to the region centroid.

    This is the "soft endpoint reached?" read — does the bridge's own motion ENTER the turn
    regime region (downstream read; nothing injected). Lower = deeper into the region.
    """
    roll = _validate_state(roll_raw, "region_entry roll")
    region = _validate_state(region_raw, "region_entry region")
    centroid = region.mean(axis=0)
    s = np.asarray(std, dtype=np.float64)
    d = np.abs((roll - centroid[None, :]) / s[None, :]).mean(axis=1)
    return float(np.min(d))


@dataclass
class CaliperScore:
    """One rollout scored under one caliper (precise or soft re-anchor)."""

    caliper: str
    best_pose_d: float
    pop: float
    pop_safe: bool
    clip_resumable: bool
    region_entry_dist: float
    resume_target_frame: int
    resume_rollout_frame: int
    n_candidates: int


def score_rollout(
    roll_raw: np.ndarray,
    target: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    caliper: str,
    *,
    goal_horizon: int,
    seam_len: int,
    regime_frac: float = DEFAULT_REGIME_FRAC,
    min_span: int = DEFAULT_SOFT_MIN_SPAN,
) -> CaliperScore:
    """Score one bridge rollout against a turn clip under the given caliper.

    Re-uses the W1d `evaluate_rollout_state_space` (identical pose-match + pop + thresholds);
    the ONLY thing the caliper changes is the resume candidate set. Soft therefore cannot
    pass anything the thresholds reject — it can only let the re-anchor pick a better-aligned
    in-regime phase.
    """
    region = resume_region(
        target,
        caliper,
        goal_horizon=goal_horizon,
        seam_len=seam_len,
        regime_frac=regime_frac,
        min_span=min_span,
    )
    state = evaluate_rollout_state_space(
        np.asarray(roll_raw, dtype=np.float64),
        np.asarray(region.frames, dtype=np.float64),
        np.asarray(std, dtype=np.float64),
        thr,
    )
    entry = region_entry_min_dist(roll_raw, region.frames, std)
    return CaliperScore(
        caliper=str(caliper),
        best_pose_d=float(state["best_pose_d"]),
        pop=float(state["pop"]),
        pop_safe=bool(state["pop_safe"]),
        clip_resumable=bool(state["clip_resumable"]),
        region_entry_dist=float(entry),
        resume_target_frame=int(region.indices[int(state["resume_target_frame"])]),
        resume_rollout_frame=int(state["resume_rollout_frame"]),
        n_candidates=int(region.n_candidates),
    )


def splice_pop(
    cut_state: np.ndarray,
    target: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    caliper: str,
    *,
    goal_horizon: int,
    seam_len: int,
    regime_frac: float = DEFAULT_REGIME_FRAC,
    min_span: int = DEFAULT_SOFT_MIN_SPAN,
) -> CaliperScore:
    """Angle-A helper: pop of a direct MM-cut from a single cut state to a resume frame.

    `cut_state` is one 281-d frame (e.g. an arbitrary-phase Walk_F frame). Treated as a
    1-frame rollout so the same scorer applies: precise splices to the fixed seam frame,
    soft re-anchors to the best in-regime phase. Shows (i) soft re-anchor reduces pop vs a
    fixed-frame splice, and (ii) the residual pop is still non-zero (no frame perfectly
    aligns at arbitrary phase) → a generated bridge is still required.
    """
    cut = _validate_state(cut_state, "splice cut_state")
    return score_rollout(
        cut,
        target,
        std,
        thr,
        caliper,
        goal_horizon=goal_horizon,
        seam_len=seam_len,
        regime_frac=regime_frac,
        min_span=min_span,
    )


# =============================================================== pre-committed decision rule
@dataclass
class SoftEndpointVerdict:
    """The pre-committed un-park / keep-park decision (decided before seeing the numbers)."""

    held_out_clips: List[str]
    precise_pass: Dict[str, bool]
    soft_pass: Dict[str, bool]
    motion_consistency_ok: Dict[str, bool]
    revived_clips: List[str]
    positive_control_pass: bool
    negative_control_holds: bool
    gate_valid: bool
    decision: str  # "UNPARK" | "KEEP_PARK" | "GATE_INVALID"
    reason: str


def soft_endpoint_decision(
    *,
    held_out_clips: Sequence[str],
    precise_pass: Mapping[str, bool],
    soft_pass: Mapping[str, bool],
    motion_consistency_ok: Mapping[str, bool],
    positive_control_pass: bool,
    negative_control_holds: bool,
) -> SoftEndpointVerdict:
    """Apply the pre-committed rule.

    A held-out clip is REVIVED iff it fails the precise caliper, passes the soft caliper, AND
    its motion consistency (realized-yaw correct + pop genuinely safe) still holds under soft.

    Gate validity guard (honesty): the soft caliper is only trusted if it still discriminates
    — the positive control (recorded turn) passes AND the negative control (non-turn motion
    scored against a turn region) still FAILS. If the gate is invalid, no conclusion is drawn
    (fix the gate first) and we hold PARK.

    Decision:
      - gate invalid               → GATE_INVALID (hold PARK; the gate, not the reframe, is at fault)
      - gate valid & any revived   → UNPARK (reframe materially moves the wall)
      - gate valid & none revived  → KEEP_PARK (formalize soft endpoint into spec; data ceiling stands)
    """
    held = [str(c) for c in held_out_clips]
    revived = [
        c
        for c in held
        if (not bool(precise_pass.get(c, False)))
        and bool(soft_pass.get(c, False))
        and bool(motion_consistency_ok.get(c, False))
    ]
    gate_valid = bool(positive_control_pass and negative_control_holds)
    if not gate_valid:
        decision = "GATE_INVALID"
        reason = (
            "soft caliper failed its discrimination guard "
            f"(positive_control_pass={bool(positive_control_pass)}, "
            f"negative_control_holds={bool(negative_control_holds)}); "
            "fix the gate before concluding → hold PARK"
        )
    elif revived:
        decision = "UNPARK"
        reason = (
            f"soft re-anchor revived held-out {revived} with motion consistency intact "
            "→ reframe materially changes the W1d wall"
        )
    else:
        decision = "KEEP_PARK"
        reason = (
            "no held-out clip revived under the soft caliper with motion consistency intact "
            "→ reframe is formalized into spec but does not break the data ceiling"
        )
    return SoftEndpointVerdict(
        held_out_clips=held,
        precise_pass={c: bool(precise_pass.get(c, False)) for c in held},
        soft_pass={c: bool(soft_pass.get(c, False)) for c in held},
        motion_consistency_ok={c: bool(motion_consistency_ok.get(c, False)) for c in held},
        revived_clips=revived,
        positive_control_pass=bool(positive_control_pass),
        negative_control_holds=bool(negative_control_holds),
        gate_valid=gate_valid,
        decision=decision,
        reason=reason,
    )
