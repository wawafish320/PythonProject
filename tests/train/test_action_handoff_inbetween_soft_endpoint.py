"""Unit tests for the soft-endpoint re-anchor caliper pure functions.

Covers the zero-training probe's scoring/decision half:
`train/action_handoff_inbetween_soft_endpoint.py`. The probe tool itself is exercised
end-to-end by its artifact run; these tests pin the pure invariants the reframe relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from train.action_handoff_inbetween_model import GateThresholds
from train.action_handoff_inbetween_soft_endpoint import (
    DEFAULT_SOFT_MIN_SPAN,
    PRECISE,
    SOFT,
    region_entry_min_dist,
    resume_region,
    score_rollout,
    seam_start,
    soft_endpoint_decision,
    splice_pop,
    turn_regime_indices,
)
from train.data.action_handoff_inbetween import (
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)

GOAL_HORIZON = 12
SEAM_LEN = 6


def _synthetic_target(
    *,
    t: int = 40,
    regime_lo: int = 12,
    regime_hi: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """Target with a clear turn regime [regime_lo, regime_hi] and a low-yaw walk-return tail."""
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.1, size=(t, STATE_DIM)).astype(np.float64)
    # Distinct, monotone pose per frame so pose-NN is deterministic.
    state[:, POSE_SLICE] = np.arange(t, dtype=np.float64)[:, None] * 0.01
    yaw = np.zeros(t, dtype=np.float64)
    yaw[regime_lo : regime_hi + 1] = 1.0  # only the turn regime ramps yaw
    state[:, YAW_RATE_SLICE] = yaw[:, None]
    return state


# --------------------------------------------------------------------------- seam_start
def test_seam_start_clamps_to_fit_seam():
    assert seam_start(40, GOAL_HORIZON, SEAM_LEN) == 12
    # short clip: g0 clamps so g0+K fits.
    assert seam_start(15, GOAL_HORIZON, SEAM_LEN) == 9


# ------------------------------------------------------------------ turn_regime_indices
def test_turn_regime_is_contiguous_starts_at_g0_excludes_walk_return():
    target = _synthetic_target(t=40, regime_lo=12, regime_hi=25)
    idx = turn_regime_indices(target, start=12)
    assert idx[0] == 12  # starts at the precise seam g0
    assert idx[-1] == 25  # stops at the last in-regime frame
    assert np.array_equal(idx, np.arange(12, 26))  # contiguous
    assert 39 not in idx  # the low-yaw walk-return tail is excluded


def test_turn_regime_min_span_fallback_when_no_turn_signal():
    target = _synthetic_target(t=40)
    target[:, YAW_RATE_SLICE] = 0.0  # no turn signal anywhere
    idx = turn_regime_indices(target, start=12, min_span=DEFAULT_SOFT_MIN_SPAN)
    assert idx[0] == 12
    assert idx.size == DEFAULT_SOFT_MIN_SPAN  # only the minimal span, never empty


# ----------------------------------------------------------------------- resume_region
def test_precise_region_is_fixed_seam_window():
    target = _synthetic_target()
    region = resume_region(target, PRECISE, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    assert np.array_equal(region.indices, np.arange(12, 18))
    assert region.frames.shape == (6, STATE_DIM)
    assert region.seam_start == 12


def test_precise_is_subset_of_soft_region():
    target = _synthetic_target()
    precise = resume_region(target, PRECISE, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    soft = resume_region(target, SOFT, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    assert set(precise.indices.tolist()).issubset(set(soft.indices.tolist()))
    assert soft.n_candidates > precise.n_candidates


def test_unknown_caliper_raises():
    target = _synthetic_target()
    with pytest.raises(ValueError):
        resume_region(target, "bogus", goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)


# ------------------------------------------------------------------ region_entry_min_dist
def test_region_entry_zero_when_rollout_sits_on_region_centroid():
    target = _synthetic_target()
    region = resume_region(target, SOFT, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    std = np.ones(STATE_DIM, dtype=np.float64)
    centroid = region.frames.mean(axis=0)
    roll = np.stack([centroid, centroid + 5.0], axis=0)
    assert region_entry_min_dist(roll, region.frames, std) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------- score_rollout superset property
def test_soft_pose_never_worse_than_precise():
    target = _synthetic_target(t=40, regime_lo=12, regime_hi=24)
    std = np.ones(STATE_DIM, dtype=np.float64)
    thr = GateThresholds()
    # Roll whose pose matches a SOFT-only frame (frame 20, in [12,24] but not [12,18)).
    roll = target[20:21].copy()
    precise = score_rollout(roll, target, std, thr, PRECISE, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    soft = score_rollout(roll, target, std, thr, SOFT, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    assert soft.best_pose_d <= precise.best_pose_d + 1e-9
    assert soft.best_pose_d == pytest.approx(0.0, abs=1e-5)  # exact re-anchor onto frame 20
    assert soft.resume_target_frame == 20
    assert precise.best_pose_d > 1e-6  # frame 20 unavailable to the precise window


def test_splice_pop_returns_caliper_score_single_frame():
    target = _synthetic_target()
    std = np.ones(STATE_DIM, dtype=np.float64)
    thr = GateThresholds()
    cut = target[5]  # a single arbitrary-phase cut frame
    sc = splice_pop(cut, target, std, thr, SOFT, goal_horizon=GOAL_HORIZON, seam_len=SEAM_LEN)
    assert np.isfinite(sc.pop)
    assert sc.n_candidates >= 1


# -------------------------------------------------------------- soft_endpoint_decision
_HELD = ["Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R"]


def test_decision_unpark_on_genuine_revival():
    v = soft_endpoint_decision(
        held_out_clips=_HELD,
        precise_pass={"Walk_L_To_L": False, "Walk_R_To_L": False, "Walk_R_To_R": False},
        soft_pass={"Walk_L_To_L": True, "Walk_R_To_L": False, "Walk_R_To_R": False},
        motion_consistency_ok={"Walk_L_To_L": True, "Walk_R_To_L": False, "Walk_R_To_R": False},
        positive_control_pass=True,
        negative_control_holds=True,
    )
    assert v.decision == "UNPARK"
    assert v.revived_clips == ["Walk_L_To_L"]


def test_decision_keep_park_when_no_revival():
    v = soft_endpoint_decision(
        held_out_clips=_HELD,
        precise_pass={c: False for c in _HELD},
        soft_pass={c: False for c in _HELD},
        motion_consistency_ok={c: False for c in _HELD},
        positive_control_pass=True,
        negative_control_holds=True,
    )
    assert v.decision == "KEEP_PARK"
    assert v.revived_clips == []


def test_soft_pass_without_motion_consistency_is_not_a_revival():
    # Soft pass but the heading ramp is wrong / pop unsafe → NOT revived (red line).
    v = soft_endpoint_decision(
        held_out_clips=_HELD,
        precise_pass={c: False for c in _HELD},
        soft_pass={"Walk_R_To_R": True, "Walk_L_To_L": False, "Walk_R_To_L": False},
        motion_consistency_ok={c: False for c in _HELD},
        positive_control_pass=True,
        negative_control_holds=True,
    )
    assert v.decision == "KEEP_PARK"
    assert v.revived_clips == []


def test_decision_gate_invalid_when_control_fails():
    # Even with an apparent revival, a broken gate blocks the conclusion → hold PARK.
    v = soft_endpoint_decision(
        held_out_clips=_HELD,
        precise_pass={c: False for c in _HELD},
        soft_pass={c: True for c in _HELD},
        motion_consistency_ok={c: True for c in _HELD},
        positive_control_pass=True,
        negative_control_holds=False,  # soft passed a non-turn negative → always-yes gate
    )
    assert v.decision == "GATE_INVALID"
    assert v.gate_valid is False
