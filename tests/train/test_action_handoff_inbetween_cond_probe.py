"""Unit tests — §7.3 3b Slice 2 cond-driven baseline probe (pure parts).

Model-free: covers the phase-wrap seeding, the cond override (the finding that a constant
override is a no-op under per-window normalization while a trajectory survives), the
rollout→egocentric conversion shape, and the per-clip aggregation schema (five headline
metrics + Walk_L_To_R as its own row). The ckpt-dependent rollout / hidden_pre capture is
exercised by the end-to-end probe run, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from train.action_handoff_inbetween_cond_probe import (
    COND_DIM,
    WALK_L_TO_R,
    aggregate_clip_record,
    build_cond_override,
    phase_seed_indices,
    rollout_to_egocentric,
    select_start_phases,
    summarize_reach,
    turn_clip_order,
)
from train.data.action_handoff_inbetween import STATE_DIM, TURN_CLIPS


# --------------------------------------------------------------- phase wrap
def test_phase_seed_indices_wraps_periodic_clip():
    idx = phase_seed_indices(phase=84, horizon=8, clip_len=87)
    assert idx.tolist() == [84, 85, 86, 0, 1, 2, 3, 4]
    assert idx.dtype == np.int64


def test_phase_seed_indices_horizon_exceeds_clip_len():
    idx = phase_seed_indices(phase=0, horizon=200, clip_len=87)
    assert idx.shape == (200,)
    assert idx.min() == 0 and idx.max() == 86


def test_phase_seed_indices_validates():
    with pytest.raises(ValueError):
        phase_seed_indices(0, 0, 87)
    with pytest.raises(ValueError):
        phase_seed_indices(0, 8, 0)


def test_select_start_phases_count_and_range():
    phases = select_start_phases(87, 24)
    assert len(phases) == 24
    assert all(0 <= p < 87 for p in phases)


# --------------------------------------------------------------- cond override
def _ramp_cond(t: int) -> np.ndarray:
    """A turn-like cond: constant act_oh, ramping cond_dir heading, ~constant speed."""
    head = np.linspace(0.0, 1.2, t)  # rad, ramping (turn)
    cond = np.zeros((t, COND_DIM), dtype=np.float32)
    cond[:, 1] = 1.0  # act_oh = [0,1,0,0]
    cond[:, 4] = np.cos(head)
    cond[:, 5] = np.sin(head)
    cond[:, 6] = 0.9
    return cond


def test_build_cond_override_trajectory_survives_normalization():
    """Finding #2: a ramping cond_dir trajectory keeps signal after per-window normalization."""
    cond = _ramp_cond(50)
    ov = build_cond_override(cond, horizon=50)
    assert ov.norm.shape == (50, COND_DIM)
    assert ov.turn_len == 50
    # cond_dir channels carry real variance after normalization (not collapsed)
    assert np.std(ov.norm[:, 4]) > 0.1 or np.std(ov.norm[:, 5]) > 0.1


def test_build_cond_override_constant_is_noop_under_normalization():
    """A CONSTANT cond_dir override collapses to ~0 — the documented no-op."""
    t = 50
    cond = np.zeros((t, COND_DIM), dtype=np.float32)
    cond[:, 1] = 1.0
    cond[:, 4] = np.cos(0.7)  # constant heading
    cond[:, 5] = np.sin(0.7)
    cond[:, 6] = 0.9
    ov = build_cond_override(cond, horizon=t)
    # constant channels → robust std floored → normalized ~0 (indistinguishable from Walk_F)
    assert np.allclose(ov.norm[:, 4:6], 0.0, atol=1e-3)


def test_build_cond_override_extends_by_holding_last_frame():
    cond = _ramp_cond(40)
    ov = build_cond_override(cond, horizon=120)
    assert ov.raw.shape == (120, COND_DIM)
    # tail holds the last turn frame
    assert np.allclose(ov.raw[40:], cond[-1][None, :])
    # stats computed over the turn trajectory only (length 40), not the extended window
    assert ov.mu.shape == (COND_DIM,) and ov.std.shape == (COND_DIM,)


def test_build_cond_override_clips_and_validates():
    ov = build_cond_override(_ramp_cond(30), horizon=30, clip=6.0)
    assert ov.norm.max() <= 6.0 + 1e-6 and ov.norm.min() >= -6.0 - 1e-6
    with pytest.raises(ValueError):
        build_cond_override(np.zeros((10, 3), dtype=np.float32), horizon=10)  # wrong cond dim


# --------------------------------------------------------------- rollout → egocentric
def test_rollout_to_egocentric_shape():
    s = 30
    rot6d = np.random.RandomState(0).randn(s, 46, 6).astype(np.float32)
    root_vel = np.random.RandomState(1).randn(s, 2).astype(np.float32)
    cond_dir = _ramp_cond(s)[:, 4:6]
    contact = np.random.RandomState(2).rand(s, 2).astype(np.float32)
    state = rollout_to_egocentric(rot6d, root_vel, cond_dir, contact)
    assert state.shape == (s, STATE_DIM)
    assert state.dtype == np.float32


def test_rollout_to_egocentric_truncates_to_common_length():
    rot6d = np.zeros((30, 46, 6), dtype=np.float32)
    root_vel = np.zeros((28, 2), dtype=np.float32)
    cond_dir = np.tile([1.0, 0.0], (25, 1)).astype(np.float32)
    contact = np.zeros((40, 2), dtype=np.float32)
    state = rollout_to_egocentric(rot6d, root_vel, cond_dir, contact)
    assert state.shape == (25, STATE_DIM)


# --------------------------------------------------------------- aggregation
def test_summarize_reach_floor_rate():
    out = summarize_reach([0.5, 1.0, 2.0, 3.0], conv_norm_thr=1.5)
    assert out["n"] == 4
    assert out["reach_floor_rate"] == pytest.approx(0.5)  # 0.5,1.0 ≤ 1.5
    assert out["reach_min_norm_min"] == pytest.approx(0.5)


def test_summarize_reach_empty():
    out = summarize_reach([], conv_norm_thr=1.5)
    assert out["n"] == 0
    assert np.isnan(out["reach_floor_rate"])


def test_aggregate_clip_record_has_five_headline_metrics():
    outcomes = [
        {"clip_resumable": True, "pop_safe": False, "fallback": False, "best_pose_d": 0.1, "pop": 0.4},
        {"clip_resumable": False, "pop_safe": True, "fallback": True, "best_pose_d": 0.3, "pop": 0.2},
    ]
    rec = aggregate_clip_record([1.0, 2.0], outcomes, conv_norm_thr=1.5)
    for key in (
        "reach_min_norm_mean",
        "reach_floor_rate",
        "clip_resumable_rate",
        "pop_safe_rate",
        "fallback_rate",
    ):
        assert key in rec
    assert rec["clip_resumable_rate"] == pytest.approx(0.5)
    assert rec["fallback_rate"] == pytest.approx(0.5)
    assert rec["reach_floor_rate"] == pytest.approx(0.5)
    assert rec["n_starts"] == 2


# --------------------------------------------------------------- L_R own row
def test_turn_clip_order_includes_walk_l_to_r():
    order = turn_clip_order()
    assert WALK_L_TO_R in order
    assert set(order) == set(TURN_CLIPS)


def test_turn_clip_order_rejects_missing_l_to_r():
    with pytest.raises(ValueError):
        turn_clip_order(["Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R"])
