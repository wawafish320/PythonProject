from __future__ import annotations

"""Egocentric state-transform + schema semantics for the in-betweening pipeline.

Covers spec §1.1 (egocentric state, yaw_rate definition) and §1.2/§1.3 schema.
Pure synthetic data; no artifacts, no model.
"""

import numpy as np

from train.data.action_handoff_inbetween import (
    CONTACT_DIM,
    CONTACT_SLICE,
    EGO_VEL_DIM,
    EGO_VEL_SLICE,
    FPS,
    POSE_DIM,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_DIM,
    YAW_RATE_SLICE,
    build_egocentric_state,
    egocentric_root_vel,
    yaw_rate_from_cond_dir,
)


def _cond_dir(headings: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(headings), np.sin(headings)], axis=1)


def test_schema_dims_and_slices() -> None:
    assert STATE_DIM == 281
    assert POSE_DIM + EGO_VEL_DIM + YAW_RATE_DIM + CONTACT_DIM == STATE_DIM
    assert (POSE_DIM, EGO_VEL_DIM, YAW_RATE_DIM, CONTACT_DIM) == (276, 2, 1, 2)
    # slices are contiguous and cover [0, 281) exactly, in order.
    assert (POSE_SLICE.start, POSE_SLICE.stop) == (0, 276)
    assert (EGO_VEL_SLICE.start, EGO_VEL_SLICE.stop) == (276, 278)
    assert (YAW_RATE_SLICE.start, YAW_RATE_SLICE.stop) == (278, 279)
    assert (CONTACT_SLICE.start, CONTACT_SLICE.stop) == (279, 281)


def test_walk_f_like_clip_has_zero_lateral_and_zero_yaw_rate() -> None:
    # Straight forward walk: constant world heading, velocity purely along heading.
    t = 40
    heading = np.full(t, 0.7)  # arbitrary fixed world heading
    cond_dir = _cond_dir(heading)
    speed = 0.6 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, t))  # forward speed oscillates
    root_vel = np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=1)

    ego = egocentric_root_vel(root_vel, cond_dir)
    yaw = yaw_rate_from_cond_dir(cond_dir, fps=FPS)

    # forward component recovers the (signed) speed; lateral ≈ 0 (phase-flat).
    assert np.allclose(ego[:, 0], speed, atol=1e-5)
    assert np.max(np.abs(ego[:, 1])) < 1e-5
    # constant heading → no turning.
    assert np.max(np.abs(yaw)) < 1e-5


def test_turn_onset_has_nonzero_yaw_rate() -> None:
    t = 30
    # heading ramps (a turn): yaw_rate must be clearly non-zero.
    heading = np.linspace(0.0, 1.2, t)
    yaw = yaw_rate_from_cond_dir(_cond_dir(heading), fps=FPS)
    assert np.max(np.abs(yaw)) > 0.5  # rad/s, clearly turning


def test_yaw_rate_definition_units_wrap_frame0_dtype_shape() -> None:
    # (1) constant heading step δ → yaw_rate = δ * FPS (units rad/s, scaled by FPS).
    delta = 0.01
    t = 12
    heading = np.arange(t) * delta
    yaw = yaw_rate_from_cond_dir(_cond_dir(heading), fps=FPS)
    assert yaw.shape == (t, YAW_RATE_DIM) == (t, 1)
    assert yaw.dtype == np.float32
    assert np.allclose(yaw[1:, 0], delta * FPS, atol=1e-4)

    # (2) frame 0 := frame 1.
    assert yaw[0, 0] == yaw[1, 0]

    # (3) wrap to [-π, π): heading jump from +3.0 to -3.0 is a +0.283 step, NOT -6.0.
    heading_wrap = np.array([0.0, 3.0, -3.0])
    yaw_wrap = yaw_rate_from_cond_dir(_cond_dir(heading_wrap), fps=FPS)
    expected_step = ((-3.0 - 3.0) + np.pi) % (2 * np.pi) - np.pi  # ≈ +0.2832
    assert expected_step > 0  # wrapped to the short way around
    assert np.isclose(yaw_wrap[2, 0], expected_step * FPS, atol=1e-3)
    # unwrapped would be -6.0*FPS; confirm we are nowhere near that.
    assert yaw_wrap[2, 0] > 0


def test_build_egocentric_state_shape_and_dtype() -> None:
    t = 20
    bone_rot6d = np.random.default_rng(0).normal(size=(t, 46, 6)).astype(np.float32)
    heading = np.linspace(0.0, 0.5, t)
    cond_dir = _cond_dir(heading)
    root_vel = np.random.default_rng(1).normal(size=(t, 2)).astype(np.float32)
    contact = np.random.default_rng(2).random((t, 2)).astype(np.float32)

    state = build_egocentric_state(bone_rot6d, root_vel, cond_dir, contact)
    assert state.shape == (t, STATE_DIM)
    assert state.dtype == np.float32
    # pose channels equal the flattened rot6d.
    assert np.allclose(state[:, POSE_SLICE], bone_rot6d.reshape(t, -1), atol=1e-5)
    # contact channels passed through.
    assert np.allclose(state[:, CONTACT_SLICE], contact, atol=1e-5)
