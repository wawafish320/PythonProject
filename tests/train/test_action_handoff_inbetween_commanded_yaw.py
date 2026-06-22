from __future__ import annotations

import numpy as np
import pytest

from tools.run_action_handoff_inbetween_masked_smoke import _yaw_metrics
from train.action_handoff_inbetween_commanded_yaw import (
    classify_f4_commanded_yaw,
    replace_yaw_rate_slice,
)
from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    FPS,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
    yaw_rate_from_cond_dir,
)

HELD = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")


def _state(h: int, dtype: np.dtype = np.float32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.2, size=(h, STATE_DIM)).astype(dtype)


def test_replace_yaw_rate_only_touches_yaw_slice() -> None:
    h = 12
    roll = _state(h, dtype=np.float64, seed=1)
    cmd = np.linspace(-0.8, 0.6, h, dtype=np.float64)
    out = replace_yaw_rate_slice(roll, cmd)

    assert out.shape == (h, STATE_DIM)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    assert np.allclose(out[:, YAW_RATE_SLICE].reshape(-1), cmd, atol=1e-12)
    assert np.allclose(out[:, POSE_SLICE], roll[:, POSE_SLICE], atol=0.0)
    assert np.allclose(out[:, EGO_VEL_SLICE], roll[:, EGO_VEL_SLICE], atol=0.0)
    assert np.allclose(out[:, CONTACT_SLICE], roll[:, CONTACT_SLICE], atol=0.0)
    assert np.allclose(roll[:, YAW_RATE_SLICE].reshape(-1), _state(h, dtype=np.float64, seed=1)[:, YAW_RATE_SLICE].reshape(-1))


def test_replace_yaw_rate_preserves_shape_dtype_for_float32_and_rank2_input() -> None:
    h = 10
    roll = _state(h, dtype=np.float32, seed=2)
    cmd = np.linspace(0.1, 0.2, h, dtype=np.float32).reshape(h, 1)
    out = replace_yaw_rate_slice(roll, cmd)
    assert out.shape == (h, STATE_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_commanded_yaw_alignment_positive_control_corr_one_mae_zero() -> None:
    h = 16
    heading = np.linspace(0.1, 1.1, h, dtype=np.float64)
    cond_dir = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    cmd = yaw_rate_from_cond_dir(cond_dir, fps=FPS).reshape(-1)

    target_middle = _state(h, dtype=np.float64, seed=3)
    target_middle[:, YAW_RATE_SLICE] = cmd.reshape(h, 1)
    yaw = _yaw_metrics(cmd, target_middle[:, YAW_RATE_SLICE].reshape(-1))
    assert yaw["corr"] == pytest.approx(1.0, abs=1e-9)
    assert yaw["heading_mae_rad"] == pytest.approx(0.0, abs=1e-9)


def test_decision_branch_gate_invalid() -> None:
    base = {c: {"yaw_corr": -0.8, "pop_safe_rate": 0.0} for c in HELD}
    cmd = {c: {"yaw_corr": 0.9, "heading_mae_rad": 0.05, "pop_safe_rate": 0.0} for c in HELD}
    v = classify_f4_commanded_yaw(
        held_out_clips=HELD,
        baseline_rows=base,
        commanded_rows=cmd,
        tau_yaw_rad=0.25,
        gate_valid=False,
    )
    assert v.decision == "GATE_INVALID"


def test_decision_branch_f4_not_explained_when_yaw_not_fixed() -> None:
    base = {
        "Walk_L_To_L": {"yaw_corr": -0.8, "pop_safe_rate": 0.0},
        "Walk_R_To_L": {"yaw_corr": -0.6, "pop_safe_rate": 0.0},
        "Walk_R_To_R": {"yaw_corr": -0.7, "pop_safe_rate": 0.0},
    }
    cmd = {
        "Walk_L_To_L": {"yaw_corr": 0.9, "heading_mae_rad": 0.05, "pop_safe_rate": 0.0},
        "Walk_R_To_L": {"yaw_corr": -0.1, "heading_mae_rad": 0.05, "pop_safe_rate": 0.0},
        "Walk_R_To_R": {"yaw_corr": 0.8, "heading_mae_rad": 0.05, "pop_safe_rate": 0.0},
    }
    v = classify_f4_commanded_yaw(
        held_out_clips=HELD,
        baseline_rows=base,
        commanded_rows=cmd,
        tau_yaw_rad=0.25,
        gate_valid=True,
    )
    assert v.decision == "F4_NOT_EXPLAINED"


def test_decision_branch_f4_control_confirmed_when_pop_still_fails() -> None:
    base = {c: {"yaw_corr": -0.8, "pop_safe_rate": 0.0} for c in HELD}
    cmd = {
        "Walk_L_To_L": {"yaw_corr": 0.7, "heading_mae_rad": 0.05, "pop_safe_rate": 0.0},
        "Walk_R_To_L": {"yaw_corr": 0.8, "heading_mae_rad": 0.03, "pop_safe_rate": 0.0},
        "Walk_R_To_R": {"yaw_corr": 0.9, "heading_mae_rad": 0.02, "pop_safe_rate": 0.0},
    }
    v = classify_f4_commanded_yaw(
        held_out_clips=HELD,
        baseline_rows=base,
        commanded_rows=cmd,
        tau_yaw_rad=0.25,
        gate_valid=True,
    )
    assert v.decision == "F4_CONTROL_CONFIRMED"


def test_decision_branch_f4_mixed_with_pop_when_pop_no_longer_fails() -> None:
    base = {c: {"yaw_corr": -0.8, "pop_safe_rate": 0.0} for c in HELD}
    cmd = {c: {"yaw_corr": 0.9, "heading_mae_rad": 0.03, "pop_safe_rate": 0.3} for c in HELD}
    v = classify_f4_commanded_yaw(
        held_out_clips=HELD,
        baseline_rows=base,
        commanded_rows=cmd,
        tau_yaw_rad=0.25,
        gate_valid=True,
    )
    assert v.decision == "F4_MIXED_WITH_POP"
