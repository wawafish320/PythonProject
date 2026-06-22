from __future__ import annotations

import numpy as np
import pytest

from train.action_handoff_inbetween_goal_injection import joint_action_binding_gate_decision
from train.action_handoff_inbetween_reach import (
    HIDDEN_DIM,
    build_hidden_pre_anchors,
    build_same_source_hidden_pre_anchors,
    summarize_absolute_self_reach,
)


def _cluster(direction: np.ndarray, *, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return d[None, :] * 6.0 + noise * rng.normal(size=(n, HIDDEN_DIM))


def test_absolute_self_reach_gate_reports_k2_k3_k5_rates() -> None:
    out = summarize_absolute_self_reach([0.01, 0.03, 0.07], self_reach_abs_cos=0.01)

    assert out["rate_by_k"]["k=2"] == pytest.approx(1 / 3)
    assert out["rate_by_k"]["k=3"] == pytest.approx(2 / 3)
    assert out["rate_by_k"]["k=5"] == pytest.approx(2 / 3)
    assert out["threshold_abs_by_k"]["k=3"] == pytest.approx(0.03)


def test_joint_gate_requires_reach_yaw_pop_and_pose_together() -> None:
    rows = {
        "Walk_L_To_R": {
            "self_reach_gate": {"rate_by_k": {"k=3": 0.8}},
            "yaw_corr": 0.5,
            "heading_mae_rad": 0.1,
            "pop_safe_rate": 0.4,
            "best_pose_d_mean": 0.10,
        }
    }
    baselines = [{"Walk_L_To_R": {"self_reach_gate": {"rate_by_k": {"k=3": 0.2}}, "best_pose_d_mean": 0.11}}]

    passed = joint_action_binding_gate_decision(rows, baseline_metrics=baselines)
    assert passed.per_clip_pass["Walk_L_To_R"] is True
    assert passed.l_to_r_pass is True

    rows["Walk_L_To_R"]["yaw_corr"] = -0.1
    failed = joint_action_binding_gate_decision(rows, baseline_metrics=baselines)
    assert failed.per_clip_pass["Walk_L_To_R"] is False
    assert failed.per_clip_checks["Walk_L_To_R"]["realized_yaw_corr_positive"] is False
    assert failed.stop is True


def test_same_source_anchor_rebuild_uses_evaluated_hidden_capture() -> None:
    rng = np.random.default_rng(0)
    old_dir = rng.normal(size=HIDDEN_DIM)
    old_dir /= np.linalg.norm(old_dir)
    new_dir = rng.normal(size=HIDDEN_DIM)
    new_dir -= old_dir * float(np.dot(new_dir, old_dir))
    new_dir /= np.linalg.norm(new_dir)

    old_hidden = np.concatenate([rng.normal(size=(20, HIDDEN_DIM)), _cluster(old_dir, n=12, noise=0.02, rng=rng)])
    new_hidden = np.concatenate([rng.normal(size=(20, HIDDEN_DIM)), _cluster(new_dir, n=12, noise=0.02, rng=rng)])

    old_anchor = build_hidden_pre_anchors({"T": old_hidden}, ("T",), end_window_k=12)["T"]
    new_anchors, diag = build_same_source_hidden_pre_anchors(
        {"T": new_hidden},
        self_check_hidden_by_clip={"T": new_hidden},
        turn_clips=("T",),
        end_window_k=12,
    )
    new_anchor = new_anchors["T"]

    assert new_anchor.min_abs_cos(new_hidden) < old_anchor.min_abs_cos(new_hidden)
    assert diag["T"].reach_available is True
    assert diag["T"].anchor_hidden_shape == (32, HIDDEN_DIM)


def test_same_source_anchor_flags_miscalibrated_self_check() -> None:
    rng = np.random.default_rng(1)
    direction = rng.normal(size=HIDDEN_DIM)
    direction /= np.linalg.norm(direction)
    anchor_hidden = np.concatenate([rng.normal(size=(20, HIDDEN_DIM)), _cluster(direction, n=12, noise=0.01, rng=rng)])
    far_check = _cluster(-direction, n=12, noise=0.01, rng=rng)

    _, diag = build_same_source_hidden_pre_anchors(
        {"T": anchor_hidden},
        self_check_hidden_by_clip={"T": far_check},
        turn_clips=("T",),
        end_window_k=12,
    )

    assert diag["T"].self_check_reached is False
    assert diag["T"].reach_available is False
    assert "reach unavailable" in diag["T"].reason
