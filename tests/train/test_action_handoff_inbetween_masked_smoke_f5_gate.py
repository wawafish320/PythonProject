from __future__ import annotations

from tools.run_action_handoff_inbetween_masked_smoke import _f5_only_gate_decision


def _baseline_row(best_pose: float = 0.10, reach_k3: float = 0.2):
    return {
        "best_pose_d_mean": best_pose,
        "self_reach_gate": {"rate_by_k": {"k=3": reach_k3}},
    }


def test_f5_gate_ignores_yaw_fields_and_passes_on_pop_pose() -> None:
    per_clip = {
        "Walk_L_To_L": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.10, "yaw_corr": -0.9, "heading_mae_rad": 3.14},
        "Walk_L_To_R": {"pop_safe_rate": 0.1, "best_pose_d_mean": 0.11, "yaw_corr": -0.8, "heading_mae_rad": 2.5},
        "Walk_R_To_L": {"pop_safe_rate": 0.3, "best_pose_d_mean": 0.09, "yaw_corr": -0.7, "heading_mae_rad": 1.8},
        "Walk_R_To_R": {"pop_safe_rate": 0.4, "best_pose_d_mean": 0.12, "yaw_corr": -0.6, "heading_mae_rad": 1.2},
    }
    baseline_free = {k: _baseline_row() for k in per_clip}
    baseline_pinned = {k: _baseline_row() for k in per_clip}
    out = _f5_only_gate_decision(
        per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.05,
        require_reach_lift=False,
    )
    assert out["all_pass"] is True
    assert out["per_clip_pass"]["Walk_L_To_R"] is True


def test_f5_gate_fails_when_pop_or_pose_fail() -> None:
    per_clip = {
        "Walk_L_To_L": {"pop_safe_rate": 0.0, "best_pose_d_mean": 0.10},
        "Walk_L_To_R": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.25},
        "Walk_R_To_L": {"pop_safe_rate": 0.3, "best_pose_d_mean": 0.10},
        "Walk_R_To_R": {"pop_safe_rate": 0.4, "best_pose_d_mean": 0.10},
    }
    baseline_free = {k: _baseline_row(best_pose=0.10) for k in per_clip}
    baseline_pinned = {k: _baseline_row(best_pose=0.10) for k in per_clip}
    out = _f5_only_gate_decision(
        per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.01,
        require_reach_lift=False,
    )
    assert out["all_pass"] is False
    assert out["per_clip_pass"]["Walk_L_To_L"] is False
    assert out["per_clip_pass"]["Walk_L_To_R"] is False
