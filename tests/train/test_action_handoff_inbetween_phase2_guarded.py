"""Pure tests for §7.3 PHASE 2 guarded fine-tune helpers.

No base checkpoint load, training, or rollout is exercised here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.run_action_handoff_inbetween_phase2_guarded_finetune_probe import (
    _drift_guard,
    _l2_to_init,
    _phase2_decision,
    _select_trainable_base_params,
    _snapshot_params,
)
from tools.run_action_handoff_inbetween_reach_honesty_probe import _g3_decision, _self_reach_rates


class ToyPhase2Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
        )
        self.residual_proj = torch.nn.Linear(8, 8)
        self.other = torch.nn.Linear(8, 8)


def test_select_trainable_base_tail_freezes_early_and_other_params():
    model = ToyPhase2Model()
    selected = _select_trainable_base_params(model, "tail")
    selected_names = {name for name, _ in selected}

    assert "shared_encoder.0.weight" not in selected_names
    assert "shared_encoder.2.weight" in selected_names
    assert "shared_encoder.4.weight" in selected_names
    assert "residual_proj.weight" in selected_names
    assert "other.weight" not in selected_names

    flags = dict(model.named_parameters())
    assert flags["shared_encoder.0.weight"].requires_grad is False
    assert flags["shared_encoder.2.weight"].requires_grad is True
    assert flags["residual_proj.weight"].requires_grad is True
    assert flags["other.weight"].requires_grad is False


def test_l2_to_init_is_finite_scalar_and_backprops_to_selected_params():
    model = ToyPhase2Model()
    selected = _select_trainable_base_params(model, "residual_only")
    init = _snapshot_params(selected)
    with torch.no_grad():
        selected[0][1].add_(0.25)

    loss = _l2_to_init(selected, init)
    assert loss.shape == ()
    assert loss.dtype == selected[0][1].dtype
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0

    loss.backward()
    for _, param in selected:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_drift_guard_allows_small_or_relative_bounded_changes():
    before = {"best_pose_d_mean": 1.0, "pop_mean": 0.5, "root_speed_mean": 2.0}
    after = {"best_pose_d_mean": 1.05, "pop_mean": 0.49, "root_speed_mean": 2.09}
    out = _drift_guard(before, after, rel_tol=0.1, abs_tol=0.01)
    assert out["passed"] is True
    assert out["checks"]["best_pose_d_mean"]["passed"] is True


def test_drift_guard_fails_large_unbounded_regression():
    before = {"best_pose_d_mean": 1.0, "pop_mean": 0.5, "root_speed_mean": 2.0}
    after = {"best_pose_d_mean": 1.3, "pop_mean": 0.9, "root_speed_mean": 2.1}
    out = _drift_guard(before, after, rel_tol=0.1, abs_tol=0.01)
    assert out["passed"] is False
    assert out["checks"]["pop_mean"]["passed"] is False


@pytest.mark.parametrize(
    ("gate", "drift_passed", "expected"),
    [
        (
            SimpleNamespace(lifted_above_floor=True, all_pass=True, l_to_r_pass=True),
            True,
            "success_all_targets_pass_walk_f_stable",
        ),
        (
            SimpleNamespace(lifted_above_floor=True, all_pass=False, l_to_r_pass=True),
            True,
            "partial_success_l_r_passes_walk_f_stable_all_targets_not_yet",
        ),
        (
            SimpleNamespace(lifted_above_floor=True, all_pass=False, l_to_r_pass=False),
            True,
            "partial_reach_lifted_walk_f_stable_l_r_not_yet",
        ),
        (
            SimpleNamespace(lifted_above_floor=True, all_pass=False, l_to_r_pass=True),
            False,
            "failure_reach_lifted_but_walk_f_drifted",
        ),
        (
            SimpleNamespace(lifted_above_floor=False, all_pass=False, l_to_r_pass=False),
            True,
            "reconsider_reach_not_lifted",
        ),
    ],
)
def test_phase2_decision_separates_l_r_partial_from_all_target_success(gate, drift_passed, expected):
    assert _phase2_decision(gate, drift_passed=drift_passed) == expected


def test_self_reach_rates_are_per_start_not_best_start_only():
    out = _self_reach_rates([0.01, 0.04, 0.08, 0.12], self_abs_floor=0.02, k_values=[2.0, 3.0, 5.0])

    assert out["rate_by_k"]["k=2"] == 0.5
    assert out["count_by_k"]["k=2"] == 2
    assert out["rate_by_k"]["k=3"] == 0.5
    assert out["rate_by_k"]["k=5"] == 0.75
    assert out["threshold_abs_by_k"]["k=5"] == pytest.approx(0.10)


def test_g3_keeps_b4_blocked_unless_reach_yaw_and_pop_all_pass():
    bc_rows = {
        "column_keys": {"pinned": "pinned", "free": "free"},
        "walk_l_to_r": {
            "pinned": {
                "self_reach_gate": {"rate_by_k": {"k=3": 0.0}},
                "yaw_heading_mae_rad_mean": 1.0,
            },
            "free": {
                "self_reach_gate": {"rate_by_k": {"k=3": 0.25}},
                "yaw_heading_mae_rad_mean": 0.5,
                "yaw_corr_mean": 0.4,
                "pop_safe_rate": 0.0,
            },
        },
    }

    out = _g3_decision(bc_rows)

    assert out["required_simultaneous_checks"]["self_reach_k=3_rate_lifted_free_vs_pinned"] is True
    assert out["required_simultaneous_checks"]["realized_yaw_corr_positive"] is True
    assert out["required_simultaneous_checks"]["heading_mae_significantly_down_vs_pinned"] is True
    assert out["required_simultaneous_checks"]["pop_safe_positive"] is False
    assert out["B4_seam_status"] == "blocked"
