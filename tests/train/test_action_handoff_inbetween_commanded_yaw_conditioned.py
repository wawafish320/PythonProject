from __future__ import annotations

import torch

from tools.run_action_handoff_inbetween_commanded_yaw_conditioned_probe import (
    CommandedYawLandingConditionedMaskedMiddlePredictor,
    CommandedYawMaskedMiddlePredictor,
    _is_significant_f5_improvement,
    _landing_ego_contact_condition,
)
from tools.run_action_handoff_inbetween_masked_smoke import (
    _f5_only_gate_decision,
    _pivot_channel_mse_weighted,
)
from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_SLICE,
    YAW_RATE_SLICE,
)


def _baseline_row(best_pose: float = 0.10, reach_k3: float = 0.2):
    return {
        "best_pose_d_mean": best_pose,
        "self_reach_gate": {"rate_by_k": {"k=3": reach_k3}},
    }


def test_commanded_yaw_predictor_forward_shape_and_input_dim() -> None:
    b, c, k, h, d = 3, 8, 12, 16, 281
    model = CommandedYawMaskedMiddlePredictor(state_dim=d, context_len=c, seam_len=k, horizon=h, hidden=64)
    ctx = torch.randn(b, c, d, dtype=torch.float32)
    seam = torch.randn(b, k, d, dtype=torch.float32)
    cmd = torch.randn(b, h, 1, dtype=torch.float32)
    out = model(ctx, seam, cmd)
    assert out.shape == (b, h, d)
    assert out.dtype == torch.float32
    assert model.input_dim == c * d + k * d + h


def test_commanded_yaw_landing_predictor_forward_shape_and_input_dim() -> None:
    b, c, k, h, d = 2, 8, 12, 16, 281
    model = CommandedYawLandingConditionedMaskedMiddlePredictor(
        state_dim=d,
        context_len=c,
        seam_len=k,
        horizon=h,
        hidden=64,
    )
    ctx = torch.randn(b, c, d, dtype=torch.float32)
    seam = torch.randn(b, k, d, dtype=torch.float32)
    cmd = torch.randn(b, h, 1, dtype=torch.float32)
    landing = torch.randn(b, 4, dtype=torch.float32)
    out = model(ctx, seam, cmd, landing)
    assert out.shape == (b, h, d)
    assert out.dtype == torch.float32
    assert model.input_dim == c * d + k * d + h + 4


def test_landing_ego_contact_condition_shape_and_order() -> None:
    b, k, d = 3, 12, 281
    seam_n = torch.zeros(b, k, d, dtype=torch.float32)
    seam_n[:, 0, EGO_VEL_SLICE] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    seam_n[:, 0, CONTACT_SLICE] = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    cond = _landing_ego_contact_condition(seam_n)
    assert cond.shape == (b, 4)
    assert torch.allclose(cond[0], torch.tensor([1.0, 2.0, 7.0, 8.0]))
    assert torch.allclose(cond[2], torch.tensor([5.0, 6.0, 11.0, 12.0]))


def test_pivot_loss_ignores_yaw_but_responds_pose_ego_contact() -> None:
    b, h, d = 2, 10, 281
    torch.manual_seed(0)
    pred = torch.randn(b, h, d, dtype=torch.float32)
    target = pred.clone()
    loss_ref = _pivot_channel_mse_weighted(pred, target, pose_w=1.0, ego_w=1.0, contact_w=1.0)

    target_yaw_changed = target.clone()
    target_yaw_changed[..., YAW_RATE_SLICE] += 123.0
    loss_yaw = _pivot_channel_mse_weighted(pred, target_yaw_changed, pose_w=1.0, ego_w=1.0, contact_w=1.0)
    assert torch.allclose(loss_ref, loss_yaw, atol=1e-9)

    target_pose_changed = target.clone()
    target_pose_changed[..., POSE_SLICE] += 0.5
    target_pose_changed[..., EGO_VEL_SLICE] += 0.5
    target_pose_changed[..., CONTACT_SLICE] += 0.5
    loss_pose = _pivot_channel_mse_weighted(pred, target_pose_changed, pose_w=1.0, ego_w=1.0, contact_w=1.0)
    assert float(loss_pose.item()) > float(loss_ref.item())


def test_commanded_yaw_input_can_change_output() -> None:
    b, c, k, h, d = 2, 8, 12, 16, 281
    torch.manual_seed(1)
    model = CommandedYawMaskedMiddlePredictor(state_dim=d, context_len=c, seam_len=k, horizon=h, hidden=64)
    ctx = torch.randn(b, c, d, dtype=torch.float32)
    seam = torch.randn(b, k, d, dtype=torch.float32)
    cmd_zero = torch.zeros(b, h, 1, dtype=torch.float32)
    cmd_one = torch.ones(b, h, 1, dtype=torch.float32)
    out_zero = model(ctx, seam, cmd_zero)
    out_one = model(ctx, seam, cmd_one)
    diff = torch.mean(torch.abs(out_one - out_zero)).item()
    assert diff > 1e-7


def test_landing_condition_input_can_change_output() -> None:
    b, c, k, h, d = 2, 8, 12, 16, 281
    torch.manual_seed(2)
    model = CommandedYawLandingConditionedMaskedMiddlePredictor(
        state_dim=d,
        context_len=c,
        seam_len=k,
        horizon=h,
        hidden=64,
    )
    ctx = torch.randn(b, c, d, dtype=torch.float32)
    seam = torch.randn(b, k, d, dtype=torch.float32)
    cmd = torch.randn(b, h, 1, dtype=torch.float32)
    landing_a = torch.zeros(b, 4, dtype=torch.float32)
    landing_b = torch.ones(b, 4, dtype=torch.float32)
    out_a = model(ctx, seam, cmd, landing_a)
    out_b = model(ctx, seam, cmd, landing_b)
    diff = torch.mean(torch.abs(out_b - out_a)).item()
    assert diff > 1e-7


def test_f5_gate_still_ignores_yaw_fields() -> None:
    per_clip_a = {
        "Walk_L_To_L": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.10, "yaw_corr": -0.8, "heading_mae_rad": 2.0},
        "Walk_L_To_R": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.10, "yaw_corr": -0.7, "heading_mae_rad": 2.1},
        "Walk_R_To_L": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.10, "yaw_corr": -0.6, "heading_mae_rad": 2.2},
        "Walk_R_To_R": {"pop_safe_rate": 0.2, "best_pose_d_mean": 0.10, "yaw_corr": -0.5, "heading_mae_rad": 2.3},
    }
    per_clip_b = {
        k: {**v, "yaw_corr": 0.95, "heading_mae_rad": 0.01}
        for k, v in per_clip_a.items()
    }
    baseline_free = {k: _baseline_row(best_pose=0.10) for k in per_clip_a}
    baseline_pinned = {k: _baseline_row(best_pose=0.10) for k in per_clip_a}
    out_a = _f5_only_gate_decision(
        per_clip_a,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.02,
        require_reach_lift=False,
    )
    out_b = _f5_only_gate_decision(
        per_clip_b,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.02,
        require_reach_lift=False,
    )
    assert out_a["all_pass"] == out_b["all_pass"]
    assert out_a["per_clip_pass"] == out_b["per_clip_pass"]


def test_significant_improvement_prefers_popsafe_or_joint_pop_contact_lift() -> None:
    assert _is_significant_f5_improvement(
        {"pop_safe_rate_delta": 0.05, "pop_mean_delta": 0.10, "ego_vel_pop_mean_delta": 0.10, "contact_pop_mean_delta": 0.10},
        pop_safe_improve_eps=0.05,
        pop_mean_improve_eps=0.02,
    )
    assert _is_significant_f5_improvement(
        {"pop_safe_rate_delta": 0.0, "pop_mean_delta": -0.03, "ego_vel_pop_mean_delta": 0.01, "contact_pop_mean_delta": -0.04},
        pop_safe_improve_eps=0.05,
        pop_mean_improve_eps=0.02,
    )
    assert not _is_significant_f5_improvement(
        {"pop_safe_rate_delta": 0.0, "pop_mean_delta": 0.01, "ego_vel_pop_mean_delta": -0.20, "contact_pop_mean_delta": -0.01},
        pop_safe_improve_eps=0.05,
        pop_mean_improve_eps=0.02,
    )
    assert not _is_significant_f5_improvement(
        {"pop_safe_rate_delta": -0.05, "pop_mean_delta": -0.10, "ego_vel_pop_mean_delta": -0.10, "contact_pop_mean_delta": -0.10},
        pop_safe_improve_eps=0.05,
        pop_mean_improve_eps=0.02,
    )
