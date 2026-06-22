from __future__ import annotations

import types

import torch

from train.action_handoff_inbetween_model import (
    MinimalGoalAR,
    ModelConfig,
    rollout_free_commanded_yaw,
)
from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)


def _model(hidden: int = 32, seam_len: int = 6) -> MinimalGoalAR:
    torch.manual_seed(0)
    return MinimalGoalAR(ModelConfig(hidden=hidden, seam_len=seam_len))


def test_rollout_free_commanded_yaw_shape_dtype_device() -> None:
    model = _model()
    b, c, k, h = 3, 16, model.cfg.seam_len, 12
    ctx = torch.randn(b, c, STATE_DIM, dtype=torch.float32)
    goal = torch.randn(b, k, STATE_DIM, dtype=torch.float32)
    cmd = torch.randn(b, h, 1, dtype=torch.float32)
    out = rollout_free_commanded_yaw(model, ctx, goal, cmd)
    assert out.shape == (b, h, STATE_DIM)
    assert out.dtype == torch.float32
    assert out.device == ctx.device


def test_rollout_free_commanded_yaw_exactly_overwrites_yaw_slice() -> None:
    model = _model()
    b, c, k, h = 2, 16, model.cfg.seam_len, 10
    ctx = torch.randn(b, c, STATE_DIM, dtype=torch.float32)
    goal = torch.randn(b, k, STATE_DIM, dtype=torch.float32)
    cmd = torch.linspace(-2.0, 2.0, h, dtype=torch.float32).view(1, h, 1).repeat(b, 1, 1)
    out = model.rollout_free_commanded_yaw(ctx, goal, cmd)
    assert torch.allclose(out[..., YAW_RATE_SLICE], cmd, atol=0.0, rtol=0.0)


def test_plain_rollout_free_does_not_force_yaw_to_command() -> None:
    model = _model()
    b, c, k, h = 2, 16, model.cfg.seam_len, 10
    ctx = torch.randn(b, c, STATE_DIM, dtype=torch.float32)
    goal = torch.randn(b, k, STATE_DIM, dtype=torch.float32)
    cmd = torch.full((b, h, 1), fill_value=9.0, dtype=torch.float32)
    plain = model.rollout_free(ctx, goal, horizon=h)
    assert not torch.allclose(plain[..., YAW_RATE_SLICE], cmd, atol=1e-4, rtol=0.0)


def test_commanded_yaw_changes_next_step_feedback_input() -> None:
    model = _model()
    b, c, k, h = 1, 8, model.cfg.seam_len, 4
    ctx = torch.zeros(b, c, STATE_DIM, dtype=torch.float32)
    goal = torch.zeros(b, k, STATE_DIM, dtype=torch.float32)
    cmd = torch.tensor([[[0.3], [-0.5], [0.7], [-1.1]]], dtype=torch.float32)

    captured_inputs: list[torch.Tensor] = []

    def spy_step(self: MinimalGoalAR, s_t: torch.Tensor, ctx_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        del ctx_emb, goal_emb
        captured_inputs.append(s_t.detach().clone())
        return s_t + 1.0

    model.step = types.MethodType(spy_step, model)
    out = model.rollout_free_commanded_yaw(ctx, goal, cmd)

    assert len(captured_inputs) == h
    assert captured_inputs[0].shape == (b, STATE_DIM)
    assert torch.allclose(captured_inputs[1][..., YAW_RATE_SLICE], cmd[:, 0, :], atol=0.0, rtol=0.0)
    assert torch.allclose(captured_inputs[2][..., YAW_RATE_SLICE], cmd[:, 1, :], atol=0.0, rtol=0.0)
    assert torch.allclose(out[..., YAW_RATE_SLICE], cmd, atol=0.0, rtol=0.0)


def test_pose_ego_contact_not_directly_replaced_by_command() -> None:
    model = _model()
    b, c, k, h = 2, 8, model.cfg.seam_len, 5
    ctx = torch.zeros(b, c, STATE_DIM, dtype=torch.float32)
    goal = torch.zeros(b, k, STATE_DIM, dtype=torch.float32)
    cmd_a = torch.zeros(b, h, 1, dtype=torch.float32)
    cmd_b = torch.ones(b, h, 1, dtype=torch.float32)

    const = torch.zeros(STATE_DIM, dtype=torch.float32)
    const[POSE_SLICE] = 2.0
    const[EGO_VEL_SLICE] = -3.0
    const[CONTACT_SLICE] = 4.0
    const[YAW_RATE_SLICE] = -8.0

    def const_step(self: MinimalGoalAR, s_t: torch.Tensor, ctx_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        del self, s_t, ctx_emb, goal_emb
        return const.view(1, STATE_DIM).repeat(b, 1)

    model.step = types.MethodType(const_step, model)
    out_a = model.rollout_free_commanded_yaw(ctx, goal, cmd_a)
    out_b = model.rollout_free_commanded_yaw(ctx, goal, cmd_b)

    assert torch.allclose(out_a[..., POSE_SLICE], out_b[..., POSE_SLICE], atol=0.0, rtol=0.0)
    assert torch.allclose(out_a[..., EGO_VEL_SLICE], out_b[..., EGO_VEL_SLICE], atol=0.0, rtol=0.0)
    assert torch.allclose(out_a[..., CONTACT_SLICE], out_b[..., CONTACT_SLICE], atol=0.0, rtol=0.0)
    assert torch.allclose(out_a[..., YAW_RATE_SLICE], cmd_a, atol=0.0, rtol=0.0)
    assert torch.allclose(out_b[..., YAW_RATE_SLICE], cmd_b, atol=0.0, rtol=0.0)
