from __future__ import annotations

"""Stage-3a B1 probe model/harness smoke (§7.3). Pure synthetic data; CPU; fast.

Validates the wiring, NOT B1: shapes, loss-decreases-on-overfit, normalizer round-trip,
state-space metric schema. The binding gate (z-reach, checkpoint init) is 3b.
"""

import numpy as np
import torch

from train.action_handoff_inbetween_model import (
    GateThresholds,
    LossWeights,
    MinimalGoalAR,
    ModelConfig,
    StateNormalizer,
    compute_losses,
    evaluate_rollout_state_space,
)
from train.data.action_handoff_inbetween import POSE_SLICE, STATE_DIM


def _clips() -> dict:
    rng = np.random.default_rng(0)
    return {
        "Walk_F": rng.normal(size=(80, STATE_DIM)).astype(np.float32),
        "Walk_R_To_R": rng.normal(size=(60, STATE_DIM)).astype(np.float32),
    }


def test_normalizer_round_trip() -> None:
    norm = StateNormalizer(_clips())
    x = torch.randn(4, STATE_DIM)
    back = norm.denormalize(norm.normalize(x))
    assert torch.allclose(back, x, atol=1e-4)


def test_model_rollout_shapes() -> None:
    C, H, K = 16, 12, 6
    model = MinimalGoalAR(ModelConfig(seam_len=K, hidden=32))
    B = 5
    ctx = torch.randn(B, C, STATE_DIM)
    goal = torch.randn(B, K, STATE_DIM)
    gt_middle = torch.randn(B, H, STATE_DIM)
    mid_pred, seam_pred = model.rollout_teacher_forced(ctx, goal, gt_middle)
    assert mid_pred.shape == (B, H, STATE_DIM)
    assert seam_pred.shape == (B, K, STATE_DIM)
    free = model.rollout_free(ctx, goal, horizon=20)
    assert free.shape == (B, 20, STATE_DIM)


def test_loss_decreases_on_overfit() -> None:
    torch.manual_seed(0)
    C, H, K = 16, 12, 6
    norm = StateNormalizer(_clips())
    model = MinimalGoalAR(ModelConfig(seam_len=K, hidden=64))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = LossWeights()
    B = 8
    ctx = norm.normalize(torch.randn(B, C, STATE_DIM))
    mid = norm.normalize(torch.randn(B, H, STATE_DIM))
    seam = norm.normalize(torch.randn(B, K, STATE_DIM))

    _, first = compute_losses(model, ctx, seam, mid, seam, w)
    for _ in range(120):
        total, _ = compute_losses(model, ctx, seam, mid, seam, w)
        opt.zero_grad()
        total.backward()
        opt.step()
    _, last = compute_losses(model, ctx, seam, mid, seam, w)
    assert last["total"] < first["total"]
    for k in ("total", "L_middle", "L_reach", "L_seam_C1"):
        assert k in last


def test_state_space_metric_schema_and_resumable_logic() -> None:
    std = np.ones(STATE_DIM, dtype=np.float32)
    thr = GateThresholds(tau_pose=0.15, tau_pop=0.30, reach_proxy_thr=1.0)
    K = 6
    goal_seam = np.zeros((K, STATE_DIM), dtype=np.float32)

    # Rollout that lands exactly on the seam pose → resumable + pop_safe.
    roll_hit = np.ones((20, STATE_DIM), dtype=np.float32)
    roll_hit[10] = 0.0  # one frame matches the seam exactly
    out = evaluate_rollout_state_space(roll_hit, goal_seam, std, thr)
    for k in ("best_pose_d", "clip_resumable", "pop", "pop_safe", "reach_proxy", "fallback"):
        assert k in out
    assert out["clip_resumable"] is True
    assert out["fallback"] is False

    # Rollout far from the seam in pose → not resumable → fallback.
    roll_miss = np.ones((20, STATE_DIM), dtype=np.float32) * 5.0
    out2 = evaluate_rollout_state_space(roll_miss, goal_seam, std, thr)
    assert out2["clip_resumable"] is False
    assert out2["fallback"] is True
