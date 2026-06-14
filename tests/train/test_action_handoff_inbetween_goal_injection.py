"""Unit tests — §7.3 §2.4 goal-injection binding probe (pure / model-free parts).

Covers the goal head shape + zero-init no-op, the residual_proj injection delta math, the
L_reach group-norm decrease on a toy, the calibration relerr, the reach_rate summary, and
the BINDING gate decision (STOP semantics, Walk_L_To_R judged on its own row). The base
model / training / full probe are exercised by the end-to-end run, not here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from train.action_handoff_inbetween_goal_injection import (
    DEFAULT_REACH_RATE_GATE,
    WALK_L_TO_R,
    GoalHead,
    calibration_relerr,
    calibration_records_all_pass,
    context_window_indices,
    egocentric_pose_egovel,
    hidden_pre_anchor_loss,
    l_reach,
    loss_plateau_status,
    reach_gate_decision,
    register_goal_injection_pre_temporal,
    summarize_reach_rate,
)
from train.data.action_handoff_inbetween import STATE_DIM


# ------------------------------------------------------------- goal head
def test_goal_head_shape_and_zero_init_is_noop():
    K = 6
    gh = GoalHead.build(goal_flat_dim=K * STATE_DIM, init_scale=0.0)
    goal = torch.randn(K * STATE_DIM)
    delta = gh(goal)
    assert delta.shape == (512,)
    # init_scale=0 → delta is exactly zero (goal head is a no-op before training)
    assert torch.allclose(delta, torch.zeros(512))


def test_goal_head_nonzero_scale_produces_signal():
    K = 6
    gh = GoalHead.build(goal_flat_dim=K * STATE_DIM, init_scale=1.0)
    delta = gh(torch.randn(K * STATE_DIM))
    assert delta.shape == (512,)
    assert torch.isfinite(delta).all()


def test_goal_head_trainable_params_exist():
    gh = GoalHead.build(goal_flat_dim=6 * STATE_DIM, init_scale=1.0)
    n = sum(p.numel() for p in gh.parameters() if p.requires_grad)
    assert n > 0


def test_goal_head_depth_and_film_shape():
    gh = GoalHead.build(goal_flat_dim=6 * STATE_DIM, hidden=128, depth=3, init_scale=1.0, mode="film")
    delta = gh(torch.randn(6 * STATE_DIM))
    assert delta.shape == (1024,)
    assert torch.isfinite(delta).all()


def test_pre_temporal_injection_hook_changes_shared_encoder_activation():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
            )

        def forward(self, x):
            return self.shared_encoder(x)

    model = ToyModel()
    x = torch.randn(2, 3, 4)
    base = model(x)
    delta = torch.ones(512) * 0.25
    handle = register_goal_injection_pre_temporal(model, delta)
    try:
        injected = model(x)
    finally:
        handle.remove()
    restored = model(x)
    assert injected.shape == base.shape == (2, 3, 512)
    assert not torch.allclose(injected, base)
    assert torch.allclose(restored, base)


def test_pre_temporal_injection_supports_early_and_multi_targets():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
            )

        def forward(self, x):
            return self.shared_encoder(x)

    model = ToyModel()
    x = torch.randn(2, 3, 4)
    base = model(x)
    delta = torch.ones(512) * 0.01
    handle = register_goal_injection_pre_temporal(model, delta, targets="shared_encoder.0,shared_encoder.1")
    try:
        injected = model(x)
    finally:
        handle.remove()
    restored = model(x)
    assert injected.shape == base.shape == (2, 3, 512)
    assert not torch.allclose(injected, base)
    assert torch.allclose(restored, base)


def test_pre_temporal_film_zero_delta_is_noop():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
            )

        def forward(self, x):
            return self.shared_encoder(x)

    model = ToyModel()
    x = torch.randn(2, 3, 4)
    base = model(x)
    delta = torch.zeros(1024)
    handle = register_goal_injection_pre_temporal(model, delta, targets="shared_encoder.1", mode="film")
    try:
        injected = model(x)
    finally:
        handle.remove()
    assert torch.allclose(injected, base)


def test_hidden_pre_anchor_loss_is_finite_and_keeps_gradient():
    hidden = torch.randn(2, 5, 512, requires_grad=True)
    centroid = torch.randn(512)
    loss = hidden_pre_anchor_loss(hidden, centroid, end_window_k=3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


# ------------------------------------------------------------- egocentric / L_reach
def test_egocentric_pose_egovel_shapes_and_heading():
    T = 6
    out_raw = torch.zeros(T, 278)
    out_raw[:, 276] = 1.0  # root_vel_x = 1
    cond_dir = torch.tensor([1.0, 0.0])  # heading +x → ego_fwd = vx
    pose, ego = egocentric_pose_egovel(out_raw, cond_dir)
    assert pose.shape == (T, 276)
    assert ego.shape == (T, 2)
    assert torch.allclose(ego[:, 0], torch.ones(T), atol=1e-5)  # forward = vx = 1
    assert torch.allclose(ego[:, 1], torch.zeros(T), atol=1e-5)  # lateral = 0


def test_l_reach_decreases_with_gradient_step():
    """A toy: a trainable delta-on-output reduces group-normalized L_reach."""
    K = 6
    seam = np.zeros((K, STATE_DIM), dtype=np.float32)
    std = np.ones(STATE_DIM, dtype=np.float32)
    cond_dir = torch.tensor([1.0, 0.0])
    base = torch.randn(K, 278)
    delta = torch.zeros(K, 278, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=0.5)
    losses = []
    for _ in range(50):
        out = base + delta
        loss = l_reach(out, seam, std, cond_dir)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


# ------------------------------------------------------------- calibration
def test_calibration_relerr_zero_for_identical():
    a = np.random.RandomState(0).randn(20, 512)
    assert calibration_relerr(a, a.copy()) == pytest.approx(0.0, abs=1e-9)


def test_calibration_relerr_truncates_to_common_len():
    a = np.random.RandomState(0).randn(20, 512)
    assert calibration_relerr(a[:15], a) == pytest.approx(0.0, abs=1e-9)


def test_context_window_indices_edge_and_wrap_modes():
    assert context_window_indices(1, 4, 10, mode="edge").tolist() == [0, 0, 0, 1]
    assert context_window_indices(1, 4, 10, mode="wrap").tolist() == [8, 9, 0, 1]


def test_calibration_records_all_pass_requires_each_clip_reached():
    records = {
        "Walk_L_To_L": {"context_self_min_norm": 0.7, "context_self_reached": True},
        "Walk_L_To_R": {"context_self_min_norm": 1.2, "context_self_reached": True},
    }
    assert calibration_records_all_pass(records, conv_norm_thr=1.5) is True
    records["Walk_L_To_R"]["context_self_reached"] = False
    assert calibration_records_all_pass(records, conv_norm_thr=1.5) is False


# ------------------------------------------------------------- reach summary
def test_summarize_reach_rate():
    out = summarize_reach_rate([0.5, 1.0, 2.0, 3.0], conv_norm_thr=1.5)
    assert out["reach_rate"] == pytest.approx(0.5)
    assert out["reach_min_norm_min"] == pytest.approx(0.5)


# ------------------------------------------------------------- BINDING gate (STOP semantics)
def test_gate_stop_when_no_lift_above_floor():
    g = reach_gate_decision({"Walk_L_To_L": 0.0, "Walk_L_To_R": 0.0, "Walk_R_To_L": 0.0, "Walk_R_To_R": 0.0})
    assert g.lifted_above_floor is False
    assert g.stop is True
    assert "did not lift" in g.reason


def test_gate_stop_when_l_to_r_fails_even_if_others_pass():
    g = reach_gate_decision(
        {"Walk_L_To_L": 0.9, "Walk_L_To_R": 0.1, "Walk_R_To_L": 0.8, "Walk_R_To_R": 0.85},
        gate=0.7,
    )
    assert g.lifted_above_floor is True
    assert g.l_to_r_pass is False
    assert g.stop is True
    assert WALK_L_TO_R in g.reason


def test_gate_pass_when_all_clear():
    g = reach_gate_decision(
        {"Walk_L_To_L": 0.9, "Walk_L_To_R": 0.75, "Walk_R_To_L": 0.8, "Walk_R_To_R": 0.85},
        gate=0.7,
    )
    assert g.all_pass is True
    assert g.stop is False


def test_gate_partial_lift_no_clean_stop():
    g = reach_gate_decision(
        {"Walk_L_To_L": 0.9, "Walk_L_To_R": 0.72, "Walk_R_To_L": 0.4, "Walk_R_To_R": 0.85},
        gate=0.7,
    )
    # L_R passes, lifted, but not all pass → not a clean STOP (proceed cautiously)
    assert g.l_to_r_pass is True and g.lifted_above_floor is True
    assert g.all_pass is False
    assert g.stop is False


def test_default_reach_gate_is_provisional_0_7():
    assert DEFAULT_REACH_RATE_GATE == pytest.approx(0.7)


# ------------------------------------------------------------- plateau ("持平") semantics
def test_plateau_insufficient_samples_is_false():
    st = loss_plateau_status([1.0, 0.9, 0.8], min_steps=10, window=2, rel_delta=0.02)
    assert st["plateau"] is False
    assert "insufficient" in st["reason"]


def test_plateau_still_descending_is_not_plateau():
    # monotone decreasing → recent window is meaningfully below prev → keep training
    vals = [1.0 - 0.05 * i for i in range(12)]
    st = loss_plateau_status(vals, min_steps=4, window=3, rel_delta=0.02)
    assert st["relative_improvement"] > 0.02
    assert st["plateau"] is False
    assert "still improving" in st["reason"]


def test_plateau_worsening_is_not_plateau():
    # recent window worse than prev (curve went up / unstable, e.g. LR too high) → no upgrade
    vals = [0.10] * 3 + [0.20] * 3
    st = loss_plateau_status(vals, min_steps=4, window=3, rel_delta=0.02)
    assert st["relative_improvement"] < -0.02
    assert st["plateau"] is False
    assert "worsened" in st["reason"]


def test_plateau_flat_is_plateau():
    # genuinely flat within rel_delta on both halves → plateau True
    vals = [0.5000, 0.5005, 0.4998, 0.5002, 0.4999, 0.5001, 0.5000, 0.4997]
    st = loss_plateau_status(vals, min_steps=4, window=3, rel_delta=0.02)
    assert abs(st["relative_improvement"]) <= 0.02
    assert st["plateau"] is True
    assert "plateaued" in st["reason"]
