from __future__ import annotations

import numpy as np
import pytest
import torch

from train.action_handoff_inbetween_model import (
    GateThresholds,
    evaluate_rollout_state_space,
    loss_middle,
    loss_seam_c1,
)
from train.data.action_handoff_inbetween import (
    EGO_VEL_SLICE,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
)


def test_loss_middle_ignores_yaw_channel() -> None:
    b, h = 2, 12
    pred = torch.zeros(b, h, STATE_DIM)
    tgt = torch.zeros(b, h, STATE_DIM)
    tgt[..., YAW_RATE_SLICE] = 5.0
    v = loss_middle(pred, tgt)
    assert v.item() == pytest.approx(0.0, abs=1e-9)


def test_loss_middle_tracks_pose_egovel_contact_channels() -> None:
    b, h = 2, 12
    pred = torch.zeros(b, h, STATE_DIM)
    tgt = torch.zeros(b, h, STATE_DIM)
    tgt[..., POSE_SLICE] = 1.0
    tgt[..., EGO_VEL_SLICE] = 2.0
    v = loss_middle(pred, tgt)
    assert v.item() > 0.0


def test_loss_seam_c1_ignores_yaw_channel() -> None:
    b, h, k = 2, 12, 6
    mid = torch.zeros(b, h, STATE_DIM)
    seam = torch.zeros(b, k, STATE_DIM)
    seam[:, 0, YAW_RATE_SLICE] = 9.0
    v = loss_seam_c1(mid, seam)
    assert v.item() == pytest.approx(0.0, abs=1e-9)


def test_evaluate_rollout_pop_ignores_yaw_gap_under_f5_policy() -> None:
    thr = GateThresholds(tau_pose=0.15, tau_pop=0.30, reach_proxy_thr=1.0)
    std = np.ones(STATE_DIM, dtype=np.float64)
    goal = np.zeros((6, STATE_DIM), dtype=np.float64)
    roll = np.zeros((8, STATE_DIM), dtype=np.float64)
    roll[3, YAW_RATE_SLICE] = 50.0
    out = evaluate_rollout_state_space(roll, goal, std, thr)
    assert out["clip_resumable"] is True
    assert out["pop"] == pytest.approx(0.0, abs=1e-9)
    assert out["pop_safe"] is True
