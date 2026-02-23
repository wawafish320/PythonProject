#!/usr/bin/env python3
"""
Minimal sanity check for the plan/meas/e_t/omega_hat loop.

Runs a forward+backward pass and checks shapes/finite values.
"""
from __future__ import annotations

import torch

from train.models import EventMotionModel


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    B = 4
    J = 2  # joints used in state/layout + output rotations
    Dy = J * 6  # 6D per joint
    Dx = 3 + 2 + J * 6 + J * 3  # RootPosition(3) + RootVelocity(2) + Rot6D + AngVel
    Dc = 8
    C = 2
    A = J * 3
    P = 24

    state_layout = {
        "RootPosition": {"start": 0, "size": 3},
        "RootVelocity": {"start": 3, "size": 2},
        "BoneRotations6D": {"start": 5, "size": J * 6},
        "BoneAngularVelocities": {"start": 5 + J * 6, "size": J * 3},
    }

    model = EventMotionModel(
        in_state_dim=Dx,
        out_motion_dim=Dy,
        cond_dim=Dc,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        contact_dim=C,
        angvel_dim=A,
        pose_hist_dim=P,
        state_layout=state_layout,
        bone_names=["j0", "j1"],
        contact_plan_enable=True,
        contact_plan_hidden=16,
        contact_plan_inject="none",
        contact_meas_enable=True,
        contact_meas_hidden=32,
        contact_meas_dropout=0.0,
    ).to(device=device, dtype=dtype)

    state = torch.randn(B, Dx, device=device, dtype=dtype, requires_grad=True)
    cond = torch.randn(B, Dc, device=device, dtype=dtype)
    angvel = torch.randn(B, A, device=device, dtype=dtype)
    pose_hist = torch.randn(B, P, device=device, dtype=dtype)
    plan_z = torch.zeros(B, int(getattr(model, "contact_plan_hidden", 0) or 0), device=device, dtype=dtype)
    gt_contacts = torch.rand(B, C, device=device, dtype=dtype)

    out = model(state, cond, angvel=angvel, pose_history=pose_hist, plan_z=plan_z)
    for k in ("out", "contacts_plan", "contacts_meas", "contacts_err", "omega_hat", "plan_z_next"):
        assert k in out, f"missing key: {k}"

    assert out["out"].shape == (B, Dy)
    assert out["contacts_plan"].shape == (B, C)
    assert out["contacts_meas"].shape == (B, C)
    assert out["contacts_err"].shape == (B, C)
    assert out["omega_hat"].shape == (B, Dy // 6, 3)
    assert out["plan_z_next"].shape == plan_z.shape

    loss = (
        (out["contacts_plan"] - gt_contacts).pow(2).mean()
        + out["out"].pow(2).mean()
        + out["omega_hat"].pow(2).mean()
    )
    assert torch.isfinite(loss).all(), "loss is not finite"
    loss.backward()
    assert torch.isfinite(state.grad).all(), "grad is not finite"
    print("[OK] debug_contact_loop: forward/backward finite, shapes OK")


if __name__ == "__main__":
    main()
