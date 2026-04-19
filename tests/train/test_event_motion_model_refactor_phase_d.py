from __future__ import annotations

import copy
import unittest

import torch

from train.checkpoint.compat import (
    maybe_upgrade_direct_pose_split_state_dict,
)
from train.models import EventMotionModel


def _make_state_layout(num_joints: int) -> dict[str, dict[str, int]]:
    rot_dim = int(num_joints) * 6
    angvel_dim = int(num_joints) * 3
    return {
        "RootPosition": {"start": 0, "size": 3},
        "RootVelocity": {"start": 3, "size": 2},
        "BoneRotations6D": {"start": 5, "size": rot_dim},
        "BoneAngularVelocities": {"start": 5 + rot_dim, "size": angvel_dim},
    }


def _make_output_layout(num_joints: int) -> dict[str, dict[str, int]]:
    return {"BoneRotations6D": {"start": 0, "size": int(num_joints) * 6}}


def _make_io(batch_size: int, steps: int, num_joints: int, cond_dim: int, contact_dim: int) -> dict[str, torch.Tensor]:
    dx = 5 + int(num_joints) * 6 + int(num_joints) * 3
    angvel_dim = int(num_joints) * 3
    pose_hist_dim = 12
    return {
        "state": torch.randn(batch_size, steps, dx, dtype=torch.float32),
        "cond": torch.randn(batch_size, steps, cond_dim, dtype=torch.float32),
        "contacts": torch.rand(batch_size, steps, contact_dim, dtype=torch.float32),
        "angvel": torch.randn(batch_size, steps, angvel_dim, dtype=torch.float32),
        "pose_history": torch.randn(batch_size, steps, pose_hist_dim, dtype=torch.float32),
    }


def _build_model(
    *,
    bone_names: list[str],
    direct_mode: str = "concat",
    use_event_clock: bool = False,
    use_phase: bool = False,
    phase_mode: str = "concat",
    split_enable: bool = False,
    leg_enable: bool = False,
    leg_bones: tuple[str, ...] | None = None,
) -> EventMotionModel:
    num_joints = len(bone_names)
    cond_dim = 8
    pose_hist_dim = 12
    torch.manual_seed(0)
    model = EventMotionModel(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=cond_dim,
        hidden_dim=48,
        num_layers=2,
        dropout=0.0,
        contact_dim=2,
        angvel_dim=num_joints * 3,
        pose_hist_dim=pose_hist_dim,
        state_layout=_make_state_layout(num_joints),
        output_layout=_make_output_layout(num_joints),
        bone_names=bone_names,
        contact_plan_enable=True,
        contact_plan_hidden=16,
        contact_plan_inject="none",
        use_event_clock=use_event_clock,
        direct_pose_enable=True,
        direct_pose_hidden=32,
        direct_pose_meas_mode=direct_mode,
        direct_pose_plan_drop_prob=0.1,
        direct_pose_meas_drop_prob=0.2,
        direct_pose_meas_noise_std=0.01,
        direct_pose_use_phase_z=use_phase,
        direct_pose_phase_z_mode=phase_mode,
        direct_pose_split_enable=split_enable,
        direct_pose_leg_enable=leg_enable,
        direct_pose_leg_bones=leg_bones,
    )
    model.train()
    return model


class EventMotionModelRefactorPhaseDTest(unittest.TestCase):
    def test_split_and_nonsplit_direct_forward_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps = 2, 3
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=2)

        for split_enable in (False, True):
            with self.subTest(split_enable=split_enable):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=split_enable,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                out = model(
                    io["state"],
                    io["cond"],
                    contacts=io["contacts"],
                    angvel=io["angvel"],
                    pose_history=io["pose_history"],
                )
                self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
                self.assertTrue(torch.isfinite(out["out_direct"]).all().item())
                if split_enable:
                    self.assertIsNotNone(model.direct_pose_leg_terminal)
                    self.assertIsNotNone(model.direct_pose_out_nonleg)
                else:
                    self.assertIsNone(model.direct_pose_leg_terminal)
                    self.assertIsNone(model.direct_pose_out_nonleg)

    def test_split_checkpoint_upgrade_from_legacy_direct_head(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        legacy_model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=False,
            leg_bones=("thigh_l", "thigh_r"),
        )
        split_model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
        )

        legacy_state = copy.deepcopy(legacy_model.state_dict())
        self.assertTrue(maybe_upgrade_direct_pose_split_state_dict(split_model, legacy_state))
        self.assertNotIn("direct_pose_head.6.weight", legacy_state)
        self.assertIn("direct_pose_leg_terminal.6.weight", legacy_state)
        self.assertIn("direct_pose_out_nonleg.weight", legacy_state)

        load_info = split_model.load_state_dict(legacy_state, strict=False)
        self.assertFalse(any(key.startswith("direct_pose_leg_terminal") for key in load_info.missing_keys))
        self.assertFalse(any(key.startswith("direct_pose_out_nonleg") for key in load_info.missing_keys))
        self.assertFalse(any(key == "direct_pose_head.6.weight" for key in load_info.unexpected_keys))

    def test_split_leg_terminal_forward_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps = 2, 3
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=2)

        model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
        )
        out = model(
            io["state"],
            io["cond"],
            contacts=io["contacts"],
            angvel=io["angvel"],
            pose_history=io["pose_history"],
        )

        self.assertTrue(model.direct_pose_split_enable)
        self.assertIsNotNone(model.direct_pose_leg_terminal)
        self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
        self.assertTrue(torch.isfinite(out["out_direct"]).all().item())

if __name__ == "__main__":
    unittest.main()
