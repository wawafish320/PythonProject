from __future__ import annotations

import unittest

import torch

from train.models import EventMotionModel, MotionJointLoss


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


def _build_model(*, bone_names: list[str]) -> EventMotionModel:
    num_joints = len(bone_names)
    torch.manual_seed(0)
    return EventMotionModel(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=8,
        hidden_dim=48,
        num_layers=2,
        dropout=0.0,
        contact_dim=2,
        angvel_dim=num_joints * 3,
        pose_hist_dim=12,
        state_layout=_make_state_layout(num_joints),
        output_layout=_make_output_layout(num_joints),
        bone_names=bone_names,
        contact_plan_enable=True,
        contact_plan_hidden=16,
        direct_pose_enable=True,
        direct_pose_hidden=32,
        direct_pose_meas_mode="concat",
        direct_pose_split_enable=True,
        direct_pose_arm_split_enable=True,
        direct_pose_leg_bones=("thigh_l", "thigh_r"),
        direct_pose_arm_bones=("arm_l", "thigh_l"),
    )


def _build_loss(*, bone_names: list[str]) -> MotionJointLoss:
    loss = MotionJointLoss(
        output_layout=_make_output_layout(len(bone_names)),
        w_direct_pose=1.0,
        direct_pose_loss_leg_split=True,
        direct_pose_leg_bones=("thigh_l", "thigh_r"),
        direct_pose_arm_split_enable=True,
        direct_pose_arm_bones=("arm_l", "thigh_l"),
    )
    loss.set_bone_names(bone_names)
    loss.root_idx = -1
    return loss


class EventMotionModelPhaseEDirectGroupsTest(unittest.TestCase):
    def test_model_and_loss_share_group_resolution_semantics(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        model = _build_model(bone_names=bone_names)
        loss = _build_loss(bone_names=bone_names)

        masks = loss._resolve_direct_group_masks(len(bone_names), device=torch.device("cpu"))
        self.assertIsNotNone(masks)
        assert masks is not None

        self.assertEqual(int(model.direct_pose_leg_out_idx.numel()), int(masks["leg"].sum().item()) * 6)
        self.assertEqual(int(model.direct_pose_arm_out_idx.numel()), int(masks["arm"].sum().item()) * 6)
        self.assertEqual(int(model.direct_pose_else_out_idx.numel()), int(masks["else"].sum().item()) * 6)
        self.assertEqual(int(model.direct_pose_nonleg_out_idx.numel()), int(masks["nonleg"].sum().item()) * 6)

        self.assertTrue(bool(masks["leg"][0].item()))
        self.assertTrue(bool(masks["leg"][1].item()))
        self.assertTrue(bool(masks["arm"][2].item()))
        self.assertFalse(bool(masks["arm"][0].item()))
        self.assertTrue(bool(masks["else"][3].item()))


if __name__ == "__main__":
    unittest.main()
