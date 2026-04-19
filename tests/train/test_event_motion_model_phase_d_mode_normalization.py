from __future__ import annotations

import unittest

from train.checkpoint.contract import (
    normalize_contact_plan_init_mode,
    normalize_direct_pose_feat_source,
    normalize_direct_pose_leg_gate_mode,
    normalize_direct_pose_leg_mode,
    normalize_direct_pose_phase_z_mode,
    normalize_lambda_fusion_mode,
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


def _build_model(**overrides) -> EventMotionModel:
    bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
    num_joints = len(bone_names)
    kwargs = dict(
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
        direct_pose_leg_enable=True,
        direct_pose_leg_bones=("thigh_l", "thigh_r"),
    )
    kwargs.update(overrides)
    return EventMotionModel(**kwargs)


class EventMotionModelPhaseDModeNormalizationTest(unittest.TestCase):
    def test_public_normalizers_cover_phase_d_modes(self) -> None:
        self.assertEqual(normalize_contact_plan_init_mode("obs+learnable"), "learnable+obs")
        self.assertEqual(normalize_direct_pose_feat_source("h_temporal"), "hidden_pre")
        self.assertEqual(normalize_direct_pose_phase_z_mode("phase_only_hint"), "replace_contacts")
        self.assertEqual(normalize_direct_pose_leg_mode("axis_angle"), "so3")
        self.assertEqual(normalize_direct_pose_leg_gate_mode("mlp"), "learned")
        self.assertEqual(normalize_direct_pose_leg_gate_mode("log_mag"), "scale")
        self.assertEqual(normalize_lambda_fusion_mode("global"), "global")

    def test_public_normalizers_strict_mode_reject_invalid(self) -> None:
        with self.assertRaises(SystemExit):
            normalize_contact_plan_init_mode("bad", strict=True)
        with self.assertRaises(SystemExit):
            normalize_direct_pose_feat_source("bad", strict=True)
        with self.assertRaises(SystemExit):
            normalize_direct_pose_phase_z_mode("bad", strict=True)
        with self.assertRaises(SystemExit):
            normalize_direct_pose_leg_mode("bad", strict=True)
        with self.assertRaises(SystemExit):
            normalize_lambda_fusion_mode("bad", strict=True)

    def test_model_ctor_canonicalizes_aliases(self) -> None:
        model = _build_model(
            contact_plan_init_mode="obs+learnable",
            direct_pose_feat_source="h_temporal",
            direct_pose_use_phase_z=True,
            direct_pose_phase_z_mode="phase_only_hint",
            direct_pose_leg_mode="axis_angle",
            direct_pose_leg_gate_mode="mlp",
            lambda_fusion_mode="global",
        )

        self.assertEqual(model.contact_plan_init_mode, "learnable+obs")
        self.assertEqual(model.direct_pose_feat_source, "hidden_pre")
        self.assertEqual(model.direct_pose_phase_z_mode, "replace_contacts")
        self.assertEqual(model.direct_pose_leg_mode, "so3")
        self.assertEqual(model.direct_pose_leg_gate_mode, "learned")
        self.assertEqual(model.lambda_fusion_mode, "global")

    def test_model_ctor_falls_back_to_defaults_for_invalid_values(self) -> None:
        model = _build_model(
            contact_plan_init_mode="bad",
            direct_pose_feat_source="bad",
            direct_pose_phase_z_mode="bad",
            direct_pose_leg_mode="bad",
            direct_pose_leg_gate_mode="bad",
            lambda_fusion_mode="bad",
        )

        self.assertEqual(model.contact_plan_init_mode, "learnable")
        self.assertEqual(model.direct_pose_feat_source, "cond")
        self.assertEqual(model.direct_pose_phase_z_mode, "concat")
        self.assertEqual(model.direct_pose_leg_mode, "rot6d_add")
        self.assertEqual(model.direct_pose_leg_gate_mode, "none")
        self.assertEqual(model.lambda_fusion_mode, "per_joint")


if __name__ == "__main__":
    unittest.main()
