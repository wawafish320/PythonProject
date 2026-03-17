from __future__ import annotations

import copy
import unittest

import torch

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
        contact_phase_state_enable=use_phase,
        phase_reset_source="contacts_meas",
        direct_pose_split_enable=split_enable,
        direct_pose_leg_enable=leg_enable,
        direct_pose_leg_bones=leg_bones,
    )
    model.train()
    return model


class EventMotionModelRefactorPhaseDTest(unittest.TestCase):
    def test_direct_override_regression_covers_event_clock_on_off(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps, contact_dim = 2, 3, 2
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=contact_dim)
        phase_init = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        cases = [
            {
                "name": "concat_override_event_clock_off",
                "kwargs": dict(direct_mode="concat", use_event_clock=False, use_phase=False, phase_mode="concat"),
                "plan_override": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "meas_override": torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
                "phase_z": None,
                "expect_event_clock": False,
                "expect_phase": False,
            },
            {
                "name": "mode_select_ignore_meas_event_clock_on",
                "kwargs": dict(direct_mode="mode_select", use_event_clock=True, use_phase=False, phase_mode="concat"),
                "plan_override": torch.tensor([[0.1, 0.9]], dtype=torch.float32),
                "meas_override": "ignore",
                "phase_z": None,
                "expect_event_clock": True,
                "expect_phase": False,
            },
            {
                "name": "replace_contacts_override_event_clock_on",
                "kwargs": dict(direct_mode="concat", use_event_clock=True, use_phase=True, phase_mode="replace_contacts"),
                "plan_override": "zero",
                "meas_override": torch.tensor(
                    [[[0.6, 0.4]], [[0.3, 0.7]]],
                    dtype=torch.float32,
                ),
                "phase_z": phase_init,
                "expect_event_clock": True,
                "expect_phase": True,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                model = _build_model(bone_names=bone_names, **case["kwargs"])
                model.direct_pose_plan_override = case["plan_override"]
                model.direct_pose_meas_override = case["meas_override"]

                torch.manual_seed(123)
                out = model(
                    io["state"],
                    io["cond"],
                    contacts=io["contacts"],
                    angvel=io["angvel"],
                    pose_history=io["pose_history"],
                    phase_z=case["phase_z"],
                )

                self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
                self.assertEqual(out["contacts_plan"].shape, (batch_size, steps, contact_dim))
                self.assertEqual(out["contacts_meas"].shape, (batch_size, steps, contact_dim))
                self.assertTrue(torch.isfinite(out["out_direct"]).all().item())
                self.assertTrue(torch.isfinite(out["contacts_plan"]).all().item())

                if case["expect_event_clock"]:
                    self.assertIn("event_clock_lambda_corr", out)
                    self.assertEqual(out["event_clock_lambda_corr"].shape[:2], (batch_size, steps))
                    self.assertTrue(torch.isfinite(out["event_clock_lambda_corr"]).all().item())
                else:
                    self.assertNotIn("event_clock_lambda_corr", out)

                if case["expect_phase"]:
                    self.assertIn("phase_z_next", out)
                    self.assertEqual(out["phase_z_next"].shape, (batch_size, 2 * contact_dim))
                    self.assertTrue(torch.isfinite(out["phase_z_next"]).all().item())

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
                    self.assertIsNotNone(model.direct_pose_out_leg)
                    self.assertIsNotNone(model.direct_pose_out_nonleg)
                else:
                    self.assertIsNone(model.direct_pose_out_leg)
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
        self.assertTrue(split_model.adapt_legacy_state_dict_(legacy_state))
        self.assertNotIn("direct_pose_head.6.weight", legacy_state)
        self.assertIn("direct_pose_out_leg.weight", legacy_state)
        self.assertIn("direct_pose_out_nonleg.weight", legacy_state)

        load_info = split_model.load_state_dict(legacy_state, strict=False)
        self.assertFalse(any(key.startswith("direct_pose_out_leg") for key in load_info.missing_keys))
        self.assertFalse(any(key.startswith("direct_pose_out_nonleg") for key in load_info.missing_keys))
        self.assertFalse(any(key == "direct_pose_head.6.weight" for key in load_info.unexpected_keys))

    def test_cross_leg_ablate_helper_supports_concat_and_replace_contacts(self) -> None:
        batch_size, steps, contact_dim = 2, 3, 2
        direct_dim = 6

        layout_concat = EventMotionModel._resolve_direct_pose_contact_layout(
            total_dim=direct_dim + contact_dim + contact_dim + (2 * contact_dim),
            direct_dim=direct_dim,
            plan_dim=contact_dim,
            meas_dim_raw=contact_dim,
            phase_dim=2 * contact_dim,
            contact_dim=contact_dim,
        )
        self.assertIsNotNone(layout_concat)
        seq_concat = torch.arange(
            batch_size * steps * (direct_dim + contact_dim + contact_dim + (2 * contact_dim)),
            dtype=torch.float32,
        ).view(batch_size, steps, -1)
        EventMotionModel._ablate_direct_pose_contact_channel(
            seq_concat,
            ablation="zero",
            channel=1,
            batch_size=batch_size,
            steps=steps,
            plan_slice=layout_concat["plan"],
            meas_slice=layout_concat["meas"],
            phase_slice=layout_concat["phase"],
        )
        self.assertTrue(torch.equal(seq_concat[..., direct_dim + 1], torch.zeros_like(seq_concat[..., direct_dim + 1])))
        self.assertTrue(
            torch.equal(seq_concat[..., direct_dim + contact_dim + 1], torch.zeros_like(seq_concat[..., direct_dim + contact_dim + 1]))
        )
        phase_start = direct_dim + contact_dim + contact_dim + 2
        self.assertTrue(torch.equal(seq_concat[..., phase_start : phase_start + 2], torch.zeros_like(seq_concat[..., phase_start : phase_start + 2])))

        layout_replace = EventMotionModel._resolve_direct_pose_contact_layout(
            total_dim=direct_dim + (2 * contact_dim),
            direct_dim=direct_dim,
            plan_dim=contact_dim,
            meas_dim_raw=0,
            phase_dim=2 * contact_dim,
            contact_dim=contact_dim,
        )
        self.assertIsNotNone(layout_replace)
        seq_replace = torch.arange(batch_size * steps * (direct_dim + (2 * contact_dim)), dtype=torch.float32).view(batch_size, steps, -1)
        original = seq_replace.clone()
        EventMotionModel._ablate_direct_pose_contact_channel(
            seq_replace,
            ablation="roll_batch",
            channel=0,
            batch_size=batch_size,
            steps=steps,
            plan_slice=layout_replace["plan"],
            meas_slice=layout_replace["meas"],
            phase_slice=layout_replace["phase"],
        )
        self.assertTrue(torch.equal(seq_replace[0, :, direct_dim : direct_dim + 2], original[1, :, direct_dim : direct_dim + 2]))
        self.assertTrue(torch.equal(seq_replace[1, :, direct_dim : direct_dim + 2], original[0, :, direct_dim : direct_dim + 2]))


if __name__ == "__main__":
    unittest.main()
