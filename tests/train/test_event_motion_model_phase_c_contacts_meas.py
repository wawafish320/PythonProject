from __future__ import annotations

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


def _build_model(*, contact_dim: int = 2, use_event_clock: bool = False) -> EventMotionModel:
    bone_names = ["thigh_l", "thigh_r"]
    num_joints = len(bone_names)
    torch.manual_seed(0)
    model = EventMotionModel(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=4,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        contact_dim=contact_dim,
        angvel_dim=num_joints * 3,
        pose_hist_dim=0,
        state_layout=_make_state_layout(num_joints),
        output_layout=_make_output_layout(num_joints),
        bone_names=bone_names,
        contact_plan_enable=True,
        contact_plan_hidden=16,
        use_event_clock=use_event_clock,
        direct_pose_enable=False,
    )
    model.eval()
    return model


class EventMotionModelContactsMeasCanonicalizationTest(unittest.TestCase):
    def test_helper_broadcasts_and_computes_delta(self) -> None:
        model = _build_model(contact_dim=2)
        contacts_meas, delta_meas, meas_prev_t = model._canonicalize_contacts_meas_inputs(
            torch.tensor([0.25, 0.75], dtype=torch.float32),
            torch.tensor([0.10, 0.40], dtype=torch.float32),
            batch_size=3,
            seq_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(contacts_meas.shape, (3, 4, 2))
        self.assertEqual(delta_meas.shape, (3, 4, 2))
        self.assertIsNotNone(meas_prev_t)
        self.assertEqual(meas_prev_t.shape, (3, 2))
        expected_meas = torch.tensor([0.25, 0.75], dtype=torch.float32).view(1, 1, 2).expand(3, 4, 2)
        expected_prev = torch.tensor([0.10, 0.40], dtype=torch.float32).view(1, 2).expand(3, 2)
        torch.testing.assert_close(contacts_meas, expected_meas)
        torch.testing.assert_close(meas_prev_t, expected_prev)
        torch.testing.assert_close(delta_meas[:, 0], expected_meas[:, 0] - expected_prev)
        torch.testing.assert_close(delta_meas[:, 1:], torch.zeros((3, 3, 2), dtype=torch.float32))

    def test_helper_pads_and_truncates_channel_dim(self) -> None:
        model = _build_model(contact_dim=4)
        contacts_meas, delta_meas, meas_prev_t = model._canonicalize_contacts_meas_inputs(
            torch.tensor([[0.2, 0.4]], dtype=torch.float32),
            torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32),
            batch_size=2,
            seq_len=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        expected_meas = torch.tensor([0.2, 0.4, 0.0, 0.0], dtype=torch.float32).view(1, 1, 4).expand(2, 3, 4)
        expected_prev = torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float32).view(1, 4).expand(2, 4)
        torch.testing.assert_close(contacts_meas, expected_meas)
        torch.testing.assert_close(meas_prev_t, expected_prev)
        torch.testing.assert_close(delta_meas[:, 0], expected_meas[:, 0] - expected_prev)
        torch.testing.assert_close(delta_meas[:, 1:], torch.zeros((2, 2, 4), dtype=torch.float32))

    def test_helper_raises_on_shape_mismatch(self) -> None:
        model = _build_model(contact_dim=2)
        with self.assertRaisesRegex(ValueError, "contacts batch mismatch"):
            model._canonicalize_contacts_meas_inputs(
                torch.zeros((2, 3, 2), dtype=torch.float32),
                None,
                batch_size=3,
                seq_len=3,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_helper_swallow_only_conversion_failure(self) -> None:
        model = _build_model(contact_dim=2)
        contacts_meas, delta_meas, meas_prev_t = model._canonicalize_contacts_meas_inputs(
            object(),
            object(),
            batch_size=2,
            seq_len=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(contacts_meas, torch.zeros((2, 3, 2), dtype=torch.float32))
        torch.testing.assert_close(delta_meas, torch.zeros((2, 3, 2), dtype=torch.float32))
        self.assertIsNone(meas_prev_t)

    def test_forward_event_clock_uses_canonicalized_measurements(self) -> None:
        model = _build_model(contact_dim=2, use_event_clock=True)
        batch_size, steps = 2, 3
        state = torch.randn(batch_size, steps, model.in_state_dim, dtype=torch.float32)
        cond = torch.randn(batch_size, steps, model.cond_dim, dtype=torch.float32)
        contacts = torch.tensor([0.25, 0.75], dtype=torch.float32)
        meas_prev = torch.tensor([0.10, 0.50], dtype=torch.float32)

        with torch.no_grad():
            out = model(state, cond, contacts=contacts, meas_logits_prev=meas_prev)

        expected_meas = contacts.view(1, 1, 2).expand(batch_size, steps, 2)
        expected_delta = torch.zeros((batch_size, steps, 2), dtype=torch.float32)
        expected_delta[:, 0] = contacts.view(1, 2).expand(batch_size, 2) - meas_prev.view(1, 2).expand(batch_size, 2)

        self.assertIn("contacts_plan", out)
        self.assertIn("contacts_meas", out)
        self.assertIn("event_clock_delta_meas", out)
        torch.testing.assert_close(out["contacts_meas"], expected_meas)
        torch.testing.assert_close(out["event_clock_delta_meas"], expected_delta, atol=1e-6, rtol=1e-6)

    def test_forward_non_event_clock_uses_canonicalized_measurements(self) -> None:
        model = _build_model(contact_dim=2, use_event_clock=False)
        batch_size, steps = 2, 3
        state = torch.randn(batch_size, steps, model.in_state_dim, dtype=torch.float32)
        cond = torch.randn(batch_size, steps, model.cond_dim, dtype=torch.float32)
        contacts = torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        with torch.no_grad():
            out = model(state, cond, contacts=contacts)

        expected_meas = torch.tensor([0.2, 0.8], dtype=torch.float32).view(1, 1, 2).expand(batch_size, steps, 2)
        self.assertIn("contacts_plan", out)
        self.assertIn("contacts_meas", out)
        self.assertNotIn("event_clock_delta_meas", out)
        torch.testing.assert_close(out["contacts_meas"], expected_meas)


if __name__ == "__main__":
    unittest.main()
