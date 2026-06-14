from __future__ import annotations

import unittest

import torch

from train.validate.injection_contracts import InjectionPayload, InjectionTensorSpec
from train.validate.run_freerun_cycles import _apply_injection_payload_hook


class FreeRunCyclesInjectionHookTest(unittest.TestCase):
    def test_no_payload_keeps_tensor_unchanged(self) -> None:
        y = torch.arange(12, dtype=torch.float32).view(1, 12)
        out, record = _apply_injection_payload_hook(
            y_used_raw=y,
            payload=None,
            requested_fields=None,
            step=5,
            step_in_cycle=1,
            rot6d_y_slice=slice(2, 8),
            rootvel_y_slice=slice(0, 2),
            angvel_y_slice=slice(8, 12),
        )
        self.assertIs(out, y)
        self.assertIsNone(record)
        torch.testing.assert_close(out, y)

    def test_payload_overrides_rootvel_rot6d_angvel_and_records_metadata(self) -> None:
        y = torch.zeros((1, 12), dtype=torch.float32)
        payload = InjectionPayload(
            rot6d_raw=torch.full((1, 6), 2.0, dtype=torch.float32),
            rootvel_raw=torch.full((1, 2), 1.0, dtype=torch.float32),
            angvel_raw=torch.full((1, 4), 3.0, dtype=torch.float32),
            source_frame_index=0,
            spec=InjectionTensorSpec(
                rot6d_shape=(1, 6),
                rootvel_shape=(1, 2),
                angvel_shape=(1, 4),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )

        out, record = _apply_injection_payload_hook(
            y_used_raw=y,
            payload=payload,
            requested_fields={"rootvel", "rot6d", "angvel"},
            step=40,
            step_in_cycle=7,
            rot6d_y_slice=slice(2, 8),
            rootvel_y_slice=slice(0, 2),
            angvel_y_slice=slice(8, 12),
        )
        self.assertIsNot(out, y)
        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
        torch.testing.assert_close(out, expected)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.step, 40)
        self.assertEqual(record.step_in_cycle, 7)
        self.assertEqual(len(record.fields_applied), 3)

        by_field = {str(item["field"]): item for item in record.fields_applied}
        self.assertSetEqual(set(by_field.keys()), {"rootvel", "rot6d", "angvel"})
        self.assertTrue(bool(by_field["rootvel"]["requested"]))
        self.assertTrue(bool(by_field["rot6d"]["requested"]))
        self.assertTrue(bool(by_field["angvel"]["requested"]))
        self.assertTrue(bool(by_field["rootvel"]["applied"]))
        self.assertTrue(bool(by_field["rot6d"]["applied"]))
        self.assertTrue(bool(by_field["angvel"]["applied"]))
        self.assertEqual(by_field["rootvel"]["reason"], "applied")
        self.assertEqual(by_field["rot6d"]["reason"], "applied")
        self.assertEqual(by_field["angvel"]["reason"], "applied")
        self.assertEqual(by_field["rootvel"]["target_slice"], {"start": 0, "stop": 2})
        self.assertEqual(by_field["rot6d"]["target_slice"], {"start": 2, "stop": 8})
        self.assertEqual(by_field["angvel"]["target_slice"], {"start": 8, "stop": 12})
        self.assertEqual(by_field["rootvel"]["payload_shape"], [1, 2])
        self.assertEqual(by_field["rot6d"]["payload_shape"], [1, 6])
        self.assertEqual(by_field["angvel"]["payload_shape"], [1, 4])
        self.assertEqual(by_field["rootvel"]["slice_start"], 0)
        self.assertEqual(by_field["rootvel"]["slice_stop"], 2)
        self.assertEqual(by_field["rot6d"]["slice_start"], 2)
        self.assertEqual(by_field["rot6d"]["slice_stop"], 8)
        self.assertEqual(by_field["angvel"]["slice_start"], 8)
        self.assertEqual(by_field["angvel"]["slice_stop"], 12)
        self.assertEqual(by_field["rootvel"]["tensor_shape"], [1, 2])
        self.assertEqual(by_field["rot6d"]["tensor_shape"], [1, 6])
        self.assertEqual(by_field["angvel"]["tensor_shape"], [1, 4])
        self.assertEqual(by_field["rootvel"]["payload_dtype"], "float32")
        self.assertEqual(by_field["rot6d"]["payload_dtype"], "float32")
        self.assertEqual(by_field["angvel"]["payload_dtype"], "float32")
        self.assertEqual(by_field["rootvel"]["payload_device"], "cpu")
        self.assertEqual(by_field["rot6d"]["payload_device"], "cpu")
        self.assertEqual(by_field["angvel"]["payload_device"], "cpu")
        self.assertEqual(by_field["rootvel"]["target_dtype"], "float32")
        self.assertEqual(by_field["rootvel"]["target_device"], "cpu")

    def test_payload_requested_angvel_without_target_slice_marks_target_slice_missing(self) -> None:
        y = torch.zeros((1, 12), dtype=torch.float32)
        payload = InjectionPayload(
            rot6d_raw=torch.full((1, 6), 2.0, dtype=torch.float32),
            rootvel_raw=torch.full((1, 2), 1.0, dtype=torch.float32),
            angvel_raw=torch.full((1, 4), 3.0, dtype=torch.float32),
            source_frame_index=0,
            spec=InjectionTensorSpec(
                rot6d_shape=(1, 6),
                rootvel_shape=(1, 2),
                angvel_shape=(1, 4),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )
        out, record = _apply_injection_payload_hook(
            y_used_raw=y,
            payload=payload,
            requested_fields={"rootvel", "rot6d", "angvel"},
            step=40,
            step_in_cycle=7,
            rot6d_y_slice=slice(2, 8),
            rootvel_y_slice=slice(0, 2),
            angvel_y_slice=None,
        )
        self.assertIsNotNone(record)
        assert record is not None
        by_field = {str(item["field"]): item for item in record.fields_applied}
        self.assertTrue(bool(by_field["rootvel"]["requested"]))
        self.assertTrue(bool(by_field["rootvel"]["applied"]))
        self.assertEqual(by_field["rootvel"]["reason"], "applied")
        self.assertTrue(bool(by_field["rot6d"]["requested"]))
        self.assertTrue(bool(by_field["rot6d"]["applied"]))
        self.assertEqual(by_field["rot6d"]["reason"], "applied")
        self.assertTrue(bool(by_field["angvel"]["requested"]))
        self.assertFalse(bool(by_field["angvel"]["applied"]))
        self.assertEqual(by_field["angvel"]["reason"], "target_slice_missing")
        self.assertIsNone(by_field["angvel"]["target_slice"])
        self.assertEqual(by_field["angvel"]["payload_shape"], [1, 4])
        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
