from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
import unittest

from train import posttrain
from train.configuration.norm_spec import ContactPretrainRuntime


def _make_cfg(**overrides):
    values = {
        "contact_meas_gate_by_hit": "auto",
        "contact_meas_vxy_mode": "ABS",
        "contact_meas_ground_z_mode": "WINDOW",
        "contact_meas_ground_z_beta": 0.07,
        "contact_meas_ground_z_window": 9,
        "contact_meas_ground_z_quantile": 0.25,
        "contact_meas_ground_z_slew_up_cm": 12.5,
        "contact_meas_ground_z_slew_down_cm": -3.0,
        "posttrain_contacts_pretrain_clamp": 1.5,
        "posttrain_contacts_pretrain_affine_stats": None,
        "lambda_reliability_mode": "warmup",
        "lambda_reliability_warmup_steps": 12,
        "lambda_reliability_contact_err_max": 0.75,
        "lambda_reliability_warmup_joint_scales": {"foot_l": 0.5},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PosttrainLocalRuntimeOverlayTest(unittest.TestCase):
    def test_contact_meas_gate_override_aliases(self) -> None:
        for raw_value in ("true", "1", "yes", "y"):
            self.assertIs(posttrain._resolve_posttrain_contact_meas_gate_override(raw_value), True)
        for raw_value in ("false", "0", "no", "n"):
            self.assertIs(posttrain._resolve_posttrain_contact_meas_gate_override(raw_value), False)
        for raw_value in ("auto", "", None, "unexpected"):
            self.assertIsNone(posttrain._resolve_posttrain_contact_meas_gate_override(raw_value))

    def test_ground_z_slew_limits_keep_pair_fallback_semantics(self) -> None:
        self.assertEqual(
            posttrain._resolve_posttrain_ground_z_slew_limits_m(
                _make_cfg(contact_meas_ground_z_slew_up_cm=12.5, contact_meas_ground_z_slew_down_cm=-3.0)
            ),
            (0.125, 0.0),
        )
        self.assertEqual(
            posttrain._resolve_posttrain_ground_z_slew_limits_m(
                _make_cfg(contact_meas_ground_z_slew_up_cm="bad", contact_meas_ground_z_slew_down_cm=9.0)
            ),
            (0.0, 0.0),
        )

    def test_resolve_posttrain_local_runtime_overlay_normalizes_contract(self) -> None:
        overlay = posttrain._resolve_posttrain_local_runtime_overlay(
            _make_cfg(contact_meas_gate_by_hit="YES")
        )

        self.assertIs(overlay.contact_meas_gate_by_hit_override, True)
        self.assertEqual(overlay.contact_meas_vxy_mode, "abs")
        self.assertEqual(overlay.contact_meas_ground_z_mode, "window")
        self.assertAlmostEqual(overlay.contact_meas_ground_z_max_up_m, 0.125)
        self.assertAlmostEqual(overlay.contact_meas_ground_z_max_down_m, 0.0)
        self.assertEqual(overlay.contacts_pretrain.clamp, 1.5)
        self.assertEqual(overlay.lambda_reliability_mode, "warmup")
        self.assertEqual(overlay.lambda_reliability_warmup_steps, 12)
        self.assertEqual(overlay.lambda_reliability_contact_err_max, 0.75)
        self.assertEqual(overlay.lambda_reliability_warmup_joint_scales, {"foot_l": 0.5})

    def test_apply_posttrain_local_runtime_overlay_maps_local_and_contact_fields(self) -> None:
        trainer = SimpleNamespace()
        overlay = posttrain.PosttrainLocalRuntimeOverlay(
            contact_meas_gate_by_hit_override=False,
            contact_meas_vxy_mode="signed",
            contact_meas_ground_z_mode="ema",
            contact_meas_ground_z_beta=0.1,
            contact_meas_ground_z_window=7,
            contact_meas_ground_z_quantile=0.3,
            contact_meas_ground_z_max_up_m=0.04,
            contact_meas_ground_z_max_down_m=0.05,
            contacts_pretrain=ContactPretrainRuntime(clamp=1.25, affine_stats=None, affine=None),
            lambda_reliability_mode="warmup",
            lambda_reliability_warmup_steps=30,
            lambda_reliability_contact_err_max=0.8,
            lambda_reliability_warmup_joint_scales={"foot_r": 0.6},
        )

        posttrain._apply_posttrain_local_runtime_overlay(trainer, overlay)

        self.assertEqual(trainer.posttrain_contacts_pretrain_clamp, 1.25)
        self.assertTrue(trainer.contacts_pretrain_runtime_attached)
        self.assertIs(trainer.contact_meas_gate_by_hit_override, False)
        self.assertEqual(trainer.contact_meas_vxy_mode, "signed")
        self.assertEqual(trainer.contact_meas_ground_z_mode, "ema")
        self.assertEqual(trainer.contact_meas_ground_z_beta, 0.1)
        self.assertEqual(trainer.contact_meas_ground_z_window, 7)
        self.assertEqual(trainer.contact_meas_ground_z_quantile, 0.3)
        self.assertEqual(trainer.contact_meas_ground_z_max_up_m, 0.04)
        self.assertEqual(trainer.contact_meas_ground_z_max_down_m, 0.05)
        self.assertEqual(trainer.lambda_reliability_mode, "warmup")
        self.assertEqual(trainer.lambda_reliability_warmup_steps, 30)
        self.assertEqual(trainer.lambda_reliability_contact_err_max, 0.8)
        self.assertEqual(trainer.lambda_reliability_warmup_joint_scales, {"foot_r": 0.6})

    def test_apply_posttrain_overlay_applies_all_dataclass_fields_except_contact_runtime(self) -> None:
        trainer = SimpleNamespace()
        overlay = posttrain.PosttrainLocalRuntimeOverlay(
            contact_meas_gate_by_hit_override=True,
            contact_meas_vxy_mode="abs",
            contact_meas_ground_z_mode="window",
            contact_meas_ground_z_beta=0.2,
            contact_meas_ground_z_window=11,
            contact_meas_ground_z_quantile=0.4,
            contact_meas_ground_z_max_up_m=0.06,
            contact_meas_ground_z_max_down_m=0.07,
            contacts_pretrain=ContactPretrainRuntime(clamp=1.0, affine_stats=None, affine=None),
            lambda_reliability_mode="none",
            lambda_reliability_warmup_steps=0,
            lambda_reliability_contact_err_max=1.0,
            lambda_reliability_warmup_joint_scales=None,
        )

        posttrain._apply_posttrain_local_runtime_overlay(trainer, overlay)

        field_names = {field.name for field in fields(overlay)}
        self.assertIn("contacts_pretrain", field_names)
        for field_name in sorted(field_names - {"contacts_pretrain"}):
            self.assertTrue(hasattr(trainer, field_name), field_name)
            self.assertEqual(getattr(trainer, field_name), getattr(overlay, field_name), field_name)
        self.assertFalse(hasattr(trainer, "contacts_pretrain"))
        self.assertTrue(trainer.contacts_pretrain_runtime_attached)


if __name__ == "__main__":
    unittest.main()
