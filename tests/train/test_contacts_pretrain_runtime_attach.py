from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from train.configuration.norm_spec import ContactPretrainRuntime
from train.runtime_attach import apply_contacts_pretrain_runtime, apply_loss_runtime_from_trainer
from train.training_MPL import Trainer


class ContactsPretrainRuntimeAttachTest(unittest.TestCase):
    def test_apply_contacts_pretrain_runtime_dual_writes_owner_and_neutral_attrs(self) -> None:
        trainer = SimpleNamespace()

        returned = apply_contacts_pretrain_runtime(
            trainer,
            owner_prefix="posttrain",
            runtime=ContactPretrainRuntime(
                clamp=1.25,
                affine_stats="stats.json",
                affine={"scale": [1.0, 1.0], "bias": [0.0, 0.0], "eps": 1e-6},
            ),
        )

        self.assertIs(returned, trainer)
        self.assertTrue(trainer.contacts_pretrain_runtime_attached)
        self.assertEqual(trainer.contacts_pretrain_clamp, 1.25)
        self.assertEqual(trainer.contacts_pretrain_affine_stats_spec, "stats.json")
        self.assertEqual(
            trainer.contacts_pretrain_affine,
            {"scale": [1.0, 1.0], "bias": [0.0, 0.0], "eps": 1e-6},
        )
        self.assertEqual(trainer.posttrain_contacts_pretrain_clamp, 1.25)
        self.assertEqual(trainer.posttrain_contacts_pretrain_affine_stats_spec, "stats.json")
        self.assertEqual(
            trainer.posttrain_contacts_pretrain_affine,
            {"scale": [1.0, 1.0], "bias": [0.0, 0.0], "eps": 1e-6},
        )

    def test_predict_pretrain_contacts_loud_fails_on_partial_neutral_attach(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.model = SimpleNamespace(
            frozen_encoder=object(),
            frozen_contact_head=object(),
            contact_dim=2,
            encoder_input_dim=4,
            _contact_meas_state_angvel_slice=None,
        )
        trainer.contacts_pretrain_runtime_attached = True
        trainer.contacts_pretrain_clamp = 1.0
        trainer.contacts_pretrain_affine_stats_spec = None

        with self.assertRaisesRegex(RuntimeError, "missing neutral attrs: contacts_pretrain_affine"):
            trainer._predict_pretrain_contacts_from_frozen(
                motion_step_t=torch.zeros(1, 4, dtype=torch.float32),
                pose_hist_step_t=None,
            )

    def test_predict_pretrain_contacts_keeps_inactive_path_non_raising(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.model = None
        trainer.contacts_pretrain_runtime_attached = False

        output = trainer._predict_pretrain_contacts_from_frozen(
            motion_step_t=torch.zeros(1, 4, dtype=torch.float32),
            pose_hist_step_t=None,
        )

        self.assertIsNone(output)

    def test_apply_loss_runtime_from_trainer_syncs_stats_and_optional_meta(self) -> None:
        trainer = SimpleNamespace(
            mu_y=[1.0, 2.0],
            std_y=[3.0, 4.0],
            _bundle_meta={"fps": 60},
        )
        loss_fn = SimpleNamespace()

        returned = apply_loss_runtime_from_trainer(loss_fn, trainer, copy_bundle_meta=True)

        self.assertIs(returned, loss_fn)
        self.assertEqual(loss_fn.mu_y, [1.0, 2.0])
        self.assertEqual(loss_fn.std_y, [3.0, 4.0])
        self.assertEqual(loss_fn.meta, {"fps": 60})

    def test_apply_loss_runtime_from_trainer_allows_missing_stats_pair(self) -> None:
        trainer = SimpleNamespace()
        loss_fn = SimpleNamespace()

        apply_loss_runtime_from_trainer(loss_fn, trainer)

        self.assertIsNone(loss_fn.mu_y)
        self.assertIsNone(loss_fn.std_y)

    def test_apply_loss_runtime_from_trainer_rejects_partial_stats_pair(self) -> None:
        trainer = SimpleNamespace(mu_y=[1.0], std_y=None)
        loss_fn = SimpleNamespace()

        with self.assertRaisesRegex(RuntimeError, "partial normalization stats"):
            apply_loss_runtime_from_trainer(loss_fn, trainer)

    def test_apply_loss_runtime_from_trainer_ignores_empty_bundle_meta(self) -> None:
        trainer = SimpleNamespace(mu_y=None, std_y=None, _bundle_meta={})
        loss_fn = SimpleNamespace()

        apply_loss_runtime_from_trainer(loss_fn, trainer, copy_bundle_meta=True)

        self.assertFalse(hasattr(loss_fn, "meta"))

    def test_apply_loss_runtime_from_trainer_swallows_meta_copy_error_without_warn(self) -> None:
        trainer = SimpleNamespace(mu_y=None, std_y=None, _bundle_meta=object())
        loss_fn = SimpleNamespace()

        returned = apply_loss_runtime_from_trainer(loss_fn, trainer, copy_bundle_meta=True)

        self.assertIs(returned, loss_fn)
        self.assertFalse(hasattr(loss_fn, "meta"))


if __name__ == "__main__":
    unittest.main()
