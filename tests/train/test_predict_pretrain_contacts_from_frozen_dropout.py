from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.training_MPL import Trainer
from train.utils import _build_pretrain_contact_encoder_input


class _RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, return_summary: bool = False) -> torch.Tensor:
        _ = return_summary
        self.last_input = x.detach().clone()
        return x + 1.0


class _RecordingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        if x.dim() == 3 and int(x.size(1)) == 1:
            return x[:, 0]
        return x


def _make_trainer(*, mode: str, prob: float) -> Trainer:
    encoder = _RecordingEncoder()
    head = _RecordingHead()
    trainer = Trainer.__new__(Trainer)
    trainer.model = SimpleNamespace(
        frozen_encoder=encoder,
        frozen_contact_head=head,
        contact_dim=2,
        encoder_input_dim=6,
        _contact_meas_state_angvel_slice=None,
    )
    trainer.angvel_x_slice = slice(0, 2)
    trainer.contacts_pretrain_runtime_attached = True
    trainer.contacts_pretrain_clamp = 0.0
    trainer.contacts_pretrain_affine_stats_spec = None
    trainer.contacts_pretrain_affine = None
    trainer.contacts_pretrain_dropout_injection_mode = mode
    trainer.contacts_pretrain_dropout_prob = prob
    return trainer


class PredictPretrainContactsFromFrozenDropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.motion = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        self.pose_hist = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)

    def _expected_encoder_input(self) -> torch.Tensor:
        return _build_pretrain_contact_encoder_input(
            self.motion,
            self.pose_hist,
            contact_dim=2,
            encoder_input_dim=6,
            angvel_slice=slice(0, 2),
            clamp_val=0.0,
        )

    def test_inject_dropout_false_keeps_route_deterministic(self) -> None:
        trainer = _make_trainer(mode="hidden", prob=0.7)
        output = trainer._predict_pretrain_contacts_from_frozen(
            motion_step_t=self.motion,
            pose_hist_step_t=self.pose_hist,
            inject_dropout=False,
        )

        encoder = trainer.model.frozen_encoder
        head = trainer.model.frozen_contact_head
        expected_input = self._expected_encoder_input().unsqueeze(1)
        self.assertTrue(torch.allclose(encoder.last_input, expected_input))
        self.assertTrue(torch.allclose(head.last_input, expected_input + 1.0))
        expected_probs = torch.sigmoid((expected_input + 1.0)[:, 0][..., :2])
        self.assertTrue(torch.allclose(output, expected_probs))

    def test_encoder_input_dropout_injects_before_encoder(self) -> None:
        trainer = _make_trainer(mode="encoder_input", prob=0.5)

        torch.manual_seed(1234)
        _ = trainer._predict_pretrain_contacts_from_frozen(
            motion_step_t=self.motion,
            pose_hist_step_t=self.pose_hist,
            inject_dropout=True,
        )

        expected_input = self._expected_encoder_input()
        torch.manual_seed(1234)
        expected_dropped = F.dropout(expected_input, p=0.5, training=True).unsqueeze(1)

        encoder = trainer.model.frozen_encoder
        head = trainer.model.frozen_contact_head
        self.assertTrue(torch.allclose(encoder.last_input, expected_dropped))
        self.assertTrue(torch.allclose(head.last_input, expected_dropped + 1.0))

    def test_hidden_dropout_injects_after_encoder_before_head(self) -> None:
        trainer = _make_trainer(mode="hidden", prob=0.5)

        torch.manual_seed(2026)
        _ = trainer._predict_pretrain_contacts_from_frozen(
            motion_step_t=self.motion,
            pose_hist_step_t=self.pose_hist,
            inject_dropout=True,
        )

        expected_input = self._expected_encoder_input().unsqueeze(1)
        expected_hidden = expected_input + 1.0
        torch.manual_seed(2026)
        expected_hidden_drop = F.dropout(expected_hidden, p=0.5, training=True)

        encoder = trainer.model.frozen_encoder
        head = trainer.model.frozen_contact_head
        self.assertTrue(torch.allclose(encoder.last_input, expected_input))
        self.assertTrue(torch.allclose(head.last_input, expected_hidden_drop))

    def test_invalid_dropout_mode_fails_fast(self) -> None:
        trainer = _make_trainer(mode="bad_mode", prob=0.0)

        with self.assertRaisesRegex(RuntimeError, "dropout injection mode"):
            trainer._predict_pretrain_contacts_from_frozen(
                motion_step_t=self.motion,
                pose_hist_step_t=self.pose_hist,
                inject_dropout=True,
            )

    def test_invalid_dropout_prob_fails_fast(self) -> None:
        trainer = _make_trainer(mode="off", prob=1.0)

        with self.assertRaisesRegex(RuntimeError, "dropout prob"):
            trainer._predict_pretrain_contacts_from_frozen(
                motion_step_t=self.motion,
                pose_hist_step_t=self.pose_hist,
                inject_dropout=True,
            )


if __name__ == "__main__":
    unittest.main()
