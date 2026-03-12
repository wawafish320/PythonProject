from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from train.history import PoseHistState, init_pose_hist_state
from train.training_MPL import Trainer


def _make_pose_hist_params(
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = 6
    scales = torch.linspace(1.1, 1.6, steps=dim, device=ref.device, dtype=ref.dtype)
    mu = torch.linspace(-0.2, 0.2, steps=dim, device=ref.device, dtype=ref.dtype)
    std = torch.linspace(0.7, 1.2, steps=dim, device=ref.device, dtype=ref.dtype)
    return scales, mu, std


def _make_trainer() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.pose_hist_len = 2
    trainer.pose_hist_dim = 6
    trainer.pose_hist_scales = torch.linspace(1.1, 1.6, steps=6, dtype=torch.float32)
    trainer.pose_hist_mu = torch.linspace(-0.2, 0.2, steps=6, dtype=torch.float32)
    trainer.pose_hist_std = torch.linspace(0.7, 1.2, steps=6, dtype=torch.float32)
    trainer.force_pose_hist_seq = False
    trainer.use_freerun_state_sync = False
    trainer.device = torch.device("cpu")
    return trainer


class _NormalizerStub:
    def denorm_x(self, x: torch.Tensor, prev_raw: torch.Tensor | None = None) -> torch.Tensor:
        return x + 20.0


class TrainingMPLPoseHistoryPhase2Test(unittest.TestCase):
    def test_prepare_pose_hist_state_matches_shared_helper(self) -> None:
        trainer = _make_trainer()
        state_seq = torch.zeros(2, 4, dtype=torch.float32)
        pose_hist_seq = torch.tensor(
            [
                [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]],
                [[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            ],
            dtype=torch.float32,
        )

        state = trainer._prepare_pose_hist_state(
            state_seq=state_seq,
            pose_hist_seq=pose_hist_seq,
            y_raw_local=None,
            rot6d_y_slice=slice(0, 3),
        )
        expected = init_pose_hist_state(
            ref_tensor=state_seq,
            pose_hist_seq=pose_hist_seq,
            y_prev_raw=None,
            rot_slice=slice(0, 3),
            pose_hist_len=2,
            pose_hist_dim=6,
            params_fn=trainer._pose_hist_params,
        )

        self.assertIsInstance(state, PoseHistState)
        self.assertTrue(state.enabled)
        torch.testing.assert_close(state.buffer_norm, expected.buffer_norm)
        torch.testing.assert_close(state.buffer_raw, expected.buffer_raw)

    def test_resolve_rollout_step_inputs_uses_shared_pose_hist_resolution(self) -> None:
        trainer = _make_trainer()
        trainer._normalize_cond_from_raw = lambda cond_raw, mu, std: None

        cond_seq = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
        angvel_seq = torch.tensor([[[3.0], [4.0]]], dtype=torch.float32)
        pose_hist_seq = torch.tensor(
            [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]],
            dtype=torch.float32,
        )
        buffer = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]], dtype=torch.float32)
        context = SimpleNamespace(
            step_idx=1,
            total_steps=2,
            motion=torch.zeros(1, 4, dtype=torch.float32),
            motion_raw_local=None,
            y_raw_local=None,
            state_seq=torch.zeros(1, 2, 4, dtype=torch.float32),
            gt_seq=None,
            cond_seq=cond_seq,
            cond_raw_seq=None,
            contacts_seq=None,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            cond_norm_mu=None,
            cond_norm_std=None,
            has_time_dim={
                "cond": True,
                "cond_raw": False,
                "contacts": False,
                "angvel": True,
                "pose_hist": True,
            },
            pose_hist_state=PoseHistState(enabled=True, length=2, dim=6, stride=3, buffer_norm=buffer),
            plan_enable=False,
            mode="teacher",
            enable_reprojection=False,
            time_base_local=None,
        )

        step_inputs = trainer._resolve_rollout_step_inputs(context)
        torch.testing.assert_close(step_inputs.pose_history_t, buffer)

        context.pose_hist_state = PoseHistState(enabled=False, length=2, dim=6, stride=3)
        step_inputs = trainer._resolve_rollout_step_inputs(context)
        torch.testing.assert_close(step_inputs.pose_history_t, pose_hist_seq[:, 1])

    def test_update_rollout_carry_state_advances_pose_hist_from_carry_raw(self) -> None:
        trainer = _make_trainer()
        trainer.normalizer = _NormalizerStub()
        trainer._raise_norm_error = lambda context, exc=None: (_ for _ in ()).throw(RuntimeError(context))
        trainer._apply_free_carry = lambda motion_raw_local, y_raw, cond_next_raw=None: y_raw + 1.0
        trainer._diag_norm_x = lambda x_raw: x_raw + 10.0
        trainer._denorm = lambda y: y + 30.0

        request = SimpleNamespace(
            step_idx=0,
            total_steps=2,
            batch_size=1,
            tf_ratio=1.0,
            state_seq=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]]], dtype=torch.float32),
            gt_seq=torch.tensor([[[11.0, 12.0, 13.0, 14.0], [3.0, 4.0, 5.0, 6.0]]], dtype=torch.float32),
            motion_raw_local=torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32),
            y_raw=torch.tensor([[50.0, 60.0, 70.0, 80.0]], dtype=torch.float32),
            y_raw_local=torch.tensor([[15.0, 16.0, 17.0, 18.0]], dtype=torch.float32),
            allow_grad=False,
            cond_raw_for_env=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            ss_chunk_len=2,
            ss_sel_hold=torch.tensor([[1.0]], dtype=torch.float32),
            pose_hist_state=PoseHistState(
                enabled=True,
                length=2,
                dim=6,
                stride=3,
                scales=trainer.pose_hist_scales,
                mu=trainer.pose_hist_mu,
                std=trainer.pose_hist_std,
                buffer_raw=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32),
            ),
            rot6d_y_slice=slice(0, 3),
        )
        request.pose_hist_state.buffer_norm = init_pose_hist_state(
            ref_tensor=torch.zeros(1, 4, dtype=torch.float32),
            pose_hist_seq=None,
            y_prev_raw=torch.tensor([[1.0, 2.0, 3.0, 0.0]], dtype=torch.float32),
            rot_slice=slice(0, 3),
            pose_hist_len=2,
            pose_hist_dim=6,
            params_fn=lambda ref: _make_pose_hist_params(ref),
        ).buffer_norm

        carry_state = trainer._update_rollout_carry_state(request)

        expected_y_raw_local = torch.tensor([[33.0, 34.0, 35.0, 36.0]], dtype=torch.float32)
        expected_buffer_raw = torch.tensor([[4.0, 5.0, 6.0, 33.0, 34.0, 35.0]], dtype=torch.float32)

        torch.testing.assert_close(carry_state.y_raw_local, expected_y_raw_local)
        torch.testing.assert_close(carry_state.pose_hist_state.buffer_raw, expected_buffer_raw)


if __name__ == "__main__":
    unittest.main()
