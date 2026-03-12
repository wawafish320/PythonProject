from __future__ import annotations

import unittest

import torch

from train.history import PoseHistState, pose_hist_inverse_vec
from train.validate.run_freerun_cycles import (
    _init_eval_pose_hist_state,
    _resolve_eval_pose_hist_input,
)


class _TrainerStub:
    def __init__(self) -> None:
        self.pose_hist_len = 2
        self.pose_hist_dim = 6

    def _pose_hist_params(
        self,
        ref_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = torch.linspace(1.1, 1.6, steps=6, device=ref_tensor.device, dtype=ref_tensor.dtype)
        mu = torch.linspace(-0.2, 0.2, steps=6, device=ref_tensor.device, dtype=ref_tensor.dtype)
        std = torch.linspace(0.7, 1.2, steps=6, device=ref_tensor.device, dtype=ref_tensor.dtype)
        return scales, mu, std


class FreeRunCyclesPoseHistoryPhase4Test(unittest.TestCase):
    def test_init_eval_pose_hist_state_uses_step_specific_seq(self) -> None:
        trainer = _TrainerStub()
        ref = torch.zeros(1, 4, dtype=torch.float32)
        pose_hist_seq = torch.tensor(
            [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]],
            dtype=torch.float32,
        )

        state = _init_eval_pose_hist_state(
            trainer,
            ref_tensor=ref,
            pose_hist_seq=pose_hist_seq,
            step=1,
            device=torch.device("cpu"),
            dtype=ref.dtype,
        )

        self.assertTrue(state.enabled)
        torch.testing.assert_close(state.buffer_norm, pose_hist_seq[:, 1])

    def test_init_eval_pose_hist_state_zero_fallback_keeps_shared_contract(self) -> None:
        trainer = _TrainerStub()
        ref = torch.zeros(1, 4, dtype=torch.float32)

        state = _init_eval_pose_hist_state(
            trainer,
            ref_tensor=ref,
            pose_hist_seq=None,
            step=3,
            device=torch.device("cpu"),
            dtype=ref.dtype,
        )

        expected_norm = torch.zeros((1, 6), dtype=ref.dtype)
        expected_raw = pose_hist_inverse_vec(expected_norm, state.scales, state.mu, state.std)

        self.assertTrue(state.enabled)
        torch.testing.assert_close(state.buffer_norm, expected_norm)
        torch.testing.assert_close(state.buffer_raw, expected_raw)

    def test_resolve_eval_pose_hist_input_supports_buffer_seq_zero_modes(self) -> None:
        state = PoseHistState(
            enabled=True,
            length=2,
            dim=6,
            stride=3,
            buffer_norm=torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]], dtype=torch.float32),
        )
        seq = torch.tensor(
            [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            _resolve_eval_pose_hist_input(
                state=state,
                pose_hist_seq=seq,
                idx=1,
                source="buffer",
                batch_size=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            state.buffer_norm,
        )
        torch.testing.assert_close(
            _resolve_eval_pose_hist_input(
                state=state,
                pose_hist_seq=seq,
                idx=1,
                source="seq",
                batch_size=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            seq[:, 1],
        )
        torch.testing.assert_close(
            _resolve_eval_pose_hist_input(
                state=state,
                pose_hist_seq=seq,
                idx=1,
                source="zero",
                batch_size=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            torch.zeros((1, 6), dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
