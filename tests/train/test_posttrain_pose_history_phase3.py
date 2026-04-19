from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
import unittest

import torch
import torch.nn as nn

from train.history import PoseHistState
from train import rollout_kernel as _rollout_kernel
from train.posttrain import (
    LambdaFusionAccum,
    LambdaFusionFinalizeContext,
    _lambda_fusion_finalize,
    _lambda_rollout_prepare_context,
    _rollout_step_common,
)
from train.training_MPL import Trainer


class _NormalizerStub:
    def __init__(self) -> None:
        self.std_y = [1.0] * 6
        self.y_to_x_map = []

    def denorm_x(self, x: torch.Tensor, prev_raw: torch.Tensor | None = None) -> torch.Tensor:
        return x + 10.0

    def x_to_y(self, x: torch.Tensor, dy: int) -> torch.Tensor:
        return x[..., :dy]


class _ModelStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.last_pose_history: torch.Tensor | None = None

    def forward(self, motion, cond, **kwargs):
        self.last_pose_history = kwargs.get("pose_history")
        return {"out": motion.new_zeros(motion.shape[0], motion.shape[1], 6)}


def _make_trainer() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.pose_hist_len = 2
    trainer.pose_hist_dim = 6
    trainer.pose_hist_scales = torch.linspace(1.1, 1.6, steps=6, dtype=torch.float32)
    trainer.pose_hist_mu = torch.linspace(-0.2, 0.2, steps=6, dtype=torch.float32)
    trainer.pose_hist_std = torch.linspace(0.7, 1.2, steps=6, dtype=torch.float32)
    trainer.rot6d_y_slice = slice(0, 6)
    trainer.rot6d_slice = slice(0, 6)
    trainer.use_freerun_state_sync = False
    trainer.normalizer = _NormalizerStub()
    trainer._prepare_cond_stat = lambda cond, ref: cond
    return trainer


class PosttrainPoseHistoryPhase3Test(unittest.TestCase):
    def test_lambda_rollout_prepare_context_uses_shared_pose_hist_state_with_offset(self) -> None:
        trainer = _make_trainer()
        model = _ModelStub()
        batch = {
            "motion": torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]],
                dtype=torch.float32,
            ),
            "gt_motion": torch.tensor(
                [[[11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [12.0, 13.0, 14.0, 15.0, 16.0, 17.0], [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]]],
                dtype=torch.float32,
            ),
            "pose_hist": torch.tensor(
                [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]],
                dtype=torch.float32,
            ),
        }

        with mock.patch("train.posttrain.torch.randint", return_value=torch.tensor([1], dtype=torch.int64)):
            prep = _lambda_rollout_prepare_context(
                trainer,
                model,
                batch,
                rollout_steps=2,
                rollout_cycles=2,
                include_boundary=False,
                boundary_weight=1.0,
                random_offset=True,
                time_weight_mode="inv",
                time_weight_max=1.0,
            )

        self.assertEqual(prep.offset, 1)
        pose_hist_state = prep.state["pose_hist_state"]
        self.assertIsInstance(pose_hist_state, PoseHistState)
        self.assertTrue(pose_hist_state.enabled)
        torch.testing.assert_close(pose_hist_state.buffer_norm, batch["pose_hist"][:, 1])

    def test_rollout_step_common_prefers_pose_hist_buffer_then_seq(self) -> None:
        trainer = _make_trainer()
        model = _ModelStub()
        pose_hist_seq = torch.tensor(
            [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]],
            dtype=torch.float32,
        )
        state = {
            "motion": torch.zeros(1, 6, dtype=torch.float32),
            "motion_raw": torch.zeros(1, 6, dtype=torch.float32),
            "y_prev_raw": torch.zeros(1, 6, dtype=torch.float32),
            "pose_hist_state": PoseHistState(
                enabled=True,
                length=2,
                dim=6,
                stride=3,
                buffer_norm=torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]], dtype=torch.float32),
            ),
            "prev_foot_pos_meas": None,
        }

        with (
            mock.patch("train.posttrain._prepare_rollout_cond", return_value=(torch.ones(1, 1), None)),
            mock.patch("train.posttrain._rollout_kernel.prepare_rollout_contacts_input", return_value=None),
            mock.patch("train.posttrain._rollout_kernel.update_rollout_recurrent_state", return_value=None),
        ):
            _rollout_step_common(
                trainer,
                model,
                state=state,
                t=0,
                idx=1,
                total_steps=2,
                cond_seq=None,
                cond_raw_tgt=None,
                cond_norm_mu=None,
                cond_norm_std=None,
                angvel_seq=None,
                pose_hist_seq=pose_hist_seq,
                time_index_mode="none",
                time_base=None,
                enable_reprojection=False,
            )
            torch.testing.assert_close(model.last_pose_history, state["pose_hist_state"].buffer_norm.unsqueeze(1))

            state["pose_hist_state"] = PoseHistState(enabled=False, length=2, dim=6, stride=3)
            _rollout_step_common(
                trainer,
                model,
                state=state,
                t=0,
                idx=1,
                total_steps=2,
                cond_seq=None,
                cond_raw_tgt=None,
                cond_norm_mu=None,
                cond_norm_std=None,
                angvel_seq=None,
                pose_hist_seq=pose_hist_seq,
                time_index_mode="none",
                time_base=None,
                enable_reprojection=False,
            )
            torch.testing.assert_close(model.last_pose_history, pose_hist_seq[:, 1:2])

    def test_apply_rollout_carry_state_advances_shared_pose_hist_state(self) -> None:
        trainer = _make_trainer()
        trainer._apply_free_carry = lambda motion_raw, y_next_raw, cond_next_raw=None: y_next_raw + 1.0
        trainer._diag_norm_x = lambda x_raw: x_raw + 2.0

        state = {
            "motion_raw": torch.zeros(1, 6, dtype=torch.float32),
            "motion": torch.zeros(1, 6, dtype=torch.float32),
            "y_prev_raw": torch.zeros(1, 6, dtype=torch.float32),
            "rot_slice": slice(0, 3),
            "pose_hist_state": PoseHistState(
                enabled=True,
                length=2,
                dim=6,
                stride=3,
                scales=trainer.pose_hist_scales,
                mu=trainer.pose_hist_mu,
                std=trainer.pose_hist_std,
                buffer_raw=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32),
            ),
        }
        y_next_raw = torch.tensor([[30.0, 31.0, 32.0, 90.0, 91.0, 92.0]], dtype=torch.float32)

        _rollout_kernel.apply_rollout_carry_state(
            trainer,
            state,
            y_next_raw=y_next_raw,
            cond_raw_step=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        )

        self.assertIsInstance(state["pose_hist_state"], PoseHistState)
        torch.testing.assert_close(state["motion_raw"], y_next_raw + 1.0)
        torch.testing.assert_close(state["motion"], y_next_raw + 3.0)
        torch.testing.assert_close(
            state["pose_hist_state"].buffer_raw,
            torch.tensor([[4.0, 5.0, 6.0, 30.0, 31.0, 32.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(state["y_prev_raw"], y_next_raw)

    def test_lambda_fusion_finalize_preserves_direct_contract_shape(self) -> None:
        trainer = _make_trainer()
        model = SimpleNamespace(direct_pose_leg_joint_names=["foot_l", "foot_r"])
        finalize_ctx = LambdaFusionFinalizeContext(
            trainer=trainer,
            model=model,
            objective="direct",
            direct_pose_leg_align_weight=0.5,
            direct_pose_leg_align_anchor_weight=0.25,
            direct_nonleg_focus_requested=1,
            direct_nonleg_focus_resolved=1,
            direct_nonleg_focus_weight_use=1.5,
            direct_nonleg_focus_applied=1.0,
        )
        accum = LambdaFusionAccum(
            loss_terms=[torch.tensor(1.0, dtype=torch.float32)],
            inc_terms=[torch.tensor(2.0, dtype=torch.float32)],
            dir_terms=[torch.tensor(3.0, dtype=torch.float32)],
            dir_base_terms=[torch.tensor(3.1, dtype=torch.float32)],
            dir_leg_base_terms=[torch.tensor(1.2, dtype=torch.float32)],
            dir_nonleg_base_terms=[torch.tensor(1.8, dtype=torch.float32)],
            dir_nonleg_plain_terms=[torch.tensor(1.7, dtype=torch.float32)],
            leg_align_terms=[torch.tensor(0.4, dtype=torch.float32)],
            leg_align_frac_terms=[torch.tensor(0.6, dtype=torch.float32)],
            leg_align_distal_terms=[torch.tensor(0.1, dtype=torch.float32)],
            leg_align_distal_frac_terms=[torch.tensor(0.2, dtype=torch.float32)],
            leg_align_proximal_terms=[torch.tensor(0.3, dtype=torch.float32)],
            leg_align_proximal_frac_terms=[torch.tensor(0.4, dtype=torch.float32)],
            lam_vals=[torch.tensor([[0.2, 0.8]], dtype=torch.float32)],
            lam_eff_vals=[torch.tensor([[0.1, 0.7]], dtype=torch.float32)],
            lam_rel_vals=[torch.tensor([[0.9, 0.95]], dtype=torch.float32)],
        )

        total, stats, aux_payload = _lambda_fusion_finalize(finalize_ctx=finalize_ctx, accum_ctx=accum)

        self.assertTrue(torch.is_tensor(total))
        self.assertTrue(
            {
                "blend_loss",
                "dir_geo",
                "dir_base",
                "dir_leg_base",
                "dir_nonleg_base",
                "dir_group_norm_used",
                "leg_align_loss",
                "lambda_mean",
                "lambda_eff_mean",
                "lambda_rel_mean",
                "total",
            }.issubset(stats.keys())
        )
        self.assertIsInstance(aux_payload, dict)
        self.assertIn("ema_update_payload", aux_payload)
        self.assertIn("leg_align_grad_probe", aux_payload)
        self.assertEqual(set(aux_payload["leg_align_grad_probe"].keys()), {"total", "distal", "proximal"})
        self.assertTrue(all(torch.is_tensor(value) for value in aux_payload["leg_align_grad_probe"].values()))


if __name__ == "__main__":
    unittest.main()
