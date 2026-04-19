from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from train.posttrain import _lambda_fusion_loss_rollout
from train.training_MPL import Trainer


def _make_trainer() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    return trainer


def _make_prep_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        include_boundary=False,
        offset=0,
        J=2,
        boundary_steps=0,
        boundary_weighted_sum=0.0,
    )


def _make_nonleg_focus_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        direct_nonleg_focus_mask_j=torch.zeros(2, dtype=torch.float32),
        direct_nonleg_focus_requested=0,
        direct_nonleg_focus_resolved=0,
        direct_nonleg_focus_weight_use=1.0,
        direct_nonleg_focus_applied=0.0,
    )


def _make_reg_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        gate_sup_weight=0.0,
        gate_sup_start=0,
        tau_rad=0.0,
        margin_rad=0.0,
        direct_group_norm_enable=False,
        direct_group_w_leg=1.0,
        direct_group_w_nonleg=1.0,
        direct_group_beta=0.95,
        direct_group_ratio_min=0.2,
        direct_group_ratio_max=5.0,
        direct_group_eps=1e-6,
    )


class PosttrainLambdaFusionSmokeTest(unittest.TestCase):
    def test_lambda_fusion_loss_rollout_blend_smoke_pins_key_stats(self) -> None:
        trainer = _make_trainer()
        model = SimpleNamespace(direct_pose_leg_joint_names=["foot_l", "foot_r"])
        batch = {
            "motion": torch.zeros(1, 2, 6, dtype=torch.float32),
            "gt_motion": torch.zeros(1, 2, 6, dtype=torch.float32),
        }

        def _populate_unroll(*, runtime_ctx, weights_ctx, accum_ctx, state_vars):
            self.assertEqual(runtime_ctx["objective"], "blend")
            self.assertEqual(int(runtime_ctx["prep_ctx"].J), 2)
            accum_ctx.loss_terms.append(torch.tensor(1.5, dtype=torch.float32))
            accum_ctx.inc_terms.append(torch.tensor(2.0, dtype=torch.float32))
            accum_ctx.dir_terms.append(torch.tensor(3.0, dtype=torch.float32))
            accum_ctx.lam_vals.append(torch.tensor([[0.2, 0.8]], dtype=torch.float32))
            accum_ctx.lam_eff_vals.append(torch.tensor([[0.1, 0.7]], dtype=torch.float32))
            accum_ctx.lam_rel_vals.append(torch.tensor([[0.9, 0.95]], dtype=torch.float32))
            return False, float(state_vars.direct_nonleg_focus_applied)

        with (
            mock.patch("train.posttrain._lambda_rollout_prepare_context", return_value=_make_prep_ctx()),
            mock.patch("train.posttrain._lambda_rollout_resolve_nonleg_focus", return_value=_make_nonleg_focus_ctx()),
            mock.patch("train.posttrain._lambda_rollout_build_reg_params", return_value=_make_reg_ctx()),
            mock.patch("train.posttrain._lambda_fusion_run_unroll", side_effect=_populate_unroll),
        ):
            loss, stats, aux_payload = _lambda_fusion_loss_rollout(
                trainer,
                model,
                batch,
                columns=("motion", "gt_motion"),
                rollout_steps=2,
                rollout_cycles=1,
                include_boundary=False,
                boundary_weight=0.0,
                random_offset=False,
                time_index_mode="none",
                time_weight_max=1.0,
                time_weight_mode="uniform",
                detach_rollout_state=False,
                lambda_entropy_weight=0.0,
                lambda_smooth_weight=0.0,
                objective="blend",
            )

        torch.testing.assert_close(loss, torch.tensor(1.5, dtype=torch.float32))
        self.assertAlmostEqual(stats["total"], 1.5, places=6)
        self.assertAlmostEqual(stats["blend_loss"], 1.5, places=6)
        self.assertAlmostEqual(stats["inc_geo"], 2.0, places=6)
        self.assertAlmostEqual(stats["dir_geo"], 3.0, places=6)
        self.assertAlmostEqual(stats["lambda_mean"], 0.5, places=6)
        self.assertAlmostEqual(stats["lambda_eff_mean"], 0.4, places=6)
        self.assertAlmostEqual(stats["lambda_rel_mean"], 0.925, places=6)
        self.assertIsInstance(aux_payload, dict)
        self.assertIn("ema_update_payload", aux_payload)
        self.assertNotIn("leg_align_grad_probe", aux_payload)
        self.assertFalse(getattr(trainer, "_posttrain_soft_fail_counts", {}))

    def test_lambda_fusion_loss_rollout_direct_smoke_preserves_grad_probe_contract(self) -> None:
        trainer = _make_trainer()
        model = SimpleNamespace(direct_pose_leg_joint_names=["foot_l", "foot_r"])
        batch = {
            "motion": torch.zeros(1, 2, 6, dtype=torch.float32),
            "gt_motion": torch.zeros(1, 2, 6, dtype=torch.float32),
        }

        def _populate_unroll(*, runtime_ctx, weights_ctx, accum_ctx, state_vars):
            self.assertEqual(runtime_ctx["objective"], "direct")
            accum_ctx.loss_terms.append(torch.tensor(1.5, dtype=torch.float32))
            accum_ctx.inc_terms.append(torch.tensor(2.0, dtype=torch.float32))
            accum_ctx.dir_terms.append(torch.tensor(3.0, dtype=torch.float32))
            accum_ctx.leg_align_terms.append(torch.tensor(0.4, dtype=torch.float32))
            accum_ctx.leg_align_distal_terms.append(torch.tensor(0.1, dtype=torch.float32))
            accum_ctx.leg_align_proximal_terms.append(torch.tensor(0.3, dtype=torch.float32))
            accum_ctx.lam_vals.append(torch.tensor([[0.25, 0.75]], dtype=torch.float32))
            accum_ctx.lam_eff_vals.append(torch.tensor([[0.2, 0.6]], dtype=torch.float32))
            accum_ctx.lam_rel_vals.append(torch.tensor([[0.8, 0.9]], dtype=torch.float32))
            return False, float(state_vars.direct_nonleg_focus_applied)

        with (
            mock.patch("train.posttrain._lambda_rollout_prepare_context", return_value=_make_prep_ctx()),
            mock.patch("train.posttrain._lambda_rollout_resolve_nonleg_focus", return_value=_make_nonleg_focus_ctx()),
            mock.patch("train.posttrain._lambda_rollout_build_reg_params", return_value=_make_reg_ctx()),
            mock.patch("train.posttrain._lambda_fusion_run_unroll", side_effect=_populate_unroll),
        ):
            loss, stats, aux_payload = _lambda_fusion_loss_rollout(
                trainer,
                model,
                batch,
                columns=("motion", "gt_motion"),
                rollout_steps=2,
                rollout_cycles=1,
                include_boundary=False,
                boundary_weight=0.0,
                random_offset=False,
                time_index_mode="none",
                time_weight_max=1.0,
                time_weight_mode="uniform",
                detach_rollout_state=False,
                lambda_entropy_weight=0.0,
                lambda_smooth_weight=0.0,
                direct_pose_leg_align_weight=0.0,
                objective="direct",
            )

        torch.testing.assert_close(loss, torch.tensor(3.0, dtype=torch.float32))
        self.assertAlmostEqual(stats["total"], 3.0, places=6)
        self.assertAlmostEqual(stats["blend_loss"], 1.5, places=6)
        self.assertAlmostEqual(stats["inc_geo"], 2.0, places=6)
        self.assertAlmostEqual(stats["dir_geo"], 3.0, places=6)
        self.assertAlmostEqual(stats["lambda_mean"], 0.5, places=6)
        self.assertIn("leg_align_grad_probe", aux_payload)
        self.assertEqual(set(aux_payload["leg_align_grad_probe"].keys()), {"total", "distal", "proximal"})
        torch.testing.assert_close(aux_payload["leg_align_grad_probe"]["total"], torch.tensor(0.4, dtype=torch.float32))
        torch.testing.assert_close(aux_payload["leg_align_grad_probe"]["distal"], torch.tensor(0.1, dtype=torch.float32))
        torch.testing.assert_close(aux_payload["leg_align_grad_probe"]["proximal"], torch.tensor(0.3, dtype=torch.float32))
        self.assertFalse(getattr(trainer, "_posttrain_soft_fail_counts", {}))


if __name__ == "__main__":
    unittest.main()
