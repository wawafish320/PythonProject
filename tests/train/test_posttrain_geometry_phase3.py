from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from train.geometry import blend_rot6d_raw_with_lambda, compose_delta_raw_to_next, matrix_to_rot6d, so3_exp_map
from train.posttrain import (
    LambdaFusionAccum,
    LambdaRolloutDataContext,
    LambdaRolloutRuntimeContext,
    LambdaRolloutStepContext,
    LambdaRolloutStepState,
    LambdaRolloutWeights,
    _lambda_rollout_unroll_single_step,
)


def _make_weights() -> LambdaRolloutWeights:
    return LambdaRolloutWeights(
        contact_meas_weight=0.0,
        direct_pose_leg_align_weight=0.0,
        direct_pose_leg_align_oracle_min_deg=0.0,
        direct_pose_leg_align_oracle_weight_deg=0.0,
        direct_pose_leg_align_mode="cos",
        direct_pose_leg_align_mag_weight=1.0,
        direct_pose_leg_align_res_weight=1.0,
        direct_pose_leg_align_sign_weight=0.0,
        direct_pose_leg_align_cos_thresh=0.0,
        direct_pose_leg_align_target_joints=None,
        direct_pose_leg_align_anchor_joints=None,
        direct_pose_leg_align_anchor_weight=0.0,
        direct_pose_leg_gate_sup_weight=0.0,
        direct_pose_loss_leg_split=False,
        direct_nonleg_focus_mask_j=None,
        direct_nonleg_focus_resolved=0,
        direct_nonleg_focus_weight_use=1.0,
        gate_sup_weight=0.0,
        gate_sup_start=0,
        tau_rad=0.0,
        margin_rad=0.0,
        lambda_plan_entropy_weight=0.0,
        lambda_plan_dyn_weight=0.0,
        lambda_early_weight=0.0,
        lambda_early_steps=0,
        lambda_entropy_weight=0.0,
        lambda_smooth_weight=0.0,
        lambda_monotonic_weight=0.0,
    )


class PosttrainGeometryPhase3Test(unittest.TestCase):
    def test_unroll_single_step_carry_uses_shared_geometry_outputs(self) -> None:
        columns = ("X", "Z")
        batch_size, joint_count = 1, 1
        rot_slice = slice(0, 6)

        y_prev_raw = matrix_to_rot6d(
            so3_exp_map(torch.tensor([[[0.10, -0.03, 0.02]]], dtype=torch.float32)),
            columns=columns,
        ).view(batch_size, 6)
        direct_raw = matrix_to_rot6d(
            so3_exp_map(torch.tensor([[[0.25, 0.04, -0.05]]], dtype=torch.float32)),
            columns=columns,
        ).view(batch_size, 6)
        delta_norm = torch.tensor([[0.03, -0.01, 0.02, -0.02, 0.01, 0.04]], dtype=torch.float32)
        lambda_fusion = torch.tensor([[0.35]], dtype=torch.float32)
        gt_seq = direct_raw.unsqueeze(1).repeat(1, 2, 1)

        trainer = SimpleNamespace(
            _denorm=lambda x: x,
            _lambda_fusion_apply_reliability=lambda lam, **kwargs: (lam, None),
        )
        model = SimpleNamespace()
        state = {"y_prev_raw": y_prev_raw.clone()}

        runtime = LambdaRolloutRuntimeContext(
            trainer=trainer,
            model=model,
            state=state,
            total_steps=2,
            cycle_len=2,
            include_boundary=False,
            steps=2,
            offset=0,
            time_index_mode="none",
            time_base=None,
            enable_reprojection=False,
            detach_rollout_state=False,
            yaw_gt_fn=None,
            columns=columns,
            B=batch_size,
            J=joint_count,
            objective="blend",
            y0_raw=None,
            gt_seq=gt_seq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            rot_len=6,
        )
        data = LambdaRolloutDataContext(
            cond_seq=None,
            cond_raw_tgt=None,
            cond_norm_mu=None,
            cond_norm_std=None,
            angvel_seq=None,
            pose_hist_seq=None,
            contacts_seq=None,
            step_weights=torch.ones(2, dtype=torch.float32),
            std_y=torch.ones(6, dtype=torch.float32),
            rot_slice=rot_slice,
        )
        accum = LambdaFusionAccum()
        state_vars = LambdaRolloutStepState(
            meas_used_logits=False,
            direct_nonleg_focus_applied=0.0,
            lam_prev=None,
            lam_prev_monot=None,
            plan_prev=None,
        )
        ctx = LambdaRolloutStepContext(
            runtime=runtime,
            data=data,
            weights=_make_weights(),
            accum=accum,
            state_vars=state_vars,
        )

        expected_y_inc = compose_delta_raw_to_next(
            y_prev_raw,
            delta_norm,
            rot_slice=rot_slice,
            columns=columns,
            omega_hat=None,
            gate_val=0.0,
            max_deg=0.0,
            omega_detach=True,
            reproject=False,
        )
        expected_y_blend = blend_rot6d_raw_with_lambda(
            expected_y_inc,
            direct_raw,
            lambda_fusion,
            rot_slice=rot_slice,
            columns=columns,
        )

        carry_capture: dict[str, torch.Tensor] = {}
        with (
            mock.patch(
                "train.posttrain._rollout_step_common",
                return_value={
                    "ret": {"out": delta_norm, "out_direct": direct_raw, "lambda_fusion": lambda_fusion},
                    "contacts_in_t": None,
                    "cond_raw_step": torch.zeros(batch_size, 1, dtype=torch.float32),
                    "rollout_step_t": None,
                },
            ),
            mock.patch(
                "train.posttrain._lambda_rollout_accumulate_plan_terms",
                side_effect=lambda **kwargs: kwargs["plan_prev"],
            ),
            mock.patch(
                "train.posttrain._lambda_rollout_accumulate_direct_objective",
                side_effect=lambda **kwargs: kwargs["direct_nonleg_focus_applied"],
            ),
            mock.patch("train.posttrain._lambda_rollout_accumulate_gate_supervision", return_value=None),
            mock.patch(
                "train.posttrain._rollout_kernel.apply_rollout_carry_state",
                side_effect=lambda trainer, state, *, y_next_raw, cond_raw_step: carry_capture.setdefault(
                    "y_next_raw", y_next_raw.detach().clone()
                ),
            ),
        ):
            _lambda_rollout_unroll_single_step(t=0, ctx=ctx)

        torch.testing.assert_close(carry_capture["y_next_raw"], expected_y_blend, atol=1e-6, rtol=1e-6)
        self.assertEqual(len(accum.loss_terms), 1)
        self.assertEqual(len(accum.lam_vals), 1)
        self.assertEqual(len(accum.lam_eff_vals), 1)


if __name__ == "__main__":
    unittest.main()
