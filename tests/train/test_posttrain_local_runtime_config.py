from __future__ import annotations

from types import SimpleNamespace
import unittest

from train.configuration import model_build as mb


class PosttrainLocalRuntimeConfigTest(unittest.TestCase):
    def test_local_runtime_defaults_are_centralized(self) -> None:
        cfg = mb.resolve_posttrain_local_runtime_config(SimpleNamespace())

        self.assertEqual(cfg.contact_meas_gate_by_hit, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_GATE_BY_HIT)
        self.assertIsNone(cfg.contact_meas_gate_by_hit_override)
        self.assertEqual(cfg.contact_meas_vxy_mode, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_VXY_MODE)
        self.assertEqual(cfg.contact_meas_ground_z_mode, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_MODE)
        self.assertEqual(cfg.contact_meas_ground_z_beta, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_BETA)
        self.assertEqual(cfg.contact_meas_ground_z_window, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_WINDOW)
        self.assertEqual(cfg.contact_meas_ground_z_quantile, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_GROUND_Z_QUANTILE)
        self.assertEqual(cfg.lambda_reliability_mode, mb.DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_MODE)
        self.assertEqual(cfg.lambda_reliability_warmup_steps, mb.DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_WARMUP_STEPS)
        self.assertEqual(cfg.lambda_reliability_contact_err_max, mb.DEFAULT_POSTTRAIN_LAMBDA_RELIABILITY_CONTACT_ERR_MAX)
        self.assertIsNone(cfg.lambda_reliability_warmup_joint_scales)

    def test_lambda_objective_defaults_are_centralized(self) -> None:
        cfg = mb.resolve_posttrain_lambda_objective_config(SimpleNamespace())

        self.assertEqual(cfg.lambda_fusion_mode, mb.DEFAULT_LAMBDA_FUSION_MODE)
        self.assertEqual(cfg.lambda_fusion_hidden, mb.DEFAULT_LAMBDA_FUSION_HIDDEN)
        self.assertEqual(cfg.lambda_fusion_dropout, mb.DEFAULT_LAMBDA_FUSION_DROPOUT)
        self.assertEqual(cfg.lambda_fusion_logit_init, mb.DEFAULT_LAMBDA_FUSION_LOGIT_INIT)
        self.assertEqual(cfg.lambda_fusion_use_rollout_step, mb.DEFAULT_LAMBDA_FUSION_USE_ROLLOUT_STEP)
        self.assertEqual(cfg.lambda_fusion_entropy_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_FUSION_ENTROPY_WEIGHT)
        self.assertEqual(cfg.lambda_fusion_smooth_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_FUSION_SMOOTH_WEIGHT)
        self.assertEqual(cfg.lambda_fusion_early_steps, mb.DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_STEPS)
        self.assertEqual(cfg.lambda_fusion_early_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_FUSION_EARLY_WEIGHT)
        self.assertEqual(cfg.lambda_fusion_monotonic_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_FUSION_MONOTONIC_WEIGHT)
        self.assertEqual(cfg.lambda_plan_entropy_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_PLAN_ENTROPY_WEIGHT)
        self.assertEqual(cfg.lambda_plan_dyn_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_PLAN_DYN_WEIGHT)
        self.assertEqual(cfg.lambda_time_weight_mode, mb.DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MODE)
        self.assertEqual(cfg.lambda_time_weight_max, mb.DEFAULT_POSTTRAIN_LAMBDA_TIME_WEIGHT_MAX)
        self.assertEqual(cfg.lambda_l2sp_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_L2SP_WEIGHT)
        self.assertEqual(cfg.lambda_boundary_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_BOUNDARY_WEIGHT)
        self.assertEqual(cfg.lambda_gate_sup_weight, mb.DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_WEIGHT)
        self.assertEqual(cfg.lambda_gate_sup_tau_deg, mb.DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_TAU_DEG)
        self.assertEqual(cfg.lambda_gate_sup_margin_deg, mb.DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_MARGIN_DEG)
        self.assertEqual(cfg.lambda_gate_sup_start_step, mb.DEFAULT_POSTTRAIN_LAMBDA_GATE_SUP_START_STEP)
        self.assertEqual(cfg.contact_meas_weight, mb.DEFAULT_POSTTRAIN_CONTACT_MEAS_WEIGHT)

    def test_explicit_zero_and_false_are_not_replaced_by_defaults(self) -> None:
        local = mb.resolve_posttrain_local_runtime_config(
            SimpleNamespace(
                contact_meas_gate_by_hit="false",
                contact_meas_ground_z_beta=0.0,
                contact_meas_ground_z_window=0,
                contact_meas_ground_z_quantile=0.0,
                contact_meas_ground_z_slew_up_cm=0.0,
                lambda_reliability_warmup_steps=0,
                lambda_reliability_contact_err_max=0.0,
            )
        )
        self.assertEqual(local.contact_meas_gate_by_hit, "false")
        self.assertIs(local.contact_meas_gate_by_hit_override, False)
        self.assertEqual(local.contact_meas_ground_z_beta, 0.0)
        self.assertEqual(local.contact_meas_ground_z_window, 1)
        self.assertEqual(local.contact_meas_ground_z_quantile, 0.0)
        self.assertEqual(local.contact_meas_ground_z_max_up_m, 0.0)
        self.assertEqual(local.lambda_reliability_warmup_steps, 0)
        self.assertEqual(local.lambda_reliability_contact_err_max, 0.0)

        objective = mb.resolve_posttrain_lambda_objective_config(
            SimpleNamespace(
                lambda_fusion_use_rollout_step=False,
                lambda_fusion_entropy_weight=0.0,
                lambda_fusion_smooth_weight=0.0,
                lambda_fusion_early_steps=0,
                lambda_fusion_early_weight=0.0,
                lambda_plan_entropy_weight=0.0,
                lambda_time_weight_max=0.0,
                lambda_l2sp_weight=0.0,
                lambda_gate_sup_weight=0.0,
                contact_meas_weight=0.0,
            )
        )
        self.assertFalse(objective.lambda_fusion_use_rollout_step)
        self.assertEqual(objective.lambda_fusion_entropy_weight, 0.0)
        self.assertEqual(objective.lambda_fusion_smooth_weight, 0.0)
        self.assertEqual(objective.lambda_fusion_early_steps, 0)
        self.assertEqual(objective.lambda_fusion_early_weight, 0.0)
        self.assertEqual(objective.lambda_plan_entropy_weight, 0.0)
        self.assertEqual(objective.lambda_time_weight_max, 1.0)
        self.assertEqual(objective.lambda_l2sp_weight, 0.0)
        self.assertEqual(objective.lambda_gate_sup_weight, 0.0)
        self.assertEqual(objective.contact_meas_weight, 0.0)

    def test_enum_normalization_and_invalid_values_fail_fast(self) -> None:
        local = mb.resolve_posttrain_local_runtime_config(
            SimpleNamespace(
                contact_meas_gate_by_hit="YES",
                contact_meas_vxy_mode="root-relative",
                contact_meas_ground_z_mode="EMA",
                lambda_reliability_mode="step_warmup,contact_err",
            )
        )
        self.assertEqual(local.contact_meas_gate_by_hit, "true")
        self.assertIs(local.contact_meas_gate_by_hit_override, True)
        self.assertEqual(local.contact_meas_vxy_mode, "root_rel")
        self.assertEqual(local.contact_meas_ground_z_mode, "ema")
        self.assertEqual(local.lambda_reliability_mode, "warmup+contacts_err")

        objective = mb.resolve_posttrain_lambda_objective_config(
            SimpleNamespace(lambda_time_weight_mode="one_over_t")
        )
        self.assertEqual(objective.lambda_time_weight_mode, "inv")

        for bad_cfg, pattern in (
            (SimpleNamespace(contact_meas_gate_by_hit="maybe"), "contact_meas_gate_by_hit"),
            (SimpleNamespace(contact_meas_vxy_mode="signed"), "contact_meas_vxy_mode"),
            (SimpleNamespace(contact_meas_ground_z_mode="bad"), "contact_meas_ground_z_mode"),
            (SimpleNamespace(lambda_reliability_mode="bad"), "lambda_reliability_mode"),
        ):
            with self.assertRaisesRegex(ValueError, pattern):
                mb.resolve_posttrain_local_runtime_config(bad_cfg)
        with self.assertRaisesRegex(ValueError, "lambda_time_weight_mode"):
            mb.resolve_posttrain_lambda_objective_config(SimpleNamespace(lambda_time_weight_mode="bad"))


if __name__ == "__main__":
    unittest.main()
