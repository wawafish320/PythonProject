from __future__ import annotations

from types import SimpleNamespace
import unittest

from train.rollout_kernel import (
    FreeCarryRuntimeConfig,
    PoseHistRuntimeConfig,
    RolloutCondRuntimeConfig,
    resolve_free_carry_runtime_config,
    resolve_pose_hist_runtime_config,
    resolve_rollout_cond_runtime_config,
)


class RolloutRuntimeConfigTest(unittest.TestCase):
    def test_rot6d_y_slice_takes_precedence_over_rot6d_slice(self) -> None:
        trainer = SimpleNamespace(
            rot6d_y_slice=slice(6, 12),
            rot6d_slice=slice(0, 6),
            eval_root_idx=3,
            eval_up_axis=1,
            _up_axis=2,
            yaw_forward_axis=0,
            yaw_forward_axis_offset=0.5,
            cond_norm_clip=4.0,
        )

        runtime_cfg = resolve_rollout_cond_runtime_config(trainer)

        self.assertIsInstance(runtime_cfg, RolloutCondRuntimeConfig)
        self.assertEqual(runtime_cfg.rot_slice, slice(6, 12))
        self.assertEqual(runtime_cfg.root_idx, 3)
        self.assertEqual(runtime_cfg.up_axis, 1)
        self.assertEqual(runtime_cfg.forward_axis, 0)
        self.assertEqual(runtime_cfg.offset, 0.5)
        self.assertEqual(runtime_cfg.cond_norm_clip, 4.0)

    def test_rot6d_y_slice_falls_back_to_rot6d_slice_when_invalid(self) -> None:
        trainer = SimpleNamespace(
            rot6d_y_slice="invalid",
            rot6d_slice=slice(0, 6),
        )

        runtime_cfg = resolve_rollout_cond_runtime_config(trainer)

        self.assertEqual(runtime_cfg.rot_slice, slice(0, 6))

    def test_rot_slice_is_none_when_both_candidates_are_invalid(self) -> None:
        trainer = SimpleNamespace(
            rot6d_y_slice="invalid",
            rot6d_slice=None,
        )

        runtime_cfg = resolve_rollout_cond_runtime_config(trainer)

        self.assertIsNone(runtime_cfg.rot_slice)

    def test_eval_up_axis_falls_back_to_private_axis(self) -> None:
        trainer = SimpleNamespace(
            eval_up_axis=None,
            _up_axis=1,
        )

        runtime_cfg = resolve_rollout_cond_runtime_config(trainer)

        self.assertEqual(runtime_cfg.up_axis, 1)

    def test_defaults_preserve_contract_for_none_and_zero(self) -> None:
        trainer = SimpleNamespace(
            eval_root_idx=None,
            eval_up_axis=None,
            _up_axis=None,
            yaw_forward_axis=None,
            yaw_forward_axis_offset=None,
            cond_norm_clip=None,
        )

        runtime_cfg = resolve_rollout_cond_runtime_config(trainer)

        self.assertEqual(runtime_cfg.root_idx, 0)
        self.assertEqual(runtime_cfg.up_axis, 2)
        self.assertEqual(runtime_cfg.forward_axis, 2)
        self.assertEqual(runtime_cfg.offset, 0.0)
        self.assertEqual(runtime_cfg.cond_norm_clip, 6.0)

        trainer.cond_norm_clip = 0.0
        runtime_cfg_zero = resolve_rollout_cond_runtime_config(trainer)
        self.assertEqual(runtime_cfg_zero.cond_norm_clip, 0.0)


class FreeCarryRuntimeConfigTest(unittest.TestCase):
    def test_resolves_free_carry_slices_columns_and_clock(self) -> None:
        trainer = SimpleNamespace(
            rot6d_x_slice=slice(0, 12),
            rot6d_y_slice=slice(3, 15),
            rot6d_slice=slice(99, 105),
            angvel_x_slice=slice(17, 23),
            rootvel_x_slice=slice(12, 14),
            rootpos_x_slice=slice(14, 17),
            bone_hz=120.0,
            loss_fn=SimpleNamespace(_rot6d_columns=(" y ", "x", "Z")),
        )

        runtime_cfg = resolve_free_carry_runtime_config(trainer)

        self.assertIsInstance(runtime_cfg, FreeCarryRuntimeConfig)
        self.assertEqual(runtime_cfg.rot6d_x_slice, slice(0, 12))
        self.assertEqual(runtime_cfg.rot6d_y_slice, slice(3, 15))
        self.assertEqual(runtime_cfg.angvel_x_slice, slice(17, 23))
        self.assertEqual(runtime_cfg.rootvel_x_slice, slice(12, 14))
        self.assertEqual(runtime_cfg.rootpos_x_slice, slice(14, 17))
        self.assertEqual(runtime_cfg.bone_hz, 120.0)
        self.assertEqual(runtime_cfg.columns, ("Y", "X"))

    def test_falls_back_to_rot6d_slice_and_safe_defaults(self) -> None:
        trainer = SimpleNamespace(
            rot6d_x_slice="invalid",
            rot6d_y_slice=None,
            rot6d_slice=slice(0, 6),
            angvel_x_slice="invalid",
            rootvel_x_slice=None,
            rootpos_x_slice="invalid",
            bone_hz=None,
            loss_fn=SimpleNamespace(_rot6d_columns=("X", "X")),
        )

        runtime_cfg = resolve_free_carry_runtime_config(trainer)

        self.assertEqual(runtime_cfg.rot6d_x_slice, slice(0, 6))
        self.assertEqual(runtime_cfg.rot6d_y_slice, slice(0, 6))
        self.assertIsNone(runtime_cfg.angvel_x_slice)
        self.assertIsNone(runtime_cfg.rootvel_x_slice)
        self.assertIsNone(runtime_cfg.rootpos_x_slice)
        self.assertEqual(runtime_cfg.bone_hz, 60.0)
        self.assertEqual(runtime_cfg.columns, ("X", "Z"))

    def test_rot6d_slices_are_none_when_all_candidates_are_invalid(self) -> None:
        trainer = SimpleNamespace(
            rot6d_x_slice="invalid",
            rot6d_y_slice="invalid",
            rot6d_slice=None,
        )

        runtime_cfg = resolve_free_carry_runtime_config(trainer)

        self.assertIsNone(runtime_cfg.rot6d_x_slice)
        self.assertIsNone(runtime_cfg.rot6d_y_slice)


class PoseHistRuntimeConfigTest(unittest.TestCase):
    def test_resolves_pose_hist_contract_fields(self) -> None:
        params_fn = lambda ref: (ref, ref, ref)
        trainer = SimpleNamespace(
            pose_hist_len=2,
            pose_hist_dim=6,
            force_pose_hist_seq=True,
            _pose_hist_params=params_fn,
        )

        runtime_cfg = resolve_pose_hist_runtime_config(trainer)

        self.assertIsInstance(runtime_cfg, PoseHistRuntimeConfig)
        self.assertEqual(runtime_cfg.pose_hist_len, 2)
        self.assertEqual(runtime_cfg.pose_hist_dim, 6)
        self.assertEqual(runtime_cfg.pose_hist_stride, 3)
        self.assertTrue(runtime_cfg.enabled)
        self.assertTrue(runtime_cfg.force_pose_hist_seq)
        self.assertIs(runtime_cfg.params_fn, params_fn)

    def test_defaults_disable_pose_hist_when_missing_or_invalid(self) -> None:
        trainer = SimpleNamespace(
            pose_hist_len=None,
            pose_hist_dim="invalid",
            force_pose_hist_seq=False,
            _pose_hist_params="invalid",
        )

        with self.assertRaises(ValueError):
            resolve_pose_hist_runtime_config(trainer)

        trainer.pose_hist_dim = None
        runtime_cfg = resolve_pose_hist_runtime_config(trainer)
        self.assertEqual(runtime_cfg.pose_hist_len, 0)
        self.assertEqual(runtime_cfg.pose_hist_dim, 0)
        self.assertEqual(runtime_cfg.pose_hist_stride, 0)
        self.assertFalse(runtime_cfg.enabled)
        self.assertFalse(runtime_cfg.force_pose_hist_seq)
        self.assertIsNone(runtime_cfg.params_fn)


if __name__ == "__main__":
    unittest.main()
