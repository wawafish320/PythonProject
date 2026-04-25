from __future__ import annotations

import math
from unittest import mock
import unittest

import torch

from train.data.normalizers import normalize_cond_tensor
from train.geometry import matrix_to_rot6d, reproject_cond_to_local_frame
from train.history import PoseHistState
from train.rollout_kernel import (
    RolloutCondRuntimeConfig,
    RolloutModelStepRequest,
    execute_rollout_model_step,
    prepare_cond_input_from_raw,
    prepare_rollout_cond,
)

_REMOVED_HELPERS = (
    "_infer_root_yaw" + "_from_rot6d",
    "_reproject_cond_to" + "_local_frame",
    "_normalize_cond" + "_from_raw",
)


class _TrainerStub:
    def __init__(self) -> None:
        self.rot6d_y_slice = slice(0, 6)
        self.rot6d_slice = slice(0, 6)
        self.eval_root_idx = 0
        self.eval_up_axis = 2
        self._up_axis = 2
        self.yaw_forward_axis = 0
        self.yaw_forward_axis_offset = 0.0
        self.cond_norm_clip = 6.0
        self.yaw_calls: list[int] = []


def _make_yaw_raw(yaw: float) -> torch.Tensor:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    root_R = torch.tensor(
        [[[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return matrix_to_rot6d(root_R.unsqueeze(0)).reshape(1, 6)


class RolloutKernelExecuteModelStepTest(unittest.TestCase):
    def test_prepare_rollout_cond_uses_shared_owner(self) -> None:
        trainer = _TrainerStub()
        cond_seq = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]],
            dtype=torch.float32,
        )
        cond_raw_seq = torch.tensor(
            [[[7.0, 1.0, 0.0, 4.0], [8.0, 0.0, 1.0, 5.0], [9.0, -1.0, 0.0, 6.0]]],
            dtype=torch.float32,
        )
        cond_norm_mu = torch.zeros((1, 4), dtype=torch.float32)
        cond_norm_std = torch.tensor([[2.0, 2.0, 2.0, 4.0]], dtype=torch.float32)

        def _yaw_gt(step_idx: int) -> torch.Tensor:
            trainer.yaw_calls.append(int(step_idx))
            return torch.tensor([0.5], dtype=torch.float32)

        runtime_cfg = RolloutCondRuntimeConfig(
            rot_slice=slice(0, 6),
            root_idx=0,
            up_axis=2,
            forward_axis=0,
            offset=0.0,
            cond_norm_clip=0.0,
        )
        expected_cond_input = cond_seq[:, 2]
        expected_cond_raw = cond_raw_seq[:, 0]
        expected_cond_proj = reproject_cond_to_local_frame(
            expected_cond_raw,
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
        )
        expected_cond_norm = normalize_cond_tensor(
            expected_cond_proj,
            cond_norm_mu,
            cond_norm_std,
            cond_norm_clip=runtime_cfg.cond_norm_clip,
        )

        with mock.patch(
            "train.rollout_kernel.prepare_cond_input_from_raw",
            wraps=prepare_cond_input_from_raw,
        ) as shared_mock, mock.patch(
            "train.rollout_kernel.resolve_rollout_cond_runtime_config",
            return_value=runtime_cfg,
        ) as runtime_mock:
            cond_input, cond_raw_step, cond_raw_for_model, reprojection_applied = prepare_rollout_cond(
                trainer,
                cond_seq=cond_seq,
                cond_raw_seq=cond_raw_seq,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
                step_idx=1,
                cond_idx=2,
                cond_has_time_dim=True,
                cond_raw_has_time_dim=True,
                cond_raw_offset=1,
                include_boundary=True,
                cycle_len=3,
                y_prev_raw=_make_yaw_raw(0.25),
                allow_reprojection=True,
                yaw_gt_fn=_yaw_gt,
            )

        self.assertEqual(trainer.yaw_calls, [2])
        runtime_mock.assert_called_once_with(trainer)
        shared_mock.assert_called_once()
        torch.testing.assert_close(cond_input, expected_cond_norm)
        torch.testing.assert_close(cond_raw_step, expected_cond_raw)
        torch.testing.assert_close(cond_raw_for_model, expected_cond_proj)
        self.assertTrue(reprojection_applied)
        torch.testing.assert_close(shared_mock.call_args.kwargs["base_cond_input"], expected_cond_input)
        self.assertEqual(shared_mock.call_args.kwargs["cond_norm_clip"], runtime_cfg.cond_norm_clip)
        self.assertEqual(shared_mock.call_args.kwargs["rot_slice"], runtime_cfg.rot_slice)
        self.assertTrue(shared_mock.call_args.kwargs["normalize_fail_open"])

    def test_execute_rollout_model_step_uses_shared_cond_yaw_pipeline(self) -> None:
        trainer = _TrainerStub()
        for attr_name in _REMOVED_HELPERS:
            self.assertFalse(hasattr(trainer, attr_name))
        model = object()
        state = {
            "motion": torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            "y_prev_raw": _make_yaw_raw(0.25),
            "pose_hist_state": PoseHistState(enabled=False, length=0, dim=0, stride=0),
            "plan_z": torch.tensor([[0.7]], dtype=torch.float32),
            "meas_logits_prev": torch.tensor([[0.9]], dtype=torch.float32),
        }
        cond_raw_seq = torch.tensor(
            [[[7.0, 1.0, 0.0, 4.0], [8.0, 0.0, 1.0, 5.0], [9.0, -1.0, 0.0, 6.0]]],
            dtype=torch.float32,
        )
        cond_norm_mu = torch.zeros((1, 4), dtype=torch.float32)
        cond_norm_std = torch.tensor([[2.0, 2.0, 2.0, 4.0]], dtype=torch.float32)
        angvel_seq = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
            dtype=torch.float32,
        )
        pose_hist_seq = torch.tensor(
            [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]],
            dtype=torch.float32,
        )

        def _yaw_gt(step_idx: int) -> torch.Tensor:
            trainer.yaw_calls.append(int(step_idx))
            return torch.tensor([0.5], dtype=torch.float32)

        expected_cond_raw = cond_raw_seq[:, 0]
        expected_cond_proj = reproject_cond_to_local_frame(
            expected_cond_raw,
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
        )
        expected_cond_norm = normalize_cond_tensor(
            expected_cond_proj,
            cond_norm_mu,
            cond_norm_std,
            cond_norm_clip=trainer.cond_norm_clip,
        )

        request = RolloutModelStepRequest(
            state=state,
            step_idx=1,
            frame_idx=2,
            total_steps=3,
            cond_seq=None,
            cond_raw_seq=cond_raw_seq,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            time_index_mode="cycle",
            time_base=torch.tensor([10], dtype=torch.int64),
            enable_reprojection=True,
            include_boundary=True,
            cycle_len=3,
            cond_raw_offset=1,
            yaw_gt_fn=_yaw_gt,
        )

        call_payload: dict[str, object] = {}

        def _fake_forward(model_obj, **kwargs):
            self.assertIs(model_obj, model)
            call_payload.update(kwargs)
            return {"out": torch.zeros(1, 1, 6, dtype=torch.float32)}, torch.zeros(1, 1, 6, dtype=torch.float32), None

        with (
            mock.patch("train.rollout_kernel.forward_rollout_model_step", side_effect=_fake_forward) as forward_mock,
            mock.patch("train.rollout_kernel.update_rollout_recurrent_state") as update_mock,
        ):
            out = execute_rollout_model_step(trainer, model, request)

        self.assertEqual(trainer.yaw_calls, [2])
        torch.testing.assert_close(out["cond_raw_step"], expected_cond_raw)
        torch.testing.assert_close(call_payload["motion"], state["motion"].unsqueeze(1))
        torch.testing.assert_close(call_payload["cond_input"], expected_cond_norm.unsqueeze(1))
        torch.testing.assert_close(call_payload["angvel_t"], angvel_seq[:, 2:3])
        torch.testing.assert_close(call_payload["pose_history_t"], pose_hist_seq[:, 2:3])
        torch.testing.assert_close(call_payload["time_index_t"], torch.tensor([12], dtype=torch.int64))
        torch.testing.assert_close(call_payload["rollout_step_t"], torch.tensor([[[0.5]]], dtype=torch.float32))
        forward_mock.assert_called_once()
        update_mock.assert_called_once()
        self.assertIs(update_mock.call_args.args[1], out["ret"])
        self.assertIs(update_mock.call_args.args[2], state)


if __name__ == "__main__":
    unittest.main()
