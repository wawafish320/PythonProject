from __future__ import annotations

import ast
import math
from pathlib import Path
from unittest import mock
import unittest

import torch

from train.data.normalizers import normalize_cond_tensor
from train.geometry import matrix_to_rot6d, reproject_cond_to_local_frame
from train.rollout_kernel import prepare_cond_input_from_raw


def _make_yaw_raw(yaw: float) -> torch.Tensor:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    root_R = torch.tensor(
        [[[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return matrix_to_rot6d(root_R.unsqueeze(0)).reshape(1, 6)


class RolloutCondSharedTest(unittest.TestCase):
    def test_prepare_cond_input_from_raw_skips_reprojection_when_disabled(self) -> None:
        base_cond_input = torch.full((1, 4), -9.0, dtype=torch.float32)
        cond_raw_step = torch.tensor([[7.0, 1.0, 0.0, 4.0]], dtype=torch.float32)
        cond_norm_mu = torch.zeros((1, 4), dtype=torch.float32)
        cond_norm_std = torch.tensor([[2.0, 2.0, 2.0, 4.0]], dtype=torch.float32)

        cond_input, cond_raw_for_model, reprojection_applied = prepare_cond_input_from_raw(
            base_cond_input=base_cond_input,
            cond_raw_step=cond_raw_step,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            cond_norm_clip=6.0,
            allow_reprojection=False,
            yaw_gt=torch.tensor([0.5], dtype=torch.float32),
            y_prev_raw=_make_yaw_raw(0.25),
            rot_slice=slice(0, 6),
            root_idx=0,
            up_axis=2,
            forward_axis=0,
            offset=0.0,
        )

        expected_cond_norm = normalize_cond_tensor(
            cond_raw_step,
            cond_norm_mu,
            cond_norm_std,
            cond_norm_clip=6.0,
        )

        torch.testing.assert_close(cond_raw_for_model, cond_raw_step)
        torch.testing.assert_close(cond_input, expected_cond_norm)
        self.assertFalse(reprojection_applied)

    def test_prepare_cond_input_from_raw_reprojects_with_gt_and_pred_yaw(self) -> None:
        base_cond_input = torch.full((1, 4), -9.0, dtype=torch.float32)
        cond_raw_step = torch.tensor([[7.0, 1.0, 0.0, 4.0]], dtype=torch.float32)
        cond_norm_mu = torch.zeros((1, 4), dtype=torch.float32)
        cond_norm_std = torch.tensor([[2.0, 2.0, 2.0, 4.0]], dtype=torch.float32)
        yaw_gt = torch.tensor([0.5], dtype=torch.float32)

        cond_input, cond_raw_for_model, reprojection_applied = prepare_cond_input_from_raw(
            base_cond_input=base_cond_input,
            cond_raw_step=cond_raw_step,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            cond_norm_clip=6.0,
            allow_reprojection=True,
            yaw_gt=yaw_gt,
            y_prev_raw=_make_yaw_raw(0.25),
            rot_slice=slice(0, 6),
            root_idx=0,
            up_axis=2,
            forward_axis=0,
            offset=0.0,
        )

        expected_cond_proj = reproject_cond_to_local_frame(
            cond_raw_step,
            yaw_gt,
            torch.tensor([0.25], dtype=torch.float32),
        )
        expected_cond_norm = normalize_cond_tensor(
            expected_cond_proj,
            cond_norm_mu,
            cond_norm_std,
            cond_norm_clip=6.0,
        )

        torch.testing.assert_close(cond_raw_for_model, expected_cond_proj)
        torch.testing.assert_close(cond_input, expected_cond_norm)
        self.assertTrue(reprojection_applied)

    def test_prepare_cond_input_from_raw_keeps_raw_when_yaw_pred_missing(self) -> None:
        base_cond_input = torch.full((1, 4), -9.0, dtype=torch.float32)
        cond_raw_step = torch.tensor([[7.0, 1.0, 0.0, 4.0]], dtype=torch.float32)
        cond_norm_mu = torch.zeros((1, 4), dtype=torch.float32)
        cond_norm_std = torch.tensor([[2.0, 2.0, 2.0, 4.0]], dtype=torch.float32)

        with mock.patch("train.rollout_kernel.root_yaw_from_raw_rot6d", return_value=None) as yaw_mock:
            cond_input, cond_raw_for_model, reprojection_applied = prepare_cond_input_from_raw(
                base_cond_input=base_cond_input,
                cond_raw_step=cond_raw_step,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
                cond_norm_clip=6.0,
                allow_reprojection=True,
                yaw_gt=torch.tensor([0.5], dtype=torch.float32),
                y_prev_raw=_make_yaw_raw(0.25),
                rot_slice=slice(0, 6),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
            )

        expected_cond_norm = normalize_cond_tensor(
            cond_raw_step,
            cond_norm_mu,
            cond_norm_std,
            cond_norm_clip=6.0,
        )

        yaw_mock.assert_called_once()
        torch.testing.assert_close(cond_raw_for_model, cond_raw_step)
        torch.testing.assert_close(cond_input, expected_cond_norm)
        self.assertFalse(reprojection_applied)

    def test_prepare_cond_input_from_raw_uses_normalize_override_when_available(self) -> None:
        base_cond_input = torch.full((1, 4), -9.0, dtype=torch.float32)
        cond_raw_step = torch.tensor([[7.0, 1.0, 0.0, 4.0]], dtype=torch.float32)
        cond_override = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

        with mock.patch(
            "train.rollout_kernel.normalize_cond_tensor",
            return_value=cond_override,
        ) as norm_mock:
            cond_input, cond_raw_for_model, reprojection_applied = prepare_cond_input_from_raw(
                base_cond_input=base_cond_input,
                cond_raw_step=cond_raw_step,
                cond_norm_mu=torch.zeros((1, 4), dtype=torch.float32),
                cond_norm_std=torch.ones((1, 4), dtype=torch.float32),
                cond_norm_clip=3.5,
                allow_reprojection=False,
                yaw_gt=None,
                y_prev_raw=None,
                rot_slice=slice(0, 6),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
            )

        torch.testing.assert_close(cond_input, cond_override)
        torch.testing.assert_close(cond_raw_for_model, cond_raw_step)
        self.assertFalse(reprojection_applied)
        self.assertEqual(norm_mock.call_args.kwargs["cond_norm_clip"], 3.5)

    def test_prepare_cond_input_from_raw_keeps_base_when_normalize_returns_none(self) -> None:
        base_cond_input = torch.full((1, 4), -9.0, dtype=torch.float32)
        cond_raw_step = torch.tensor([[7.0, 1.0, 0.0, 4.0]], dtype=torch.float32)

        with mock.patch("train.rollout_kernel.normalize_cond_tensor", return_value=None) as norm_mock:
            cond_input, cond_raw_for_model, reprojection_applied = prepare_cond_input_from_raw(
                base_cond_input=base_cond_input,
                cond_raw_step=cond_raw_step,
                cond_norm_mu=torch.zeros((1, 4), dtype=torch.float32),
                cond_norm_std=torch.ones((1, 4), dtype=torch.float32),
                cond_norm_clip=0.0,
                allow_reprojection=False,
                yaw_gt=None,
                y_prev_raw=None,
                rot_slice=slice(0, 6),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
            )

        torch.testing.assert_close(cond_input, base_cond_input)
        torch.testing.assert_close(cond_raw_for_model, cond_raw_step)
        self.assertFalse(reprojection_applied)
        self.assertEqual(norm_mock.call_args.kwargs["cond_norm_clip"], 0.0)


class RolloutCondSharedSentinelTest(unittest.TestCase):
    _FORBIDDEN_RUNTIME_ATTRS = frozenset(
        {
            "rot6d_y_slice",
            "rot6d_slice",
            "eval_root_idx",
            "eval_up_axis",
            "yaw_forward_axis",
            "yaw_forward_axis_offset",
            "cond_norm_clip",
        }
    )

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _parse_module(cls, rel_path: str) -> ast.AST:
        path = cls._repo_root() / rel_path
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    @staticmethod
    def _call_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
        return names

    @staticmethod
    def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    @classmethod
    def _forbidden_runtime_reads(
        cls,
        node: ast.AST,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[tuple[str, int]]:
        reads: list[tuple[str, int]] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            lineno = getattr(child, "lineno", None)
            if lineno is None:
                continue
            if start_line is not None and lineno < start_line:
                continue
            if end_line is not None and lineno > end_line:
                continue
            if not isinstance(child.func, ast.Name) or child.func.id not in {"getattr", "hasattr"}:
                continue
            if len(child.args) < 2:
                continue
            attr_arg = child.args[1]
            if not isinstance(attr_arg, ast.Constant) or not isinstance(attr_arg.value, str):
                continue
            if attr_arg.value in cls._FORBIDDEN_RUNTIME_ATTRS:
                reads.append((attr_arg.value, lineno))
        return reads

    @staticmethod
    def _find_call(node: ast.AST, name: str) -> ast.Call:
        return next(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and (
                (isinstance(child.func, ast.Name) and child.func.id == name)
                or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
            )
        )

    def test_prepare_rollout_cond_delegates_to_runtime_resolver(self) -> None:
        tree = self._parse_module("train/rollout_kernel.py")
        prepare_rollout_cond_node = self._find_function(tree, "prepare_rollout_cond")
        call_names = self._call_names(prepare_rollout_cond_node)

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertIn("prepare_cond_input_from_raw", call_names)
        self.assertNotIn("normalize_cond_tensor", call_names)
        self.assertNotIn("reproject_cond_to_local_frame", call_names)
        self.assertEqual(self._forbidden_runtime_reads(prepare_rollout_cond_node), [])

    def test_eval_utils_freerun_cond_path_uses_runtime_resolver(self) -> None:
        tree = self._parse_module("train/eval_utils.py")
        evaluate_freerun_node = self._find_function(tree, "evaluate_freerun")
        call_names = self._call_names(evaluate_freerun_node)
        prepare_call = self._find_call(evaluate_freerun_node, "prepare_cond_input_from_raw")

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertIn("prepare_cond_input_from_raw", call_names)
        self.assertNotIn("normalize_cond_tensor", call_names)
        self.assertNotIn("reproject_cond_to_local_frame", call_names)
        self.assertEqual(
            self._forbidden_runtime_reads(
                evaluate_freerun_node,
                start_line=max(1, prepare_call.lineno - 40),
                end_line=prepare_call.lineno,
            ),
            [],
        )

    def test_training_and_posttrain_yaw_paths_use_runtime_resolver(self) -> None:
        training_tree = self._parse_module("train/training_MPL.py")
        training_node = self._find_function(training_tree, "_resolve_rollout_gt_yaw")
        training_calls = self._call_names(training_node)
        self.assertIn("resolve_rollout_cond_runtime_config", training_calls)
        self.assertEqual(self._forbidden_runtime_reads(training_node), [])

        posttrain_tree = self._parse_module("train/posttrain.py")
        posttrain_node = self._find_function(posttrain_tree, "_lambda_fusion_run_unroll")
        posttrain_calls = self._call_names(posttrain_node)
        self.assertIn("resolve_rollout_cond_runtime_config", posttrain_calls)
        self.assertEqual(self._forbidden_runtime_reads(posttrain_node), [])

    def test_run_freerun_cycles_uses_shared_owner_without_local_pipeline(self) -> None:
        call_names = self._call_names(self._parse_module("train/validate/run_freerun_cycles.py"))

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertGreaterEqual(call_names.count("prepare_cond_input_from_raw"), 2)
        self.assertNotIn("normalize_cond_tensor", call_names)
        self.assertNotIn("reproject_cond_to_local_frame", call_names)


if __name__ == "__main__":
    unittest.main()
