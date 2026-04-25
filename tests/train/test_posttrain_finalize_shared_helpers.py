from __future__ import annotations

import ast
import inspect
import math
import textwrap
import unittest

import torch

from train import posttrain
from train.posttrain_shared import (
    reduce_optional_term_totals,
    safe_float_scalar,
    summarize_lambda_finalize_stats,
)


def _function_node(fn) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


class PosttrainFinalizeSharedHelpersTest(unittest.TestCase):
    def test_reduce_optional_term_totals_keeps_zero_for_empty_lists(self) -> None:
        reduced = reduce_optional_term_totals(
            zero=torch.tensor(0.0, dtype=torch.float32),
            named_terms={
                "present": [torch.tensor(1.0, dtype=torch.float32), torch.tensor(2.0, dtype=torch.float32)],
                "empty": [],
            },
        )

        torch.testing.assert_close(reduced["present"], torch.tensor(3.0, dtype=torch.float32))
        torch.testing.assert_close(reduced["empty"], torch.tensor(0.0, dtype=torch.float32))

    def test_summarize_lambda_finalize_stats_matches_legacy_values(self) -> None:
        recorded: list[str] = []

        stats = summarize_lambda_finalize_stats(
            lam_vals=[torch.tensor([[0.2, 0.8]], dtype=torch.float32)],
            lam_eff_vals=[torch.tensor([[0.1, 0.7]], dtype=torch.float32)],
            lam_rel_vals=[torch.tensor([[0.9, 0.95]], dtype=torch.float32)],
            record_soft_fail=recorded.append,
        )

        self.assertEqual(recorded, [])
        self.assertAlmostEqual(stats["lambda_mean"], 0.5, places=6)
        self.assertAlmostEqual(stats["lambda_std"], 0.3, places=6)
        self.assertAlmostEqual(stats["lambda_eff_mean"], 0.4, places=6)
        self.assertAlmostEqual(stats["lambda_eff_std"], 0.3, places=6)
        self.assertAlmostEqual(stats["lambda_rel_mean"], 0.925, places=6)

    def test_summarize_lambda_finalize_stats_empty_rel_returns_nan(self) -> None:
        stats = summarize_lambda_finalize_stats(
            lam_vals=[torch.tensor([[0.25, 0.75]], dtype=torch.float32)],
            lam_eff_vals=[torch.tensor([[0.2, 0.6]], dtype=torch.float32)],
            lam_rel_vals=[],
        )

        self.assertAlmostEqual(stats["lambda_mean"], 0.5, places=6)
        self.assertAlmostEqual(stats["lambda_eff_mean"], 0.4, places=6)
        self.assertTrue(math.isnan(stats["lambda_rel_mean"]))

    def test_safe_float_scalar_preserves_finalize_tensor_conversion(self) -> None:
        self.assertAlmostEqual(safe_float_scalar(torch.tensor(1.25, dtype=torch.float32)), 1.25, places=6)
        self.assertAlmostEqual(safe_float_scalar(2.5), 2.5, places=6)
        self.assertTrue(math.isnan(safe_float_scalar(None)))
        self.assertEqual(safe_float_scalar(float("inf"), default=7.0), 7.0)


class PosttrainFinalizeSentinelTest(unittest.TestCase):
    def test_lambda_fusion_finalize_has_no_nested_sum_or_float_helpers(self) -> None:
        fn_node = _function_node(posttrain._lambda_fusion_finalize)
        nested_names = {
            node.name
            for node in ast.walk(fn_node)
            if isinstance(node, ast.FunctionDef) and node is not fn_node
        }

        self.assertNotIn("_sum_terms", nested_names)
        self.assertNotIn("_to_float", nested_names)

    def test_finalize_direct_group_norm_delegates_base_payload_to_shared_owner(self) -> None:
        fn_node = _function_node(posttrain._finalize_direct_group_norm)
        call_targets = {
            ast.unparse(node.func)
            for node in ast.walk(fn_node)
            if isinstance(node, ast.Call)
        }
        arm_else_binops = [
            {
                name.id
                for name in ast.walk(node)
                if isinstance(name, ast.Name)
            }
            for node in ast.walk(fn_node)
            if isinstance(node, ast.BinOp)
        ]

        self.assertIn("loss_fn._compute_direct_pose_group_base_payload", call_targets)
        self.assertIn("loss_fn._compute_direct_pose_group_norm_shared", call_targets)
        self.assertFalse(any({"dir_arm_base", "dir_else_base"} <= names for names in arm_else_binops))


if __name__ == "__main__":
    unittest.main()
