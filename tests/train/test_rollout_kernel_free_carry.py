from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

import torch

from train.geometry import angvel_vec_from_R_seq, matrix_to_rot6d
from train.rollout_kernel import apply_free_carry_raw


def _rot_z(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


class RolloutKernelFreeCarryTest(unittest.TestCase):
    def test_apply_free_carry_raw_updates_rot_angvel_root_velocity_and_position(self) -> None:
        columns = ("X", "Z")
        prev_R = torch.eye(3, dtype=torch.float32).repeat(1, 2, 1, 1)
        next_R = prev_R.clone()
        next_R[:, 0] = _rot_z(math.pi / 2.0)

        prev_6d = matrix_to_rot6d(prev_R, columns=columns).reshape(1, -1)
        next_6d = matrix_to_rot6d(next_R, columns=columns).reshape(1, -1)

        x_prev = torch.zeros(1, 23, dtype=torch.float32)
        x_prev[..., 0:12] = prev_6d
        x_prev[..., 14:17] = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)

        y_next_raw = torch.full((1, 18), -99.0, dtype=torch.float32)
        y_next_raw[..., 3:15] = next_6d

        out = apply_free_carry_raw(
            x_prev=x_prev,
            y_next_raw=y_next_raw,
            cond_next_raw=torch.tensor([[7.0, 8.0, 3.0, 4.0, 2.0]], dtype=torch.float64),
            rot6d_x_slice=slice(0, 12),
            rot6d_y_slice=slice(3, 15),
            angvel_x_slice=slice(17, 23),
            rootvel_x_slice=slice(12, 14),
            rootpos_x_slice=slice(14, 17),
            bone_hz=20.0,
            columns=columns,
        )

        expected_angvel = angvel_vec_from_R_seq(
            torch.stack([prev_R, next_R], dim=1),
            fps=20.0,
        )[:, -1].reshape(1, -1)

        torch.testing.assert_close(out[..., 0:12], next_6d)
        torch.testing.assert_close(out[..., 12:14], torch.tensor([[1.2, 1.6]], dtype=torch.float32))
        torch.testing.assert_close(out[..., 14:17], torch.tensor([[10.06, 20.08, 30.0]], dtype=torch.float32))
        torch.testing.assert_close(out[..., 17:23], expected_angvel, atol=1e-5, rtol=1e-5)

    def test_apply_free_carry_raw_expands_1d_cond_to_batch(self) -> None:
        out = apply_free_carry_raw(
            x_prev=torch.zeros(2, 10, dtype=torch.float32),
            y_next_raw=torch.zeros(2, 6, dtype=torch.float32),
            cond_next_raw=torch.tensor([5.0, 3.0, 4.0, 2.0], dtype=torch.float32),
            rot6d_x_slice=slice(0, 6),
            rot6d_y_slice=slice(0, 6),
            angvel_x_slice=None,
            rootvel_x_slice=slice(6, 8),
            rootpos_x_slice=slice(8, 10),
            bone_hz=20.0,
            columns=("X", "Z"),
        )

        expected_root_vel = torch.tensor([[1.2, 1.6], [1.2, 1.6]], dtype=torch.float32)
        expected_root_pos = torch.tensor([[0.06, 0.08], [0.06, 0.08]], dtype=torch.float32)
        torch.testing.assert_close(out[..., 6:8], expected_root_vel)
        torch.testing.assert_close(out[..., 8:10], expected_root_pos)

    def test_apply_free_carry_raw_raises_for_invalid_inputs(self) -> None:
        base_kwargs = dict(
            x_prev=torch.zeros(1, 12, dtype=torch.float32),
            y_next_raw=torch.zeros(1, 12, dtype=torch.float32),
            cond_next_raw=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
            rot6d_x_slice=slice(0, 6),
            rot6d_y_slice=slice(0, 6),
            angvel_x_slice=None,
            rootvel_x_slice=slice(6, 8),
            rootpos_x_slice=slice(8, 10),
            bone_hz=20.0,
            columns=("X", "Z"),
        )

        cases = (
            {"name": "missing_rot6d_x_slice", "override": {"rot6d_x_slice": None}},
            {"name": "missing_rot6d_y_slice", "override": {"rot6d_y_slice": None}},
            {"name": "mismatched_rot6d_slices", "override": {"rot6d_y_slice": slice(0, 12), "y_next_raw": torch.zeros(1, 12, dtype=torch.float32)}},
            {"name": "cond_next_raw_none", "override": {"cond_next_raw": None}},
            {"name": "cond_dim_too_small", "override": {"cond_next_raw": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}},
            {"name": "missing_rootvel_x_slice", "override": {"rootvel_x_slice": None}},
            {"name": "missing_rootpos_x_slice", "override": {"rootpos_x_slice": None}},
            {"name": "missing_columns", "override": {"columns": ()}},
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                kwargs = dict(base_kwargs)
                kwargs.update(case["override"])
                with self.assertRaises(ValueError):
                    apply_free_carry_raw(**kwargs)


class FreeCarryRefactorSentinelTest(unittest.TestCase):
    _FORBIDDEN_RUNTIME_ATTRS = frozenset(
        {
            "rot6d_x_slice",
            "rot6d_y_slice",
            "rot6d_slice",
            "angvel_x_slice",
            "rootvel_x_slice",
            "rootpos_x_slice",
            "bone_hz",
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
    def _call_lines(node: ast.AST, name: str) -> list[int]:
        lines: list[int] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if (
                (isinstance(func, ast.Name) and func.id == name)
                or (isinstance(func, ast.Attribute) and func.attr == name)
            ):
                lines.append(int(child.lineno))
        return sorted(lines)

    @classmethod
    def _forbidden_runtime_reads(
        cls,
        node: ast.AST,
        *,
        start_line: int,
        end_line: int,
    ) -> list[tuple[str, int]]:
        reads: list[tuple[str, int]] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            lineno = getattr(child, "lineno", None)
            if lineno is None or lineno < start_line or lineno > end_line:
                continue
            if not isinstance(child.func, ast.Name) or child.func.id != "getattr":
                continue
            if len(child.args) < 2:
                continue
            attr_arg = child.args[1]
            if not isinstance(attr_arg, ast.Constant) or not isinstance(attr_arg.value, str):
                continue
            if attr_arg.value in cls._FORBIDDEN_RUNTIME_ATTRS:
                reads.append((attr_arg.value, int(lineno)))
        return reads

    def test_repo_no_private_apply_free_carry_definition_or_calls(self) -> None:
        root = self._repo_root()
        def_pattern = "".join(("def ", "_apply_free_", "carry"))
        call_pattern = "".join((".", "_apply_free_", "carry("))
        offenders: list[str] = []

        for folder in ("train", "tests", "tools"):
            for path in (root / folder).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if def_pattern in text or call_pattern in text:
                    offenders.append(str(path.relative_to(root)))

        self.assertEqual(offenders, [])

    def test_free_carry_callers_use_runtime_resolver_without_local_reads(self) -> None:
        for rel_path, expected_calls in (
            ("train/rollout_kernel.py", 2),
            ("train/eval_utils.py", 1),
            ("train/validate/run_freerun_cycles.py", 2),
            ("train/validate/run_gait_speed_scaling_whitebox.py", 1),
        ):
            with self.subTest(path=rel_path):
                tree = self._parse_module(rel_path)
                apply_lines = self._call_lines(tree, "apply_free_carry_raw")
                resolver_lines = self._call_lines(tree, "resolve_free_carry_runtime_config")

                self.assertEqual(len(apply_lines), expected_calls)
                for line in apply_lines:
                    start_line = max(1, line - 30)
                    self.assertTrue(
                        any(start_line <= resolver_line <= line for resolver_line in resolver_lines),
                        f"{rel_path}:{line} should resolve free-carry runtime before apply_free_carry_raw",
                    )
                    self.assertEqual(
                        self._forbidden_runtime_reads(tree, start_line=start_line, end_line=line),
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
