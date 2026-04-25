from __future__ import annotations

import ast
from pathlib import Path
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


class FreeRunCyclesRuntimeResolverSentinelTest(unittest.TestCase):
    _FORBIDDEN_RUNTIME_ATTRS = frozenset(
        {
            "rot6d_y_slice",
            "rot6d_x_slice",
            "rot6d_slice",
            "eval_root_idx",
            "eval_up_axis",
            "yaw_forward_axis",
            "yaw_forward_axis_offset",
            "cond_norm_clip",
            "pose_hist_len",
            "pose_hist_dim",
            "_pose_hist_params",
            "rootvel_x_slice",
            "rootpos_x_slice",
            "angvel_x_slice",
        }
    )

    @staticmethod
    def _parse_module() -> ast.AST:
        path = Path(__file__).resolve().parents[2] / "train/validate/run_freerun_cycles.py"
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

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
    def _call_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                names.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.append(child.func.attr)
        return names

    @staticmethod
    def _find_prepare_call(node: ast.AST, cond_arg_name: str) -> ast.Call:
        return next(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "prepare_cond_input_from_raw"
            and any(
                keyword.arg == "cond_raw_step"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == cond_arg_name
                for keyword in child.keywords
            )
        )

    def test_main_cond_path_uses_runtime_resolver_without_local_parser(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_run_freerun_cycles")
        call_names = self._call_names(node)
        prepare_call = self._find_prepare_call(node, "cond_raw_step")

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertEqual(
            self._forbidden_runtime_reads(
                node,
                start_line=max(1, prepare_call.lineno - 45),
                end_line=prepare_call.lineno,
            ),
            [],
        )

    def test_donor_cond_path_uses_runtime_resolver_without_local_parser(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_advance_pose_hist_hybrid_donor_step")
        call_names = self._call_names(node)
        prepare_call = self._find_prepare_call(node, "cond_raw_step_shared")

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertEqual(
            self._forbidden_runtime_reads(
                node,
                start_line=max(1, prepare_call.lineno - 45),
                end_line=prepare_call.lineno,
            ),
            [],
        )

    def test_eval_pose_hist_init_uses_shared_runtime_resolver(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_init_eval_pose_hist_state")
        call_names = self._call_names(node)

        self.assertIn("resolve_pose_hist_runtime_config", call_names)
        pose_hist_reads = [
            read
            for read in self._forbidden_runtime_reads(node)
            if read[0] in {"pose_hist_len", "pose_hist_dim", "_pose_hist_params"}
        ]
        self.assertEqual(pose_hist_reads, [])

    def test_run_freerun_cycles_pose_hist_setup_uses_shared_runtime_resolver(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_run_freerun_cycles")
        resolver_call = next(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "resolve_pose_hist_runtime_config"
        )
        pose_hist_reads = [
            read
            for read in self._forbidden_runtime_reads(
                node,
                start_line=max(1, resolver_call.lineno - 5),
                end_line=resolver_call.lineno + 20,
            )
            if read[0] in {"pose_hist_len", "pose_hist_dim"}
        ]
        self.assertEqual(pose_hist_reads, [])

    def test_run_freerun_cycles_uses_shared_free_carry_and_rot_runtime(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_run_freerun_cycles")
        call_names = self._call_names(node)

        self.assertIn("resolve_rollout_cond_runtime_config", call_names)
        self.assertIn("resolve_free_carry_runtime_config", call_names)
        self.assertEqual(self._forbidden_runtime_reads(node), [])


if __name__ == "__main__":
    unittest.main()
