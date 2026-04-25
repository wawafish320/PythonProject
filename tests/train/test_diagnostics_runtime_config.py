from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from train.diagnostics import DiagnosticsRuntimeConfig, resolve_diagnostics_runtime_config


class DiagnosticsRuntimeConfigTest(unittest.TestCase):
    def test_resolves_shared_geometry_and_free_carry_runtime(self) -> None:
        trainer = SimpleNamespace(
            _diag_scope="eval",
            rot6d_y_slice=slice(6, 12),
            rot6d_x_slice=slice(0, 12),
            rot6d_slice=slice(99, 105),
            yaw_x_slice=slice(30, 31),
            rootvel_x_slice=slice(12, 14),
            angvel_x_slice=slice(17, 23),
            eval_align_root0=False,
            eval_root_idx=None,
            root_idx=3,
            eval_up_axis=1,
            _up_axis=2,
            yaw_forward_axis=0,
            yaw_forward_axis_offset=0.5,
            bone_hz=None,
            fps=120.0,
            foot_contact_threshold=2.5,
            diag_input_stats=True,
            eval_angvel_beta=0.7,
            eval_angvel_mag_threshold=0.3,
            angvel_eps=1e-5,
            angvel_dir_threshold=0.2,
            loss_fn=SimpleNamespace(_rot6d_columns=("X", "Z")),
        )

        runtime_cfg = resolve_diagnostics_runtime_config(trainer, bone_names=("hip", "foot"))

        self.assertIsInstance(runtime_cfg, DiagnosticsRuntimeConfig)
        self.assertEqual(runtime_cfg.diag_scope, "eval")
        self.assertEqual(runtime_cfg.rot6d_y, slice(6, 12))
        self.assertEqual(runtime_cfg.rot6d_x, slice(0, 12))
        self.assertEqual(runtime_cfg.yaw_x_slice, slice(30, 31))
        self.assertEqual(runtime_cfg.rv_x, slice(12, 14))
        self.assertEqual(runtime_cfg.angvel_slice, slice(17, 23))
        self.assertFalse(runtime_cfg.eval_align_root)
        self.assertEqual(runtime_cfg.root_idx, 3)
        self.assertEqual(runtime_cfg.up_axis, 1)
        self.assertEqual(runtime_cfg.fps_eval, 120.0)
        self.assertEqual(runtime_cfg.contact_threshold, 2.5)
        self.assertTrue(runtime_cfg.diag_input_stats)
        self.assertEqual(runtime_cfg.yaw_forward_axis_offset, 0.5)
        self.assertEqual(runtime_cfg.mag_rel_beta, 0.7)
        self.assertEqual(runtime_cfg.mag_rel_threshold, 0.3)
        self.assertEqual(runtime_cfg.angvel_eps, 1e-5)
        self.assertEqual(runtime_cfg.angvel_dir_threshold, 0.2)
        self.assertEqual(runtime_cfg.bone_names, ("hip", "foot"))

    def test_defaults_preserve_safe_contract(self) -> None:
        trainer = SimpleNamespace(
            rot6d_y_slice="invalid",
            rot6d_x_slice="invalid",
            rot6d_slice=None,
            yaw_x_slice="invalid",
            rootvel_x_slice="invalid",
            angvel_x_slice="invalid",
            eval_root_idx=None,
            eval_up_axis=None,
            _up_axis=None,
            yaw_forward_axis=None,
            yaw_forward_axis_offset=None,
            bone_hz=None,
            fps=None,
        )

        runtime_cfg = resolve_diagnostics_runtime_config(trainer)

        self.assertEqual(runtime_cfg.diag_scope, "free_run")
        self.assertIsNone(runtime_cfg.rot6d_y)
        self.assertIsNone(runtime_cfg.rot6d_x)
        self.assertIsNone(runtime_cfg.yaw_x_slice)
        self.assertIsNone(runtime_cfg.rv_x)
        self.assertIsNone(runtime_cfg.angvel_slice)
        self.assertTrue(runtime_cfg.eval_align_root)
        self.assertEqual(runtime_cfg.root_idx, 0)
        self.assertEqual(runtime_cfg.up_axis, 2)
        self.assertEqual(runtime_cfg.fps_eval, 60.0)
        self.assertEqual(runtime_cfg.contact_threshold, 1.5)
        self.assertFalse(runtime_cfg.diag_input_stats)
        self.assertEqual(runtime_cfg.yaw_forward_axis_offset, 0.0)
        self.assertEqual(runtime_cfg.mag_rel_beta, 0.25)
        self.assertEqual(runtime_cfg.mag_rel_threshold, 0.10)
        self.assertEqual(runtime_cfg.angvel_eps, 1e-6)
        self.assertEqual(runtime_cfg.angvel_dir_threshold, 0.1)
        self.assertEqual(runtime_cfg.bone_names, ())


class DiagnosticsRuntimeResolverSentinelTest(unittest.TestCase):
    _FORBIDDEN_RUNTIME_ATTRS = frozenset(
        {
            "rot6d_y_slice",
            "rot6d_x_slice",
            "rot6d_slice",
            "yaw_x_slice",
            "rootvel_x_slice",
            "angvel_x_slice",
            "eval_root_idx",
            "eval_up_axis",
            "yaw_forward_axis_offset",
        }
    )

    @staticmethod
    def _parse_module() -> ast.AST:
        path = Path(__file__).resolve().parents[2] / "train/diagnostics.py"
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    @staticmethod
    def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

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

    @classmethod
    def _forbidden_runtime_reads(cls, node: ast.AST) -> list[tuple[str, int]]:
        reads: list[tuple[str, int]] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if not isinstance(child.func, ast.Name) or child.func.id not in {"getattr", "hasattr"}:
                continue
            if len(child.args) < 2:
                continue
            attr_arg = child.args[1]
            lineno = getattr(child, "lineno", None)
            if not isinstance(attr_arg, ast.Constant) or not isinstance(attr_arg.value, str) or lineno is None:
                continue
            if attr_arg.value in cls._FORBIDDEN_RUNTIME_ATTRS:
                reads.append((attr_arg.value, int(lineno)))
        return reads

    def test_diagnose_free_run_uses_shared_runtime_resolver(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "diagnose_free_run")
        call_names = self._call_names(node)

        self.assertIn("resolve_diagnostics_runtime_config", call_names)
        self.assertEqual(self._forbidden_runtime_reads(node), [])

    def test_history_drift_geo_stats_use_shared_runtime_resolver(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "_compute_history_drift_geo_local_stats")
        call_names = self._call_names(node)

        self.assertIn("resolve_diagnostics_runtime_config", call_names)
        self.assertEqual(self._forbidden_runtime_reads(node), [])

    def test_step_debug_record_uses_shared_runtime_resolver(self) -> None:
        tree = self._parse_module()
        node = self._find_function(tree, "collect_freerun_step_debug_record")
        call_names = self._call_names(node)

        self.assertIn("resolve_diagnostics_runtime_config", call_names)
        self.assertEqual(self._forbidden_runtime_reads(node), [])


if __name__ == "__main__":
    unittest.main()
