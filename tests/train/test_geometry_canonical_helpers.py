from __future__ import annotations

import math
from pathlib import Path
import unittest

import torch

from train.geometry import (
    matrix_to_rot6d,
    reproject_cond_to_local_frame,
    root_yaw_from_raw_rot6d,
    root_yaw_from_rot6d_torch,
)

_REMOVED_HELPERS = (
    "_infer_root_yaw" + "_from_rot6d",
    "_reproject_cond_to" + "_local_frame",
    "_normalize_cond" + "_from_raw",
)


def _yaw_matrix_z(yaw: float) -> torch.Tensor:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return torch.tensor(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


class GeometryCanonicalHelpersTest(unittest.TestCase):
    def test_root_yaw_from_raw_rot6d_matches_tensor_helper(self) -> None:
        root_rot = torch.stack([_yaw_matrix_z(0.2), _yaw_matrix_z(1.1)], dim=0).unsqueeze(0)
        rot6d = matrix_to_rot6d(root_rot)
        raw = torch.cat(
            [
                torch.tensor([[99.0]], dtype=torch.float32),
                rot6d.reshape(1, -1),
                torch.tensor([[-7.0]], dtype=torch.float32),
            ],
            dim=-1,
        )

        actual = root_yaw_from_raw_rot6d(
            raw,
            rot_slice=slice(1, 13),
            root_idx=1,
            up_axis=2,
            forward_axis=0,
            offset=0.15,
            reproject=True,
        )
        expected = root_yaw_from_rot6d_torch(
            rot6d,
            root_idx=1,
            up_axis=2,
            forward_axis=0,
            offset=0.15,
            reproject=True,
        )

        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, expected)

    def test_root_yaw_from_raw_rot6d_returns_none_on_invalid_input(self) -> None:
        self.assertIsNone(
            root_yaw_from_raw_rot6d(
                None,
                rot_slice=slice(0, 6),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
                reproject=True,
            )
        )
        self.assertIsNone(
            root_yaw_from_raw_rot6d(
                torch.zeros(6, dtype=torch.float32),
                rot_slice=slice(0, 6),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
                reproject=True,
            )
        )
        self.assertIsNone(
            root_yaw_from_raw_rot6d(
                torch.zeros((1, 7), dtype=torch.float32),
                rot_slice=slice(0, 7),
                root_idx=0,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
                reproject=True,
            )
        )
        self.assertIsNone(
            root_yaw_from_raw_rot6d(
                torch.zeros((1, 12), dtype=torch.float32),
                rot_slice=slice(0, 12),
                root_idx=2,
                up_axis=2,
                forward_axis=0,
                offset=0.0,
                reproject=True,
            )
        )

    def test_cond_reprojection_rotates_direction_and_preserves_tail(self) -> None:
        cond_raw = torch.tensor(
            [[5.0, -2.0, 1.0, 0.0, 3.0], [6.0, -3.0, 0.0, 1.0, 4.0]],
            dtype=torch.float32,
        )
        yaw_gt = torch.zeros(2, dtype=torch.float32)
        yaw_pred = torch.full((2,), math.pi / 2.0, dtype=torch.float32)

        actual = reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred)
        expected = torch.tensor(
            [[5.0, -2.0, 0.0, -1.0, 3.0], [6.0, -3.0, 1.0, 0.0, 4.0]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_cond_reprojection_rejects_short_cond(self) -> None:
        with self.assertRaises(ValueError):
            reproject_cond_to_local_frame(
                torch.zeros((1, 2), dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
            )

    def test_runtime_files_use_shared_canonical_helpers(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        expected_tokens = {
            "train/rollout_kernel.py": (
                "prepare_cond_input_from_raw",
                "root_yaw_from_raw_rot6d",
                "reproject_cond_to_local_frame",
                "normalize_cond_tensor",
            ),
            "train/eval_utils.py": ("prepare_cond_input_from_raw",),
            "train/validate/run_freerun_cycles.py": ("prepare_cond_input_from_raw",),
            "train/validate/run_gait_speed_scaling_whitebox.py": ("normalize_cond_tensor",),
            "train/posttrain.py": ("root_yaw_from_raw_rot6d",),
        }

        for rel_path, tokens in expected_tokens.items():
            text = (repo_root / rel_path).read_text(encoding="utf-8")
            for token in tokens:
                self.assertIn(token, text, msg=f"{rel_path} should reference {token}")

    def test_repo_no_longer_defines_or_calls_removed_trainer_helpers(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        forbidden = tuple(
            [f"def {name}" for name in _REMOVED_HELPERS]
            + [f".{name}(" for name in _REMOVED_HELPERS]
        )
        skip_paths = {Path(__file__).resolve()}

        for rel_root in ("train", "tests", "tools"):
            for path in (repo_root / rel_root).rglob("*.py"):
                if path in skip_paths:
                    continue
                text = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, text, msg=f"Forbidden token {token!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
