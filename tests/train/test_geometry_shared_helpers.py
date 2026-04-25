from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.geometry import (
    apply_direct_leg_so3_to_raw,
    blend_rot6d_raw_with_lambda,
    compose_delta_raw_to_next,
    matrix_to_rot6d,
    rot6d_to_matrix,
    so3_exp_map,
    so3_geodesic_blend,
    so3_log_map,
)


ATOL = 1e-6
RTOL = 1e-6
FIXTURE_DIR = Path(__file__).with_name("fixtures") / "baseline_numeric"


def _load_fixture(file_name: str) -> dict:
    return torch.load(FIXTURE_DIR / file_name, map_location="cpu")


def _spec_to_slice(spec: list[int | None]) -> slice:
    return slice(spec[0], spec[1], spec[2])


def _compose_delta_raw_from_fixture(fixture: dict) -> torch.Tensor:
    inputs = fixture["inputs"]
    std_y = torch.tensor(fixture["config"]["std_y"], dtype=inputs["delta_norm"].dtype)
    while std_y.dim() < inputs["delta_norm"].dim():
        std_y = std_y.unsqueeze(0)
    return inputs["delta_norm"] * std_y.clamp_min(1e-6)


def _denorm_from_fixture(fixture: dict) -> torch.Tensor:
    direct_norm = fixture["inputs"]["direct_norm"]
    mean = torch.tensor(fixture["config"]["denorm_mean"], dtype=direct_norm.dtype)
    std = torch.tensor(fixture["config"]["denorm_std"], dtype=direct_norm.dtype)
    if direct_norm.dim() == 3 and direct_norm.size(1) == 1:
        direct_norm = direct_norm[:, 0]
    return direct_norm * std + mean


class GeometrySharedHelpersTest(unittest.TestCase):
    def test_compose_delta_raw_to_next_matches_baseline_without_omega(self) -> None:
        fixture = _load_fixture("compose_delta_raw_seed000.pt")
        config = fixture["config"]
        actual = compose_delta_raw_to_next(
            fixture["inputs"]["y_prev_raw"],
            _compose_delta_raw_from_fixture(fixture),
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
            omega_hat=None,
            reproject=False,
        )
        torch.testing.assert_close(actual, fixture["outputs"]["y_next_raw"], atol=ATOL, rtol=RTOL)

    def test_compose_delta_raw_to_next_matches_baseline_with_bj3_omega(self) -> None:
        fixture = _load_fixture("compose_delta_raw_seed001.pt")
        config = fixture["config"]
        omega_hat = fixture["inputs"]["omega_hat"][:, 0]
        gate_val = float(torch.sigmoid(torch.tensor(config["so3_corr_gate_logit"])).item())
        actual = compose_delta_raw_to_next(
            fixture["inputs"]["y_prev_raw"],
            _compose_delta_raw_from_fixture(fixture),
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
            omega_hat=omega_hat,
            gate_val=gate_val,
            max_deg=float(config["so3_corr_max_deg"]),
            reproject=False,
        )
        torch.testing.assert_close(actual, fixture["outputs"]["y_next_raw"], atol=ATOL, rtol=RTOL)

    def test_compose_delta_raw_to_next_matches_baseline_with_b1j3_omega(self) -> None:
        fixture = _load_fixture("compose_delta_raw_seed001.pt")
        config = fixture["config"]
        gate_val = float(torch.sigmoid(torch.tensor(config["so3_corr_gate_logit"])).item())
        actual = compose_delta_raw_to_next(
            fixture["inputs"]["y_prev_raw"],
            _compose_delta_raw_from_fixture(fixture),
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
            omega_hat=fixture["inputs"]["omega_hat"],
            gate_val=gate_val,
            max_deg=float(config["so3_corr_max_deg"]),
            reproject=False,
        )
        torch.testing.assert_close(actual, fixture["outputs"]["y_next_raw"], atol=ATOL, rtol=RTOL)

    def test_so3_geodesic_blend_matches_canonical_formula(self) -> None:
        batch_size = 2
        joint_count = 3
        generator = torch.Generator(device="cpu")
        generator.manual_seed(101)
        R_inc = so3_exp_map(torch.randn(batch_size, joint_count, 3, generator=generator) * 0.2)
        R_dir = so3_exp_map(torch.randn(batch_size, joint_count, 3, generator=generator) * 0.3)
        lam = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]], dtype=torch.float32)

        actual = so3_geodesic_blend(R_inc, R_dir, lam)
        R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
        omega = so3_log_map(R_res)
        expected = torch.matmul(so3_exp_map(omega * lam.unsqueeze(-1)), R_inc)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    def test_so3_geodesic_blend_rejects_noncanonical_lambda_shape(self) -> None:
        R_inc = so3_exp_map(torch.zeros(2, 3, 3, dtype=torch.float32))
        R_dir = so3_exp_map(torch.zeros(2, 3, 3, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "lam must have canonical shape"):
            so3_geodesic_blend(R_inc, R_dir, torch.zeros(2, 1, 3, dtype=torch.float32))

    def test_blend_rot6d_raw_with_lambda_matches_baseline_canonical_bj_fixture(self) -> None:
        fixture = _load_fixture("lambda_blend_seed011.pt")
        config = fixture["config"]
        actual = blend_rot6d_raw_with_lambda(
            fixture["inputs"]["y_inc_raw"],
            _denorm_from_fixture(fixture),
            fixture["inputs"]["lambda_fusion"],
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
        )
        torch.testing.assert_close(actual, fixture["outputs"]["y_blend_raw"], atol=ATOL, rtol=RTOL)

    def test_apply_direct_leg_so3_to_raw_matches_baseline_without_exclusion(self) -> None:
        fixture = _load_fixture("direct_leg_adjust_seed020.pt")
        config = fixture["config"]
        actual, R_leg_base_used, idx_use, keep_mask = apply_direct_leg_so3_to_raw(
            fixture["inputs"]["direct_raw_base"],
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
            omega_leg=fixture["inputs"]["ret"]["direct_leg_omega"],
            leg_idx=torch.tensor(config["leg_idx"], dtype=torch.long),
            exclude_idx=None,
            stopgrad_base=bool(config["stopgrad_base"]),
        )
        torch.testing.assert_close(actual, fixture["outputs"]["direct_raw_base"], atol=ATOL, rtol=RTOL)
        self.assertIsNone(keep_mask)
        self.assertEqual(idx_use.tolist(), config["leg_idx"])
        self.assertEqual(tuple(R_leg_base_used.shape), (2, len(config["leg_idx"]), 3, 3))

    def test_apply_direct_leg_so3_to_raw_matches_baseline_with_exclusion(self) -> None:
        fixture = _load_fixture("direct_leg_adjust_seed021.pt")
        config = fixture["config"]
        actual, R_leg_base_used, idx_use, keep_mask = apply_direct_leg_so3_to_raw(
            fixture["inputs"]["direct_raw_base"],
            rot_slice=_spec_to_slice(config["rot_slice"]),
            columns=tuple(config["columns"]),
            omega_leg=fixture["inputs"]["ret"]["direct_leg_omega"],
            leg_idx=torch.tensor(config["leg_idx"], dtype=torch.long),
            exclude_idx=int(config["root_idx"]),
            stopgrad_base=bool(config["stopgrad_base"]),
        )
        torch.testing.assert_close(actual, fixture["outputs"]["direct_raw_base"], atol=ATOL, rtol=RTOL)
        self.assertEqual(idx_use.tolist(), [2, 3])
        self.assertIsNotNone(keep_mask)
        self.assertEqual(keep_mask.tolist(), [False, True, True])
        self.assertEqual(tuple(R_leg_base_used.shape), (2, 2, 3, 3))

    def test_apply_direct_leg_so3_to_raw_root_only_exclusion_returns_unchanged_empty_idx(self) -> None:
        batch_size = 2
        joint_count = 2
        columns = ("X", "Z")
        direct_raw = matrix_to_rot6d(
            so3_exp_map(torch.zeros(batch_size, joint_count, 3, dtype=torch.float32)),
            columns=columns,
        ).reshape(batch_size, joint_count * 6)
        actual, R_leg_base_used, idx_use, keep_mask = apply_direct_leg_so3_to_raw(
            direct_raw,
            rot_slice=slice(0, joint_count * 6),
            columns=columns,
            omega_leg=torch.ones(batch_size, 1, 3, dtype=torch.float32) * 0.05,
            leg_idx=torch.tensor([0], dtype=torch.long),
            exclude_idx=0,
            stopgrad_base=False,
        )
        torch.testing.assert_close(actual, direct_raw, atol=ATOL, rtol=RTOL)
        self.assertEqual(idx_use.numel(), 0)
        self.assertIsNotNone(keep_mask)
        self.assertEqual(keep_mask.tolist(), [False])
        self.assertEqual(tuple(R_leg_base_used.shape), (batch_size, 0, 3, 3))


if __name__ == "__main__":
    unittest.main()
