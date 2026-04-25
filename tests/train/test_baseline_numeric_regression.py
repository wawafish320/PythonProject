from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.geometry import matrix_to_rot6d, rot6d_to_matrix, so3_exp_map
from train.posttrain import _lambda_rollout_apply_direct_leg_adjustments
from train.training_MPL import Trainer


ATOL = 1e-6
RTOL = 1e-6
FIXTURE_DIR = Path(__file__).with_name("fixtures") / "baseline_numeric"


class _AffineNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean.detach().cpu().clone().to(dtype=torch.float32)
        self._std = std.detach().cpu().clone().to(dtype=torch.float32)

    def denorm(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=y.device, dtype=y.dtype)
        std = self._std.to(device=y.device, dtype=y.dtype)
        return y * std + mean


def _slice_to_spec(s: slice) -> list[int | None]:
    return [s.start, s.stop, s.step]


def _spec_to_slice(spec: list[int | None]) -> slice:
    return slice(spec[0], spec[1], spec[2])


def _to_cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _to_cpu_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(value) for value in obj)
    return obj


def _randn(seed: int, shape: tuple[int, ...], *, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32) * float(scale)


def _rand(seed: int, shape: tuple[int, ...], *, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return low + (high - low) * torch.rand(shape, generator=generator, dtype=torch.float32)


def _make_rot6d_batch(*, seed: int, batch_size: int, joint_count: int, columns: tuple[str, str]) -> torch.Tensor:
    omega = _randn(seed, (batch_size, joint_count, 3), scale=0.20)
    return matrix_to_rot6d(so3_exp_map(omega), columns=columns).reshape(batch_size, joint_count * 6)


def _make_trainer(*, fixture_config: dict) -> Trainer:
    rot_slice = _spec_to_slice(fixture_config["rot_slice"])
    feature_dim = int(fixture_config["feature_dim"])
    columns = tuple(fixture_config["columns"])
    denorm_mean = torch.tensor(fixture_config.get("denorm_mean", [0.0] * feature_dim), dtype=torch.float32)
    denorm_std = torch.tensor(fixture_config.get("denorm_std", [1.0] * feature_dim), dtype=torch.float32)

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.rot6d_y_slice = rot_slice
    trainer.rot6d_slice = rot_slice
    trainer._norm_cache = {}
    trainer.StdY = fixture_config.get("std_y")
    trainer.std_y = fixture_config.get("std_y")
    trainer.normalizer = _AffineNormalizer(denorm_mean, denorm_std)
    trainer.loss_fn = SimpleNamespace(
        _rot6d_columns=columns,
        root_idx=int(fixture_config.get("root_idx", 0)),
    )
    trainer.model = SimpleNamespace(
        so3_corr_gate_logit=torch.tensor(
            float(fixture_config.get("so3_corr_gate_logit", 0.0)),
            dtype=torch.float32,
        ),
    )
    trainer.so3_corr_max_deg = float(fixture_config.get("so3_corr_max_deg", 20.0))
    return trainer


def _make_direct_model(*, fixture_config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        direct_pose_leg_joint_idx_tensor=torch.tensor(fixture_config["leg_idx"], dtype=torch.long),
        direct_pose_leg_stopgrad_main=bool(fixture_config["stopgrad_base"]),
        direct_pose_leg_joint_names=list(fixture_config["leg_joint_names"]),
    )


def _empty_direct_term_lists() -> dict[str, list[torch.Tensor]]:
    return {
        "leg_align_terms": [],
        "leg_align_frac_terms": [],
        "leg_align_joint_num_terms": [],
        "leg_align_joint_den_terms": [],
        "leg_align_joint_frac_terms": [],
        "leg_align_distal_terms": [],
        "leg_align_distal_frac_terms": [],
        "leg_align_proximal_terms": [],
        "leg_align_proximal_frac_terms": [],
        "leg_align_anchor_terms": [],
        "leg_align_anchor_frac_terms": [],
        "leg_gate_sup_terms": [],
        "leg_gate_sup_tgt_frac_terms": [],
        "leg_gate_sup_pred_mean_terms": [],
    }


def _build_compose_fixture(
    *,
    file_name: str,
    seed: int,
    with_omega_hat: bool,
) -> dict:
    batch_size = 2
    joint_count = 2
    tail_dim = 2
    feature_dim = joint_count * 6 + tail_dim
    rot_slice = slice(0, joint_count * 6)
    columns = ("X", "Z")
    std_y = torch.linspace(0.75, 1.25, feature_dim, dtype=torch.float32).tolist()

    y_prev_raw = torch.cat(
        [
            _make_rot6d_batch(seed=seed + 11, batch_size=batch_size, joint_count=joint_count, columns=columns),
            _randn(seed + 12, (batch_size, tail_dim), scale=0.08),
        ],
        dim=-1,
    )
    delta_norm = torch.cat(
        [
            _randn(seed + 13, (batch_size, joint_count * 6), scale=0.04),
            _randn(seed + 14, (batch_size, tail_dim), scale=0.03),
        ],
        dim=-1,
    )

    fixture_config = {
        "case_name": file_name.removesuffix(".pt"),
        "feature_dim": feature_dim,
        "rot_slice": _slice_to_spec(rot_slice),
        "columns": list(columns),
        "std_y": std_y,
        "compose_reproject_result": False,
        "so3_corr_gate_logit": -0.45,
        "so3_corr_max_deg": 17.5,
    }
    trainer = _make_trainer(fixture_config=fixture_config)

    compose_kwargs = {}
    omega_hat = None
    if with_omega_hat:
        omega_hat = _randn(seed + 15, (batch_size, 1, joint_count, 3), scale=0.07)
        compose_kwargs["omega_hat"] = omega_hat
    output = trainer._compose_delta_to_raw(y_prev_raw, delta_norm, **compose_kwargs)

    return {
        "file_name": file_name,
        "target": "Trainer._compose_delta_to_raw",
        "seed": seed,
        "config": fixture_config,
        "inputs": {
            "y_prev_raw": y_prev_raw,
            "delta_norm": delta_norm,
            "omega_hat": omega_hat,
        },
        "outputs": {
            "y_next_raw": output,
        },
    }


def _build_lambda_fixture(
    *,
    file_name: str,
    seed: int,
    lambda_shape: str,
    direct_norm_shape: str,
) -> dict:
    batch_size = 2
    joint_count = 3
    tail_dim = 2
    feature_dim = joint_count * 6 + tail_dim
    rot_slice = slice(0, joint_count * 6)
    columns = ("X", "Z")

    y_inc_raw = torch.cat(
        [
            _make_rot6d_batch(seed=seed + 21, batch_size=batch_size, joint_count=joint_count, columns=columns),
            _randn(seed + 22, (batch_size, tail_dim), scale=0.06),
        ],
        dim=-1,
    )
    direct_norm_base = _randn(seed + 23, (batch_size, feature_dim), scale=0.18)
    if direct_norm_shape == "B1D":
        direct_norm = direct_norm_base.unsqueeze(1)
    else:
        direct_norm = direct_norm_base

    lambda_base = _rand(seed + 24, (batch_size, joint_count), low=0.15, high=0.85)
    if lambda_shape == "B1":
        lambda_fusion = lambda_base.mean(dim=-1, keepdim=True)
    elif lambda_shape == "BJ":
        lambda_fusion = lambda_base
    elif lambda_shape == "B1J":
        lambda_fusion = lambda_base.unsqueeze(1)
    elif lambda_shape == "BJ1":
        lambda_fusion = lambda_base.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported lambda_shape={lambda_shape}")

    fixture_config = {
        "case_name": file_name.removesuffix(".pt"),
        "feature_dim": feature_dim,
        "rot_slice": _slice_to_spec(rot_slice),
        "columns": list(columns),
        "denorm_mean": torch.linspace(-0.25, 0.25, feature_dim, dtype=torch.float32).tolist(),
        "denorm_std": torch.linspace(0.85, 1.15, feature_dim, dtype=torch.float32).tolist(),
        "lambda_shape": lambda_shape,
        "direct_norm_shape": direct_norm_shape,
    }
    trainer = _make_trainer(fixture_config=fixture_config)
    output = trainer._apply_lambda_fusion_to_raw(
        y_inc_raw,
        direct_norm=direct_norm,
        lambda_fusion=lambda_fusion,
    )

    return {
        "file_name": file_name,
        "target": "Trainer._apply_lambda_fusion_to_raw",
        "seed": seed,
        "config": fixture_config,
        "inputs": {
            "y_inc_raw": y_inc_raw,
            "direct_norm": direct_norm,
            "lambda_fusion": lambda_fusion,
        },
        "outputs": {
            "y_blend_raw": output,
        },
    }


def _build_direct_leg_fixture(
    *,
    file_name: str,
    seed: int,
    leg_idx: list[int],
    root_idx: int,
    stopgrad_base: bool,
    align_mode: str,
    omega_shape: str,
) -> dict:
    batch_size = 2
    joint_count = 4
    feature_dim = joint_count * 6
    rot_slice = slice(0, feature_dim)
    columns = ("X", "Z")

    direct_raw_base = _make_rot6d_batch(seed=seed + 31, batch_size=batch_size, joint_count=joint_count, columns=columns)
    r_gt_raw = _make_rot6d_batch(seed=seed + 32, batch_size=batch_size, joint_count=joint_count, columns=columns)
    r_gt = rot6d_to_matrix(r_gt_raw.view(batch_size, joint_count, 6), columns=columns)

    leg_joint_names = ["root", "calf_l", "foot_l"] if root_idx in leg_idx else ["thigh_r", "foot_r"]
    leg_count = len(leg_idx)
    omega_base = _randn(seed + 33, (batch_size, leg_count, 3), scale=0.05)
    if omega_shape == "B1L3":
        direct_leg_omega = omega_base.unsqueeze(1)
        gate_logits = _randn(seed + 34, (batch_size, 1, leg_count), scale=0.6)
    elif omega_shape == "BL3":
        direct_leg_omega = omega_base
        gate_logits = _randn(seed + 34, (batch_size, leg_count), scale=0.6)
    else:
        raise ValueError(f"Unsupported omega_shape={omega_shape}")

    fixture_config = {
        "case_name": file_name.removesuffix(".pt"),
        "feature_dim": feature_dim,
        "rot_slice": _slice_to_spec(rot_slice),
        "columns": list(columns),
        "root_idx": root_idx,
        "leg_idx": list(leg_idx),
        "leg_joint_names": leg_joint_names,
        "stopgrad_base": bool(stopgrad_base),
        "direct_pose_leg_align_weight": 1.0,
        "direct_pose_leg_align_oracle_min_deg": 1.25,
        "direct_pose_leg_align_oracle_weight_deg": 30.0,
        "direct_pose_leg_align_mode": align_mode,
        "direct_pose_leg_align_mag_weight": 1.2,
        "direct_pose_leg_align_res_weight": 0.6,
        "direct_pose_leg_align_sign_weight": 0.2,
        "direct_pose_leg_align_cos_thresh": 0.35,
        "direct_pose_leg_align_target_joints": "foot_l,calf_l" if root_idx in leg_idx else "thigh_r,foot_r",
        "direct_pose_leg_align_anchor_joints": "calf_l" if root_idx in leg_idx else "thigh_r",
        "direct_pose_leg_align_anchor_weight": 0.15,
        "direct_pose_leg_gate_sup_weight": 0.3,
        "omega_shape": omega_shape,
    }
    trainer = _make_trainer(fixture_config=fixture_config)
    model = _make_direct_model(fixture_config=fixture_config)
    term_lists = _empty_direct_term_lists()

    output = _lambda_rollout_apply_direct_leg_adjustments(
        trainer=trainer,
        model=model,
        ret={
            "direct_leg_omega": direct_leg_omega,
            "direct_leg_gate_logits": gate_logits,
        },
        direct_raw_base=direct_raw_base,
        R_gt=r_gt,
        B=batch_size,
        J=joint_count,
        device=torch.device("cpu"),
        dtype=torch.float32,
        columns=columns,
        rot_slice=rot_slice,
        rot_len=feature_dim,
        direct_pose_leg_align_weight=float(fixture_config["direct_pose_leg_align_weight"]),
        direct_pose_leg_align_oracle_min_deg=float(fixture_config["direct_pose_leg_align_oracle_min_deg"]),
        direct_pose_leg_align_oracle_weight_deg=float(fixture_config["direct_pose_leg_align_oracle_weight_deg"]),
        direct_pose_leg_align_mode=str(fixture_config["direct_pose_leg_align_mode"]),
        direct_pose_leg_align_mag_weight=float(fixture_config["direct_pose_leg_align_mag_weight"]),
        direct_pose_leg_align_res_weight=float(fixture_config["direct_pose_leg_align_res_weight"]),
        direct_pose_leg_align_sign_weight=float(fixture_config["direct_pose_leg_align_sign_weight"]),
        direct_pose_leg_align_cos_thresh=float(fixture_config["direct_pose_leg_align_cos_thresh"]),
        direct_pose_leg_align_target_joints=str(fixture_config["direct_pose_leg_align_target_joints"]),
        direct_pose_leg_align_anchor_joints=str(fixture_config["direct_pose_leg_align_anchor_joints"]),
        direct_pose_leg_align_anchor_weight=float(fixture_config["direct_pose_leg_align_anchor_weight"]),
        direct_pose_leg_gate_sup_weight=float(fixture_config["direct_pose_leg_gate_sup_weight"]),
        step_weight=torch.tensor(0.75, dtype=torch.float32),
        leg_align_terms=term_lists["leg_align_terms"],
        leg_align_frac_terms=term_lists["leg_align_frac_terms"],
        leg_align_joint_num_terms=term_lists["leg_align_joint_num_terms"],
        leg_align_joint_den_terms=term_lists["leg_align_joint_den_terms"],
        leg_align_joint_frac_terms=term_lists["leg_align_joint_frac_terms"],
        leg_align_distal_terms=term_lists["leg_align_distal_terms"],
        leg_align_distal_frac_terms=term_lists["leg_align_distal_frac_terms"],
        leg_align_proximal_terms=term_lists["leg_align_proximal_terms"],
        leg_align_proximal_frac_terms=term_lists["leg_align_proximal_frac_terms"],
        leg_align_anchor_terms=term_lists["leg_align_anchor_terms"],
        leg_align_anchor_frac_terms=term_lists["leg_align_anchor_frac_terms"],
        leg_gate_sup_terms=term_lists["leg_gate_sup_terms"],
        leg_gate_sup_tgt_frac_terms=term_lists["leg_gate_sup_tgt_frac_terms"],
        leg_gate_sup_pred_mean_terms=term_lists["leg_gate_sup_pred_mean_terms"],
    )

    return {
        "file_name": file_name,
        "target": "_lambda_rollout_apply_direct_leg_adjustments",
        "seed": seed,
        "config": fixture_config,
        "inputs": {
            "direct_raw_base": direct_raw_base,
            "R_gt": r_gt,
            "step_weight": torch.tensor(0.75, dtype=torch.float32),
            "ret": {
                "direct_leg_omega": direct_leg_omega,
                "direct_leg_gate_logits": gate_logits,
            },
        },
        "outputs": {
            "direct_raw_base": output,
        },
    }


def _build_all_fixtures() -> list[dict]:
    return [
        _build_compose_fixture(file_name="compose_delta_raw_seed000.pt", seed=0, with_omega_hat=False),
        _build_compose_fixture(file_name="compose_delta_raw_seed001.pt", seed=1, with_omega_hat=True),
        _build_lambda_fixture(file_name="lambda_blend_seed010.pt", seed=10, lambda_shape="B1", direct_norm_shape="BD"),
        _build_lambda_fixture(file_name="lambda_blend_seed011.pt", seed=11, lambda_shape="BJ", direct_norm_shape="BD"),
        _build_lambda_fixture(file_name="lambda_blend_seed012.pt", seed=12, lambda_shape="B1J", direct_norm_shape="B1D"),
        _build_lambda_fixture(file_name="lambda_blend_seed013.pt", seed=13, lambda_shape="BJ1", direct_norm_shape="BD"),
        _build_direct_leg_fixture(
            file_name="direct_leg_adjust_seed020.pt",
            seed=20,
            leg_idx=[1, 3],
            root_idx=0,
            stopgrad_base=False,
            align_mode="cos",
            omega_shape="BL3",
        ),
        _build_direct_leg_fixture(
            file_name="direct_leg_adjust_seed021.pt",
            seed=21,
            leg_idx=[0, 2, 3],
            root_idx=0,
            stopgrad_base=True,
            align_mode="proj",
            omega_shape="B1L3",
        ),
    ]


def _write_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for fixture in _build_all_fixtures():
        path = FIXTURE_DIR / fixture["file_name"]
        torch.save(_to_cpu_tree(fixture), path)
        print(f"wrote {path}")


def _load_fixture(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _assert_close_case(*, case_name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    except AssertionError as exc:
        raise AssertionError(f"{case_name}: {exc}") from exc


def _replay_compose_fixture(fixture: dict) -> torch.Tensor:
    trainer = _make_trainer(fixture_config=fixture["config"])
    inputs = fixture["inputs"]
    return trainer._compose_delta_to_raw(
        inputs["y_prev_raw"],
        inputs["delta_norm"],
        omega_hat=inputs["omega_hat"],
    )


def _replay_lambda_fixture(fixture: dict) -> torch.Tensor:
    trainer = _make_trainer(fixture_config=fixture["config"])
    inputs = fixture["inputs"]
    return trainer._apply_lambda_fusion_to_raw(
        inputs["y_inc_raw"],
        direct_norm=inputs["direct_norm"],
        lambda_fusion=inputs["lambda_fusion"],
    )


def _replay_direct_leg_fixture(fixture: dict) -> torch.Tensor:
    config = fixture["config"]
    trainer = _make_trainer(fixture_config=config)
    model = _make_direct_model(fixture_config=config)
    inputs = fixture["inputs"]
    term_lists = _empty_direct_term_lists()
    return _lambda_rollout_apply_direct_leg_adjustments(
        trainer=trainer,
        model=model,
        ret=inputs["ret"],
        direct_raw_base=inputs["direct_raw_base"],
        R_gt=inputs["R_gt"],
        B=int(inputs["direct_raw_base"].shape[0]),
        J=int(config["feature_dim"] // 6),
        device=torch.device("cpu"),
        dtype=torch.float32,
        columns=tuple(config["columns"]),
        rot_slice=_spec_to_slice(config["rot_slice"]),
        rot_len=int(config["feature_dim"]),
        direct_pose_leg_align_weight=float(config["direct_pose_leg_align_weight"]),
        direct_pose_leg_align_oracle_min_deg=float(config["direct_pose_leg_align_oracle_min_deg"]),
        direct_pose_leg_align_oracle_weight_deg=float(config["direct_pose_leg_align_oracle_weight_deg"]),
        direct_pose_leg_align_mode=str(config["direct_pose_leg_align_mode"]),
        direct_pose_leg_align_mag_weight=float(config["direct_pose_leg_align_mag_weight"]),
        direct_pose_leg_align_res_weight=float(config["direct_pose_leg_align_res_weight"]),
        direct_pose_leg_align_sign_weight=float(config["direct_pose_leg_align_sign_weight"]),
        direct_pose_leg_align_cos_thresh=float(config["direct_pose_leg_align_cos_thresh"]),
        direct_pose_leg_align_target_joints=str(config["direct_pose_leg_align_target_joints"]),
        direct_pose_leg_align_anchor_joints=str(config["direct_pose_leg_align_anchor_joints"]),
        direct_pose_leg_align_anchor_weight=float(config["direct_pose_leg_align_anchor_weight"]),
        direct_pose_leg_gate_sup_weight=float(config["direct_pose_leg_gate_sup_weight"]),
        step_weight=inputs["step_weight"],
        leg_align_terms=term_lists["leg_align_terms"],
        leg_align_frac_terms=term_lists["leg_align_frac_terms"],
        leg_align_joint_num_terms=term_lists["leg_align_joint_num_terms"],
        leg_align_joint_den_terms=term_lists["leg_align_joint_den_terms"],
        leg_align_joint_frac_terms=term_lists["leg_align_joint_frac_terms"],
        leg_align_distal_terms=term_lists["leg_align_distal_terms"],
        leg_align_distal_frac_terms=term_lists["leg_align_distal_frac_terms"],
        leg_align_proximal_terms=term_lists["leg_align_proximal_terms"],
        leg_align_proximal_frac_terms=term_lists["leg_align_proximal_frac_terms"],
        leg_align_anchor_terms=term_lists["leg_align_anchor_terms"],
        leg_align_anchor_frac_terms=term_lists["leg_align_anchor_frac_terms"],
        leg_gate_sup_terms=term_lists["leg_gate_sup_terms"],
        leg_gate_sup_tgt_frac_terms=term_lists["leg_gate_sup_tgt_frac_terms"],
        leg_gate_sup_pred_mean_terms=term_lists["leg_gate_sup_pred_mean_terms"],
    )


class BaselineNumericRegressionTest(unittest.TestCase):
    def test_compose_delta_to_raw_replays_saved_fixtures(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("compose_delta_raw_seed*.pt"))
        self.assertGreaterEqual(
            len(fixture_paths),
            2,
            f"Missing compose fixtures in {FIXTURE_DIR}; run `python3 {Path(__file__)} --write-fixtures`.",
        )
        for path in fixture_paths:
            with self.subTest(case=path.name):
                fixture = _load_fixture(path)
                actual = _replay_compose_fixture(fixture)
                _assert_close_case(
                    case_name=path.name,
                    actual=actual,
                    expected=fixture["outputs"]["y_next_raw"],
                )

    def test_lambda_fusion_to_raw_replays_saved_fixtures(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("lambda_blend_seed*.pt"))
        self.assertGreaterEqual(
            len(fixture_paths),
            4,
            f"Missing lambda fixtures in {FIXTURE_DIR}; run `python3 {Path(__file__)} --write-fixtures`.",
        )
        for path in fixture_paths:
            with self.subTest(case=path.name):
                fixture = _load_fixture(path)
                actual = _replay_lambda_fixture(fixture)
                _assert_close_case(
                    case_name=path.name,
                    actual=actual,
                    expected=fixture["outputs"]["y_blend_raw"],
                )

    def test_direct_leg_adjustments_replay_saved_fixtures(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("direct_leg_adjust_seed*.pt"))
        self.assertGreaterEqual(
            len(fixture_paths),
            2,
            f"Missing direct-leg fixtures in {FIXTURE_DIR}; run `python3 {Path(__file__)} --write-fixtures`.",
        )
        for path in fixture_paths:
            with self.subTest(case=path.name):
                fixture = _load_fixture(path)
                actual = _replay_direct_leg_fixture(fixture)
                _assert_close_case(
                    case_name=path.name,
                    actual=actual,
                    expected=fixture["outputs"]["direct_raw_base"],
                )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Write or replay deterministic numeric baseline fixtures.")
    parser.add_argument("--write-fixtures", action="store_true", help="Regenerate `.pt` baseline fixtures in-place.")
    args, remaining = parser.parse_known_args()
    if args.write_fixtures:
        _write_fixtures()
        return
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    _main()
