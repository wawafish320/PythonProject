from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tools.run_cp015_tailk7_exit_optizability_audit import _seed_group_norm_ema as _seed_exit_group_norm_ema
from tools.run_cp015_tailk7_replace_efficiency_audit import _seed_group_norm_ema as _seed_replace_group_norm_ema
from train.models import MotionJointLoss
from train.posttrain import LambdaFusionAccum, LambdaFusionFinalizeContext, _finalize_direct_group_norm, _lambda_rollout_accumulate_direct_objective
from train.training_MPL import Trainer


def _make_output_layout(num_joints: int) -> dict[str, dict[str, int]]:
    return {"BoneRotations6D": {"start": 0, "size": int(num_joints) * 6}}


def _build_loss(
    *,
    arm_else_balance_enable: bool,
    arm_weight: float,
    else_weight: float,
    group_norm_beta: float = 0.5,
) -> MotionJointLoss:
    loss = MotionJointLoss(
        output_layout=_make_output_layout(3),
        w_direct_pose=1.0,
        direct_pose_loss_leg_split=True,
        direct_pose_leg_bones=("leg",),
        direct_pose_arm_split_enable=True,
        direct_pose_arm_bones=("arm",),
        direct_pose_loss_arm_else_balance_enable=arm_else_balance_enable,
        direct_pose_loss_arm_weight=arm_weight,
        direct_pose_loss_else_weight=else_weight,
        direct_pose_loss_group_norm_enable=True,
        direct_pose_loss_group_norm_ema_beta=group_norm_beta,
        direct_pose_loss_group_norm_ratio_min=0.2,
        direct_pose_loss_group_norm_ratio_max=5.0,
        direct_pose_loss_group_norm_eps=1e-6,
    )
    loss.set_bone_names(["leg", "arm", "spine"])
    loss.root_idx = -1
    return loss


def _make_trainer(loss_fn: MotionJointLoss) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.loss_fn = loss_fn
    return trainer


def _make_weights(*, nonleg_focus_mask: torch.Tensor | None = None, nonleg_focus_weight: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        direct_pose_loss_leg_split=True,
        direct_nonleg_focus_mask_j=nonleg_focus_mask,
        direct_nonleg_focus_resolved=int(nonleg_focus_mask.sum().item()) if torch.is_tensor(nonleg_focus_mask) else 0,
        direct_nonleg_focus_weight_use=float(nonleg_focus_weight),
    )


def _sum_term(terms: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(terms).sum() if terms else torch.tensor(float("nan"), dtype=torch.float32)


class PosttrainDirectGroupNormPhase5Test(unittest.TestCase):
    def _run_accumulate(self, *, trainer: Trainer, e_dir: torch.Tensor, weights: SimpleNamespace) -> LambdaFusionAccum:
        accum = LambdaFusionAccum()
        _lambda_rollout_accumulate_direct_objective(
            trainer=trainer,
            model=SimpleNamespace(direct_pose_leg_joint_idx_tensor=torch.tensor([0], dtype=torch.long)),
            weights=weights,
            accum=accum,
            objective="direct",
            e_dir=e_dir,
            step_weight=torch.tensor(1.0, dtype=e_dir.dtype),
            J=int(e_dir.shape[-1]),
            direct_nonleg_focus_applied=0.0,
        )
        return accum

    def test_posttrain_default_semantics_keep_zero_drift(self) -> None:
        loss = _build_loss(arm_else_balance_enable=False, arm_weight=3.0, else_weight=1.0)
        trainer = _make_trainer(loss)
        loss._direct_pose_group_norm_ema = {
            "leg": torch.tensor(4.0, dtype=torch.float32),
            "nonleg": torch.tensor(10.0, dtype=torch.float32),
        }

        accum = self._run_accumulate(
            trainer=trainer,
            e_dir=torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
            weights=_make_weights(),
        )

        dir_geo, stats, _ = _finalize_direct_group_norm(
            finalize_ctx=LambdaFusionFinalizeContext(
                trainer=trainer,
                model=SimpleNamespace(),
                objective="direct",
                direct_pose_loss_arm_else_balance_enable=False,
                direct_pose_loss_arm_weight=3.0,
                direct_pose_loss_else_weight=1.0,
                direct_group_norm_enable=True,
                direct_group_w_leg=1.0,
                direct_group_w_nonleg=1.0,
                direct_group_beta=0.5,
                direct_group_ratio_min=0.2,
                direct_group_ratio_max=5.0,
                direct_group_eps=1e-6,
            ),
            trainer=trainer,
            blend_loss_total=torch.tensor(0.0, dtype=torch.float32),
            objective="direct",
            dir_geo=torch.tensor(0.0, dtype=torch.float32),
            dir_leg_base_terms=accum.dir_leg_base_terms,
            dir_nonleg_base_terms=accum.dir_nonleg_base_terms,
            dir_arm_base_terms=accum.dir_arm_base_terms,
            dir_else_base_terms=accum.dir_else_base_terms,
            dir_leg_base=_sum_term(accum.dir_leg_base_terms),
            dir_nonleg_base=_sum_term(accum.dir_nonleg_base_terms),
            dir_arm_base=_sum_term(accum.dir_arm_base_terms),
            dir_else_base=_sum_term(accum.dir_else_base_terms),
        )

        self.assertAlmostEqual(stats["dir_nonleg_base"], 5.0, places=6)
        self.assertAlmostEqual(stats["dir_nonleg_effective_base"], 5.0, places=6)
        self.assertAlmostEqual(stats["direct_pose_arm_else_balance_active"], 0.0, places=6)
        self.assertAlmostEqual(float(dir_geo), 1.0, places=6)

    def test_arm_else_balance_aligns_posttrain_with_shared_helper(self) -> None:
        loss = _build_loss(arm_else_balance_enable=True, arm_weight=3.0, else_weight=1.0)
        trainer = _make_trainer(loss)
        loss._direct_pose_group_norm_ema = {
            "leg": torch.tensor(4.0, dtype=torch.float32),
            "nonleg": torch.tensor(9.0, dtype=torch.float32),
        }

        accum = self._run_accumulate(
            trainer=trainer,
            e_dir=torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
            weights=_make_weights(),
        )
        dir_leg_base = _sum_term(accum.dir_leg_base_terms)
        dir_nonleg_base = _sum_term(accum.dir_nonleg_base_terms)
        dir_arm_base = _sum_term(accum.dir_arm_base_terms)
        dir_else_base = _sum_term(accum.dir_else_base_terms)

        shared_base = loss._compute_direct_pose_group_base_payload(
            dir_leg_base=dir_leg_base,
            dir_nonleg_base=dir_nonleg_base,
            dir_arm_base=dir_arm_base,
            dir_else_base=dir_else_base,
            arm_else_balance_enable=True,
            arm_weight=3.0,
            else_weight=1.0,
            eps=1e-6,
        )
        assert shared_base is not None
        expected_geo, _, _ = loss._compute_direct_pose_group_norm_shared(
            dir_leg_base,
            dir_nonleg_base,
            shared_base["dir_nonleg_effective_base"],
            direct_group_w_leg=1.0,
            direct_group_w_nonleg=1.0,
            direct_group_beta=0.5,
            direct_group_ratio_min=0.2,
            direct_group_ratio_max=5.0,
            direct_group_eps=1e-6,
            update_ema_state=False,
        )

        dir_geo, stats, _ = _finalize_direct_group_norm(
            finalize_ctx=LambdaFusionFinalizeContext(
                trainer=trainer,
                model=SimpleNamespace(),
                objective="direct",
                direct_pose_loss_arm_else_balance_enable=True,
                direct_pose_loss_arm_weight=3.0,
                direct_pose_loss_else_weight=1.0,
                direct_group_norm_enable=True,
                direct_group_w_leg=1.0,
                direct_group_w_nonleg=1.0,
                direct_group_beta=0.5,
                direct_group_ratio_min=0.2,
                direct_group_ratio_max=5.0,
                direct_group_eps=1e-6,
            ),
            trainer=trainer,
            blend_loss_total=torch.tensor(0.0, dtype=torch.float32),
            objective="direct",
            dir_geo=torch.tensor(0.0, dtype=torch.float32),
            dir_leg_base_terms=accum.dir_leg_base_terms,
            dir_nonleg_base_terms=accum.dir_nonleg_base_terms,
            dir_arm_base_terms=accum.dir_arm_base_terms,
            dir_else_base_terms=accum.dir_else_base_terms,
            dir_leg_base=dir_leg_base,
            dir_nonleg_base=dir_nonleg_base,
            dir_arm_base=dir_arm_base,
            dir_else_base=dir_else_base,
        )

        self.assertAlmostEqual(stats["dir_nonleg_base"], 5.0, places=6)
        self.assertAlmostEqual(stats["dir_nonleg_effective_base"], 4.5, places=6)
        self.assertAlmostEqual(stats["direct_pose_arm_else_balance_active"], 1.0, places=6)
        self.assertAlmostEqual(float(dir_geo), float(expected_geo), places=6)
        for key, expected_value in shared_base.items():
            if key == "dir_base":
                continue
            expected_float = float(expected_value.detach().cpu()) if torch.is_tensor(expected_value) else float(expected_value)
            self.assertIn(key, stats)
            self.assertAlmostEqual(stats[key], expected_float, places=6)

    def test_ema_state_uses_loss_fn_as_canonical_owner(self) -> None:
        loss = _build_loss(arm_else_balance_enable=False, arm_weight=1.0, else_weight=1.0, group_norm_beta=0.5)
        trainer = _make_trainer(loss)
        loss._direct_pose_group_norm_ema = {
            "leg": torch.tensor(4.0, dtype=torch.float32),
            "nonleg": torch.tensor(10.0, dtype=torch.float32),
        }

        accum = self._run_accumulate(
            trainer=trainer,
            e_dir=torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
            weights=_make_weights(),
        )

        _, stats, ema_update_payload = _finalize_direct_group_norm(
            finalize_ctx=LambdaFusionFinalizeContext(
                trainer=trainer,
                model=SimpleNamespace(),
                objective="direct",
                direct_pose_loss_arm_else_balance_enable=False,
                direct_pose_loss_arm_weight=1.0,
                direct_pose_loss_else_weight=1.0,
                direct_group_norm_enable=True,
                direct_group_w_leg=1.0,
                direct_group_w_nonleg=1.0,
                direct_group_beta=0.5,
                direct_group_ratio_min=0.2,
                direct_group_ratio_max=5.0,
                direct_group_eps=1e-6,
            ),
            trainer=trainer,
            blend_loss_total=torch.tensor(0.0, dtype=torch.float32),
            objective="direct",
            dir_geo=torch.tensor(0.0, dtype=torch.float32),
            dir_leg_base_terms=accum.dir_leg_base_terms,
            dir_nonleg_base_terms=accum.dir_nonleg_base_terms,
            dir_arm_base_terms=accum.dir_arm_base_terms,
            dir_else_base_terms=accum.dir_else_base_terms,
            dir_leg_base=_sum_term(accum.dir_leg_base_terms),
            dir_nonleg_base=_sum_term(accum.dir_nonleg_base_terms),
            dir_arm_base=_sum_term(accum.dir_arm_base_terms),
            dir_else_base=_sum_term(accum.dir_else_base_terms),
        )

        self.assertAlmostEqual(stats["dir_group_norm_leg_ema"], 4.0, places=6)
        self.assertAlmostEqual(stats["dir_group_norm_nonleg_ema"], 10.0, places=6)
        self.assertIsInstance(ema_update_payload, dict)
        self.assertEqual(set(ema_update_payload.keys()), {"leg", "nonleg"})
        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["leg"]), 3.0, places=6)
        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["nonleg"]), 7.5, places=6)
        self.assertFalse(hasattr(trainer, "_direct_pose_group_norm_ema"))

    def test_tool_exit_seed_uses_loss_fn_canonical_state(self) -> None:
        loss = _build_loss(arm_else_balance_enable=False, arm_weight=1.0, else_weight=1.0)
        trainer = _make_trainer(loss)

        _seed_exit_group_norm_ema(
            trainer,
            {
                "dir_group_norm_used": 1.0,
                "dir_group_norm_leg_ema": 4.0,
                "dir_group_norm_nonleg_ema": 10.0,
            },
        )

        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["leg"]), 4.0, places=6)
        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["nonleg"]), 10.0, places=6)
        self.assertFalse(hasattr(trainer, "_direct_pose_group_norm_ema"))

    def test_finalize_direct_group_norm_runtime_delegates_to_loss_fn_seams(self) -> None:
        loss = _build_loss(arm_else_balance_enable=True, arm_weight=3.0, else_weight=1.0)
        trainer = _make_trainer(loss)
        accum = self._run_accumulate(
            trainer=trainer,
            e_dir=torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
            weights=_make_weights(),
        )

        with (
            mock.patch.object(
                loss,
                "_compute_direct_pose_group_base_payload",
                wraps=loss._compute_direct_pose_group_base_payload,
            ) as base_payload,
            mock.patch.object(
                loss,
                "_compute_direct_pose_group_norm_shared",
                wraps=loss._compute_direct_pose_group_norm_shared,
            ) as norm_shared,
        ):
            _finalize_direct_group_norm(
                finalize_ctx=LambdaFusionFinalizeContext(
                    trainer=trainer,
                    model=SimpleNamespace(),
                    objective="direct",
                    direct_pose_loss_arm_else_balance_enable=True,
                    direct_pose_loss_arm_weight=3.0,
                    direct_pose_loss_else_weight=1.0,
                    direct_group_norm_enable=True,
                    direct_group_w_leg=1.0,
                    direct_group_w_nonleg=1.0,
                    direct_group_beta=0.5,
                    direct_group_ratio_min=0.2,
                    direct_group_ratio_max=5.0,
                    direct_group_eps=1e-6,
                ),
                trainer=trainer,
                blend_loss_total=torch.tensor(0.0, dtype=torch.float32),
                objective="direct",
                dir_geo=torch.tensor(0.0, dtype=torch.float32),
                dir_leg_base_terms=accum.dir_leg_base_terms,
                dir_nonleg_base_terms=accum.dir_nonleg_base_terms,
                dir_arm_base_terms=accum.dir_arm_base_terms,
                dir_else_base_terms=accum.dir_else_base_terms,
                dir_leg_base=_sum_term(accum.dir_leg_base_terms),
                dir_nonleg_base=_sum_term(accum.dir_nonleg_base_terms),
                dir_arm_base=_sum_term(accum.dir_arm_base_terms),
                dir_else_base=_sum_term(accum.dir_else_base_terms),
            )

        base_payload.assert_called_once()
        norm_shared.assert_called_once()

    def test_tool_replace_seed_uses_loss_fn_canonical_state(self) -> None:
        loss = _build_loss(arm_else_balance_enable=False, arm_weight=1.0, else_weight=1.0)
        trainer = _make_trainer(loss)

        _seed_replace_group_norm_ema(
            trainer,
            {
                "dir_group_norm_used": 1.0,
                "dir_group_norm_leg_ema": 5.0,
                "dir_group_norm_nonleg_ema": 11.0,
            },
        )

        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["leg"]), 5.0, places=6)
        self.assertAlmostEqual(float(loss._direct_pose_group_norm_ema["nonleg"]), 11.0, places=6)
        self.assertFalse(hasattr(trainer, "_direct_pose_group_norm_ema"))


if __name__ == "__main__":
    unittest.main()
