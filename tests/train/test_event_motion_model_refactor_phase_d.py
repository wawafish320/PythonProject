from __future__ import annotations

import copy
import hashlib
import unittest
from unittest import mock

import torch
from torch import nn

from train.checkpoint.compat import (
    maybe_upgrade_direct_pose_split_state_dict,
)
from train.models import EventMotionModel


def _make_state_layout(num_joints: int) -> dict[str, dict[str, int]]:
    rot_dim = int(num_joints) * 6
    angvel_dim = int(num_joints) * 3
    return {
        "RootPosition": {"start": 0, "size": 3},
        "RootVelocity": {"start": 3, "size": 2},
        "BoneRotations6D": {"start": 5, "size": rot_dim},
        "BoneAngularVelocities": {"start": 5 + rot_dim, "size": angvel_dim},
    }


def _make_output_layout(num_joints: int) -> dict[str, dict[str, int]]:
    return {"BoneRotations6D": {"start": 0, "size": int(num_joints) * 6}}


def _make_io(batch_size: int, steps: int, num_joints: int, cond_dim: int, contact_dim: int) -> dict[str, torch.Tensor]:
    dx = 5 + int(num_joints) * 6 + int(num_joints) * 3
    angvel_dim = int(num_joints) * 3
    pose_hist_dim = 12
    return {
        "state": torch.randn(batch_size, steps, dx, dtype=torch.float32),
        "cond": torch.randn(batch_size, steps, cond_dim, dtype=torch.float32),
        "contacts": torch.rand(batch_size, steps, contact_dim, dtype=torch.float32),
        "angvel": torch.randn(batch_size, steps, angvel_dim, dtype=torch.float32),
        "pose_history": torch.randn(batch_size, steps, pose_hist_dim, dtype=torch.float32),
    }


def _build_model(
    *,
    bone_names: list[str],
    direct_mode: str = "concat",
    use_event_clock: bool = False,
    contact_time_pe_dim: int = 0,
    use_phase: bool = False,
    phase_mode: str = "concat",
    split_enable: bool = False,
    arm_split_enable: bool = False,
    leg_enable: bool = False,
    leg_bones: tuple[str, ...] | None = None,
    arm_bones: tuple[str, ...] | None = None,
    nonleg_proj_dim: int = 0,
    leg_mode: str = "rot6d_add",
    leg_gate_mode: str = "none",
    leg_side_routing: bool = False,
    leg_side_cue: str = "none",
    leg_side_embed_dim: int = 0,
    leg_side_sign_gate: bool = False,
    leg_side_rank1: bool = False,
) -> EventMotionModel:
    num_joints = len(bone_names)
    cond_dim = 8
    pose_hist_dim = 12
    torch.manual_seed(0)
    model = EventMotionModel(
        in_state_dim=5 + num_joints * 6 + num_joints * 3,
        out_motion_dim=num_joints * 6,
        cond_dim=cond_dim,
        hidden_dim=48,
        num_layers=2,
        dropout=0.0,
        contact_dim=2,
        angvel_dim=num_joints * 3,
        pose_hist_dim=pose_hist_dim,
        state_layout=_make_state_layout(num_joints),
        output_layout=_make_output_layout(num_joints),
        bone_names=bone_names,
        contact_plan_enable=True,
        contact_plan_hidden=16,
        contact_plan_inject="none",
        use_event_clock=use_event_clock,
        contact_plan_time_pe_dim=contact_time_pe_dim,
        direct_pose_enable=True,
        direct_pose_hidden=32,
        direct_pose_meas_mode=direct_mode,
        direct_pose_plan_drop_prob=0.1,
        direct_pose_meas_drop_prob=0.2,
        direct_pose_meas_noise_std=0.01,
        direct_pose_use_phase_z=use_phase,
        direct_pose_phase_z_mode=phase_mode,
        direct_pose_split_enable=split_enable,
        direct_pose_arm_split_enable=arm_split_enable,
        direct_pose_leg_enable=leg_enable,
        direct_pose_leg_mode=leg_mode,
        direct_pose_leg_gate_mode=leg_gate_mode,
        direct_pose_leg_side_routing=leg_side_routing,
        direct_pose_leg_side_cue=leg_side_cue,
        direct_pose_leg_side_embed_dim=leg_side_embed_dim,
        direct_pose_leg_side_sign_gate=leg_side_sign_gate,
        direct_pose_leg_side_rank1=leg_side_rank1,
        direct_pose_leg_bones=leg_bones,
        direct_pose_arm_bones=arm_bones,
        direct_pose_nonleg_proj_dim=nonleg_proj_dim,
    )
    model.train()
    return model


_FORWARD_SNAPSHOT_EXPECTED = {
    "attn": {"shape": [2, 3, 3], "sum": 5.999999910593033, "mean": 0.3333333283662796, "l2": 1.4185537344259154},
    "contacts_err": {"shape": [2, 3, 2], "sum": 0.3161674439907074, "mean": 0.026347286999225616, "l2": 1.024909898986513},
    "contacts_meas": {"shape": [2, 3, 2], "sum": 5.846404492855072, "mean": 0.48720037440458935, "l2": 1.9690011362325317},
    "contacts_plan": {"shape": [2, 3, 2], "sum": 6.162571936845779, "mean": 0.5135476614038149, "l2": 1.7805717946189827},
    "contacts_plan_logits": {"shape": [2, 3, 2], "sum": 0.6518216542899609, "mean": 0.054318471190830074, "l2": 0.35554002375418364},
    "delta": {"shape": [2, 3, 24], "sum": 7.731795372441411, "mean": 0.05369302341973202, "l2": 3.5947518831066905},
    "h_final": {"shape": [2, 3, 48], "sum": -9.762588888406754e-07, "mean": -3.3897878084745674e-09, "l2": 16.97053025456694},
    "omega_hat": {"shape": [2, 3, 4, 3], "sum": 0.0, "mean": 0.0, "l2": 0.0},
    "out": {"shape": [2, 3, 24], "sum": 7.731795372441411, "mean": 0.05369302341973202, "l2": 3.5947518831066905},
    "out_direct": {"shape": [2, 3, 24], "sum": 5.253597396425903, "mean": 0.03648331525295766, "l2": 1.5652745171242561},
    "plan_z_next": {"shape": [2, 16], "sum": -0.05935699865221977, "mean": -0.0018549062078818679, "l2": 1.294231819640348},
}


def _make_forward_snapshot_output() -> dict[str, torch.Tensor]:
    bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
    torch.manual_seed(12345)
    io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
    model = _build_model(
        bone_names=bone_names,
        direct_mode="concat",
        split_enable=True,
        leg_bones=("thigh_l", "thigh_r"),
    )
    model.eval()
    torch.manual_seed(999)
    with torch.no_grad():
        return model(
            io["state"],
            io["cond"],
            contacts=io["contacts"],
            angvel=io["angvel"],
            pose_history=io["pose_history"],
        )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    keys = sorted(state_dict.keys())
    tensors = {}
    aggregate = hashlib.sha256()
    for key in keys:
        value = state_dict[key]
        meta = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        tensors[key] = meta
        aggregate.update(key.encode("utf-8"))
        aggregate.update(str(meta["shape"]).encode("utf-8"))
        aggregate.update(str(meta["dtype"]).encode("utf-8"))
        aggregate.update(str(meta["sha256"]).encode("utf-8"))
    return {"keys": keys, "tensors": tensors, "aggregate_sha256": aggregate.hexdigest()}


class EventMotionModelRefactorPhaseDTest(unittest.TestCase):
    def test_split_and_nonsplit_direct_forward_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps = 2, 3
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=2)

        for split_enable in (False, True):
            with self.subTest(split_enable=split_enable):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=split_enable,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                out = model(
                    io["state"],
                    io["cond"],
                    contacts=io["contacts"],
                    angvel=io["angvel"],
                    pose_history=io["pose_history"],
                )
                self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
                self.assertTrue(torch.isfinite(out["out_direct"]).all().item())
                if split_enable:
                    self.assertIsNotNone(model.direct_pose_leg_terminal)
                    self.assertIsNotNone(model.direct_pose_out_nonleg)
                else:
                    self.assertIsNone(model.direct_pose_leg_terminal)
                    self.assertIsNone(model.direct_pose_out_nonleg)

    def test_split_checkpoint_upgrade_from_legacy_direct_head(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        legacy_model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=False,
            leg_bones=("thigh_l", "thigh_r"),
        )
        split_model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
        )

        legacy_state = copy.deepcopy(legacy_model.state_dict())
        self.assertTrue(maybe_upgrade_direct_pose_split_state_dict(split_model, legacy_state))
        self.assertNotIn("direct_pose_head.6.weight", legacy_state)
        self.assertIn("direct_pose_leg_terminal.6.weight", legacy_state)
        self.assertIn("direct_pose_out_nonleg.weight", legacy_state)

        load_info = split_model.load_state_dict(legacy_state, strict=False)
        self.assertFalse(any(key.startswith("direct_pose_leg_terminal") for key in load_info.missing_keys))
        self.assertFalse(any(key.startswith("direct_pose_out_nonleg") for key in load_info.missing_keys))
        self.assertFalse(any(key == "direct_pose_head.6.weight" for key in load_info.unexpected_keys))

    def test_split_leg_terminal_forward_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps = 2, 3
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=2)

        model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
        )
        out = model(
            io["state"],
            io["cond"],
            contacts=io["contacts"],
            angvel=io["angvel"],
            pose_history=io["pose_history"],
        )

        self.assertTrue(model.direct_pose_split_enable)
        self.assertIsNotNone(model.direct_pose_leg_terminal)
        self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
        self.assertTrue(torch.isfinite(out["out_direct"]).all().item())

    def test_split_common_branch_init_is_stable_under_terminal_construction_noise(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "calf_l", "calf_r", "arm_l", "arm_r", "spine", "head"]
        model_ref = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )

        original_terminal = EventMotionModel._build_direct_pose_terminal_block

        def noisy_terminal(
            self: EventMotionModel,
            *,
            trunk_dim: int,
            out_dim: int,
            drop: float,
        ) -> nn.Sequential:
            torch.rand(4096)
            return original_terminal(
                self,
                trunk_dim=trunk_dim,
                out_dim=out_dim,
                drop=drop,
            )

        with mock.patch.object(
            EventMotionModel,
            "_build_direct_pose_terminal_block",
            autospec=True,
            side_effect=noisy_terminal,
        ):
            model_noisy_terminal = _build_model(
                bone_names=bone_names,
                direct_mode="concat",
                split_enable=True,
                arm_split_enable=True,
                leg_bones=("thigh_l", "thigh_r"),
                arm_bones=("arm_l", "arm_r"),
                nonleg_proj_dim=16,
                leg_enable=True,
            )

        state_ref = model_ref.state_dict()
        state_noisy = model_noisy_terminal.state_dict()
        for key in (
            "direct_pose_head.0.weight",
            "direct_pose_head.0.bias",
            "direct_pose_head.3.weight",
            "direct_pose_head.3.bias",
            "direct_pose_arm_proj.0.weight",
            "direct_pose_arm_proj.0.bias",
            "direct_pose_else_proj.0.weight",
            "direct_pose_else_proj.0.bias",
            "direct_pose_out_arm.weight",
            "direct_pose_out_arm.bias",
            "direct_pose_out_else.weight",
            "direct_pose_out_else.bias",
            "direct_pose_leg_head.0.weight",
            "direct_pose_leg_head.0.bias",
            "direct_pose_leg_head.3.weight",
            "direct_pose_leg_head.3.bias",
        ):
            with self.subTest(key=key):
                self.assertTrue(torch.equal(state_ref[key], state_noisy[key]), key)

    def test_split_common_branch_init_is_stable_under_early_terminal_layout_flip(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "calf_l", "calf_r", "arm_l", "arm_r", "spine", "head"]
        model_ref = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )

        original_split = EventMotionModel._build_split_head_branch

        def split_with_early_terminal(
            self: EventMotionModel,
            *,
            trunk_dim: int,
            out_dim: int,
            proj_dim: int = 0,
            out_name: str,
            proj_name: str,
            generator=None,
        ):
            if proj_name == "direct_pose_arm_proj":
                _ = self._build_direct_pose_terminal_block(
                    trunk_dim=trunk_dim,
                    out_dim=max(1, int(out_dim)),
                    drop=0.0,
                )
            return original_split(
                self,
                trunk_dim=trunk_dim,
                out_dim=out_dim,
                proj_dim=proj_dim,
                out_name=out_name,
                proj_name=proj_name,
                generator=generator,
            )

        with mock.patch.object(
            EventMotionModel,
            "_build_split_head_branch",
            autospec=True,
            side_effect=split_with_early_terminal,
        ):
            model_layout_flip = _build_model(
                bone_names=bone_names,
                direct_mode="concat",
                split_enable=True,
                arm_split_enable=True,
                leg_bones=("thigh_l", "thigh_r"),
                arm_bones=("arm_l", "arm_r"),
                nonleg_proj_dim=16,
                leg_enable=True,
            )

        state_ref = model_ref.state_dict()
        state_layout_flip = model_layout_flip.state_dict()
        for key in (
            "direct_pose_head.0.weight",
            "direct_pose_head.0.bias",
            "direct_pose_head.3.weight",
            "direct_pose_head.3.bias",
            "direct_pose_arm_proj.0.weight",
            "direct_pose_arm_proj.0.bias",
            "direct_pose_else_proj.0.weight",
            "direct_pose_else_proj.0.bias",
            "direct_pose_out_arm.weight",
            "direct_pose_out_arm.bias",
            "direct_pose_out_else.weight",
            "direct_pose_out_else.bias",
            "direct_pose_leg_head.0.weight",
            "direct_pose_leg_head.0.bias",
            "direct_pose_leg_head.3.weight",
            "direct_pose_leg_head.3.bias",
        ):
            with self.subTest(key=key):
                self.assertTrue(torch.equal(state_ref[key], state_layout_flip[key]), key)

    def test_split_branch_init_is_repeated_construction_deterministic(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "calf_l", "calf_r", "arm_l", "arm_r", "spine", "head"]
        model_a = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )
        torch.rand(4096)
        model_b = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        self.assertEqual(set(state_a.keys()), set(state_b.keys()))
        for key in state_a.keys():
            with self.subTest(key=key):
                self.assertTrue(torch.equal(state_a[key], state_b[key]), key)

    def test_forward_output_snapshot_deterministic_regression(self) -> None:
        out = _make_forward_snapshot_output()

        self.assertEqual(sorted(out.keys()), sorted(_FORWARD_SNAPSHOT_EXPECTED.keys()))
        for key, expected in _FORWARD_SNAPSHOT_EXPECTED.items():
            with self.subTest(key=key):
                value = out[key]
                self.assertTrue(torch.is_tensor(value))
                self.assertEqual(list(value.shape), expected["shape"])
                self.assertTrue(torch.isfinite(value).all().item())
                value64 = value.detach().cpu().double()
                self.assertAlmostEqual(float(value64.sum().item()), expected["sum"], places=6)
                self.assertAlmostEqual(float(value64.mean().item()), expected["mean"], places=6)
                self.assertAlmostEqual(float(torch.linalg.vector_norm(value64).item()), expected["l2"], places=6)

    def test_forward_shell_dispatch_smoke_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
        )
        model.eval()

        calls: list[str] = []
        original_prepare = EventMotionModel._prepare_forward_inputs
        original_init_contact_clock = EventMotionModel._init_contact_clock_forward_defaults
        original_finalize_contact_plan = EventMotionModel._finalize_contact_plan_outputs
        original_build = EventMotionModel._build_forward_base_result
        original_should_run_direct = EventMotionModel._should_run_direct_pose_forward
        original_init_direct = EventMotionModel._init_direct_pose_forward_runtime
        original_write_direct = EventMotionModel._write_forward_direct_pose_outputs
        original_write_lambda = EventMotionModel._write_forward_lambda_fusion_outputs
        original_write_so3 = EventMotionModel._write_forward_so3_delta_outputs
        original_write_period = EventMotionModel._write_forward_period_output

        def prepare_spy(self: EventMotionModel, **kwargs):
            calls.append("prepare_inputs")
            return original_prepare(self, **kwargs)

        def init_contact_clock_spy(self: EventMotionModel):
            calls.append("init_contact_clock_defaults")
            return original_init_contact_clock(self)

        def finalize_contact_plan_spy(self: EventMotionModel, **kwargs):
            calls.append("finalize_contact_plan_outputs")
            return original_finalize_contact_plan(self, **kwargs)

        def build_spy(self: EventMotionModel, **kwargs):
            calls.append("build_base_result")
            return original_build(self, **kwargs)

        def should_run_direct_spy(self: EventMotionModel, contacts_plan):
            calls.append("should_run_direct_pose")
            return original_should_run_direct(self, contacts_plan)

        def init_direct_spy(self: EventMotionModel, runtime_controls):
            calls.append("init_direct_pose_runtime")
            return original_init_direct(self, runtime_controls)

        def write_direct_spy(self: EventMotionModel, result, **kwargs):
            calls.append("write_direct_outputs")
            return original_write_direct(self, result, **kwargs)

        def write_lambda_spy(self: EventMotionModel, result, **kwargs):
            calls.append("write_lambda_fusion_outputs")
            return original_write_lambda(self, result, **kwargs)

        def write_so3_spy(self: EventMotionModel, result, **kwargs):
            calls.append("write_so3_delta_outputs")
            return original_write_so3(self, result, **kwargs)

        def write_period_spy(result, **kwargs):
            calls.append("write_period_output")
            return original_write_period(result, **kwargs)

        with (
            mock.patch.object(EventMotionModel, "_prepare_forward_inputs", autospec=True, side_effect=prepare_spy),
            mock.patch.object(
                EventMotionModel,
                "_init_contact_clock_forward_defaults",
                autospec=True,
                side_effect=init_contact_clock_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_finalize_contact_plan_outputs",
                autospec=True,
                side_effect=finalize_contact_plan_spy,
            ),
            mock.patch.object(EventMotionModel, "_build_forward_base_result", autospec=True, side_effect=build_spy),
            mock.patch.object(
                EventMotionModel,
                "_should_run_direct_pose_forward",
                autospec=True,
                side_effect=should_run_direct_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_init_direct_pose_forward_runtime",
                autospec=True,
                side_effect=init_direct_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_write_forward_direct_pose_outputs",
                autospec=True,
                side_effect=write_direct_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_write_forward_lambda_fusion_outputs",
                autospec=True,
                side_effect=write_lambda_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_write_forward_so3_delta_outputs",
                autospec=True,
                side_effect=write_so3_spy,
            ),
            mock.patch.object(
                EventMotionModel,
                "_write_forward_period_output",
                autospec=True,
                side_effect=write_period_spy,
            ),
        ):
            torch.manual_seed(999)
            with torch.no_grad():
                out = model(
                    io["state"],
                    io["cond"],
                    contacts=io["contacts"],
                    angvel=io["angvel"],
                    pose_history=io["pose_history"],
                )

        self.assertEqual(
            calls,
            [
                "prepare_inputs",
                "init_contact_clock_defaults",
                "finalize_contact_plan_outputs",
                "build_base_result",
                "should_run_direct_pose",
                "init_direct_pose_runtime",
                "write_direct_outputs",
                "write_lambda_fusion_outputs",
                "write_so3_delta_outputs",
                "write_period_output",
            ],
        )
        self.assertIn("out_direct", out)
        self.assertEqual(out["out_direct"].shape, (2, 3, len(bone_names) * 6))

    def test_forward_leg_gate_helper_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_apply_leg_gate = EventMotionModel._apply_direct_pose_leg_gate_outputs

        for side_routing in (False, True):
            with self.subTest(gate_mode="learned", side_routing=side_routing):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode="so3",
                    leg_gate_mode="learned",
                    leg_side_routing=side_routing,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def apply_leg_gate_spy(self: EventMotionModel, **kwargs):
                    calls.append(
                        {
                            "side_routing": kwargs.get("side_positions") is not None,
                            "gate_head_name": kwargs.get("gate_head_name"),
                        }
                    )
                    return original_apply_leg_gate(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_apply_direct_pose_leg_gate_outputs",
                    autospec=True,
                    side_effect=apply_leg_gate_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["side_routing"], side_routing)
                self.assertEqual(
                    calls[0]["gate_head_name"],
                    "direct_pose_leg_gate_head_shared" if side_routing else "direct_pose_leg_gate_head",
                )
                self.assertIn("direct_leg_gate", out)
                self.assertEqual(out["direct_leg_gate"].shape, (2, 3, 2))
                self.assertTrue(torch.isfinite(out["direct_leg_gate"]).all().item())

    def test_forward_leg_scale_helper_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_apply_leg_gate = EventMotionModel._apply_direct_pose_leg_gate_outputs

        for side_routing in (False, True):
            with self.subTest(gate_mode="scale", side_routing=side_routing):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode="so3",
                    leg_gate_mode="scale",
                    leg_side_routing=side_routing,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def apply_leg_gate_spy(self: EventMotionModel, **kwargs):
                    calls.append(
                        {
                            "side_routing": kwargs.get("side_positions") is not None,
                            "gate_head_name": kwargs.get("gate_head_name"),
                        }
                    )
                    return original_apply_leg_gate(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_apply_direct_pose_leg_gate_outputs",
                    autospec=True,
                    side_effect=apply_leg_gate_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["side_routing"], side_routing)
                self.assertEqual(
                    calls[0]["gate_head_name"],
                    "direct_pose_leg_gate_head_shared" if side_routing else "direct_pose_leg_gate_head",
                )
                self.assertIn("direct_leg_scale", out)
                self.assertIn("direct_leg_scale_log", out)
                self.assertIn("direct_leg_scale_log_raw", out)
                self.assertEqual(out["direct_leg_scale"].shape, (2, 3, 2))
                self.assertTrue(torch.isfinite(out["direct_leg_scale"]).all().item())

    def test_forward_side_routed_leg_residual_shell_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_shell = EventMotionModel._forward_side_routed_leg_residual

        cases = (
            {
                "name": "scale",
                "leg_gate_mode": "scale",
                "leg_side_sign_gate": False,
                "leg_side_rank1": False,
                "expected_present": {"direct_leg_omega", "direct_leg_scale", "direct_leg_scale_log", "direct_leg_scale_log_raw"},
                "expected_absent": {"direct_leg_gate", "direct_leg_side_sign_gate"},
            },
            {
                "name": "sign_gate",
                "leg_gate_mode": "none",
                "leg_side_sign_gate": True,
                "leg_side_rank1": False,
                "expected_present": {"direct_leg_omega", "direct_leg_side_sign_gate"},
                "expected_absent": {"direct_leg_gate", "direct_leg_scale"},
            },
            {
                "name": "rank1",
                "leg_gate_mode": "none",
                "leg_side_sign_gate": False,
                "leg_side_rank1": True,
                "expected_present": {"direct_leg_omega"},
                "expected_absent": {"direct_leg_gate", "direct_leg_scale", "direct_leg_side_sign_gate"},
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode="so3",
                    leg_gate_mode=str(case["leg_gate_mode"]),
                    leg_side_routing=True,
                    leg_side_sign_gate=bool(case["leg_side_sign_gate"]),
                    leg_side_rank1=bool(case["leg_side_rank1"]),
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def shell_spy(self: EventMotionModel, **kwargs):
                    calls.append(
                        {
                            "batch_size": kwargs.get("batch_size"),
                            "query_steps": kwargs.get("query_steps"),
                            "plan_other_ablate_mode": kwargs.get("leg_side_plan_other_ablate_mode"),
                        }
                    )
                    return original_shell(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_forward_side_routed_leg_residual",
                    autospec=True,
                    side_effect=shell_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(calls, [{"batch_size": 2, "query_steps": 3, "plan_other_ablate_mode": "none"}])
                self.assertEqual(out["out_direct"].shape, (2, 3, 24))
                for key in case["expected_present"]:
                    self.assertIn(key, out)
                for key in case["expected_absent"]:
                    self.assertNotIn(key, out)

    def test_forward_leg_omega_pre_gate_helper_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_prepare_leg_omega = EventMotionModel._prepare_direct_pose_leg_omega

        for gate_mode in ("learned", "scale"):
            for side_routing in (False, True):
                with self.subTest(gate_mode=gate_mode, side_routing=side_routing):
                    model = _build_model(
                        bone_names=bone_names,
                        direct_mode="concat",
                        split_enable=True,
                        leg_enable=True,
                        leg_mode="so3",
                        leg_gate_mode=gate_mode,
                        leg_side_routing=side_routing,
                        leg_bones=("thigh_l", "thigh_r"),
                    )
                    model.eval()
                    calls: list[dict[str, object]] = []

                    def prepare_leg_omega_spy(self: EventMotionModel, **kwargs):
                        calls.append(
                            {
                                "side_routing": kwargs.get("side_positions") is not None,
                                "joint_count": kwargs.get("joint_count"),
                            }
                        )
                        return original_prepare_leg_omega(self, **kwargs)

                    with mock.patch.object(
                        EventMotionModel,
                        "_prepare_direct_pose_leg_omega",
                        autospec=True,
                        side_effect=prepare_leg_omega_spy,
                    ):
                        torch.manual_seed(999)
                        with torch.no_grad():
                            out = model(
                                io["state"],
                                io["cond"],
                                contacts=io["contacts"],
                                angvel=io["angvel"],
                                pose_history=io["pose_history"],
                            )

                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]["side_routing"], side_routing)
                    self.assertEqual(calls[0]["joint_count"], 2)
                    self.assertIn("direct_leg_omega", out)
                    self.assertEqual(out["direct_leg_omega"].shape, (2, 3, 2, 3))

    def test_forward_side_leg_omega_resolver_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_resolve_side_omegas = EventMotionModel._resolve_direct_pose_side_leg_omegas

        cases = (
            {"name": "sign_gate", "leg_side_sign_gate": True, "leg_side_rank1": False},
            {"name": "rank1", "leg_side_sign_gate": False, "leg_side_rank1": True},
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode="so3",
                    leg_side_routing=True,
                    leg_side_sign_gate=bool(case["leg_side_sign_gate"]),
                    leg_side_rank1=bool(case["leg_side_rank1"]),
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def resolve_side_omegas_spy(self: EventMotionModel, **kwargs):
                    calls.append({"branch_joint_count": kwargs.get("branch_joint_count")})
                    return original_resolve_side_omegas(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_resolve_direct_pose_side_leg_omegas",
                    autospec=True,
                    side_effect=resolve_side_omegas_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(calls, [{"branch_joint_count": 1}])
                self.assertIn("direct_leg_omega", out)
                self.assertEqual(out["direct_leg_omega"].shape, (2, 3, 2, 3))
                if case["name"] == "sign_gate":
                    self.assertIn("direct_leg_side_sign_gate", out)
                    self.assertEqual(out["direct_leg_side_sign_gate"].shape, (2, 3, 2))

    def test_forward_non_side_leg_delta_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_resolve_leg_delta = EventMotionModel._resolve_direct_pose_non_side_leg_delta

        for leg_mode in ("so3", "rot6d_add"):
            with self.subTest(leg_mode=leg_mode):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode=leg_mode,
                    leg_side_routing=False,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def resolve_leg_delta_spy(self: EventMotionModel, **kwargs):
                    calls.append({"joint_count": kwargs.get("joint_count"), "ablation_mode": kwargs.get("ablation_mode")})
                    return original_resolve_leg_delta(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_resolve_direct_pose_non_side_leg_delta",
                    autospec=True,
                    side_effect=resolve_leg_delta_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(calls, [{"joint_count": 2, "ablation_mode": "none"}])
                self.assertEqual(out["out_direct"].shape, (2, 3, 24))

    def test_event_clock_loop_shell_phase_cue_time_bias_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        batch_size, steps = 2, 3
        torch.manual_seed(12345)
        io = _make_io(batch_size, steps, len(bone_names), cond_dim=8, contact_dim=2)
        model = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            use_event_clock=True,
            contact_time_pe_dim=6,
            use_phase=True,
            split_enable=True,
            leg_enable=True,
            leg_mode="so3",
            leg_side_routing=True,
            leg_side_cue="phase_event_age",
            leg_bones=("thigh_l", "thigh_r"),
        )
        model.eval()
        phase_z = torch.randn(batch_size, steps, 4, dtype=torch.float32)
        phase_event_age = torch.rand(batch_size, steps, 2, dtype=torch.float32)
        captured: dict[str, list[tuple[int, ...]]] = {}
        original_finalize = EventMotionModel._finalize_contact_plan_outputs
        test_case = self

        def finalize_spy(model_self: EventMotionModel, **kwargs):
            phase_seq = kwargs.get("phase_in_direct_seq")
            cue_seq = kwargs.get("leg_side_cue_seq")
            test_case.assertIsInstance(phase_seq, list)
            test_case.assertIsInstance(cue_seq, list)
            captured["phase_shapes"] = [tuple(int(dim) for dim in tensor.shape) for tensor in phase_seq]
            captured["cue_shapes"] = [tuple(int(dim) for dim in tensor.shape) for tensor in cue_seq]
            return original_finalize(model_self, **kwargs)

        with (
            mock.patch.object(
                EventMotionModel,
                "_finalize_contact_plan_outputs",
                autospec=True,
                side_effect=finalize_spy,
            ),
            mock.patch.object(
                model.contact_plan_time_head,
                "forward",
                wraps=model.contact_plan_time_head.forward,
            ) as time_head_forward,
        ):
            torch.manual_seed(999)
            with torch.no_grad():
                out = model(
                    io["state"],
                    io["cond"],
                    contacts=io["contacts"],
                    angvel=io["angvel"],
                    pose_history=io["pose_history"],
                    phase_z=phase_z,
                    phase_event_age=phase_event_age,
                )

        self.assertEqual(captured["phase_shapes"], [(batch_size, 4)] * steps)
        self.assertEqual(captured["cue_shapes"], [(batch_size, 2)] * steps)
        self.assertEqual(time_head_forward.call_count, steps)
        expected_keys = {
            "contacts_plan",
            "contacts_plan_logits",
            "contacts_meas",
            "event_clock_delta_meas",
            "event_clock_lr_diff",
            "event_clock_lambda_corr",
            "event_clock_lambda_logit",
            "event_clock_dynamic_prior",
            "event_clock_delta_z",
            "out_direct",
            "plan_z_next",
        }
        self.assertTrue(expected_keys.issubset(out.keys()))
        self.assertEqual(out["out_direct"].shape, (batch_size, steps, len(bone_names) * 6))
        self.assertEqual(out["event_clock_lambda_corr"].shape[:2], (batch_size, steps))
        self.assertEqual(out["event_clock_lambda_logit"].shape[:2], (batch_size, steps))
        self.assertEqual(out["event_clock_dynamic_prior"].shape[:2], (batch_size, steps))
        self.assertEqual(out["event_clock_delta_z"].shape[:2], (batch_size, steps))
        self.assertTrue(torch.isfinite(out["contacts_plan_logits"]).all().item())
        self.assertTrue(torch.isfinite(out["event_clock_lambda_corr"]).all().item())

    def test_forward_non_side_leg_residual_shell_dispatch_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "arm_l", "arm_r"]
        torch.manual_seed(12345)
        io = _make_io(batch_size=2, steps=3, num_joints=len(bone_names), cond_dim=8, contact_dim=2)
        original_shell = EventMotionModel._forward_non_side_leg_residual

        cases = (
            {
                "name": "non_side_learned",
                "leg_mode": "so3",
                "leg_gate_mode": "learned",
                "expected_present": {"direct_leg_omega", "direct_leg_gate"},
                "expected_absent": {"direct_leg_scale", "direct_leg_side_sign_gate"},
            },
            {
                "name": "non_side_rot6d",
                "leg_mode": "rot6d_add",
                "leg_gate_mode": "none",
                "expected_present": set(),
                "expected_absent": {"direct_leg_omega", "direct_leg_gate", "direct_leg_scale", "direct_leg_side_sign_gate"},
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                model = _build_model(
                    bone_names=bone_names,
                    direct_mode="concat",
                    split_enable=True,
                    leg_enable=True,
                    leg_mode=str(case["leg_mode"]),
                    leg_gate_mode=str(case["leg_gate_mode"]),
                    leg_side_routing=False,
                    leg_bones=("thigh_l", "thigh_r"),
                )
                model.eval()
                calls: list[dict[str, object]] = []

                def shell_spy(self: EventMotionModel, **kwargs):
                    calls.append(
                        {
                            "batch_size": kwargs.get("batch_size"),
                            "query_steps": kwargs.get("query_steps"),
                            "cross_leg_ablate_mode": kwargs.get("leg_cross_leg_ablate_mode"),
                        }
                    )
                    return original_shell(self, **kwargs)

                with mock.patch.object(
                    EventMotionModel,
                    "_forward_non_side_leg_residual",
                    autospec=True,
                    side_effect=shell_spy,
                ):
                    torch.manual_seed(999)
                    with torch.no_grad():
                        out = model(
                            io["state"],
                            io["cond"],
                            contacts=io["contacts"],
                            angvel=io["angvel"],
                            pose_history=io["pose_history"],
                        )

                self.assertEqual(calls, [{"batch_size": 2, "query_steps": 3, "cross_leg_ablate_mode": "none"}])
                self.assertEqual(out["out_direct"].shape, (2, 3, 24))
                for key in case["expected_present"]:
                    self.assertIn(key, out)
                for key in case["expected_absent"]:
                    self.assertNotIn(key, out)

    def test_state_dict_fingerprint_repeated_construction_regression(self) -> None:
        bone_names = ["thigh_l", "thigh_r", "calf_l", "calf_r", "arm_l", "arm_r", "spine", "head"]
        model_a = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )
        torch.rand(4096)
        model_b = _build_model(
            bone_names=bone_names,
            direct_mode="concat",
            split_enable=True,
            arm_split_enable=True,
            leg_bones=("thigh_l", "thigh_r"),
            arm_bones=("arm_l", "arm_r"),
            nonleg_proj_dim=16,
            leg_enable=True,
        )

        fingerprint_a = _state_dict_fingerprint(model_a.state_dict())
        fingerprint_b = _state_dict_fingerprint(model_b.state_dict())

        self.assertEqual(fingerprint_a["keys"], fingerprint_b["keys"])
        self.assertEqual(fingerprint_a["aggregate_sha256"], fingerprint_b["aggregate_sha256"])

        tensor_meta_a = fingerprint_a["tensors"]
        tensor_meta_b = fingerprint_b["tensors"]
        self.assertIsInstance(tensor_meta_a, dict)
        self.assertIsInstance(tensor_meta_b, dict)
        self.assertEqual(set(tensor_meta_a.keys()), set(tensor_meta_b.keys()))
        for key in fingerprint_a["keys"]:
            with self.subTest(key=key):
                meta_a = tensor_meta_a[key]
                meta_b = tensor_meta_b[key]
                self.assertEqual(meta_a["shape"], meta_b["shape"])
                self.assertEqual(meta_a["dtype"], meta_b["dtype"])
                self.assertEqual(meta_a["sha256"], meta_b["sha256"])

if __name__ == "__main__":
    unittest.main()
